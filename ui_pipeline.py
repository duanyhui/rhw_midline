import numpy as np
import cv2
import traceback

import pcammls
from pcammls import PercipioSDK, PERCIPIO_STREAM_COLOR, PERCIPIO_STREAM_DEPTH, image_data, pointcloud_data_list

# 复用现有 detect.py 中的核心函数（不修改原文件，避免破坏注释）
# 函数包括：create_corrected_mask, extract_nearest_surface_mask, extract_skeleton_universal,
#          extract_skeleton_points_and_visualize, compare_centerlines, calculate_deviation_vectors,
#          visualize_deviation_vectors
try:
    import detect as detect_mod
except Exception as e:
    detect_mod = None
    print("警告：无法导入 detect.py，请确认文件与本程序在同一路径或 PYTHONPATH 中。错误：", e)

def ndarray_to_bgr(img):
    """确保输出为 BGR 三通道图像，便于在 UI 中显示。"""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

class CameraSnapshot:
    def __init__(self, depth_render_bgr, depth_raw, pointcloud, color_bgr=None):
        self.depth_render_bgr = depth_render_bgr  # (H,W,3) 8U
        self.depth_raw = depth_raw                # (H,W) or (H,W,1) 16U mm
        self.pointcloud = pointcloud              # (H',W',3) float64 or float32 (mm)
        self.color_bgr = color_bgr                # 可选

def capture_one_frame(map_depth_to_color=True) -> CameraSnapshot:
    """连接设备并抓取一帧数据，返回深度渲染、原始深度、点云（以及可选彩色）。"""
    cl = PercipioSDK()
    devs = cl.ListDevice()
    if not devs:
        raise RuntimeError("未找到 Percipio 设备")
    handle = cl.Open(devs[0].id)
    if not cl.isValidHandle(handle):
        raise RuntimeError("打开设备失败")

    # 仅启用所需流，减少带宽
    cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    cl.DeviceStreamOn(handle)

    # 读取一帧
    depth_frame = None
    color_frame = None
    for _ in range(80):
        image_list = cl.DeviceStreamRead(handle, 2000)
        for f in image_list:
            if f.streamID == PERCIPIO_STREAM_DEPTH:
                depth_frame = f
            elif f.streamID == PERCIPIO_STREAM_COLOR:
                color_frame = f
        if depth_frame is not None and color_frame is not None:
            break

    if depth_frame is None:
        cl.DeviceStreamOff(handle); cl.Close(handle)
        raise RuntimeError("未获取到深度帧")

    # 深度渲染与原始深度
    img_render = image_data()
    cl.DeviceStreamDepthRender(depth_frame, img_render)
    depth_render_bgr = img_render.as_nparray().copy()  # (H,W,3)

    depth_mapped = image_data()
    depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
    # 将深度图映射到彩色分辨率，利于与理论线进行像素级比较
    color_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_COLOR)
    if color_frame is not None:
        cl.DeviceStreamMapDepthImageToColorCoordinate(
            depth_calib, depth_frame, scale_unit, color_calib,
            color_frame.width, color_frame.height, depth_mapped
        )
        depth_raw = depth_mapped.as_nparray().squeeze().copy()
    else:
        # 回退：使用原分辨率深度
        depth_raw = depth_frame.as_nparray().squeeze().copy()

    # 点云
    pointcloud = pointcloud_data_list()
    cl.DeviceStreamMapDepthImageToPoint3D(depth_frame, depth_calib, scale_unit, pointcloud)
    p3d = pointcloud.as_nparray().copy()

    # 彩色
    color_bgr = None
    if color_frame is not None:
        decoded = image_data()
        cl.DeviceStreamImageDecode(color_frame, decoded)
        color_bgr = decoded.as_nparray().copy()

    cl.DeviceStreamOff(handle); cl.Close(handle)
    return CameraSnapshot(depth_render_bgr, depth_raw, p3d, color_bgr)

def apply_tilt_correction(p3d: np.ndarray, rotation_matrix_path='tilt_correction_matrix.npy'):
    """如果存在倾斜校正矩阵，则应用到点云。返回(修正点云, R或None)。"""
    R = None
    try:
        R = np.load(rotation_matrix_path)
        if R.shape == (3,3):
            pts = p3d.reshape(-1,3)
            valid = pts[:,2] > 0
            pts[valid] = pts[valid] @ R.T
            p3d = pts.reshape(p3d.shape)
    except FileNotFoundError:
        pass
    return p3d, R

def load_theoretical_centerline(path='theoretical_centerline.npy'):
    try:
        arr = np.load(path)
        return arr
    except Exception:
        return None

def load_hand_eye(path='hand_eye_transform.npy'):
    try:
        M = np.load(path)
        return M
    except Exception:
        return None

def process_roi(snapshot: CameraSnapshot, roi_xyxy, pixel_size_mm=0.5, depth_margin=3.5,
                keypoints=12):
    """
    针对 ROI 计算：表面掩码 -> 骨架 -> 实际中轴线点 -> 与理论线比较 -> 偏移向量 -> 汇总。
    返回字典包含可视化图和指标。
    """
    if detect_mod is None:
        raise RuntimeError("未能导入 detect.py，无法复用核心算法")

    x1,y1,x2,y2 = roi_xyxy
    x1,x2 = int(x1), int(x2)
    y1,y2 = int(y1), int(y2)

    p3d = snapshot.pointcloud
    # 直接在点云的像素坐标上裁剪（与深度渲染坐标一致的情况下使用）
    roi_cloud = p3d[y1:y2, x1:x2].copy()

    # 倾斜校正（若有）
    roi_cloud, R = apply_tilt_correction(roi_cloud)

    # 表面掩码
    if R is not None and hasattr(detect_mod, "create_corrected_mask"):
        surface_mask, z_min = detect_mod.create_corrected_mask(roi_cloud, depth_margin, pixel_size_mm=pixel_size_mm)
    else:
        surface_mask, z_min = detect_mod.extract_nearest_surface_mask(roi_cloud, depth_margin)

    if surface_mask is None:
        raise RuntimeError("无法从 ROI 提取有效表面掩码")

    # 骨架（中轴线）
    skeleton_bgr = detect_mod.extract_skeleton_universal(surface_mask, visualize=False)
    if skeleton_bgr is None:
        raise RuntimeError("骨架提取失败")
    # 提取像素点（加上 ROI 偏移成为全图坐标）
    actual_pts = detect_mod.extract_skeleton_points_and_visualize(skeleton_bgr, origin_offset=(x1,y1), visualize=False)

    # 理论线
    theoretical = load_theoretical_centerline()

    # 对比与偏移
    comparison_vis, match_score = detect_mod.compare_centerlines(actual_pts if actual_pts.size else np.zeros((0,2)),
                                                                 theoretical if theoretical is not None else np.zeros((0,2)),
                                                                 (snapshot.depth_render_bgr.shape[0], snapshot.depth_render_bgr.shape[1]))
    deviation_vectors = []
    avg_pixel_offset = None
    avg_physical_offset = None
    if theoretical is not None and actual_pts.size:
        deviation_vectors = detect_mod.calculate_deviation_vectors(actual_pts, theoretical, num_key_points=keypoints)
        # 统计平均像素偏移
        if deviation_vectors:
            vectors = np.array([vec for _, vec in deviation_vectors])
            avg_pixel_offset = np.mean(vectors, axis=0)

            # 如有手眼矩阵，换算物理坐标
            M = load_hand_eye()
            if M is not None:
                linear = M[:,:2]  # 2x2
                avg_physical_offset = linear @ avg_pixel_offset

        # 可视化偏移箭头
        final_vis = detect_mod.visualize_deviation_vectors(comparison_vis, deviation_vectors)
    else:
        final_vis = comparison_vis

    # 打包可视化图像
    out = {
        "surface_mask": ndarray_to_bgr(surface_mask),
        "skeleton": ndarray_to_bgr(skeleton_bgr),
        "comparison": ndarray_to_bgr(comparison_vis),
        "final": ndarray_to_bgr(final_vis),
        "actual_points": actual_pts,
        "theoretical_points": theoretical,
        "match_score": float(match_score),
        "avg_pixel_offset": None if avg_pixel_offset is None else avg_pixel_offset.astype(float),
        "avg_physical_offset": None if avg_physical_offset is None else avg_physical_offset.astype(float),
        "deviation_vectors": deviation_vectors,  # list of (key_pt, vec) as ndarrays
    }
    return out

def export_deviation_csv(path, deviation_vectors, avg_pixel_offset=None, avg_physical_offset=None):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "key_point_x", "key_point_y", "dx(px)", "dy(px)"])
        for idx, (kp, vec) in enumerate(deviation_vectors):
            writer.writerow([idx, float(kp[0]), float(kp[1]), float(vec[0]), float(vec[1])])
        writer.writerow([])
        if avg_pixel_offset is not None:
            writer.writerow(["avg_pixel_dx", "avg_pixel_dy", float(avg_pixel_offset[0]), float(avg_pixel_offset[1])])
        if avg_physical_offset is not None:
            writer.writerow(["avg_physical_dx(mm)", "avg_physical_dy(mm)", float(avg_physical_offset[0]), float(avg_physical_offset[1])])
    return path
