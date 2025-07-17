import numpy as np
import pcammls
from pcammls import *
import cv2
import time
import json
import open3d as o3d
from datetime import datetime

# --- 用于 OpenCV 鼠标回调的全局变量 ---
roi_pts = []
roi_selecting = False
frame_for_roi = None


def select_roi_callback(event, x, y, flags, param):
    """OpenCV 鼠标回调函数，用于选择 ROI"""
    global roi_pts, roi_selecting, frame_for_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_pts = [(x, y)]
        roi_selecting = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_selecting:
            img_copy = frame_for_roi.copy()
            cv2.rectangle(img_copy, roi_pts[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_selecting = False
        # 确保 x1<x2, y1<y2
        x1, y1 = roi_pts[0]
        x2, y2 = x, y
        roi_pts = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]
        cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_for_roi)


def calculate_and_visualize_plane(points_mm):
    """
    从点云数据中拟合平面，计算旋转矩阵，并进行可视化。

    :param points_mm: (N, 3) 的点云数据，单位为毫米 (mm)。
    :return: 3x3 的旋转矩阵，如果失败则返回 None。
    """
    if len(points_mm) < 500:
        print("错误：ROI内的有效点数量过少，无法进行平面拟合。")
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_mm.astype(np.float64))

    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=10.0,
                                                 ransac_n=3,
                                                 num_iterations=1000)
    except Exception as e:
        print(f"Open3D 平面分割失败: {e}")
        return None

    [a, b, c, d] = plane_model
    print(f"检测到的平面方程: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    plane_normal = np.array([a, b, c])
    norm = np.linalg.norm(plane_normal)
    if norm == 0:
        return None
    plane_normal /= norm
    if plane_normal[2] < 0:
        plane_normal = -plane_normal

    target_normal = np.array([0.0, 0.0, 1.0])
    v = np.cross(plane_normal, target_normal)
    s = np.linalg.norm(v)
    c_dot = np.dot(plane_normal, target_normal)

    if np.isclose(s, 0):
        rotation_matrix = np.identity(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.identity(3) + vx + vx @ vx * ((1 - c_dot) / (s ** 2))

    print("\n计算出的倾斜校正旋转矩阵 R:")
    print(rotation_matrix)

    # --- 可视化代码 ---
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 0.8, 0])  # 平面内点为绿色
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # ROI内但非平面的点为灰色

    try:
        hull, _ = inlier_cloud.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color([1, 0, 0])  # 红色边框
        geometries_to_draw = [inlier_cloud, outlier_cloud, hull_ls]
    except Exception as e:
        print(f"警告：凸包计算失败 ({e})，仅显示点云。")
        geometries_to_draw = [inlier_cloud, outlier_cloud]

    print("正在显示可视化窗口... 关闭窗口后程序将继续。")
    o3d.visualization.draw_geometries(geometries_to_draw)

    return rotation_matrix


class PythonPercipioDeviceEvent(pcammls.DeviceEvent):
    Offline = False

    def __init__(self):
        pcammls.DeviceEvent.__init__(self)

    def run(self, handle, eventID):
        if eventID == TY_EVENT_DEVICE_OFFLINE:
            print('=== Event Callback: Device Offline!')
            self.Offline = True
        return 0

    def IsOffline(self):
        return self.Offline


def extract_nearest_surface_mask(roi_p3d_aligned, depth_margin):
    """
    (旧方法) 从 ROI 点云中提取最浅表面的 mask（基于 Z 值层提取）
    此方法会产生透视畸变。
    """
    z_img = roi_p3d_aligned[:, :, 2].copy()
    valid_mask = z_img > 0
    if not np.any(valid_mask):
        return None, None
    z_min_val = np.min(z_img[valid_mask])
    surface_mask = ((z_img >= z_min_val) & (z_img <= z_min_val + depth_margin)).astype(np.uint8) * 255
    return surface_mask, z_min_val


### --- NEW --- ###
def create_orthographic_mask(roi_p3d_rotated, depth_margin, pixel_size_mm=0.5):
    """
    (新方法) 使用正交投影创建无畸变的掩码。

    Args:
        roi_p3d_rotated (np.ndarray): (H, W, 3) 旋转后的ROI点云。
        depth_margin (float): 表面层提取的容差 (mm)。
        pixel_size_mm (float): 输出掩码中每个像素代表的真实世界尺寸 (mm)。

    Returns:
        tuple: (ortho_mask, area_pixels, area_mm2, ortho_visualization)
               - ortho_mask: 生成的二进制掩码 (uint8)。
               - area_pixels: 像素面积。
               - area_mm2: 真实物理面积 (mm²)。
               - ortho_visualization: 用于调试的彩色俯视深度图。
               失败时返回 (None, 0, 0, None)。
    """
    # 1. 将点云重塑为点列表并过滤无效点
    points = roi_p3d_rotated.reshape(-1, 3)
    valid_points = points[points[:, 2] > 0]

    if valid_points.shape[0] == 0:
        print("正交投影错误: ROI内无有效点。")
        return None, 0, 0, None

    # 2. 基于旋转后的Z轴提取最表层
    z_min_val = np.min(valid_points[:, 2])
    surface_points = valid_points[(valid_points[:, 2] >= z_min_val) &
                                  (valid_points[:, 2] <= z_min_val + depth_margin)]

    if surface_points.shape[0] == 0:
        print("正交投影错误: 未在最表层找到点。")
        return None, 0, 0, None

    # 3. 获取表层点在XY平面上的包围盒
    x_min, y_min, _ = np.min(surface_points, axis=0)
    x_max, y_max, _ = np.max(surface_points, axis=0)

    # 4. 计算正交掩码的像素尺寸
    width_px = int(np.ceil((x_max - x_min) / pixel_size_mm)) + 1
    height_px = int(np.ceil((y_max - y_min) / pixel_size_mm)) + 1

    if width_px <= 1 or height_px <= 1:
        print("正交投影错误: 计算出的掩码尺寸无效。")
        return None, 0, 0, None

    # 5. 创建正交掩码和可视化图像
    ortho_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    ortho_vis_depth = np.zeros((height_px, width_px), dtype=np.float32)

    # 6. 将三维点投影到二维网格
    px_coords = np.floor((surface_points[:, 0] - x_min) / pixel_size_mm).astype(int)
    py_coords = np.floor((surface_points[:, 1] - y_min) / pixel_size_mm).astype(int)

    # 填充掩码和深度图
    ortho_mask[py_coords, px_coords] = 255
    ortho_vis_depth[py_coords, px_coords] = surface_points[:, 2]

    # 7. 从深度数据创建可供调试的可视化图像
    vis_img = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    valid_vis_mask = ortho_vis_depth > 0
    if np.any(valid_vis_mask):
        z_vals = ortho_vis_depth[valid_vis_mask]
        # 将深度值归一化到0-255以便显示
        z_norm = 255 * (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
        vis_img[valid_vis_mask] = cv2.applyColorMap(z_norm.astype(np.uint8), cv2.COLORMAP_JET)[0]
        vis_img[~valid_vis_mask] = 0  # 无数据区域为黑色

    # 8. 计算面积
    area_pixels = np.count_nonzero(ortho_mask)
    area_mm2 = area_pixels * (pixel_size_mm ** 2)

    print(f"\t已创建正交投影掩码: {width_px}x{height_px} px")
    print(f"\t真实物理面积: {area_mm2:.2f} mm^2")

    return ortho_mask, area_pixels, area_mm2, vis_img


def calculate_mask_area(surface_mask):
    """计算掩码面积（非零像素的数量）"""
    if surface_mask is None:
        return 0
    return np.count_nonzero(surface_mask)


def visualize_mask_with_info(surface_mask, title, area, z_min, area_mm2=None):
    """可视化掩码并显示相关信息"""
    if surface_mask is None:
        return

    vis_img = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(vis_img, f"Area: {area} pixels", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_img, f"Z_min: {z_min:.2f} mm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if area_mm2 is not None:
        cv2.putText(vis_img, f"Area: {area_mm2:.2f} mm^2", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(title, vis_img)


class MaskAreaComparator:
    """掩码面积对比器"""

    ### --- MODIFIED --- ###
    def __init__(self, roi_coords=None, pixel_size_mm=0.5):
        if roi_coords is None:
            self.ROI_X1, self.ROI_Y1 = 707, 600
            self.ROI_X2, self.ROI_Y2 = 927, 804
        else:
            self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = roi_coords

        self.depth_margin = 4.0
        self.pixel_size_mm = pixel_size_mm  # 新增：正交投影的像素大小
        self.cl = None
        self.handle = None
        self.event = None
        self.depth_calib_data = None
        self.scale_unit = None
        self.pointcloud_data_arr = None

        try:
            self.rotation_matrix = np.load('tilt_correction_matrix.npy')
            print("成功加载倾斜校正矩阵。")
        except FileNotFoundError:
            self.rotation_matrix = None
            print("警告: 未找到倾斜校正矩阵。")

    def initialize_camera(self):
        """初始化相机"""
        self.cl = PercipioSDK()
        dev_list = self.cl.ListDevice()
        if len(dev_list) == 0:
            print('没有找到设备');
            return False
        selected_idx = 0;
        sn = dev_list[selected_idx].id
        self.handle = self.cl.Open(sn)
        if not self.cl.isValidHandle(self.handle):
            print('设备打开失败');
            return False
        self.event = PythonPercipioDeviceEvent()
        self.cl.DeviceRegiststerCallBackEvent(self.event)
        depth_fmt_list = self.cl.DeviceStreamFormatDump(self.handle, PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamFormatConfig(self.handle, PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])
        self.cl.DeviceLoadDefaultParameters(self.handle)
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(self.handle)
        self.depth_calib_data = self.cl.DeviceReadCalibData(self.handle, PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamEnable(self.handle, PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamOn(self.handle)
        self.pointcloud_data_arr = pointcloud_data_list()
        print("相机初始化成功")
        return True

    ### --- MODIFIED --- ###
    def capture_and_analyze(self, measurement_name, apply_tilt_correction=False):
        """
        捕获当前帧并分析掩码面积。
        当 apply_tilt_correction 为 True 时，使用正交投影。
        """
        if self.handle is None:
            print("相机未初始化");
            return None

        print(f"\n开始捕获 {measurement_name} 状态的数据...")
        for _ in range(5): self.cl.DeviceStreamRead(self.handle, 2000)

        image_list = self.cl.DeviceStreamRead(self.handle, 2000)
        if len(image_list) == 0: print("无法获取图像数据"); return None

        img_depth = image_list[0]
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data,
                                                   self.scale_unit, self.pointcloud_data_arr)
        p3d_nparray = self.pointcloud_data_arr.as_nparray()

        result = None
        # --- 新的逻辑分支：应用倾斜校正和正交投影 ---
        if apply_tilt_correction and self.rotation_matrix is not None:
            print(f"  对 {measurement_name} 应用倾斜校正和正交投影...")

            # 1. 旋转整个点云
            original_shape = p3d_nparray.shape
            points = p3d_nparray.reshape(-1, 3)
            valid_points_mask = points[:, 2] > 0
            rotated_points = points.copy()
            rotated_points[valid_points_mask] = rotated_points[valid_points_mask] @ self.rotation_matrix.T
            p3d_rotated_nparray = rotated_points.reshape(original_shape)

            # 2. 从旋转后的点云中提取ROI
            roi_cloud_rotated = p3d_rotated_nparray[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]

            # 3. 使用正交投影生成无畸变掩码
            ortho_mask, area_pixels, area_mm2, ortho_vis = create_orthographic_mask(
                roi_cloud_rotated, self.depth_margin, self.pixel_size_mm
            )

            if ortho_mask is None:
                print(f"  {measurement_name} 状态: 创建正交掩码失败。");
                return None

            z_min_val = np.min(roi_cloud_rotated[roi_cloud_rotated[:, :, 2] > 0][:, 2])

            # --- 可视化对比 ---
            print("  生成旧方法（畸变）掩码用于对比...")
            distorted_mask, _ = extract_nearest_surface_mask(roi_cloud_rotated, self.depth_margin)
            if distorted_mask is not None:
                distorted_area = calculate_mask_area(distorted_mask)
                h1, w1 = distorted_mask.shape[:2];
                h2, w2 = ortho_mask.shape[:2]
                max_h = max(h1, h2)
                # 创建并排显示的画布
                comp_vis = np.zeros((max_h, w1 + w2 + 10, 3), dtype=np.uint8)
                # 左侧：畸变掩码
                distorted_vis = cv2.cvtColor(distorted_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(distorted_vis, "Distorted (Old)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(distorted_vis, f"Area: {distorted_area} px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
                comp_vis[0:h1, 0:w1] = distorted_vis
                # 右侧：校正后掩码
                corrected_vis = cv2.cvtColor(ortho_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(corrected_vis, "Corrected (Ortho)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(corrected_vis, f"Area: {area_pixels} px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                cv2.putText(corrected_vis, f"({area_mm2:.1f} mm^2)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                comp_vis[0:h2, w1 + 10:w1 + 10 + w2] = corrected_vis

                cv2.imshow(f"Correction Comparison - {measurement_name}", comp_vis)
                if ortho_vis is not None:
                    cv2.imshow(f"Ortho Top-Down View - {measurement_name}", ortho_vis)

            result = {
                'name': measurement_name, 'timestamp': datetime.now().isoformat(),
                'area_pixels': area_pixels, 'area_mm2': area_mm2,  # 新增真实面积
                'z_min_mm': float(z_min_val), 'depth_margin_mm': self.depth_margin,
                'roi_coords': (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2),
                'tilt_corrected': True, 'correction_method': 'Orthographic'  # 标明校正方法
            }

        # --- 旧的逻辑分支：不校正或校正矩阵不可用 ---
        else:
            if apply_tilt_correction: print("  警告: 请求倾斜校正但矩阵不可用。")
            print(f"  分析 {measurement_name} (无校正)...")
            roi_cloud = p3d_nparray[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]
            surface_mask, z_min = extract_nearest_surface_mask(roi_cloud, self.depth_margin)
            if surface_mask is None:
                print(f"  {measurement_name} 状态: 提取表面掩码失败。");
                return None
            area = calculate_mask_area(surface_mask)
            visualize_mask_with_info(surface_mask, f"{measurement_name} - Mask (Uncorrected)", area, z_min)
            result = {
                'name': measurement_name, 'timestamp': datetime.now().isoformat(),
                'area_pixels': area, 'area_mm2': None,  # 无真实面积
                'z_min_mm': float(z_min) if z_min is not None else -1, 'depth_margin_mm': self.depth_margin,
                'roi_coords': (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2), 'tilt_corrected': False
            }

        print(f"  {measurement_name} 状态分析完成:")
        print(f"    像素面积: {result['area_pixels']} 像素")
        if result.get('area_mm2'):
            print(f"    真实面积: {result['area_mm2']:.2f} mm^2")
        print(f"    最小深度: {result['z_min_mm']:.2f} mm")
        return result

    ### --- MODIFIED --- ###
    def compare_measurements(self, normal_result, tilted_result):
        """对比正对和倾斜状态的测量结果，增加真实面积对比"""
        if normal_result is None or tilted_result is None:
            print("测量结果不完整，无法对比");
            return

        normal_area_px = normal_result['area_pixels']
        tilted_area_px = tilted_result['area_pixels']
        normal_area_mm2 = normal_result.get('area_mm2')
        tilted_area_mm2 = tilted_result.get('area_mm2')

        print("\n" + "=" * 60 + "\n掩码面积对比分析结果\n" + "=" * 60)
        print(f"正对状态:")
        print(f"  掩码面积: {normal_area_px} px" + (
            f" ({normal_area_mm2:.2f} mm^2)" if normal_area_mm2 is not None else ""))
        print(f"倾斜状态 (已校正):")
        print(f"  掩码面积: {tilted_area_px} px" + (
            f" ({tilted_area_mm2:.2f} mm^2)" if tilted_area_mm2 is not None else ""))
        print(f"差异分析:")
        area_diff_px = tilted_area_px - normal_area_px
        area_change_px = (area_diff_px / normal_area_px) * 100 if normal_area_px > 0 else 0
        print(f"  像素面积差异: {area_diff_px} 像素 ({area_change_px:+.2f}%)")
        if normal_area_mm2 is not None and tilted_area_mm2 is not None:
            area_diff_mm2 = tilted_area_mm2 - normal_area_mm2
            area_change_mm2 = (area_diff_mm2 / normal_area_mm2) * 100 if normal_area_mm2 > 0 else 0
            print(f"  真实面积差异: {area_diff_mm2:+.2f} mm^2 ({area_change_mm2:+.2f}%)")
        print("=" * 60)

        comparison_result = {
            'comparison_timestamp': datetime.now().isoformat(),
            'normal_measurement': normal_result, 'tilted_measurement': tilted_result,
            'pixel_area_difference': int(area_diff_px), 'pixel_area_change_percent': float(area_change_px),
            'real_area_difference_mm2': float(area_diff_mm2) if normal_area_mm2 is not None else None,
            'real_area_change_percent': float(area_change_mm2) if normal_area_mm2 is not None else None
        }
        self.save_comparison_result(comparison_result)

    def save_comparison_result(self, result):
        """保存对比结果到文件"""
        filename = f"mask_area_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"对比结果已保存到: {filename}")
        except Exception as e:
            print(f"保存结果失败: {e}")

    def run_interactive_comparison(self):
        """运行交互式对比测试（增强版）"""
        if not self.initialize_camera(): return
        try:
            print("\n【第一阶段：正对状态测量】")
            input("请将物体正对相机放置，然后按 Enter 键开始设置...")
            if not self.setup_for_measurement("正对状态"): return
            normal_result = self.capture_and_analyze("正对状态", apply_tilt_correction=True)
            if normal_result is None: return

            print("\n【第二阶段：倾斜状态测量】")
            input("请将物体倾斜放置，然后按 Enter 键开始设置...")
            if not self.setup_for_measurement("倾斜状态"): return
            tilted_result = self.capture_and_analyze("倾斜状态", apply_tilt_correction=True)
            if tilted_result is None: return

            print("\n【第三阶段：结果对比分析】")
            self.compare_measurements(normal_result, tilted_result)
            print("\n测试完成！按任意键关闭所有窗口...")
            cv2.waitKey(0)
        finally:
            self.cleanup()

    def run_simple_comparison(self):
        """运行简单对比测试"""
        if not self.initialize_camera(): return
        try:
            input("\n1. 请将物体正对相机放置，然后按 Enter 键开始测量...")
            # 正对时，不应用旋转，但依然用正交投影计算真实面积
            normal_result = self.capture_and_analyze("正对状态", apply_tilt_correction=True)
            if normal_result is None: return

            input("\n2. 请将物体倾斜放置，然后按 Enter 键开始测量...")
            tilted_result = self.capture_and_analyze("倾斜状态", apply_tilt_correction=True)
            if tilted_result is None: return

            self.compare_measurements(normal_result, tilted_result)
            print("\n测试完成！按任意键关闭窗口...")
            cv2.waitKey(0)
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.handle is not None:
            self.cl.DeviceStreamOff(self.handle)
            self.cl.Close(self.handle)
        cv2.destroyAllWindows()

    def select_roi_interactive(self):
        """交互式ROI选择功能"""
        global roi_pts, frame_for_roi
        roi_pts = []
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", select_roi_callback)
        print("请在 'Select ROI' 窗口中用鼠标拖拽出一个矩形区域。完成后，按 'c' 键确认，或按 'q' 键取消。")
        while True:
            image_list = self.cl.DeviceStreamRead(self.handle, 2000)
            img_depth = next((f for f in image_list if f.streamID == PERCIPIO_STREAM_DEPTH), None)
            if img_depth:
                depth_render_image = image_data()
                self.cl.DeviceStreamDepthRender(img_depth, depth_render_image)
                frame_for_roi = depth_render_image.as_nparray()
                if len(roi_pts) == 2:
                    cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame_for_roi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Select ROI"); return None
            elif key == ord('c'):
                if len(roi_pts) != 2: print("错误：请先选择一个有效的ROI区域再按 'c'。"); continue
                cv2.destroyWindow("Select ROI");
                return (roi_pts[0][0], roi_pts[0][1], roi_pts[1][0], roi_pts[1][1])

    def calculate_tilt_correction_matrix(self, roi_coords=None):
        """基于选定的ROI区域计算倾斜校正矩阵"""
        if roi_coords is None: roi_coords = (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2)
        x1, y1, x2, y2 = roi_coords
        print(f"正在从ROI区域 ({x1}, {y1}) - ({x2}, {y2}) 计算校正矩阵...")
        for _ in range(3): self.cl.DeviceStreamRead(self.handle, 2000)
        image_list = self.cl.DeviceStreamRead(self.handle, 2000)
        img_depth = next((f for f in image_list if f.streamID == PERCIPIO_STREAM_DEPTH), None)
        if img_depth is None: print("无法获取深度图像数据"); return None
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data,
                                                   self.scale_unit, self.pointcloud_data_arr)
        p3d_nparray = self.pointcloud_data_arr.as_nparray()
        roi_p3d = p3d_nparray[y1:y2, x1:x2, :]
        valid_points = roi_p3d[roi_p3d[:, :, 2] > 0]
        print(f"已从ROI采集点云，共 {len(valid_points)} 个有效点。")
        if len(valid_points) < 500: print("ROI内有效点数量不足"); return None
        rotation_matrix = calculate_and_visualize_plane(valid_points)
        if rotation_matrix is not None:
            self.rotation_matrix = rotation_matrix
            print("倾斜校正矩阵计算成功并已更新到实例中")
        return rotation_matrix

    def setup_for_measurement(self, measurement_name):
        """为测量设置平面校正和ROI"""
        print(f"\n=== 设置 {measurement_name} 状态 ===")
        print("步骤1: 校正倾斜平面")
        input("请在设备视野中放置一个平整的参考平面，准备好后按 Enter 键开始平面校正...")
        if not self.calibrate_plane_for_measurement():
            print("平面校正失败");
            return False
        print("步骤2: 选择测量ROI区域")
        input("现在请将目标物体放置在合适位置，准备好后按 Enter 键开始选择ROI区域...")
        roi_coords = self.select_roi_interactive()
        if roi_coords is None:
            print("ROI选择被取消");
            return False
        self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = roi_coords
        print(f"ROI已设置为: ({self.ROI_X1}, {self.ROI_Y1}) - ({self.ROI_X2}, {self.ROI_Y2})")
        return True

    def calibrate_plane_for_measurement(self):
        """为当前测量状态校正平面"""
        print("请用鼠标在 'Select Plane ROI' 窗口中选择平整的参考平面区域")
        global roi_pts, frame_for_roi
        roi_pts = []
        cv2.namedWindow("Select Plane ROI")
        cv2.setMouseCallback("Select Plane ROI", select_roi_callback)
        plane_roi_coords = None
        while True:
            image_list = self.cl.DeviceStreamRead(self.handle, 2000)
            img_depth = next((f for f in image_list if f.streamID == PERCIPIO_STREAM_DEPTH), None)
            if img_depth:
                depth_render_image = image_data()
                self.cl.DeviceStreamDepthRender(img_depth, depth_render_image)
                frame_for_roi = depth_render_image.as_nparray()
                if len(roi_pts) == 2: cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
                cv2.imshow("Select Plane ROI", frame_for_roi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Select Plane ROI"); return False
            elif key == ord('c'):
                if len(roi_pts) != 2: print("错误：请先选择一个有效的平面区域。"); continue
                plane_roi_coords = (roi_pts[0][0], roi_pts[0][1], roi_pts[1][0], roi_pts[1][1])
                cv2.destroyWindow("Select Plane ROI");
                break
        if plane_roi_coords is None: return False
        rotation_matrix = self.calculate_tilt_correction_matrix(plane_roi_coords)
        if rotation_matrix is None: print("校正矩阵计算失败"); return False
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        matrix_filename = f"tilt_correction_matrix_{timestamp}.npy"
        np.save(matrix_filename, rotation_matrix)
        print(f"平面校正矩阵已保存到: {matrix_filename}")
        return True


def main():
    """主函数"""
    print("掩码面积对比工具 (已升级为正交投影方案)")
    print("=" * 50)
    print("1. 增强版 - 动态选择ROI和计算校正矩阵")
    print("2. 简单版 - 使用固定ROI和现有校正矩阵")
    print("3. 退出")
    while True:
        choice = input("\n请选择模式 (1/2/3): ").strip()
        if choice == '1':
            MaskAreaComparator().run_interactive_comparison()
            break
        elif choice == '2':
            MaskAreaComparator().run_simple_comparison()
            break
        elif choice == '3':
            print("退出程序");
            break
        else:
            print("无效选择")


if __name__ == '__main__':
    main()
