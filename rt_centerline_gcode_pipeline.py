# -*- coding: utf-8 -*-
# 实时中轴线识别 + 基于 G 代码路径的理论中轴线匹配 + 偏移发布 主程序
#
# 适用场景：
# - 相机侧置、工件随加工逐步升高 → 识别区域（ROI）需要随着工件高度自适应更新
# - 以车床 G 代码工具路径作为“理论中轴线”，将其投影到相机深度图像坐标
# - 实时对比“实际中轴线”（由深度/点云提取）与“理论中轴线”，输出纠偏量，向车床发布
#
# 依赖：
# - Percipio SDK (pcammls)
# - OpenCV, NumPy, SciPy, scikit-image, open3d
# - 已完成的手眼标定矩阵 hand_eye_transform.npy (2x3，像素→机床 mm 仿射)
# - 可选：倾斜校正矩阵 tilt_correction_matrix.npy (3x3，用于点云 Z 轴校正)
# - 复用 detect_simplified.py 中的掩码/骨架/比较函数（保持与现有项目的一致输出）
#
# 安全声明：
# - 向真实机床发送纠偏前，务必先在仿真/空跑/软限位环境验证！
# - 建议设置最大步进、死区（deadband）、滤波，防止震荡与超程。
import numpy as np
import cv2, time
import pcammls
from pcammls import *
from pathlib import Path

# 复用你的简化检测实现（仅 mode==2 中用到的函数）
import detect_simplified as det  # 需与本脚本同目录
import gcode_utils as gu
from publisher import OffsetPublisher

# ========== 可配置参数 ==========
class Cfg:
    # --- 摄像头/点云 ---
    DEPTH_MARGIN_MM = 3.5       # 最近表面层厚度（mm）
    PIXEL_SIZE_MM = 0.5         # 正交投影像素物理尺寸（mm/px），用于 create_corrected_mask
    ROI_MARGIN_PX = 40          # ROI 扩展边距（像素），用于自适应ROI
    ROI_MIN_WH = (180, 140)     # ROI 最小宽高
    DEAD_BAND_PX = 1.0          # 小于该像素偏移视为 0（抑制抖动）
    MAX_STEP_MM = 0.10          # 每周期最大输出步长（mm），防止过大调整
    EMA_ALPHA = 0.4             # 指数平滑系数，0~1 越大响应越快
    KEY_POINTS = 12             # 偏移采样的理论关键点数量

    # --- G 代码 ---
    GCODE_PATH = "path.nc"      # 你的 G 代码文件（lathe，X/Z）
    GCODE_UNIT_MM = 1.0         # 若为英寸，改为 25.4
    G_ARC_SEG_MM = 0.8          # 圆弧离散段长

    # --- 发布/联机 ---
    CSV_LOG = "offset_log.csv"
    TCP_ADDR = None             # 例如 ("127.0.0.1", 9000)
    UDP_ADDR = None             # 例如 ("127.0.0.1", 9100)
    SERIAL_PORT = None          # 例如 "COM5" / "/dev/ttyUSB0"
    SERIAL_BAUD = 115200
    GCODE_MODE = False          # True 则额外生成 G91 U.. W.. 片段（谨慎！）

    # --- 其他 ---
    SHOW_UI = True              # 打开 OpenCV 可视化
    EXIT_KEY = ord('q')

# ========== 理论中轴线：由 G 代码生成并映射到像素系 ==========
def build_theoretical_centerline_pix(depth_shape_hw, hand_eye_path='hand_eye_transform.npy'):
    # 读取 G 代码并生成机床系折线
    txt = Path(Cfg.GCODE_PATH).read_text(encoding='utf-8', errors='ignore')
    lines = gu.parse_gcode(txt)
    poly_m = gu.gcode_to_machine_polyline(lines, unit_scale=Cfg.GCODE_UNIT_MM, arc_seglen=Cfg.G_ARC_SEG_MM)

    # 加载像素→机床的仿射，求机床→像素
    A = gu.load_hand_eye(hand_eye_path)   # (2,3)
    uv = gu.machine_to_pixel(poly_m, A)   # (N,2)

    # 重采样 & 裁剪到图像范围
    uv = gu.resample_polyline(uv, step=2.0)  # 2像素步长
    uv = gu.clip_to_image(uv, depth_shape_hw, margin=20)
    return uv

# ========== 自适应 ROI 更新：根据上一帧的 mask 或骨架位置扩展 ==========
def update_roi(prev_roi, new_mask, img_shape):
    H,W = img_shape
    if new_mask is not None and np.any(new_mask>0):
        ys, xs = np.where(new_mask>0)
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        x1 = max(0, x1 - Cfg.ROI_MARGIN_PX)
        y1 = max(0, y1 - Cfg.ROI_MARGIN_PX)
        x2 = min(W-1, x2 + Cfg.ROI_MARGIN_PX)
        y2 = min(H-1, y2 + Cfg.ROI_MARGIN_PX)
        # 保证最小尺寸
        w = x2-x1+1; h = y2-y1+1
        if w < Cfg.ROI_MIN_WH[0]:
            cx = (x1+x2)//2; half = Cfg.ROI_MIN_WH[0]//2
            x1 = max(0, cx-half); x2 = min(W-1, cx+half)
        if h < Cfg.ROI_MIN_WH[1]:
            cy = (y1+y2)//2; half = Cfg.ROI_MIN_WH[1]//2
            y1 = max(0, cy-half); y2 = min(H-1, cy+half)
        return [x1,y1,x2,y2]
    # 回退到上一帧
    if prev_roi is not None:
        return prev_roi
    # 否则返回整幅图
    return [0,0,W-1,H-1]

# ========== 偏移滤波与限幅 ==========
class OffsetFilter:
    def __init__(self, alpha=0.4, max_step=0.1, dead_px=1.0, A2x3=None):
        self.alpha = alpha
        self.max_step = max_step
        self.dead_px = dead_px
        self.prev = None
        self.A = A2x3

    def pixel_to_mm(self, dv_px):
        """通过 hand-eye 的线性部分把像素增量换算到 mm（近似）"""
        if self.A is None:
            return np.array([0.0,0.0])
        M = self.A[:,:2]
        # 像素→机床：dXZ = M * dUV
        dmm = (M @ dv_px.reshape(2,1)).flatten()
        return dmm

    def apply(self, avg_dev_px):
        if avg_dev_px is None:
            return None, None
        dv = np.array(avg_dev_px, dtype=float)  # [dx, dy] in pixel
        # 死区
        if np.linalg.norm(dv) < self.dead_px:
            dv[:] = 0.0
        # EMA 平滑（像素）
        if self.prev is None:
            smoothed_px = dv
        else:
            smoothed_px = self.alpha*dv + (1-self.alpha)*self.prev
        self.prev = smoothed_px
        # 换算到 mm，并限幅
        dmm = self.pixel_to_mm(smoothed_px)
        n = np.linalg.norm(dmm)
        if n > self.max_step and n > 1e-9:
            dmm = dmm * (self.max_step / n)
        return smoothed_px, dmm

# ========== 主循环：采集-提取-比较-输出 ==========
def main():
    # 设备初始化
    cl = PercipioSDK()
    devs = cl.ListDevice()
    if not devs:
        print('no device'); return
    handle = cl.Open(devs[0].id)
    if not cl.isValidHandle(handle):
        print('open failed'); return

    # 读取标定
    depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    color_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_COLOR)
    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)

    # 读取矩阵
    rot_R = None
    try:
        rot_R = np.load('tilt_correction_matrix.npy')
    except Exception:
        rot_R = None
        print('[WARN] 未找到倾斜校正矩阵，将跳过 Z 轴校正。')
    A = gu.load_hand_eye('hand_eye_transform.npy')  # (2,3) 像素→机床

    # 启动数据流
    cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    cl.DeviceStreamOn(handle)

    # 预分配
    pc_list = pointcloud_data_list()
    img_depth_render = image_data()

    # 获取一帧，建立理论中轴线（像素）
    # 先读一帧以拿到深度图尺寸
    while True:
        lst = cl.DeviceStreamRead(handle, 2000)
        if lst: break
    depth_frame = next(f for f in lst if f.streamID==PERCIPIO_STREAM_DEPTH)
    cl.DeviceStreamDepthRender(depth_frame, img_depth_render)
    depth_bgr = img_depth_render.as_nparray()
    H, W = depth_bgr.shape[:2]

    theory_uv = build_theoretical_centerline_pix((H,W), hand_eye_path='hand_eye_transform.npy')
    if len(theory_uv)==0:
        print('[ERR] 理论中轴线为空，请检查G代码与手眼矩阵。')
        return

    # 偏移输出
    pub = OffsetPublisher(csv_path=Cfg.CSV_LOG,
                          tcp_addr=Cfg.TCP_ADDR, udp_addr=Cfg.UDP_ADDR,
                          serial_port=Cfg.SERIAL_PORT, serial_baud=Cfg.SERIAL_BAUD,
                          gcode_mode=Cfg.GCODE_MODE)

    filt = OffsetFilter(alpha=Cfg.EMA_ALPHA, max_step=Cfg.MAX_STEP_MM, dead_px=Cfg.DEAD_BAND_PX, A2x3=A)

    # 自适应 ROI
    roi = [0,0,W-1,H-1]
    last_mask = None

    try:
        while True:
            imgs = cl.DeviceStreamRead(handle, 2000)
            if not imgs: continue
            depth_frame = next(f for f in imgs if f.streamID==PERCIPIO_STREAM_DEPTH)
            color_frame = next((f for f in imgs if f.streamID==PERCIPIO_STREAM_COLOR), None)

            # 渲染深度便于显示
            cl.DeviceStreamDepthRender(depth_frame, img_depth_render)
            depth_bgr = img_depth_render.as_nparray()

            # 点云
            cl.DeviceStreamMapDepthImageToPoint3D(depth_frame, depth_calib, scale_unit, pc_list)
            p3d = pc_list.as_nparray().copy()  # (Hc, Wc, 3) 取决于设备

            # 将 p3d resize 到深度渲染尺寸（如尺寸不同，可近似缩放/或从SDK拿相同分辨率）
            if p3d.shape[0]!=H or p3d.shape[1]!=W:
                p3d = cv2.resize(p3d, (W,H), interpolation=cv2.INTER_NEAREST)

            # 倾斜校正
            if rot_R is not None:
                pts = p3d.reshape(-1,3)
                valid = pts[:,2]>0
                pts[valid] = pts[valid] @ rot_R.T
                p3d = pts.reshape(H,W,3)

            # ROI 更新
            roi = update_roi(roi, last_mask, (H,W))
            x1,y1,x2,y2 = roi
            roi_cloud = p3d[y1:y2+1, x1:x2+1]

            # 掩码（根据是否有 R 选择方法）
            if rot_R is None:
                surface_mask, zmin = det.extract_nearest_surface_mask(roi_cloud, depth_margin=Cfg.DEPTH_MARGIN_MM)
            else:
                surface_mask, zmin = det.create_corrected_mask(roi_cloud, depth_margin=Cfg.DEPTH_MARGIN_MM, pixel_size_mm=Cfg.PIXEL_SIZE_MM)
            if surface_mask is None:
                last_mask = None
                if Cfg.SHOW_UI:
                    cv2.imshow('depth', depth_bgr)
                    if cv2.waitKey(1)==Cfg.EXIT_KEY: break
                continue

            # 骨架（实际中轴线，局部→全局像素）
            skel_bgr = det.extract_skeleton_universal(surface_mask, visualize=False)
            actual_pts = det.extract_skeleton_points_and_visualize(skel_bgr, origin_offset=(x1,y1), visualize=False)

            # 记录 mask 以便下次更新 ROI
            last_mask = np.zeros((H,W), dtype=np.uint8)
            last_mask[y1:y2+1, x1:x2+1] = surface_mask

            # 对比与偏移（全部像素坐标系）
            comp_vis, match = det.compare_centerlines(actual_pts if actual_pts.size else np.zeros((0,2)),
                                                      theory_uv,
                                                      (H,W))
            devs = det.calculate_deviation_vectors(actual_pts, theory_uv, num_key_points=Cfg.KEY_POINTS) if actual_pts.size else []

            avg_px = None
            if devs:
                vv = np.array([v for _,v in devs], dtype=float)
                avg_px = vv.mean(axis=0)

            # 滤波/换算/限幅
            sm_px, dmm = filt.apply(avg_px)
            if dmm is not None:
                pub.publish(float(dmm[0]), float(dmm[1]), float(match), note='rt')

            # 可视化叠加
            if Cfg.SHOW_UI:
                # 在对比图上画 ROI 与理论线
                vis = comp_vis.copy()
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
                # 理论线
                for i in range(len(theory_uv)-1):
                    p1 = tuple(theory_uv[i].astype(int))
                    p2 = tuple(theory_uv[i+1].astype(int))
                    cv2.line(vis, p1,p2,(0,255,0),1)
                # 显示平均偏移
                if sm_px is not None:
                    cv2.putText(vis, f"avg_px=({sm_px[0]:.1f},{sm_px[1]:.1f})", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                    cv2.putText(vis, f"step_mm=({dmm[0]:.3f},{dmm[1]:.3f})", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                cv2.putText(vis, f"match={match*100:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.imshow('rt_centerline', vis)
                if cv2.waitKey(1)==Cfg.EXIT_KEY:
                    break

    finally:
        pub.close()
        cl.DeviceStreamOff(handle); cl.Close(handle)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
