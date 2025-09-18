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
    plane_normal = np.array([a, b, c])
    norm = np.linalg.norm(plane_normal)
    if norm == 0: return None
    plane_normal /= norm
    if plane_normal[2] < 0: plane_normal = -plane_normal

    target_normal = np.array([0.0, 0.0, 1.0])
    v = np.cross(plane_normal, target_normal)
    s = np.linalg.norm(v)
    c_dot = np.dot(plane_normal, target_normal)

    if np.isclose(s, 0):
        rotation_matrix = np.identity(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.identity(3) + vx + vx @ vx * ((1 - c_dot) / (s ** 2))

    print("\n计算出的倾斜校正旋转矩阵 R:\n", rotation_matrix)

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([0, 0.8, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    try:
        hull, _ = inlier_cloud.compute_convex_hull()
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color([1, 0, 0])
        geometries_to_draw = [inlier_cloud, outlier_cloud, hull_ls]
    except Exception as e:
        geometries_to_draw = [inlier_cloud, outlier_cloud]

    print("正在显示3D可视化窗口... 关闭窗口后程序将继续。")
    o3d.visualization.draw_geometries(geometries_to_draw)

    return rotation_matrix


class PythonPercipioDeviceEvent(pcammls.DeviceEvent):
    Offline = False

    def __init__(self): pcammls.DeviceEvent.__init__(self)

    def run(self, handle, eventID):
        if eventID == TY_EVENT_DEVICE_OFFLINE:
            print('=== Event Callback: Device Offline!')
            self.Offline = True
        return 0

    def IsOffline(self): return self.Offline


def extract_nearest_surface_mask(roi_p3d_aligned, depth_margin):
    """(旧方法) 从 ROI 点云中提取最浅表面的 mask，会产生透视畸变。"""
    z_img = roi_p3d_aligned[:, :, 2].copy()
    valid_mask = z_img > 0
    if not np.any(valid_mask): return None, None
    z_min_val = np.min(z_img[valid_mask])
    surface_mask = ((z_img >= z_min_val) & (z_img <= z_min_val + depth_margin)).astype(np.uint8) * 255
    return surface_mask, z_min_val


### --- FINAL VERSION OF THE FUNCTION --- ###
def create_orthographic_mask(roi_p3d_rotated, depth_margin, pixel_size_mm=0.5):
    """
    (最终优化版) 使用形态学闭运算连接稀疏点，再通过轮廓层级分析处理中空物体。
    """
    points = roi_p3d_rotated.reshape(-1, 3)
    valid_points = points[points[:, 2] > 0]
    if valid_points.shape[0] < 3:
        print("正交投影错误: 有效点数量过少。")
        return None, 0, 0, None

    z_min_val = np.min(valid_points[:, 2])
    surface_points = valid_points[(valid_points[:, 2] >= z_min_val) & (valid_points[:, 2] <= z_min_val + depth_margin)]
    if surface_points.shape[0] < 3:
        print("正交投影错误: 表面点数量过少。")
        return None, 0, 0, None

    x_min, y_min, _ = np.min(surface_points, axis=0)
    x_max, y_max, _ = np.max(surface_points, axis=0)

    width_px = int(np.ceil((x_max - x_min) / pixel_size_mm)) + 1
    height_px = int(np.ceil((y_max - y_min) / pixel_size_mm)) + 1
    if width_px <= 1 or height_px <= 1: return None, 0, 0, None

    # --- 步骤 1: 创建一个临时二值图像，将所有2D点画上去 ---
    point_img = np.zeros((height_px, width_px), dtype=np.uint8)
    px_coords = np.floor((surface_points[:, 0] - x_min) / pixel_size_mm).astype(int)
    py_coords = np.floor((surface_points[:, 1] - y_min) / pixel_size_mm).astype(int)
    point_img[py_coords, px_coords] = 255

    # --- 新增步骤: 形态学闭运算，连接断开的轮廓 ---
    # 内核大小和迭代次数可能需要根据实际的点云稀疏程度进行微调
    kernel_size = int(np.ceil(1.0 / pixel_size_mm)) * 2 + 1  # 保证内核尺寸覆盖约2mm
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed_img = cv2.morphologyEx(point_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 步骤 2: 在“连接好”的图像上查找所有轮廓及其层级关系 ---
    contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("轮廓查找失败: 未找到任何轮廓。")
        return None, 0, 0, None

    # --- 步骤 3: 计算真实面积 (外轮廓面积 - 内轮廓面积) ---
    total_area_pixels = 0
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # 如果 parent 是 -1, 说明这是一个外轮廓
            if hierarchy[0][i][3] == -1:
                total_area_pixels += cv2.contourArea(contour)
            # 否则，这是一个内轮廓（洞），需要减去它的面积
            else:
                total_area_pixels -= cv2.contourArea(contour)
    else:  # 如果没有层级关系，直接计算所有轮廓面积
        for contour in contours:
            total_area_pixels += cv2.contourArea(contour)

    area_mm2 = total_area_pixels * (pixel_size_mm ** 2)

    # --- 步骤 4: 创建最终的可视化掩码 ---
    # 先画外轮廓（白色填充），再画内轮廓（黑色填充）来“挖洞”
    ortho_mask = np.zeros((height_px, width_px), dtype=np.uint8)
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(ortho_mask, [contour], 0, 255, -1)
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(ortho_mask, [contour], 0, 0, -1)
    else:  # 如果没有层级，直接画所有找到的轮廓
        cv2.drawContours(ortho_mask, contours, -1, 255, -1)

    # (俯视深度图逻辑保持不变，仅用于调试)
    vis_img = None  # 可选

    print(f"\t已创建层级轮廓掩码: {width_px}x{height_px} px")
    print(f"\t真实物理面积 (处理中空后): {area_mm2:.2f} mm^2")

    return ortho_mask, total_area_pixels, area_mm2, vis_img


def show_final_comparison_image(normal_result, tilted_result):
    """
    创建一个专门的窗口，用于并排显示和对比最终的测量结果。
    """
    canvas = np.full((300, 800, 3), 255, dtype=np.uint8)  # 白色画布

    # --- 第一列: 正对状态结果 ---
    cv2.putText(canvas, "Normal State", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    normal_px = normal_result['area_pixels']
    normal_mm2 = normal_result.get('area_mm2')
    cv2.putText(canvas, f"Final Area: {normal_px:.0f} px^2", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100),
                1)
    if normal_mm2 is not None:
        cv2.putText(canvas, "REAL AREA:", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2)
        cv2.putText(canvas, f"{normal_mm2:.2f} mm^2", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 0, 0), 3)

    # --- 分割线 ---
    cv2.line(canvas, (400, 20), (400, 280), (150, 150, 150), 2)

    # --- 第二列: 倾斜状态结果 ---
    cv2.putText(canvas, "Tilted (Corrected)", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    tilted_px = tilted_result['area_pixels']
    tilted_mm2 = tilted_result.get('area_mm2')
    cv2.putText(canvas, f"Final Area: {tilted_px:.0f} px^2", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100),
                1)
    if tilted_mm2 is not None:
        cv2.putText(canvas, "REAL AREA:", (450, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 0), 2)
        cv2.putText(canvas, f"{tilted_mm2:.2f} mm^2", (450, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 150, 0), 3)

    # --- 差异百分比 ---
    if normal_mm2 is not None and tilted_mm2 is not None and normal_mm2 > 0:
        diff_percent = ((tilted_mm2 - normal_mm2) / normal_mm2) * 100
        cv2.putText(canvas, f"Difference: {diff_percent:+.2f}%", (450, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200),
                    2)

    cv2.imshow("Final Measurement Summary", canvas)


class MaskAreaComparator:
    """掩码面积对比器"""

    def __init__(self, roi_coords=None, pixel_size_mm=0.5):
        if roi_coords is None:
            self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = 707, 600, 927, 804
        else:
            self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = roi_coords
        self.depth_margin = 4.0
        self.pixel_size_mm = pixel_size_mm
        self.cl, self.handle, self.event, self.depth_calib_data, self.scale_unit, self.pointcloud_data_arr = (None,) * 6
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
        if len(dev_list) == 0: print('没有找到设备'); return False
        self.handle = self.cl.Open(dev_list[0].id)
        if not self.cl.isValidHandle(self.handle): print('设备打开失败'); return False
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
        print("相机初始化成功");
        return True

    def capture_and_analyze(self, measurement_name, apply_tilt_correction=False):
        if self.handle is None: print("相机未初始化"); return None
        print(f"\n开始捕获 {measurement_name} 状态的数据...")
        for _ in range(5): self.cl.DeviceStreamRead(self.handle, 2000)
        image_list = self.cl.DeviceStreamRead(self.handle, 2000)
        if len(image_list) == 0: print("无法获取图像数据"); return None

        img_depth = image_list[0]
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data, self.scale_unit,
                                                   self.pointcloud_data_arr)
        p3d_nparray = self.pointcloud_data_arr.as_nparray()
        result = None

        if apply_tilt_correction and self.rotation_matrix is not None:
            print(f"  对 {measurement_name} 应用倾斜校正和正交投影...")
            original_shape = p3d_nparray.shape
            points = p3d_nparray.reshape(-1, 3)
            valid_mask = points[:, 2] > 0
            rotated_points = points.copy()
            rotated_points[valid_mask] = rotated_points[valid_mask] @ self.rotation_matrix.T
            p3d_rotated_nparray = rotated_points.reshape(original_shape)
            roi_cloud_rotated = p3d_rotated_nparray[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]

            ortho_mask, area_pixels, area_mm2, ortho_vis = create_orthographic_mask(
                roi_cloud_rotated, self.depth_margin, self.pixel_size_mm)
            if ortho_mask is None: print(f"创建正交掩码失败。"); return None

            z_min_val = np.min(roi_cloud_rotated[roi_cloud_rotated[:, :, 2] > 0][:, 2])

            distorted_mask, _ = extract_nearest_surface_mask(roi_cloud_rotated, self.depth_margin)
            if distorted_mask is not None:
                distorted_area = np.count_nonzero(distorted_mask)
                h1, w1 = distorted_mask.shape[:2];
                h2, w2 = ortho_mask.shape[:2]
                comp_vis = np.zeros((max(h1, h2), w1 + w2 + 10, 3), dtype=np.uint8)
                distorted_vis = cv2.cvtColor(distorted_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(distorted_vis, "Distorted (Old)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(distorted_vis, f"Area: {distorted_area} px", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
                comp_vis[0:h1, 0:w1] = distorted_vis

                corrected_vis = cv2.cvtColor(ortho_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(corrected_vis, "Corrected (Final)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(corrected_vis, f"Final Area: {area_pixels:.0f} px^2", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
                cv2.putText(corrected_vis, f"{area_mm2:.1f} mm^2", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            2)
                comp_vis[0:h2, w1 + 10:w1 + 10 + w2] = corrected_vis

                cv2.imshow(f"Correction Comparison - {measurement_name}", comp_vis)
                if ortho_vis is not None: cv2.imshow(f"Ortho Top-Down View - {measurement_name}", ortho_vis)

            result = {'name': measurement_name, 'timestamp': datetime.now().isoformat(), 'area_pixels': area_pixels,
                      'area_mm2': area_mm2, 'z_min_mm': float(z_min_val), 'depth_margin_mm': self.depth_margin,
                      'roi_coords': (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2), 'tilt_corrected': True,
                      'correction_method': 'Hierarchical Contour Area'}
        else:
            if apply_tilt_correction: print("警告: 请求倾斜校正但矩阵不可用。")
            print(f"分析 {measurement_name} (无校正)...")
            roi_cloud = p3d_nparray[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]
            surface_mask, z_min = extract_nearest_surface_mask(roi_cloud, self.depth_margin)
            if surface_mask is None: print(f"提取表面掩码失败。"); return None
            area = np.count_nonzero(surface_mask)
            result = {'name': measurement_name, 'timestamp': datetime.now().isoformat(), 'area_pixels': area,
                      'area_mm2': None, 'z_min_mm': float(z_min) if z_min is not None else -1,
                      'depth_margin_mm': self.depth_margin,
                      'roi_coords': (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2), 'tilt_corrected': False}

        print(f"  {measurement_name} 状态分析完成:")
        print(f"    最终像素面积: {result['area_pixels']:.2f} px^2")
        if result.get('area_mm2'): print(f"    最终真实面积: {result['area_mm2']:.2f} mm^2")
        print(f"    最小深度: {result['z_min_mm']:.2f} mm")
        return result

    def compare_measurements(self, normal_result, tilted_result):
        """对比测量结果，并调用新的汇总窗口"""
        if normal_result is None or tilted_result is None:
            print("测量结果不完整，无法对比");
            return

        normal_area_mm2 = normal_result.get('area_mm2')
        tilted_area_mm2 = tilted_result.get('area_mm2')

        print("\n" + "=" * 60 + "\n最终面积对比分析结果\n" + "=" * 60)
        print(f"正对状态: {normal_area_mm2:.2f} mm^2" if normal_area_mm2 else "N/A")
        print(f"倾斜状态 (已校正): {tilted_area_mm2:.2f} mm^2" if tilted_area_mm2 else "N/A")
        if normal_area_mm2 is not None and tilted_area_mm2 is not None and normal_area_mm2 > 0:
            area_diff_mm2 = tilted_area_mm2 - normal_area_mm2
            area_change_mm2 = (area_diff_mm2 / normal_area_mm2) * 100
            print(f"真实面积差异: {area_diff_mm2:+.2f} mm^2 ({area_change_mm2:+.2f}%)")
        print("=" * 60)

        print("\n正在生成最终结果汇总图...")
        show_final_comparison_image(normal_result, tilted_result)

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
            print("\n测试完成！请查看所有窗口，按任意键退出...")
            cv2.waitKey(0)
        finally:
            self.cleanup()

    def run_simple_comparison(self):
        """运行简单对比测试"""
        if not self.initialize_camera(): return
        try:
            input("\n1. 请将物体正对相机放置，然后按 Enter 键开始测量...")
            normal_result = self.capture_and_analyze("正对状态", apply_tilt_correction=True)
            if normal_result is None: return

            input("\n2. 请将物体倾斜放置，然后按 Enter 键开始测量...")
            tilted_result = self.capture_and_analyze("倾斜状态", apply_tilt_correction=True)
            if tilted_result is None: return

            self.compare_measurements(normal_result, tilted_result)
            print("\n测试完成！请查看所有窗口，按任意键退出...")
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
        global roi_pts, frame_for_roi;
        roi_pts = []
        cv2.namedWindow("Select ROI");
        cv2.setMouseCallback("Select ROI", select_roi_callback)
        print("请在 'Select ROI' 窗口中用鼠标拖拽出一个矩形区域。完成后，按 'c' 键确认，或按 'q' 键取消。")
        while True:
            img_depth = next(
                (f for f in self.cl.DeviceStreamRead(self.handle, 2000) if f.streamID == PERCIPIO_STREAM_DEPTH), None)
            if img_depth:
                depth_render_image = image_data();
                self.cl.DeviceStreamDepthRender(img_depth, depth_render_image)
                frame_for_roi = depth_render_image.as_nparray()
                if len(roi_pts) == 2: cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame_for_roi)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Select ROI"); return None
            elif key == ord('c'):
                if len(roi_pts) != 2: print("错误：请先选择一个有效的ROI区域再按 'c'。"); continue
                cv2.destroyWindow("Select ROI");
                return (roi_pts[0][0], roi_pts[0][1], roi_pts[1][0], roi_pts[1][1])

    def calculate_tilt_correction_matrix(self, roi_coords=None):
        if roi_coords is None: roi_coords = (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2)
        x1, y1, x2, y2 = roi_coords
        print(f"正在从ROI区域 ({x1}, {y1}) - ({x2}, {y2}) 计算校正矩阵...")
        for _ in range(3): self.cl.DeviceStreamRead(self.handle, 2000)
        img_depth = next(
            (f for f in self.cl.DeviceStreamRead(self.handle, 2000) if f.streamID == PERCIPIO_STREAM_DEPTH), None)
        if img_depth is None: print("无法获取深度图像数据"); return None
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data, self.scale_unit,
                                                   self.pointcloud_data_arr)
        p3d_nparray = self.pointcloud_data_arr.as_nparray()
        valid_points = p3d_nparray[y1:y2, x1:x2, :][p3d_nparray[y1:y2, x1:x2, 2] > 0]
        if len(valid_points) < 500: print("ROI内有效点数量不足"); return None
        rotation_matrix = calculate_and_visualize_plane(valid_points)
        if rotation_matrix is not None: self.rotation_matrix = rotation_matrix
        return rotation_matrix

    def setup_for_measurement(self, measurement_name):
        print(f"\n=== 设置 {measurement_name} 状态 ===")
        input("步骤1: 请在视野中放置参考平面，按 Enter 开始校正...")
        if not self.calibrate_plane_for_measurement(): print("平面校正失败"); return False
        input("步骤2: 请放置目标物体，按 Enter 开始选择ROI...")
        roi_coords = self.select_roi_interactive()
        if roi_coords is None: print("ROI选择被取消"); return False
        self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = roi_coords
        print(f"ROI已设置为: {roi_coords}");
        return True

    def calibrate_plane_for_measurement(self):
        print("请用鼠标在 'Select Plane ROI' 窗口中选择平整的参考平面区域")
        roi_coords = self.select_roi_interactive()  # 复用ROI选择函数
        if roi_coords is None: return False
        rotation_matrix = self.calculate_tilt_correction_matrix(roi_coords)
        if rotation_matrix is None: print("校正矩阵计算失败"); return False
        matrix_filename = f"tilt_correction_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
        np.save(matrix_filename, rotation_matrix)
        print(f"平面校正矩阵已保存到: {matrix_filename}");
        return True


def main():
    """主函数"""
    print("掩码面积对比工具 (最终版 - 支持中空物体)")
    print("=" * 50)
    print("1. 增强版 - 动态选择ROI和计算校正矩阵")
    print("2. 简单版 - 使用固定ROI和现有校正矩阵")
    print("3. 退出")
    while True:
        choice = input("\n请选择模式 (1/2/3): ").strip()
        if choice == '1':
            MaskAreaComparator().run_interactive_comparison(); break
        elif choice == '2':
            MaskAreaComparator().run_simple_comparison(); break
        elif choice == '3':
            print("退出程序"); break
        else:
            print("无效选择")


if __name__ == '__main__':
    main()
