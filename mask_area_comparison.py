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

    # 使用内点的凸包来可视化检测到的平面区域
    try:
        # 计算内点云的凸包，它会生成一个紧密包裹这些点的网格
        hull, _ = inlier_cloud.compute_convex_hull()
        # 为了更清晰地显示，我们只画出凸包的红色边框
        hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color([1, 0, 0])  # 红色边框
        geometries_to_draw = [inlier_cloud, outlier_cloud, hull_ls]
    except Exception as e:
        print(f"警告：凸包计算失败 ({e})，仅显示点云。")
        geometries_to_draw = [inlier_cloud, outlier_cloud]

    print("正在显示可视化窗口... 关闭窗口后程序将继���。")
    o3d.visualization.draw_geometries(geometries_to_draw)
    # --- 可视化结束 ---

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
    从 ROI 点云中提取最浅表面的 mask（基于 Z 值层提取）

    参数:
        roi_p3d_aligned: np.ndarray, 形状 (H, W, 3)，点云数据
        depth_margin: 容差范围（单位: mm），用于构造深度层厚度

    返回:
        surface_mask: np.ndarray, 形状 (H, W)，uint8 类型二值掩码，255表示最浅层区域
        z_min_val: 最小深度值（距离最近的 mm 值）
    """

    # 1. 提取 Z 值图
    z_img = roi_p3d_aligned[:, :, 2].copy()
    z_img[z_img <= 0] = 0  # 过滤无效值

    valid_mask = z_img > 0
    if not np.any(valid_mask):
        print("无有效深度值")
        return None, None

    z_min_val = np.min(z_img[valid_mask])
    print("\t最小深度值:{}mm".format(z_min_val))

    # 2. 创建 mask，提取 z_min 附近的一层
    lower = z_min_val
    upper = z_min_val + depth_margin
    surface_mask = ((z_img >= lower) & (z_img <= upper)).astype(np.uint8) * 255  # 二值掩码

    return surface_mask, z_min_val


def calculate_mask_area(surface_mask):
    """
    计算掩码面积（非零像素的数量）

    参数:
        surface_mask: np.ndarray, 二值掩码

    返回:
        area: int, 掩码面积（像素数）
    """
    if surface_mask is None:
        return 0
    return np.count_nonzero(surface_mask)


def visualize_mask_with_info(surface_mask, title, area, z_min):
    """
    可视化掩码并显示相关信息

    参数:
        surface_mask: 二值掩码
        title: 窗口标题
        area: 掩码面积
        z_min: 最小深度值
    """
    if surface_mask is None:
        return

    # 创��彩色图像用于显示
    vis_img = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)

    # 在图像上添加文本信息
    cv2.putText(vis_img, f"Area: {area} pixels", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis_img, f"Z_min: {z_min:.2f} mm", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(title, vis_img)


class MaskAreaComparator:
    """掩码面积对比器"""

    def __init__(self, roi_coords=None):
        """
        初始化

        参数:
            roi_coords: tuple, (x1, y1, x2, y2) ROI坐标，如果为None则使用默认值
        """
        # 默认ROI坐标
        if roi_coords is None:
            self.ROI_X1, self.ROI_Y1 = 707, 600
            self.ROI_X2, self.ROI_Y2 = 927, 804
        else:
            self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = roi_coords

        self.measurements = []
        self.depth_margin = 4.0  # 默认深度容差

        # 初始化相机
        self.cl = None
        self.handle = None
        self.event = None
        self.depth_calib_data = None
        self.scale_unit = None
        self.pointcloud_data_arr = None

        # 加载倾斜校正矩阵
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
            print('没有找到设备')
            return False

        print('找到设备:')
        for idx in range(len(dev_list)):
            dev = dev_list[idx]
            print('{} -- {} \t {}'.format(idx, dev.id, dev.iface.id))

        # 默认选择第一个设备
        selected_idx = 0
        sn = dev_list[selected_idx].id

        self.handle = self.cl.Open(sn)
        if not self.cl.isValidHandle(self.handle):
            err = self.cl.TYGetLastErrorCodedescription()
            print('设备打开失败: ', err)
            return False

        self.event = PythonPercipioDeviceEvent()
        self.cl.DeviceRegiststerCallBackEvent(self.event)

        # 配置深度流
        depth_fmt_list = self.cl.DeviceStreamFormatDump(self.handle, PERCIPIO_STREAM_DEPTH)
        if len(depth_fmt_list) == 0:
            print('设备没有深度流')
            return False
        self.cl.DeviceStreamFormatConfig(self.handle, PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])

        # 加载默认参数
        err = self.cl.DeviceLoadDefaultParameters(self.handle)
        if err:
            print('加载默认参数失败: ', self.cl.TYGetLastErrorCodedescription())

        # 读取校准数据
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(self.handle)
        self.depth_calib_data = self.cl.DeviceReadCalibData(self.handle, PERCIPIO_STREAM_DEPTH)

        # 启用深度流
        err = self.cl.DeviceStreamEnable(self.handle, PERCIPIO_STREAM_DEPTH)
        if err:
            print('启用深度流失败: {}'.format(err))
            return False

        self.cl.DeviceStreamOn(self.handle)

        # 初始化点云数据
        self.pointcloud_data_arr = pointcloud_data_list()

        print("相机初始化成功")
        return True

    def capture_and_analyze(self, measurement_name, apply_tilt_correction=False):
        """
        捕获当前帧并分析掩码面积

        参数:
            measurement_name: str, 测量名称（如 "正对", "倾斜"）
            apply_tilt_correction: bool, 是否应用倾斜校正

        返回:
            dict: 包含测量结果的字典
        """
        if self.handle is None:
            print("相机未初始化")
            return None

        print(f"\n开始捕获 {measurement_name} 状态的数据...")

        # 等待几帧稳定
        for _ in range(5):
            image_list = self.cl.DeviceStreamRead(self.handle, 2000)
            if len(image_list) == 0:
                continue

        # 捕获
        image_list = self.cl.DeviceStreamRead(self.handle, 2000)
        if len(image_list) == 0:
            print("无法获取图像数据")
            return None

        img_depth = image_list[0]  # 深度图

        # 生成点云
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data,
                                                   self.scale_unit, self.pointcloud_data_arr)
        p3d_nparray = self.pointcloud_data_arr.as_nparray()

        # 应用倾斜校正（如果需要）
        if apply_tilt_correction and self.rotation_matrix is not None:
            original_shape = p3d_nparray.shape
            points = p3d_nparray.reshape(-1, 3)
            valid_points_mask = points[:, 2] > 0
            points[valid_points_mask] = points[valid_points_mask] @ self.rotation_matrix.T
            p3d_nparray = points.reshape(original_shape)
            print(f"  已对 {measurement_name} 状态应用倾斜校正")

        # 提取ROI点云
        roi_cloud = p3d_nparray[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]

        # 提取表面掩码
        surface_mask, z_min = extract_nearest_surface_mask(roi_cloud, self.depth_margin)

        if surface_mask is None:
            print(f"  {measurement_name} 状态: 无法提取有效的表面掩码")
            return None

        # 计算面积
        area = calculate_mask_area(surface_mask)

        # 创建测量结果
        result = {
            'name': measurement_name,
            'timestamp': datetime.now().isoformat(),
            'area_pixels': area,
            'z_min_mm': float(z_min),
            'depth_margin_mm': self.depth_margin,
            'roi_coords': (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2),
            'tilt_corrected': apply_tilt_correction
        }

        print(f"  {measurement_name} 状态分析完成:")
        print(f"    掩码面积: {area} 像素")
        print(f"    最小深度: {z_min:.2f} mm")

        # 可视化
        visualize_mask_with_info(surface_mask, f"{measurement_name} - Surface Mask", area, z_min)

        return result

    def compare_measurements(self, normal_result, tilted_result):
        """
        对比正对和倾斜状态的测量结果

        参数:
            normal_result: dict, 正对状态的测量结果
            tilted_result: dict, 倾斜状态的测量结果
        """
        if normal_result is None or tilted_result is None:
            print("测量结果不完整，无法进行对比")
            return

        normal_area = normal_result['area_pixels']
        tilted_area = tilted_result['area_pixels']
        normal_z_min = normal_result['z_min_mm']
        tilted_z_min = tilted_result['z_min_mm']

        area_diff = tilted_area - normal_area
        area_change_percent = (area_diff / normal_area) * 100 if normal_area > 0 else 0
        z_min_diff = tilted_z_min - normal_z_min

        print("\n" + "="*60)
        print("掩码面积对比分析结果")
        print("="*60)
        print(f"正对状态:")
        print(f"  掩码面积: {normal_area} 像素")
        print(f"  最小深度: {normal_z_min:.2f} mm")
        print(f"倾斜状态:")
        print(f"  掩码面积: {tilted_area} 像素")
        print(f"  最小深度: {tilted_z_min:.2f} mm")
        print(f"差异分析:")
        print(f"  面积差异: {area_diff} 像素 ({area_change_percent:+.2f}%)")
        print(f"  深度差异: {z_min_diff:+.2f} mm")
        print("="*60)

        # 保存对比结果
        comparison_result = {
            'comparison_timestamp': datetime.now().isoformat(),
            'normal_measurement': normal_result,
            'tilted_measurement': tilted_result,
            'area_difference_pixels': int(area_diff),
            'area_change_percent': float(area_change_percent),
            'z_min_difference_mm': float(z_min_diff)
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
        """运行交互式对比测试（增强版：支持动态ROI选择和校正矩阵计算）"""
        if not self.initialize_camera():
            return

        try:
            print("\n掩码面积对比测试（增强版）")
            print("="*50)
            print("此版本支持动态ROI选择和倾斜校正矩阵计算")
            print("="*50)

            # === 测量正对状态 ===
            print("\n【第一阶段：正对状态测量】")
            input("请将物体正对相机放置，然后按 Enter 键开始设置...")

            # 设置正对状态的ROI和参数
            if not self.setup_for_measurement("正对状态"):
                print("正对状态设置失败，退出测试")
                return

            input("ROI设置完成。按 Enter 键开始正对状态测量...")
            normal_result = self.capture_and_analyze("正对状态", apply_tilt_correction=True)

            if normal_result is None:
                print("正对状态测量失败，退出测试")
                return

            # === 测量倾斜状态 ===
            print("\n【第二阶段：倾斜状态测量】")
            input("请将物体倾斜放置，然后按 Enter 键开始设置...")

            # 设置倾斜状态的ROI和校正矩阵
            if not self.setup_for_measurement("倾斜状态"):
                print("倾斜状态设���失败，但仍可查看正对状态结果")
                return

            input("倾斜状态设置完成。按 Enter 键开始倾斜状态测量...")
            tilted_result = self.capture_and_analyze("倾斜状态", apply_tilt_correction=True)

            if tilted_result is None:
                print("倾斜状态测量失败，但仍可查看正对状态结果")
                return

            # === 对比分析结果 ===
            print("\n【第三阶段：结果对比分析】")
            self.compare_measurements(normal_result, tilted_result)

            # 显示详细的ROI和校正信息
            print("\n" + "="*60)
            print("测量配置总结")
            print("="*60)
            print(f"正对状态ROI: {normal_result['roi_coords']}")
            print(f"倾斜状态ROI: {tilted_result['roi_coords']}")
            print(f"深度容差: {self.depth_margin} mm")
            print(f"倾斜校正: {'已应用' if self.rotation_matrix is not None else '未应用'}")
            print("="*60)

            print("\n测试完成！按任意键关闭所有窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\n用户中断测试")
        finally:
            self.cleanup()

    def run_simple_comparison(self):
        """运行简单对比测试（使用固定ROI和现有校正矩阵）"""
        if not self.initialize_camera():
            return

        try:
            print("\n掩码面积对比测试（简单模式）")
            print("="*40)
            print("使用固定ROI和现有校正矩阵进行快速测试")

            # 测量正对状态
            input("\n1. 请将物体正对相机放置，然后按 Enter 键开始测量...")
            normal_result = self.capture_and_analyze("正对状态", apply_tilt_correction=False)

            if normal_result is None:
                print("正对状态测量失败，退出测试")
                return

            # 测量倾斜状态
            input("\n2. 请将物体倾斜放置，然后按 Enter 键开始测量...")
            tilted_result = self.capture_and_analyze("倾斜状态", apply_tilt_correction=True)

            if tilted_result is None:
                print("倾斜状态测量失败，但仍可查看正对状态结果")
                return

            # 对比结果
            self.compare_measurements(normal_result, tilted_result)

            print("\n测试完成！按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\n用户中断测试")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.handle is not None:
            self.cl.DeviceStreamOff(self.handle)
            self.cl.Close(self.handle)
        cv2.destroyAllWindows()

    def select_roi_interactive(self):
        """
        交互式ROI选择功能

        返回:
            tuple: (x1, y1, x2, y2) ROI坐标，如果失败则返回None
        """
        global roi_pts, frame_for_roi
        roi_pts = []  # 重置全局变量

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", select_roi_callback)

        print("请在 'Select ROI' 窗口中用鼠标拖��出一个矩形区��。")
        print("完成后，按 'c' 键确认，或按 'q' 键取消。")

        while True:
            image_list = self.cl.DeviceStreamRead(self.handle, 2000)
            img_depth = None
            for frame in image_list:
                if frame.streamID == PERCIPIO_STREAM_DEPTH:
                    img_depth = frame
                    break

            if img_depth:
                # 渲染深度图用于显示和ROI选择
                depth_render_image = image_data()
                self.cl.DeviceStreamDepthRender(img_depth, depth_render_image)
                frame_for_roi = depth_render_image.as_nparray()

                # 如果已选择ROI，在帧上画出矩形
                if len(roi_pts) == 2:
                    cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)

                cv2.imshow("Select ROI", frame_for_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Select ROI")
                return None
            elif key == ord('c'):
                if len(roi_pts) != 2:
                    print("错误：请先选择一个有效的ROI区域再按 'c'。")
                    continue

                x1, y1 = roi_pts[0]
                x2, y2 = roi_pts[1]
                cv2.destroyWindow("Select ROI")
                return (x1, y1, x2, y2)

    def calculate_tilt_correction_matrix(self, roi_coords=None):
        """
        基于选定的ROI区域计算倾斜校正矩阵

        参数:
            roi_coords: tuple, (x1, y1, x2, y2) ROI坐标，如果为None则使用当前设置的ROI

        返回:
            np.ndarray: 3x3倾斜校正旋转矩阵，如果失败则返回None
        """
        if roi_coords is None:
            roi_coords = (self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2)

        x1, y1, x2, y2 = roi_coords

        print(f"���在从ROI区域 ({x1}, {y1}) - ({x2}, {y2}) 计算校正矩阵...")

        # 等待几帧稳定
        for _ in range(3):
            image_list = self.cl.DeviceStreamRead(self.handle, 2000)

        # 捕获用于校准的���
        image_list = self.cl.DeviceStreamRead(self.handle, 2000)
        img_depth = None
        for frame in image_list:
            if frame.streamID == PERCIPIO_STREAM_DEPTH:
                img_depth = frame
                break

        if img_depth is None:
            print("无法获取深度图像数据")
            return None

        # 将深度图转换为点云
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data,
                                                   self.scale_unit, self.pointcloud_data_arr)
        p3d_nparray = self.pointcloud_data_arr.as_nparray()

        # 从选择的像素坐标提取点云ROI
        roi_p3d = p3d_nparray[y1:y2, x1:x2, :]
        valid_points = roi_p3d[roi_p3d[:, :, 2] > 0]

        print(f"已从ROI采集点云，共 {len(valid_points)} 个有效点。")

        if len(valid_points) < 500:
            print("ROI内有效点数量不足，无法进行平面拟合")
            return None

        print("正在计算校正矩阵并显示可视化结果...")
        rotation_matrix = calculate_and_visualize_plane(valid_points)

        if rotation_matrix is not None:
            self.rotation_matrix = rotation_matrix
            print("��斜校正矩阵计算成功并已更新到实例中")

        return rotation_matrix

    def setup_for_measurement(self, measurement_name):
        """
        为测量设置平面校正和ROI
        新流程：先校正平面，再选择ROI

        参数:
            measurement_name: str, 测量名称

        返回:
            bool: 设置是否成功
        """
        print(f"\n=== 设置 {measurement_name} 状态 ===")

        # 步骤1: 计算平面校正矩阵（每次测量都重新计算）
        print("步骤1: 校正倾斜平面")
        print("请在设备视野中放置一个平整的参考平面（如桌面、平板等）")
        input("准备好后按 Enter 键开始平面校正...")

        if not self.calibrate_plane_for_measurement():
            print("平面校正失败")
            return False

        # 步骤2: 选择ROI区域（在校正后的数据上选择）
        print("步骤2: 选择测量ROI区域")
        print("现在请将目标物体放置在合适位置")
        input("准备好后按 Enter 键开始选择ROI区域...")

        roi_coords = self.select_roi_interactive()
        if roi_coords is None:
            print("ROI选择被取消")
            return False

        # 更新ROI坐标
        self.ROI_X1, self.ROI_Y1, self.ROI_X2, self.ROI_Y2 = roi_coords
        print(f"ROI已设置为: ({self.ROI_X1}, {self.ROI_Y1}) - ({self.ROI_X2}, {self.ROI_Y2})")

        return True

    def calibrate_plane_for_measurement(self):
        """
        为当前测量状态校正平面

        返回:
            bool: 校正是否成功
        """
        print("正在进行平面校正...")
        print("请用鼠标在 'Select Plane ROI' 窗口中选择平整的参考平面区域")

        global roi_pts, frame_for_roi
        roi_pts = []  # 重置全局变量

        cv2.namedWindow("Select Plane ROI")
        cv2.setMouseCallback("Select Plane ROI", select_roi_callback)

        print("请用鼠标拖拽选择平面区域，完成后按 'c' 键确认，或按 'q' 键取消。")

        plane_roi_coords = None
        while True:
            image_list = self.cl.DeviceStreamRead(self.handle, 2000)
            img_depth = None
            for frame in image_list:
                if frame.streamID == PERCIPIO_STREAM_DEPTH:
                    img_depth = frame
                    break

            if img_depth:
                # 渲染深度图用于显示和ROI选择
                depth_render_image = image_data()
                self.cl.DeviceStreamDepthRender(img_depth, depth_render_image)
                frame_for_roi = depth_render_image.as_nparray()

                # 如果已选择ROI，在帧上画出矩形
                if len(roi_pts) == 2:
                    cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)

                cv2.imshow("Select Plane ROI", frame_for_roi)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Select Plane ROI")
                return False
            elif key == ord('c'):
                if len(roi_pts) != 2:
                    print("错误：请先选择一个有效的平面区域再按 'c'。")
                    continue

                x1, y1 = roi_pts[0]
                x2, y2 = roi_pts[1]
                plane_roi_coords = (x1, y1, x2, y2)
                cv2.destroyWindow("Select Plane ROI")
                break

        if plane_roi_coords is None:
            return False

        # 基于选定的平面ROI计算校正矩阵
        rotation_matrix = self.calculate_tilt_correction_matrix(plane_roi_coords)

        if rotation_matrix is None:
            print("校正矩阵计算失败")
            return False

        # 保存当前的校正矩阵
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        matrix_filename = f"tilt_correction_matrix_{timestamp}.npy"
        np.save(matrix_filename, rotation_matrix)
        print(f"平面校正矩阵已保存到: {matrix_filename}")

        return True

def main():
    """主函数"""
    print("掩码面积对比工具")
    print("="*40)
    print("1. 增强版 - 支持动态ROI选择和倾斜校正矩阵计算")
    print("2. 简单版 - 使用固定ROI和现有校正矩阵")
    print("3. 退出")

    while True:
        try:
            choice = input("\n请选择模式 (1/2/3): ").strip()

            if choice == '1':
                print("\n启动增强版模式...")
                comparator = MaskAreaComparator()
                comparator.run_interactive_comparison()
                break

            elif choice == '2':
                print("\n启动简单版模式...")
                comparator = MaskAreaComparator()
                comparator.run_simple_comparison()
                break

            elif choice == '3':
                print("退出程序")
                break

            else:
                print("无效选择，请输入 1、2 或 3")

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break

if __name__ == '__main__':
    main()
