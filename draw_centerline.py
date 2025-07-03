import cv2
import numpy as np
import pcammls
from pcammls import *
import os

# 全局变量
line_points = []
drawing_mode = 'point'  # 'point' 或 'freehand'
drawing = False         # 在自由绘制模式下，当鼠标按住时为 True

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，处理逐点模式和自由绘制模式"""
    global line_points, drawing_mode, drawing

    if drawing_mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            line_points.append((x, y))
            print(f"逐点模式: 添加新点 ({x}, {y})。")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if line_points:
                line_points.pop()
                print("逐点模式: 移除了最后一个点。")

    elif drawing_mode == 'freehand':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            line_points.clear()  # 开始一条新线
            line_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                line_points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 在自由绘制模式下，右键单击清除当前线条
            if line_points:
                line_points.clear()
                print("自由绘制模式: 线条已清除。")

def interpolate_polyline(points, step=1.0):
    """
    通过在给定的顶点之间插入点来填充折线。

    :param points: 代表折线顶点的 (x, y) 元组列表。
    :param step: 插值点之间的距离（以像素为单位）。
    :return: 一个新的 (x, y) 元组列表，代表密集的折线。
    """
    if len(points) < 2:
        return points

    dense_points = []
    for i in range(len(points) - 1):
        p1 = np.array(points[i], dtype=float)
        p2 = np.array(points[i+1], dtype=float)

        # 添加段的起点
        dense_points.append(tuple(p1.astype(int)))

        distance = np.linalg.norm(p2 - p1)
        if distance > step:
            # 计算方向向量
            direction = (p2 - p1) / distance

            # 沿着线段以'step'为步长添加点
            current_dist = step
            while current_dist < distance:
                interpolated_point = p1 + current_dist * direction
                dense_points.append(tuple(interpolated_point.astype(int)))
                current_dist += step

    # 添加最后一个点
    if points:
        dense_points.append(points[-1])

    # 移除可能因四舍五入而产生的重复点
    seen = set()
    unique_points = []
    for p in dense_points:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)

    return unique_points


# --- 从 detect.py 移植过来的函数 ---

def extract_contours_adaptive(surface_mask, min_vertices=4, max_vertices=12, max_iterations=10,
                              initial_epsilon_factor=0.01):
    """
    自适应轮廓提取：
    通过迭代增加epsilon，将轮廓拟合到指定的顶点数范围内，以适应包含曲线和直线的形状。
    (移植自 detect.py)
    """
    contours, _ = cv2.findContours(surface_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    fitted_contours = []
    for cnt in contours[:2]:
        hull = cv2.convexHull(cnt)
        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            continue

        epsilon_factor = initial_epsilon_factor
        best_approx = hull
        for _ in range(max_iterations):
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(hull, epsilon, closed=True)
            if min_vertices <= len(approx) <= max_vertices:
                best_approx = approx
                break
            if len(approx) > max_vertices:
                epsilon_factor *= 1.5
                best_approx = approx
            else:
                break
        fitted_contours.append(best_approx)
    return fitted_contours


def extract_skeleton_universal(surface_mask: np.ndarray, visualize: bool = True):
    """
    通用的骨架提取函数，能同时处理实心和空心（有镂空）的物体。
    (移植自 detect.py)
    """
    if surface_mask is None or np.count_nonzero(surface_mask) == 0:
        return None

    contours = extract_contours_adaptive(surface_mask, min_vertices=4, max_vertices=16, initial_epsilon_factor=0.001)
    if not contours:
        return None

    target_mask = np.zeros_like(surface_mask, dtype=np.uint8)
    outer_cnt = contours[0]
    inner_cnt = None

    if len(contours) >= 2:
        inner_cnt = contours[1]
        cv2.drawContours(target_mask, [outer_cnt], -1, 255, cv2.FILLED)
        cv2.drawContours(target_mask, [inner_cnt], -1, 0, cv2.FILLED)
    else:
        cv2.drawContours(target_mask, [outer_cnt], -1, 255, cv2.FILLED)

    from skimage.morphology import skeletonize
    skeleton = skeletonize(target_mask > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    dilated_skeleton = cv2.dilate(skeleton_img, kernel, iterations=1)

    if visualize:
        vis_img = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_img, [outer_cnt], -1, (0, 0, 255), 2)
        if inner_cnt is not None:
            cv2.drawContours(vis_img, [inner_cnt], -1, (0, 255, 0), 2)
        vis_img[dilated_skeleton != 0] = (255, 0, 0)
        cv2.imshow("Universal Skeleton Extraction", vis_img)

    return cv2.cvtColor(dilated_skeleton, cv2.COLOR_GRAY2BGR)


def extract_skeleton_points(skeleton_image, origin_offset=(0, 0)):
    """
    从骨架图中提取所有离散点的坐标。
    (移植自 detect.py)
    """
    if skeleton_image is None or skeleton_image.size == 0:
        return np.array([])
    if skeleton_image.ndim == 3:
        gray_skeleton = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_skeleton = skeleton_image

    rows, cols = np.where(gray_skeleton > 0)
    if len(rows) == 0:
        return np.array([])

    local_points = np.vstack((cols, rows)).T
    global_points = local_points + np.array(origin_offset)
    return global_points


def extract_nearest_surface_mask(roi_p3d_aligned, depth_margin):
    """
    从 ROI 点云中提取最浅表面的 mask。
    (移植自 detect.py)
    """
    z_img = roi_p3d_aligned[:, :, 2].copy()
    z_img[z_img <= 0] = 0
    valid_mask = z_img > 0
    if not np.any(valid_mask):
        return None
    z_min_val = np.min(z_img[valid_mask])
    lower = z_min_val
    upper = z_min_val + depth_margin
    surface_mask = ((z_img >= lower) & (z_img <= upper)).astype(np.uint8) * 255
    return surface_mask

# --- 自动提取模式 ---
roi_points = []
drawing_roi = False

def roi_selection_callback(event, x, y, flags, param):
    """用于在自动模式下选择ROI的鼠标回调"""
    global roi_points, drawing_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing_roi = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_roi:
            # 实时显示矩形
            img_copy = param.copy()
            cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing_roi = False
        # 确保 x1 < x2, y1 < y2
        x1, y1 = roi_points[0]
        x2, y2 = roi_points[1]
        roi_points = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]
        cv2.rectangle(param, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI", param)

def auto_extract_centerline(cl, handle):
    """
    自动提取模式的主函数。
    """
    print("\n--- 自动提取模式 ---")
    print("1. 将在预览窗口显示实时深度图。")
    print("2. 按 'c' 键捕获当前帧。")
    print("3. 在捕获的图像上用鼠标拖拽一个矩形区域作为ROI。")
    print("4. 程序将自动提取ROI内的中轴线并显示预览。")
    print("5. 按 's' 保存, 'r' 重新选择ROI, 'q' 退出。")

    # 获取必要的相机参数
    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
    depth_calib_data = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    pointcloud_data_arr = pointcloud_data_list()
    img_render = image_data()

    captured_frame = None
    p3d_nparray = None

    # 步骤1 & 2: 捕获
    while True:
        image_list = cl.DeviceStreamRead(handle, 2000)
        if not image_list:
            cv2.waitKey(10)
            continue

        depth_frame = None
        for frame in image_list:
            if frame.streamID == PERCIPIO_STREAM_DEPTH:
                depth_frame = frame
                cl.DeviceStreamDepthRender(frame, img_render)
                live_view = img_render.as_nparray().copy()
                cv2.imshow("Live View - Press 'c' to capture", live_view)
                break

        key = cv2.waitKey(20)
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
        elif key == ord('c') and depth_frame:
            captured_frame = live_view
            cl.DeviceStreamMapDepthImageToPoint3D(depth_frame, depth_calib_data, scale_unit, pointcloud_data_arr)
            p3d_nparray = pointcloud_data_arr.as_nparray()
            cv2.destroyWindow("Live View - Press 'c' to capture")
            print("图像已捕获。请选择ROI。")
            break

    if captured_frame is None or p3d_nparray is None:
        print("未能捕获图像或点云。")
        return

    # 步骤3 & 4 & 5: ROI选择和提取
    global roi_points
    while True:
        roi_points = []
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", roi_selection_callback, captured_frame)
        cv2.imshow("Select ROI", captured_frame)
        print("请在 'Select ROI' 窗口中拖拽出一个矩形区域, 然后按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyWindow("Select ROI")

        if len(roi_points) != 2:
            print("未选择有效的ROI，退出。")
            return

        (x1, y1), (x2, y2) = roi_points
        print(f"选择的ROI: ({x1}, {y1}) to ({x2}, {y2})")

        # 提取点云ROI
        roi_cloud = p3d_nparray[y1:y2, x1:x2]

        # 提取表面
        surface_mask = extract_nearest_surface_mask(roi_cloud, depth_margin=2.5)
        if surface_mask is None:
            print("在ROI内未找到有效表面，请重试。")
            continue

        # 提取骨架
        skeleton_vis = extract_skeleton_universal(surface_mask, visualize=True)
        if skeleton_vis is None:
            print("无法提取骨架，请重试。")
            continue

        # 提取点
        actual_points = extract_skeleton_points(skeleton_vis, origin_offset=(x1, y1))
        print(f"提取到 {len(actual_points)} 个点。")

        # 可视化最终结果
        final_vis = captured_frame.copy()
        for point in actual_points:
            cv2.circle(final_vis, tuple(point.astype(int)), radius=2, color=(0, 255, 255), thickness=-1)
        cv2.rectangle(final_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Extraction Result", final_vis)

        print("按 's' 保存, 'r' 重新选择ROI, 'q' 退出。")
        key = cv2.waitKey(0)
        if key == ord('s'):
            if len(actual_points) > 0:
                np.save('theoretical_centerline.npy', actual_points)
                print(f"成功保存 {len(actual_points)} 个理论中轴线坐标点到 theoretical_centerline.npy")
            else:
                print("没有提取到任何点，未保存。")
            break
        elif key == ord('r'):
            cv2.destroyWindow("Universal Skeleton Extraction")
            cv2.destroyWindow("Extraction Result")
            continue
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


def draw_centerline():
    """
    在深度图上通过多点标记或自由绘制的方式绘制理论中轴线并保存坐标。

    使用方法:
        - 运行脚本，会显示一个实时深度图预览窗口。
        - 按 'c' 键捕获当前帧，然后开始在捕获的图像上绘制。
        - 在绘图窗口中:
            - 按 'm' 键切换“逐点模式”和“自由绘制模式”。
            - 逐点模式:
                - 左键点击: 添加一个点。
                - 右键点击: 移除上一个添加的点。
            - 自由绘制模式:
                - 按住左键并拖动: 绘制一条线。
                - 右键点击: 清除当前绘制的线。
            - 按 'r' 键: 清除所有已绘制的点，重新开始。
        - 绘制完成后，按 's' 键保存坐标。
        - 按 'q' 键退出。
    """
    # 0. 选择模式
    print("请选择操作模式:")
    print("1: 手动相机模式 (在捕获的深度图上手动绘制)")
    print("2: 手动模板模式 (在空白画布上手动绘制)")
    print("3: 自动提取模式 (从捕获的图像中自动提取中轴线)")
    mode_choice = input("请输入选项 (1, 2, 或 3): ")

    # --- 自动提取模式 ---
    if mode_choice == '3':
        cl = PercipioSDK()
        dev_list = cl.ListDevice()
        if not dev_list:
            print('未找到设备')
            return
        handle = cl.Open(dev_list[0].id)
        if not cl.isValidHandle(handle):
            print('打开设备失败')
            return
        err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_DEPTH)
        if err:
            print('启用深度数据流失败: {}'.format(err))
            cl.Close(handle)
            return
        cl.DeviceStreamOn(handle)

        auto_extract_centerline(cl, handle)

        cl.DeviceStreamOff(handle)
        cl.Close(handle)
        return

    captured_image = None
    # --- 手动相机模式 ---
    if mode_choice == '1':
        # 1. 初始化SDK和设备
        cl = PercipioSDK()
        dev_list = cl.ListDevice()
        if not dev_list:
            print('未找到设备')
            return

        handle = cl.Open(dev_list[0].id)
        if not cl.isValidHandle(handle):
            print('打开设备失败')
            return

        # 3. 启用彩色和深度数据流
        err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
        if err:
            print('启用数据流失败: {}'.format(err))
            cl.Close(handle)
            exit()

        # 4. 启动数据流
        cl.DeviceStreamOn(handle)

        depth_render = image_data()

        # 2. 捕获一张深度图用于标注
        print("正在捕获深度图像，请将物体放置在相机视野内...")
        print("按 'c' 键捕获当前帧并开始绘制，按 'q' 键退出。")

        captured_image = None

        while True:
            image_list = cl.DeviceStreamRead(handle, 2000)
            if not image_list:
                cv2.waitKey(10)
                continue

            for frame in image_list:
                if frame.streamID == PERCIPIO_STREAM_DEPTH:
                    cl.DeviceStreamDepthRender(frame, depth_render)
                    live_view = depth_render.as_nparray().copy()
                    cv2.imshow("Live Depth View - Press 'c' to capture, 'q' to quit", live_view)
                    break

            key = cv2.waitKey(20)
            if key == ord('q'):
                cl.DeviceStreamOff(handle)
                cl.Close(handle)
                cv2.destroyAllWindows()
                return
            elif key == ord('c'):
                captured_image = live_view
                cv2.destroyWindow("Live Depth View - Press 'c' to capture, 'q' to quit")
                print("图像已捕获，请开始绘制。")
                break

        if captured_image is None:
            print("未能捕获图像。")
            cl.DeviceStreamOff(handle)
            cl.Close(handle)
            return

        cl.DeviceStreamOff(handle)
        cl.Close(handle)

    # --- 手动模板模式 ---
    elif mode_choice == '2':
        # --- 模板模式 ---
        print("已进入模板模式。将在空白画布上绘制。")
        # 尺寸可以根据需要调整，这里使用一个常见的深度图尺寸
        captured_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    else:
        print("无效的选项。退出程序。")
        return

    if captured_image is None:
        print("未能获取用于绘制的图像。")
        return

    # 3. 在捕获的图像上绘制中轴线
    cv2.namedWindow("Draw Theoretical Centerline")
    cv2.setMouseCallback("Draw Theoretical Centerline", mouse_callback)

    print("\n--- 绘制模式 ---")
    print("按 'm' 键在“逐点模式”和“自由绘制模式”之间切换。")
    print("\n--- 逐点模式 ---")
    print("左键点击: 添加一个点。")
    print("右键点击: 移除上一个点。")
    print("\n--- 自由绘制模式 ---")
    print("按住左键并拖动: 绘制一条线。")
    print("右键点击: 清除当前线条。")
    print("\n--- 通用操作 ---")
    print("按 'r' 键: 重置所有点。")
    print("按 's' 键: 保存并退出。")
    print("按 'q' 键: 不保存直接退出。")

    vis_image = captured_image.copy()

    global drawing_mode, line_points

    while True:
        # 创建一个副本用于实时显示，避免在原图上重复画线
        temp_vis_image = vis_image.copy()

        # 显示当前模式
        mode_text = f"Mode: {'Point' if drawing_mode == 'point' else 'Freehand'}"
        cv2.putText(temp_vis_image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # 在逐点模式下，绘制各个顶点
        if drawing_mode == 'point':
            for point in line_points:
                cv2.circle(temp_vis_image, point, radius=4, color=(0, 0, 255), thickness=-1)

        # 绘制连接线（对两种模式都适用）
        if len(line_points) > 1:
            pts = np.array(line_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_vis_image, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow("Draw Theoretical Centerline", temp_vis_image)

        key = cv2.waitKey(20)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if line_points:
                # 插值以创建密集的点集
                dense_line_points = interpolate_polyline(line_points, step=1.0)
                points_to_save = np.array(dense_line_points)

                # 保存为 .npy 文件
                np.save('theoretical_centerline.npy', points_to_save)
                print(f"成功保存 {len(points_to_save)} 个理论中轴线坐标点到 theoretical_centerline.npy (折线)")
            else:
                print("没有绘制任何点，未保存。")
            break
        elif key == ord('r'):
            line_points.clear()
            print("所有点都已清除。")
        elif key == ord('m'):
            if drawing_mode == 'point':
                drawing_mode = 'freehand'
            else:
                drawing_mode = 'point'
            line_points.clear() # 切换模式时清除点
            print(f"已切换到 {drawing_mode.capitalize()} 模式。点已清除。")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    draw_centerline()
