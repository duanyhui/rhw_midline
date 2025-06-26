import cv2
import numpy as np
import pcammls
from pcammls import *
import os

# 全局变量用于存储绘制的点
line_points = []

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于通过点击添加/删除点来绘制中轴线"""
    global line_points

    if event == cv2.EVENT_LBUTTONDOWN:
        line_points.append((x, y))
        print(f"添加新点: ({x}, {y})。")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if line_points:
            line_points.pop()
            print("移除了最后一个点。")

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

def draw_centerline():
    """
    在深度图上通过多点标记的方式绘制理论中轴线并保存坐标。

    使用方法:
        - 运行脚本，会显示一个实时深度图预览窗口。
        - 按 'c' 键捕获当前帧，然后开始在捕获的图像上绘制。
        - 在绘图窗口中:
            - 左键点击: 添加一个点。点与点之间会自动连线。
            - 右键点击: 移除上一个添加的点。
            - 按 'r' 键: 清除所有已绘制的点，重新开始。
        - 绘制完成后，按 's' 键保存坐标。
        - 按 'q' 键退出。
    """
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

    # 3. 在捕获的图像上绘制中轴线
    cv2.namedWindow("Draw Theoretical Centerline")
    cv2.setMouseCallback("Draw Theoretical Centerline", mouse_callback)

    print("\n--- 开始绘制中轴线 ---")
    print("左键点击: 添加一个点")
    print("右键点击: 移除上一个点")
    print("按 'r' 键: 清除所有点 (Reset)")
    print("按 's' 键: 保存并退出")
    print("按 'q' 键: 不保存直接退出")

    vis_image = captured_image.copy()

    while True:
        # 创建一个副本用于实时显示，避免在原图上重复画线
        temp_vis_image = vis_image.copy()

        # 绘制各个顶点
        for point in line_points:
            cv2.circle(temp_vis_image, point, radius=4, color=(0, 0, 255), thickness=-1)

        # 绘制连接线
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
                dense_line_points = interpolate_polyline(line_points, step=2.0)
                points_to_save = np.array(dense_line_points)

                # 保存为 .npy 文件
                np.save('theoretical_centerline.npy', points_to_save)
                print(f"成功保存 {len(points_to_save)} 个理论中轴线坐标点到 theoretical_centerline.npy (已插值)")
            else:
                print("没有绘制任何点，未保存。")
            break
        elif key == ord('r'):
            line_points.clear()
            print("所有点都已清除。")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    draw_centerline()
