# calibrate_hand_eye.py

import cv2
import numpy as np
import pcammls
from pcammls import *

# --- 可配置参数 ---
NUM_CALIB_POINTS = 4  # 需要采集的标定点数量

# --- 全局变量，用于存储鼠标点击坐标 ---
clicked_point = None

def mouse_callback(event, x, y, flags, param):
    """处理鼠标点击事件，记录点击的坐标。"""
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"已选择像素点: {clicked_point}")


def visualize_calibration_result(camera_pts, robot_pts, transform_matrix, img_shape):
    """
    可视化标定结果，对比相机点和映射后的机器人点。
    """
    vis_image = np.ones((img_shape[0], img_shape[1], 3), dtype=np.uint8) * 50

    # 逆变换矩阵，用于将机器人坐标映射回像素坐标以进行可视化
    try:
        inv_transform_matrix = cv2.invertAffineTransform(transform_matrix)
    except cv2.error:
        print("无法计算逆矩阵，跳过结果可视化。")
        return

    for i in range(len(camera_pts)):
        # 原始相机点 (蓝色圆点)
        cam_pt = tuple(map(int, camera_pts[i]))
        cv2.circle(vis_image, cam_pt, 8, (255, 100, 100), -1)
        cv2.putText(vis_image, f"C{i+1}", (cam_pt[0] + 10, cam_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

        # 将机器人坐标点通过逆矩阵映射回像素坐标 (红色叉)
        robot_pt_homogeneous = np.array([robot_pts[i][0], robot_pts[i][1], 1])
        cam_pt_estimated = inv_transform_matrix @ robot_pt_homogeneous
        cam_pt_est_int = tuple(map(int, cam_pt_estimated))

        cv2.drawMarker(vis_image, cam_pt_est_int, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(vis_image, f"R{i+1}", (cam_pt_est_int[0] + 10, cam_pt_est_int[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 连接对应的点对
        cv2.line(vis_image, cam_pt, cam_pt_est_int, (0, 255, 255), 1)

    cv2.imshow("Calibration Result", vis_image)
    print("\n在 'Calibration Result' 窗口中查看标定精度。按任意键退出。")
    cv2.waitKey(0)


def main():
    global clicked_point

    # --- 初始化相机 ---
    cl = PercipioSDK()
    dev_list = cl.ListDevice()
    if not dev_list:
        print("未找到设备")
        return
    handle = cl.Open(dev_list[0].id)
    if not cl.isValidHandle(handle):
        print("打开设备失败")
        return

    cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR)
    cl.DeviceStreamOn(handle)

    # --- 设置窗口和鼠标回调 ---
    cv2.namedWindow("Calibration View")
    cv2.setMouseCallback("Calibration View", mouse_callback)

    print("=" * 50)
    print("手眼标定程序 (按 'q' 退出)")
    print(f"请准备采集 {NUM_CALIB_POINTS} 个标定点。")
    print("=" * 50)

    robot_points = []  # 存储机械臂坐标 (X, Y)
    camera_points = []  # 存储相机像素坐标 (u, v)
    img_shape = None

    while len(robot_points) < NUM_CALIB_POINTS:
        print(f"\n--- 正在采集第 {len(robot_points) + 1} 个点 ---")

        try:
            robot_x = float(input("1. 移动机械臂到标定位置后，输入当前 X 坐标: "))
            robot_y = float(input("2. 输入当前 Y 坐标: "))
        except ValueError:
            print("输入无效，请输入数字。")
            continue

        # 重置上次点击的点
        clicked_point = None
        print("3. 在 'Calibration View' 窗口中用鼠标左键点击标记点。")
        print("4. 点击后，按 's' 键确认。")

        while True:
            image_list = cl.DeviceStreamRead(handle, 2000)
            if image_list:
                img_color_raw = image_list[0]
                if img_shape is None:
                    img_shape = (img_color_raw.height, img_color_raw.width)
                img_color_decoded = image_data()
                cl.DeviceStreamImageDecode(img_color_raw, img_color_decoded)
                color_frame = img_color_decoded.as_nparray()

                # 绘制已采集的点
                for i, p in enumerate(camera_points):
                    pt_tuple = tuple(map(int, p))
                    cv2.circle(color_frame, pt_tuple, 7, (255, 0, 0), -1)
                    cv2.putText(color_frame, f"P{i+1}", (pt_tuple[0] + 10, pt_tuple[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # 如果用户已点击，绘制一个临时的绿色准星
                if clicked_point:
                    cv2.drawMarker(color_frame, clicked_point, (0, 255, 0), cv2.MARKER_CROSS, 15, 2)

                cv2.imshow("Calibration View", color_frame)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('s'):
                if clicked_point:
                    break  # 确认成功，跳出循环
                else:
                    print("尚未点击选择像素点，无法确认！")
            elif key == ord('q'):
                print("用户中断操作。")
                cl.DeviceStreamOff(handle)
                cl.Close(handle)
                cv2.destroyAllWindows()
                return

        # 存储点对
        robot_points.append([robot_x, robot_y])
        camera_points.append(list(clicked_point))
        print(f"已记录点对: Robot{tuple(robot_points[-1])} <-> Camera{tuple(camera_points[-1])}")

    print("\n所有点采集完毕，正在计算变换矩阵...")
    robot_pts_np = np.array(robot_points, dtype=np.float32)
    camera_pts_np = np.array(camera_points, dtype=np.float32)

    M, _ = cv2.estimateAffine2D(camera_pts_np, robot_pts_np)

    if M is not None:
        print("变换矩阵计算成功:")
        print(M)
        np.save('hand_eye_transform.npy', M)
        print("\n变换矩阵已保存到 'hand_eye_transform.npy'")

        print("\n--- 验证变换效果 ---")
        for i in range(NUM_CALIB_POINTS):
            cam_pt = np.array([camera_pts_np[i][0], camera_pts_np[i][1], 1], dtype=np.float32)
            robot_coord_est = M @ cam_pt
            print(
                f"相机点 {camera_pts_np[i]} -> 机器人坐标 (实际): {robot_pts_np[i]}, (估算): [{robot_coord_est[0]:.2f}, {robot_coord_est[1]:.2f}]")

        # 调用可视化结果函数
        visualize_calibration_result(camera_pts_np, robot_pts_np, M, img_shape)

    else:
        print("计算变换矩阵失败！请检查采集的点是否共线。")

    cl.DeviceStreamOff(handle)
    cl.Close(handle)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()