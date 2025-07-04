import numpy as np
import pcammls
from pcammls import *
import open3d as o3d
import cv2

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


    print("正在显示可视化窗口... 关闭窗口后程序将继续。")
    o3d.visualization.draw_geometries(geometries_to_draw)
    # --- 可视化结束 ---

    return rotation_matrix


def main():
    global roi_pts, frame_for_roi
    cl = PercipioSDK()
    dev_list = cl.ListDevice()
    if not dev_list:
        print('未找到设备')
        return

    handle = cl.Open(dev_list[0].id)
    if not cl.isValidHandle(handle):
        print(f'打开设备失败')
        return
    cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    cl.DeviceStreamOn(handle)

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi_callback)

    print("请在 'Select ROI' 窗口中用鼠标拖拽出一个矩形区域。")
    print("完成后，按 'c' 键确认并开始计算，或按 'q' 键退出。")

    p3d_nparray = None
    while True:
        image_list = cl.DeviceStreamRead(handle, 2000)
        img_depth = None
        for frame in image_list:
            if frame.streamID == PERCIPIO_STREAM_DEPTH:
                img_depth = frame
                break

        if img_depth:
            # 渲染深度图用于显示和ROI选择
            depth_render_image = image_data()
            cl.DeviceStreamDepthRender(img_depth, depth_render_image)
            frame_for_roi = depth_render_image.as_nparray()

            # 如果已选择ROI，在帧上画出矩形
            if len(roi_pts) == 2:
                cv2.rectangle(frame_for_roi, roi_pts[0], roi_pts[1], (0, 255, 0), 2)

            cv2.imshow("Select ROI", frame_for_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if len(roi_pts) != 2:
                print("错误：请先选择一个有效的ROI区域再按 'c'。")
                continue

            print("ROI 已确认，正在处理...")
            scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
            print('depth image scale unit :{}'.format(scale_unit))  # 0.125
            depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)

            # 将深度图转换为点云
            pointcloud_data = pointcloud_data_list()
            cl.DeviceStreamMapDepthImageToPoint3D(img_depth, depth_calib, scale_unit, pointcloud_data)
            p3d_nparray = pointcloud_data.as_nparray()
            break

    cv2.destroyAllWindows()

    if p3d_nparray is not None:
        # 从选择的像素坐标提取点云ROI
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        roi_p3d = p3d_nparray[y1:y2, x1:x2, :]
        valid_points = roi_p3d[roi_p3d[:, :, 2] > 0]

        print(f"已从ROI采集点云，共 {len(valid_points)} 个有效点。")
        print("正在计算校正矩阵并显示可视化结果...")

        rotation_matrix = calculate_and_visualize_plane(valid_points)

        if rotation_matrix is not None:
            np.save('tilt_correction_matrix.npy', rotation_matrix)
            print("\n标定成功！旋转矩阵已保存到 'tilt_correction_matrix.npy'")
        else:
            print("\n标定失败。")

    cl.DeviceStreamOff(handle)
    cl.Close(handle)


if __name__ == '__main__':
    main()