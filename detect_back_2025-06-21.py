import numpy as np
import pcammls
from pcammls import *
import cv2
import open3d
from skimage.morphology import skeletonize
from scipy.ndimage import label


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


def extract_outer_inner_contours(surface_mask, simplify_epsilon=4):
    """
    从表面 mask 中提取外轮廓和内轮廓，并进行多边形拟合

    参数:
        surface_mask: np.ndarray, 二值图像 (H, W)，255 表示目标区域
        simplify_epsilon: 拟合精度，单位像素

    返回:
        fitted_contours: list of np.ndarray, 拟合后的轮廓 [outer, inner]
    """
    contours, _ = cv2.findContours(surface_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print("总共找到轮廓数量：", len(contours))

    if len(contours) < 1:
        return [], []

    # 按面积排序（大到小），通常最大的是外轮廓，第二大是内轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    fitted_contours = []
    for i in range(min(2, len(contours))):  # 只处理前两个
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, simplify_epsilon, True)  # 多边形拟合
        fitted_contours.append(approx)

    return fitted_contours


def extract_outer_inner_contours_002(surface_mask, simplify_epsilon=0.02):
    """
    改进版：用凸包+动态epsilon强制拟合四边形轮廓
    """
    contours, hierarchy = cv2.findContours(surface_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"找到轮廓数量：{len(contours)}，层级结构：{hierarchy}")

    fitted_contours = []
    if len(contours) == 0:
        return fitted_contours

    # 按面积从大到小排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i in range(min(2, len(contours))):  # 只处理前两个最大轮廓
        cnt = contours[i]

        # 步骤1：计算凸包（消除凹陷）
        hull = cv2.convexHull(cnt)

        # 步骤2：动态计算epsilon（基于周长比例）
        perimeter = cv2.arcLength(hull, True)
        epsilon = simplify_epsilon * perimeter  # 关键参数，可调整

        # 步骤3：多边形近似
        approx = cv2.approxPolyDP(hull, epsilon, closed=True)

        # 步骤4：强制四边形（如果顶点超过4，重新拟合）
        if len(approx) > 4:
            # 如果顶点过多，适当增大epsilon
            epsilon *= 1.5
            approx = cv2.approxPolyDP(hull, epsilon, closed=True)

        print(f"轮廓{i}拟合后顶点数：{len(approx)}")
        fitted_contours.append(approx)

    return fitted_contours


def fit_plane_and_extract_height_map(roi_p3d: np.ndarray,
                                     distance_threshold,
                                     ransac_n: int = 65,
                                     num_iterations: int = 1000,
                                     visualize: bool = True):
    """
    对点云 ROI 使用 RANSAC 拟合平面，提取平面区域高度图并可视化

    参数:
        roi_p3d: (H, W, 3) 的点云数据（单位 mm）
        distance_threshold: RANSAC 拟合平面内点最大距离（mm）
        ransac_n: 拟合平面时用于最小拟合的点数
        num_iterations: RANSAC 最大迭代次数
        visualize: 是否显示可视化窗口

    返回:
        plane_model: 平面参数 [a, b, c, d]
        inlier_mask: (H, W) 的 bool 掩码，True 表示属于平面
        height_map: (H, W) 的 float 高度图，仅平面区域有效
    """
    H, W, _ = roi_p3d.shape
    valid_mask = np.all(~np.isnan(roi_p3d), axis=2) & (roi_p3d[:, :, 2] > 0)
    points = roi_p3d[valid_mask].reshape(-1, 3)

    if len(points) < ransac_n:
        print("点云中有效点不足以拟合平面")
        return None, None, None

    # Open3D 点云构建与拟合
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points / 1000.0)  # 转为米
    plane_model, inliers = pcd.segment_plane(distance_threshold / 1000.0, ransac_n, num_iterations)
    a, b, c, d = plane_model
    print(f"拟合平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # inlier mask 映射回 2D 图像
    inlier_mask = np.zeros((H, W), dtype=bool)
    inlier_points = np.asarray(pcd.points)[inliers] * 1000  # 转回 mm

    # 构建快速 KDTree 来映射回像素位置
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    _, indices = tree.query(inlier_points, k=1)

    flat_indices = np.flatnonzero(valid_mask)
    selected_pixels = flat_indices[indices]
    y_coords, x_coords = np.unravel_index(selected_pixels, (H, W))
    inlier_mask[y_coords, x_coords] = True

    # --- 高度图计算 ---
    height_map = np.zeros((H, W), dtype=np.float32)
    normal = np.array([a, b, c])
    norm = np.linalg.norm(normal)
    normal_unit = normal / norm

    # 提取平面点
    yy, xx = np.where(inlier_mask)
    pts = roi_p3d[yy, xx, :]
    dot = pts @ normal.T
    signed_dist = (dot + d * 1000) / norm
    height_map[yy, xx] = signed_dist

    # --- 可视化 ---
    if visualize:
        color_map = np.zeros((H, W, 3), dtype=np.uint8)
        norm_heights = (signed_dist - signed_dist.min()) / (signed_dist.max() - signed_dist.min() + 1e-6)
        height_img = np.uint8(norm_heights * 255)
        color = cv2.applyColorMap(height_img, cv2.COLORMAP_JET)
        color = color.reshape(-1, 3)  # 🔧 修复这里的 shape
        color_map[yy, xx] = color

        vis_mask = np.zeros((H, W), dtype=np.uint8)
        vis_mask[yy, xx] = 255

        contour_vis = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(vis_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)

        # cv2.imshow("Inlier Mask", vis_mask)
        # cv2.imshow("Relative Height Map", color_map)
        # cv2.imshow("Plane Region Contour", contour_vis)

    # FIXME plane_model, inlier_mask, height_map, conter_vis = fit_plane_and_extract_height_map( ValueError: not enough values to unpack(expected 4, got 3)
    return plane_model, inlier_mask, height_map, contour_vis


def extract_skeleton_from_contour_image(image: np.ndarray, invert_binary=True):
    """
    从图像中提取中心封闭轮廓区域并生成骨架线图像，用OpenCV显示

    参数:
        image: 输入图像，可以是彩色 (H, W, 3) 或灰度 (H, W)
        invert_binary: 是否进行二值图反转（适用于白底黑轮廓的图）

    显示:
        使用 OpenCV 显示掩码区域和骨架热力图叠加图
    """

    # 如果是彩色图，转换为灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 二值化
    threshold_type = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 127, 255, threshold_type)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到轮廓")
        return None

    # 找到最接近图像中心的轮廓
    image_center = np.array([gray.shape[1] // 2, gray.shape[0] // 2])
    contour_centers = [np.mean(cnt[:, 0, :], axis=0) for cnt in contours]
    distances = [np.linalg.norm(c - image_center) for c in contour_centers]
    closest_index = np.argmin(distances)
    selected_contour = contours[closest_index]

    # 创建掩码并填充选中轮廓区域
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, [selected_contour], -1, 255, thickness=cv2.FILLED)

    # 提取骨架
    masked_binary = cv2.bitwise_and(binary, mask)
    # skeleton = skeletonize(masked_binary > 0)

    # 热力图叠加效果
    # skeleton_uint8 = np.uint8(skeleton * 255)
    # skeleton_colored = cv2.applyColorMap(skeleton_uint8, cv2.COLORMAP_HOT)

    # 显示原图 + 骨架热力图叠加
    # image_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # overlay = cv2.addWeighted(image_color, 0.7, skeleton_colored, 0.7, 0)

    # 显示图像
    cv2.imshow("Selected Region Binary", masked_binary)
    # cv2.imshow("Skeleton Overlay", overlay)

    # 可视化，叠加轮廓线
    contours = extract_outer_inner_contours(masked_binary)
    # contours = extract_outer_inner_contours_002(surface_mask)
    vis_img = cv2.cvtColor(masked_binary, cv2.COLOR_GRAY2BGR)

    colors = [(0, 0, 255), (0, 255, 0)]  # 外轮廓红色，内轮廓绿色
    # for contour in contours:
    #     print(contour)
    for i, contour in enumerate(contours):
        cv2.drawContours(vis_img, [contour], -1, colors[i], 2)
        # cv2.drawContours(vis, [contour], -1, colors[i], 2)

    # 计算并绘制中轴线()
    dilation = None
    if len(contours) >= 2:
        outer_cnt = contours[0]
        inner_cnt = contours[1]

        # 创建环状区域掩膜
        ring_mask = np.zeros_like(masked_binary, dtype=np.uint8)
        cv2.drawContours(ring_mask, [outer_cnt], -1, 255, cv2.FILLED)  # 填充外轮廓
        cv2.drawContours(ring_mask, [inner_cnt], -1, 0, cv2.FILLED)  # 挖空内轮廓

        # 自定义骨架提取函数
        def skeletonize_(img):
            skel = np.zeros(img.shape, np.uint8)
            img = img.copy()
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(img, kernel)
                temp = cv2.dilate(eroded, kernel)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                if cv2.countNonZero(img) == 0:
                    break
            return skel

        # 提取骨架
        thinned = skeletonize_(ring_mask)  # (480,560)

        # 将骨架绘制为蓝色（BGR格式）
        vis_img[thinned != 0] = (255, 0, 0)  # 蓝色通道

        # 创建黑色背景视图
        black_view = np.zeros_like(vis_img)
        # 将骨架绘制为蓝色（BGR格式）在黑色视图上
        black_view[thinned != 0] = (255, 255, 255)  # white

        # 显示独立骨架视图
        cv2.imshow("Independent Skeleton View", black_view)

        # 膨胀操作，扩大骨架线 (腐蚀直接消失)
        _copy = black_view.copy()
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(_copy, kernel)
        cv2.imshow("dilation", dilation)

        # update
        # 转灰度
        if len(dilation.shape) == 3:
            dilation_gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
        else:
            dilation_gray = dilation

        #
        _, binary = cv2.threshold(dilation_gray, 127, 255, cv2.THRESH_BINARY)

        return dilation

    return dilation



def main():
    cl = PercipioSDK()

    dev_list = cl.ListDevice()
    for idx in range(len(dev_list)):
        dev = dev_list[idx]
        print('{} -- {} \t {}'.format(idx, dev.id, dev.iface.id))
    if len(dev_list) == 0:
        print('no device')
        return
    if len(dev_list) == 1:
        selected_idx = 0
    else:
        selected_idx = int(input('select a device:'))
    if selected_idx < 0 or selected_idx >= len(dev_list):
        return

    sn = dev_list[selected_idx].id

    handle = cl.Open(sn)
    if not cl.isValidHandle(handle):
        err = cl.TYGetLastErrorCodedescription()
        print('no device found : ', end='')
        print(err)
        return

    event = PythonPercipioDeviceEvent()
    cl.DeviceRegiststerCallBackEvent(event)

    ## START
    color_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_COLOR)
    if len(color_fmt_list) == 0:
        print('device has no color stream.')
        return

    print('color image format list:')
    for idx in range(len(color_fmt_list)):
        fmt = color_fmt_list[idx]
        print('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, cl.Width(fmt), cl.Height(fmt), fmt.getDesc()))
    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_COLOR, color_fmt_list[0])

    depth_fmt_list = cl.DeviceStreamFormatDump(handle, PERCIPIO_STREAM_DEPTH)
    if len(depth_fmt_list) == 0:
        print('device has no depth stream.')
        return

    print('depth image format list:')
    for idx in range(len(depth_fmt_list)):
        fmt = depth_fmt_list[idx]
        print('\t{} -size[{}x{}]\t-\t desc:{}'.format(idx, cl.Width(fmt), cl.Height(fmt), fmt.getDesc()))

    cl.DeviceStreamFormatConfig(handle, PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])

    err = cl.DeviceLoadDefaultParameters(handle)
    if err:
        print('Load default parameters fail: ', end='')
        print(cl.TYGetLastErrorCodedescription())
    else:
        print('Load default parameters successful')

    # 读取校准数据
    scale_unit = cl.DeviceReadCalibDepthScaleUnit(handle)
    print('depth image scale unit :{}'.format(scale_unit))  # 0.125
    depth_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    color_calib = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_COLOR)
    depth_calib_data = cl.DeviceReadCalibData(handle, PERCIPIO_STREAM_DEPTH)
    depth_calib_width = depth_calib_data.Width()
    depth_calib_height = depth_calib_data.Height()
    depth_calib_intr = depth_calib_data.Intrinsic()
    depth_calib_extr = depth_calib_data.Extrinsic()
    depth_calib_dis = depth_calib_data.Distortion()

    err = cl.DeviceStreamEnable(handle, PERCIPIO_STREAM_COLOR | PERCIPIO_STREAM_DEPTH)
    if err:
        print('device stream enable err:{}'.format(err))
        return

    print('{} -- {} \t'.format(0, "Map depth to color coordinate(suggest)"))
    print('{} -- {} \t'.format(1, "Map color to depth coordinate"))
    # 模式0：将深度图映射到彩色图坐标系（常用，彩色图分辨率通常更高）。
    # 模式1：将彩色图映射到深度图坐标系。
    # registration_mode = int(input('select registration mode(0 or 1):'))
    # if selected_idx < 0 or selected_idx >= 2:
    #     registration_mode = 0
    registration_mode = 2

    cl.DeviceStreamOn(handle)

    # 点云数据
    pointcloud_data_arr = pointcloud_data_list()

    img_registration_depth = image_data()
    img_registration_render = image_data()
    img_parsed_color = image_data()
    img_undistortion_color = image_data()
    img_registration_color = image_data()

    count = 0

    # 深度图分辨率：(1920, 2560)
    # 点云图分辨率：(1536, 2048）
    ROI_X1, ROI_Y1 = 900, 800  # 左上角坐标
    ROI_X2, ROI_Y2 = 1600, 1400  # 右下角坐标

    while True:
        if event.IsOffline():
            break
        image_list = cl.DeviceStreamRead(handle, 2000)
        if len(image_list) == 2:
            frame = []
            for i in range(len(image_list)):
                frame = image_list[i]
                if frame.streamID == PERCIPIO_STREAM_DEPTH:
                    img_depth = frame
                if frame.streamID == PERCIPIO_STREAM_COLOR:
                    img_color = frame

            if registration_mode == 0:  # 深度图映射到彩色图坐标系

                # 深度映射到彩色坐标系，得到 raw 深度和渲染用的伪彩
                cl.DeviceStreamMapDepthImageToColorCoordinate(
                    depth_calib, img_depth, scale_unit, color_calib,
                    img_color.width, img_color.height, img_registration_depth
                )
                cl.DeviceStreamDepthRender(img_registration_depth, img_registration_render)

                # 原始深度图（16bit, 单位 mm）
                raw_depth = img_registration_depth.as_nparray()  # (1920, 2560, 1)
                print('depth image scale unit :{}'.format(raw_depth.shape))
                roi_raw_depth = raw_depth[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                print('roi depth image shape :{}'.format(roi_raw_depth.shape))  # (600, 700, 1)

                roi_raw_depth = np.squeeze(roi_raw_depth)

                valid_mask = roi_raw_depth > 0
                if not np.any(valid_mask):
                    print("没有有效深度！")
                    continue
                min_depth = np.min(roi_raw_depth[valid_mask])
                thickness = 50
                lower = max(min_depth - thickness, 1)
                upper = min_depth + thickness
                print("lower:{} upper:{}".format(lower, upper))

                layer_mask = ((roi_raw_depth >= lower) & (roi_raw_depth <= upper)).astype(np.uint8) * 255
                # cv2.imshow('Layer Mask', layer_mask)
                print("layer_mask 非零像素数量: ", np.count_nonzero(layer_mask))

                contours, _ = cv2.findContours(layer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print("找到轮廓数量: ", len(contours))

                # 渲染可视化图像
                # cl.DeviceStreamDepthRender(img_registration_depth, img_registration_render)
                mat_depth_render = img_registration_render.as_nparray()  # (1920, 2560, 3)
                # cv2.imshow("mat_depth_render", mat_depth_render)
                cv2.rectangle(mat_depth_render, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 0), 2)
                roi_depth = mat_depth_render[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]  # [800:1400,900,1600] => [600,700]
                # cv2.imshow('ROI Depth', roi_depth)
                print(roi_depth.shape)  # (600, 700, 3)

                roi_depth_vis = roi_depth.copy()
                # cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
                cv2.imshow('ROI Depth VIS', roi_depth_vis)  # (600,700,3)

                # 点云转换
                cl.DeviceStreamMapDepthImageToPoint3D(frame, depth_calib_data, scale_unit, pointcloud_data_arr)
                sz = pointcloud_data_arr.size()
                print('get p3d size : {}'.format(sz))  # get p3d size : 3145728
                p3d_nparray = pointcloud_data_arr.as_nparray()  # (1536, 2048, 3)

                # ★★★ 自动缩放比例计算
                scale_x = p3d_nparray.shape[1] / raw_depth.shape[1]  # 2048 / 2560 = 0.8
                scale_y = p3d_nparray.shape[0] / raw_depth.shape[0]  # 1536 / 1920 = 0.8

                # ★★★ 根据深度 ROI 映射计算出点云 ROI 的坐标
                ROI_PCD_X1 = int(ROI_X1 * scale_x)  # 900 * 0.8 = 720
                ROI_PCD_X2 = int(ROI_X2 * scale_x)  # 1600 * 0.8 = 1280
                ROI_PCD_Y1 = int(ROI_Y1 * scale_y)  # 800 * 0.8 = 640
                ROI_PCD_Y2 = int(ROI_Y2 * scale_y)  # 1400 * 0.8 = 1120
                # Y2-Y1 = 480
                # X2-X1 = 560

                print("映射后点云ROI坐标：({}, {}) ~ ({}, {})".format(ROI_PCD_X1, ROI_PCD_Y1, ROI_PCD_X2, ROI_PCD_Y2))
                # 映射后点云ROI坐标：(720, 640) ~ (1280, 1120)

                # ★★★ ROI 对齐提取
                roi_p3d = p3d_nparray[ROI_PCD_Y1:ROI_PCD_Y2, ROI_PCD_X1:ROI_PCD_X2, :]  # [640:1120,720:1280]
                print("roi_p3d_shape", roi_p3d.shape)  # (480, 560, 3)
                cv2.imshow("roi_p3d", roi_p3d)

                # *** (480,560,3) => (600,700,3)
                # resize 到 (600, 700) 对齐深度图 ROI 形状
                # roi_p3d_aligned = cv2.resize(roi_p3d, (roi_raw_depth.shape[1], roi_raw_depth.shape[0]),
                #                              interpolation=cv2.INTER_NEAREST)
                # print("roi_3d_shape:", roi_p3d_aligned.shape)  # (600, 700, 3)
                # cv2.imshow("roi_p3d_aligned", roi_p3d_aligned)
                # cv2.imshow('p3d_z', roi_p3d_aligned[:, :, 2].astype(np.uint16))  # 显示 z 值

                # FIXME plane_model, inlier_mask, height_map, conter_vis = fit_plane_and_extract_height_map( ValueError: not enough values to unpack(expected 4, got 3)
                # RANSAC 拟合平面 + 掩码生成 + 可视化渲染
                plane_model, inlier_mask, height_map, conter_vis = fit_plane_and_extract_height_map(
                    roi_p3d,
                    # roi_p3d_aligned,
                    distance_threshold=3.8,
                    visualize=True
                )

                # 骨架提取
                skeletonize_vis = extract_skeleton_from_contour_image(conter_vis)
                # cv2.imshow("skeletonize_vis", skeletonize_vis)

            elif registration_mode == 1:  # 彩色图映射到深度图坐标系。
                cl.DeviceStreamImageDecode(img_color, img_parsed_color)
                cl.DeviceStreamDoUndistortion(color_calib, img_parsed_color, img_undistortion_color)

                # 将彩色图像映射到深度坐标系，适用于需要深度图分辨率的场景。
                cl.DeviceStreamMapRGBImageToDepthCoordinate(depth_calib, img_depth, scale_unit, color_calib,
                                                            img_undistortion_color, img_registration_color)

                cl.DeviceStreamDepthRender(img_depth, img_registration_render)
                mat_depth_render = img_registration_render.as_nparray()
                cv2.imshow('depth', mat_depth_render)

                mat_registration_color = img_registration_color.as_nparray()
                cv2.imshow('registration rgb', mat_registration_color)

            elif registration_mode == 2:  # 彩色点云显示

                ROI_X1, ROI_Y1 = 800, 700  # 左上角坐标
                ROI_X2, ROI_Y2 = 1200, 1050  # 右下角坐标

                # 点云转换
                cl.DeviceStreamMapDepthImageToPoint3D(frame, depth_calib_data, scale_unit, pointcloud_data_arr)
                sz = pointcloud_data_arr.size()
                print('get p3d size : {}'.format(sz))  # get p3d size : 3145728
                p3d_nparray = pointcloud_data_arr.as_nparray()  # (1536, 2048, 3)
                # cv2.imshow("p3d_nparray", p3d_nparray)

                # 深度图显示
                cl.DeviceStreamDepthRender(frame, img_registration_depth)
                depth = img_registration_depth.as_nparray()
                # cv2.imshow('depth', depth)  # (1536, 2048, 3)
                # Depth at (1000, 1000): R=35, G=255, B=219

                # roi depth
                roi_raw_depth = depth[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                # print('roi depth image shape :{}'.format(roi_raw_depth.shape))
                cv2.imshow("roi_raw_depth", roi_raw_depth)

                # roi cloud
                roi_cloud = p3d_nparray[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                cv2.imshow("roi_p3d_nparray", roi_cloud)

                # process
                plane_model, inlier_mask, height_map, conter_vis = fit_plane_and_extract_height_map(
                    roi_cloud,
                    # roi_p3d_aligned,
                    distance_threshold=3.8,
                    visualize=True
                )
                dilation_vis = extract_skeleton_from_contour_image(conter_vis)

                if dilation_vis is not None:
                    if dilation_vis.ndim == 3 and dilation_vis.shape[2] == 3:
                        if np.array_equal(dilation_vis[:, :, 0], dilation_vis[:, :, 1]) and \
                                np.array_equal(dilation_vis[:, :, 0], dilation_vis[:, :, 2]):
                            dilation_vis = dilation_vis[:, :, 0].copy()
                        else:
                            dilation_vis = cv2.cvtColor(dilation_vis, cv2.COLOR_BGR2GRAY)

                    # resize to depth size
                    full_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=dilation_vis.dtype)
                    full_mask[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = dilation_vis
                    # cv2.imshow("full_mask", full_mask)

                    # create red mask
                    mask_color = np.zeros_like(depth)
                    mask_color[full_mask == 255] = (102, 255, 255)  # G
                    # print("green pixel count:", np.sum(np.all(mask_color == (0, 255, 0), axis=2))) # 3233
                    # cv2.imshow("mask_color", mask_color)
                    # addWeight
                    overlay = cv2.addWeighted(depth, 0.6, mask_color, 0.4, 0)
                    # draw
                    cv2.rectangle(overlay, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
                    # cv2.imshow("result", overlay)

            k = cv2.waitKey(10)
            if k == ord('q'):
                break

    cl.DeviceStreamOff(handle)
    cl.Close(handle)


if __name__ == '__main__':
    main()
