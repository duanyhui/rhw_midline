import numpy as np
import pcammls
from pcammls import *
import cv2
import open3d
from skimage.morphology import skeletonize
from scipy.ndimage import label
from scipy.spatial import cKDTree


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

### --- 新增辅助函数：用于保持纵横比的缩放 --- ###
def resize_with_padding(img, target_shape):
    """
    通过添加黑边（Letterboxing）来缩放图像，同时保持其原始纵横比。
    这可以防止在缩放过程中发生不希望的拉伸或压缩，从而解决叠加时的位置偏移问题。

    :param img: 需要缩放的源图像。
    :param target_shape: (height, width) 格式的目标尺寸。
    :return: 经过缩放和填充后，尺寸与 target_shape 完全一致的新图像。
    """
    target_h, target_w = target_shape
    img_h, img_w = img.shape[:2]

    # 计算目标和源图像的纵横比
    target_aspect = target_w / target_h
    img_aspect = img_w / img_h

    if img_aspect > target_aspect:
        # 源图像比目标“更宽”，因此以宽度为基准进行缩放
        new_w = target_w
        new_h = int(new_w / img_aspect)
    else:
        # 源图像比目标“更高”，因此以高度为基准进行缩放
        new_h = target_h
        new_w = int(new_h * img_aspect)

    # 使用计算出的新尺寸进行缩放，这会保持原始比例
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # 创建一个符合目标尺寸的黑色背景画布
    padded_img = np.zeros(target_shape, dtype=img.dtype)

    # 计算将缩放后图像放置在画布中心所需的偏移量（即黑边的尺寸）
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    # 将缩放后的图像粘贴到画布中心
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return padded_img


### --- 新增函数：用于生成无畸变掩码的核心 --- ###
def create_corrected_mask(roi_p3d_rotated, depth_margin, pixel_size_mm=0.5):
    """
    (新方案) 通过正交投影和形态学闭运算，从旋转后的点云创建无畸变且密实的表面掩码。
    这个函数将替代旧的 extract_nearest_surface_mask 来处理校正后的点云。

    :param roi_p3d_rotated: (H, W, 3) 的 NumPy 数组，代表已经过旋转校正的点云ROI。
    :param depth_margin: 从最近表面提取点云层的厚度容差 (mm)。
    :param pixel_size_mm: 正交投影后，每个像素代表的真实世界尺寸 (mm)。
    :return: (final_mask, z_min_val)
             - final_mask: (H', W') 的 uint8 数组，最终生成的无畸变、密实的二值掩码。
             - z_min_val: 校正后点云中的最小Z值。
    """
    # 1. 将点云ROI重塑为点列表，并过滤无效点
    points = roi_p3d_rotated.reshape(-1, 3)
    valid_points = points[points[:, 2] > 0]
    if valid_points.shape[0] < 3:
        print("正交投影错误: 有效点数量过少。")
        return None, None

    # 2. 在校正后的Z轴上找到最近的表面
    z_min_val = np.min(valid_points[:, 2])
    print(f"\t最近表面Z值 (校正后): {z_min_val:.2f}mm")
    surface_points = valid_points[(valid_points[:, 2] >= z_min_val) & (valid_points[:, 2] <= z_min_val + depth_margin)]
    if surface_points.shape[0] < 3:
        print("正交投影错误: 表面点数量过少。")
        return None, None

    # 3. --- 正交投影 ---
    # 计算点云在真实世界XY平面上的边界
    x_min, y_min, _ = np.min(surface_points, axis=0)
    x_max, y_max, _ = np.max(surface_points, axis=0)

    # 根据真实世界尺寸和设定的像素分辨率，计算新掩码图像的像素尺寸
    width_px = int(np.ceil((x_max - x_min) / pixel_size_mm)) + 1
    height_px = int(np.ceil((y_max - y_min) / pixel_size_mm)) + 1
    if width_px <= 1 or height_px <= 1:
        return None, None

    # 4. 将3D表面点“画”到一个新的2D画布（稀疏的点集）上
    point_img = np.zeros((height_px, width_px), dtype=np.uint8)
    px_coords = np.floor((surface_points[:, 0] - x_min) / pixel_size_mm).astype(int)
    py_coords = np.floor((surface_points[:, 1] - y_min) / pixel_size_mm).astype(int)
    point_img[py_coords, px_coords] = 255
    cv2.imshow("Orthographic Projection (Sparse Points)", point_img)  # 用于调试，显示稀疏的点

    # 5. --- 形态学闭运算 ---
    # 这是连接稀疏点，形成完整轮廓的关键步骤
    # 内核大小需要根据点的稀疏程度进行调整，一个较大的内核可以连接更远的点
    kernel_size = 3  # 可以从 5, 7, 9, 11 开始尝试
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 迭代次数越多，填充效果越强
    closed_img = cv2.morphologyEx(point_img, cv2.MORPH_CLOSE, kernel, iterations=4)

    print(f"\t已创建无畸变掩码，尺寸: {closed_img.shape}")
    ### 将校正后的掩码resize回原始ROI的尺寸 ###
    # 这解决了后续步骤中因尺寸不匹配导致的ValueError
    # 注意：这可能会引入轻微的纵横比失真，但保留了正确的拓扑结构（如孔洞）
    ### --- 关键修复：使用带padding的resize来保持纵横比，解决偏移问题 --- ###
    original_roi_shape = (roi_p3d_rotated.shape[0], roi_p3d_rotated.shape[1]) # (height, width)
    resized_mask = resize_with_padding(closed_img, original_roi_shape)


    print(f"\t已创建无畸变掩码，并从 {closed_img.shape} resize到 {resized_mask.shape} 以匹配ROI")
    return resized_mask, z_min_val


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
    lower = z_min_val
    upper = z_min_val + depth_margin
    surface_mask = ((z_img >= lower) & (z_img <= upper)).astype(np.uint8) * 255  # 二值掩码

    return surface_mask, z_min_val

def extract_nearest_surface_mask_ransac(roi_p3d_aligned, depth_margin,
                                        ransac_iters=400,
                                        dist_thresh=0.8,
                                        front_percentile=20.0,
                                        subsample=50000,
                                        random_state=None):
    """
    使用 RANSAC 在粗糙场景下提取“最近表面”的掩码（斜面近端带厚度）。

    参数:
        roi_p3d_aligned: np.ndarray, (H, W, 3)，ROI 内点云，单位 mm
        depth_margin: float，最近表面沿相机 Z 方向的厚度阈值（mm）
        ransac_iters: int，RANSAC 迭代次数
        dist_thresh: float，RANSAC 评估平面的正交距离阈值（mm）
        front_percentile: float，仅从靠前的这部分百分位（按Z从小到大）里随机采样构模，减少采到背景面的概率
        subsample: int，为加速在大ROI下进行下采样的上限点数
        random_state: Optional[int]，随机种子，便于复现实验

    返回:
        surface_mask: np.ndarray, (H, W) uint8，255表示属于最近表面区域
        z_min_val: float，拟合后内点中的最小Z（mm），便于日志/对齐参考
    """
    import numpy as np

    H, W, _ = roi_p3d_aligned.shape
    z_img = roi_p3d_aligned[:, :, 2].copy()
    z_img[z_img <= 0] = 0  # 过滤无效值
    valid_mask = z_img > 0
    if not np.any(valid_mask):
        print("无有效深度值")
        return None, None

    rng = np.random.default_rng(random_state)

    # --- 准备点集 ---
    ys, xs = np.where(valid_mask)
    pts = roi_p3d_aligned[ys, xs].reshape(-1, 3)  # Nx3
    N = pts.shape[0]

    # 仅在靠前(front)的点里采样建模，降低选到背景的概率
    z_sorted = np.sort(pts[:, 2])
    z_front_cut = z_sorted[int(np.clip((front_percentile / 100.0) * (N - 1), 0, N - 1))]
    front_idx = np.where(pts[:, 2] <= z_front_cut)[0]
    if front_idx.size < 3:
        # 退化情况下扩大采样集合
        front_idx = np.arange(N)

    # 下采样以加速
    if N > subsample:
        keep = rng.choice(N, subsample, replace=False)
        pts_eval = pts[keep]
    else:
        pts_eval = pts

    # --- RANSAC: 拟合最“靠前”的大面 ---
    best_inliers = None
    best_inlier_count = 0
    best_mean_z = np.inf
    best_plane = None  # (a,b,c,d) with ||(a,b,c)||=1

    for _ in range(int(ransac_iters)):
        # 1) 随机取3点(来自前景子集)拟合候选平面
        tri = rng.choice(front_idx, 3, replace=False)
        p1, p2, p3 = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -np.dot(n, p1)  # 平面 ax+by+cz+d=0

        # 2) 计算评估集到平面的正交距离
        dists = np.abs(pts_eval @ n + d)  # 因为n单位化，无需再除以||n||
        inliers_eval = dists < dist_thresh
        inlier_count = int(inliers_eval.sum())
        if inlier_count == 0:
            continue

        # 3) 在全体点上复核（只在更优时做）
        if inlier_count > best_inlier_count:
            dists_all = np.abs(pts @ n + d)
            inliers_all = dists_all < dist_thresh
            inlier_count_all = int(inliers_all.sum())
            mean_z = float(pts[inliers_all, 2].mean()) if inlier_count_all > 0 else np.inf

            # 选择：优先更多内点；若相同，更靠前（平均Z更小）
            if (inlier_count_all > best_inlier_count) or \
               (inlier_count_all == best_inlier_count and mean_z < best_mean_z):
                best_inliers = inliers_all
                best_inlier_count = inlier_count_all
                best_mean_z = mean_z
                best_plane = (n, d)

    if best_inliers is None or best_inlier_count < 3:
        # RANSAC 失败，退回到最简单“z层”方法
        z_min_val = float(np.min(pts[:, 2]))
        print("\t[RANSAC回退] 最小深度值:{}mm".format(z_min_val))
        lower, upper = z_min_val, z_min_val + float(depth_margin)
        surface_mask = ((z_img >= lower) & (z_img <= upper)).astype(np.uint8) * 255
        return surface_mask, z_min_val

    # --- 用最佳内点做一次 SVD 精拟合平面 ---
    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    X = inlier_pts - centroid
    # SVD: 最小奇异值对应的法向
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n_ref = Vt[-1, :]
    n_ref_norm = np.linalg.norm(n_ref)
    if n_ref_norm < 1e-9:
        # 退化情况：沿用RANSAC的n,d
        n_ref, d_ref = best_plane
    else:
        n_ref = n_ref / n_ref_norm
        d_ref = -np.dot(n_ref, centroid)

    # --- 生成掩码：沿相机Z方向的“上行厚度” ---
    # 平面: a x + b y + c z + d = 0
    a, b, c = n_ref
    d = d_ref

    surface_mask = np.zeros((H, W), dtype=np.uint8)
    z_min_val = float(inlier_pts[:, 2].min())
    print("\t[RANSAC] 最小深度值:{}mm".format(z_min_val))

    if abs(c) >= 1e-8:
        # 可用 z_plane 预测，保留 z - z_plane ∈ [0, depth_margin]
        # 对有效像素进行向量化计算
        xs_f = roi_p3d_aligned[valid_mask, 0]
        ys_f = roi_p3d_aligned[valid_mask, 1]
        zs_f = roi_p3d_aligned[valid_mask, 2]
        z_plane = -(a * xs_f + b * ys_f + d) / c
        dz = zs_f - z_plane
        keep = (dz >= 0.0) & (dz <= float(depth_margin))
        surface_mask[ys, xs] = (keep.astype(np.uint8) * 255)
    else:
        # 平面近似与Z轴平行，退而用“正交距离 + z阈值”的保守判定
        dists_all = np.abs(roi_p3d_aligned[:, :, 0] * a +
                           roi_p3d_aligned[:, :, 1] * b +
                           roi_p3d_aligned[:, :, 2] * c + d)
        # 取在平面附近且 z 不超过最近内点 + depth_margin
        keep = (dists_all < float(dist_thresh)) & (z_img > 0) & (z_img <= (z_min_val + float(depth_margin)))
        surface_mask[keep] = 255

    return surface_mask, z_min_val







def extract_contours_adaptive(surface_mask, min_vertices=4, max_vertices=12, max_iterations=10,
                              initial_epsilon_factor=0.01):
    """
    自适应轮廓提取：
    通过迭代增加epsilon，将轮廓拟合到指定的顶点数范围内，以适应包含曲线和直线的形状。

    :param surface_mask: 输入的二值图像掩码。
    :param min_vertices: 拟合后多边形的最小顶点数。
    :param max_vertices: 拟合后多边形的最大顶点数。
    :param max_iterations: 为找到合适epsilon值的最大迭代次数。
    :param initial_epsilon_factor: 初始的epsilon系数（相对于周长）。
    :return: 包含拟合后轮廓的列表。
    """
    contours, _ = cv2.findContours(surface_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # 按面积从大到小排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    fitted_contours = []
    # 只处理前两个最大轮廓
    for cnt in contours[:2]:
        # 步骤1：计算凸包以平滑轮廓
        hull = cv2.convexHull(cnt)
        perimeter = cv2.arcLength(hull, True)

        if perimeter == 0:
            continue

        # 步骤2：迭代寻找最佳epsilon
        epsilon_factor = initial_epsilon_factor
        best_approx = hull  # 默认值为原始凸包

        for _ in range(max_iterations):
            epsilon = epsilon_factor * perimeter
            approx = cv2.approxPolyDP(hull, epsilon, closed=True)

            # 如果顶点数在理想范围内，则采用此结果并退出循环
            if min_vertices <= len(approx) <= max_vertices:
                best_approx = approx
                break

            # 如果顶点数太多，说明epsilon太小，需要增加它以简化更多
            if len(approx) > max_vertices:
                epsilon_factor *= 1.5  # 增加epsilon的系数
                best_approx = approx  # 暂存当前最接近的结果
            # 如果顶点数太少，说明epsilon过大，已经过度简化了。
            # 此时可以停止，并使用上一次迭代的结果（如果需要更精确控制）。
            # 在这个简化模型中，我们只在顶点过多时调整，最终会得到一个结果。
            else:  # len(approx) < min_vertices
                # 已经过度简化，跳出循环，使用上一次或当前的结果
                break

        fitted_contours.append(best_approx)

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
        contour_vis: 可视化轮廓图像 (H, W, 3)
    """
    H, W, _ = roi_p3d.shape
    valid_mask = np.all(~np.isnan(roi_p3d), axis=2) & (roi_p3d[:, :, 2] > 0)
    points = roi_p3d[valid_mask].reshape(-1, 3)

    if len(points) < ransac_n:
        print("点云中有效点不足以拟合平面")
        return None, None, None,None

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

        cv2.imshow("Inlier Mask", vis_mask)
        cv2.imshow("Relative Height Map", color_map)
        cv2.imshow("Plane Region Contour", contour_vis)

    # FIXME plane_model, inlier_mask, height_map, conter_vis = fit_plane_and_extract_height_map( ValueError: not enough values to unpack(expected 4, got 3)
    return plane_model, inlier_mask, height_map, contour_vis



def overlay_skeleton_with_scaling(cl: pcammls.PercipioSDK,
                                  depth_calib,
                                  img_depth: pcammls.image_data,
                                  scale_unit: float,
                                  color_calib,
                                  img_color: pcammls.image_data,
                                  skeleton_img: np.ndarray):
    """
    使用SDK的坐标映射功能，将深度坐标系下的骨架图精确地叠加到彩色图上。
    (修正版：解决掩码与彩色图维度不匹配的索引错误)

    Args:
        cl: PercipioSDK 实例.
        depth_calib: 深度相机的标定数据.
        img_depth: 原始的深度图像数据 (将被临时修改).
        scale_unit: 深度值缩放单位.
        color_calib: 彩色相机的标定数据.
        img_color: 原始的彩色图像数据.
        skeleton_img: 在深度图坐标系下生成的骨架二值图 (uint8).
    """
    if skeleton_img is None:
        print("输入的骨架图为空，无法叠加。")
        return

    # 1. 备份原始深度数据
    original_depth_np = img_depth.as_nparray().copy()

    # 2. 创建包含真实深度值的“伪深度图”
    fake_depth_np = np.zeros_like(original_depth_np)
    skeleton_mask = (skeleton_img == 255)
    fake_depth_np[skeleton_mask] = original_depth_np[skeleton_mask]

    # 3. 将伪深度图数据临时写入 img_depth 的缓冲区
    img_depth.as_nparray()[:] = fake_depth_np

    # 4. 调用SDK函数进行坐标映射
    mapped_skeleton_image = pcammls.image_data()
    cl.DeviceStreamMapDepthImageToColorCoordinate(
        depth_calib, img_depth, scale_unit, color_calib,
        img_color.width, img_color.height, mapped_skeleton_image
    )

    # 5. 立即恢复原始深度数据
    img_depth.as_nparray()[:] = original_depth_np

    # 6. 从映射结果中生成最终的、对齐的二值掩码
    mapped_skeleton_np = mapped_skeleton_image.as_nparray()
    if mapped_skeleton_np is None:
        print("坐标映射失败，无法生成叠加图。")
        return

    final_aligned_mask = (mapped_skeleton_np > 0).astype(np.uint8) * 255

    # 7. 解码彩色图并进行叠加
    decoded_color = pcammls.image_data()
    cl.DeviceStreamImageDecode(img_color, decoded_color)
    color_arr = decoded_color.as_nparray()

    overlay_color = np.zeros_like(color_arr)

    # --- 修正 IndexError 的核心代码 ---
    # 创建布尔掩码
    boolean_mask = (final_aligned_mask == 255)
    # 检查掩码是否为 (H, W, 1) 的3D形状，如果是，则压缩为 (H, W) 的2D形状
    if boolean_mask.ndim == 3 and boolean_mask.shape[2] == 1:
        boolean_mask = np.squeeze(boolean_mask, axis=2)

    # 现在使用2D掩码为3D图像的像素赋值
    overlay_color[boolean_mask] = (0, 255, 0)  # BGR: Green

    result = cv2.addWeighted(color_arr, 1.0, overlay_color, 0.8, 0)

    # 为了对比，显示映射后的原始深度图渲染效果
    img_registration_render = pcammls.image_data()
    mapped_depth_for_render = pcammls.image_data()
    cl.DeviceStreamMapDepthImageToColorCoordinate(
        depth_calib, img_depth, scale_unit, color_calib,
        img_color.width, img_color.height, mapped_depth_for_render
    )
    cl.DeviceStreamDepthRender(mapped_depth_for_render, img_registration_render)
    cv2.imshow("MappedDepthRender", img_registration_render.as_nparray())

    # 显示最终的叠加结果
    result = cv2.resize(result,(0,0),fx=0.5, fy=0.5)
    cv2.imshow("ColorSkeletonOverlay (Aligned)", result)


def extract_skeleton_from_surface_mask(surface_mask: np.ndarray, visualize: bool = True):
    """
    直接从表面掩码（如 Z 值提取的掩码）中提取内外轮廓并计算中轴线（骨架）。
    此函数绕过了 RANSAC 拟合，直接处理给定的二值掩码。

    参数:
        surface_mask: (H, W) 的二值掩码 (uint8), 255 代表目标区域。
        visualize: 是否显示中间过程和结果的可视化窗口。

    返回:
        dilated_skeleton: (H, W, 3) 的 BGR 图像，包含膨胀后的骨架线。
                         如果无法生成，则返回 None。
    修改时间:  2025-06-25 17:19

    """
    if surface_mask is None or np.count_nonzero(surface_mask) == 0:
        print("输入的 surface_mask 为空或不包含有效区域。")
        return None

    # 1. 从掩码中提取内外轮廓（通常是面积最大的两个）

    contours = extract_contours_adaptive(surface_mask, min_vertices=4, max_vertices=16,initial_epsilon_factor=0.001)

    if len(contours) < 2:
        print("未能找到足够的内外轮廓来生成环状区域。")
        return None

    outer_cnt = contours[0]
    inner_cnt = contours[1]

    # 2. 创建环状区域掩膜 (Ring Mask)
    ring_mask = np.zeros_like(surface_mask, dtype=np.uint8)
    cv2.drawContours(ring_mask, [outer_cnt], -1, 255, cv2.FILLED)  # 填充外轮廓
    cv2.drawContours(ring_mask, [inner_cnt], -1, 0, cv2.FILLED)   # 在内轮廓区域挖一个洞

    # 3. 提取骨架
    # 使用 scikit-image 的 skeletonize 函数，效果较好
    from skimage.morphology import skeletonize
    skeleton = skeletonize(ring_mask > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    # 4. 膨胀骨架使其更粗，便于观察和后续处理
    kernel = np.ones((1, 1), np.uint8)
    dilated_skeleton = cv2.dilate(skeleton_img, kernel, iterations=1)

    # 5. 可视化（如果需要）
    if visualize:
        # 创建一个彩色图像用于可视化
        vis_img = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)
        # 绘制内外轮廓
        cv2.drawContours(vis_img, [outer_cnt], -1, (0, 0, 255), 2)  # 外轮廓红色
        cv2.drawContours(vis_img, [inner_cnt], -1, (0, 255, 0), 2)  # 内轮廓绿色
        # 将骨架叠加为蓝色
        vis_img[dilated_skeleton != 0] = (255, 0, 0)

        cv2.imshow("Ring Mask", ring_mask)
        cv2.imshow("Skeleton from Surface Mask", vis_img)

    # 返回一个三通道的图像，方便后续的颜色叠加等操作
    return cv2.cvtColor(dilated_skeleton, cv2.COLOR_GRAY2BGR)


def extract_skeleton_universal(surface_mask: np.ndarray, visualize: bool = True):
    """
    通用的骨架提取函数，能同时处理实心和空心（有镂空）的物体。

    该函数通过检测轮廓数量来自动判断物体类型，并相应地创建用于骨架化的目标掩码。
    - 如果只有一个轮廓，则处理为实心物体。
    - 如果有多个轮廓，则取最大的两个创建环形区域进行处理。

    :param surface_mask: (H, W) 的二值掩码 (uint8), 255 代表目标区域。
    :param visualize: 是否显示中间过程和结果的可视化窗口。
    :return: (H, W, 3) 的 BGR 图像，包含膨胀后的骨架线。如果无法生成，则返回 None。
    """
    if surface_mask is None or np.count_nonzero(surface_mask) == 0:
        print("输入的 surface_mask 为空或不包含有效区域。")
        return None

    # 1. 提取轮廓并进行自适应拟合
    contours = extract_contours_adaptive(surface_mask, min_vertices=4, max_vertices=16, initial_epsilon_factor=0.001)

    if not contours:
        print("未能找到任何轮廓。")
        return None

    # 2. ★★★ 核心改动：根据轮廓数量决定处理方式 ★★★
    target_mask = np.zeros_like(surface_mask, dtype=np.uint8)
    outer_cnt = contours[0]
    inner_cnt = None

    if len(contours) >= 2:
        # 情况一：空心物体（至少有两个轮廓）
        print("检测到空心物体，生成环形掩码。")
        inner_cnt = contours[1]
        # 创建环形区域
        cv2.drawContours(target_mask, [outer_cnt], -1, 255, cv2.FILLED)
        cv2.drawContours(target_mask, [inner_cnt], -1, 0, cv2.FILLED)
    else:
        # 情况二：实心物体（只有一个轮廓）
        print("检测到实心物体，生成填充掩码。")
        # 创建实心区域
        cv2.drawContours(target_mask, [outer_cnt], -1, 255, cv2.FILLED)

    # 3. 提取骨架
    skeleton = skeletonize(target_mask > 0)
    skeleton_img = (skeleton * 255).astype(np.uint8)

    # 4. 膨胀骨架使其更粗
    kernel = np.ones((3, 3), np.uint8)
    dilated_skeleton = cv2.dilate(skeleton_img, kernel, iterations=1)

    # 5. 可视化
    if visualize:
        vis_img = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_img, [outer_cnt], -1, (0, 0, 255), 2)  # 外轮廓红色
        if inner_cnt is not None:
            cv2.drawContours(vis_img, [inner_cnt], -1, (0, 255, 0), 2)  # 内轮廓绿色

        # 将骨架叠加为蓝色
        vis_img[dilated_skeleton != 0] = (255, 0, 0)

        cv2.imshow("Target Mask for Skeletonization", target_mask)
        cv2.imshow("Universal Skeleton Extraction", vis_img)

    return cv2.cvtColor(dilated_skeleton, cv2.COLOR_GRAY2BGR)

def extract_skeleton_points_and_visualize(skeleton_image, origin_offset=(0, 0), visualize=True):
    """
    从骨架图中提取所有离散点的坐标，并可选择性地进行可视化。
    该函数处理坐标系转换，将ROI内的局部坐标映射为全局坐标。

    :param skeleton_image: 输入的骨架二值图 (H, W) 或 (H, W, 3)。通常是 extract_skeleton_* 函数的输出。
    :param origin_offset: (x, y) 格式的偏移量，即ROI在原图中的左上角坐标。
    :param visualize: 是否创建一个新窗口来绘制并显示这些离散点。
    :return: 一个 (N, 2) 的 NumPy 数组，包含所有骨架点的 (x, y) 全局坐标。如果无有效点则返回空数组。
    """
    # 1. 确保输入是 2D 灰度图
    if skeleton_image is None or skeleton_image.size == 0:
        print("输入的骨架图为空，无法提取点。")
        return np.array([])

    if skeleton_image.ndim == 3:
        gray_skeleton = cv2.cvtColor(skeleton_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_skeleton = skeleton_image

    # 2. 提取所有非零像素的坐标 (y, x)
    rows, cols = np.where(gray_skeleton > 0)

    if len(rows) == 0:
        print("骨架图中未找到有效点。")
        return np.array([])

    # 3. 将坐标转换为 (x, y) 格式并应用全局偏移量
    # np.vstack((cols, rows)).T 将 (y,x) 坐标对转换为 (N, 2) 的 [x, y] 格式数组
    local_points = np.vstack((cols, rows)).T
    global_points = local_points + np.array(origin_offset)

    # 4. 在新窗口中可视化离散点
    if visualize:
        # 创建一个与输入骨架图同样大小的黑色画布
        vis_canvas = np.zeros((skeleton_image.shape[0], skeleton_image.shape[1], 3), dtype=np.uint8)

        # 在画布上绘制每个离散点（使用局部坐标）
        for point in local_points:
            # point 是 [x, y] 格式
            cv2.circle(vis_canvas, tuple(point), radius=1, color=(0, 255, 255), thickness=-1)  # 绘制黄色的点

        cv2.imshow("Discrete Skeleton Points (in ROI)", vis_canvas)

    return global_points


def compare_centerlines(actual_points, theoretical_points, image_shape):
    """
    比较实际中轴线和理论中轴线，并生成可视化结果。

    :param actual_points: (N, 2) numpy数组，实际中轴线坐标。
    :param theoretical_points: (M, 2) numpy数组，理论中轴线坐标。
    :param image_shape: (height, width) 的元组，用于创建可视化图像。
    :return:
        - vis_image: (H, W, 3) BGR图像，可视化比较结果。
        - match_score: 0到1之间的浮点数，表示匹配程度。
    """
    if actual_points.size == 0 or theoretical_points.size == 0:
        return np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8), 0.0

    vis_image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # 绘制理论中轴线 (绿色)
    for i in range(len(theoretical_points) - 1):
        p1 = tuple(theoretical_points[i].astype(int))
        p2 = tuple(theoretical_points[i+1].astype(int))
        cv2.line(vis_image, p1, p2, (0, 255, 0), 2)

    # 绘制实际中轴线 (红色)
    for point in actual_points:
        cv2.circle(vis_image, tuple(point.astype(int)), radius=2, color=(0, 0, 255), thickness=-1)

    # 计算匹配程度
    tree = cKDTree(theoretical_points)
    distances, _ = tree.query(actual_points, k=1)

    # 将距离超过一定阈值（例如5个像素）的点视为不匹配
    threshold = 5.0
    matched_points = np.sum(distances < threshold)
    match_score = matched_points / len(actual_points) if len(actual_points) > 0 else 0.0

    # 在图像上显示匹配分数
    cv2.putText(vis_image, f"Match Score: {match_score*100:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return vis_image, match_score

def calculate_deviation_vectors(actual_points, theoretical_points, num_key_points=10):
    """
    计算实际中轴线相对于理论中轴线的偏移向量。

    该函数在理论中轴线上选取若干关键点，然后为每个关键点找到
    实际中轴线上最近的对应点，并计算两者之间的偏移向量。

    :param actual_points: (N, 2) numpy数组，实际中轴线坐标。
    :param theoretical_points: (M, 2) numpy数组，理论中轴线坐标。
    :param num_key_points: 要在理论中轴线上选取的关键点数量。
    :return: 一个列表，每个元素是一个元组 (key_point, deviation_vector)，
             其中 key_point 是理论关键点的坐标，deviation_vector 是 (dx, dy) 形式的偏移向量。
             如果输入为空，则返回空列表。
    """
    if actual_points.size == 0 or theoretical_points.size == 0:
        return []

    # 1. 在理论中轴线上均匀选取关键点
    key_point_indices = np.linspace(0, len(theoretical_points) - 1, num_key_points, dtype=int)
    theory_key_points = theoretical_points[key_point_indices]

    # 2. 构建实际中轴线点的 KDTree 以便快速查找最近邻
    tree = cKDTree(actual_points)

    deviation_results = []
    # 3. 为每个理论关键点找到实际对应点并计算向量
    for key_point in theory_key_points:
        # 查找最近的实际点
        distance, nearest_index = tree.query(key_point)
        actual_corresponding_point = actual_points[nearest_index]

        # 计算偏移向量
        deviation_vector = actual_corresponding_point - key_point
        deviation_results.append((key_point, deviation_vector))

    return deviation_results

def visualize_deviation_vectors(vis_image, deviation_results):
    """
    在图像上将偏移向量可视化为箭头。

    :param vis_image: 要绘制的 BGR 图像。
    :param deviation_results: calculate_deviation_vectors 函数的输出结果。
    :return: 绘制了向量后的图像。
    """
    img_with_vectors = vis_image.copy()
    for key_point, vector in deviation_results:
        start_point = tuple(key_point.astype(int))
        end_point = tuple((key_point + vector).astype(int))

        # 绘制从实际点指向理论点的箭头，表示偏差方向

        cv2.arrowedLine(img_with_vectors, end_point, start_point, (0, 255, 255), 2, tipLength=0.4) # 黄色箭头
        # 标记理论关键点位置
        cv2.circle(img_with_vectors, start_point, radius=3, color=(255, 0, 0), thickness=-1) # 蓝色圆点

    return img_with_vectors

def visualize_z_channel(p3d_array, window_name="Z-Channel Visualization"):
    """
    将点云的 Z 通道可视化为一张伪彩色深度图。

    :param p3d_array: (H, W, 3) 的点云 NumPy 数组。
    :param window_name: 显示窗口的名称。
    """
    # 1. 提取 Z 通道
    z_channel = p3d_array[:, :, 2].copy()

    # 2. 找到有效的 Z 值 (非无穷大或 NaN)
    valid_mask = np.isfinite(z_channel) & (z_channel != 0)
    if not np.any(valid_mask):
        # 如果没有有效点，显示一张黑图
        cv2.imshow(window_name, np.zeros((z_channel.shape[0], z_channel.shape[1], 3), dtype=np.uint8))
        return

    # 3. 将有效 Z 值归一化到 0-255 范围以便显示
    z_valid = z_channel[valid_mask]
    z_min, z_max = np.min(z_valid), np.max(z_valid)

    if z_max - z_min > 0:
        # 归一化
        normalized_z = (z_channel - z_min) / (z_max - z_min) * 255
    else:
        # 如果所有值都一样，则设为中间值
        normalized_z = np.full(z_channel.shape, 128)

    # 将归一化后的值转换为 8 位无符号整数
    z_uint8 = normalized_z.astype(np.uint8)

    # 4. 应用伪彩色映射表以增强可视化效果
    colored_z = cv2.applyColorMap(z_uint8, cv2.COLORMAP_JET)

    # 在无效区域显示为黑色
    colored_z[~valid_mask] = [0, 0, 0]

    # 5. 显示图像
    cv2.imshow(window_name, colored_z)

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

    # 加载理论中轴线
    try:
        theoretical_centerline = np.load('theoretical_centerline.npy')
    except FileNotFoundError:
        print("警告: 未找到 'theoretical_centerline.npy'。请先运行 draw_centerline.py 创建理论中轴线。")
        theoretical_centerline = None
    try:
        rotation_matrix = np.load('tilt_correction_matrix.npy')
        print("成功加载倾斜校正矩阵。")
    except FileNotFoundError:
        rotation_matrix = None
        print("警告: 未找到 'tilt_correction_matrix.npy'。")
        print("程序将不进行倾斜校正。请先运行 my_calibrate.py 进行校准。")
    try:
        hand_eye_matrix = np.load('hand_eye_transform.npy')
        print("成功加载手眼标定矩阵 'hand_eye_transform.npy'。")
    except FileNotFoundError:
        print("警告: 未找到手眼标定矩阵 'hand_eye_transform.npy'。")
        print("将无法计算物理偏移量。请先运行 calibrate_hand_eye.py。")
        hand_eye_matrix = None

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

            if registration_mode == 2:  # 彩色点云显示

                # ROI_X1, ROI_Y1 = 800, 700  # 左上角坐标
                # ROI_X2, ROI_Y2 = 1200, 1050  # 右下角坐标
                # 这里使用的是深度图分辨率的 ROI
                # 正方形的
                ROI_X1, ROI_Y1 = 993, 614  # 左上角坐标
                ROI_X2, ROI_Y2 = 1186, 790  # 右下角坐标
                #backup
                # ROI_X1, ROI_Y1 = 915, 709  # 左上角坐标
                # ROI_X2, ROI_Y2 = 980, 790  # 右下角坐标


                # 点云转换
                cl.DeviceStreamMapDepthImageToPoint3D(frame, depth_calib_data, scale_unit, pointcloud_data_arr)
                sz = pointcloud_data_arr.size()
                print('get p3d size : {}'.format(sz))  # get p3d size : 3145728
                p3d_nparray = pointcloud_data_arr.as_nparray()  # (1536, 2048, 3)
                visualize_z_channel(p3d_nparray, "Z-Channel (Before Correction)")
                # 点云校准
                if rotation_matrix is not None:
                    # 获取点云的原始形状
                    original_shape = p3d_nparray.shape
                    # 将点云数据从 (H, W, 3) 变形为 (N, 3) 以便进行矩阵乘法
                    points = p3d_nparray.reshape(-1, 3)

                    # 过滤掉无效点 (z<=0)，避免不必要的计算
                    valid_points_mask = points[:, 2] > 0

                    # 使用矩阵进行旋转：p' = R * p^T  (这里用 p @ R.T 更高效)
                    # 只对有效点进行旋转
                    points[valid_points_mask] = points[valid_points_mask] @ rotation_matrix.T

                    # 将旋转后的点云数据恢复为原始的 (H, W, 3) 形状
                    p3d_nparray = points.reshape(original_shape)
                    print("已对点云应用倾斜校正。")
                    visualize_z_channel(p3d_nparray, "Z-Channel (After Correction)")

                # 深度图显示
                cl.DeviceStreamDepthRender(frame, img_registration_depth)
                depth = img_registration_depth.as_nparray()

                # 提取点云 ROI
                roi_cloud = p3d_nparray[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
                cv2.imshow("roi_p3d_nparray", roi_cloud)


                if rotation_matrix is None:
                    print("未进行倾斜校正，使用原始方法提取掩码。")
                    surface_mask, z_min = extract_nearest_surface_mask(roi_cloud, depth_margin=3.5)
                else:
                    # 如果已校正，则使用正交投影生成无畸变掩码
                    print("已进行倾斜校正，使用正交投影生成无畸变掩码。")
                    # surface_mask, z_min = create_corrected_mask(roi_cloud, depth_margin=5.5, pixel_size_mm= 0.5)
                    # 使用原始方式，能更好地处理空心物体
                    # surface_mask, z_min = extract_nearest_surface_mask(roi_cloud, depth_margin=5.8)
                    # 使用 RANSAC 方法提取表面掩码，适应粗糙表面
                    surface_mask, z_min = extract_nearest_surface_mask_ransac(roi_cloud, depth_margin=3.8, ransac_iters=400,
                                                        dist_thresh=0.8,front_percentile=20.0,subsample=50000,random_state=1)
                if surface_mask is None:
                    print("无法提取表面掩码，跳过此帧。")
                    continue
                cv2.imshow("Initial Surface Mask", surface_mask)

                # # 直接从 surface_mask 计算中轴线，不再使用 RANSAC
                # dilation_vis = extract_skeleton_from_surface_mask(surface_mask, visualize=True)
                # # 使用通用骨架提取方法，能处理实心和空心物体
                dilation_vis = extract_skeleton_universal(surface_mask, visualize=True)

                # # 使用RANSAC拟合平面并提取高度图，鲁棒性高，但准确度可能不如直接提取最近一层点云区域。
                # # plane_model, inlier_mask, height_map, conter_vis = fit_plane_and_extract_height_map(
                # #     roi_cloud,
                # #     distance_threshold=4.8,
                # #     visualize=True,
                # # )

                if dilation_vis is not None:

                    # 提取中轴线的离散点，并传入ROI的偏移量(ROI_X1, ROI_Y1)来获取全局坐标
                    skeleton_points = extract_skeleton_points_and_visualize(
                        dilation_vis,
                        origin_offset=(ROI_X1, ROI_Y1),
                        visualize=True
                    )

                    # 打印返回的数据，供后续比较和分析
                    print(f"提取到 {len(skeleton_points)} 个中轴线离散点。")
                    if len(skeleton_points) > 5:
                        print("前5个点的全局坐标 (x, y):")
                        print(skeleton_points[:5])


                    # --- 比较中轴线并计算和显示偏移向量 ---
                    if theoretical_centerline is not None:
                        # 步骤 1: 生成基础的对比图
                        comparison_vis, match_score = compare_centerlines(
                            skeleton_points,
                            theoretical_centerline,
                            (depth.shape[0], depth.shape[1])
                        )
                        print(f"中轴线匹配度: {match_score:.2f}")

                        # 步骤 2: 计算偏移向量
                        deviation_vectors = calculate_deviation_vectors(
                            skeleton_points,
                            theoretical_centerline,
                            num_key_points=12
                        )

                        # 打印输出纠偏数据
                        print("--- 关键点偏移向量 (理论点 -> 实际点) ---")
                        for i, (key_pt, vec) in enumerate(deviation_vectors):
                            print(f"  关键点 {i}: 理论位置 {key_pt.astype(int)}, 偏移 (dx, dy) = {vec.astype(int)}")

                        # 计算并输出平均偏移量
                        if deviation_vectors:
                            # 提取所有的偏移向量 (dx, dy)
                            all_vectors = np.array([vec for _, vec in deviation_vectors])
                            # 计算平均偏移量
                            average_offset = np.mean(all_vectors, axis=0)
                            print("--- 平均偏移量 (dx, dy) ---")
                            print(f"  ({average_offset[0]:.2f}, {average_offset[1]:.2f})")

                            if hand_eye_matrix is not None:
                                # 提取矩阵的线性部分 (旋转和缩放)
                                linear_transform = hand_eye_matrix[:, :2]

                                # 计算物理世界中的偏移量 (dX, dY)
                                physical_offset = linear_transform @ average_offset

                                print("--- 纠正后的平均物理偏移量 (dX, dY) ---")
                                print(f"  ({physical_offset[0]:.3f}, {physical_offset[1]:.3f}) mm")

                        # 步骤 3: 在对比图上可视化偏移向量
                        final_vis = visualize_deviation_vectors(comparison_vis, deviation_vectors)
                        # # 窗口大小调整成一半
                        # final_vis = cv2.resize(final_vis, (0, 0), fx=0.5, fy=0.5)

                        cv2.imshow("Centerline Comparison with Deviation Vectors", final_vis)

                    # --- 结束新增 ---

                    if dilation_vis.ndim == 3 and dilation_vis.shape[2] == 3:
                        if np.array_equal(dilation_vis[:, :, 0], dilation_vis[:, :, 1]) and \
                                np.array_equal(dilation_vis[:, :, 0], dilation_vis[:, :, 2]):
                            dilation_vis = dilation_vis[:, :, 0].copy()
                        else:
                            dilation_vis = cv2.cvtColor(dilation_vis, cv2.COLOR_BGR2GRAY)

                    # 创建与完整深度图同样大小的掩码
                    full_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=dilation_vis.dtype)
                    # 将ROI区域的骨架填充到正确位置
                    full_mask[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = dilation_vis
                    cv2.imshow("full_mask", full_mask)

                    # 在深度图上叠加骨架
                    mask_color = np.zeros_like(depth)
                    mask_color[full_mask == 255] = (102, 255, 255)  # 浅绿色
                    overlay = cv2.addWeighted(depth, 0.6, mask_color, 0.4, 0)
                    # draw
                    cv2.rectangle(overlay, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 2)
                    cv2.imshow("result", overlay)

                    original_color = image_data()
                    cl.DeviceStreamImageDecode(img_color,original_color)
                    color_arr = original_color.as_nparray()
                    cv2.imshow("original_color_arr", color_arr)
                    # 在彩色图上叠加骨架线
                    overlay_skeleton_with_scaling(
                        cl, depth_calib, img_depth, scale_unit,
                        color_calib, img_color, full_mask
                    )

            k = cv2.waitKey(10)
            if k == ord('q'):
                break

    cl.DeviceStreamOff(handle)
    cl.Close(handle)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
