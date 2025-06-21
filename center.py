# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.morphology import skeletonize
# from skimage.measure import label, regionprops
#
# # 读取图像（灰度）
# image_path = "/home/oneko/projects/precipio/camport_multi_language-master/img_res/5-19/Plane Region Contour_screenshot_22.05.2025.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# # 二值化
# _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
#
# # 查找轮廓
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 找到最接近中心的封闭区域（按距离图像中心点排序）
# image_center = np.array([image.shape[1] // 2, image.shape[0] // 2])
# contour_centers = [np.mean(cnt[:, 0, :], axis=0) for cnt in contours]
# distances = [np.linalg.norm(c - image_center) for c in contour_centers]
# closest_index = np.argmin(distances)
# selected_contour = contours[closest_index]
#
# # 创建 mask 并填充所选区域
# mask = np.zeros_like(binary)
# cv2.drawContours(mask, [selected_contour], -1, 255, thickness=cv2.FILLED)
#
# # 在 mask 区域内提取骨架线
# masked_binary = cv2.bitwise_and(binary, mask)
# skeleton = skeletonize(masked_binary > 0)
#
# # 可视化结果
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(masked_binary, cmap='gray')
# ax[0].set_title("Selected Region Binary")
# ax[1].imshow(image, cmap='gray')
# ax[1].imshow(skeleton, cmap='hot', alpha=0.8)
# ax[1].set_title("Skeleton in Center Region")
# for a in ax:
#     a.axis('off')
# plt.tight_layout()
# plt.show()

import cv2
import numpy as np
from skimage.morphology import skeletonize

# 读取图像（灰度）
image_path = "/home/oneko/projects/precipio/camport_multi_language-master/img_res/5-19/Plane Region Contour_screenshot_22.05.2025.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 二值化
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最接近中心的封闭区域
image_center = np.array([image.shape[1] // 2, image.shape[0] // 2])
contour_centers = [np.mean(cnt[:, 0, :], axis=0) for cnt in contours]
distances = [np.linalg.norm(c - image_center) for c in contour_centers]
closest_index = np.argmin(distances)
selected_contour = contours[closest_index]

# 创建 mask 并填充所选区域
mask = np.zeros_like(binary)
cv2.drawContours(mask, [selected_contour], -1, 255, thickness=cv2.FILLED)

# 提取骨架线
masked_binary = cv2.bitwise_and(binary, mask)
skeleton = skeletonize(masked_binary > 0)  # bool array

# 把骨架转换为可视图（使用热力图效果）
skeleton_uint8 = np.uint8(skeleton * 255)
skeleton_colored = cv2.applyColorMap(skeleton_uint8, cv2.COLORMAP_HOT)

# 显示原图 + 骨架叠加效果
image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(image_color, 0.7, skeleton_colored, 0.7, 0)

# 显示窗口
cv2.imshow("Selected Region Binary", masked_binary)
cv2.imshow("Skeleton Overlay", overlay)

try:
    # 等待按键或捕获Ctrl+C
    while True:
        key = cv2.waitKey(100)  # 每100毫秒检查一次
        if key != -1:  # 检测到按键
            break
except KeyboardInterrupt:
    print("程序被用户中断。")
finally:
    # 确保关闭所有OpenCV窗口
    cv2.destroyAllWindows()
