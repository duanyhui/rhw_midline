# -*- coding: utf-8 -*-
"""
UI Processing Pipeline

职责：
- 在 GUI 中调用用户现有的算法函数（detect_simplified.py），完成 ROI→掩码→骨架→比较→偏移→CSV 的闭环。
- 自动加载 theoretical_centerline.npy、tilt_correction_matrix.npy、hand_eye_transform.npy。
- 仅以 import 方式复用，绝不改动原脚本注释。
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import csv
import numpy as np
import cv2

# 导入用户算法模块
try:
    import detect_simplified as detect
except Exception:
    try:
        import detect as detect
    except Exception as e:
        detect = None
        print("[WARN] detect module import failed:", e)

@dataclass
class ProcessingResult:
    surface_mask: Optional[np.ndarray] = None           # (H_roi, W_roi) uint8
    skeleton_overlay: Optional[np.ndarray] = None       # (H, W, 3) BGR on depth
    comparison_with_vectors: Optional[np.ndarray] = None# (H, W, 3) BGR
    deviation_results: Optional[list] = None            # [(key_pt, vec), ...]
    avg_pixel_offset: Optional[Tuple[float, float]] = None
    avg_mm_offset: Optional[Tuple[float, float]] = None
    msg: str = ""

class Pipeline:
    def __init__(self):
        self.theoretical_centerline = None      # (N,2)
        self.tilt_matrix = None                 # (3,3)
        self.hand_eye = None                    # (2,3)
        self._last_result: Optional[ProcessingResult] = None
        self.load_assets()

    # ---------- Assets ----------
    def load_assets(self):
        """从当前工作目录自动加载 .npy 资产。"""
        self.theoretical_centerline = self._load_npy("theoretical_centerline.npy")
        self.tilt_matrix = self._load_npy("tilt_correction_matrix.npy")
        self.hand_eye = self._load_npy("hand_eye_transform.npy")
        self.theory_ok = self.theoretical_centerline is not None
        self.tilt_ok = self.tilt_matrix is not None
        self.handeye_ok = self.hand_eye is not None

    @staticmethod
    def _load_npy(path: str):
        try:
            return np.load(path)
        except Exception:
            return None

    # ---------- Processing ----------
    def process_frame(self,
                      p3d_nparray: np.ndarray,
                      depth_render_bgr: np.ndarray,
                      roi_xyxy: Tuple[int, int, int, int],
                      num_key_points: int = 12) -> ProcessingResult:
        """
        在一帧数据上执行完整流程。
        """
        if detect is None:
            raise RuntimeError("detect 模块导入失败，无法执行算法。")

        x1, y1, x2, y2 = roi_xyxy
        # 保护边界
        h, w = p3d_nparray.shape[:2]
        x1 = max(0, min(x1, w-2)); x2 = max(1, min(x2, w-1))
        y1 = max(0, min(y1, h-2)); y2 = max(1, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            return ProcessingResult(msg="ROI 无效。")

        # 1) ROI 点云提取
        roi_cloud = p3d_nparray[y1:y2, x1:x2].copy()  # (H_roi, W_roi, 3) 单位mm

        # 2) 倾斜校正（可选）
        if self.tilt_matrix is not None:
            pts = roi_cloud.reshape(-1, 3)
            valid = pts[:, 2] > 0
            pts[valid] = pts[valid] @ self.tilt_matrix.T
            roi_cloud = pts.reshape(roi_cloud.shape)

        # 3) 表面掩码
        if self.tilt_matrix is not None and hasattr(detect, "create_corrected_mask"):
            surface_mask, z_min = detect.create_corrected_mask(roi_cloud, depth_margin=3.5, pixel_size_mm=0.5)  # 可能弹窗
        else:
            surface_mask, z_min = detect.extract_nearest_surface_mask(roi_cloud, depth_margin=3.5)
        if surface_mask is None:
            return ProcessingResult(msg="未能提取表面掩码。")

        # 4) 骨架（实际中轴线）
        # 优先使用通用骨架提取，visualize=False 避免开启窗口
        skel_bgr = detect.extract_skeleton_universal(surface_mask, visualize=False)
        if skel_bgr is None:
            return ProcessingResult(surface_mask=surface_mask, msg="骨架提取失败。")

        # 转灰度用于叠加
        skel_gray = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY) if skel_bgr.ndim == 3 else skel_bgr

        # 5) 提取骨架离散点（映射到全局坐标）
        if hasattr(detect, "extract_skeleton_points_and_visualize"):
            actual_points = detect.extract_skeleton_points_and_visualize(
                skel_bgr, origin_offset=(x1, y1), visualize=False
            )
        else:
            # 兼容 draw_centerline.py 移植的简化版本接口
            from draw_centerline import extract_skeleton_points as _esp  # type: ignore
            actual_points = _esp(skel_bgr, origin_offset=(x1, y1))
        if actual_points is None or len(actual_points) == 0:
            return ProcessingResult(surface_mask=surface_mask, msg="骨架点为空。")

        # 6) 在深度渲染图上叠加骨架（UI展示）
        depth_h, depth_w = depth_render_bgr.shape[:2]
        full_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = skel_gray
        mask_color = np.zeros_like(depth_render_bgr)
        mask_color[full_mask > 0] = (102, 255, 255)
        overlay = cv2.addWeighted(depth_render_bgr, 0.7, mask_color, 0.6, 0.0)

        # 7) 与理论中轴线比较 + 偏移
        comparison_img = None
        deviation_results = []
        avg_px = None
        avg_mm = None

        if self.theoretical_centerline is not None and hasattr(detect, "compare_centerlines"):
            comparison_img, _ = detect.compare_centerlines(
                actual_points, self.theoretical_centerline, (depth_h, depth_w)
            )
            # 偏移关键点
            deviation_results = detect.calculate_deviation_vectors(
                actual_points, self.theoretical_centerline, num_key_points=num_key_points
            )
            # 平均像素偏移
            if deviation_results:
                vecs = np.array([v for _, v in deviation_results], dtype=np.float32)
                avg_px = (float(vecs[:,0].mean()), float(vecs[:,1].mean()))
                # 物理(mm)偏移（若有手眼 2x3）
                if self.hand_eye is not None:
                    linear = self.hand_eye[:, :2]  # 2x2
                    mm = linear @ np.array([[avg_px[0]], [avg_px[1]]], dtype=np.float32)
                    avg_mm = (float(mm[0,0]), float(mm[1,0]))
            # 偏移箭头可视化
            if comparison_img is not None and deviation_results:
                comparison_img = detect.visualize_deviation_vectors(comparison_img, deviation_results)

        self._last_result = ProcessingResult(
            surface_mask=surface_mask,
            skeleton_overlay=overlay,
            comparison_with_vectors=comparison_img,
            deviation_results=deviation_results,
            avg_pixel_offset=avg_px,
            avg_mm_offset=avg_mm,
            msg=f"提取点数: {len(actual_points)}"
        )
        return self._last_result

    def has_result(self) -> bool:
        return self._last_result is not None and self._last_result.deviation_results is not None

    # ---------- Export ----------
    def export_csv(self, path: str):
        """导出关键点偏移与平均值汇总。"""
        if not self.has_result():
            raise RuntimeError("没有可导出的偏移结果。")
        res = self._last_result
        rows = []
        # 每个关键点
        for i, (kp, vec) in enumerate(res.deviation_results):
            kp = np.array(kp).astype(float)
            vec = np.array(vec).astype(float)
            mag_px = float(np.linalg.norm(vec))
            mm_x, mm_y, mag_mm = "", "", ""
            if res.avg_mm_offset is not None and self.hand_eye is not None:
                # 对单个向量也做 mm 变换（2x2）
                linear = self.hand_eye[:, :2]
                mm_vec = linear @ vec.reshape(2,1)
                mm_x = float(mm_vec[0,0]); mm_y = float(mm_vec[1,0])
                mag_mm = float(np.linalg.norm([mm_x, mm_y]))
            rows.append([
                i, float(kp[0]), float(kp[1]), float(vec[0]), float(vec[1]), mag_px, mm_x, mm_y, mag_mm
            ])
        # 平均
        avg_px = res.avg_pixel_offset if res.avg_pixel_offset is not None else (None, None)
        avg_mm = res.avg_mm_offset if res.avg_mm_offset is not None else (None, None)
        rows.append(["AVERAGE", "", "", avg_px[0], avg_px[1],
                     "" if avg_px[0] is None else float(np.linalg.norm(avg_px)),
                     avg_mm[0] if avg_mm[0] is not None else "",
                     avg_mm[1] if avg_mm[1] is not None else "",
                     "" if avg_mm[0] is None else float(np.linalg.norm(avg_mm))])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index","key_x","key_y","dx(px)","dy(px)","|d|(px)","dx(mm)","dy(mm)","|d|(mm)"])
            writer.writerows(rows)
