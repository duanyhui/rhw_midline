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

# 导入用户算法模块（保持不改动原脚本）
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
     # --- 新增：用于导出 ---
    theoretical_centerline: Optional[np.ndarray] = None               # (N,2)
    actual_centerline_points: Optional[List[Tuple[float, float]]] = None  # [(x,y),...]


@dataclass
class PipelineConfig:
    depth_margin: float = 3.5      # mm，离表面最近层的容忍范围
    pixel_size_mm: float = 0.5     # mm/px，用于 create_corrected_mask 的栅格化

    # 新增：RANSAC 相关默认参数（粗糙场景）
    ransac_iters: int = 400
    ransac_dist_thresh: float = 0.8
    ransac_front_percentile: float = 20.0
    ransac_subsample: int = 50000
    ransac_seed: Optional[int] = None

    # 新增：表面掩码提取方法
    # 可选：'nearest'（最近Z层，快速）、'corrected'（create_corrected_mask，需倾斜校正）、'ransac'（粗糙场景）
    surface_method: str = "nearest"

class Pipeline:
    def __init__(self):
        self.theoretical_centerline = None      # (N,2)
        self.tilt_matrix = None                 # (3,3)
        self.hand_eye = None                    # (2,3)
        self._last_result: Optional[ProcessingResult] = None
        self.cfg = PipelineConfig()
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

    # 允许在外部更新配置（GUI 可调用）
    def set_config(self,
                   depth_margin: Optional[float] = None,
                   pixel_size_mm: Optional[float] = None,
                   surface_method: Optional[str] = None,
                   ransac_iters: Optional[int] = None,
                   ransac_dist_thresh: Optional[float] = None,
                   ransac_front_percentile: Optional[float] = None,
                   ransac_subsample: Optional[int] = None,
                   ransac_seed: Optional[int] = None):
        if depth_margin is not None:
            self.cfg.depth_margin = float(depth_margin)
        if pixel_size_mm is not None:
            self.cfg.pixel_size_mm = float(pixel_size_mm)
        if surface_method is not None:
            surface_method = str(surface_method).lower()
            if surface_method in ("nearest", "corrected", "ransac"):
                self.cfg.surface_method = surface_method
        if ransac_iters is not None:
            self.cfg.ransac_iters = int(ransac_iters)
        if ransac_dist_thresh is not None:
            self.cfg.ransac_dist_thresh = float(ransac_dist_thresh)
        if ransac_front_percentile is not None:
            self.cfg.ransac_front_percentile = float(ransac_front_percentile)
        if ransac_subsample is not None:
            self.cfg.ransac_subsample = int(ransac_subsample)
        if ransac_seed is not None:
            # -1 表示随机；其他整数固定种子
            self.cfg.ransac_seed = None if int(ransac_seed) < 0 else int(ransac_seed)

    # ---------- Processing ----------
    def process_frame(self,
                      p3d_nparray: np.ndarray,
                      depth_render_bgr: np.ndarray,
                      roi_xyxy: Tuple[int, int, int, int],
                      num_key_points: int = 12,
                      depth_margin: Optional[float] = None,
                      pixel_size_mm: Optional[float] = None,
                      surface_method: Optional[str] = None,
                      # RANSAC 可覆写参数（可选）
                      ransac_iters: Optional[int] = None,
                      ransac_dist_thresh: Optional[float] = None,
                      ransac_front_percentile: Optional[float] = None,
                      ransac_subsample: Optional[int] = None,
                      ransac_seed: Optional[int] = None
                      ) -> ProcessingResult:
        """
        在一帧数据上执行完整流程。
        可选参数：
        - depth_margin: 覆盖默认配置的 mm 容差
        - pixel_size_mm: 覆盖默认配置的 mm/px，create_corrected_mask 时有效
        - surface_method: 'nearest' | 'corrected' | 'ransac'
        - RANSAC 相关：iters / dist_thresh / front_percentile / subsample / seed
        """
        if detect is None:
            raise RuntimeError("detect 模块导入失败，无法执行算法。")

        # 解析参数（若未传则落到配置默认）
        dm = float(depth_margin) if depth_margin is not None else self.cfg.depth_margin
        pmm = float(pixel_size_mm) if pixel_size_mm is not None else self.cfg.pixel_size_mm
        method = (surface_method or self.cfg.surface_method).lower()

        r_iters = int(ransac_iters) if ransac_iters is not None else self.cfg.ransac_iters
        r_dist = float(ransac_dist_thresh) if ransac_dist_thresh is not None else self.cfg.ransac_dist_thresh
        r_front = float(ransac_front_percentile) if ransac_front_percentile is not None else self.cfg.ransac_front_percentile
        r_subs = int(ransac_subsample) if ransac_subsample is not None else self.cfg.ransac_subsample
        r_seed = (None if (ransac_seed is None and self.cfg.ransac_seed is None)
                  else (None if (ransac_seed is not None and int(ransac_seed) < 0)
                        else (self.cfg.ransac_seed if ransac_seed is None else int(ransac_seed))))

        x1, y1, x2, y2 = roi_xyxy
        # 保护边界
        h, w = p3d_nparray.shape[:2]
        x1 = max(0, min(x1, w-2)); x2 = max(1, min(x2, w-1))
        y1 = max(0, min(y1, h-2)); y2 = max(1, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            return ProcessingResult(msg="ROI 无效。")

        # 1) ROI 点云
        roi_cloud = p3d_nparray[y1:y2, x1:x2].copy()  # (H_roi, W_roi, 3) mm

        # 2) 倾斜校正（可选）
        if self.tilt_matrix is not None:
            pts = roi_cloud.reshape(-1, 3)
            valid = pts[:, 2] > 0
            pts[valid] = pts[valid] @ self.tilt_matrix.T
            roi_cloud = pts.reshape(roi_cloud.shape)

        # 3) 表面掩码（根据选择的方法）
        surface_mask, z_min = None, None
        used_method = method

        if method == "ransac":
            if hasattr(detect, "extract_nearest_surface_mask_ransac"):
                surface_mask, z_min = detect.extract_nearest_surface_mask_ransac(
                    roi_cloud,
                    depth_margin=dm,
                    ransac_iters=r_iters,
                    dist_thresh=r_dist,
                    front_percentile=r_front,
                    subsample=r_subs,
                    random_state=r_seed
                )
            else:
                print("[WARN] detect 中未找到 RANSAC 方法，回退到最近Z层。")
                used_method = "nearest"

        if surface_mask is None and used_method == "corrected":
            if self.tilt_matrix is not None and hasattr(detect, "create_corrected_mask"):
                surface_mask, z_min = detect.create_corrected_mask(
                    roi_cloud, depth_margin=dm, pixel_size_mm=pmm
                )
            else:
                print("[WARN] create_corrected_mask 不可用或缺少倾斜校正矩阵，回退到最近Z层。")
                used_method = "nearest"

        if surface_mask is None and used_method == "nearest":
            # 原逻辑：最近 Z 层
            # todo  都使用 create_corrected_mask方法，extract_nearest_surface_mask方法会导致物体中轴线映射到深度图后出现畸变，要寻找新的方法
            surface_mask, z_min = detect.extract_nearest_surface_mask(
                roi_cloud, depth_margin=dm
            )

        if surface_mask is None:
            return ProcessingResult(msg="未能提取表面掩码。")

        # 4) 骨架（实际中轴线）
        # 优先使用通用骨架提取，visualize=False 避免开启窗口
        skel_bgr = detect.extract_skeleton_universal(surface_mask, visualize=False)
        if skel_bgr is None:
            return ProcessingResult(surface_mask=surface_mask, msg="骨架提取失败。")

        # 5) 骨架离散点（映射到全局）
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

        # 6) 叠加到深度渲染
        depth_h, depth_w = depth_render_bgr.shape[:2]
        skel_gray = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY) if skel_bgr.ndim == 3 else skel_bgr
        full_mask = np.zeros((depth_h, depth_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = skel_gray
        mask_color = np.zeros_like(depth_render_bgr)
        mask_color[full_mask > 0] = (102, 255, 255)
        overlay = cv2.addWeighted(depth_render_bgr, 0.7, mask_color, 0.6, 0.0)

        # 7) 比较 + 偏移
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
                    linear = self.hand_eye[:, :2]
                    mm = linear @ np.array([[avg_px[0]], [avg_px[1]]], dtype=np.float32)
                    avg_mm = (float(mm[0,0]), float(mm[1,0]))
            # 偏移箭头可视化
            if comparison_img is not None and deviation_results:
                comparison_img = detect.visualize_deviation_vectors(comparison_img, deviation_results)

        # 结果信息汇总
        ransac_info = ""
        if used_method == "ransac":
            ransac_info = f" | RANSAC: iters={r_iters}, dist={r_dist}, front%={r_front}, subsample={r_subs}, seed={r_seed}"

        self._last_result = ProcessingResult(
            surface_mask=surface_mask,
            skeleton_overlay=overlay,
            comparison_with_vectors=comparison_img,
            deviation_results=deviation_results,
            avg_pixel_offset=avg_px,
            avg_mm_offset=avg_mm,
            msg=f"提取点数: {len(actual_points)} | method={used_method} | depth_margin={dm}, pixel_size_mm={pmm}{ransac_info}"
            # --- 新增：用于导出 ---
            theoretical_centerline=self.theoretical_centerline if self.theoretical_centerline is not None else None,
            actual_centerline_points=[(float(p[0]), float(p[1])) for p in actual_points] if actual_points is not None else None

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
            rows.append([i, float(kp[0]), float(kp[1]), float(vec[0]), float(vec[1]), mag_px, mm_x, mm_y, mag_mm])
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
            # --- 追加区段：实际中轴线起点（初始值） ---
            # 仅写入一个起点，满足“实际中轴线坐标初始值”的需求
            writer.writerow([])
            writer.writerow(["ACTUAL_CENTERLINE_START", "x", "y"])
            if (res.actual_centerline_points is not None) and len(res.actual_centerline_points) > 0:
                ax0, ay0 = res.actual_centerline_points[0]
                writer.writerow(["A0", float(ax0), float(ay0)])
            else:
                writer.writerow(["A0", "", ""])  # 无则留空

            # --- 追加区段：理论中轴线全坐标 ---
            writer.writerow([])
            writer.writerow(["THEORETICAL_CENTERLINE", "index", "x", "y"])
            if (res.theoretical_centerline is not None) and (len(res.theoretical_centerline) > 0):
                for i, (tx, ty) in enumerate(res.theoretical_centerline):
                    writer.writerow(["T", i, float(tx), float(ty)])
            else:
                writer.writerow(["T", "", "", ""])  # 无则留空
