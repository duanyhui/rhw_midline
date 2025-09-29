
# -*- coding: utf-8 -*-
"""
Controller: 以“调用方式”使用现有算法文件中的函数/类，
不修改 align_centerline_to_gcode_pro_edit_max.py。

功能：
- 预先启动相机（避免慢启动），随用随取单帧。
- 单帧处理流水线（ROI→顶视→展平→最近表面→引导中心线→可视化）。
- 导出偏移表 CSV 与纠偏 G 代码。
- 可选：保存当前帧的 bias_comp（per_index）。
"""
from __future__ import annotations
import os, sys, json, math, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 允许从项目根目录导入你现有的算法文件
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2

# === 导入现有算法模块（保持不改动源文件） ===
import align_centerline_to_gcode_pro_edit_max as core

# ===== Fallbacks: 当 core 缺少私有函数时，使用本地实现 =====
def _fallback_detect_corner_knots_from_tangent(T, ang_thresh_deg=35.0, min_gap_pts=3):
    """根据相邻切向量夹角检测拐点；返回索引（含首尾）。"""
    import numpy as _np
    n = len(T)
    if n <= 1:
        return [0] if n else []
    ang = _np.arctan2(T[:,1], T[:,0])
    d = _np.diff(ang)
    d = (d + _np.pi) % (2*_np.pi) - _np.pi  # wrap to [-pi, pi]
    jumps = _np.where(_np.abs(_np.degrees(d)) >= float(ang_thresh_deg))[0] + 1
    keep = []
    for i in jumps:
        if not keep or (i - keep[-1]) >= int(max(1, min_gap_pts)):
            keep.append(int(i))
    knots = [0] + keep + [n-1]
    return sorted(set([k for k in knots if 0 <= k < n]))

def _fallback_remap_knots_index(k_cal, n_cal, n_in):
    """把标定轨迹结点索引映射到 bias 数组索引空间（线性比例）"""
    if n_cal <= 1 or n_in <= 1:
        return [0, max(0, n_in-1)]
    return [int(round(k*(n_in-1)/(n_cal-1))) for k in k_cal]

def _fallback_resample_per_index_bias_piecewise(arr, target_len, k_in, k_cur):
    """按对应结点分段线性插值，把 per-index bias 重采样到 target_len。"""
    import numpy as _np
    arr = _np.asarray(arr, _np.float32).ravel()
    out = _np.empty(int(target_len), _np.float32)
    if len(k_cur) < 2 or len(k_in) < 2:
        return _fallback_resample_per_index_bias_for_length(arr, int(target_len))
    out[:k_cur[0]+1] = arr[k_in[0]]
    out[k_cur[-1]:] = arr[k_in[-1]]
    for (a,b,aa,bb) in zip(k_in[:-1], k_in[1:], k_cur[:-1], k_cur[1:]):
        if bb <= aa:
            continue
        x = _np.linspace(0.0, 1.0, bb-aa+1, dtype=_np.float32)
        out[aa:bb+1] = (1-x)*arr[a] + x*arr[b]
    return out

def _fallback_resample_per_index_bias_for_length(arr, target_len):
    """全长等比例 1D 插值兜底。"""
    import numpy as _np
    arr = _np.asarray(arr, _np.float32).ravel()
    target_len = int(target_len)
    if arr.size == 0:
        return _np.zeros(target_len, _np.float32)
    if arr.size == 1 or target_len <= 1:
        return _np.full(target_len, float(arr[0]), _np.float32)
    x_src = _np.linspace(0.0, 1.0, arr.size, dtype=_np.float32)
    x_dst = _np.linspace(0.0, 1.0, target_len, dtype=_np.float32)
    return _np.interp(x_dst, x_src, arr).astype(_np.float32)

# ---------- 工具：Qt 图像转换（BGR/灰度 → QImage） ----------
try:
    from PyQt5.QtGui import QImage
except Exception:
    QImage = None  # 仅为类型占位

def np_to_qimage(img: np.ndarray) -> Optional[QImage]:
    if QImage is None or img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        buf = img.astype(np.uint8).tobytes()
        q = QImage(buf, w, h, w, QImage.Format_Grayscale8)
        return q.copy()
    elif img.ndim == 3:
        h, w, c = img.shape
        if c == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            return q.copy()
        elif c == 4:
            q = QImage(img.data, w, h, w * 4, QImage.Format_ARGB32)
            return q.copy()
    return None

# ---------- 配置容器 ----------
@dataclass
class GUIConfig:
    T_path: str = "T_cam2machine.npy"
    gcode_path: str = "args/example.gcode"
    bias_enable: bool = True
    bias_path: str = "bias_comp.json"

    # ROI / 投影
    roi_mode: str = "gcode_bounds"  # none / camera_rect / machine / gcode_bounds
    cam_roi_xywh: Tuple[int,int,int,int] = (682, 847, 228, 185)
    roi_center_xy: Tuple[float,float] = (50.0, 50.0)
    roi_size_mm: float = 550.0
    bounds_margin_mm: float = 20.0
    pixel_size_mm: float = 0.8

    # 最近表面
    z_select: str = "max"
    nearest_use_percentile: bool = True
    nearest_qlo: float = 1.0
    nearest_qhi: float = 99.0
    depth_margin_mm: float = 3.0
    morph_open: int = 3
    morph_close: int = 5
    min_component_area_px: int = 600

    # 引导中心线
    guide_enable: bool = True
    guide_step_mm: float = 1.0
    guide_halfwidth_mm: float = 6.0
    guide_smooth_win: int = 7
    guide_max_offset_mm: float = 8.0
    guide_max_grad_mm_per_mm: float = 0.08
    max_gap_pts: int = 5
    curvature_adaptive: bool = True
    curvature_gamma: float = 35.0
    min_smooth_win: int = 3

    # 拐角忽略
    corner_ignore_enable: bool = False
    corner_angle_thr_deg: float = 35.0
    corner_ignore_span_mm: float = 2.0

    # 遮挡
    occ_enable: bool = True
    occ_polys: List[List[Tuple[float,float]]] = field(default_factory=lambda: [[(-50,-50), (30,-30), (30,200), (-50,200)]])
    occ_dilate_mm: float = 3.0
    occ_synthesize_band: bool = True
    occ_band_halfwidth_mm: Optional[float] = None

    # Guard
    guard_min_valid_ratio: float = 0.60
    guard_max_abs_p95_mm: float = 8.80
    guard_min_plane_inlier_ratio: float = 0.25
    guard_long_missing_max_mm: float = 20.0
    guard_grad_max_mm_per_mm: float = 0.08

    # 导出
    out_dir: str = "out"
    offset_csv: str = "out/offset_table.csv"
    corrected_gcode: str = "out/corrected.gcode"
    centerline_gcode: str = "out/centerline.gcode"
    export_centerline: bool = False
    preview_corrected: bool = True
    save_corrected_preview: bool = True
    dump_quicklook: bool = True
    dump_report: bool = True

    # ====== 新增：平面展平/拟合 ======
    plane_enable: bool = True
    plane_ransac_thresh_mm: float = 0.8
    plane_ransac_iters: int = 500
    plane_sample_cap: int = 120000

    # ====== 新增：可视化/调试 ======
    draw_normal_probes: bool = True
    arrow_stride: int = 12
    debug_normals_window: bool = True
    debug_normals_stride: int = 25
    debug_normals_max: int = 40
    debug_normals_len_mm: Optional[float] = None
    debug_normals_text: bool = True

    def to_core_params(self) -> Dict[str, Any]:
        """把 GUIConfig 合并到 core.PARAMS 的深拷贝里，返回新 dict。"""
        cfg = json.loads(json.dumps(core.PARAMS))  # 深复制（避免直接引用）
        cfg.update({
            "T_path": self.T_path,
            "gcode_path": self.gcode_path,
            "roi_mode": self.roi_mode,
            "cam_roi_xywh": list(self.cam_roi_xywh),
            "roi_center_xy": list(self.roi_center_xy),
            "roi_size_mm": float(self.roi_size_mm),
            "bounds_margin_mm": float(self.bounds_margin_mm),
            "pixel_size_mm": float(self.pixel_size_mm),
            # 最近表面
            "z_select": self.z_select,
            "nearest_use_percentile": bool(self.nearest_use_percentile),
            "nearest_qlo": float(self.nearest_qlo),
            "nearest_qhi": float(self.nearest_qhi),
            "depth_margin_mm": float(self.depth_margin_mm),
            "morph_open": int(self.morph_open),
            "morph_close": int(self.morph_close),
            "min_component_area_px": int(self.min_component_area_px),
            # 引导
            "guide_enable": bool(self.guide_enable),
            "guide_step_mm": float(self.guide_step_mm),
            "guide_halfwidth_mm": float(self.guide_halfwidth_mm),
            "guide_smooth_win": int(self.guide_smooth_win),
            "guide_max_offset_mm": float(self.guide_max_offset_mm),
            "guide_max_grad_mm_per_mm": float(self.guide_max_grad_mm_per_mm),
            "max_gap_pts": int(self.max_gap_pts),
            "curvature_adaptive": bool(self.curvature_adaptive),
            "curvature_gamma": float(self.curvature_gamma),
            "min_smooth_win": int(self.min_smooth_win),
            # 拐角
            "corner_ignore_enable": bool(self.corner_ignore_enable),
            "corner_angle_thr_deg": float(self.corner_angle_thr_deg),
            "corner_ignore_span_mm": float(self.corner_ignore_span_mm),
            # 遮挡
            "occlusion": {
                "enable": bool(self.occ_enable),
                "polys": [[(float(x), float(y)) for (x,y) in poly] for poly in (self.occ_polys or [])],
                "dilate_mm": float(self.occ_dilate_mm),
                "synthesize_band": bool(self.occ_synthesize_band),
                "band_halfwidth_mm": (None if self.occ_band_halfwidth_mm is None else float(self.occ_band_halfwidth_mm)),
            },
            # Guard
            "Guard": {
                "enable": True,
                "min_valid_ratio": float(self.guard_min_valid_ratio),
                "max_abs_p95_mm": float(self.guard_max_abs_p95_mm),
                "min_plane_inlier_ratio": float(self.guard_min_plane_inlier_ratio),
                "long_missing_max_mm": float(self.guard_long_missing_max_mm),
                "grad_max_mm_per_mm": float(self.guard_grad_max_mm_per_mm),
            },
            # 导出
            "out_dir": self.out_dir,
            "offset_csv": self.offset_csv,
            "corrected_gcode": self.corrected_gcode,
            "centerline_gcode": self.centerline_gcode,
            "export_centerline": bool(self.export_centerline),
            "preview_corrected": bool(self.preview_corrected),
            "save_corrected_preview": bool(self.save_corrected_preview),
            "dump_quicklook": bool(self.dump_quicklook),
            "dump_report": bool(self.dump_report),
            "bias_comp": {
                "enable": bool(self.bias_enable),
                "path": self.bias_path,
            },
            # 平面展平/拟合
            "plane_enable": bool(self.plane_enable),
            "plane_ransac_thresh_mm": float(self.plane_ransac_thresh_mm),
            "plane_ransac_iters": int(self.plane_ransac_iters),
            "plane_sample_cap": int(self.plane_sample_cap),

            # 调试/可视化（与 core.PARAMS 命名一致）
            "draw_normal_probes": bool(self.draw_normal_probes),
            "arrow_stride": int(self.arrow_stride),
            "debug_normals_window": bool(self.debug_normals_window),
            "debug_normals_stride": int(self.debug_normals_stride),
            "debug_normals_max": int(self.debug_normals_max),
            "debug_normals_len_mm": (None if self.debug_normals_len_mm is None else float(self.debug_normals_len_mm)),
            "debug_normals_text": bool(self.debug_normals_text),
        })
        return cfg

# ---------- 主控制器 ----------
class AlignController:
    def __init__(self, gui_cfg: Optional[GUIConfig] = None):
        self.cfg = gui_cfg or GUIConfig()
        self.stream: Optional[core.PCamMLSStream] = None
        self.R = None; self.t = None
        # 最近一次处理结果缓存（用于导出）
        self.last: Dict[str, Any] = {}

    # ---- 相机 ----
    def start_camera(self) -> str:
        """预先启动相机（阻塞式，建议在后台线程调用）。"""
        if self.stream is not None:
            return "camera_already_on"
        try:
            self.stream = core.PCamMLSStream()
            self.stream.open()
            return "camera_ready"
        except Exception as e:
            self.stream = None
            return f"camera_error: {e}"

    def read_frame(self, timeout_ms: int = 2000):
        if self.stream is None:
            raise RuntimeError("camera not started")
        return self.stream.read_pointcloud(timeout_ms)

    def close(self):
        try:
            if self.stream is not None:
                self.stream.close()
        finally:
            self.stream = None

    # ---- 单帧处理 ----
    def process_single_frame(self, *, for_export: bool = False) -> Dict[str, Any]:
        """抓取相机单帧并完整跑一次流水线，返回可视化与关键数据。"""
        P_cam, _ = self.read_frame(2000)
        if P_cam is None:
            raise RuntimeError("未获取到深度帧")
        cfg = self.cfg.to_core_params()

        # 1) 外参与 G 代码
        self.R, self.t, _ = core.load_extrinsic(cfg['T_path'])
        
        # 使用增强的G代码解析器
        try:
            from gcode_parser_patch import parse_gcode_xy_enhanced
            g_raw, feed = parse_gcode_xy_enhanced(cfg['gcode_path'], step_mm=cfg['guide_step_mm'])
            print(f"[INFO] 使用增强G代码解析器，解析到 {len(g_raw)} 个点")
        except ImportError:
            # 如果补丁不可用，使用原版解析器
            g_raw, feed = core.parse_gcode_xy(cfg['gcode_path'], step_mm=cfg['guide_step_mm'])
            print(f"[INFO] 使用原版G代码解析器，解析到 {len(g_raw)} 个点")
        step_mm = float(cfg['guide_step_mm'])
        g_xy = core.resample_polyline(g_raw, max(0.2, step_mm)) if g_raw.size > 0 else g_raw
        T_ref, N_ref = core.tangent_normal(g_xy) if g_xy.size > 0 else (np.zeros((0,2)), np.zeros((0,2)))
        s_ref = core.arc_length_s(g_xy)

        # 拐角忽略
        ignore_mask = None
        if cfg.get('corner_ignore_enable', False) and g_xy.size > 2:
            ignore_mask = core.compute_corner_ignore_mask(
                g_xy,
                angle_thr_deg=cfg.get('corner_angle_thr_deg', 35.0),
                span_mm=cfg.get('corner_ignore_span_mm', 2.0),
                step_mm=step_mm,
            )

        # 2) 相机坐标→机床坐标
        P_mach = core.transform_cam_to_machine_grid(P_cam, self.R, self.t)

        # 3) ROI
        H, W, _ = P_mach.shape
        m_valid = core.valid_mask_hw(P_mach)
        roi_mode = str(cfg.get('roi_mode', 'none')).lower()
        if roi_mode == 'camera_rect':
            m_roi = core.camera_rect_mask(H, W, tuple(cfg['cam_roi_xywh']))
            m_select = m_valid & m_roi
        elif roi_mode == 'machine':
            m_roi = core.machine_rect_mask(P_mach, tuple(cfg['roi_center_xy']), float(cfg['roi_size_mm']))
            m_select = m_valid & m_roi
        elif roi_mode == 'gcode_bounds' and g_xy.size>0:
            gx0,gy0 = g_xy.min(0); gx1,gy1 = g_xy.max(0)
            cx, cy = (gx0+gx1)*0.5, (gy0+gy1)*0.5
            sz = max(gx1-gx0, gy1-gy0) + cfg['bounds_margin_mm']*2
            m_roi = core.machine_rect_mask(P_mach, (cx,cy), sz)
            m_select = m_valid & m_roi
        else:
            m_select = m_valid

        # 4) 顶视边界与分辨率
        x0,x1,y0,y1 = core.compute_bounds_xy_from_mask(
            P_mach, m_select,
            cfg['bounds_qlo'], cfg['bounds_qhi'], cfg['bounds_margin_mm']
        )
        pix_mm = core.adjust_pixel_size(x0,x1,y0,y1, float(cfg['pixel_size_mm']), cfg['max_grid_pixels'])

        # 5) 顶视投影
        height, mask_top, origin_xy = core.project_topdown_from_grid(P_mach, m_select, pix_mm, (x0,x1,y0,y1))

        # 5.0 遮挡
        occ_cfg = cfg.get('occlusion', {})
        occ_top = None
        if bool(occ_cfg.get('enable', False)) and len(occ_cfg.get('polys', [])) > 0:
            Ht, Wt = height.shape
            occ_top = core._rasterize_polygons_topdown(
                occ_cfg.get('polys', []), origin_xy, pix_mm, Ht, Wt,
                dilate_mm=float(occ_cfg.get('dilate_mm', 0.0))
            )
            height[occ_top > 0] = np.nan
            mask_top[occ_top > 0] = 0

        # 5.1 平面展平
        plane = None; inlier_ratio = float('nan')
        if cfg.get('plane_enable', True) and np.isfinite(height).any():
            pts = P_mach[m_select & (np.isfinite(P_mach).all(axis=2))]
            plane, inlier_ratio, inl_mask = core._fit_plane_ransac(
                pts,
                thr=cfg['plane_ransac_thresh_mm'],
                iters=cfg['plane_ransac_iters'],
                sample_cap=cfg['plane_sample_cap'],
            )
            height_flat = core._flatten_height_with_plane(height, origin_xy, pix_mm, plane)
            src_for_nearest = height_flat
        else:
            height_flat = height.copy()
            src_for_nearest = height

        # 6) 最近表面掩码
        nearest_mask, z_ref, (z_low, z_high) = core.extract_nearest_surface_mask_from_height(
            src_for_nearest, (mask_top > 0),
            z_select=cfg['z_select'],
            depth_margin_mm=cfg['depth_margin_mm'],
            use_percentile=cfg['nearest_use_percentile'],
            qlo=cfg['nearest_qlo'],
            qhi=cfg['nearest_qhi'],
            morph_open=cfg['morph_open'],
            morph_close=cfg['morph_close'],
            min_component_area_px=cfg['min_component_area_px'],
        )

        # 6.1 遮挡内按 G 代码合成掩码
        if occ_top is not None and g_xy.size > 0 and bool(occ_cfg.get('synthesize_band', True)):
            nearest_mask = core._synthesize_mask_in_occlusion(
                nearest_mask, occ_top, g_xy, origin_xy, pix_mm,
                band_halfwidth_mm=occ_cfg.get('band_halfwidth_mm', None)
            )

        # 7) 引导中心线（严格一一对应）
        use_guided = bool(cfg.get('guide_enable', True)) and g_xy.size > 1
        if use_guided:
            centerline_xy, delta_n, valid_mask, rep = core.gcode_guided_centerline_strict(
                nearest_mask, origin_xy, pix_mm, g_xy, N_ref,
                halfwidth_mm=cfg['guide_halfwidth_mm'],
                base_smooth_win=cfg['guide_smooth_win'],
                max_abs_mm=cfg['guide_max_offset_mm'],
                max_gap_pts=cfg['max_gap_pts'],
                curvature_adaptive=cfg['curvature_adaptive'],
                curvature_gamma=cfg['curvature_gamma'],
                min_smooth_win=cfg['min_smooth_win'],
                ignore_mask=ignore_mask,
            )
            e_idx = np.arange(len(delta_n), dtype=int)
            e_n = np.nan_to_num(delta_n, nan=0.0)
        else:
            centerline_xy = g_xy.copy(); delta_n = np.zeros(len(g_xy), np.float32)
            valid_mask = np.zeros(len(g_xy), np.bool_)
            rep = dict(ratio=0.0)
            e_idx = np.arange(len(g_xy), dtype=int); e_n = np.zeros(len(g_xy), np.float32)

        # 8) 可视化拼装
        vis_top = core.render_topdown(height, mask_top, origin_xy, pix_mm, gcode_xy=g_xy)
        overlay = cv2.addWeighted(vis_top, 1.0, cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR), 0.25, 0)
        vis_cmp = core.draw_deviation_overlay(
            overlay, origin_xy, pix_mm,
            g_xy, centerline_xy,
            e_idx, e_n,
            arrow_stride=int(cfg['arrow_stride']),
            draw_probes=cfg.get('draw_normal_probes', True),
            N_ref=N_ref, valid_mask=valid_mask,
        )
        vis_cmp = core.draw_machine_axes_overlay(vis_cmp, origin_xy, pix_mm)

        # 统计/HUD/Guard
        dev_mean = float(np.mean(e_n[valid_mask])) if valid_mask.any() else 0.0
        dev_med  = float(np.median(e_n[valid_mask])) if valid_mask.any() else 0.0
        dev_p95  = float(np.percentile(np.abs(e_n[valid_mask]), 95)) if valid_mask.any() else 0.0
        valid_ratio = float(np.count_nonzero(valid_mask)) / max(1, g_xy.shape[0])
        plane_info = f"inlier={inlier_ratio:.2f}" if np.isfinite(inlier_ratio) else "inlier=nan"

        # 长缺失（把忽略点视作有效）
        eff_valid = valid_mask.copy()
        if ignore_mask is not None and len(ignore_mask) == len(eff_valid):
            eff_valid = eff_valid | ignore_mask
        runs = core.find_missing_runs(eff_valid)
        longest_pts = max([r-l+1 for (l,r) in runs], default=0)
        longest_mm = longest_pts * max(1e-9, float(cfg['guide_step_mm']))

        guard = cfg['Guard']; guard_ok = True; reasons = []
        if rep.get('ratio',0.0) < guard.get('min_valid_ratio', 0.60):
            guard_ok = False; reasons.append(f"valid_ratio {rep.get('ratio',0.0):.2f} < {guard.get('min_valid_ratio')}")
        if dev_p95 > guard.get('max_abs_p95_mm', 8.8):
            guard_ok = False; reasons.append(f"p95 {dev_p95:.2f} > {guard.get('max_abs_p95_mm')}")
        if np.isfinite(inlier_ratio) and inlier_ratio < guard.get('min_plane_inlier_ratio', 0.25):
            guard_ok = False; reasons.append(f"plane_inlier {inlier_ratio:.2f} < {guard.get('min_plane_inlier_ratio')}")
        if longest_mm > guard.get('long_missing_max_mm', 20.0):
            guard_ok = False; reasons.append(f"long_missing {longest_mm:.1f}mm over limit")
        # 梯度检查
        if len(delta_n) > 1 and len(s_ref) == len(delta_n):
            grad = np.abs(np.diff(delta_n)) / np.maximum(1e-9, np.diff(s_ref))
            gmax_chk = guard.get('grad_max_mm_per_mm', cfg.get('guide_max_grad_mm_per_mm', 0.08))
            if np.isfinite(grad).any() and float(np.nanpercentile(grad, 98)) > gmax_chk * 1.15:
                guard_ok = False; reasons.append("gradient @p98 over limit")

        txt = (
            'plane[%s]  band=[%.2f,%.2f]mm  pix=%.2fmm  dev(mean/med/p95)=%+.3f/%+.3f/%.3f  '
            'guided=%.2f  miss_long=%.1fmm  Guard=%s' % (
                plane_info,
                *((z_low, z_high) if np.isfinite(z_low) and np.isfinite(z_high) else (0.0, 0.0)),
                pix_mm, dev_mean, dev_med, dev_p95, rep.get('ratio', 0.0), longest_mm, "PASS" if guard_ok else "FAIL"
            )
        )
        cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,0) if guard_ok else (0,0,255), 1, cv2.LINE_AA)

        # Normal-Probes（少量）
        vis_probe = None
        try:
            vis_probe = core.render_normals_probe_window(
                overlay, origin_xy, pix_mm,
                g_xy, N_ref,
                delta_n=delta_n,
                valid_mask=(eff_valid),
                stride=int(core.PARAMS.get('debug_normals_stride', 25)),
                max_count=int(core.PARAMS.get('debug_normals_max', 40)),
                len_mm=core.PARAMS.get('debug_normals_len_mm', None),
                draw_text=bool(core.PARAMS.get('debug_normals_text', True)),
            )
        except Exception:
            pass

        # Bias 可视化直方图（可选）
        panel = None
        if self.cfg.bias_enable and g_xy.size > 1 and len(delta_n) == len(g_xy):
            try:
                with open(self.cfg.bias_path, 'r', encoding='utf-8') as f:
                    bc = json.load(f)
                bias = self._build_bias_vector(
                    bc, target_len=len(delta_n),
                    g_xy=g_xy, N_ref=N_ref, T_ref=T_ref,
                    ignore_mask=ignore_mask
                )
                delta_corr = delta_n.copy()
                delta_corr[np.isfinite(delta_corr)] -= bias[np.isfinite(delta_corr)]
                panel = core._render_biascomp_panel(np.asarray(delta_n, np.float32), np.asarray(delta_corr, np.float32),
                                                     title='BiasComp Δn (raw → corrected)', width=960, height=480, bins=48)
            except Exception:
                panel = None

        # 顶视图与最近表面（单独页显示）
        vis_top_only = core.render_topdown(height, mask_top, origin_xy, pix_mm, gcode_xy=None)
        nearest_vis = cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR)

        # 指标卡 - 改进：使用更有意义的轨迹偏离指标
        
        # 计算实际轨迹与理论轨迹的直线距离偏差
        if centerline_xy is not None and g_xy is not None and len(centerline_xy) == len(g_xy):
            # 计算实际中轴线与理论轨迹的欧几里得距离
            trajectory_distances = np.linalg.norm(centerline_xy - g_xy, axis=1)
            trajectory_distances_valid = trajectory_distances[valid_mask] if valid_mask.any() else trajectory_distances
            
            # 计算轨迹跟踪精度指标
            traj_mean_dist = float(np.mean(trajectory_distances_valid)) if len(trajectory_distances_valid) > 0 else 0.0
            traj_median_dist = float(np.median(trajectory_distances_valid)) if len(trajectory_distances_valid) > 0 else 0.0
            traj_p95_dist = float(np.percentile(trajectory_distances_valid, 95)) if len(trajectory_distances_valid) > 0 else 0.0
            traj_max_dist = float(np.max(trajectory_distances_valid)) if len(trajectory_distances_valid) > 0 else 0.0
            
            # 计算轨迹一致性（相邻点距离的标准差，反映轨迹平滑度）
            if len(trajectory_distances_valid) > 1:
                traj_consistency = float(np.std(trajectory_distances_valid))
            else:
                traj_consistency = 0.0
                
            # 计算轨迹覆盖率（有效测量点占总轨迹长度的比例）
            valid_ratio = float(np.count_nonzero(valid_mask)) / max(1, g_xy.shape[0])
            
        else:
            # 如果没有有效的轨迹数据，使用原有的法向偏移作为备用
            traj_mean_dist = abs(float(dev_mean)) if 'dev_mean' in locals() else 0.0
            traj_median_dist = abs(float(dev_med)) if 'dev_med' in locals() else 0.0
            traj_p95_dist = float(dev_p95) if 'dev_p95' in locals() else 0.0
            traj_max_dist = float(dev_p95) if 'dev_p95' in locals() else 0.0
            traj_consistency = 0.0
            valid_ratio = float(np.count_nonzero(valid_mask)) / max(1, len(g_xy)) if 'valid_mask' in locals() and g_xy is not None else 0.0
        
        metrics = dict(
            valid_ratio=valid_ratio,
            # 新的轨迹跟踪精度指标
            trajectory_mean_distance=traj_mean_dist,
            trajectory_median_distance=traj_median_dist, 
            trajectory_p95_distance=traj_p95_dist,
            trajectory_max_distance=traj_max_dist,
            trajectory_consistency=traj_consistency,
            # 保留原有指标作为详细信息（但不在主界面显示）
            dev_mean_raw=dev_mean if 'dev_mean' in locals() else 0.0,
            dev_median_raw=dev_med if 'dev_med' in locals() else 0.0,
            dev_p95_raw=dev_p95 if 'dev_p95' in locals() else 0.0,
            plane_inlier_ratio=float(inlier_ratio) if np.isfinite(inlier_ratio) else float('nan'),
            longest_missing_mm=float(longest_mm)
        )

        # --- 用“Bias Corrected”的 Δn 重绘 Centerline vs G-code（无黄线） ---
        vis_corr = None
        try:
            # 1) 计算经过 bias_comp.json 修正后的 delta（Δn_corrected）
            delta_corr = delta_n.copy()
            if cfg.get('bias_comp', {}).get('enable', False):
                with open(cfg['bias_comp']['path'], 'r', encoding='utf-8') as f:
                    bc = json.load(f)
                bias = self._build_bias_vector(
                    bc, target_len=len(delta_corr),
                    g_xy=g_xy, N_ref=N_ref, T_ref=T_ref,  # T_ref 此前已算过
                    ignore_mask=ignore_mask
                )
                m = np.isfinite(delta_corr)
                delta_corr[m] = delta_corr[m] - bias[m]

            # 2) 用修正后的 Δn 得到“修正后的实测中轴线”（不做导出那套平滑/限幅）
            e_n_corr = np.nan_to_num(delta_corr, nan=0.0)
            centerline_xy_corr = g_xy + N_ref * e_n_corr[:, None]

            # 3) 重画“Centerline vs G-code”叠加（与 vis_cmp 的画法一致，但用修正后的中心线与 Δn）
            vis_corr = core.draw_deviation_overlay(
                overlay, origin_xy, pix_mm,
                g_xy, centerline_xy_corr,
                e_idx, e_n_corr,
                arrow_stride=int(cfg['arrow_stride']),
                draw_probes=cfg.get('draw_normal_probes', True),
                N_ref=N_ref, valid_mask=valid_mask,
            )
            vis_corr = core.draw_machine_axes_overlay(vis_corr, origin_xy, pix_mm)

            # 可选：加一行小标题，便于区分（不需要黄线）
            cv2.putText(vis_corr, 'Centerline vs G-code (Bias Corrected)',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis_corr, 'Centerline vs G-code (Bias Corrected)',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        except Exception as _e:
            print('[WARN] build vis_corr (bias-corrected overlay) failed:', _e)

        # 缓存导出所需
        self.last = dict(
            cfg=cfg, g_xy=g_xy, N_ref=N_ref, s_ref=s_ref, delta_n=delta_n,
            valid_mask=valid_mask, ignore_mask=ignore_mask,
            origin_xy=origin_xy, pix_mm=pix_mm,
            vis_cmp=vis_cmp, vis_probe=vis_probe, hist_panel=panel,
            feed=feed,
            vis_top=vis_top_only,
            vis_nearest=nearest_vis,
            metrics=metrics,
            vis_corr=vis_corr,

        )

        return self.last

    def _build_bias_vector(self, bc: dict, target_len: int,
                           g_xy: np.ndarray, N_ref: np.ndarray, T_ref: Optional[np.ndarray],
                           ignore_mask: Optional[np.ndarray]) -> np.ndarray:
        mode = str(bc.get('mode', 'vector')).lower()
        if mode in ('per_index', 'table'):
            arr = np.asarray(bc.get('delta_bias_mm', []), np.float32).ravel()
            if arr.size == target_len:
                bias = arr
            else:
                # --- 分段重采样（优先） ---
                thr = float(self.cfg.corner_angle_thr_deg)
                if T_ref is None:
                    T_ref, _ = core.tangent_normal(g_xy)
                detect_knots = getattr(core, '_detect_corner_knots_from_tangent',
                                       _fallback_detect_corner_knots_from_tangent)
                k_cur = detect_knots(T_ref, ang_thresh_deg=thr, min_gap_pts=3)

                bias = None
                gpath = bc.get('gcode_path', '')
                step_cal = float(bc.get('guide_step_mm', self.cfg.guide_step_mm))
                if gpath:
                    # 使用增强的G代码解析器
                    try:
                        from gcode_parser_patch import parse_gcode_xy_enhanced
                        g_cal_raw, _ = parse_gcode_xy_enhanced(gpath, step_mm=step_cal)
                    except ImportError:
                        g_cal_raw, _ = core.parse_gcode_xy(gpath, step_mm=step_cal)
                    g_cal = core.resample_polyline(g_cal_raw, max(0.2, step_cal)) if g_cal_raw.size > 0 else g_cal_raw
                    if g_cal.size > 0:
                        T_cal, _ = core.tangent_normal(g_cal)
                        k_cal = detect_knots(T_cal, ang_thresh_deg=thr, min_gap_pts=3)
                        remap_knots = getattr(core, '_remap_knots_index', _fallback_remap_knots_index)
                        k_in = remap_knots(k_cal, len(g_cal), arr.size)
                        if len(k_in) == len(k_cur) and arr.size >= 2:
                            resample_piece = getattr(core, '_resample_per_index_bias_piecewise',
                                                     _fallback_resample_per_index_bias_piecewise)
                            bias = resample_piece(arr, target_len, k_in, k_cur)

                # 兜底：全长等比例重采样
                if bias is None:
                    bias = core._resample_per_index_bias_for_length(arr, target_len)
        else:
            v = np.asarray(bc.get('v', [0.0, 0.0]), np.float32)
            b0 = float(bc.get('b', 0.0))
            bias = (N_ref[:, 0] * v[0] + N_ref[:, 1] * v[1] + b0).astype(np.float32)

        # 与你的忽略策略一致：忽略区间置零
        if ignore_mask is not None and len(ignore_mask) == len(bias):
            bias = bias.copy();
            bias[ignore_mask] = 0.0
        return bias

    # ---- 导出纠偏 ----
    def export_corrected(self) -> Dict[str, Any]:
        if not self.last:
            raise RuntimeError("请先进行一次预览处理（获得当前帧的偏移）")
        L = self.last
        cfg = L['cfg']
        g_xy = L['g_xy']; N_ref = L['N_ref']; s_ref = L['s_ref']
        delta_n_meas = L['delta_n'].copy()
        ignore_mask = L.get('ignore_mask')
        feed = L.get('feed', None)

        # 偏差补偿（在 auto_flip 前）
        if cfg.get('bias_comp',{}).get('enable', False):
            try:
                with open(cfg['bias_comp']['path'], 'r', encoding='utf-8') as f:
                    bc = json.load(f)

                bias = self._build_bias_vector(
                    bc, target_len=len(delta_n_meas),
                    g_xy=g_xy, N_ref=N_ref, T_ref=None,  # T_ref 导出时可现算
                    ignore_mask=ignore_mask
                )
                m = np.isfinite(delta_n_meas)
                delta_n_meas[m] = delta_n_meas[m] - bias[m]
                # ↑ 不要重复减 bias
            except Exception as e:
                print("[BIAS] skip:", e)

        # 自动符号判定
        if len(delta_n_meas) > 5 and np.isfinite(delta_n_meas).any():
            med_dn = float(np.nanmedian(delta_n_meas))
            raw_dy = float(np.median((g_xy + N_ref * np.nan_to_num(delta_n_meas)[:,None])[:min(64,len(g_xy)),1] - g_xy[:min(64,len(g_xy)),1]))
            if abs(raw_dy) > 0.01 and abs(med_dn) > 0.01 and raw_dy * med_dn < 0:
                delta_n_meas = -delta_n_meas

        # 应用模式：默认 invert（朝向理论轨迹纠偏）
        apply_sign = -1.0
        delta_apply_raw = apply_sign * delta_n_meas

        # 分段平滑/限幅/梯度限幅
        kappa = core.curvature_kappa(g_xy)
        delta_apply = core.postprocess_delta_segmentwise_for_export(
            delta_apply_raw,
            ignore_mask=ignore_mask if (ignore_mask is not None and len(ignore_mask) == len(delta_apply_raw)) else None,
            s=s_ref, kappa=kappa,
            base_win=cfg['guide_smooth_win'],
            curvature_adaptive=cfg['curvature_adaptive'],
            curvature_gamma=cfg['curvature_gamma'],
            win_min=cfg['min_smooth_win'],
            max_abs_mm=cfg['guide_max_offset_mm'],
            grad_max_mm_per_mm=cfg['guide_max_grad_mm_per_mm'],
            max_gap_pts=cfg['max_gap_pts'],
        )

        g_xy_corr = g_xy + N_ref * delta_apply[:,None]
        dxy_vec = (N_ref * delta_apply[:,None])

        # 保存
        out_dir = cfg['out_dir']; os.makedirs(out_dir, exist_ok=True)
        core.save_offset_csv(s_ref, delta_apply, dxy_vec, cfg['offset_csv'])
        core.write_linear_gcode(g_xy_corr, cfg['corrected_gcode'], feed=feed)
        if cfg.get('export_centerline', False):
            core.write_linear_gcode((g_xy + N_ref * np.nan_to_num(L['delta_n'])[:,None])[:len(g_xy_corr)], cfg['centerline_gcode'], feed=feed)

        # 预览图保存
        if cfg.get('preview_corrected', True) and L.get('vis_cmp') is not None:
            try:
                vis_prev = L['vis_cmp'].copy()
                Hprev, Wprev = vis_prev.shape[:2]
                def xy_to_px(xy_arr):
                    if xy_arr.size == 0: return np.empty((0,2), int)
                    x0,y0 = L['origin_xy']; pix_mm = L['pix_mm']; y1 = y0 + Hprev * pix_mm
                    xs = np.clip(((xy_arr[:,0]-x0)/pix_mm).astype(int), 0, Wprev-1)
                    ys = np.clip(((y1 - xy_arr[:,1])/pix_mm).astype(int), 0, Hprev-1)
                    return np.stack([xs,ys], axis=1)
                pts_corr = xy_to_px(g_xy_corr)
                for ii in range(len(pts_corr)-1):
                    cv2.line(vis_prev, tuple(pts_corr[ii]), tuple(pts_corr[ii+1]), (0,255,255), 2, cv2.LINE_AA)
                cv2.putText(vis_prev, 'corrected preview', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis_prev, 'corrected preview', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
                prev_path = os.path.join(out_dir, 'corrected_preview.png')
                cv2.imwrite(prev_path, vis_prev)
            except Exception as e:
                print('[WARN] preview save failed:', e)

        # quicklook & report
        if cfg.get('dump_quicklook', True) and L.get('vis_cmp') is not None:
            try:
                hist = core._render_histogram(delta_apply, title='applied delta_n histogram (mm)')
                H, W = L['vis_cmp'].shape[:2]
                quick = core._compose_quicklook(L['vis_cmp'], None, hist, hud_text='Exported: offsets & corrected.gcode')
                qp = os.path.join(out_dir, 'quicklook.png')
                cv2.imwrite(qp, quick)
            except Exception:
                pass
        if cfg.get('dump_report', True):
            try:
                rep_json = dict(
                    valid_ratio=float(np.mean(L['valid_mask'])) if len(L.get('valid_mask',[]))>0 else 0.0,
                    dev_p95=float(np.nanpercentile(np.abs(L['delta_n']), 95)) if np.isfinite(L['delta_n']).any() else 0.0,
                    gcode_points=int(len(g_xy)),
                    pixel_size_mm=float(L['pix_mm']),
                )
                rp = os.path.join(out_dir, 'report.json')
                with open(rp, 'w', encoding='utf-8') as f:
                    json.dump(rep_json, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return dict(
            offset_csv=cfg['offset_csv'],
            corrected_gcode=cfg['corrected_gcode'],
            out_dir=out_dir,
        )

    # ---- 保存 bias_comp（per_index） ----
    def save_bias_from_current(self) -> str:
        if not self.last:
            raise RuntimeError("请先预览一次以获得当前帧的 Δn")
        L = self.last
        cfg = L['cfg']
        outp = cfg.get('bias_comp',{}).get('path','bias_comp.json')
        delta_n = L['delta_n']
        ignore_mask = L.get('ignore_mask')
        data = {
            "version": "bias_comp/v1",
            "mode": "per_index",
            "guide_step_mm": float(cfg.get('guide_step_mm', 1.0)),
            "n_points": int(len(delta_n)),
            "corner_ignore": bool(cfg.get('corner_ignore_enable', False)),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gcode_path": cfg.get('gcode_path', ''),
            "delta_bias_mm": np.nan_to_num(np.asarray(delta_n, dtype=float), nan=0.0).tolist(),
        }
        if ignore_mask is not None and len(ignore_mask) == len(delta_n):
            data["ignored_indices"] = np.where(ignore_mask)[0].astype(int).tolist()
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return outp

