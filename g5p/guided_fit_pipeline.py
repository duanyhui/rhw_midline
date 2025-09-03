# -*- coding: utf-8 -*-
"""
guided_fit_pipeline.py  (enhanced)
----------------------------------
相对上一版，新增：
- compute_arclength()
- align_external_offsets()：把外部偏差（按 s 或按点数）对齐到 g_xy
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import json

try:
    import align_centerline_to_gcode_pro_edit as core
except Exception as e:
    raise SystemExit("请将本文件与 align_centerline_to_gcode_pro_edit.py 放在同一目录。\n原始错误: {}".format(e))

class GuidedFitEngine:
    def __init__(self):
        self.stream = None
        self.last_g = None
        self.last_N = None
        self.last_feed = None

    def open_camera(self):
        if self.stream is None:
            self.stream = core.PCamMLSStream()
            self.stream.open()
        return True

    def close_camera(self):
        if self.stream is not None:
            try: self.stream.close()
            except Exception: pass
            self.stream = None

    def run_once(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        R, t, _ = core.load_extrinsic(cfg['T_path'])
        g_raw, feed = core.parse_gcode_xy(cfg['gcode_path'], step_mm=cfg['guide_step_mm'])
        self.last_feed = feed

        step_mm = float(cfg.get('guide_step_mm', 1.0))
        if g_raw.size > 0:
            g_xy = core.resample_polyline(g_raw, max(0.2, step_mm))
            ref_tree = core.build_kdtree(g_xy)
            T_ref, N_ref = core.tangent_normal(g_xy)
        else:
            g_xy = g_raw; ref_tree = None
            T_ref = np.zeros((0,2)); N_ref = np.zeros((0,2))
        self.last_g = g_xy; self.last_N = N_ref

        if self.stream is None:
            self.open_camera()
        P_cam, _ = self.stream.read_pointcloud(2000)
        if P_cam is None: raise RuntimeError("未读取到深度帧")
        H, W, _ = P_cam.shape
        P_mach = core.transform_cam_to_machine_grid(P_cam, R, t)

        # ROI
        roi_mode = str(cfg.get('roi_mode', 'none')).lower()
        m_valid = core.valid_mask_hw(P_mach)
        if roi_mode == 'camera_rect':
            m_roi = core.camera_rect_mask(H, W, cfg['cam_roi_xywh'])
            m_select = m_valid & m_roi
        elif roi_mode == 'machine':
            m_roi = core.machine_rect_mask(P_mach, cfg['roi_center_xy'], cfg['roi_size_mm'])
            m_select = m_valid & m_roi
        elif roi_mode == 'gcode_bounds' and g_xy.size > 0:
            gx0, gy0 = g_xy.min(0); gx1, gy1 = g_xy.max(0)
            cx, cy = (gx0+gx1)*0.5, (gy0+gy1)*0.5
            sz = max(gx1-gx0, gy1-gy0) + float(cfg['bounds_margin_mm'])*2.0
            m_roi = core.machine_rect_mask(P_mach, (cx, cy), sz)
            m_select = m_valid & m_roi
        else:
            m_select = m_valid

        # 顶视投影范围 & 分辨率
        x0,x1,y0,y1 = core.compute_bounds_xy_from_mask(
            P_mach, m_select, cfg['bounds_qlo'], cfg['bounds_qhi'], cfg['bounds_margin_mm'])
        pix_mm = core.adjust_pixel_size(x0, x1, y0, y1, float(cfg['pixel_size_mm']), cfg['max_grid_pixels'])

        # 顶视（高度图）
        height, mask_top, origin_xy = core.project_topdown_from_grid(P_mach, m_select, pix_mm, (x0,x1,y0,y1))

        # 平面拟合/展平
        plane = None; inlier_ratio = float('nan')
        if bool(cfg.get('plane_enable', True)) and np.isfinite(height).any():
            pts = P_mach[m_select & (np.isfinite(P_mach).all(axis=2))]
            plane, inlier_ratio, _ = core._fit_plane_ransac(
                pts, thr=cfg['plane_ransac_thresh_mm'],
                iters=cfg['plane_ransac_iters'],
                sample_cap=cfg['plane_sample_cap']
            )
            height_flat = core._flatten_height_with_plane(height, origin_xy, pix_mm, plane)
            src_for_nearest = height_flat
        else:
            height_flat = height.copy(); src_for_nearest = height

        # 最近层
        nearest_mask, z_ref, (z_low, z_high) = core.extract_nearest_surface_mask_from_height(
            src_for_nearest, (mask_top > 0),
            z_select=cfg['z_select'], depth_margin_mm=cfg['depth_margin_mm'],
            use_percentile=cfg['nearest_use_percentile'],
            qlo=cfg['nearest_qlo'], qhi=cfg['nearest_qhi'],
            morph_open=cfg['morph_open'], morph_close=cfg['morph_close'],
            min_component_area_px=cfg['min_component_area_px']
        )

        # 骨架 & 引导中心线
        skel_bgr = core.extract_skeleton_universal(nearest_mask, visualize=True)
        if skel_bgr is None: raise RuntimeError("最近层不足以生成骨架。")
        skel_gray = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY)
        path_px = core.skeleton_to_path_px_topo(skel_gray)

        use_guided = bool(cfg.get('guide_enable', True)) and g_xy.size > 1
        if use_guided:
            centerline_xy, delta_n, rep = core.gcode_guided_centerline_v2(
                nearest_mask, origin_xy, pix_mm, g_xy, N_ref,
                halfwidth_mm=cfg['guide_halfwidth_mm'],
                smooth_win=cfg['guide_smooth_win'],
                max_abs_mm=cfg['guide_max_offset_mm']
            )
            if rep['ratio'] < cfg['guide_min_valid_ratio']:
                centerline_xy = core.px_to_mach_xy(path_px, origin_xy, pix_mm, height.shape[0])
                e_idx, _, e_n = core.project_points_to_path(centerline_xy, g_xy, core.build_kdtree(g_xy), N_ref)
            else:
                e_idx = np.arange(len(delta_n), dtype=int)
                e_n = delta_n
        else:
            centerline_xy = core.px_to_mach_xy(path_px, origin_xy, pix_mm, height.shape[0])
            e_idx, _, e_n = core.project_points_to_path(centerline_xy, g_xy, core.build_kdtree(g_xy), N_ref)

        # 可视化
        vis_top_raw = core.render_topdown(height, mask_top, origin_xy, pix_mm, gcode_xy=g_xy)
        vis_overlay = cv2.addWeighted(vis_top_raw, 1.0, cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR), 0.25, 0)
        vis_cmp = core.draw_deviation_overlay(
            vis_overlay, origin_xy, pix_mm, g_xy, centerline_xy, e_idx, e_n,
            arrow_stride=int(cfg['arrow_stride']), draw_probes=bool(cfg.get('draw_normal_probes', True)),
            N_ref=N_ref
        )
        vis_cmp = core.draw_machine_axes_overlay(vis_cmp, origin_xy, pix_mm)

        dev_mean = float(np.mean(e_n)) if e_n.size > 0 else 0.0
        dev_med  = float(np.median(e_n)) if e_n.size > 0 else 0.0
        dev_p95  = float(np.percentile(np.abs(e_n), 95)) if e_n.size > 0 else 0.0
        valid_ratio = float(len(e_n)) / max(1, g_xy.shape[0]) if e_n.size>0 else 0.0
        stats = dict(dev_mean=dev_mean, dev_median=dev_med, dev_p95=dev_p95,
                     valid_ratio=valid_ratio, plane_inlier_ratio=inlier_ratio)

        return dict(
            vis_base=vis_cmp, origin_xy=origin_xy, pix_mm=float(pix_mm),
            height_shape=height.shape, g_xy=g_xy, N_ref=N_ref, centerline_xy=centerline_xy,
            e_idx=e_idx, e_n=e_n, nearest_mask=nearest_mask, stats=stats, feed=feed
        )

    def export_offsets_and_gcode(self, result: Dict[str, Any], cfg: Dict[str, Any],
                                 out_dir: Optional[Path]=None) -> Dict[str, Any]:
        if result is None or result.get('g_xy') is None or result.get('centerline_xy') is None:
            raise ValueError("缺少必要数据：请先运行一次并得到中心线。")
        g_xy: np.ndarray = result['g_xy']
        centerline_xy: np.ndarray = result['centerline_xy']
        N_ref: np.ndarray = result['N_ref']

        # 测得偏移
        delta_n_meas, dxy_vec = core.compute_offsets_along_gcode(centerline_xy, g_xy, N_ref)
        if delta_n_meas.size == 0:
            raise RuntimeError("无法计算偏移：中心线与参考轨迹重叠度不足。")

        if bool(cfg.get('auto_flip_offset', True)):
            Mchk = min(len(delta_n_meas), len(centerline_xy), len(g_xy))
            if Mchk > 5:
                raw_dy = float(np.median(centerline_xy[:Mchk,1] - g_xy[:Mchk,1]))
                med_dn = float(np.median(delta_n_meas[:Mchk])) if np.isfinite(delta_n_meas).any() else 0.0
                if abs(raw_dy) > 0.01 and abs(med_dn) > 0.01 and raw_dy * med_dn < 0:
                    delta_n_meas = -delta_n_meas

        mode = str(cfg.get('offset_apply_mode', 'invert')).lower()
        apply_sign = -1.0 if mode in ('invert','opposite','opp','toward_target','correct') else 1.0
        delta_n_apply = apply_sign * delta_n_meas

        delta_n_apply = np.clip(core.moving_average_1d(delta_n_apply, cfg['guide_smooth_win']),
                                -cfg['guide_max_offset_mm'], cfg['guide_max_offset_mm'])
        M = len(delta_n_apply)
        g_xy_corr = g_xy[:M] + result['N_ref'][:M] * delta_n_apply[:, None]

        seg = np.linalg.norm(np.diff(g_xy[:M], axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])

        out_dir = Path(out_dir or cfg.get('out_dir', 'out'))
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = Path(cfg.get('offset_csv', out_dir / 'offset_table.csv'))
        gcode_path = Path(cfg.get('corrected_gcode', out_dir / 'corrected.gcode'))

        core.save_offset_csv(s, delta_n_apply, result['N_ref'][:M]*delta_n_apply[:,None], csv_path)
        core.write_linear_gcode(g_xy_corr, gcode_path, feed=self.last_feed)

        return dict(csv_path=str(csv_path), gcode_path=str(gcode_path),
                    g_xy_corr=g_xy_corr, delta_n_apply=delta_n_apply, s=s)

# -------------------- 工具函数 --------------------
def compute_arclength(xy: np.ndarray) -> np.ndarray:
    if xy is None or xy.shape[0] < 2:
        return np.zeros((0,), float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])

def align_external_offsets(g_xy: np.ndarray, N_ref: np.ndarray,
                           packet, apply_mode: str = 'as_is') -> Tuple[np.ndarray, np.ndarray]:
    """
    将外部传入的偏差对齐到 g_xy：
    - 若提供 s：按 s 对齐插值到 len(g_xy)；
    - 否则若长度等于 len(g_xy)：按点一一对应；
    - 否则：按均匀弧长插值。
    返回：delta_n_apply, g_xy_corr
    """
    import numpy as np
    s_ref = compute_arclength(g_xy)
    M = len(g_xy)
    if packet.mode == 'delta_xy' and packet.delta_xy is not None:
        dxy = np.asarray(packet.delta_xy, float)
        if packet.s is not None and len(packet.s) == len(dxy):
            dn = np.interp(s_ref, packet.s, (dxy * N_ref[:len(dxy)]).sum(1),
                           left=(dxy[0]*N_ref[0]).sum(), right=(dxy[-1]*N_ref[min(len(N_ref)-1,len(dxy)-1)]).sum())
        else:
            if len(dxy) != M:
                idx = np.linspace(0, M-1, len(dxy))
                dn = np.interp(np.arange(M), idx, (dxy * N_ref[:len(dxy)]).sum(1))
            else:
                dn = (dxy * N_ref[:M]).sum(1)
    else:  # delta_n
        dn_src = np.asarray(packet.delta_n, float)
        if packet.s is not None and len(packet.s) == len(dn_src):
            dn = np.interp(s_ref, packet.s, dn_src, left=dn_src[0], right=dn_src[-1])
        else:
            if len(dn_src) != M:
                idx = np.linspace(0, M-1, len(dn_src))
                dn = np.interp(np.arange(M), idx, dn_src)
            else:
                dn = dn_src

    if str(apply_mode).lower() in ('invert','opposite','opp','toward_target','correct'):
        dn = -dn

    g_xy_corr = g_xy[:M] + N_ref[:M] * dn[:M, None]
    return dn[:M], g_xy_corr[:M]

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
