#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias Compensation Calibrator
----------------------------
用于“事先标定并保存”偏移向量/常数补偿（v / b），并在主流程中加载使用。
复用 align_centerline_to_gcode_pro_edit_max.py 里的几何/取样/法向匹配逻辑，确保一致性。

快捷键：
  A  累积当前帧样本（valid & 非拐角）到样本池
  C  清空样本池
  M  切换模型：auto → vector → scalar
  W  保存补偿到 JSON（默认 out/bias_comp.json）
  V  预览开关：显示/隐藏（vector / scalar / both）叠加效果
  Q  退出
"""

import json, time, math, os
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import cv2

# === 导入你现有主程序（仅作为“库”使用，不会触发 run()） ===
import align_centerline_to_gcode_pro_edit_max as main  # 同目录导入

# ----------------------- 参数（可按需修改） -----------------------
CFG = dict(
    bias_json='out/bias_comp.json',    # 保存路径
    kappa_gamma=35.0,                  # 曲率减权强度（用于稳健估计）
    clip_abs_mm=3.0,                   # 参与拟合的 δ 绝对值上限（鲁棒性）
    corner_alpha_span_mm=None,         # 角点渐隐半径(None=跟随主程序 corner_ignore_span_mm)
    min_samples=80,                    # 最小样本数
    model='auto',                      # 'auto' | 'vector' | 'scalar'
    show_preview=True,                 # 是否在顶视图上叠加三种补偿预览
)

# ----------------------- 工具：估计与评分 -----------------------
def _r2_score(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size < 2 or not np.isfinite(y).any(): return 0.0
    ss_res = float(np.nansum((y - yhat)**2))
    ss_tot = float(np.nansum((y - np.nanmean(y))**2)) + 1e-12
    return max(0.0, 1.0 - ss_res/ss_tot)

def estimate_bias(N: np.ndarray, delta: np.ndarray, kappa: Optional[np.ndarray]=None,
                  clip_abs_mm: float=3.0, gamma: float=35.0) -> Dict:
    """
    输入：
      N      : (n,2) 法向
      delta  : (n,)  法向位移
      kappa  : (n,)  曲率（可选，用于减权）
    返回：
      dict(mode_best, v, b, r2_vector, r2_scalar, used_n, used_ratio)
    """
    N = np.asarray(N, float)
    z = np.asarray(delta, float).copy()

    # 1) 鲁棒裁剪（去掉极端 5% 或 clip_abs_mm 限幅）
    if np.isfinite(z).any():
        q = np.nanpercentile(np.abs(z), 95)
        thr = float(min(q, clip_abs_mm))
        z = np.clip(z, -thr, +thr)

    # 2) 曲率减权（高曲率减小权重）
    if kappa is not None and kappa.size == z.size:
        w = 1.0 / (1.0 + float(gamma) * np.asarray(kappa, float))
    else:
        w = np.ones_like(z, float)

    # 3) 去除 NaN/Inf
    good = np.isfinite(z) & np.isfinite(N).all(axis=1) & (w > 1e-9)
    N = N[good]; z = z[good]; w = w[good]
    used_n = int(len(z))

    result = dict(v=[0.0,0.0], b=0.0, r2_vector=0.0, r2_scalar=0.0,
                  used_n=used_n, used_ratio= float(used_n))

    if used_n < 4:
        return result

    # 4) 向量模型：z ≈ N · v
    A = N.copy()
    # 加权最小二乘：等价于对 sqrt(w)A, sqrt(w)z 做最小二乘
    sw = np.sqrt(w)[:,None]
    try:
        v_hat, *_ = np.linalg.lstsq(A*sw, z*sw[:,0], rcond=None)
        z_vec = (N @ v_hat)
        r2_vec = _r2_score(z, z_vec)
    except Exception:
        v_hat = np.zeros(2, float); z_vec = np.zeros_like(z); r2_vec = 0.0

    # 5) 标量模型：z ≈ b（用加权中位数近似）
    #   简化：直接取中位数（较稳健）；也可改为加权中位数
    b_hat = float(np.median(z))
    z_sca = np.full_like(z, b_hat)
    r2_sca = _r2_score(z, z_sca)

    mode_best = 'vector' if r2_vec >= r2_sca else 'scalar'
    result.update(mode_best=mode_best, v=[float(v_hat[0]), float(v_hat[1])],
                  b=float(b_hat), r2_vector=float(r2_vec), r2_scalar=float(r2_sca))
    return result

# ----------------------- 可视化：散点/玫瑰/直方 -----------------------
def render_bias_charts(N: np.ndarray, delta: np.ndarray,
                       est: Dict, size=(560, 520)) -> np.ndarray:
    W,H = size[0], size[1]
    canvas = np.full((H, W, 3), 20, np.uint8)

    if N.size == 0 or delta.size == 0:
        cv2.putText(canvas, 'No samples yet', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)
        return canvas

    # θ = 法向角度
    theta = np.arctan2(N[:,1], N[:,0])
    z = delta

    # --- 左：δ-θ 散点 ---
    plot = np.full((H, W//2, 3), 25, np.uint8)
    # 坐标换算
    pad = 40
    ww, hh = W//2 - 2*pad, H - 2*pad
    # y 轴范围
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    lo, hi = float(lo), float(hi)
    if hi - lo < 1e-6:
        hi, lo = max(hi,0.5), min(lo,-0.5)

    def to_px(t, zz):
        x = pad + ( (t + math.pi) / (2*math.pi) ) * ww
        y = pad + (1.0 - ( (zz - lo) / max(1e-6, (hi - lo)) )) * hh
        return int(x), int(y)

    # 网格
    for k in range(0, 361, 30):
        x,_ = to_px(math.radians(k-180), (lo+hi)/2)
        cv2.line(plot, (x, pad), (x, pad+hh), (60,60,60), 1, cv2.LINE_AA)
    for t in np.linspace(lo, hi, 5):
        _,y = to_px(0, t)
        cv2.line(plot, (pad, y), (pad+ww, y), (60,60,60), 1, cv2.LINE_AA)
    # 散点
    for t, zz in zip(theta, z):
        x,y = to_px(t, zz)
        cv2.circle(plot, (x,y), 2, (80,200,255), -1, cv2.LINE_AA)

    # 画预测曲线
    # 向量：z_hat = N·v = |v| cos(theta - phi)
    v = np.array(est.get('v', [0.0,0.0]), float)
    if np.linalg.norm(v) > 1e-9:
        phi = math.atan2(v[1], v[0])
        amp = np.linalg.norm(v)
        xs = np.linspace(-math.pi, math.pi, 360)
        ys = amp * np.cos(xs - phi)
        pts = [to_px(x,y) for x,y in zip(xs, ys)]
        cv2.polylines(plot, [np.array(pts, np.int32)], False, (0,255,180), 2, cv2.LINE_AA)
    # 标量：一条水平线
    b = float(est.get('b', 0.0))
    _, yb = to_px(0, b)
    cv2.line(plot, (pad, yb), (pad+ww, yb), (0,165,255), 2, cv2.LINE_AA)

    cv2.putText(plot, 'delta vs theta', (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

    # --- 右：极坐标玫瑰 + 直方图 ---
    right = np.full((H, W - W//2, 3), 25, np.uint8)
    cx, cy = (right.shape[1]//2, H//2)
    R = int(min(cx, cy) * 0.9)
    cv2.circle(right, (cx,cy), R, (60,60,60), 1, cv2.LINE_AA)
    # 玫瑰：按角度分箱，长度=对应箱内 |delta| 的均值
    bins = 24
    edges = np.linspace(-math.pi, math.pi, bins+1)
    idx = np.digitize(theta, edges) - 1
    for k in range(bins):
        sel = (idx == k)
        if not np.any(sel): continue
        val = float(np.mean(np.abs(z[sel])))
        r = int(min(R, (val / max(1e-6, np.percentile(np.abs(z), 98))) * R))
        ang = (edges[k] + edges[k+1]) * 0.5
        x = cx + int(r * math.cos(ang))
        y = cy + int(r * math.sin(ang))
        cv2.line(right, (cx,cy), (x,y), (80,200,255), 2, cv2.LINE_AA)
    cv2.putText(right, 'polar rose |delta|', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2)

    # 组合
    canvas[:, :W//2] = plot
    canvas[:, W//2:] = right

    # HUD
    txt = (f"mode*={est.get('mode_best')}  "
           f"v=({est.get('v',[0,0])[0]:+.3f},{est.get('v',[0,0])[1]:+.3f})  "
           f"b={est.get('b',0.0):+.3f}  "
           f"R2(vec/sca)={est.get('r2_vector',0.0):.2f}/{est.get('r2_scalar',0.0):.2f}  "
           f"samples={est.get('used_n',0)}")
    cv2.putText(canvas, txt, (14, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(canvas, txt, (14, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    return canvas

# ----------------------- 主流程：采样 → 拟合 → 可视化 → 存档 -----------------------
def run():
    P = main.PARAMS  # 直接使用主程序参数
    out_dir = Path(P['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 外参与 G 代码
    R, t, _ = main.load_extrinsic(P['T_path'])
    g_raw, feed = main.parse_gcode_xy(P['gcode_path'], step_mm=P['guide_step_mm'])
    step_mm = float(P.get('guide_step_mm', 1.0))
    g_xy  = main.resample_polyline(g_raw, max(0.2, step_mm)) if g_raw.size > 0 else g_raw
    T_ref, N_ref = main.tangent_normal(g_xy) if g_xy.size > 0 else (np.zeros((0,2)), np.zeros((0,2)))
    s_ref = main.arc_length_s(g_xy)

    # 2) 角点忽略掩码（与主程序保持一致）
    if P.get('corner_ignore_enable', True) and g_xy.size > 2:
        ignore_mask = main.compute_corner_ignore_mask(
            g_xy,
            angle_thr_deg=P.get('corner_angle_thr_deg', 35.0),
            span_mm=P.get('corner_ignore_span_mm', 6.0),
            step_mm=step_mm
        )
    else:
        ignore_mask = None

    # 3) 打开相机
    stream = main.PCamMLSStream(); stream.open()

    # 样本池
    S_N = []      # (nx, ny)
    S_dn = []     # delta_n
    S_kappa = []  # 曲率
    kappa = main.curvature_kappa(g_xy)

    frame = 0
    preview_on = bool(CFG.get('show_preview', True))
    model_sel = str(CFG.get('model','auto')).lower()

    print('[BiasCal] 按 A 累积样本, W 保存, C 清空, M 切换模型, V 预览开关, Q 退出')
    try:
        while True:
            # === 拉一帧，构建顶视 & 最近表面 ===
            P_cam, _ = stream.read_pointcloud(2000)
            if P_cam is None:
                print('[BiasCal] 无深度帧'); continue
            H, W, _ = P_cam.shape
            P_mach = main.transform_cam_to_machine_grid(P_cam, R, t)

            # ROI (沿用主程序 gcode_bounds)
            m_valid = main.valid_mask_hw(P_mach)
            gx0,gy0 = g_xy.min(0); gx1,gy1 = g_xy.max(0)
            cx, cy = (gx0+gx1)*0.5, (gy0+gy1)*0.5
            sz = max(gx1-gx0, gy1-gy0) + P['bounds_margin_mm']*2
            m_roi = main.machine_rect_mask(P_mach, (cx,cy), sz)
            m_select = m_valid & m_roi

            x0,x1,y0,y1 = main.compute_bounds_xy_from_mask(P_mach, m_select,
                                                           P['bounds_qlo'], P['bounds_qhi'], P['bounds_margin_mm'])
            pix_mm = main.adjust_pixel_size(x0,x1,y0,y1, float(P['pixel_size_mm']), P['max_grid_pixels'])
            height, mask_top, origin_xy = main.project_topdown_from_grid(P_mach, m_select, pix_mm, (x0,x1,y0,y1))

            # 平面展平 + 最近表面掩码
            if P.get('plane_enable', True) and np.isfinite(height).any():
                pts = P_mach[m_select & (np.isfinite(P_mach).all(axis=2))]
                plane, inlier_ratio, _ = main._fit_plane_ransac(pts,
                    thr=P['plane_ransac_thresh_mm'],
                    iters=P['plane_ransac_iters'],
                    sample_cap=P['plane_sample_cap'])
                height_flat = main._flatten_height_with_plane(height, origin_xy, pix_mm, plane)
                src_for_nearest = height_flat
            else:
                src_for_nearest = height
                inlier_ratio = float('nan')

            nearest_mask, z_ref, (z_lo, z_hi) = main.extract_nearest_surface_mask_from_height(
                src_for_nearest, (mask_top > 0),
                z_select=P['z_select'],
                depth_margin_mm=P['depth_margin_mm'],
                use_percentile=P['nearest_use_percentile'],
                qlo=P['nearest_qlo'], qhi=P['nearest_qhi'],
                morph_open=P['morph_open'], morph_close=P['morph_close'],
                min_component_area_px=P['min_component_area_px']
            )

            # 严格法向单点决策 —— 拿到 delta_n 与 valid_mask
            centerline_xy, delta_n, valid_mask, rep = main.gcode_guided_centerline_strict(
                nearest_mask, origin_xy, pix_mm, g_xy, N_ref,
                halfwidth_mm=P['guide_halfwidth_mm'],
                base_smooth_win=P['guide_smooth_win'],
                max_abs_mm=P['guide_max_offset_mm'],
                max_gap_pts=P['max_gap_pts'],
                curvature_adaptive=P['curvature_adaptive'],
                curvature_gamma=P['curvature_gamma'],
                min_smooth_win=P['min_smooth_win'],
                ignore_mask=ignore_mask
            )

            # 可视化底图
            vis_top = main.render_topdown(height, mask_top, origin_xy, pix_mm, gcode_xy=g_xy)
            overlay = cv2.addWeighted(vis_top, 1.0, cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR), 0.25, 0)
            vis = main.draw_machine_axes_overlay(overlay, origin_xy, pix_mm)

            # 在线预览（叠加：原始 centerline）
            e_idx = np.arange(len(delta_n), dtype=int); e_n = np.nan_to_num(delta_n, nan=0.0)
            vis = main.draw_deviation_overlay(vis, origin_xy, pix_mm, g_xy, centerline_xy,
                                              e_idx, e_n, arrow_stride=int(P['arrow_stride']),
                                              draw_probes=P.get('draw_normal_probes', True),
                                              N_ref=N_ref, valid_mask=valid_mask)

            # ===== 累积样本（仅 valid & 非拐角） =====
            idx_eff = np.where(valid_mask)[0]
            if ignore_mask is not None and len(ignore_mask)==len(valid_mask):
                idx_eff = idx_eff[~ignore_mask[idx_eff]]
            Ni = N_ref[idx_eff]; zi = e_n[idx_eff]; ki = kappa[idx_eff]
            # 提示：按 A 才加入样本池
            hud = f"[BiasCal] samples={len(S_dn)}  inlier={inlier_ratio:.2f}  guided={rep.get('ratio',0.0):.2f}  A:add  C:clear  M:model  W:save  V:preview  Q:quit"
            cv2.putText(vis, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, hud, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,0), 1, cv2.LINE_AA)

            # 当前帧即时估计（不入库）
            est_now = estimate_bias(Ni, zi, ki, clip_abs_mm=CFG['clip_abs_mm'], gamma=CFG['kappa_gamma'])

            # 叠加：三种补偿预览（仅显示，不写盘）
            if preview_on and g_xy.size > 1:
                # 向量
                v = np.array(est_now['v'], float)
                proj_v = (N_ref @ v)  # 每点法向投影
                g_v = g_xy + N_ref * proj_v[:,None]   # 仅向量补偿后“预期中心线”
                # 标量
                b = float(est_now['b'])
                alpha = (~ignore_mask).astype(float) if ignore_mask is not None else np.ones(len(g_xy), float)
                g_b = g_xy + N_ref * (b*alpha)[:,None]
                # 叠加画三条线
                def draw_path(img, path, color, label, y):
                    # 映射到像素
                    Ht,Wt = img.shape[:2]
                    def xy_to_px(xy):
                        x0,y0 = origin_xy; y1 = y0 + Ht * pix_mm
                        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, Wt-1)
                        ys = np.clip(((y1 - xy[:,1])/pix_mm).astype(int), 0, Ht-1)
                        return np.stack([xs,ys], axis=1)
                    px = xy_to_px(path)
                    for i in range(len(px)-1):
                        cv2.line(img, tuple(px[i]), tuple(px[i+1]), color, 2, cv2.LINE_AA)
                    cv2.putText(img, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                draw_path(vis, g_v, (0,255,180), 'preview: vector only', 48)
                draw_path(vis, g_b, (0,165,255), 'preview: scalar only', 70)
                draw_path(vis, g_xy + N_ref*((proj_v + b*alpha)[:,None]), (0,255,255), 'preview: vector+scalar', 92)

            # 样本池估计 & 可视化面板
            if len(S_dn) >= 2:
                est_pool = estimate_bias(np.vstack(S_N), np.hstack(S_dn), np.hstack(S_kappa),
                                         clip_abs_mm=CFG['clip_abs_mm'], gamma=CFG['kappa_gamma'])
                panel = render_bias_charts(np.vstack(S_N), np.hstack(S_dn), est_pool, size=(560,520))
            else:
                panel = np.full((520, 560, 3), 20, np.uint8)
                cv2.putText(panel, 'Press A to accumulate samples...', (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 2)

            cv2.imshow('BiasCal Live', vis)
            cv2.imshow('BiasCal Plots', panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                if Ni.size and zi.size:
                    S_N.append(Ni.copy()); S_dn.append(zi.copy()); S_kappa.append(ki.copy())
                    print(f'[BiasCal] add samples: +{len(zi)} (total {sum(len(x) for x in S_dn)})')
            elif key == ord('c'):
                S_N.clear(); S_dn.clear(); S_kappa.clear()
                print('[BiasCal] samples cleared.')
            elif key == ord('m'):
                model_sel = {'auto':'vector', 'vector':'scalar', 'scalar':'auto'}[model_sel]
                print('[BiasCal] model ->', model_sel)
            elif key == ord('v'):
                preview_on = not preview_on
            elif key == ord('w'):
                if sum(len(x) for x in S_dn) < CFG['min_samples']:
                    print(f'[BiasCal] too few samples (<{CFG["min_samples"]}), not saved.')
                    continue
                EST = estimate_bias(np.vstack(S_N), np.hstack(S_dn), np.hstack(S_kappa),
                                    clip_abs_mm=CFG['clip_abs_mm'], gamma=CFG['kappa_gamma'])
                # 手动覆盖选择
                if model_sel != 'auto': EST['mode_best'] = model_sel
                payload = dict(
                    version='bias_comp/v1',
                    mode=EST['mode_best'],
                    v=EST['v'],
                    b=EST['b'],
                    r2_vector=EST['r2_vector'],
                    r2_scalar=EST['r2_scalar'],
                    used_samples=int(sum(len(x) for x in S_dn)),
                    guide_step_mm=float(P['guide_step_mm']),
                    corner_ignore=bool(P.get('corner_ignore_enable', True)),
                    corner_span_mm=float(P.get('corner_ignore_span_mm', 6.0)),
                    kappa_gamma=float(CFG['kappa_gamma']),
                    clip_abs_mm=float(CFG['clip_abs_mm']),
                    timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                    gcode_path=str(P['gcode_path']),
                    notes='Apply vector to all points (incl. corners); scalar with corner fade-out.'
                )
                outp = Path(CFG['bias_json']); outp.parent.mkdir(parents=True, exist_ok=True)
                with outp.open('w', encoding='utf-8') as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print('[BiasCal] saved ->', outp)

            frame += 1
    finally:
        try: stream.close()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
