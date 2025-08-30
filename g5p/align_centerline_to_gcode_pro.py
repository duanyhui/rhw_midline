#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centerline ↔ G-code (Guided Fit) with Plane Flattening & Guarded Export — Improved
----------------------------------------------------------------------------------
改进要点：
- 最近表“平面” — ROI 内 RANSAC 平面拟合 + 展平，再在残差高度上取薄层；
- ROI 新增 gcode_bounds；
- G 代码解析支持 G2/G3 圆弧重采样，尽量保留 F；
- 质量门槛（Guard）：valid_ratio / p95 / plane_inlier_ratio 等统一判定；
- 可视化增强：phi=0 等距带、法向剖面覆盖、直方图、quicklook 导出、report.json；
- 其余保持与你原始流程一致，向后兼容。
"""

from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
import numpy as np
import cv2
import math
import collections
import time
import os
import json

# ===================== 参数 =====================
PARAMS = dict(
    # 文件
    T_path='T_cam2machine.npy',               # 相机->机床外参 (含 R,t,T)，来自手眼标定
    gcode_path='args/example.gcode',          # 理论 G 代码路径

    # 顶视投影（右手系：+X 向右，+Y 向上）
    pixel_size_mm=0.8,                        # 初始像素尺寸（mm）
    bounds_qlo=1.0, bounds_qhi=99.0,          # XY 分位数范围（用于粗边界）
    bounds_margin_mm=20.0,                    # 边界外扩（mm）
    max_grid_pixels=1_200_000,                # 顶视网格最大像素数

    # ROI 选择：'none' / 'camera_rect' / 'machine' / 'gcode_bounds'    # NEW
    roi_mode='gcode_bounds',                  # 默认按 G 代码包围盒
    cam_roi_xywh=(682, 847, 228, 185),        # 相机像素系矩形 ROI (x,y,w,h)
    roi_center_xy=(50.0,50.0),                # 机床系 ROI 中心 (mm)（用于 roi_mode='machine'）
    roi_size_mm=250.0,                        # ROI 边长 (mm)

    # 最近表面提取（在“展平后的残差高度图”上进行）
    z_select='max',                           # 'max' 最高层，'min' 最低层（含义不变，针对残差高度）
    nearest_use_percentile=True,
    nearest_qlo=1.0, nearest_qhi=99.0,
    depth_margin_mm=3.0,                      # 最近层厚度（mm，残差域）
    morph_open=3, morph_close=5,
    min_component_area_px=600,

    # 平面拟合/展平（NEW）
    plane_enable=True,
    plane_ransac_thresh_mm=0.8,
    plane_ransac_iters=500,
    plane_sample_cap=120000,                  # 拟合时的点数上限（采样）
    plane_min_inlier_ratio=0.55,              # Guard 指标之一

    # 骨架/折线
    rdp_epsilon_px=3,
    show_skeleton_dilate=True,
    resample_step_px=1.0,

    # G 代码引导中轴线（核心）
    guide_enable=True,
    guide_step_mm=2.0,                        # 将 G 代码重采样为该步长（G0/G1/G2/G3）
    guide_halfwidth_mm=6.0,                   # 法向扫描半宽
    guide_use_dt=True,
    guide_min_on_count=3,
    guide_smooth_win=7,
    guide_max_offset_mm=8.0,
    guide_min_valid_ratio=0.60,               # CHG: 从 0.35 提到 0.60（更安全）
    guide_fallback_to_skeleton=True,

    # 偏差可视化
    arrow_stride=12,
    draw_normal_probes=True,                  # NEW: 显示法向采样线（调试）

    # 纠偏（EMA 在线）
    ema_alpha=0.35,
    deadband_mm=0.05,
    clip_mm=2.0,
    max_step_mm=0.15,
    print_corr=True,

    # 导出（离线）
    export_on_key='c',
    out_dir='out',
    offset_csv='out/offset_table.csv',
    corrected_gcode='out/corrected.gcode',

    # 质量门槛（Guard） NEW
    Guard=dict(
        enable=True,
        min_valid_ratio=0.60,                 # 与 guide_min_valid_ratio 一致或略高
        max_abs_p95_mm=8.80,
        min_plane_inlier_ratio=0.25,
    ),

    # 可视化/报告 NEW
    colormap=getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
    dump_quicklook=True,                      # 导出 quicklook PNG
    dump_report=True,                         # 导出 JSON 报告
)

# ======================= Percipio SDK（原样保留） =======================
try:
    import pcammls
except Exception:
    pcammls = None

class PCamMLSStream:
    def __init__(self):
        if pcammls is None:
            raise SystemExit('未安装 pcammls，无法使用相机。')
        self.cl = pcammls.PercipioSDK()
        self.h = None
        self.depth_calib = None
        self.scale_unit = 0.125
        self.pcl_buf = pcammls.pointcloud_data_list()

    def open(self):
        devs = self.cl.ListDevice()
        if len(devs) == 0: raise SystemExit('未发现设备。')
        print('检测到设备:')
        for i, d in enumerate(devs):
            print('  {}: {}\t{}'.format(i, d.id, d.iface.id))
        idx = 0
        idx = max(0, min(idx, len(devs)-1))
        sn = devs[idx].id

        h = self.cl.Open(sn)
        if not self.cl.isValidHandle(h):
            raise SystemExit('打开设备失败: {}'.format(self.cl.TYGetLastErrorCodedescription()))
        self.h = h
        depth_fmts = self.cl.DeviceStreamFormatDump(h, pcammls.PERCIPIO_STREAM_DEPTH)
        if not depth_fmts: raise SystemExit('无深度流。')
        self.cl.DeviceStreamFormatConfig(h, pcammls.PERCIPIO_STREAM_DEPTH, depth_fmts[0])
        self.cl.DeviceLoadDefaultParameters(h)
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(h)
        self.depth_calib = self.cl.DeviceReadCalibData(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamEnable(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamOn(h)

    def read_pointcloud(self, timeout_ms: int = 2000):
        frames = self.cl.DeviceStreamRead(self.h, timeout_ms)
        depth_fr = None
        for fr in frames:
            if fr.streamID == pcammls.PERCIPIO_STREAM_DEPTH:
                depth_fr = fr; break
        if depth_fr is None:
            return None, None
        self.cl.DeviceStreamMapDepthImageToPoint3D(depth_fr, self.depth_calib, self.scale_unit, self.pcl_buf)
        P_cam = self.pcl_buf.as_nparray()  # (H,W,3) mm
        return P_cam, depth_fr

    def close(self):
        if self.h is not None:
            try: self.cl.DeviceStreamOff(self.h)
            except Exception: pass
            try: self.cl.Close(self.h)
            except Exception: pass
            self.h = None

# ======================= 基础 IO/几何（在原有基础上少量增强） =======================
def load_extrinsic(T_path: Union[str, Path]):
    data = np.load(str(T_path), allow_pickle=True).item()
    R = np.asarray(data['R'], dtype=float)
    t = np.asarray(data['t'], dtype=float).reshape(1, 3)
    T = np.asarray(data['T'], dtype=float)
    return R, t, T

# ---- G-code 解析（支持 G0/G1 + G2/G3 圆弧） NEW ----
def _interp_arc_xy(xy0, xy1, ij=None, R=None, cw=True, step=1.0):
    # 仅处理平面 XY 圆弧。优先 IJ 圆心；否则用 R。
    p0 = np.array(xy0, float); p1 = np.array(xy1, float)
    if ij is not None:
        c = p0 + np.array(ij, float)
    else:
        # R 法：两点到圆心距离相等，选择小圆或大圆；此处选小圆。
        # 解法简化：计算中垂线与两点夹角，根据 R 求圆心。
        chord = p1 - p0; L = np.linalg.norm(chord)
        if L < 1e-9 or R is None:
            return np.vstack([p0, p1])
        h = math.sqrt(max(R*R - (L*0.5)**2, 0.0))
        mid = (p0 + p1) * 0.5
        n = np.array([-chord[1], chord[0]]) / (L + 1e-12)
        # 方向选择（cw/ccw），取最近的那个
        c1 = mid + n * h
        c2 = mid - n * h
        # 选使得方向与 cw 匹配的圆心
        def ang(p): return math.atan2(p[1]-c[1], p[0]-c[0])
        for cand in (c1, c2):
            c = cand
            a0, a1 = ang(p0), ang(p1)
            da = (a1 - a0)
            if cw and da > 0: da -= 2*math.pi
            if (not cw) and da < 0: da += 2*math.pi
            arc_len = abs(da) * R
            if arc_len > 1e-3: break
    # 角度展开
    def ang(p): return math.atan2(p[1]-c[1], p[0]-c[0])
    a0, a1 = ang(p0), ang(p1)
    da = (a1 - a0)
    # 顺/逆时针校正
    if cw and da > 0: da -= 2*math.pi
    if (not cw) and da < 0: da += 2*math.pi
    r = np.linalg.norm(p0 - c)
    arc_len = abs(da) * r
    n = max(2, int(math.ceil(arc_len / max(1e-6, step))))
    aa = np.linspace(a0, a0 + da, n)
    pts = np.stack([c[0] + r*np.cos(aa), c[1] + r*np.sin(aa)], axis=1)
    pts[0] = p0; pts[-1] = p1
    return pts

def parse_gcode_xy(path: Union[str, Path], step_mm: float = 1.0) -> Tuple[np.ndarray, Optional[float]]:
    p = Path(path) if not isinstance(path, Path) else path
    if (not path) or (not p.exists()):
        return np.empty((0,2), float), None
    pts = []
    feed = None
    cur = {'X': None, 'Y': None}
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith(';') or line.startswith('('):
                continue
            if ';' in line: line = line.split(';',1)[0]
            while '(' in line and ')' in line:
                a = line.find('('); b = line.find(')');
                if a < 0 or b < 0 or b <= a: break
                line = (line[:a] + ' ' + line[b+1:]).strip()
            toks = line.split()
            if not toks: continue
            cmd = toks[0].upper()
            # 读 F（若有）
            for u in toks[1:]:
                U = u.upper()
                if U.startswith('F'):
                    try: feed = float(U[1:])
                    except: pass
            # 读运动
            if cmd in ('G0','G00','G1','G01','G2','G02','G3','G03'):
                x = cur['X']; y = cur['Y']
                I = J = R = None
                for u in toks[1:]:
                    U = u.upper()
                    if U.startswith('X'):
                        try: x = float(U[1:])
                        except: pass
                    elif U.startswith('Y'):
                        try: y = float(U[1:])
                        except: pass
                    elif U.startswith('I'):
                        try: I = float(U[1:])
                        except: pass
                    elif U.startswith('J'):
                        try: J = float(U[1:])
                        except: pass
                    elif U.startswith('R'):
                        try: R = float(U[1:])
                        except: pass
                if x is None or y is None:
                    continue
                if cmd in ('G2','G02','G3','G03'):
                    cw = cmd in ('G2','G02')
                    if len(pts)==0 and (cur['X'] is None or cur['Y'] is None):
                        # 没有起点无法画圆弧
                        pts.append([x, y])
                    else:
                        p0 = np.array([cur['X'], cur['Y']]) if cur['X'] is not None else np.array(pts[-1])
                        p1 = np.array([x, y])
                        ij = (I,J) if (I is not None and J is not None) else None
                        arc = _interp_arc_xy(p0, p1, ij=ij, R=R, cw=cw, step=step_mm)
                        if len(pts)>0 and np.allclose(pts[-1], arc[0]):
                            pts.extend(arc[1:].tolist())
                        else:
                            pts.extend(arc.tolist())
                else:  # 直线
                    if len(pts)==0 or not np.allclose([x,y], pts[-1]):
                        pts.append([x,y])
                cur['X'], cur['Y'] = x, y
    P = np.asarray(pts, float) if pts else np.empty((0,2), float)
    return P, feed

def resample_polyline(poly: np.ndarray, step: float) -> np.ndarray:
    if poly.shape[0] < 2: return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = float(seg.sum());
    if L <= 1e-9: return poly[[0]].copy()
    n = max(2, int(math.ceil(L / max(1e-6, step))))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = []; j = 0
    for si in s:
        while j < len(seg) and si > cs[j+1]: j += 1
        if j >= len(seg): out.append(poly[-1]); continue
        t = (si - cs[j]) / max(seg[j], 1e-9)
        out.append(poly[j]*(1-t) + poly[j+1]*t)
    return np.asarray(out, float)

def tangent_normal(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if poly.shape[0] < 2:
        return np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    d = np.gradient(poly, axis=0)
    T = d / np.maximum(1e-9, np.linalg.norm(d, axis=1, keepdims=True))
    N = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, N

def build_kdtree(pts: np.ndarray):
    try:
        from scipy.spatial import cKDTree as KDTree
        return KDTree(pts) if pts.size else None
    except Exception:
        if pts.size == 0: return None
        class _Lin:
            def __init__(self, A): self.A = A
            def query(self, B):
                B = np.atleast_2d(B)
                d2 = ((B[:,None,:]-self.A[None,:,:])**2).sum(2)
                idx = d2.argmin(1)
                return np.sqrt(d2[np.arange(len(B)), idx]), idx
        return _Lin(pts)

def transform_cam_to_machine_grid(P_cam_hw3: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    H, W, _ = P_cam_hw3.shape
    P = P_cam_hw3.reshape(-1, 3).astype(np.float32)
    Pm = (R @ P.T).T + t
    return Pm.reshape(H, W, 3)

# ================= 掩码/投影/可视化（右手系） =================
def valid_mask_hw(P_hw3: np.ndarray) -> np.ndarray:
    X = P_hw3[:, :, 0]; Y = P_hw3[:, :, 1]; Z = P_hw3[:, :, 2]
    m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    m &= (np.abs(X) + np.abs(Y) + np.abs(Z) > 1e-6)
    return m

def camera_rect_mask(H: int, W: int, xywh: Tuple[int,int,int,int]) -> np.ndarray:
    x, y, w, h = xywh
    x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
    w = int(np.clip(w, 1, W-x)); h = int(np.clip(h, 1, H-y))
    m = np.zeros((H, W), np.bool_)
    m[y:y+h, x:x+w] = True
    return m

def machine_rect_mask(P_mach_hw3: np.ndarray, center_xy: Tuple[float,float], size_mm: float) -> np.ndarray:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    half = float(size_mm) * 0.5
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    X = P_mach_hw3[:, :, 0]; Y = P_mach_hw3[:, :, 1]
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)

def compute_bounds_xy_from_mask(P_mach_hw3: np.ndarray, mask_hw: np.ndarray,
                                qlo: float, qhi: float, margin: float) -> Tuple[float,float,float,float]:
    m = mask_hw.copy()
    if not np.any(m):
        m = valid_mask_hw(P_mach_hw3)
        if not np.any(m):
            return -100, 100, -100, 100
    X = P_mach_hw3[:, :, 0][m]; Y = P_mach_hw3[:, :, 1][m]
    x0, x1 = np.percentile(X, qlo), np.percentile(X, qhi)
    y0, y1 = np.percentile(Y, qlo), np.percentile(Y, qhi)
    return float(x0 - margin), float(x1 + margin), float(y0 - margin), float(y1 + margin)

def adjust_pixel_size(x0,x1,y0,y1,pix_mm,max_pixels) -> float:
    W = max(2, int(round((x1-x0)/max(1e-6,pix_mm))))
    H = max(2, int(round((y1-y0)/max(1e-6,pix_mm))))
    while W*H > max_pixels:
        pix_mm *= 1.25
        W = max(2, int(round((x1-x0)/pix_mm)))
        H = max(2, int(round((y1-y0)/pix_mm)))
    return pix_mm

def project_topdown_from_grid(P_mach_hw3: np.ndarray, select_mask_hw: np.ndarray,
                              pix_mm: float,
                              bounds_xy: Tuple[float,float,float,float]) -> Tuple[np.ndarray,np.ndarray,Tuple[float,float]]:
    x0,x1,y0,y1 = bounds_xy
    X = P_mach_hw3[:, :, 0]; Y = P_mach_hw3[:, :, 1]; Z = P_mach_hw3[:, :, 2]
    m = select_mask_hw & valid_mask_hw(P_mach_hw3)

    Wg = int(max(2, round((x1 - x0) / max(1e-6, pix_mm))))
    Hg = int(max(2, round((y1 - y0) / max(1e-6, pix_mm))))
    height = np.full((Hg, Wg), np.nan, np.float32)
    count  = np.zeros((Hg, Wg), np.int32)
    if not np.any(m):
        return height, (count>0).astype(np.uint8)*255, (x0, y0)

    Xs = X[m].astype(np.float32); Ys = Y[m].astype(np.float32); Zs = Z[m].astype(np.float32)
    ix = np.clip(((Xs - x0) / pix_mm).astype(np.int32), 0, Wg-1)
    iy = np.clip(((y1 - Ys) / pix_mm).astype(np.int32), 0, Hg-1)  # 右手系：+Y 向上

    for gx, gy, gz in zip(ix, iy, Zs):
        if (not np.isfinite(height[gy, gx])) or (gz > height[gy, gx]):
            height[gy, gx] = gz
        count[gy, gx] += 1

    mask = (count > 0).astype(np.uint8) * 255
    return height, mask, (x0, y0)

def render_topdown(height: np.ndarray, mask: np.ndarray,
                   origin_xy: Tuple[float,float], pix_mm: float,
                   gcode_xy: Optional[np.ndarray]=None) -> np.ndarray:
    H,W = height.shape
    if np.isfinite(height).any():
        vmin = float(np.nanpercentile(height, 5)); vmax = float(np.nanpercentile(height, 95))
        vspan = max(1e-6, vmax - vmin)
        gray = np.clip(((height - vmin)/vspan) * 255, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(gray, PARAMS['colormap'])
    else:
        vis = np.zeros((H,W,3), np.uint8)
    vis = cv2.addWeighted(vis, 0.9, np.dstack([mask]*3), 0.1, 0)

    def xy_to_px(xy):
        x0,y0 = origin_xy
        y1 = y0 + H * pix_mm
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((y1 - xy[:,1])/pix_mm).astype(int), 0, H-1)  # 右手系：+Y 向上
        return np.stack([xs,ys], axis=1)

    if gcode_xy is not None and gcode_xy.size>0:
        pts = xy_to_px(gcode_xy)
        for i in range(len(pts)-1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), (255,255,255), 1, cv2.LINE_AA)
    return vis

def draw_machine_axes_overlay(img: np.ndarray,
                              origin_xy: Tuple[float,float],
                              pix_mm: float) -> np.ndarray:
    H, W = img.shape[:2]
    vis = img.copy()
    base = (40, H-40)
    ax = int(60.0 / max(1e-6, pix_mm))
    cv2.arrowedLine(vis, base, (base[0] + ax, base[1]), (0, 255, 0), 2, cv2.LINE_AA)   # +X
    cv2.arrowedLine(vis, base, (base[0], base[1] - ax), (0, 200, 255), 2, cv2.LINE_AA) # +Y
    cv2.putText(vis, '+X', (base[0] + ax + 6, base[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(vis, '+Y', (base[0] - 18, base[1] - ax - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
    return vis

# ======================= 最近表面（残差域） =======================
def morph_cleanup(mask_u8: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_k), int(close_k)))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_k), int(open_k)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    return m

def extract_nearest_surface_mask_from_height(
    height: np.ndarray,
    valid_mask: np.ndarray,
    *,
    z_select: str = 'max',
    depth_margin_mm: float = 30.0,
    use_percentile: bool = True,
    qlo: float = 1.0,
    qhi: float = 99.0,
    morph_open: int = 3,
    morph_close: int = 5,
    min_component_area_px: int = 600
) -> Tuple[np.ndarray, float, Tuple[float,float]]:
    H, W = height.shape[:2]
    vm = (np.asarray(valid_mask).astype(np.uint8) > 0) & np.isfinite(height)
    if not np.any(vm):
        return np.zeros((H, W), np.uint8), float('nan'), (float('nan'), float('nan'))

    vals = height[vm]
    z_select = str(z_select).lower()

    if use_percentile:
        if z_select.startswith('max'):
            z_ref = float(np.nanpercentile(vals, qhi))
            low, high = z_ref - float(depth_margin_mm), z_ref + 1e-6
        else:
            z_ref = float(np.nanpercentile(vals, qlo))
            low, high = z_ref, z_ref + float(depth_margin_mm)
    else:
        if z_select.startswith('max'):
            z_ref = float(np.nanmax(vals))
            low, high = z_ref - float(depth_margin_mm), z_ref + 1e-6
        else:
            z_ref = float(np.nanmin(vals))
            low, high = z_ref, z_ref + float(depth_margin_mm)

    band = (height >= low) & (height <= high) & vm
    m = (band.astype(np.uint8)) * 255
    m = morph_cleanup(m, morph_open, morph_close)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if num <= 1:
        return np.zeros((H, W), np.uint8), z_ref, (low, high)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas)) if areas.size > 0 else 0
    if stats[idx, cv2.CC_STAT_AREA] < max(1, int(min_component_area_px)):
        return np.zeros((H, W), np.uint8), z_ref, (low, high)
    keep = (labels == idx).astype(np.uint8) * 255
    return keep, z_ref, (low, high)

# ======================= 骨架提取（沿用你的通用实现） =======================
try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except Exception:
    _sk_skeletonize = None
try:
    import cv2.ximgproc as xip
except Exception:
    xip = None

def _remove_small_blobs(binary, min_area=300):
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    keep = np.zeros_like(binary, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep

def _clean_mask_for_skeleton(mask):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3, iterations=1)
    mask = _remove_small_blobs(mask, min_area=600)
    return mask

def _prune_skeleton_spurs(skel_u8, min_branch_len=12, max_pass=6):
    skel = (skel_u8 > 0).astype(np.uint8)
    def neighbors(y, x):
        ys = [y-1, y-1, y-1, y,   y,   y+1, y+1, y+1]
        xs = [x-1, x,   x+1, x-1, x+1, x-1, x,   x+1]
        out = []
        H, W = skel.shape
        for yy, xx in zip(ys, xs):
            if 0 <= yy < H and 0 <= xx < W and skel[yy, xx]:
                out.append((yy, xx))
        return out
    k = np.array([[1,1,1],[1,10,1],[1,1,1]], np.uint8)
    for _ in range(max_pass):
        nbr_cnt = cv2.filter2D(skel, -1, k, borderType=cv2.BORDER_CONSTANT)
        endpoints = np.argwhere((skel == 1) & (nbr_cnt == 11))
        if len(endpoints) == 0:
            break
        removed_any = False
        for y, x in endpoints:
            if skel[y, x] == 0: continue
            path = [(y, x)]
            prev = None; cur = (y, x)
            while True:
                nbrs = neighbors(*cur)
                if prev is not None and prev in nbrs: nbrs.remove(prev)
                if len(nbrs) == 0 or len(nbrs) > 1: break
                nxt = nbrs[0]; path.append(nxt); prev, cur = cur, nxt
                y2, x2 = cur
                if cv2.filter2D(skel, -1, k, borderType=cv2.BORDER_CONSTANT)[y2, x2] == 11 and len(path) > 1:
                    break
            if len(path) < min_branch_len:
                for yy, xx in path: skel[yy, xx] = 0; removed_any = True
        if not removed_any: break
    return (skel * 255).astype(np.uint8)

def _skeletonize_bool(bw_bool: np.ndarray) -> np.ndarray:
    if _sk_skeletonize is not None:
        sk = _sk_skeletonize(bw_bool); return (sk.astype(np.uint8) * 255)
    if xip is not None:
        try:
            u8 = (bw_bool.astype(np.uint8) * 255)
            sk = xip.thinning(u8, thinningType=xip.THINNING_ZHANGSUEN)
            return (sk > 0).astype(np.uint8) * 255
        except Exception:
            pass
    u8 = (bw_bool.astype(np.uint8) * 255)
    dist = cv2.distanceTransform(u8, cv2.DIST_L2, 3)
    sk = np.zeros_like(u8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0: continue
            sk = np.maximum(sk, (dist < np.roll(np.roll(dist, dy, 0), dx, 1)).astype(np.uint8))
    ridge = (sk == 0).astype(np.uint8) & (u8 > 0)
    return ridge.astype(np.uint8) * 255

def extract_skeleton_universal(surface_mask: np.ndarray, visualize: bool = True):
    if surface_mask is None or np.count_nonzero(surface_mask) == 0:
        print("输入的 surface_mask 为空或不包含有效区域。"); return None
    ring_mask = _clean_mask_for_skeleton(surface_mask)
    if np.count_nonzero(ring_mask) == 0:
        print("净化后掩码为空。"); return None

    contours, hierarchy = cv2.findContours(ring_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        print("未找到轮廓。"); return None
    hierarchy = hierarchy[0]
    outer_idx = -1; outer_area = -1
    for i, h in enumerate(hierarchy):
        if h[3] == -1:
            a = cv2.contourArea(contours[i])
            if a > outer_area:
                outer_area = a; outer_idx = i
    if outer_idx < 0:
        print("未找到外轮廓。"); return None
    child = hierarchy[outer_idx][2]
    inner_idx = -1; inner_area = -1
    while child != -1:
        a = cv2.contourArea(contours[child])
        if a > inner_area:
            inner_area = a; inner_idx = child
        child = hierarchy[child][0]
    outer_cnt = contours[outer_idx]
    inner_cnt = contours[inner_idx] if inner_idx >= 0 else None

    if inner_cnt is None:
        skel_u8 = _skeletonize_bool(ring_mask > 0)
        skel_u8 = _prune_skeleton_spurs(skel_u8, 12, 6)
        if visualize:
            vis = cv2.cvtColor(surface_mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, [outer_cnt], -1, (0, 0, 255), 2)
            vis[skel_u8 > 0] = (255, 0, 0)
            cv2.imshow("Target Mask for Skeletonization", ring_mask)
            cv2.imshow("Universal Skeleton Extraction", vis)
        return cv2.cvtColor(skel_u8, cv2.COLOR_GRAY2BGR)

    # 有内环
    H, W = ring_mask.shape
    outer_line = np.full((H, W), 255, np.uint8)
    inner_line = np.full((H, W), 255, np.uint8)
    cv2.drawContours(outer_line, [outer_cnt], -1, 0, 1)
    cv2.drawContours(inner_line,  [inner_cnt], -1, 0, 1)

    d_out = cv2.distanceTransform(outer_line, cv2.DIST_L2, 5).astype(np.float32)
    d_in  = cv2.distanceTransform(inner_line,  cv2.DIST_L2, 5).astype(np.float32)

    dist_sum = d_out + d_in + 1e-6
    diff = np.abs(d_out - d_in)
    alpha = 0.04
    tau = np.maximum(1.0, alpha * dist_sum)
    equi_band = ((diff <= tau) & (ring_mask > 0)).astype(np.uint8) * 255
    equi_band = cv2.morphologyEx(equi_band, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

    skel_u8 = _skeletonize_bool(equi_band > 0)
    skel_u8 = _remove_small_blobs(skel_u8, min_area=50)
    skel_u8 = _prune_skeleton_spurs(skel_u8, min_branch_len=12, max_pass=6)

    if PARAMS.get('show_skeleton_dilate', True):
        skel_show = cv2.dilate(skel_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
    else:
        skel_show = skel_u8

    if visualize:
        vis = cv2.cvtColor(np.zeros_like(ring_mask), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, [outer_cnt], -1, (0, 0, 255), 2)
        if inner_cnt is not None:
            cv2.drawContours(vis, [inner_cnt], -1, (0, 255, 0), 2)
        vis[skel_show > 0] = (255, 0, 0)
        cv2.imshow("Target Mask for Skeletonization", ring_mask)
        cv2.imshow("Universal Skeleton Extraction", vis)
        cv2.imshow("Equidistance Band", equi_band)

    return cv2.cvtColor(skel_u8, cv2.COLOR_GRAY2BGR)

# =================== 骨架 → 折线（原样） ===================
def _skeleton_graph(skel_u8: np.ndarray):
    m = (skel_u8 > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    nodes = list(zip(ys.tolist(), xs.tolist()))
    S = set(nodes)
    adj = {n: set() for n in nodes}
    H, W = m.shape
    for y, x in nodes:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                yy, xx = y+dy, x+dx
                if 0 <= yy < H and 0 <= xx < W and (yy,xx) in S:
                    adj[(y,x)].add((yy,xx))
    deg = {n: len(adj[n]) for n in nodes}
    endpoints = [n for n,d in deg.items() if d == 1]
    return adj, endpoints, deg

def _longest_path_from_graph(adj, endpoints):
    def bfs(start):
        vis = {start: None}
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in vis:
                    vis[v] = u; q.append(v)
        far = max(vis.keys(), key=lambda k: (abs(k[0]-start[0]) + abs(k[1]-start[1])))
        return far, vis
    start = endpoints[0] if endpoints else next(iter(adj.keys()))
    b, pre = bfs(start)
    c, pre2 = bfs(b)
    path = [c]; p = pre2[c]
    while p is not None: path.append(p); p = pre2[p]
    return path[::-1]

def _trace_closed_loop(adj, deg) -> List[Tuple[int,int]]:
    start = None
    for n, d in deg.items():
        if d == 2: start = n; break
    if start is None:
        start = next(iter(adj.keys()))
    path = [start]
    prev = None
    cur = start
    visited = set([start])
    for _ in range(200000):
        nbrs = list(adj[cur])
        if prev is not None and len(nbrs) > 1:
            if prev in nbrs: nbrs.remove(prev)
        nxt = None
        if len(nbrs) == 1:
            nxt = nbrs[0]
        else:
            if prev is None:
                nxt = nbrs[0]
            else:
                v = np.array([cur[1]-prev[1], cur[0]-prev[0]], float)
                best = -1e9
                for cnd in nbrs:
                    w = np.array([cnd[1]-cur[1], cnd[0]-cur[0]], float)
                    score = float(np.dot(v, w) / (np.linalg.norm(v)*np.linalg.norm(w) + 1e-9))
                    if score > best:
                        best = score; nxt = cnd
        if nxt is None: break
        if nxt == start:
            path.append(nxt)
            break
        path.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt
        if len(path) > 1 and deg.get(cur, 0) != 2:
            break
    if len(path) >= 2 and path[-1] == path[0]:
        path = path[:-1]
    return path

def _resample_px_chain(path_px: np.ndarray, step_px: float) -> np.ndarray:
    if len(path_px) < 2: return path_px
    seg = np.linalg.norm(np.diff(path_px, axis=0), axis=1)
    L = float(seg.sum());
    if L < 1e-6: return path_px[[0]].copy()
    n = max(2, int(np.ceil(L / max(1e-6, step_px))))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = []; j=0
    for si in s:
        while j < len(seg) and si > cs[j+1]: j += 1
        if j >= len(seg): out.append(path_px[-1]); continue
        t = (si - cs[j]) / max(seg[j], 1e-9)
        out.append(path_px[j]*(1-t) + path_px[j+1]*t)
    return np.asarray(out, float)

def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    P = np.asarray(points, dtype=float)
    n = P.shape[0]
    if n < 3: return P.copy()
    a = P[0]; b = P[-1]; ab = b - a
    lab2 = float(np.dot(ab, ab))
    if lab2 < 1e-12:
        d2 = np.sum((P - a) ** 2, axis=1)
        idx = int(np.argmax(d2))
        if idx == 0 or idx == n - 1:
            return np.vstack([a, b]).astype(P.dtype)
        left = _rdp(P[:idx + 1], eps)
        right = _rdp(P[idx:], eps)
        return np.vstack([left[:-1], right]).astype(P.dtype)
    ap = P[1:-1] - a
    cross = np.abs(ab[0] * ap[:, 1] - ab[1] * ap[:, 0])
    dist = cross / (np.sqrt(lab2) + 1e-12)
    i_rel = int(np.argmax(dist))
    dmax = float(dist[i_rel])
    i = 1 + i_rel
    if dmax > eps:
        left = _rdp(P[:i + 1], eps)
        right = _rdp(P[i:], eps)
        return np.vstack([left[:-1], right]).astype(P.dtype)
    else:
        return np.vstack([a, b]).astype(P.dtype)

def skeleton_to_path_px_topo(skel_gray_u8: np.ndarray) -> np.ndarray:
    if skel_gray_u8.ndim == 3:
        skel_gray_u8 = cv2.cvtColor(skel_gray_u8, cv2.COLOR_BGR2GRAY)
    if np.count_nonzero(skel_gray_u8) < 2:
        return np.empty((0,2), float)
    adj, endpoints, deg = _skeleton_graph(skel_gray_u8)
    is_closed = (len(endpoints) == 0) and all((d == 2) for d in deg.values())
    if is_closed:
        cyc_nodes = _trace_closed_loop(adj, deg)
        path_px = np.array([[x, y] for (y, x) in cyc_nodes], dtype=np.float32)
        step_px = float(PARAMS.get('resample_step_px', 1.0))
        path_px = _resample_px_chain(path_px, max(0.5, step_px))
    else:
        nodes = _longest_path_from_graph(adj, endpoints)
        path_px = np.array([[x, y] for (y, x) in nodes], dtype=np.float32)
        eps = float(PARAMS.get('rdp_epsilon_px', 0.0))
        if eps > 1e-6 and len(path_px) >= 3:
            path_px = _rdp(path_px, eps)
    return path_px

# =============== 像素/机床坐标互换（统一口径） ===============
def px_to_mach_xy(path_px: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int) -> np.ndarray:
    if path_px.size == 0: return np.empty((0,2), float)
    x0, y0 = origin_xy; y1 = y0 + Himg * pix_mm
    X = x0 + (path_px[:,0] + 0.5) * pix_mm
    Y = y1 - (path_px[:,1] + 0.5) * pix_mm
    return np.stack([X, Y], axis=1)

def px_float_to_mach_xy(px: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int) -> np.ndarray:
    if px.size == 0: return np.empty((0,2), float)
    x0, y0 = origin_xy; y1 = y0 + Himg * pix_mm
    X = x0 + (px[:,0]) * pix_mm
    Y = y1 - (px[:,1]) * pix_mm
    return np.stack([X, Y], axis=1)

def mach_xy_to_px_float(xy: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int) -> np.ndarray:
    x0, y0 = origin_xy; y1 = y0 + Himg * pix_mm
    xs = (xy[:,0]-x0)/pix_mm
    ys = (y1 - xy[:,1])/pix_mm
    return np.stack([xs, ys], axis=1)

# =================== 偏差计算（KDTree 兜底） ===================
def project_points_to_path(points_xy: np.ndarray, ref_xy: np.ndarray, ref_tree, ref_normals: np.ndarray):
    if points_xy.size == 0 or ref_xy.size == 0:
        return np.empty((0,), int), np.empty((0, 2)), np.empty((0,))
    _, idx = ref_tree.query(points_xy)
    idx = np.asarray(idx, dtype=int)
    nearest = ref_xy[idx]
    N = ref_normals[np.clip(idx, 0, len(ref_normals) - 1)]
    e_n = ((points_xy - nearest) * N).sum(axis=1)
    return idx, nearest, e_n

class EmaCorrector:
    def __init__(self, alpha=0.35, deadband=0.05, clip_mm=2.0, max_step_mm=0.15):
        self.a = float(alpha)
        self.dead = float(deadband)
        self.clip = float(clip_mm)
        self.step = float(max_step_mm)
        self._ema = 0.0
        self._out_prev = np.zeros(2, float)
        self.last_idx = -1
        self.last_nvec = np.zeros(2, float)
        self.last_med = 0.0
        self.last_target = np.zeros(2, float)

    def update(self, e_idx: np.ndarray, e_n: np.ndarray, N_ref: np.ndarray):
        if e_n.size == 0 or N_ref.size == 0:
            return np.zeros(2, float), dict(mean=0.0, median=0.0, p95=0.0, n=0)
        stats = dict(mean=float(np.mean(e_n)), median=float(np.median(e_n)),
                     p95=float(np.percentile(np.abs(e_n), 95)), n=int(len(e_n)))
        k = int(np.argsort(e_n)[len(e_n)//2])
        i = int(np.clip(e_idx[k], 0, len(N_ref)-1))
        nvec = N_ref[i]
        med = stats['median']
        if abs(med) < self.dead:
            med = 0.0
        med = float(np.clip(med, -self.clip, self.clip))
        self._ema = self.a * med + (1.0 - self.a) * self._ema
        target = nvec * self._ema
        delta = target - self._out_prev
        mag = float(np.linalg.norm(delta))
        if mag > self.step and mag > 0:
            delta *= (self.step / mag)
        out = self._out_prev + delta
        self._out_prev = out
        self.last_idx = i
        self.last_nvec = nvec.copy()
        self.last_med = med
        self.last_target = out.copy()
        return out, stats

class VectorEmaCorrector:
    def __init__(self, alpha=0.35, deadband=0.05, clip_mm=2.0, max_step_mm=0.15, use_mean: bool=False):
        self.a = float(alpha)
        self.dead = float(deadband)
        self.clip = float(clip_mm)
        self.step = float(max_step_mm)
        self.use_mean = bool(use_mean)
        self._ema_vec = np.zeros(2, float)
        self._out_prev = np.zeros(2, float)
        self.last_v_med = np.zeros(2, float)
        self.last_target = np.zeros(2, float)
        self.last_after_clip = np.zeros(2, float)
    def update(self, e_idx: np.ndarray, e_n: np.ndarray, N_ref: np.ndarray):
        if e_n.size == 0 or N_ref.size == 0:
            return np.zeros(2, float), dict(mean=0.0, median=0.0, p95=0.0, n=0)
        stats = dict(mean=float(np.mean(e_n)), median=float(np.median(e_n)),
                     p95=float(np.percentile(np.abs(e_n), 95)), n=int(len(e_n)))
        idx_clipped = np.clip(e_idx, 0, len(N_ref)-1)
        Ni = N_ref[idx_clipped]
        Vi = Ni * e_n[:, None]
        mask_core = np.abs(e_n) <= (self.clip * 2.0)
        Vi_core = Vi[mask_core] if mask_core.any() else Vi
        if Vi_core.size == 0:
            return np.zeros(2, float), stats
        v_med = Vi_core.mean(axis=0) if self.use_mean else np.median(Vi_core, axis=0)
        mag = float(np.linalg.norm(v_med))
        if mag < self.dead:
            v_med[:] = 0.0
        v_med = np.clip(v_med, -self.clip, self.clip)
        self._ema_vec = self.a * v_med + (1.0 - self.a) * self._ema_vec
        target = self._ema_vec
        delta = target - self._out_prev
        dmag = float(np.linalg.norm(delta))
        if dmag > self.step and dmag > 0:
            delta *= (self.step / dmag)
        out = self._out_prev + delta
        self._out_prev = out
        self.last_v_med = v_med.copy()
        self.last_after_clip = v_med.copy()
        self.last_target = out.copy()
        return out, stats

def draw_correction_debug(img: np.ndarray,
                          origin_xy: Tuple[float,float], pix_mm: float,
                          gcode_xy: np.ndarray,
                          corr_obj,
                          mode: str) -> np.ndarray:
    if gcode_xy is None or gcode_xy.size == 0: return img
    vis = img.copy()
    H, W = vis.shape[:2]
    def xy_to_px(xy):
        x0, y0 = origin_xy; y1 = y0 + H * pix_mm
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((y1 - xy[:,1])/pix_mm).astype(int), 0, H-1)
        return np.stack([xs,ys], axis=1)
    if isinstance(corr_obj, EmaCorrector) and hasattr(corr_obj, 'last_idx') and corr_obj.last_idx >= 0:
        idx = min(corr_obj.last_idx, gcode_xy.shape[0]-1)
        p = gcode_xy[idx:idx+1]
        pp = xy_to_px(p)[0]
        cv2.circle(vis, tuple(pp), 5, (0,255,255), 2, cv2.LINE_AA)
        tgt = corr_obj.last_target
        base = p[0]
        arrow_end = base + tgt
        bpx = xy_to_px(base[None,:])[0]; epx = xy_to_px(arrow_end[None,:])[0]
        cv2.arrowedLine(vis, tuple(bpx), tuple(epx), (255,0,255), 2, cv2.LINE_AA, tipLength=0.35)
        txt = f"MED={corr_obj.last_med:+.3f}  EMA={corr_obj._ema:+.3f}"
        cv2.putText(vis, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1, cv2.LINE_AA)
    elif isinstance(corr_obj, VectorEmaCorrector):
        base = np.array([gcode_xy[:,0].mean(), gcode_xy[:,1].mean()])
        tgt = corr_obj.last_target
        bpx = xy_to_px(base[None,:])[0]; epx = xy_to_px((base + tgt)[None,:])[0]
        cv2.arrowedLine(vis, tuple(bpx), tuple(epx), (255,0,128), 3, cv2.LINE_AA, tipLength=0.35)
        txt = f"Vmed=({corr_obj.last_v_med[0]:+.3f},{corr_obj.last_v_med[1]:+.3f})  EMA=({corr_obj._ema_vec[0]:+.3f},{corr_obj._ema_vec[1]:+.3f})"
        cv2.putText(vis, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,128,255), 1, cv2.LINE_AA)
    return vis

def draw_deviation_overlay(vis_top: np.ndarray,
                           origin_xy: Tuple[float,float], pix_mm: float,
                           gcode_xy: Optional[np.ndarray],
                           centerline_xy: Optional[np.ndarray],
                           e_idx: np.ndarray, e_n: np.ndarray,
                           arrow_stride: int = 10,
                           draw_probes: bool = False, N_ref: Optional[np.ndarray] = None) -> np.ndarray:
    H, W = vis_top.shape[:2]
    out = vis_top.copy()
    def xy_to_px(xy):
        x0, y0 = origin_xy; y1 = y0 + H * pix_mm
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((y1 - xy[:,1])/pix_mm).astype(int), 0, H-1)
        return np.stack([xs,ys], axis=1)
    if gcode_xy is not None and gcode_xy.size > 0:
        p = xy_to_px(gcode_xy)
        for i in range(len(p)-1):
            cv2.line(out, tuple(p[i]), tuple(p[i+1]), (240,240,240), 1, cv2.LINE_AA)
    if centerline_xy is not None and centerline_xy.size > 0:
        q = xy_to_px(centerline_xy)
        for i in range(len(q)-1):
            cv2.line(out, tuple(q[i]), tuple(q[i+1]), (200,255,255), 2, cv2.LINE_AA)
    if e_n.size > 0 and gcode_xy is not None and gcode_xy.shape[0] > 1 and centerline_xy is not None:
        seg = np.diff(gcode_xy, axis=0)
        T = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
        N = np.stack([-T[:,1], T[:,0]], axis=1)
        stride = max(1, arrow_stride)
        for k in range(0, len(e_idx), stride):
            i = int(np.clip(e_idx[k], 0, len(N)-1))
            n = N[i]
            base = centerline_xy[k]
            tip = base + n * e_n[k]
            b = xy_to_px(base[None,:])[0]
            t = xy_to_px(tip[None,:])[0]
            col = (0,255,0) if abs(e_n[k]) < 0.5 else (0,165,255)
            cv2.arrowedLine(out,  tuple(t),tuple(b), col, 2, cv2.LINE_AA, tipLength=0.35)
            if draw_probes and N_ref is not None:
                # 在该处绘制法向采样范围（用于调参观测）
                nref = N_ref[min(i, len(N_ref)-1)]
                L = int( PARAMS['guide_halfwidth_mm'] / max(1e-9, pix_mm) )
                gp0 = b - (nref * L)[::-1].astype(int)
                gp1 = b + (nref * L)[::-1].astype(int)
                cv2.line(out, tuple(gp0), tuple(gp1), (180,180,255), 1, cv2.LINE_AA)
    return out

# ===================== Guided Centerline（核心） =====================
def _bilinear_at(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x = xy[:,0]; y = xy[:,1]
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    x0 = np.clip(x0, 0, W-1); x1 = np.clip(x1, 0, W-1)
    y0 = np.clip(y0, 0, H-1); y1 = np.clip(y1, 0, H-1)
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return (Ia*wa + Ib*wb + Ic*wc + Id*wd)

def _smooth_polyline(xy: np.ndarray, win: int) -> np.ndarray:
    if xy.shape[0] < 3 or win <= 2 or win % 2 == 0:
        return xy.copy()
    k = win//2
    out = xy.copy()
    for i in range(xy.shape[0]):
        a = max(0, i-k); b = min(xy.shape[0], i+k+1)
        out[i] = xy[a:b].mean(axis=0)
    return out

def _outer_inner_distance_fields(nearest_mask_u8: np.ndarray):
    ring_mask = _clean_mask_for_skeleton(nearest_mask_u8)
    if np.count_nonzero(ring_mask) == 0:
        return None, None, None
    contours, hierarchy = cv2.findContours(ring_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None: return None, None, None
    hierarchy = hierarchy[0]
    outer_idx, outer_area = -1, -1
    for i, h in enumerate(hierarchy):
        if h[3] == -1:
            a = cv2.contourArea(contours[i])
            if a > outer_area:
                outer_area = a; outer_idx = i
    if outer_idx < 0: return None, None, None
    child = hierarchy[outer_idx][2]
    inner_idx, inner_area = -1, -1
    while child != -1:
        a = cv2.contourArea(contours[child])
        if a > inner_area:
            inner_area = a; inner_idx = child
        child = hierarchy[child][0]
    if inner_idx < 0:
        return ring_mask, None, None

    H, W = ring_mask.shape
    outer_line = np.full((H, W), 255, np.uint8)
    inner_line = np.full((H, W), 255, np.uint8)
    cv2.drawContours(outer_line, [contours[outer_idx]], -1, 0, 1)
    cv2.drawContours(inner_line,  [contours[inner_idx]], -1, 0, 1)

    d_out = cv2.distanceTransform(outer_line, cv2.DIST_L2, 5).astype(np.float32)
    d_in  = cv2.distanceTransform(inner_line,  cv2.DIST_L2, 5).astype(np.float32)
    return ring_mask, d_out, d_in

def gcode_guided_centerline_v2(
    nearest_mask: np.ndarray,
    origin_xy: Tuple[float,float],
    pix_mm: float,
    gcode_xy: np.ndarray,
    gcode_normals: np.ndarray,
    *,
    halfwidth_mm: float = None,
    smooth_win: int = None,
    max_abs_mm: float = None
):
    H, W = nearest_mask.shape[:2]
    halfw_mm = float(halfwidth_mm if halfwidth_mm is not None else PARAMS['guide_halfwidth_mm'])
    halfw_px = max(1.0, halfw_mm / max(1e-9, pix_mm))
    win = int(smooth_win if smooth_win is not None else PARAMS['guide_smooth_win'])
    max_abs = float(max_abs_mm if max_abs_mm is not None else PARAMS['guide_max_offset_mm'])

    ring_mask, d_out, d_in = _outer_inner_distance_fields(nearest_mask)
    use_equidist = (d_out is not None) and (d_in is not None)
    if not use_equidist:
        dt = cv2.distanceTransform((nearest_mask>0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

    g_px = mach_xy_to_px_float(gcode_xy, origin_xy, pix_mm, H)
    delta_n = np.full(g_px.shape[0], np.nan, np.float32)

    def bil(img, xy):
        return _bilinear_at(img, xy)

    for i in range(g_px.shape[0]):
        p  = g_px[i]
        n  = gcode_normals[min(i, len(gcode_normals)-1)]
        n_px = np.array([n[0]/pix_mm, -n[1]/pix_mm], dtype=np.float32)
        n_px /= (np.linalg.norm(n_px) + 1e-12)

        M  = max(7, int(np.ceil(2*halfw_px)))
        ts = np.linspace(-halfw_px, +halfw_px, M).astype(np.float32)
        line_xy = p[None,:] + ts[:,None] * n_px[None,:]

        if use_equidist:
            phi = bil(d_out, line_xy) - bil(d_in, line_xy)      # 目标：phi ≈ 0
            in_ring = bil((ring_mask>0).astype(np.float32), line_xy) >= 0.5
            phi[~in_ring] = np.nan
            idx = np.where(np.isfinite(phi))[0]
            if idx.size == 0:
                continue
            z = None
            for a,b in zip(idx[:-1], idx[1:]):
                if np.sign(phi[a]) == 0:
                    z = ts[a]; break
                if np.sign(phi[a]) * np.sign(phi[b]) < 0:
                    t0,t1, f0,f1 = ts[a], ts[b], phi[a], phi[b]
                    z = t0 - f0*(t1-t0)/max(1e-9, (f1-f0))
                    break
            if z is None:
                k = int(np.nanargmin(np.abs(phi)))
                z = ts[k]
            delta_n[i] = float(z * pix_mm)
        else:
            on = bil((nearest_mask>0).astype(np.float32), line_xy) >= 0.5
            if not np.any(on):
                continue
            dvals = bil(dt, line_xy); dvals[~on] = -1.0
            k = int(np.argmax(dvals))
            if dvals[k] > 0:
                delta_n[i] = float(ts[k] * pix_mm)

    ok = np.isfinite(delta_n)
    if ok.any() and (~ok).any():
        I = np.arange(len(delta_n))
        delta_n[~ok] = np.interp(I[~ok], I[ok], delta_n[ok])
    delta_n = moving_average_1d(delta_n, win)
    delta_n = np.clip(delta_n, -max_abs, +max_abs)
    centerline_xy = gcode_xy + gcode_normals * delta_n[:,None]
    ratio = float(np.count_nonzero(ok)) / max(1,len(delta_n))
    stats = dict(valid=int(np.count_nonzero(ok)), total=int(len(delta_n)), ratio=ratio,
                 use_equidist=bool(use_equidist))
    return centerline_xy.astype(np.float32), delta_n.astype(np.float32), stats

# ===================== 偏移曲线 & G 代码导出 =====================
def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.copy()
    win = int(win) | 1
    k = win//2
    y = x.copy()
    for i in range(len(x)):
        a = max(0, i-k); b = min(len(x), i+k+1)
        y[i] = np.mean(x[a:b])
    return y

def compute_offsets_along_gcode(centerline_xy: np.ndarray, gcode_xy: np.ndarray, N: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = min(len(centerline_xy), len(gcode_xy), len(N))
    if M == 0:
        return np.empty((0,), float), np.empty((0,2), float)
    if len(centerline_xy) == len(gcode_xy):
        d = centerline_xy[:M] - gcode_xy[:M]
        delta_n = (d * N[:M]).sum(axis=1)
    else:
        tree = build_kdtree(gcode_xy)
        _, idx = tree.query(centerline_xy)
        delta_n = np.zeros(len(gcode_xy)); cnt = np.zeros(len(gcode_xy))
        for k, i in enumerate(np.asarray(idx, int)):
            dn = (centerline_xy[k] - gcode_xy[i]).dot(N[i])
            delta_n[i] += dn; cnt[i] += 1
        ok = cnt > 0
        if not ok.any():
            return np.empty((0,), float), np.empty((0,2), float)
        delta_n[ok] /= cnt[ok]
        ii = np.arange(len(delta_n))
        delta_n[~ok] = np.interp(ii[~ok], ii[ok], delta_n[ok])
    dxy = N[:M] * delta_n[:M, None]
    return delta_n[:M], dxy

def save_offset_csv(s: np.ndarray, delta_n: np.ndarray, dxy: np.ndarray, out_csv: Union[str, Path]):
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('s_mm,delta_n_mm,dx_mm,dy_mm\n')
        for i in range(len(delta_n)):
            f.write(f"{s[i]:.6f},{delta_n[i]:.6f},{dxy[i,0]:.6f},{dxy[i,1]:.6f}\n")
    print('[SAVE]', out_csv)

def write_linear_gcode(xy: np.ndarray, out_path: Union[str, Path], feed: Optional[float]=None):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('; corrected by guided centerline (flattened-plane v2)\n')
        f.write('G90 ; absolute\n')
        if xy.shape[0] > 0:
            x0, y0 = xy[0]
            f.write(f'G0 X{x0:.4f} Y{y0:.4f}\n')
        for i in range(1, xy.shape[0]):
            x, y = xy[i]
            if feed is not None and i == 1:
                f.write(f'G1 X{x:.4f} Y{y:.4f} F{feed:.1f}\n')
            else:
                f.write(f'G1 X{x:.4f} Y{y:.4f}\n')
    print('[SAVE]', out_path)

# ===================== 平面拟合/展平（NEW） =====================
def _fit_plane_ransac(XYZ: np.ndarray, thr: float=0.8, iters: int=500, sample_cap: int=120000) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    输入：N×3 点（mm）
    返回：plane=(a,b,c,d)（ax+by+cz+d=0），inlier_ratio，inlier_mask
    """
    pts = np.asarray(XYZ, float)
    if pts.shape[0] < 50:
        return np.array([0,0,1, -np.median(pts[:,2])]), 0.0, np.zeros(len(pts), bool)
    if pts.shape[0] > sample_cap:
        idx = np.random.choice(pts.shape[0], sample_cap, replace=False)
        pts = pts[idx]
    N = pts.shape[0]
    best_inl = -1; best_plane = None; best_mask = None
    rng = np.random.default_rng(12345)
    for _ in range(max(50, iters)):
        ids = rng.choice(N, 3, replace=False)
        p0,p1,p2 = pts[ids]
        v1 = p1 - p0; v2 = p2 - p0
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        d = -np.dot(n, p0)
        dist = np.abs(pts @ n + d)
        inlier = dist <= thr
        cnt = int(np.count_nonzero(inlier))
        if cnt > best_inl:
            best_inl = cnt; best_plane = np.append(n, d); best_mask = inlier
    if best_plane is None:
        return np.array([0,0,1, -np.median(pts[:,2])]), 0.0, np.zeros(len(pts), bool)
    # 最小二乘细化（用内点）
    P = pts[best_mask]
    X = np.c_[P[:,0], P[:,1], np.ones(len(P))]
    y = P[:,2]
    # z ≈ a*x + b*y + c
    try:
        abC, *_ = np.linalg.lstsq(X, y, rcond=None)
        a,b,c = abC
        n = np.array([-a, -b, 1.0]); n /= np.linalg.norm(n)
        d = -c
        plane = np.array([n[0], n[1], n[2], d])
    except Exception:
        plane = best_plane
    inlier_ratio = float(best_inl) / float(N)
    return plane, inlier_ratio, best_mask

def _plane_z(ax: float, by: float, c: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # 与 lstsq 一致：z ≈ a*x + b*y + c
    return ax * x + by * y + c

def _flatten_height_with_plane(height: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float,
                               plane: np.ndarray) -> np.ndarray:
    """
    把顶视高度图按平面展平：h_flat = Z - planeZ(X,Y)
    """
    H, W = height.shape
    x0, y0 = origin_xy
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = x0 + xx * pix_mm
    Y = (y0 + H*pix_mm) - yy * pix_mm
    n = plane[:3]; d = plane[3]
    # 由 n,d 表达式换成 z = a*x + b*y + c
    if abs(n[2]) < 1e-6:
        # 垂直平面，直接返回原高度（不展平）
        return height.copy()
    a = -n[0]/n[2]; b = -n[1]/n[2]; c = -d/n[2]
    Zp = _plane_z(a, b, c, X, Y)
    h_flat = height - Zp.astype(np.float32)
    return h_flat

# ===================== 可视化增强（NEW） =====================
def _render_histogram(values: np.ndarray, width=360, height=140, bins=40, title:str='') -> np.ndarray:
    img = np.full((height, width, 3), 20, np.uint8)
    if values.size == 0 or not np.isfinite(values).any():
        return img
    v = values[np.isfinite(values)]
    hist, edges = np.histogram(v, bins=bins)
    hist = hist.astype(float);
    if hist.max() > 0: hist = hist / hist.max()
    for i, h in enumerate(hist):
        x0 = int(i * width / bins); x1 = int((i+1) * width / bins)
        y1 = height - 10
        y0 = int(y1 - h * (height-30))
        cv2.rectangle(img, (x0, y0), (x1-1, y1), (80,200,255), -1)
    cv2.rectangle(img, (0,0), (width-1, height-1), (90,90,90), 1)
    if title:
        cv2.putText(img, title, (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
    return img

def _compose_quicklook(vis_top, phi_vis=None, hist=None, hud_text:str=''):
    h_list = [vis_top]
    if phi_vis is not None:
        h_list.append(phi_vis)
    if hist is not None:
        h_list.append(hist)
    # 垂直拼接
    out = h_list[0]
    for im in h_list[1:]:
        w = min(out.shape[1], im.shape[1])
        out = cv2.resize(out, (w, int(out.shape[0]*w/out.shape[1])))
        im = cv2.resize(im, (w, int(im.shape[0]*w/im.shape[1])))
        out = np.vstack([out, im])
    if hud_text:
        cv2.putText(out, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return out

# ============================ 主流程（集成改进） ============================
def run():
    cfg = PARAMS
    out_dir = Path(cfg['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 外参与 G 代码
    R, t, _ = load_extrinsic(cfg['T_path'])                         # 来自手眼标定（mm）  （参考你原实现）  :contentReference[oaicite:12]{index=12}
    g_raw, feed = parse_gcode_xy(cfg['gcode_path'], step_mm=cfg['guide_step_mm'])
    step_mm = float(cfg.get('guide_step_mm', 1.0))
    g_xy  = resample_polyline(g_raw, max(0.2, step_mm)) if g_raw.size > 0 else g_raw
    ref_tree = build_kdtree(g_xy) if g_xy.size > 0 else None
    T_ref, N_ref = tangent_normal(g_xy) if g_xy.size > 0 else (np.zeros((0,2)), np.zeros((0,2)))

    # 2) 相机
    stream = PCamMLSStream(); stream.open()

    roi_mode = str(cfg.get('roi_mode', 'none')).lower()
    corr_mode = str(cfg.get('corr_mode', 'vector_median')).lower()  # 默认更稳健：向量 EMA
    if corr_mode in ('vector','vector_mean','vector_median'):
        corr = VectorEmaCorrector(cfg['ema_alpha'], cfg['deadband_mm'], cfg['clip_mm'], cfg['max_step_mm'],
                                  use_mean=('mean' in corr_mode))
        print('[INFO] 使用 VectorEmaCorrector 模式 =', corr_mode)
    else:
        corr = EmaCorrector(cfg['ema_alpha'], cfg['deadband_mm'], cfg['clip_mm'], cfg['max_step_mm'])
        print('[INFO] 使用 EmaCorrector (median_scalar)')

    print('[INFO] 拉流中… (q 退出, s 截图, {} 导出纠偏)  mode={}  plane_enable={}  guard={}'
          .format(cfg['export_on_key'], roi_mode, cfg['plane_enable'], cfg['Guard']))

    frame_id = 0
    try:
        while True:
            P_cam, _ = stream.read_pointcloud(2000)
            if P_cam is None:
                print('[WARN] 无深度帧'); continue
            H, W, _ = P_cam.shape
            P_mach = transform_cam_to_machine_grid(P_cam, R, t)

            # 3) ROI（新增 gcode_bounds）
            m_valid = valid_mask_hw(P_mach)
            if roi_mode == 'camera_rect':
                m_roi = camera_rect_mask(H, W, cfg['cam_roi_xywh'])
                m_select = m_valid & m_roi
            elif roi_mode == 'machine':
                m_roi = machine_rect_mask(P_mach, cfg['roi_center_xy'], cfg['roi_size_mm'])
                m_select = m_valid & m_roi
            elif roi_mode == 'gcode_bounds' and g_xy.size>0:
                gx0,gy0 = g_xy.min(0); gx1,gy1 = g_xy.max(0)
                cx, cy = (gx0+gx1)*0.5, (gy0+gy1)*0.5
                sz = max(gx1-gx0, gy1-gy0) + cfg['bounds_margin_mm']*2
                m_roi = machine_rect_mask(P_mach, (cx,cy), sz)
                m_select = m_valid & m_roi
            else:
                m_select = m_valid

            # 4) 顶视边界/分辨率
            x0,x1,y0,y1 = compute_bounds_xy_from_mask(P_mach, m_select,
                                                       cfg['bounds_qlo'], cfg['bounds_qhi'],
                                                       cfg['bounds_margin_mm'])
            pix_mm = adjust_pixel_size(x0,x1,y0,y1, float(cfg['pixel_size_mm']), cfg['max_grid_pixels'])

            # 5) 顶视投影（右手系，取最高 Z） —— “未展平”的 Z 高度图
            height, mask_top, origin_xy = project_topdown_from_grid(P_mach, m_select, pix_mm, (x0,x1,y0,y1))

            # 5.1 平面拟合与展平（NEW）
            plane = None; inlier_ratio = float('nan')
            if cfg.get('plane_enable', True) and np.isfinite(height).any():
                # 从 ROI 内点云直接拟合平面
                pts = P_mach[m_select & (np.isfinite(P_mach).all(axis=2))]
                plane, inlier_ratio, inl_mask = _fit_plane_ransac(pts,
                    thr=cfg['plane_ransac_thresh_mm'],
                    iters=cfg['plane_ransac_iters'],
                    sample_cap=cfg['plane_sample_cap'])
                height_flat = _flatten_height_with_plane(height, origin_xy, pix_mm, plane)
                src_for_nearest = height_flat
            else:
                height_flat = height.copy()
                src_for_nearest = height

            # 6) 最近表层（在“展平后残差高度”域上取层）
            nearest_mask, z_ref, (z_low, z_high) = extract_nearest_surface_mask_from_height(
                src_for_nearest, (mask_top > 0),
                z_select=cfg['z_select'],
                depth_margin_mm=cfg['depth_margin_mm'],
                use_percentile=cfg['nearest_use_percentile'],
                qlo=cfg['nearest_qlo'],
                qhi=cfg['nearest_qhi'],
                morph_open=cfg['morph_open'],
                morph_close=cfg['morph_close'],
                min_component_area_px=cfg['min_component_area_px']
            )

            # 7) 骨架（用于可视化与回退）
            skel_bgr = extract_skeleton_universal(nearest_mask, visualize=True)
            if skel_bgr is None:
                cv2.imshow('Centerline vs G-code (RHR)', np.zeros((480, 640, 3), np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            skel_gray = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY)
            path_px = skeleton_to_path_px_topo(skel_gray)

            # 8) G 代码引导的中轴线（核心）
            use_guided = bool(PARAMS.get('guide_enable', True)) and g_xy.size > 1
            if use_guided:
                centerline_xy, delta_n, rep = gcode_guided_centerline_v2(
                    nearest_mask, origin_xy, pix_mm, g_xy, N_ref,
                    halfwidth_mm=PARAMS['guide_halfwidth_mm'],
                    smooth_win=PARAMS['guide_smooth_win'],
                    max_abs_mm=PARAMS['guide_max_offset_mm']
                )
                if rep['ratio'] < PARAMS['guide_min_valid_ratio']:
                    centerline_xy = px_to_mach_xy(path_px, origin_xy, pix_mm, height.shape[0])
                    e_idx, _, e_n = project_points_to_path(centerline_xy, g_xy, ref_tree, N_ref)
                else:
                    e_idx = np.arange(len(delta_n), dtype=int)
                    e_n = delta_n
            else:
                centerline_xy = px_to_mach_xy(path_px, origin_xy, pix_mm, height.shape[0])
                e_idx, _, e_n = project_points_to_path(centerline_xy, g_xy, ref_tree, N_ref)

            # 9) 可视化叠加（顶视 + 最近表层 + 偏差箭头 + 法向采样线）
            vis_top_raw = render_topdown(height, mask_top, origin_xy, pix_mm, gcode_xy=g_xy)
            # 把最近层叠加显示
            overlay = cv2.addWeighted(vis_top_raw, 1.0,
                                      cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR), 0.25, 0)
            vis_cmp = draw_deviation_overlay(overlay, origin_xy, pix_mm, g_xy, centerline_xy, e_idx, e_n,
                                             arrow_stride=int(PARAMS['arrow_stride']),
                                             draw_probes=PARAMS.get('draw_normal_probes', True),
                                             N_ref=N_ref)
            vis_cmp = draw_machine_axes_overlay(vis_cmp, origin_xy, pix_mm)

            # 10) 在线纠偏（EMA）
            if cfg.get('print_corr', True) and e_n.size > 0 and N_ref.size > 0:
                dxdy, stats = corr.update(e_idx, e_n, N_ref)
                print("CORR frame={:06d}  mean={:+.3f}  med={:+.3f}  p95={:.3f}  ->  dx={:+.3f}  dy={:+.3f}  [mm]"
                      .format(frame_id, float(np.mean(e_n)) if e_n.size>0 else 0.0,
                              float(np.median(e_n)) if e_n.size>0 else 0.0,
                              float(np.percentile(np.abs(e_n), 95)) if e_n.size>0 else 0.0,
                              dxdy[0], dxdy[1]))
            else:
                dxdy = np.zeros(2, float); stats = dict(mean=0, median=0, p95=0)

            # 11) HUD & Guard 判定
            dev_mean = float(np.mean(e_n)) if e_n.size > 0 else 0.0
            dev_med  = float(np.median(e_n)) if e_n.size > 0 else 0.0
            dev_p95  = float(np.percentile(np.abs(e_n), 95)) if e_n.size > 0 else 0.0
            valid_ratio = float(len(e_n)) / max(1, g_xy.shape[0]) if e_n.size>0 else 0.0
            plane_info = f"inlier={inlier_ratio:.2f}" if np.isfinite(inlier_ratio) else "inlier=nan"

            guard = cfg['Guard']
            guard_enable = guard.get('enable', True)
            guard_ok = True
            reasons = []
            if guard_enable:
                if rep.get('ratio', 0.0) < guard.get('min_valid_ratio', 0.60):
                    guard_ok = False; reasons.append(f"valid_ratio {rep.get('ratio',0.0):.2f} < {guard.get('min_valid_ratio')}")
                if dev_p95 > guard.get('max_abs_p95_mm', 0.80):
                    guard_ok = False; reasons.append(f"p95 {dev_p95:.2f} > {guard.get('max_abs_p95_mm')}")
                if np.isfinite(inlier_ratio) and inlier_ratio < guard.get('min_plane_inlier_ratio', 0.55):
                    guard_ok = False; reasons.append(f"plane_inlier {inlier_ratio:.2f} < {guard.get('min_plane_inlier_ratio')}")
            guard_str = "PASS" if guard_ok else "FAIL"

            Ht, Wt = height.shape
            txt = ('plane[%s]  band=[%.2f,%.2f]mm  pix=%.2fmm  grid=%dx%d  '
                   'dev(mean/med/p95)=%+.3f/%+.3f/%.3f  guided=%.2f  Guard=%s'
                   % (plane_info, *( (z_low, z_high) if np.isfinite(z_low) and np.isfinite(z_high) else (0.0,0.0) ),
                      pix_mm, Wt, Ht, dev_mean, dev_med, dev_p95, rep.get('ratio', 0.0), guard_str))
            cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (0,255,0) if guard_ok else (0,0,255), 1, cv2.LINE_AA)

            # 调试纠偏矢量
            if cfg.get('debug_vis', True):
                vis_cmp_dbg = draw_correction_debug(vis_cmp, origin_xy, pix_mm, g_xy, corr, corr_mode)
            else:
                vis_cmp_dbg = vis_cmp

            cv2.imshow('Top-Down + Nearest Surface + Skeleton', overlay)
            cv2.imshow('NearestSurfaceMask', nearest_mask)
            cv2.imshow('Centerline vs G-code (RHR)', vis_cmp_dbg)

            # 12) 导出/截图/分辨率调节
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                outp = out_dir / f'centerline_cmp_{frame_id:06d}.png'
                cv2.imwrite(str(outp), vis_cmp); print('[SAVE]', outp)
            elif key in (ord('='), ord('+')):
                PARAMS['pixel_size_mm'] = max(0.1, float(PARAMS['pixel_size_mm'])*0.8)
            elif key == ord('-'):
                PARAMS['pixel_size_mm'] = min(5.0, float(PARAMS['pixel_size_mm'])/0.8)
            elif key == ord(cfg['export_on_key']) and g_xy.size>1 and centerline_xy.size>0:
                # Guard：不通过就拒绝导出
                if guard_enable and (not guard_ok):
                    print('[GUARD] 导出被拒绝：', '; '.join(reasons))
                else:
                    # 计算沿轨迹的有符号法向偏移
                    delta_n_off, dxy_vec = compute_offsets_along_gcode(centerline_xy, g_xy, N_ref)
                    if delta_n_off.size > 0:
                        delta_n_off = np.clip(moving_average_1d(delta_n_off, PARAMS['guide_smooth_win']),
                                              -PARAMS['guide_max_offset_mm'], PARAMS['guide_max_offset_mm'])
                        g_xy_corr = g_xy[:len(delta_n_off)] + N_ref[:len(delta_n_off)] * delta_n_off[:,None]
                        seg = np.linalg.norm(np.diff(g_xy[:len(delta_n_off)], axis=0), axis=1)
                        s = np.concatenate([[0.0], np.cumsum(seg)])
                        save_offset_csv(s, delta_n_off, N_ref[:len(delta_n_off)]*delta_n_off[:,None], PARAMS['offset_csv'])
                        write_linear_gcode(g_xy_corr, PARAMS['corrected_gcode'], feed=feed)

                        # quicklook & report
                        if PARAMS.get('dump_quicklook', True):
                            hist = _render_histogram(delta_n_off, title='delta_n histogram (mm)')
                            quick = _compose_quicklook(vis_cmp_dbg, None, hist,
                                hud_text='Exported: offsets & corrected.gcode')
                            qp = out_dir / 'quicklook.png'
                            cv2.imwrite(str(qp), quick); print('[SAVE]', qp)
                        if PARAMS.get('dump_report', True):
                            rep_json = dict(
                                valid_ratio=rep.get('ratio', 0.0),
                                dev_mean=dev_mean, dev_median=dev_med, dev_p95=dev_p95,
                                plane_inlier_ratio=inlier_ratio,
                                guard=guard, guard_ok=guard_ok, reasons=reasons,
                                gcode_points=int(len(g_xy)), centerline_points=int(len(centerline_xy)),
                                pixel_size_mm=pix_mm
                            )
                            rp = out_dir / 'report.json'
                            with rp.open('w', encoding='utf-8') as f:
                                json.dump(rep_json, f, ensure_ascii=False, indent=2)
                            print('[SAVE]', rp)
            frame_id += 1

    finally:
        try: stream.close()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
