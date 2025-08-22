#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSW 在线视觉管线（All-in-One 单文件，修正版）
===========================================

修复要点
--------
- **ROI 内无有效点导致黑屏**：增加了**有效点筛选**（过滤 Z<=0、(0,0,0)、NaN），并基于有效点**自动居中/自适应** ROI。
- **首帧自动居中更鲁棒**：只用“有效点”计算中位数；若还为空，提供 `c/C/a/A` 四个热键快速校正 ROI。
- **可视化诊断**：HUD 中显示“总有效点数/ROI 内有效点数”；当 ROI 命中为 0 时在画面顶端显示红色提示条。

功能概述（与上一版一致）
----------------------
- 读取外参 `T_cam2machine.npy`，相机点云→机床坐标系。
- 机床系下 **Z 轴俯视正交投影**，得到高度图与“最近表面”掩码。
- 掩码→**中轴线拟合**（骨架/最长路/RDP/平滑/等弧长采样），输出机床系 XY 曲线与调试图。
- 解析 G 代码（G0/G1），重采样、法向场；计算横向偏差 eₙ，EMA/死区/限速/限步输出 `(dx,dy)`。
- 丰富可视化：高度伪彩、掩码、理论路径、实际中轴线、偏差箭头与统计条、中轴线调试图。

热键
----
- `q` 退出；`s` 截图；`-`/`=` 改栅格分辨率；`i/k/j/l` 平移 ROI；`[`/`]` 缩放 ROI；`r` 重置。
- `c`：把 ROI **居中到全局有效点中位数**；`C`：居中到**当前 ROI**内有效点中位数。
- `a`：按全局有效点 **5–95% 分位**自适应 ROI；`A`：按 **1–99% 分位**自适应 ROI。
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import collections
import math
import numpy as np
import cv2

# --- 可选依赖 ---
try:
    import pcammls  # PercipioSDK
except Exception:
    pcammls = None
try:
    import cv2.ximgproc as xip
except Exception:
    xip = None
try:
    from skimage.morphology import skeletonize as sk_skeletonize
except Exception:
    sk_skeletonize = None
try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None
try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

# =============================
# 配置
# =============================
CONFIG: Dict = dict(
    T_path='T_cam2machine.npy',
    gcode_path='path/example.gcode',
    pixel_size_mm=0.5,
    roi_size_mm=120.0,
    roi_center_xy=[0.0, 0.0],
    z_select='max',                 # 'max' 或 'min'
    min_points_per_cell=1,
    morph_open=3,
    morph_close=5,
    # 有效点判定
    require_positive_z=True,        # 过滤 Z<=0
    reject_zero_xyz=True,           # 过滤 (0,0,0)
    # 中轴线拟合
    rdp_epsilon_px=2.0,
    sg_window=11,
    sg_polyorder=2,
    resample_step_mm=1.0,
    # 偏差控制
    ema_alpha=0.3,
    deadband_mm=0.05,
    clip_mm=2.0,
    max_step_mm=0.15,
    max_rate_mm_s=5.0,
    # 显示/输出
    colormap=getattr(cv2, 'COLORMAP_TURBO', getattr(cv2, 'COLORMAP_JET', 2)),
    out_dir='out/frames'
)

# =============================
# 工具
# =============================

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_extrinsic(T_path: str | Path):
    data = np.load(T_path, allow_pickle=True).item()
    return np.asarray(data['R'], float), np.asarray(data['t'], float), np.asarray(data['T'], float)


def transform_cam_to_machine(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P_cam.T).T + t

# =============================
# G 代码与几何
# =============================

def parse_gcode_xy(path: str | Path) -> np.ndarray:
    pts: List[List[float]] = []
    if not path or not Path(path).exists():
        return np.empty((0,2), float)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        x=y=None
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(';') or line.startswith('('):
                continue
            if ';' in line:
                line = line.split(';',1)[0]
            while '(' in line and ')' in line:
                a,b = line.find('('), line.find(')')
                if a<0 or b<0 or b<=a: break
                line = (line[:a] + ' ' + line[b+1:]).strip()
            toks = line.split()
            if not toks: continue
            cmd = toks[0].upper()
            if cmd in ('G0','G00','G1','G01'):
                for t in toks[1:]:
                    u=t.upper()
                    if u.startswith('X'):
                        try: x=float(u[1:])
                        except: pass
                    elif u.startswith('Y'):
                        try: y=float(u[1:])
                        except: pass
                if x is not None and y is not None:
                    pts.append([x,y])
    return np.asarray(pts, float) if pts else np.empty((0,2), float)


def resample_polyline(poly: np.ndarray, step: float) -> np.ndarray:
    if poly.shape[0] < 2: return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = float(seg.sum())
    if L <= 1e-9: return poly[[0]].copy()
    n = max(2, int(math.ceil(L/max(1e-6, step))))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out=[]; j=0
    for si in s:
        while j < len(seg) and si > cs[j+1]: j+=1
        if j >= len(seg): out.append(poly[-1]); continue
        t = (si - cs[j]) / max(seg[j],1e-9)
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
    if pts.size == 0: return None
    if KDTree is not None:
        return KDTree(pts)
    class _Lin:
        def __init__(self, A): self.A = A
        def query(self, B):
            B = np.atleast_2d(B)
            d2 = ((B[:,None,:]-self.A[None,:,:])**2).sum(axis=2)
            idx = d2.argmin(axis=1)
            return np.sqrt(d2[np.arange(len(B)), idx]), idx
    return _Lin(pts)

# =============================
# 俯视正交投影
# =============================

def orthographic_project_top(P_mach: np.ndarray, roi_center: np.ndarray, roi_size: float, pix_mm: float,
                             z_select: str='max', min_points_per_cell: int=1) -> tuple[np.ndarray,np.ndarray,tuple[float,float],int]:
    half = roi_size*0.5
    cx, cy = float(roi_center[0]), float(roi_center[1])
    x0, x1 = cx-half, cx+half
    y0, y1 = cy-half, cy+half

    X,Y,Z = P_mach[:,0], P_mach[:,1], P_mach[:,2]
    # ROI + 有效性：Z>0、不是零向量、不是 NaN
    valid = np.isfinite(X)&np.isfinite(Y)&np.isfinite(Z)
    if CONFIG['require_positive_z']:
        valid &= (Z > 0)
    if CONFIG['reject_zero_xyz']:
        valid &= (np.abs(X)+np.abs(Y)+np.abs(Z) > 1e-6)
    inroi = (X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)
    sel = valid & inroi

    W = H = int(max(2, round(roi_size/pix_mm)))
    if not np.any(sel):
        return np.full((H,W), np.nan, np.float32), np.zeros((H,W), np.uint8), (x0,y0), 0

    X, Y, Z = X[sel], Y[sel], Z[sel]
    ix = np.clip(((X-x0)/pix_mm).astype(np.int32), 0, W-1)
    iy = np.clip(((Y-y0)/pix_mm).astype(np.int32), 0, H-1)

    height = np.full((H,W), np.nan, np.float32)
    count  = np.zeros((H,W), np.int32)
    if z_select=='max':
        for xg,yg,zg in zip(ix,iy,Z):
            if not np.isfinite(height[yg,xg]) or zg>height[yg,xg]: height[yg,xg]=zg
            count[yg,xg]+=1
    else:
        for xg,yg,zg in zip(ix,iy,Z):
            if not np.isfinite(height[yg,xg]) or zg<height[yg,xg]: height[yg,xg]=zg
            count[yg,xg]+=1
    mask = (count>=min_points_per_cell).astype(np.uint8)*255
    return height, mask, (x0,y0), int(sel.sum())

# =============================
# 中轴线拟合（骨架→路径）
# =============================

def _skeletonize(mask: np.ndarray, use_ximgproc_first: bool=True) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if xip is not None and use_ximgproc_first:
        try:
            sk = xip.thinning(m, thinningType=xip.THINNING_ZHANGSUEN)
            return (sk>0).astype(np.uint8)*255
        except Exception:
            pass
    if sk_skeletonize is not None:
        try:
            sk = sk_skeletonize(m.astype(bool))
            return (sk.astype(np.uint8))*255
        except Exception:
            pass
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    sk = np.zeros_like(m)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==0 and dy==0: continue
            sk = np.maximum(sk, (dist < np.roll(np.roll(dist, dy, 0), dx, 1)).astype(np.uint8))
    ridge = (sk==0).astype(np.uint8) & (m>0)
    return ridge.astype(np.uint8)*255


def _skeleton_graph(skel: np.ndarray):
    ys, xs = np.where(skel>0)
    nodes = list(zip(ys.tolist(), xs.tolist()))
    S = set(nodes)
    adj = {n:set() for n in nodes}
    H, W = skel.shape
    for y, x in nodes:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                yy, xx = y+dy, x+dx
                if 0<=yy<H and 0<=xx<W and (yy,xx) in S:
                    adj[(y,x)].add((yy,xx))
    deg = {n:len(adj[n]) for n in nodes}
    endpoints = [n for n,d in deg.items() if d==1]
    junctions = [n for n,d in deg.items() if d>=3]
    return nodes, adj, endpoints, junctions


def _longest_path_from_graph(adj, endpoints):
    import collections
    def bfs(start):
        vis={start:None}; q=collections.deque([start])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if v not in vis:
                    vis[v]=u; q.append(v)
        far = max(vis.keys(), key=lambda k:(abs(k[0]-start[0])+abs(k[1]-start[1])))
        return far, vis
    if endpoints:
        a=endpoints[0]; b,pre=bfs(a); c,pre2=bfs(b)
        path=[c]; p=pre2[c]
        while p is not None: path.append(p); p=pre2[p]
        return path[::-1]
    anyn=next(iter(adj.keys())); b,pre=bfs(anyn); c,pre2=bfs(b)
    path=[c]; p=pre2[c]
    while p is not None: path.append(p); p=pre2[p]
    return path[::-1]


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    if len(points) < 3: return points
    a, b = points[0], points[-1]
    ab = b - a; lab2 = (ab*ab).sum() + 1e-12
    dmax = -1; idx=-1
    for i in range(1, len(points)-1):
        ap = points[i] - a
        t = np.clip(np.dot(ap, ab)/lab2, 0, 1)
        proj = a + t*ab
        d = np.linalg.norm(points[i]-proj)
        if d>dmax: dmax=d; idx=i
    if dmax>eps:
        left = _rdp(points[:idx+1], eps)
        right= _rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


def _smooth_polyline(px: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if len(px) < max(5, window): return px
    if savgol_filter is not None and window%2==1:
        xs = savgol_filter(px[:,0], window, polyorder)
        ys = savgol_filter(px[:,1], window, polyorder)
        return np.stack([xs, ys], axis=1)
    k = max(3, window|1); pad=k//2
    ext = np.pad(px, ((pad,pad),(0,0)), mode='edge')
    ker = np.ones((k,1))/k
    xs = np.convolve(ext[:,0], ker[:,0], mode='valid')
    ys = np.convolve(ext[:,1], ker[:,0], mode='valid')
    return np.stack([xs, ys], axis=1)


def _resample_polyline(px: np.ndarray, step: float) -> np.ndarray:
    if len(px) < 2: return px
    seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
    L = float(seg.sum());
    if L < 1e-9: return px[[0]].copy()
    n = max(2, int(np.ceil(L/step)))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out=[]; j=0
    for si in s:
        while j < len(seg) and si > cs[j+1]: j+=1
        if j >= len(seg): out.append(px[-1]); continue
        t = (si - cs[j]) / max(seg[j],1e-9)
        out.append(px[j]*(1-t) + px[j+1]*t)
    return np.asarray(out, float)


def _curvature(px: np.ndarray) -> np.ndarray:
    if len(px) < 3: return np.zeros((len(px),), float)
    d1 = np.gradient(px, axis=0); d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0])
    den = (d1[:,0]**2 + d1[:,1]**2)**1.5 + 1e-12
    kappa = num/den; kappa[~np.isfinite(kappa)] = 0
    return kappa


def fit_centerline(mask: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float,
                   background: Optional[np.ndarray]=None,
                   cfg: Optional[Dict]=None):
    C = dict(
        use_ximgproc_first=True,
        rdp_epsilon_px=CONFIG['rdp_epsilon_px'],
        sg_window=CONFIG['sg_window'],
        sg_polyorder=CONFIG['sg_polyorder'],
        resample_step_mm=CONFIG['resample_step_mm'],
        curvature_amp=200.0,
        colormap=CONFIG['colormap'],
    )
    if cfg: C.update(cfg)

    skel = _skeletonize(mask, C['use_ximgproc_first'])
    ys, xs = np.where(skel>0)
    if len(xs)==0:
        dbg = background if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.empty((0,2), float), dbg, dict(reason='empty-mask')

    nodes, adj, endpoints, junctions = _skeleton_graph(skel)
    path_nodes = _longest_path_from_graph(adj, endpoints)
    path_px = np.array([[x,y] for (y,x) in path_nodes], float)

    if len(path_px)>=3:
        path_px = _rdp(path_px, C['rdp_epsilon_px'])
    path_px = _smooth_polyline(path_px, C['sg_window'], C['sg_polyorder'])

    step_px = max(1.0, float(C['resample_step_mm'])/max(1e-6, pix_mm))
    path_px = _resample_polyline(path_px, step_px)

    # 曲率可视化
    kappa = _curvature(path_px)
    d = np.gradient(path_px, axis=0)
    T = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    N = np.stack([-T[:,1], T[:,0]], axis=1)

    # 背景
    H, W = mask.shape
    vis = background.copy() if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if vis.shape[:2] != mask.shape:
        vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_NEAREST)

    # 骨架淡显
    sk3 = cv2.cvtColor(((skel>0)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(vis, 0.9, sk3, 0.3, 0)

    # 端点/分叉
    for y,x in endpoints: cv2.circle(vis, (x,y), 3, (0,255,255), -1)
    for y,x in junctions: cv2.circle(vis, (x,y), 3, (0,165,255), -1)

    # 主路径曲率着色
    if len(path_px) >= 2:
        k = kappa.copy()
        if k.size != len(path_px):
            if kappa.size>1:
                k = np.interp(np.linspace(0,1,len(path_px)), np.linspace(0,1,len(kappa)), kappa)
            else:
                k = np.zeros(len(path_px))
        k = np.clip(k * C['curvature_amp'], 0, 1)
        cm = cv2.applyColorMap((k*255).astype(np.uint8), C['colormap'])
        for i in range(len(path_px)-1):
            c = tuple(int(v) for v in cm[i,0].tolist())
            cv2.line(vis, tuple(path_px[i][::-1]), tuple(path_px[i+1][::-1]), c, 2, cv2.LINE_AA)

    # 法线与索引
    for i in range(0, len(path_px), 20):
        x,y = path_px[i]; nx,ny = N[i]*12
        cv2.arrowedLine(vis, (int(x),int(y)), (int(x+nx),int(y+ny)), (0,255,0), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(vis, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    legend = np.zeros((60, vis.shape[1], 3), np.uint8)
    cv2.putText(legend, 'yellow=endpoints  orange=junctions  line=curvature  green=normals',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    dbg = np.vstack([vis, legend])

    # 像素路径 -> 机床系 XY
    x0, y0 = origin_xy
    X = x0 + (path_px[:,0] + 0.5) * pix_mm
    Y = y0 + (path_px[:,1] + 0.5) * pix_mm
    centerline_xy = np.stack([X, Y], axis=1)

    return centerline_xy, dbg, dict(path_px=path_px, curvature=kappa)

# =============================
# 偏差计算/控制
# =============================

def project_points_to_path(points: np.ndarray, ref_xy: np.ndarray, ref_tree, ref_normals: np.ndarray):
    if points.size == 0 or ref_xy.size == 0:
        return np.empty((0,), int), np.empty((0,2)), np.empty((0,))
    _, idx = ref_tree.query(points)
    idx = np.asarray(idx, dtype=int)
    nearest = ref_xy[idx]
    N = ref_normals[np.clip(idx, 0, len(ref_normals)-1)]
    e_n = ((points - nearest) * N).sum(axis=1)
    return idx, nearest, e_n

@dataclass
class ControllerConfig:
    ema_alpha: float = CONFIG['ema_alpha']
    deadband_mm: float = CONFIG['deadband_mm']
    clip_mm: float = CONFIG['clip_mm']
    max_step_mm: float = CONFIG['max_step_mm']
    max_rate_mm_s: float = CONFIG['max_rate_mm_s']

class DeviationController:
    def __init__(self, cfg: Optional[ControllerConfig]=None):
        self.cfg = cfg or ControllerConfig()
        self._ema = 0.0
        self._prev_out = np.zeros(2, float)
        self._t_last = None
    def _now(self):
        import time
        return time.time()
    def _smooth(self, e: float) -> float:
        a = float(np.clip(self.cfg.ema_alpha, 0.0, 1.0))
        self._ema = a*e + (1-a)*self._ema
        return self._ema
    def update(self, e_n: np.ndarray, normals: np.ndarray):
        if e_n.size == 0:
            return np.zeros(2, float), dict(mean=0.0, median=0.0, p95=0.0, n=0)
        med = float(np.median(e_n))
        idx_med = int(np.argsort(e_n)[len(e_n)//2])
        nvec = normals[np.clip(idx_med, 0, len(normals)-1)]
        if abs(med) < self.cfg.deadband_mm: med = 0.0
        med = float(np.clip(med, -self.cfg.clip_mm, self.cfg.clip_mm))
        med_s = self._smooth(med)
        target = nvec * med_s
        t = self._now()
        dt = 0.0 if self._t_last is None else max(1e-3, t - self._t_last)
        self._t_last = t
        delta = target - self._prev_out
        mag = float(np.linalg.norm(delta))
        max_delta = self.cfg.max_rate_mm_s * dt
        if mag > max_delta:
            delta = delta * (max_delta / (mag + 1e-9))
        delta = np.clip(delta, -self.cfg.max_step_mm, self.cfg.max_step_mm)
        out = self._prev_out + delta
        self._prev_out = out
        stats = dict(mean=float(np.mean(e_n)), median=float(np.median(e_n)), p95=float(np.percentile(e_n,95)), n=int(len(e_n)))
        return out, stats

# =============================
# 可视化
# =============================

def render_topdown(height, mask, origin_xy, pix_mm, gcode_xy, centerline_xy):
    H,W = height.shape
    vis = np.zeros((H,W,3), np.uint8)
    h = height.copy(); h[~np.isfinite(h)] = np.nan
    if np.isfinite(h).any():
        vmin = np.nanpercentile(h,5); vmax=np.nanpercentile(h,95); vspan=max(1e-6,vmax-vmin)
        gray = np.clip(((h-vmin)/vspan)*255,0,255).astype(np.uint8)
        vis = cv2.applyColorMap(gray, CONFIG['colormap'])
    vis = cv2.addWeighted(vis, 0.9, np.dstack([mask]*3), 0.1, 0)
    def xy_to_px(xy):
        x0,y0 = origin_xy
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((xy[:,1]-y0)/pix_mm).astype(int), 0, H-1)
        return np.stack([xs,ys], axis=1)
    if gcode_xy is not None and gcode_xy.size>0:
        pts = xy_to_px(gcode_xy)
        for i in range(len(pts)-1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), (255,255,255), 1, cv2.LINE_AA)
    if centerline_xy is not None and centerline_xy.size>0:
        pts = xy_to_px(centerline_xy)
        for p in pts:
            cv2.circle(vis, tuple(p), 1, (0,255,0), -1)
    return vis


def draw_deviation_overlay(vis_top: np.ndarray, ref_xy: np.ndarray, actual_xy: np.ndarray,
                           idx: np.ndarray, e_n: np.ndarray,
                           origin_xy: Tuple[float,float], pix_mm: float,
                           stride: int=10) -> np.ndarray:
    H, W = vis_top.shape[:2]
    out = vis_top.copy()
    def xy_to_px(xy):
        x0, y0 = origin_xy
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((xy[:,1]-y0)/pix_mm).astype(int), 0, H-1)
        return np.stack([xs, ys], axis=1)
    if ref_xy is not None and ref_xy.size>0:
        p = xy_to_px(ref_xy)
        for i in range(len(p)-1):
            cv2.line(out, tuple(p[i]), tuple(p[i+1]), (220,220,220), 1, cv2.LINE_AA)
    if actual_xy is not None and actual_xy.size>0:
        q = xy_to_px(actual_xy)
        for i in range(len(q)-1):
            cv2.line(out, tuple(q[i]), tuple(q[i+1]), (200,255,255), 1, cv2.LINE_AA)
    if e_n.size>0 and ref_xy.size>1:
        ref_seg = np.diff(ref_xy, axis=0)
        ref_T = ref_seg / (np.linalg.norm(ref_seg, axis=1, keepdims=True) + 1e-9)
        ref_N = np.stack([-ref_T[:,1], ref_T[:,0]], axis=1)
        for k in range(0, len(idx), max(1,stride)):
            i = int(np.clip(idx[k], 0, len(ref_N)-1))
            n = ref_N[i]
            base = actual_xy[k]
            tip  = base + n * e_n[k]
            b = xy_to_px(base[None,:])[0]
            t = xy_to_px(tip[None,:])[0]
            cv2.arrowedLine(out, tuple(b), tuple(t), (0,255,0) if abs(e_n[k])<0.5 else (0,165,255), 2, cv2.LINE_AA, tipLength=0.3)
        # 统计条
        h = 80
        bar = np.full((h, out.shape[1], 3), 30, np.uint8)
        mid = h//2
        xs = np.linspace(0, bar.shape[1]-1, len(e_n)).astype(int)
        scale_px = (h*0.4)/max(1e-6, np.percentile(np.abs(e_n),95))
        ys = (mid - e_n*scale_px).astype(int)
        for i in range(len(xs)-1):
            cv2.line(bar, (xs[i], ys[i]), (xs[i+1], ys[i+1]), (0,128,255), 2, cv2.LINE_AA)
        cv2.line(bar, (0, mid), (bar.shape[1]-1, mid), (120,120,120), 1, cv2.LINE_AA)
        out = np.vstack([out, bar])
    return out

# =============================
# 相机
# =============================
class PCamMLSStream:
    def __init__(self):
        if pcammls is None:
            raise SystemExit('未安装 pcammls，无法使用相机模式。')
        self.cl = pcammls.PercipioSDK()
        self.h = None
        self.depth_calib = None
        self.scale_unit = 1.0
        self.pointcloud = pcammls.pointcloud_data_list()
    def open(self):
        devs = self.cl.ListDevice()
        if len(devs)==0: raise SystemExit('未发现设备。')
        print('检测到设备:');
        for i,d in enumerate(devs): print(f'  {i}: {d.id}\t{d.iface.id}')
        idx = 0 if len(devs)==1 else int(input('选择设备索引: '))
        sn = devs[idx].id
        h = self.cl.Open(sn)
        if not self.cl.isValidHandle(h):
            raise SystemExit(f'打开设备失败: {self.cl.TYGetLastErrorCodedescription()}')
        self.h = h
        depth_fmts = self.cl.DeviceStreamFormatDump(h, pcammls.PERCIPIO_STREAM_DEPTH)
        if not depth_fmts: raise SystemExit('无深度流。')
        self.cl.DeviceStreamFormatConfig(h, pcammls.PERCIPIO_STREAM_DEPTH, depth_fmts[0])
        self.cl.DeviceLoadDefaultParameters(h)
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(h)
        self.depth_calib = self.cl.DeviceReadCalibData(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamEnable(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamOn(h)
    def read_pointcloud(self, timeout_ms=2000):
        imgs = self.cl.DeviceStreamRead(self.h, timeout_ms)
        depth_img = None
        for fr in imgs:
            if fr.streamID == pcammls.PERCIPIO_STREAM_DEPTH:
                depth_img = fr; break
        if depth_img is None: return None
        self.cl.DeviceStreamMapDepthImageToPoint3D(depth_img, self.depth_calib, self.scale_unit, self.pointcloud)
        return self.pointcloud.as_nparray()  # (H, W, 3) in mm
    def close(self):
        if self.h is not None:
            try: self.cl.DeviceStreamOff(self.h)
            except Exception: pass
            try: self.cl.Close(self.h)
            except Exception: pass
            self.h = None

# =============================
# 主程序
# =============================

def _auto_center_from_valid(P_mach_flat: np.ndarray) -> Optional[np.ndarray]:
    X,Y,Z = P_mach_flat[:,0], P_mach_flat[:,1], P_mach_flat[:,2]
    valid = np.isfinite(X)&np.isfinite(Y)&np.isfinite(Z)
    if CONFIG['require_positive_z']:
        valid &= (Z>0)
    if CONFIG['reject_zero_xyz']:
        valid &= (np.abs(X)+np.abs(Y)+np.abs(Z) > 1e-6)
    if not np.any(valid):
        return None
    med = np.median(np.stack([X[valid], Y[valid]], axis=1), axis=0)
    return med


def _auto_size_from_valid(P_mach_flat: np.ndarray, qlo=5.0, qhi=95.0) -> tuple[np.ndarray, float]:
    X,Y,Z = P_mach_flat[:,0], P_mach_flat[:,1], P_mach_flat[:,2]
    valid = np.isfinite(X)&np.isfinite(Y)&np.isfinite(Z)
    if CONFIG['require_positive_z']: valid &= (Z>0)
    if CONFIG['reject_zero_xyz']: valid &= (np.abs(X)+np.abs(Y)+np.abs(Z) > 1e-6)
    if not np.any(valid):
        return np.array(CONFIG['roi_center_xy'], float), CONFIG['roi_size_mm']
    xs, ys = X[valid], Y[valid]
    x0, x1 = np.percentile(xs, qlo), np.percentile(xs, qhi)
    y0, y1 = np.percentile(ys, qlo), np.percentile(ys, qhi)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    size = float(max(x1-x0, y1-y0)) * 1.1
    size = np.clip(size, 20.0, 1200.0)
    return np.array([cx, cy], float), size


def main():
    # 外参
    T_path = input(f"外参路径（默认 {CONFIG['T_path']}）：").strip() or CONFIG['T_path']
    if not Path(T_path).exists():
        raise SystemExit(f'外参不存在：{T_path}')
    R,t,T = load_extrinsic(T_path)

    # G 代码
    gcode_path = input(f"G 代码路径（默认 {CONFIG['gcode_path']}）：").strip() or CONFIG['gcode_path']
    g_raw = parse_gcode_xy(gcode_path) if gcode_path and Path(gcode_path).exists() else np.empty((0,2))
    g_xy = resample_polyline(g_raw, CONFIG['resample_step_mm']) if g_raw.size>0 else g_raw
    ref_tree = build_kdtree(g_xy) if g_xy.size>0 else None
    T_ref, N_ref = tangent_normal(g_xy) if g_xy.size>0 else (np.zeros((0,2)), np.zeros((0,2)))

    # ROI & 栅格
    pix_mm = CONFIG['pixel_size_mm']
    roi_size = CONFIG['roi_size_mm']
    roi_center = np.array(CONFIG['roi_center_xy'], float)

    # 相机
    stream = PCamMLSStream(); stream.open()

    ctrl = DeviationController()
    dev_hist = collections.deque(maxlen=7)
    out_dir = Path(CONFIG['out_dir']); ensure_dir(out_dir)
    frame_id = 0

    try:
        while True:
            P_cam = stream.read_pointcloud()
            if P_cam is None: continue
            P_mach = transform_cam_to_machine(P_cam.reshape(-1,3).astype(np.float32), R, t)

            # 首帧/按键自动居中
            if frame_id == 0 and (roi_center == 0).all():
                med = _auto_center_from_valid(P_mach)
                if med is not None:
                    roi_center = med

            # 投影 & 掩码（返回 ROI 内有效点数量）
            height, mask, origin_xy, n_roi_valid = orthographic_project_top(
                P_mach, roi_center, roi_size, pix_mm,
                z_select=CONFIG['z_select'], min_points_per_cell=CONFIG['min_points_per_cell']
            )
            mask = morph_cleanup(mask, CONFIG['morph_open'], CONFIG['morph_close'])

            # 顶视图
            vis_top = render_topdown(height, mask, origin_xy, pix_mm, g_xy, np.empty((0,2)))

            # 如果 ROI 内 0 点，给出明显提示
            if n_roi_valid == 0:
                warn = np.full((40, vis_top.shape[1], 3), (30,30,30), np.uint8)
                cv2.putText(warn, 'ROI 命中 0 個有效點：請調整 ROI (i/j/k/l,[,],c,a)', (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                vis_top = np.vstack([warn, vis_top])

            # 中轴线拟合
            centerline_xy, dbg_centerline, diag = fit_centerline(
                mask, origin_xy, pix_mm, background=vis_top,
                cfg=dict(resample_step_mm=CONFIG['resample_step_mm'])
            )

            # 偏差与控制
            e_n = np.array([]); idx = np.array([], int); dxdy = np.zeros(2,float); stats=dict(mean=0.0, median=0.0, p95=0.0, n=0)
            if g_xy.size>0 and centerline_xy.size>0 and ref_tree is not None and N_ref.size>0:
                idx, nearest, e_n = project_points_to_path(centerline_xy, g_xy, ref_tree, N_ref)
                dxdy, stats = ctrl.update(e_n, N_ref[np.clip(idx,0,len(N_ref)-1)])
                dev_hist.append(stats['mean'])

            # 偏差叠加 + 调试图拼接
            vis_dev = draw_deviation_overlay(vis_top, g_xy, centerline_xy, idx, e_n, origin_xy, pix_mm, stride=10)
            dbg_scaled = cv2.resize(dbg_centerline, (vis_dev.shape[1], vis_dev.shape[0]), interpolation=cv2.INTER_AREA)
            vis = np.vstack([vis_dev, dbg_scaled])

            # HUD：增加有效点统计
            avg = np.mean(dev_hist) if len(dev_hist)>0 else 0.0
            text1 = f'pixel={pix_mm:.2f}mm  roi={roi_size:.0f}mm  dev_avg={avg: .3f}mm'
            text2 = f'dx={dxdy[0]: .3f} mm  dy={dxdy[1]: .3f} mm  valid(ROI/ALL)={n_roi_valid}/'+str(P_mach.shape[0])
            for (y,txt) in [(24,text1),(48,text2)]:
                cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow('FSW All-in-One (Top-Down + Centerline + Deviations)', vis)

            # 交互
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                outp = out_dir/f'vis_{frame_id:06d}.png'; cv2.imwrite(str(outp), vis); print(f'[SAVE] {outp}')
            elif key in (ord('='), ord('+')): pix_mm = max(0.1, pix_mm*0.8)
            elif key == ord('-'): pix_mm = min(5.0, pix_mm/0.8)
            elif key == ord('i'): roi_center[1] += roi_size*0.1
            elif key == ord('k'): roi_center[1] -= roi_size*0.1
            elif key == ord('j'): roi_center[0] -= roi_size*0.1
            elif key == ord('l'): roi_center[0] += roi_size*0.1
            elif key == ord('['): roi_size = max(20.0, roi_size*0.8)
            elif key == ord(']'): roi_size = min(1200.0, roi_size/0.8)
            elif key == ord('r'):
                roi_center = np.array(CONFIG['roi_center_xy'], float)
                pix_mm = CONFIG['pixel_size_mm']
                roi_size = CONFIG['roi_size_mm']
            elif key == ord('c'):
                med = _auto_center_from_valid(P_mach)
                if med is not None: roi_center = med
            elif key == ord('C'):
                # 仅基于当前 ROI 内点
                half = roi_size*0.5
                x0,x1 = roi_center[0]-half, roi_center[0]+half
                y0,y1 = roi_center[1]-half, roi_center[1]+half
                X,Y,Z = P_mach[:,0], P_mach[:,1], P_mach[:,2]
                valid = np.isfinite(X)&np.isfinite(Y)&np.isfinite(Z)
                if CONFIG['require_positive_z']: valid &= (Z>0)
                if CONFIG['reject_zero_xyz']: valid &= (np.abs(X)+np.abs(Y)+np.abs(Z) > 1e-6)
                inroi = (X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)
                sel = valid & inroi
                if np.any(sel):
                    roi_center = np.median(np.stack([X[sel],Y[sel]], axis=1), axis=0)
            elif key == ord('a'):
                roi_center, roi_size = _auto_size_from_valid(P_mach, 5, 95)
            elif key == ord('A'):
                roi_center, roi_size = _auto_size_from_valid(P_mach, 1, 99)

            frame_id += 1
    finally:
        try: stream.close()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
