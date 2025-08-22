#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSW 在线视觉管线（All-in-One 单文件，v3：先全场→再按 ROI 裁剪）
=========================================================

本版变更（保留原有注释，新增不删）
--------------------------------
- **不在投影阶段先框选 ROI**：先把整幅相机点云转换到机床坐标系后，
  根据**全场**的稳健边界（分位数）进行俯视正交投影，得到**全场高度图/掩码**。
- **两种 ROI 使用方式（均为“后截取”）**：
  1) `roi_mode='machine'`：在**机床系**下用参数 `roi_center_xy + roi_size_mm` 对投影结果做矩形裁剪，
     用于中轴线拟合与偏差计算（推荐，逻辑清晰，单位统一）。
  2) `roi_mode='camera_rect'`：在**相机原始深度图像**上给定像素矩形 `[x, y, w, h]`，
     仅对该像素区域映射 3D→机床系后再投影/拟合。适合硬件侧已有像素 ROI 的场景。
- **全场自适应边界**：自动用 1–99% 分位（可调）估计 XY 范围，并自动放大像素尺寸，避免生成过大的栅格。
- **诊断可视化**：叠加当前 ROI（机床/相机两种）轮廓与 HUD 统计，帮助快速定位“ROI 命中 0 点”的情况。

两种 ROI 方案的取舍（简评）
----------------------------
- 机床系 ROI：
  - ✅ 优点：矩形定义就是**真实物理尺寸**；与 G 代码/机床控制坐标天然对齐；
           外参只影响一次坐标变换，投影/裁剪逻辑简洁；更适合在线纠偏闭环。
  - ⚠️ 注意：需要先完成全场投影，图像分辨率过细时会稍耗时（v3 已加入自适应像素尺寸上限）。
- 相机像素 ROI：
  - ✅ 优点：可直接复用相机侧已有像素矩形；数据量更小，前端过滤更早。
  - ⚠️ 注意：像素矩形经 3D→机床系变换后**不再是矩形**（透视/姿态影响），
           逻辑上仍以“像素集合→三维点集合”的方式处理即可；当镜头或外参变化时需重新评估像素→物理覆盖。

使用提示
--------
1) 先生成 `T_cam2machine.npy`；
2) 运行脚本，输入外参与（可选）G 代码路径；
3) 缺省 `roi_mode='none'`：显示全场投影；
   - 若要只在 ROI 内拟合中轴线，将 `roi_mode` 设为 `'machine'` 或 `'camera_rect'` 并配置对应参数；
4) 热键（保留原有）：`q` 退出；`s` 截图；`-`/`=` 改分辨率；`r` 重置；
   `i/j/k/l` 平移**机床 ROI**；`[`/`]` 缩放**机床 ROI**。

依赖
----
必需：`numpy`, `opencv-python`, `pcammls`
可选：`scikit-image`（骨架）、`scipy`（KDTree/Savitzky–Golay）；缺省均有回退实现。
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import collections
import math
import numpy as np
import cv2

# --- 可选依赖（保留原有写法） ---
try:
    import pcammls  # PercipioSDK
except Exception:
    pcammls = None
try:
    import cv2.ximgproc as xip  # 骨架 thinning
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
# 全局配置（新增项不删旧注释）
# =============================
CONFIG: Dict = dict(
    # 文件
    T_path='T_cam2machine.npy',
    gcode_path='path/example.gcode',

    # 栅格分辨率与全场边界
    pixel_size_mm=0.6,                 # 初始像素尺寸（投影到全场时使用）
    bounds_qlo=1.0, bounds_qhi=99.0,  # 全场边界分位（更稳健）
    bounds_margin_mm=20.0,             # 全场边界外扩（mm）
    max_grid_pixels=1_200_000,         # 全场栅格像素上限（自动增大 pixel_size_mm）

    # ROI 模式：'none' / 'machine' / 'camera_rect'
    roi_mode='none',

    # 机床系 ROI（后裁剪用）
    roi_center_xy=[0.0, 0.0],          # 机床 ROI 中心（mm）
    roi_size_mm=120.0,                 # 机床 ROI 边长（mm）

    # 相机像素 ROI（后裁剪用）
    cam_roi_xywh=[100, 100, 300, 200], # (x,y,w,h)，单位：像素

    # 最近表面策略/形态学
    z_select='max',
    min_points_per_cell=1,
    morph_open=3,
    morph_close=5,

    # 有效点判定
    require_positive_z=False,          # 关闭/或设为 'auto'
    reject_zero_xyz=True,

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
# 通用工具（保留原有函数，必要处增强）
# =============================

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_extrinsic(T_path: str | Path):
    data = np.load(T_path, allow_pickle=True).item()
    return np.asarray(data['R'], float), np.asarray(data['t'], float), np.asarray(data['T'], float)


def transform_cam_to_machine(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P_cam.T).T + t

# =============================
# G 代码解析与几何场（保留）
# =============================

def parse_gcode_xy(path: str | Path) -> np.ndarray:
    pts: List[List[float]] = []
    if not path or not Path(path).exists():
        return np.empty((0, 2), float)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        x = y = None
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(';') or line.startswith('('):
                continue
            if ';' in line:
                line = line.split(';', 1)[0]
            while '(' in line and ')' in line:
                a, b = line.find('('), line.find(')')
                if a < 0 or b < 0 or b <= a: break
                line = (line[:a] + ' ' + line[b+1:]).strip()
            toks = line.split()
            if not toks: continue
            cmd = toks[0].upper()
            if cmd in ('G0','G00','G1','G01'):
                for t in toks[1:]:
                    u = t.upper()
                    if u.startswith('X'):
                        try: x = float(u[1:])
                        except: pass
                    elif u.startswith('Y'):
                        try: y = float(u[1:])
                        except: pass
                if x is not None and y is not None:
                    pts.append([x, y])
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
# 形态学清理（补齐）
# =============================

def morph_cleanup(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """对二值掩码做开/闭运算清理。"""
    m = (mask > 0).astype(np.uint8) * 255
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

# =============================
# 全场边界估计 & 自适应像素尺寸
# =============================

def _valid_mask_xyz(P: np.ndarray) -> np.ndarray:
    X, Y, Z = P[:,0], P[:,1], P[:,2]
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    if CONFIG.get('reject_zero_xyz', True):
        valid &= (np.abs(X) + np.abs(Y) + np.abs(Z) > 1e-6)
    rpz = CONFIG.get('require_positive_z', False)
    if rpz is True:
        valid &= (Z > 0)
    elif rpz == 'auto':
        pos = np.count_nonzero(Z > 0); neg = np.count_nonzero(Z < 0)
        valid &= (Z > 0) if pos >= neg else (Z < 0)
    return valid


def compute_global_bounds(P_mach: np.ndarray, qlo: float, qhi: float, margin_mm: float) -> Tuple[float,float,float,float]:
    """基于全场有效点的分位数估计 XY 边界，并外扩 margin。返回 (x0,x1,y0,y1)。"""
    valid = _valid_mask_xyz(P_mach)
    if not np.any(valid):
        return -100, 100, -100, 100  # 回退边界
    X, Y = P_mach[valid,0], P_mach[valid,1]
    x0, x1 = np.percentile(X, qlo), np.percentile(X, qhi)
    y0, y1 = np.percentile(Y, qlo), np.percentile(Y, qhi)
    return float(x0 - margin_mm), float(x1 + margin_mm), float(y0 - margin_mm), float(y1 + margin_mm)


def adjust_pixel_size_to_limit(x0,x1,y0,y1,pix_mm,max_pixels) -> float:
    """若全场像素数超过上限，则增大像素尺寸。"""
    W = max(2, int(round((x1-x0)/max(1e-6,pix_mm))))
    H = max(2, int(round((y1-y0)/max(1e-6,pix_mm))))
    while W*H > max_pixels:
        pix_mm *= 1.25
        W = max(2, int(round((x1-x0)/pix_mm)))
        H = max(2, int(round((y1-y0)/pix_mm)))
    return pix_mm

# =============================
# 全场俯视正交投影（无 ROI）
# =============================

def project_global_topdown(P_mach: np.ndarray, pix_mm: float,
                           bounds: Tuple[float,float,float,float],
                           z_select: str='max', min_points_per_cell: int=1):
    x0,x1,y0,y1 = bounds
    X,Y,Z = P_mach[:,0], P_mach[:,1], P_mach[:,2]
    valid = _valid_mask_xyz(P_mach)
    inb = (X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)
    sel = valid & inb
    W = H = int(max(2, round(max(x1-x0, y1-y0)/pix_mm)))
    if not np.any(sel):
        return np.full((H,H), np.nan, np.float32), np.zeros((H,H), np.uint8), (x0,y0), 0
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
# 中轴线拟合（骨架→路径→平滑→采样）
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
    """从掩码拟合机床系 XY 中轴线，并返回调试可视化。"""
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

    # 1) 骨架
    skel = _skeletonize(mask, C['use_ximgproc_first'])
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        dbg = background if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.empty((0,2), float), dbg, dict(reason='empty-mask')

    # 2) 建图 → 最长路 → 像素路径
    nodes, adj, endpoints, junctions = _skeleton_graph(skel)
    path_nodes = _longest_path_from_graph(adj, endpoints)  # [(y,x)]
    path_px = np.array([[x, y] for (y, x) in path_nodes], dtype=np.float32)

    # 去除非有限值
    if path_px.size == 0:
        dbg = background if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.empty((0,2), float), dbg, dict(reason='empty-path')
    path_px = path_px[np.all(np.isfinite(path_px), axis=1)]
    if len(path_px) < 2:
        dbg = background if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.empty((0,2), float), dbg, dict(reason='short-path')

    # 3) 简化 + 平滑 + 等弧长重采样（像素域）
    if len(path_px) >= 3:
        path_px = _rdp(path_px, float(C['rdp_epsilon_px']))
    path_px = _smooth_polyline(path_px, int(C['sg_window']), int(C['sg_polyorder']))
    step_px = max(1.0, float(C['resample_step_mm'])/max(1e-6, pix_mm))
    path_px = _resample_polyline(path_px, step_px)

    # 再次去除非有限值并裁剪到图像范围，转 int 用于绘制
    H, W = mask.shape
    path_px = path_px[np.all(np.isfinite(path_px), axis=1)]
    if len(path_px) < 2:
        dbg = background if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.empty((0,2), float), dbg, dict(reason='short-after-clean')
    path_px[:, 0] = np.clip(path_px[:, 0], 0, W - 1)
    path_px[:, 1] = np.clip(path_px[:, 1], 0, H - 1)
    path_int = np.round(path_px).astype(np.int32)

    # 4) 曲率/法线（用于调试绘制）
    def _curv(px):
        if len(px) < 3: return np.zeros((len(px),), float)
        d1 = np.gradient(px, axis=0); d2 = np.gradient(d1, axis=0)
        num = np.abs(d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0])
        den = (d1[:,0]**2 + d1[:,1]**2)**1.5 + 1e-12
        kappa = num/den; kappa[~np.isfinite(kappa)] = 0
        return kappa
    kappa = _curv(path_px)
    d = np.gradient(path_px, axis=0)
    T = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    N = np.stack([-T[:,1], T[:,0]], axis=1)

    # 5) 调试可视化合成
    if background is None:
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        vis = background.copy()
        if vis.shape[:2] != mask.shape:
            vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_NEAREST)

    # 骨架淡显
    sk3 = cv2.cvtColor(((skel>0)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(vis, 0.9, sk3, 0.3, 0)

    # 端点/分叉
    for y,x in endpoints: cv2.circle(vis, (int(x),int(y)), 3, (0,255,255), -1)
    for y,x in junctions: cv2.circle(vis, (int(x),int(y)), 3, (0,165,255), -1)

    # 主路径曲率着色（安全处理 NaN）
    if len(path_int) >= 2:
        k = np.clip(np.nan_to_num(kappa * float(C['curvature_amp']), nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        # 做成 Nx1 的单通道“图像”喂给 applyColorMap
        lut_in = (k * 255).astype(np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(lut_in, int(C['colormap'])).reshape(-1, 3)
        for i in range(len(path_int) - 1):
            c = tuple(int(v) for v in colors[i])
            p1 = (int(path_int[i,0]),   int(path_int[i,1]))
            p2 = (int(path_int[i+1,0]), int(path_int[i+1,1]))
            cv2.line(vis, p1, p2, c, 2, cv2.LINE_AA)

    # 法线箭头 + 索引
    for i in range(0, len(path_int), 20):
        x, y = int(path_int[i,0]), int(path_int[i,1])
        nx, ny = N[i] * 12.0
        tx, ty = int(np.clip(x + nx, 0, W-1)), int(np.clip(y + ny, 0, H-1))
        cv2.arrowedLine(vis, (x,y), (tx,ty), (0,255,0), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(vis, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    legend = np.zeros((60, vis.shape[1], 3), np.uint8)
    cv2.putText(legend, 'yellow=endpoints  orange=junctions  line=curvature  green=normals',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    dbg = np.vstack([vis, legend])

    # 6) 像素路径 -> 机床系 XY
    x0, y0 = origin_xy
    X = x0 + (path_px[:,0] + 0.5) * pix_mm
    Y = y0 + (path_px[:,1] + 0.5) * pix_mm
    centerline_xy = np.stack([X, Y], axis=1)

    diag = dict(path_px=path_px, curvature=kappa)
    return centerline_xy, dbg, diag

# =============================
# 偏差计算与控制（保留）
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
# 可视化（保留，新增 ROI 叠加）
# =============================

def render_topdown(height, mask, origin_xy, pix_mm, gcode_xy, centerline_xy,
                   roi_rect_px: Optional[Tuple[int,int,int,int]]=None,
                   roi_mode_text: str=''):
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
    if roi_rect_px is not None:
        x,y,w,hp = roi_rect_px
        cv2.rectangle(vis, (x,y), (x+w, y+hp), (0,0,255), 2, cv2.LINE_AA)
        if roi_mode_text:
            cv2.putText(vis, roi_mode_text, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中心线提取增强补丁（骨架补齐 + 去毛刺 + 分支选择 + 可视化）
====================================================

用途
----
将本文件中的函数粘贴到你的 `fsw_pipeline_all_in_one_v2.py` 中（**替换原有的中轴线相关实现**），
即可获得更稳健的“实际中轴线”提取：
- 原生骨架 → **端点桥接（补齐）** → **毛刺剪枝** → **按参考路径选择主分支** → RDP/平滑/等弧长采样。
- 提供丰富调试图：原掩码、原始骨架、桥接线、剪枝后骨架、最终主路径（曲率着色）、端点/分叉、法线箭头、指标条。

如何接入（两行改动）
--------------------
1) 在文件顶部保留/导入所需依赖（本补丁仅用到已有依赖：numpy、cv2、可选 ximgproc/skimage/scipy）。
2) 在主循环里**把原来的** `fit_centerline(...)` **调用替换为**：
   ```python
   centerline_xy, dbg_centerline, metrics = fit_centerline_plus(
       mask_used, origin_xy, pix_mm,
       background=vis_top,
       ref_xy=g_xy,            # 可为 None；若提供则按参考路径选择主分支
       ref_tree=ref_tree,      # 可为 None；若提供用于快速距离评估
       cfg=dict(               # 可不传，沿用 CONFIG 缺省
           spur_len_mm=5.0,
           bridge_max_gap_mm=8.0,
           min_component_len_mm=30.0,
       )
   )
   ```
   其它代码不需要改。

注意
----
- 本补丁**不删除**你原文件的注释与函数命名；你可以保留原 `fit_centerline`，另行调用 `fit_centerline_plus`；
  若希望完全替换，只需把旧调用点换成 `fit_centerline_plus` 即可。
- 需要 `KDTree` 时若未安装 `scipy`，会自动回退到线性搜索实现（已有）。
"""
from typing import Tuple, Optional, Dict
import numpy as np
import cv2

# 若你的文件里已有以下符号，请删除本段重复定义或保持一致：
try:
    import cv2.ximgproc as xip
except Exception:
    xip = None
try:
    from skimage.morphology import skeletonize as sk_skeletonize
except Exception:
    sk_skeletonize = None

# =============== 基础工具（与原文件一致/兼容） ===============

def _safe_uint8(bw: np.ndarray) -> np.ndarray:
    m = (bw > 0).astype(np.uint8) * 255
    return m


def _skeletonize_enhanced(mask: np.ndarray) -> np.ndarray:
    """优先用 ximgproc thinning；失败时 skimage；再失败走距离场脊线回退。"""
    m = (mask > 0).astype(np.uint8)
    if xip is not None:
        try:
            sk = xip.thinning(m, thinningType=xip.THINNING_ZHANGSUEN)
            return (sk > 0).astype(np.uint8) * 255
        except Exception:
            pass
    if sk_skeletonize is not None:
        try:
            sk = sk_skeletonize(m.astype(bool))
            return (sk.astype(np.uint8)) * 255
        except Exception:
            pass
    # 回退：距离变换脊线近似
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    sk = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            sk = np.maximum(sk, (dist < np.roll(np.roll(dist, dy, 0), dx, 1)).astype(np.uint8))
    ridge = (sk == 0).astype(np.uint8) & (m > 0)
    return ridge.astype(np.uint8) * 255


def _skeleton_graph(skel: np.ndarray):
    ys, xs = np.where(skel > 0)
    nodes = list(zip(ys.tolist(), xs.tolist()))
    S = set(nodes)
    adj = {n: set() for n in nodes}
    H, W = skel.shape
    for y, x in nodes:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and (yy, xx) in S:
                    adj[(y, x)].add((yy, xx))
    deg = {n: len(adj[n]) for n in nodes}
    endpoints = [n for n, d in deg.items() if d == 1]
    junctions = [n for n, d in deg.items() if d >= 3]
    return nodes, adj, endpoints, junctions


def _longest_path_from_graph(adj, endpoints):
    import collections
    def bfs(start):
        vis = {start: None}; q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in vis:
                    vis[v] = u; q.append(v)
        far = max(vis.keys(), key=lambda k: (abs(k[0] - start[0]) + abs(k[1] - start[1])))
        return far, vis
    if endpoints:
        a = endpoints[0]; b, pre = bfs(a); c, pre2 = bfs(b)
        path = [c]; p = pre2[c]
        while p is not None:
            path.append(p); p = pre2[p]
        return path[::-1]
    anyn = next(iter(adj.keys())); b, pre = bfs(anyn); c, pre2 = bfs(b)
    path = [c]; p = pre2[c]
    while p is not None:
        path.append(p); p = pre2[p]
    return path[::-1]


def _connected_components(skel: np.ndarray):
    nodes, adj, endpoints, junctions = _skeleton_graph(skel)
    seen = set(); comps = []
    for n in nodes:
        if n in seen: continue
        stack = [n]; comp = set([n]); seen.add(n)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); comp.add(v); stack.append(v)
        comps.append(comp)
    return comps, adj, endpoints, junctions


def _spur_prune(skel: np.ndarray, spur_len_px: int) -> np.ndarray:
    """多轮移除短毛刺：从端点向内累积像素距离，短于阈值则删除。"""
    sk = (skel > 0).astype(np.uint8)
    if spur_len_px <= 1: return sk * 255
    for _ in range(5):  # 最多 5 轮
        nodes, adj, endpoints, _ = _skeleton_graph(sk * 255)
        if not endpoints: break
        removed = 0
        for epi in endpoints:
            # 沿端点向内走 spur_len_px 步
            path = [epi]; cur = epi; prev = None
            for _step in range(spur_len_px):
                nxts = [v for v in adj.get(cur, []) if v != prev]
                if not nxts: break
                nxt = nxts[0] if len(nxts) == 1 else nxts[0]
                path.append(nxt); prev, cur = cur, nxt
                # 到达分叉即停止
                if len(adj[cur]) >= 3: break
            # 若没有遇到分叉，认为是毛刺，删除 path
            if len(adj.get(cur, [])) <= 2:  # 仍未进入主干
                for y, x in path:
                    sk[y, x] = 0; removed += 1
        if removed == 0: break
    return sk.astype(np.uint8) * 255


def _bridge_endpoints(mask: np.ndarray, skel: np.ndarray, max_gap_px: int, max_bridges: int = 20) -> Tuple[np.ndarray, int, np.ndarray]:
    """在小间隙内桥接端点对：返回新掩码、桥接次数、桥接可视化层。"""
    if max_gap_px <= 1: return mask.copy(), 0, np.zeros((*mask.shape, 3), np.uint8)
    nodes, adj, endpoints, _ = _skeleton_graph(skel)
    pts = np.array([[x, y] for (y, x) in endpoints], np.int32)
    if len(pts) < 2:
        return mask.copy(), 0, np.zeros((*mask.shape, 3), np.uint8)
    M = _safe_uint8(mask)
    vis = np.zeros((*mask.shape, 3), np.uint8)
    dil = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    used = np.zeros(len(pts), np.bool_)
    bridges = 0
    for i in range(len(pts)):
        if used[i]: continue
        pi = pts[i]
        # 选最近的未使用端点
        d2 = np.sum((pts - pi) ** 2, axis=1)
        order = np.argsort(d2)
        for j in order:
            if j == i or used[j]:
                continue
            pj = pts[j]
            dist = np.sqrt(float(d2[j]))
            if dist > max_gap_px: break
            # 直线是否“可行”：落在 dil 区域的比例 >= 0.6
            line_mask = np.zeros_like(M)
            cv2.line(line_mask, tuple(pi), tuple(pj), 255, 1, cv2.LINE_8)
            inter = cv2.bitwise_and(line_mask, dil)
            ratio = float(np.count_nonzero(inter)) / max(1, int(dist))
            if ratio >= 0.6:
                # 桥接：在原掩码上画线
                cv2.line(M, tuple(pi), tuple(pj), 255, 1, cv2.LINE_8)
                cv2.line(vis, tuple(pi), tuple(pj), (255, 0, 255), 1, cv2.LINE_AA)
                used[i] = used[j] = True
                bridges += 1
                break
        if bridges >= max_bridges:
            break
    return M, bridges, vis


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    if len(points) < 3: return points
    a, b = points[0], points[-1]
    ab = b - a; lab2 = (ab * ab).sum() + 1e-12
    dmax = -1; idx = -1
    for i in range(1, len(points) - 1):
        ap = points[i] - a
        t = np.clip(np.dot(ap, ab) / lab2, 0, 1)
        proj = a + t * ab
        d = np.linalg.norm(points[i] - proj)
        if d > dmax: dmax = d; idx = i
    if dmax > eps:
        left = _rdp(points[:idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


def _smooth_polyline(px: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
    except Exception:
        savgol_filter = None
    if len(px) < max(5, window): return px
    if savgol_filter is not None and window % 2 == 1:
        xs = savgol_filter(px[:, 0], window, polyorder)
        ys = savgol_filter(px[:, 1], window, polyorder)
        return np.stack([xs, ys], axis=1)
    # 回退：移动平均
    k = max(3, window | 1); pad = k // 2
    ext = np.pad(px, ((pad, pad), (0, 0)), mode='edge')
    ker = np.ones((k, 1)) / k
    xs = np.convolve(ext[:, 0], ker[:, 0], mode='valid')
    ys = np.convolve(ext[:, 1], ker[:, 0], mode='valid')
    return np.stack([xs, ys], axis=1)


def _resample_polyline(px: np.ndarray, step: float) -> np.ndarray:
    if len(px) < 2: return px
    seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
    L = float(seg.sum())
    if L < 1e-9: return px[[0]].copy()
    n = max(2, int(np.ceil(L / step)))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = []; j = 0
    for si in s:
        while j < len(seg) and si > cs[j + 1]: j += 1
        if j >= len(seg): out.append(px[-1]); continue
        t = (si - cs[j]) / max(seg[j], 1e-9)
        out.append(px[j] * (1 - t) + px[j + 1] * t)
    return np.asarray(out, float)


def _curvature(px: np.ndarray) -> np.ndarray:
    if len(px) < 3: return np.zeros((len(px),), float)
    d1 = np.gradient(px, axis=0); d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
    kappa = num / den; kappa[~np.isfinite(kappa)] = 0
    return kappa


# =============== 入口函数：fit_centerline_plus ===============

def fit_centerline_plus(mask: np.ndarray,
                        origin_xy: Tuple[float, float],
                        pix_mm: float,
                        background: Optional[np.ndarray] = None,
                        ref_xy: Optional[np.ndarray] = None,
                        ref_tree: Optional[object] = None,
                        cfg: Optional[Dict] = None):
    """从掩码拟合机床系 XY 中轴线（增强版），并返回调试可视化与指标。"""
    # 默认参数（可从外部 cfg 覆盖）
    C = dict(
        spur_len_mm=5.0,
        bridge_max_gap_mm=8.0,
        min_component_len_mm=30.0,
        rdp_epsilon_px=2.0,
        sg_window=11,
        sg_polyorder=2,
        resample_step_mm=1.0,
        curvature_amp=200.0,
        colormap=getattr(cv2, 'COLORMAP_TURBO', getattr(cv2, 'COLORMAP_JET', 2)),
        normal_stride_px=20,
    )
    if cfg: C.update(cfg)

    H, W = mask.shape
    spur_len_px = max(1, int(round(C['spur_len_mm'] / max(1e-6, pix_mm))))
    bridge_gap_px = max(1, int(round(C['bridge_max_gap_mm'] / max(1e-6, pix_mm))))
    min_comp_len_px = max(2, int(round(C['min_component_len_mm'] / max(1e-6, pix_mm))))

    # ---------- (1) 基线：原始骨架 ----------
    M0 = _safe_uint8(mask)
    sk_raw = _skeletonize_enhanced(M0)

    # ---------- (2) 端点桥接（补齐） ----------
    M_bridge, n_bridges, vis_bridge = _bridge_endpoints(M0, sk_raw, bridge_gap_px)

    # ---------- (3) 再骨架 + 毛刺剪枝 ----------
    sk_b = _skeletonize_enhanced(M_bridge)
    sk_pruned = _spur_prune(sk_b, spur_len_px)

    # ---------- (4) 组件与主路径选择 ----------
    comps, adj, endpoints, junctions = _connected_components(sk_pruned)
    chosen_path_px = None
    best_score = 1e18
    for comp in comps:
        # 组件像素集合 → 子图 → 最长路
        sub_nodes = list(comp)
        sub_adj = {n: (adj[n] & comp) for n in sub_nodes}
        sub_endpoints = [n for n in sub_nodes if len(sub_adj[n]) == 1]
        if len(sub_nodes) < min_comp_len_px:
            continue
        path_nodes = _longest_path_from_graph(sub_adj, sub_endpoints)
        path_px = np.array([[x, y] for (y, x) in path_nodes], dtype=np.float32)
        if path_px.shape[0] < 2:
            continue
        # 距参考路径的得分（若无参考，则用负长度当分数）
        if ref_xy is not None and ref_xy.size > 0:
            # 用简化后的少量采样做 KD 查询
            sample = path_px[::max(1, path_px.shape[0] // 100)]  # 最多 100 个点
            # 像素→机床系
            x0, y0 = origin_xy
            SX = x0 + (sample[:, 0] + 0.5) * pix_mm
            SY = y0 + (sample[:, 1] + 0.5) * pix_mm
            S = np.stack([SX, SY], axis=1)
            if ref_tree is None:
                # 线性回退
                d2 = np.min(np.sum((S[:, None, :] - ref_xy[None, :, :]) ** 2, axis=2), axis=1)
            else:
                _d, _i = ref_tree.query(S)
                d2 = _d ** 2
            score = float(np.mean(d2))
        else:
            # 没有参考路径时，优先最长
            score = -float(path_px.shape[0])
        if score < best_score:
            best_score = score
            chosen_path_px = path_px

    if chosen_path_px is None:
        # 回退：对全图取最长路
        nodes, adj_all, endpoints_all, _ = _skeleton_graph(sk_pruned)
        if len(nodes) >= 2:
            chosen = _longest_path_from_graph(adj_all, endpoints_all)
            chosen_path_px = np.array([[x, y] for (y, x) in chosen], dtype=np.float32)
        else:
            dbg = cv2.cvtColor(M0, cv2.COLOR_GRAY2BGR)
            cv2.putText(dbg, 'no-skeleton', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return np.empty((0, 2), float), dbg, dict(reason='no-skeleton')

    # ---------- (5) RDP/平滑/等弧长采样 ----------
    path_px = chosen_path_px
    if path_px.shape[0] >= 3:
        path_px = _rdp(path_px, float(C['rdp_epsilon_px']))
    path_px = _smooth_polyline(path_px, int(C['sg_window']), int(C['sg_polyorder']))
    step_px = max(1.0, float(C['resample_step_mm']) / max(1e-6, pix_mm))
    path_px = _resample_polyline(path_px, step_px)
    # 清理/裁剪/取整
    path_px = path_px[np.all(np.isfinite(path_px), axis=1)]
    if len(path_px) < 2:
        dbg = cv2.cvtColor(M0, cv2.COLOR_GRAY2BGR)
        cv2.putText(dbg, 'short-path', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return np.empty((0, 2), float), dbg, dict(reason='short-path')
    path_px[:, 0] = np.clip(path_px[:, 0], 0, W - 1)
    path_px[:, 1] = np.clip(path_px[:, 1], 0, H - 1)
    path_int = np.round(path_px).astype(np.int32)

    # ---------- (6) 曲率/法线 ----------
    kappa = _curvature(path_px)
    d = np.gradient(path_px, axis=0)
    T = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)

    # ---------- (7) 可视化合成 ----------
    if background is None:
        vis = cv2.cvtColor(M0, cv2.COLOR_GRAY2BGR)
    else:
        vis = background.copy()
        if vis.shape[:2] != mask.shape:
            vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_NEAREST)

    # 原始骨架（青）
    sk3 = cv2.cvtColor(sk_raw, cv2.COLOR_GRAY2BGR)
    sk3[:, :, 1] = np.maximum(sk3[:, :, 1], sk3[:, :, 0])  # 偏青
    vis = cv2.addWeighted(vis, 0.85, sk3, 0.35, 0)

    # 桥接线（品红）
    vis = cv2.addWeighted(vis, 1.0, vis_bridge, 0.9, 0)

    # 剪枝后骨架（黄）
    skp = cv2.cvtColor(sk_pruned, cv2.COLOR_GRAY2BGR)
    skp[:, :, 2] = 0; skp[:, :, 1] = np.maximum(skp[:, :, 1], skp[:, :, 0])  # 偏黄
    vis = cv2.addWeighted(vis, 0.9, skp, 0.4, 0)

    # 端点/分叉
    nodes2, adj2, endpoints2, junctions2 = _skeleton_graph(sk_pruned)
    for y, x in endpoints2: cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)
    for y, x in junctions2: cv2.circle(vis, (int(x), int(y)), 3, (0, 165, 255), -1)

    # 主路径曲率着色
    if len(path_int) >= 2:
        k = np.clip(np.nan_to_num(kappa * float(C['curvature_amp']), nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        lut_in = (k * 255).astype(np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(lut_in, int(C['colormap'])).reshape(-1, 3)
        for i in range(len(path_int) - 1):
            c = tuple(int(v) for v in colors[i])
            p1 = (int(path_int[i, 0]), int(path_int[i, 1]))
            p2 = (int(path_int[i + 1, 0]), int(path_int[i + 1, 1]))
            cv2.line(vis, p1, p2, c, 2, cv2.LINE_AA)
        # 法线箭头
        stride = int(C['normal_stride_px'])
        for i in range(0, len(path_int), max(1, stride)):
            x, y = int(path_int[i, 0]), int(path_int[i, 1])
            nx, ny = N[i] * 12.0
            tx, ty = int(np.clip(x + nx, 0, W - 1)), int(np.clip(y + ny, 0, H - 1))
            cv2.arrowedLine(vis, (x, y), (tx, ty), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(vis, str(i), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 指标条
    legend = np.zeros((64, vis.shape[1], 3), np.uint8)
    comps_count = len(comps)
    spur_px_est = spur_len_px
    Lpx = float(np.linalg.norm(np.diff(path_px, axis=0), axis=1).sum()) if len(path_px) > 1 else 0.0
    Lmm = Lpx * pix_mm
    info = (f"components={comps_count}  bridges={n_bridges}  spur_prune~{spur_px_est}px  "
            f"len={Lmm:.1f}mm  step={C['resample_step_mm']:.1f}mm")
    cv2.putText(legend, info, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    dbg = np.vstack([vis, legend])

    # ---------- (8) 像素路径 -> 机床系 XY ----------
    x0, y0 = origin_xy
    X = x0 + (path_px[:, 0] + 0.5) * pix_mm
    Y = y0 + (path_px[:, 1] + 0.5) * pix_mm
    centerline_xy = np.stack([X, Y], axis=1)

    metrics = dict(
        components=int(comps_count),
        bridges=int(n_bridges),
        spur_len_px=int(spur_px_est),
        length_mm=float(Lmm),
        n_points=int(centerline_xy.shape[0])
    )
    return centerline_xy, dbg, metrics


# =============================
# 相机（保留）
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
        print('检测到设备:')
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
        if depth_img is None: return None, None
        self.cl.DeviceStreamMapDepthImageToPoint3D(depth_img, self.depth_calib, self.scale_unit, self.pointcloud)
        return self.pointcloud.as_nparray(), depth_img  # (H,W,3) in mm, 原始深度帧
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

    # 相机
    stream = PCamMLSStream(); stream.open()

    # 控制器与统计
    ctrl = DeviationController()
    dev_hist = collections.deque(maxlen=7)
    out_dir = Path(CONFIG['out_dir']); ensure_dir(out_dir)
    frame_id = 0

    try:
        while True:
            P_cam, depth_fr = stream.read_pointcloud()
            if P_cam is None: continue
            H, W, _ = P_cam.shape

            # 选择点集：根据 ROI 模式决定是否先在相机像素域裁剪
            roi_mode = CONFIG.get('roi_mode','none')
            if roi_mode == 'camera_rect':
                x,y,w,h = CONFIG.get('cam_roi_xywh',[0,0,W,H])
                x = int(np.clip(x,0,W-1)); y=int(np.clip(y,0,H-1))
                w = int(np.clip(w,1,W-x)); h=int(np.clip(h,1,H-y))
                pixmask = np.zeros((H,W), np.bool_)
                pixmask[y:y+h, x:x+w] = True
                P_sel = P_cam[pixmask].reshape(-1,3)
            else:
                P_sel = P_cam.reshape(-1,3)

            # 相机→机床
            P_mach = transform_cam_to_machine(P_sel.astype(np.float32), R, t)

            # 全场边界 + 像素尺寸自适应
            x0,x1,y0,y1 = compute_global_bounds(P_mach, CONFIG['bounds_qlo'], CONFIG['bounds_qhi'], CONFIG['bounds_margin_mm'])
            pix_mm = float(CONFIG['pixel_size_mm'])
            pix_mm = adjust_pixel_size_to_limit(x0,x1,y0,y1,pix_mm, CONFIG['max_grid_pixels'])

            # 全场投影（无 ROI）
            height, mask, origin_xy, n_valid = project_global_topdown(
                P_mach, pix_mm, (x0,x1,y0,y1),
                z_select=CONFIG['z_select'], min_points_per_cell=CONFIG['min_points_per_cell']
            )
            mask_clean = morph_cleanup(mask, CONFIG['morph_open'], CONFIG['morph_close'])

            # 机床 ROI 后裁剪（仅用于中轴线与偏差；原图仍显示全场）
            roi_rect_px = None
            mask_for_fit = mask_clean.copy()
            if roi_mode == 'machine':
                cx, cy = CONFIG['roi_center_xy']; size = CONFIG['roi_size_mm']
                half = size*0.5
                x0r, x1r = cx-half, cx+half
                y0r, y1r = cy-half, cy+half
                # 转像素
                x0p = int(np.clip((x0r - origin_xy[0])/pix_mm, 0, mask_for_fit.shape[1]-1))
                x1p = int(np.clip((x1r - origin_xy[0])/pix_mm, 0, mask_for_fit.shape[1]-1))
                y0p = int(np.clip((y0r - origin_xy[1])/pix_mm, 0, mask_for_fit.shape[0]-1))
                y1p = int(np.clip((y1r - origin_xy[1])/pix_mm, 0, mask_for_fit.shape[0]-1))
                roi_rect_px = (min(x0p,x1p), min(y0p,y1p), abs(x1p-x0p), abs(y1p-y0p))
                # 生成 ROI mask
                M = np.zeros_like(mask_for_fit)
                x,y,w,hp = roi_rect_px
                M[y:y+hp, x:x+w] = 255
                mask_for_fit = cv2.bitwise_and(mask_for_fit, M)

            # 顶视图（全场）+ ROI 边框
            vis_top = render_topdown(height, mask_clean, origin_xy, pix_mm, g_xy, np.empty((0,2)),
                                     roi_rect_px=roi_rect_px,
                                     roi_mode_text=f'ROI({roi_mode})' if roi_mode!='none' else '')

            # 中轴线拟合（若开启 ROI 则用裁剪后掩码；否则直接用全场掩码，可能较慢/分叉多）
            mask_used = mask_for_fit if roi_mode!='camera_rect' else mask_clean
            centerline_xy, dbg_centerline, diag = fit_centerline_plus(
                mask_used, origin_xy, pix_mm, background=vis_top,
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

            # HUD
            avg = np.mean(dev_hist) if len(dev_hist)>0 else 0.0
            text1 = f'pixel={pix_mm:.2f}mm  grid={height.shape[1]}x{height.shape[0]}  dev_avg={avg: .3f}mm'
            text2 = f'dx={dxdy[0]: .3f} mm  dy={dxdy[1]: .3f} mm  mode={roi_mode}  valid={n_valid} pts'
            for (y,txt) in [(24,text1),(48,text2)]:
                cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow('FSW v3 (Global Top-Down + ROI after transform)', vis)

            # 交互（保留，主要用于机床 ROI 微调）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                outp = out_dir/f'vis_{frame_id:06d}.png'; cv2.imwrite(str(outp), vis); print(f'[SAVE] {outp}')
            elif key in (ord('='), ord('+')): CONFIG['pixel_size_mm'] = max(0.1, float(CONFIG['pixel_size_mm'])*0.8)
            elif key == ord('-'): CONFIG['pixel_size_mm'] = min(5.0, float(CONFIG['pixel_size_mm'])/0.8)
            elif key == ord('r'):
                CONFIG['roi_center_xy'] = [0.0, 0.0]; CONFIG['roi_size_mm'] = 120.0
            elif roi_mode == 'machine':
                if key == ord('i'): CONFIG['roi_center_xy'][1] += CONFIG['roi_size_mm']*0.1
                elif key == ord('k'): CONFIG['roi_center_xy'][1] -= CONFIG['roi_size_mm']*0.1
                elif key == ord('j'): CONFIG['roi_center_xy'][0] -= CONFIG['roi_size_mm']*0.1
                elif key == ord('l'): CONFIG['roi_center_xy'][0] += CONFIG['roi_size_mm']*0.1
                elif key == ord('['): CONFIG['roi_size_mm'] = max(20.0, CONFIG['roi_size_mm']*0.8)
                elif key == ord(']'): CONFIG['roi_size_mm'] = min(1200.0, CONFIG['roi_size_mm']/0.8)

            frame_id += 1
    finally:
        try: stream.close()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
