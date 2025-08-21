#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中轴线拟合（机床系管线的独立模块）
================================

目标
----
- 从二值掩码（最近表面区域）出发，提取骨架，构建像素级骨架图，找到**主路径**，
  对主路径进行**有序化、简化、平滑与等弧长重采样**，最终输出 **机床坐标系 XY 中轴线**。
- 提供丰富的可视化：
  - 在背景图（高度图伪彩或原始掩码）上叠加骨架点、端点/分叉点、主路径、
    切线/法线箭头、曲率色条、关键索引标注。

依赖
----
- 必需：numpy, opencv-python
- 可选：scikit-image（skeletonize），scipy（Savitzky-Golay 平滑）
- 若缺少可选依赖，会使用内置回退实现（效果略逊）。

如何使用
--------
from centerline_fitting import fit_centerline
centerline_xy, debug_vis, diag = fit_centerline(mask, origin_xy, pix_mm, background=vis_top)

返回
----
- centerline_xy: (N,2) 机床系 XY 的等弧长重采样中轴线
- debug_vis: 可直接显示/保存的调试图（BGR）
- diag: 诊断信息 dict（包含多阶段路径、曲率等）
"""
from __future__ import annotations
from typing import Tuple, Dict, Optional
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# 可选依赖
try:
    from skimage.morphology import skeletonize as sk_skeletonize  # type: ignore
except Exception:
    sk_skeletonize = None

try:
    from scipy.signal import savgol_filter  # type: ignore
except Exception:
    savgol_filter = None

# OpenCV ximgproc (可选)
try:
    import cv2.ximgproc as xip  # type: ignore
except Exception:
    xip = None


# =============================
# 默认配置
# =============================
DEFAULT_CFG: Dict = dict(
    # 骨架提取
    use_ximgproc_first=True,
    # RDP 简化（像素单位）
    rdp_epsilon_px=2.0,
    # Savitzky-Golay 平滑（像素单位），窗口需为奇数；若无 scipy，退化为移动平均
    sg_window=11,
    sg_polyorder=2,
    # 等弧长重采样（毫米单位；将自动转像素步长）
    resample_step_mm=1.0,
    # 可视化
    normal_stride_px=20,       # 每隔多少像素画一根法线
    show_indices_every=20,     # 每隔多少点标一个索引
    curvature_amp=200.0,       # 曲率可视化增强系数
    colormap=getattr(cv2, 'COLORMAP_TURBO', getattr(cv2, 'COLORMAP_JET', 2)),
)


# =============================
# 工具函数
# =============================

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return (img.astype(np.uint8))*255
    m = img.copy()
    if m.ndim == 2:
        m = np.clip(m, 0, 255).astype(np.uint8)
    else:
        m = np.clip(m, 0, 255).astype(np.uint8)
    return m


def _skeletonize(mask: np.ndarray, use_ximgproc_first: bool=True) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if xip is not None and use_ximgproc_first:
        try:
            sk = xip.thinning(m, thinningType=xip.THINNING_ZHANGSUEN)
            return (sk > 0).astype(np.uint8)*255
        except Exception:
            pass
    if sk_skeletonize is not None:
        try:
            sk = sk_skeletonize(m.astype(bool))
            return (sk.astype(np.uint8))*255
        except Exception:
            pass
    # 回退：距离变换脊线近似
    if cv2 is None:
        return m*255
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    # 简单非极大抑制
    sk = np.zeros_like(m)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx==0 and dy==0: continue
            sk = np.maximum(sk, (dist < np.roll(np.roll(dist, dy, 0), dx, 1)).astype(np.uint8))
    ridge = (sk == 0).astype(np.uint8) & (m>0)
    return ridge.astype(np.uint8)*255


def _skeleton_graph(skel: np.ndarray):
    """把骨架像素转换为无向图：节点=像素坐标(y,x)，边=8邻接。
    返回：nodes(list[(y,x)]), adj(dict[node] -> set(neighbors))，endpoints，junctions
    """
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
                if 0 <= yy < H and 0 <= xx < W and (yy,xx) in S:
                    adj[(y,x)].add((yy,xx))
    deg = {n:len(adj[n]) for n in nodes}
    endpoints = [n for n,d in deg.items() if d==1]
    junctions = [n for n,d in deg.items() if d>=3]
    return nodes, adj, endpoints, junctions


def _longest_path_from_graph(adj, endpoints):
    """选主路径：若有端点，则计算端点对的最远 geodesic；否则从任意点做两次 BFS 近似最长路径。"""
    import collections
    def bfs(start):
        vis = {start: None}
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in vis:
                    vis[v] = u
                    q.append(v)
        # 返回距离最远的点及其前驱链表
        far = max(vis.keys(), key=lambda k: _l1dist(start, k))
        return far, vis

    def _l1dist(a,b):
        return abs(a[0]-b[0])+abs(a[1]-b[1])

    if endpoints:
        # 选一个端点 a，BFS 找到最远 b；再从 b 反向找回路径
        a = endpoints[0]
        b, pre = bfs(a)
        # 再次 BFS 以增大找最远端
        c, pre2 = bfs(b)
        # 用 pre2 从 c 回溯到 b
        path = [c]
        p = pre2[c]
        while p is not None:
            path.append(p)
            p = pre2[p]
        path = path[::-1]
        return path
    # 无端点：从任意点两次 BFS
    anyn = next(iter(adj.keys()))
    b, pre = bfs(anyn)
    c, pre2 = bfs(b)
    path = [c]
    p = pre2[c]
    while p is not None:
        path.append(p)
        p = pre2[p]
    path = path[::-1]
    return path


def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer–Douglas–Peucker 简化。points: (N,2)；eps: 容许误差（像素）。"""
    if len(points) < 3:
        return points
    # 端点
    a, b = points[0], points[-1]
    # 到线段距离最大值
    ab = b - a
    lab2 = (ab*ab).sum() + 1e-12
    dmax = -1
    idx = -1
    for i in range(1, len(points)-1):
        ap = points[i] - a
        t = np.clip(np.dot(ap, ab)/lab2, 0, 1)
        proj = a + t*ab
        d = np.linalg.norm(points[i]-proj)
        if d > dmax:
            dmax = d; idx = i
    if dmax > eps:
        left = _rdp(points[:idx+1], eps)
        right = _rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


def _smooth_polyline(px: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    if len(px) < max(5, window):
        return px
    if savgol_filter is not None and window % 2 == 1:
        xs = savgol_filter(px[:,0], window, polyorder)
        ys = savgol_filter(px[:,1], window, polyorder)
        return np.stack([xs, ys], axis=1)
    # 回退：移动平均
    k = max(3, window|1)  # odd
    pad = k//2
    ext = np.pad(px, ((pad,pad),(0,0)), mode='edge')
    ker = np.ones((k,1))/k
    xs = np.convolve(ext[:,0], ker[:,0], mode='valid')
    ys = np.convolve(ext[:,1], ker[:,0], mode='valid')
    return np.stack([xs, ys], axis=1)


def _polyline_length(px: np.ndarray) -> float:
    if len(px) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(px, axis=0), axis=1).sum())


def _resample_polyline(px: np.ndarray, step: float) -> np.ndarray:
    if len(px) < 2:
        return px
    seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
    L = float(seg.sum())
    if L < 1e-9:
        return px[[0]].copy()
    n = max(2, int(np.ceil(L/step)))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    j = 0
    for si in s:
        while j < len(seg) and si > cs[j+1]:
            j += 1
        if j >= len(seg):
            out.append(px[-1])
        else:
            t = (si - cs[j]) / max(seg[j], 1e-9)
            p = px[j]*(1-t) + px[j+1]*t
            out.append(p)
    return np.asarray(out, dtype=float)


def _curvature(px: np.ndarray) -> np.ndarray:
    if len(px) < 3:
        return np.zeros((len(px),), dtype=float)
    # 一阶/二阶差分近似曲率 κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    d1 = np.gradient(px, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0])
    den = (d1[:,0]**2 + d1[:,1]**2)**1.5 + 1e-12
    kappa = num/den
    kappa[~np.isfinite(kappa)] = 0
    return kappa


def _pix_to_machine(px: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float) -> np.ndarray:
    x0, y0 = origin_xy
    X = x0 + (px[:,0] + 0.5) * pix_mm
    Y = y0 + (px[:,1] + 0.5) * pix_mm
    return np.stack([X, Y], axis=1)


# =============================
# 可视化
# =============================

def _render_debug(background: Optional[np.ndarray], mask: np.ndarray, skel: np.ndarray,
                  path_px: np.ndarray, endpoints, junctions, normals_px: np.ndarray,
                  kappa: np.ndarray, cfg: Dict) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError('需要安装 OpenCV (cv2)。')
    H, W = mask.shape
    if background is None:
        bg = cv2.cvtColor(_ensure_uint8(mask), cv2.COLOR_GRAY2BGR)
    else:
        bg = background.copy()
        if bg.shape[:2] != mask.shape:
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_NEAREST)
    vis = bg

    # 骨架（细灰）
    sk3 = cv2.cvtColor(_ensure_uint8((skel>0)*255), cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(vis, 0.9, sk3, 0.3, 0)

    # 端点与分叉点
    for y,x in endpoints:
        cv2.circle(vis, (x,y), 3, (0,255,255), -1)
    for y,x in junctions:
        cv2.circle(vis, (x,y), 3, (0,165,255), -1)

    # 主路径（曲率着色）
    if len(path_px) >= 2:
        # 归一化曲率
        k = kappa.copy()
        if k.size != len(path_px):
            k = np.interp(np.linspace(0,1,len(path_px)), np.linspace(0,1,len(kappa)), kappa) if kappa.size>1 else np.zeros(len(path_px))
        k = np.clip(k * cfg['curvature_amp'], 0, 1)
        k8 = (k*255).astype(np.uint8)
        cm = cv2.applyColorMap(k8, cfg['colormap'])
        for i in range(len(path_px)-1):
            c = tuple(int(v) for v in cm[i,0].tolist())
            cv2.line(vis, tuple(path_px[i][::-1]), tuple(path_px[i+1][::-1]), c, 2, cv2.LINE_AA)

    # 法线
    for (x,y, nx,ny) in normals_px:
        p1 = (int(x), int(y))
        p2 = (int(x + nx), int(y + ny))
        cv2.arrowedLine(vis, p1, p2, (0,255,0), 1, cv2.LINE_AA, tipLength=0.3)

    # 关键索引
    for i in range(0, len(path_px), max(1,int(cfg['show_indices_every']))):
        x, y = path_px[i]
        cv2.putText(vis, str(i), (int(x)+3, int(y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    legend = np.zeros((60, vis.shape[1], 3), np.uint8)
    cv2.putText(legend, 'yellow=endpoints  orange=junctions  line=curvature-colored  green=normal',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    vis_out = np.vstack([vis, legend])
    return vis_out


# =============================
# 主入口
# =============================

def fit_centerline(mask: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float,
                   background: Optional[np.ndarray]=None, cfg: Optional[Dict]=None):
    """
    输入：
      - mask: HxW uint8 最近表面掩码（>0 为前景）
      - origin_xy: 投影左下角像素(0,0)对应的机床坐标 (x0,y0)
      - pix_mm: 每像素毫米数
      - background: 可选，HxW 或 HxW×3 的可视化背景（如高度图伪彩）
      - cfg: 可选配置（不提供则用 DEFAULT_CFG）
    输出：
      - centerline_xy: (N,2) 机床系 XY 中轴线（等弧长重采样、已平滑）
      - debug_vis: 调试可视化图（BGR）
      - diag: 诊断信息字典
    """
    if cv2 is None:
        raise RuntimeError('需要安装 OpenCV (cv2)。')
    C = dict(DEFAULT_CFG)
    if cfg:
        C.update(cfg)

    # 1) 骨架
    skel = _skeletonize(mask, C['use_ximgproc_first'])
    nodes, adj, endpoints, junctions = _skeleton_graph(skel)
    if len(nodes) == 0:
        # 直接返回空
        H, W = mask.shape
        dbg = background if background is not None else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.empty((0,2), float), dbg, dict(reason='empty-mask')

    # 2) 主路径（像素坐标，(x,y) 顺序）
    path_nodes = _longest_path_from_graph(adj, endpoints)  # 列表元素是 (y,x)
    path_px = np.array([[x,y] for (y,x) in path_nodes], dtype=float)

    # 3) RDP 简化 → 平滑
    if len(path_px) >= 3:
        path_px = _rdp(path_px, C['rdp_epsilon_px'])
    path_px = _smooth_polyline(path_px, C['sg_window'], C['sg_polyorder'])

    # 4) 等弧长重采样（像素步长 = resample_step_mm / pix_mm）
    step_px = max(1.0, float(C['resample_step_mm'])/max(1e-6, pix_mm))
    path_px = _resample_polyline(path_px, step_px)

    # 5) 曲率
    kappa = _curvature(path_px)

    # 6) 法线可视化
    normals_vis = []
    if len(path_px) >= 2:
        d = np.gradient(path_px, axis=0)
        T = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        N = np.stack([-T[:,1], T[:,0]], axis=1)
        stride = max(1, int(C['normal_stride_px']))
        L = 12.0  # 可视化长度（像素）
        for i in range(0, len(path_px), stride):
            x,y = path_px[i]
            nx,ny = N[i]*L
            normals_vis.append((x,y,nx,ny))
    normals_vis = np.array(normals_vis) if normals_vis else np.zeros((0,4))

    # 7) 可视化合成
    debug_vis = _render_debug(background, mask, skel, path_px, endpoints, junctions, normals_vis, kappa, C)

    # 8) 换算到机床系 XY
    centerline_xy = _pix_to_machine(path_px, origin_xy, pix_mm)

    diag = dict(
        path_px=path_px,
        endpoints=endpoints,
        junctions=junctions,
        curvature=kappa,
        normals_px=normals_vis,
        cfg=C,
    )
    return centerline_xy, debug_vis, diag


if __name__ == '__main__':
    print('本文件作为模块供管线调用；如需单独测试，请在其它脚本中构造 mask / origin_xy / pix_mm 并调用 fit_centerline()。')
