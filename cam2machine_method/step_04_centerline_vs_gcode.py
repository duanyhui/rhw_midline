#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 04 — 最近表面 → 中轴线 → 与 G 代码对比（右手系 +X→右, +Y→上）
----------------------------------------------------------------
- 输入：在线相机点云 (H,W,3) -> 机床系网格 (H,W,3)
- ROI：camera_rect / machine / none（ROI 仅做掩码，保持网格形状）
- 顶视投影：右手系映射（Y 越大像素行号越小）
- 最近表面：在 height(Hg,Wg) 上提取薄层掩码
- 中轴线：通用等距骨架（保留你全部可视化窗口）
- 对比：中轴线机床 XY vs G 代码路径，给出法向偏差箭头/统计
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import cv2
import math
import collections

# ================ 配置 ================
CONFIG = dict(
    # 文件
    T_path='T_cam2machine.npy',
    gcode_path='path/example.gcode',

    # 顶视投影
    pixel_size_mm=0.8,
    bounds_qlo=1.0, bounds_qhi=99.0,
    bounds_margin_mm=20.0,
    max_grid_pixels=1_200_000,

    # ROI：'none' / 'camera_rect' / 'machine'
    roi_mode='camera_rect',
    cam_roi_xywh=(714, 839, 228, 191),  # x,y,w,h in px
    roi_center_xy=(0.0, 0.0),           # mm
    roi_size_mm=150.0,                  # mm

    # 最近表面
    z_select='max',            # 'max': Z越大越近；'min': Z越小越近
    nearest_use_percentile=True,
    nearest_qlo=1.0,
    nearest_qhi=99.0,
    depth_margin_mm=5.0,
    morph_open=3,
    morph_close=5,
    min_component_area_px=600,

    # 中轴线显示/对比
    arrow_stride=12,          # 偏差箭头抽样步长（沿中轴点序号）
    rdp_epsilon_px=2.0,       # 中轴像素路径RDP简化（可选）
    show_skeleton_dilate=True, # 仅用于观察的轻度膨胀

    # 可视化
    colormap=getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
    out_dir='out'
)

# ============== Percipio SDK ==============
try:
    import pcammls
except Exception:
    pcammls = None


# ============== 基础 IO/几何 ==============

def load_extrinsic(T_path: Union[str, Path]):
    data = np.load(str(T_path), allow_pickle=True).item()
    R = np.asarray(data['R'], dtype=float)
    t = np.asarray(data['t'], dtype=float).reshape(1, 3)
    T = np.asarray(data['T'], dtype=float)
    return R, t, T


def parse_gcode_xy(path: Union[str, Path]) -> np.ndarray:
    p = Path(path) if not isinstance(path, Path) else path
    if (not path) or (not p.exists()):
        return np.empty((0, 2), float)
    pts = []
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        x = None; y = None
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith(';') or line.startswith('('):
                continue
            if ';' in line: line = line.split(';', 1)[0]
            while '(' in line and ')' in line:
                a = line.find('('); b = line.find(')')
                if a < 0 or b < 0 or b <= a: break
                line = (line[:a] + ' ' + line[b+1:]).strip()
            toks = line.split()
            if not toks: continue
            cmd = toks[0].upper()
            if cmd in ('G0','G00','G1','G01'):
                for tkn in toks[1:]:
                    u = tkn.upper()
                    if u.startswith('X'):
                        try: x = float(u[1:])
                        except: pass
                    elif u.startswith('Y'):
                        try: y = float(u[1:])
                        except: pass
                if x is not None and y is not None:
                    pts.append([x,y])
    return np.asarray(pts, float) if pts else np.empty((0,2), float)


def resample_polyline(poly: np.ndarray, step: float) -> np.ndarray:
    if poly.shape[0] < 2: return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = float(seg.sum())
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
    Pm = (R @ P.T).T + t  # (N,3)
    return Pm.reshape(H, W, 3)


# ============== 相机封装 ==============

class PCamMLSStream:
    def __init__(self):
        if pcammls is None:
            raise SystemExit('未安装 pcammls，无法使用相机。')
        self.cl = pcammls.PercipioSDK()
        self.h = None
        self.depth_calib = None
        self.scale_unit = 1.0
        self.pcl_buf = pcammls.pointcloud_data_list()

    def open(self):
        devs = self.cl.ListDevice()
        if len(devs) == 0: raise SystemExit('未发现设备。')
        print('检测到设备:')
        for i, d in enumerate(devs):
            print('  {}: {}\t{}'.format(i, d.id, d.iface.id))
        idx_str = input('选择设备索引 (回车默认0): ').strip()
        idx = int(idx_str) if idx_str else 0
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
        P_cam = self.pcl_buf.as_nparray()  # (H,W,3) mm, camera frame
        return P_cam, depth_fr

    def close(self):
        if self.h is not None:
            try: self.cl.DeviceStreamOff(self.h)
            except Exception: pass
            try: self.cl.Close(self.h)
            except Exception: pass
            self.h = None


# ============== 掩码/投影/可视化（右手系） ==============

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


def masked_Pmach_hw3(P_mach_hw3: np.ndarray, m_select: np.ndarray) -> np.ndarray:
    out = P_mach_hw3.copy().astype(np.float32)
    out[~m_select] = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    return out


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
                   gcode_xy: Optional[np.ndarray]=None,
                   machine_roi_rect_px: Optional[Tuple[int,int,int,int]]=None,
                   roi_mode_text: str='') -> np.ndarray:
    H,W = height.shape
    if np.isfinite(height).any():
        vmin = float(np.nanpercentile(height, 5)); vmax = float(np.nanpercentile(height, 95))
        vspan = max(1e-6, vmax - vmin)
        gray = np.clip(((height - vmin)/vspan) * 255, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(gray, CONFIG['colormap'])
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

    if machine_roi_rect_px is not None:
        x,y,w,h = machine_roi_rect_px
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2, cv2.LINE_AA)
        if roi_mode_text:
            cv2.putText(vis, roi_mode_text, (x+6, y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

    return vis


def draw_machine_axes_overlay(img: np.ndarray,
                              origin_xy: Tuple[float,float],
                              pix_mm: float) -> np.ndarray:
    H, W = img.shape[:2]
    vis = img.copy()
    base = (40, H-40)
    ax = int(60.0 / max(1e-6, pix_mm))
    cv2.arrowedLine(vis, base, (base[0] + ax, base[1]), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)   # +X
    cv2.arrowedLine(vis, base, (base[0], base[1] - ax), (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.25) # +Y
    cv2.putText(vis, '+X', (base[0] + ax + 6, base[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(vis, '+Y', (base[0] - 18, base[1] - ax - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)
    return vis


# ============== 最近表面（与 Step03 相同） ==============

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


# ============== 骨架（你提供的方法 + 回退） ==============

# 回退 skeletonize
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

def _get_outer_inner_contours(ring_mask):
    contours, hierarchy = cv2.findContours(ring_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None, None
    hierarchy = hierarchy[0]
    outer_idx = -1; outer_area = -1
    for i, h in enumerate(hierarchy):
        if h[3] == -1:
            a = cv2.contourArea(contours[i])
            if a > outer_area:
                outer_area = a; outer_idx = i
    if outer_idx < 0:
        return None, None
    child = hierarchy[outer_idx][2]
    inner_idx = -1; inner_area = -1
    while child != -1:
        a = cv2.contourArea(contours[child])
        if a > inner_area:
            inner_area = a; inner_idx = child
        child = hierarchy[child][0]
    outer_cnt = contours[outer_idx]
    inner_cnt = contours[inner_idx] if inner_idx >= 0 else None
    return outer_cnt, inner_cnt

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
            if skel[y, x] == 0:
                continue
            path = [(y, x)]
            prev = None
            cur = (y, x)
            while True:
                nbrs = neighbors(*cur)
                if prev is not None and prev in nbrs:
                    nbrs.remove(prev)
                if len(nbrs) == 0 or len(nbrs) > 1:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
                y2, x2 = cur
                if cv2.filter2D(skel, -1, k, borderType=cv2.BORDER_CONSTANT)[y2, x2] == 11 and len(path) > 1:
                    break
            if len(path) < min_branch_len:
                for yy, xx in path:
                    skel[yy, xx] = 0
                removed_any = True
        if not removed_any:
            break
    return (skel * 255).astype(np.uint8)

def _skeletonize_bool(bw_bool: np.ndarray) -> np.ndarray:
    """返回 uint8 0/255 骨架，尽量使用 skimage；失败则 ximgproc；再失败距离场回退。"""
    if _sk_skeletonize is not None:
        sk = _sk_skeletonize(bw_bool)
        return (sk.astype(np.uint8) * 255)
    if xip is not None:
        try:
            u8 = (bw_bool.astype(np.uint8) * 255)
            sk = xip.thinning(u8, thinningType=xip.THINNING_ZHANGSUEN)
            return (sk > 0).astype(np.uint8) * 255
        except Exception:
            pass
    # 距离场回退
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
    """
    你的通用等距中轴线算法（保持接口与可视化）。
    返回：BGR 图像（蓝色像素为 1px 骨架，若 visualize=True 会弹出可视化窗口）
    """
    if surface_mask is None or np.count_nonzero(surface_mask) == 0:
        print("输入的 surface_mask 为空或不包含有效区域。")
        return None

    ring_mask = _clean_mask_for_skeleton(surface_mask)
    if np.count_nonzero(ring_mask) == 0:
        print("净化后掩码为空。")
        return None

    outer_cnt, inner_cnt = _get_outer_inner_contours(ring_mask)
    if outer_cnt is None:
        print("未找到外轮廓。")
        return None

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

    H, W = ring_mask.shape
    outer_line = np.full((H, W), 255, np.uint8)
    inner_line = np.full((H, W), 255, np.uint8)
    cv2.drawContours(outer_line, [outer_cnt], -1, 0, 1)
    cv2.drawContours(inner_line, [inner_cnt], -1, 0, 1)

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

    if CONFIG.get('show_skeleton_dilate', True):
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
        # 如需观察等距带：
        cv2.imshow("Equidistance Band", equi_band)

    return cv2.cvtColor(skel_u8, cv2.COLOR_GRAY2BGR)


# ============== 骨架像素路径 → 有序折线（最长路） ==============

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
    return adj, endpoints

def _longest_path_from_graph(adj, endpoints):
    import collections
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
    return path[::-1]  # [(y,x), ...]

def skeleton_to_ordered_polyline_px(skel_gray_u8: np.ndarray) -> np.ndarray:
    """输入灰度骨架(0/255) → 返回 Nx2 像素坐标序列 [x,y]（最长路径）"""
    if skel_gray_u8.ndim == 3:
        skel_gray_u8 = cv2.cvtColor(skel_gray_u8, cv2.COLOR_BGR2GRAY)
    if np.count_nonzero(skel_gray_u8) < 2:
        return np.empty((0,2), float)
    adj, endpoints = _skeleton_graph(skel_gray_u8)
    path_nodes = _longest_path_from_graph(adj, endpoints)
    path_px = np.array([[x, y] for (y, x) in path_nodes], dtype=np.float32)
    # 可选：RDP 简化
    eps = float(CONFIG.get('rdp_epsilon_px', 0.0))
    if eps > 1e-6 and len(path_px) >= 3:
        path_px = _rdp(path_px, eps)
    return path_px

def _rdp(points: np.ndarray, eps: float) -> np.ndarray:
    if len(points) < 3: return points
    a, b = points[0], points[-1]
    ab = b - a; lab2 = (ab*ab).sum() + 1e-12
    dmax = -1; idx = -1
    for i in range(1, len(points)-1):
        ap = points[i] - a
        t = np.clip(np.dot(ap, ab)/lab2, 0, 1)
        proj = a + t*ab
        d = np.linalg.norm(points[i] - proj)
        if d > dmax: dmax = d; idx = i
    if dmax > eps:
        left = _rdp(points[:idx+1], eps)
        right = _rdp(points[idx:], eps)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([a, b])


# ============== 像素/机床坐标互换（右手系） ==============

def px_to_mach_xy(path_px: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int) -> np.ndarray:
    """像素路径 → 机床系 XY；注意右手系 y 反向：Y = y1 - (py+0.5)*pix"""
    if path_px.size == 0:
        return np.empty((0,2), float)
    x0, y0 = origin_xy
    y1 = y0 + Himg * pix_mm
    X = x0 + (path_px[:,0] + 0.5) * pix_mm
    Y = y1 - (path_px[:,1] + 0.5) * pix_mm
    return np.stack([X, Y], axis=1)

def mach_xy_to_px(xy: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int, Wimg: int) -> np.ndarray:
    x0, y0 = origin_xy
    y1 = y0 + Himg * pix_mm
    xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, Wimg-1)
    ys = np.clip(((y1 - xy[:,1])/pix_mm).astype(int), 0, Himg-1)
    return np.stack([xs,ys], axis=1)


# ============== 偏差计算与可视化 ==============

def project_points_to_path(points_xy: np.ndarray, ref_xy: np.ndarray, ref_tree, ref_normals: np.ndarray):
    if points_xy.size == 0 or ref_xy.size == 0:
        return np.empty((0,), int), np.empty((0, 2)), np.empty((0,))
    _, idx = ref_tree.query(points_xy)
    idx = np.asarray(idx, dtype=int)
    nearest = ref_xy[idx]
    N = ref_normals[np.clip(idx, 0, len(ref_normals) - 1)]
    e_n = ((points_xy - nearest) * N).sum(axis=1)
    return idx, nearest, e_n

def draw_deviation_overlay(vis_top: np.ndarray,
                           origin_xy: Tuple[float,float], pix_mm: float,
                           gcode_xy: Optional[np.ndarray],
                           centerline_xy: Optional[np.ndarray],
                           e_idx: np.ndarray, e_n: np.ndarray,
                           arrow_stride: int = 10) -> np.ndarray:
    H, W = vis_top.shape[:2]
    out = vis_top.copy()

    def xy_to_px(xy):
        return mach_xy_to_px(xy, origin_xy, pix_mm, H, W)

    # G代码（白）
    if gcode_xy is not None and gcode_xy.size > 0:
        p = xy_to_px(gcode_xy)
        for i in range(len(p)-1):
            cv2.line(out, tuple(p[i]), tuple(p[i+1]), (240,240,240), 1, cv2.LINE_AA)

    # 中轴（青色）
    if centerline_xy is not None and centerline_xy.size > 0:
        q = xy_to_px(centerline_xy)
        for i in range(len(q)-1):
            cv2.line(out, tuple(q[i]), tuple(q[i+1]), (200,255,255), 2, cv2.LINE_AA)

    # 偏差箭头（绿/橙）
    if e_n.size > 0 and gcode_xy is not None and gcode_xy.shape[0] > 1:
        seg = np.diff(gcode_xy, axis=0)
        T = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9)
        N = np.stack([-T[:,1], T[:,0]], axis=1)
        for k in range(0, len(e_idx), max(1, arrow_stride)):
            i = int(np.clip(e_idx[k], 0, len(N)-1))
            n = N[i]
            base = centerline_xy[k]
            tip = base + n * e_n[k]
            b = xy_to_px(base[None,:])[0]
            t = xy_to_px(tip[None,:])[0]
            col = (0,255,0) if abs(e_n[k]) < 0.5 else (0,165,255)
            cv2.arrowedLine(out, tuple(b), tuple(t), col, 2, cv2.LINE_AA, tipLength=0.35)
    return out


# ============== 主流程 ==============

def main():
    # 读取外参与 G 代码
    T_path = input('外参路径 (默认 {}): '.format(CONFIG['T_path'])).strip() or CONFIG['T_path']
    G_path = input('G代码路径 (默认 {}): '.format(CONFIG['gcode_path'])).strip() or CONFIG['gcode_path']
    if not Path(T_path).exists():
        raise SystemExit('外参文件不存在: {}'.format(T_path))
    R, t, _ = load_extrinsic(T_path)
    g_raw = parse_gcode_xy(G_path)
    g_xy = resample_polyline(g_raw, 1.0) if g_raw.size > 0 else g_raw
    ref_tree = build_kdtree(g_xy) if g_xy.size > 0 else None
    _, N_ref = tangent_normal(g_xy) if g_xy.size > 0 else (np.zeros((0,2)), np.zeros((0,2)))

    # 相机
    stream = PCamMLSStream(); stream.open()
    out_dir = Path(CONFIG['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)
    frame_id = 0

    roi_mode = str(CONFIG.get('roi_mode', 'none')).lower()
    print('[INFO] 拉流中… (q 退出, s 保存, -/= 像素尺寸)  mode={}'.format(roi_mode))
    try:
        while True:
            P_cam, _ = stream.read_pointcloud(2000)
            if P_cam is None:
                print('[WARN] 无深度帧'); continue

            H, W, _ = P_cam.shape
            P_mach = transform_cam_to_machine_grid(P_cam, R, t)

            # ROI
            m_valid = valid_mask_hw(P_mach)
            if roi_mode == 'camera_rect':
                m_roi = camera_rect_mask(H, W, CONFIG['cam_roi_xywh'])
                m_select = m_valid & m_roi
                roi_text = 'ROI(camera_rect)'
            elif roi_mode == 'machine':
                m_roi = machine_rect_mask(P_mach, CONFIG['roi_center_xy'], CONFIG['roi_size_mm'])
                m_select = m_valid & m_roi
                roi_text = 'ROI(machine)'
            else:
                m_select = m_valid
                roi_text = ''

            P_mach_roi = masked_Pmach_hw3(P_mach, m_select)

            # 顶视边界/分辨率
            x0,x1,y0,y1 = compute_bounds_xy_from_mask(P_mach, m_select,
                                                       CONFIG['bounds_qlo'], CONFIG['bounds_qhi'],
                                                       CONFIG['bounds_margin_mm'])
            pix_mm = adjust_pixel_size(x0,x1,y0,y1, float(CONFIG['pixel_size_mm']), CONFIG['max_grid_pixels'])

            # 顶视投影（右手系）
            height, mask_top, origin_xy = project_topdown_from_grid(P_mach, m_select, pix_mm, (x0,x1,y0,y1))

            # 最近表面：height → 薄层掩码
            nearest_mask, z_ref, (z_low, z_high) = extract_nearest_surface_mask_from_height(
                height, (mask_top > 0),
                z_select=CONFIG['z_select'],
                depth_margin_mm=CONFIG['depth_margin_mm'],
                use_percentile=CONFIG['nearest_use_percentile'],
                qlo=CONFIG['nearest_qlo'],
                qhi=CONFIG['nearest_qhi'],
                morph_open=CONFIG['morph_open'],
                morph_close=CONFIG['morph_close'],
                min_component_area_px=CONFIG['min_component_area_px']
            )

            # === 骨架提取（保留你的可视化窗口） ===
            skel_bgr = extract_skeleton_universal(nearest_mask, visualize=True)
            if skel_bgr is None:
                # 正常会在函数里有提示并显示清理后的目标；这里继续下一帧
                cv2.imshow('Centerline vs G-code (RHR)', np.zeros((480, 640, 3), np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            skel_gray = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY)

            # 像素骨架 → 有序像素路径
            path_px = skeleton_to_ordered_polyline_px(skel_gray)
            if path_px.size == 0:
                cv2.imshow('Centerline vs G-code (RHR)', np.zeros((480, 640, 3), np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # 像素路径 → 机床 XY（右手系）
            Ht, Wt = height.shape
            centerline_xy = px_to_mach_xy(path_px, origin_xy, pix_mm, Ht)

            # 顶视渲染 + 最近表面叠加
            vis_top = render_topdown(height, mask_top, origin_xy, pix_mm,
                                     gcode_xy=g_xy, machine_roi_rect_px=None, roi_mode_text=roi_text)
            overlay = vis_top.copy()
            # recent surface（绿）
            gmask = cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR); gmask[:,:,1] = np.maximum(gmask[:,:,1], gmask[:,:,0])
            overlay = cv2.addWeighted(overlay, 1.0, gmask, 0.25, 0)

            # 偏差计算
            e_idx = np.array([], int); e_n = np.array([])
            if g_xy.size > 1 and centerline_xy.size > 0 and ref_tree is not None and N_ref.size > 0:
                e_idx, nearest_on_ref, e_n = project_points_to_path(centerline_xy, g_xy, ref_tree, N_ref)

            # 叠加：中轴线 + G 代码 + 偏差箭头
            vis_cmp = draw_deviation_overlay(overlay, origin_xy, pix_mm, g_xy, centerline_xy, e_idx, e_n,
                                             arrow_stride=int(CONFIG['arrow_stride']))
            vis_cmp = draw_machine_axes_overlay(vis_cmp, origin_xy, pix_mm)

            # HUD
            dev_mean = float(np.mean(e_n)) if e_n.size > 0 else 0.0
            dev_med  = float(np.median(e_n)) if e_n.size > 0 else 0.0
            dev_p95  = float(np.percentile(np.abs(e_n), 95)) if e_n.size > 0 else 0.0
            txt = 'z_ref={:.2f}  band=[{:.2f},{:.2f}]mm  pix={:.2f}mm  grid={}x{}  dev(mean/med/p95)={:.3f}/{:.3f}/{:.3f} mm'.format(
                z_ref, z_low, z_high, pix_mm, Wt, Ht, dev_mean, dev_med, dev_p95
            )
            cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)

            # 展示
            cv2.imshow('Top-Down + Nearest Surface', overlay)
            cv2.imshow('NearestSurfaceMask', nearest_mask)
            cv2.imshow('Centerline vs G-code (RHR)', vis_cmp)

            # 交互
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                outp = out_dir / f'centerline_cmp_{frame_id:06d}.png'
                cv2.imwrite(str(outp), vis_cmp); print('[SAVE]', outp); frame_id += 1
            elif k in (ord('='), ord('+')):
                CONFIG['pixel_size_mm'] = max(0.1, float(CONFIG['pixel_size_mm'])*0.8)
            elif k == ord('-'):
                CONFIG['pixel_size_mm'] = min(5.0, float(CONFIG['pixel_size_mm'])/0.8)

    finally:
        try: stream.close()
        except Exception: pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
