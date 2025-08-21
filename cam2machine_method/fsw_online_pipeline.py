#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSW 在线视觉管线（机床系）
=========================

功能概述
--------
- 读取外参 `T_cam2machine.npy`（R, t），将相机点云转换到**机床坐标系**（单位 mm）。
- 通过 PCamMLS（PercipioSDK）实时采集深度并映射点云。
- 在机床系下做 **Z 轴向下的正交投影**，生成 XY 栅格上的“最近表面”高度图与掩码。
- 对掩码做清理并**提取中轴线/骨架**（多种后端：OpenCV ximgproc / scikit-image / 简易回退）。
- 解析 G 代码得到**理论路径**（机床系 XY），可重采样并计算切线/法线场。
- 将“实际中轴线”与“理论路径”对比，计算**横向偏差 e_n(s)**，并可视化叠加：
  - 机床系俯视高度图（Z→颜色）
  - 理论路径与实际中轴线
  - 偏差数值与稳定统计

使用方式
--------
- 无需命令行参数，运行后按提示输入路径或直接使用 `CONFIG` 默认值。
- 依赖：`numpy`, `opencv-python`; 可选：`scikit-image`（骨架提取）、`scipy`（KDTree）。

键盘交互
--------
- `q` 退出；`s` 保存当前可视化到 `out/frames/`；`-`/`=` 调整投影分辨率；
- `i/k/j/l` 平移 ROI；`[`/`]` 改变 ROI 大小；`r` 复位 ROI。

注意
----
- 单位统一为 **mm**。
- 若 OpenCV 没有 ximgproc 或系统无 scikit-image，将使用简易回退方案（精度较低）。
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple, Optional
import time
import json
import math
import collections

import numpy as np

# 可选依赖
try:
    import cv2
except Exception:
    cv2 = None

try:
    import pcammls  # PercipioSDK Python
except Exception:
    pcammls = None

try:
    from skimage.morphology import skeletonize as sk_skeletonize  # type: ignore
except Exception:
    sk_skeletonize = None

try:
    import cv2.ximgproc as xip  # type: ignore
except Exception:
    xip = None

try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
except Exception:
    KDTree = None


# =============================
# 参数区（可直接修改默认）
# =============================
CONFIG = dict(
    # 文件路径
    T_path='T_cam2machine.npy',       # 外参（由 calibrate_extrinsic_3d.py 生成）
    gcode_path='path/example.gcode',  # G 代码路径（如果为空，将只显示点云与骨架）

    # 投影与 ROI（机床系）
    pixel_size_mm=0.5,                # 俯视栅格分辨率（越小越细致，计算也越慢）
    roi_size_mm=120.0,                # 初始 ROI 边长（正方形）
    roi_center_xy=[0.0, 0.0],         # 初始 ROI 中心（机床系 XY），为空则自动取首帧点云中位数

    # 最近表面提取与形态学
    z_select='max',                   # 'max' 取最高点（靠近相机/向上），或 'min' 取最低点
    min_points_per_cell=1,            # 每个栅格至少多少点才算有效
    morph_open=3,                     # 开运算核尺寸（像素）
    morph_close=5,                    # 闭运算核尺寸（像素）

    # 骨架提取
    use_ximgproc_first=True,          # 优先尝试 OpenCV ximgproc.thinning

    # 偏差计算
    resample_step_mm=1.0,             # 理论路径重采样间距（mm）
    smooth_window=7,                  # 偏差滑动平均窗口（帧）
    max_dev_to_show_mm=2.0,           # 显示用限幅（mm）

    # UI 与输出
    colormap=getattr(cv2, 'COLORMAP_TURBO', getattr(cv2, 'COLORMAP_JET', 2)),
    out_dir='out/frames',
)


# =============================
# 工具函数
# =============================

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_extrinsic(T_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(T_path, allow_pickle=True).item()
    R = np.asarray(data['R'], dtype=float)
    t = np.asarray(data['t'], dtype=float)
    T = np.asarray(data['T'], dtype=float)
    return R, t, T


def transform_cam_to_machine(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P_cam.T).T + t


def parse_gcode_xy(path: str | Path) -> np.ndarray:
    pts = []
    if not path or not Path(path).exists():
        return np.empty((0, 2), dtype=float)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        x, y = None, None
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(';') or line.startswith('('):
                continue
            # 去注释
            if ';' in line:
                line = line.split(';', 1)[0].strip()
            if '(' in line and ')' in line:
                # 简单去掉小括号注释
                while '(' in line and ')' in line:
                    a, b = line.find('('), line.find(')')
                    if a < 0 or b < 0 or b <= a:
                        break
                    line = (line[:a] + ' ' + line[b+1:]).strip()
            tokens = line.split()
            if not tokens:
                continue
            cmd = tokens[0].upper()
            if cmd in ('G0', 'G00', 'G1', 'G01'):
                for tok in tokens[1:]:
                    m = tok.upper()
                    if m.startswith('X'):
                        try: x = float(m[1:])
                        except: pass
                    elif m.startswith('Y'):
                        try: y = float(m[1:])
                        except: pass
                if x is not None and y is not None:
                    pts.append([x, y])
            # G2/G3 圆弧：此处简单忽略（可后续扩展采样）
    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pts, dtype=float)


def resample_polyline(poly: np.ndarray, step: float) -> np.ndarray:
    if poly.shape[0] < 2:
        return poly.copy()
    seglens = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = float(np.sum(seglens))
    if L <= 1e-9:
        return poly[[0]].copy()
    n = max(2, int(math.ceil(L / max(1e-6, step))))
    s = np.linspace(0.0, L, n)
    # 原曲线累计弧长
    cs = np.concatenate([[0.0], np.cumsum(seglens)])
    out = []
    j = 0
    for si in s:
        while j < len(seglens) and si > cs[j+1]:
            j += 1
        if j >= len(seglens):
            out.append(poly[-1])
        else:
            t = (si - cs[j]) / max(seglens[j], 1e-9)
            p = poly[j] * (1-t) + poly[j+1] * t
            out.append(p)
    return np.asarray(out, dtype=float)


def poly_tangent_normal(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if poly.shape[0] < 2:
        T = np.array([[1.0, 0.0]])
        N = np.array([[0.0, 1.0]])
        return T, N
    d = np.gradient(poly, axis=0)
    T = d / np.maximum(1e-9, np.linalg.norm(d, axis=1, keepdims=True))
    N = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, N


def build_kdtree(pts: np.ndarray):
    if pts.size == 0:
        return None
    if KDTree is not None:
        return KDTree(pts)
    # 退化为简单封装：线性扫描
    class _Lin:
        def __init__(self, A):
            self.A = A
        def query(self, B):
            B = np.atleast_2d(B)
            d2 = ((B[:,None,:]-self.A[None,:,:])**2).sum(axis=2)
            idx = d2.argmin(axis=1)
            return np.sqrt(d2[np.arange(len(B)), idx]), idx
    return _Lin(pts)


# =============================
# 投影与掩码
# =============================

def orthographic_project_top(P_mach: np.ndarray, roi_center: np.ndarray, roi_size: float, pix_mm: float,
                             z_select: str='max', min_points_per_cell: int=1) -> Tuple[np.ndarray, np.ndarray, Tuple[float,float]]:
    """将点云在机床系下正交投影到 XY 栅格。
    返回： (height, mask, origin_xy)
      - height: HxW float32，高度图（Z 值，mm；无效为 NaN）
      - mask  : HxW uint8，最近表面掩码
      - origin_xy: (x0, y0) 为像素 (0,0) 对应的机床系 XY 左下角坐标
    """
    # ROI 方形边界（以中心为 (cx,cy)）
    half = roi_size * 0.5
    cx, cy = float(roi_center[0]), float(roi_center[1])
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    # 选择落入 ROI 的点
    X = P_mach[:,0]; Y = P_mach[:,1]; Z = P_mach[:,2]
    m = (X>=x0) & (X<x1) & (Y>=y0) & (Y<y1) & np.isfinite(Z)
    if not np.any(m):
        H = int(max(2, round(roi_size / pix_mm)))
        W = H
        return np.full((H,W), np.nan, np.float32), np.zeros((H,W), np.uint8), (x0, y0)
    X, Y, Z = X[m], Y[m], Z[m]

    W = int(max(2, round(roi_size / pix_mm)))
    H = W
    ix = np.clip(((X - x0) / pix_mm).astype(np.int32), 0, W-1)
    iy = np.clip(((Y - y0) / pix_mm).astype(np.int32), 0, H-1)

    height = np.full((H, W), np.nan, np.float32)
    count  = np.zeros((H, W), np.int32)
    if z_select == 'max':
        # 取同一像素内 Z 最大（更“靠近相机”/“越高”）
        for xg, yg, zg in zip(ix, iy, Z):
            if not np.isfinite(height[yg, xg]) or zg > height[yg, xg]:
                height[yg, xg] = zg
            count[yg, xg] += 1
    else:
        for xg, yg, zg in zip(ix, iy, Z):
            if not np.isfinite(height[yg, xg]) or zg < height[yg, xg]:
                height[yg, xg] = zg
            count[yg, xg] += 1
    mask = (count >= min_points_per_cell).astype(np.uint8) * 255
    return height, mask, (x0, y0)


def morph_cleanup(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    if mask.size == 0 or cv2 is None:
        return mask
    m = mask.copy()
    if open_k and open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k and close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


# =============================
# 骨架 / 中轴线
# =============================

def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    m = (mask > 0).astype(np.uint8)
    if xip is not None and CONFIG['use_ximgproc_first']:
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
    # 简易回退：用距离变换的脊线近似（质量较差）
    if cv2 is not None:
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        # 非极大值抑制：与邻域比较
        sk = np.zeros_like(m)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                sk = np.maximum(sk, (dist < np.roll(np.roll(dist, dy, 0), dx, 1)).astype(np.uint8))
        ridge = (sk == 0).astype(np.uint8) & (m>0)
        return ridge.astype(np.uint8)*255
    return m*255


def skeleton_pixels_to_machine_xy(skel: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float) -> np.ndarray:
    ys, xs = np.where(skel>0)
    if len(xs)==0:
        return np.empty((0,2), dtype=float)
    x0, y0 = origin_xy
    X = x0 + (xs + 0.5) * pix_mm
    Y = y0 + (ys + 0.5) * pix_mm
    return np.stack([X, Y], axis=1)


# =============================
# 偏差计算
# =============================

def deviations_to_path(actual_xy: np.ndarray, ref_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if actual_xy.size==0 or ref_xy.size==0:
        return np.empty((0,)), np.empty((0,2)), np.empty((0,2))
    tree = build_kdtree(ref_xy)
    d, idx = tree.query(actual_xy)
    nearest = ref_xy[idx]
    # 局部切线/法线
    T, N = poly_tangent_normal(ref_xy)
    Tn = T[np.clip(idx, 0, len(T)-1)]
    Nn = N[np.clip(idx, 0, len(N)-1)]
    dev_vec = actual_xy - nearest
    e_n = (dev_vec * Nn).sum(axis=1)
    return e_n, nearest, Nn


# =============================
# 可视化
# =============================

def render_topdown(height: np.ndarray, mask: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float,
                   gcode_xy: np.ndarray, centerline_xy: np.ndarray, cfg=CONFIG) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError('需要安装 OpenCV (cv2)。')
    H, W = height.shape
    vis = np.zeros((H, W, 3), np.uint8)
    # 高度着色
    h = height.copy()
    h[~np.isfinite(h)] = np.nan
    if np.isfinite(h).any():
        vmin = np.nanpercentile(h, 5)
        vmax = np.nanpercentile(h, 95)
        vspan = max(1e-6, vmax - vmin)
        norm = np.clip((h - vmin)/vspan, 0, 1)
        gray = (norm * 255).astype(np.uint8)
        color = cv2.applyColorMap(gray, cfg['colormap'])
        vis = color
    # 掩码淡显
    if mask is not None:
        vis = cv2.addWeighted(vis, 0.9, np.dstack([mask]*3), 0.1, 0)
    # 画 G 代码路径与中心线
    def xy_to_px(xy):
        x0, y0 = origin_xy
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((xy[:,1]-y0)/pix_mm).astype(int), 0, H-1)
        return np.stack([xs, ys], axis=1)
    if gcode_xy is not None and gcode_xy.size>0:
        pts = xy_to_px(gcode_xy)
        for i in range(len(pts)-1):
            cv2.line(vis, tuple(pts[i]), tuple(pts[i+1]), (255,255,255), 1, cv2.LINE_AA)
    if centerline_xy is not None and centerline_xy.size>0:
        pts = xy_to_px(centerline_xy)
        for p in pts:
            cv2.circle(vis, tuple(p), 1, (0,255,0), -1)
    return vis


def draw_deviation_bar(img: np.ndarray, e_n: np.ndarray, max_abs: float, title: str='e_n (mm)'):
    if cv2 is None:
        return img
    h = 80
    w = img.shape[1]
    bar = np.full((h, w, 3), 30, np.uint8)
    cv2.putText(bar, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)
    if e_n.size>1:
        # 映射到中线±max_abs
        mid = h//2
        scale = (h*0.4)/max(1e-6, max_abs)
        xs = np.linspace(0, w-1, len(e_n)).astype(int)
        ys = (mid - e_n*scale).astype(int)
        for i in range(len(xs)-1):
            c = (0, 255, 255) if abs(e_n[i])<max_abs*0.5 else (0, 128, 255)
            cv2.line(bar, (xs[i], ys[i]), (xs[i+1], ys[i+1]), c, 2, cv2.LINE_AA)
        cv2.line(bar, (0, mid), (w-1, mid), (120,120,120), 1, cv2.LINE_AA)
        cv2.putText(bar, f"avg={np.mean(e_n): .3f} mm", (w-180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
    return bar


# =============================
# 相机（PCamMLS）
# =============================
class PCamMLSStream:
    def __init__(self):
        if pcammls is None:
            raise RuntimeError('未安装 pcammls。')
        if cv2 is None:
            raise RuntimeError('需要 OpenCV 用于显示。')
        self.cl = pcammls.PercipioSDK()
        self.h = None
        self.depth_calib = None
        self.scale_unit = 1.0
        self.pointcloud = pcammls.pointcloud_data_list()

    def open(self):
        devs = self.cl.ListDevice()
        if len(devs)==0:
            raise RuntimeError('未发现设备。')
        print('检测到设备:')
        for i,d in enumerate(devs):
            print(f'  {i}: {d.id}\t{d.iface.id}')
        idx = 0 if len(devs)==1 else int(input('选择设备索引: '))
        sn = devs[idx].id
        h = self.cl.Open(sn)
        if not self.cl.isValidHandle(h):
            raise RuntimeError(f'打开设备失败: {self.cl.TYGetLastErrorCodedescription()}')
        self.h = h
        depth_fmts = self.cl.DeviceStreamFormatDump(h, pcammls.PERCIPIO_STREAM_DEPTH)
        if not depth_fmts:
            raise RuntimeError('无深度流。')
        self.cl.DeviceStreamFormatConfig(h, pcammls.PERCIPIO_STREAM_DEPTH, depth_fmts[0])
        self.cl.DeviceLoadDefaultParameters(h)
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(h)
        self.depth_calib = self.cl.DeviceReadCalibData(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamEnable(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamOn(h)

    def read_pointcloud(self, timeout_ms=2000) -> Optional[np.ndarray]:
        imgs = self.cl.DeviceStreamRead(self.h, timeout_ms)
        depth_img = None
        for fr in imgs:
            if fr.streamID == pcammls.PERCIPIO_STREAM_DEPTH:
                depth_img = fr
                break
        if depth_img is None:
            return None
        self.cl.DeviceStreamMapDepthImageToPoint3D(depth_img, self.depth_calib, self.scale_unit, self.pointcloud)
        return self.pointcloud.as_nparray()  # (H, W, 3) in mm

    def close(self):
        if self.h is not None:
            try:
                self.cl.DeviceStreamOff(self.h)
            except Exception: pass
            try:
                self.cl.Close(self.h)
            except Exception: pass
            self.h = None


# =============================
# 主流程
# =============================

def main():
    if cv2 is None:
        raise SystemExit('需要安装 OpenCV (cv2)。')

    # 加载外参
    T_path = input(f"外参路径（默认 {CONFIG['T_path']}）：").strip() or CONFIG['T_path']
    if not Path(T_path).exists():
        raise SystemExit(f'外参文件不存在：{T_path}')
    R, t, T = load_extrinsic(T_path)

    # 加载 G 代码并重采样
    gcode_path = input(f"G 代码路径（默认 {CONFIG['gcode_path']}）：").strip() or CONFIG['gcode_path']
    g_raw = parse_gcode_xy(gcode_path) if gcode_path and Path(gcode_path).exists() else np.empty((0,2))
    g_xy = resample_polyline(g_raw, CONFIG['resample_step_mm']) if g_raw.size>0 else g_raw

    # ROI 初始化
    pix_mm = CONFIG['pixel_size_mm']
    roi_size = CONFIG['roi_size_mm']
    roi_center = np.array(CONFIG['roi_center_xy'], dtype=float)
    if roi_center.shape[0] != 2:
        roi_center = np.array([0.0, 0.0], dtype=float)

    # 打开相机
    stream = PCamMLSStream()
    stream.open()

    # 平滑统计
    dev_hist = collections.deque(maxlen=max(1, int(CONFIG['smooth_window'])))

    ensure_dir(CONFIG['out_dir'])
    frame_id = 0

    try:
        while True:
            P_cam = stream.read_pointcloud()
            if P_cam is None:
                continue
            H, W, _ = P_cam.shape
            P_cam = P_cam.reshape(-1, 3).astype(np.float32)
            # 相机系 -> 机床系
            P_mach = transform_cam_to_machine(P_cam, R, t)

            # 自动初始化 ROI 中心（首帧用点云中位数）
            if frame_id == 0 and (roi_center==0).all():
                med = np.nanmedian(P_mach, axis=0)
                if np.isfinite(med[0]) and np.isfinite(med[1]):
                    roi_center = med[:2]

            # 正交投影 & 掩码
            height, mask, origin_xy = orthographic_project_top(
                P_mach, roi_center, roi_size, pix_mm,
                z_select=CONFIG['z_select'], min_points_per_cell=CONFIG['min_points_per_cell']
            )
            mask = morph_cleanup(mask, CONFIG['morph_open'], CONFIG['morph_close'])

            # 骨架与实际中轴线（机床系 XY）
            skel = skeletonize_mask(mask)
            centerline_xy = skeleton_pixels_to_machine_xy(skel, origin_xy, pix_mm)

            # 偏差计算
            e_n = np.array([])
            if g_xy.size>0 and centerline_xy.size>0:
                e_n, nearest, normals = deviations_to_path(centerline_xy, g_xy)
                dev_hist.append(np.nanmean(e_n) if e_n.size>0 else 0.0)

            # 可视化
            vis_top = render_topdown(height, mask, origin_xy, pix_mm, g_xy, centerline_xy)
            bar = draw_deviation_bar(vis_top, e_n, CONFIG['max_dev_to_show_mm']) if e_n.size>0 else None
            if bar is not None:
                vis = np.vstack([vis_top, bar])
            else:
                vis = vis_top
            # 文本叠加
            cv2.putText(vis, f'pixel={pix_mm:.2f}mm  roi={roi_size:.0f}mm  dev_avg={np.mean(dev_hist) if dev_hist else 0.0: .3f}mm',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f'pixel={pix_mm:.2f}mm  roi={roi_size:.0f}mm  dev_avg={np.mean(dev_hist) if dev_hist else 0.0: .3f}mm',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow('FSW Top-Down (Machine XY)', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                outp = Path(CONFIG['out_dir'])/f'vis_{frame_id:06d}.png'
                cv2.imwrite(str(outp), vis)
                print(f'[SAVE] {outp}')
            elif key in (ord('='), ord('+')):
                pix_mm = max(0.1, pix_mm*0.8)
            elif key == ord('-'):
                pix_mm = min(5.0, pix_mm/0.8)
            elif key == ord('i'):
                roi_center[1] += roi_size*0.1
            elif key == ord('k'):
                roi_center[1] -= roi_size*0.1
            elif key == ord('j'):
                roi_center[0] -= roi_size*0.1
            elif key == ord('l'):
                roi_center[0] += roi_size*0.1
            elif key == ord('['):
                roi_size = max(20.0, roi_size*0.8)
            elif key == ord(']'):
                roi_size = min(1000.0, roi_size/0.8)
            elif key == ord('r'):
                roi_center = np.array(CONFIG['roi_center_xy'], dtype=float)
                pix_mm = CONFIG['pixel_size_mm']
                roi_size = CONFIG['roi_size_mm']

            frame_id += 1

    finally:
        stream.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
