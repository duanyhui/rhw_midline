#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSW 在线视觉管线 v2（模块化：中轴线拟合 + 偏差控制）
=================================================

新增内容
--------
- 集成 `centerline_fitting.fit_centerline`：从掩码提取主路径，简化/平滑/等弧长重采样，输出机床系 XY 中轴线；
  同时产出**调试可视化**（骨架、端点/分叉、曲率着色、法线箭头、索引）。
- 集成 `deviation_controller`：与理论路径比对，计算横向偏差 e_n，
  并通过 EMA/死区/限速/限步 生成平滑的 (dx, dy) 建议输出；
  在俯视图上叠加**偏差箭头**与统计条。

依赖
----
- 必需：numpy, opencv-python, pcammls（PercipioSDK）
- 可选：scikit-image, scipy

用法
----
- 直接运行脚本，按提示输入外参与 G 代码路径；或修改 CONFIG 默认项。
- 窗口热键：`q` 退出；`s` 截图；`i/k/j/l` 平移 ROI；`[`/`]` 缩放 ROI；`-`/`=` 改变栅格分辨率；`r` 重置。
"""
from __future__ import annotations
from pathlib import Path
import collections
import math
import numpy as np
import json

import cv2  # 必需

# 相机与模块
import pcammls
from centerline_fitting import fit_centerline
from deviation_controller import (
    DeviationController, ControllerConfig,
    build_kdtree, tangent_normal, project_points_to_path,
    draw_deviation_overlay,
)

# =============================
# 配置
# =============================
CONFIG = dict(
    T_path='T_cam2machine.npy',
    gcode_path='path/example.gcode',
    pixel_size_mm=0.5,
    roi_size_mm=120.0,
    roi_center_xy=[0.0, 0.0],
    z_select='max',
    min_points_per_cell=1,
    morph_open=3,
    morph_close=5,
    resample_step_mm=1.0,
    smooth_window=7,
    max_dev_to_show_mm=2.0,
    colormap=getattr(cv2, 'COLORMAP_TURBO', getattr(cv2, 'COLORMAP_JET', 2)),
    out_dir='out/frames'
)

# =============================
# 基础工具
# =============================

def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_extrinsic(T_path: str | Path):
    data = np.load(T_path, allow_pickle=True).item()
    return np.asarray(data['R'], float), np.asarray(data['t'], float), np.asarray(data['T'], float)


def transform_cam_to_machine(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ P_cam.T).T + t

# G 代码解析（G0/G1）

def parse_gcode_xy(path: str | Path) -> np.ndarray:
    pts = []
    if not path or not Path(path).exists():
        return np.empty((0, 2), float)
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
    n = max(2, int(np.ceil(L/max(1e-6, step))))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out=[]; j=0
    for si in s:
        while j < len(seg) and si > cs[j+1]: j+=1
        if j >= len(seg): out.append(poly[-1]); continue
        t = (si - cs[j]) / max(seg[j],1e-9)
        out.append(poly[j]*(1-t) + poly[j+1]*t)
    return np.asarray(out, float)

# 投影与掩码

def orthographic_project_top(P_mach: np.ndarray, roi_center: np.ndarray, roi_size: float, pix_mm: float,
                             z_select: str='max', min_points_per_cell: int=1):
    half = roi_size*0.5
    cx, cy = float(roi_center[0]), float(roi_center[1])
    x0, x1 = cx-half, cx+half
    y0, y1 = cy-half, cy+half
    X,Y,Z = P_mach[:,0], P_mach[:,1], P_mach[:,2]
    m = (X>=x0)&(X<x1)&(Y>=y0)&(Y<y1)&np.isfinite(Z)
    W = H = int(max(2, round(roi_size/pix_mm)))
    if not np.any(m):
        return np.full((H,W), np.nan, np.float32), np.zeros((H,W), np.uint8), (x0,y0)
    X, Y, Z = X[m], Y[m], Z[m]
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
    return height, mask, (x0,y0)


def morph_cleanup(mask, open_k: int, close_k: int):
    m = mask.copy()
    if open_k and open_k>1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k and close_k>1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k,close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m

# 可视化

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

# 相机（PCamMLS）
class PCamMLSStream:
    def __init__(self):
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
            raise SystemExit(f'打开失败: {self.cl.TYGetLastErrorCodedescription()}')
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
        return self.pointcloud.as_nparray()
    def close(self):
        if self.h is not None:
            try: self.cl.DeviceStreamOff(self.h)
            except Exception: pass
            try: self.cl.Close(self.h)
            except Exception: pass
            self.h = None

# =============================
# 主流程
# =============================

def main():
    # 外参与 G 代码
    T_path = input(f"外参路径（默认 {CONFIG['T_path']}）：").strip() or CONFIG['T_path']
    if not Path(T_path).exists():
        raise SystemExit(f'外参不存在：{T_path}')
    R,t,T = load_extrinsic(T_path)

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

    # 控制器与统计
    ctrl = DeviationController(ControllerConfig(
        ema_alpha=0.3, deadband_mm=0.05, clip_mm=CONFIG['max_dev_to_show_mm'],
        max_step_mm=0.15, max_rate_mm_s=5.0, lead_distance_mm=5.0
    ))
    dev_hist = collections.deque(maxlen=max(1,int(CONFIG['smooth_window'])))

    out_dir = Path(CONFIG['out_dir']); ensure_dir(out_dir)
    frame_id = 0

    try:
        while True:
            P_cam = stream.read_pointcloud()
            if P_cam is None: continue
            P_mach = transform_cam_to_machine(P_cam.reshape(-1,3).astype(np.float32), R, t)

            # 首帧自动 ROI
            if frame_id == 0 and (roi_center==0).all():
                med = np.nanmedian(P_mach, axis=0)
                if np.isfinite(med[0]) and np.isfinite(med[1]):
                    roi_center = med[:2]

            height, mask, origin_xy = orthographic_project_top(
                P_mach, roi_center, roi_size, pix_mm,
                z_select=CONFIG['z_select'], min_points_per_cell=CONFIG['min_points_per_cell']
            )
            mask = morph_cleanup(mask, CONFIG['morph_open'], CONFIG['morph_close'])

            # 中轴线拟合（带可视化）
            vis_top = render_topdown(height, mask, origin_xy, pix_mm, g_xy, np.empty((0,2)))
            centerline_xy, dbg_centerline, diag = fit_centerline(
                mask, origin_xy, pix_mm, background=vis_top,
                cfg=dict(resample_step_mm=CONFIG['resample_step_mm'])
            )

            # 偏差与控制
            e_n = np.array([]); idx = np.array([], int)
            if g_xy.size>0 and centerline_xy.size>0 and ref_tree is not None and N_ref.size>0:
                idx, nearest, e_n = project_points_to_path(centerline_xy, g_xy, ref_tree, N_ref)
                dxdy, stats = ctrl.update(e_n, N_ref[np.clip(idx,0,len(N_ref)-1)])
                dev_hist.append(stats['mean'])
            else:
                dxdy = np.zeros(2,float); stats = dict(mean=0.0, median=0.0, p95=0.0, n=0)

            # 可视化叠加
            vis_dev = draw_deviation_overlay(vis_top, g_xy, centerline_xy, idx, e_n, origin_xy, pix_mm, stride=10)
            dbg_scaled = cv2.resize(dbg_centerline, (vis_dev.shape[1], vis_dev.shape[0]), interpolation=cv2.INTER_AREA)
            vis = np.vstack([vis_dev, dbg_scaled])

            # 文本信息
            cv2.putText(vis, f'pixel={pix_mm:.2f}mm  roi={roi_size:.0f}mm  dev_avg={np.mean(dev_hist) if len(dev_hist)>0 else 0.0: .3f}mm',
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f'pixel={pix_mm:.2f}mm  roi={roi_size:.0f}mm  dev_avg={np.mean(dev_hist) if len(dev_hist)>0 else 0.0: .3f}mm',
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(vis, f'dx={dxdy[0]: .3f} mm  dy={dxdy[1]: .3f} mm  n={stats.get("n",0)}',
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f'dx={dxdy[0]: .3f} mm  dy={dxdy[1]: .3f} mm  n={stats.get("n",0)}',
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow('FSW v2 (Top-Down + Centerline + Deviations)', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('s'):
                outp = out_dir/f'vis_{frame_id:06d}.png'
                cv2.imwrite(str(outp), vis); print(f'[SAVE] {outp}')
            elif key in (ord('='), ord('+')): pix_mm = max(0.1, pix_mm*0.8)
            elif key == ord('-'): pix_mm = min(5.0, pix_mm/0.8)
            elif key == ord('i'): roi_center[1] += roi_size*0.1
            elif key == ord('k'): roi_center[1] -= roi_size*0.1
            elif key == ord('j'): roi_center[0] -= roi_size*0.1
            elif key == ord('l'): roi_center[0] += roi_size*0.1
            elif key == ord('['): roi_size = max(20.0, roi_size*0.8)
            elif key == ord(']'): roi_size = min(1000.0, roi_size/0.8)
            elif key == ord('r'):
                roi_center = np.array(CONFIG['roi_center_xy'], float)
                pix_mm = CONFIG['pixel_size_mm']
                roi_size = CONFIG['roi_size_mm']
            frame_id += 1
    finally:
        stream.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
