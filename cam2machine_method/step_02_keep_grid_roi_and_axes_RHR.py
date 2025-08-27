#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 03 — 最近表平面提取（基于 height(Hg,Wg)）+ 可视化
---------------------------------------------------
- 继续沿用：机床系网格 (H,W,3) -> ROI 掩码 -> 顶视投影 height/mask
- 在 height 上提取“最近表面薄层”掩码（分位数稳健参考 + 厚度带 + 形态学 + 最大连通域）
- 右手系：+X 右、+Y 上（仅影响投影/绘制，和掩码逻辑无关）
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
import cv2

# ================= 配置 =================
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
    cam_roi_xywh=(714, 839, 228, 191),  # 像素 ROI
    roi_center_xy=(0.0, 0.0),           # 机床 ROI 中心 (mm)
    roi_size_mm=150.0,                  # 机床 ROI 边长 (mm)

    # 最近表面提取（可按需改 z_select，见注释）
    z_select='max',            # 'max'：Z 越大越近；'min'：Z 越小越近
    nearest_use_percentile=True,
    nearest_qlo=1.0,
    nearest_qhi=99.0,
    depth_margin_mm=3.0,      # 薄层厚度（mm）
    morph_open=3,
    morph_close=5,
    min_component_area_px=600,

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


# ============== 最近表面（核心） ==============

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
    z_select: str = 'max',              # 'max'：Z 越大越近；'min'：Z 越小越近
    depth_margin_mm: float = 30.0,
    use_percentile: bool = True,
    qlo: float = 1.0,
    qhi: float = 99.0,
    morph_open: int = 3,
    morph_close: int = 5,
    min_component_area_px: int = 600
) -> Tuple[np.ndarray, float, Tuple[float,float]]:
    """
    在高度图上提取“最近表面”的薄层掩码（0/255）。

    返回：(mask_u8, z_ref, (low, high))
    """
    H, W = height.shape[:2]
    vm = (np.asarray(valid_mask).astype(np.uint8) > 0) & np.isfinite(height)
    if not np.any(vm):
        return np.zeros((H, W), np.uint8), float('nan'), (float('nan'), float('nan'))

    vals = height[vm]
    z_select = str(z_select).lower()

    # 参考 z_ref
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

    # 保留最大连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if num <= 1:
        return np.zeros((H, W), np.uint8), z_ref, (low, high)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas)) if areas.size > 0 else 0
    if stats[idx, cv2.CC_STAT_AREA] < max(1, int(min_component_area_px)):
        return np.zeros((H, W), np.uint8), z_ref, (low, high)
    keep = (labels == idx).astype(np.uint8) * 255
    return keep, z_ref, (low, high)


def extract_nearest_surface_mask_from_grid(
    P_mach_roi_hw3: np.ndarray,
    *,
    z_select: str = 'max',
    depth_margin_mm: float = 30.0,
    **kwargs
) -> Tuple[np.ndarray, float, Tuple[float,float]]:
    """
    兼容包装：输入 (H,W,3) 网格点云，直接在 Z 通道上做“最近表面”提取。
    """
    Z = P_mach_roi_hw3[:, :, 2].astype(np.float32)
    valid = np.isfinite(Z) & (Z != 0)
    return extract_nearest_surface_mask_from_height(
        Z, valid,
        z_select=z_select,
        depth_margin_mm=depth_margin_mm,
        **kwargs
    )


# ============== 主流程 ==============

def main():
    # 读取外参与 G 代码
    T_path = input('外参路径 (默认 {}): '.format(CONFIG['T_path'])).strip() or CONFIG['T_path']
    G_path = input('G代码路径 (默认 {}): '.format(CONFIG['gcode_path'])).strip() or CONFIG['gcode_path']
    if not Path(T_path).exists():
        raise SystemExit('外参文件不存在: {}'.format(T_path))
    R, t, _ = load_extrinsic(T_path)
    g_xy = parse_gcode_xy(G_path)

    # 相机
    stream = PCamMLSStream(); stream.open()
    out_dir = Path(CONFIG['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)

    roi_mode = str(CONFIG.get('roi_mode', 'none')).lower()
    frame = 0
    try:
        while True:
            P_cam, _ = stream.read_pointcloud(2000)
            if P_cam is None:
                print('[WARN] 无深度帧'); continue

            H, W, _ = P_cam.shape
            P_mach = transform_cam_to_machine_grid(P_cam, R, t)

            # ROI 选择
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

            # ===== 最近表面：在 height 上提取薄层掩码 =====
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

            # 可视化：顶视 + 最近表面叠加
            vis_top = render_topdown(height, mask_top, origin_xy, pix_mm,
                                     gcode_xy=g_xy, machine_roi_rect_px=None, roi_mode_text=roi_text)
            # 叠加薄层（绿色）
            overlay = vis_top.copy()
            gmask = cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR)
            gmask[:, :, 1] = np.maximum(gmask[:, :, 1], gmask[:, :, 0])  # 强化绿
            overlay = cv2.addWeighted(overlay, 1.0, gmask, 0.35, 0)

            # HUD
            txt = f'Nearest surface: z_ref={z_ref:.2f}  band=[{z_low:.2f},{z_high:.2f}] mm  pix={pix_mm:.2f}mm  grid={height.shape[1]}x{height.shape[0]}'
            overlay = draw_machine_axes_overlay(overlay, origin_xy, pix_mm)
            cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(overlay, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('Top-Down + Nearest Surface', overlay)

            # 仅掩码窗口
            cv2.imshow('NearestSurfaceMask', nearest_mask)

            # 交互
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                outp = out_dir / f'nearest_{frame:06d}.png'
                cv2.imwrite(str(outp), overlay); print('[SAVE]', outp); frame += 1
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
