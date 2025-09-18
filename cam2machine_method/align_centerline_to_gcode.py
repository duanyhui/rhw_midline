# -*- coding: utf-8 -*-
"""
1_align_centerline_to_gcode_pro.py

目标：
  输入：点云（相机系）、外参（相机->机床）、G 代码
  过程：投影顶视 -> 最近表面 -> G 代码引导中轴线
  输出：全局偏移 (δx, δy)【法向最小二乘】、(δx, δy, δθ)【2D 刚体配准】、
       按位法向误差 e_n 序列、修正后的 G 代码（可选导出）

要点：
  - 中轴线在 G 代码的法向上做一维扫描（距离变换取峰）
  - 无效点插值 + 限幅 + 平滑 + （必要时）骨架回退
  - “偏移量”用全局最小二乘求解，给出清晰可解释的 (δx, δy[, δθ])
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2

# --------------------- 基础 I/O 与几何（复用/简化） ---------------------

def load_extrinsic(T_path: str | Path):
    """读取 hand–eye 标定外参（calibrate_3d.py 的输出）"""
    data = np.load(str(T_path), allow_pickle=True).item()
    R = np.asarray(data['R'], float)
    t = np.asarray(data['t'], float).reshape(1, 3)
    T = np.asarray(data['T'], float)
    return R, t, T

def parse_gcode_xy(path: str | Path) -> np.ndarray:
    """解析 G0/G1 的 XY 序列（忽略注释、其它轴/指令）"""
    p = Path(path)
    if (not path) or (not p.exists()): return np.empty((0,2), float)
    pts = []
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        x = y = None
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith(';') or line.startswith('('): continue
            if ';' in line: line = line.split(';', 1)[0]
            # 去除括号注释
            while '(' in line and ')' in line:
                a = line.find('('); b = line.find(')')
                if a < 0 or b < 0 or b <= a: break
                line = (line[:a] + ' ' + line[b+1:]).strip()
            toks = line.split()
            if not toks: continue
            cmd = toks[0].upper()
            if cmd in ('G0','G00','G1','G01'):
                for u in toks[1:]:
                    U = u.upper()
                    if U.startswith('X'):
                        try: x = float(U[1:])
                        except: pass
                    elif U.startswith('Y'):
                        try: y = float(U[1:])
                        except: pass
                if x is not None and y is not None:
                    pts.append([x,y])
    return np.asarray(pts, float) if pts else np.empty((0,2), float)

def resample_polyline(poly: np.ndarray, step: float) -> np.ndarray:
    """按弧长等距重采样"""
    if poly.shape[0] < 2: return poly.copy()
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    L = float(seg.sum());
    if L <= 1e-9: return poly[[0]].copy()
    n = max(2, int(np.ceil(L / max(1e-6, step))))
    s = np.linspace(0.0, L, n)
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    out = []; j = 0
    for si in s:
        while j < len(seg) and si > cs[j+1]: j += 1
        if j >= len(seg): out.append(poly[-1]); continue
        t = (si - cs[j]) / max(seg[j], 1e-9)
        out.append(poly[j]*(1-t) + poly[j+1]*t)
    return np.asarray(out, float)

def tangent_normal(poly: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """单位切向/法向（右手旋转：N = [-Ty, Tx]）"""
    if poly.shape[0] < 2:
        return np.array([[1.0,0.0]]), np.array([[0.0,1.0]])
    d = np.gradient(poly, axis=0)
    T = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    N = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, N

def transform_cam_to_machine(P_cam: np.ndarray, R: np.ndarray, t_row: np.ndarray) -> np.ndarray:
    """
    相机系 -> 机床系。
    输入可为 HxWx3 或 Nx3（单位：mm）
    """
    P = P_cam.reshape(-1, 3).astype(np.float32)
    Pm = (R @ P.T).T + t_row
    return Pm.reshape(P_cam.shape)

# --------------------- 顶视投影与最近表面 ---------------------

@dataclass
class TopdownCfg:
    pixel_size_mm: float = 0.8
    bounds_qlo: float   = 1.0
    bounds_qhi: float   = 99.0
    bounds_margin_mm: float = 20.0
    max_grid_pixels: int = 1_200_000
    # 最近表面
    z_select: str       = 'max'    # 'max' or 'min'
    nearest_use_percentile: bool = True
    nearest_qlo: float  = 1.0
    nearest_qhi: float  = 99.0
    depth_margin_mm: float = 3.0
    morph_open: int     = 3
    morph_close: int    = 5
    min_component_area_px: int = 600

def _valid_mask(P_hw3: np.ndarray) -> np.ndarray:
    X,Y,Z = P_hw3[...,0], P_hw3[...,1], P_hw3[...,2]
    m = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & ((np.abs(X)+np.abs(Y)+np.abs(Z))>1e-6)
    return m

def _compute_bounds_xy(P_mach_hw3: np.ndarray, mask_hw: np.ndarray, qlo, qhi, margin):
    m = mask_hw.copy()
    if not np.any(m):
        m = _valid_mask(P_mach_hw3)
        if not np.any(m): return -100,100,-100,100
    X = P_mach_hw3[...,0][m]; Y = P_mach_hw3[...,1][m]
    x0, x1 = np.percentile(X, qlo), np.percentile(X, qhi)
    y0, y1 = np.percentile(Y, qlo), np.percentile(Y, qhi)
    return float(x0 - margin), float(x1 + margin), float(y0 - margin), float(y1 + margin)

def _adjust_pixel_size(x0,x1,y0,y1,pix_mm,max_pixels) -> float:
    W = max(2, int(round((x1-x0)/max(1e-6,pix_mm))))
    H = max(2, int(round((y1-y0)/max(1e-6,pix_mm))))
    while W*H > max_pixels:
        pix_mm *= 1.25
        W = max(2, int(round((x1-x0)/max(1e-6,pix_mm))))
        H = max(2, int(round((y1-y0)/max(1e-6,pix_mm))))
    return pix_mm

def project_topdown(P_mach_hw3: np.ndarray,
                    select_mask_hw: np.ndarray,
                    cfg: TopdownCfg):
    """把 3D 网格投成顶视 height 图（右手系 +Y 向上）；返回 height, mask, origin(x0,y0), pix_mm"""
    x0,x1,y0,y1 = _compute_bounds_xy(P_mach_hw3, select_mask_hw, cfg.bounds_qlo, cfg.bounds_qhi, cfg.bounds_margin_mm)
    pix_mm = _adjust_pixel_size(x0,x1,y0,y1, cfg.pixel_size_mm, cfg.max_grid_pixels)
    X,Y,Z = P_mach_hw3[...,0], P_mach_hw3[...,1], P_mach_hw3[...,2]
    m = select_mask_hw & _valid_mask(P_mach_hw3)
    Wg = int(max(2, round((x1-x0)/max(1e-6,pix_mm))))
    Hg = int(max(2, round((y1-y0)/max(1e-6,pix_mm))))
    height = np.full((Hg,Wg), np.nan, np.float32)
    count  = np.zeros((Hg,Wg), np.int32)
    if not np.any(m):
        return height, (count>0).astype(np.uint8)*255, (x0,y0), pix_mm
    Xs = X[m].astype(np.float32); Ys = Y[m].astype(np.float32); Zs = Z[m].astype(np.float32)
    ix = np.clip(((Xs - x0)/pix_mm).astype(np.int32), 0, Wg-1)
    iy = np.clip(((y1 - Ys)/pix_mm).astype(np.int32), 0, Hg-1) # +Y 向上
    for gx,gy,gz in zip(ix,iy,Zs):
        if (not np.isfinite(height[gy,gx])) or (gz > height[gy,gx]): height[gy,gx] = gz
        count[gy,gx] += 1
    mask = (count>0).astype(np.uint8)*255
    return height, mask, (x0,y0), pix_mm

def _morph_cleanup(mask_u8: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    m = (mask_u8>0).astype(np.uint8)*255
    if close_k and close_k>1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k,close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    if open_k and open_k>1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    return m

def extract_nearest_surface_mask(height: np.ndarray, valid_mask: np.ndarray, cfg: TopdownCfg):
    """从 height 里提“最近表面”薄层掩码 + 最大连通域"""
    H,W = height.shape
    vm = (valid_mask.astype(np.uint8)>0) & np.isfinite(height)
    if not np.any(vm):
        return np.zeros((H,W), np.uint8), np.nan, (np.nan,np.nan)
    vals = height[vm]
    if cfg.nearest_use_percentile:
        if cfg.z_select.lower().startswith('max'):
            z_ref = float(np.nanpercentile(vals, cfg.nearest_qhi))
            low, high = z_ref - float(cfg.depth_margin_mm), z_ref + 1e-6
        else:
            z_ref = float(np.nanpercentile(vals, cfg.nearest_qlo))
            low, high = z_ref, z_ref + float(cfg.depth_margin_mm)
    else:
        if cfg.z_select.lower().startswith('max'):
            z_ref = float(np.nanmax(vals)); low, high = z_ref - float(cfg.depth_margin_mm), z_ref + 1e-6
        else:
            z_ref = float(np.nanmin(vals)); low, high = z_ref, z_ref + float(cfg.depth_margin_mm)
    band = (height>=low)&(height<=high)&vm
    m = (band.astype(np.uint8))*255
    m = _morph_cleanup(m, cfg.morph_open, cfg.morph_close)
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m>0).astype(np.uint8), 8)
    if num<=1: return np.zeros((H,W),np.uint8), z_ref, (low,high)
    areas = stats[1:, cv2.CC_STAT_AREA]; idx = 1 + int(np.argmax(areas))
    if stats[idx, cv2.CC_STAT_AREA] < max(1, int(cfg.min_component_area_px)):
        return np.zeros((H,W),np.uint8), z_ref, (low,high)
    keep = (labels==idx).astype(np.uint8)*255
    return keep, z_ref, (low,high)

# --------------------- G 代码引导的中轴线（核心） ---------------------

@dataclass
class GuideCfg:
    step_mm: float = 1.0
    halfwidth_mm: float = 6.0
    use_dt: bool = True                 # True=距离变换峰值；False=边界中点
    min_on_count: int = 3
    smooth_win: int = 7
    max_offset_mm: float = 8.0
    min_valid_ratio: float = 0.35
    fallback_use_skeleton: bool = True

def _bilinear_at(img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    H,W = img.shape[:2]
    x = xy[:,0]; y = xy[:,1]
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.clip(x0+1, 0, W-1); y1 = np.clip(y0+1, 0, H-1)
    x0 = np.clip(x0,0,W-1); y0 = np.clip(y0,0,H-1)
    Ia = img[y0,x0]; Ib = img[y0,x1]; Ic = img[y1,x0]; Id = img[y1,x1]
    wa = (x1 - x) * (y1 - y); wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0); wd = (x - x0) * (y - y0)
    return Ia*wa + Ib*wb + Ic*wc + Id*wd

def _smooth_polyline(xy: np.ndarray, win: int) -> np.ndarray:
    if xy.shape[0] < 3 or win<=2 or win%2==0: return xy.copy()
    k = win//2; out = xy.copy()
    for i in range(len(xy)):
        a = max(0, i-k); b = min(len(xy), i+k+1)
        out[i] = xy[a:b].mean(axis=0)
    return out

def mach_xy_to_px_float(xy: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int) -> np.ndarray:
    x0,y0 = origin_xy; y1 = y0 + Himg * pix_mm
    xs = (xy[:,0]-x0)/pix_mm
    ys = (y1 - xy[:,1])/pix_mm
    return np.stack([xs,ys], axis=1)

def px_float_to_mach_xy(px: np.ndarray, origin_xy: Tuple[float,float], pix_mm: float, Himg: int) -> np.ndarray:
    x0,y0 = origin_xy; y1 = y0 + Himg * pix_mm
    X = x0 + px[:,0]*pix_mm
    Y = y1 - px[:,1]*pix_mm
    return np.stack([X,Y], axis=1)

def gcode_guided_centerline(nearest_mask: np.ndarray,
                            origin_xy: Tuple[float,float],
                            pix_mm: float,
                            gcode_xy: np.ndarray,
                            gcode_normals: np.ndarray,
                            guide: GuideCfg,
                            path_px_skeleton: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """按 G 代码法向逐点扫描，生成中轴线（机床系）"""
    H,W = nearest_mask.shape[:2]
    dt = cv2.distanceTransform((nearest_mask>0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
    g_px = mach_xy_to_px_float(gcode_xy, origin_xy, pix_mm, H)
    halfw_px = max(1.0, guide.halfwidth_mm/max(1e-9,pix_mm))
    out_px = np.full_like(g_px, np.nan)
    valid = 0
    # 简易骨架回退索引
    sk_tree = None
    if guide.fallback_use_skeleton and path_px_skeleton is not None and path_px_skeleton.size>0:
        try:
            from scipy.spatial import cKDTree
            sk_tree = cKDTree(path_px_skeleton)
        except Exception:
            sk_tree = None
    for i in range(len(g_px)):
        p = g_px[i]; n = gcode_normals[min(i, len(gcode_normals)-1)]
        n_px = np.array([n[0]/pix_mm, -n[1]/pix_mm], float); n_px /= (np.linalg.norm(n_px)+1e-12)
        p0 = p - n_px*halfw_px; p1 = p + n_px*halfw_px
        L = np.linalg.norm(p1-p0); M = max(3, int(np.ceil(L/1.0)))
        ts = np.linspace(0.0, 1.0, M)
        line_xy = (p0[None,:]*(1.0-ts[:,None]) + p1[None,:]*ts[:,None])
        mvals = _bilinear_at((nearest_mask>0).astype(np.float32), line_xy)
        cp = None
        if guide.use_dt:
            dvals = _bilinear_at(dt, line_xy); dvals[mvals<0.5] = -1.0
            k = int(np.argmax(dvals))
            if dvals[k] > 0: cp = line_xy[k]
        else:
            on_idx = np.where(mvals>=0.5)[0]
            if on_idx.size >= guide.min_on_count:
                j0, j1 = on_idx[0], on_idx[-1]; cp = (line_xy[j0]+line_xy[j1])*0.5
        if cp is None and sk_tree is not None:
            # 回退：就近骨架像素
            dist, idx = sk_tree.query(p[None,:])
            if np.isfinite(dist).all(): cp = path_px_skeleton[int(idx)]
        if cp is not None and np.isfinite(cp).all():
            out_px[i] = cp; valid += 1
    # 插值填补
    ok = np.isfinite(out_px[:,0]) & np.isfinite(out_px[:,1])
    if ok.any() and (~ok).any():
        idxs = np.arange(len(out_px))
        for d in (0,1):
            vals = out_px[ok,d]
            out_px[~ok,d] = np.interp(idxs[~ok], idxs[ok], vals)
    # 限幅
    dpx = out_px - g_px
    dmm = dpx * pix_mm
    r = np.linalg.norm(dmm, axis=1)
    too_far = r > guide.max_offset_mm
    if np.any(too_far):
        scale = (guide.max_offset_mm/(r[too_far]+1e-9))
        dmm[too_far] *= scale[:,None]
        out_px[too_far] = g_px[too_far] + dmm[too_far]/pix_mm
    # 平滑
    out_px = _smooth_polyline(out_px, guide.smooth_win)
    centerline_xy = px_float_to_mach_xy(out_px, origin_xy, pix_mm, H)
    stats = dict(valid=int(valid), total=int(len(out_px)), ratio=float(valid/max(1,len(out_px))))
    return centerline_xy, stats

# --------------------- 偏移解算（关键改动） ---------------------

def normals_from_poly(poly: np.ndarray) -> np.ndarray:
    T,N = tangent_normal(poly); return N

def project_normal_errors(centerline_xy: np.ndarray, gcode_xy: np.ndarray, N_ref: np.ndarray) -> np.ndarray:
    """在参考法向上计算按位法向误差 e_n[i]，两者长度应一致或近似一致"""
    m = min(len(centerline_xy), len(gcode_xy), len(N_ref))
    C = centerline_xy[:m]; G = gcode_xy[:m]; N = N_ref[:m]
    e = ((C - G) * N).sum(axis=1)
    return e, C, G, N

def solve_offset_normal_only(centerline_xy: np.ndarray, gcode_xy: np.ndarray, N_ref: np.ndarray,
                             weights: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Dict]:
    """
    法向最小二乘求全局平移 v=(δx,δy)：
       sum_i w_i * ((C_i - (G_i + v)) · N_i)^2 最小
    闭式解： (Σ w_i N_i N_i^T) v = Σ w_i e_i N_i
    """
    e, C, G, N = project_normal_errors(centerline_xy, gcode_xy, N_ref)
    if len(e) < 3:
        return np.zeros(2,float), dict(n=0, rms=0.0)
    if weights is None: w = np.ones(len(e), float)
    else: w = np.asarray(weights, float).reshape(-1)[:len(e)]
    A = np.zeros((2,2), float); b = np.zeros(2, float)
    for i in range(len(e)):
        Ni = N[i].reshape(2,1); wi = float(w[i]); ei = float(e[i])
        A += wi * (Ni @ Ni.T)
        b += wi * (ei * N[i])
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        v = np.linalg.pinv(A) @ b
    # 拟合质量：法向残差
    r = e - (N @ v)
    return v, dict(n=int(len(e)), rms=float(np.sqrt(np.mean(r**2))), mean=float(np.mean(r)),
                   p95=float(np.percentile(np.abs(r),95)))

def solve_offset_rigid_2d(centerline_xy: np.ndarray, gcode_xy: np.ndarray,
                          weights: Optional[np.ndarray]=None) -> Tuple[np.ndarray, float, Dict]:
    """
    2D 刚体配准（SVD）：求 R(θ), t 使 ||C - (R G + t)||^2 最小
    返回：t=(δx,δy), θ(rad)
    """
    m = min(len(centerline_xy), len(gcode_xy))
    if m < 2: return np.zeros(2,float), 0.0, dict(n=0, rms=0.0)
    C = centerline_xy[:m]; G = gcode_xy[:m]
    if weights is None: w = np.ones(m, float)
    else: w = np.asarray(weights, float).reshape(-1)[:m]
    W = w / (np.sum(w)+1e-12)
    cC = (W[:,None] * C).sum(0); cG = (W[:,None]*G).sum(0)
    C0 = C - cC; G0 = G - cG
    H = (W[:,None,None] * (G0[:,:,None] @ C0[:,None,:])).sum(0)   # 2x2
    U,S,Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = cC - (R @ cG)
    theta = float(np.arctan2(R[1,0], R[0,0]))
    # 质量
    pred = (G @ R.T) + t
    rms = float(np.sqrt(np.mean(np.sum((C - pred)**2, axis=1))))
    return t, theta, dict(n=int(m), rms=rms)

# --------------------- G 代码写回 ---------------------

def export_corrected_gcode(src_path: str | Path,
                           dst_path: str | Path,
                           offset_xy: Tuple[float,float],
                           theta_rad: float = 0.0,
                           center: Optional[Tuple[float,float]] = None):
    """
    把 (δx,δy,δθ) 应用于 G0/G1 中的 X/Y 坐标并写出新文件。
    旋转参考点默认为原 G 代码的质心（若给 center 则绕其旋转）。
    """
    pts = parse_gcode_xy(src_path)
    if pts.size == 0:
        # 直接拷贝
        Path(dst_path).write_text(Path(src_path).read_text(encoding='utf-8'), encoding='utf-8')
        return
    G = pts.copy()
    if center is None:
        c = G.mean(0)
    else:
        c = np.asarray(center, float).reshape(2)
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad),  np.cos(theta_rad)]], float)
    G_rot = ((G - c) @ R.T) + c
    G_new = G_rot + np.asarray(offset_xy, float).reshape(1,2)
    # 逐行重写
    out_lines = []
    with open(src_path, 'r', encoding='utf-8', errors='ignore') as f:
        idx = 0
        x_cur = None; y_cur = None
        for raw in f:
            line = raw.rstrip('\n')
            if idx < len(pts):
                # 查该行是否是走刀，若包含 X/Y 则替换
                toks = line.strip().split()
                if toks and toks[0].upper() in ('G0','G00','G1','G01'):
                    # 只在原行“出现 X/Y”的时候替换，避免改伤其它行
                    hasX = any(u.upper().startswith('X') for u in toks[1:])
                    hasY = any(u.upper().startswith('Y') for u in toks[1:])
                    if hasX or hasY:
                        gx, gy = G_new[idx]
                        new_toks = [toks[0]]
                        for u in toks[1:]:
                            U = u.upper()
                            if U.startswith('X'):
                                new_toks.append(f"X{gx:.4f}")
                                x_cur = gx
                            elif U.startswith('Y'):
                                new_toks.append(f"Y{gy:.4f}")
                                y_cur = gy
                            else:
                                new_toks.append(u)
                        line = ' '.join(new_toks)
                        idx += 1
            out_lines.append(line)
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

# --------------------- 顶层接口 ---------------------

@dataclass
class Options:
    # 顶视
    topdown: TopdownCfg = TopdownCfg()
    # 引导
    guide: GuideCfg = GuideCfg()
    # ROI 选择（可选）：机床系下正方形 ROI
    roi_center_xy: Tuple[float,float] = (0.0, 0.0)
    roi_size_mm: Optional[float] = None    # None 表示无 ROI；否则裁剪到正方形
    # 导出
    save_debug_png: Optional[str] = None
    corrected_gcode_out: Optional[str] = None
    # 重采样步长（mm）
    gcode_step_mm: float = 1.0

def compute_offsets(P_cam_hw3: np.ndarray,
                    T_extrinsic_path: str | Path,
                    gcode_path: str | Path,
                    options: Options = Options()) -> Dict:
    """
    入口函数：点云（相机系）+ 外参 + G 代码 → 偏移量与可选产物
    """
    # 1) 外参与 G 代码
    R, t_row, _ = load_extrinsic(T_extrinsic_path)   # 来自 calibrate_3d.py 的输出（正确使用）  # 参见：:contentReference[oaicite:3]{index=3}
    G_raw = parse_gcode_xy(gcode_path)
    if G_raw.size == 0:
        raise RuntimeError("G 代码中未解析到任何 XY 点。")
    G = resample_polyline(G_raw, max(0.2, float(options.gcode_step_mm)))
    T_ref, N_ref = tangent_normal(G)

    # 2) 点云 → 机床系
    P_mach = transform_cam_to_machine(P_cam_hw3, R, t_row)

    # 3) ROI（机床系）
    m_valid = _valid_mask(P_mach)
    if options.roi_size_mm is not None:
        cx, cy = options.roi_center_xy; half = options.roi_size_mm*0.5
        X,Y = P_mach[...,0], P_mach[...,1]
        m_roi = (X>=cx-half)&(X<=cx+half)&(Y>=cy-half)&(Y<=cy+half)
        m_select = m_valid & m_roi
    else:
        m_select = m_valid

    # 4) 顶视投影 & 最近表面
    height, mask_top, origin_xy, pix_mm = project_topdown(P_mach, m_select, options.topdown)
    nearest_mask, z_ref, (z_low, z_high) = extract_nearest_surface_mask(height, (mask_top>0), options.topdown)

    # 5) 骨架（可选，作为失败回退；若无 skimage 也可以跳过）
    path_px_skeleton = None
    try:
        from skimage.morphology import skeletonize
        if np.count_nonzero(nearest_mask)>0:
            bw = (nearest_mask>0).astype(np.uint8)*255
            sk = skeletonize(bw>0).astype(np.uint8)*255
            ys,xs = np.where(sk>0)
            path_px_skeleton = np.stack([xs,ys], axis=1).astype(np.float32) if len(xs)>0 else None
    except Exception:
        pass

    # 6) 引导中轴线（机床系）
    C_xy, rep = gcode_guided_centerline(nearest_mask, origin_xy, pix_mm, G, N_ref, options.guide, path_px_skeleton)
    if rep['ratio'] < options.guide.min_valid_ratio or C_xy.size == 0:
        raise RuntimeError("中轴线引导失败（掩码过少/断裂过多）。")

    # 7) 偏移解算（两条口径）
    v_normal, q_normal = solve_offset_normal_only(C_xy, G, N_ref, None)
    t_rigid, theta_rigid, q_rigid = solve_offset_rigid_2d(C_xy, G, None)

    # 8) 导出修正 G 代码（可选）
    out_gcode_path = None
    if options.corrected_gcode_out:
        out_gcode_path = options.corrected_gcode_out
        export_corrected_gcode(gcode_path, out_gcode_path,
                               offset_xy=(t_rigid[0], t_rigid[1]), theta_rad=theta_rigid)

    # 9) 可选可视化保存
    dbg = None
    if options.save_debug_png:
        vis = _render_debug(height, mask_top, origin_xy, pix_mm, G, C_xy, N_ref)
        cv2.imwrite(options.save_debug_png, vis)
        dbg = options.save_debug_png

    # 10) 统计/返回
    e_n, Cc, Gc, Nc = project_normal_errors(C_xy, G, N_ref)
    return dict(
        pixel_size_mm=float(pix_mm),
        origin_xy=(float(origin_xy[0]), float(origin_xy[1])),
        centerline_xy=C_xy,
        gcode_xy=G,
        en_profile=e_n,
        quality=dict(nearest_ratio=rep['ratio'],
                     normal=q_normal, rigid=q_rigid),
        offset_normal=(float(v_normal[0]), float(v_normal[1])),
        offset_rigid=dict(dx=float(t_rigid[0]), dy=float(t_rigid[1]), dtheta_deg=float(np.degrees(theta_rigid))),
        corrected_gcode_path=out_gcode_path,
        debug_png_path=dbg
    )

def _render_debug(height: np.ndarray, mask_top: np.ndarray, origin_xy, pix_mm, g_xy, c_xy, N_ref):
    """简单叠加可视化：height 伪彩 + G 代码 + 中轴线 + 偏差箭头"""
    H,W = height.shape
    if np.isfinite(height).any():
        vmin = float(np.nanpercentile(height, 5)); vmax = float(np.nanpercentile(height,95))
        gray = np.clip((height - vmin)/(max(1e-6, vmax-vmin)), 0,1)
        vis = cv2.applyColorMap((gray*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    else:
        vis = np.zeros((H,W,3), np.uint8)
    vis = cv2.addWeighted(vis, 0.9, np.dstack([mask_top]*3), 0.1, 0)

    def xy_to_px(xy):
        x0,y0 = origin_xy; y1 = y0 + H*pix_mm
        xs = np.clip(((xy[:,0]-x0)/pix_mm).astype(int), 0, W-1)
        ys = np.clip(((y1 - xy[:,1])/pix_mm).astype(int), 0, H-1)
        return np.stack([xs,ys], 1)

    if g_xy is not None and len(g_xy)>1:
        p = xy_to_px(g_xy)
        for i in range(len(p)-1):
            cv2.line(vis, tuple(p[i]), tuple(p[i+1]), (240,240,240), 1, cv2.LINE_AA)
    if c_xy is not None and len(c_xy)>1:
        q = xy_to_px(c_xy)
        for i in range(len(q)-1):
            cv2.line(vis, tuple(q[i]), tuple(q[i+1]), (200,255,255), 2, cv2.LINE_AA)
    # 画部分法向偏差箭头
    e_n, Cc, Gc, Nc = project_normal_errors(c_xy, g_xy, N_ref)
    step = max(1, len(e_n)//60)
    P = xy_to_px(Cc)
    for k in range(0, len(e_n), step):
        n = Nc[k]; base = Cc[k]; tip = base + n*e_n[k]
        b = xy_to_px(base[None,:])[0]; t = xy_to_px(tip[None,:])[0]
        col = (0,255,0) if abs(e_n[k])<0.5 else (0,165,255)
        cv2.arrowedLine(vis, tuple(b), tuple(t), col, 2, cv2.LINE_AA, tipLength=0.35)
    return vis

# --------------------- CLI 示例 ---------------------
if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument('--pcam', type=str, required=True, help='相机点云 .npy（HxWx3 或 Nx3，单位 mm）')
    ap.add_argument('--textr', type=str, required=True, help='外参 .npy (calibrate_3d.py 生成)')
    ap.add_argument('--gcode', type=str, required=True, help='G 代码文件')
    ap.add_argument('--outpng', type=str, default=None, help='保存调试可视化 PNG')
    ap.add_argument('--outgcode', type=str, default=None, help='导出修正后的 G 代码')
    ap.add_argument('--step', type=float, default=1.0, help='G 代码重采样步长 (mm)')
    ap.add_argument('--roi', type=float, default=None, help='ROI 正方形边长 (mm)，缺省为全帧')
    args = ap.parse_args()

    P_cam = np.load(args.pcam).astype(float)
    opt = Options(
        topdown=TopdownCfg(),
        guide=GuideCfg(),
        roi_size_mm=args.roi if args.roi is not None else None,
        save_debug_png=args.outpng,
        corrected_gcode_out=args.outgcode,
        gcode_step_mm=args.step
    )
    try:
        rep = compute_offsets(P_cam, args.textr, args.gcode, opt)
    except Exception as e:
        print('[ERROR]', e)
        sys.exit(2)

    print('\n=== 偏移解（法向最小二乘） ===')
    print('δx={:+.3f}  δy={:+.3f}  [mm]'.format(*rep['offset_normal']))
    print('质量：', rep['quality']['normal'])
    print('\n=== 偏移解（2D 刚体配准） ===')
    print('δx={:+.3f}  δy={:+.3f}  δθ={:+.3f} deg'.format(rep['offset_rigid']['dx'],
                                                       rep['offset_rigid']['dy'],
                                                       rep['offset_rigid']['dtheta_deg']))
    print('质量：', rep['quality']['rigid'])
    if rep['corrected_gcode_path']:
        print('已导出修正 G 代码：', rep['corrected_gcode_path'])
    if rep['debug_png_path']:
        print('已保存调试图：', rep['debug_png_path'])
