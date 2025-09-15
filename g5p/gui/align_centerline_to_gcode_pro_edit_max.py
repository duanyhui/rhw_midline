#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centerline ↔ G-code (Guided Fit) — STRICT One-to-One Edition
------------------------------------------------------------
重构要点（与原版相比的实质改动）：
- [CHG] 严格一一对应：以 G 代码等弧长采样点 X(s_i) 为唯一主序列，逐点沿法线搜索对应点；
       导出阶段不再引入 KDTree 聚合或骨架回退结果，彻底消除“跨段映射”。
- [CHG] 缺失策略：仅对“短缺口”做局部插值（≤ max_gap_pts）；遇到“长段缺失”直接判 FAIL。
- [NEW] 梯度门限：|Δδ/Δs| ≤ guide_max_grad_mm_per_mm，平滑后再做梯度限幅，抑制抖动。
- [NEW] 可选曲率自适应平滑窗口（高曲率区域缩短窗口减少拖尾）。
- [CHG] Guard 扩展：新增 long_missing_max_mm / grad_limit 两项，严格把关导出。
- 其余：保留你的平面展平、最近表层提取、等距带 φ≈0 搜索、直方图/quicklook/报告等。

右手系：+X → 右，+Y → 上。
"""
import re
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict
import numpy as np
import cv2
import math
import collections
import time
import os
import json

# ===================== 参数（含新增） =====================
PARAMS = dict(
    # 文件
    T_path='T_cam2machine.npy',               # 相机->机床外参 (含 R,t,T)
    gcode_path='args/example.gcode',          # 理论 G 代码路径

    # 顶视投影（右手系：+X 向右，+Y 向上）
    pixel_size_mm=0.8,
    bounds_qlo=1.0, bounds_qhi=99.0,
    bounds_margin_mm=20.0,
    max_grid_pixels=1_200_000,

    # ROI 选择：'none' / 'camera_rect' / 'machine' / 'gcode_bounds'
    roi_mode='gcode_bounds',
    cam_roi_xywh=(682, 847, 228, 185),
    roi_center_xy=(50.0,50.0),
    roi_size_mm=550.0,

    # 最近表面提取（在“展平后的残差高度图”上进行）
    z_select='max',
    nearest_use_percentile=True,
    nearest_qlo=1.0, nearest_qhi=99.0,
    depth_margin_mm=3.0,
    morph_open=3, morph_close=5,
    min_component_area_px=600,

    # 平面拟合/展平
    plane_enable=True,
    plane_ransac_thresh_mm=0.8,
    plane_ransac_iters=500,
    plane_sample_cap=120000,
    plane_min_inlier_ratio=0.55,              # Guard 指标之一

    # 骨架/折线（仅用于可视化，不参与导出）
    rdp_epsilon_px=3,
    show_skeleton_dilate=True,
    resample_step_px=1.0,

    # === G 代码引导中轴线（核心） ===
    guide_enable=True,
    guide_step_mm=1.0,                        # 等弧长重采样步长（G0/G1/G2/G3）
    guide_halfwidth_mm=6.0,                   # 法向扫描半宽（±）
    guide_use_dt=True,
    guide_min_on_count=3,
    guide_smooth_win=7,
    guide_max_offset_mm=8.0,

    # === Corner ignoring（拐角点附近忽略取点） ===  # NEW
    corner_ignore_enable=False,  # 开启/关闭拐角忽略
    corner_angle_thr_deg=35.0,  # 拐角判定阈值（夹角≥该角度判为拐角）
    corner_ignore_span_mm=2.0,  # 在拐角两侧各忽略的弧长半径（mm）

    # === 遮挡（固定位置设备） ===  # [OCCLUSION]
    occlusion=dict(
        enable=True,  # 设为 True 启用
        # 在机床坐标系 XY(mm) 中给出一个或多个多边形（顺/逆时针均可）
        # 示例：左下角 62x52mm 矩形（请按现场一次性标定后填写）
        polys=[
            [(-50,-50), (30,-30), (30,70), (-50,70)]
        ],
        dilate_mm=3.0,  # 安全扩张，确保完全覆盖遮挡
        synthesize_band=True,  # 是否在遮挡区内按 G 代码合成环带掩码
        band_halfwidth_mm=None  # None=自动从可见区估计；或手工指定半宽
    ),

    # [NEW] 严格一一对应与缺失处理
    strict_one_to_one=True,                   # 强制一一对应（导出仅基于法向单点决策）
    max_gap_pts=5,                            # 仅对 ≤ 该点数的缺口做局部插值
    long_missing_max_mm=20.0,                 # 长段缺失阈值（mm）；超过则 Guard FAIL
    long_missing_max_ratio=0.08,              # 或者按比例限制（相对总弧长）

    # [NEW] 梯度门限（抑制抖动/跳变）
    guide_max_grad_mm_per_mm=0.08,            # |Δδ| ≤ gmax * Δs（默认 0.08 mm/mm）
    curvature_adaptive=True,                  # 高曲率区域缩短平滑窗口
    curvature_gamma=35.0,                     # 窗口缩放强度（越大越敏感）
    min_smooth_win=3,                         # 自适应窗口下限

    guide_min_valid_ratio=0.60,               # 最低命中率（法向找到解的比例）
    guide_fallback_to_skeleton=True,          # 仅用于可视化（导出不使用回退）

    # === Debug：法线-交点最小可视化窗口 ===  # NEW
    debug_normals_window=True,  # 是否开启“法线-交点”调试窗口
    debug_normals_stride=25,  # 抽样步长：每隔多少个G点画一根法线
    debug_normals_max=40,  # 最多显示多少根法线（避免太密）
    debug_normals_len_mm=None,  # 法线长度（mm），None=自动用1.2×guide_halfwidth_mm
    debug_normals_text=True,  # 是否在交点旁标注 delta_n（mm）

    # 偏差可视化
    arrow_stride=12,
    draw_normal_probes=True,                  # 显示法向采样线（调试）

    # 纠偏（EMA 在线预估，保留）
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
    # 纠偏应用策略：'invert'（朝向理论轨迹纠偏）或 'follow'（直接跟随测得中心线）
    offset_apply_mode='invert',

    centerline_gcode='out/centerline.gcode',  # 可选：导出“测得中心线”（仅用于对比）
    export_centerline=False,
    auto_flip_offset=True,                    # 自动检测符号并翻转
    # === 偏差补偿（可选） ===
    bias_comp=dict(
        enable=True,  # True 时启用补偿
        path='bias_comp.json'  # 标定文件路径（支持 mode="vector"/"per_index"）
    ),
    preview_corrected=True,                   # 预览叠加 corrected 轨迹
    save_corrected_preview=True,

    # 质量门槛（Guard）
    Guard=dict(
        enable=True,
        min_valid_ratio=0.60,                 # 命中率
        max_abs_p95_mm=8.80,                  # 偏移绝对值 P95 上限
        min_plane_inlier_ratio=0.25,          # 平面内点率下限
        long_missing_max_mm=20.0,             # [NEW] 长段缺失 mm 上限
        grad_max_mm_per_mm=0.08               # [NEW] 导出前复核梯度上限
    ),

    # 可视化/报告
    colormap=getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET),
    dump_quicklook=True,
    dump_report=True,
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

# ======================= 基础 IO/几何（保留/增强） =======================
def load_extrinsic(T_path: Union[str, Path]):
    data = np.load(str(T_path), allow_pickle=True).item()
    R = np.asarray(data['R'], dtype=float)
    t = np.asarray(data['t'], dtype=float).reshape(1, 3)
    T = np.asarray(data['T'], dtype=float)
    return R, t, T

# ---- G-code 解析（支持 G0/G1 + G2/G3 圆弧） ----
def _interp_arc_xy(xy0, xy1, ij=None, R=None, cw=True, step=1.0):
    # 仅处理平面 XY 圆弧。优先 IJ 圆心；否则用 R。
    p0 = np.array(xy0, float); p1 = np.array(xy1, float)
    if ij is not None:
        c = p0 + np.array(ij, float)
    else:
        chord = p1 - p0; L = np.linalg.norm(chord)
        if L < 1e-9 or R is None:
            return np.vstack([p0, p1])
        h = math.sqrt(max(R*R - (L*0.5)**2, 0.0))
        mid = (p0 + p1) * 0.5
        n = np.array([-chord[1], chord[0]]) / (L + 1e-12)
        c1 = mid + n * h
        c2 = mid - n * h
        def ang(p): return math.atan2(p[1]-c[1], p[0]-c[0])
        for cand in (c1, c2):
            c = cand
            a0, a1 = ang(p0), ang(p1)
            da = (a1 - a0)
            if cw and da > 0: da -= 2*math.pi
            if (not cw) and da < 0: da += 2*math.pi
            arc_len = abs(da) * R
            if arc_len > 1e-3: break
    def ang(p): return math.atan2(p[1]-c[1], p[0]-c[0])
    a0, a1 = ang(p0), ang(p1)
    da = (a1 - a0)
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
            toks = re.findall(r'[A-Za-z]+-?\d+(?:\.\d+)?', line)

            if not toks: continue
            cmd = toks[0].upper()
            for u in toks[1:]:
                U = u.upper()
                if U.startswith('F'):
                    try: feed = float(U[1:])
                    except: pass
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
                else:
                    if len(pts)==0 or not np.allclose([x,y], pts[-1]):
                        pts.append([x,y])
                cur['X'], cur['Y'] = x, y
    P = np.asarray(pts, float) if pts else np.empty((0,2), float)
    return P, feed

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

def arc_length_s(poly: np.ndarray) -> np.ndarray:
    """等弧长参数（累计弧长）"""
    if poly.shape[0] < 2: return np.zeros((poly.shape[0],), float)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s

def tangent_normal(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if poly.shape[0] < 2:
        return np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    d = np.gradient(poly, axis=0)
    T = d / np.maximum(1e-9, np.linalg.norm(d, axis=1, keepdims=True))
    N = np.stack([-T[:,1], T[:,0]], axis=1)
    return T, N

def curvature_kappa(poly: np.ndarray) -> np.ndarray:
    """简易曲率估计：||ΔT|| / Δs"""
    if poly.shape[0] < 3: return np.zeros((poly.shape[0],), float)
    s = arc_length_s(poly)
    T, _ = tangent_normal(poly)
    kappa = np.zeros((len(poly),), float)
    for i in range(1, len(poly)-1):
        dT = T[i+1] - T[i-1]
        ds = max(1e-9, s[i+1] - s[i-1])
        kappa[i] = float(np.linalg.norm(dT) / ds)
    kappa[0] = kappa[1]; kappa[-1] = kappa[-2]
    return kappa

# === [NEW] per-index 偏差重用：拐角锁定的分段重采样 ===
def _detect_corner_knots_from_tangent(T_ref, ang_thresh_deg=25.0, min_gap_pts=3):
    """
    根据相邻切向夹角检测拐角，返回分段端点索引列表（含首尾）。
    """
    import numpy as _np
    t = _np.asarray(T_ref, _np.float32)
    n = t.shape[0]
    if n < 3:
        return [0, max(0, n-1)]
    cosang = _np.clip((t[:-1] * t[1:]).sum(axis=1), -1.0, 1.0)
    ang = _np.degrees(_np.arccos(cosang))
    idx = _np.where(ang >= float(ang_thresh_deg))[0] + 1  # 顶点落在后一点
    # 去除过近的拐角
    keep, last = [], -10**9
    for i in idx.tolist():
        if i - last >= int(min_gap_pts):
            keep.append(int(i)); last = int(i)
    knots = [0] + keep + [n-1]
    return sorted(set(knots))

def _resample_per_index_bias_for_length(delta_bias_mm, n_out):
    """
    全局归一化弧长的 1D 线性插值（仅作为兜底，不跨段混合）。
    """
    import numpy as _np
    b = _np.asarray(delta_bias_mm, _np.float32).ravel()
    n_in = int(b.size)
    if n_in <= 1 or n_out <= 0:
        return _np.zeros((max(0, n_out),), _np.float32)
    if n_in == n_out:
        return b.copy()
    t_in  = _np.linspace(0.0, 1.0, n_in,  dtype=_np.float32)
    t_out = _np.linspace(0.0, 1.0, n_out, dtype=_np.float32)
    return _np.interp(t_out, t_in, b).astype(_np.float32)

def _remap_knots_index(knots_src, n_src, n_dst):
    """
    把源端点索引（基于 n_src）按比例映射到目标长度 n_dst 的索引。
    """
    import numpy as _np
    if n_src <= 1 or n_dst <= 0:
        return [0, max(0, n_dst-1)]
    out = []
    for k in knots_src:
        v = int(round(k * (n_dst - 1) / max(1, n_src - 1)))
        out.append(int(_np.clip(v, 0, n_dst - 1)))
    # 保证首尾存在
    if out[0] != 0:
        out = [0] + out
    if out[-1] != (n_dst - 1):
        out = out + [n_dst - 1]
    return sorted(set(out))

def _resample_per_index_bias_piecewise(arr_in, n_out, knots_in, knots_out):
    """
    在给定输入/输出分段端点（含首尾，等长）约束下逐段线性插值，段间绝不混合。
    """
    import numpy as _np
    b = _np.asarray(arr_in, _np.float32).ravel()
    n_in = b.size
    out = _np.zeros((int(n_out),), _np.float32)
    if n_in <= 1 or n_out <= 0 or len(knots_in) < 2 or len(knots_in) != len(knots_out):
        return out
    K = len(knots_in) - 1
    for j in range(K):
        a0, a1 = int(knots_in[j]),  int(knots_in[j+1])
        b0, b1 = int(knots_out[j]), int(knots_out[j+1])
        seg_in = b[max(0, a0):min(n_in-1, a1)+1]
        m_in   = seg_in.size
        m_out  = max(0, b1 - b0 + 1)
        if m_in <= 1 or m_out <= 0:
            continue
        t_in  = _np.linspace(0.0, 1.0, m_in,  dtype=_np.float32)
        t_out = _np.linspace(0.0, 1.0, m_out, dtype=_np.float32)
        out[b0:b1+1] = _np.interp(t_out, t_in, seg_in).astype(_np.float32)
    return out


def compute_corner_ignore_mask(poly: np.ndarray,
                               angle_thr_deg: float,
                               span_mm: float,
                               step_mm: float) -> np.ndarray:
    """
    根据折线夹角阈值 + 弧长半径生成忽略取点的掩码。
    - 角度：对相邻两段 v[i-1], v[i] 计算转角，角度≥阈值视为拐角（索引为顶点 i）。
    - 半径：在每个拐角左右各扩展 span_mm 的等弧长范围（按 step_mm 转为索引）。
    - 两端点也视作拐角以保障稳健。
    返回：bool 掩码，True 表示“忽略取点/不参与统计”。
    """
    n = int(poly.shape[0])
    mask = np.zeros(n, dtype=np.bool_)
    if n < 3:
        mask[[0, n-1]] = True
        return mask
    v = np.diff(poly, axis=0)
    ln = np.linalg.norm(v, axis=1) + 1e-12
    cosang = np.sum(v[:-1] * v[1:], axis=1) / (ln[:-1] * ln[1:])
    cosang = np.clip(cosang, -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cosang))             # 0=直线, 值越大越“拐”
    thr = float(angle_thr_deg)
    corner_idx = np.where(ang_deg >= thr)[0] + 1        # 顶点索引落在中间点 i
    # 端点也算拐角
    corner_idx = np.unique(np.concatenate([corner_idx, [0, n-1]]))
    k = max(1, int(round(float(span_mm) / max(1e-9, float(step_mm)))))
    for c in corner_idx:
        l = max(0, c - k); r = min(n, c + k + 1)
        mask[l:r] = True
    return mask


def build_kdtree(pts: np.ndarray):
    """保留（在线 EMA 可视化时仍会用到最近邻查询）"""
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
def render_normals_probe_window(base_img: np.ndarray,
                                origin_xy: Tuple[float,float],
                                pix_mm: float,
                                gcode_xy: np.ndarray,
                                N_ref: np.ndarray,
                                delta_n: Optional[np.ndarray],
                                valid_mask: Optional[np.ndarray],
                                *,
                                stride: int = 25,
                                max_count: int = 40,
                                len_mm: Optional[float] = None,
                                draw_text: bool = True) -> np.ndarray:
    """
    在一张底图上，稀疏地画出：
      - 理论中轴线上的少量采样点（白点）
      - 对应点的法线（浅蓝色线段）
      - 法线与“实际中轴线（沿法向单点决策得到）”的交点（品红圆点）
    说明：
      - 交点位置 = gcode_xy[i] + N_ref[i] * delta_n[i]
      - 仅在 valid_mask[i] 为 True 且 delta_n[i] 有效时绘制交点
      - 为保持一致的像素映射，这里与主渲染一致：右手系 +Y 向上
    """
    vis = base_img.copy()
    H, W = vis.shape[:2]

    # 像素变换（与其它可视化函数保持一致）
    def xy_to_px(xy):
        x0, y0 = origin_xy
        y1 = y0 + H * pix_mm
        xs = np.clip(((xy[:, 0] - x0) / pix_mm).astype(int), 0, W - 1)
        ys = np.clip(((y1 - xy[:, 1]) / pix_mm).astype(int), 0, H - 1)
        return np.stack([xs, ys], axis=1)

    if gcode_xy.size == 0:
        return vis

    n_pts = len(gcode_xy)
    stride = max(1, int(stride))
    cand_idx = list(range(0, n_pts, stride))
    if max_count is not None and len(cand_idx) > int(max_count):
        # 均匀抽取 max_count 个索引
        idx_uniform = np.linspace(0, len(cand_idx) - 1, int(max_count)).round().astype(int)
        cand_idx = [cand_idx[i] for i in idx_uniform]

    # 法线长度（像素）
    if len_mm is None:
        len_mm = float(PARAMS.get('guide_halfwidth_mm', 6.0)) * 1.2
    L_px = int(max(4, round(len_mm / max(1e-9, pix_mm))))

    has_delta = (delta_n is not None and len(delta_n) == n_pts)
    has_valid = (valid_mask is not None and len(valid_mask) == n_pts)

    for i in cand_idx:
        base_xy = gcode_xy[i]
        base_px = xy_to_px(base_xy[None, :])[0]

        # 画理论点（白点）
        cv2.circle(vis, tuple(base_px), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 画法线（浅蓝）：以理论点为中心，延伸 ±L_px
        n = N_ref[min(i, len(N_ref) - 1)]
        # 注意像素坐标系与右手系的Y方向相反，这里按既有约定转换
        n_px = np.array([n[0], -n[1]], dtype=np.float32)
        n_px /= (np.linalg.norm(n_px) + 1e-12)
        p0 = (base_px - (n_px * L_px)).astype(int)
        p1 = (base_px + (n_px * L_px)).astype(int)
        cv2.line(vis, tuple(p0), tuple(p1), (180, 200, 255), 1, cv2.LINE_AA)

        # 画交点（品红）：仅在有效且有 δ 值时
        if has_delta and has_valid and bool(valid_mask[i]) and np.isfinite(delta_n[i]):
            q_xy = base_xy + n * float(delta_n[i])
            q_px = xy_to_px(q_xy[None, :])[0]
            cv2.circle(vis, tuple(q_px), 3, (255, 0, 255), -1, cv2.LINE_AA)
            if draw_text:
                txt = f"{float(delta_n[i]):+.2f}mm"
                # 文本稍微偏移，避免遮住点
                tpos = (int(q_px[0] + 6), int(q_px[1] - 6))
                cv2.putText(vis, txt, tpos, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)

    # 角落图例
    # cv2.rectangle(vis, (8, 8), (216, 56), (0, 0, 0), -1)
    # cv2.putText(vis, "white: G-code point", (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(vis, "cyan: normal",        (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 255), 1, cv2.LINE_AA)
    # cv2.putText(vis, "magenta: intersection",(14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

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

# ======================= 骨架提取（沿用你的通用实现；仅用于可视化） =======================
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
    """仅用于可视化与调试；导出不使用骨架结果。"""
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

    # 有内环：构建等距带
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

    return cv2.cvtColor(skel_u8, cv2.COLOR_BGR2RGB)

# =================== 像素/机床坐标互换 ===================
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

# =================== 偏差/EMA（保留，用于在线观测） ===================
def project_points_to_path(points_xy: np.ndarray, ref_xy: np.ndarray, ref_tree, ref_normals: np.ndarray):
    """保留：在线 EMA 调试用；导出不使用此聚合。"""
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
        self.a = float(alpha); self.dead = float(deadband)
        self.clip = float(clip_mm); self.step = float(max_step_mm)
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

# ===================== 可视化偏差叠加（支持有效掩码） =====================
def draw_deviation_overlay(vis_top: np.ndarray,
                           origin_xy: Tuple[float,float], pix_mm: float,
                           gcode_xy: Optional[np.ndarray],
                           centerline_xy: Optional[np.ndarray],
                           e_idx: np.ndarray, e_n: np.ndarray,
                           arrow_stride: int = 10,
                           draw_probes: bool = False, N_ref: Optional[np.ndarray] = None,
                           valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    [CHG] 新增 valid_mask：仅对有效点绘制箭头。
    """
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
            if valid_mask is not None and not bool(valid_mask[i]):   # 仅绘制有效点
                continue
            n = N[i]
            base = gcode_xy[i]  # 以 G 代码点为基准
            tip  = base + n * e_n[i]
            b = xy_to_px(base[None,:])[0]
            t = xy_to_px(tip[None,:])[0]
            col = (0,255,0) if abs(e_n[i]) < 0.5 else (0,165,255)
            cv2.arrowedLine(out,  tuple(b), tuple(t), col, 2, cv2.LINE_AA, tipLength=0.35)
            if draw_probes and N_ref is not None:
                nref = N_ref[min(i, len(N_ref)-1)]
                L = int( PARAMS['guide_halfwidth_mm'] / max(1e-9, pix_mm) )
                gp0 = b - (nref * L)[::-1].astype(int)
                gp1 = b + (nref * L)[::-1].astype(int)
                cv2.line(out, tuple(gp0), tuple(gp1), (180,180,255), 1, cv2.LINE_AA)
    return out

# ===================== 工具：插补/缺失/梯度限幅 =====================
def find_missing_runs(valid_mask: np.ndarray) -> List[Tuple[int,int]]:
    """找出 ~valid 的连续区间 [l, r]（闭区间）"""
    runs = []
    v = np.asarray(valid_mask, bool)
    n = len(v)
    i = 0
    while i < n:
        if v[i]:
            i += 1; continue
        l = i
        while i+1 < n and not v[i+1]:
            i += 1
        r = i
        runs.append((l, r))
        i += 1
    return runs

def local_interpolate_short_gaps(x: np.ndarray, valid_mask: np.ndarray, max_gap_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    仅对长度 ≤ max_gap_pts 的缺口做线性插值；其余保持 NaN。
    返回：y, valid_new
    """
    y = x.copy()
    v = valid_mask.copy()
    runs = find_missing_runs(v)
    for l, r in runs:
        gap = r - l + 1
        if gap <= max_gap_pts:
            a = l-1
            b = r+1
            if a >= 0 and b < len(x) and np.isfinite(x[a]) and np.isfinite(x[b]):
                for i in range(gap):
                    t = (i+1) / (gap+1)
                    y[l+i] = y[a]*(1-t) + y[b]*t
                    v[l+i] = True
    return y, v

def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.copy()
    win = int(win) | 1
    k = win//2
    y = x.copy()
    for i in range(len(x)):
        a = max(0, i-k); b = min(len(x), i+k+1)
        y[i] = np.mean(x[a:b])
    return y

def moving_average_1d_variable(x: np.ndarray, base_win: int, kappa: np.ndarray,
                               gamma: float, win_min: int = 3) -> np.ndarray:
    """
    [NEW] 曲率自适应平滑：高曲率处窗口缩小，减少拐角拖尾。
    win_i = max(win_min, int(round(base_win / (1 + gamma * kappa_i))))
    """
    if base_win <= 1 or kappa.size != x.size:
        return moving_average_1d(x, base_win)
    y = x.copy()
    for i in range(len(x)):
        ki = float(kappa[i])
        win_i = max(win_min, int(round(base_win / (1.0 + gamma * ki))))
        win_i |= 1  # 奇数
        k = win_i // 2
        a = max(0, i-k); b = min(len(x), i+k+1)
        y[i] = np.mean(x[a:b])
    return y

def gradient_clamp(delta_n: np.ndarray, s: np.ndarray, gmax: float) -> np.ndarray:
    """
    [NEW] 梯度限幅：逐点约束 |Δδ| ≤ gmax * Δs
    简单单向扫描两遍（forward/backward）以防局部溢出。
    """
    y = delta_n.copy()
    for _ in range(2):  # 两遍
        # forward
        for i in range(1, len(y)):
            ds = max(1e-9, s[i] - s[i-1])
            hi = y[i-1] + gmax * ds
            lo = y[i-1] - gmax * ds
            if y[i] > hi: y[i] = hi
            if y[i] < lo: y[i] = lo
        # backward
        for i in range(len(y)-2, -1, -1):
            ds = max(1e-9, s[i+1] - s[i])
            hi = y[i+1] + gmax * ds
            lo = y[i+1] - gmax * ds
            if y[i] > hi: y[i] = hi
            if y[i] < lo: y[i] = lo
    return y

# ===================== Guided Centerline（严格版） =====================
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

# === [OCCLUSION] 顶视遮挡多边形光栅化（机床XY -> 顶视像素） ===
def _rasterize_polygons_topdown(polys_xy, origin_xy, pix_mm, H, W, dilate_mm=0.0):
    """
    输入：多边形列表（机床XY, mm），顶视原点/分辨率，目标图尺寸
    输出：uint8 遮挡掩码（255=遮挡）
    """
    mask = np.zeros((H, W), np.uint8)
    if not polys_xy:
        return mask
    for poly in polys_xy:
        P = np.asarray(poly, np.float32)
        px = mach_xy_to_px_float(P, origin_xy, pix_mm, H).astype(np.int32)
        cv2.fillPoly(mask, [px.reshape(-1,1,2)], 255)
    if dilate_mm and float(dilate_mm) > 0:
        r = int(round(float(dilate_mm) / max(1e-9, pix_mm)))
        if r > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r*2+1, r*2+1))
            mask = cv2.dilate(mask, k, 1)
    return mask

# === [OCCLUSION] 从可见区域估计“轨迹环带半宽”（像素）
def _estimate_band_halfwidth_px(nearest_mask_u8, exclude_u8=None):
    ring = (nearest_mask_u8 > 0).astype(np.uint8) * 255
    if exclude_u8 is not None:
        ring = cv2.bitwise_and(ring, cv2.bitwise_not((exclude_u8 > 0).astype(np.uint8)))
    ring_mask, d_out, d_in = _outer_inner_distance_fields(ring)
    # 首选：外/内距离场的等距带
    if d_out is not None and d_in is not None:
        dist_sum = d_out + d_in + 1e-6
        diff = np.abs(d_out - d_in)
        tau = np.maximum(1.0, 0.04 * dist_sum)
        equi = ((diff <= tau) & (ring_mask > 0))
        vals = dist_sum[equi]
        if vals.size > 20:
            return float(0.5 * np.median(vals))
    # 回退：骨架 + 距离变换
    dt = cv2.distanceTransform((ring > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    sk = _skeletonize_bool(ring > 0)
    v = dt[sk > 0]
    if v.size > 0:
        return float(np.median(v))
    return 6.0  # 兜底：6px

# === [OCCLUSION] 在遮挡区内按 G 代码合成“环带掩码”，并与真实掩码拼合
def _synthesize_mask_in_occlusion(nearest_mask_u8, occ_u8, gcode_xy, origin_xy, pix_mm,
                                  band_halfwidth_mm=None):
    if occ_u8 is None or np.count_nonzero(occ_u8) == 0 or gcode_xy.size == 0:
        return nearest_mask_u8
    H, W = nearest_mask_u8.shape[:2]
    occ = (occ_u8 > 0)
    g_px = mach_xy_to_px_float(gcode_xy, origin_xy, pix_mm, H)
    idx = []
    for i in range(len(g_px)):
        x = int(np.clip(round(g_px[i,0]), 0, W-1))
        y = int(np.clip(round(g_px[i,1]), 0, H-1))
        if occ[y, x]: idx.append(i)
    if not idx:
        return nearest_mask_u8

    # 连续片段
    segs = []
    s0 = idx[0]; pre = idx[0]
    for k in idx[1:]:
        if k == pre + 1:
            pre = k
        else:
            segs.append((s0, pre)); s0 = k; pre = k
    segs.append((s0, pre))

    # 半宽（像素）
    if band_halfwidth_mm is None:
        half_px = _estimate_band_halfwidth_px(nearest_mask_u8, exclude_u8=occ_u8)
    else:
        half_px = float(band_halfwidth_mm) / max(1e-9, pix_mm)
    thick = max(1, int(round(half_px * 2.0)))

    # 只在遮挡区内画“加粗折线带”
    synth = np.zeros((H, W), np.uint8)
    for l, r in segs:
        pts = g_px[l:r+1].astype(np.int32).reshape(-1,1,2)
        cv2.polylines(synth, [pts], False, 255, thickness=thick, lineType=cv2.LINE_AA)
    synth = cv2.bitwise_and(synth, occ_u8)

    out = nearest_mask_u8.copy()
    out[occ] = 0                 # 清空遮挡区的“伪”测量
    out = cv2.bitwise_or(out, synth)
    # 轻微闭运算，避免缝隙
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
    return out


def postprocess_delta_segmentwise_for_export(
    delta_in: np.ndarray,
    ignore_mask: Optional[np.ndarray],
    s: np.ndarray,
    kappa: np.ndarray,
    *,
    base_win: int,
    curvature_adaptive: bool,
    curvature_gamma: float,
    win_min: int,
    max_abs_mm: float,
    grad_max_mm_per_mm: float,
    max_gap_pts: int
) -> np.ndarray:
    """
    目的：在导出前进行“分段”平滑/限幅，禁止跨越拐角传播。
    规则：
      - ignore_mask=True 的点：视为“硬边界 + 零偏移”，导出保持原始G代码坐标。
      - 其它点：在各自连续片段内做   短缺口插补 → 平滑 → 幅值限幅 → 梯度限幅。
    返回：处理后的 delta（不含 NaN）
    """
    x = np.asarray(delta_in, dtype=np.float32).copy()
    n = len(x)
    if n == 0:
        return x

    sep = (ignore_mask.astype(bool) if (ignore_mask is not None and len(ignore_mask)==n)
           else np.zeros(n, dtype=bool))

    # 被忽略的点固定为 0 偏移（保持顶点不动）
    x[sep] = 0.0

    out = x.copy()
    i = 0
    while i < n:
        # 跳过分隔点
        if sep[i]:
            i += 1
            continue
        # 找一个非忽略的连续片段 [l, r]
        l = i
        while i+1 < n and (not sep[i+1]):
            i += 1
        r = i

        seg = x[l:r+1].copy()
        s_seg = s[l:r+1]
        k_seg = kappa[l:r+1] if (kappa is not None and len(kappa)==n) else np.zeros_like(seg)

        # 片段内：仅对“短缺口”插补（保持与测量阶段一致）
        v = np.isfinite(seg)
        seg_fill, v2 = local_interpolate_short_gaps(seg, v, max_gap_pts=max_gap_pts)
        # 平滑
        if curvature_adaptive:
            seg_smooth = moving_average_1d_variable(seg_fill, base_win=base_win,
                                                    kappa=k_seg, gamma=curvature_gamma, win_min=win_min)
        else:
            seg_smooth = moving_average_1d(seg_fill, base_win)
        # 幅值限幅
        seg_smooth = np.clip(seg_smooth, -max_abs_mm, +max_abs_mm)
        # 梯度限幅（用片段自身的弧长）
        seg_final = gradient_clamp(seg_smooth, s_seg, gmax=grad_max_mm_per_mm)
        # 写回
        out[l:r+1] = seg_final

        i += 1

    # 保险：任何残存 NaN 置零
    out = np.nan_to_num(out, nan=0.0, posinf=max_abs_mm, neginf=-max_abs_mm)
    return out

def gcode_guided_centerline_strict(
    nearest_mask: np.ndarray,
    origin_xy: Tuple[float,float],
    pix_mm: float,
    gcode_xy: np.ndarray,
    gcode_normals: np.ndarray,
    *,
    halfwidth_mm: float,
    base_smooth_win: int,
    max_abs_mm: float,
    max_gap_pts: int,
    curvature_adaptive: bool,
    curvature_gamma: float,
    min_smooth_win: int,
    ignore_mask: Optional[np.ndarray] = None              # NEW
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    返回：
      centerline_xy  —— 仅用于可视化（短缺口插补后）
      delta_n_final  —— 与 gcode_xy 等长的法向偏移（含短缺口插补、平滑、限幅、梯度限幅）
      valid_mask     —— 原始命中（未插补）标记
      stats          —— 命中率/是否使用等距带等
    """
    H, W = nearest_mask.shape[:2]
    halfw_mm = float(halfwidth_mm)
    halfw_px = max(1.0, halfw_mm / max(1e-9, pix_mm))
    win = int(base_smooth_win)
    max_abs = float(max_abs_mm)

    ring_mask, d_out, d_in = _outer_inner_distance_fields(nearest_mask)
    use_equidist = (d_out is not None) and (d_in is not None)
    if not use_equidist:
        dt = cv2.distanceTransform((nearest_mask>0).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)

    g_px = mach_xy_to_px_float(gcode_xy, origin_xy, pix_mm, H)
    delta_n = np.full(g_px.shape[0], np.nan, np.float32)
    valid_mask = np.zeros(g_px.shape[0], np.bool_)

    g_px = mach_xy_to_px_float(gcode_xy, origin_xy, pix_mm, H)
    delta_n = np.full(g_px.shape[0], np.nan, np.float32)
    valid_mask = np.zeros(g_px.shape[0], np.bool_)
    skip = (ignore_mask.astype(bool) if (ignore_mask is not None and len(ignore_mask)==len(g_px))
            else np.zeros(len(g_px), np.bool_))                                     # NEW

    def bil(img, xy):
        return _bilinear_at(img, xy)

    for i in range(g_px.shape[0]):
        if skip[i]:                      # NEW: 拐角附近直接跳过（既不取点也不计入“未命中”）
            continue
        p  = g_px[i]
        n  = gcode_normals[min(i, len(gcode_normals)-1)]
        # 像素系法向（注意右手系到像素行列方向的转换）
        n_px = np.array([n[0]/pix_mm, -n[1]/pix_mm], dtype=np.float32)
        n_px /= (np.linalg.norm(n_px) + 1e-12)

        M  = max(9, int(np.ceil(2*halfw_px)))            # 采样密度略提高
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
            valid_mask[i] = True
        else:
            on = bil((nearest_mask>0).astype(np.float32), line_xy) >= 0.5
            if not np.any(on):
                continue
            dvals = bil(dt, line_xy); dvals[~on] = -1.0
            k = int(np.argmax(dvals))
            if dvals[k] > 0:
                delta_n[i] = float(ts[k] * pix_mm)
                valid_mask[i] = True

    # === 仅对“短缺口”插补 ===
    delta_n_fill, valid_fill = local_interpolate_short_gaps(delta_n, valid_mask, max_gap_pts=max_gap_pts)

    # 曲率自适应平滑（按 G 代码曲率）+ 幅值限幅
    kappa = curvature_kappa(gcode_xy)
    if curvature_adaptive:
        delta_sm = moving_average_1d_variable(delta_n_fill, base_win=win, kappa=kappa,
                                              gamma=curvature_gamma, win_min=int(min_smooth_win))
    else:
        delta_sm = moving_average_1d(delta_n_fill, win)
    delta_sm = np.clip(delta_sm, -max_abs, +max_abs)

    # 梯度限幅（按等弧长 s）
    s = arc_length_s(gcode_xy)
    gmax = float(PARAMS.get('guide_max_grad_mm_per_mm', 0.08))
    delta_final = gradient_clamp(delta_sm, s, gmax=gmax)

    # 可视化：短缺口已插补；长段缺失保持“未命中”，导出时由 Guard 判定
    centerline_xy = gcode_xy + gcode_normals * np.nan_to_num(delta_final, nan=0.0)[:,None]
    denom = int(np.count_nonzero(~skip)) if (ignore_mask is not None) else int(len(delta_n))
    ratio = float(np.count_nonzero(valid_mask)) / max(1, denom)
    stats = dict(valid=int(np.count_nonzero(valid_mask)), total=int(len(delta_n)),
                 ratio=ratio, use_equidist=bool(use_equidist))
    return centerline_xy.astype(np.float32), delta_final.astype(np.float32), valid_mask, stats

# ===================== 偏移曲线 & G 代码导出（严格版） =====================
def save_offset_csv(s: np.ndarray, delta_n: np.ndarray, dxy: np.ndarray, out_csv: Union[str, Path]):
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('s_mm,delta_n_mm,dx_mm,dy_mm\n')
        for i in range(len(delta_n)):
            f.write(f"{s[i]:.6f},{delta_n[i]:.6f},{dxy[i,0]:.6f},{dxy[i,1]:.6f}\n")
    print('[SAVE]', out_csv)
def save_ref_and_offsets_csv(s: np.ndarray, ref_xy: np.ndarray, dxy: np.ndarray, out_csv: Union[str, Path]):
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', encoding='utf-8') as f:
        f.write('s_mm,x_ref_mm,y_ref_mm,dx_mm,dy_mm,x_corr_mm,y_corr_mm\n')
        for i in range(len(s)):
            xr, yr = ref_xy[i]; dx, dy = dxy[i]
            f.write(f"{s[i]:.6f},{xr:.6f},{yr:.6f},{dx:.6f},{dy:.6f},{xr+dx:.6f},{yr+dy:.6f}\n")
    print('[SAVE]', out_csv)

def write_linear_gcode(xy: np.ndarray, out_path: Union[str, Path], feed: Optional[float]=None):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('; corrected by guided centerline (STRICT one-to-one)\n')
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

# ===================== 平面拟合/展平（沿用） =====================
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
    if abs(n[2]) < 1e-6:
        return height.copy()
    a = -n[0]/n[2]; b = -n[1]/n[2]; c = -d/n[2]
    Zp = _plane_z(a, b, c, X, Y)
    h_flat = height - Zp.astype(np.float32)
    return h_flat

# ===================== 可视化增强（沿用） =====================
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

def _render_biascomp_panel(raw_vals: np.ndarray,
                           corr_vals: np.ndarray,
                           title: str = 'BiasComp Δn (raw → corrected)',
                           width: int = 440, height: int = 300, bins: int = 48) -> np.ndarray:
    import numpy as _np
    import cv2 as _cv2

    img = _np.full((height, width, 3), 20, _np.uint8)

    # --- 数据清理 ---
    raw = _np.asarray(raw_vals, _np.float32)
    cor = _np.asarray(corr_vals, _np.float32)
    raw = raw[_np.isfinite(raw)]
    cor = cor[_np.isfinite(cor)]
    if raw.size == 0 and cor.size == 0:
        _cv2.putText(img, 'no valid data', (12, height//2), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, _cv2.LINE_AA)
        return img

    allv = raw if cor.size == 0 else (cor if raw.size == 0 else _np.concatenate([raw, cor]))
    # 统一、对称的范围（围绕 0）
    xmax = float(_np.percentile(_np.abs(allv), 98)) * 1.05
    xmax = max(0.5, xmax)
    edges = _np.linspace(-xmax, +xmax, bins+1)

    h_raw, _ = _np.histogram(raw, bins=edges)
    h_cor, _ = _np.histogram(cor, bins=edges)
    hmax = float(max(h_raw.max() if h_raw.size else 1, h_cor.max() if h_cor.size else 1, 1))

    # --- 画布布局 ---
    # --- 画布布局 ---  # 让绘图区更小，避免遮挡左上/左下角文字
    L, R, T, B = 88, 28, 72, 92
    Wp, Hp = width - L - R, height - T - B
    origin = (L, height - B)

    # 坐标轴
    _cv2.rectangle(img, (L-1, T-1), (L+Wp+1, T+Hp+1), (70,70,70), 1)
    # 0 参考线（竖线）
    x0_px = int(L + Wp * (0 - (-xmax)) / (2*xmax))
    _cv2.line(img, (x0_px, T), (x0_px, T+Hp), (80,80,80), 1, _cv2.LINE_AA)

    # x 轴刻度（-xmax, -xmax/2, 0, xmax/2, xmax）
    ticks = [-xmax, -0.5*xmax, 0.0, 0.5*xmax, xmax]
    for t in ticks:
        tx = int(L + Wp * (t + xmax) / (2*xmax))
        _cv2.line(img, (tx, T+Hp), (tx, T+Hp+4), (120,120,120), 1)
        s = f'{t:+.1f}'
        _cv2.putText(img, s, (tx-14, T+Hp+28), _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170,170,170), 1, _cv2.LINE_AA)

    # --- 画柱（同一坐标，双柱并排对比） ---
    w_bin = Wp / bins
    bar_pad = max(1, int(round(w_bin*0.10)))
    bw = max(1, int(round((w_bin - 3*bar_pad) / 2)))

    for i in range(bins):
        x_left = int(L + i*w_bin)
        # raw 在左、corrected 在右
        h1 = int(Hp * (h_raw[i] / hmax)) if i < len(h_raw) else 0
        h2 = int(Hp * (h_cor[i] / hmax)) if i < len(h_cor) else 0
        # raw 颜色：橙黄；corrected 颜色：青绿
        _cv2.rectangle(img,
            (x_left + bar_pad,          T+Hp-h1),
            (x_left + bar_pad + bw,     T+Hp),
            (80,200,255), -1)           # raw
        _cv2.rectangle(img,
            (x_left + 2*bar_pad + bw,   T+Hp-h2),
            (x_left + 2*bar_pad + 2*bw, T+Hp),
            (70,230,120), -1)           # corrected

    # --- 统计信息 ---
    def _stat(v):
        if v.size == 0:
            return dict(n=0, mean=0.0, med=0.0, p95=0.0, std=0.0)
        return dict(
            n=int(v.size),
            mean=float(_np.mean(v)),
            med=float(_np.median(v)),
            p95=float(_np.percentile(_np.abs(v), 95)),
            std=float(_np.std(v))
        )
    s_raw = _stat(raw); s_cor = _stat(cor)
    dp95 = s_cor['p95'] - s_raw['p95']
    dp95_pct = (dp95 / s_raw['p95'] * 100.0) if s_raw['p95'] > 1e-9 else 0.0

    # 标题与图例
    _cv2.putText(img, title, (12, 24), _cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 1, _cv2.LINE_AA)
    _cv2.rectangle(img, (12, 36), (28, 52), (80,200,255), -1);  _cv2.putText(img, 'raw', (34, 49), _cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210,210,210), 1, _cv2.LINE_AA)
    _cv2.rectangle(img, (74, 36), (90, 52), (70,230,120), -1);  _cv2.putText(img, 'corrected', (96, 49), _cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210,210,210), 1, _cv2.LINE_AA)

    # 数值卡片
    y0 = height - B + 6
    _cv2.putText(img, f"raw: n={s_raw['n']}  mean={s_raw['mean']:+.3f}  med={s_raw['med']:+.3f}  p95={s_raw['p95']:.3f}",
                 (12, y0+56), _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, _cv2.LINE_AA)
    _cv2.putText(img, f"cor: n={s_cor['n']}  mean={s_cor['mean']:+.3f}  med={s_cor['med']:+.3f}  p95={s_cor['p95']:.3f}  Δp95={dp95:+.3f}mm ({dp95_pct:+.1f}%)",
                 (12, y0+72), _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, _cv2.LINE_AA)

    # 范围说明
    _cv2.putText(img, f"range=[{-xmax:.1f},{xmax:.1f}]mm  bins={bins}  zero-line at x=0",
                 (12, T-8), _cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170,170,170), 1, _cv2.LINE_AA)

    return img


# ============================ 主流程（STRICT 集成） ============================
def run():
    cfg = PARAMS
    out_dir = Path(cfg['out_dir']); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 外参与 G 代码
    R, t, _ = load_extrinsic(cfg['T_path'])                         # 手眼外参（mm）
    g_raw, feed = parse_gcode_xy(cfg['gcode_path'], step_mm=cfg['guide_step_mm'])
    step_mm = float(cfg.get('guide_step_mm', 1.0))
    g_xy  = resample_polyline(g_raw, max(0.2, step_mm)) if g_raw.size > 0 else g_raw
    ref_tree = build_kdtree(g_xy) if g_xy.size > 0 else None
    T_ref, N_ref = tangent_normal(g_xy) if g_xy.size > 0 else (np.zeros((0,2)), np.zeros((0,2)))
    s_ref = arc_length_s(g_xy)

    # === 拐角忽略掩码（可选） ===                                   # NEW
    if PARAMS.get('corner_ignore_enable', True) and g_xy.size > 2:
        ignore_mask = compute_corner_ignore_mask(
            g_xy,
            angle_thr_deg=PARAMS.get('corner_angle_thr_deg', 35.0),
            span_mm=PARAMS.get('corner_ignore_span_mm', 6.0),
            step_mm=step_mm
        )
    else:
        ignore_mask = None

    # --- [ADD] 预加载偏差补偿（用于实时可视化对比，不影响原逻辑/窗口） ---
    bias_vis = None
    try:
        bc_cfg = PARAMS.get('bias_comp', {})
        if bc_cfg.get('enable', False):
            with open(bc_cfg.get('path', 'bias_comp.json'), 'r', encoding='utf-8') as f:
                bc = json.load(f)
            # 步长一致性校验（向量模式不强制点数一致）
            step_ok = abs(float(bc.get('guide_step_mm', step_mm)) - float(step_mm)) < 1e-9
            if step_ok and g_xy.size > 1:
                mode = str(bc.get('mode', 'vector')).lower()
                if mode in ('per_index', 'table'):
                    arr = np.asarray(bc.get('delta_bias_mm', []), dtype=np.float32)
                    if arr.size == len(g_xy):
                        bias_arr = arr
                    else:
                        # --- [NEW] per-index → 拐角锁定分段重采样 ---
                        thr = float(PARAMS.get('corner_angle_thr_deg', 35.0))
                        T_cur, _ = tangent_normal(g_xy)
                        k_cur = _detect_corner_knots_from_tangent(T_cur, ang_thresh_deg=thr, min_gap_pts=3)

                        gpath = bc.get('gcode_path', '')
                        step_cal = float(bc.get('guide_step_mm', step_mm))
                        bias_arr = None
                        if gpath:
                            g_cal_raw, _ = parse_gcode_xy(gpath, step_mm=step_cal)
                            g_cal = resample_polyline(g_cal_raw,
                                                      max(0.2, step_cal)) if g_cal_raw.size > 0 else g_cal_raw
                            if g_cal.size > 0:
                                T_cal, _ = tangent_normal(g_cal)
                                k_cal = _detect_corner_knots_from_tangent(T_cal, ang_thresh_deg=thr, min_gap_pts=3)
                                # 把“标定层拐角索引（基于 g_cal 长度）”映射到“arr.size”
                                k_in = _remap_knots_index(k_cal, len(g_cal), arr.size)
                                if len(k_in) == len(k_cur) and arr.size >= 2:
                                    bias_arr = _resample_per_index_bias_piecewise(arr, len(g_xy), k_in, k_cur)
                        if bias_arr is None:
                            bias_arr = _resample_per_index_bias_for_length(arr, len(g_xy))
                        print(f"[BIAS][vis] per_index resample: {arr.size}->{len(g_xy)} (segments={len(k_cur) - 1})")
                else:
                    # 向量模型：bias_i = dot(v, N_ref[i]) + b
                    v = np.asarray(bc.get('v', [0.0, 0.0]), dtype=np.float32)
                    b0 = float(bc.get('b', 0.0))
                    T_ref, N_ref = tangent_normal(g_xy)  # 确保已得到法向
                    bias_arr = (N_ref[:, 0] * v[0] + N_ref[:, 1] * v[1] + b0).astype(np.float32)
                # 与拐点忽略策略一致：忽略区间的补偿置零，避免跨段传播
                if (ignore_mask is not None) and (len(ignore_mask) == len(bias_arr)):
                    bias_arr = bias_arr.copy()
                    bias_arr[ignore_mask] = 0.0

                bias_vis = bias_arr
                print('[BIAS][vis] preload ok')
            else:
                print('[BIAS][vis] skipped (guide_step不匹配或g_xy过短)')
    except Exception as _e:
        print('[BIAS][vis] preload failed:', _e)

    # 2) 相机
    stream = PCamMLSStream(); stream.open()

    roi_mode = str(cfg.get('roi_mode', 'none')).lower()
    corr_mode = str(cfg.get('corr_mode', 'vector_median')).lower()  # 在线观测：向量 EMA
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

            # 3) ROI（含 gcode_bounds）
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

            # 5) 顶视投影（取最高 Z）
            height, mask_top, origin_xy = project_topdown_from_grid(P_mach, m_select, pix_mm, (x0,x1,y0,y1))

            # 5.0 [OCCLUSION] 在顶视网格中抠除固定遮挡
            occ_cfg = PARAMS.get('occlusion', {})
            occ_enable = bool(occ_cfg.get('enable', False)) and len(occ_cfg.get('polys', [])) > 0
            occ_top = None
            if occ_enable:
                Ht, Wt = height.shape
                occ_top = _rasterize_polygons_topdown(
                    occ_cfg.get('polys', []), origin_xy, pix_mm, Ht, Wt,
                    dilate_mm=float(occ_cfg.get('dilate_mm', 0.0))
                )
                height[occ_top > 0] = np.nan  # 遮挡记为缺测
                mask_top[occ_top > 0] = 0

            # 5.1 平面拟合与展平
            plane = None; inlier_ratio = float('nan')
            if cfg.get('plane_enable', True) and np.isfinite(height).any():
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

            # 6.1 [OCCLUSION] 遮挡区内按 G 代码 + 可见半宽合成“虚拟最近表面掩码”
            if occ_enable and occ_top is not None and g_xy.size > 0 and bool(occ_cfg.get('synthesize_band', True)):
                nearest_mask = _synthesize_mask_in_occlusion(
                    nearest_mask, occ_top, g_xy, origin_xy, pix_mm,
                    band_halfwidth_mm=occ_cfg.get('band_halfwidth_mm', None)
                )
                cv2.imshow('OcclusionTop', occ_top)

            # 7) 骨架（仅作可视化；不参与导出）
            skel_bgr = extract_skeleton_universal(nearest_mask, visualize=True)
            if skel_bgr is None:
                cv2.imshow('Centerline vs G-code (RHR)', np.zeros((480, 640, 3), np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            skel_gray = cv2.cvtColor(skel_bgr, cv2.COLOR_BGR2GRAY)

            # 8) G 代码引导的中轴线（严格一一对应）
            use_guided = bool(PARAMS.get('guide_enable', True)) and g_xy.size > 1
            if use_guided:
                centerline_xy, delta_n, valid_mask, rep = gcode_guided_centerline_strict(
                    nearest_mask, origin_xy, pix_mm, g_xy, N_ref,
                    halfwidth_mm=PARAMS['guide_halfwidth_mm'],
                    base_smooth_win=PARAMS['guide_smooth_win'],
                    max_abs_mm=PARAMS['guide_max_offset_mm'],
                    max_gap_pts=PARAMS['max_gap_pts'],
                    curvature_adaptive=PARAMS['curvature_adaptive'],
                    curvature_gamma=PARAMS['curvature_gamma'],
                    min_smooth_win=PARAMS['min_smooth_win'],
                    ignore_mask=ignore_mask  # NEW
                )
                e_idx = np.arange(len(delta_n), dtype=int)
                e_n   = np.nan_to_num(delta_n, nan=0.0)  # 在线可视化采用 0 占位；导出时仍看 valid_mask
            else:
                centerline_xy = g_xy.copy()
                valid_mask = np.zeros(len(g_xy), bool)
                e_idx = np.arange(len(g_xy), dtype=int)
                e_n = np.zeros(len(g_xy), float)
                rep = dict(ratio=0.0)

            # 9) 可视化叠加（顶视 + 最近表层 + 偏差箭头 + 法向采样线）
            vis_top_raw = render_topdown(height, mask_top, origin_xy, pix_mm, gcode_xy=g_xy)
            overlay = cv2.addWeighted(vis_top_raw, 1.0,
                                      cv2.cvtColor(nearest_mask, cv2.COLOR_GRAY2BGR), 0.25, 0)
            vis_cmp = draw_deviation_overlay(overlay, origin_xy, pix_mm, g_xy, centerline_xy, e_idx, e_n,
                                             arrow_stride=int(PARAMS['arrow_stride']),
                                             draw_probes=PARAMS.get('draw_normal_probes', True),
                                             N_ref=N_ref, valid_mask=valid_mask)
            vis_cmp = draw_machine_axes_overlay(vis_cmp, origin_xy, pix_mm)

            # 10) 在线纠偏（EMA，仅用“有效点”的统计）
            if cfg.get('print_corr', True) and N_ref.size > 0:
                idx_eff = np.where(valid_mask)[0]
                e_idx_eff = e_idx[idx_eff]
                e_n_eff   = e_n[idx_eff]
                dxdy, stats = corr.update(e_idx_eff, e_n_eff, N_ref) if e_n_eff.size > 0 else (np.zeros(2,float), dict(mean=0,median=0,p95=0,n=0))
                print("CORR frame={:06d}  mean={:+.3f}  med={:+.3f}  p95={:.3f}  ->  dx={:+.3f}  dy={:+.3f}  [mm]"
                      .format(frame_id, float(np.mean(e_n_eff)) if e_n_eff.size>0 else 0.0,
                              float(np.median(e_n_eff)) if e_n_eff.size>0 else 0.0,
                              float(np.percentile(np.abs(e_n_eff), 95)) if e_n_eff.size>0 else 0.0,
                              dxdy[0], dxdy[1]))
            else:
                dxdy = np.zeros(2, float); stats = dict(mean=0, median=0, p95=0)

            # 11) HUD & Guard 判定
            dev_mean = float(np.mean(e_n[valid_mask])) if valid_mask.any() else 0.0
            dev_med  = float(np.median(e_n[valid_mask])) if valid_mask.any() else 0.0
            dev_p95  = float(np.percentile(np.abs(e_n[valid_mask]), 95)) if valid_mask.any() else 0.0
            valid_ratio = float(np.count_nonzero(valid_mask)) / max(1, g_xy.shape[0])
            plane_info = f"inlier={inlier_ratio:.2f}" if np.isfinite(inlier_ratio) else "inlier=nan"

            # 长段缺失检测
            # 长段缺失检测（把“忽略点”当作有效）                      # NEW / CHG
            eff_valid_mask = valid_mask.copy()
            if ignore_mask is not None and len(ignore_mask) == len(eff_valid_mask):
                eff_valid_mask = eff_valid_mask | ignore_mask
            runs = find_missing_runs(eff_valid_mask)
            step_mm_eff = max(1e-9, float(PARAMS['guide_step_mm']))
            long_mm_max = float(PARAMS['long_missing_max_mm'])
            long_ratio_max = float(PARAMS['long_missing_max_ratio'])
            longest_pts = max([r - l + 1 for (l,r) in runs], default=0)
            longest_mm  = longest_pts * step_mm_eff
            missing_ratio = float(sum((r - l + 1) for (l, r) in runs)) / max(1, len(eff_valid_mask))

            guard = cfg['Guard']; guard_enable = guard.get('enable', True)
            guard_ok = True; reasons = []
            if guard_enable:
                if rep.get('ratio', 0.0) < guard.get('min_valid_ratio', 0.60):
                    guard_ok = False; reasons.append(f"valid_ratio {rep.get('ratio',0.0):.2f} < {guard.get('min_valid_ratio')}")
                if dev_p95 > guard.get('max_abs_p95_mm', 0.80):
                    guard_ok = False; reasons.append(f"p95 {dev_p95:.2f} > {guard.get('max_abs_p95_mm')}")
                if np.isfinite(inlier_ratio) and inlier_ratio < guard.get('min_plane_inlier_ratio', 0.55):
                    guard_ok = False; reasons.append(f"plane_inlier {inlier_ratio:.2f} < {guard.get('min_plane_inlier_ratio')}")
                if longest_mm > guard.get('long_missing_max_mm', long_mm_max) or missing_ratio > cfg.get('long_missing_max_ratio', long_ratio_max):
                    guard_ok = False; reasons.append(f"long_missing {longest_mm:.1f}mm or ratio {missing_ratio:.2f} over limit")
                # 梯度上限复核
                gmax_chk = guard.get('grad_max_mm_per_mm', cfg.get('guide_max_grad_mm_per_mm', 0.08))
                grad = np.abs(np.diff(delta_n)) / np.maximum(1e-9, np.diff(s_ref))
                if np.isfinite(grad).any() and float(np.nanpercentile(grad, 98)) > gmax_chk * 1.15:
                    guard_ok = False; reasons.append(f"grad@p98 {float(np.nanpercentile(grad, 98)):.3f} > {gmax_chk*1.15:.3f}")

            guard_str = "PASS" if guard_ok else "FAIL"

            Ht, Wt = height.shape
            txt = ('plane[%s]  band=[%.2f,%.2f]mm  pix=%.2fmm  grid=%dx%d  '
                   'dev(mean/med/p95)=%+.3f/%+.3f/%.3f  guided=%.2f  miss_long=%.1fmm  Guard=%s'
                   % (plane_info, *( (z_low, z_high) if np.isfinite(z_low) and np.isfinite(z_high) else (0.0,0.0) ),
                      pix_mm, Wt, Ht, dev_mean, dev_med, dev_p95, rep.get('ratio', 0.0), longest_mm, guard_str))
            cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis_cmp, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (0,255,0) if guard_ok else (0,0,255), 1, cv2.LINE_AA)

            cv2.imshow('Top-Down + Nearest Surface + Skeleton', overlay)
            cv2.imshow('NearestSurfaceMask', nearest_mask)
            cv2.imshow('Centerline vs G-code (RHR)', vis_cmp)

            # --- 新增：法线-交点最小可视化窗口（少量抽样） ---  # NEW
            if PARAMS.get('debug_normals_window', True) and g_xy.size > 1:
                vis_probe = render_normals_probe_window(
                    overlay, origin_xy, pix_mm,
                    g_xy, N_ref,
                    delta_n=delta_n if 'delta_n' in locals() else None,
                    valid_mask=( (valid_mask | ignore_mask) if (ignore_mask is not None and len(ignore_mask)==len(valid_mask)) else valid_mask ),  # NEW（可选）
                    stride=int(PARAMS.get('debug_normals_stride', 25)),
                    max_count=int(PARAMS.get('debug_normals_max', 40)),
                    len_mm=PARAMS.get('debug_normals_len_mm', None),
                    draw_text=bool(PARAMS.get('debug_normals_text', True))
                )
                cv2.imshow('Normal-Probes (few)', vis_probe)

                # --- [ADD] 偏差补偿后的可视化窗口（不影响原逻辑/窗口） ---
                if (bias_vis is not None) and (g_xy.size > 1) and ('delta_n' in locals()) and (
                        len(delta_n) == len(bias_vis)):
                    # 1) 计算纠正后的法向偏差（处于“测量域”：auto_flip之前的一致定义）
                    delta_n_corr = delta_n.copy()
                    mvis = np.isfinite(delta_n_corr)
                    delta_n_corr[mvis] = delta_n_corr[mvis] - bias_vis[mvis]

                    # 2) Normal-Probes (few) —— Bias Corrected（你特别要求的窗口）
                    vis_probe_corr = render_normals_probe_window(
                        overlay, origin_xy, pix_mm,
                        g_xy, N_ref,
                        delta_n=delta_n_corr,
                        valid_mask=((valid_mask | ignore_mask) if (
                                    ignore_mask is not None and len(ignore_mask) == len(valid_mask)) else valid_mask),
                        stride=int(PARAMS.get('debug_normals_stride', 25)),
                        max_count=int(PARAMS.get('debug_normals_max', 40)),
                        len_mm=PARAMS.get('debug_normals_len_mm', None),
                        draw_text=bool(PARAMS.get('debug_normals_text', True))
                    )
                    cv2.imshow('Normal-Probes (few) [Bias Corrected]', vis_probe_corr)

                    # 3) 叠加图：Centerline vs G-code（Bias Corrected）
                    e_idx_corr = np.arange(len(delta_n_corr), dtype=int)
                    e_n_corr = np.nan_to_num(delta_n_corr, nan=0.0)
                    centerline_corr = g_xy + N_ref * e_n_corr[:, None]
                    vis_cmp_corr = draw_deviation_overlay(
                        overlay, origin_xy, pix_mm,
                        g_xy, centerline_corr,
                        e_idx_corr, e_n_corr,
                        arrow_stride=int(PARAMS['arrow_stride']),
                        draw_probes=PARAMS.get('draw_normal_probes', True),
                        N_ref=N_ref, valid_mask=valid_mask
                    )
                    vis_cmp_corr = draw_machine_axes_overlay(vis_cmp_corr, origin_xy, pix_mm)
                    cv2.putText(vis_cmp_corr, 'bias compensated (visualization)', (12, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(vis_cmp_corr, 'bias compensated (visualization)', (12, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Centerline vs G-code [Bias Corrected]', vis_cmp_corr)

                    # 4) Δn 直方图对比（raw vs corrected），便于快速观察补偿量级
                    panel = _render_biascomp_panel(
                        np.asarray(delta_n, np.float32),
                        np.asarray(delta_n_corr, np.float32),
                        title='BiasComp Δn (raw → corrected)',
                        width=960, height=840, bins=48
                    )
                    cv2.imshow('BiasComp Δn Hist', panel)

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
            elif key == ord(cfg['export_on_key']) and g_xy.size>1:
                # Guard：不通过就拒绝导出（严格一一对应）
                if guard_enable and (not guard_ok):
                    print('[GUARD] 导出被拒绝：', '; '.join(reasons))
                else:
                    # === 1) 采用“严格法向单点决策”的 δ(s)（不再 KDTree 聚合） ===
                    delta_n_meas = delta_n.copy()
                    # === [ADD] 偏差补偿（在 auto_flip 之前，处于“测量域”） ===
                    try:
                        bc_cfg = PARAMS.get('bias_comp', {})
                        if bc_cfg.get('enable', False):
                            with open(bc_cfg.get('path', 'bias_comp.json'), 'r', encoding='utf-8') as f:
                                bc = json.load(f)

                            step_ok = abs(float(bc.get('guide_step_mm', PARAMS['guide_step_mm'])) - float(
                                PARAMS['guide_step_mm'])) < 1e-9
                            mode = str(bc.get('mode', 'vector')).lower()

                            if step_ok:
                                if mode in ('per_index', 'table'):
                                    arr = np.asarray(bc.get('delta_bias_mm', []), dtype=np.float32)
                                    if arr.size == len(delta_n_meas):
                                        bias = arr
                                    else:
                                        # --- [NEW] per-index → 拐角锁定分段重采样 ---
                                        thr = float(PARAMS.get('corner_angle_thr_deg', 35.0))
                                        # T_ref 已在 run() 顶部计算；若不在作用域，安全再算一次
                                        T_here, _ = tangent_normal(g_xy)
                                        k_cur = _detect_corner_knots_from_tangent(T_here, ang_thresh_deg=thr,
                                                                                  min_gap_pts=3)

                                        gpath = bc.get('gcode_path', '')
                                        step_cal = float(bc.get('guide_step_mm', PARAMS['guide_step_mm']))
                                        bias = None
                                        if gpath:
                                            g_cal_raw, _ = parse_gcode_xy(gpath, step_mm=step_cal)
                                            g_cal = resample_polyline(g_cal_raw, max(0.2,
                                                                                     step_cal)) if g_cal_raw.size > 0 else g_cal_raw
                                            if g_cal.size > 0:
                                                T_cal, _ = tangent_normal(g_cal)
                                                k_cal = _detect_corner_knots_from_tangent(T_cal, ang_thresh_deg=thr,
                                                                                          min_gap_pts=3)
                                                k_in = _remap_knots_index(k_cal, len(g_cal), arr.size)
                                                if len(k_in) == len(k_cur) and arr.size >= 2:
                                                    bias = _resample_per_index_bias_piecewise(arr, len(delta_n_meas),
                                                                                              k_in, k_cur)
                                        if bias is None:
                                            bias = _resample_per_index_bias_for_length(arr, len(delta_n_meas))
                                        print(f"[BIAS] per_index resample: {arr.size}->{len(delta_n_meas)}")
                                else:
                                    # 向量模型：bias_i = dot(v, N_ref[i]) + b
                                    v = np.asarray(bc.get('v', [0.0, 0.0]), dtype=np.float32)
                                    b0 = float(bc.get('b', 0.0))
                                    bias = (N_ref[:, 0] * v[0] + N_ref[:, 1] * v[1] + b0).astype(np.float32)

                                # 拐角忽略区间的补偿置零，避免跨段传播（保持你原逻辑）
                                if ('ignore_mask' in locals()) and (ignore_mask is not None) and (
                                        len(ignore_mask) == len(bias)):
                                    bias = bias.copy()
                                    bias[ignore_mask] = 0.0

                                # 只在有效测量处相减，NaN（长缺口）保持不变
                                m = np.isfinite(delta_n_meas)
                                delta_n_meas[m] = delta_n_meas[m] - bias[m]
                                print("[BIAS] applied: mean={:+.3f} p95={:.3f}".format(
                                    float(np.nanmean(bias)), float(np.nanpercentile(np.abs(bias), 95))
                                ))
                            else:
                                print("[BIAS] skipped: guide_step不匹配")
                    except Exception as _e:
                        print("[BIAS] failed:", _e)
                    # === [ADD END] ===

                    # === 2) 自动符号判定（可选） ===
                    if PARAMS.get('auto_flip_offset', True):
                        Mchk = min(len(delta_n_meas), len(g_xy))
                        if Mchk > 5 and np.isfinite(delta_n_meas[:Mchk]).any():
                            med_dn = float(np.nanmedian(delta_n_meas[:Mchk]))
                            # 以 +Y 差作为参考，若与法向位移中位数方向相反则翻转
                            raw_dy = float(np.median((g_xy + N_ref * np.nan_to_num(delta_n_meas)[:,None])[:Mchk,1] - g_xy[:Mchk,1]))
                            if abs(raw_dy) > 0.01 and abs(med_dn) > 0.01 and raw_dy * med_dn < 0:
                                delta_n_meas = -delta_n_meas
                                print('[AUTO] measured offset sign flipped raw_dy={:.3f} med_dn(before_flip)={:.3f}'
                                      .format(raw_dy, med_dn))

                    # === 3) 将“测得偏移”转换为“应用偏移” ===
                    _mode = str(PARAMS.get('offset_apply_mode', 'invert')).lower()
                    _apply_sign = -1.0 if _mode in ('invert', 'opposite', 'opp', 'toward_target', 'correct') else 1.0
                    delta_n_apply_raw = _apply_sign * delta_n_meas

                    # === 4) 分段平滑 & 限幅 & 梯度限幅（拐角为硬边界/零偏移） ===  # NEW
                    kappa = curvature_kappa(g_xy)
                    delta_n_apply = postprocess_delta_segmentwise_for_export(
                        delta_n_apply_raw,
                        ignore_mask=ignore_mask if ('ignore_mask' in locals() and ignore_mask is not None
                                                    and len(ignore_mask) == len(delta_n_apply_raw)) else None,
                        s=s_ref, kappa=kappa,
                        base_win=PARAMS['guide_smooth_win'],
                        curvature_adaptive=PARAMS['curvature_adaptive'],
                        curvature_gamma=PARAMS['curvature_gamma'],
                        win_min=PARAMS['min_smooth_win'],
                        max_abs_mm=PARAMS['guide_max_offset_mm'],
                        grad_max_mm_per_mm=PARAMS['guide_max_grad_mm_per_mm'],
                        max_gap_pts=PARAMS['max_gap_pts']
                    )

                    if ignore_mask is not None and len(ignore_mask) == len(delta_n_apply):
                        n_ign = int(np.count_nonzero(ignore_mask))
                        print(f"[EXPORT] corner-ignored points: {n_ign}")

                    # === 5) 生成纠偏 G 代码（严格按索引一一对应） ===
                    g_xy_corr = g_xy + N_ref * delta_n_apply[:,None]
                    dxy_vec = (N_ref * delta_n_apply[:,None])

                    # === 6) 导出 CSV & Gcode ===
                    dxy_vec = (N_ref * delta_n_apply[:, None])
                    save_offset_csv(s_ref, delta_n_apply, dxy_vec, PARAMS['offset_csv'])
                    save_ref_and_offsets_csv(s_ref, g_xy, dxy_vec, Path(PARAMS['out_dir']) / 'offset_table_full.csv')
                    write_linear_gcode(g_xy_corr, PARAMS['corrected_gcode'], feed=feed)
                    if PARAMS.get('export_centerline', False):
                        write_linear_gcode((g_xy + N_ref * np.nan_to_num(delta_n)[:,None])[:len(g_xy_corr)],
                                           PARAMS['centerline_gcode'], feed=feed)

                    # === 7) 预览叠加 ===
                    if PARAMS.get('preview_corrected', True):
                        try:
                            vis_prev = vis_cmp.copy()
                            Hprev, Wprev = vis_prev.shape[:2]
                            def xy_to_px(xy_arr):
                                if xy_arr.size == 0: return np.empty((0,2), int)
                                x0,y0 = origin_xy; y1 = y0 + Hprev * pix_mm
                                xs = np.clip(((xy_arr[:,0]-x0)/pix_mm).astype(int), 0, Wprev-1)
                                ys = np.clip(((y1 - xy_arr[:,1])/pix_mm).astype(int), 0, Hprev-1)
                                return np.stack([xs,ys], axis=1)
                            pts_corr = xy_to_px(g_xy_corr)
                            for ii in range(len(pts_corr)-1):
                                cv2.line(vis_prev, tuple(pts_corr[ii]), tuple(pts_corr[ii+1]), (0,255,255), 2, cv2.LINE_AA)
                            cv2.putText(vis_prev, 'corrected preview', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0,0,0), 2, cv2.LINE_AA)
                            cv2.putText(vis_prev, 'corrected preview', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0,255,255), 1, cv2.LINE_AA)
                            cv2.imshow('CorrectedPreview', vis_prev)
                            if PARAMS.get('save_corrected_preview', True):
                                prev_path = out_dir / 'corrected_preview.png'
                                cv2.imwrite(str(prev_path), vis_prev)
                                print('[SAVE]', prev_path)
                        except Exception as _e:
                            print('[WARN] preview failed:', _e)

                    # quicklook & report
                    if PARAMS.get('dump_quicklook', True):
                        hist = _render_histogram(delta_n_apply, title='applied delta_n histogram (mm)')
                        quick = _compose_quicklook(vis_cmp, None, hist,
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
                            pixel_size_mm=pix_mm,
                            longest_missing_mm=float(longest_mm)
                        )
                        rp = out_dir / 'report.json'
                        with rp.open('w', encoding='utf-8') as f:
                            json.dump(rep_json, f, ensure_ascii=False, indent=2)
                        print('[SAVE]', rp)
            elif key == ord('b') and g_xy.size > 1 and use_guided:
                # === [ADD] 标定导出：把当前帧逐点“固有偏差”写成 bias_comp.json ===
                try:
                    bc_cfg = PARAMS.get('bias_comp', {})
                    outp = Path(bc_cfg.get('path', 'bias_comp.json'))
                    data = {
                        "version": "bias_comp/v1",
                        "mode": "per_index",  # 逐点补偿表
                        "guide_step_mm": float(PARAMS.get('guide_step_mm', 1.0)),
                        "n_points": int(len(delta_n)),
                        "corner_ignore": bool(PARAMS.get('corner_ignore_enable', False)),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "gcode_path": PARAMS.get('gcode_path', ''),
                        # JSON 不支持 NaN，这里把缺测点置 0.0；应用时只在有效测量点相减
                        "delta_bias_mm": np.nan_to_num(np.asarray(delta_n, dtype=float), nan=0.0).tolist()
                    }
                    # 可选：保存拐点忽略索引，便于审计
                    if ('ignore_mask' in locals()) and (ignore_mask is not None) and len(ignore_mask) == len(delta_n):
                        data["ignored_indices"] = np.where(ignore_mask)[0].astype(int).tolist()
                    outp.parent.mkdir(parents=True, exist_ok=True)
                    with open(outp, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"[BIAS] saved to {outp} (n={data['n_points']}, step={data['guide_step_mm']})")
                except Exception as e:
                    print("[BIAS] save failed:", e)

            frame_id += 1

    finally:
        try: stream.close()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == '__main__':
    run()
