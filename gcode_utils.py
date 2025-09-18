# -*- coding: utf-8 -*-
# G-code 工具模块：解析 G 代码并将刀具路径转换为像素坐标系下的理论中轴线。
#
# 设计目标：
# 1) 支持常见指令：G0/G00（快移）、G1/G01（直线插补）、G2/G3（圆弧，近似离散化）
# 2) 支持 G90 绝对坐标 / G91 相对坐标 模式切换
# 3) 支持 X/Z 两轴（车床常用）；如存在 Y 轴可扩展
# 4) 将机床坐标系下路径（mm）映射到像素坐标（u,v）：需要 hand_eye_transform.npy 提供的仿射
#    - 假设 hand_eye_transform.npy 形状为 (2,3)：  [X]   [a11 a12 a13] [u]
#                                             [Z] = [a21 a22 a23] [v]
#                                                                    [1]
#    - 则可由二维线性部分 M = [[a11,a12],[a21,a22]] 与偏置 b = [a13,a23]^T，
#      求逆得到： [u v]^T = M^{-1} ( [X Z]^T - b )
# 5) 提供像素轨迹的重采样与裁剪（只保留相机 FOV 内段）
#
# 提示：不同机床/后处理器的 G 代码方言可能有差异，此处实现覆盖通用子集。
#
import re
import numpy as np

# ========== 基础解析 ==========

_re_word = re.compile(r'([A-Za-z])\s*(-?\d+(?:\.\d+)?)')

class GCodeState(object):
    """维护解析状态：当前位置、模式（绝对/相对）、平面等。"""
    def __init__(self):
        self.abs_mode = True  # G90 绝对；G91 相对
        self.pos = np.array([0.0, 0.0], dtype=float)  # [X, Z]
        self.feed = None

def parse_gcode(text):
    """解析 G 代码文本，产出规范化的指令序列。"""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('(') or line.startswith(';'):
            continue
        # 去掉分号后注释
        if ';' in line:
            line = line.split(';',1)[0].strip()
        lines.append(line)
    return lines

def _words(line):
    return dict((m.group(1).upper(), float(m.group(2))) for m in _re_word.finditer(line))

# ========== 轨迹生成 ==========

def gcode_to_machine_polyline(lines, unit_scale=1.0, arc_seglen=0.5):
    """
    将 G 代码转换为机床坐标系（X,Z）的折线点列（单位：mm）。

    参数：
    - lines: 经过 parse_gcode 处理的字符串行列表
    - unit_scale: 单位缩放（若 G 代码为英寸，可传 25.4）
    - arc_seglen: 圆弧离散化段长（mm）

    返回：N×2 的 ndarray（列为 [X,Z]）。
    """
    st = GCodeState()
    pts = [st.pos.copy()]

    for line in lines:
        wd = _words(line)
        # 模式切换
        if 'G' in wd:
            g = int(wd['G'])
            if g == 90:  # 绝对
                st.abs_mode = True
            elif g == 91:  # 相对
                st.abs_mode = False

        # 位置指令
        has_motion = any(k in wd for k in ('X','Z','I','K'))
        if not has_motion:
            continue

        tgt = st.pos.copy()
        if 'X' in wd:
            x = wd['X'] * unit_scale
            tgt[0] = x if st.abs_mode else tgt[0] + x
        if 'Z' in wd:
            z = wd['Z'] * unit_scale
            tgt[1] = z if st.abs_mode else tgt[1] + z

        # 直线/快移
        if 'G' in wd and int(wd['G']) in (0,1):
            if not np.allclose(tgt, st.pos):
                pts.append(tgt.copy())
                st.pos = tgt
            continue

        # 圆弧（G2/G3，顺/逆时针） —— 使用 I/K 偏置；离散为等弧长线段
        if 'G' in wd and int(wd['G']) in (2,3):
            cw = int(wd['G']) == 2
            cx = st.pos[0] + (wd.get('I',0.0)*unit_scale)
            cz = st.pos[1] + (wd.get('K',0.0)*unit_scale)
            center = np.array([cx, cz])
            r = np.linalg.norm(st.pos - center)
            if r < 1e-6:
                # 退化为直线
                if not np.allclose(tgt, st.pos):
                    pts.append(tgt.copy()); st.pos = tgt
                continue
            # 起止角
            a0 = np.arctan2(st.pos[1]-cz, st.pos[0]-cx)
            a1 = np.arctan2(tgt[1]-cz, tgt[0]-cx)
            # 角度差（考虑顺/逆时针）
            if cw:
                if a1 > a0: a1 -= 2*np.pi
            else:
                if a1 < a0: a1 += 2*np.pi
            arc_len = abs(a1 - a0) * r
            nseg = max(2, int(np.ceil(arc_len / max(1e-3, arc_seglen))))
            for i in range(1, nseg+1):
                a = a0 + (a1-a0)*i/nseg
                p = np.array([cx + r*np.cos(a), cz + r*np.sin(a)])
                pts.append(p)
            st.pos = tgt
            continue

        # 其他：忽略
        st.pos = tgt

    return np.vstack(pts)

# ========== 手眼映射：机床(mm) -> 像素(u,v) / 像素 -> 机床 ==========

def load_hand_eye(path='hand_eye_transform.npy'):
    arr = np.load(path)
    if arr.shape != (2,3):
        raise ValueError("hand_eye_transform.npy 预期形状为 (2,3)")
    return arr

def pixel_to_machine(uv, A2x3):
    """像素 -> 机床： [X;Z] = A [u;v;1]"""
    uv1 = np.c_[uv, np.ones((uv.shape[0],1))]
    XZ = (A2x3 @ uv1.T).T
    return XZ

def machine_to_pixel(XZ, A2x3):
    """机床 -> 像素： [u;v] = M^{-1} ([X;Z]-b)，其中 A=[M|b]"""
    M = A2x3[:, :2]    # 2x2
    b = A2x3[:, 2]     # 2,
    Minv = np.linalg.inv(M)
    uv = (Minv @ (XZ.T - b[:,None])).T
    return uv

# ========== 轨迹重采样/裁剪 ==========

def resample_polyline(poly, step=2.0):
    """按等距（近似）重采样折线，返回新点列。单位同输入。"""
    if len(poly) < 2:
        return poly.copy()
    seglens = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    total = float(np.sum(seglens))
    if total < 1e-9:
        return poly.copy()
    n = max(2, int(total/step))
    # 逐段线性插值
    new_pts = [poly[0]]
    dist_acc = 0.0
    i = 0
    for k in range(1, n):
        target = k * total / (n-1)
        while dist_acc + seglens[i] < target and i < len(seglens)-1:
            dist_acc += seglens[i]; i += 1
        t = (target - dist_acc) / max(1e-9, seglens[i])
        p = poly[i] + t*(poly[i+1]-poly[i])
        new_pts.append(p)
    if not np.allclose(new_pts[-1], poly[-1]):
        new_pts.append(poly[-1])
    return np.vstack(new_pts)

def clip_to_image(uv, shape_hw, margin=0):
    """保留落在图像内（含边界 margin）的点。"""
    H, W = shape_hw
    u = uv[:,0]; v = uv[:,1]
    mask = (u>=-margin) & (u<W+margin) & (v>=-margin) & (v<H+margin)
    return uv[mask]
