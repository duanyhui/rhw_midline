#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01 — IO 初始化与全局点云预览（Python 3.7+ 全兼容）
----------------------------------------------------
1) 加载外参 (R,t,T) 与 G 代码路径点
2) 初始化 Percipio 相机流
3) 读取一帧全局点云 P_cam (H,W,3, 单位:mm)
4) 用 Z 通道做伪彩可视化预览；支持循环采集/保存/退出

热键：
- q: 退出
- s: 保存当前深度伪彩图到 ./out/depth_preview_xxx.png
"""

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import cv2
import sys
from pathlib import Path

# ==== 可选依赖：Percipio SDK ====
try:
    import pcammls  # Percipio SDK
except Exception:
    pcammls = None


# ==== 基础 IO：外参 / G 代码 ====

def load_extrinsic(T_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 .npy 读取: dict(R, t, T)，均为 numpy 数组
    R: (3,3) 旋转；t: (1,3) 或 (3,) 平移；T: (4,4) 齐次
    """
    data = np.load(str(T_path), allow_pickle=True).item()
    R = np.asarray(data['R'], dtype=float)
    t = np.asarray(data['t'], dtype=float).reshape(1, 3)
    T = np.asarray(data['T'], dtype=float)
    return R, t, T


def parse_gcode_xy(path: Union[str, Path]) -> np.ndarray:
    """
    解析 G0/G1 指令中的 X/Y，忽略注释与圆弧等高级指令。
    返回 (N,2) mm；若路径不存在则返回空数组。
    """
    p = Path(path) if not isinstance(path, Path) else path
    if (not path) or (not p.exists()):
        return np.empty((0, 2), float)

    pts = []
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        x = None
        y = None
        for raw in f:
            line = raw.strip()
            if (not line) or line.startswith(';') or line.startswith('('):
                continue
            if ';' in line:
                line = line.split(';', 1)[0]
            # 去括号注释
            while '(' in line and ')' in line:
                a = line.find('(')
                b = line.find(')')
                if a < 0 or b < 0 or b <= a:
                    break
                line = (line[:a] + ' ' + line[b + 1:]).strip()
            toks = line.split()
            if not toks:
                continue
            cmd = toks[0].upper()
            if cmd in ('G0', 'G00', 'G1', 'G01'):
                for tkn in toks[1:]:
                    u = tkn.upper()
                    if u.startswith('X'):
                        try:
                            x = float(u[1:])
                        except Exception:
                            pass
                    elif u.startswith('Y'):
                        try:
                            y = float(u[1:])
                        except Exception:
                            pass
                if x is not None and y is not None:
                    pts.append([x, y])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 2), float)


# ==== 相机封装 ====

class PCamMLSStream:
    """
    最小封装：打开 → 拉流 → 点云映射 → 关闭
    read_pointcloud() -> (P_cam, depth_frame)
    - P_cam: (H,W,3) float32, 单位 mm
    - depth_frame: Percipio 原始深度帧（用于调试/标注）
    """
    def __init__(self):
        if pcammls is None:
            raise SystemExit('未安装 pcammls，无法使用相机。请先安装 Percipio SDK。')
        self.cl = pcammls.PercipioSDK()
        self.h = None
        self.depth_calib = None
        self.scale_unit = 1.0
        self.pointcloud = pcammls.pointcloud_data_list()

    def open(self) -> None:
        devs = self.cl.ListDevice()
        if len(devs) == 0:
            raise SystemExit('未发现设备。')
        print('检测到设备:')
        for i, d in enumerate(devs):
            print('  {}: {}\t{}'.format(i, d.id, d.iface.id))
        try:
            idx_str = input('选择设备索引 (回车默认 0): ').strip()
            idx = int(idx_str) if idx_str else 0
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(devs) - 1))
        sn = devs[idx].id

        h = self.cl.Open(sn)
        if not self.cl.isValidHandle(h):
            raise SystemExit('打开设备失败: {}'.format(self.cl.TYGetLastErrorCodedescription()))
        self.h = h

        depth_fmts = self.cl.DeviceStreamFormatDump(h, pcammls.PERCIPIO_STREAM_DEPTH)
        if not depth_fmts:
            raise SystemExit('无深度流。')
        self.cl.DeviceStreamFormatConfig(h, pcammls.PERCIPIO_STREAM_DEPTH, depth_fmts[0])
        self.cl.DeviceLoadDefaultParameters(h)
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(h)
        self.depth_calib = self.cl.DeviceReadCalibData(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamEnable(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.cl.DeviceStreamOn(h)

    def read_pointcloud(self, timeout_ms: int = 2000):
        imgs = self.cl.DeviceStreamRead(self.h, timeout_ms)
        depth_img = None
        for fr in imgs:
            if fr.streamID == pcammls.PERCIPIO_STREAM_DEPTH:
                depth_img = fr
                break
        if depth_img is None:
            return None, None
        # 映射到 3D 点云（相机坐标系，单位 mm）
        self.cl.DeviceStreamMapDepthImageToPoint3D(
            depth_img, self.depth_calib, self.scale_unit, self.pointcloud
        )
        return self.pointcloud.as_nparray(), depth_img

    def close(self) -> None:
        if self.h is not None:
            try:
                self.cl.DeviceStreamOff(self.h)
            except Exception:
                pass
            try:
                self.cl.Close(self.h)
            except Exception:
                pass
            self.h = None


# ==== 可视化工具 ====

def depth_colormap_from_pcam(P_cam: np.ndarray) -> np.ndarray:
    """
    把 P_cam 的 Z(mm) 映射为伪彩图，供快速预览
    - 自适应 5~95 分位增强对比度
    - 非法/0 值置为无效不参与拉伸
    """
    Z = P_cam[:, :, 2].astype(np.float32).copy()
    mask = np.isfinite(Z)
    if not np.any(mask):
        return np.zeros((Z.shape[0], Z.shape[1], 3), np.uint8)
    # 过滤 0（多数设备 0 表示无效距离）
    valid = (mask & (Z != 0))
    if not np.any(valid):
        return np.zeros((Z.shape[0], Z.shape[1], 3), np.uint8)
    z = Z[valid]
    z_min = float(np.percentile(z, 5))
    z_max = float(np.percentile(z, 95))
    z_span = max(1.0, z_max - z_min)
    Zn = np.clip((Z - z_min) / z_span, 0.0, 1.0)
    Zn[~valid] = 0.0
    img = (Zn * 255.0).astype(np.uint8)
    cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
    vis = cv2.applyColorMap(img, cmap)
    return vis


# ==== 主流程 ====

def main() -> None:
    # 1) 读取外参与 G 代码
    default_T = 'T_cam2machine.npy'
    default_gcode = 'path/example.gcode'
    try:
        T_path = input('外参路径 (默认 {}): '.format(default_T)).strip() or default_T
    except Exception:
        T_path = default_T
    try:
        G_path = input('G代码路径 (默认 {}): '.format(default_gcode)).strip() or default_gcode
    except Exception:
        G_path = default_gcode

    if not Path(T_path).exists():
        raise SystemExit('[错误] 外参文件不存在: {}'.format(T_path))
    R, t, T = load_extrinsic(T_path)
    print('[OK] 外参加载: R={}, t={}, T={}'.format(R.shape, t.shape, T.shape))

    g_xy = parse_gcode_xy(G_path)
    print('[OK] G代码点数: {} (mm)'.format(len(g_xy)))

    # 2) 初始化相机
    stream = PCamMLSStream()
    stream.open()
    out_dir = Path('out')
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_id = 0

    print('[INFO] 开始拉流（q 退出，s 保存当前帧）')
    try:
        while True:
            P_cam, depth_fr = stream.read_pointcloud(timeout_ms=2000)
            if P_cam is None:
                print('[WARN] 未获取到深度帧')
                continue

            H, W, _ = P_cam.shape
            # 3) 可视化：Z(mm) 伪彩
            vis = depth_colormap_from_pcam(P_cam)
            hud = vis.copy()
            txt = 'P_cam: {}x{}  points={}'.format(W, H, W * H)
            cv2.putText(hud, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(hud, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Depth Z (mm) — from P_cam', hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                outp = out_dir / 'depth_preview_{:06d}.png'.format(frame_id)
                cv2.imwrite(str(outp), hud)
                print('[SAVE] {}'.format(outp))
                frame_id += 1
                # 保存点云为npy
                outp_pcam = out_dir / 'P_cam_{:06d}.npy'.format(frame_id)
                np.save(str(outp_pcam), P_cam)
                print('[SAVE] {}'.format(outp_pcam))

    finally:
        try:
            stream.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
