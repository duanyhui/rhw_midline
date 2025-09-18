#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_3d_pro.py
单帧交互标定：在“彩色→深度坐标”的注册彩色图上点击采点，
使用同帧点云获取相机系3D点（深度相机坐标系），拟合相机->机床外参，
输出与现有脚本一致的 T_cam2machine.npy（含 R/t/T），并提供丰富可视化。

依赖：pcammls, opencv-python, numpy
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import json
import time

try:
    import pcammls
except Exception:
    pcammls = None


# =================== 与现版 calibrate_3d.py 保持一致的基础函数 ===================

def estimate_rigid_transform(cam_pts: np.ndarray, mach_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(cam_pts, dtype=float)
    B = np.asarray(mach_pts, dtype=float)
    if A.shape != B.shape or A.shape[1] != 3:
        raise ValueError(f"形状不匹配: cam {A.shape}, mach {B.shape}，需要 Nx3")
    if A.shape[0] < 3:
        raise ValueError("3D 刚体拟合至少需要 3 对点。")
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t

def apply_transform(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T + t

def build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def eval_alignment(R: np.ndarray, t: np.ndarray, cam_pts: np.ndarray, mach_pts: np.ndarray) -> dict:
    pred = apply_transform(R, t, cam_pts)
    err = np.linalg.norm(pred - mach_pts, axis=1)
    return {
        'rms_mm': float(np.sqrt(np.mean(err ** 2))) if len(err) else float('nan'),
        'p95_mm': float(np.percentile(err, 95)) if len(err) else float('nan'),
        'max_mm': float(np.max(err)) if len(err) else float('nan'),
        'errors_mm': err
    }

def save_transform(R: np.ndarray, t: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {'R': R.astype(float), 't': t.astype(float), 'T': build_T(R, t).astype(float)}
    np.save(out_path, data)
    # 可选：同时输出 yaml/json 便于审阅（不改变主输出结构）
    try:
        import yaml
        with open(out_path.with_suffix('.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump({k: v.tolist() for k, v in data.items()}, f, sort_keys=False, allow_unicode=True)
    except Exception:
        with open(out_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump({k: v.tolist() for k, v in data.items()}, f, indent=2, ensure_ascii=False)

def trim_outliers_once(cam_pts: np.ndarray, mach_pts: np.ndarray, trim_percent: float):
    if trim_percent <= 0:
        idx = np.arange(len(cam_pts))
        return cam_pts, mach_pts, idx
    R0, t0 = estimate_rigid_transform(cam_pts, mach_pts)
    pred = apply_transform(R0, t0, cam_pts)
    err = np.linalg.norm(pred - mach_pts, axis=1)
    k = int(np.floor(len(err) * (1.0 - trim_percent / 100.0)))
    if k < 3:
        raise ValueError("修剪后剩余点数不足 3 个。")
    idx = np.argsort(err)[:k]
    return cam_pts[idx], mach_pts[idx], idx


# =================== 单帧采集与对齐（参考 SDK registration 示例） ===================
class SingleShotGrabber:
    """抓取单帧彩色+深度，并计算：
       - registration_rgb: 彩色→深度坐标的注册彩色图（用于点击）
       - depth_render: 深度伪彩（调试）
       - p3d: 深度帧对应的点云 (H,W,3) [mm]，在“深度相机坐标系”
    """
    def __init__(self):
        if pcammls is None:
            raise RuntimeError("未安装 pcammls，无法访问设备。")
        self.cl = pcammls.PercipioSDK()
        self.h = None
        self.scale_unit = 1.0
        self.depth_calib = None
        self.color_calib = None

        self.img_parsed_color = pcammls.image_data()
        self.img_undistortion_color = pcammls.image_data()
        self.img_registration_color = pcammls.image_data()
        self.img_registration_render = pcammls.image_data()
        self.pointcloud_data_arr = pcammls.pointcloud_data_list()

    def open(self):
        dev_list = self.cl.ListDevice()
        if len(dev_list) == 0:
            raise RuntimeError("未发现设备。")
        # 默认取第一个设备；如需选择可自行扩展
        sn = dev_list[0].id
        h = self.cl.Open(sn)
        if not self.cl.isValidHandle(h):
            raise RuntimeError(f"打开设备失败: {self.cl.TYGetLastErrorCodedescription()}")
        self.h = h

        # 配置流格式
        color_fmt_list = self.cl.DeviceStreamFormatDump(h, pcammls.PERCIPIO_STREAM_COLOR)
        depth_fmt_list = self.cl.DeviceStreamFormatDump(h, pcammls.PERCIPIO_STREAM_DEPTH)
        if len(depth_fmt_list) == 0:
            raise RuntimeError("设备无深度流。")
        self.cl.DeviceStreamFormatConfig(h, pcammls.PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])
        if len(color_fmt_list) > 0:
            self.cl.DeviceStreamFormatConfig(h, pcammls.PERCIPIO_STREAM_COLOR, color_fmt_list[0])

        # 载入参数
        self.cl.DeviceLoadDefaultParameters(h)
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(h)
        self.depth_calib = self.cl.DeviceReadCalibData(h, pcammls.PERCIPIO_STREAM_DEPTH)
        self.color_calib = self.cl.DeviceReadCalibData(h, pcammls.PERCIPIO_STREAM_COLOR)

        # 启动拉流
        enable_mask = pcammls.PERCIPIO_STREAM_DEPTH | pcammls.PERCIPIO_STREAM_COLOR
        self.cl.DeviceStreamEnable(h, enable_mask)
        self.cl.DeviceStreamOn(h)

    def close(self):
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

    def grab_one(self, timeout_ms=2000):
        """抓取单帧，并构建 registration_rgb / depth_render / p3d"""
        image_list = self.cl.DeviceStreamRead(self.h, timeout_ms)
        img_depth = None
        img_color = None
        for fr in image_list:
            if fr.streamID == pcammls.PERCIPIO_STREAM_DEPTH:
                img_depth = fr
            elif fr.streamID == pcammls.PERCIPIO_STREAM_COLOR:
                img_color = fr
        if img_depth is None or img_color is None:
            return None

        # 1) 解码+去畸变彩色
        self.cl.DeviceStreamImageDecode(img_color, self.img_parsed_color)
        self.cl.DeviceStreamDoUndistortion(self.color_calib, self.img_parsed_color, self.img_undistortion_color)

        # 2) 彩色 → 深度坐标注册（用于在“彩色图”上点击但像素与深度完全对齐）
        self.cl.DeviceStreamMapRGBImageToDepthCoordinate(
            self.depth_calib, img_depth, self.scale_unit,
            self.color_calib, self.img_undistortion_color, self.img_registration_color
        )

        # 3) 深度渲染（调试展示）
        self.cl.DeviceStreamDepthRender(img_depth, self.img_registration_render)

        # 4) 深度 → 点云（用于从像素获取 3D 相机系坐标）
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib, self.scale_unit, self.pointcloud_data_arr)

        # 转 numpy（复制单帧，供后续点击使用）
        registration_rgb = self.img_registration_color.as_nparray().copy()
        depth_render = self.img_registration_render.as_nparray().copy()
        p3d = self.pointcloud_data_arr.as_nparray().copy()  # (H,W,3) in mm（深度相机坐标系）

        return dict(registration_rgb=registration_rgb, depth_render=depth_render, p3d=p3d)


# =================== 交互 UI：在注册彩色图上点击、拟合、可视化 ===================
class CalibUI:
    def __init__(self, single_frame: dict, out_path: str, trim_percent: float):
        self.rgb = single_frame['registration_rgb']   # 显示&点击的图（彩色→深度坐标）
        self.depth_vis = single_frame['depth_render'] # 伪彩深度
        self.p3d = single_frame['p3d']                # (H,W,3) mm，深度相机坐标系
        self.H, self.W = self.rgb.shape[:2]
        self.overlay = self.rgb.copy()
        self.out_path = out_path
        self.trim_percent = float(trim_percent)

        self.cam_pts: List[np.ndarray] = []
        self.mach_pts: List[np.ndarray] = []
        self.valid_mask = np.isfinite(self.p3d).all(axis=2) & (self.p3d[...,2] > 0)

        self.R = None
        self.t = None
        self.rep = None
        self.kept_idx = None  # 剔除离群后保留的索引

    def _draw_hud(self):
        img = self.overlay
        cv2.putText(img, 'Click on Registration RGB (color->depth). Keys: z=undo, f=fit, s=save, c=clear, q=quit',
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'Click on Registration RGB (color->depth). Keys: z=undo, f=fit, s=save, c=clear, q=quit',
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        n = len(self.cam_pts)
        txt = f'Points: {n}    trim_percent={self.trim_percent:.1f}%'
        if self.rep is not None:
            txt += f'    RMS={self.rep["rms_mm"]:.3f}  P95={self.rep["p95_mm"]:.3f}  Max={self.rep["max_mm"]:.3f} [mm]'
        cv2.putText(img, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

    def _redraw(self):
        self.overlay = self.rgb.copy()
        # 标注点与编号
        for i, p in enumerate(self.cam_pts):
            # 反找像素位置（用最近点搜索）：这里我们直接用记录的点击像素
            pass
        # 在 overlay 上画编号和误差
        for i, (px, py) in enumerate(self._clicked_pixels):
            col = (0,255,0)
            if self.kept_idx is not None and (i not in set(self.kept_idx.tolist())):
                col = (0,0,255)  # 被剔除的点
            cv2.circle(self.overlay, (px, py), 5, col, 2, cv2.LINE_AA)
            label = f'{i+1}'
            cv2.putText(self.overlay, label, (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        # 如果已拟合，显示每点误差
        if self.rep is not None and len(self.cam_pts) == len(self.mach_pts) and len(self.cam_pts) >= 3:
            R, t = self.R, self.t
            pred = apply_transform(R, t, np.asarray(self.cam_pts))
            err = np.linalg.norm(pred - np.asarray(self.mach_pts), axis=1)
            for i, (px, py) in enumerate(self._clicked_pixels):
                e = float(err[i])
                col = (0,255,0) if e < 0.6 else (0,165,255) if e < 1.0 else (0,0,255)
                cv2.putText(self.overlay, f'{e:.2f}mm', (px+6, py+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(self.overlay, f'{e:.2f}mm', (px+6, py+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        self._draw_hud()

    def _pixel_to_cam3d(self, x: int, y: int) -> np.ndarray | None:
        if not (0 <= x < self.W and 0 <= y < self.H):
            return None
        if not self.valid_mask[y, x]:
            return None
        p = self.p3d[y, x].astype(float)
        if not np.isfinite(p).all() or p[2] <= 0:
            return None
        return p

    def _ask_mach_point(self, idx: int) -> np.ndarray | None:
        while True:
            s = input(f'点 #{idx} 机床 XYZ（mm），例如 125.0 40.5 0.0：').strip()
            if s == '':
                return None
            try:
                x, y, z = map(float, s.split())
                return np.array([x, y, z], dtype=float)
            except Exception:
                print('格式不正确，请按：X Y Z 输入。')

    def _fit_and_report(self):
        if len(self.cam_pts) < 3 or len(self.mach_pts) < 3:
            print('至少需要 3 个点才能拟合。')
            return
        cam = np.asarray(self.cam_pts)
        mach = np.asarray(self.mach_pts)
        # 可选剔除离群点
        try:
            cam2, mach2, kept_idx = trim_outliers_once(cam, mach, self.trim_percent)
        except Exception as e:
            print('[WARN] 修剪失败：', e)
            cam2, mach2, kept_idx = cam, mach, np.arange(len(cam))
        self.kept_idx = kept_idx

        R, t = estimate_rigid_transform(cam2, mach2)
        rep = eval_alignment(R, t, cam, mach)
        self.R, self.t, self.rep = R, t, rep

        print("\n=== 标定结果 (Camera[depth] -> Machine) ===")
        print('R =')
        print(np.array2string(R, formatter={'float_kind': lambda x: f"{x: .6f}"}))
        print('t =', np.array2string(t, formatter={'float_kind': lambda x: f"{x: .6f}"}))
        print('T =')
        print(np.array2string(build_T(R, t), formatter={'float_kind': lambda x: f"{x: .6f}"}))
        print("\n配准误差 (mm):")
        print(f"RMS = {rep['rms_mm']:.4f} | P95 = {rep['p95_mm']:.4f} | Max = {rep['max_mm']:.4f}")

    def run(self):
        self._clicked_pixels: List[Tuple[int,int]] = []

        win_rgb = 'Registration RGB (click here)'
        win_depth = 'Depth'
        win_overlay = 'Overlay'
        cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_overlay, cv2.WINDOW_NORMAL)

        cv2.imshow(win_rgb, self.rgb)
        cv2.imshow(win_depth, self.depth_vis)
        self._redraw()
        cv2.imshow(win_overlay, self.overlay)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                p_cam = self._pixel_to_cam3d(x, y)
                if p_cam is None:
                    print(f'[警告] 点击像素 ({x},{y}) 无有效深度/点云，忽略。')
                    return
                idx = len(self.cam_pts) + 1
                print(f'拾取像素=({x},{y}) -> cam3D={p_cam} [mm]')
                p_mach = self._ask_mach_point(idx)
                if p_mach is None:
                    print('放弃本次添加。')
                    return
                self.cam_pts.append(p_cam)
                self.mach_pts.append(p_mach)
                self._clicked_pixels.append((x, y))
                self._fit_and_report()
                self._redraw()
                cv2.imshow(win_overlay, self.overlay)

        cv2.setMouseCallback(win_rgb, on_mouse)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('z'):  # 撤销
                if self.cam_pts:
                    self.cam_pts.pop(); self.mach_pts.pop(); self._clicked_pixels.pop()
                    self._fit_and_report()
                    self._redraw(); cv2.imshow(win_overlay, self.overlay)
            elif key == ord('c'):  # 清空
                self.cam_pts.clear(); self.mach_pts.clear(); self._clicked_pixels.clear()
                self.R = self.t = self.rep = self.kept_idx = None
                self._redraw(); cv2.imshow(win_overlay, self.overlay)
            elif key == ord('f'):  # 拟合
                self._fit_and_report()
                self._redraw(); cv2.imshow(win_overlay, self.overlay)
            elif key == ord('s'):  # 保存
                if self.R is None or self.t is None:
                    print('尚未拟合，无法保存。请先按 f 进行拟合。')
                else:
                    save_transform(self.R, self.t, self.out_path)
                    print(f'已保存外参到: {self.out_path}（同时生成 .yaml/.json）')

        cv2.destroyAllWindows()


# =================== 主函数 ===================
def main():
    print("=== 3D 外参标定（彩色点击 → 深度取3D → 相机[深度]→机床）===\n")
    out_path = input("外参输出路径（默认 T_cam2machine.npy）：").strip() or "T_cam2machine.npy"
    try:
        trim_percent = float(input("剔除最差残差百分比 0-50（默认 0）：").strip() or "0")
    except Exception:
        trim_percent = 0.0

    grabber = SingleShotGrabber()
    try:
        print("[INFO] 打开设备、配置流…")
        grabber.open()
        print("[INFO] 抓取单帧…")
        frame = None
        for _ in range(60):  # 给设备一点时间出帧
            frame = grabber.grab_one(2000)
            if frame is not None:
                break
            time.sleep(0.03)
        if frame is None:
            raise RuntimeError("未能获取到有效的彩色/深度帧。")

        print("[INFO] 单帧已捕获。请在“Registration RGB”窗口点击添加相机点，每次点击后在终端输入机床XYZ。")
        ui = CalibUI(frame, out_path, trim_percent)
        ui.run()

    finally:
        try:
            grabber.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
