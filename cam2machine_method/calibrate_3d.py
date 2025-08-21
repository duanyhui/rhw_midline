#!/usr/bin/env python3
"""
3D 外参标定（相机 -> 机床坐标系）— 交互版 + PCamMLS 相机接入
---------------------------------------------------------

- 终端交互式选择；也可修改下方 CONFIG 默认值。
- 支持三种采点模式：文件 Nx3、离线图片点击、PCamMLS 实时相机点击。
- 输出：T_cam2machine.npy（含 R、t、T 4x4，单位 mm）及 .yaml/.json；控制台打印 RMS/P95/Max。
- 建议采 ≥8–12 个、覆盖工作区的点对；可用 trim_percent 剔除最差残差后再拟合。
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import json
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pcammls  # 使用命名空间引用以避免命名污染
except Exception:
    pcammls = None


CONFIG = {
    'default_out_path': 'T_cam2machine.npy',
    'trim_percent': 0.0,
}


def load_points_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if path.suffix.lower() == '.npy':
        pts = np.load(path)
    else:
        pts = np.genfromtxt(path, delimiter=',')
        if pts.ndim == 1:
            pts = np.expand_dims(pts, 0)
    pts = np.asarray(pts, dtype=float)
    if pts.shape[-1] != 3:
        raise ValueError(f"期望 Nx3 点集，得到 {pts.shape}")
    return pts


def save_transform(R: np.ndarray, t: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {'R': R.astype(float), 't': t.astype(float), 'T': build_T(R, t).astype(float)}
    np.save(out_path, data)
    try:
        import yaml  # 可选
        with open(out_path.with_suffix('.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump({k: v.tolist() for k, v in data.items()}, f, sort_keys=False, allow_unicode=True)
    except Exception:
        with open(out_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump({k: v.tolist() for k, v in data.items()}, f, indent=2, ensure_ascii=False)


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
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def eval_alignment(R: np.ndarray, t: np.ndarray, cam_pts: np.ndarray, mach_pts: np.ndarray) -> dict:
    pred = apply_transform(R, t, cam_pts)
    err = np.linalg.norm(pred - mach_pts, axis=1)
    return {'rms_mm': float(np.sqrt(np.mean(err ** 2))), 'p95_mm': float(np.percentile(err, 95)), 'max_mm': float(np.max(err)), 'errors_mm': err}


def trim_outliers_once(cam_pts: np.ndarray, mach_pts: np.ndarray, trim_percent: float):
    R0, t0 = estimate_rigid_transform(cam_pts, mach_pts)
    pred = apply_transform(R0, t0, cam_pts)
    err = np.linalg.norm(pred - mach_pts, axis=1)
    k = int(np.floor(len(err) * (1.0 - trim_percent / 100.0)))
    if k < 3:
        raise ValueError("修剪后剩余点数不足 3 个。")
    idx = np.argsort(err)[:k]
    return cam_pts[idx], mach_pts[idx], idx


def read_intrinsics(json_path: str | Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        intr = json.load(f)
    for k in ['fx', 'fy', 'cx', 'cy']:
        if k not in intr:
            raise ValueError(f"缺少相机内参字段: {k}")
    intr.setdefault('depth_scale', 1.0)
    intr.setdefault('invalid_depth', None)
    return intr


def read_image_any(path: str | Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("需要安装 OpenCV (cv2)。")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def read_depth_any(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() == '.npy':
        d = np.load(p)
        if d.ndim != 2:
            raise ValueError("深度 .npy 必须是 HxW 数组")
        return d.astype(float)
    if cv2 is None:
        raise RuntimeError("需要安装 OpenCV (cv2)。")
    if p.suffix.lower() == '.exr':
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"无法读取深度: {path}")
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(float)
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"无法读取深度: {path}")
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return d.astype(float)


def backproject(u: float, v: float, z_mm: float, intr: dict) -> np.ndarray:
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    X = (u - cx) * z_mm / fx
    Y = (v - cy) * z_mm / fy
    return np.array([X, Y, z_mm], dtype=float)


class ClickCollectorImages:
    def __init__(self, rgb: np.ndarray, depth: np.ndarray, intr: dict):
        self.rgb = rgb.copy()
        self.depth = depth
        self.intr = intr
        self.points_cam: List[np.ndarray] = []
        self.overlay = rgb.copy()
        self.invalid_depth = intr.get('invalid_depth', None)
        self.depth_scale = float(intr.get('depth_scale', 1.0))

    def on_mouse(self, event, x, y, flags, param):
        if event == 1:
            z_raw = float(self.depth[y, x])
            if self.invalid_depth is not None and z_raw == self.invalid_depth:
                print(f"[警告] ({x},{y}) 深度无效，跳过。")
                return
            z_mm = z_raw * self.depth_scale
            if z_mm <= 0:
                print(f"[警告] ({x},{y}) 深度非正值: {z_mm}，跳过。")
                return
            p_cam = backproject(x, y, z_mm, self.intr)
            print(f"拾取像素=({x},{y}) -> cam3D={p_cam}")
            self.points_cam.append(p_cam)
            cv2.circle(self.overlay, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(self.overlay, str(len(self.points_cam)), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        if cv2 is None:
            raise RuntimeError("需要安装 OpenCV (cv2)。")
        win = '离线图片点击采点 (q 结束)'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            cv2.imshow(win, self.overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
        cv2.destroyWindow(win)
        return np.array(self.points_cam, dtype=float)


class ClickCollectorPCamMLS:
    def __init__(self):
        if pcammls is None:
            raise RuntimeError('未安装 pcammls，无法使用相机模式。')
        if cv2 is None:
            raise RuntimeError('相机模式需要显示窗口，请安装 OpenCV (cv2)。')
        self.cl = pcammls.PercipioSDK()
        self.handle = None
        self.current_p3d = None
        self.points_cam: List[np.ndarray] = []
        self.img_registration_depth = None
        self.depth_calib_data = None
        self.scale_unit = 1.0

    def open_device(self):
        dev_list = self.cl.ListDevice()
        if len(dev_list) == 0:
            raise RuntimeError('未发现任何设备。')
        print('检测到以下设备:')
        for idx, dev in enumerate(dev_list):
            print(f"  {idx} -- {dev.id}\t{dev.iface.id}")
        selected_idx = 0 if len(dev_list) == 1 else int(input('请选择设备索引: '))
        if selected_idx < 0 or selected_idx >= len(dev_list):
            raise RuntimeError('无效的设备索引。')
        sn = dev_list[selected_idx].id

        handle = self.cl.Open(sn)
        if not self.cl.isValidHandle(handle):
            err = self.cl.TYGetLastErrorCodedescription()
            raise RuntimeError(f'打开设备失败: {err}')
        self.handle = handle

        class _DevEvent(pcammls.DeviceEvent):
            def __init__(self):
                pcammls.DeviceEvent.__init__(self)
                self._offline = False
            def run(self, h, eventID):
                if eventID == pcammls.TY_EVENT_DEVICE_OFFLINE:
                    print('=== 事件: 设备离线 ===')
                    self._offline = True
                return 0
            def IsOffline(self):
                return self._offline
        self.event = _DevEvent()
        if hasattr(self.cl, 'DeviceRegisterCallBackEvent'):
            self.cl.DeviceRegisterCallBackEvent(self.event)
        elif hasattr(self.cl, 'DeviceRegiststerCallBackEvent'):
            self.cl.DeviceRegiststerCallBackEvent(self.event)

        color_fmt_list = self.cl.DeviceStreamFormatDump(handle, pcammls.PERCIPIO_STREAM_COLOR)
        depth_fmt_list = self.cl.DeviceStreamFormatDump(handle, pcammls.PERCIPIO_STREAM_DEPTH)
        if len(depth_fmt_list) == 0:
            raise RuntimeError('设备无深度流。')
        self.cl.DeviceStreamFormatConfig(handle, pcammls.PERCIPIO_STREAM_DEPTH, depth_fmt_list[0])
        if len(color_fmt_list) > 0:
            self.cl.DeviceStreamFormatConfig(handle, pcammls.PERCIPIO_STREAM_COLOR, color_fmt_list[0])

        err = self.cl.DeviceLoadDefaultParameters(handle)
        if err:
            print('加载默认参数失败，继续尝试。')
        self.scale_unit = self.cl.DeviceReadCalibDepthScaleUnit(handle)
        print(f'depth scale unit: {self.scale_unit}')
        self.depth_calib_data = self.cl.DeviceReadCalibData(handle, pcammls.PERCIPIO_STREAM_DEPTH)

        enable_mask = pcammls.PERCIPIO_STREAM_DEPTH
        if len(color_fmt_list) > 0:
            enable_mask |= pcammls.PERCIPIO_STREAM_COLOR
        err = self.cl.DeviceStreamEnable(handle, enable_mask)
        if err:
            raise RuntimeError(f'DeviceStreamEnable 失败: {err}')

        self.cl.DeviceStreamOn(handle)
        self.img_registration_depth = pcammls.image_data()
        self.pointcloud_data_arr = pcammls.pointcloud_data_list()

    def close_device(self):
        if self.handle is not None:
            try:
                self.cl.DeviceStreamOff(self.handle)
            except Exception:
                pass
            try:
                self.cl.Close(self.handle)
            except Exception:
                pass
            self.handle = None
        if cv2 is not None:
            cv2.destroyAllWindows()

    def _update_frame(self):
        image_list = self.cl.DeviceStreamRead(self.handle, 2000)
        img_depth = None
        for i in range(len(image_list)):
            frame = image_list[i]
            if frame.streamID == pcammls.PERCIPIO_STREAM_DEPTH:
                img_depth = frame
        if img_depth is None:
            return False
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib_data, self.scale_unit, self.pointcloud_data_arr)
        self.current_p3d = self.pointcloud_data_arr.as_nparray()
        self.cl.DeviceStreamDepthRender(img_depth, self.img_registration_depth)
        return True

    def run(self):
        self.open_device()
        win = '相机深度渲染（在此窗口左键添加点，q 结束）'
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        clicked_pixels: List[tuple[int,int]] = []

        def on_mouse(event, x, y, flags, param):
            if event == 1:
                if self.current_p3d is None:
                    print('[警告] 当前无点云数据。')
                    return
                H, W = self.current_p3d.shape[:2]
                if not (0 <= x < W and 0 <= y < H):
                    print('[警告] 点击超出范围。')
                    return
                p_cam = self.current_p3d[y, x].astype(float)
                if not np.isfinite(p_cam).all() or p_cam[2] <= 0:
                    print('[警告] 无效 3D 点（可能无深度）。')
                    return
                print(f'拾取像素=({x},{y}) -> cam3D={p_cam}')
                self.points_cam.append(p_cam)
                clicked_pixels.append((x, y))
        cv2.setMouseCallback(win, on_mouse)
        print('提示：点击窗口添加相机点；每次点击后在终端输入对应机床 XYZ；按 q 结束。')

        mach_list: List[List[float]] = []
        try:
            while True:
                if self.event.IsOffline():
                    print('[错误] 设备离线。')
                    break
                if not self._update_frame():
                    continue
                depth_vis = self.img_registration_depth.as_nparray().copy()
                for i, (x, y) in enumerate(clicked_pixels):
                    cv2.circle(depth_vis, (x, y), 5, (0, 255, 0), 2)
                    cv2.putText(depth_vis, str(i+1), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win, depth_vis)
                key = cv2.waitKey(10) & 0xFF
                if key in (ord('q'), 27):
                    break
                while len(mach_list) < len(self.points_cam):
                    idx = len(mach_list) + 1
                    s = input(f'点 #{idx} 机床 XYZ（mm），例如 125.0 40.5 0.0：').strip()
                    try:
                        x, y, z = map(float, s.split())
                        mach_list.append([x, y, z])
                    except Exception:
                        print('格式不正确，请按：X Y Z 输入。')
        finally:
            self.close_device()
        cam_pts = np.array(self.points_cam, dtype=float)
        mach_pts = np.array(mach_list, dtype=float)
        return cam_pts, mach_pts


def prompt_choice() -> int:
    print("\n请选择标定模式：")
    print('  1) 从文件加载 Nx3 点对 (files)')
    print('  2) 点击采点（离线图片+深度+内参）')
    print('  3) 点击采点（PCamMLS 实时相机）')
    while True:
        s = input('输入 1/2/3：').strip()
        if s in ('1', '2', '3'):
            return int(s)
        print('无效选择，请重试。')


def main():
    out_path = input(f"外参输出路径（默认 {CONFIG['default_out_path']}）：").strip() or CONFIG['default_out_path']
    try:
        trim_percent = float(input(f"剔除最差残差百分比 0-50（默认 {CONFIG['trim_percent']}）：").strip() or CONFIG['trim_percent'])
    except Exception:
        trim_percent = CONFIG['trim_percent']

    mode = prompt_choice()

    if mode == 1:
        cam_file = input('相机 3D 点文件路径 (.npy/.csv)：').strip()
        mach_file = input('机床 3D 点文件路径 (.npy/.csv)：').strip()
        cam_pts = load_points_file(cam_file)
        mach_pts = load_points_file(mach_file)

    elif mode == 2:
        if cv2 is None:
            print('需要安装 OpenCV (cv2) 才能使用该模式。')
            return
        rgb_path = input('RGB 图像路径：').strip()
        depth_path = input('深度图路径 (npy/exr/png)：').strip()
        intr_json = input('内参 JSON 路径（含 fx,fy,cx,cy,depth_scale）：').strip()
        intr = read_intrinsics(intr_json)
        rgb = read_image_any(rgb_path)
        depth = read_depth_any(depth_path)
        cam_pts = ClickCollectorImages(rgb, depth, intr).run()
        if len(cam_pts) < 3:
            print('标定至少需要 3 个点。')
            return
        print('请为每个点击点输入机床 XYZ（mm）')
        mach_list = []
        for i in range(len(cam_pts)):
            while True:
                s = input(f'点 #{i+1} 机床 XYZ：').strip()
                try:
                    x, y, z = map(float, s.split())
                    mach_list.append([x, y, z])
                    break
                except Exception:
                    print('格式不正确，请按：X Y Z 输入。')
        mach_pts = np.array(mach_list, dtype=float)

    else:
        collector = ClickCollectorPCamMLS()
        cam_pts, mach_pts = collector.run()
        if len(cam_pts) < 3:
            print('标定至少需要 3 个点。')
            return

    if trim_percent and trim_percent > 0:
        cam_pts, mach_pts, kept_idx = trim_outliers_once(cam_pts, mach_pts, trim_percent)
        print(f'已剔除最差 {trim_percent}% 残差点；保留 {len(kept_idx)} 个点。')

    R, t = estimate_rigid_transform(cam_pts, mach_pts)
    rep = eval_alignment(R, t, cam_pts, mach_pts)

    print("\n=== 标定结果 (Camera -> Machine) ===")
    print('R =')
    print(np.array2string(R, formatter={'float_kind': lambda x: f"{x: .6f}"}))
    print('t =', np.array2string(t, formatter={'float_kind': lambda x: f"{x: .6f}"}))
    print('T =')
    print(np.array2string(build_T(R, t), formatter={'float_kind': lambda x: f"{x: .6f}"}))
    print("\n配准误差 (mm):")
    print(f"RMS = {rep['rms_mm']:.4f} | P95 = {rep['p95_mm']:.4f} | Max = {rep['max_mm']:.4f}")
    save_transform(R, t, out_path)
    print(f"\n已保存外参到: {out_path}（同时生成 .yaml/.json）")


if __name__ == '__main__':
    main()
