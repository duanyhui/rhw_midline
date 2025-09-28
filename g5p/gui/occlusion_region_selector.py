#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
occlusion_region_selector.py
遮挡区域框选工具：基于相机深度数据可视化框选机床坐标系下的遮挡区域
结合calibrate_3d_pro.py的相机调用逻辑和align_centerline_to_gcode_pro_edit_max.py的坐标变换
最终输出可直接用于align_centerline_to_gcode_pro_edit_max.py中polys参数的坐标数据

依赖：pcammls, opencv-python, numpy
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import json
import time

try:
    import pcammls
except Exception:
    pcammls = None


# =================== 相机数据采集（基于calibrate_3d_pro.py逻辑） ===================
class OcclusionCameraGrabber:
    """基于calibrate_3d_pro.py的相机调用逻辑，获取深度数据用于遮挡区域可视化"""
    
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
        """打开相机设备并配置流"""
        dev_list = self.cl.ListDevice()
        if len(dev_list) == 0:
            raise RuntimeError("未发现设备。")
        
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
        """关闭相机设备"""
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

    def grab_frame(self, timeout_ms=2000):
        """抓取单帧数据并返回注册彩色图和点云数据"""
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

        # 解码+去畸变彩色
        self.cl.DeviceStreamImageDecode(img_color, self.img_parsed_color)
        self.cl.DeviceStreamDoUndistortion(self.color_calib, self.img_parsed_color, self.img_undistortion_color)

        # 彩色→深度坐标注册
        self.cl.DeviceStreamMapRGBImageToDepthCoordinate(
            self.depth_calib, img_depth, self.scale_unit,
            self.color_calib, self.img_undistortion_color, self.img_registration_color
        )

        # 深度→点云（相机坐标系，mm）
        self.cl.DeviceStreamMapDepthImageToPoint3D(img_depth, self.depth_calib, self.scale_unit, self.pointcloud_data_arr)

        # 转numpy
        registration_rgb = self.img_registration_color.as_nparray().copy()
        p3d_cam = self.pointcloud_data_arr.as_nparray().copy()  # (H,W,3) 相机坐标系 mm

        return dict(registration_rgb=registration_rgb, p3d_cam=p3d_cam)


# =================== 坐标变换（基于align_centerline_to_gcode_pro_edit_max.py） ===================
def load_extrinsic(T_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载外参矩阵"""
    data = np.load(str(T_path), allow_pickle=True).item()
    R = np.asarray(data['R'], dtype=float)
    t = np.asarray(data['t'], dtype=float).reshape(1, 3)
    T = np.asarray(data['T'], dtype=float)
    return R, t, T

def transform_cam_to_machine(P_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """将相机坐标系点云转换为机床坐标系"""
    H, W, _ = P_cam.shape
    P = P_cam.reshape(-1, 3).astype(np.float32)
    # 应用外参变换：P_machine = R @ P_cam + t
    Pm = (R @ P.T).T + t
    return Pm.reshape(H, W, 3)

def compute_topdown_bounds(P_machine: np.ndarray, margin_mm: float = 20.0) -> Tuple[float, float, float, float]:
    """计算机床坐标系下的俯视投影边界"""
    valid_mask = np.isfinite(P_machine).all(axis=2)
    if not np.any(valid_mask):
        return 0.0, 100.0, 0.0, 100.0
    
    X = P_machine[:, :, 0][valid_mask]
    Y = P_machine[:, :, 1][valid_mask]
    
    x_min, x_max = float(np.percentile(X, 1)), float(np.percentile(X, 99))
    y_min, y_max = float(np.percentile(Y, 1)), float(np.percentile(Y, 99))
    
    # 添加边界
    x_min -= margin_mm
    x_max += margin_mm
    y_min -= margin_mm
    y_max += margin_mm
    
    return x_min, x_max, y_min, y_max

def create_topdown_view(P_machine: np.ndarray, pixel_size_mm: float = 0.8) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """创建机床坐标系俯视图"""
    # 计算边界
    x_min, x_max, y_min, y_max = compute_topdown_bounds(P_machine)
    
    # 计算图像尺寸
    width_mm = x_max - x_min
    height_mm = y_max - y_min
    W = int(np.ceil(width_mm / pixel_size_mm))
    H = int(np.ceil(height_mm / pixel_size_mm))
    
    # 限制最大尺寸
    max_pixels = 1200 * 1000
    if W * H > max_pixels:
        scale = np.sqrt(max_pixels / (W * H))
        W = int(W * scale)
        H = int(H * scale)
        pixel_size_mm = width_mm / W
    
    # 创建俯视图（取最高Z值）
    topdown_img = np.full((H, W), np.nan, dtype=np.float32)
    valid_mask = np.isfinite(P_machine).all(axis=2) & (P_machine[:, :, 2] > 0)
    
    if np.any(valid_mask):
        X = P_machine[:, :, 0][valid_mask]
        Y = P_machine[:, :, 1][valid_mask]
        Z = P_machine[:, :, 2][valid_mask]
        
        # 转换为像素坐标（右手系：+X向右，+Y向上）
        px_x = np.clip(((X - x_min) / pixel_size_mm).astype(int), 0, W-1)
        px_y = np.clip(((y_max - Y) / pixel_size_mm).astype(int), 0, H-1)  # Y翻转
        
        # 填充俯视图（取最高Z值）
        for i in range(len(px_x)):
            x, y, z = px_x[i], px_y[i], Z[i]
            if np.isnan(topdown_img[y, x]) or z > topdown_img[y, x]:
                topdown_img[y, x] = z
    
    origin_xy = (x_min, y_min)
    return topdown_img, origin_xy, pixel_size_mm


# =================== 遮挡区域框选界面 ===================
class OcclusionRegionSelector:
    """遮挡区域框选交互界面"""
    
    def __init__(self, topdown_img: np.ndarray, origin_xy: Tuple[float, float], pixel_size_mm: float):
        self.topdown_img = topdown_img
        self.origin_xy = origin_xy
        self.pixel_size_mm = pixel_size_mm
        self.H, self.W = topdown_img.shape
        
        # 创建可视化图像
        self.vis_img = self._create_visualization()
        self.overlay = self.vis_img.copy()
        
        # 框选状态
        self.is_selecting = False
        self.start_point = None
        self.current_rect = None
        self.selected_regions = []  # 存储选中的区域 [(x1,y1,x2,y2), ...]
        
        print("=== 遮挡区域框选工具 ===")
        print("操作说明:")
        print("- 鼠标左键拖拽框选遮挡区域")
        print("- 按 'c' 清除所有选择")
        print("- 按 'z' 撤销上一个选择")
        print("- 按 's' 保存并输出坐标")
        print("- 按 'q' 或 ESC 退出")
        print("- 可以框选多个区域")
        print()
    
    def _create_visualization(self) -> np.ndarray:
        """创建俯视图的可视化图像"""
        # 将深度数据转为8位灰度图
        valid_mask = np.isfinite(self.topdown_img)
        if not np.any(valid_mask):
            return np.zeros((self.H, self.W, 3), dtype=np.uint8)
        
        # 归一化深度值
        z_data = self.topdown_img[valid_mask]
        z_min, z_max = np.percentile(z_data, [5, 95])
        
        normalized = np.zeros_like(self.topdown_img)
        normalized[valid_mask] = np.clip((self.topdown_img[valid_mask] - z_min) / max(z_max - z_min, 1e-6), 0, 1)
        
        # 转为8位并应用颜色映射
        gray_img = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray_img, cv2.COLORMAP_TURBO)
        
        # 将无效区域设为黑色
        colored[~valid_mask] = [0, 0, 0]
        
        return colored
    
    def _pixel_to_machine_xy(self, px: int, py: int) -> Tuple[float, float]:
        """像素坐标转机床坐标"""
        x0, y0 = self.origin_xy
        y1 = y0 + self.H * self.pixel_size_mm
        
        machine_x = x0 + (px + 0.5) * self.pixel_size_mm
        machine_y = y1 - (py + 0.5) * self.pixel_size_mm
        
        return machine_x, machine_y
    
    def _rect_to_machine_poly(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[float, float]]:
        """矩形框转机床坐标系多边形"""
        # 确保坐标顺序正确
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        
        # 四个角点转机床坐标
        corners = [
            self._pixel_to_machine_xy(left, top),     # 左上
            self._pixel_to_machine_xy(right, top),    # 右上
            self._pixel_to_machine_xy(right, bottom), # 右下
            self._pixel_to_machine_xy(left, bottom)   # 左下
        ]
        
        return corners
    
    def _draw_overlay(self):
        """绘制叠加层（选中区域+当前框选）"""
        self.overlay = self.vis_img.copy()
        
        # 绘制已选中的区域
        for i, (x1, y1, x2, y2) in enumerate(self.selected_regions):
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            
            # 绘制矩形框
            cv2.rectangle(self.overlay, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # 添加区域编号
            label = f"Region {i+1}"
            text_pos = (left + 5, top + 20)
            cv2.putText(self.overlay, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 添加半透明填充
            mask = np.zeros(self.overlay.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_SPRING)
            self.overlay = cv2.addWeighted(self.overlay, 0.8, colored_mask, 0.2, 0)
        
        # 绘制当前框选（如果正在框选）
        if self.is_selecting and self.start_point and self.current_rect:
            x1, y1 = self.start_point
            x2, y2 = self.current_rect
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            cv2.rectangle(self.overlay, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # 绘制坐标轴和信息
        self._draw_info_overlay()
    
    def _draw_info_overlay(self):
        """绘制信息叠加层"""
        # 状态信息
        info_text = f"Selected regions: {len(self.selected_regions)}  |  Pixel size: {self.pixel_size_mm:.2f}mm"
        cv2.putText(self.overlay, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.overlay, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # 操作提示
        help_text = "Drag to select | c=clear | z=undo | s=save | q=quit"
        cv2.putText(self.overlay, help_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.overlay, help_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
        
        # 机床坐标系指示
        x0, y0 = self.origin_xy
        coord_text = f"Origin: ({x0:.1f}, {y0:.1f})mm  |  +X→ +Y↑"
        cv2.putText(self.overlay, coord_text, (10, self.H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.overlay, coord_text, (10, self.H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    def _on_mouse(self, event, x, y, flags, param):
        """鼠标事件处理"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_selecting = True
            self.start_point = (x, y)
            self.current_rect = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE and self.is_selecting:
            self.current_rect = (x, y)
            self._draw_overlay()
        
        elif event == cv2.EVENT_LBUTTONUP and self.is_selecting:
            self.is_selecting = False
            if self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # 检查框选区域是否有效（至少10x10像素）
                if abs(x2 - x1) >= 10 and abs(y2 - y1) >= 10:
                    self.selected_regions.append((x1, y1, x2, y2))
                    
                    # 显示机床坐标
                    poly = self._rect_to_machine_poly(x1, y1, x2, y2)
                    print(f"添加区域 {len(self.selected_regions)}: {poly}")
                else:
                    print("框选区域过小，已忽略（最小10x10像素）")
            
            self.start_point = None
            self.current_rect = None
            self._draw_overlay()
    
    def get_machine_polygons(self) -> List[List[Tuple[float, float]]]:
        """获取所有选中区域的机床坐标多边形"""
        polygons = []
        for x1, y1, x2, y2 in self.selected_regions:
            poly = self._rect_to_machine_poly(x1, y1, x2, y2)
            polygons.append(poly)
        return polygons
    
    def run(self) -> List[List[Tuple[float, float]]]:
        """运行交互界面"""
        window_name = "Occlusion Region Selector - Machine Coordinate System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._on_mouse)
        
        self._draw_overlay()
        
        while True:
            cv2.imshow(window_name, self.overlay)
            key = cv2.waitKey(30) & 0xFF
            
            if key in (ord('q'), 27):  # 'q' 或 ESC
                break
            elif key == ord('c'):  # 清除所有选择
                self.selected_regions.clear()
                print("已清除所有选择")
                self._draw_overlay()
            elif key == ord('z'):  # 撤销上一个选择
                if self.selected_regions:
                    removed = self.selected_regions.pop()
                    print(f"已撤销区域 {len(self.selected_regions) + 1}")
                    self._draw_overlay()
            elif key == ord('s'):  # 保存并输出
                if self.selected_regions:
                    polygons = self.get_machine_polygons()
                    self._save_polygons(polygons)
                    break
                else:
                    print("没有选中任何区域")
        
        cv2.destroyAllWindows()
        return self.get_machine_polygons()
    
    def _save_polygons(self, polygons: List[List[Tuple[float, float]]]):
        """保存多边形到文件并打印代码"""
        # 保存到JSON文件
        output_file = "occlusion_regions.json"
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pixel_size_mm": self.pixel_size_mm,
            "origin_xy": self.origin_xy,
            "regions": polygons
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 保存结果到 {output_file} ===")
        print(f"共选择了 {len(polygons)} 个遮挡区域")
        
        # 打印可直接复制到代码中的格式
        print("\n=== 可直接复制到 align_centerline_to_gcode_pro_edit_max.py 的 polys 参数 ===")
        print("polys=[")
        for i, poly in enumerate(polygons):
            formatted_poly = ", ".join([f"({x:.1f},{y:.1f})" for x, y in poly])
            print(f"    [{formatted_poly}],  # 区域 {i+1}")
        print("],")
        
        print(f"\n提示：将上述 polys 数据复制到 align_centerline_to_gcode_pro_edit_max.py 第80-91行的 occlusion 配置中")


# =================== 主函数 ===================
def main():
    print("=== 遮挡区域框选工具 ===")
    print("用于在机床坐标系下可视化框选遮挡区域，输出可直接用于 align_centerline_to_gcode_pro_edit_max.py")
    print()
    
    # 参数配置
    T_path = input("外参文件路径（默认 T_cam2machine.npy）: ").strip() or "T_cam2machine.npy"
    if not Path(T_path).exists():
        print(f"错误：外参文件 {T_path} 不存在")
        return
    
    try:
        pixel_size_mm = float(input("俯视图像素大小（mm，默认 0.8）: ").strip() or "0.8")
    except ValueError:
        pixel_size_mm = 0.8
    
    # 加载外参
    try:
        R, t, T = load_extrinsic(T_path)
        print(f"成功加载外参: {T_path}")
    except Exception as e:
        print(f"错误：无法加载外参文件 {T_path}: {e}")
        return
    
    # 初始化相机
    grabber = OcclusionCameraGrabber()
    try:
        print("正在打开相机设备...")
        grabber.open()
        
        print("正在抓取相机数据...")
        # 等待几帧让相机稳定
        frame_data = None
        for attempt in range(10):
            frame_data = grabber.grab_frame(timeout_ms=3000)
            if frame_data is not None:
                break
            time.sleep(0.1)
        
        if frame_data is None:
            print("错误：无法获取相机数据")
            return
        
        print("成功获取相机数据，正在处理...")
        
        # 转换相机坐标到机床坐标
        P_cam = frame_data['p3d_cam']  # (H,W,3) 相机坐标系
        P_machine = transform_cam_to_machine(P_cam, R, t)
        
        # 创建俯视图
        print("正在生成俯视图...")
        topdown_img, origin_xy, actual_pixel_size = create_topdown_view(P_machine, pixel_size_mm)
        
        print(f"俯视图尺寸: {topdown_img.shape}")
        print(f"实际像素大小: {actual_pixel_size:.3f}mm")
        print(f"坐标原点: ({origin_xy[0]:.1f}, {origin_xy[1]:.1f})mm")
        print()
        
        # 启动选择界面
        selector = OcclusionRegionSelector(topdown_img, origin_xy, actual_pixel_size)
        polygons = selector.run()
        
        print(f"\n=== 框选完成 ===")
        print(f"共选择了 {len(polygons)} 个遮挡区域")
        
        if polygons:
            print("\n=== 最终结果（可直接复制使用） ===")
            print("occlusion=dict(")
            print("    enable=True,")
            print("    polys=[")
            for i, poly in enumerate(polygons):
                formatted_poly = ", ".join([f"({x:.1f},{y:.1f})" for x, y in poly])
                print(f"        [{formatted_poly}],  # 区域 {i+1}")
            print("    ],")
            print("    dilate_mm=3.0,")
            print("    synthesize_band=True,")
            print("    band_halfwidth_mm=None")
            print("),")
            print("\n请将上述配置复制到 align_centerline_to_gcode_pro_edit_max.py 的第80-91行")
        
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            grabber.close()
        except Exception:
            pass
        print("\n程序结束")


if __name__ == '__main__':
    main()