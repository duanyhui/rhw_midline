#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
occlusion_selector_standalone.py
独立的遮挡区域框选工具 - 不依赖实时相机数据的版本
可以使用现有的点云数据或示例数据进行遮挡区域框选
最终输出可直接用于 align_centerline_to_gcode_pro_edit_max.py

使用方法：
1. 如果有现有的机床坐标点云数据(.npy)，可以直接加载
2. 如果没有，程序会生成示例数据用于演示框选功能
3. 支持多个区域框选，输出标准格式的坐标数据
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import json
import time
import sys


# =================== 坐标变换和数据处理 ===================
def load_extrinsic(T_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载外参矩阵"""
    try:
        data = np.load(str(T_path), allow_pickle=True).item()
        R = np.asarray(data['R'], dtype=float)
        t = np.asarray(data['t'], dtype=float).reshape(1, 3)
        T = np.asarray(data['T'], dtype=float)
        return R, t, T
    except Exception as e:
        print(f"警告：无法加载外参文件 {T_path}: {e}")
        print("将使用示例数据进行演示")
        # 返回示例外参（单位矩阵）
        R = np.eye(3, dtype=float)
        t = np.zeros((1, 3), dtype=float)
        T = np.eye(4, dtype=float)
        return R, t, T

def generate_sample_machine_points(x_range=(-100, 200), y_range=(-50, 150), z_range=(0, 20), n_points=50000):
    """生成示例机床坐标系点云数据"""
    print("生成示例机床坐标系点云数据...")
    
    # 生成基础平面
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    z = np.random.uniform(z_range[0], z_range[1], n_points//2)
    
    # 添加一些高度变化模拟工件
    z2 = np.random.uniform(z_range[1], z_range[1]+10, n_points//2)
    z_all = np.concatenate([z, z2])
    
    # 组合点云
    points = np.column_stack([x, y, z_all])
    
    # 添加一些噪声和缺失
    valid_mask = np.random.random(len(points)) > 0.05  # 5%缺失
    points = points[valid_mask]
    
    print(f"生成了 {len(points)} 个点的示例数据")
    print(f"X范围: [{x_range[0]}, {x_range[1]}]mm")
    print(f"Y范围: [{y_range[0]}, {y_range[1]}]mm") 
    print(f"Z范围: [{z_range[0]}, {z_range[1]+10}]mm")
    
    return points

def compute_topdown_bounds(points: np.ndarray, margin_mm: float = 20.0) -> Tuple[float, float, float, float]:
    """计算俯视投影边界"""
    if len(points) == 0:
        return 0.0, 100.0, 0.0, 100.0
    
    X, Y = points[:, 0], points[:, 1]
    
    x_min, x_max = float(np.percentile(X, 1)), float(np.percentile(X, 99))
    y_min, y_max = float(np.percentile(Y, 1)), float(np.percentile(Y, 99))
    
    # 添加边界
    x_min -= margin_mm
    x_max += margin_mm
    y_min -= margin_mm
    y_max += margin_mm
    
    return x_min, x_max, y_min, y_max

def create_topdown_view(points: np.ndarray, pixel_size_mm: float = 0.8) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """从点云创建俯视图"""
    if len(points) == 0:
        return np.zeros((100, 100), dtype=np.float32), (0.0, 0.0), pixel_size_mm
    
    # 计算边界
    x_min, x_max, y_min, y_max = compute_topdown_bounds(points)
    
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
    
    # 转换为像素坐标（右手系：+X向右，+Y向上）
    px_x = np.clip(((points[:, 0] - x_min) / pixel_size_mm).astype(int), 0, W-1)
    px_y = np.clip(((y_max - points[:, 1]) / pixel_size_mm).astype(int), 0, H-1)  # Y翻转
    
    # 填充俯视图（取最高Z值）
    for i in range(len(points)):
        x, y, z = px_x[i], px_y[i], points[i, 2]
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
        
        print("\n=== 遮挡区域框选工具 ===")
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
            # 如果没有有效数据，创建示例图案
            demo_img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            cv2.putText(demo_img, "No valid depth data", (20, self.H//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(demo_img, "Use sample data for demo", (20, self.H//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            return demo_img
        
        # 归一化深度值
        z_data = self.topdown_img[valid_mask]
        z_min, z_max = np.percentile(z_data, [5, 95])
        
        normalized = np.zeros_like(self.topdown_img)
        normalized[valid_mask] = np.clip((self.topdown_img[valid_mask] - z_min) / max(z_max - z_min, 1e-6), 0, 1)
        
        # 转为8位并应用颜色映射
        gray_img = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray_img, cv2.COLORMAP_TURBO)
        
        # 将无效区域设为深灰色
        colored[~valid_mask] = [50, 50, 50]
        
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
        
        # 绘制坐标轴指示器
        arrow_len = 50
        origin_px = (60, self.H - 60)
        # X轴（红色）
        cv2.arrowedLine(self.overlay, origin_px, (origin_px[0] + arrow_len, origin_px[1]), (0, 0, 255), 3)
        cv2.putText(self.overlay, "+X", (origin_px[0] + arrow_len + 5, origin_px[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Y轴（绿色）
        cv2.arrowedLine(self.overlay, origin_px, (origin_px[0], origin_px[1] - arrow_len), (0, 255, 0), 3)
        cv2.putText(self.overlay, "+Y", (origin_px[0] - 15, origin_px[1] - arrow_len - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
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
                    print(f"添加区域 {len(self.selected_regions)}:")
                    print(f"  像素坐标: ({x1},{y1}) -> ({x2},{y2})")
                    print(f"  机床坐标: {[(f'{x:.1f},{y:.1f}') for x,y in poly]}")
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
        cv2.resizeWindow(window_name, min(1200, self.W), min(800, self.H))
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
                    return polygons
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
        print("\n" + "="*60)
        print("=== 可直接复制到 align_centerline_to_gcode_pro_edit_max.py ===")
        print("="*60)
        print("# 将以下内容替换第80-91行的 occlusion 配置:")
        print()
        print("occlusion=dict(")
        print("    enable=True,  # 启用遮挡处理")
        print("    polys=[")
        for i, poly in enumerate(polygons):
            formatted_poly = ", ".join([f"({x:.1f},{y:.1f})" for x, y in poly])
            print(f"        [{formatted_poly}],  # 区域 {i+1}")
        print("    ],")
        print("    dilate_mm=3.0,  # 安全扩张，确保完全覆盖遮挡")
        print("    synthesize_band=True,  # 是否在遮挡区内按G代码合成环带掩码")
        print("    band_halfwidth_mm=None  # None=自动从可见区估计；或手工指定半宽")
        print("),")
        print()
        print("="*60)


# =================== 主函数 ===================
def main():
    print("=== 遮挡区域框选工具（独立版本）===")
    print("用于在机床坐标系下可视化框选遮挡区域")
    print("输出格式可直接用于 align_centerline_to_gcode_pro_edit_max.py")
    print()
    
    # 参数配置
    mode = input("选择数据源 [1]现有点云文件 [2]示例数据 (默认2): ").strip() or "2"
    
    machine_points = None
    
    if mode == "1":
        # 尝试加载现有数据
        data_path = input("点云数据文件路径 (.npy格式): ").strip()
        if data_path and Path(data_path).exists():
            try:
                data = np.load(data_path)
                if data.ndim == 3 and data.shape[2] >= 3:
                    # (H,W,3) 格式
                    H, W, _ = data.shape
                    valid_mask = np.isfinite(data).all(axis=2)
                    machine_points = data[valid_mask]
                    print(f"成功加载点云数据: {machine_points.shape}")
                elif data.ndim == 2 and data.shape[1] >= 3:
                    # (N,3) 格式
                    machine_points = data[:, :3]
                    print(f"成功加载点云数据: {machine_points.shape}")
                else:
                    print("数据格式不支持，使用示例数据")
                    machine_points = generate_sample_machine_points()
            except Exception as e:
                print(f"加载数据失败: {e}")
                machine_points = generate_sample_machine_points()
        else:
            print("文件不存在，使用示例数据")
            machine_points = generate_sample_machine_points()
    else:
        # 使用示例数据
        machine_points = generate_sample_machine_points()
    
    try:
        pixel_size_mm = float(input("俯视图像素大小（mm，默认 0.8）: ").strip() or "0.8")
    except ValueError:
        pixel_size_mm = 0.8
    
    try:
        print("\n正在生成俯视图...")
        topdown_img, origin_xy, actual_pixel_size = create_topdown_view(machine_points, pixel_size_mm)
        
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
            print("\n框选结果已保存到 occlusion_regions.json")
            print("请查看控制台输出的配置代码，可直接复制使用")
        else:
            print("\n未选择任何区域")
            
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
    
    print("\n程序结束")


if __name__ == '__main__':
    main()