# -*- coding: utf-8 -*-
"""
偏移数据处理和分批传输模块
处理offset_table.csv文件，进行偏差检测和分批传输到PLC
"""
import csv
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from plc_data_structures import OffsetPoint, PLCDataBlocks, PLCErrorCodes


@dataclass
class OffsetDataConfig:
    """偏移数据处理配置"""
    max_offset_mm: float = 20.0          # 最大允许偏移量(mm)
    max_gradient: float = 0.5            # 最大允许梯度变化
    max_invalid_ratio: float = 0.1       # 最大无效数据比例
    interpolation_method: str = "linear"  # 插值方法
    scale_factor: int = 1000             # PLC数据缩放因子
    enable_safety_check: bool = True     # 启用安全检查
    enable_filtering: bool = True        # 启用数据过滤


@dataclass
class ProcessingResult:
    """数据处理结果"""
    success: bool
    original_count: int
    processed_count: int
    filtered_count: int
    error_count: int
    max_offset: float
    max_gradient: float
    processing_time: float
    error_message: str = ""
    warnings: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class OffsetDataValidator:
    """偏移数据验证器"""
    
    def __init__(self, config: OffsetDataConfig):
        self.config = config
    
    def validate_offset_point(self, point: OffsetPoint, index: int) -> Tuple[bool, str]:
        """验证单个偏移点"""
        # 检查偏移量大小
        magnitude = (point.dx_mm**2 + point.dy_mm**2)**0.5
        if magnitude > self.config.max_offset_mm:
            return False, f"偏移量过大: {magnitude:.3f}mm > {self.config.max_offset_mm}mm (索引: {index})"
        
        # 检查是否为NaN或无穷大
        if not (np.isfinite(point.dx_mm) and np.isfinite(point.dy_mm)):
            return False, f"偏移值无效: dx={point.dx_mm}, dy={point.dy_mm} (索引: {index})"
        
        return True, ""
    
    def validate_gradient(self, points: List[OffsetPoint]) -> Tuple[bool, str, float]:
        """验证梯度变化"""
        if len(points) < 2:
            return True, "", 0.0
        
        max_gradient = 0.0
        for i in range(1, len(points)):
            prev_point = points[i-1]
            curr_point = points[i]
            
            dx_diff = abs(curr_point.dx_mm - prev_point.dx_mm)
            dy_diff = abs(curr_point.dy_mm - prev_point.dy_mm)
            gradient = max(dx_diff, dy_diff)
            
            max_gradient = max(max_gradient, gradient)
            
            if gradient > self.config.max_gradient:
                return False, f"梯度变化过大: {gradient:.3f} > {self.config.max_gradient} (索引: {i-1} -> {i})", max_gradient
        
        return True, "", max_gradient
    
    def validate_data_consistency(self, points: List[OffsetPoint]) -> Tuple[bool, str]:
        """验证数据一致性"""
        if not points:
            return False, "偏移数据为空"
        
        # 统计有效点数
        valid_count = 0
        for point in points:
            if np.isfinite(point.dx_mm) and np.isfinite(point.dy_mm):
                valid_count += 1
        
        valid_ratio = valid_count / len(points)
        if valid_ratio < (1.0 - self.config.max_invalid_ratio):
            return False, f"有效数据比例过低: {valid_ratio:.1%} < {(1.0 - self.config.max_invalid_ratio):.1%}"
        
        return True, ""


class OffsetDataFilter:
    """偏移数据过滤器"""
    
    def __init__(self, config: OffsetDataConfig):
        self.config = config
    
    def filter_outliers(self, points: List[OffsetPoint]) -> Tuple[List[OffsetPoint], List[int]]:
        """过滤异常值"""
        if not points:
            return points, []
        
        # 提取偏移量数据
        dx_values = np.array([p.dx_mm for p in points])
        dy_values = np.array([p.dy_mm for p in points])
        
        # 计算偏移量大小
        magnitudes = np.sqrt(dx_values**2 + dy_values**2)
        
        # 使用3-sigma规则检测异常值
        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        threshold = mean_mag + 3 * std_mag
        
        # 同时检查配置的最大偏移量
        max_threshold = min(threshold, self.config.max_offset_mm)
        
        # 标记异常值
        outlier_indices = []
        filtered_points = []
        
        for i, (point, magnitude) in enumerate(zip(points, magnitudes)):
            if magnitude <= max_threshold and np.isfinite(magnitude):
                filtered_points.append(point)
            else:
                outlier_indices.append(i)
        
        return filtered_points, outlier_indices
    
    def interpolate_missing_values(self, points: List[OffsetPoint], invalid_indices: List[int]) -> List[OffsetPoint]:
        """插值补全缺失值"""
        if not invalid_indices or len(points) < 3:
            return points
        
        result_points = points.copy()
        
        for idx in invalid_indices:
            # 线性插值
            if self.config.interpolation_method == "linear":
                interpolated_point = self._linear_interpolate(points, idx)
            else:
                # 默认使用相邻值
                interpolated_point = self._nearest_neighbor_interpolate(points, idx)
            
            if interpolated_point:
                result_points[idx] = interpolated_point
        
        return result_points
    
    def _linear_interpolate(self, points: List[OffsetPoint], target_idx: int) -> Optional[OffsetPoint]:
        """线性插值"""
        # 找前后有效点
        prev_idx = target_idx - 1
        next_idx = target_idx + 1
        
        # 向前查找有效点
        while prev_idx >= 0:
            point = points[prev_idx]
            if np.isfinite(point.dx_mm) and np.isfinite(point.dy_mm):
                break
            prev_idx -= 1
        
        # 向后查找有效点
        while next_idx < len(points):
            point = points[next_idx]
            if np.isfinite(point.dx_mm) and np.isfinite(point.dy_mm):
                break
            next_idx += 1
        
        # 如果找到前后有效点，进行插值
        if prev_idx >= 0 and next_idx < len(points):
            prev_point = points[prev_idx]
            next_point = points[next_idx]
            
            # 线性插值
            weight = (target_idx - prev_idx) / (next_idx - prev_idx)
            dx_interp = prev_point.dx_mm + weight * (next_point.dx_mm - prev_point.dx_mm)
            dy_interp = prev_point.dy_mm + weight * (next_point.dy_mm - prev_point.dy_mm)
            
            return OffsetPoint(dx_mm=dx_interp, dy_mm=dy_interp)
        
        return None
    
    def _nearest_neighbor_interpolate(self, points: List[OffsetPoint], target_idx: int) -> Optional[OffsetPoint]:
        """最近邻插值"""
        # 查找最近的有效点
        min_distance = float('inf')
        nearest_point = None
        
        for i, point in enumerate(points):
            if i != target_idx and np.isfinite(point.dx_mm) and np.isfinite(point.dy_mm):
                distance = abs(i - target_idx)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = point
        
        return nearest_point


class OffsetDataHandler:
    """偏移数据处理器主类"""
    
    def __init__(self, config: Optional[OffsetDataConfig] = None):
        self.config = config or OffsetDataConfig()
        self.validator = OffsetDataValidator(self.config)
        self.filter = OffsetDataFilter(self.config)
    
    def load_offset_table(self, file_path: str) -> Tuple[List[OffsetPoint], ProcessingResult]:
        """从offset_table.csv加载偏移数据"""
        start_time = time.time()
        
        try:
            points = []
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return [], ProcessingResult(
                    success=False,
                    original_count=0,
                    processed_count=0,
                    filtered_count=0,
                    error_count=0,
                    max_offset=0.0,
                    max_gradient=0.0,
                    processing_time=time.time() - start_time,
                    error_message=f"文件不存在: {file_path}"
                )
            
            # 读取CSV文件
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    try:
                        dx_mm = float(row.get('dx_mm', 0))
                        dy_mm = float(row.get('dy_mm', 0))
                        points.append(OffsetPoint(dx_mm=dx_mm, dy_mm=dy_mm))
                    except (ValueError, TypeError) as e:
                        print(f"警告: 跳过无效行 {row_idx}: {e}")
                        points.append(OffsetPoint(dx_mm=0.0, dy_mm=0.0))  # 用零值占位
            
            original_count = len(points)
            
            # 处理数据
            processed_points, result = self.process_offset_data(points)
            result.original_count = original_count
            result.processing_time = time.time() - start_time
            
            return processed_points, result
            
        except Exception as e:
            return [], ProcessingResult(
                success=False,
                original_count=0,
                processed_count=0,
                filtered_count=0,
                error_count=0,
                max_offset=0.0,
                max_gradient=0.0,
                processing_time=time.time() - start_time,
                error_message=f"加载文件失败: {str(e)}"
            )
    
    def process_offset_data(self, points: List[OffsetPoint]) -> Tuple[List[OffsetPoint], ProcessingResult]:
        """处理偏移数据，包括验证和过滤"""
        if not points:
            return [], ProcessingResult(
                success=False,
                original_count=0,
                processed_count=0,
                filtered_count=0,
                error_count=0,
                max_offset=0.0,
                max_gradient=0.0,
                processing_time=0.0,
                error_message="输入数据为空"
            )
        
        warnings = []
        error_count = 0
        
        # 数据一致性检查
        if self.config.enable_safety_check:
            is_consistent, consistency_msg = self.validator.validate_data_consistency(points)
            if not is_consistent:
                return [], ProcessingResult(
                    success=False,
                    original_count=len(points),
                    processed_count=0,
                    filtered_count=0,
                    error_count=len(points),
                    max_offset=0.0,
                    max_gradient=0.0,
                    processing_time=0.0,
                    error_message=consistency_msg
                )
        
        # 逐点验证
        invalid_indices = []
        for i, point in enumerate(points):
            is_valid, error_msg = self.validator.validate_offset_point(point, i)
            if not is_valid:
                invalid_indices.append(i)
                if self.config.enable_safety_check:
                    warnings.append(error_msg)
                    error_count += 1
        
        # 数据过滤
        processed_points = points.copy()
        filtered_count = 0
        
        if self.config.enable_filtering and invalid_indices:
            # 过滤异常值
            filtered_points, outlier_indices = self.filter.filter_outliers(points)
            
            # 插值补全
            if outlier_indices:
                processed_points = self.filter.interpolate_missing_values(points, outlier_indices)
                filtered_count = len(outlier_indices)
                warnings.append(f"已过滤 {filtered_count} 个异常值并进行插值补全")
            else:
                processed_points = filtered_points
        
        # 梯度验证
        max_gradient = 0.0
        if self.config.enable_safety_check and len(processed_points) > 1:
            is_gradient_ok, gradient_msg, max_gradient = self.validator.validate_gradient(processed_points)
            if not is_gradient_ok:
                if self.config.enable_filtering:
                    warnings.append(f"梯度检查警告: {gradient_msg}")
                else:
                    return [], ProcessingResult(
                        success=False,
                        original_count=len(points),
                        processed_count=0,
                        filtered_count=0,
                        error_count=error_count,
                        max_offset=0.0,
                        max_gradient=max_gradient,
                        processing_time=0.0,
                        error_message=gradient_msg,
                        warnings=warnings
                    )
        
        # 计算最大偏移量
        max_offset = 0.0
        if processed_points:
            offsets = [(p.dx_mm**2 + p.dy_mm**2)**0.5 for p in processed_points]
            max_offset = max(offsets)
        
        return processed_points, ProcessingResult(
            success=True,
            original_count=len(points),
            processed_count=len(processed_points),
            filtered_count=filtered_count,
            error_count=error_count,
            max_offset=max_offset,
            max_gradient=max_gradient,
            processing_time=0.0,
            warnings=warnings
        )
    
    def save_processed_data(self, points: List[OffsetPoint], output_path: str) -> bool:
        """保存处理后的偏移数据"""
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['dx_mm', 'dy_mm'])
                
                for point in points:
                    writer.writerow([f"{point.dx_mm:.6f}", f"{point.dy_mm:.6f}"])
            
            return True
        except Exception as e:
            print(f"保存处理后数据失败: {e}")
            return False
    
    def create_batch_transmission_plan(self, points: List[OffsetPoint]) -> Dict[str, Any]:
        """制定分批传输计划"""
        if not points:
            return {
                "total_points": 0,
                "total_batches": 0,
                "max_points_per_batch": PLCDataBlocks.POINTS_PER_BLOCK * 3,  # 3个数据块
                "batches": []
            }
        
        max_points_per_batch = PLCDataBlocks.POINTS_PER_BLOCK * 3  # 384个点
        total_batches = (len(points) + max_points_per_batch - 1) // max_points_per_batch
        
        batches = []
        for batch_idx in range(total_batches):
            start_idx = batch_idx * max_points_per_batch
            end_idx = min(start_idx + max_points_per_batch, len(points))
            batch_points = points[start_idx:end_idx]
            
            batches.append({
                "batch_index": batch_idx,
                "start_index": start_idx,
                "end_index": end_idx,
                "point_count": len(batch_points),
                "max_offset": max((p.dx_mm**2 + p.dy_mm**2)**0.5 for p in batch_points) if batch_points else 0.0
            })
        
        return {
            "total_points": len(points),
            "total_batches": total_batches,
            "max_points_per_batch": max_points_per_batch,
            "batches": batches
        }


class OffsetDataLoader:
    """偏移数据加载器，集成多种数据源"""
    
    def __init__(self, config: Optional[OffsetDataConfig] = None):
        self.config = config or OffsetDataConfig()
        self.handler = OffsetDataHandler(self.config)
    
    def load_from_layer_output(self, layer_id: int, base_output_dir: str = "output") -> Tuple[List[OffsetPoint], ProcessingResult]:
        """从层输出目录加载偏移数据"""
        output_dir = Path(base_output_dir) / f"layer_{layer_id:02d}_out"
        offset_table_path = output_dir / "offset_table.csv"
        
        if not offset_table_path.exists():
            return [], ProcessingResult(
                success=False,
                original_count=0,
                processed_count=0,
                filtered_count=0,
                error_count=0,
                max_offset=0.0,
                max_gradient=0.0,
                processing_time=0.0,
                error_message=f"偏移表文件不存在: {offset_table_path}"
            )
        
        return self.handler.load_offset_table(str(offset_table_path))
    
    def load_from_correction_data(self, correction_data: Dict[str, Any]) -> Tuple[List[OffsetPoint], ProcessingResult]:
        """从纠偏数据字典加载偏移数据"""
        if "offset_table_path" in correction_data:
            return self.handler.load_offset_table(correction_data["offset_table_path"])
        elif "offset_points" in correction_data:
            # 直接从数据中加载
            offset_points_data = correction_data["offset_points"]
            points = []
            
            for point_data in offset_points_data:
                if isinstance(point_data, dict):
                    dx = point_data.get("dx_mm", 0.0)
                    dy = point_data.get("dy_mm", 0.0)
                    points.append(OffsetPoint(dx_mm=dx, dy_mm=dy))
                elif isinstance(point_data, (list, tuple)) and len(point_data) >= 2:
                    points.append(OffsetPoint(dx_mm=point_data[0], dy_mm=point_data[1]))
            
            return self.handler.process_offset_data(points)
        else:
            return [], ProcessingResult(
                success=False,
                original_count=0,
                processed_count=0,
                filtered_count=0,
                error_count=0,
                max_offset=0.0,
                max_gradient=0.0,
                processing_time=0.0,
                error_message="纠偏数据中未找到偏移信息"
            )
    
    def create_test_data(self, count: int = 1000, max_offset: float = 5.0) -> List[OffsetPoint]:
        """创建测试数据"""
        np.random.seed(42)  # 确保可重复性
        
        points = []
        for i in range(count):
            # 添加一些趋势
            trend_x = 0.01 * i * np.sin(i * 0.1)
            trend_y = 0.005 * i * np.cos(i * 0.15)
            
            # 添加随机噪声
            noise_x = np.random.normal(0, max_offset * 0.1)
            noise_y = np.random.normal(0, max_offset * 0.1)
            
            dx_mm = trend_x + noise_x
            dy_mm = trend_y + noise_y
            
            # 限制在最大偏移量范围内
            magnitude = (dx_mm**2 + dy_mm**2)**0.5
            if magnitude > max_offset:
                scale = max_offset / magnitude
                dx_mm *= scale
                dy_mm *= scale
            
            points.append(OffsetPoint(dx_mm=dx_mm, dy_mm=dy_mm))
        
        return points


if __name__ == "__main__":
    # 测试代码
    print("偏移数据处理模块测试...")
    
    # 创建配置
    config = OffsetDataConfig(
        max_offset_mm=10.0,
        max_gradient=0.3,
        enable_safety_check=True,
        enable_filtering=True
    )
    
    # 创建加载器
    loader = OffsetDataLoader(config)
    
    # 创建测试数据
    test_points = loader.create_test_data(count=500, max_offset=3.0)
    print(f"创建了 {len(test_points)} 个测试偏移点")
    
    # 处理数据
    handler = OffsetDataHandler(config)
    processed_points, result = handler.process_offset_data(test_points)
    
    print(f"处理结果:")
    print(f"  成功: {result.success}")
    print(f"  原始点数: {result.original_count}")
    print(f"  处理点数: {result.processed_count}")
    print(f"  过滤点数: {result.filtered_count}")
    print(f"  错误点数: {result.error_count}")
    print(f"  最大偏移: {result.max_offset:.3f}mm")
    print(f"  最大梯度: {result.max_gradient:.3f}")
    print(f"  处理时间: {result.processing_time:.3f}s")
    
    if result.warnings:
        print(f"  警告:")
        for warning in result.warnings:
            print(f"    - {warning}")
    
    # 创建分批传输计划
    transmission_plan = handler.create_batch_transmission_plan(processed_points)
    print(f"\n分批传输计划:")
    print(f"  总点数: {transmission_plan['total_points']}")
    print(f"  总批次: {transmission_plan['total_batches']}")
    print(f"  每批次最大点数: {transmission_plan['max_points_per_batch']}")
    
    for batch in transmission_plan['batches'][:3]:  # 只显示前3个批次
        print(f"  批次 {batch['batch_index']}: {batch['point_count']} 点 (索引 {batch['start_index']}-{batch['end_index']})")
    
    print("测试完成!")