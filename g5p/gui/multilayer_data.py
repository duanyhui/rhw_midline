# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 数据结构定义
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class LayerInfo:
    """单层信息"""
    layer_id: int
    gcode_path: str
    gcode_data: List[str] = field(default_factory=list)
    bias_comp: Optional[Dict] = None
    processing_result: Optional[Dict] = None
    correction_result: Optional[Dict] = None
    timestamp: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, error
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "layer_id": self.layer_id,
            "gcode_path": self.gcode_path,
            "status": self.status,
            "timestamp": self.timestamp,
            "has_bias_comp": self.bias_comp is not None,
            "has_processing_result": self.processing_result is not None,
            "has_correction_result": self.correction_result is not None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerInfo':
        """从字典创建"""
        return cls(
            layer_id=data["layer_id"],
            gcode_path=data["gcode_path"],
            status=data.get("status", "pending"),
            timestamp=data.get("timestamp")
        )

@dataclass
class ProjectConfig:
    """项目配置"""
    project_name: str = "多层加工项目"
    total_layers: int = 1
    layer_thickness_mm: float = 0.5
    auto_next_layer: bool = False
    
    # PLC通信配置
    use_plc: bool = False
    plc_type: str = "tcp"  # tcp, s7
    plc_ip: str = "192.168.1.100"
    plc_port: int = 502
    current_layer_address: str = "DB1.DBD0"  # S7地址
    start_signal_address: str = "DB1.DBX4.0"
    
    # 相机配置
    camera_config: Dict = field(default_factory=lambda: {
        "T_path": "T_cam2machine.npy",
        "roi_mode": "gcode_bounds",
        "pixel_size_mm": 0.8,
        "bounds_margin_mm": 20.0
    })
    
    # 算法配置 
    algorithm_config: Dict = field(default_factory=lambda: {
        "guide_step_mm": 1.0,
        "guide_halfwidth_mm": 6.0,
        "guide_smooth_win": 7,
        "guide_max_offset_mm": 8.0,
        "guide_max_grad_mm_per_mm": 0.08,
        "plane_enable": True,
        "plane_ransac_thresh_mm": 0.8,
        "nearest_qlo": 1.0,
        "nearest_qhi": 99.0,
        "depth_margin_mm": 3.0
    })
    
    # 保存路径
    project_dir: str = "projects"
    backup_enabled: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "project_name": self.project_name,
            "total_layers": self.total_layers,
            "layer_thickness_mm": self.layer_thickness_mm,
            "auto_next_layer": self.auto_next_layer,
            "use_plc": self.use_plc,
            "plc_type": self.plc_type,
            "plc_ip": self.plc_ip,
            "plc_port": self.plc_port,
            "current_layer_address": self.current_layer_address,
            "start_signal_address": self.start_signal_address,
            "camera_config": self.camera_config,
            "algorithm_config": self.algorithm_config,
            "project_dir": self.project_dir,
            "backup_enabled": self.backup_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProjectConfig':
        """从字典创建"""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

@dataclass 
class ProcessingMetrics:
    """处理指标"""
    layer_id: int
    valid_ratio: float = 0.0
    dev_mean: float = 0.0
    dev_median: float = 0.0
    dev_p95: float = 0.0
    plane_inlier_ratio: float = 0.0
    longest_missing_mm: float = 0.0
    processing_time: float = 0.0
    guard_passed: bool = False
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "layer_id": self.layer_id,
            "valid_ratio": self.valid_ratio,
            "dev_mean": self.dev_mean,
            "dev_median": self.dev_median,
            "dev_p95": self.dev_p95,
            "plane_inlier_ratio": self.plane_inlier_ratio,
            "longest_missing_mm": self.longest_missing_mm,
            "processing_time": self.processing_time,
            "guard_passed": self.guard_passed
        }

@dataclass
class CorrectionData:
    """纠偏数据"""
    layer_id: int
    offset_csv_path: str
    corrected_gcode_path: str
    centerline_gcode_path: Optional[str] = None
    preview_image_path: Optional[str] = None
    applied_bias_comp: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "layer_id": self.layer_id,
            "offset_csv_path": self.offset_csv_path,
            "corrected_gcode_path": self.corrected_gcode_path,
            "centerline_gcode_path": self.centerline_gcode_path,
            "preview_image_path": self.preview_image_path,
            "has_applied_bias_comp": self.applied_bias_comp is not None
        }