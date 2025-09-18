# -*- coding: utf-8 -*-
"""
高级参数调节界面 - 完整版本
提供丰富的参数调节和实时预览功能
"""
import os
import sys
import json
import time
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QPixmap

# 本地导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from controller import AlignController, GUIConfig, np_to_qimage
import numpy as np
import cv2

class ParameterPreviewThread(QThread):
    """参数预览线程"""
    preview_ready = pyqtSignal(dict)  # 预览结果
    preview_failed = pyqtSignal(str)  # 预览失败
    
    def __init__(self, controller: AlignController, params: Dict):
        super().__init__()
        self.controller = controller
        self.params = params
        
    def run(self):
        """运行预览"""
        try:
            # 应用参数到控制器
            for key, value in self.params.items():
                if hasattr(self.controller.cfg, key):
                    setattr(self.controller.cfg, key, value)
                    
            # 处理单帧
            result = self.controller.process_single_frame()
            self.preview_ready.emit(result)
            
        except Exception as e:
            self.preview_failed.emit(str(e))

class AdvancedParametersDialog(QDialog):
    """高级参数调节对话框"""
    
    parameters_applied = pyqtSignal(dict)  # 参数应用信号
    
    def __init__(self, controller: AlignController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.original_config = self._backup_config()
        self.preview_timer = QTimer()
        self.preview_thread = None
        self.current_preview_result = None
        
        self.setup_ui()
        self.setup_connections()
        self.load_current_parameters()
        
        # 设置预览定时器
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.start_preview)
        
    def _backup_config(self):
        """备份当前配置"""
        return {
            'roi_mode': self.controller.cfg.roi_mode,
            'cam_roi_xywh': self.controller.cfg.cam_roi_xywh,
            'pixel_size_mm': self.controller.cfg.pixel_size_mm,
            'bounds_margin_mm': self.controller.cfg.bounds_margin_mm,
            'plane_enable': self.controller.cfg.plane_enable,
            'plane_ransac_thresh_mm': self.controller.cfg.plane_ransac_thresh_mm,
            'plane_ransac_iters': self.controller.cfg.plane_ransac_iters,
            'z_select': self.controller.cfg.z_select,
            'nearest_qlo': self.controller.cfg.nearest_qlo,
            'nearest_qhi': self.controller.cfg.nearest_qhi,
            'depth_margin_mm': self.controller.cfg.depth_margin_mm,
            'morph_open': self.controller.cfg.morph_open,
            'morph_close': self.controller.cfg.morph_close,
            'min_component_area_px': self.controller.cfg.min_component_area_px,
            'draw_normal_probes': self.controller.cfg.draw_normal_probes,
            'arrow_stride': self.controller.cfg.arrow_stride,
            'debug_normals_window': self.controller.cfg.debug_normals_window,
            'debug_normals_stride': self.controller.cfg.debug_normals_stride,
            'debug_normals_max': self.controller.cfg.debug_normals_max,
            'debug_normals_len_mm': self.controller.cfg.debug_normals_len_mm,
            'debug_normals_text': self.controller.cfg.debug_normals_text,
            'occ_enable': self.controller.cfg.occ_enable,
            'occ_dilate_mm': self.controller.cfg.occ_dilate_mm,
            'occ_polys': self.controller.cfg.occ_polys,
            'occ_synthesize_band': self.controller.cfg.occ_synthesize_band,
            'occ_band_halfwidth_mm': self.controller.cfg.occ_band_halfwidth_mm,
            'guide_step_mm': self.controller.cfg.guide_step_mm,
            'guide_halfwidth_mm': self.controller.cfg.guide_halfwidth_mm,
            'guide_smooth_win': self.controller.cfg.guide_smooth_win,
            'guide_max_offset_mm': self.controller.cfg.guide_max_offset_mm,
            'guide_max_grad_mm_per_mm': self.controller.cfg.guide_max_grad_mm_per_mm,
            'max_gap_pts': self.controller.cfg.max_gap_pts,
            'curvature_adaptive': self.controller.cfg.curvature_adaptive,
            'curvature_gamma': self.controller.cfg.curvature_gamma,
            'min_smooth_win': self.controller.cfg.min_smooth_win,
            'corner_ignore_enable': self.controller.cfg.corner_ignore_enable,
            'corner_angle_thr_deg': self.controller.cfg.corner_angle_thr_deg,
            'corner_ignore_span_mm': self.controller.cfg.corner_ignore_span_mm,
        }
        
    def setup_ui(self):
        """设置UI界面"""
        self.setWindowTitle("高级参数调节 - 实时预览")
        self.setModal(True)
        self.resize(1400, 900)
        
        # 主布局
        main_layout = QHBoxLayout(self)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # 左侧：参数面板
        params_panel = self.create_parameters_panel()
        main_splitter.addWidget(params_panel)
        
        # 右侧：预览面板
        preview_panel = self.create_preview_panel()
        main_splitter.addWidget(preview_panel)
        
        # 设置分割比例
        main_splitter.setStretchFactor(0, 1)  # 参数面板
        main_splitter.setStretchFactor(1, 2)  # 预览面板占更大空间
        main_splitter.setSizes([450, 950])
        
    def create_parameters_panel(self):
        """创建参数面板"""
        panel = QWidget()
        panel.setMaximumWidth(500)
        panel_layout = QVBoxLayout(panel)
        
        # 参数选项卡
        self.params_tabs = QTabWidget()
        
        # ROI投影参数
        self.params_tabs.addTab(self.create_roi_tab(), "ROI投影")
        
        # 最近表面参数
        self.params_tabs.addTab(self.create_surface_tab(), "最近表面")
        
        # 法线可视化参数
        self.params_tabs.addTab(self.create_visualization_tab(), "法线可视化")
        
        # 遮挡区域参数
        self.params_tabs.addTab(self.create_occlusion_tab(), "遮挡区域")
        
        # 引导中心线参数
        self.params_tabs.addTab(self.create_guide_tab(), "引导中心线")
        
        panel_layout.addWidget(self.params_tabs)
        
        # 底部按钮
        buttons_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_parameters)
        buttons_layout.addWidget(self.btn_reset)
        
        self.btn_load_preset = QPushButton("加载预设")
        self.btn_load_preset.clicked.connect(self.load_preset)
        buttons_layout.addWidget(self.btn_load_preset)
        
        self.btn_save_preset = QPushButton("保存预设")
        self.btn_save_preset.clicked.connect(self.save_preset)
        buttons_layout.addWidget(self.btn_save_preset)
        
        buttons_layout.addStretch()
        
        self.btn_apply = QPushButton("应用")
        self.btn_apply.clicked.connect(self.apply_parameters)
        buttons_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        buttons_layout.addWidget(self.btn_cancel)
        
        panel_layout.addLayout(buttons_layout)
        
        return panel
        
    def create_roi_tab(self):
        """创建ROI投影参数选项卡"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        # ROI模式选择
        roi_mode_group = QGroupBox("ROI模式选择")
        roi_mode_layout = QFormLayout(roi_mode_group)
        
        self.cmb_roi_mode = QComboBox()
        self.cmb_roi_mode.addItems(["none", "camera_rect", "machine", "gcode_bounds"])
        self.cmb_roi_mode.currentTextChanged.connect(self.on_parameter_changed)
        roi_mode_layout.addRow("ROI模式:", self.cmb_roi_mode)
        
        # 相机矩形ROI
        camera_roi_group = QGroupBox("相机矩形ROI")
        camera_roi_layout = QFormLayout(camera_roi_group)
        
        self.spn_roi_x = QSpinBox()
        self.spn_roi_x.setRange(0, 2000)
        self.spn_roi_x.valueChanged.connect(self.on_parameter_changed)
        camera_roi_layout.addRow("X坐标:", self.spn_roi_x)
        
        self.spn_roi_y = QSpinBox()
        self.spn_roi_y.setRange(0, 2000)
        self.spn_roi_y.valueChanged.connect(self.on_parameter_changed)
        camera_roi_layout.addRow("Y坐标:", self.spn_roi_y)
        
        self.spn_roi_w = QSpinBox()
        self.spn_roi_w.setRange(50, 2000)
        self.spn_roi_w.valueChanged.connect(self.on_parameter_changed)
        camera_roi_layout.addRow("宽度:", self.spn_roi_w)
        
        self.spn_roi_h = QSpinBox()
        self.spn_roi_h.setRange(50, 2000)
        self.spn_roi_h.valueChanged.connect(self.on_parameter_changed)
        camera_roi_layout.addRow("高度:", self.spn_roi_h)
        
        # 投影参数
        projection_group = QGroupBox("投影参数")
        projection_layout = QFormLayout(projection_group)
        
        self.spn_pixel_size = QDoubleSpinBox()
        self.spn_pixel_size.setRange(0.05, 10.0)
        self.spn_pixel_size.setDecimals(3)
        self.spn_pixel_size.valueChanged.connect(self.on_parameter_changed)
        projection_layout.addRow("像素尺寸(mm):", self.spn_pixel_size)
        
        self.spn_bounds_margin = QDoubleSpinBox()
        self.spn_bounds_margin.setRange(0, 100)
        self.spn_bounds_margin.setDecimals(1)
        self.spn_bounds_margin.valueChanged.connect(self.on_parameter_changed)
        projection_layout.addRow("边界扩展(mm):", self.spn_bounds_margin)
        
        layout.addWidget(roi_mode_group)
        layout.addWidget(camera_roi_group)
        layout.addWidget(projection_group)
        layout.addStretch()
        
        return scroll