# -*- coding: utf-8 -*-
"""
高级参数调节界面
提供丰富的参数调节和实时预览功能
"""
import os
import sys
import json
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, 
    QScrollArea, QFrame, QSlider, QMessageBox, QFileDialog
)
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
        roi_tab = self.create_roi_tab()
        self.params_tabs.addTab(roi_tab, "ROI投影")
        
        # 最近表面参数
        surface_tab = self.create_surface_tab()
        self.params_tabs.addTab(surface_tab, "最近表面")
        
        # 法线可视化参数
        visual_tab = self.create_visualization_tab()
        self.params_tabs.addTab(visual_tab, "法线可视化")
        
        # 遮挡区域参数
        occlusion_tab = self.create_occlusion_tab()
        self.params_tabs.addTab(occlusion_tab, "遮挡区域")
        
        # 引导中心线参数
        guide_tab = self.create_guide_tab()
        self.params_tabs.addTab(guide_tab, "引导中心线")
        
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
        
    def create_surface_tab(self):
        """创建最近表面参数选项卡"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        # 平面拟合
        plane_group = QGroupBox("平面展平")
        plane_layout = QFormLayout(plane_group)
        
        self.chk_plane_enable = QCheckBox("启用平面展平")
        self.chk_plane_enable.stateChanged.connect(self.on_parameter_changed)
        plane_layout.addRow(self.chk_plane_enable)
        
        self.spn_plane_thresh = QDoubleSpinBox()
        self.spn_plane_thresh.setRange(0.1, 10.0)
        self.spn_plane_thresh.setDecimals(2)
        self.spn_plane_thresh.setSingleStep(0.1)
        self.spn_plane_thresh.valueChanged.connect(self.on_parameter_changed)
        plane_layout.addRow("RANSAC阈值(mm):", self.spn_plane_thresh)
        
        self.spn_plane_iters = QSpinBox()
        self.spn_plane_iters.setRange(50, 5000)
        self.spn_plane_iters.setSingleStep(50)
        self.spn_plane_iters.valueChanged.connect(self.on_parameter_changed)
        plane_layout.addRow("RANSAC迭代次数:", self.spn_plane_iters)
        
        # 最近表面提取
        surface_group = QGroupBox("最近表面提取")
        surface_layout = QFormLayout(surface_group)
        
        self.cmb_z_select = QComboBox()
        self.cmb_z_select.addItems(["max", "min"])
        self.cmb_z_select.currentTextChanged.connect(self.on_parameter_changed)
        surface_layout.addRow("Z选择模式:", self.cmb_z_select)
        
        self.spn_nearest_qlo = QDoubleSpinBox()
        self.spn_nearest_qlo.setRange(0.0, 50.0)
        self.spn_nearest_qlo.setDecimals(1)
        self.spn_nearest_qlo.valueChanged.connect(self.on_parameter_changed)
        surface_layout.addRow("下分位数(%):", self.spn_nearest_qlo)
        
        self.spn_nearest_qhi = QDoubleSpinBox()
        self.spn_nearest_qhi.setRange(50.0, 100.0)
        self.spn_nearest_qhi.setDecimals(1)
        self.spn_nearest_qhi.valueChanged.connect(self.on_parameter_changed)
        surface_layout.addRow("上分位数(%):", self.spn_nearest_qhi)
        
        self.spn_depth_margin = QDoubleSpinBox()
        self.spn_depth_margin.setRange(0.1, 50.0)
        self.spn_depth_margin.setDecimals(2)
        self.spn_depth_margin.valueChanged.connect(self.on_parameter_changed)
        surface_layout.addRow("深度边界(mm):", self.spn_depth_margin)
        
        # 形态学处理
        morphology_group = QGroupBox("形态学处理")
        morphology_layout = QFormLayout(morphology_group)
        
        self.spn_morph_open = QSpinBox()
        self.spn_morph_open.setRange(0, 31)
        self.spn_morph_open.valueChanged.connect(self.on_parameter_changed)
        morphology_layout.addRow("开运算核大小:", self.spn_morph_open)
        
        self.spn_morph_close = QSpinBox()
        self.spn_morph_close.setRange(0, 31)
        self.spn_morph_close.valueChanged.connect(self.on_parameter_changed)
        morphology_layout.addRow("闭运算核大小:", self.spn_morph_close)
        
        self.spn_min_area = QSpinBox()
        self.spn_min_area.setRange(1, 100000)
        self.spn_min_area.setSingleStep(100)
        self.spn_min_area.valueChanged.connect(self.on_parameter_changed)
        morphology_layout.addRow("最小连通域:", self.spn_min_area)
        
        layout.addWidget(plane_group)
        layout.addWidget(surface_group)
        layout.addWidget(morphology_group)
        layout.addStretch()
        
        return scroll
        
    def create_visualization_tab(self):
        """创建法线可视化参数选项卡"""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        layout = QVBoxLayout(tab)
        
        # 法线显示
        normal_group = QGroupBox("法线显示设置")
        normal_layout = QFormLayout(normal_group)
        
        self.chk_draw_normals = QCheckBox("显示法向采样线")
        self.chk_draw_normals.stateChanged.connect(self.on_parameter_changed)
        normal_layout.addRow(self.chk_draw_normals)
        
        self.spn_arrow_stride = QSpinBox()
        self.spn_arrow_stride.setRange(1, 200)
        self.spn_arrow_stride.valueChanged.connect(self.on_parameter_changed)
        normal_layout.addRow("箭头间距:", self.spn_arrow_stride)
        
        # 法线调试窗口
        debug_window_group = QGroupBox("法线调试窗口")
        debug_window_layout = QFormLayout(debug_window_group)
        
        self.chk_debug_window = QCheckBox("启用法线调试窗口")
        self.chk_debug_window.stateChanged.connect(self.on_parameter_changed)
        debug_window_layout.addRow(self.chk_debug_window)
        
        self.spn_debug_stride = QSpinBox()
        self.spn_debug_stride.setRange(1, 200)
        self.spn_debug_stride.valueChanged.connect(self.on_parameter_changed)
        debug_window_layout.addRow("调试步长:", self.spn_debug_stride)
        
        self.spn_debug_max = QSpinBox()
        self.spn_debug_max.setRange(1, 200)
        self.spn_debug_max.valueChanged.connect(self.on_parameter_changed)
        debug_window_layout.addRow("最大显示数:", self.spn_debug_max)
        
        self.spn_debug_length = QDoubleSpinBox()
        self.spn_debug_length.setRange(0.0, 100.0)
        self.spn_debug_length.setDecimals(2)
        self.spn_debug_length.valueChanged.connect(self.on_parameter_changed)
        debug_window_layout.addRow("法线长度(mm):", self.spn_debug_length)
        
        self.chk_debug_text = QCheckBox("显示Δn文本标注")
        self.chk_debug_text.stateChanged.connect(self.on_parameter_changed)
        debug_window_layout.addRow(self.chk_debug_text)
        
        layout.addWidget(normal_group)
        layout.addWidget(debug_window_group)
        layout.addStretch()
        
        return scroll