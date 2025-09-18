# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 可视化组件
"""
import os
import numpy as np
import cv2
import json
import time
from typing import Dict, Optional, List
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QSlider, QCheckBox, QTextEdit, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QTabWidget,
    QSplitter, QScrollArea, QFrame, QMessageBox, QDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QPixmap, QFont

from controller import np_to_qimage

# ==================== 多层加工参数调节对话框 ====================

class MultilayerAdvancedParametersDialog(QDialog):
    """多层加工系统高级参数调节对话框"""
    
    parameters_applied = pyqtSignal(dict)  # 参数应用信号
    
    def __init__(self, project_config, controller=None, parent=None):
        super().__init__(parent)
        self.project_config = project_config
        self.controller = controller
        self.setWindowTitle("多层加工系统 - 高级参数调节")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.setup_ui()
        self.load_current_parameters()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        # 项目配置选项卡
        project_tab = self.create_project_tab()
        tabs.addTab(project_tab, "项目配置")
        
        # 相机配置选项卡
        camera_tab = self.create_camera_tab()
        tabs.addTab(camera_tab, "相机配置")
        
        # 算法配置选项卡
        algorithm_tab = self.create_algorithm_tab()
        tabs.addTab(algorithm_tab, "算法配置")
        
        # PLC通信选项卡
        plc_tab = self.create_plc_tab()
        tabs.addTab(plc_tab, "PLC通信")
        
        # 处理配置选项卡
        processing_tab = self.create_processing_tab()
        tabs.addTab(processing_tab, "处理配置")
        
        layout.addWidget(tabs)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("重置")
        self.btn_reset.clicked.connect(self.reset_parameters)
        btn_layout.addWidget(self.btn_reset)
        
        self.btn_save_preset = QPushButton("保存预设")
        self.btn_save_preset.clicked.connect(self.save_preset)
        btn_layout.addWidget(self.btn_save_preset)
        
        self.btn_load_preset = QPushButton("加载预设")
        self.btn_load_preset.clicked.connect(self.load_preset)
        btn_layout.addWidget(self.btn_load_preset)
        
        btn_layout.addStretch()
        
        self.btn_apply = QPushButton("应用")
        self.btn_apply.clicked.connect(self.apply_parameters)
        btn_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)
        
    def create_project_tab(self):
        """创建项目配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 基础项目信息
        basic_group = QGroupBox("基础信息")
        basic_layout = QFormLayout(basic_group)
        
        self.edit_project_name = QLineEdit()
        basic_layout.addRow("项目名称:", self.edit_project_name)
        
        self.spn_total_layers = QSpinBox()
        self.spn_total_layers.setRange(1, 1000)
        basic_layout.addRow("总层数:", self.spn_total_layers)
        
        self.spn_layer_thickness = QDoubleSpinBox()
        self.spn_layer_thickness.setRange(0.01, 100.0)
        self.spn_layer_thickness.setDecimals(3)
        basic_layout.addRow("层厚(mm):", self.spn_layer_thickness)
        
        self.chk_auto_next = QCheckBox("自动处理下一层")
        basic_layout.addRow(self.chk_auto_next)
        
        layout.addWidget(basic_group)
        
        # 保存配置
        save_group = QGroupBox("保存配置")
        save_layout = QFormLayout(save_group)
        
        self.edit_project_dir = QLineEdit()
        save_layout.addRow("项目目录:", self.edit_project_dir)
        
        self.chk_backup_enabled = QCheckBox("启用自动备份")
        save_layout.addRow(self.chk_backup_enabled)
        
        layout.addWidget(save_group)
        layout.addStretch()
        
        return widget
        
    def create_camera_tab(self):
        """创建相机配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 变换矩阵配置
        transform_group = QGroupBox("变换矩阵")
        transform_layout = QFormLayout(transform_group)
        
        self.edit_t_path = QLineEdit()
        self.btn_browse_t = QPushButton("浏览")
        t_layout = QHBoxLayout()
        t_layout.addWidget(self.edit_t_path)
        t_layout.addWidget(self.btn_browse_t)
        transform_layout.addRow("T矩阵路径:", t_layout)
        
        layout.addWidget(transform_group)
        
        # ROI配置
        roi_group = QGroupBox("ROI配置")
        roi_layout = QFormLayout(roi_group)
        
        self.cmb_roi_mode = QComboBox()
        self.cmb_roi_mode.addItems(["none", "camera_rect", "machine", "gcode_bounds"])
        roi_layout.addRow("ROI模式:", self.cmb_roi_mode)
        
        self.spn_pixel_size = QDoubleSpinBox()
        self.spn_pixel_size.setRange(0.001, 100.0)
        self.spn_pixel_size.setDecimals(4)
        roi_layout.addRow("像素尺寸(mm):", self.spn_pixel_size)
        
        self.spn_bounds_margin = QDoubleSpinBox()
        self.spn_bounds_margin.setRange(0, 1000)
        self.spn_bounds_margin.setDecimals(1)
        roi_layout.addRow("边界扩展(mm):", self.spn_bounds_margin)
        
        layout.addWidget(roi_group)
        layout.addStretch()
        
        return widget
        
    def create_algorithm_tab(self):
        """创建算法配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 引导中心线算法
        guide_group = QGroupBox("引导中心线算法")
        guide_layout = QFormLayout(guide_group)
        
        self.spn_guide_step = QDoubleSpinBox()
        self.spn_guide_step.setRange(0.1, 10.0)
        self.spn_guide_step.setDecimals(2)
        guide_layout.addRow("引导步长(mm):", self.spn_guide_step)
        
        self.spn_guide_halfwidth = QDoubleSpinBox()
        self.spn_guide_halfwidth.setRange(0.5, 100.0)
        self.spn_guide_halfwidth.setDecimals(1)
        guide_layout.addRow("搜索半宽(mm):", self.spn_guide_halfwidth)
        
        self.spn_smooth_win = QSpinBox()
        self.spn_smooth_win.setRange(1, 99)
        guide_layout.addRow("平滑窗口:", self.spn_smooth_win)
        
        self.spn_max_offset = QDoubleSpinBox()
        self.spn_max_offset.setRange(0.1, 100.0)
        self.spn_max_offset.setDecimals(1)
        guide_layout.addRow("最大偏移(mm):", self.spn_max_offset)
        
        self.spn_max_grad = QDoubleSpinBox()
        self.spn_max_grad.setRange(0.001, 1.0)
        self.spn_max_grad.setDecimals(4)
        guide_layout.addRow("最大梯度(mm/mm):", self.spn_max_grad)
        
        layout.addWidget(guide_group)
        
        # 平面展平算法
        plane_group = QGroupBox("平面展平算法")
        plane_layout = QFormLayout(plane_group)
        
        self.chk_plane_enable = QCheckBox("启用平面展平")
        plane_layout.addRow(self.chk_plane_enable)
        
        self.spn_plane_thresh = QDoubleSpinBox()
        self.spn_plane_thresh.setRange(0.1, 10.0)
        self.spn_plane_thresh.setDecimals(2)
        plane_layout.addRow("RANSAC阈值(mm):", self.spn_plane_thresh)
        
        layout.addWidget(plane_group)
        
        # 最近表面提取
        surface_group = QGroupBox("最近表面提取")
        surface_layout = QFormLayout(surface_group)
        
        self.spn_nearest_qlo = QDoubleSpinBox()
        self.spn_nearest_qlo.setRange(0.0, 50.0)
        self.spn_nearest_qlo.setDecimals(1)
        surface_layout.addRow("下分位数(%):", self.spn_nearest_qlo)
        
        self.spn_nearest_qhi = QDoubleSpinBox()
        self.spn_nearest_qhi.setRange(50.0, 100.0)
        self.spn_nearest_qhi.setDecimals(1)
        surface_layout.addRow("上分位数(%):", self.spn_nearest_qhi)
        
        self.spn_depth_margin = QDoubleSpinBox()
        self.spn_depth_margin.setRange(0.1, 100.0)
        self.spn_depth_margin.setDecimals(2)
        surface_layout.addRow("深度边界(mm):", self.spn_depth_margin)
        
        layout.addWidget(surface_group)
        layout.addStretch()
        
        return widget
        
    def create_plc_tab(self):
        """创建PLC通信选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # PLC基础配置
        basic_group = QGroupBox("基础配置")
        basic_layout = QFormLayout(basic_group)
        
        self.chk_use_plc = QCheckBox("启用PLC通信")
        basic_layout.addRow(self.chk_use_plc)
        
        self.cmb_plc_type = QComboBox()
        self.cmb_plc_type.addItems(["tcp", "s7", "mock"])
        basic_layout.addRow("通信类型:", self.cmb_plc_type)
        
        self.edit_plc_ip = QLineEdit()
        basic_layout.addRow("PLC IP:", self.edit_plc_ip)
        
        self.spn_plc_port = QSpinBox()
        self.spn_plc_port.setRange(1, 65535)
        basic_layout.addRow("端口:", self.spn_plc_port)
        
        layout.addWidget(basic_group)
        
        # 地址配置
        address_group = QGroupBox("地址配置")
        address_layout = QFormLayout(address_group)
        
        self.edit_layer_address = QLineEdit()
        address_layout.addRow("当前层地址:", self.edit_layer_address)
        
        self.edit_start_address = QLineEdit()
        address_layout.addRow("启动信号地址:", self.edit_start_address)
        
        layout.addWidget(address_group)
        layout.addStretch()
        
        return widget
        
    def create_processing_tab(self):
        """创建处理配置选项卡"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 处理策略
        strategy_group = QGroupBox("处理策略")
        strategy_layout = QFormLayout(strategy_group)
        
        self.chk_auto_process = QCheckBox("自动处理模式")
        strategy_layout.addRow(self.chk_auto_process)
        
        self.spn_max_retry = QSpinBox()
        self.spn_max_retry.setRange(0, 10)
        strategy_layout.addRow("最大重试次数:", self.spn_max_retry)
        
        self.spn_timeout = QSpinBox()
        self.spn_timeout.setRange(10, 600)
        strategy_layout.addRow("处理超时(秒):", self.spn_timeout)
        
        layout.addWidget(strategy_group)
        
        # 质量控制
        quality_group = QGroupBox("质量控制")
        quality_layout = QFormLayout(quality_group)
        
        self.spn_min_valid_ratio = QDoubleSpinBox()
        self.spn_min_valid_ratio.setRange(0.0, 1.0)
        self.spn_min_valid_ratio.setDecimals(2)
        quality_layout.addRow("最小有效率:", self.spn_min_valid_ratio)
        
        self.spn_max_dev_p95 = QDoubleSpinBox()
        self.spn_max_dev_p95.setRange(0.1, 100.0)
        self.spn_max_dev_p95.setDecimals(2)
        quality_layout.addRow("最大P95偏差(mm):", self.spn_max_dev_p95)
        
        layout.addWidget(quality_group)
        layout.addStretch()
        
        return widget
        
    def load_current_parameters(self):
        """加载当前参数到界面控件"""
        config = self.project_config
        
        # 项目配置
        self.edit_project_name.setText(config.project_name)
        self.spn_total_layers.setValue(config.total_layers)
        self.spn_layer_thickness.setValue(config.layer_thickness_mm)
        self.chk_auto_next.setChecked(config.auto_next_layer)
        self.edit_project_dir.setText(config.project_dir)
        self.chk_backup_enabled.setChecked(config.backup_enabled)
        
        # 相机配置
        camera_config = config.camera_config
        self.edit_t_path.setText(camera_config.get("T_path", ""))
        self.cmb_roi_mode.setCurrentText(camera_config.get("roi_mode", "gcode_bounds"))
        self.spn_pixel_size.setValue(camera_config.get("pixel_size_mm", 0.8))
        self.spn_bounds_margin.setValue(camera_config.get("bounds_margin_mm", 20.0))
        
        # 算法配置
        algo_config = config.algorithm_config
        self.spn_guide_step.setValue(algo_config.get("guide_step_mm", 1.0))
        self.spn_guide_halfwidth.setValue(algo_config.get("guide_halfwidth_mm", 6.0))
        self.spn_smooth_win.setValue(algo_config.get("guide_smooth_win", 7))
        self.spn_max_offset.setValue(algo_config.get("guide_max_offset_mm", 8.0))
        self.spn_max_grad.setValue(algo_config.get("guide_max_grad_mm_per_mm", 0.08))
        self.chk_plane_enable.setChecked(algo_config.get("plane_enable", True))
        self.spn_plane_thresh.setValue(algo_config.get("plane_ransac_thresh_mm", 0.8))
        self.spn_nearest_qlo.setValue(algo_config.get("nearest_qlo", 1.0))
        self.spn_nearest_qhi.setValue(algo_config.get("nearest_qhi", 99.0))
        self.spn_depth_margin.setValue(algo_config.get("depth_margin_mm", 3.0))
        
        # PLC配置
        self.chk_use_plc.setChecked(config.use_plc)
        self.cmb_plc_type.setCurrentText(config.plc_type)
        self.edit_plc_ip.setText(config.plc_ip)
        self.spn_plc_port.setValue(config.plc_port)
        self.edit_layer_address.setText(config.current_layer_address)
        self.edit_start_address.setText(config.start_signal_address)
        
        # 处理配置 - 使用默认值
        self.chk_auto_process.setChecked(False)
        self.spn_max_retry.setValue(3)
        self.spn_timeout.setValue(120)
        self.spn_min_valid_ratio.setValue(0.3)
        self.spn_max_dev_p95.setValue(15.0)
        
    def collect_current_parameters(self):
        """收集当前界面参数"""
        # 更新项目配置
        self.project_config.project_name = self.edit_project_name.text()
        self.project_config.total_layers = self.spn_total_layers.value()
        self.project_config.layer_thickness_mm = self.spn_layer_thickness.value()
        self.project_config.auto_next_layer = self.chk_auto_next.isChecked()
        self.project_config.project_dir = self.edit_project_dir.text()
        self.project_config.backup_enabled = self.chk_backup_enabled.isChecked()
        
        # 更新相机配置
        self.project_config.camera_config.update({
            "T_path": self.edit_t_path.text(),
            "roi_mode": self.cmb_roi_mode.currentText(),
            "pixel_size_mm": self.spn_pixel_size.value(),
            "bounds_margin_mm": self.spn_bounds_margin.value()
        })
        
        # 更新算法配置
        self.project_config.algorithm_config.update({
            "guide_step_mm": self.spn_guide_step.value(),
            "guide_halfwidth_mm": self.spn_guide_halfwidth.value(),
            "guide_smooth_win": self.spn_smooth_win.value(),
            "guide_max_offset_mm": self.spn_max_offset.value(),
            "guide_max_grad_mm_per_mm": self.spn_max_grad.value(),
            "plane_enable": self.chk_plane_enable.isChecked(),
            "plane_ransac_thresh_mm": self.spn_plane_thresh.value(),
            "nearest_qlo": self.spn_nearest_qlo.value(),
            "nearest_qhi": self.spn_nearest_qhi.value(),
            "depth_margin_mm": self.spn_depth_margin.value()
        })
        
        # 更新PLC配置
        self.project_config.use_plc = self.chk_use_plc.isChecked()
        self.project_config.plc_type = self.cmb_plc_type.currentText()
        self.project_config.plc_ip = self.edit_plc_ip.text()
        self.project_config.plc_port = self.spn_plc_port.value()
        self.project_config.current_layer_address = self.edit_layer_address.text()
        self.project_config.start_signal_address = self.edit_start_address.text()
        
        return self.project_config
        
    def save_preset(self):
        """保存参数预设"""
        try:
            from PyQt5.QtWidgets import QInputDialog
            preset_name, ok = QInputDialog.getText(self, "保存预设", "预设名称:")
            if ok and preset_name:
                config = self.collect_current_parameters()
                preset_data = {
                    "name": preset_name,
                    "config": config.to_dict(),
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                preset_file = f"configs/preset_{preset_name}.json"
                os.makedirs("configs", exist_ok=True)
                with open(preset_file, 'w', encoding='utf-8') as f:
                    json.dump(preset_data, f, ensure_ascii=False, indent=2)
                    
                QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存预设失败: {e}")
            
    def load_preset(self):
        """加载参数预设"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            preset_file, _ = QFileDialog.getOpenFileName(
                self, "加载预设", "configs", "JSON文件 (*.json)"
            )
            if preset_file:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                    
                from multilayer_data import ProjectConfig
                config = ProjectConfig.from_dict(preset_data["config"])
                self.project_config = config
                self.load_current_parameters()
                
                preset_name = preset_data.get("name", "未知")
                QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已加载")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载预设失败: {e}")
            
    def reset_parameters(self):
        """重置参数"""
        try:
            from multilayer_data import ProjectConfig
            self.project_config = ProjectConfig()  # 使用默认配置
            self.load_current_parameters()
            QMessageBox.information(self, "重置", "参数已重置为默认值")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重置失败: {e}")
            
    def apply_parameters(self):
        """应用参数"""
        try:
            config = self.collect_current_parameters()
            self.parameters_applied.emit(config.to_dict())
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用参数失败: {e}")

# ==================== 层可视化组件 ====================

class LayerVisualizationWidget(QWidget):
    """层可视化组件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.layers_data = {}  # {layer_id: visualization_data}
        self.current_layer_id = None
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 顶部控制面板
        top_controls = QHBoxLayout()
        
        # 高级参数调节按钮
        self.btn_advanced_params = QPushButton("高级参数调节")
        self.btn_advanced_params.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.btn_advanced_params.clicked.connect(self.open_advanced_params)
        top_controls.addWidget(self.btn_advanced_params)
        
        # 快速操作按钮
        self.btn_refresh = QPushButton("刷新")
        self.btn_refresh.clicked.connect(self.update_visualization)
        top_controls.addWidget(self.btn_refresh)
        
        self.btn_export = QPushButton("导出图像")
        self.btn_export.clicked.connect(self.export_current_view)
        top_controls.addWidget(self.btn_export)
        
        top_controls.addStretch()
        layout.addLayout(top_controls)
        
        # 主控制面板
        controls = QHBoxLayout()
        
        # 层选择
        self.layer_selector = QComboBox()
        self.layer_selector.currentTextChanged.connect(self.on_layer_changed)
        controls.addWidget(QLabel("选择层:"))
        controls.addWidget(self.layer_selector)
        
        # 视图模式
        self.view_mode = QComboBox()
        self.view_mode.addItems([
            "原始vs理论", "纠偏后vs理论", "误差对比图", "G代码3D可视化", 
            "中轴线分析", "偏差分布", "纠偏前后对比", "顶视高度", "最近表面", "统计对比"
        ])
        self.view_mode.currentTextChanged.connect(self.update_visualization)
        controls.addWidget(QLabel("视图模式:"))
        controls.addWidget(self.view_mode)
        
        # 对比模式
        self.compare_mode = QCheckBox("对比模式")
        self.compare_mode.toggled.connect(self.update_visualization)
        controls.addWidget(self.compare_mode)
        
        # 自动刷新
        self.auto_refresh = QCheckBox("自动刷新")
        self.auto_refresh.toggled.connect(self.toggle_auto_refresh)
        controls.addWidget(self.auto_refresh)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(600)
        self.image_label.setStyleSheet("QLabel { border: 1px solid gray; background-color: white; }")
        layout.addWidget(self.image_label)
        
        # 统计信息显示
        self.stats_layout = QHBoxLayout()
        
        # 左侧：当前层统计
        self.current_stats = QTextEdit()
        self.current_stats.setMaximumHeight(150)
        self.current_stats.setReadOnly(True)
        self.stats_layout.addWidget(self.current_stats)
        
        # 右侧：对比统计（当启用对比模式时）
        self.compare_stats = QTextEdit()
        self.compare_stats.setMaximumHeight(150)
        self.compare_stats.setReadOnly(True)
        self.compare_stats.setVisible(False)
        self.stats_layout.addWidget(self.compare_stats)
        
        layout.addLayout(self.stats_layout)
        
        # 自动刷新定时器
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_visualization)
        
    def export_current_view(self):
        """导出当前视图"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出图像", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)"
            )
            if file_path:
                success = self.export_current_image(file_path)
                if success:
                    QMessageBox.information(self, "成功", f"图像已导出至: {file_path}")
                else:
                    QMessageBox.warning(self, "错误", "导出失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")
            
    def generate_statistics_comparison(self):
        """生成统计对比图"""
        try:
            # 创建统计对比图像
            img = np.ones((600, 800, 3), dtype=np.uint8) * 240
            
            # 添加标题
            cv2.putText(img, "Multi-layer Statistics Comparison", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            if self.layers_data:
                # 绘制统计信息
                y_pos = 100
                for layer_id, data in self.layers_data.items():
                    metrics = data.get('metrics', {})
                    valid_ratio = metrics.get('valid_ratio', 0)
                    dev_p95 = metrics.get('dev_p95', 0)
                    
                    text = f"Layer {layer_id}: Valid {valid_ratio:.1%}, P95 {dev_p95:.3f}mm"
                    cv2.putText(img, text, (50, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                    y_pos += 40
            else:
                cv2.putText(img, "No data available", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                           
            return img
        except Exception as e:
            print(f"生成统计对比图错误: {e}")
            return None
        
    def open_advanced_params(self):
        """打开高级参数调节对话框"""
        try:
            # 需要从父窗口获取project_config和controller
            parent_window = self.window()
            if hasattr(parent_window, 'project_config'):
                project_config = parent_window.project_config
                controller = getattr(parent_window, 'controller', None)
                
                self.advanced_params_dialog = MultilayerAdvancedParametersDialog(
                    project_config, controller, self
                )
                self.advanced_params_dialog.parameters_applied.connect(self.on_advanced_params_applied)
                self.advanced_params_dialog.exec_()
            else:
                QMessageBox.warning(self, "警告", "无法获取项目配置信息")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开高级参数调节失败: {e}")
            
    def on_advanced_params_applied(self, params_dict):
        """高级参数应用回调"""
        try:
            # 通知父窗口参数已更新
            parent_window = self.window()
            if hasattr(parent_window, 'on_advanced_params_updated'):
                parent_window.on_advanced_params_updated(params_dict)
            
            QMessageBox.information(self, "成功", "高级参数已应用")
            self.update_visualization()  # 刷新可视化
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用参数失败: {e}")
            
    def toggle_auto_refresh(self, enabled):
        """切换自动刷新"""
        if enabled:
            self.refresh_timer.start(2000)  # 2秒刷新一次
        else:
            self.refresh_timer.stop()
        
    def add_layer_data(self, layer_id: int, data: Dict):
        """添加层数据"""
        self.layers_data[layer_id] = data
        
        # 更新层选择器
        current_items = [self.layer_selector.itemText(i) 
                        for i in range(self.layer_selector.count())]
        layer_text = f"第{layer_id}层"
        if layer_text not in current_items:
            self.layer_selector.addItem(layer_text)
            
        # 自动选择最新层
        self.layer_selector.setCurrentText(layer_text)
        
    def on_layer_changed(self, layer_text: str):
        """层选择变化"""
        if layer_text and layer_text.startswith("第") and layer_text.endswith("层"):
            try:
                self.current_layer_id = int(layer_text[1:-1])  # "第1层" -> 1
                self.update_visualization()
            except ValueError:
                pass
                
    def update_visualization(self):
        """更新可视化"""
        if not self.current_layer_id or self.current_layer_id not in self.layers_data:
            self.image_label.setText("请选择有效的层")
            return
            
        try:
            data = self.layers_data[self.current_layer_id]
            view_mode = self.view_mode.currentText()
            
            # 根据视图模式获取图像
            img = None
            if view_mode == "原始vs理论":
                img = data.get('vis_cmp')
            elif view_mode == "纠偏后vs理论":
                img = data.get('vis_corr')
            elif view_mode == "误差对比图":
                img = data.get('error_comparison')
            elif view_mode == "G代码3D可视化":
                img = data.get('gcode_3d_viz')
            elif view_mode == "中轴线分析":
                img = data.get('centerline_analysis')
            elif view_mode == "偏差分布":
                img = data.get('hist_panel')
            elif view_mode == "纠偏前后对比":
                img = data.get('before_after_comparison')
            elif view_mode == "顶视高度":
                img = data.get('vis_top')
            elif view_mode == "最近表面":
                img = data.get('vis_nearest')
            elif view_mode == "统计对比":
                img = self.generate_statistics_comparison()
            
            if img is not None:
                self.display_image(img)
            else:
                self.image_label.setText(f"无{view_mode}数据")
                
            # 更新统计信息
            self.update_statistics()
                
        except Exception as e:
            self.image_label.setText(f"可视化更新错误: {e}")
            print(f"可视化更新错误: {e}")
            
    def display_image(self, img_array):
        """显示图像"""
        if img_array is None:
            self.image_label.setText("无图像数据")
            return
            
        try:
            qimage = np_to_qimage(img_array)
            if qimage:
                pixmap = QPixmap.fromImage(qimage)
                # 缩放以适应显示区域，保持宽高比
                label_size = self.image_label.size()
                scaled_pixmap = pixmap.scaled(
                    label_size.width() - 20, label_size.height() - 20,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("图像格式不支持")
        except Exception as e:
            self.image_label.setText(f"图像显示错误: {e}")
            
    def update_statistics(self):
        """更新统计信息"""
        if not self.current_layer_id or self.current_layer_id not in self.layers_data:
            return
            
        try:
            data = self.layers_data[self.current_layer_id]
            metrics = data.get('metrics', {})
            
            # 当前层统计
            stats_text = self.format_layer_statistics(self.current_layer_id, metrics, data)
            self.current_stats.setPlainText(stats_text)
            
            # 对比模式
            if self.compare_mode.isChecked():
                self.compare_stats.setVisible(True)
                # 可以实现层间对比逻辑
                compare_text = self.get_comparison_statistics()
                self.compare_stats.setPlainText(compare_text)
            else:
                self.compare_stats.setVisible(False)
                
        except Exception as e:
            print(f"统计信息更新错误: {e}")
            
    def format_layer_statistics(self, layer_id: int, metrics: Dict, data: Dict) -> str:
        """格式化层统计信息"""
        stats_lines = [f"=== 第{layer_id}层统计信息 ==="]
        
        # 基础指标
        valid_ratio = metrics.get('valid_ratio', 0)
        stats_lines.append(f"有效率: {valid_ratio:.1%}")
        
        dev_p95 = metrics.get('dev_p95', 0)
        stats_lines.append(f"偏差P95: {dev_p95:.3f} mm")
        
        dev_mean = metrics.get('dev_mean', 0)
        dev_median = metrics.get('dev_median', 0)
        stats_lines.append(f"偏差均值: {dev_mean:+.3f} mm")
        stats_lines.append(f"偏差中位数: {dev_median:+.3f} mm")
        
        # 平面拟合信息
        plane_inlier_ratio = metrics.get('plane_inlier_ratio', 0)
        if not np.isnan(plane_inlier_ratio):
            stats_lines.append(f"平面内点率: {plane_inlier_ratio:.1%}")
        
        # 缺失信息
        longest_missing_mm = metrics.get('longest_missing_mm', 0)
        stats_lines.append(f"最长缺失: {longest_missing_mm:.1f} mm")
        
        # 处理信息
        processing_time = data.get('processing_time', 0)
        stats_lines.append(f"处理耗时: {processing_time:.1f} 秒")
        
        layer_type = data.get('layer_type', 'unknown')
        stats_lines.append(f"层类型: {'标定层' if layer_type == 'calibration' else '纠偏层'}")
        
        timestamp = data.get('timestamp', '')
        if timestamp:
            stats_lines.append(f"处理时间: {timestamp}")
            
        return '\n'.join(stats_lines)
        
    def get_comparison_statistics(self) -> str:
        """获取对比统计信息"""
        if not self.current_layer_id:
            return "无对比数据"
            
        # 可以实现与前一层或平均值的对比
        compare_lines = ["=== 对比信息 ==="]
        
        # 与前一层对比
        prev_layer_id = self.current_layer_id - 1
        if prev_layer_id in self.layers_data:
            prev_data = self.layers_data[prev_layer_id]
            prev_metrics = prev_data.get('metrics', {})
            curr_metrics = self.layers_data[self.current_layer_id].get('metrics', {})
            
            # 有效率变化
            prev_valid = prev_metrics.get('valid_ratio', 0)
            curr_valid = curr_metrics.get('valid_ratio', 0)
            valid_change = curr_valid - prev_valid
            compare_lines.append(f"有效率变化: {valid_change:+.1%}")
            
            # 偏差变化
            prev_p95 = prev_metrics.get('dev_p95', 0)
            curr_p95 = curr_metrics.get('dev_p95', 0)
            p95_change = curr_p95 - prev_p95
            compare_lines.append(f"P95偏差变化: {p95_change:+.3f} mm")
            
        # 整体趋势
        if len(self.layers_data) >= 3:
            compare_lines.append("\n=== 整体趋势 ===")
            all_valid_ratios = [data.get('metrics', {}).get('valid_ratio', 0) 
                              for data in self.layers_data.values()]
            avg_valid = np.mean(all_valid_ratios) if all_valid_ratios else 0
            compare_lines.append(f"平均有效率: {avg_valid:.1%}")
            
            all_p95 = [data.get('metrics', {}).get('dev_p95', 0) 
                      for data in self.layers_data.values()]
            avg_p95 = np.mean(all_p95) if all_p95 else 0
            compare_lines.append(f"平均P95偏差: {avg_p95:.3f} mm")
            
        return '\n'.join(compare_lines)
        
    def clear_all_data(self):
        """清空所有数据"""
        self.layers_data.clear()
        self.layer_selector.clear()
        self.current_layer_id = None
        self.image_label.clear()
        self.current_stats.clear()
        self.compare_stats.clear()
        
    def export_current_image(self, file_path: str):
        """导出当前显示的图像"""
        try:
            pixmap = self.image_label.pixmap()
            if pixmap:
                pixmap.save(file_path)
                return True
            return False
        except Exception as e:
            print(f"导出图像失败: {e}")
            return False

# ==================== 3D轨迹可视化组件 ====================

class TrajectoryVisualizationWidget(QWidget):
    """3D轨迹可视化组件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.trajectory_data = {}  # {layer_id: trajectory_info}
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 控制面板
        controls = QHBoxLayout()
        
        self.show_theoretical = QCheckBox("显示理论轨迹")
        self.show_theoretical.setChecked(True)
        self.show_theoretical.toggled.connect(self.update_view)
        controls.addWidget(self.show_theoretical)
        
        self.show_actual = QCheckBox("显示实际轨迹")
        self.show_actual.setChecked(True)
        self.show_actual.toggled.connect(self.update_view)
        controls.addWidget(self.show_actual)
        
        self.show_corrected = QCheckBox("显示纠偏轨迹")
        self.show_corrected.setChecked(True)
        self.show_corrected.toggled.connect(self.update_view)
        controls.addWidget(self.show_corrected)
        
        # 层选择滑块
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.valueChanged.connect(self.update_view)
        controls.addWidget(QLabel("层选择:"))
        controls.addWidget(self.layer_slider)
        
        self.layer_label = QLabel("第1层")
        controls.addWidget(self.layer_label)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # 3D显示区域（这里用标签代替真实的3D控件）
        self.view_label = QLabel("3D轨迹视图")
        self.view_label.setAlignment(Qt.AlignCenter)
        self.view_label.setMinimumHeight(400)
        self.view_label.setStyleSheet("QLabel { border: 1px solid gray; background-color: #f0f0f0; }")
        layout.addWidget(self.view_label)
        
    def add_trajectory_data(self, layer_id: int, theoretical: np.ndarray, 
                          actual: np.ndarray, corrected: Optional[np.ndarray] = None):
        """添加轨迹数据"""
        self.trajectory_data[layer_id] = {
            'theoretical': theoretical,
            'actual': actual,
            'corrected': corrected
        }
        
        # 更新滑块范围
        max_layer = max(self.trajectory_data.keys())
        self.layer_slider.setRange(1, max_layer)
        self.layer_slider.setValue(layer_id)
        
        self.update_view()
        
    def update_view(self):
        """更新3D视图"""
        current_layer = self.layer_slider.value()
        self.layer_label.setText(f"第{current_layer}层")
        
        if current_layer not in self.trajectory_data:
            self.view_label.setText("无轨迹数据")
            return
            
        # 这里应该实现真实的3D渲染
        # 目前只显示文本信息
        data = self.trajectory_data[current_layer]
        info_lines = [f"第{current_layer}层轨迹信息:"]
        
        if self.show_theoretical.isChecked() and 'theoretical' in data:
            theo_len = len(data['theoretical'])
            info_lines.append(f"理论轨迹: {theo_len} 点")
            
        if self.show_actual.isChecked() and 'actual' in data:
            actual_len = len(data['actual'])
            info_lines.append(f"实际轨迹: {actual_len} 点")
            
        if self.show_corrected.isChecked() and data.get('corrected') is not None:
            corr_len = len(data['corrected'])
            info_lines.append(f"纠偏轨迹: {corr_len} 点")
            
        self.view_label.setText('\n'.join(info_lines))

# ==================== 多层对比图表组件 ====================

class MultiLayerComparisonWidget(QWidget):
    """多层对比图表组件"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.comparison_data = {}
        
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        
        # 图表显示区域
        self.chart_label = QLabel("多层对比图表")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setMinimumHeight(400)
        self.chart_label.setStyleSheet("QLabel { border: 1px solid gray; }")
        layout.addWidget(self.chart_label)
        
    def add_layer_metrics(self, layer_id: int, metrics: Dict):
        """添加层指标数据"""
        self.comparison_data[layer_id] = metrics
        self.update_chart()
        
    def update_chart(self):
        """更新图表"""
        if not self.comparison_data:
            self.chart_label.setText("无对比数据")
            return
            
        # 这里应该绘制真实的图表
        # 目前显示文本统计
        lines = ["多层对比统计:"]
        for layer_id, metrics in sorted(self.comparison_data.items()):
            valid_ratio = metrics.get('valid_ratio', 0)
            dev_p95 = metrics.get('dev_p95', 0)
            lines.append(f"第{layer_id}层: 有效率{valid_ratio:.1%}, P95={dev_p95:.3f}mm")
            
        self.chart_label.setText('\n'.join(lines))
        
    def export_current_view(self):
        """导出当前视图"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出图像", "", "PNG文件 (*.png);;JPEG文件 (*.jpg)"
            )
            if file_path:
                success = self.export_current_image(file_path)
                if success:
                    QMessageBox.information(self, "成功", f"图像已导出至: {file_path}")
                else:
                    QMessageBox.warning(self, "错误", "导出失败")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")