# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 可视化组件
"""
import numpy as np
import cv2
from typing import Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QSlider, QCheckBox, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

from controller import np_to_qimage

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
        
        # 控制面板
        controls = QHBoxLayout()
        
        # 层选择
        self.layer_selector = QComboBox()
        self.layer_selector.currentTextChanged.connect(self.on_layer_changed)
        controls.addWidget(QLabel("选择层:"))
        controls.addWidget(self.layer_selector)
        
        # 视图模式
        self.view_mode = QComboBox()
        self.view_mode.addItems(["原始vs理论", "纠偏后vs理论", "误差分布", "顶视高度", "最近表面"])
        self.view_mode.currentTextChanged.connect(self.update_visualization)
        controls.addWidget(QLabel("视图模式:"))
        controls.addWidget(self.view_mode)
        
        # 对比模式
        self.compare_mode = QCheckBox("对比模式")
        self.compare_mode.toggled.connect(self.update_visualization)
        controls.addWidget(self.compare_mode)
        
        # 刷新按钮
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.update_visualization)
        controls.addWidget(self.refresh_btn)
        
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
        self.current_stats.setMaximumHeight(120)
        self.current_stats.setReadOnly(True)
        self.stats_layout.addWidget(self.current_stats)
        
        # 右侧：对比统计（当启用对比模式时）
        self.compare_stats = QTextEdit()
        self.compare_stats.setMaximumHeight(120)
        self.compare_stats.setReadOnly(True)
        self.compare_stats.setVisible(False)
        self.stats_layout.addWidget(self.compare_stats)
        
        layout.addLayout(self.stats_layout)
        
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
            elif view_mode == "误差分布":
                img = data.get('hist_panel')
            elif view_mode == "顶视高度":
                img = data.get('vis_top')
            elif view_mode == "最近表面":
                img = data.get('vis_nearest')
            
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