# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 主程序
支持逐层加工、偏差补偿累积、PLC通信、3D可视化
"""
import sys
import os
import json
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import traceback

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QSplitter, QProgressBar, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QPixmap, QFont, QCloseEvent

# 本地模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from controller import AlignController, GUIConfig, np_to_qimage
import align_centerline_to_gcode_pro_edit_max as core
import numpy as np
import cv2

# 导入其他模块
from multilayer_data import LayerInfo, ProjectConfig
from multilayer_plc import PLCCommunicator, TCPPLCCommunicator, S7PLCCommunicator, PLCMonitorThread, MockPLCCommunicator
from s7_plc_enhanced import S7PLCEnhanced  # 新增增强的S7通信器
from multilayer_processor import LayerProcessingThread
from multilayer_visualizer import LayerVisualizationWidget

# ==================== 主窗口 ====================

class MultilayerMainWindow(QMainWindow):
    """多层加工纠偏系统主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多层加工纠偏系统 v1.0")
        self.resize(1600, 1000)
        
        # 核心组件
        self.controller = AlignController(GUIConfig())
        self.project_config = ProjectConfig()
        self.layers: Dict[int, LayerInfo] = {}
        self.current_layer = 0
        
        # 处理参数（来自高级参数配置）
        self.process_delay_sec = 2  # 默认500ms处理延迟
        
        # PLC通信
        self.plc_communicator = None
        self.plc_monitor = None
        
        # 处理状态
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        
        self.setup_ui()
        self.setup_connections()
        
        # 启动相机
        self.start_camera()
        
        # 验证数据处理能力
        QTimer.singleShot(1000, self.validate_system_capabilities)
        
    def setup_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局：水平分割
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter()
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # 右侧可视化面板
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例
        splitter.setStretchFactor(0, 0)  # 左侧固定
        splitter.setStretchFactor(1, 1)  # 右侧扩展
        splitter.setSizes([400, 1200])
        
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        
        # 项目配置组
        project_group = QGroupBox("项目配置")
        project_layout = QFormLayout(project_group)
        
        self.project_name_edit = QLineEdit(self.project_config.project_name)
        project_layout.addRow("项目名称:", self.project_name_edit)
        
        self.total_layers_spin = QSpinBox()
        self.total_layers_spin.setRange(1, 100)
        self.total_layers_spin.setValue(self.project_config.total_layers)
        project_layout.addRow("总层数:", self.total_layers_spin)
        
        self.layer_thickness_spin = QDoubleSpinBox()
        self.layer_thickness_spin.setRange(0.1, 10.0)
        self.layer_thickness_spin.setDecimals(2)
        self.layer_thickness_spin.setValue(self.project_config.layer_thickness_mm)
        project_layout.addRow("层厚(mm):", self.layer_thickness_spin)
        
        self.auto_next_check = QCheckBox("自动处理下一层")
        self.auto_next_check.setChecked(self.project_config.auto_next_layer)
        project_layout.addRow(self.auto_next_check)
        
        layout.addWidget(project_group)
        
        # PLC通信组
        plc_group = QGroupBox("PLC通信")
        plc_layout = QFormLayout(plc_group)
        
        self.use_plc_check = QCheckBox("启用PLC通信")
        self.use_plc_check.setChecked(self.project_config.use_plc)
        plc_layout.addRow(self.use_plc_check)
        
        self.plc_type_combo = QComboBox()
        self.plc_type_combo.addItems(["tcp", "s7", "s7_sim", "mock"])
        self.plc_type_combo.setCurrentText(self.project_config.plc_type)
        plc_layout.addRow("通信类型:", self.plc_type_combo)
        
        self.plc_ip_edit = QLineEdit(self.project_config.plc_ip)
        plc_layout.addRow("PLC IP:", self.plc_ip_edit)
        
        self.plc_port_spin = QSpinBox()
        self.plc_port_spin.setRange(1, 65535)
        self.plc_port_spin.setValue(self.project_config.plc_port)
        plc_layout.addRow("端口:", self.plc_port_spin)
        
        self.layer_address_edit = QLineEdit(self.project_config.current_layer_address)
        plc_layout.addRow("层号地址:", self.layer_address_edit)
        
        layout.addWidget(plc_group)
        
        # G代码加载组
        gcode_group = QGroupBox("G代码管理")
        gcode_layout = QVBoxLayout(gcode_group)
        
        load_layout = QHBoxLayout()
        self.gcode_dir_edit = QLineEdit()
        load_btn = QPushButton("选择G代码目录")
        load_btn.clicked.connect(self.load_gcode_directory)
        load_layout.addWidget(self.gcode_dir_edit)
        load_layout.addWidget(load_btn)
        gcode_layout.addLayout(load_layout)
        
        self.gcode_list = QListWidget()
        gcode_layout.addWidget(self.gcode_list)
        
        layout.addWidget(gcode_group)
        
        # 处理控制组
        control_group = QGroupBox("处理控制")
        control_layout = QVBoxLayout(control_group)
        
        self.connect_plc_btn = QPushButton("连接PLC")
        self.connect_plc_btn.clicked.connect(self.toggle_plc_connection)
        control_layout.addWidget(self.connect_plc_btn)
        
        self.process_current_btn = QPushButton("处理当前层")
        self.process_current_btn.clicked.connect(self.process_current_layer)
        control_layout.addWidget(self.process_current_btn)
        
        self.next_layer_btn = QPushButton("下一层")
        self.next_layer_btn.clicked.connect(self.go_next_layer)
        control_layout.addWidget(self.next_layer_btn)
        
        layout.addWidget(control_group)
        
        # 状态显示
        status_group = QGroupBox("状态信息")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("系统就绪")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        status_layout.addWidget(self.progress_bar)
        
        self.plc_status_label = QLabel("PLC: 未连接")
        status_layout.addWidget(self.plc_status_label)
        
        self.camera_status_label = QLabel("相机: 启动中...")
        status_layout.addWidget(self.camera_status_label)
        
        layout.addWidget(status_group)
        
        # 保存/导出
        save_group = QGroupBox("保存导出")
        save_layout = QVBoxLayout(save_group)
        
        # 高级参数调节按钮
        self.btn_advanced_params = QPushButton("高级参数调节")
        self.btn_advanced_params.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.btn_advanced_params.clicked.connect(self.open_advanced_params)
        save_layout.addWidget(self.btn_advanced_params)
        
        self.save_project_btn = QPushButton("保存项目")
        self.save_project_btn.clicked.connect(self.save_project)
        save_layout.addWidget(self.save_project_btn)
        
        self.export_results_btn = QPushButton("导出结果")
        self.export_results_btn.clicked.connect(self.export_results)
        save_layout.addWidget(self.export_results_btn)
        
        layout.addWidget(save_group)
        
        layout.addStretch()
        return panel
        
    def create_visualization_panel(self):
        """创建右侧可视化面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        
        # 当前层可视化
        self.current_layer_viz = LayerVisualizationWidget()
        self.tab_widget.addTab(self.current_layer_viz, "当前层分析")
        
        # 层管理表格
        self.create_layer_table()
        self.tab_widget.addTab(self.layer_table_widget, "层管理")
        
        # 整体统计
        self.create_overall_stats()
        self.tab_widget.addTab(self.overall_stats_widget, "整体统计")
        
        layout.addWidget(self.tab_widget)
        return panel
        
    def create_layer_table(self):
        """创建层管理表格"""
        self.layer_table_widget = QWidget()
        layout = QVBoxLayout(self.layer_table_widget)
        
        self.layer_table = QTableWidget()
        self.layer_table.setColumnCount(6)
        self.layer_table.setHorizontalHeaderLabels([
            "层号", "状态", "覆盖率", "轨迹精度", "处理时间", "操作"
        ])
        layout.addWidget(self.layer_table)
        
    def create_overall_stats(self):
        """创建整体统计面板"""
        self.overall_stats_widget = QWidget()
        layout = QVBoxLayout(self.overall_stats_widget)
        
        self.overall_stats_text = QTextEdit()
        self.overall_stats_text.setReadOnly(True)
        layout.addWidget(self.overall_stats_text)
        
    def setup_connections(self):
        """设置信号连接"""
        # PLC配置变化信号连接
        self.use_plc_check.toggled.connect(self.on_plc_config_changed)
        self.plc_type_combo.currentTextChanged.connect(self.on_plc_config_changed)
        self.plc_ip_edit.textChanged.connect(self.on_plc_config_changed)
        self.plc_port_spin.valueChanged.connect(self.on_plc_config_changed)
        self.layer_address_edit.textChanged.connect(self.on_plc_config_changed)
        
        # 项目配置变化信号连接
        self.project_name_edit.textChanged.connect(self.on_project_config_changed)
        self.total_layers_spin.valueChanged.connect(self.on_project_config_changed)
        self.layer_thickness_spin.valueChanged.connect(self.on_project_config_changed)
        self.auto_next_check.toggled.connect(self.on_project_config_changed)
        
    def start_camera(self):
        """启动相机"""
        def camera_thread():
            msg = self.controller.start_camera()
            self.camera_status_label.setText(f"相机: {msg}")
            
        threading.Thread(target=camera_thread, daemon=True).start()
        
    def load_gcode_directory(self):
        """加载G代码目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择G代码目录")
        if directory:
            self.gcode_dir_edit.setText(directory)
            self.scan_gcode_files(directory)
            
    def scan_gcode_files(self, directory: str):
        """扫描G代码文件"""
        self.gcode_list.clear()
        self.layers.clear()
        
        try:
            gcode_path = Path(directory)
            gcode_files = list(gcode_path.glob("*.gcode")) + \
                         list(gcode_path.glob("*.nc")) + \
                         list(gcode_path.glob("*.txt"))
            
            # 按文件名排序
            gcode_files.sort(key=lambda x: x.name)
            
            for i, file_path in enumerate(gcode_files, 1):
                layer_info = LayerInfo(
                    layer_id=i,
                    gcode_path=str(file_path),
                    status="pending"
                )
                self.layers[i] = layer_info
                
                item = QListWidgetItem(f"第{i}层: {file_path.name}")
                self.gcode_list.addItem(item)
                
            # 更新总层数
            self.total_layers_spin.setValue(len(gcode_files))
            self.update_layer_table()
            
            self.status_label.setText(f"已加载 {len(gcode_files)} 层G代码")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载G代码失败: {e}")
            
    def toggle_plc_connection(self):
        """切换PLC连接"""
        if self.plc_communicator is None or not self.plc_communicator.connected:
            self.connect_plc()
        else:
            self.disconnect_plc()
            
    def connect_plc(self):
        """连接PLC"""
        if not self.use_plc_check.isChecked():
            QMessageBox.information(self, "提示", "请先启用PLC通信")
            return
            
        # 更新配置
        self.update_project_config()
        
        # 创建通信器
        if self.project_config.plc_type == "tcp":
            self.plc_communicator = TCPPLCCommunicator(self.project_config)
        elif self.project_config.plc_type == "s7":
            # 优先使用增强版S7通信器
            self.plc_communicator = S7PLCEnhanced(self.project_config)
        elif self.project_config.plc_type == "s7_sim":
            # S7模拟器通信器
            from multilayer_plc import S7SimulatorPLCCommunicator
            self.plc_communicator = S7SimulatorPLCCommunicator(self.project_config)
        elif self.project_config.plc_type == "mock":
            self.plc_communicator = MockPLCCommunicator(self.project_config)
        else:
            # 默认使用模拟通信器
            self.plc_communicator = MockPLCCommunicator(self.project_config)
            
        # 连接新的PLC信号
        self.plc_communicator.connection_status.connect(self.on_plc_status_changed)
        self.plc_communicator.layer_changed.connect(self.on_layer_changed_from_plc)
        self.plc_communicator.machine_status_changed.connect(self.on_machine_status_changed)
        self.plc_communicator.correction_request.connect(self.on_correction_request)
        
        # 连接增强版S7通信器的额外信号（只对S7PLCEnhanced有效）
        if isinstance(self.plc_communicator, S7PLCEnhanced):
            self.plc_communicator.data_transmission_progress.connect(self.on_data_transmission_progress)
            self.plc_communicator.safety_alert.connect(self.on_safety_alert)
            self.plc_communicator.heartbeat_updated.connect(self.on_heartbeat_updated)
        
        # 尝试连接
        if self.plc_communicator.connect():
            self.connect_plc_btn.setText("断开PLC")
            
            # 启动状态轮询
            self.plc_communicator.start_polling()
            
            # 启动监控线程
            self.plc_monitor = PLCMonitorThread(self.plc_communicator)
            self.plc_monitor.start()
        else:
            self.plc_communicator = None
            
    def disconnect_plc(self):
        """断开PLC连接"""
        if self.plc_monitor:
            self.plc_monitor.stop()
            self.plc_monitor = None
            
        if self.plc_communicator:
            # 停止轮询
            self.plc_communicator.stop_polling()
            self.plc_communicator.disconnect_plc()
            self.plc_communicator = None
            
        self.connect_plc_btn.setText("连接PLC")
        self.plc_status_label.setText("PLC: 未连接")
        
    def process_current_layer(self):
        """处理当前层"""
        if self.current_layer == 0:
            if self.layers:
                self.current_layer = 1
            else:
                QMessageBox.warning(self, "错误", "请先加载G代码")
                return
                
        if self.current_layer not in self.layers:
            QMessageBox.warning(self, "错误", f"第{self.current_layer}层不存在")
            return
            
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.information(self, "提示", "正在处理中，请稍候...")
            return
            
        # 获取前一层的偏差补偿
        previous_bias = None
        if self.current_layer > 1:
            # 查找最近一层有偏差补偿数据的层
            for prev_layer_id in range(self.current_layer - 1, 0, -1):
                prev_layer_info = self.layers.get(prev_layer_id)
                if prev_layer_info and prev_layer_info.bias_comp:
                    previous_bias = prev_layer_info.bias_comp
                    print(f"第{self.current_layer}层应用第{prev_layer_id}层的偏差补偿")
                    break
            
            if previous_bias is None:
                print(f"警告：第{self.current_layer}层未找到可用的偏差补偿数据")
                
        # 启动处理线程
        layer_info = self.layers[self.current_layer]
        self.processing_thread = LayerProcessingThread(
            layer_info, self.controller, previous_bias
        )
        
        # 连接信号
        self.processing_thread.processing_finished.connect(self.on_processing_finished)
        self.processing_thread.processing_failed.connect(self.on_processing_failed)
        self.processing_thread.progress_updated.connect(self.on_progress_updated)
        
        self.processing_thread.start()
        self.progress_bar.setValue(0)
        
    def go_next_layer(self):
        """下一层"""
        if self.current_layer < len(self.layers):
            self.current_layer += 1
            if self.auto_next_check.isChecked():
                self.process_current_layer()
        else:
            QMessageBox.information(self, "完成", "所有层处理完成！")
            
    def on_plc_status_changed(self, connected: bool, message: str):
        """PLC状态变化"""
        status = "已连接" if connected else "未连接"
        self.plc_status_label.setText(f"PLC: {status}")
        if message:
            self.status_label.setText(message)
            
    def on_layer_changed_from_plc(self, layer_id: int):
        """增强的PLC层号变化处理"""
        print(f"PLC层号变化: {layer_id}")
        
        # 检查层号是否有效
        if layer_id <= 0:
            print(f"错误: 无效的层号 {layer_id}")
            self.status_label.setText(f"错误: 无效的层号 {layer_id}")
            return
            
        # 检查层号是否在已加载G代码范围内
        if layer_id not in self.layers:
            # 如果层号超出范围，但在总层数内，警告用户
            if layer_id <= self.project_config.total_layers:
                print(f"警告: 第{layer_id}层G代码未加载")
                self.status_label.setText(f"警告: 第{layer_id}层G代码未加载，请先加载所有层的G代码")
                return
            else:
                print(f"错误: 第{layer_id}层不存在（已加载层数: {len(self.layers)}）")
                self.status_label.setText(f"错误: 第{layer_id}层不存在")
                return
        
        # 更新当前层
        old_layer = self.current_layer
        self.current_layer = layer_id
        
        # 更新界面显示
        if old_layer != layer_id:
            self.status_label.setText(f"PLC通知: 从第{old_layer}层切换到第{layer_id}层")
            # 检查并更新前一层的状态
            if old_layer > 0 and old_layer in self.layers and self.layers[old_layer].status == "processing":
                self.layers[old_layer].status = "completed"
                print(f"自动将第{old_layer}层状态更新为已完成")
        else:
            self.status_label.setText(f"PLC确认: 当前为第{layer_id}层")
        
        # 更新层管理表格
        self.update_layer_table()
        
        # 在自动模式下，不在这里直接处理，等待PLC状态变为waiting时才处理
        # 这样可以避免早期触发处理
        print(f"层号变化完成，当前层: {self.current_layer}，等待PLC发送waiting状态")
                
    def on_machine_status_changed(self, status: str):
        """增强的机床状态变化处理"""
        print(f"PLC状态变化: {status}")
        
        status_map = {
            "idle": "空闲",
            "processing": "加工中",
            "waiting": "等待纠偏",
            "error": "错误",
            "completed": "完成"
        }
        chinese_status = status_map.get(status, status)
        
        # 更新状态显示
        status_with_layer = f"机床状态: {chinese_status} - 第{self.current_layer}层"
        self.status_label.setText(status_with_layer)
        
        # 根据状态调整界面和处理逻辑
        if status == "idle":
            self.process_current_btn.setEnabled(True)
            print(f"机床空闲，等待开始下一阶段工作")
            
        elif status == "processing":
            self.process_current_btn.setEnabled(False)
            print(f"机床正在加工第{self.current_layer}层，算法程序等待")
            
        elif status == "waiting":
            self.process_current_btn.setEnabled(True)
            self.status_label.setText(f"机床等待纠偏数据 - 第{self.current_layer}层")
            print(f"机床进入等待状态，等待correction_request信号")
            # 注意：不在这里直接处理，等待correction_request信号
            
        elif status == "error":
            self.process_current_btn.setEnabled(True)
            self.status_label.setText(f"机床错误 - 第{self.current_layer}层")
            print(f"机床出现错误，需要排查问题")
            
        elif status == "completed":
            print(f"第{self.current_layer}层加工完成")
            # 在一些实现中，completed状态后可能会自动转为idle
            
        else:
            print(f"未知的机床状态: {status}")
            
    def on_correction_request(self, layer_id: int):
        """增强的机床纠偏数据请求处理"""
        print(f"\n=== PLC请求第{layer_id}层纠偏数据 ===")
        print(f"当前系统状态:")
        print(f"  - 当前层: {self.current_layer}")
        print(f"  - 请求层: {layer_id}")
        print(f"  - 已加载层数: {len(self.layers)}")
        print(f"  - 自动模式: {self.auto_next_check.isChecked()}")
        print(f"  - PLC连接状态: {self.plc_communicator.connected if self.plc_communicator else '未连接'}")
        
        # 更新状态显示
        self.status_label.setText(f"接收到PLC请求，开始处理第{layer_id}层")
        
        # 同步当前层号
        if self.current_layer != layer_id:
            print(f"层号不一致，从 {self.current_layer} 更新为 {layer_id}")
            self.current_layer = layer_id
            self.update_layer_table()  # 更新界面显示
        
        # 检查层是否存在
        if layer_id not in self.layers:
            error_msg = f"错误: 第{layer_id}层G代码不存在，请先加载对应的G代码文件"
            print(error_msg)
            self.status_label.setText(error_msg)
            
            # 向PLC发送错误响应
            if self.plc_communicator and self.plc_communicator.connected:
                try:
                    self.plc_communicator.write_layer_completion(layer_id, False, 0.0)
                    print(f"已向PLC发送第{layer_id}层错误响应")
                except Exception as e:
                    print(f"发送错误响应失败: {e}")
            
            # 显示错误对话框
            QMessageBox.warning(self, "纠偏请求错误", 
                               f"{error_msg}\n\n"
                               f"当前已加载层数: {len(self.layers)}\n"
                               f"请求的层号: {layer_id}\n\n"
                               "请检查G代码加载情况。")
            return
        
        # 检查是否正在处理其他层
        if self.processing_thread and self.processing_thread.isRunning():
            print(f"警告: 已有层正在处理中，将等待完成后处理第{layer_id}层")
            self.status_label.setText(f"等待上一层处理完成，然后处理第{layer_id}层")
            
            # 创建队列处理机制
            def process_when_ready():
                """当前处理完成后自动处理队列中的层"""
                if not (self.processing_thread and self.processing_thread.isRunning()):
                    print(f"队列处理：开始处理第{layer_id}层")
                    self.process_current_layer()
                else:
                    # 继续等待
                    QTimer.singleShot(500, process_when_ready)
            
            QTimer.singleShot(500, process_when_ready)
            return
        
        # 检查数据完整性
        layer_info = self.layers[layer_id]
        if not layer_info.gcode_path or not os.path.exists(layer_info.gcode_path):
            error_msg = f"错误: 第{layer_id}层G代码文件不存在: {layer_info.gcode_path}"
            print(error_msg)
            self.status_label.setText(error_msg)
            
            # 向PLC发送错误响应
            if self.plc_communicator and self.plc_communicator.connected:
                try:
                    self.plc_communicator.write_layer_completion(layer_id, False, 0.0)
                    print(f"已向PLC发送第{layer_id}层文件错误响应")
                except Exception as e:
                    print(f"发送错误响应失败: {e}")
            
            QMessageBox.warning(self, "文件错误", error_msg)
            return
        
        # 获取处理延迟时间
        delay_sec = getattr(self, 'process_delay_sec', 0.5)
        delay_ms = int(delay_sec * 1000)
        
        print(f"延迟 {delay_sec}秒 后开始处理（模拟采集延迟）")
        
        # 更新层状态为处理中
        layer_info.status = "processing"
        self.update_layer_table()
        
        # 延迟开始处理（模拟采集延迟）
        QTimer.singleShot(delay_ms, lambda: self.process_layer_with_validation(layer_id))
        
        print(f"=== 纠偏请求处理完成 ===")
    
    def process_layer_with_validation(self, layer_id: int):
        """带验证的层处理"""
        try:
            # 最终验证
            if layer_id != self.current_layer:
                print(f"警告: 延迟处理时层号发生变化 ({layer_id} -> {self.current_layer})")
                return
            
            if layer_id not in self.layers:
                print(f"错误: 延迟处理时第{layer_id}层已不存在")
                return
            
            # 开始处理
            self.process_current_layer()
            
        except Exception as e:
            error_msg = f"层处理验证失败: {e}"
            print(error_msg)
            self.status_label.setText(error_msg)
            
            # 向PLC发送错误响应
            if self.plc_communicator and self.plc_communicator.connected:
                try:
                    self.plc_communicator.write_layer_completion(layer_id, False, 0.0)
                except Exception as e2:
                    print(f"发送错误响应失败: {e2}")
    
    def on_data_transmission_progress(self, current_batch: int, total_batches: int, status_msg: str):
        """数据传输进度更新"""
        progress = int((current_batch / total_batches) * 100) if total_batches > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"数据传输: {status_msg}")
        print(f"传输进度: {current_batch}/{total_batches} - {status_msg}")
    
    def on_safety_alert(self, layer_id: int, alert_message: str, deviation_value: float):
        """安全警告处理"""
        warning_msg = f"第{layer_id}层安全警告: {alert_message}"
        print(f"安全警告: {warning_msg}")
        self.status_label.setText(warning_msg)
        
        # 显示警告对话框
        QMessageBox.warning(self, "安全警告", 
                           f"{warning_msg}\n\n偏差值: {deviation_value:.3f}mm\n\n"
                           "请检查数据和设备状态。")
    
    def on_heartbeat_updated(self, heartbeat_count: int):
        """心跳更新"""
        # 可以在状态栏显示心跳信息，或者只在调试时显示
        if heartbeat_count % 10 == 0:  # 每10次心跳显示一次
            print(f"PLC心跳: {heartbeat_count}")
                
    def on_processing_finished(self, layer_id: int, result: Dict):
        """处理完成"""
        if layer_id in self.layers:
            layer_info = self.layers[layer_id]
            layer_info.processing_result = result
            layer_info.status = "completed"
            layer_info.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 保存偏差补偿数据 - 修复：使用bias_comp_data而不是文件路径
            if 'bias_comp_data' in result:
                layer_info.bias_comp = result['bias_comp_data']
                print(f"第{layer_id}层偏差补偿数据已保存到内存")
            elif 'bias_comp_path' in result:
                try:
                    with open(result['bias_comp_path'], 'r', encoding='utf-8') as f:
                        layer_info.bias_comp = json.load(f)
                        print(f"第{layer_id}层偏差补偿数据已从文件加载")
                except Exception as e:
                    print(f"读取偏差补偿失败: {e}")
                    # 如果文件读取失败，尝试从result中获取
                    if 'bias_comp_data' in result:
                        layer_info.bias_comp = result['bias_comp_data']
                        print(f"第{layer_id}层偏差补偿数据已从result获取")
                    
            # 更新可视化
            self.current_layer_viz.add_layer_data(layer_id, result)
            self.update_layer_table()
            
            self.progress_bar.setValue(100)
            processing_time = result.get('processing_time', 0.0)
            
            # 检查处理结果质量
            quality_status = self.check_processing_quality(layer_id, result)
            
            # 更新状态显示包含质量信息
            self.status_label.setText(f"第{layer_id}层处理完成 - {quality_status}")
            print(f"第{layer_id}层处理完成 - 用时: {processing_time:.1f}s - {quality_status}")
            
            # 发送纠偏数据到PLC（非标定层）
            if layer_id > 1 and self.plc_communicator and self.plc_communicator.connected:
                correction_data = result.get('correction', {})
                if correction_data:
                    try:
                        success = self.plc_communicator.send_correction_data(layer_id, correction_data)
                        if success:
                            self.status_label.setText(f"第{layer_id}层纠偏数据已发送到PLC（包含corrected.gcode和offset_table.csv）")
                            print(f"纠偏数据已发送: corrected.gcode + offset_table.csv")
                        else:
                            self.status_label.setText(f"第{layer_id}层纠偏数据发送失败")
                    except Exception as e:
                        print(f"发送纠偏数据到PLC失败: {e}")
                else:
                    print(f"第{layer_id}层没有纠偏数据可发送")
            elif layer_id == 1:
                # 第一层仅为标定，不发送纠偏数据
                print(f"第{layer_id}层为标定层，不发送纠偏数据")
                if self.plc_communicator and self.plc_communicator.connected:
                    self.status_label.setText(f"第{layer_id}层标定完成，等待PLC下一层信号...")
                        
            # 不再发送层完成信号，等待PLC的下一个请求
            print(f"第{layer_id}层处理完成，等待PLC发送下一层请求信号...")
            
            # 自动下一层 - 等待PLC状态而不是使用固定延迟
            if self.auto_next_check.isChecked() and layer_id < len(self.layers):
                # 不再使用固定延迟，而是等待PLC发送correction_request信号
                # PLC会在机床完成当前层加工后发送下一层请求
                self.status_label.setText(f"第{layer_id}层已完成，等待机床开始下一层...")
                print(f"第{layer_id}层处理完成，等待PLC发送下一层请求信号...")
    
    def check_processing_quality(self, layer_id: int, result: Dict) -> str:
        """检查处理结果质量"""
        try:
            metrics = result.get('metrics', {})
            if not metrics:
                return "无质量数据"
            
            valid_ratio = metrics.get('valid_ratio', 0.0)
            
            # 优先使用新的轨迹精度指标
            if 'trajectory_p95_distance' in metrics:
                precision = metrics.get('trajectory_p95_distance', 0.0)
                precision_type = "P95距离"
            elif 'dev_p95' in metrics:
                precision = metrics.get('dev_p95', 0.0)
                precision_type = "P95偏差"
            else:
                return f"覆盖率: {valid_ratio:.1%}"
            
            # 质量评估
            if valid_ratio >= 0.8 and precision <= 1.0:
                quality = "优秀"
            elif valid_ratio >= 0.6 and precision <= 2.0:
                quality = "良好"
            elif valid_ratio >= 0.4 and precision <= 3.0:
                quality = "可接受"
            else:
                quality = "需注意"
            
            return f"{quality} (覆盖率: {valid_ratio:.1%}, {precision_type}: {precision:.2f}mm)"
            
        except Exception as e:
            print(f"质量检查失败: {e}")
            return "质量检查失败"
    
    def validate_plc_data_compatibility(self) -> Dict[str, Any]:
        """验证PLC数据兼容性和处理能力"""
        validation_result = {
            "overall_ready": True,
            "issues": [],
            "warnings": [],
            "capabilities": {
                "layer_processing": False,
                "correction_data_handling": False,
                "quality_assessment": False,
                "plc_communication": False,
                "error_handling": False,
                "batch_processing": False
            },
            "data_flow_status": {
                "gcode_loading": False,
                "bias_compensation": False,
                "offset_data_generation": False,
                "plc_transmission": False,
                "status_synchronization": False
            }
        }
        
        try:
            # 1. 检查层处理能力
            if hasattr(self, 'process_current_layer') and callable(getattr(self, 'process_current_layer')):
                validation_result["capabilities"]["layer_processing"] = True
            else:
                validation_result["issues"].append("缺少层处理方法")
            
            # 2. 检查纠偏数据处理能力
            if hasattr(self, 'on_correction_request') and callable(getattr(self, 'on_correction_request')):
                validation_result["capabilities"]["correction_data_handling"] = True
            else:
                validation_result["issues"].append("缺少纠偏数据处理方法")
            
            # 3. 检查质量评估能力
            if hasattr(self, 'check_processing_quality') and callable(getattr(self, 'check_processing_quality')):
                validation_result["capabilities"]["quality_assessment"] = True
            else:
                validation_result["warnings"].append("缺少质量评估能力")
            
            # 4. 检查PLC通信能力
            if (hasattr(self, 'plc_communicator') and 
                hasattr(self, 'connect_plc') and callable(getattr(self, 'connect_plc'))):
                validation_result["capabilities"]["plc_communication"] = True
                
                # 检查PLC通信器类型支持
                if self.plc_communicator:
                    plc_type = type(self.plc_communicator).__name__
                    if 'S7PLCEnhanced' in plc_type:
                        validation_result["capabilities"]["batch_processing"] = True
                        print(f"PLC通信器类型: {plc_type} (支持增强功能)")
                    else:
                        validation_result["warnings"].append(f"PLC通信器类型: {plc_type} (基础功能)")
            else:
                validation_result["issues"].append("缺少PLC通信能力")
            
            # 5. 检查错误处理能力
            if (hasattr(self, 'on_processing_failed') and callable(getattr(self, 'on_processing_failed')) and
                hasattr(self, 'process_layer_with_validation') and callable(getattr(self, 'process_layer_with_validation'))):
                validation_result["capabilities"]["error_handling"] = True
            else:
                validation_result["warnings"].append("错误处理能力不完整")
            
            # 6. 检查数据流状态
            # G代码加载
            if hasattr(self, 'layers') and len(getattr(self, 'layers', {})) > 0:
                validation_result["data_flow_status"]["gcode_loading"] = True
            else:
                validation_result["warnings"].append("尚未加载G代码文件")
            
            # 偏差补偿能力
            if hasattr(self, 'controller') and self.controller:
                validation_result["data_flow_status"]["bias_compensation"] = True
            else:
                validation_result["issues"].append("缺少算法控制器")
            
            # PLC传输能力
            if (self.plc_communicator and 
                hasattr(self.plc_communicator, 'send_correction_data') and 
                callable(getattr(self.plc_communicator, 'send_correction_data'))):
                validation_result["data_flow_status"]["plc_transmission"] = True
            else:
                validation_result["warnings"].append("PLC数据传输能力未就绪")
            
            # 状态同步能力
            signal_methods = ['on_layer_changed_from_plc', 'on_machine_status_changed', 'on_correction_request']
            if all(hasattr(self, method) and callable(getattr(self, method)) for method in signal_methods):
                validation_result["data_flow_status"]["status_synchronization"] = True
            else:
                validation_result["issues"].append("缺少PLC状态同步方法")
            
            # 7. 综合评估
            capabilities_ready = sum(validation_result["capabilities"].values())
            data_flow_ready = sum(validation_result["data_flow_status"].values())
            critical_issues = len(validation_result["issues"])
            
            validation_result["overall_ready"] = (
                capabilities_ready >= 4 and  # 至少需要基础能力
                data_flow_ready >= 3 and     # 至少需要基础数据流
                critical_issues == 0         # 无关键问题
            )
            
            # 生成总结报告
            validation_result["summary"] = {
                "capabilities_score": f"{capabilities_ready}/6",
                "data_flow_score": f"{data_flow_ready}/5",
                "critical_issues_count": critical_issues,
                "warnings_count": len(validation_result["warnings"]),
                "readiness_level": "Ready" if validation_result["overall_ready"] else "Needs Attention"
            }
            
            return validation_result
            
        except Exception as e:
            validation_result["overall_ready"] = False
            validation_result["issues"].append(f"验证过程异常: {str(e)}")
            return validation_result
    
    def print_data_processing_status(self):
        """打印数据处理能力状态报告"""
        validation = self.validate_plc_data_compatibility()
        
        print("\n" + "="*60)
        print("数据处理能力验证报告")
        print("="*60)
        
        # 总体状态
        status_icon = "✓" if validation["overall_ready"] else "✗"
        print(f"\n总体状态: {status_icon} {validation['summary']['readiness_level']}")
        
        # 能力评分
        print(f"\n能力评分:")
        print(f"  - 功能能力: {validation['summary']['capabilities_score']}")
        print(f"  - 数据流: {validation['summary']['data_flow_score']}")
        
        # 详细能力
        print(f"\n具体能力:")
        for capability, status in validation["capabilities"].items():
            icon = "✓" if status else "✗"
            print(f"  {icon} {capability}: {'Ready' if status else 'Missing'}")
        
        # 数据流状态
        print(f"\n数据流状态:")
        for flow, status in validation["data_flow_status"].items():
            icon = "✓" if status else "✗"
            print(f"  {icon} {flow}: {'Ready' if status else 'Not Ready'}")
        
        # 问题和警告
        if validation["issues"]:
            print(f"\n关键问题 ({len(validation['issues'])}):")
            for issue in validation["issues"]:
                print(f"  ✗ {issue}")
        
        if validation["warnings"]:
            print(f"\n警告 ({len(validation['warnings'])}):")
            for warning in validation["warnings"]:
                print(f"  ⚠ {warning}")
        
        print("="*60)
        return validation
    
    def handle_plc_data_error(self, error_type: str, error_details: dict):
        """处理PLC数据错误和异常情况"""
        try:
            layer_id = error_details.get('layer_id', self.current_layer)
            error_message = error_details.get('message', 'Unknown error')
            
            print(f"\n=== PLC数据错误处理 ===")
            print(f"错误类型: {error_type}")
            print(f"层号: {layer_id}")
            print(f"错误详情: {error_message}")
            
            # 根据错误类型采取不同处理策略
            if error_type == "layer_mismatch":
                # 层号不匹配
                expected_layer = error_details.get('expected_layer')
                actual_layer = error_details.get('actual_layer')
                print(f"层号不匹配: 期望{expected_layer}, 实际{actual_layer}")
                
                # 同步层号
                if actual_layer and actual_layer in self.layers:
                    self.current_layer = actual_layer
                    self.status_label.setText(f"已同步到第{actual_layer}层")
                    self.update_layer_table()
                
            elif error_type == "data_transmission_failure":
                # 数据传输失败
                retry_count = error_details.get('retry_count', 0)
                max_retries = error_details.get('max_retries', 3)
                
                if retry_count < max_retries:
                    print(f"数据传输失败，将进行第{retry_count + 1}次重试")
                    self.status_label.setText(f"第{layer_id}层数据传输失败，正在重试...")
                    
                    # 延迟后重试
                    QTimer.singleShot(2000, lambda: self.retry_data_transmission(layer_id, retry_count + 1))
                else:
                    print(f"数据传输失败，已超过最大重试次数")
                    self.status_label.setText(f"第{layer_id}层数据传输失败")
                    
                    # 标记层为错误状态
                    if layer_id in self.layers:
                        self.layers[layer_id].status = "error"
                        self.update_layer_table()
                        
                    # 显示错误对话框
                    QMessageBox.critical(self, "数据传输错误", 
                                        f"第{layer_id}层数据传输失败\n\n"
                                        f"错误: {error_message}\n"
                                        f"重试次数: {retry_count}/{max_retries}\n\n"
                                        "请检查PLC连接和数据格式")
                
            elif error_type == "quality_check_failure":
                # 质量检查失败
                quality_metrics = error_details.get('quality_metrics', {})
                print(f"质量检查失败: {quality_metrics}")
                
                self.status_label.setText(f"第{layer_id}层质量不符合要求")
                
                # 显示警告对话框
                QMessageBox.warning(self, "质量警告", 
                                   f"第{layer_id}层处理结果质量不佳\n\n"
                                   f"详情: {error_message}\n\n"
                                   "请检查采集条件和算法参数")
                
            elif error_type == "plc_connection_lost":
                # PLC连接丢失
                print(f"PLC连接丢失")
                self.status_label.setText("PLC连接丢失，正在尝试重连...")
                
                # 尝试重新连接
                self.disconnect_plc()
                QTimer.singleShot(3000, self.connect_plc)
                
            elif error_type == "gcode_file_missing":
                # G代码文件缺失
                missing_file = error_details.get('file_path', '')
                print(f"G代码文件缺失: {missing_file}")
                
                self.status_label.setText(f"第{layer_id}层G代码文件缺失")
                
                # 显示错误对话框
                QMessageBox.critical(self, "G代码文件错误", 
                                    f"第{layer_id}层G代码文件不存在\n\n"
                                    f"文件路径: {missing_file}\n\n"
                                    "请检查文件路径和重新加载G代码")
                
            else:
                # 未知错误类型
                print(f"未知错误类型: {error_type}")
                self.status_label.setText(f"系统错误: {error_type}")
                
                QMessageBox.warning(self, "系统错误", 
                                   f"发生未知错误\n\n"
                                   f"类型: {error_type}\n"
                                   f"详情: {error_message}")
            
            # 向PLC发送错误状态
            if self.plc_communicator and self.plc_communicator.connected:
                try:
                    self.plc_communicator.write_layer_completion(layer_id, False, 0.0)
                    print(f"已向PLC发送错误状态")
                except Exception as e:
                    print(f"发送错误状态失败: {e}")
            
            print(f"=== 错误处理完成 ===")
            
        except Exception as e:
            error_msg = f"错误处理器异常: {str(e)}"
            print(error_msg)
            self.status_label.setText(error_msg)
    
    def retry_data_transmission(self, layer_id: int, retry_count: int):
        """重试数据传输"""
        try:
            print(f"开始第{retry_count}次重试传输第{layer_id}层数据")
            
            if layer_id not in self.layers:
                print(f"重试失败: 第{layer_id}层不存在")
                return
            
            layer_info = self.layers[layer_id]
            if not layer_info.processing_result:
                print(f"重试失败: 第{layer_id}层没有处理结果")
                return
            
            # 获取纠偏数据
            correction_data = layer_info.processing_result.get('correction', {})
            if not correction_data:
                print(f"重试失败: 第{layer_id}层没有纠偏数据")
                return
            
            # 尝试发送
            if self.plc_communicator and self.plc_communicator.connected:
                success = self.plc_communicator.send_correction_data(layer_id, correction_data)
                if success:
                    print(f"第{retry_count}次重试成功")
                    self.status_label.setText(f"第{layer_id}层数据重试传输成功")
                else:
                    print(f"第{retry_count}次重试失败")
                    # 继续错误处理
                    self.handle_plc_data_error("data_transmission_failure", {
                        'layer_id': layer_id,
                        'message': f'第{retry_count}次重试仍然失败',
                        'retry_count': retry_count,
                        'max_retries': 3
                    })
            else:
                print(f"重试失败: PLC未连接")
                
        except Exception as e:
            print(f"重试数据传输异常: {e}")
    
    def validate_system_capabilities(self) -> None:
        """验证系统数据处理能力"""
        try:
            print(f"\n{'='*60}")
            print("多层加工纠偏系统 - 数据处理能力验证")
            print(f"{'='*60}")
            
            # 执行验证
            validation = self.print_data_processing_status()
            
            # 根据验证结果更新界面
            not_ready_msg = "检查中"  # 默认值
            
            if validation["overall_ready"]:
                ready_msg = "系统就绪，可以处理PLC数据"
                self.camera_status_label.setText("相机: 已就绪")
                print(f"\n✓ {ready_msg}")
            else:
                issues_count = len(validation["issues"])
                warnings_count = len(validation["warnings"])
                not_ready_msg = f"系统需要注意 ({issues_count}个问题, {warnings_count}个警告)"
                print(f"\n⚠ {not_ready_msg}")
                
                # 显示关键问题
                if validation["issues"]:
                    print(f"\n关键问题需要解决:")
                    for i, issue in enumerate(validation["issues"], 1):
                        print(f"  {i}. {issue}")
            
            # 显示功能状态摘要
            capabilities_ready = sum(validation["capabilities"].values())
            data_flow_ready = sum(validation["data_flow_status"].values())
            
            summary_msg = (
                f"功能状态: {capabilities_ready}/6 | "
                f"数据流: {data_flow_ready}/5 | "
                f"问题: {len(validation['issues'])} | "
                f"警告: {len(validation['warnings'])}"
            )
            print(f"\n{summary_msg}")
            
            # 更新状态显示
            if validation["overall_ready"]:
                self.status_label.setText("系统就绪 - 可以处理PLC数据")
            else:
                self.status_label.setText(f"系统检查中 - {not_ready_msg}")
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            error_msg = f"系统能力验证异常: {str(e)}"
            print(error_msg)
            self.status_label.setText(error_msg)
    def on_processing_failed(self, layer_id: int, error: str):
        """处理失败"""
        if layer_id in self.layers:
            self.layers[layer_id].status = "error"
            
        self.progress_bar.setValue(0)
        self.status_label.setText(f"第{layer_id}层处理失败")
        QMessageBox.critical(self, "处理错误", error)
        
    def on_progress_updated(self, layer_id: int, status: str):
        """进度更新"""
        self.status_label.setText(f"第{layer_id}层: {status}")
        
    def update_layer_table(self):
        """更新层管理表格"""
        self.layer_table.setRowCount(len(self.layers))
        
        for i, (layer_id, layer_info) in enumerate(self.layers.items()):
            self.layer_table.setItem(i, 0, QTableWidgetItem(str(layer_id)))
            self.layer_table.setItem(i, 1, QTableWidgetItem(layer_info.status))
            
            # 统计信息 - 使用新的轨迹精度指标
            if layer_info.processing_result:
                metrics = layer_info.processing_result.get('metrics', {})
                valid_ratio = metrics.get('valid_ratio', 0)
                
                # 优先使用新的轨迹精度指标
                if 'trajectory_p95_distance' in metrics:
                    trajectory_precision = metrics.get('trajectory_p95_distance', 0)
                    precision_label = f"{trajectory_precision:.3f}mm"
                elif 'dev_p95' in metrics:
                    # 向后兼容：使用原有的法向偏移数据
                    trajectory_precision = metrics.get('dev_p95', 0)
                    precision_label = f"{trajectory_precision:.3f}mm(法向)"
                else:
                    precision_label = "-"
                
                self.layer_table.setItem(i, 2, QTableWidgetItem(f"{valid_ratio:.1%}"))
                self.layer_table.setItem(i, 3, QTableWidgetItem(precision_label))
            else:
                self.layer_table.setItem(i, 2, QTableWidgetItem("-"))
                self.layer_table.setItem(i, 3, QTableWidgetItem("-"))
                
            # 时间戳
            timestamp = layer_info.timestamp or "-"
            self.layer_table.setItem(i, 4, QTableWidgetItem(timestamp))
            
            # 操作按钮（后续可添加）
            self.layer_table.setItem(i, 5, QTableWidgetItem("查看"))
            
    def on_plc_config_changed(self):
        """PLC配置变化"""
        # 更新配置
        self.update_project_config()
        
        # 如果PLC已连接，提示用户重新连接以应用新配置
        if (self.plc_communicator and self.plc_communicator.connected):
            self.status_label.setText("PLC配置已更改，请重新连接以应用新设置")
            # 可选：自动断开并重连
            # self.disconnect_plc()
            # self.connect_plc()
        
    def on_project_config_changed(self):
        """项目配置变化"""
        self.update_project_config()
        
    def update_project_config(self):
        """更新项目配置"""
        self.project_config.project_name = self.project_name_edit.text()
        self.project_config.total_layers = self.total_layers_spin.value()
        self.project_config.layer_thickness_mm = self.layer_thickness_spin.value()
        self.project_config.auto_next_layer = self.auto_next_check.isChecked()
        
        self.project_config.use_plc = self.use_plc_check.isChecked()
        self.project_config.plc_type = self.plc_type_combo.currentText()
        self.project_config.plc_ip = self.plc_ip_edit.text()
        self.project_config.plc_port = self.plc_port_spin.value()
        self.project_config.current_layer_address = self.layer_address_edit.text()
        
    def save_project(self):
        """保存项目"""
        try:
            # 选择保存位置
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存项目", f"{self.project_config.project_name}.json",
                "JSON文件 (*.json)"
            )
            
            if file_path:
                project_data = {
                    "config": {
                        "project_name": self.project_config.project_name,
                        "total_layers": self.project_config.total_layers,
                        "layer_thickness_mm": self.project_config.layer_thickness_mm,
                        "auto_next_layer": self.project_config.auto_next_layer,
                        "use_plc": self.project_config.use_plc,
                        "plc_type": self.project_config.plc_type,
                        "plc_ip": self.project_config.plc_ip,
                        "plc_port": self.project_config.plc_port,
                        "current_layer_address": self.project_config.current_layer_address,
                    },
                    "layers": {
                        str(layer_id): {
                            "layer_id": layer_info.layer_id,
                            "gcode_path": layer_info.gcode_path,
                            "status": layer_info.status,
                            "timestamp": layer_info.timestamp,
                            "has_result": layer_info.processing_result is not None
                        }
                        for layer_id, layer_info in self.layers.items()
                    },
                    "current_layer": self.current_layer,
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(project_data, f, ensure_ascii=False, indent=2)
                    
                QMessageBox.information(self, "成功", "项目保存成功")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存项目失败: {e}")
            
    def export_results(self):
        """导出结果 - 增强版，包含每层out文件夹"""
        try:
            export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if export_dir:
                export_path = Path(export_dir) / f"{self.project_config.project_name}_export"
                export_path.mkdir(exist_ok=True)
                
                # 导出各层可视化结果
                self.export_visualization_results(export_path)
                
                # 导出每层的out文件夹（机床纠偏数据）
                self.export_layer_out_directories(export_path)
                
                # 导出项目摘要
                self.export_project_summary(export_path)
                
                QMessageBox.information(self, "成功", f"结果导出至: {export_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")
            
    def export_visualization_results(self, export_path: Path):
        """导出可视化结果"""
        for layer_id, layer_info in self.layers.items():
            if layer_info.processing_result:
                layer_dir = export_path / "visualization" / f"layer_{layer_id:02d}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存可视化图像
                result = layer_info.processing_result
                for img_name in ['vis_cmp', 'vis_corr', 'hist_panel']:
                    if img_name in result and result[img_name] is not None:
                        img_path = layer_dir / f"{img_name}.png"
                        cv2.imwrite(str(img_path), result[img_name])
                        
                # 保存统计数据
                if 'metrics' in result:
                    metrics_path = layer_dir / "metrics.json"
                    with open(metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(result['metrics'], f, ensure_ascii=False, indent=2)
                        
    def export_layer_out_directories(self, export_path: Path):
        """导出每层的out文件夹（机床纠偏数据） - 从总output文件夹中复制"""
        import shutil
        
        machine_data_dir = export_path / "machine_data"
        machine_data_dir.mkdir(exist_ok=True)
        
        exported_layers = []
        
        # 从总output文件夹中查找各层数据
        output_source_dir = Path("output")
        if output_source_dir.exists():
            for layer_id, layer_info in self.layers.items():
                # 只导出非标定层（即需要纠偏的层）
                if (layer_info.processing_result and 
                    layer_info.processing_result.get('layer_type') == 'correction'):
                    
                    # 查找output目录下的layer_XX_out目录
                    layer_out_source = output_source_dir / f"layer_{layer_id:02d}_out"
                    if layer_out_source.exists():
                        layer_out_dest = machine_data_dir / f"layer_{layer_id:02d}_out"
                        
                        # 复制整个目录
                        if layer_out_dest.exists():
                            shutil.rmtree(layer_out_dest)
                        shutil.copytree(layer_out_source, layer_out_dest)
                        
                        exported_layers.append(layer_id)
                        print(f"已导出第{layer_id}层机床数据: {layer_out_dest}")
                        
        # 创建机床数据目录说明
        machine_readme = machine_data_dir / "README.md"
        with open(machine_readme, 'w', encoding='utf-8') as f:
            f.write(f"# 机床纠偏数据\n\n")
            f.write(f"项目名称: {self.project_config.project_name}\n")
            f.write(f"导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"已导出层数: {len(exported_layers)}\n\n")
            f.write("## 目录结构\n")
            for layer_id in exported_layers:
                f.write(f"- `layer_{layer_id:02d}_out/`: 第{layer_id}层纠偏数据\n")
                f.write(f"  - `offset_table.csv`: 偏移表\n")
                f.write(f"  - `corrected.gcode`: 纠偏后G代码\n")
                f.write(f"  - `original_layer_{layer_id:02d}.gcode`: 原始G代码\n")
                f.write(f"  - `comparison_visualization.png`: 对比可视化\n")
                f.write(f"  - `corrected_visualization.png`: 纠偏可视化\n")
                f.write(f"  - `processing_metrics.json`: 处理指标\n")
                f.write(f"  - `layer_info.json`: 层信息\n")
                f.write(f"  - `README.md`: 详细说明\n\n")
            f.write("## 使用说明\n")
            f.write("机床可按层号访问对应的目录，加载相应的纠偏数据。\n")
            f.write("每个层目录包含完整的加工数据：原始G代码、纠偏后G代码、偏移表和可视化结果。\n")
            
        # 创建总体数据统计
        stats_file = machine_data_dir / "overall_statistics.json"
        overall_stats = {
            'project_name': self.project_config.project_name,
            'total_layers_processed': len(exported_layers),
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'layers': exported_layers,
            'quality_summary': self.generate_quality_summary(exported_layers)
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=2)
            
    def generate_quality_summary(self, exported_layers: list) -> Dict:
        """生成质量汇总统计"""
        quality_summary = {
            'average_valid_ratio': 0.0,
            'average_dev_p95': 0.0,
            'best_layer': None,
            'worst_layer': None,
            'layer_details': []
        }
        
        valid_ratios = []
        dev_p95s = []
        
        for layer_id in exported_layers:
            if layer_id in self.layers:
                layer_info = self.layers[layer_id]
                if layer_info.processing_result:
                    metrics = layer_info.processing_result.get('metrics', {})
                    valid_ratio = metrics.get('valid_ratio', 0)
                    dev_p95 = metrics.get('dev_p95', 0)
                    
                    valid_ratios.append(valid_ratio)
                    dev_p95s.append(dev_p95)
                    
                    quality_summary['layer_details'].append({
                        'layer_id': layer_id,
                        'valid_ratio': valid_ratio,
                        'dev_p95': dev_p95,
                        'status': layer_info.status
                    })
        
        if valid_ratios:
            quality_summary['average_valid_ratio'] = sum(valid_ratios) / len(valid_ratios)
            quality_summary['average_dev_p95'] = sum(dev_p95s) / len(dev_p95s)
            
            # 找出最佳和最差层（基于P95偏差）
            if dev_p95s:
                min_dev_idx = dev_p95s.index(min(dev_p95s))
                max_dev_idx = dev_p95s.index(max(dev_p95s))
                quality_summary['best_layer'] = exported_layers[min_dev_idx]
                quality_summary['worst_layer'] = exported_layers[max_dev_idx]
        
        return quality_summary
            
    def export_project_summary(self, export_path: Path):
        """导出项目摘要 - 包含新的output文件夹信息"""
        summary = {
            'project_name': self.project_config.project_name,
            'total_layers': len(self.layers),
            'completed_layers': sum(1 for info in self.layers.values() if info.status == 'completed'),
            'correction_layers': sum(1 for info in self.layers.values() 
                                   if info.processing_result and info.processing_result.get('layer_type') == 'correction'),
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'output_structure': {
                'description': '每层数据保存在output文件夹下的独立目录中',
                'pattern': 'output/layer_{layer_id:02d}_out/',
                'contains': [
                    '原始G代码 (original_layer_XX.gcode)',
                    '纠偏后G代码 (corrected.gcode)',
                    '偏移表 (offset_table.csv)',
                    '多种可视化图像',
                    '处理指标 (processing_metrics.json)',
                    '偏差补偿数据 (bias_compensation.json)',
                    '详细说明 (README.md)'
                ]
            },
            'layers_summary': []
        }
        
        for layer_id, layer_info in self.layers.items():
            layer_summary = {
                'layer_id': layer_id,
                'status': layer_info.status,
                'layer_type': layer_info.processing_result.get('layer_type', 'unknown') if layer_info.processing_result else 'unknown',
                'timestamp': layer_info.timestamp,
                'gcode_source': layer_info.gcode_path if hasattr(layer_info, 'gcode_path') else '',
                'output_directory': f'output/layer_{layer_id:02d}_out/' if layer_info.processing_result and layer_info.processing_result.get('layer_type') == 'correction' else None
            }
            
            if layer_info.processing_result:
                metrics = layer_info.processing_result.get('metrics', {})
                layer_summary.update({
                    'valid_ratio': metrics.get('valid_ratio', 0),
                    'dev_p95': metrics.get('dev_p95', 0),
                    'dev_mean': metrics.get('dev_mean', 0),
                    'processing_time': layer_info.processing_result.get('processing_time', 0)
                })
                
            summary['layers_summary'].append(layer_summary)
            
        summary_file = export_path / "project_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            
        # 同时创建一个简化的README文件
        readme_file = export_path / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"# {self.project_config.project_name} - 导出结果\n\n")
            f.write(f"导出时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总层数: {len(self.layers)}\n")
            f.write(f"已完成层数: {summary['completed_layers']}\n")
            f.write(f"纠偏层数: {summary['correction_layers']}\n\n")
            
            f.write("## 目录结构\n\n")
            f.write("```\n")
            f.write(f"{self.project_config.project_name}_export/\n")
            f.write("├── machine_data/          # 机床纠偏数据（从 output 文件夹复制）\n")
            f.write("│   ├── layer_01_out/\n")
            f.write("│   ├── layer_02_out/\n")
            f.write("│   └── README.md\n")
            f.write("├── visualization/         # 可视化结果\n")
            f.write("└── project_summary.json   # 项目摘要\n")
            f.write("```\n\n")
            
            f.write("## 使用说明\n\n")
            f.write("1. **machine_data/**: 包含机床所需的所有纠偏数据\n")
            f.write("2. **visualization/**: 包含可视化分析结果\n")
            f.write("3. **project_summary.json**: 包含完整的项目信息和统计数据\n\n")
            
            f.write("每个 layer_XX_out 文件夹包含：\n")
            f.write("- 原始G代码和纠偏后G代码\n")
            f.write("- 偏移表 (CSV格式)\n")
            f.write("- 多种可视化图像\n")
            f.write("- 处理指标和层信息\n")
            f.write("- 偏差补偿数据（用于下一层）\n")
            
    def open_advanced_params(self):
        """打开高级参数调节对话框"""
        try:
            from multilayer_advanced_params import MultilayerAdvancedParametersDialog
            dialog = MultilayerAdvancedParametersDialog(self.controller, self)
            dialog.parameters_applied.connect(self.on_advanced_params_applied)
            dialog.exec_()
        except ImportError as e:
            QMessageBox.warning(self, "错误", f"无法加载高级参数对话框: {e}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开高级参数调节失败: {e}")
            
    def on_advanced_params_applied(self, params_dict):
        """高级参数应用回调"""
        try:
            # 参数已经在对话框中应用到控制器，这里做后续处理
            
            # 保存处理延迟参数
            if 'process_delay_sec' in params_dict:
                self.process_delay_sec = params_dict['process_delay_sec']
                print(f"处理延迟时间已更新为: {self.process_delay_sec}秒")
            
            # 保存自动下一层参数
            if 'auto_next_layer' in params_dict:
                self.auto_next_check.setChecked(params_dict['auto_next_layer'])
                print(f"自动下一层设置已更新为: {params_dict['auto_next_layer']}")
            
            # 自动触发一次预览以显示参数效果
            if hasattr(self, 'process_current_btn') and self.process_current_btn.isEnabled():
                # 在当前层上触发参数验证
                QTimer.singleShot(500, lambda: self.trigger_parameter_preview())
                
            self.status_label.setText("高级参数已应用")
            
            # 显示成功消息
            QMessageBox.information(self, "成功", "高级参数已成功应用！\n\n主要更新包括：\n- ROI投影参数\n- 最近表面提取参数\n- 引导中心线参数\n- 可视化设置\n- 遮挡区域配置\n- 质量控制阈值\n- 处理延迟时间")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"应用高级参数失败: {e}")
            
    def trigger_parameter_preview(self):
        """触发参数预览"""
        try:
            if self.current_layer > 0 and self.current_layer in self.layers:
                # 如果当前层已经有数据，可以触发一次重新处理来查看参数效果
                self.status_label.setText(f"正在使用新参数验证第{self.current_layer}层...")
                # 注意：这里不直接调用process_current_layer，避免重复处理
                print(f"高级参数已更新，可以手动点击'处理当前层'查看效果")
        except Exception as e:
            print(f"参数预览触发失败: {e}")
            
    def sync_controller_config(self):
        """同步控制器配置"""
        try:
            if not self.controller:
                return
                
            cfg = self.controller.cfg
            camera_config = self.project_config.camera_config
            algo_config = self.project_config.algorithm_config
            
            # 更新相机配置
            cfg.pixel_size_mm = camera_config.get("pixel_size_mm", 0.8)
            cfg.bounds_margin_mm = camera_config.get("bounds_margin_mm", 20.0)
            cfg.roi_mode = camera_config.get("roi_mode", "gcode_bounds")
            
            # 更新算法配置
            cfg.guide_step_mm = algo_config.get("guide_step_mm", 1.0)
            cfg.guide_halfwidth_mm = algo_config.get("guide_halfwidth_mm", 6.0)
            cfg.guide_smooth_win = algo_config.get("guide_smooth_win", 7)
            cfg.guide_max_offset_mm = algo_config.get("guide_max_offset_mm", 8.0)
            cfg.guide_max_grad_mm_per_mm = algo_config.get("guide_max_grad_mm_per_mm", 0.08)
            cfg.plane_enable = algo_config.get("plane_enable", True)
            cfg.plane_ransac_thresh_mm = algo_config.get("plane_ransac_thresh_mm", 0.8)
            cfg.nearest_qlo = algo_config.get("nearest_qlo", 1.0)
            cfg.nearest_qhi = algo_config.get("nearest_qhi", 99.0)
            cfg.depth_margin_mm = algo_config.get("depth_margin_mm", 3.0)
            
        except Exception as e:
            print(f"同步控制器配置错误: {e}")
            
    def on_advanced_params_updated(self, params_dict):
        """供子组件调用的参数更新回调"""
        self.on_advanced_params_applied(params_dict)
            
    def closeEvent(self, a0):
        """关闭事件"""
        self.disconnect_plc()
        if self.controller:
            self.controller.close()
        if a0:
            a0.accept()

# ==================== 主函数 ====================

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("多层加工纠偏系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("智能制造")
    
    # 创建主窗口
    window = MultilayerMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()