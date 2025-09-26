#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7 PLC模拟器 - 支持snap7通信协议的可视化PLC模拟器
模拟西门子PLC的数据块读写操作，用于测试主程序的S7通信功能
"""

import sys
import threading
import time
import struct
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QComboBox, QTextEdit, QTabWidget,
    QHeaderView, QMessageBox, QCheckBox, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor

# 导入TCP服务器
from s7_tcp_server import S7SimulatorTCPServer

class S7DataBlock:
    """S7数据块模拟类"""
    
    def __init__(self, db_number: int, size: int = 512):
        self.db_number = db_number
        self.size = size
        self.data = bytearray(size)  # 初始化为全0
        self.lock = threading.Lock()
        
    def read(self, start: int, length: int) -> bytes:
        """读取数据块"""
        with self.lock:
            if start < 0 or start + length > self.size:
                raise ValueError(f"读取范围超出数据块大小: {start}+{length} > {self.size}")
            return bytes(self.data[start:start + length])
    
    def write(self, start: int, data: bytes) -> None:
        """写入数据块"""
        with self.lock:
            if start < 0 or start + len(data) > self.size:
                raise ValueError(f"写入范围超出数据块大小: {start}+{len(data)} > {self.size}")
            self.data[start:start + len(data)] = data
    
    def read_int16(self, address: int) -> int:
        """读取16位整数"""
        data = self.read(address, 2)
        return struct.unpack('>h', data)[0]  # 大端序
    
    def write_int16(self, address: int, value: int) -> None:
        """写入16位整数"""
        data = struct.pack('>h', value)  # 大端序
        self.write(address, data)
    
    def read_int32(self, address: int) -> int:
        """读取32位整数"""
        data = self.read(address, 4)
        return struct.unpack('>i', data)[0]
    
    def write_int32(self, address: int, value: int) -> None:
        """写入32位整数"""
        data = struct.pack('>i', value)
        self.write(address, data)

class S7PLCSimulator(QObject):
    """S7 PLC模拟器核心"""
    
    # 信号定义
    data_changed = pyqtSignal(int, int, int)  # db_number, address, value
    status_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # 数据块定义
        self.data_blocks = {
            9044: S7DataBlock(9044),  # 控制状态数据块
            9045: S7DataBlock(9045),  # 偏移数据块1
            9046: S7DataBlock(9046),  # 偏移数据块2  
            9047: S7DataBlock(9047),  # 偏移数据块3
        }
        
        # PLC状态
        self.machine_status = "idle"  # idle, running, completed, error
        self.current_layer = 1
        self.total_layers = 6
        self.processing_lock = False
        
        # 初始化数据块
        self.init_data_blocks()
        
        # 自动状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status_data)
        self.status_timer.start(1000)  # 每秒更新一次
    
    def init_data_blocks(self):
        """初始化数据块数据"""
        db_control = self.data_blocks[9044]
        
        # DB9044 控制数据块布局 (0-50字节):
        # 0-1: 机床状态 (0=idle, 1=running, 2=completed, 3=error)
        # 2-3: 当前层号
        # 4-5: 总层数
        # 6-7: 处理锁 (0=unlock, 1=locked)
        # 8-9: 偏移点总数
        # 10-11: 当前批次号
        # 12-13: 总批次数
        # 14-15: 数据就绪标志 (0=not_ready, 1=ready)
        # 16-17: 心跳计数
        # 18-19: 错误码
        # 20-21: 算法处理延时(ms)
        
        db_control.write_int16(0, 0)   # 机床状态: idle
        db_control.write_int16(2, self.current_layer)
        db_control.write_int16(4, self.total_layers)
        db_control.write_int16(6, 0)   # 处理锁: unlock
        db_control.write_int16(8, 0)   # 偏移点总数
        db_control.write_int16(10, 0)  # 当前批次号
        db_control.write_int16(12, 0)  # 总批次数
        db_control.write_int16(14, 0)  # 数据就绪: not_ready
        db_control.write_int16(16, 0)  # 心跳计数
        db_control.write_int16(18, 0)  # 错误码
        db_control.write_int16(20, 2000)  # 处理延时: 2000ms
    
    def db_read(self, db_number: int, start: int, length: int) -> bytes:
        """模拟snap7的db_read操作"""
        if db_number not in self.data_blocks:
            raise ValueError(f"数据块 DB{db_number} 不存在")
        
        return self.data_blocks[db_number].read(start, length)
    
    def db_write(self, db_number: int, start: int, data: bytes) -> None:
        """模拟snap7的db_write操作"""
        if db_number not in self.data_blocks:
            raise ValueError(f"数据块 DB{db_number} 不存在")
        
        self.data_blocks[db_number].write(start, data)
        
        # 发送数据变化信号
        self.data_changed.emit(db_number, start, len(data))
    
    def get_machine_status(self) -> str:
        """获取机床状态"""
        status_code = self.data_blocks[9044].read_int16(0)
        status_map = {0: "idle", 1: "running", 2: "completed", 3: "error"}
        return status_map.get(status_code, "unknown")
    
    def set_machine_status(self, status: str):
        """设置机床状态"""
        status_map = {"idle": 0, "running": 1, "completed": 2, "error": 3}
        if status in status_map:
            self.machine_status = status
            self.data_blocks[9044].write_int16(0, status_map[status])
            self.status_changed.emit(f"机床状态: {status}")
    
    def get_current_layer(self) -> int:
        """获取当前层号"""
        return self.data_blocks[9044].read_int16(2)
    
    def set_current_layer(self, layer: int):
        """设置当前层号"""
        self.current_layer = layer
        self.data_blocks[9044].write_int16(2, layer)
        self.status_changed.emit(f"当前层号: {layer}")
    
    def get_processing_lock(self) -> bool:
        """获取处理锁状态"""
        return self.data_blocks[9044].read_int16(6) == 1
    
    def set_processing_lock(self, locked: bool):
        """设置处理锁"""
        self.processing_lock = locked
        self.data_blocks[9044].write_int16(6, 1 if locked else 0)
        lock_status = "locked" if locked else "unlocked"
        self.status_changed.emit(f"处理锁: {lock_status}")
    
    def update_status_data(self):
        """更新状态数据"""
        # 更新心跳计数
        heartbeat = self.data_blocks[9044].read_int16(16)
        heartbeat = (heartbeat + 1) % 65536
        self.data_blocks[9044].write_int16(16, heartbeat)
        
        # 检查数据就绪状态
        offset_count = self.data_blocks[9044].read_int16(8)
        if offset_count > 0:
            self.data_blocks[9044].write_int16(14, 1)  # 数据就绪
        else:
            self.data_blocks[9044].write_int16(14, 0)  # 数据未就绪
    
    def simulate_layer_processing(self):
        """模拟一层的处理流程"""
        if self.get_machine_status() != "idle":
            return False
            
        # 开始加工
        self.set_machine_status("running")
        
        # 3秒后自动完成
        def complete_layer():
            time.sleep(3)
            self.set_machine_status("completed")
            
        threading.Thread(target=complete_layer, daemon=True).start()
        return True
    
    def next_layer(self):
        """进入下一层"""
        if self.current_layer < self.total_layers:
            self.set_current_layer(self.current_layer + 1)
            self.set_machine_status("idle")
            # 清除偏移数据
            self.data_blocks[9044].write_int16(8, 0)  # 偏移点数清零
            return True
        return False

class S7PLCSimulatorGUI(QMainWindow):
    """S7 PLC模拟器GUI界面"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S7 PLC模拟器 - 多层加工系统")
        self.resize(1200, 800)
        
        # 创建PLC模拟器
        self.plc = S7PLCSimulator()
        self.plc.data_changed.connect(self.on_data_changed)
        self.plc.status_changed.connect(self.on_status_changed)
        
        self.setup_ui()
        
        # 创建并启动TCP服务器（在UI创建完成后）
        self.tcp_server = S7SimulatorTCPServer(self.plc)
        if self.tcp_server.start():
            self.log_message("TCP服务器启动成功 (端口: 8502)")
        else:
            self.log_message("TCP服务器启动失败")
        
        # 定时刷新界面
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_display)
        self.refresh_timer.start(500)  # 每500ms刷新一次
    
    def setup_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        layout.addWidget(left_panel)
        
        # 右侧数据监控面板
        right_panel = self.create_monitor_panel()
        layout.addWidget(right_panel)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 2)
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # PLC连接状态
        status_group = QGroupBox("PLC状态")
        status_layout = QFormLayout(status_group)
        
        self.connection_label = QLabel("已连接")
        self.connection_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addRow("连接状态:", self.connection_label)
        
        self.machine_status_label = QLabel("idle")
        status_layout.addRow("机床状态:", self.machine_status_label)
        
        self.current_layer_label = QLabel("1")
        status_layout.addRow("当前层号:", self.current_layer_label)
        
        self.processing_lock_label = QLabel("unlocked")
        status_layout.addRow("处理锁:", self.processing_lock_label)
        
        layout.addWidget(status_group)
        
        # 手动控制
        control_group = QGroupBox("手动控制")
        control_layout = QVBoxLayout(control_group)
        
        # 机床状态控制
        status_control_layout = QHBoxLayout()
        self.status_combo = QComboBox()
        self.status_combo.addItems(["idle", "running", "completed", "error"])
        self.status_combo.currentTextChanged.connect(self.on_status_changed_manual)
        
        status_control_layout.addWidget(QLabel("状态:"))
        status_control_layout.addWidget(self.status_combo)
        control_layout.addLayout(status_control_layout)
        
        # 层号控制
        layer_control_layout = QHBoxLayout()
        self.layer_spin = QSpinBox()
        self.layer_spin.setRange(1, 100)
        self.layer_spin.setValue(1)
        self.layer_spin.valueChanged.connect(self.on_layer_changed_manual)
        
        layer_control_layout.addWidget(QLabel("层号:"))
        layer_control_layout.addWidget(self.layer_spin)
        control_layout.addLayout(layer_control_layout)
        
        # 处理锁控制
        self.lock_checkbox = QCheckBox("处理锁定")
        self.lock_checkbox.toggled.connect(self.on_lock_changed_manual)
        control_layout.addWidget(self.lock_checkbox)
        
        layout.addWidget(control_group)
        
        # 自动化控制
        auto_group = QGroupBox("自动化控制")
        auto_layout = QVBoxLayout(auto_group)
        
        self.start_layer_btn = QPushButton("开始当前层")
        self.start_layer_btn.clicked.connect(self.start_current_layer)
        auto_layout.addWidget(self.start_layer_btn)
        
        self.next_layer_btn = QPushButton("下一层")
        self.next_layer_btn.clicked.connect(self.go_next_layer)
        auto_layout.addWidget(self.next_layer_btn)
        
        self.simulate_correction_btn = QPushButton("模拟接收纠偏数据")
        self.simulate_correction_btn.clicked.connect(self.simulate_correction_data)
        auto_layout.addWidget(self.simulate_correction_btn)
        
        layout.addWidget(auto_group)
        
        # 状态日志
        log_group = QGroupBox("状态日志")
        log_layout = QVBoxLayout(log_group)
        
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(150)
        self.status_log.setReadOnly(True)
        log_layout.addWidget(self.status_log)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        return panel
    
    def create_monitor_panel(self):
        """创建数据监控面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 选项卡
        tab_widget = QTabWidget()
        
        # DB9044 控制数据
        self.create_control_data_tab(tab_widget)
        
        # DB9045-9047 偏移数据
        self.create_offset_data_tab(tab_widget)
        
        # 原始数据视图
        self.create_raw_data_tab(tab_widget)
        
        layout.addWidget(tab_widget)
        return panel
    
    def create_control_data_tab(self, parent):
        """创建控制数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 控制数据表格
        self.control_table = QTableWidget()
        self.control_table.setColumnCount(4)
        self.control_table.setHorizontalHeaderLabels(["地址", "名称", "值", "描述"])
        
        # 控制数据项目定义
        control_items = [
            (0, "机床状态", "0", "0=idle, 1=running, 2=completed, 3=error"),
            (2, "当前层号", "1", "当前正在处理的层号"),
            (4, "总层数", "6", "项目总层数"),
            (6, "处理锁", "0", "0=unlock, 1=locked"),
            (8, "偏移点总数", "0", "当前层的偏移点数量"),
            (10, "当前批次", "0", "当前传输的批次号"),
            (12, "总批次数", "0", "总的传输批次数"),
            (14, "数据就绪", "0", "0=not_ready, 1=ready"),
            (16, "心跳计数", "0", "PLC心跳计数器"),
            (18, "错误码", "0", "错误代码"),
            (20, "处理延时", "2000", "算法处理延时(ms)")
        ]
        
        self.control_table.setRowCount(len(control_items))
        for i, (addr, name, value, desc) in enumerate(control_items):
            self.control_table.setItem(i, 0, QTableWidgetItem(str(addr)))
            self.control_table.setItem(i, 1, QTableWidgetItem(name))
            self.control_table.setItem(i, 2, QTableWidgetItem(value))
            self.control_table.setItem(i, 3, QTableWidgetItem(desc))
        
        self.control_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.control_table)
        
        parent.addTab(widget, "DB9044 控制数据")
    
    def create_offset_data_tab(self, parent):
        """创建偏移数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 偏移数据统计
        stats_group = QGroupBox("偏移数据统计")
        stats_layout = QFormLayout(stats_group)
        
        self.offset_count_label = QLabel("0")
        self.batch_info_label = QLabel("0 / 0")
        self.data_ready_label = QLabel("未就绪")
        
        stats_layout.addRow("偏移点总数:", self.offset_count_label)
        stats_layout.addRow("批次进度:", self.batch_info_label)
        stats_layout.addRow("数据状态:", self.data_ready_label)
        
        layout.addWidget(stats_group)
        
        # 偏移数据预览表格
        preview_group = QGroupBox("偏移数据预览 (前20个点)")
        preview_layout = QVBoxLayout(preview_group)
        
        self.offset_table = QTableWidget()
        self.offset_table.setColumnCount(3)
        self.offset_table.setHorizontalHeaderLabels(["序号", "X偏移(μm)", "Y偏移(μm)"])
        self.offset_table.setRowCount(20)
        
        preview_layout.addWidget(self.offset_table)
        layout.addWidget(preview_group)
        
        parent.addTab(widget, "偏移数据")
    
    def create_raw_data_tab(self, parent):
        """创建原始数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 数据块选择
        db_select_layout = QHBoxLayout()
        db_select_layout.addWidget(QLabel("数据块:"))
        
        self.db_combo = QComboBox()
        self.db_combo.addItems(["DB9044", "DB9045", "DB9046", "DB9047"])
        self.db_combo.currentTextChanged.connect(self.refresh_raw_data)
        db_select_layout.addWidget(self.db_combo)
        
        db_select_layout.addStretch()
        layout.addLayout(db_select_layout)
        
        # 原始数据表格
        self.raw_data_table = QTableWidget()
        self.raw_data_table.setColumnCount(17)  # 16字节为一行，加上地址列
        
        headers = ["地址"] + [f"+{i}" for i in range(16)]
        self.raw_data_table.setHorizontalHeaderLabels(headers)
        
        layout.addWidget(self.raw_data_table)
        
        parent.addTab(widget, "原始数据")
    
    def on_status_changed_manual(self, status):
        """手动状态变化"""
        self.plc.set_machine_status(status)
    
    def on_layer_changed_manual(self, layer):
        """手动层号变化"""
        self.plc.set_current_layer(layer)
    
    def on_lock_changed_manual(self, locked):
        """手动锁定变化"""
        self.plc.set_processing_lock(locked)
    
    def start_current_layer(self):
        """开始当前层"""
        if self.plc.simulate_layer_processing():
            self.log_message("开始当前层加工...")
        else:
            self.log_message("无法开始 - 机床不在idle状态")
    
    def go_next_layer(self):
        """下一层"""
        if self.plc.next_layer():
            self.log_message(f"进入第{self.plc.current_layer}层")
        else:
            self.log_message("已到达最后一层")
    
    def simulate_correction_data(self):
        """模拟接收纠偏数据"""
        import random
        
        # 检查是否被锁定
        if self.plc.get_processing_lock():
            self.log_message("数据传输被锁定，无法接收纠偏数据")
            return
        
        # 设置处理锁
        self.plc.set_processing_lock(True)
        self.log_message("开始接收纠偏数据，设置处理锁")
        
        # 模拟分批接收数据
        total_points = 256  # 模拟256个偏移点
        batch_size = 128
        total_batches = (total_points + batch_size - 1) // batch_size
        
        # 清除旧数据
        self.plc.data_blocks[9044].write_int16(8, 0)   # 清零偏移点数
        self.plc.data_blocks[9044].write_int16(14, 0)  # 数据未就绪
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_points)
            batch_points = end_idx - start_idx
            
            # 选择数据块
            db_numbers = [9045, 9046, 9047]
            db_number = db_numbers[batch_num % len(db_numbers)]
            db_offset = self.plc.data_blocks[db_number]
            
            # 写入批次数据
            for i in range(batch_points):
                # 模拟真实偏移数据 (-2000 到 2000微米, 0.1mm精度)
                dx = random.randint(-2000, 2000)
                dy = random.randint(-2000, 2000)
                
                addr = i * 4
                db_offset.write_int16(addr, dx)
                db_offset.write_int16(addr + 2, dy)
            
            # 更新批次信息
            if batch_num == 0:
                self.plc.data_blocks[9044].write_int16(8, total_points)  # 总点数
            
            self.plc.data_blocks[9044].write_int16(10, batch_num + 1)     # 当前批次
            self.plc.data_blocks[9044].write_int16(12, total_batches)     # 总批次
            
            self.log_message(f"接收批次 {batch_num + 1}/{total_batches}: {batch_points}个点 → DB{db_number}")
            
            # 模拟传输延时
            import time
            time.sleep(0.2)
        
        # 数据接收完成
        self.plc.data_blocks[9044].write_int16(14, 1)  # 数据就绪
        self.plc.set_processing_lock(False)  # 释放处理锁
        
        self.log_message(f"纠偏数据接收完成: {total_points}个点，{total_batches}个批次，释放处理锁")
    
    def on_data_changed(self, db_number, address, length):
        """数据变化处理"""
        pass  # 由refresh_display统一处理
    
    def on_status_changed(self, message):
        """状态变化处理"""
        self.log_message(message)
    
    def refresh_display(self):
        """刷新显示"""
        # 更新控制面板状态
        self.machine_status_label.setText(self.plc.get_machine_status())
        self.current_layer_label.setText(str(self.plc.get_current_layer()))
        lock_status = "locked" if self.plc.get_processing_lock() else "unlocked"
        self.processing_lock_label.setText(lock_status)
        
        # 更新控制数据表格
        self.refresh_control_table()
        
        # 更新偏移数据
        self.refresh_offset_data()
        
        # 更新原始数据
        self.refresh_raw_data()
    
    def refresh_control_table(self):
        """刷新控制数据表格"""
        db = self.plc.data_blocks[9044]
        
        # 地址映射到表格行
        addr_to_row = {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8, 18: 9, 20: 10}
        
        for addr, row in addr_to_row.items():
            try:
                value = db.read_int16(addr)
                self.control_table.setItem(row, 2, QTableWidgetItem(str(value)))
            except:
                pass
    
    def refresh_offset_data(self):
        """刷新偏移数据"""
        try:
            # 更新统计信息
            offset_count = self.plc.data_blocks[9044].read_int16(8)
            current_batch = self.plc.data_blocks[9044].read_int16(10)
            total_batches = self.plc.data_blocks[9044].read_int16(12)
            data_ready = self.plc.data_blocks[9044].read_int16(14)
            processing_lock = self.plc.data_blocks[9044].read_int16(6)
            
            self.offset_count_label.setText(str(offset_count))
            self.batch_info_label.setText(f"{current_batch} / {total_batches}")
            
            # 增强状态显示
            if processing_lock:
                ready_text = "🔒 传输中" if current_batch < total_batches else "🔒 处理中"
                self.data_ready_label.setStyleSheet("color: orange; font-weight: bold;")
            elif data_ready:
                ready_text = "✅ 就绪"
                self.data_ready_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                ready_text = "❌ 未就绪"
                self.data_ready_label.setStyleSheet("color: red;")
            
            self.data_ready_label.setText(ready_text)
            
            # 更新偏移数据预览 - 显示来自不同数据块的数据
            display_count = min(20, offset_count) if offset_count > 0 else 0
            
            for i in range(20):
                if i < display_count:
                    # 根据批次分布确定数据来源
                    batch_for_point = (i // 128) + 1
                    db_index = (batch_for_point - 1) % 3
                    db_numbers = [9045, 9046, 9047]
                    db_number = db_numbers[db_index]
                    
                    db_offset = self.plc.data_blocks[db_number]
                    point_in_batch = i % 128
                    addr = point_in_batch * 4
                    
                    try:
                        dx = db_offset.read_int16(addr)
                        dy = db_offset.read_int16(addr + 2)
                        
                        # 显示点信息，包括来源数据块
                        self.offset_table.setItem(i, 0, QTableWidgetItem(f"{i}(DB{db_number})"))
                        self.offset_table.setItem(i, 1, QTableWidgetItem(f"{dx}μm"))
                        self.offset_table.setItem(i, 2, QTableWidgetItem(f"{dy}μm"))
                        
                        # 根据偏移量大小设置颜色
                        magnitude = (dx**2 + dy**2)**0.5
                        if magnitude > 1500:  # 大于1.5mm
                            # 浅红色背景
                            color = QColor(255, 204, 204)  # #FFCCCC
                            self.offset_table.item(i, 1).setBackground(color)
                            self.offset_table.item(i, 2).setBackground(color)
                        elif magnitude > 500:  # 大于0.5mm
                            # 浅黄色背景
                            color = QColor(255, 255, 204)  # #FFFFCC
                            self.offset_table.item(i, 1).setBackground(color)
                            self.offset_table.item(i, 2).setBackground(color)
                        else:
                            # 正常颜色（白色）
                            self.offset_table.item(i, 1).setBackground(QColor(255, 255, 255))
                            self.offset_table.item(i, 2).setBackground(QColor(255, 255, 255))
                            
                    except Exception as e:
                        self.offset_table.setItem(i, 0, QTableWidgetItem(f"{i}(ERR)"))
                        self.offset_table.setItem(i, 1, QTableWidgetItem("--"))
                        self.offset_table.setItem(i, 2, QTableWidgetItem("--"))
                else:
                    self.offset_table.setItem(i, 0, QTableWidgetItem(""))
                    self.offset_table.setItem(i, 1, QTableWidgetItem(""))
                    self.offset_table.setItem(i, 2, QTableWidgetItem(""))
                    
                    # 清除背景颜色
                    if self.offset_table.item(i, 1):
                        self.offset_table.item(i, 1).setBackground(QColor(255, 255, 255))
                    if self.offset_table.item(i, 2):
                        self.offset_table.item(i, 2).setBackground(QColor(255, 255, 255))
                        
        except Exception as e:
            print(f"刷新偏移数据失败: {e}")
    
    def refresh_raw_data(self):
        """刷新原始数据"""
        try:
            db_name = self.db_combo.currentText()
            db_number = int(db_name[2:])  # 从"DB9044"中提取9044
            
            if db_number not in self.plc.data_blocks:
                return
            
            db = self.plc.data_blocks[db_number]
            
            # 计算行数 (每行16字节)
            rows = (db.size + 15) // 16
            self.raw_data_table.setRowCount(rows)
            
            for row in range(rows):
                start_addr = row * 16
                self.raw_data_table.setItem(row, 0, QTableWidgetItem(f"{start_addr:04X}"))
                
                for col in range(16):
                    addr = start_addr + col
                    if addr < db.size:
                        try:
                            data = db.read(addr, 1)
                            value = f"{data[0]:02X}"
                        except:
                            value = "00"
                    else:
                        value = "--"
                    
                    self.raw_data_table.setItem(row, col + 1, QTableWidgetItem(value))
        except:
            pass
    
    def log_message(self, message):
        """记录日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")
        
        # 限制日志行数
        if self.status_log.document().lineCount() > 100:
            cursor = self.status_log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
    
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止TCP服务器
        if hasattr(self, 'tcp_server'):
            self.tcp_server.stop()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    window = S7PLCSimulatorGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()