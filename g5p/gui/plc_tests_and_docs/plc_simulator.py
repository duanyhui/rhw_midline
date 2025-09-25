# -*- coding: utf-8 -*-
"""
S7协议PLC模拟器
模拟西门子PLC的数据块读写操作，用于测试多层加工纠偏系统
"""
import sys
import os
import time
import threading
import socket
import struct
from pathlib import Path
from typing import Dict, Optional, Any
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QTextEdit, QComboBox,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QTabWidget,
    QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QCloseEvent

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from plc_data_structures import (
        PLCDataBlocks, PLCDataManager, ControlBlockManager, 
        MachineStatus, ProgramStatus, DataLockStatus, OffsetPoint
    )
    from offset_data_handler import OffsetDataLoader
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的目录下运行此程序")
    sys.exit(1)


class S7Server:
    """S7协议服务器模拟"""
    
    def __init__(self, ip: str = "192.168.1.3", port: int = 102):
        self.ip = ip
        self.port = port
        self.data_manager = PLCDataManager()
        self.server_active = False
        self.client_connections = {}
        
        # 初始化默认状态
        self._initialize_default_data()
    
    def _initialize_default_data(self):
        """初始化默认数据"""
        self.data_manager.control_block.set_current_layer(1)
        self.data_manager.control_block.set_machine_status(MachineStatus.IDLE)
        self.data_manager.control_block.set_program_status(ProgramStatus.DISCONNECTED)
    
    def start_server(self):
        """启动模拟服务器"""
        self.server_active = True
        print(f"S7模拟服务器已启动: {self.ip}:{self.port}")
    
    def stop_server(self):
        """停止模拟服务器"""
        self.server_active = False
        print("S7模拟服务器已停止")
    
    def read_db(self, db_number: int, offset: int, size: int) -> bytes:
        """读取数据块"""
        if db_number == PLCDataBlocks.DB_CONTROL:
            data = self.data_manager.control_block.get_data()
        elif db_number in self.data_manager.offset_blocks:
            data = self.data_manager.offset_blocks[db_number].get_data()
        else:
            return b'\x00' * size
        
        return data[offset:offset + size]
    
    def write_db(self, db_number: int, offset: int, data: bytes):
        """写入数据块"""
        if db_number == PLCDataBlocks.DB_CONTROL:
            current_data = bytearray(self.data_manager.control_block.get_data())
            current_data[offset:offset + len(data)] = data
            self.data_manager.control_block.set_data(bytes(current_data))
        elif db_number in self.data_manager.offset_blocks:
            current_data = bytearray(self.data_manager.offset_blocks[db_number].get_data())
            current_data[offset:offset + len(data)] = data
            self.data_manager.offset_blocks[db_number].set_data(bytes(current_data))


class PLCSimulatorMainWindow(QMainWindow):
    """PLC模拟器主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S7 PLC模拟器 - 多层加工纠偏系统")
        self.resize(1200, 800)
        
        # S7服务器
        self.s7_server = S7Server()
        
        # 定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # 1秒更新一次
        
        self.setup_ui()
        self.start_simulator()
    
    def setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        splitter = QSplitter()
        layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧数据显示
        data_panel = self.create_data_panel()
        splitter.addWidget(data_panel)
        
        splitter.setSizes([300, 900])
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        
        # 服务器配置
        server_group = QGroupBox("服务器配置")
        server_layout = QFormLayout(server_group)
        
        self.ip_edit = QLineEdit(self.s7_server.ip)
        server_layout.addRow("IP地址:", self.ip_edit)
        
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(self.s7_server.port)
        server_layout.addRow("端口:", self.port_spin)
        
        self.server_status_label = QLabel("已启动")
        server_layout.addRow("状态:", self.server_status_label)
        
        layout.addWidget(server_group)
        
        # 机床控制
        machine_group = QGroupBox("机床控制")
        machine_layout = QFormLayout(machine_group)
        
        self.layer_spin = QSpinBox()
        self.layer_spin.setRange(1, 100)
        self.layer_spin.setValue(1)
        machine_layout.addRow("当前层号:", self.layer_spin)
        
        self.machine_status_combo = QComboBox()
        self.machine_status_combo.addItems(["空闲", "加工中", "等待纠偏", "错误", "完成"])
        machine_layout.addRow("机床状态:", self.machine_status_combo)
        
        apply_btn = QPushButton("应用设置")
        apply_btn.clicked.connect(self.apply_machine_settings)
        machine_layout.addRow(apply_btn)
        
        layout.addWidget(machine_group)
        
        # 自动测试
        test_group = QGroupBox("自动测试")
        test_layout = QVBoxLayout(test_group)
        
        self.start_test_btn = QPushButton("开始多层测试")
        self.start_test_btn.clicked.connect(self.start_multilayer_test)
        test_layout.addWidget(self.start_test_btn)
        
        self.test_status_label = QLabel("未开始")
        test_layout.addWidget(self.test_status_label)
        
        layout.addWidget(test_group)
        
        layout.addStretch()
        return panel
    
    def create_data_panel(self):
        """创建数据显示面板"""
        tab_widget = QTabWidget()
        
        # 控制数据块
        control_tab = self.create_control_data_tab()
        tab_widget.addTab(control_tab, "控制数据块")
        
        # 偏移数据块
        offset_tab = self.create_offset_data_tab()
        tab_widget.addTab(offset_tab, "偏移数据块")
        
        return tab_widget
    
    def create_control_data_tab(self):
        """创建控制数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.control_table = QTableWidget()
        self.control_table.setColumnCount(3)
        self.control_table.setHorizontalHeaderLabels(["字段", "值", "描述"])
        layout.addWidget(self.control_table)
        
        return widget
    
    def create_offset_data_tab(self):
        """创建偏移数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.offset_text = QTextEdit()
        self.offset_text.setReadOnly(True)
        self.offset_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.offset_text)
        
        return widget
    
    def apply_machine_settings(self):
        """应用机床设置"""
        layer = self.layer_spin.value()
        status_map = {
            "空闲": MachineStatus.IDLE,
            "加工中": MachineStatus.PROCESSING,
            "等待纠偏": MachineStatus.WAITING,
            "错误": MachineStatus.ERROR,
            "完成": MachineStatus.COMPLETED
        }
        status = status_map[self.machine_status_combo.currentText()]
        
        self.s7_server.data_manager.control_block.set_current_layer(layer)
        self.s7_server.data_manager.control_block.set_machine_status(status)
        self.s7_server.data_manager.control_block.update_timestamp()
        
        print(f"设置: 层{layer}, 状态{self.machine_status_combo.currentText()}")
    
    def start_multilayer_test(self):
        """开始多层测试"""
        self.test_status_label.setText("测试中...")
        print("开始多层自动测试...")
        
        # 简单的测试流程
        def test_sequence():
            for layer in range(1, 6):  # 测试5层
                print(f"测试第{layer}层...")
                
                # 设置层号和等待状态
                self.s7_server.data_manager.control_block.set_current_layer(layer)
                self.s7_server.data_manager.control_block.set_machine_status(MachineStatus.WAITING)
                
                time.sleep(2)  # 等待算法程序响应
                
                # 模拟加工中
                self.s7_server.data_manager.control_block.set_machine_status(MachineStatus.PROCESSING)
                time.sleep(3)
            
            self.test_status_label.setText("测试完成")
            print("多层测试完成")
        
        test_thread = threading.Thread(target=test_sequence, daemon=True)
        test_thread.start()
    
    def update_display(self):
        """更新显示"""
        self.update_control_table()
        self.update_offset_display()
    
    def update_control_table(self):
        """更新控制表格"""
        control_info = self.s7_server.data_manager.control_block.to_dict()
        
        self.control_table.setRowCount(len(control_info))
        
        for i, (key, value) in enumerate(control_info.items()):
            self.control_table.setItem(i, 0, QTableWidgetItem(key))
            self.control_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.control_table.setItem(i, 2, QTableWidgetItem(self.get_field_description(key)))
    
    def update_offset_display(self):
        """更新偏移数据显示"""
        text_lines = []
        for db_num, block in self.s7_server.data_manager.offset_blocks.items():
            summary = block.get_summary()
            text_lines.append(f"=== DB{db_num} ===")
            text_lines.append(f"非零点数: {summary['non_zero_points']}")
            text_lines.append(f"X范围: {summary['dx_range_mm']}")
            text_lines.append(f"Y范围: {summary['dy_range_mm']}")
            text_lines.append("")
        
        self.offset_text.setPlainText("\n".join(text_lines))
    
    def get_field_description(self, field: str) -> str:
        """获取字段描述"""
        descriptions = {
            "current_layer": "当前处理层号",
            "machine_status": "机床工作状态",
            "program_status": "程序连接状态",
            "total_points": "总偏移点数",
            "current_batch": "当前传输批次",
            "total_batches": "总批次数量",
            "data_lock": "数据锁状态",
            "layer_type": "层类型(0=标定,1=纠偏)",
            "heartbeat": "心跳计数器"
        }
        return descriptions.get(field, "")
    
    def start_simulator(self):
        """启动模拟器"""
        self.s7_server.start_server()
    
    def closeEvent(self, a0):
        """关闭事件"""
        self.s7_server.stop_server()
        if a0:
            a0.accept()


def main():
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("S7 PLC模拟器")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = PLCSimulatorMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()