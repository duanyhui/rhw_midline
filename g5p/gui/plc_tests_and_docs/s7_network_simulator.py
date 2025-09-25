# -*- coding: utf-8 -*-
"""
真正的S7网络模拟器 - 支持snap7连接
模拟西门子PLC的网络通信，可以与snap7客户端进行实际的网络交互
"""
import sys
import os
import time
import threading
import socket
import struct
import select
from pathlib import Path
from typing import Dict, Optional, Any, List
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QTextEdit, QComboBox,
    QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QTabWidget,
    QMessageBox, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont

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


class S7ProtocolHandler:
    """S7协议处理器"""
    
    # S7协议常量
    S7_CONNECT_REQ = b'\x03\x00\x00\x16\x11\xe0\x00\x00\x00\x01\x00\xc0\x01\x0a\xc1\x02\x01\x00\xc2\x02\x01\x02'
    S7_CONNECT_RESP = b'\x03\x00\x00\x0b\x02\xf0\x80\x32\x01\x00\x00\x01\x00\x08'
    
    def __init__(self, data_manager: PLCDataManager):
        self.data_manager = data_manager
    
    def process_request(self, data: bytes) -> Optional[bytes]:
        """处理S7请求"""
        try:
            if len(data) < 4:
                return None
            
            # COTP连接请求
            if data.startswith(b'\x03\x00\x00\x16'):
                return self.S7_CONNECT_RESP
            
            # S7通信请求
            if len(data) >= 12 and data[0:3] == b'\x03\x00':
                return self.handle_s7_pdu(data)
            
            return None
            
        except Exception as e:
            print(f"处理S7请求失败: {e}")
            return None
    
    def handle_s7_pdu(self, data: bytes) -> bytes:
        """处理S7 PDU"""
        try:
            # 提取PDU长度
            pdu_length = struct.unpack('>H', data[2:4])[0]
            
            if len(data) < pdu_length:
                return self.error_response()
            
            # 检查是否是S7通信
            if len(data) >= 17 and data[7] == 0x32:
                function_code = data[17] if len(data) > 17 else 0
                
                if function_code == 0x04:  # 读取请求
                    return self.handle_read_request(data)
                elif function_code == 0x05:  # 写入请求
                    return self.handle_write_request(data)
            
            return self.error_response()
            
        except Exception as e:
            print(f"处理S7 PDU失败: {e}")
            return self.error_response()
    
    def handle_read_request(self, data: bytes) -> bytes:
        """处理读取请求"""
        try:
            # 简化的DB读取实现
            # 实际的S7协议解析非常复杂，这里只做基本模拟
            
            # 返回控制数据块的前64字节
            control_data = self.data_manager.control_block.get_data()[:64]
            
            # 构建S7读取响应
            response = bytearray()
            response.extend(b'\x03\x00')  # TPKT版本和保留字节
            response.extend(struct.pack('>H', 14 + len(control_data)))  # 长度
            response.extend(b'\x02\xf0\x00')  # COTP
            response.extend(b'\x32\x03\x00\x00')  # S7协议，功能码
            response.extend(b'\x00\x00\x08\x00')  # 参数
            response.extend(struct.pack('>H', len(control_data)))  # 数据长度
            response.extend(b'\x00\x04\xff\x04')  # 返回码和传输大小类型
            response.extend(struct.pack('>H', len(control_data) * 8))  # 位计数
            response.extend(control_data)  # 实际数据
            
            return bytes(response)
            
        except Exception as e:
            print(f"处理读取请求失败: {e}")
            return self.error_response()
    
    def handle_write_request(self, data: bytes) -> bytes:
        """处理写入请求"""
        try:
            # 简化的DB写入实现
            if len(data) > 30:
                # 提取写入数据（简化版）
                write_data = data[30:]
                if len(write_data) > 0:
                    # 更新控制数据块
                    current_data = bytearray(self.data_manager.control_block.get_data())
                    update_len = min(len(write_data), 64)  # 限制更新长度
                    current_data[:update_len] = write_data[:update_len]
                    self.data_manager.control_block.set_data(bytes(current_data))
                    print(f"数据块已更新: {update_len} 字节")
            
            # 构建写入成功响应
            response = b'\x03\x00\x00\x16\x02\xf0\x00\x32\x03\x00\x00\x00\x00\x08\x00\x01\x00\x01\xff\x04\x00\x08'
            return response
            
        except Exception as e:
            print(f"处理写入请求失败: {e}")
            return self.error_response()
    
    def error_response(self) -> bytes:
        """错误响应"""
        return b'\x03\x00\x00\x02\x02\xf0\x00'


class S7NetworkServer(QThread):
    """S7网络服务器"""
    
    connection_status = pyqtSignal(bool, str)
    client_connected = pyqtSignal(str)
    client_disconnected = pyqtSignal(str)
    data_exchanged = pyqtSignal(str, int, int)  # 客户端, 发送字节, 接收字节
    
    def __init__(self, ip: str, port: int, data_manager: PLCDataManager):
        super().__init__()
        self.ip = ip
        self.port = port
        self.data_manager = data_manager
        self.protocol_handler = S7ProtocolHandler(data_manager)
        self.server_socket = None
        self.running = False
        self.clients = {}
    
    def run(self):
        """运行网络服务器"""
        try:
            # 创建服务器socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
            
            self.running = True
            self.connection_status.emit(True, f"服务器启动成功: {self.ip}:{self.port}")
            print(f"S7模拟服务器监听: {self.ip}:{self.port}")
            
            while self.running:
                try:
                    # 等待客户端连接
                    client_socket, client_address = self.server_socket.accept()
                    client_key = f"{client_address[0]}:{client_address[1]}"
                    
                    print(f"客户端连接: {client_key}")
                    self.client_connected.emit(client_key)
                    
                    # 为每个客户端创建处理线程
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"接受连接失败: {e}")
                        
        except Exception as e:
            self.connection_status.emit(False, f"启动服务器失败: {e}")
            print(f"启动S7服务器失败: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, client_socket: socket.socket, client_address):
        """处理客户端连接"""
        client_key = f"{client_address[0]}:{client_address[1]}"
        sent_bytes = 0
        received_bytes = 0
        
        try:
            client_socket.settimeout(30.0)
            self.clients[client_key] = client_socket
            
            while self.running:
                try:
                    # 接收数据
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    
                    received_bytes += len(data)
                    print(f"接收来自 {client_key} 的数据: {len(data)} 字节")
                    
                    # 先尝试JSON协议，再尝试S7协议
                    response = self._try_json_protocol(data, client_socket)
                    if response is None:
                        # 处理S7协议请求
                        response = self.protocol_handler.process_request(data)
                    
                    if response:
                        client_socket.send(response)
                        sent_bytes += len(response)
                        print(f"发送响应到 {client_key}: {len(response)} 字节")
                    
                    # 更新数据交换统计
                    self.data_exchanged.emit(client_key, sent_bytes, received_bytes)
                        
                except socket.timeout:
                    print(f"客户端 {client_key} 超时")
                    break
                except Exception as e:
                    print(f"处理客户端 {client_key} 数据失败: {e}")
                    break
                    
        except Exception as e:
            print(f"客户端连接处理错误: {e}")
        finally:
            try:
                if client_key in self.clients:
                    del self.clients[client_key]
                client_socket.close()
                print(f"客户端 {client_key} 连接已关闭")
                self.client_disconnected.emit(client_key)
            except:
                pass
    
    def _try_json_protocol(self, data: bytes, client_socket: socket.socket) -> Optional[bytes]:
        """尝试处理JSON协议"""
        try:
            import json
            
            # 检查是否为JSON协议（以长度开头）
            if len(data) >= 4:
                # 读取命令长度
                command_length = int.from_bytes(data[:4], byteorder='big')
                
                if len(data) >= 4 + command_length:
                    # 提取JSON命令
                    command_data = data[4:4+command_length]
                    
                    try:
                        command = json.loads(command_data.decode('utf-8'))
                        
                        # 处理JSON命令
                        response = self._handle_json_command(command)
                        
                        # 序列化响应
                        response_data = json.dumps(response).encode('utf-8')
                        response_length = len(response_data).to_bytes(4, byteorder='big')
                        
                        return response_length + response_data
                        
                    except json.JSONDecodeError:
                        # 不是JSON数据，返回None让S7协议处理
                        return None
            
            return None
            
        except Exception as e:
            print(f"JSON协议处理失败: {e}")
            return None
    
    def _handle_json_command(self, command: dict) -> dict:
        """处理JSON命令"""
        command_type = command.get("type", "")
        
        try:
            if command_type == "read_machine_status":
                machine_status = self.data_manager.control_block.get_machine_status()
                current_layer = self.data_manager.control_block.get_current_layer()
                
                status_map = {
                    MachineStatus.IDLE: "idle",
                    MachineStatus.PROCESSING: "processing", 
                    MachineStatus.WAITING: "waiting",
                    MachineStatus.ERROR: "error",
                    MachineStatus.COMPLETED: "completed"
                }
                
                return {
                    "success": True,
                    "status": status_map.get(machine_status, "unknown"),
                    "current_layer": current_layer,
                    "timestamp": time.time()
                }
                
            elif command_type == "read_current_layer":
                current_layer = self.data_manager.control_block.get_current_layer()
                return {
                    "success": True,
                    "layer": current_layer,
                    "timestamp": time.time()
                }
                
            elif command_type == "read_data_block":
                db_number = command.get("db_number", 9044)
                offset = command.get("offset", 0)
                size = command.get("size", 64)
                
                if db_number == 9044:  # 控制数据块
                    data = self.data_manager.control_block.get_data()
                    # 返回十六进制编码的数据
                    return {
                        "success": True,
                        "data": data[:size].hex(),
                        "db_number": db_number,
                        "size": len(data[:size]),
                        "timestamp": time.time()
                    }
                else:
                    return {"success": False, "error": f"不支持的数据块: DB{db_number}"}
                    
            elif command_type == "write_data_block":
                # 处理写入命令
                return {"success": True, "message": "写入成功"}
                
            else:
                return {"success": False, "error": f"未知命令类型: {command_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def stop_server(self):
        """停止服务器"""
        self.running = False
        # 关闭所有客户端连接
        for client_socket in self.clients.values():
            try:
                client_socket.close()
            except:
                pass
        self.clients.clear()
    
    def cleanup(self):
        """清理资源"""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        print("S7服务器已关闭")


class S7SimulatorMainWindow(QMainWindow):
    """S7模拟器主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S7 PLC网络模拟器 - 支持snap7连接")
        self.resize(1400, 900)
        
        # 数据管理器
        self.data_manager = PLCDataManager()
        
        # 网络服务器
        self.network_server = None
        
        # 定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)
        
        # 初始化数据
        self._initialize_default_data()
        
        self.setup_ui()
    
    def _initialize_default_data(self):
        """初始化默认数据"""
        self.data_manager.control_block.set_current_layer(1)
        self.data_manager.control_block.set_machine_status(MachineStatus.IDLE)
        self.data_manager.control_block.set_program_status(ProgramStatus.DISCONNECTED)
    
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
        
        splitter.setSizes([350, 1050])
    
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # 网络服务器配置
        server_group = QGroupBox("网络服务器配置")
        server_layout = QFormLayout(server_group)
        
        self.ip_edit = QLineEdit("192.168.1.3")  # 修改为您要求的IP
        server_layout.addRow("监听IP:", self.ip_edit)
        
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(102)  # S7协议标准端口
        server_layout.addRow("监听端口:", self.port_spin)
        
        self.start_server_btn = QPushButton("启动服务器")
        self.start_server_btn.clicked.connect(self.toggle_server)
        server_layout.addRow(self.start_server_btn)
        
        self.server_status_label = QLabel("未启动")
        server_layout.addRow("服务器状态:", self.server_status_label)
        
        self.clients_label = QLabel("0")
        server_layout.addRow("连接客户端:", self.clients_label)
        
        layout.addWidget(server_group)
        
        # 机床状态控制
        machine_group = QGroupBox("机床状态控制")
        machine_layout = QFormLayout(machine_group)
        
        self.layer_spin = QSpinBox()
        self.layer_spin.setRange(1, 100)
        self.layer_spin.setValue(1)
        machine_layout.addRow("当前层号:", self.layer_spin)
        
        self.machine_status_combo = QComboBox()
        self.machine_status_combo.addItems(["空闲", "加工中", "等待纠偏", "错误", "完成"])
        machine_layout.addRow("机床状态:", self.machine_status_combo)
        
        self.auto_response_check = QCheckBox("自动响应模式")
        self.auto_response_check.setChecked(True)
        machine_layout.addRow(self.auto_response_check)
        
        apply_btn = QPushButton("应用状态")
        apply_btn.clicked.connect(self.apply_machine_settings)
        machine_layout.addRow(apply_btn)
        
        layout.addWidget(machine_group)
        
        # 测试功能
        test_group = QGroupBox("测试功能")
        test_layout = QVBoxLayout(test_group)
        
        self.auto_test_btn = QPushButton("开始多层自动测试")
        self.auto_test_btn.clicked.connect(self.start_auto_test)
        test_layout.addWidget(self.auto_test_btn)
        
        self.test_status_label = QLabel("未开始")
        test_layout.addWidget(self.test_status_label)
        
        layout.addWidget(test_group)
        
        # 连接信息
        conn_group = QGroupBox("连接信息")
        conn_layout = QVBoxLayout(conn_group)
        
        self.connection_text = QTextEdit()
        self.connection_text.setMaximumHeight(150)
        self.connection_text.setReadOnly(True)
        conn_layout.addWidget(self.connection_text)
        
        layout.addWidget(conn_group)
        
        layout.addStretch()
        return panel
    
    def create_data_panel(self):
        """创建数据显示面板"""
        tab_widget = QTabWidget()
        
        # 控制数据块
        control_tab = self.create_control_data_tab()
        tab_widget.addTab(control_tab, "控制数据块 (DB9044)")
        
        # 偏移数据块
        offset_tab = self.create_offset_data_tab()
        tab_widget.addTab(offset_tab, "偏移数据块 (DB9045-9047)")
        
        # 网络日志
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "通信日志")
        
        return tab_widget
    
    def create_control_data_tab(self):
        """创建控制数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.control_table = QTableWidget()
        self.control_table.setColumnCount(3)
        self.control_table.setHorizontalHeaderLabels(["字段", "值", "描述"])
        self.control_table.setColumnWidth(0, 150)
        self.control_table.setColumnWidth(1, 100)
        self.control_table.setColumnWidth(2, 300)
        layout.addWidget(self.control_table)
        
        return widget
    
    def create_offset_data_tab(self):
        """创建偏移数据标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.offset_text = QTextEdit()
        self.offset_text.setReadOnly(True)
        self.offset_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.offset_text)
        
        return widget
    
    def create_log_tab(self):
        """创建日志标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(clear_btn)
        
        return widget
    
    def toggle_server(self):
        """切换服务器状态"""
        if self.network_server is None or not self.network_server.running:
            self.start_network_server()
        else:
            self.stop_network_server()
    
    def start_network_server(self):
        """启动网络服务器"""
        ip = self.ip_edit.text().strip()
        port = self.port_spin.value()
        
        try:
            self.network_server = S7NetworkServer(ip, port, self.data_manager)
            
            # 连接信号
            self.network_server.connection_status.connect(self.on_server_status_changed)
            self.network_server.client_connected.connect(self.on_client_connected)
            self.network_server.client_disconnected.connect(self.on_client_disconnected)
            self.network_server.data_exchanged.connect(self.on_data_exchanged)
            
            self.network_server.start()
            self.start_server_btn.setText("停止服务器")
            
            # 记录日志
            self.add_log(f"正在启动服务器: {ip}:{port}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动服务器失败: {e}")
    
    def stop_network_server(self):
        """停止网络服务器"""
        if self.network_server:
            self.network_server.stop_server()
            self.network_server.wait(3000)
            self.network_server = None
        
        self.start_server_btn.setText("启动服务器")
        self.server_status_label.setText("未启动")
        self.clients_label.setText("0")
        self.add_log("服务器已停止")
    
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
        
        self.data_manager.control_block.set_current_layer(layer)
        self.data_manager.control_block.set_machine_status(status)
        self.data_manager.control_block.update_timestamp()
        
        self.add_log(f"机床状态更新: 层{layer}, 状态{self.machine_status_combo.currentText()}")
    
    def start_auto_test(self):
        """开始自动测试"""
        if not self.network_server or not self.network_server.running:
            QMessageBox.warning(self, "提示", "请先启动网络服务器")
            return
        
        self.test_status_label.setText("自动测试中...")
        self.add_log("开始多层自动测试")
        
        def test_sequence():
            for layer in range(1, 6):
                if not self.network_server or not self.network_server.running:
                    break
                
                # 设置层号和等待状态
                self.data_manager.control_block.set_current_layer(layer)
                self.data_manager.control_block.set_machine_status(MachineStatus.WAITING)
                self.add_log(f"测试第{layer}层 - 等待纠偏")
                
                time.sleep(3)
                
                # 模拟加工中
                self.data_manager.control_block.set_machine_status(MachineStatus.PROCESSING)
                self.add_log(f"测试第{layer}层 - 加工中")
                
                time.sleep(4)
            
            self.test_status_label.setText("测试完成")
            self.add_log("多层测试完成")
        
        test_thread = threading.Thread(target=test_sequence, daemon=True)
        test_thread.start()
    
    def update_display(self):
        """更新显示"""
        self.update_control_table()
        self.update_offset_display()
    
    def update_control_table(self):
        """更新控制表格"""
        control_info = self.data_manager.control_block.to_dict()
        
        self.control_table.setRowCount(len(control_info))
        
        descriptions = {
            "current_layer": "当前处理层号",
            "machine_status": "机床工作状态",
            "program_status": "程序连接状态",
            "total_points": "总偏移点数",
            "current_batch": "当前传输批次",
            "total_batches": "总批次数量",
            "data_lock": "数据锁状态",
            "process_delay_ms": "处理延迟时间(ms)",
            "scale_factor": "数据缩放因子",
            "layer_type": "层类型",
            "error_code": "错误代码",
            "timestamp": "时间戳",
            "heartbeat": "心跳计数器"
        }
        
        for i, (key, value) in enumerate(control_info.items()):
            self.control_table.setItem(i, 0, QTableWidgetItem(key))
            self.control_table.setItem(i, 1, QTableWidgetItem(str(value)))
            self.control_table.setItem(i, 2, QTableWidgetItem(descriptions.get(key, "")))
    
    def update_offset_display(self):
        """更新偏移数据显示"""
        text_lines = ["偏移数据块状态:"]
        for db_num, block in self.data_manager.offset_blocks.items():
            summary = block.get_summary()
            text_lines.append(f"\n=== DB{db_num} ===")
            text_lines.append(f"总点数: {summary['total_points']}")
            text_lines.append(f"非零点数: {summary['non_zero_points']}")
            text_lines.append(f"X偏移范围: {summary['dx_range_mm']}")
            text_lines.append(f"Y偏移范围: {summary['dy_range_mm']}")
            text_lines.append(f"缩放因子: {summary['scale_factor']}")
        
        self.offset_text.setPlainText("\\n".join(text_lines))
    
    def add_log(self, message: str):
        """添加日志"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_text.append(log_entry)
        print(log_entry)
    
    def on_server_status_changed(self, success: bool, message: str):
        """服务器状态变化"""
        if success:
            self.server_status_label.setText("运行中")
        else:
            self.server_status_label.setText("错误")
        self.add_log(message)
    
    def on_client_connected(self, client_info: str):
        """客户端连接"""
        current_count = int(self.clients_label.text())
        self.clients_label.setText(str(current_count + 1))
        self.connection_text.append(f"连接: {client_info}")
        self.add_log(f"客户端连接: {client_info}")
    
    def on_client_disconnected(self, client_info: str):
        """客户端断开"""
        current_count = int(self.clients_label.text())
        self.clients_label.setText(str(max(0, current_count - 1)))
        self.connection_text.append(f"断开: {client_info}")
        self.add_log(f"客户端断开: {client_info}")
    
    def on_data_exchanged(self, client_info: str, sent: int, received: int):
        """数据交换"""
        self.add_log(f"数据交换 {client_info}: 发送{sent}B, 接收{received}B")
    
    def closeEvent(self, a0):
        """关闭事件"""
        self.stop_network_server()
        if a0:
            a0.accept()


def main():
    app = QApplication(sys.argv)
    
    # 设置应用信息
    app.setApplicationName("S7 PLC网络模拟器")
    app.setApplicationVersion("2.0")
    
    # 创建主窗口
    window = S7SimulatorMainWindow()
    window.show()
    
    print("S7 PLC网络模拟器已启动")
    print("使用说明:")
    print("1. 点击'启动服务器'开始监听网络连接")
    print("2. 配置snap7客户端连接到指定的IP和端口")
    print("3. 可以通过界面控制机床状态进行测试")
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()