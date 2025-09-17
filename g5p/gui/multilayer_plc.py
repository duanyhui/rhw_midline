# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - PLC通信模块
"""
import json
import time
import threading
from typing import Optional
from PyQt5.QtCore import QThread, QObject, pyqtSignal

from multilayer_data import ProjectConfig

# ==================== PLC通信基类 ====================

class PLCCommunicator(QObject):
    """PLC通信基类"""
    layer_changed = pyqtSignal(int)  # 层号变化信号
    connection_status = pyqtSignal(bool, str)  # 连接状态信号
    
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config
        self.connected = False
        self.running = False
        
    def connect(self):
        """连接PLC"""
        raise NotImplementedError
        
    def disconnect(self):
        """断开连接"""
        self.running = False
        self.connected = False
        
    def read_current_layer(self) -> int:
        """读取当前层号"""
        raise NotImplementedError
        
    def read_start_signal(self) -> bool:
        """读取开始信号"""
        raise NotImplementedError
        
    def write_completion_signal(self, layer_id: int, success: bool):
        """写入完成信号"""
        pass  # 可选实现

# ==================== TCP方式PLC通信 ====================

class TCPPLCCommunicator(PLCCommunicator):
    """TCP方式PLC通信"""
    
    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        self.socket = None
        
    def connect(self):
        try:
            import socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.config.plc_ip, self.config.plc_port))
            self.connected = True
            self.connection_status.emit(True, "TCP连接成功")
            return True
        except Exception as e:
            self.connection_status.emit(False, f"TCP连接失败: {e}")
            return False
            
    def disconnect(self):
        super().disconnect()
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
    def read_current_layer(self) -> int:
        if not self.connected or not self.socket:
            return -1
        try:
            # 发送读取层号请求
            request = {"type": "read_layer", "timestamp": time.time()}
            request_data = json.dumps(request).encode('utf-8')
            self.socket.send(len(request_data).to_bytes(4, byteorder='big'))
            self.socket.send(request_data)
            
            # 接收响应
            length_bytes = self.socket.recv(4)
            if len(length_bytes) < 4:
                return -1
            length = int.from_bytes(length_bytes, byteorder='big')
            
            response_data = self.socket.recv(length)
            response = json.loads(response_data.decode('utf-8'))
            return response.get("layer", -1)
        except Exception as e:
            print(f"TCP读取层号失败: {e}")
            return -1
            
    def read_start_signal(self) -> bool:
        if not self.connected or not self.socket:
            return False
        try:
            request = {"type": "read_start", "timestamp": time.time()}
            request_data = json.dumps(request).encode('utf-8')
            self.socket.send(len(request_data).to_bytes(4, byteorder='big'))
            self.socket.send(request_data)
            
            length_bytes = self.socket.recv(4)
            if len(length_bytes) < 4:
                return False
            length = int.from_bytes(length_bytes, byteorder='big')
            
            response_data = self.socket.recv(length)
            response = json.loads(response_data.decode('utf-8'))
            return response.get("start", False)
        except Exception as e:
            print(f"TCP读取开始信号失败: {e}")
            return False
            
    def write_completion_signal(self, layer_id: int, success: bool):
        """写入完成信号"""
        if not self.connected or not self.socket:
            return
        try:
            request = {
                "type": "write_completion",
                "layer": layer_id,
                "success": success,
                "timestamp": time.time()
            }
            request_data = json.dumps(request).encode('utf-8')
            self.socket.send(len(request_data).to_bytes(4, byteorder='big'))
            self.socket.send(request_data)
        except Exception as e:
            print(f"TCP写入完成信号失败: {e}")

# ==================== S7协议PLC通信 ====================

class S7PLCCommunicator(PLCCommunicator):
    """S7协议PLC通信"""
    
    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        self.client = None
        self._snap7_available = False
        
        # 检查snap7是否可用
        try:
            import snap7
            self._snap7_available = True
        except ImportError:
            self._snap7_available = False
        
    def connect(self):
        if not self._snap7_available:
            self.connection_status.emit(False, "S7连接失败: 未安装 python-snap7 库")
            return False
            
        try:
            import snap7
            self.client = snap7.client.Client()
            self.client.connect(self.config.plc_ip, 0, 1)
            self.connected = True
            self.connection_status.emit(True, "S7连接成功")
            return True
        except Exception as e:
            self.connection_status.emit(False, f"S7连接失败: {e}")
            return False
            
    def disconnect(self):
        super().disconnect()
        if self.client:
            try:
                self.client.disconnect()
            except:
                pass
            self.client = None
            
    def _parse_s7_address(self, address: str) -> tuple:
        """解析S7地址格式
        例如: "DB1.DBD0" -> (1, "DBD", 0)
        """
        try:
            parts = address.split('.')
            db_part = parts[0]  # DB1
            data_part = parts[1]  # DBD0
            
            db_num = int(db_part[2:])  # 1
            
            if data_part.startswith('DBD'):
                data_type = 'DBD'
                offset = int(data_part[3:])
                size = 4
            elif data_part.startswith('DBW'):
                data_type = 'DBW'
                offset = int(data_part[3:])
                size = 2
            elif data_part.startswith('DBB'):
                data_type = 'DBB'
                offset = int(data_part[3:])
                size = 1
            elif data_part.startswith('DBX'):
                # 位地址，例如 DBX4.0
                data_type = 'DBX'
                bit_parts = data_part[3:].split('.')
                offset = int(bit_parts[0])
                bit_offset = int(bit_parts[1])
                size = 1
                return db_num, data_type, offset, bit_offset, size
            else:
                raise ValueError(f"不支持的数据类型: {data_part}")
                
            return db_num, data_type, offset, size
        except Exception as e:
            raise ValueError(f"地址解析失败 '{address}': {e}")
            
    def read_current_layer(self) -> int:
        if not self.connected or not self.client:
            return -1
        try:
            db_num, data_type, offset, size = self._parse_s7_address(
                self.config.current_layer_address
            )
            
            data = self.client.db_read(db_num, offset, size)
            
            if data_type == 'DBD':
                # 32位整数，大端序
                layer = int.from_bytes(data, byteorder='big', signed=False)
            elif data_type == 'DBW':
                # 16位整数，大端序
                layer = int.from_bytes(data, byteorder='big', signed=False)
            elif data_type == 'DBB':
                # 8位整数
                layer = data[0]
            else:
                layer = -1
                
            return layer
        except Exception as e:
            print(f"S7读取层号失败: {e}")
            return -1
            
    def read_start_signal(self) -> bool:
        if not self.connected or not self.client:
            return False
        try:
            address_parts = self._parse_s7_address(self.config.start_signal_address)
            
            if len(address_parts) == 5:  # 位地址
                db_num, data_type, offset, bit_offset, size = address_parts
                data = self.client.db_read(db_num, offset, size)
                byte_val = data[0]
                bit_val = (byte_val >> bit_offset) & 1
                return bool(bit_val)
            else:  # 字节/字/双字地址
                db_num, data_type, offset, size = address_parts
                data = self.client.db_read(db_num, offset, size)
                return data[0] != 0  # 非零即真
                
        except Exception as e:
            print(f"S7读取开始信号失败: {e}")
            return False
            
    def write_completion_signal(self, layer_id: int, success: bool):
        """写入完成信号到PLC"""
        if not self.connected or not self.client:
            return
        try:
            # 这里可以定义完成信号的写入地址
            # 例如 DB1.DBX5.0 表示完成标志
            # DB1.DBD8 表示完成的层号
            # 具体地址需要根据PLC程序定义
            pass
        except Exception as e:
            print(f"S7写入完成信号失败: {e}")

# ==================== 模拟PLC通信器（用于测试） ====================

class MockPLCCommunicator(PLCCommunicator):
    """模拟PLC通信器，用于无PLC环境下的测试"""
    
    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        self.mock_layer = 1
        self.mock_start = False
        self.auto_increment = False
        self.last_layer_time = time.time()
        
    def connect(self):
        self.connected = True
        self.connection_status.emit(True, "模拟PLC连接成功")
        return True
        
    def disconnect(self):
        super().disconnect()
        
    def read_current_layer(self) -> int:
        if not self.connected:
            return -1
            
        # 模拟自动递增层号（每30秒递增一层）
        if self.auto_increment:
            current_time = time.time()
            if current_time - self.last_layer_time > 30:  # 30秒
                self.mock_layer += 1
                self.last_layer_time = current_time
                
        return self.mock_layer
        
    def read_start_signal(self) -> bool:
        return self.mock_start
        
    def set_mock_layer(self, layer: int):
        """设置模拟层号"""
        old_layer = self.mock_layer
        self.mock_layer = layer
        if layer != old_layer:
            self.layer_changed.emit(layer)
            
    def set_mock_start(self, start: bool):
        """设置模拟开始信号"""
        self.mock_start = start
        
    def enable_auto_increment(self, enable: bool):
        """启用自动递增"""
        self.auto_increment = enable
        if enable:
            self.last_layer_time = time.time()

# ==================== PLC监控线程 ====================

class PLCMonitorThread(QThread):
    """PLC监控线程"""
    
    def __init__(self, communicator: PLCCommunicator):
        super().__init__()
        self.communicator = communicator
        self.last_layer = -1
        self.last_start = False
        self.running = False
        self.poll_interval = 0.5  # 500ms轮询间隔
        
    def run(self):
        self.running = True
        error_count = 0
        max_errors = 5
        
        while self.running:
            try:
                # 读取当前层号
                current_layer = self.communicator.read_current_layer()
                if current_layer != self.last_layer and current_layer > 0:
                    self.last_layer = current_layer
                    self.communicator.layer_changed.emit(current_layer)
                    
                # 读取开始信号
                start_signal = self.communicator.read_start_signal()
                if start_signal != self.last_start:
                    self.last_start = start_signal
                    # 可以添加开始信号变化的处理
                    
                # 重置错误计数
                error_count = 0
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                error_count += 1
                print(f"PLC监控错误 ({error_count}/{max_errors}): {e}")
                
                if error_count >= max_errors:
                    self.communicator.connection_status.emit(
                        False, f"PLC通信错误过多，已断开连接"
                    )
                    break
                    
                time.sleep(1.0)  # 错误时等待更长时间
                
    def stop(self):
        """停止监控"""
        self.running = False
        self.wait(3000)  # 等待最多3秒