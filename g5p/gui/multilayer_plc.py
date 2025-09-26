# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - PLC通信模块
"""
import json
import socket
import time
import threading
from typing import Optional
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer

from multilayer_data import ProjectConfig

# ==================== PLC数据交换协议定义 ====================

class PLCDataProtocol:
    """PLC数据交换协议定义"""
    
    # 数据交换命令类型
    CMD_READ_LAYER = "read_current_layer"      # 读取当前层号
    CMD_READ_STATUS = "read_machine_status"    # 读取机床状态
    CMD_WRITE_COMPLETE = "write_layer_complete" # 写入层完成信号
    CMD_SEND_CORRECTION = "send_correction_data" # 发送纠偏数据
    CMD_ALERT_ERROR = "alert_deviation_error"   # 偏差过大警告
    
    # 机床状态枚举
    STATUS_IDLE = "idle"           # 空闲
    STATUS_PROCESSING = "processing" # 加工中
    STATUS_WAITING = "waiting"     # 等待纠偏数据
    STATUS_ERROR = "error"         # 错误状态
    
    # 纠偏数据状态
    CORRECTION_VALID = "valid"     # 有效纠偏数据
    CORRECTION_SKIP = "skip"       # 跳过纠偏（偏差过大）
    CORRECTION_ERROR = "error"     # 纠偏数据错误

# ==================== PLC通信基类 ====================

class PLCCommunicator(QObject):
    """PLC通信基类"""
    layer_changed = pyqtSignal(int)  # 层号变化信号
    connection_status = pyqtSignal(bool, str)  # 连接状态信号
    machine_status_changed = pyqtSignal(str)  # 机床状态变化信号
    correction_request = pyqtSignal(int)  # 纠偏请求信号
    
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config
        self.connected = False
        self.running = False
        self.current_layer = 0
        self.machine_status = PLCDataProtocol.STATUS_IDLE
        
        # 轮询定时器
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_plc_status)
        self.poll_interval = 1000  # 1秒轮询一次
        
        # 状态缓存
        self.last_machine_status = ""
        self.last_layer = -1
        
    def start_polling(self):
        """开始轮询PLC状态"""
        if self.connected and not self.poll_timer.isActive():
            self.poll_timer.start(self.poll_interval)
            print("PLC状态轮询已启动")
            
    def stop_polling(self):
        """停止轮询PLC状态"""
        if self.poll_timer.isActive():
            self.poll_timer.stop()
            print("PLC状态轮询已停止")
            
    def _poll_plc_status(self):
        """轮询PLC状态，检测变化并发出信号"""
        if not self.connected:
            return
            
        try:
            # 读取机床状态
            current_status = self.read_machine_status()
            if current_status != self.last_machine_status:
                self.last_machine_status = current_status
                self.machine_status_changed.emit(current_status)
                print(f"PLC状态变化: {current_status}")
                
                # 如果机床状态为waiting，表示完成了加工，等待纠偏数据
                if current_status == PLCDataProtocol.STATUS_WAITING:
                    current_layer = self.read_current_layer()
                    if current_layer > 0:
                        print(f"PLC请求第{current_layer}层纠偏数据")
                        self.correction_request.emit(current_layer)
                        
        except Exception as e:
            print(f"PLC状态轮询错误: {e}")
        
    def connect(self):
        """连接PLC"""
        raise NotImplementedError
        
    def disconnect_plc(self):
        """断开PLC连接"""
        self.running = False
        self.connected = False
        
    def read_current_layer(self) -> int:
        """读取当前层号"""
        raise NotImplementedError
        
    def read_machine_status(self) -> str:
        """读取机床状态"""
        raise NotImplementedError
        
    def read_start_signal(self) -> bool:
        """读取开始信号（兼容旧接口）"""
        # 默认实现，基于机床状态判断
        status = self.read_machine_status()
        return status == PLCDataProtocol.STATUS_WAITING
        
    def write_layer_completion(self, layer_id: int, success: bool, processing_time: float = 0.0):
        """写入层完成信号"""
        raise NotImplementedError
        
    def send_correction_data(self, layer_id: int, correction_data: dict) -> bool:
        """发送纠偏数据到PLC"""
        raise NotImplementedError
        
    def send_deviation_alert(self, layer_id: int, alert_message: str, deviation_value: float):
        """发送偏差过大警告"""
        raise NotImplementedError

# ==================== TCP方式PLC通信（支持JSON数据交换） ====================

class TCPPLCCommunicator(PLCCommunicator):
    """TCP方式PLC通信，支持JSON数据交换"""
    
    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        self.socket = None
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 配置Socket选项
            self.socket.settimeout(10.0)  # 增加超时时间到10秒
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # 启用保活
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用Nagle算法
            
            # 连接到PLC
            self.socket.connect((self.config.plc_ip, self.config.plc_port))
            self.connected = True
            self.connection_status.emit(True, "TCP连接成功")
            print(f"[TCP] 成功连接到 {self.config.plc_ip}:{self.config.plc_port}")
            return True
        except socket.timeout:
            self.connection_status.emit(False, "TCP连接超时")
            return False
        except socket.error as e:
            self.connection_status.emit(False, f"TCP连接失败: {e}")
            return False
        except Exception as e:
            self.connection_status.emit(False, f"TCP连接异常: {e}")
            return False
            
    def disconnect_plc(self):
        super().disconnect_plc()
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
            
    def _send_command(self, command: dict) -> dict:
        """发送命令并接收响应，带自动重连"""
        if not self.connected or not self.socket:
            return {"success": False, "error": "连接未建立"}
        
        # 尝试3次通信，失败后自动重连
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # 检查连接状态
                if not self._check_connection():
                    if attempt < max_attempts - 1:
                        print(f"[TCP] 连接检查失败，尝试重连... ({attempt + 1}/{max_attempts})")
                        if self._reconnect():
                            continue
                    return {"success": False, "error": "连接检查失败"}
                
                # 发送命令
                command_data = json.dumps(command, ensure_ascii=False).encode('utf-8')
                length_bytes = len(command_data).to_bytes(4, byteorder='big')
                
                # 验证要发送的数据长度
                if len(command_data) > 1024 * 1024:  # 1MB
                    return {"success": False, "error": f"命令数据过大: {len(command_data)} 字节"}
                
                # 使用sendall确保数据完整发送
                self.socket.sendall(length_bytes)
                self.socket.sendall(command_data)
                
                # 精确接收响应长度
                length_bytes = self._recv_exact(4)
                if not length_bytes:
                    if attempt < max_attempts - 1:
                        print(f"[TCP] 接收长度失败，尝试重连... ({attempt + 1}/{max_attempts})")
                        if self._reconnect():
                            continue
                    return {"success": False, "error": "服务器关闭连接"}
                
                length = int.from_bytes(length_bytes, byteorder='big')
                
                # 验证长度合理性
                if length <= 0 or length > 1024 * 1024:  # 最大1MB
                    print(f"[TCP] 响应长度异常: {length}, 长度字节: {length_bytes.hex()}")
                    if attempt < max_attempts - 1:
                        print(f"[TCP] 尝试清理并重连... ({attempt + 1}/{max_attempts})")
                        self._clear_socket_buffer()
                        if self._reconnect():
                            continue
                    return {"success": False, "error": f"响应数据长度异常: {length}"}
                
                # 精确接收响应数据
                response_data = self._recv_exact(length)
                if not response_data:
                    if attempt < max_attempts - 1:
                        print(f"[TCP] 接收数据失败，尝试重连... ({attempt + 1}/{max_attempts})")
                        if self._reconnect():
                            continue
                    return {"success": False, "error": "接收响应数据失败"}
                
                # 安全解析JSON
                try:
                    response_str = response_data.decode('utf-8')
                    if not response_str.strip():
                        return {"success": False, "error": "服务器返回空响应"}
                        
                    response = json.loads(response_str)
                    return response
                except json.JSONDecodeError as je:
                    print(f"[TCP] JSON解析错误: {je}")
                    print(f"[TCP] 原始数据: {response_data[:100]}...")
                    if attempt < max_attempts - 1:
                        print(f"[TCP] 数据损坏，尝试重连... ({attempt + 1}/{max_attempts})")
                        if self._reconnect():
                            continue
                    return {"success": False, "error": f"JSON解析失败: {str(je)[:100]}"}
                except UnicodeDecodeError as ue:
                    return {"success": False, "error": f"编码解析失败: {str(ue)}"}
                    
            except socket.timeout:
                if attempt < max_attempts - 1:
                    print(f"[TCP] 通信超时，尝试重连... ({attempt + 1}/{max_attempts})")
                    if self._reconnect():
                        continue
                return {"success": False, "error": "通信超时"}
            except socket.error as se:
                self.connected = False
                if attempt < max_attempts - 1:
                    print(f"[TCP] 网络错误 {se}，尝试重连... ({attempt + 1}/{max_attempts})")
                    if self._reconnect():
                        continue
                return {"success": False, "error": f"网络错误: {str(se)}"}
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"[TCP] 通信异常 {e}，尝试重连... ({attempt + 1}/{max_attempts})")
                    if self._reconnect():
                        continue
                return {"success": False, "error": f"通信异常: {str(e)}"}
        
        return {"success": False, "error": "多次重试失败"}
    
    def _recv_exact(self, n: int) -> bytes:
        """精确接收指定长度的数据"""
        if not self.socket:
            return b''
            
        data = b''
        while len(data) < n:
            try:
                chunk = self.socket.recv(n - len(data))
                if not chunk:
                    return b''  # 连接关闭
                data += chunk
            except socket.timeout:
                print(f"[TCP] 接收数据超时，已接收 {len(data)}/{n} 字节")
                return b''
            except socket.error:
                return b''
        return data
    
    def _check_connection(self) -> bool:
        """检查连接是否有效"""
        if not self.socket or not self.connected:
            return False
        
        try:
            # 发送一个小的测试包检查连接
            self.socket.send(b'')
            return True
        except socket.error:
            self.connected = False
            return False
    
    def _reconnect(self) -> bool:
        """尝试重新连接"""
        try:
            self.disconnect_plc()
            time.sleep(0.5)  # 短暂等待
            return self.connect()
        except Exception as e:
            print(f"[TCP] 重连失败: {e}")
            return False
    
    def _clear_socket_buffer(self):
        """清理客户端Socket缓冲区"""
        if not self.socket:
            return
        
        try:
            self.socket.settimeout(0.1)
            while True:
                data = self.socket.recv(1024)
                if not data:
                    break
                print(f"[TCP] 清理缓冲区数据: {len(data)} 字节")
        except socket.timeout:
            pass  # 正常，缓冲区已清空
        except Exception:
            pass
        finally:
            self.socket.settimeout(10.0)  # 恢复原超时
            
    def read_current_layer(self) -> int:
        command = {
            "type": PLCDataProtocol.CMD_READ_LAYER,
            "timestamp": time.time()
        }
        response = self._send_command(command)
        return response.get("layer", -1) if response.get("success") else -1
        
    def read_machine_status(self) -> str:
        command = {
            "type": PLCDataProtocol.CMD_READ_STATUS,
            "timestamp": time.time()
        }
        response = self._send_command(command)
        return response.get("status", PLCDataProtocol.STATUS_ERROR) if response.get("success") else PLCDataProtocol.STATUS_ERROR
            
    def write_layer_completion(self, layer_id: int, success: bool, processing_time: float = 0.0):
        """写入层完成信号"""
        command = {
            "type": PLCDataProtocol.CMD_WRITE_COMPLETE,
            "layer": layer_id,
            "success": success,
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        response = self._send_command(command)
        return response.get("success", False)
        
    def send_correction_data(self, layer_id: int, correction_data: dict) -> bool:
        """发送纠偏数据到PLC（包含文件路径）"""
        # 整理纠偏数据，确保包含文件路径
        enhanced_data = self._prepare_correction_data(layer_id, correction_data)
        
        # 检查纠偏数据安全性
        safety_check = self._check_correction_safety(enhanced_data)
        if not safety_check["safe"]:
            self.send_deviation_alert(layer_id, safety_check["message"], safety_check["max_deviation"])
            return False
            
        command = {
            "type": PLCDataProtocol.CMD_SEND_CORRECTION,
            "layer": layer_id,
            "correction_status": PLCDataProtocol.CORRECTION_VALID,
            "data": enhanced_data,
            "timestamp": time.time()
        }
        response = self._send_command(command)
        return response.get("success", False)
        
    def _prepare_correction_data(self, layer_id: int, correction_data: dict) -> dict:
        """准备纠偏数据，包含corrected.gcode和offset_table.csv路径"""
        import os
        from pathlib import Path
        
        enhanced_data = correction_data.copy()
        
        # 找到当前层的输出目录
        output_dir = Path("output") / f"layer_{layer_id:02d}_out"
        
        if output_dir.exists():
            # 添加corrected.gcode路径
            corrected_gcode_path = output_dir / "corrected.gcode"
            if corrected_gcode_path.exists():
                enhanced_data["corrected_gcode_path"] = str(corrected_gcode_path)
                
            # 添加offset_table.csv路径
            offset_table_path = output_dir / "offset_table.csv"
            if offset_table_path.exists():
                enhanced_data["offset_table_path"] = str(offset_table_path)
                
            # 添加其他相关文件
            processing_metrics_path = output_dir / "processing_metrics.json"
            if processing_metrics_path.exists():
                enhanced_data["processing_metrics_path"] = str(processing_metrics_path)
                
            bias_comp_path = output_dir / "bias_compensation.json"
            if bias_comp_path.exists():
                enhanced_data["bias_compensation_path"] = str(bias_comp_path)
                
            # 添加文件列表信息
            enhanced_data["output_directory"] = str(output_dir)
            enhanced_data["available_files"] = [f.name for f in output_dir.iterdir() if f.is_file()]
            
        else:
            print(f"警告: 第{layer_id}层输出目录不存在: {output_dir}")
            
        return enhanced_data
        
    def send_deviation_alert(self, layer_id: int, alert_message: str, deviation_value: float):
        """发送偏差过大警告"""
        command = {
            "type": PLCDataProtocol.CMD_ALERT_ERROR,
            "layer": layer_id,
            "alert_message": alert_message,
            "deviation_value": deviation_value,
            "correction_status": PLCDataProtocol.CORRECTION_SKIP,
            "timestamp": time.time()
        }
        self._send_command(command)
        
    def _check_correction_safety(self, correction_data: dict) -> dict:
        """检查纠偏数据安全性"""
        try:
            # 安全阈值设置
            MAX_OFFSET_MM = 20.0  # 最大允许偏移量
            MAX_GRADIENT = 0.5    # 最大允许梯度
            
            # 检查偏移表数据
            if "offset_table_path" in correction_data:
                # 尝试使用pandas，如果没有则使用csv模块
                try:
                    # Type: ignore for pandas import
                    import pandas as pd  # type: ignore
                    offset_df = pd.read_csv(correction_data["offset_table_path"])
                    
                    # 检查最大偏移量
                    if "dx_mm" in offset_df.columns and "dy_mm" in offset_df.columns:
                        max_offset = ((offset_df["dx_mm"]**2 + offset_df["dy_mm"]**2)**0.5).max()
                        if max_offset > MAX_OFFSET_MM:
                            return {
                                "safe": False,
                                "message": f"偏移量过大: {max_offset:.2f}mm > {MAX_OFFSET_MM}mm",
                                "max_deviation": max_offset
                            }
                    
                    # 检查梯度变化
                    if len(offset_df) > 1:
                        dx_diff = offset_df["dx_mm"].diff().abs().max()
                        dy_diff = offset_df["dy_mm"].diff().abs().max()
                        max_gradient = max(dx_diff, dy_diff)
                        if max_gradient > MAX_GRADIENT:
                            return {
                                "safe": False,
                                "message": f"梯度变化过大: {max_gradient:.3f} > {MAX_GRADIENT}",
                                "max_deviation": max_gradient
                            }
                            
                except ImportError:
                    # 如果没有pandas，使用csv模块读取
                    import csv
                    with open(correction_data["offset_table_path"], 'r') as f:
                        reader = csv.DictReader(f)
                        data = list(reader)
                        if not data:
                            return {"safe": True, "message": "空数据文件", "max_deviation": 0.0}
                        
                        # 检查最大偏移量
                        max_offset = 0.0
                        max_gradient = 0.0
                        prev_dx, prev_dy = 0.0, 0.0
                        
                        for i, row in enumerate(data):
                            dx = float(row.get("dx_mm", 0))
                            dy = float(row.get("dy_mm", 0))
                            offset = (dx**2 + dy**2)**0.5
                            max_offset = max(max_offset, offset)
                            
                            # 检查梯度
                            if i > 0:
                                gradient = max(abs(dx - prev_dx), abs(dy - prev_dy))
                                max_gradient = max(max_gradient, gradient)
                            prev_dx, prev_dy = dx, dy
                            
                        if max_offset > MAX_OFFSET_MM:
                            return {
                                "safe": False,
                                "message": f"偏移量过大: {max_offset:.2f}mm > {MAX_OFFSET_MM}mm",
                                "max_deviation": max_offset
                            }
                            
                        if max_gradient > MAX_GRADIENT:
                            return {
                                "safe": False,
                                "message": f"梯度变化过大: {max_gradient:.3f} > {MAX_GRADIENT}",
                                "max_deviation": max_gradient
                            }
            
            return {"safe": True, "message": "数据安全", "max_deviation": 0.0}
            
        except Exception as e:
            return {
                "safe": False,
                "message": f"数据检查错误: {str(e)}",
                "max_deviation": float('inf')
            }

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
            
    def disconnect_plc(self):
        super().disconnect_plc()
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
        
    def disconnect_plc(self):
        super().disconnect_plc()
        
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
        self.last_status = ""
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
                    
                # 读取机床状态
                machine_status = self.communicator.read_machine_status()
                if machine_status != self.last_status:
                    self.last_status = machine_status
                    if hasattr(self.communicator, 'machine_status_changed'):
                        self.communicator.machine_status_changed.emit(machine_status)
                    
                    # 如果机床进入等待状态，发出纠偏请求信号
                    if (machine_status == "waiting" and 
                        hasattr(self.communicator, 'correction_request')):
                        self.communicator.correction_request.emit(current_layer)
                
                # 兼容旧接口：读取开始信号
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