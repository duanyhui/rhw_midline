# -*- coding: utf-8 -*-
"""
增强的S7协议PLC通信模块
集成snap7库和数据块操作，实现完整的PLC数据交换功能
"""
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
import traceback

from multilayer_data import ProjectConfig
from multilayer_plc import PLCCommunicator, PLCDataProtocol
from plc_data_structures import (
    PLCDataBlocks, PLCErrorCodes, ControlBlockManager, OffsetBlockManager, 
    PLCDataManager, MachineStatus, ProgramStatus, DataLockStatus
)
from offset_data_handler import OffsetDataHandler, OffsetDataLoader, OffsetDataConfig


class S7PLCEnhanced(PLCCommunicator):
    """增强的S7协议PLC通信器"""
    
    # 额外信号
    data_transmission_progress = pyqtSignal(int, int, str)  # 当前批次, 总批次, 状态信息
    safety_alert = pyqtSignal(int, str, float)  # 层号, 警告信息, 偏差值
    heartbeat_updated = pyqtSignal(int)  # 心跳计数
    
    def __init__(self, config: ProjectConfig):
        super().__init__(config)
        
        # S7连接参数
        self.plc_ip = config.plc_ip
        self.rack = 0  # 西门子PLC默认机架号
        self.slot = 2  # 西门子PLC默认插槽号
        
        # snap7客户端
        self.client = None
        self._snap7_available = False
        
        # 模拟模式标志
        self._simulation_mode = False
        
        # TCP模拟连接
        self._tcp_socket = None
        
        # 数据管理器
        self.data_manager = PLCDataManager()
        self.offset_handler = OffsetDataHandler(OffsetDataConfig(
            max_offset_mm=20.0,
            max_gradient=0.5,
            enable_safety_check=True,
            enable_filtering=True
        ))
        self.offset_loader = OffsetDataLoader()
        
        # 传输状态
        self.transmission_lock = threading.Lock()
        self.current_transmission = None
        self.transmission_active = False
        
        # 心跳定时器
        self.heartbeat_timer = QTimer()
        self.heartbeat_timer.timeout.connect(self._update_heartbeat)
        self.heartbeat_interval = 1000  # 1秒心跳
        
        # 检查snap7是否可用
        self._check_snap7_availability()
    
    def _check_snap7_availability(self):
        """检查snap7库是否可用"""
        try:
            import snap7
            self._snap7_available = True
            print("snap7库可用")
        except ImportError:
            self._snap7_available = False
            print("警告: 未安装python-snap7库，S7通信将不可用")
    
    def connect(self) -> bool:
        """连接PLC - 智能切换连接方式"""
        if not self._snap7_available:
            self.connection_status.emit(False, "S7连接失败: 未安装python-snap7库")
            return False
        
        # 根据用户记忆：智能判断IP地址和协议类型
        is_local_ip = self.plc_ip in ['127.0.0.1', 'localhost'] or self.plc_ip.startswith('192.168.')
        
        # 根据用户记忆：避免127.0.0.1与S7协议组合
        if self.plc_ip == '127.0.0.1' and self.config.plc_type == 's7':
            print("警告: 检测到127.0.0.1与S7协议组合，自动切换到TCP模式")
            return self._try_tcp_connection()
        
        try:
            import snap7
            
            # 创建S7客户端
            self.client = snap7.client.Client()
            
            # 连接到PLC
            self.client.connect(self.plc_ip, self.rack, self.slot)
            
            # 验证连接
            if self.client.get_connected():
                self.connected = True
                
                # 初始化数据管理器
                self._initialize_plc_data()
                
                # 启动心跳
                self.heartbeat_timer.start(self.heartbeat_interval)
                
                self.connection_status.emit(True, f"S7连接成功 ({self.plc_ip})")
                print(f"S7 PLC连接成功: {self.plc_ip}")
                return True
            else:
                self.connection_status.emit(False, "S7连接验证失败")
                return False
                
        except Exception as e:
            error_msg = str(e)
            print(f"S7连接错误: {e}")
            
            # 根据用户记忆：智能切换策略
            if 'Bad PDU format' in error_msg or 'Unreachable peer' in error_msg:
                if is_local_ip:
                    print("检测到PDU格式错误或连接不可达，尝试切换到TCP模式...")
                    return self._try_tcp_connection()
                else:
                    print("远程PLC连接失败，尝试回退到模拟器...")
                    return self._try_fallback_to_simulator()
            else:
                self.connection_status.emit(False, f"S7连接失败: {error_msg}")
                print(traceback.format_exc())
                return False
    
    def _try_tcp_connection(self) -> bool:
        """尝试TCP协议连接（回退到模拟器）"""
        try:
            # 使用配置中的端口号，而不是固定的502
            target_port = self.config.plc_port
            print(f"尝试TCP连接到 {self.plc_ip}:{target_port}...")
            
            import socket
            import json
            
            # 创建持久的TCP连接
            self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._tcp_socket.settimeout(5.0)
            
            try:
                self._tcp_socket.connect((self.plc_ip if self.plc_ip != 'localhost' else '127.0.0.1', target_port))
                print(f"TCP端口{target_port}可达，建立模拟器连接")
                
                # 测试通信：发送一个简单的状态查询
                test_command = {
                    "type": "read_machine_status",
                    "timestamp": time.time()
                }
                
                command_data = json.dumps(test_command).encode('utf-8')
                # 发送命令长度和命令数据
                self._tcp_socket.send(len(command_data).to_bytes(4, byteorder='big'))
                self._tcp_socket.send(command_data)
                
                # 接收响应
                response_length_data = self._tcp_socket.recv(4)
                if len(response_length_data) == 4:
                    response_length = int.from_bytes(response_length_data, byteorder='big')
                    response_data = self._tcp_socket.recv(response_length)
                    response = json.loads(response_data.decode('utf-8'))
                    
                    if response.get('success', False):
                        print(f"模拟器通信成功，当前状态: {response.get('status', 'unknown')}")
                        
                        self.connected = True
                        self._simulation_mode = True
                        
                        # 初始化数据管理器（模拟模式）
                        self._initialize_plc_data_simulation()
                        
                        # 启动心跳（模拟模式）
                        self.heartbeat_timer.start(self.heartbeat_interval)
                        
                        self.connection_status.emit(True, f"模拟器连接成功 ({self.plc_ip}:{target_port})")
                        return True
                    else:
                        print(f"模拟器响应错误: {response}")
                        self._tcp_socket.close()
                        self._tcp_socket = None
                        return False
                else:
                    print(f"模拟器响应格式错误")
                    self._tcp_socket.close()
                    self._tcp_socket = None
                    return False
                    
            except Exception as e:
                print(f"TCP通信失败: {e}")
                if self._tcp_socket:
                    self._tcp_socket.close()
                    self._tcp_socket = None
                return False
                
        except Exception as e:
            print(f"TCP连接尝试失败: {e}")
            return False
    
    def _send_tcp_command(self, command: dict) -> dict:
        """发送TCP命令到模拟器"""
        if not self._tcp_socket:
            return {"success": False, "error": "TCP连接未建立"}
        
        try:
            import json
            
            # 序列化命令
            command_data = json.dumps(command).encode('utf-8')
            
            # 发送命令长度和命令数据
            self._tcp_socket.send(len(command_data).to_bytes(4, byteorder='big'))
            self._tcp_socket.send(command_data)
            
            # 接收响应长度
            response_length_data = self._tcp_socket.recv(4)
            if len(response_length_data) != 4:
                return {"success": False, "error": "响应长度错误"}
            
            response_length = int.from_bytes(response_length_data, byteorder='big')
            
            # 接收响应数据
            response_data = b''
            while len(response_data) < response_length:
                chunk = self._tcp_socket.recv(response_length - len(response_data))
                if not chunk:
                    return {"success": False, "error": "响应数据不完整"}
                response_data += chunk
            
            # 解析响应
            response = json.loads(response_data.decode('utf-8'))
            return response
            
        except Exception as e:
            print(f"TCP命令发送失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _try_fallback_to_simulator(self) -> bool:
        """回退到本地模拟器连接"""
        original_ip = self.plc_ip  # 在函数开始处声明
        original_port = self.config.plc_port  # 保存原始端口
        
        try:
            print("尝试回退到本地模拟器 (127.0.0.1:502)...")
            
            # 更改连接参数到本地模拟器
            self.plc_ip = '127.0.0.1'
            self.config.plc_port = 502  # 使用标准模拟器端口
            
            # 尝试TCP连接到本地模拟器
            success = self._try_tcp_connection()
            
            if success:
                print(f"成功回退到本地模拟器 (原始IP: {original_ip}:{original_port})")
                self.connection_status.emit(True, f"已回退到本地模拟器 (原始IP: {original_ip}:{original_port})")
                return True
            else:
                # 恢复原始配置
                self.plc_ip = original_ip
                self.config.plc_port = original_port
                print("回退到本地模拟器失败")
                self.connection_status.emit(False, f"S7连接失败，本地模拟器也不可用")
                return False
                
        except Exception as e:
            # 恢复原始配置
            self.plc_ip = original_ip
            self.config.plc_port = original_port
            print(f"回退到模拟器失败: {e}")
            self.connection_status.emit(False, f"连接失败: {str(e)}")
            return False
    
    def _initialize_plc_data_simulation(self):
        """初始化PLC数据（模拟模式）"""
        try:
            print("初始化PLC数据（模拟模式）...")
            
            # 设置程序状态为已连接
            self.data_manager.control_block.set_program_status(ProgramStatus.CONNECTED)
            self.data_manager.control_block.update_timestamp()
            
            # 在模拟模式下，我们不实际写入数据块，只是记录状态
            print("PLC数据初始化完成（模拟模式）")
            
        except Exception as e:
            print(f"PLC数据初始化失败（模拟模式）: {e}")
    
    def disconnect_plc(self):
        """断开PLC连接"""
        super().disconnect_plc()
        
        # 停止心跳
        if self.heartbeat_timer.isActive():
            self.heartbeat_timer.stop()
        
        # 断开S7连接
        if self.client:
            try:
                self.client.disconnect()
                print("S7 PLC连接已断开")
            except:
                pass
            self.client = None
        
        # 断开TCP模拟连接
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
                print("TCP模拟连接已断开")
            except:
                pass
            self._tcp_socket = None
        
        # 重置模拟模式
        self._simulation_mode = False
    
    def _initialize_plc_data(self):
        """初始化PLC数据"""
        try:
            # 设置程序状态为已连接
            self.data_manager.control_block.set_program_status(ProgramStatus.CONNECTED)
            self.data_manager.control_block.update_timestamp()
            
            # 写入控制数据块
            self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
            
            print("PLC数据初始化完成")
            
        except Exception as e:
            print(f"PLC数据初始化失败: {e}")
    
    def _read_data_block(self, db_number: int, offset: int = 0, size: Optional[int] = None) -> Optional[bytes]:
        """读取PLC数据块"""
        if not self.connected:
            return None
        
        # 模拟模式下通过TCP获取实时数据
        if getattr(self, '_simulation_mode', False):
            try:
                # 发送读取命令到模拟器
                command = {
                    "type": "read_data_block",
                    "db_number": db_number,
                    "offset": offset,
                    "size": size or PLCDataBlocks.DB_SIZE,
                    "timestamp": time.time()
                }
                
                response = self._send_tcp_command(command)
                if response.get("success", False):
                    # 从响应中获取数据
                    data_hex = response.get("data", "")
                    if data_hex:
                        # 将十六进制字符串转换为字节
                        data_bytes = bytes.fromhex(data_hex)
                        
                        # 更新本地数据管理器
                        if db_number == PLCDataBlocks.DB_CONTROL:
                            self.data_manager.control_block.set_data(data_bytes)
                        
                        return data_bytes
                    else:
                        # 如果没有数据，返回默认值
                        return self.data_manager.control_block.get_data()[:size or PLCDataBlocks.DB_SIZE]
                else:
                    print(f"模拟器读取失败: {response.get('error', 'unknown')}")
                    # 返回本地缓存数据作为备用
                    return self.data_manager.control_block.get_data()[:size or PLCDataBlocks.DB_SIZE]
                    
            except Exception as e:
                print(f"模拟模式读取失败: {e}")
                # 返回本地缓存数据作为备用
                return self.data_manager.control_block.get_data()[:size or PLCDataBlocks.DB_SIZE]
            
        if not self.client:
            return None
        
        if size is None:
            size = PLCDataBlocks.DB_SIZE
        
        try:
            data = self.client.db_read(db_number, offset, size)
            return bytes(data)
        except Exception as e:
            print(f"读取数据块DB{db_number}失败: {e}")
            return None
    
    def _write_data_block(self, db_number: int, data: bytes, offset: int = 0) -> bool:
        """写入PLC数据块"""
        if not self.connected:
            return False
        
        # 模拟模式下模拟写入操作
        if getattr(self, '_simulation_mode', False):
            # 更新本地数据管理器
            if db_number == PLCDataBlocks.DB_CONTROL:
                self.data_manager.control_block.set_data(data)
                print(f"模拟写入数据块DB{db_number}成功 ({len(data)}字节)")
                return True
            return True
            
        if not self.client:
            return False
        
        try:
            self.client.db_write(db_number, offset, bytearray(data))
            return True
        except Exception as e:
            print(f"写入数据块DB{db_number}失败: {e}")
            return False
    
    def _update_heartbeat(self):
        """更新心跳计数"""
        if not self.connected:
            return
        
        try:
            # 模拟模式下的心跳更新
            if getattr(self, '_simulation_mode', False):
                # 直接更新本地数据管理器
                self.data_manager.control_block.increment_heartbeat()
                self.data_manager.control_block.update_timestamp()
                
                # 发出心跳信号
                heartbeat = self.data_manager.control_block.get_heartbeat()
                self.heartbeat_updated.emit(heartbeat)
                
                # 每10次心跳显示一次信息
                if heartbeat % 10 == 0:
                    print(f"模拟模式心跳: {heartbeat}")
                return
            
            # 读取当前控制块
            control_data = self._read_data_block(PLCDataBlocks.DB_CONTROL)
            if control_data:
                self.data_manager.control_block.set_data(control_data)
                
                # 更新心跳和时间戳
                self.data_manager.control_block.increment_heartbeat()
                self.data_manager.control_block.update_timestamp()
                
                # 写回控制块
                self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
                
                # 发出心跳信号
                heartbeat = self.data_manager.control_block.get_heartbeat()
                self.heartbeat_updated.emit(heartbeat)
                
        except Exception as e:
            print(f"心跳更新失败: {e}")
    
    def read_current_layer(self) -> int:
        """读取当前层号"""
        if not self.connected:
            return -1
        
        try:
            control_data = self._read_data_block(PLCDataBlocks.DB_CONTROL)
            if control_data:
                self.data_manager.control_block.set_data(control_data)
                return self.data_manager.control_block.get_current_layer()
        except Exception as e:
            print(f"读取层号失败: {e}")
        
        return -1
    
    def read_machine_status(self) -> str:
        """读取机床状态"""
        if not self.connected:
            return PLCDataProtocol.STATUS_ERROR
        
        try:
            control_data = self._read_data_block(PLCDataBlocks.DB_CONTROL)
            if control_data:
                self.data_manager.control_block.set_data(control_data)
                machine_status = self.data_manager.control_block.get_machine_status()
                
                # 转换为协议状态
                status_map = {
                    MachineStatus.IDLE: PLCDataProtocol.STATUS_IDLE,
                    MachineStatus.PROCESSING: PLCDataProtocol.STATUS_PROCESSING,
                    MachineStatus.WAITING: PLCDataProtocol.STATUS_WAITING,
                    MachineStatus.ERROR: PLCDataProtocol.STATUS_ERROR,
                    MachineStatus.COMPLETED: PLCDataProtocol.STATUS_IDLE
                }
                
                return status_map.get(machine_status, PLCDataProtocol.STATUS_ERROR)
                
        except Exception as e:
            print(f"读取机床状态失败: {e}")
        
        return PLCDataProtocol.STATUS_ERROR
    
    def write_layer_completion(self, layer_id: int, success: bool, processing_time: float = 0.0):
        """写入层完成信号"""
        if not self.connected:
            return False
        
        try:
            # 更新控制块状态
            if success:
                self.data_manager.control_block.set_program_status(ProgramStatus.COMPLETED)
                self.data_manager.control_block.set_error_code(PLCErrorCodes.NO_ERROR)
            else:
                self.data_manager.control_block.set_program_status(ProgramStatus.ERROR)
                self.data_manager.control_block.set_error_code(PLCErrorCodes.UNKNOWN_ERROR)
            
            self.data_manager.control_block.set_current_layer(layer_id)
            self.data_manager.control_block.update_timestamp()
            
            # 写入控制块
            success_write = self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
            
            if success_write:
                print(f"层{layer_id}完成信号已写入PLC")
            
            return success_write
            
        except Exception as e:
            print(f"写入层完成信号失败: {e}")
            return False
    
    def send_correction_data(self, layer_id: int, correction_data: dict) -> bool:
        """发送纠偏数据到PLC"""
        if not self.connected:
            print("PLC未连接，无法发送纠偏数据")
            return False
        
        # 第1层仅做标定，不发送纠偏数据
        if layer_id == 1:
            print(f"第{layer_id}层为标定层，跳过纠偏数据发送")
            self.data_manager.control_block.set_layer_type(0)  # 标定层
            self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
            return True
        
        try:
            # 加载偏移数据
            offset_points, processing_result = self.offset_loader.load_from_correction_data(correction_data)
            
            if not processing_result.success:
                error_msg = f"加载偏移数据失败: {processing_result.error_message}"
                print(error_msg)
                self.safety_alert.emit(layer_id, error_msg, 0.0)
                return False
            
            if not offset_points:
                print(f"第{layer_id}层没有有效的偏移数据")
                return False
            
            # 安全检查
            if processing_result.max_offset > self.offset_handler.config.max_offset_mm:
                alert_msg = f"偏移量过大: {processing_result.max_offset:.3f}mm"
                print(f"安全警告: {alert_msg}")
                self.safety_alert.emit(layer_id, alert_msg, processing_result.max_offset)
                
                # 根据配置决定是否继续
                if not self.offset_handler.config.enable_filtering:
                    return False
            
            # 设置为纠偏层
            self.data_manager.control_block.set_layer_type(1)  # 纠偏层
            
            # 开始分批传输
            return self._transmit_offset_data_in_batches(layer_id, offset_points)
            
        except Exception as e:
            error_msg = f"发送纠偏数据失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            self.safety_alert.emit(layer_id, error_msg, 0.0)
            return False
    
    def _transmit_offset_data_in_batches(self, layer_id: int, offset_points: List) -> bool:
        """分批传输偏移数据"""
        with self.transmission_lock:
            if self.transmission_active:
                print("数据传输正在进行中，请稍候...")
                return False
            
            self.transmission_active = True
        
        try:
            # 创建传输计划
            transmission_plan = self.offset_handler.create_batch_transmission_plan(offset_points)
            total_batches = transmission_plan['total_batches']
            
            print(f"开始分批传输第{layer_id}层偏移数据:")
            print(f"  总点数: {transmission_plan['total_points']}")
            print(f"  总批次: {total_batches}")
            print(f"  每批次最大点数: {transmission_plan['max_points_per_batch']}")
            
            # 逐批传输
            for batch_info in transmission_plan['batches']:
                batch_index = batch_info['batch_index']
                
                # 发出进度信号
                status_msg = f"传输第{batch_index + 1}/{total_batches}批次 ({batch_info['point_count']}个点)"
                self.data_transmission_progress.emit(batch_index + 1, total_batches, status_msg)
                
                # 检查数据锁
                if not self._wait_for_data_unlock():
                    print(f"等待数据解锁超时，批次{batch_index}传输失败")
                    return False
                
                # 设置锁定状态
                self._set_data_lock(DataLockStatus.LOCKED)
                
                try:
                    # 获取当前批次的偏移点
                    start_idx = batch_info['start_index']
                    end_idx = batch_info['end_index']
                    batch_points = offset_points[start_idx:end_idx]
                    
                    # 设置批次数据到数据管理器
                    success = self.data_manager.set_offset_batch(offset_points, batch_index)
                    if not success:
                        print(f"设置批次{batch_index}数据失败")
                        return False
                    
                    # 更新控制信息
                    self.data_manager.control_block.set_current_layer(layer_id)
                    self.data_manager.control_block.set_current_batch(batch_index)
                    self.data_manager.control_block.set_program_status(ProgramStatus.PROCESSING)
                    self.data_manager.control_block.update_timestamp()
                    
                    # 写入所有数据块
                    all_data_blocks = self.data_manager.get_all_data_blocks()
                    
                    write_success = True
                    for db_num, db_data in all_data_blocks.items():
                        if not self._write_data_block(db_num, db_data):
                            write_success = False
                            print(f"写入数据块DB{db_num}失败")
                            break
                    
                    if not write_success:
                        return False
                    
                    print(f"批次{batch_index}传输完成 ({batch_info['point_count']}个点)")
                    
                finally:
                    # 解除锁定
                    self._set_data_lock(DataLockStatus.UNLOCKED)
                
                # 批次间延迟
                time.sleep(0.1)
            
            # 传输完成，更新状态
            self.data_manager.control_block.set_program_status(ProgramStatus.COMPLETED)
            self.data_manager.control_block.set_error_code(PLCErrorCodes.NO_ERROR)
            self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
            
            print(f"第{layer_id}层偏移数据传输完成！")
            self.data_transmission_progress.emit(total_batches, total_batches, "传输完成")
            
            return True
            
        except Exception as e:
            error_msg = f"分批传输失败: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            
            # 设置错误状态
            self.data_manager.control_block.set_program_status(ProgramStatus.ERROR)
            self.data_manager.control_block.set_error_code(PLCErrorCodes.UNKNOWN_ERROR)
            self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
            
            return False
            
        finally:
            with self.transmission_lock:
                self.transmission_active = False
    
    def _wait_for_data_unlock(self, timeout_seconds: float = 10.0) -> bool:
        """等待数据解锁"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            try:
                control_data = self._read_data_block(PLCDataBlocks.DB_CONTROL)
                if control_data:
                    self.data_manager.control_block.set_data(control_data)
                    lock_status = self.data_manager.control_block.get_data_lock()
                    
                    if lock_status == DataLockStatus.UNLOCKED:
                        return True
                    
                time.sleep(0.1)  # 100ms检查间隔
                
            except Exception as e:
                print(f"检查数据锁状态失败: {e}")
                time.sleep(0.5)
        
        print(f"等待数据解锁超时 ({timeout_seconds}秒)")
        return False
    
    def _set_data_lock(self, lock_status: DataLockStatus) -> bool:
        """设置数据锁状态"""
        try:
            self.data_manager.control_block.set_data_lock(lock_status)
            return self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
        except Exception as e:
            print(f"设置数据锁失败: {e}")
            return False
    
    def send_deviation_alert(self, layer_id: int, alert_message: str, deviation_value: float):
        """发送偏差过大警告"""
        try:
            print(f"发送偏差警告 - 层{layer_id}: {alert_message} (偏差: {deviation_value:.3f})")
            
            # 设置错误状态
            self.data_manager.control_block.set_current_layer(layer_id)
            self.data_manager.control_block.set_program_status(ProgramStatus.ERROR)
            self.data_manager.control_block.set_error_code(PLCErrorCodes.OFFSET_TOO_LARGE)
            self.data_manager.control_block.update_timestamp()
            
            # 写入控制块
            self._write_data_block(PLCDataBlocks.DB_CONTROL, self.data_manager.control_block.get_data())
            
            # 发出信号
            self.safety_alert.emit(layer_id, alert_message, deviation_value)
            
        except Exception as e:
            print(f"发送偏差警告失败: {e}")
    
    def get_plc_status_summary(self) -> Dict[str, Any]:
        """获取PLC状态摘要"""
        try:
            if not self.connected:
                return {
                    "connected": False,
                    "error": "PLC未连接"
                }
            
            # 读取控制块
            control_data = self._read_data_block(PLCDataBlocks.DB_CONTROL)
            if control_data:
                self.data_manager.control_block.set_data(control_data)
            
            # 获取系统状态
            system_status = self.data_manager.get_system_status()
            
            # 补充连接信息
            system_status["connection"] = {
                "connected": self.connected,
                "plc_ip": self.plc_ip,
                "rack": self.rack,
                "slot": self.slot,
                "transmission_active": self.transmission_active
            }
            
            return system_status
            
        except Exception as e:
            return {
                "connected": self.connected,
                "error": f"获取状态失败: {str(e)}"
            }
    
    def read_offset_data_from_plc(self, batch_index: Optional[int] = None) -> Dict[str, Any]:
        """从PLC读取偏移数据（用于验证和调试）"""
        try:
            if not self.connected:
                return {"success": False, "error": "PLC未连接"}
            
            # 读取控制块
            control_data = self._read_data_block(PLCDataBlocks.DB_CONTROL)
            if control_data:
                self.data_manager.control_block.set_data(control_data)
            
            # 读取偏移数据块
            offset_data = {}
            for db_num in [PLCDataBlocks.DB_OFFSET_1, PLCDataBlocks.DB_OFFSET_2, PLCDataBlocks.DB_OFFSET_3]:
                db_data = self._read_data_block(db_num)
                if db_data:
                    self.data_manager.offset_blocks[db_num].set_data(db_data)
                    points = self.data_manager.offset_blocks[db_num].get_offset_points()
                    
                    # 只保留非零点
                    non_zero_points = []
                    for i, point in enumerate(points):
                        if abs(point.dx_mm) > 0.001 or abs(point.dy_mm) > 0.001:
                            non_zero_points.append({
                                "index": i,
                                "dx_mm": point.dx_mm,
                                "dy_mm": point.dy_mm
                            })
                    
                    offset_data[f"DB{db_num}"] = {
                        "total_points": len(points),
                        "non_zero_points": len(non_zero_points),
                        "data": non_zero_points[:10]  # 只显示前10个点
                    }
            
            control_info = self.data_manager.control_block.to_dict()
            
            return {
                "success": True,
                "control": control_info,
                "offset_data": offset_data,
                "read_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"读取偏移数据失败: {str(e)}"
            }


if __name__ == "__main__":
    # 测试代码
    print("S7增强通信模块测试...")
    
    # 创建测试配置
    from multilayer_data import ProjectConfig
    
    test_config = ProjectConfig()
    test_config.plc_ip = "192.168.0.100"  # 测试IP
    test_config.plc_type = "s7"
    
    # 创建通信器
    s7_comm = S7PLCEnhanced(test_config)
    
    # 显示状态
    status = s7_comm.get_plc_status_summary()
    print(f"PLC状态摘要: {status}")
    
    print("测试完成!")