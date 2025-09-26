#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7模拟器通信类 - 扩展主程序支持S7模拟器
在主程序中添加"s7_sim"通信类型，连接到S7模拟器进行测试
"""

import sys
import os
import time
import struct
from typing import Dict, Any, Optional

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from multilayer_plc import PLCCommunicator
from s7_simulator.mock_s7_communicator import MockS7Communicator

class S7SimulatorPLCCommunicator(PLCCommunicator):
    """S7模拟器PLC通信类"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8502):
        super().__init__()
        self.host = host
        self.port = port
        self.s7_client = None
        self.safety_limits = {
            "max_offset_mm": 20.0,        # 最大偏移量20mm
            "max_gradient": 0.5,          # 最大梯度0.5mm/mm
        }
    
    def connect(self) -> bool:
        """连接到S7模拟器"""
        try:
            self.s7_client = MockS7Communicator(self.host, self.port)
            
            # 连接到模拟器 (模拟真实S7参数)
            success = self.s7_client.connect(ip="192.168.0.2", rack=0, slot=1)
            
            if success:
                self.connected = True
                self.status_message = "已连接到S7模拟器"
                return True
            else:
                self.status_message = "连接S7模拟器失败"
                return False
                
        except Exception as e:
            self.status_message = f"S7模拟器连接错误: {e}"
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.s7_client:
            try:
                self.s7_client.disconnect()
            except:
                pass
            
            self.s7_client = None
        
        self.connected = False
        self.status_message = "已断开S7模拟器连接"
    
    def read_current_layer(self) -> int:
        """读取当前层号"""
        if not self.connected or not self.s7_client:
            return 0
        
        try:
            return self.s7_client.get_current_layer()
        except Exception as e:
            print(f"读取层号失败: {e}")
            return 0
    
    def read_machine_status(self) -> str:
        """读取机床状态"""
        if not self.connected or not self.s7_client:
            return "disconnected"
        
        try:
            return self.s7_client.get_machine_status()
        except Exception as e:
            print(f"读取机床状态失败: {e}")
            return "error"
    
    def send_correction_data(self, layer_id: int, data: Dict[str, Any]) -> bool:
        """发送纠偏数据到S7模拟器"""
        if not self.connected or not self.s7_client:
            return False
        
        try:
            # 提取偏移数据
            offset_table = data.get("offset_table", [])
            if not offset_table:
                print("警告: 没有偏移数据")
                return True  # 空数据视为成功
            
            # 安全检查
            safe_offsets = self._validate_safety(offset_table)
            if not safe_offsets:
                print("警告: 偏移数据安全检查失败，跳过发送")
                return True  # 安全起见，返回成功但不发送数据
            
            # 设置处理锁
            self.s7_client.set_processing_lock(True)
            
            # 分批发送数据 (每批128个点)
            batch_size = 128
            total_points = len(safe_offsets)
            total_batches = (total_points + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_points)
                batch_data = safe_offsets[start_idx:end_idx]
                
                # 发送批次数据
                self.s7_client.write_offset_batch(batch_data, batch_num + 1, total_batches)
                
                print(f"发送批次 {batch_num + 1}/{total_batches}: {len(batch_data)} 个偏移点")
                
                # 批次间短暂延时
                if batch_num < total_batches - 1:
                    time.sleep(0.1)
            
            # 释放处理锁
            self.s7_client.set_processing_lock(False)
            
            print(f"纠偏数据发送完成: 第{layer_id}层, {total_points}个偏移点, {total_batches}个批次")
            return True
            
        except Exception as e:
            print(f"发送纠偏数据失败: {e}")
            
            # 确保释放锁
            try:
                if self.s7_client:
                    self.s7_client.set_processing_lock(False)
            except:
                pass
            
            return False
    
    def _validate_safety(self, offset_table: list) -> list:
        """安全验证偏移数据"""
        if not offset_table:
            return []
        
        safe_offsets = []
        max_offset = self.safety_limits["max_offset_mm"]
        max_gradient = self.safety_limits["max_gradient"]
        
        for i, (x, y, dx, dy) in enumerate(offset_table):
            # 检查偏移量大小
            offset_magnitude = (dx**2 + dy**2)**0.5
            if offset_magnitude > max_offset:
                print(f"警告: 点{i} 偏移量过大 {offset_magnitude:.3f}mm > {max_offset}mm，设为0")
                dx, dy = 0.0, 0.0
            
            # 检查梯度变化 (与前一个点比较)
            if i > 0:
                prev_dx, prev_dy = safe_offsets[-1][2], safe_offsets[-1][3]
                gradient_dx = abs(dx - prev_dx)
                gradient_dy = abs(dy - prev_dy)
                
                if gradient_dx > max_gradient:
                    print(f"警告: 点{i} X梯度过大 {gradient_dx:.3f}mm/mm，使用前值")
                    dx = prev_dx
                
                if gradient_dy > max_gradient:
                    print(f"警告: 点{i} Y梯度过大 {gradient_dy:.3f}mm/mm，使用前值")
                    dy = prev_dy
            
            safe_offsets.append((x, y, dx, dy))
        
        return safe_offsets
    
    def get_status_info(self) -> Dict[str, Any]:
        """获取详细状态信息"""
        if not self.connected or not self.s7_client:
            return {
                "connected": False,
                "machine_status": "disconnected",
                "current_layer": 0,
                "processing_lock": False,
                "heartbeat": 0
            }
        
        try:
            # 获取基本状态
            status_info = {
                "connected": True,
                "machine_status": self.s7_client.get_machine_status(),
                "current_layer": self.s7_client.get_current_layer(),
                "processing_lock": self.s7_client.get_processing_lock(),
            }
            
            # 获取偏移数据信息
            offset_info = self.s7_client.get_offset_data_info()
            status_info.update(offset_info)
            
            # 获取心跳
            try:
                heartbeat = self.s7_client.read_int16(9044, 16)
                status_info["heartbeat"] = heartbeat
            except:
                status_info["heartbeat"] = 0
            
            return status_info
            
        except Exception as e:
            print(f"获取状态信息失败: {e}")
            return {
                "connected": False,
                "machine_status": "error",
                "current_layer": 0,
                "processing_lock": False,
                "heartbeat": 0,
                "error": str(e)
            }

# 工厂函数，用于在主程序中创建S7模拟器通信器
def create_s7_simulator_communicator(config: Dict[str, Any]) -> S7SimulatorPLCCommunicator:
    """创建S7模拟器通信器"""
    host = config.get("plc_ip", "127.0.0.1")
    port = config.get("s7_sim_port", 8502)  # 使用专用端口
    
    return S7SimulatorPLCCommunicator(host, port)