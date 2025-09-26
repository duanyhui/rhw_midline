#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟S7通信类 - 兼容主程序的S7通信接口
连接到S7 PLC模拟器，提供与真实S7通信相同的接口
"""

import socket
import struct
import time
import json
from typing import Dict, Any, Optional

class MockS7Communicator:
    """模拟S7通信类"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8502):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        # S7数据块映射
        self.db_mappings = {
            # DB9044 - 控制状态数据块
            "machine_status": (9044, 0),      # 机床状态
            "current_layer": (9044, 2),       # 当前层号
            "total_layers": (9044, 4),        # 总层数
            "processing_lock": (9044, 6),     # 处理锁
            "offset_count": (9044, 8),        # 偏移点总数
            "current_batch": (9044, 10),      # 当前批次
            "total_batches": (9044, 12),      # 总批次数
            "data_ready": (9044, 14),         # 数据就绪
            "heartbeat": (9044, 16),          # 心跳计数
            "error_code": (9044, 18),         # 错误码
            "process_delay": (9044, 20),      # 处理延时
        }
    
    def connect(self, ip: str, rack: int = 0, slot: int = 1) -> bool:
        """连接到PLC模拟器"""
        try:
            # 模拟snap7连接参数，但实际连接到模拟器
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            
            # 尝试连接到模拟器的TCP服务
            self.socket.connect((self.host, self.port))
            
            # 发送连接请求
            connect_msg = {
                "type": "connect",
                "ip": ip,
                "rack": rack,
                "slot": slot,
                "timestamp": time.time()
            }
            
            self._send_message(connect_msg)
            response = self._receive_message()
            
            if response and response.get("status") == "connected":
                self.connected = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"连接S7模拟器失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            try:
                disconnect_msg = {"type": "disconnect", "timestamp": time.time()}
                self._send_message(disconnect_msg)
            except:
                pass
            
            self.socket.close()
            self.socket = None
            self.connected = False
    
    def db_read(self, db_number: int, start: int, size: int) -> bytes:
        """读取数据块"""
        if not self.connected:
            raise Exception("未连接到PLC")
        
        read_msg = {
            "type": "db_read",
            "db_number": db_number,
            "start": start,
            "size": size,
            "timestamp": time.time()
        }
        
        self._send_message(read_msg)
        response = self._receive_message()
        
        if response and response.get("status") == "success":
            # 将十六进制字符串转换为bytes
            hex_data = response.get("data", "")
            return bytes.fromhex(hex_data)
        else:
            error_msg = response.get("error", "读取失败") if response else "通信错误"
            raise Exception(f"读取DB{db_number}失败: {error_msg}")
    
    def db_write(self, db_number: int, start: int, data: bytes):
        """写入数据块"""
        if not self.connected:
            raise Exception("未连接到PLC")
        
        write_msg = {
            "type": "db_write",
            "db_number": db_number,
            "start": start,
            "data": data.hex(),  # 转换为十六进制字符串
            "timestamp": time.time()
        }
        
        self._send_message(write_msg)
        response = self._receive_message()
        
        if not response or response.get("status") != "success":
            error_msg = response.get("error", "写入失败") if response else "通信错误"
            raise Exception(f"写入DB{db_number}失败: {error_msg}")
    
    def read_int16(self, db_number: int, address: int) -> int:
        """读取16位整数"""
        data = self.db_read(db_number, address, 2)
        return struct.unpack('>h', data)[0]
    
    def write_int16(self, db_number: int, address: int, value: int):
        """写入16位整数"""
        data = struct.pack('>h', value)
        self.db_write(db_number, address, data)
    
    def read_int32(self, db_number: int, address: int) -> int:
        """读取32位整数"""
        data = self.db_read(db_number, address, 4)
        return struct.unpack('>i', data)[0]
    
    def write_int32(self, db_number: int, address: int, value: int):
        """写入32位整数"""
        data = struct.pack('>i', value)
        self.db_write(db_number, address, data)
    
    def get_machine_status(self) -> str:
        """获取机床状态"""
        status_code = self.read_int16(9044, 0)
        status_map = {0: "idle", 1: "running", 2: "completed", 3: "error"}
        return status_map.get(status_code, "unknown")
    
    def get_current_layer(self) -> int:
        """获取当前层号"""
        return self.read_int16(9044, 2)
    
    def get_processing_lock(self) -> bool:
        """获取处理锁状态"""
        return self.read_int16(9044, 6) == 1
    
    def set_processing_lock(self, locked: bool):
        """设置处理锁"""
        self.write_int16(9044, 6, 1 if locked else 0)
    
    def get_offset_data_info(self) -> Dict[str, int]:
        """获取偏移数据信息"""
        return {
            "offset_count": self.read_int16(9044, 8),
            "current_batch": self.read_int16(9044, 10),
            "total_batches": self.read_int16(9044, 12),
            "data_ready": self.read_int16(9044, 14)
        }
    
    def write_offset_batch(self, batch_data: list, batch_number: int, total_batches: int):
        """写入偏移数据批次"""
        # 设置批次信息
        if batch_number == 1:
            # 清零偏移点数，将在每批次累加
            self.write_int16(9044, 8, 0)  # 重置总点数
            self.write_int16(9044, 14, 0)  # 重置数据就绪状态
        
        # 累加偏移点数
        current_total = self.read_int16(9044, 8)
        new_total = current_total + len(batch_data)
        self.write_int16(9044, 8, new_total)  # 更新总点数
        
        self.write_int16(9044, 10, batch_number)     # 当前批次
        self.write_int16(9044, 12, total_batches)    # 总批次数
        
        # 选择数据块 (9045, 9046, 9047轮换使用)
        db_numbers = [9045, 9046, 9047]
        db_number = db_numbers[(batch_number - 1) % len(db_numbers)]
        
        print(f"[S7模拟器] 写入批次 {batch_number}/{total_batches} 到 DB{db_number}: {len(batch_data)} 个点")
        
        # 写入偏移数据
        for i, (dx, dy) in enumerate(batch_data):
            if i >= 128:  # 每个数据块最多128个点
                break
            
            addr = i * 4
            # 转换为微米并限制范围
            dx_um = max(-32767, min(32767, int(dx * 1000)))  # mm转微米
            dy_um = max(-32767, min(32767, int(dy * 1000)))
            
            self.write_int16(db_number, addr, dx_um)
            self.write_int16(db_number, addr + 2, dy_um)
            
            # 调试输出前几个点的数据
            if i < 3:
                print(f"  点{i}: dx={dx:.3f}mm({dx_um}μm), dy={dy:.3f}mm({dy_um}μm)")
        
        # 最后一批次设置数据就绪
        if batch_number == total_batches:
            self.write_int16(9044, 14, 1)  # 数据就绪
            print(f"[S7模拟器] 所有批次传输完成，数据就绪")
        
        return True
    
    def _send_message(self, message: dict):
        """发送消息"""
        json_data = json.dumps(message).encode('utf-8')
        length = struct.pack('>I', len(json_data))
        self.socket.sendall(length + json_data)
    
    def _receive_message(self) -> Optional[dict]:
        """接收消息"""
        try:
            # 接收长度
            length_data = self._recv_exact(4)
            if not length_data:
                return None
            
            length = struct.unpack('>I', length_data)[0]
            
            # 接收JSON数据
            json_data = self._recv_exact(length)
            if not json_data:
                return None
            
            return json.loads(json_data.decode('utf-8'))
            
        except Exception as e:
            print(f"接收消息失败: {e}")
            return None
    
    def _recv_exact(self, size: int) -> bytes:
        """精确接收指定大小的数据"""
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                break
            data += chunk
        return data