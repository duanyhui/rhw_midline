# -*- coding: utf-8 -*-
"""
PLC数据块结构定义
根据西门子S7协议定义数据块结构，用于多层加工纠偏系统的PLC通信
"""
import struct
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import IntEnum


class MachineStatus(IntEnum):
    """机床状态枚举"""
    IDLE = 0          # 空闲
    PROCESSING = 1    # 加工中
    WAITING = 2       # 等待纠偏数据
    ERROR = 3         # 错误状态
    COMPLETED = 4     # 全部完成


class ProgramStatus(IntEnum):
    """程序状态枚举"""
    DISCONNECTED = 0  # 未连接
    CONNECTED = 1     # 已连接
    PROCESSING = 2    # 处理中
    COMPLETED = 3     # 完成
    ERROR = 4         # 错误


class DataLockStatus(IntEnum):
    """数据锁状态枚举"""
    UNLOCKED = 0      # 可读写
    LOCKED = 1        # 锁定，禁止写入


@dataclass
class OffsetPoint:
    """偏移点数据结构"""
    dx_mm: float      # X方向偏移量(毫米)
    dy_mm: float      # Y方向偏移量(毫米)
    
    def to_plc_data(self, scale_factor: int = 1000) -> Tuple[int, int]:
        """转换为PLC数据格式（放大为整数）"""
        dx_int = int(self.dx_mm * scale_factor)
        dy_int = int(self.dy_mm * scale_factor)
        # 限制在16位整数范围内
        dx_int = max(-32767, min(32767, dx_int))
        dy_int = max(-32767, min(32767, dy_int))
        return dx_int, dy_int
    
    @classmethod
    def from_plc_data(cls, dx_int: int, dy_int: int, scale_factor: int = 1000) -> 'OffsetPoint':
        """从PLC数据格式创建偏移点"""
        dx_mm = dx_int / scale_factor
        dy_mm = dy_int / scale_factor
        return cls(dx_mm=dx_mm, dy_mm=dy_mm)


class PLCDataBlocks:
    """PLC数据块定义"""
    
    # 数据块编号
    DB_CONTROL = 9044    # 控制和状态信息数据块
    DB_OFFSET_1 = 9045   # 偏移数据块1
    DB_OFFSET_2 = 9046   # 偏移数据块2
    DB_OFFSET_3 = 9047   # 偏移数据块3
    
    # 数据块大小
    DB_SIZE = 512        # 每个数据块大小（字节）
    
    # 控制数据块偏移地址定义 (DB9044)
    class ControlAddresses:
        CURRENT_LAYER = 0           # INT: 当前层号 (2字节)
        MACHINE_STATUS = 2          # INT: 机床状态 (2字节)
        PROGRAM_STATUS = 4          # INT: 程序状态 (2字节)
        TOTAL_POINTS = 6            # INT: 总偏移点数量 (2字节)
        CURRENT_BATCH = 8           # INT: 当前批次索引 (2字节)
        TOTAL_BATCHES = 10          # INT: 总批次数量 (2字节)
        DATA_LOCK = 12              # INT: 数据锁状态 (2字节)
        PROCESS_DELAY_MS = 14       # INT: 处理延迟时间(毫秒) (2字节)
        SCALE_FACTOR = 16           # INT: 数据缩放因子 (2字节)
        LAYER_TYPE = 18             # INT: 层类型(0=标定, 1=纠偏) (2字节)
        ERROR_CODE = 20             # INT: 错误代码 (2字节)
        TIMESTAMP_HIGH = 22         # INT: 时间戳高位 (2字节)
        TIMESTAMP_LOW = 24          # INT: 时间戳低位 (2字节)
        HEARTBEAT = 26              # INT: 心跳计数器 (2字节)
        RESERVED = 28               # 预留空间 (484字节)
    
    # 偏移数据块结构
    POINTS_PER_BLOCK = 128  # 每个数据块存储的偏移点数量 (512字节 / 4字节每点 = 128点)
    BYTES_PER_POINT = 4     # 每个偏移点占用字节数 (dx + dy, 每个2字节)
    
    # 数据缩放因子（浮点数转整数的放大倍数）
    DEFAULT_SCALE_FACTOR = 1000  # 1mm = 1000


class ControlBlockManager:
    """控制数据块管理器 (DB9044)"""
    
    def __init__(self, scale_factor: int = PLCDataBlocks.DEFAULT_SCALE_FACTOR):
        self.scale_factor = scale_factor
        self.data = bytearray(PLCDataBlocks.DB_SIZE)
        self._init_default_values()
    
    def _init_default_values(self):
        """初始化默认值"""
        self.set_current_layer(0)
        self.set_machine_status(MachineStatus.IDLE)
        self.set_program_status(ProgramStatus.DISCONNECTED)
        self.set_scale_factor(self.scale_factor)
        self.set_data_lock(DataLockStatus.UNLOCKED)
        self.set_process_delay_ms(2000)  # 默认2秒延迟
        self.update_timestamp()
    
    def set_current_layer(self, layer: int):
        """设置当前层号"""
        self._write_int16(PLCDataBlocks.ControlAddresses.CURRENT_LAYER, layer)
    
    def get_current_layer(self) -> int:
        """获取当前层号"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.CURRENT_LAYER)
    
    def set_machine_status(self, status: MachineStatus):
        """设置机床状态"""
        self._write_int16(PLCDataBlocks.ControlAddresses.MACHINE_STATUS, int(status))
    
    def get_machine_status(self) -> MachineStatus:
        """获取机床状态"""
        status_val = self._read_int16(PLCDataBlocks.ControlAddresses.MACHINE_STATUS)
        return MachineStatus(status_val) if status_val in MachineStatus.__members__.values() else MachineStatus.ERROR
    
    def set_program_status(self, status: ProgramStatus):
        """设置程序状态"""
        self._write_int16(PLCDataBlocks.ControlAddresses.PROGRAM_STATUS, int(status))
    
    def get_program_status(self) -> ProgramStatus:
        """获取程序状态"""
        status_val = self._read_int16(PLCDataBlocks.ControlAddresses.PROGRAM_STATUS)
        return ProgramStatus(status_val) if status_val in ProgramStatus.__members__.values() else ProgramStatus.ERROR
    
    def set_total_points(self, count: int):
        """设置总偏移点数量"""
        self._write_int16(PLCDataBlocks.ControlAddresses.TOTAL_POINTS, count)
    
    def get_total_points(self) -> int:
        """获取总偏移点数量"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.TOTAL_POINTS)
    
    def set_current_batch(self, batch: int):
        """设置当前批次索引"""
        self._write_int16(PLCDataBlocks.ControlAddresses.CURRENT_BATCH, batch)
    
    def get_current_batch(self) -> int:
        """获取当前批次索引"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.CURRENT_BATCH)
    
    def set_total_batches(self, count: int):
        """设置总批次数量"""
        self._write_int16(PLCDataBlocks.ControlAddresses.TOTAL_BATCHES, count)
    
    def get_total_batches(self) -> int:
        """获取总批次数量"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.TOTAL_BATCHES)
    
    def set_data_lock(self, status: DataLockStatus):
        """设置数据锁状态"""
        self._write_int16(PLCDataBlocks.ControlAddresses.DATA_LOCK, int(status))
    
    def get_data_lock(self) -> DataLockStatus:
        """获取数据锁状态"""
        lock_val = self._read_int16(PLCDataBlocks.ControlAddresses.DATA_LOCK)
        return DataLockStatus(lock_val) if lock_val in DataLockStatus.__members__.values() else DataLockStatus.UNLOCKED
    
    def set_process_delay_ms(self, delay_ms: int):
        """设置处理延迟时间（毫秒）"""
        self._write_int16(PLCDataBlocks.ControlAddresses.PROCESS_DELAY_MS, delay_ms)
    
    def get_process_delay_ms(self) -> int:
        """获取处理延迟时间（毫秒）"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.PROCESS_DELAY_MS)
    
    def set_scale_factor(self, factor: int):
        """设置数据缩放因子"""
        self._write_int16(PLCDataBlocks.ControlAddresses.SCALE_FACTOR, factor)
    
    def get_scale_factor(self) -> int:
        """获取数据缩放因子"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.SCALE_FACTOR)
    
    def set_layer_type(self, layer_type: int):
        """设置层类型 (0=标定, 1=纠偏)"""
        self._write_int16(PLCDataBlocks.ControlAddresses.LAYER_TYPE, layer_type)
    
    def get_layer_type(self) -> int:
        """获取层类型"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.LAYER_TYPE)
    
    def set_error_code(self, error_code: int):
        """设置错误代码"""
        self._write_int16(PLCDataBlocks.ControlAddresses.ERROR_CODE, error_code)
    
    def get_error_code(self) -> int:
        """获取错误代码"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.ERROR_CODE)
    
    def update_timestamp(self):
        """更新时间戳"""
        timestamp = int(time.time())
        timestamp_high = (timestamp >> 16) & 0xFFFF
        timestamp_low = timestamp & 0xFFFF
        self._write_int16(PLCDataBlocks.ControlAddresses.TIMESTAMP_HIGH, timestamp_high)
        self._write_int16(PLCDataBlocks.ControlAddresses.TIMESTAMP_LOW, timestamp_low)
    
    def get_timestamp(self) -> int:
        """获取时间戳"""
        timestamp_high = self._read_int16(PLCDataBlocks.ControlAddresses.TIMESTAMP_HIGH)
        timestamp_low = self._read_int16(PLCDataBlocks.ControlAddresses.TIMESTAMP_LOW)
        return (timestamp_high << 16) | timestamp_low
    
    def increment_heartbeat(self):
        """递增心跳计数器"""
        current_heartbeat = self._read_int16(PLCDataBlocks.ControlAddresses.HEARTBEAT)
        new_heartbeat = (current_heartbeat + 1) % 65536  # 16位整数循环
        self._write_int16(PLCDataBlocks.ControlAddresses.HEARTBEAT, new_heartbeat)
    
    def get_heartbeat(self) -> int:
        """获取心跳计数器"""
        return self._read_int16(PLCDataBlocks.ControlAddresses.HEARTBEAT)
    
    def _write_int16(self, offset: int, value: int):
        """写入16位整数（大端序）"""
        struct.pack_into('>H', self.data, offset, value & 0xFFFF)
    
    def _read_int16(self, offset: int) -> int:
        """读取16位整数（大端序）"""
        return struct.unpack_from('>H', self.data, offset)[0]
    
    def get_data(self) -> bytes:
        """获取完整数据块内容"""
        return bytes(self.data)
    
    def set_data(self, data: bytes):
        """设置完整数据块内容"""
        if len(data) != PLCDataBlocks.DB_SIZE:
            raise ValueError(f"数据长度必须为 {PLCDataBlocks.DB_SIZE} 字节")
        self.data = bytearray(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于调试和显示"""
        return {
            "current_layer": self.get_current_layer(),
            "machine_status": self.get_machine_status().name,
            "program_status": self.get_program_status().name,
            "total_points": self.get_total_points(),
            "current_batch": self.get_current_batch(),
            "total_batches": self.get_total_batches(),
            "data_lock": self.get_data_lock().name,
            "process_delay_ms": self.get_process_delay_ms(),
            "scale_factor": self.get_scale_factor(),
            "layer_type": "calibration" if self.get_layer_type() == 0 else "correction",
            "error_code": self.get_error_code(),
            "timestamp": self.get_timestamp(),
            "heartbeat": self.get_heartbeat()
        }


class OffsetBlockManager:
    """偏移数据块管理器 (DB9045, DB9046, DB9047)"""
    
    def __init__(self, db_number: int, scale_factor: int = PLCDataBlocks.DEFAULT_SCALE_FACTOR):
        if db_number not in [PLCDataBlocks.DB_OFFSET_1, PLCDataBlocks.DB_OFFSET_2, PLCDataBlocks.DB_OFFSET_3]:
            raise ValueError(f"无效的偏移数据块编号: {db_number}")
        
        self.db_number = db_number
        self.scale_factor = scale_factor
        self.data = bytearray(PLCDataBlocks.DB_SIZE)
        self.max_points = PLCDataBlocks.POINTS_PER_BLOCK
    
    def set_offset_points(self, points: List[OffsetPoint], start_index: int = 0):
        """设置偏移点数据"""
        if start_index + len(points) > self.max_points:
            raise ValueError(f"偏移点数量超出限制: {start_index + len(points)} > {self.max_points}")
        
        for i, point in enumerate(points):
            offset = (start_index + i) * PLCDataBlocks.BYTES_PER_POINT
            dx_int, dy_int = point.to_plc_data(self.scale_factor)
            
            # 写入dx (2字节, 大端序, 有符号)
            struct.pack_into('>h', self.data, offset, dx_int)
            # 写入dy (2字节, 大端序, 有符号)
            struct.pack_into('>h', self.data, offset + 2, dy_int)
    
    def get_offset_points(self, count: Optional[int] = None, start_index: int = 0) -> List[OffsetPoint]:
        """获取偏移点数据"""
        if count is None:
            count = self.max_points - start_index
        
        if start_index + count > self.max_points:
            count = self.max_points - start_index
        
        points = []
        for i in range(count):
            offset = (start_index + i) * PLCDataBlocks.BYTES_PER_POINT
            
            # 读取dx (2字节, 大端序, 有符号)
            dx_int = struct.unpack_from('>h', self.data, offset)[0]
            # 读取dy (2字节, 大端序, 有符号)
            dy_int = struct.unpack_from('>h', self.data, offset + 2)[0]
            
            point = OffsetPoint.from_plc_data(dx_int, dy_int, self.scale_factor)
            points.append(point)
        
        return points
    
    def clear_data(self):
        """清空数据块"""
        self.data = bytearray(PLCDataBlocks.DB_SIZE)
    
    def get_data(self) -> bytes:
        """获取完整数据块内容"""
        return bytes(self.data)
    
    def set_data(self, data: bytes):
        """设置完整数据块内容"""
        if len(data) != PLCDataBlocks.DB_SIZE:
            raise ValueError(f"数据长度必须为 {PLCDataBlocks.DB_SIZE} 字节")
        self.data = bytearray(data)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取数据块摘要信息"""
        # 统计非零偏移点数量
        points = self.get_offset_points()
        non_zero_count = sum(1 for p in points if abs(p.dx_mm) > 0.001 or abs(p.dy_mm) > 0.001)
        
        # 计算偏移量范围
        if non_zero_count > 0:
            dx_values = [p.dx_mm for p in points if abs(p.dx_mm) > 0.001 or abs(p.dy_mm) > 0.001]
            dy_values = [p.dy_mm for p in points if abs(p.dx_mm) > 0.001 or abs(p.dy_mm) > 0.001]
            
            dx_range = (min(dx_values), max(dx_values)) if dx_values else (0, 0)
            dy_range = (min(dy_values), max(dy_values)) if dy_values else (0, 0)
        else:
            dx_range = (0, 0)
            dy_range = (0, 0)
        
        return {
            "db_number": self.db_number,
            "total_points": self.max_points,
            "non_zero_points": non_zero_count,
            "dx_range_mm": dx_range,
            "dy_range_mm": dy_range,
            "scale_factor": self.scale_factor
        }


class PLCDataManager:
    """PLC数据管理器，统一管理所有数据块"""
    
    def __init__(self, scale_factor: int = PLCDataBlocks.DEFAULT_SCALE_FACTOR):
        self.scale_factor = scale_factor
        self.control_block = ControlBlockManager(scale_factor)
        self.offset_blocks = {
            PLCDataBlocks.DB_OFFSET_1: OffsetBlockManager(PLCDataBlocks.DB_OFFSET_1, scale_factor),
            PLCDataBlocks.DB_OFFSET_2: OffsetBlockManager(PLCDataBlocks.DB_OFFSET_2, scale_factor),
            PLCDataBlocks.DB_OFFSET_3: OffsetBlockManager(PLCDataBlocks.DB_OFFSET_3, scale_factor)
        }
        self.max_points_per_batch = PLCDataBlocks.POINTS_PER_BLOCK * len(self.offset_blocks)  # 384个点
    
    def calculate_batches_needed(self, total_points: int) -> int:
        """计算需要的批次数量"""
        return (total_points + self.max_points_per_batch - 1) // self.max_points_per_batch
    
    def set_offset_batch(self, points: List[OffsetPoint], batch_index: int) -> bool:
        """设置一个批次的偏移数据"""
        if not points:
            return False
        
        total_batches = self.calculate_batches_needed(len(points))
        if batch_index >= total_batches:
            return False
        
        # 清空所有偏移数据块
        for block in self.offset_blocks.values():
            block.clear_data()
        
        # 计算当前批次的偏移点范围
        start_idx = batch_index * self.max_points_per_batch
        end_idx = min(start_idx + self.max_points_per_batch, len(points))
        batch_points = points[start_idx:end_idx]
        
        # 分配到各个数据块
        points_written = 0
        for db_num, block in self.offset_blocks.items():
            if points_written >= len(batch_points):
                break
            
            # 计算当前块要写入的点数
            remaining_points = len(batch_points) - points_written
            points_for_this_block = min(PLCDataBlocks.POINTS_PER_BLOCK, remaining_points)
            
            if points_for_this_block > 0:
                block_points = batch_points[points_written:points_written + points_for_this_block]
                block.set_offset_points(block_points)
                points_written += points_for_this_block
        
        # 更新控制块信息
        self.control_block.set_total_points(len(points))
        self.control_block.set_current_batch(batch_index)
        self.control_block.set_total_batches(total_batches)
        
        return True
    
    def get_offset_batch(self, batch_index: int) -> List[OffsetPoint]:
        """获取一个批次的偏移数据"""
        all_points = []
        
        for db_num in [PLCDataBlocks.DB_OFFSET_1, PLCDataBlocks.DB_OFFSET_2, PLCDataBlocks.DB_OFFSET_3]:
            block = self.offset_blocks[db_num]
            points = block.get_offset_points()
            all_points.extend(points)
        
        return all_points
    
    def get_all_data_blocks(self) -> Dict[int, bytes]:
        """获取所有数据块的完整数据"""
        data_blocks = {
            PLCDataBlocks.DB_CONTROL: self.control_block.get_data()
        }
        
        for db_num, block in self.offset_blocks.items():
            data_blocks[db_num] = block.get_data()
        
        return data_blocks
    
    def set_data_block(self, db_num: int, data: bytes):
        """设置指定数据块的数据"""
        if db_num == PLCDataBlocks.DB_CONTROL:
            self.control_block.set_data(data)
        elif db_num in self.offset_blocks:
            self.offset_blocks[db_num].set_data(data)
        else:
            raise ValueError(f"未知的数据块编号: {db_num}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        control_info = self.control_block.to_dict()
        
        offset_summaries = {}
        for db_num, block in self.offset_blocks.items():
            offset_summaries[f"DB{db_num}"] = block.get_summary()
        
        return {
            "control": control_info,
            "offsets": offset_summaries,
            "system": {
                "max_points_per_batch": self.max_points_per_batch,
                "scale_factor": self.scale_factor,
                "total_data_blocks": len(self.offset_blocks) + 1
            }
        }


# 错误代码定义
class PLCErrorCodes(IntEnum):
    """PLC错误代码定义"""
    NO_ERROR = 0
    CONNECTION_ERROR = 1
    DATA_VALIDATION_ERROR = 2
    OFFSET_TOO_LARGE = 3
    BATCH_INDEX_ERROR = 4
    TIMEOUT_ERROR = 5
    UNKNOWN_ERROR = 99


if __name__ == "__main__":
    # 测试代码
    print("PLC数据结构测试...")
    
    # 创建数据管理器
    manager = PLCDataManager()
    
    # 创建测试偏移点
    test_points = [
        OffsetPoint(dx_mm=1.5, dy_mm=-2.3),
        OffsetPoint(dx_mm=0.8, dy_mm=1.2),
        OffsetPoint(dx_mm=-1.1, dy_mm=0.5)
    ]
    
    # 设置控制信息
    manager.control_block.set_current_layer(2)
    manager.control_block.set_machine_status(MachineStatus.WAITING)
    manager.control_block.set_program_status(ProgramStatus.PROCESSING)
    
    # 设置偏移数据
    manager.set_offset_batch(test_points, 0)
    
    # 显示系统状态
    status = manager.get_system_status()
    print("系统状态:")
    print(f"  控制信息: {status['control']}")
    print(f"  偏移数据: {status['offsets']}")
    
    print("测试完成!")