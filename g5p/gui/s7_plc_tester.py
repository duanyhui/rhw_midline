#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层加工系统 PLC-S7 通信测试程序（单文件）

目的：在不使用相机的情况下，验证 DB9044 (控制块) 与 DB9045/9046/9047 (偏移块)
的读写流程是否符合预期，生成详细的过程日志，便于在上位机/HMI观察数据变化。

依赖：
  - python-snap7 (pip install python-snap7)

最简读写逻辑示例（已在本脚本中使用）：
  import snap7
  client = snap7.client.Client()
  client.connect('192.168.10.100', 0, 2)
  data = client.db_read(1, 0, 3)
  client.db_write(1, 0, b'\x11')

注意：运行前请确保 PLC 中已创建 DB9044、DB9045、DB9046、DB9047，尺寸为 512 字节。
"""

import sys
import time
import struct
import argparse
from typing import List, Tuple

try:
    import snap7  # pip install python-snap7
except Exception as e:
    print("未安装 python-snap7 或加载失败:", e)
    sys.exit(1)


# === 常量（与项目中 plc_data_structures.py 保持一致） ===
DB_CONTROL = 9044
DB_OFFSET_1 = 9045
DB_OFFSET_2 = 9046
DB_OFFSET_3 = 9047
DB_SIZE = 512

# 控制块字段偏移（INT16，大端，无符号写入）
CURRENT_LAYER = 0
MACHINE_STATUS = 2
PROGRAM_STATUS = 4
TOTAL_POINTS = 6
CURRENT_BATCH = 8
TOTAL_BATCHES = 10
DATA_LOCK = 12
PROCESS_DELAY_MS = 14
SCALE_FACTOR = 16
LAYER_TYPE = 18
ERROR_CODE = 20
TIMESTAMP_HIGH = 22
TIMESTAMP_LOW = 24
HEARTBEAT = 26

# 语义枚举（与项目一致）
MS_IDLE = 0
MS_PROCESSING = 1
MS_WAITING = 2
MS_ERROR = 3
MS_COMPLETED = 4

PS_DISCONNECTED = 0
PS_CONNECTED = 1
PS_PROCESSING = 2
PS_COMPLETED = 3
PS_ERROR = 4

LOCK_UNLOCKED = 0
LOCK_LOCKED = 1

POINTS_PER_BLOCK = 128   # 每个偏移DB最多点数
BYTES_PER_POINT = 4      # dx(2字节) + dy(2字节)，有符号16位，大端
SCALE_DEFAULT = 1000     # 1mm = 1000


# === 终端展示与工具函数 ===
USE_COLOR = True

def _supports_color() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def color(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def c_info(s: str) -> str: return color(s, '36')      # 青色
def c_ok(s: str) -> str: return color(s, '32')        # 绿色
def c_warn(s: str) -> str: return color(s, '33')      # 黄色
def c_err(s: str) -> str: return color(s, '31')       # 红色
def c_step(s: str) -> str: return color(s, '35;1')    # 品红加粗
def c_title(s: str) -> str: return color(s, '34;1')   # 蓝色加粗

# 轻量进度条
def progress(duration_s: float, label: str = "", width: int = 28):
    if duration_s <= 0:
        return
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        ratio = min(1.0, elapsed / duration_s)
        filled = int(width * ratio)
        bar = "#" * filled + "." * (width - filled)
        pct = int(ratio * 100)
        sys.stdout.write(f"\r{c_info(label)} [{bar}] {pct:3d}%")
        sys.stdout.flush()
        if ratio >= 1.0:
            break
        time.sleep(0.05)
    sys.stdout.write("\n")
    sys.stdout.flush()

_step_counter = 0
def step(title: str):
    global _step_counter
    _step_counter += 1
    print(c_step(f"\n==> [步骤 {_step_counter}] {title}"))
def u16_get(data: bytes, offset: int) -> int:
    return struct.unpack_from('>H', data, offset)[0]

def u16_set(buf: bytearray, offset: int, value: int):
    struct.pack_into('>H', buf, offset, value & 0xFFFF)

def i16_set(buf: bytearray, offset: int, value: int):
    struct.pack_into('>h', buf, offset, int(value))

def now_ts_parts() -> Tuple[int, int]:
    t = int(time.time())
    return (t >> 16) & 0xFFFF, t & 0xFFFF

def log(event: str, **kwargs):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    kv = ' '.join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{ts}] {event} {kv}".rstrip())


def read_db(client, db_num: int, size: int = DB_SIZE, start: int = 0) -> bytes:
    return bytes(client.db_read(db_num, start, size))

def write_db(client, db_num: int, data: bytes, start: int = 0):
    client.db_write(db_num, start, data)


# === 控制块（DB9044）读/改/写 ===
def read_control(client) -> bytes:
    return read_db(client, DB_CONTROL, DB_SIZE, 0)

def parse_control(data: bytes) -> dict:
    return {
        'current_layer': u16_get(data, CURRENT_LAYER),
        'machine_status': u16_get(data, MACHINE_STATUS),
        'program_status': u16_get(data, PROGRAM_STATUS),
        'total_points': u16_get(data, TOTAL_POINTS),
        'current_batch': u16_get(data, CURRENT_BATCH),
        'total_batches': u16_get(data, TOTAL_BATCHES),
        'data_lock': u16_get(data, DATA_LOCK),
        'process_delay_ms': u16_get(data, PROCESS_DELAY_MS),
        'scale_factor': u16_get(data, SCALE_FACTOR),
        'layer_type': u16_get(data, LAYER_TYPE),
        'error_code': u16_get(data, ERROR_CODE),
        'timestamp': (u16_get(data, TIMESTAMP_HIGH) << 16) | u16_get(data, TIMESTAMP_LOW),
        'heartbeat': u16_get(data, HEARTBEAT),
    }

def write_control_fields(client, patch: dict):
    """按需修改控制块字段（只覆盖提供的键），再整体写回。"""
    raw = bytearray(read_control(client))

    def set_field(name: str, val: int):
        if name == 'current_layer':
            u16_set(raw, CURRENT_LAYER, val)
        elif name == 'machine_status':
            u16_set(raw, MACHINE_STATUS, val)
        elif name == 'program_status':
            u16_set(raw, PROGRAM_STATUS, val)
        elif name == 'total_points':
            u16_set(raw, TOTAL_POINTS, val)
        elif name == 'current_batch':
            u16_set(raw, CURRENT_BATCH, val)
        elif name == 'total_batches':
            u16_set(raw, TOTAL_BATCHES, val)
        elif name == 'data_lock':
            u16_set(raw, DATA_LOCK, val)
        elif name == 'process_delay_ms':
            u16_set(raw, PROCESS_DELAY_MS, val)
        elif name == 'scale_factor':
            u16_set(raw, SCALE_FACTOR, val)
        elif name == 'layer_type':
            u16_set(raw, LAYER_TYPE, val)
        elif name == 'error_code':
            u16_set(raw, ERROR_CODE, val)
        elif name == 'timestamp_now' and val:
            hi, lo = now_ts_parts()
            u16_set(raw, TIMESTAMP_HIGH, hi)
            u16_set(raw, TIMESTAMP_LOW, lo)
        elif name == 'heartbeat_inc' and val:
            hb = u16_get(raw, HEARTBEAT)
            u16_set(raw, HEARTBEAT, (hb + 1) & 0xFFFF)

    for k, v in patch.items():
        set_field(k, v)

    write_db(client, DB_CONTROL, bytes(raw), 0)
    return parse_control(bytes(raw))


def ms_name(v: int) -> str:
    return {
        0: '空闲', 1: '加工', 2: '等待', 3: '错误', 4: '完成'
    }.get(v, str(v))

def ps_name(v: int) -> str:
    return {
        0: '未连接', 1: '已连接', 2: '处理中', 3: '已完成', 4: '错误'
    }.get(v, str(v))

def print_control_brief(ctrl: dict):
    line = (
        f"层={ctrl['current_layer']} "
        f"批次={ctrl['current_batch']}/{ctrl['total_batches']} "
        f"状态={ps_name(ctrl['program_status'])}/{ms_name(ctrl['machine_status'])} "
        f"锁={'锁定' if ctrl['data_lock'] else '解锁'} "
        f"心跳={ctrl['heartbeat']} 时间戳={ctrl['timestamp']}"
    )
    print(c_info("控制 ") + line)


def print_plan(ip: str, rack: int, slot: int, layer: int, total_points: int, total_batches: int, delay_ms: int):
    print(c_title("\n=== 测试计划 ==="))
    print(c_info("目标    ") + f"ip={ip} rack={rack} slot={slot}")
    print(c_info("层号    ") + f"{layer} (类型={'纠偏' if layer>1 else '标定'})")
    print(c_info("点数    ") + f"{total_points}  批次数={total_batches}  每批最大≤384")
    print(c_info("时序    ") + f"process_delay_ms={delay_ms}")
    print("- 连接 -> 初始化控制块 -> 心跳×3")
    print("- 每批流程: 锁定 -> 写入偏移 -> 解锁 -> 等待 -> 心跳")
    print("- 完成 -> 错误演示 -> 清除 -> 断开")


# === 偏移块（DB9045/46/47）写入/读取 ===
def build_offset_db(points: List[Tuple[int, int]]) -> bytes:
    """points: [(dx_int, dy_int), ...]，每点为有符号16位整数，放大后值。"""
    raw = bytearray(DB_SIZE)
    max_pts = min(len(points), POINTS_PER_BLOCK)
    for i in range(max_pts):
        off = i * BYTES_PER_POINT
        dx, dy = points[i]
        i16_set(raw, off, dx)
        i16_set(raw, off + 2, dy)
    return bytes(raw)

def write_offset_batch(client, all_points_mm: List[Tuple[float, float]], batch_index: int, scale: int):
    """将 all_points_mm 的一个批次写入三个偏移DB。"""
    max_per_batch = POINTS_PER_BLOCK * 3  # 384
    start = batch_index * max_per_batch
    end = min(start + max_per_batch, len(all_points_mm))
    batch = all_points_mm[start:end]

    # 切分到三个 DB
    part1 = batch[:POINTS_PER_BLOCK]
    part2 = batch[POINTS_PER_BLOCK:POINTS_PER_BLOCK*2]
    part3 = batch[POINTS_PER_BLOCK*2:]

    def to_i(points_f):
        out = []
        for (dx_mm, dy_mm) in points_f:
            dx_i = max(-32767, min(32767, int(round(dx_mm * scale))))
            dy_i = max(-32767, min(32767, int(round(dy_mm * scale))))
            out.append((dx_i, dy_i))
        return out

    db1 = build_offset_db(to_i(part1))
    db2 = build_offset_db(to_i(part2))
    db3 = build_offset_db(to_i(part3))

    write_db(client, DB_OFFSET_1, db1, 0)
    write_db(client, DB_OFFSET_2, db2, 0)
    write_db(client, DB_OFFSET_3, db3, 0)

    log("写入偏移数据", batch_index=batch_index, points_in_batch=len(batch),
        db1_points=len(part1), db2_points=len(part2), db3_points=len(part3))


def read_offset_summary(client) -> dict:
    def summarize(db_num: int) -> dict:
        raw = read_db(client, db_num, DB_SIZE, 0)
        non_zero = 0
        first_pts = []
        for i in range(POINTS_PER_BLOCK):
            off = i * BYTES_PER_POINT
            dx = struct.unpack_from('>h', raw, off)[0]
            dy = struct.unpack_from('>h', raw, off + 2)[0]
            if dx != 0 or dy != 0:
                non_zero += 1
                if len(first_pts) < 6:
                    first_pts.append((dx, dy))
        return {"db": db_num, "non_zero": non_zero, "first_points": first_pts}

    return {
        'DB9045': summarize(DB_OFFSET_1),
        'DB9046': summarize(DB_OFFSET_2),
        'DB9047': summarize(DB_OFFSET_3),
    }


# === 测试流程 ===
def run_test(ip: str, rack: int, slot: int, total_points: int, process_delay_ms: int, layer: int, preview_pts: int):
    client = snap7.client.Client()
    step("连接PLC")
    client.connect(ip, rack, slot)
    if not client.get_connected():
        print(c_err("连接失败"))
        return
    log("已连接", ip=ip, rack=rack, slot=slot)

    max_per_batch = POINTS_PER_BLOCK * 3
    total_batches = (total_points + max_per_batch - 1) // max_per_batch
    print_plan(ip, rack, slot, layer, total_points, total_batches, process_delay_ms)

    # 1) 初始化/连接状态
    step("初始化控制块 (DB9044)")
    scale = SCALE_DEFAULT
    ctrl = write_control_fields(client, {
        'scale_factor': scale,
        'program_status': PS_CONNECTED,
        'machine_status': MS_IDLE,
        'data_lock': LOCK_UNLOCKED,
        'process_delay_ms': process_delay_ms,
        'current_layer': layer,
        'total_points': total_points,
        'current_batch': 0,
        'total_batches': (total_points + (POINTS_PER_BLOCK*3) - 1) // (POINTS_PER_BLOCK*3),
        'layer_type': 1 if layer > 1 else 0,  # 第1层为标定，其余为纠偏
        'error_code': 0,
        'timestamp_now': 1,
    })
    print_control_brief(ctrl)

    # 2) 心跳三次
    step("预热心跳（3次）")
    for i in range(3):
        ctrl = write_control_fields(client, {'heartbeat_inc': 1, 'timestamp_now': 1})
        print_control_brief(ctrl)
        progress(0.6, label=f"心跳 {i+1}/3")

    # 3) 构造测试偏移点（正弦/锯齿混合，便于观察变化）
    #    约 total_points 个点，范围在 ±2.000 mm 内
    points_mm: List[Tuple[float, float]] = []
    import math
    for i in range(total_points):
        dx = 2.0 * math.sin(i / 15.0)
        dy = (i % 40) / 10.0 - 2.0  # -2.0 .. +2.0 的锯齿
        points_mm.append((dx, dy))

    max_per_batch = POINTS_PER_BLOCK * 3
    total_batches = (total_points + max_per_batch - 1) // max_per_batch

    # 4) 批次写入流程
    step("开始分批传输")
    for b in range(total_batches):
        print(c_title(f"\n[批次 {b+1}/{total_batches}]"))
        # 4.1 设置为锁定，写控制信息（PROCESSING）
        ctrl = write_control_fields(client, {
            'data_lock': LOCK_LOCKED,
            'program_status': PS_PROCESSING,
            'machine_status': MS_WAITING,
            'current_batch': b,
            'timestamp_now': 1,
        })
        print(c_ok("[锁定] ") + "已锁定并设置状态 处理/等待")
        print_control_brief(ctrl)

        # 4.2 写偏移数据到三个块
        write_offset_batch(client, points_mm, b, scale)
        offs = read_offset_summary(client)
        for key in ('DB9045', 'DB9046', 'DB9047'):
            info = offs[key]
            preview = info['first_points'][:preview_pts]
            print(c_info(f"[写入] {key} 非零点={info['non_zero']} 预览={preview}"))

        # 4.3 解锁（供PLC侧读取），并保持 WAITING/PROCESSING 状态
        ctrl = write_control_fields(client, {
            'data_lock': LOCK_UNLOCKED,
            'timestamp_now': 1,
        })
        print(c_ok("[解锁] ") + "已解锁，供PLC读取")
        print_control_brief(ctrl)

        # 4.4 等待 process_delay_ms（让上位机/PLC有时间反应和显示）
        progress(max(process_delay_ms, 200) / 1000.0, label="等待PLC读取")

        # 4.5 心跳 + 时间戳
        ctrl = write_control_fields(client, {'heartbeat_inc': 1, 'timestamp_now': 1})
        print(c_info("[心跳] 批次后心跳"))
        print_control_brief(ctrl)

    # 5) 完成状态
    step("标记完成")
    ctrl = write_control_fields(client, {
        'program_status': PS_COMPLETED,
        'machine_status': MS_COMPLETED,
        'error_code': 0,
        'timestamp_now': 1,
    })
    print_control_brief(ctrl)

    # 6) 故障演示（可选）：写入一次错误码再清除
    step("错误演示（设置并清除）")
    ctrl = write_control_fields(client, {
        'program_status': PS_ERROR,
        'machine_status': MS_ERROR,
        'error_code': 3,  # OFFSET_TOO_LARGE 示例
        'timestamp_now': 1,
    })
    print(c_warn("[错误] 设置示例错误码 3"))
    print_control_brief(ctrl)
    progress(1.0, label="保留错误显示")
    ctrl = write_control_fields(client, {
        'program_status': PS_CONNECTED,
        'machine_status': MS_IDLE,
        'error_code': 0,
        'timestamp_now': 1,
    })
    print(c_ok("[错误] 已清除"))
    print_control_brief(ctrl)

    client.disconnect()
    step("断开连接")
    log("已断开")


def main():
    p = argparse.ArgumentParser(description='S7 PLC 多层加工通信单文件测试程序')
    p.add_argument('--ip', default='192.168.10.100', help='PLC IP 地址')
    p.add_argument('--rack', type=int, default=0, help='机架号，默认 0')
    p.add_argument('--slot', type=int, default=2, help='插槽号，默认 2')
    p.add_argument('--layer', type=int, default=2, help='测试层号，默认 2')
    p.add_argument('--points', type=int, default=500, help='总偏移点数量，默认 500')
    p.add_argument('--delay', type=int, default=800, help='每批处理延时 ms，默认 800')
    p.add_argument('--preview', type=int, default=3, help='偏移块预览点数，默认 3')
    p.add_argument('--no-color', action='store_true', help='关闭彩色终端输出')
    args = p.parse_args()

    global USE_COLOR
    USE_COLOR = (not args.no_color) and _supports_color()

    run_test(args.ip, args.rack, args.slot, args.points, args.delay, args.layer, args.preview)


if __name__ == '__main__':
    main()
