# -*- coding: utf-8 -*-
# 偏移量发布模块
# 提供多种输出形式：
#   - Stdout JSON 行：便于与上位机/PLC对接
#   - CSV 追加：记录历史
#   - TCP/UDP Socket：向指定主机端口推送
#   - 串口（可选）：通过 RS-232/USB-Serial 输出（需 pyserial）
#   - G-code 片段生成（FANUC风格示例：G91 U.. W..），仅作演示，需谨慎使用
#
# 注意：实际车床品牌/控制系统的接口差异较大。请在仿真环境反复验证后再上机。
import json, socket, time, csv, sys
from typing import Optional

try:
    import serial
except Exception:
    serial = None

class OffsetPublisher:
    def __init__(self,
                 csv_path: Optional[str]=None,
                 tcp_addr: Optional[tuple]=None,
                 udp_addr: Optional[tuple]=None,
                 serial_port: Optional[str]=None,
                 serial_baud: int=115200,
                 gcode_mode: bool=False):
        self.csv_path = csv_path
        self.tcp_addr = tcp_addr
        self.udp_addr = udp_addr
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.gcode_mode = gcode_mode

        self.tcp_sock = None
        self.udp_sock = None
        self.ser = None

        if self.tcp_addr:
            self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_sock.settimeout(1.0)
            try:
                self.tcp_sock.connect(self.tcp_addr)
            except Exception as e:
                print(f"[Publisher] TCP 连接失败: {e}")
                self.tcp_sock = None

        if self.udp_addr:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if self.serial_port:
            if serial is None:
                print("[Publisher] 未安装 pyserial，无法打开串口。")
            else:
                try:
                    self.ser = serial.Serial(self.serial_port, self.serial_baud, timeout=0.1)
                except Exception as e:
                    print(f"[Publisher] 串口打开失败: {e}")
                    self.ser = None

        # CSV 标题
        if self.csv_path:
            try:
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['timestamp', 'dx_mm', 'dy_mm', 'match', 'note'])
            except Exception as e:
                print(f"[Publisher] 打开CSV失败: {e}")

    def _make_payload(self, dx_mm: float, dy_mm: float, match: float):
        data = {
            'ts': time.time(),
            'dx_mm': float(dx_mm),
            'dy_mm': float(dy_mm),
            'match': float(match)
        }
        if self.gcode_mode:
            # 仅示例：FANUC 车床常以 U 表示 X 直径方向增量，W 表示 Z 方向增量（需确认机床配置！）
            # 采用 G91（增量）+ 极小步长，真实项目应使用刀补/工件坐标系偏置更安全。
            data['gcode'] = f"G91 U{dx_mm:.003f} W{dy_mm:.003f} ; 由视觉纠偏生成"
        return data

    def publish(self, dx_mm: float, dy_mm: float, match: float, note: str=''):
        payload = self._make_payload(dx_mm, dy_mm, match)
        line = json.dumps(payload, ensure_ascii=False)

        # 1) 标准输出
        print(line); sys.stdout.flush()

        # 2) CSV
        if self.csv_path:
            try:
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow([payload['ts'], dx_mm, dy_mm, match, note])
            except Exception as e:
                print(f"[Publisher] 写入CSV失败: {e}")

        # 3) TCP
        if self.tcp_sock:
            try:
                self.tcp_sock.sendall((line + "\n").encode('utf-8'))
            except Exception as e:
                print(f"[Publisher] TCP 发送失败: {e}")

        # 4) UDP
        if self.udp_sock and self.udp_addr:
            try:
                self.udp_sock.sendto((line + "\n").encode('utf-8'), self.udp_addr)
            except Exception as e:
                print(f"[Publisher] UDP 发送失败: {e}")

        # 5) 串口
        if self.ser:
            try:
                self.ser.write((line + "\n").encode('utf-8'))
            except Exception as e:
                print(f"[Publisher] 串口发送失败: {e}")

    def close(self):
        try:
            if self.tcp_sock: self.tcp_sock.close()
        except Exception: pass
        try:
            if self.udp_sock: self.udp_sock.close()
        except Exception: pass
        try:
            if self.ser: self.ser.close()
        except Exception: pass
