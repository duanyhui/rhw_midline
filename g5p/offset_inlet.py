# -*- coding: utf-8 -*-
"""
offset_inlet.py
---------------
为 GUI 预留“输入偏差数据的外部接口”。提供两种方式：
1) 文件接口：加载 CSV/JSON 偏差文件；
2) 轻量本地 HTTP 接口：POST JSON 到 http://127.0.0.1:8765/offsets 即可推送偏差。

数据协议（JSON / CSV）
---------------------
JSON Schema（推荐）：
{
  "source": "string 标识",
  "units": "mm",
  "mode": "delta_n" | "delta_xy",
  "apply_mode": "as_is" | "invert",      # 可选；as_is 表示直接应用；invert 表示取反再应用
  "s": [0.0, 1.2, ...],                  # 可选：与 delta 同长度的弧长（mm），若省略则按序号均匀插值
  "delta_n": [..],                       # 当 mode=delta_n 时必需：法向位移（mm）
  "delta_xy": [[dx,dy], ...]             # 当 mode=delta_xy 时必需：在 XY 平面上的位移（mm）
}

CSV（兼容你现有的 offset_table.csv）：
- 列名：s_mm,delta_n_mm,dx_mm,dy_mm
- 也兼容只有 s_mm 与 delta_n_mm 的两列形式。

注意：该模块**不依赖**核心算法文件，纯工具。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Tuple
import json
import threading
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
import csv
import time


# -------------------- 数据载体 --------------------
@dataclass
class OffsetPacket:
    mode: str                          # 'delta_n' | 'delta_xy'
    s: Optional[np.ndarray]            # 弧长（mm），可以为 None
    delta_n: Optional[np.ndarray]      # (N,)  法向位移（mm）
    delta_xy: Optional[np.ndarray]     # (N,2)  XY 位移（mm）
    meta: Dict[str, Any]               # 附加信息（source, timestamp, apply_mode 等）

    def is_valid(self) -> bool:
        if self.mode == 'delta_n' and self.delta_n is not None and len(self.delta_n) > 1:
            return True
        if self.mode == 'delta_xy' and self.delta_xy is not None and len(self.delta_xy) > 1:
            return True
        return False


# -------------------- 文件接口 --------------------
def load_offsets_from_csv(path: str | Path) -> OffsetPacket:
    p = Path(path)
    s_list, dn_list, dx_list, dy_list = [], [], [], []
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        rd = csv.DictReader(f)
        for row in rd:
            s = row.get('s_mm') or row.get('s') or row.get('S')
            dn = row.get('delta_n_mm') or row.get('delta_n') or row.get('dn')
            dx = row.get('dx_mm') or row.get('dx') or row.get('DX')
            dy = row.get('dy_mm') or row.get('dy') or row.get('DY')
            if s is not None: s_list.append(float(s))
            if dn is not None and dn != '': dn_list.append(float(dn))
            if (dx is not None and dy is not None and dx != '' and dy != ''):
                dx_list.append(float(dx)); dy_list.append(float(dy))
    s = np.asarray(s_list, float) if s_list else None
    dn = np.asarray(dn_list, float) if dn_list else None
    dxy = None
    if dx_list and dy_list and len(dx_list) == len(dy_list):
        dxy = np.stack([dx_list, dy_list], axis=1).astype(float)
    mode = 'delta_xy' if dxy is not None else 'delta_n'
    return OffsetPacket(mode=mode, s=s, delta_n=dn, delta_xy=dxy,
                        meta=dict(source=str(p), timestamp=time.time(), apply_mode='as_is'))

def load_offsets_from_json(path: str | Path) -> OffsetPacket:
    p = Path(path)
    data = json.loads(p.read_text(encoding='utf-8'))
    mode = str(data.get('mode','delta_n')).lower()
    s = np.asarray(data['s'], float) if 's' in data and data['s'] is not None else None
    delta_n = np.asarray(data['delta_n'], float) if 'delta_n' in data and data['delta_n'] is not None else None
    delta_xy = None
    if 'delta_xy' in data and data['delta_xy'] is not None:
        arr = np.asarray(data['delta_xy'], float)
        if arr.ndim == 2 and arr.shape[1] == 2: delta_xy = arr
    meta = {k:v for k,v in data.items() if k not in ('s','delta_n','delta_xy','mode')}
    meta.setdefault('source', str(p)); meta.setdefault('timestamp', time.time())
    meta.setdefault('apply_mode', data.get('apply_mode','as_is'))
    return OffsetPacket(mode=mode, s=s, delta_n=delta_n, delta_xy=delta_xy, meta=meta)


# -------------------- HTTP 接口（轻量内置） --------------------
class _HttpHandler(BaseHTTPRequestHandler):
    cb: Optional[Callable[[OffsetPacket], None]] = None
    def do_POST(self):
        if self.path != '/offsets':
            self.send_response(404); self.end_headers(); self.wfile.write(b'Not Found'); return
        length = int(self.headers.get('Content-Length', '0') or '0')
        body = self.rfile.read(length) if length > 0 else b''
        try:
            data = json.loads(body.decode('utf-8'))
            pkt = OffsetPacket(
                mode=str(data.get('mode','delta_n')).lower(),
                s=np.asarray(data['s'], float) if data.get('s') is not None else None,
                delta_n=np.asarray(data['delta_n'], float) if data.get('delta_n') is not None else None,
                delta_xy=np.asarray(data['delta_xy'], float) if data.get('delta_xy') is not None else None,
                meta={k:v for k,v in data.items() if k not in ('s','delta_n','delta_xy','mode')}
            )
            pkt.meta.setdefault('source', 'http:post'); pkt.meta.setdefault('timestamp', time.time())
            pkt.meta.setdefault('apply_mode', data.get('apply_mode','as_is'))
            ok = pkt.is_valid()
            if ok and _HttpHandler.cb is not None:
                _HttpHandler.cb(pkt)
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps({'ok': ok}).encode('utf-8'))
        except Exception as e:
            self.send_response(400); self.end_headers()
            self.wfile.write(('error: ' + str(e)).encode('utf-8'))

    def log_message(self, fmt, *args):
        # 静默日志，避免刷屏
        pass

class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

class HttpOffsetServer:
    """在单独线程里启动一个超轻量 HTTP 服务器，接收外部 POST /offsets JSON。"""
    def __init__(self, host='127.0.0.1', port=8765, callback: Optional[Callable[[OffsetPacket], None]] = None):
        self.host = host; self.port = int(port); self.cb = callback
        self._srv: Optional[_ThreadingHTTPServer] = None
        self._th: Optional[threading.Thread] = None

    def start(self):
        if self._srv is not None: return
        _HttpHandler.cb = self.cb
        self._srv = _ThreadingHTTPServer((self.host, self.port), _HttpHandler)
        self._th = threading.Thread(target=self._srv.serve_forever, name='HttpOffsetServer', daemon=True)
        self._th.start()
        return True

    def stop(self):
        if self._srv is not None:
            try: self._srv.shutdown()
            except Exception: pass
            self._srv.server_close()
            self._srv = None
        self._th = None

    def running(self) -> bool:
        return self._srv is not None
