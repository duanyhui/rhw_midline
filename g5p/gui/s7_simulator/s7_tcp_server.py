#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7模拟器TCP服务器 - 处理主程序与S7模拟器之间的通信
"""

import socket
import threading
import json
import struct
import time
from typing import Dict, Any, Optional

class S7SimulatorTCPServer:
    """S7模拟器TCP服务器"""
    
    def __init__(self, plc_simulator, host: str = "127.0.0.1", port: int = 8502):
        self.plc_simulator = plc_simulator
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.client_sockets = []
        
    def start(self):
        """启动TCP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            print(f"S7模拟器TCP服务器启动: {self.host}:{self.port}")
            
            # 开始接受连接
            accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
            accept_thread.start()
            
            return True
            
        except Exception as e:
            print(f"启动TCP服务器失败: {e}")
            return False
    
    def stop(self):
        """停止TCP服务器"""
        self.running = False
        
        # 关闭所有客户端连接
        for client_socket in self.client_sockets[:]:
            try:
                client_socket.close()
            except:
                pass
        
        # 关闭服务器socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("S7模拟器TCP服务器已停止")
    
    def _accept_connections(self):
        """接受客户端连接"""
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                print(f"S7客户端连接: {address}")
                
                self.client_sockets.append(client_socket)
                
                # 为每个客户端创建处理线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"接受连接失败: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket, address):
        """处理客户端请求"""
        try:
            while self.running:
                # 接收消息
                message = self._receive_message(client_socket)
                if not message:
                    break
                
                # 处理消息
                response = self._process_message(message)
                
                # 发送响应
                if response:
                    self._send_message(client_socket, response)
                    
        except Exception as e:
            print(f"处理客户端{address}失败: {e}")
        finally:
            # 清理连接
            if client_socket in self.client_sockets:
                self.client_sockets.remove(client_socket)
            
            try:
                client_socket.close()
            except:
                pass
            
            print(f"客户端{address}断开连接")
    
    def _receive_message(self, client_socket: socket.socket) -> Optional[dict]:
        """接收客户端消息"""
        try:
            # 接收长度
            length_data = self._recv_exact(client_socket, 4)
            if not length_data:
                return None
            
            length = struct.unpack('>I', length_data)[0]
            
            # 接收JSON数据
            json_data = self._recv_exact(client_socket, length)
            if not json_data:
                return None
            
            return json.loads(json_data.decode('utf-8'))
            
        except Exception as e:
            print(f"接收消息失败: {e}")
            return None
    
    def _send_message(self, client_socket: socket.socket, message: dict):
        """发送消息给客户端"""
        try:
            json_data = json.dumps(message).encode('utf-8')
            length = struct.pack('>I', len(json_data))
            client_socket.sendall(length + json_data)
        except Exception as e:
            print(f"发送消息失败: {e}")
    
    def _recv_exact(self, client_socket: socket.socket, size: int) -> bytes:
        """精确接收指定大小的数据"""
        data = b''
        while len(data) < size:
            chunk = client_socket.recv(size - len(data))
            if not chunk:
                break
            data += chunk
        return data
    
    def _process_message(self, message: dict) -> Optional[dict]:
        """处理客户端消息"""
        msg_type = message.get("type")
        
        try:
            if msg_type == "connect":
                return self._handle_connect(message)
            elif msg_type == "disconnect":
                return self._handle_disconnect(message)
            elif msg_type == "db_read":
                return self._handle_db_read(message)
            elif msg_type == "db_write":
                return self._handle_db_write(message)
            else:
                return {"status": "error", "error": f"未知消息类型: {msg_type}"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _handle_connect(self, message: dict) -> dict:
        """处理连接请求"""
        ip = message.get("ip", "")
        rack = message.get("rack", 0)
        slot = message.get("slot", 1)
        
        print(f"S7连接请求: IP={ip}, Rack={rack}, Slot={slot}")
        
        # 模拟连接成功
        return {
            "status": "connected",
            "ip": ip,
            "rack": rack,
            "slot": slot,
            "timestamp": time.time()
        }
    
    def _handle_disconnect(self, message: dict) -> dict:
        """处理断开连接请求"""
        print("S7断开连接请求")
        return {"status": "disconnected", "timestamp": time.time()}
    
    def _handle_db_read(self, message: dict) -> dict:
        """处理数据块读取请求"""
        db_number = message.get("db_number")
        start = message.get("start")
        size = message.get("size")
        
        try:
            # 从PLC模拟器读取数据
            data = self.plc_simulator.db_read(db_number, start, size)
            
            return {
                "status": "success",
                "db_number": db_number,
                "start": start,
                "size": size,
                "data": data.hex(),  # 转换为十六进制字符串
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "db_number": db_number,
                "start": start,
                "size": size
            }
    
    def _handle_db_write(self, message: dict) -> dict:
        """处理数据块写入请求"""
        db_number = message.get("db_number")
        start = message.get("start")
        hex_data = message.get("data", "")
        
        try:
            # 将十六进制字符串转换为bytes
            data = bytes.fromhex(hex_data)
            
            # 写入PLC模拟器
            self.plc_simulator.db_write(db_number, start, data)
            
            return {
                "status": "success",
                "db_number": db_number,
                "start": start,
                "size": len(data),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "db_number": db_number,
                "start": start
            }