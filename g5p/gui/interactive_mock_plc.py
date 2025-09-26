# -*- coding: utf-8 -*-
"""
交互式模拟PLC服务器 - 手动命令行控制
支持手动控制每层的加工完成时机，更贴近真实机床操作
"""
import json
import socket
import threading
import time
import sys
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class InteractiveMachineState:
    """交互式机床状态"""
    current_layer: int = 1
    machine_status: str = "idle"  # idle, processing, waiting, error
    correction_received: bool = False
    layer_start_time: Optional[float] = None
    total_layers: int = 10
    
    # 统计信息
    processed_layers: int = 0
    failed_layers: int = 0
    total_processing_time: float = 0.0
    layer_times: Dict[int, float] = field(default_factory=dict)
    
    # 纠偏数据应用状态
    correction_data: Dict[int, Dict] = field(default_factory=dict)  # 每层的纠偏数据
    correction_applied: Dict[int, bool] = field(default_factory=dict)  # 纠偏是否已应用
    original_path: Optional[str] = None  # 原始加工路径
    corrected_path: Optional[str] = None  # 纠偏后路径
    deviation_stats: Dict[int, Dict] = field(default_factory=dict)  # 偏差统计


class InteractiveMockPLCServer:
    """交互式模拟PLC服务器"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 502):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = []
        self.client_threads = []  # 跟踪客户端线程
        self.machine_state = InteractiveMachineState()
        self.lock = threading.Lock()
        self.max_clients = 5  # 最大客户端连接数
        self.command_help = {
            "start": "开始当前层加工",
            "complete": "完成当前层加工",
            "next": "进入下一层",
            "error": "设置错误状态", 
            "reset": "重置到第1层",
            "status": "显示当前状态",
            "stats": "显示统计信息",
            "layers": "设置总层数 <数量>",
            "correction": "显示纠偏数据应用情况",
            "simulate": "模拟机床应用纠偏数据 <层号>",
            "path": "显示加工路径信息",
            "deviation": "显示偏差统计信息",
            "help": "显示帮助信息",
            "quit": "退出服务器"
        }
        
    def start(self):
        """启动服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(self.max_clients)
            self.running = True
            
            print(f"\n交互式模拟PLC服务器启动: {self.host}:{self.port}")
            print(f"最大客户端连接数: {self.max_clients}")
            print("=" * 60)
            self.print_status()
            self.print_help()
            
            # 启动网络服务线程
            network_thread = threading.Thread(target=self._network_server, daemon=True)
            network_thread.start()
            
            # 启动连接清理线程
            cleanup_thread = threading.Thread(target=self._cleanup_connections, daemon=True)
            cleanup_thread.start()
            
            # 主线程处理命令行输入
            self._command_loop()
                        
        except Exception as e:
            print(f"服务器启动失败: {e}")
            
    def _cleanup_connections(self):
        """定期清理死连接"""
        import time
        while self.running:
            try:
                time.sleep(5)  # 每5秒检查一次
                with self.lock:
                    # 清理已结束的线程
                    self.client_threads = [t for t in self.client_threads if t.is_alive()]
                    
                    if len(self.client_threads) > 0:
                        print(f"[维护] 当前活跃连接数: {len(self.client_threads)}")
            except Exception as e:
                print(f"[维护] 连接清理错误: {e}")
                
    def stop(self):
        """停止服务器"""
        print("\n正在停止服务器...")
        self.running = False
        
        # 关闭所有客户端连接
        with self.lock:
            for client in self.clients[:]:
                try:
                    client.close()
                except:
                    pass
            self.clients.clear()
        
        # 关闭服务器Socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        print("服务器已停止")
            
    def _network_server(self):
        """网络服务线程"""
        if not self.socket:
            return
            
        while self.running:
            try:
                self.socket.settimeout(1.0)  # 设置超时，允许定期检查running状态
                client_socket, address = self.socket.accept()
                
                # 检查客户端连接数限制
                with self.lock:
                    if len(self.client_threads) >= self.max_clients:
                        print(f"[网络] 拒绝连接 {address}: 已达到最大连接数 {self.max_clients}")
                        client_socket.close()
                        continue
                
                print(f"\n[网络] 客户端连接: {address} (连接数: {len(self.client_threads) + 1})")
                
                # 添加到客户端列表
                with self.lock:
                    self.clients.append(client_socket)
                
                client_thread = threading.Thread(
                    target=self._handle_client, 
                    args=(client_socket, address),
                    daemon=True
                )
                
                with self.lock:
                    self.client_threads.append(client_thread)
                
                client_thread.start()
                
            except socket.timeout:
                continue  # 超时是正常的，继续检查running状态
            except Exception as e:
                if self.running:
                    print(f"[网络] 接受连接错误: {e}")
                    
    def _handle_client(self, client_socket: socket.socket, address):
        """处理客户端连接"""
        try:
            # 设置Socket超时和选项
            client_socket.settimeout(30.0)  # 30秒超时
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            while self.running:
                # 安全接收数据长度
                length_bytes = self._recv_exact(client_socket, 4)
                if not length_bytes:
                    print(f"[网络] 客户端 {address} 关闭连接")
                    break
                
                # 验证长度字段的合理性
                is_valid, length = self._validate_length_bytes(length_bytes, address)
                
                if not is_valid:
                    if length == -1:  # 边界错位，需要修复
                        print(f"[边界修复] 尝试修复数据包边界...")
                        if not self._recover_packet_boundary(client_socket, length_bytes, address):
                            print(f"[边界修复] 修复失败，断开连接")
                            break
                        continue
                    else:
                        # 长度异常，尝试清空缓冲区
                        print(f"[网络] 客户端 {address} 发送的数据长度异常: {length}")
                        print(f"[调试] 长度字节: {length_bytes.hex()} -> {length}")
                        
                        if not self._clear_socket_buffer(client_socket, address):
                            break
                        continue
                
                # 安全接收命令数据
                command_data = self._recv_exact(client_socket, length)
                if not command_data:
                    print(f"[网络] 客户端 {address} 数据接收不完整")
                    break
                
                # 安全解析JSON
                try:
                    command_str = command_data.decode('utf-8')
                    if not command_str.strip():
                        print(f"[网络] 客户端 {address} 发送空数据")
                        continue
                        
                    command = json.loads(command_str)
                except json.JSONDecodeError as je:
                    print(f"[网络] 客户端 {address} JSON解析错误: {je}")
                    print(f"[调试] 长度: {length}, 实际数据长度: {len(command_data)}")
                    print(f"[调试] 原始数据: {command_data[:100]}...")
                    print(f"[调试] 数据hex: {command_data[:50].hex()}")
                    
                    # 检查是否是数据包边界问题
                    if b'{' not in command_data[:100]:
                        print(f"[恢复] 数据不包含JSON起始字符，可能是边界错位")
                        # 尝试在数据中查找JSON起始位置
                        json_start = command_data.find(b'{')
                        if json_start > 0:
                            print(f"[恢复] 在位置 {json_start} 找到JSON起始")
                            try:
                                fixed_data = command_data[json_start:]
                                command = json.loads(fixed_data.decode('utf-8'))
                                print(f"[恢复] JSON解析成功: {command.get('type', 'unknown')}")
                            except Exception:
                                print(f"[恢复] JSON修复失败")
                                continue
                        else:
                            print(f"[恢复] 未找到JSON数据，跳过此数据包")
                            continue
                    else:
                        # 发送错误响应
                        error_response = {"success": False, "error": f"JSON解析失败: {str(je)}"}
                        self._send_response(client_socket, error_response)
                        continue
                except UnicodeDecodeError as ue:
                    print(f"[网络] 客户端 {address} 编码解析错误: {ue}")
                    continue
                
                # 处理命令
                response = self._process_command(command)
                
                # 安全发送响应
                if not self._send_response(client_socket, response):
                    print(f"[网络] 客户端 {address} 响应发送失败")
                    break
                
        except socket.timeout:
            print(f"[网络] 客户端 {address} 连接超时")
        except socket.error as se:
            print(f"[网络] 客户端 {address} Socket错误: {se}")
        except Exception as e:
            print(f"[网络] 客户端 {address} 处理错误: {e}")
            import traceback
            print(f"[调试] 详细错误信息:\n{traceback.format_exc()}")
        finally:
            # 从客户端列表中移除
            with self.lock:
                if client_socket in self.clients:
                    self.clients.remove(client_socket)
            
            try:
                client_socket.close()
            except:
                pass
            print(f"[网络] 客户端 {address} 断开连接 (当前连接数: {len(self.clients)})")
    
    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """精确接收指定长度的数据，带边界检测"""
        data = b''
        retry_count = 0
        max_retries = 3
        
        while len(data) < n and retry_count < max_retries:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return b''  # 连接关闭
                data += chunk
            except socket.timeout:
                retry_count += 1
                print(f"[网络] 接收超时 (重试 {retry_count}/{max_retries})，已接收 {len(data)}/{n} 字节")
                if retry_count >= max_retries:
                    return b''
                continue
            except socket.error:
                return b''
        return data
    
    def _validate_length_bytes(self, length_bytes: bytes, address) -> Tuple[bool, int]:
        """验证长度字段的合理性"""
        if len(length_bytes) != 4:
            return False, 0
            
        length = int.from_bytes(length_bytes, byteorder='big')
        
        # 检查长度是否合理（JSON命令通常在10-10000字节之间）
        if 10 <= length <= 1024 * 1024:  # 10字节到1MB
            return True, length
        
        # 长度异常，尝试修复
        print(f"[边界检测] 客户端 {address} 长度异常: {length}")
        print(f"[边界检测] 长度字节: {length_bytes.hex()}")
        
        # 检查是否包含JSON起始标记
        if b'{' in length_bytes:
            print(f"[边界检测] 长度字段中包含JSON数据，可能是边界错位")
            return False, -1  # 特殊标记，表示需要边界修复
            
        return False, length
    
    def _clear_socket_buffer(self, client_socket: socket.socket, address) -> bool:
        """清空Socket缓冲区"""
        try:
            print(f"[恢复] 尝试清空Socket缓冲区...")
            client_socket.settimeout(0.1)  # 短超时
            total_discarded = 0
            
            while True:
                discard = client_socket.recv(4096)
                if not discard:
                    break
                total_discarded += len(discard)
                print(f"[恢复] 丢弃 {len(discard)} 字节")
                
                # 防止无限循环
                if total_discarded > 1024 * 1024:  # 1MB
                    print(f"[恢复] 已丢弃 {total_discarded} 字节，停止清理")
                    break
                    
        except socket.timeout:
            pass  # 超时是正常的，表示缓冲区已清空
        except Exception as e:
            print(f"[恢复] 清理缓冲区失败: {e}")
            return False
        finally:
            client_socket.settimeout(30.0)  # 恢复原超时
            
        print(f"[恢复] 缓冲区已清空，总计丢弃 {total_discarded} 字节")
        return True
    
    def _recover_packet_boundary(self, client_socket: socket.socket, length_bytes: bytes, address) -> bool:
        """尝试修复数据包边界错位"""
        try:
            print(f"[边界修复] 原始长度字节: {length_bytes.hex()}")
            
            # 查找JSON起始位置
            json_start = length_bytes.find(b'{')
            if json_start >= 0:
                print(f"[边界修复] 在长度字段位置 {json_start} 找到JSON起始")
                
                # 构建完整的JSON数据
                remaining_data = length_bytes[json_start:]
                
                # 尝试接收更多数据以完成JSON
                client_socket.settimeout(1.0)
                try:
                    more_data = client_socket.recv(1024)
                    remaining_data += more_data
                    print(f"[边界修复] 接收到额外 {len(more_data)} 字节")
                except socket.timeout:
                    pass
                
                # 尝试解析JSON
                try:
                    # 查找完整的JSON结束
                    json_str = remaining_data.decode('utf-8')
                    bracket_count = 0
                    json_end = -1
                    
                    for i, char in enumerate(json_str):
                        if char == '{':
                            bracket_count += 1
                        elif char == '}':
                            bracket_count -= 1
                            if bracket_count == 0:
                                json_end = i
                                break
                    
                    if json_end >= 0:
                        complete_json = json_str[:json_end + 1]
                        command = json.loads(complete_json)
                        
                        print(f"[边界修复] JSON解析成功: {command.get('type', 'unknown')}")
                        
                        # 处理命令并发送响应
                        response = self._process_command(command)
                        self._send_response(client_socket, response)
                        return True
                        
                except Exception as e:
                    print(f"[边界修复] JSON修复失败: {e}")
                    
            return False
            
        except Exception as e:
            print(f"[边界修复] 修复过程出错: {e}")
            return False
        finally:
            client_socket.settimeout(30.0)  # 恢复原超时
    
    def _send_response(self, sock: socket.socket, response: dict) -> bool:
        """安全发送响应"""
        try:
            response_data = json.dumps(response, ensure_ascii=False).encode('utf-8')
            length_bytes = len(response_data).to_bytes(4, byteorder='big')
            
            # 发送长度
            sock.sendall(length_bytes)
            # 发送数据
            sock.sendall(response_data)
            return True
        except Exception as e:
            print(f"[网络] 发送响应失败: {e}")
            return False
            
    def _process_command(self, command: Dict) -> Dict:
        """处理接收到的命令"""
        cmd_type = command.get("type", "")
        
        with self.lock:
            if cmd_type == "read_current_layer":
                return {
                    "success": True,
                    "layer": self.machine_state.current_layer,
                    "timestamp": time.time()
                }
                
            elif cmd_type == "read_machine_status":
                return {
                    "success": True,
                    "status": self.machine_state.machine_status,
                    "current_layer": self.machine_state.current_layer,
                    "processed_layers": self.machine_state.processed_layers,
                    "timestamp": time.time()
                }
                
            elif cmd_type == "write_layer_complete":
                layer_id = command.get("layer", 0)
                success = command.get("success", False)
                processing_time = command.get("processing_time", 0.0)
                
                print(f"\n[系统] 接收到层完成信号: 第{layer_id}层, 成功: {success}, 耗时: {processing_time:.1f}s")
                
                if success:
                    self.machine_state.processed_layers += 1
                    self.machine_state.total_processing_time += processing_time
                    self.machine_state.layer_times[layer_id] = processing_time
                    print(f"[系统] 第{layer_id}层处理完成，等待手动命令进入下一层")
                else:
                    self.machine_state.failed_layers += 1
                    self.machine_state.machine_status = "error"
                    print(f"[系统] 第{layer_id}层处理失败")
                    
                return {"success": True, "message": "完成信号已接收"}
                
            elif cmd_type == "send_correction_data":
                layer_id = command.get("layer", 0)
                correction_status = command.get("correction_status", "valid")
                data = command.get("data", {})
                
                print(f"\n[系统] 接收到纠偏数据: 第{layer_id}层, 状态: {correction_status}")
                
                if correction_status == "valid":
                    # 保存纠偏数据
                    self.machine_state.correction_data[layer_id] = data
                    self.machine_state.correction_received = True
                    self.machine_state.machine_status = "idle"
                    
                    # 解析和显示纠偏数据详情（包含文件信息）
                    self._process_correction_data(layer_id, data)
                    
                    print(f"[系统] 第{layer_id}层纠偏数据有效，机床准备就绪")
                    print("[提示] 纠偏数据已接收，包含corrected.gcode和offset_table.csv")
                    print("[提示] 使用 'simulate' 命令模拟应用纠偏数据")
                    print("[提示] 使用 'next' 命令进入下一层，或 'start' 开始下一层加工")
                    
                elif correction_status == "skip":
                    print(f"[系统] 第{layer_id}层跳过纠偏（偏差过大），使用原始数据")
                    self.machine_state.correction_applied[layer_id] = False
                    self.machine_state.machine_status = "idle"
                    
                elif correction_status == "warning":
                    print(f"[警告] 第{layer_id}层纠偏数据超出安全范围，已自动丢弃")
                    print("[安全] 机床使用原始加工路径，不应用纠偏")
                    self.machine_state.correction_applied[layer_id] = False
                    
                return {"success": True, "message": "纠偏数据已接收"}
                
            elif cmd_type == "alert_deviation_error":
                layer_id = command.get("layer", 0)
                alert_message = command.get("alert_message", "")
                deviation_value = command.get("deviation_value", 0.0)
                
                print(f"\n[警告] 第{layer_id}层偏差过大: {alert_message} (偏差值: {deviation_value:.3f}mm)")
                
                return {"success": True, "message": "警告已接收"}
                
            else:
                return {"success": False, "error": f"未知命令类型: {cmd_type}"}
                
    def _process_correction_data(self, layer_id: int, data: Dict):
        """处理纠偏数据并显示详情（增强版，支持文件信息）"""
        try:
            # 解析纠偏数据的关键信息
            gcode_adjustments = data.get('gcode_adjustments', [])
            correction_summary = data.get('correction_summary', {})
            processing_info = data.get('processing_info', {})
            
            # 新增: 文件路径信息
            corrected_gcode_path = data.get('corrected_gcode_path', '')
            offset_table_path = data.get('offset_table_path', '')
            output_directory = data.get('output_directory', '')
            available_files = data.get('available_files', [])
            
            print(f"\n━━━ 第{layer_id}层纠偏数据详情 ━━━")
            
            # 显示文件信息（重点！）
            if output_directory:
                print(f"• 输出目录: {output_directory}")
                
            if corrected_gcode_path:
                import os
                file_size = os.path.getsize(corrected_gcode_path) if os.path.exists(corrected_gcode_path) else 0
                print(f"• 纠偏后G代码: {corrected_gcode_path} ({file_size} 字节)")
            else:
                print(f"• 纠偏后G代码: 未提供")
                
            if offset_table_path:
                import os
                file_size = os.path.getsize(offset_table_path) if os.path.exists(offset_table_path) else 0
                print(f"• 偏移表文件: {offset_table_path} ({file_size} 字节)")
            else:
                print(f"• 偏移表文件: 未提供")
                
            if available_files:
                print(f"• 相关文件: {', '.join(available_files)}")
            
            # 显示处理信息
            if processing_info:
                valid_ratio = processing_info.get('valid_ratio', 0)
                total_points = processing_info.get('total_correction_points', 0)
                print(f"• 数据质量: 有效点数 {total_points}, 有效率 {valid_ratio:.1%}")
            
            # 显示纠偏统计
            if correction_summary:
                avg_correction = correction_summary.get('avg_correction_mm', 0)
                max_correction = correction_summary.get('max_correction_mm', 0) 
                affected_lines = correction_summary.get('affected_gcode_lines', 0)
                
                print(f"• 纠偏统计: 平均 {avg_correction:.3f}mm, 最大 {max_correction:.3f}mm")
                print(f"• 影响范围: {affected_lines} 行G代码被修正")
                
                # 保存偏差统计
                self.machine_state.deviation_stats[layer_id] = {
                    'avg_correction': avg_correction,
                    'max_correction': max_correction,
                    'affected_lines': affected_lines,
                    'total_points': processing_info.get('total_correction_points', 0),
                    'valid_ratio': processing_info.get('valid_ratio', 0)
                }
            
            # 显示 G代码调整信息
            if gcode_adjustments:
                print(f"• G代码调整: {len(gcode_adjustments)} 个调整点")
                # 显示前几个调整点作为示例
                for i, adj in enumerate(gcode_adjustments[:3]):
                    line_num = adj.get('line_number', 0)
                    original = adj.get('original_line', '')
                    corrected = adj.get('corrected_line', '')
                    offset = adj.get('offset_mm', [0, 0, 0])
                    print(f"  第{line_num}行: 偏移({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})mm")
                    
                if len(gcode_adjustments) > 3:
                    print(f"  ... 还有 {len(gcode_adjustments)-3} 个调整点")
            
            # 模拟机床接收和应用纠偏数据
            print(f"• 机床状态: 纠偏数据已接收，等待应用")
            if corrected_gcode_path and offset_table_path:
                print(f"• 文件状态: corrected.gcode 和 offset_table.csv 已接收")
                print(f"• 存储位置: 纠偏数据已保存到机床内存")
            else:
                print(f"• 文件状态: 缺少部分文件路径")
            print(f"• 应用状态: 等待手动确认应用 (使用 'simulate {layer_id}' 命令)")
            print("━" * 50)
            
        except Exception as e:
            print(f"[错误] 解析纠偏数据失败: {e}")
                
    def _command_loop(self):
        """命令行处理循环"""
        print("\n请输入命令 (输入 'help' 查看帮助):")
        
        while self.running:
            try:
                command = input(f"[第{self.machine_state.current_layer}层-{self.machine_state.machine_status}] > ").strip().lower()
                
                if not command:
                    continue
                    
                parts = command.split()
                cmd = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                self._handle_command(cmd, args)
                
            except KeyboardInterrupt:
                print("\n\n收到停止信号...")
                self.stop()
                break
            except EOFError:
                print("\n\n输入结束，退出...")
                self.stop()
                break
            except Exception as e:
                print(f"命令处理错误: {e}")
                
    def _handle_command(self, cmd: str, args: list):
        """处理命令行命令"""
        with self.lock:
            if cmd == "start":
                if self.machine_state.machine_status == "idle":
                    self.machine_state.machine_status = "processing"
                    self.machine_state.layer_start_time = time.time()
                    print(f"开始第{self.machine_state.current_layer}层加工...")
                    print("[提示] 使用 'complete' 命令完成当前层加工")
                else:
                    print(f"错误: 当前状态 '{self.machine_state.machine_status}' 无法开始加工")
                    
            elif cmd == "complete":
                if self.machine_state.machine_status == "processing":
                    # 计算加工时间
                    if self.machine_state.layer_start_time:
                        layer_time = time.time() - self.machine_state.layer_start_time
                        self.machine_state.layer_times[self.machine_state.current_layer] = layer_time
                        print(f"第{self.machine_state.current_layer}层加工完成，耗时: {layer_time:.1f}秒")
                    else:
                        print(f"第{self.machine_state.current_layer}层加工完成")
                        
                    self.machine_state.machine_status = "waiting"
                    print("机床进入等待纠偏数据状态...")
                    print("[提示] 系统将自动开始处理纠偏数据")
                else:
                    print(f"错误: 当前状态 '{self.machine_state.machine_status}' 无法完成加工")
                    
            elif cmd == "next":
                if self.machine_state.machine_status in ["idle", "waiting"]:
                    if self.machine_state.current_layer < self.machine_state.total_layers:
                        self.machine_state.current_layer += 1
                        self.machine_state.machine_status = "idle"
                        self.machine_state.correction_received = False
                        print(f"进入第{self.machine_state.current_layer}层")
                        print("[提示] 使用 'start' 开始加工")
                    else:
                        print("所有层已完成！")
                        self._print_summary()
                else:
                    print(f"错误: 当前状态 '{self.machine_state.machine_status}' 无法进入下一层")
                    
            elif cmd == "error":
                self.machine_state.machine_status = "error"
                print("机床设置为错误状态")
                print("[提示] 使用 'reset' 重置状态")
                
            elif cmd == "reset":
                self.machine_state.current_layer = 1
                self.machine_state.machine_status = "idle"
                self.machine_state.correction_received = False
                self.machine_state.layer_start_time = None
                print("机床已重置到第1层空闲状态")
                
            elif cmd == "status":
                self.print_status()
                
            elif cmd == "stats":
                self.print_statistics()
                
            elif cmd == "layers":
                if args and args[0].isdigit():
                    new_total = int(args[0])
                    if 1 <= new_total <= 100:
                        self.machine_state.total_layers = new_total
                        print(f"总层数设置为: {new_total}")
                    else:
                        print("错误: 层数范围应为 1-100")
                else:
                    print("错误: 请指定层数，例如: layers 6")
                    
            elif cmd == "correction":
                self._show_correction_status()
                
            elif cmd == "simulate":
                if args and args[0].isdigit():
                    layer_id = int(args[0])
                    self._simulate_apply_correction(layer_id)
                else:
                    print("错误: 请指定层号，例如: simulate 2")
                    
            elif cmd == "path":
                self._show_path_info()
                
            elif cmd == "deviation":
                self._show_deviation_stats()
                    
            elif cmd == "help":
                self.print_help()
                
            elif cmd == "quit":
                print("正在退出服务器...")
                self.stop()
                
            else:
                print(f"未知命令: {cmd}")
                print("输入 'help' 查看帮助")
                
    def print_status(self):
        """打印当前状态"""
        print(f"\n当前状态:")
        print(f"  层号: {self.machine_state.current_layer}/{self.machine_state.total_layers}")
        print(f"  状态: {self.machine_state.machine_status}")
        print(f"  已处理层数: {self.machine_state.processed_layers}")
        print(f"  失败层数: {self.machine_state.failed_layers}")
        print(f"  纠偏数据已接收: {self.machine_state.correction_received}")
        
        # 显示纠偏数据应用状态
        applied_count = sum(1 for applied in self.machine_state.correction_applied.values() if applied)
        received_count = len(self.machine_state.correction_data)
        print(f"  纠偏数据: 已接收{received_count}层, 已应用{applied_count}层")
        
        if self.machine_state.layer_start_time and self.machine_state.machine_status == "processing":
            elapsed = time.time() - self.machine_state.layer_start_time
            print(f"  当前层加工时间: {elapsed:.1f}秒")
        print(f"  总处理时间: {self.machine_state.total_processing_time:.1f}秒")
        
    def print_statistics(self):
        """打印详细统计"""
        print(f"\n详细统计:")
        print(f"  总层数: {self.machine_state.total_layers}")
        print(f"  已处理: {self.machine_state.processed_layers}")
        print(f"  失败: {self.machine_state.failed_layers}")
        print(f"  总耗时: {self.machine_state.total_processing_time:.1f}秒")
        
        if self.machine_state.layer_times:
            print(f"  各层耗时:")
            for layer_id, layer_time in sorted(self.machine_state.layer_times.items()):
                print(f"    第{layer_id}层: {layer_time:.1f}秒")
            
            if self.machine_state.processed_layers > 0:
                avg_time = self.machine_state.total_processing_time / self.machine_state.processed_layers
                print(f"  平均每层: {avg_time:.1f}秒")
        
    def print_help(self):
        """打印帮助信息"""
        print(f"\n可用命令:")
        for cmd, desc in self.command_help.items():
            print(f"  {cmd:<10} - {desc}")
        print()
        
    def _print_summary(self):
        """打印最终总结"""
        print("\n" + "="*60)
        print("多层加工完成总结")
        print("="*60)
        self.print_statistics()
        print("="*60)
        
    def _show_correction_status(self):
        """显示纠偏数据应用情况"""
        print(f"\n纠偏数据应用情况:")
        print(f"  当前层: {self.machine_state.current_layer}")
        print(f"  纠偏数据已接收: {self.machine_state.correction_received}")
        
        if self.machine_state.correction_data:
            print(f"  已接收纠偏数据的层:")
            for layer_id in sorted(self.machine_state.correction_data.keys()):
                applied = self.machine_state.correction_applied.get(layer_id, False)
                status = "✅ 已应用" if applied else "⏳ 未应用"
                print(f"    第{layer_id}层: {status}")
        else:
            print(f"  暂无纠偏数据")
    
    def _simulate_apply_correction(self, layer_id: int):
        """模拟机床应用纠偏数据"""
        if layer_id not in self.machine_state.correction_data:
            print(f"错误: 第{layer_id}层没有纠偏数据")
            return
            
        print(f"\n正在模拟应用第{layer_id}层纠偏数据...")
        
        # 模拟机床应用过程
        correction_data = self.machine_state.correction_data[layer_id]
        gcode_adjustments = correction_data.get('gcode_adjustments', [])
        
        print(f"• 步骤1: 验证纠偏数据安全性...")
        time.sleep(0.5)
        print(f"• 步骤2: 加载原始加工程序...")
        time.sleep(0.3)
        print(f"• 步骤3: 应用 {len(gcode_adjustments)} 个位置调整...")
        time.sleep(0.8)
        print(f"• 步骤4: 更新机床坐标系统...")
        time.sleep(0.4)
        print(f"• 步骤5: 纠偏数据应用完成！")
        
        # 标记为已应用
        self.machine_state.correction_applied[layer_id] = True
        
        # 显示应用结果
        if layer_id in self.machine_state.deviation_stats:
            stats = self.machine_state.deviation_stats[layer_id]
            print(f"\n应用结果:")
            print(f"  平均纠偏: {stats['avg_correction']:.3f}mm")
            print(f"  最大纠偏: {stats['max_correction']:.3f}mm")
            print(f"  调整点数: {stats['affected_lines']}")
            print(f"  机床状态: 纠偏已生效，准备加工")
        
        print(f"\n✅ 第{layer_id}层纠偏数据已成功应用到机床！")
    
    def _show_path_info(self):
        """显示加工路径信息"""
        print(f"\n加工路径信息:")
        if self.machine_state.original_path:
            print(f"  原始路径: {self.machine_state.original_path}")
        else:
            print(f"  原始路径: 未设置")
            
        if self.machine_state.corrected_path:
            print(f"  纠偏后路径: {self.machine_state.corrected_path}")
        else:
            print(f"  纠偏后路径: 未生成")
            
        applied_layers = [layer_id for layer_id, applied in self.machine_state.correction_applied.items() if applied]
        if applied_layers:
            print(f"  已应用纠偏的层: {sorted(applied_layers)}")
        else:
            print(f"  已应用纠偏的层: 无")
    
    def _show_deviation_stats(self):
        """显示偏差统计信息"""
        if not self.machine_state.deviation_stats:
            print(f"\n暂无偏差统计数据")
            return
            
        print(f"\n偏差统计数据:")
        print(f"  层号    平均纠偏    最大纠偏    调整点数    有效率")
        print(f"  " + "-" * 55)
        
        total_avg = 0
        total_max = 0
        total_points = 0
        
        for layer_id in sorted(self.machine_state.deviation_stats.keys()):
            stats = self.machine_state.deviation_stats[layer_id]
            avg_corr = stats['avg_correction']
            max_corr = stats['max_correction']
            points = stats['affected_lines']
            valid_ratio = stats['valid_ratio']
            
            print(f"  {layer_id:>4}    {avg_corr:>8.3f}mm    {max_corr:>8.3f}mm    {points:>8}    {valid_ratio:>6.1%}")
            
            total_avg += avg_corr
            total_max = max(total_max, max_corr)
            total_points += points
        
        if len(self.machine_state.deviation_stats) > 0:
            avg_avg = total_avg / len(self.machine_state.deviation_stats)
            print(f"  " + "-" * 55)
            print(f"  平均  {avg_avg:>8.3f}mm    {total_max:>8.3f}mm    {total_points:>8}    -")
        
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("交互式模拟PLC服务器已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='交互式模拟PLC服务器')
    parser.add_argument('--host', default='127.0.0.1', help='服务器IP地址')
    parser.add_argument('--port', type=int, default=502, help='服务器端口')
    parser.add_argument('--layers', type=int, default=6, help='总层数')
    
    args = parser.parse_args()
    
    server = InteractiveMockPLCServer(args.host, args.port)
    server.machine_state.total_layers = args.layers
    
    print(f"启动交互式模拟PLC服务器")
    print(f"配置: {args.host}:{args.port}, 总层数: {args.layers}")
    print("使用命令行手动控制每层的加工流程")
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        server.stop()


if __name__ == "__main__":
    main()