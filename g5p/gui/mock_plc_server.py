# -*- coding: utf-8 -*-
"""
模拟PLC服务器 - 用于测试多层加工纠偏系统
支持TCP/JSON协议，模拟真实机床的工作流程
"""
import json
import socket
import threading
import time
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MockMachineState:
    """模拟机床状态"""
    current_layer: int = 1
    machine_status: str = "idle"  # idle, processing, waiting, error
    manual_mode: bool = True  # 手动模式
    correction_received: bool = False
    layer_start_time: Optional[float] = None
    total_layers: int = 10
    
    # 统计信息
    processed_layers: int = 0
    failed_layers: int = 0
    total_processing_time: float = 0.0


class MockPLCServer:
    """模拟PLC服务器"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 502):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.clients = []
        self.machine_state = MockMachineState()
        self.lock = threading.Lock()
        
    def start(self):
        """启动服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True
            
            print(f"模拟PLC服务器启动: {self.host}:{self.port}")
            print("支持的命令:")
            print("  - read_current_layer: 读取当前层号")
            print("  - read_machine_status: 读取机床状态")
            print("  - write_layer_complete: 写入层完成信号")
            print("  - send_correction_data: 发送纠偏数据")
            print("  - alert_deviation_error: 偏差过大警告")
            
            # 启动机床状态模拟线程
            machine_thread = threading.Thread(target=self._simulate_machine_work, daemon=True)
            machine_thread.start()
            
            # 主循环接收客户端连接
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    print(f"客户端连接: {address}")
                    
                    client_thread = threading.Thread(
                        target=self._handle_client, 
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"接受连接错误: {e}")
                        
        except Exception as e:
            print(f"服务器启动失败: {e}")
            
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("模拟PLC服务器已停止")
        
    def _handle_client(self, client_socket: socket.socket, address):
        """处理客户端连接"""
        try:
            while self.running:
                # 接收数据长度
                length_bytes = client_socket.recv(4)
                if len(length_bytes) < 4:
                    break
                    
                length = int.from_bytes(length_bytes, byteorder='big')
                
                # 接收命令数据
                command_data = client_socket.recv(length)
                if len(command_data) < length:
                    break
                    
                command = json.loads(command_data.decode('utf-8'))
                
                # 处理命令
                response = self._process_command(command)
                
                # 发送响应
                response_data = json.dumps(response).encode('utf-8')
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
                
        except Exception as e:
            print(f"客户端 {address} 处理错误: {e}")
        finally:
            client_socket.close()
            print(f"客户端 {address} 断开连接")
            
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
                
                print(f"接收到层完成信号: 第{layer_id}层, 成功: {success}, 耗时: {processing_time:.1f}s")
                
                if success:
                    self.machine_state.processed_layers += 1
                    self.machine_state.total_processing_time += processing_time
                    
                    # 如果是自动模式，准备下一层
                    if self.machine_state.auto_mode and layer_id < self.machine_state.total_layers:
                        # 延迟一段时间后开始下一层
                        def start_next_layer():
                            time.sleep(2.0)  # 2秒准备时间
                            with self.lock:
                                self.machine_state.current_layer += 1
                                self.machine_state.machine_status = "processing"
                                self.machine_state.layer_start_time = time.time()
                                print(f"自动开始第{self.machine_state.current_layer}层加工")
                                
                        threading.Thread(target=start_next_layer, daemon=True).start()
                else:
                    self.machine_state.failed_layers += 1
                    self.machine_state.machine_status = "error"
                    
                return {"success": True, "message": "完成信号已接收"}
                
            elif cmd_type == "send_correction_data":
                layer_id = command.get("layer", 0)
                correction_status = command.get("correction_status", "valid")
                data = command.get("data", {})
                
                print(f"接收到纠偏数据: 第{layer_id}层, 状态: {correction_status}")
                
                if correction_status == "valid":
                    # 模拟接收有效纠偏数据
                    self.machine_state.correction_received = True
                    self.machine_state.machine_status = "idle"
                    print("  纠偏数据有效，机床准备就绪")
                    
                    # 打印部分数据信息
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) > 50:
                            print(f"    {key}: {value[:50]}...")
                        else:
                            print(f"    {key}: {value}")
                            
                elif correction_status == "skip":
                    print("  跳过纠偏（偏差过大），使用原始数据")
                    self.machine_state.machine_status = "idle"
                    
                return {"success": True, "message": "纠偏数据已接收"}
                
            elif cmd_type == "alert_deviation_error":
                layer_id = command.get("layer", 0)
                alert_message = command.get("alert_message", "")
                deviation_value = command.get("deviation_value", 0.0)
                
                print(f"偏差警告: 第{layer_id}层 - {alert_message} (偏差值: {deviation_value:.3f}mm)")
                
                return {"success": True, "message": "警告已接收"}
                
            else:
                return {"success": False, "error": f"未知命令类型: {cmd_type}"}
                
    def _simulate_machine_work(self):
        """模拟机床工作流程"""
        print("机床工作模拟线程启动")
        
        while self.running:
            try:
                with self.lock:
                    current_time = time.time()
                    
                    # 模拟加工过程
                    if self.machine_state.machine_status == "processing":
                        if (self.machine_state.layer_start_time and 
                            current_time - self.machine_state.layer_start_time > self.machine_state.layer_processing_time):
                            # 加工完成，等待纠偏数据
                            self.machine_state.machine_status = "waiting"
                            print(f"第{self.machine_state.current_layer}层加工完成，等待纠偏数据...")
                            
                    # 检查是否所有层都完成
                    elif (self.machine_state.machine_status == "idle" and 
                          self.machine_state.current_layer > self.machine_state.total_layers):
                        print("所有层加工完成！")
                        self._print_summary()
                        break
                        
                time.sleep(0.5)  # 500ms检查间隔
                
            except Exception as e:
                print(f"机床模拟错误: {e}")
                time.sleep(1.0)
                
    def _print_summary(self):
        """打印加工总结"""
        print("\n" + "="*50)
        print("加工总结")
        print("="*50)
        print(f"总层数: {self.machine_state.total_layers}")
        print(f"成功层数: {self.machine_state.processed_layers}")
        print(f"失败层数: {self.machine_state.failed_layers}")
        print(f"总耗时: {self.machine_state.total_processing_time:.1f} 秒")
        if self.machine_state.processed_layers > 0:
            avg_time = self.machine_state.total_processing_time / self.machine_state.processed_layers
            print(f"平均每层耗时: {avg_time:.1f} 秒")
        print("="*50)
        
    # 手动控制方法（用于测试）
    def manual_start_layer(self, layer_id: int):
        """手动开始指定层加工"""
        with self.lock:
            self.machine_state.current_layer = layer_id
            self.machine_state.machine_status = "processing"
            self.machine_state.layer_start_time = time.time()
            print(f"手动开始第{layer_id}层加工")
            
    def manual_complete_layer(self):
        """手动完成当前层加工"""
        with self.lock:
            if self.machine_state.machine_status == "processing":
                self.machine_state.machine_status = "waiting"
                print(f"手动完成第{self.machine_state.current_layer}层加工，等待纠偏数据")
                
    def set_processing_time(self, seconds: float):
        """设置每层加工时间"""
        with self.lock:
            self.machine_state.layer_processing_time = seconds
            print(f"每层加工时间设置为: {seconds} 秒")
            
    def get_status(self) -> Dict:
        """获取当前状态"""
        with self.lock:
            return {
                "current_layer": self.machine_state.current_layer,
                "status": self.machine_state.machine_status,
                "processed_layers": self.machine_state.processed_layers,
                "failed_layers": self.machine_state.failed_layers,
                "total_processing_time": self.machine_state.total_processing_time,
                "auto_mode": self.machine_state.auto_mode
            }


def main():
    """主函数 - 运行模拟PLC服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模拟PLC服务器')
    parser.add_argument('--host', default='127.0.0.1', help='服务器IP地址')
    parser.add_argument('--port', type=int, default=502, help='服务器端口')
    parser.add_argument('--layers', type=int, default=6, help='总层数')
    parser.add_argument('--time', type=float, default=5.0, help='每层加工时间（秒）')
    
    args = parser.parse_args()
    
    server = MockPLCServer(args.host, args.port)
    server.machine_state.total_layers = args.layers
    server.machine_state.layer_processing_time = args.time
    
    print(f"启动模拟PLC服务器: {args.host}:{args.port}")
    print(f"总层数: {args.layers}, 每层加工时间: {args.time}秒")
    print("按 Ctrl+C 停止服务器")
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        server.stop()


if __name__ == "__main__":
    main()