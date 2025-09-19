#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动交互式模拟PLC服务器
专用于手动控制多层加工流程测试
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def main():
    """主函数"""
    print("=" * 60)
    print("交互式模拟PLC服务器")
    print("手动控制多层加工流程")
    print("=" * 60)
    
    from interactive_mock_plc import InteractiveMockPLCServer
    
    # 创建服务器实例
    server = InteractiveMockPLCServer(host="127.0.0.1", port=502)
    server.machine_state.total_layers = 6
    
    print("配置:")
    print(f"  服务器地址: 127.0.0.1:502")
    print(f"  总层数: 6")
    print(f"  控制模式: 手动命令行")
    
    print("\n工作流程:")
    print("1. 启动此PLC服务器")
    print("2. 在主程序中连接PLC (TCP, 127.0.0.1:502)")
    print("3. 使用命令控制每层的加工流程:")
    print("   start → complete → (等待系统处理) → next → start → ...")
    
    print("\n常用命令:")
    print("  start    - 开始当前层加工")
    print("  complete - 完成当前层加工 (触发纠偏处理)")
    print("  next     - 进入下一层")
    print("  status   - 查看当前状态")
    print("  help     - 显示所有命令")
    print("  quit     - 退出服务器")
    
    print("\n准备启动...")
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        server.stop()
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()