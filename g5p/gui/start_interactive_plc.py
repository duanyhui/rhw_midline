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
    except OSError as e:
        if "Address already in use" in str(e) or "通常每个套接字地址" in str(e):
            print(f"启动失败: 端口502已被占用")
            print("解决方案:")
            print("1. 检查是否有其他PLC服务器正在运行")
            print("2. 等待几秒后重试")
            print("3. 重启计算机释放端口")
        else:
            print(f"启动失败: {e}")
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        print(f"详细错误信息:\n{traceback.format_exc()}")
        
    print("\n程序结束")
    input("按回车键退出...")

if __name__ == "__main__":
    main()