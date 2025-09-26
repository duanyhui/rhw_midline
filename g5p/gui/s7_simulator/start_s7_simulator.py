#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7 PLC模拟器启动脚本 - 完整版
提供环境检查、依赖验证和用户友好的启动体验
"""

import sys
import os
import socket
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def print_banner():
    """显示启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                   S7 PLC模拟器启动器                     ║
    ║                                                          ║
    ║  🔧 模拟西门子S7 PLC数据块读写                            ║
    ║  🌐 TCP/JSON通信协议 (端口8502)                          ║
    ║  🛡️ 完整安全机制和偏移数据处理                            ║
    ║  📊 PyQt5 GUI实时监控界面                                ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查依赖库"""
    print("🔍 检查依赖库...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QThread, QTimer, pyqtSignal
        print("   ✅ PyQt5 - GUI框架")
    except ImportError:
        print("   ❌ PyQt5 - 安装命令: pip install PyQt5")
        return False
    
    # 检查其他标准库
    for lib in ['json', 'socket', 'threading', 'struct', 'time']:
        try:
            __import__(lib)
            print(f"   ✅ {lib} - 标准库")
        except ImportError:
            print(f"   ❌ {lib} - 标准库缺失")
            return False
    
    print("✅ 所有依赖检查通过")
    return True

def check_port():
    """检查端口8502可用性"""
    print("\n🌐 检查端口8502...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', 8502))
        sock.close()
        
        if result == 0:
            print("   ❌ 端口8502已被占用")
            print("   💡 请关闭其他S7模拟器实例")
            return False
        else:
            print("   ✅ 端口8502可用")
            return True
    except Exception as e:
        print(f"   ⚠️ 端口检查失败: {e}")
        return True  # 继续运行

def check_files():
    """检查必要文件"""
    print("\n📁 检查文件结构...")
    
    required_files = [
        's7_plc_simulator.py',
        's7_tcp_server.py', 
        'mock_s7_communicator.py'
    ]
    
    for filename in required_files:
        filepath = current_dir / filename
        if filepath.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} - 文件缺失")
            return False
    
    print("✅ 文件结构完整")
    return True

def show_config_info():
    """显示配置信息"""
    print("\n" + "="*60)
    print("📋 S7 PLC模拟器配置信息")
    print("="*60)
    
    print("🔧 数据块结构:")
    print("  • DB9044: 控制数据块")
    print("    - 0: 机器状态 (0=空闲,1=加工中,2=等待纠偏)")
    print("    - 2: 当前层号") 
    print("    - 6: 处理锁 (0=空闲,1=锁定)")
    print("    - 8: 偏移点数量")
    print("    - 10: 当前批次")
    print("    - 12: 总批次数")
    print("    - 14: 数据就绪标志")
    
    print("\n  • DB9045-9047: 偏移数据块")
    print("    - 每块128个偏移点")
    print("    - 格式: dx(2字节) + dy(2字节)")
    print("    - 单位: 微米(μm)")
    
    print("\n🌐 通信配置:")
    print("  • 协议: TCP/JSON")
    print("  • 端口: 8502")
    print("  • 格式: 长度前缀 + JSON数据")
    
    print("\n🛡️ 安全机制:")
    print("  • 最大偏移: 20mm (20000μm)")
    print("  • 梯度限制: 0.5mm/mm")
    print("  • 处理锁保护: 防止数据冲突")

def main():
    """主启动流程"""
    try:
        # 显示横幅
        print_banner()
        
        # 环境检查
        if not check_dependencies():
            print("\n❌ 依赖检查失败")
            return False
            
        if not check_port():
            print("\n❌ 端口检查失败")
            return False
            
        if not check_files():
            print("\n❌ 文件检查失败") 
            return False
        
        # 显示配置信息
        show_config_info()
        
        # 用户确认
        print("\n" + "="*60)
        print("🚀 准备启动S7 PLC模拟器")
        print("="*60)
        
        print("\n💡 启动后操作提示:")
        print("  1. 等待GUI界面出现")
        print("  2. 确认TCP服务器状态为绿色")
        print("  3. 在主程序中选择PLC类型 's7_sim'")
        print("  4. 连接地址: 127.0.0.1:8502")
        
        input("\n按回车键开始启动...")
        
        # 启动模拟器
        print("\n🔄 正在启动S7 PLC模拟器...")
        
        from s7_plc_simulator import S7PLCSimulatorGUI
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        app.setApplicationName("S7 PLC模拟器")
        app.setApplicationVersion("1.0")
        
        simulator = S7PLCSimulatorGUI()
        simulator.show()
        
        print("✅ S7模拟器GUI已启动")
        print("📊 监控界面已打开，TCP服务器正在初始化...")
        
        # 运行应用
        exit_code = app.exec_()
        return exit_code == 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断启动")
        return False
    except ImportError as e:
        print(f"\n❌ 导入错误: {e}")
        print("请确保PyQt5已正确安装: pip install PyQt5")
        return False
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        print(f"详细错误:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🔧 正在初始化S7 PLC模拟器启动器...")
    
    result = main()
    
    if result:
        print("\n🎉 S7 PLC模拟器已正常退出")
    else:
        print("\n💡 故障排除建议:")
        print("  1. 检查PyQt5安装: pip install PyQt5")
        print("  2. 确认端口8502未被占用")
        print("  3. 检查文件完整性")
        print("  4. 查看详细错误信息")
        print("  5. 参考USAGE_GUIDE.md获取帮助")
        
        input("\n按回车键退出...")
        sys.exit(1)
    
    print("👋 谢谢使用!")