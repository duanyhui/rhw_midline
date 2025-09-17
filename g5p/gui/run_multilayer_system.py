#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层加工纠偏系统 - 启动脚本
运行此脚本启动完整的多层加工纠偏系统
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def check_dependencies():
    """检查依赖包"""
    missing_packages = []
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
        
    try:
        import cv2
    except ImportError:
        missing_packages.append("opencv-python")
        
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        missing_packages.append("PyQt5")
        
    # 检查可选依赖
    optional_missing = []
    
    try:
        import snap7
    except ImportError:
        optional_missing.append("python-snap7 (S7通信)")
        
    if missing_packages:
        print("错误：缺少必需的依赖包：")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n请使用以下命令安装：")
        print(f"pip install {' '.join(missing_packages)}")
        return False
        
    if optional_missing:
        print("提示：缺少可选依赖包：")
        for pkg in optional_missing:
            print(f"  - {pkg}")
        print("这些包不是必需的，但可能影响某些功能。")
        print()
        
    return True

def setup_directories():
    """设置必需的目录"""
    directories = [
        "projects",
        "configs", 
        "backups",
        "temp",
        "out"
    ]
    
    for dir_name in directories:
        dir_path = current_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        
def check_camera_files():
    """检查相机相关文件"""
    required_files = [
        "T_cam2machine.npy",
        "align_centerline_to_gcode_pro_edit_max.py",
        "controller.py"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = current_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
            
    if missing_files:
        print("警告：缺少以下文件：")
        for file_name in missing_files:
            print(f"  - {file_name}")
        print("这可能影响相机功能的正常使用。")
        return False
        
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("多层加工纠偏系统 v1.0")
    print("=" * 60)
    
    # 检查依赖
    print("检查依赖包...")
    if not check_dependencies():
        sys.exit(1)
        
    # 设置目录
    print("设置工作目录...")
    setup_directories()
    
    # 检查相机文件
    print("检查相机文件...")
    camera_ok = check_camera_files()
    
    if not camera_ok:
        print("\n注意：相机功能可能无法正常工作。")
        print("请确保相关文件存在于程序目录中。")
        
    print("\n启动系统...")
    
    try:
        # 导入并启动主程序
        from multilayer_main import main as run_main
        run_main()
    except Exception as e:
        print(f"启动失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()