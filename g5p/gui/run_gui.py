#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GUI 启动脚本 - 确保正确的路径设置
"""
import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加必要的路径
paths_to_add = [
    current_dir,  # gui目录
    os.path.dirname(current_dir),  # g5p目录
    os.path.dirname(os.path.dirname(current_dir))  # PYTHON_WIN目录
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# 导入并运行主程序
if __name__ == "__main__":
    from main import MainWindow
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
