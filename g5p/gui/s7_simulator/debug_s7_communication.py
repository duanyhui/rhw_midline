#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7通信调试工具 - 测试主程序向S7模拟器发送数据
"""

import sys
import os
import time
import json
import random

# 添加父目录到路径
sys.path.insert(0, '..')

from mock_s7_communicator import MockS7Communicator

def test_send_offset_data():
    """测试发送偏移数据"""
    print("🔧 开始测试S7偏移数据发送...")
    
    # 创建S7通信对象
    s7_comm = MockS7Communicator("127.0.0.1", 8502)
    
    try:
        # 连接到S7模拟器
        print("📡 连接到S7模拟器...")
        if not s7_comm.connect("127.0.0.1"):
            print("❌ 连接失败")
            return False
        
        print("✅ 连接成功")
        
        # 生成测试偏移数据
        test_offsets = []
        for i in range(256):
            dx = random.uniform(-2, 2)  # -2mm到2mm的随机偏移
            dy = random.uniform(-2, 2)
            test_offsets.append((dx, dy))
        
        print(f"📊 生成测试数据: {len(test_offsets)}个偏移点")
        
        # 设置处理锁（模拟主程序的行为）
        print("🔒 设置处理锁...")
        s7_comm.set_processing_lock(True)
        
        # 分批发送数据
        batch_size = 128
        total_batches = (len(test_offsets) + batch_size - 1) // batch_size
        
        print(f"📤 开始分批发送数据: {total_batches}批")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(test_offsets))
            batch_data = test_offsets[start_idx:end_idx]
            
            print(f"  批次 {batch_num + 1}/{total_batches}: {len(batch_data)}个点")
            
            # 发送批次数据
            success = s7_comm.write_offset_batch(batch_data, batch_num + 1, total_batches)
            if not success:
                print(f"❌ 批次 {batch_num + 1} 发送失败")
                return False
            
            time.sleep(0.2)  # 批次间延时
        
        # 释放处理锁
        print("🔓 释放处理锁...")
        s7_comm.set_processing_lock(False)
        
        # 验证数据状态
        time.sleep(0.5)  # 等待模拟器更新状态
        
        data_info = s7_comm.get_offset_data_info()
        print(f"📈 数据状态验证:")
        print(f"  偏移点总数: {data_info['offset_count']}")
        print(f"  当前批次: {data_info['current_batch']}")
        print(f"  总批次数: {data_info['total_batches']}")
        print(f"  数据就绪: {'是' if data_info['data_ready'] else '否'}")
        
        if data_info['data_ready'] and data_info['offset_count'] == len(test_offsets):
            print("✅ 偏移数据发送成功！")
            return True
        else:
            print("❌ 偏移数据状态不正确")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        # 清理连接
        s7_comm.disconnect()

def main():
    print("=" * 60)
    print("S7通信调试工具")
    print("=" * 60)
    
    # 提醒启动模拟器
    print("⚠️  请确保S7模拟器已启动（python s7_plc_simulator.py）")
    input("按回车键继续...")
    
    # 测试发送偏移数据
    success = test_send_offset_data()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！")
        print("💡 模拟器应该显示偏移数据为'就绪'状态，并显示256个偏移点")
    else:
        print("❌ 测试失败！")
        print("💡 请检查S7模拟器是否正常运行，端口8502是否可用")
    print("=" * 60)

if __name__ == "__main__":
    main()