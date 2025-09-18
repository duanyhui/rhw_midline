#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON to NPY转换脚本
将JSON文件转换为NPY文件，文件名保持不变
"""

import json
import numpy as np
import os
import sys
import argparse
from pathlib import Path


def json_to_npy(json_file_path, output_dir=None):
    """
    将JSON文件转换为NPY文件

    Args:
        json_file_path (str): JSON文件路径
        output_dir (str, optional): 输出目录，默认为输入文件的同级目录

    Returns:
        str: 生成的NPY文件路径
    """
    json_path = Path(json_file_path)

    # 检查文件是否存在
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_file_path}")

    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取JSON文件: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")
    except Exception as e:
        raise Exception(f"读取JSON文件时出错: {e}")

    # 确定输出目录
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 生成NPY文件路径（保持相同文件名，只改扩展名）
    npy_path = output_dir / (json_path.stem + '.npy')

    # 保存为NPY文件
    try:
        np.save(npy_path, data)
        print(f"成功保存NPY文件: {npy_path}")
        return str(npy_path)
    except Exception as e:
        raise Exception(f"保存NPY文件时出错: {e}")


def batch_convert(directory, recursive=False):
    """
    批量转换目录下的所有JSON文件

    Args:
        directory (str): 目录路径
        recursive (bool): 是否递归搜索子目录
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    # 搜索JSON文件
    if recursive:
        json_files = list(dir_path.rglob('*.json'))
    else:
        json_files = list(dir_path.glob('*.json'))

    if not json_files:
        print("未找到JSON文件")
        return

    print(f"找到 {len(json_files)} 个JSON文件")

    success_count = 0
    error_count = 0

    for json_file in json_files:
        try:
            json_to_npy(json_file)
            success_count += 1
        except Exception as e:
            print(f"转换失败 {json_file}: {e}")
            error_count += 1

    print(f"\n转换完成: 成功 {success_count} 个, 失败 {error_count} 个")


def main():
    parser = argparse.ArgumentParser(description='JSON to NPY转换工具')
    parser.add_argument('input', help='输入JSON文件路径或目录路径')
    parser.add_argument('-o', '--output', help='输出目录（可选）')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归搜索子目录（仅在输入为目录时有效）')
    parser.add_argument('-b', '--batch', action='store_true',
                       help='批量转换模式（将输入视为目录）')

    args = parser.parse_args()

    input_path = Path(args.input)

    try:
        if args.batch or input_path.is_dir():
            # 批量转换模式
            batch_convert(args.input, args.recursive)
        else:
            # 单文件转换模式
            json_to_npy(args.input, args.output)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # 如果没有命令行参数，提供交互式使用方式
    if len(sys.argv) == 1:
        print("JSON to NPY转换工具")
        print("-" * 30)

        # 交互式模式
        while True:
            print("\n选择操作:")
            print("1. 转换单个JSON文件")
            print("2. 批量转换目录下的JSON文件")
            print("3. 退出")

            choice = input("请选择 (1-3): ").strip()

            if choice == '1':
                json_file = input("请输入JSON文件路径: ").strip()
                output_dir = input("请输入输出目录（回车使用默认位置）: ").strip()

                try:
                    if not output_dir:
                        output_dir = None
                    result = json_to_npy(json_file, output_dir)
                    print(f"转换成功！文件保存到: {result}")
                except Exception as e:
                    print(f"转换失败: {e}")

            elif choice == '2':
                directory = input("请输入目录路径: ").strip()
                recursive = input("是否递归搜索子目录？(y/n): ").strip().lower() == 'y'

                try:
                    batch_convert(directory, recursive)
                except Exception as e:
                    print(f"批量转换失败: {e}")

            elif choice == '3':
                print("退出程序")
                break

            else:
                print("无效选择，请重新输入")
    else:
        # 命令行模式
        main()
