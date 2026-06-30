#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
综合脚本：先对关系词去重，然后替换triple文件中的关系谓词
"""

import os
import sys
import json
import subprocess

def check_and_run_deduplication(relation_file='relation2id.txt', 
                                 output_dir='.',
                                 threshold=0.85,
                                 device='cuda'):
    """
    检查并运行去重工具
    
    Returns:
        tuple: (alignment_file, relation2id_file) 或 (None, None)
    """
    alignment_file = os.path.join(output_dir, 'relation_alignment.json')
    relation2id_file = os.path.join(output_dir, 'relation2id_deduplicated.txt')
    
    # 检查是否已经存在映射文件
    if os.path.exists(alignment_file) and os.path.exists(relation2id_file):
        print(f"发现已存在的映射文件:")
        print(f"  - {alignment_file}")
        print(f"  - {relation2id_file}")
        print("跳过去重步骤，直接使用现有映射文件。")
        return alignment_file, relation2id_file
    
    # 运行去重工具
    print("="*60)
    print("步骤1: 运行关系词去重工具")
    print("="*60)
    
    dedup_script = os.path.join(os.path.dirname(__file__), 'relation_deduplication_optimized.py')
    
    if not os.path.exists(dedup_script):
        print(f"错误: 未找到去重脚本: {dedup_script}")
        return None, None
    
    # 构建命令
    cmd = [
        sys.executable,
        dedup_script,
        '--input', relation_file,
        '--output', output_dir,
        '--threshold', str(threshold),
        '--device', device
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        
        # 检查输出文件
        if os.path.exists(alignment_file) and os.path.exists(relation2id_file):
            print(f"\n去重完成！生成的文件:")
            print(f"  - {alignment_file}")
            print(f"  - {relation2id_file}")
            return alignment_file, relation2id_file
        else:
            print("错误: 去重工具运行完成，但未找到输出文件")
            return None, None
            
    except subprocess.CalledProcessError as e:
        print(f"错误: 去重工具运行失败")
        print(f"返回码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"错误: {e}")
        return None, None


def run_replacement(input_file, alignment_file, relation2id_file, output_file=None):
    """
    运行关系词替换工具
    
    Returns:
        str: 输出文件路径
    """
    print("\n" + "="*60)
    print("步骤2: 替换triple文件中的关系谓词")
    print("="*60)
    
    replace_script = os.path.join(os.path.dirname(__file__), 'replace_relations_in_triples.py')
    
    if not os.path.exists(replace_script):
        print(f"错误: 未找到替换脚本: {replace_script}")
        return None
    
    # 构建命令
    cmd = [
        sys.executable,
        replace_script,
        '--input', input_file,
        '--alignment', alignment_file
    ]
    
    if output_file:
        cmd.extend(['--output', output_file])
    
    print(f"运行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        
        # 确定输出文件
        if output_file and os.path.exists(output_file):
            return output_file
        else:
            # 自动生成的输出文件
            base_name = os.path.splitext(input_file)[0]
            auto_output = f"{base_name}_relations_replaced.json"
            if os.path.exists(auto_output):
                return auto_output
        
        return None
            
    except subprocess.CalledProcessError as e:
        print(f"错误: 替换工具运行失败")
        print(f"返回码: {e.returncode}")
        print(f"错误信息: {e.stderr}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='综合脚本：去重关系词并替换triple文件')
    parser.add_argument('--triple-file', '-t', type=str, required=True,
                        help='输入的triple JSON文件路径')
    parser.add_argument('--relation-file', '-r', type=str, default='relation2id.txt',
                        help='关系词文件路径（默认: relation2id.txt）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出的JSON文件路径（默认自动生成）')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='去重文件的输出目录（默认: 当前目录）')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='相似度阈值（默认: 0.85）')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='计算设备（默认: cuda）')
    parser.add_argument('--skip-dedup', action='store_true',
                        help='跳过去重步骤，直接使用现有映射文件')
    
    args = parser.parse_args()
    
    # 步骤1: 去重（如果需要）
    if not args.skip_dedup:
        alignment_file, relation2id_file = check_and_run_deduplication(
            relation_file=args.relation_file,
            output_dir=args.output_dir,
            threshold=args.threshold,
            device=args.device
        )
        
        if alignment_file is None or relation2id_file is None:
            print("\n错误: 去重步骤失败，无法继续")
            return 1
    else:
        # 使用现有文件
        alignment_file = os.path.join(args.output_dir, 'relation_alignment.json')
        relation2id_file = os.path.join(args.output_dir, 'relation2id_deduplicated.txt')
        
        if not os.path.exists(alignment_file):
            print(f"错误: 未找到对齐文件: {alignment_file}")
            return 1
    
    # 步骤2: 替换
    output_file = run_replacement(
        input_file=args.triple_file,
        alignment_file=alignment_file,
        relation2id_file=relation2id_file,
        output_file=args.output
    )
    
    if output_file:
        print("\n" + "="*60)
        print("完成！")
        print("="*60)
        print(f"输出文件: {output_file}")
        return 0
    else:
        print("\n错误: 替换步骤失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())






