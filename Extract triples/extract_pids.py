#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取 output_PHD_with_wiki_ref.json 中 wiki_ref 字段的所有 PID 并去重
"""

import json
import re
import os
from typing import Set, List

def extract_pids_from_wiki_ref(wiki_ref: List[List[str]]) -> Set[str]:
    """
    从 wiki_ref 列表中提取 PID
    
    Args:
        wiki_ref: wiki_ref 列表，每个元素是 [subject, predicate, object] 格式
        
    Returns:
        提取到的 PID 集合
    """
    pids = set()
    
    # PID 的正则表达式：P 后跟一个或多个数字
    pid_pattern = re.compile(r'^P\d+$')
    
    for ref in wiki_ref:
        if isinstance(ref, list) and len(ref) >= 2:
            # 检查第二个元素（谓词位置）是否是 PID
            predicate = ref[1]
            if isinstance(predicate, str) and pid_pattern.match(predicate):
                pids.add(predicate)
    
    return pids

def extract_all_pids(json_file: str) -> Set[str]:
    """
    从 JSON 文件中提取所有 PID
    
    Args:
        json_file: JSON 文件路径
        
    Returns:
        所有 PID 的集合
    """
    all_pids = set()
    
    print(f"正在读取文件: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件读取成功，开始提取 PID...")
        
        processed_entries = 0
        entries_with_wiki_ref = 0
        
        # 遍历所有数据集
        for dataset_name, entries in data.items():
            if isinstance(entries, list):
                print(f"正在处理数据集: {dataset_name}")
                
                for entry in entries:
                    if isinstance(entry, dict):
                        processed_entries += 1
                        
                        # 提取 wiki_ref 字段中的 PID
                        if 'wiki_ref' in entry and entry['wiki_ref']:
                            entries_with_wiki_ref += 1
                            pids = extract_pids_from_wiki_ref(entry['wiki_ref'])
                            all_pids.update(pids)
        
        print(f"处理完成:")
        print(f"  总处理条目数: {processed_entries}")
        print(f"  包含 wiki_ref 的条目数: {entries_with_wiki_ref}")
        print(f"  提取到的唯一 PID 数量: {len(all_pids)}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file}")
        return set()
    except json.JSONDecodeError as e:
        print(f"错误：JSON 格式错误 - {e}")
        return set()
    except Exception as e:
        print(f"错误：处理文件时出现异常 - {e}")
        return set()
    
    return all_pids

def save_pids_to_file(pids: Set[str], output_file: str):
    """
    将 PID 保存到文件
    
    Args:
        pids: PID 集合
        output_file: 输出文件路径
    """
    # 排序后保存
    sorted_pids = sorted(pids, key=lambda x: int(x[1:]))  # 按数字排序
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for pid in sorted_pids:
                f.write(pid + '\n')
        
        print(f"PID 列表已保存到: {output_file}")
        
        # 同时保存为 JSON 格式
        json_output_file = output_file.replace('.txt', '.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_pids, f, ensure_ascii=False, indent=2)
        
        print(f"PID 列表也已保存为 JSON 格式: {json_output_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

def analyze_pids(pids: Set[str]):
    """
    分析 PID 的统计信息
    
    Args:
        pids: PID 集合
    """
    if not pids:
        print("没有找到任何 PID")
        return
    
    # 转换为数字列表进行分析
    pid_numbers = [int(pid[1:]) for pid in pids]
    pid_numbers.sort()
    
    print(f"\n=== PID 统计分析 ===")
    print(f"总 PID 数量: {len(pids)}")
    print(f"最小 PID: P{min(pid_numbers)}")
    print(f"最大 PID: P{max(pid_numbers)}")
    
    # 显示前 20 个 PID
    print(f"\n前 20 个 PID:")
    sorted_pids = [f"P{num}" for num in pid_numbers[:20]]
    for i, pid in enumerate(sorted_pids):
        print(f"  {i+1:2d}. {pid}")
    
    if len(pids) > 20:
        print(f"  ... 还有 {len(pids) - 20} 个")
    
    # 显示最后 10 个 PID
    if len(pids) > 10:
        print(f"\n最后 10 个 PID:")
        last_pids = [f"P{num}" for num in pid_numbers[-10:]]
        for i, pid in enumerate(last_pids, start=len(pids)-9):
            print(f"  {i:2d}. {pid}")

def main():
    # 文件路径
    input_file = "./output/output_PHD_with_wiki_ref.json"
    output_file = "./output/extracted_pids.txt"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    # 提取所有 PID
    all_pids = extract_all_pids(input_file)
    
    if all_pids:
        # 分析和显示统计信息
        analyze_pids(all_pids)
        
        # 保存到文件
        save_pids_to_file(all_pids, output_file)
    else:
        print("没有提取到任何 PID")

if __name__ == "__main__":
    main()








