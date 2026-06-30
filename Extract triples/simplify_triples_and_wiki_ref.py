#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化 output_PHD_with_wiki_ref.json 中的 triples 和 wiki_ref 字段
将它们转换为形如 (subject, predicate, object) 的三元组数组
"""

import json
import os
from typing import Dict, List, Any

def simplify_triples(triples: List[Dict[str, str]]) -> List[str]:
    """
    简化 triples 字段
    
    Args:
        triples: 原始的 triples 列表，每个元素是 {"triple": "(s, p, o)"} 格式
    
    Returns:
        简化后的三元组字符串列表
    """
    simplified = []
    for triple_obj in triples:
        if isinstance(triple_obj, dict) and 'triple' in triple_obj:
            simplified.append(triple_obj['triple'])
        elif isinstance(triple_obj, str):
            # 如果已经是字符串格式，直接添加
            simplified.append(triple_obj)
    return simplified

def simplify_wiki_ref(wiki_ref: List[List[str]]) -> List[str]:
    """
    简化 wiki_ref 字段
    
    Args:
        wiki_ref: 原始的 wiki_ref 列表，每个元素是 [subject, predicate, object] 格式
    
    Returns:
        简化后的三元组字符串列表
    """
    simplified = []
    for ref in wiki_ref:
        if isinstance(ref, list) and len(ref) >= 3:
            # 转换为 (subject, predicate, object) 格式
            triple_str = f"({ref[0]}, {ref[1]}, {ref[2]})"
            simplified.append(triple_str)
    return simplified

def process_json_file(input_file: str, output_file: str):
    """
    处理 JSON 文件，简化 triples 和 wiki_ref 字段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件读取成功，开始处理数据...")
        
        # 处理数据
        processed_count = 0
        for dataset_name, entries in data.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        # 简化 triples 字段
                        if 'triples' in entry:
                            entry['triples'] = simplify_triples(entry['triples'])
                        
                        # 简化 wiki_ref 字段
                        if 'wiki_ref' in entry:
                            entry['wiki_ref'] = simplify_wiki_ref(entry['wiki_ref'])
                        
                        processed_count += 1
        
        print(f"数据处理完成，共处理了 {processed_count} 个条目")
        
        # 保存处理后的数据
        print(f"正在保存到文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"文件保存成功！")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 格式错误 - {e}")
    except Exception as e:
        print(f"错误：处理文件时出现异常 - {e}")

def main():
    # 设置文件路径
    input_file = "./output/output_PHD_with_wiki_ref.json"
    output_file = "./output/output_PHD_with_wiki_ref_simplified.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理文件
    process_json_file(input_file, output_file)
    
    # 显示简单的统计信息
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            simplified_data = json.load(f)
        
        print("\n=== 简化结果统计 ===")
        for dataset_name, entries in simplified_data.items():
            if isinstance(entries, list):
                total_entries = len(entries)
                entries_with_triples = sum(1 for entry in entries if 'triples' in entry and entry['triples'])
                entries_with_wiki_ref = sum(1 for entry in entries if 'wiki_ref' in entry and entry['wiki_ref'])
                
                print(f"{dataset_name}:")
                print(f"  总条目数: {total_entries}")
                print(f"  包含 triples 的条目: {entries_with_triples}")
                print(f"  包含 wiki_ref 的条目: {entries_with_wiki_ref}")
                
                # 显示一个示例
                if entries_with_triples > 0:
                    for entry in entries:
                        if 'triples' in entry and entry['triples']:
                            print(f"  Triples 示例: {entry['triples'][:2]}...")
                            break
                
                if entries_with_wiki_ref > 0:
                    for entry in entries:
                        if 'wiki_ref' in entry and entry['wiki_ref']:
                            print(f"  Wiki_ref 示例: {entry['wiki_ref'][:2]}...")
                            break
                print()
    
    except Exception as e:
        print(f"显示统计信息时出错: {e}")

if __name__ == "__main__":
    main()
