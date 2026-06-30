#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
利用 pred2id.txt 中的映射关系，将 output_PHD_with_wiki_ref_simplified.json
中的 wiki_ref 字段中的 PID 替换为真正的谓词名称
"""

import json
import re
from typing import Dict, List


def load_pid_mappings(pred2id_file: str) -> Dict[str, str]:
    """
    从 pred2id.txt 文件中加载 PID 到谓词标签的映射

    Args:
        pred2id_file: pred2id.txt 文件路径

    Returns:
        PID 到标签的映射字典
    """
    pid_mappings = {}

    try:
        with open(pred2id_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释行和空行
                if line.startswith('#') or not line:
                    continue

                # 分割 PID 和标签
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    pid, label = parts
                    pid_mappings[pid] = label

        print(f"成功加载 {len(pid_mappings)} 个 PID 映射")

    except FileNotFoundError:
        print(f"错误：找不到文件 {pred2id_file}")
    except Exception as e:
        print(f"读取 {pred2id_file} 时出错: {e}")

    return pid_mappings


def replace_pid_in_triple(triple_str: str, pid_mappings: Dict[str, str]) -> str:
    """
    替换三元组字符串中的 PID

    Args:
        triple_str: 三元组字符串，格式如 "(subject, P123, object)"
        pid_mappings: PID 到标签的映射字典

    Returns:
        替换后的三元组字符串
    """
    # 使用正则表达式匹配三元组格式：(subject, predicate, object)
    pattern = r'\(([^,]+),\s*(P\d+),\s*([^)]+)\)'
    match = re.match(pattern, triple_str)

    if match:
        subject, pid, obj = match.groups()
        # 替换 PID 为对应的标签
        predicate = pid_mappings.get(pid, pid)  # 如果找不到映射，保持原 PID
        return f"({subject}, {predicate}, {obj})"
    else:
        # 如果格式不匹配，返回原字符串
        return triple_str


def process_wiki_ref_field(wiki_ref: List[str], pid_mappings: Dict[str, str]) -> List[str]:
    """
    处理 wiki_ref 字段，替换其中的 PID

    Args:
        wiki_ref: wiki_ref 字段的三元组列表
        pid_mappings: PID 到标签的映射字典

    Returns:
        替换后的三元组列表
    """
    processed_wiki_ref = []

    for triple_str in wiki_ref:
        if isinstance(triple_str, str):
            processed_triple = replace_pid_in_triple(triple_str, pid_mappings)
            processed_wiki_ref.append(processed_triple)
        else:
            # 如果不是字符串，保持原样
            processed_wiki_ref.append(triple_str)

    return processed_wiki_ref


def process_json_file(input_file: str, output_file: str, pid_mappings: Dict[str, str]):
    """
    处理 JSON 文件，替换其中的 PID

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        pid_mappings: PID 到标签的映射字典
    """
    try:
        print(f"正在读取文件: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("文件读取成功，开始处理数据...")

        # 统计信息
        total_entries = 0
        processed_entries = 0
        total_replacements = 0

        # 处理数据
        for dataset_name, entries in data.items():
            if isinstance(entries, list):
                for entry in entries:
                    total_entries += 1

                    if isinstance(entry, dict) and 'wiki_ref' in entry:
                        original_wiki_ref = entry['wiki_ref']
                        processed_wiki_ref = process_wiki_ref_field(original_wiki_ref, pid_mappings)
                        entry['wiki_ref'] = processed_wiki_ref
                        processed_entries += 1

                        # 统计替换次数
                        for orig, proc in zip(original_wiki_ref, processed_wiki_ref):
                            if orig != proc:
                                total_replacements += 1

        print(f"数据处理完成:")
        print(f"  总条目数: {total_entries}")
        print(f"  包含 wiki_ref 的条目: {processed_entries}")
        print(f"  总替换次数: {total_replacements}")

        # 保存处理后的数据
        print(f"正在保存到文件: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("文件保存成功！")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 格式错误 - {e}")
    except Exception as e:
        print(f"错误：处理文件时出现异常 - {e}")


def show_sample_results(input_file: str, pid_mappings: Dict[str, str], num_samples: int = 5):
    """
    显示一些替换示例

    Args:
        input_file: 输入文件路径
        pid_mappings: PID 到标签的映射字典
        num_samples: 显示的示例数量
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n=== 替换示例 (前 {num_samples} 个) ===")

        sample_count = 0
        for dataset_name, entries in data.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and 'wiki_ref' in entry and entry['wiki_ref']:
                        for triple_str in entry['wiki_ref'][:3]:  # 每个条目最多显示3个
                            if sample_count >= num_samples:
                                return

                            original = triple_str
                            processed = replace_pid_in_triple(triple_str, pid_mappings)

                            if original != processed:
                                print(f"原始: {original}")
                                print(f"替换: {processed}")
                                print("-" * 50)
                                sample_count += 1

                            if sample_count >= num_samples:
                                return

    except Exception as e:
        print(f"显示示例时出错: {e}")


def main():
    """主函数"""
    # 文件路径
    pred2id_file = "./output/pred2id.txt"
    input_file = "./output/output_PHD_with_wiki_ref_simplified.json"
    output_file = "./output/output_PHD_with_wiki_ref_processed.json"

    print("开始处理 wiki_ref 字段中的 PID 替换...")

    # 1. 加载 PID 映射
    pid_mappings = load_pid_mappings(pred2id_file)
    if not pid_mappings:
        print("无法加载 PID 映射，程序退出")
        return

    # 2. 显示一些替换示例（处理前）
    show_sample_results(input_file, pid_mappings)

    # 3. 处理 JSON 文件
    process_json_file(input_file, output_file, pid_mappings)

    # 4. 显示最终统计信息
    print(f"\n=== 最终结果 ===")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"使用的映射数量: {len(pid_mappings)}")

    # 显示一些成功映射的 PID 示例
    print(f"\n=== PID 映射示例 ===")
    sample_pids = list(pid_mappings.keys())[:10]
    for pid in sample_pids:
        print(f"{pid} -> {pid_mappings[pid]}")


if __name__ == "__main__":
    main()