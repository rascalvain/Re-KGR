#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用去重后的映射，替换triple文件中的关系谓词
"""

import json
import os
import re
from tqdm import tqdm
from collections import defaultdict

class TripleRelationReplacer:
    """Triple关系谓词替换器"""
    
    def __init__(self, alignment_file=None, relation2id_file=None):
        """
        初始化
        
        Args:
            alignment_file: relation_alignment.json文件路径
            relation2id_file: relation2id_deduplicated.txt文件路径（可选，用于验证）
        """
        self.relation_mapping = {}  # 关系词到主关系词的映射
        self.stats = {
            'total_triples': 0,
            'replaced_triples': 0,
            'not_found_relations': defaultdict(int),
            'relation_counts': defaultdict(int)
        }
        
        if alignment_file and os.path.exists(alignment_file):
            self.load_alignment(alignment_file)
        elif relation2id_file and os.path.exists(relation2id_file):
            # 如果没有对齐文件，从去重后的relation2id文件构建映射
            print("未找到对齐文件，从去重后的relation2id文件构建映射...")
            self.build_mapping_from_relation2id(relation2id_file)
        else:
            print("警告: 未找到映射文件，将尝试自动查找...")
            self.auto_load_mapping()
    
    def load_alignment(self, alignment_file):
        """从对齐文件加载映射"""
        print(f"加载对齐文件: {alignment_file}")
        with open(alignment_file, 'r', encoding='utf-8') as f:
            alignment_data = json.load(f)
        
        # 构建关系词到主关系词的映射
        for main_relation, aligned_relations in alignment_data.items():
            for relation in aligned_relations:
                self.relation_mapping[relation] = main_relation
        
        print(f"加载了 {len(self.relation_mapping)} 个关系词映射")
    
    def build_mapping_from_relation2id(self, relation2id_file):
        """从去重后的relation2id文件构建映射（假设所有关系词都是主关系词）"""
        print(f"从relation2id文件构建映射: {relation2id_file}")
        with open(relation2id_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    relation = parts[0]
                    # 每个关系词映射到自己（因为没有对齐信息）
                    self.relation_mapping[relation] = relation
        
        print(f"加载了 {len(self.relation_mapping)} 个关系词")
    
    def auto_load_mapping(self):
        """自动查找映射文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 尝试查找对齐文件
        alignment_file = os.path.join(current_dir, 'relation_alignment.json')
        if os.path.exists(alignment_file):
            self.load_alignment(alignment_file)
            return
        
        # 尝试查找去重后的relation2id文件
        relation2id_file = os.path.join(current_dir, 'relation2id_deduplicated.txt')
        if os.path.exists(relation2id_file):
            self.build_mapping_from_relation2id(relation2id_file)
            return
        
        print("错误: 未找到映射文件！")
        print("请先运行去重工具生成映射文件，或指定映射文件路径。")
        raise FileNotFoundError("未找到映射文件")
    
    def parse_triple(self, triple_str):
        """
        解析triple字符串
        
        Args:
            triple_str: triple字符串，格式为 "(subject, relation, object)"
            
        Returns:
            tuple: (subject, relation, object) 或 None（如果解析失败）
        """
        # 移除首尾的括号和空格
        triple_str = triple_str.strip()
        if not triple_str.startswith('(') or not triple_str.endswith(')'):
            return None
        
        # 移除括号
        content = triple_str[1:-1].strip()
        
        # 使用正则表达式分割，考虑逗号可能在对象中出现
        # 匹配格式: (subject, relation, object)
        # 注意：对象可能包含逗号，所以需要更智能的解析
        pattern = r'^\(([^,]+),\s*([^,]+),\s*(.+)\)$'
        match = re.match(pattern, triple_str)
        if match:
            return match.groups()
        
        # 如果正则匹配失败，尝试简单的分割（假设对象中没有逗号）
        parts = [p.strip() for p in content.split(',')]
        if len(parts) >= 3:
            subject = parts[0]
            relation = parts[1]
            object_part = ', '.join(parts[2:])  # 对象可能包含逗号
            return (subject, relation, object_part)
        
        return None
    
    def replace_relation_in_triple(self, triple_str):
        """
        替换triple中的关系谓词
        
        Args:
            triple_str: 原始triple字符串
            
        Returns:
            str: 替换后的triple字符串
        """
        parsed = self.parse_triple(triple_str)
        if parsed is None:
            return triple_str  # 如果解析失败，返回原字符串
        
        subject, relation, object_part = parsed
        
        # 记录关系词出现次数
        self.stats['relation_counts'][relation] += 1
        
        # 查找映射
        if relation in self.relation_mapping:
            new_relation = self.relation_mapping[relation]
            if new_relation != relation:
                self.stats['replaced_triples'] += 1
            # 构建新的triple字符串
            return f"({subject}, {new_relation}, {object_part})"
        else:
            # 关系词不在映射中
            self.stats['not_found_relations'][relation] += 1
            return triple_str  # 保持原样
    
    def process_json_file(self, input_file, output_file=None):
        """
        处理JSON文件，替换所有triple中的关系谓词
        
        Args:
            input_file: 输入的JSON文件路径
            output_file: 输出的JSON文件路径（如果为None，则自动生成）
        """
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_relations_replaced.json"
        
        print(f"读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"处理 {len(data)} 条记录...")
        
        # 处理每条记录
        for item in tqdm(data, desc="处理记录"):
            if 'context_triples' in item:
                for triple_obj in item['context_triples']:
                    if 'triple' in triple_obj:
                        self.stats['total_triples'] += 1
                        original_triple = triple_obj['triple']
                        new_triple = self.replace_relation_in_triple(original_triple)
                        triple_obj['triple'] = new_triple
        
        # 保存结果
        print(f"保存结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        self.print_stats()
        
        return output_file
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("统计信息:")
        print(f"  总triple数: {self.stats['total_triples']}")
        print(f"  替换的triple数: {self.stats['replaced_triples']}")
        print(f"  未替换的triple数: {self.stats['total_triples'] - self.stats['replaced_triples']}")
        
        if self.stats['not_found_relations']:
            print(f"\n  未找到映射的关系词数: {len(self.stats['not_found_relations'])}")
            print("  前10个未找到的关系词:")
            sorted_not_found = sorted(self.stats['not_found_relations'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
            for relation, count in sorted_not_found:
                print(f"    {relation}: {count} 次")
        
        print("="*50)
    
    def save_stats(self, stats_file='replacement_stats.json'):
        """保存统计信息到文件"""
        stats_data = {
            'total_triples': self.stats['total_triples'],
            'replaced_triples': self.stats['replaced_triples'],
            'not_replaced_triples': self.stats['total_triples'] - self.stats['replaced_triples'],
            'not_found_relations': dict(self.stats['not_found_relations']),
            'top_relations': dict(sorted(self.stats['relation_counts'].items(), 
                                       key=lambda x: x[1], reverse=True)[:20])
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存到: {stats_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='替换triple文件中的关系谓词')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入的JSON文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出的JSON文件路径（默认自动生成）')
    parser.add_argument('--alignment', '-a', type=str, default=None,
                        help='relation_alignment.json文件路径（默认自动查找）')
    parser.add_argument('--relation2id', '-r', type=str, default=None,
                        help='relation2id_deduplicated.txt文件路径（可选）')
    
    args = parser.parse_args()
    
    # 创建替换器
    replacer = TripleRelationReplacer(
        alignment_file=args.alignment,
        relation2id_file=args.relation2id
    )
    
    # 处理文件
    output_file = replacer.process_json_file(args.input, args.output)
    
    # 保存统计信息
    stats_file = os.path.join(os.path.dirname(output_file), 'replacement_stats.json')
    replacer.save_stats(stats_file)
    
    print(f"\n完成！输出文件: {output_file}")


if __name__ == '__main__':
    main()






