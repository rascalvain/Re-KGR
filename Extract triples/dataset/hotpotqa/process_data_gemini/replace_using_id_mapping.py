#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用id_mapping.json进行关系词替换
通过ID映射构建关系词到关系词的映射
"""

import json
import os
import re
from tqdm import tqdm
from collections import defaultdict

class RelationReplacerFromIdMapping:
    """从ID映射构建关系词替换器"""
    
    def __init__(self, id_mapping_file, original_relation2id_file, deduplicated_relation2id_file=None):
        """
        初始化
        
        Args:
            id_mapping_file: id_mapping.json文件路径（旧ID -> 新ID的映射）
            original_relation2id_file: 原始的relation2id.txt文件路径
            deduplicated_relation2id_file: 去重后的relation2id_deduplicated.txt文件路径（可选）
        """
        self.relation_mapping = {}  # 关系词到主关系词的映射
        self.stats = {
            'total_triples': 0,
            'replaced_triples': 0,
            'not_found_relations': defaultdict(int),
            'relation_counts': defaultdict(int)
        }
        
        # 加载映射
        self.load_mappings(id_mapping_file, original_relation2id_file, deduplicated_relation2id_file)
    
    def load_relation2id(self, relation2id_file):
        """
        加载relation2id文件
        
        Returns:
            dict: {关系词: id}
            dict: {id: 关系词}
        """
        relation_to_id = {}
        id_to_relation = {}
        
        with open(relation2id_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    relation = parts[0]
                    rel_id = int(parts[1])
                    relation_to_id[relation] = rel_id
                    id_to_relation[rel_id] = relation
        
        return relation_to_id, id_to_relation
    
    def load_mappings(self, id_mapping_file, original_relation2id_file, deduplicated_relation2id_file):
        """加载所有映射并构建关系词替换映射"""
        print("加载映射文件...")
        
        # 1. 加载ID映射
        print(f"  1. 加载ID映射: {id_mapping_file}")
        if not os.path.exists(id_mapping_file):
            raise FileNotFoundError(f"未找到ID映射文件: {id_mapping_file}")
        
        with open(id_mapping_file, 'r', encoding='utf-8') as f:
            id_mapping = json.load(f)
        
        # 转换ID为整数（如果JSON中存储为字符串）
        id_mapping_int = {}
        for old_id, new_id in id_mapping.items():
            old_id_int = int(old_id) if isinstance(old_id, str) else old_id
            new_id_int = int(new_id) if isinstance(new_id, str) else new_id
            id_mapping_int[old_id_int] = new_id_int
        
        print(f"    加载了 {len(id_mapping_int)} 个ID映射")
        
        # 2. 加载原始关系词到ID的映射
        print(f"  2. 加载原始关系词: {original_relation2id_file}")
        if not os.path.exists(original_relation2id_file):
            raise FileNotFoundError(f"未找到原始关系词文件: {original_relation2id_file}")
        
        original_relation_to_id, original_id_to_relation = self.load_relation2id(original_relation2id_file)
        print(f"    加载了 {len(original_relation_to_id)} 个原始关系词")
        
        # 3. 加载去重后的关系词到ID的映射
        if deduplicated_relation2id_file and os.path.exists(deduplicated_relation2id_file):
            print(f"  3. 加载去重后的关系词: {deduplicated_relation2id_file}")
            deduplicated_relation_to_id, deduplicated_id_to_relation = self.load_relation2id(deduplicated_relation2id_file)
            print(f"    加载了 {len(deduplicated_relation_to_id)} 个去重后的关系词")
        else:
            # 如果没有去重后的文件，从ID映射推断
            print(f"  3. 从ID映射推断去重后的关系词")
            deduplicated_id_to_relation = {}
            # 获取所有唯一的新ID
            unique_new_ids = set(id_mapping_int.values())
            # 对于每个新ID，找到映射到它的第一个旧ID对应的关系词作为主关系词
            for new_id in unique_new_ids:
                # 找到映射到这个新ID的所有旧ID
                old_ids_for_new_id = [old_id for old_id, mapped_new_id in id_mapping_int.items() 
                                     if mapped_new_id == new_id]
                # 使用最小的旧ID对应的关系词作为主关系词
                if old_ids_for_new_id:
                    min_old_id = min(old_ids_for_new_id)
                    if min_old_id in original_id_to_relation:
                        deduplicated_id_to_relation[new_id] = original_id_to_relation[min_old_id]
        
        # 4. 构建关系词到主关系词的映射
        print("  4. 构建关系词替换映射...")
        for old_relation, old_id in original_relation_to_id.items():
            if old_id in id_mapping_int:
                new_id = id_mapping_int[old_id]
                if new_id in deduplicated_id_to_relation:
                    new_relation = deduplicated_id_to_relation[new_id]
                    self.relation_mapping[old_relation] = new_relation
                else:
                    # 如果新ID没有对应的关系词，保持原样
                    self.relation_mapping[old_relation] = old_relation
            else:
                # 如果旧ID不在映射中，保持原样
                self.relation_mapping[old_relation] = old_relation
        
        # 统计
        replaced_count = sum(1 for old, new in self.relation_mapping.items() if old != new)
        print(f"    构建了 {len(self.relation_mapping)} 个关系词映射")
        print(f"    其中 {replaced_count} 个关系词将被替换")
    
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
        
        print(f"\n读取文件: {input_file}")
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
        print(f"\n保存结果到: {output_file}")
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


def find_mapping_files(directory='.'):
    """自动查找映射文件"""
    id_mapping_file = None
    deduplicated_relation2id_file = None
    
    # 查找id_mapping.json
    possible_paths = [
        os.path.join(directory, 'id_mapping.json'),
        os.path.join(directory, '..', 'id_mapping.json'),
        os.path.join(directory, '..', '..', 'id_mapping.json'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            id_mapping_file = path
            break
    
    # 查找relation2id_deduplicated.txt
    possible_paths = [
        os.path.join(directory, 'relation2id_deduplicated.txt'),
        os.path.join(directory, '..', 'relation2id_deduplicated.txt'),
        os.path.join(directory, '..', '..', 'relation2id_deduplicated.txt'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            deduplicated_relation2id_file = path
            break
    
    return id_mapping_file, deduplicated_relation2id_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用id_mapping.json替换triple文件中的关系谓词')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入的JSON文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出的JSON文件路径（默认自动生成）')
    parser.add_argument('--id-mapping', '-m', type=str, default=None,
                        help='id_mapping.json文件路径（默认自动查找）')
    parser.add_argument('--original-relation2id', '-r', type=str, default='relation2id.txt',
                        help='原始relation2id.txt文件路径')
    parser.add_argument('--deduplicated-relation2id', '-d', type=str, default=None,
                        help='去重后的relation2id_deduplicated.txt文件路径（默认自动查找）')
    
    args = parser.parse_args()
    
    # 自动查找映射文件
    if args.id_mapping is None:
        id_mapping_file, deduplicated_file = find_mapping_files()
        if id_mapping_file:
            args.id_mapping = id_mapping_file
            print(f"自动找到id_mapping文件: {id_mapping_file}")
        if deduplicated_file and args.deduplicated_relation2id is None:
            args.deduplicated_relation2id = deduplicated_file
            print(f"自动找到去重后的relation2id文件: {deduplicated_file}")
    
    if args.id_mapping is None:
        print("错误: 未找到id_mapping.json文件！")
        print("请指定--id-mapping参数或确保文件在当前目录。")
        return 1
    
    # 创建替换器
    try:
        replacer = RelationReplacerFromIdMapping(
            id_mapping_file=args.id_mapping,
            original_relation2id_file=args.original_relation2id,
            deduplicated_relation2id_file=args.deduplicated_relation2id
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    
    # 处理文件
    output_file = replacer.process_json_file(args.input, args.output)
    
    # 保存统计信息
    stats_file = os.path.join(os.path.dirname(output_file), 'replacement_stats.json')
    replacer.save_stats(stats_file)
    
    print(f"\n完成！输出文件: {output_file}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())






