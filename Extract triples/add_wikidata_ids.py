#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
为output_PHD.json中的每条数据添加wikidata_id属性
该属性包含三元组中所有匹配到Wikidata的实体ID
"""

import json
import re
from collections import defaultdict

def load_entity_mappings(match_file):
    """
    从match_entity.txt文件加载实体到Wikidata ID的映射
    
    Args:
        match_file (str): 匹配实体文件路径
    
    Returns:
        dict: 实体名到Wikidata ID的映射字典
    """
    entity_to_wikidata = {}
    
    try:
        with open(match_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释行和空行
                if line.startswith('#') or not line:
                    continue
                
                # 解析每行数据: 实体名\tWikidata_ID\t标签\t描述\t相似度
                parts = line.split('\t')
                if len(parts) >= 2:
                    entity_name = parts[0]
                    wikidata_id = parts[1]
                    entity_to_wikidata[entity_name] = wikidata_id
        
        print(f"成功加载 {len(entity_to_wikidata)} 个实体映射")
        return entity_to_wikidata
        
    except Exception as e:
        print(f"加载实体映射时出错: {str(e)}")
        return {}

def parse_triple(triple_str):
    """
    解析三元组字符串，提取头实体和尾实体
    
    Args:
        triple_str (str): 三元组字符串，格式: "(头实体, 关系, 尾实体)"
    
    Returns:
        list: 提取到的实体列表
    """
    entities = []
    
    # 去除括号
    triple_str = triple_str.strip()
    if triple_str.startswith('(') and triple_str.endswith(')'):
        triple_str = triple_str[1:-1]
    
    # 尝试按逗号分割，但要处理可能包含逗号的实体名
    parts = []
    current_part = ""
    paren_count = 0
    quote_count = 0
    
    for char in triple_str:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == '"':
            quote_count = (quote_count + 1) % 2
        elif char == ',' and paren_count == 0 and quote_count == 0:
            parts.append(current_part.strip())
            current_part = ""
            continue
        current_part += char
    
    # 添加最后一部分
    if current_part:
        parts.append(current_part.strip())
    
    # 提取头实体和尾实体（跳过中间的关系）
    if len(parts) >= 3:
        head_entity = parts[0].strip()
        tail_entity = parts[2].strip()
        
        # 清理实体名称（去除多余的引号等）
        head_entity = clean_entity_name(head_entity)
        tail_entity = clean_entity_name(tail_entity)
        
        entities.extend([head_entity, tail_entity])
    
    return entities

def clean_entity_name(entity_name):
    """
    清理实体名称，去除引号和多余的空格
    
    Args:
        entity_name (str): 原始实体名称
    
    Returns:
        str: 清理后的实体名称
    """
    # 去除引号
    entity_name = entity_name.strip().strip('"').strip("'")
    # 合并多个空格
    entity_name = re.sub(r'\s+', ' ', entity_name)
    return entity_name.strip()

def find_wikidata_ids_for_data(data_item, entity_mappings):
    """
    为单条数据找到所有匹配的Wikidata ID
    
    Args:
        data_item (dict): 单条数据
        entity_mappings (dict): 实体映射字典
    
    Returns:
        list: 匹配到的Wikidata ID列表
    """
    wikidata_ids = []
    all_entities = set()
    
    # 添加主实体
    if 'entity' in data_item:
        main_entity = clean_entity_name(data_item['entity'])
        all_entities.add(main_entity)
    
    # 从triples中提取实体
    if 'triples' in data_item:
        for triple_obj in data_item['triples']:
            if 'triple' in triple_obj:
                triple_str = triple_obj['triple']
                entities = parse_triple(triple_str)
                all_entities.update(entities)
    
    # 查找匹配的Wikidata ID
    for entity in all_entities:
        if entity in entity_mappings:
            wikidata_id = entity_mappings[entity]
            if wikidata_id not in wikidata_ids:
                wikidata_ids.append(wikidata_id)
    
    return wikidata_ids

def process_json_file(input_file, output_file, entity_mappings):
    """
    处理JSON文件，为每条数据添加wikidata_id属性
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
        entity_mappings (dict): 实体映射字典
    """
    try:
        # 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_count = 0
        total_wikidata_ids = 0
        
        # 处理每个数据集
        for dataset_name, dataset_items in data.items():
            print(f"处理数据集: {dataset_name}")
            
            for i, item in enumerate(dataset_items):
                # 找到匹配的Wikidata ID
                wikidata_ids = find_wikidata_ids_for_data(item, entity_mappings)
                
                # 添加wikidata_id属性
                item['wikidata_id'] = wikidata_ids
                
                processed_count += 1
                total_wikidata_ids += len(wikidata_ids)
                
                # 显示处理进度
                if (i + 1) % 100 == 0:
                    print(f"  已处理 {i + 1} 条数据...")
        
        # 写入更新后的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n处理完成！")
        print(f"总共处理了 {processed_count} 条数据")
        print(f"总共添加了 {total_wikidata_ids} 个Wikidata ID")
        print(f"平均每条数据包含 {total_wikidata_ids/processed_count:.2f} 个Wikidata ID")
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"处理JSON文件时出错: {str(e)}")

def main():
    """主函数"""
    # 文件路径
    match_file = "output/match_entity.txt"
    input_file = "output/output_PHD.json"
    output_file = "output/output_PHD_with_wikidata.json"
    
    print("开始处理...")
    
    # 加载实体映射
    print("1. 加载实体映射...")
    entity_mappings = load_entity_mappings(match_file)
    
    if not entity_mappings:
        print("错误：无法加载实体映射，请检查文件是否存在")
        return
    
    # 处理JSON文件
    print("2. 处理JSON文件...")
    process_json_file(input_file, output_file, entity_mappings)
    
    print("\n处理完毕！")

if __name__ == "__main__":
    main()








