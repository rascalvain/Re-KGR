#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从entity2id.txt中提取匹配成功的实体，保存到match_entity.txt文件
"""

import os

def extract_matched_entities(input_file, output_file):
    """
    从映射结果文件中提取匹配成功的实体
    
    Args:
        input_file (str): 输入的映射结果文件路径
        output_file (str): 输出的匹配成功实体文件路径
    """
    matched_entities = []
    unmatched_count = 0
    total_count = 0
    
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：文件 {input_file} 不存在")
            return
        
        # 读取映射结果文件
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释行和空行
                if line.startswith('#') or not line:
                    continue
                
                # 解析每行数据
                parts = line.split('\t')
                if len(parts) >= 5:
                    entity_name = parts[0]
                    wikidata_id = parts[1]
                    label = parts[2]
                    description = parts[3]
                    similarity = parts[4]
                    
                    total_count += 1
                    
                    # 检查是否为匹配成功的实体
                    if wikidata_id != 'NO_MATCH':
                        matched_entities.append({
                            'entity_name': entity_name,
                            'wikidata_id': wikidata_id,
                            'label': label,
                            'description': description,
                            'similarity': similarity
                        })
                    else:
                        unmatched_count += 1
        
        # 写入匹配成功的实体文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 匹配成功的实体列表\n")
            f.write("# Format: entity_name\twikidata_id\tlabel\tdescription\tsimilarity_score\n")
            f.write(f"# 总共 {len(matched_entities)} 个匹配成功实体\n")
            f.write(f"# 匹配成功: {len(matched_entities)} 个\n")
            f.write(f"# 匹配失败: {unmatched_count} 个\n")
            f.write(f"# 成功率: {(len(matched_entities)/total_count)*100:.2f}%\n")
            f.write("\n")
            
            # 写入匹配成功的实体信息
            for entity in matched_entities:
                f.write(f"{entity['entity_name']}\t{entity['wikidata_id']}\t"
                       f"{entity['label']}\t{entity['description']}\t{entity['similarity']}\n")
        
        # 输出统计信息
        print(f"处理完成！")
        print(f"总实体数量: {total_count}")
        print(f"匹配成功: {len(matched_entities)} 个")
        print(f"匹配失败: {unmatched_count} 个")
        print(f"成功率: {(len(matched_entities)/total_count)*100:.2f}%")
        print(f"匹配成功实体已保存到: {output_file}")
        
        return matched_entities
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return []

def main():
    """主函数"""
    # 输入和输出文件路径
    input_file = "output/entity2id.txt"
    output_file = "output/match_entity.txt"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到映射结果文件 {input_file}")
        print("请先运行 mapping2wiki.py 生成映射结果")
        return
    
    # 提取匹配成功实体
    matched = extract_matched_entities(input_file, output_file)
    
    if matched:
        print(f"\n前10个匹配成功实体示例:")
        for i, entity in enumerate(matched[:10], 1):
            print(f"{i:2d}. {entity['entity_name']} -> {entity['wikidata_id']} ({entity['label']})")
        
        if len(matched) > 10:
            print(f"... 还有 {len(matched) - 10} 个匹配成功实体")
        
        # 按相似度排序显示最高匹配度的实体
        print(f"\n相似度最高的前5个实体:")
        sorted_entities = sorted(matched, key=lambda x: float(x['similarity']), reverse=True)
        for i, entity in enumerate(sorted_entities[:5], 1):
            print(f"{i}. {entity['entity_name']} -> {entity['wikidata_id']} "
                  f"({entity['label']}) - 相似度: {entity['similarity']}")

if __name__ == "__main__":
    main()








