#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从entity2id.txt中提取未匹配的实体，保存到unmatched.txt文件
"""

import os

def extract_unmatched_entities(input_file, output_file):
    """
    从映射结果文件中提取未匹配的实体
    
    Args:
        input_file (str): 输入的映射结果文件路径
        output_file (str): 输出的未匹配实体文件路径
    """
    unmatched_entities = []
    matched_count = 0
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
                if len(parts) >= 2:
                    entity_name = parts[0]
                    wikidata_id = parts[1]
                    
                    total_count += 1
                    
                    # 检查是否为未匹配的实体
                    if wikidata_id == 'NO_MATCH':
                        unmatched_entities.append(entity_name)
                    else:
                        matched_count += 1
        
        # 写入未匹配实体文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 未匹配到Wikidata ID的实体列表\n")
            f.write(f"# 总共 {len(unmatched_entities)} 个未匹配实体\n")
            f.write(f"# 匹配成功: {matched_count} 个\n")
            f.write(f"# 匹配失败: {len(unmatched_entities)} 个\n")
            f.write(f"# 成功率: {(matched_count/total_count)*100:.2f}%\n")
            f.write("\n")
            
            # 写入未匹配的实体名称
            for entity in unmatched_entities:
                f.write(f"{entity}\n")
        
        # 输出统计信息
        print(f"处理完成！")
        print(f"总实体数量: {total_count}")
        print(f"匹配成功: {matched_count} 个")
        print(f"匹配失败: {len(unmatched_entities)} 个")
        print(f"成功率: {(matched_count/total_count)*100:.2f}%")
        print(f"未匹配实体已保存到: {output_file}")
        
        return unmatched_entities
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return []

def main():
    """主函数"""
    # 输入和输出文件路径
    input_file = "output/entity2id.txt"
    output_file = "output/unmatched.txt"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到映射结果文件 {input_file}")
        print("请先运行 mapping2wiki.py 生成映射结果")
        return
    
    # 提取未匹配实体
    unmatched = extract_unmatched_entities(input_file, output_file)
    
    if unmatched:
        print(f"\n前10个未匹配实体示例:")
        for i, entity in enumerate(unmatched[:10], 1):
            print(f"{i:2d}. {entity}")
        
        if len(unmatched) > 10:
            print(f"... 还有 {len(unmatched) - 10} 个未匹配实体")

if __name__ == "__main__":
    main()








