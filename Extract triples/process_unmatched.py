#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简化版的未匹配实体提取脚本
"""

def process_unmatched_entities():
    """处理entity2id.txt文件，提取未匹配的实体"""
    
    input_file = "output/entity2id.txt"
    output_file = "output/unmatched.txt"
    
    unmatched_entities = []
    matched_count = 0
    total_count = 0
    
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
    
    # 显示前10个未匹配实体示例
    if unmatched_entities:
        print(f"\n前10个未匹配实体示例:")
        for i, entity in enumerate(unmatched_entities[:10], 1):
            print(f"{i:2d}. {entity}")
        
        if len(unmatched_entities) > 10:
            print(f"... 还有 {len(unmatched_entities) - 10} 个未匹配实体")

if __name__ == "__main__":
    process_unmatched_entities()








