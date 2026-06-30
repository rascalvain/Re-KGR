import json
import re
import os

def parse_triple(triple_str):
    """
    解析三元组字符串，提取头实体、关系和尾实体
    输入格式: "(头实体, 关系, 尾实体)"
    """
    # 去除括号并按逗号分割
    triple_str = triple_str.strip()
    if triple_str.startswith('(') and triple_str.endswith(')'):
        triple_str = triple_str[1:-1]  # 去除首尾括号
    
    # 分割三元组，处理可能包含逗号的实体名
    parts = []
    current_part = ""
    paren_count = 0
    
    for char in triple_str:
        if char == '(':
            paren_count += 1
            current_part += char
        elif char == ')':
            paren_count -= 1
            current_part += char
        elif char == ',' and paren_count == 0:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    
    # 添加最后一部分
    if current_part:
        parts.append(current_part.strip())
    
    # 如果分割结果不是3个部分，尝试其他方法
    if len(parts) != 3:
        # 使用正则表达式匹配
        match = re.match(r'\s*([^,]+),\s*([^,]+),\s*(.+)\s*', triple_str)
        if match:
            parts = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
    
    if len(parts) == 3:
        head_entity = parts[0].strip()
        relation = parts[1].strip()
        tail_entity = parts[2].strip()
        return head_entity, relation, tail_entity
    else:
        print(f"警告：无法解析三元组: {triple_str}")
        return None, None, None

def extract_entities_and_relations(json_file_path):
    """
    从JSON文件中提取所有实体和关系
    """
    entities = set()
    relations = set()
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计处理的三元组数量
    total_triples = 0
    successful_parses = 0
    
    # 遍历数据集
    for dataset_name, dataset_items in data.items():
        print(f"处理数据集: {dataset_name}")
        
        for item in dataset_items:
            if 'triples' in item:
                for triple_obj in item['triples']:
                    total_triples += 1
                    if 'triple' in triple_obj:
                        triple_str = triple_obj['triple']
                        head, relation, tail = parse_triple(triple_str)
                        
                        if head and relation and tail:
                            # 添加实体（头实体和尾实体）
                            entities.add(head)
                            entities.add(tail)
                            # 添加关系
                            relations.add(relation)
                            successful_parses += 1
    
    print(f"总共处理了 {total_triples} 个三元组")
    print(f"成功解析了 {successful_parses} 个三元组")
    print(f"提取到 {len(entities)} 个唯一实体")
    print(f"提取到 {len(relations)} 个唯一关系")
    
    return entities, relations

def save_to_files(entities, relations, output_dir="Extract triples/output"):
    """
    将实体和关系保存到txt文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实体到entity.txt
    entity_file = os.path.join(output_dir, "entity.txt")
    with open(entity_file, 'w', encoding='utf-8') as f:
        for entity in sorted(entities):
            f.write(entity + '\n')
    print(f"实体已保存到: {entity_file}")
    
    # 保存关系到rel.txt
    relation_file = os.path.join(output_dir, "rel.txt")
    with open(relation_file, 'w', encoding='utf-8') as f:
        for relation in sorted(relations):
            f.write(relation + '\n')
    print(f"关系已保存到: {relation_file}")


def main():
    """
    主函数 - 手动设置输入和输出路径
    """
    # 手动设置输入文件路径（修改这里的路径）
    input_file = r"g:\小论文\第三章\GCA-main\Extract triples\output\output_PHD.json"

    # 手动设置输出目录路径（修改这里的路径）
    output_dir = r"g:\小论文\第三章\GCA-main\Extract triples\output"

    print("开始处理三元组数据...")
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return

    # 检查输出目录是否存在，如不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    # 提取实体和关系
    entities, relations = extract_entities_and_relations(input_file)

    # 保存到文件
    save_to_files(entities, relations, output_dir)

    print("\n处理完成！")
    print(f"实体数量: {len(entities)}")
    print(f"关系数量: {len(relations)}")
    print(f"\n输出文件:")
    print(f"- 实体文件: {os.path.join(output_dir, 'entity.txt')}")
    print(f"- 关系文件: {os.path.join(output_dir, 'rel.txt')}")

if __name__ == "__main__":
    main()
