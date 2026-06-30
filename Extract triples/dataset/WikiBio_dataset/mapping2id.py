import json
from collections import OrderedDict

def extract_entities_and_relations(json_file_path):
    """
    从JSON文件中提取所有实体和关系，并去重
    """
    entities = set()
    relations = set()
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 遍历所有记录
    for record in data:
        # 处理original字段
        if 'original' in record:
            for triple in record['original']:
                # 解析三元组格式: (实体1, 关系, 实体2)
                triple = triple.strip()
                if triple.startswith('(') and triple.endswith(')'):
                    # 去掉括号
                    triple_content = triple[1:-1]
                    # 分割成三部分
                    parts = []
                    current = ""
                    depth = 0
                    for char in triple_content:
                        if char == ',' and depth == 0:
                            parts.append(current.strip())
                            current = ""
                        else:
                            if char == '(':
                                depth += 1
                            elif char == ')':
                                depth -= 1
                            current += char
                    parts.append(current.strip())
                    
                    if len(parts) == 3:
                        head_entity = parts[0]
                        relation = parts[1]
                        tail_entity = parts[2]
                        
                        entities.add(head_entity)
                        entities.add(tail_entity)
                        relations.add(relation)
        
        # 处理wiki_ref字段
        if 'wiki_ref' in record:
            for triple in record['wiki_ref']:
                # 同样的处理逻辑
                triple = triple.strip()
                if triple.startswith('(') and triple.endswith(')'):
                    triple_content = triple[1:-1]
                    parts = []
                    current = ""
                    depth = 0
                    for char in triple_content:
                        if char == ',' and depth == 0:
                            parts.append(current.strip())
                            current = ""
                        else:
                            if char == '(':
                                depth += 1
                            elif char == ')':
                                depth -= 1
                            current += char
                    parts.append(current.strip())
                    
                    if len(parts) == 3:
                        head_entity = parts[0]
                        relation = parts[1]
                        tail_entity = parts[2]
                        
                        entities.add(head_entity)
                        entities.add(tail_entity)
                        relations.add(relation)
    
    return entities, relations

def save_to_file(items, output_file):
    """
    将实体或关系保存到文件，格式为：item\tid
    """
    # 排序以保持一致性
    sorted_items = sorted(list(items))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(sorted_items):
            f.write(f"{item}\t{idx}\n")
    
    print(f"已保存 {len(sorted_items)} 个项目到 {output_file}")

def main():
    # 输入文件路径
    json_file = r"g:\小论文\第三章\GCA-main\Extract triples\dataset\WikiBio_dataset\wikibio_with_triples.json"
    
    # 输出文件路径
    entity_output = r"g:\小论文\第三章\GCA-main\Extract triples\dataset\WikiBio_dataset\entity2id.txt"
    relation_output = r"g:\小论文\第三章\GCA-main\Extract triples\dataset\WikiBio_dataset\rel2id.txt"
    
    print("开始提取实体和关系...")
    
    # 提取实体和关系
    entities, relations = extract_entities_and_relations(json_file)
    
    print(f"\n提取完成!")
    print(f"总共找到 {len(entities)} 个唯一实体")
    print(f"总共找到 {len(relations)} 个唯一关系")
    
    # 保存到文件
    print("\n保存实体到文件...")
    save_to_file(entities, entity_output)
    
    print("\n保存关系到文件...")
    save_to_file(relations, relation_output)
    
    print("\n完成!")
    
    # 显示一些示例
    print("\n实体示例 (前10个):")
    for entity in sorted(list(entities))[:10]:
        print(f"  - {entity}")
    
    print("\n关系示例 (前10个):")
    for relation in sorted(list(relations))[:10]:
        print(f"  - {relation}")

if __name__ == "__main__":
    main()