import json
import re
from collections import OrderedDict

def parse_triple(triple_str):
    """
    解析三元组字符串，格式: "(头实体, 关系, 尾实体)"
    返回: (头实体, 关系, 尾实体)
    """
    # 移除首尾的括号
    triple_str = triple_str.strip()
    if triple_str.startswith('(') and triple_str.endswith(')'):
        triple_str = triple_str[1:-1]
    
    # 分割三元组，考虑逗号可能出现在实体名称中
    # 使用正则表达式来更准确地分割
    parts = []
    current = ''
    paren_count = 0
    
    for char in triple_str:
        if char == ',' and paren_count == 0:
            parts.append(current.strip())
            current = ''
        else:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            current += char
    
    if current:
        parts.append(current.strip())
    
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        return None, None, None

def extract_entities_and_relations(json_file):
    """
    从JSON文件中提取所有实体和关系
    """
    print(f"正在读取文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共有 {len(data)} 条记录")
    
    # 使用OrderedDict保持插入顺序并去重
    entities = OrderedDict()
    relations = OrderedDict()
    
    # 遍历所有记录
    for idx, item in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"已处理 {idx + 1} 条记录...")
        
        # 处理 context_triples
        if 'context_triples' in item and item['context_triples']:
            for triple_obj in item['context_triples']:
                if 'triple' in triple_obj:
                    head, relation, tail = parse_triple(triple_obj['triple'])
                    if head and relation and tail:
                        entities[head] = None
                        entities[tail] = None
                        relations[relation] = None
        
        # 处理 gpt_sentence_triples
        if 'gpt_sentence_triples' in item and item['gpt_sentence_triples']:
            for triple_obj in item['gpt_sentence_triples']:
                if 'triple' in triple_obj:
                    head, relation, tail = parse_triple(triple_obj['triple'])
                    if head and relation and tail:
                        entities[head] = None
                        entities[tail] = None
                        relations[relation] = None
    
    print(f"\n提取完成!")
    print(f"总计实体数量: {len(entities)}")
    print(f"总计关系数量: {len(relations)}")
    
    return list(entities.keys()), list(relations.keys())

def save_to_file(items, filename):
    """
    保存实体或关系到文件，格式: 项目\t编号
    """
    print(f"正在保存到 {filename}...")
    
    with open(filename, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(items):
            f.write(f"{item}\t{idx}\n")
    
    print(f"已保存 {len(items)} 条记录到 {filename}")

def main():
    # 输入文件路径
    input_file = 'hotpot_dev_with_triples_aligned.json'
    
    # 输出文件路径
    entity_file = 'entity2id.txt'
    relation_file = 'relation2id.txt'
    
    # 提取实体和关系
    entities, relations = extract_entities_and_relations(input_file)
    
    # 保存到文件
    save_to_file(entities, entity_file)
    save_to_file(relations, relation_file)
    
    print("\n处理完成!")
    print(f"实体文件: {entity_file}")
    print(f"关系文件: {relation_file}")

if __name__ == '__main__':
    main()

