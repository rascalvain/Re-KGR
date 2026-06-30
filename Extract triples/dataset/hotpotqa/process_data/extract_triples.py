import json

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

def extract_context_triples(json_file, output_file):
    """
    从JSON文件中提取所有context_triples并保存到文件
    格式: head \t tail \t relation
    """
    print(f"正在读取文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共有 {len(data)} 条记录")
    
    triples_list = []
    
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
                        triples_list.append((head, tail, relation))
    
    print(f"\n提取完成!")
    print(f"总计三元组数量: {len(triples_list)}")
    
    # 保存到文件
    print(f"正在保存到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for head, tail, relation in triples_list:
            f.write(f"{head}\t{tail}\t{relation}\n")
    
    print(f"已保存 {len(triples_list)} 条三元组到 {output_file}")
    
    return len(triples_list)

def main():
    # 输入文件路径
    input_file = 'hotpot_dev_with_triples_aligned.json'
    
    # 输出文件路径
    output_file = 'triples.txt'
    
    # 提取三元组
    total_triples = extract_context_triples(input_file, output_file)
    
    print("\n处理完成!")
    print(f"输出文件: {output_file}")
    print(f"总三元组数: {total_triples}")

if __name__ == '__main__':
    main()

