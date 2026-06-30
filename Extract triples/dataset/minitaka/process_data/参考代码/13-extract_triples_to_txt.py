import json
from tqdm import tqdm

def parse_triple(triple_str):
    """
    解析三元组字符串 "(subject, relation, object)"
    返回 (subject, relation, object)
    """
    # 移除外层括号
    triple_str = triple_str.strip()
    if triple_str.startswith('(') and triple_str.endswith(')'):
        triple_str = triple_str[1:-1]
    
    # 分割三元组，注意处理可能包含逗号的实体
    parts = []
    current = []
    paren_count = 0
    
    for char in triple_str:
        if char == '(':
            paren_count += 1
            current.append(char)
        elif char == ')':
            paren_count -= 1
            current.append(char)
        elif char == ',' and paren_count == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        parts.append(''.join(current).strip())
    
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        # 解析失败，返回None
        return None, None, None


def extract_triples_to_txt(json_file, output_txt_file):
    """
    从JSON文件中提取context_triples到txt文件
    格式: head\ttail\trelation
    """
    print(f"正在加载文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共有 {len(data)} 条数据")
    
    # 统计信息
    total_triples = 0
    success_count = 0
    failed_count = 0
    
    # 提取三元组
    print(f"正在提取三元组...")
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("head\ttail\trelation\n")
        
        for item in tqdm(data, desc="处理数据"):
            if 'context_triples' in item:
                for triple_obj in item['context_triples']:
                    if 'triple' in triple_obj:
                        total_triples += 1
                        triple_str = triple_obj['triple']
                        
                        # 解析三元组
                        subject, relation, obj = parse_triple(triple_str)
                        
                        if subject is not None:
                            # 写入格式: head\ttail\trelation
                            f.write(f"{subject}\t{obj}\t{relation}\n")
                            success_count += 1
                        else:
                            failed_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("提取完成！统计信息：")
    print("=" * 70)
    print(f"总三元组数:   {total_triples}")
    print(f"成功提取数:   {success_count}")
    print(f"解析失败数:   {failed_count}")
    if total_triples > 0:
        print(f"成功率:       {success_count/total_triples*100:.2f}%")
    print("=" * 70)
    
    return success_count, failed_count


def main():
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    input_file = os.path.join(script_dir, 'data/hotpot_dev_with_triples_aligned_cleaned_relations_replaced.json')
    output_file = os.path.join(script_dir, 'triples.txt')
    
    print("=" * 70)
    print("从JSON数据集提取context_triples到TXT文件")
    print("=" * 70)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"\n错误: 找不到输入文件 {input_file}")
        return
    
    # 提取三元组
    print(f"\n开始提取...")
    success_count, failed_count = extract_triples_to_txt(input_file, output_file)
    
    print(f"\n完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"成功提取: {success_count} 个三元组")
    
    # 显示前几行作为示例
    print(f"\n输出文件前5行示例:")
    print("-" * 70)
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 6:  # 表头 + 5行数据
                break
            print(line.rstrip())
    print("-" * 70)


if __name__ == "__main__":
    main()





