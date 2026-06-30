import json
from tqdm import tqdm

def is_valid_triple(triple_str):
    """
    检查三元组是否有效
    返回 (is_valid, reason)
    """
    if not triple_str:
        return False, "空字符串"
    
    triple_str = triple_str.strip()
    
    # 检查是否以括号开头和结尾
    if not triple_str.startswith('(') or not triple_str.endswith(')'):
        return False, "缺少括号"
    
    # 检查基本长度
    if len(triple_str) < 5:  # 至少需要 "(a,b,c)"
        return False, "长度过短"
    
    # 移除外层括号检查逗号数量
    inner = triple_str[1:-1]
    
    # 计算顶层逗号数量（不在括号内的逗号）
    paren_count = 0
    comma_count = 0
    
    for char in inner:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        elif char == ',' and paren_count == 0:
            comma_count += 1
    
    # 应该有恰好2个顶层逗号
    if comma_count < 2:
        return False, f"逗号不足({comma_count}个)"
    
    if comma_count > 2:
        return False, f"逗号过多({comma_count}个)"
    
    # 检查括号是否匹配
    if paren_count != 0:
        return False, "括号不匹配"
    
    # 尝试解析三部分
    parts = []
    current = []
    paren_count = 0
    
    for char in inner:
        if char == '(':
            paren_count += 1
            current.append(char)
        elif char == ')':
            paren_count -= 1
            current.append(char)
        elif char == ',' and paren_count == 0:
            part = ''.join(current).strip()
            if not part:  # 空部分
                return False, "包含空部分"
            parts.append(part)
            current = []
        else:
            current.append(char)
    
    if current:
        part = ''.join(current).strip()
        if not part:
            return False, "包含空部分"
        parts.append(part)
    
    if len(parts) != 3:
        return False, f"部分数量错误({len(parts)}个)"
    
    return True, "正常"


def clean_json_file(input_file, output_file):
    """
    清理JSON文件中的异常三元组
    """
    print(f"正在加载文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"共有 {len(data)} 条数据")
    
    # 统计信息
    stats = {
        'context_triples': {
            'original': 0,
            'removed': 0,
            'kept': 0,
            'reasons': {}
        },
        'gpt_sentence_triples': {
            'original': 0,
            'removed': 0,
            'kept': 0,
            'reasons': {}
        }
    }
    
    # 存储异常三元组示例
    abnormal_examples = []
    
    # 处理每条数据
    for item_idx, item in enumerate(tqdm(data, desc="清理数据")):
        # 处理 context_triples
        if 'context_triples' in item:
            original_triples = item['context_triples']
            cleaned_triples = []
            
            for triple_obj in original_triples:
                if 'triple' in triple_obj:
                    stats['context_triples']['original'] += 1
                    triple_str = triple_obj['triple']
                    
                    is_valid, reason = is_valid_triple(triple_str)
                    
                    if is_valid:
                        cleaned_triples.append(triple_obj)
                        stats['context_triples']['kept'] += 1
                    else:
                        stats['context_triples']['removed'] += 1
                        # 统计原因
                        if reason not in stats['context_triples']['reasons']:
                            stats['context_triples']['reasons'][reason] = 0
                        stats['context_triples']['reasons'][reason] += 1
                        
                        # 保存示例（最多10个）
                        if len(abnormal_examples) < 10:
                            abnormal_examples.append({
                                'item_idx': item_idx,
                                'field': 'context_triples',
                                'triple': triple_str,
                                'reason': reason
                            })
            
            item['context_triples'] = cleaned_triples
        
        # 处理 gpt_sentence_triples
        if 'gpt_sentence_triples' in item:
            original_triples = item['gpt_sentence_triples']
            cleaned_triples = []
            
            for triple_obj in original_triples:
                if 'triple' in triple_obj:
                    stats['gpt_sentence_triples']['original'] += 1
                    triple_str = triple_obj['triple']
                    
                    is_valid, reason = is_valid_triple(triple_str)
                    
                    if is_valid:
                        cleaned_triples.append(triple_obj)
                        stats['gpt_sentence_triples']['kept'] += 1
                    else:
                        stats['gpt_sentence_triples']['removed'] += 1
                        # 统计原因
                        if reason not in stats['gpt_sentence_triples']['reasons']:
                            stats['gpt_sentence_triples']['reasons'][reason] = 0
                        stats['gpt_sentence_triples']['reasons'][reason] += 1
                        
                        # 保存示例（最多10个）
                        if len(abnormal_examples) < 10:
                            abnormal_examples.append({
                                'item_idx': item_idx,
                                'field': 'gpt_sentence_triples',
                                'triple': triple_str,
                                'reason': reason
                            })
            
            item['gpt_sentence_triples'] = cleaned_triples
    
    # 保存清理后的文件
    print(f"\n正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print("\n" + "=" * 70)
    print("清理完成！统计信息：")
    print("=" * 70)
    
    total_original = 0
    total_removed = 0
    total_kept = 0
    
    for field_name, field_stats in stats.items():
        print(f"\n【{field_name}】")
        print(f"  原始三元组数: {field_stats['original']}")
        print(f"  保留三元组数: {field_stats['kept']}")
        print(f"  移除三元组数: {field_stats['removed']}")
        
        if field_stats['original'] > 0:
            removal_rate = field_stats['removed'] / field_stats['original'] * 100
            print(f"  移除比例: {removal_rate:.2f}%")
        
        if field_stats['reasons']:
            print(f"  移除原因统计:")
            for reason, count in sorted(field_stats['reasons'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"    - {reason}: {count} 个")
        
        total_original += field_stats['original']
        total_removed += field_stats['removed']
        total_kept += field_stats['kept']
    
    print(f"\n【总计】")
    print(f"  原始三元组总数: {total_original}")
    print(f"  保留三元组总数: {total_kept}")
    print(f"  移除三元组总数: {total_removed}")
    
    if total_original > 0:
        print(f"  总移除比例: {total_removed/total_original*100:.2f}%")
    
    # 显示异常示例
    if abnormal_examples:
        print(f"\n【异常三元组示例】")
        for i, example in enumerate(abnormal_examples, 1):
            print(f"\n示例 {i}:")
            print(f"  位置: 数据[{example['item_idx']}].{example['field']}")
            print(f"  内容: {example['triple'][:100]}{'...' if len(example['triple']) > 100 else ''}")
            print(f"  原因: {example['reason']}")
    
    print("=" * 70)


def main():
    import os
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, 'hotpot_dev_with_triples_aligned_filtered.json')
    output_file = os.path.join(script_dir, 'hotpot_dev_with_triples_aligned_cleaned.json')
    
    clean_json_file(input_file, output_file)
    
    print(f"\n✓ 清理完成！")
    print(f"✓ 原文件: {input_file}")
    print(f"✓ 新文件: {output_file}")


if __name__ == "__main__":
    main()

