import json
import re


def is_complete_triple(triple_str, return_details=False):
    """
    检查 triple 是否完整
    完整的 triple 应该：
    1. 以 '(' 开头，')' 结尾
    2. 包含恰好 3 个部分，用逗号分隔（主语、谓语、宾语）
    3. 每个部分都不为空
    
    参数：
    - triple_str: 三元组字符串
    - return_details: 是否返回详细信息（用于调试）
    
    返回：
    - 如果 return_details=False，返回 bool
    - 如果 return_details=True，返回 (bool, dict)，dict 包含详细信息
    """
    details = {
        'valid': False,
        'error': None,
        'parts_count': 0,
        'parts': []
    }
    
    # 检查基本格式
    if not triple_str.startswith('(') or not triple_str.endswith(')'):
        details['error'] = '格式错误：不是以括号包围'
        return (False, details) if return_details else False

    # 移除括号
    content = triple_str[1:-1].strip()
    
    if not content:
        details['error'] = '格式错误：括号内容为空'
        return (False, details) if return_details else False

    # 分割成多个部分（正确处理嵌套括号和引号中的逗号）
    parts = []
    current_part = ""
    paren_depth = 0
    quote_depth = 0

    for char in content:
        if char == '"':
            quote_depth = 1 - quote_depth
            current_part += char
        elif char == '(' or char == '[':
            paren_depth += 1
            current_part += char
        elif char == ')' or char == ']':
            paren_depth -= 1
            current_part += char
        elif char == ',' and paren_depth == 0 and quote_depth == 0:
            parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char

    # 添加最后一部分
    if current_part:
        parts.append(current_part.strip())

    details['parts_count'] = len(parts)
    details['parts'] = parts

    # 检查是否有恰好3个部分
    if len(parts) != 3:
        if len(parts) < 3:
            details['error'] = f'成分不足：只有 {len(parts)} 个成分，需要 3 个（主语、谓语、宾语）'
        else:
            details['error'] = f'成分过多：有 {len(parts)} 个成分，应该只有 3 个（主语、谓语、宾语）'
        return (False, details) if return_details else False

    # 检查每个部分都不为空
    for i, part in enumerate(parts):
        if not part or part.isspace():
            details['error'] = f'格式错误：第 {i+1} 个成分为空'
            return (False, details) if return_details else False

    details['valid'] = True
    return (True, details) if return_details else True


def filter_incomplete_triples(input_file, output_file, save_report=True):
    """
    读取 JSON 文件，过滤掉不完整的 triples，保存到新文件
    
    参数：
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    - save_report: 是否保存详细报告
    """
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_records = len(data)
    total_removed = 0
    total_kept = 0
    
    # 统计不同错误类型
    error_stats = {
        '成分不足': 0,
        '成分过多': 0,
        '格式错误': 0,
        '成分为空': 0
    }
    
    # 保存一些错误样例用于检查
    error_examples = {
        '成分不足': [],
        '成分过多': [],
        '格式错误': [],
        '成分为空': []
    }
    max_examples_per_type = 5

    print(f"总共有 {total_records} 条记录")

    # 遍历每条记录
    for idx, record in enumerate(data):
        if (idx + 1) % 100 == 0:
            print(f"处理进度: {idx + 1}/{total_records}")

        # 处理 context_triples
        if 'context_triples' in record:
            original_count = len(record['context_triples'])
            filtered_triples = []

            for triple_obj in record['context_triples']:
                triple_str = triple_obj.get('triple', '')
                is_valid, details = is_complete_triple(triple_str, return_details=True)
                
                if is_valid:
                    filtered_triples.append(triple_obj)
                else:
                    total_removed += 1
                    # 统计错误类型
                    error_type = '格式错误'
                    if '成分不足' in details['error']:
                        error_type = '成分不足'
                    elif '成分过多' in details['error']:
                        error_type = '成分过多'
                    elif '成分为空' in details['error']:
                        error_type = '成分为空'
                    
                    error_stats[error_type] += 1
                    
                    # 保存错误样例
                    if len(error_examples[error_type]) < max_examples_per_type:
                        error_examples[error_type].append({
                            'record_id': record.get('_id', 'N/A'),
                            'triple': triple_str,
                            'error': details['error'],
                            'parts_count': details['parts_count'],
                            'parts': details['parts']
                        })

            record['context_triples'] = filtered_triples
            total_kept += len(filtered_triples)

            removed = original_count - len(filtered_triples)
            if removed > 0:
                print(
                    f"  记录 {idx + 1} (_id: {record.get('_id', 'N/A')}): 移除了 {removed} 个不完整的 context_triples")

        # 处理 gpt_sentence_triples
        if 'gpt_sentence_triples' in record:
            original_count = len(record['gpt_sentence_triples'])
            filtered_triples = []

            for triple_obj in record['gpt_sentence_triples']:
                triple_str = triple_obj.get('triple', '')
                is_valid, details = is_complete_triple(triple_str, return_details=True)
                
                if is_valid:
                    filtered_triples.append(triple_obj)
                else:
                    total_removed += 1
                    # 统计错误类型
                    error_type = '格式错误'
                    if '成分不足' in details['error']:
                        error_type = '成分不足'
                    elif '成分过多' in details['error']:
                        error_type = '成分过多'
                    elif '成分为空' in details['error']:
                        error_type = '成分为空'
                    
                    error_stats[error_type] += 1
                    
                    # 保存错误样例
                    if len(error_examples[error_type]) < max_examples_per_type:
                        error_examples[error_type].append({
                            'record_id': record.get('_id', 'N/A'),
                            'triple': triple_str,
                            'error': details['error'],
                            'parts_count': details['parts_count'],
                            'parts': details['parts']
                        })

            record['gpt_sentence_triples'] = filtered_triples
            total_kept += len(filtered_triples)

            removed = original_count - len(filtered_triples)
            if removed > 0:
                print(
                    f"  记录 {idx + 1} (_id: {record.get('_id', 'N/A')}): 移除了 {removed} 个不完整的 gpt_sentence_triples")

    # 保存结果
    print(f"\n正在保存到文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"{'='*60}")
    print(f"总共移除了 {total_removed} 个不完整的 triples")
    print(f"保留了 {total_kept} 个完整的 triples")
    print(f"\n错误类型统计：")
    for error_type, count in error_stats.items():
        if count > 0:
            percentage = (count / total_removed * 100) if total_removed > 0 else 0
            print(f"  - {error_type}: {count} 个 ({percentage:.1f}%)")
    
    # 保存详细报告
    if save_report:
        report_file = output_file.replace('.json', '_report.json')
        report = {
            'summary': {
                'total_records': total_records,
                'total_removed': total_removed,
                'total_kept': total_kept,
                'error_stats': error_stats
            },
            'error_examples': error_examples
        }
        
        print(f"\n正在保存详细报告到: {report_file}")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n查看报告文件可以了解具体哪些三元组被过滤以及原因")
        
        # 打印一些错误样例
        print(f"\n{'='*60}")
        print(f"错误样例展示（每种类型最多显示 3 个）：")
        print(f"{'='*60}")
        for error_type, examples in error_examples.items():
            if examples:
                print(f"\n【{error_type}】:")
                for i, example in enumerate(examples[:3]):
                    print(f"\n  样例 {i+1}:")
                    print(f"    记录ID: {example['record_id']}")
                    print(f"    三元组: {example['triple']}")
                    print(f"    错误: {example['error']}")
                    print(f"    成分数量: {example['parts_count']}")
                    if example['parts']:
                        print(f"    成分内容:")
                        for j, part in enumerate(example['parts']):
                            print(f"      [{j+1}] {part}")


if __name__ == "__main__":
    input_file = "hotpot_dev_with_triples.json"
    output_file = "hotpot_dev_with_triples_filtered.json"

    filter_incomplete_triples(input_file, output_file)