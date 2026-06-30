import json
import os

"""
诊断脚本：检查为什么三元组字段变空
"""

INPUT_FILE = "hotpot_dev_merged_triples_filtered.json"
OUTPUT_FILE = "hotpot_dev_merged_triples_aligned.json"

def diagnose():
    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        return
    
    # 检查输出文件
    if not os.path.exists(OUTPUT_FILE):
        print(f"❌ 输出文件不存在: {OUTPUT_FILE}")
        return
    
    print("=" * 70)
    print("🔍 三元组字段诊断工具")
    print("=" * 70)
    
    # 读取文件
    print(f"\n📂 读取输入文件: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    print(f"📂 读取输出文件: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    
    print(f"\n📊 数据统计:")
    print(f"   输入记录数: {len(input_data)}")
    print(f"   输出记录数: {len(output_data)}")
    
    # 统计
    stats = {
        'input_empty_context': 0,
        'input_empty_gpt': 0,
        'output_empty_context': 0,
        'output_empty_gpt': 0,
        'became_empty_context': 0,
        'became_empty_gpt': 0
    }
    
    # 详细分析
    became_empty_records = []
    
    for i in range(min(len(input_data), len(output_data))):
        input_record = input_data[i]
        output_record = output_data[i]
        record_id = input_record.get('_id', f'index_{i}')
        
        # 检查 context_triples
        input_context = input_record.get('context_triples', [])
        output_context = output_record.get('context_triples', [])
        
        if not input_context:
            stats['input_empty_context'] += 1
        if not output_context:
            stats['output_empty_context'] += 1
        if input_context and not output_context:
            stats['became_empty_context'] += 1
            became_empty_records.append({
                'id': record_id,
                'index': i,
                'field': 'context_triples',
                'input_count': len(input_context),
                'output_count': 0
            })
        
        # 检查 gpt_sentence_triples
        input_gpt = input_record.get('gpt_sentence_triples', [])
        output_gpt = output_record.get('gpt_sentence_triples', [])
        
        if not input_gpt:
            stats['input_empty_gpt'] += 1
        if not output_gpt:
            stats['output_empty_gpt'] += 1
        if input_gpt and not output_gpt:
            stats['became_empty_gpt'] += 1
            became_empty_records.append({
                'id': record_id,
                'index': i,
                'field': 'gpt_sentence_triples',
                'input_count': len(input_gpt),
                'output_count': 0
            })
    
    # 打印统计
    print(f"\n📊 详细统计:")
    print(f"\n  context_triples:")
    print(f"    - 输入文件中为空: {stats['input_empty_context']}")
    print(f"    - 输出文件中为空: {stats['output_empty_context']}")
    print(f"    - 处理后变空: {stats['became_empty_context']}")
    
    print(f"\n  gpt_sentence_triples:")
    print(f"    - 输入文件中为空: {stats['input_empty_gpt']}")
    print(f"    - 输出文件中为空: {stats['output_empty_gpt']}")
    print(f"    - 处理后变空: {stats['became_empty_gpt']}")
    
    # 显示变空的记录
    if became_empty_records:
        print(f"\n⚠️ 发现 {len(became_empty_records)} 个字段在处理后变空:")
        print(f"\n前 10 个案例:")
        for record in became_empty_records[:10]:
            print(f"  - 记录 [{record['index']}] {record['id']}")
            print(f"    字段: {record['field']}")
            print(f"    输入: {record['input_count']} 个三元组 → 输出: {record['output_count']} 个")
            
            # 显示具体内容
            if record['index'] < len(input_data):
                input_rec = input_data[record['index']]
                field_data = input_rec.get(record['field'], [])
                if field_data:
                    print(f"    输入内容示例:")
                    for j, item in enumerate(field_data[:3]):
                        print(f"      {j+1}. {item.get('triple', 'N/A')}")
            print()
    else:
        print(f"\n✅ 没有发现字段在处理后变空的情况")
    
    print("=" * 70)

if __name__ == "__main__":
    diagnose()

