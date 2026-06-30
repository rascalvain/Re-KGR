import json
import os
from typing import List, Dict

"""
筛选脚本：移除 context_triples 和 gpt_sentence_triples 都为空的数据
"""

INPUT_FILE = "hotpot_dev_merged_triples.json"
OUTPUT_FILE = "hotpot_dev_merged_triples_filtered.json"

def is_empty_triples(triples: List[Dict]) -> bool:
    """
    检查三元组列表是否为空
    空的情况包括：
    1. None
    2. 空列表 []
    3. 列表中的元素都是空字典或无效数据
    """
    if not triples:
        return True
    
    # 检查是否所有元素都是空的或无效的
    for item in triples:
        if isinstance(item, dict):
            # 检查是否有有效的 triple 字段
            triple = item.get('triple', '')
            if triple and triple.strip():
                return False
        elif isinstance(item, str) and item.strip():
            return False
        elif item:  # 其他非空值
            return False
    
    return True


def filter_empty_records(data: List[Dict]) -> tuple:
    """
    筛选掉 context_triples 和 gpt_sentence_triples 都为空的数据
    
    返回: (筛选后的数据, 统计信息)
    """
    filtered_data = []
    stats = {
        'total': len(data),
        'removed': 0,
        'kept': 0,
        'empty_context_only': 0,  # 只有 context_triples 为空
        'empty_gpt_only': 0,      # 只有 gpt_sentence_triples 为空
        'both_empty': 0,          # 两者都为空
        'both_non_empty': 0       # 两者都不为空
    }
    
    for record in data:
        context_triples = record.get('context_triples', [])
        gpt_triples = record.get('gpt_sentence_triples', [])
        
        is_context_empty = is_empty_triples(context_triples)
        is_gpt_empty = is_empty_triples(gpt_triples)
        
        # 统计
        if is_context_empty and is_gpt_empty:
            stats['both_empty'] += 1
            stats['removed'] += 1
            # 不添加到结果中
        else:
            # 保留这条记录
            filtered_data.append(record)
            stats['kept'] += 1
            
            if is_context_empty and not is_gpt_empty:
                stats['empty_context_only'] += 1
            elif not is_context_empty and is_gpt_empty:
                stats['empty_gpt_only'] += 1
            else:
                stats['both_non_empty'] += 1
    
    return filtered_data, stats


def main():
    """主函数"""
    print("=" * 70)
    print("🔍 三元组数据筛选工具")
    print("=" * 70)
    
    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 错误: 输入文件不存在: {INPUT_FILE}")
        return
    
    # 读取数据
    print(f"\n📂 读取文件: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    print(f"📊 原始数据: {len(data)} 条记录")
    
    # 筛选数据
    print(f"\n🔍 开始筛选...")
    filtered_data, stats = filter_empty_records(data)
    
    # 打印统计信息
    print(f"\n{'=' * 70}")
    print(f"📊 筛选统计:")
    print(f"   - 总记录数: {stats['total']}")
    print(f"   - 保留记录: {stats['kept']} ({stats['kept']/stats['total']*100:.2f}%)")
    print(f"   - 移除记录: {stats['removed']} ({stats['removed']/stats['total']*100:.2f}%)")
    print(f"\n   详细分类:")
    print(f"   - 两者都为空（已移除）: {stats['both_empty']}")
    print(f"   - 两者都不为空: {stats['both_non_empty']}")
    print(f"   - 仅 context_triples 为空: {stats['empty_context_only']}")
    print(f"   - 仅 gpt_sentence_triples 为空: {stats['empty_gpt_only']}")
    print(f"{'=' * 70}")
    
    # 保存结果
    print(f"\n💾 保存筛选后的数据到: {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 保存成功！")
        print(f"📊 最终数据: {len(filtered_data)} 条记录")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return
    
    print(f"\n{'=' * 70}")
    print(f"✅ 筛选完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()







