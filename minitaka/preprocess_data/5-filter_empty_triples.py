"""
Mintaka 数据集处理 — 第 4 步：过滤空三元组数据

过滤逻辑：
  - entity_triples（Wikidata 子图）或 gpt_triples（GPT 提取三元组）
    任意一方为空，即移除该条记录。
  - 仅两者均不为空时才保留。

输入：第 3 步输出 mintaka_dev_with_all_triples.json
输出：mintaka_dev_with_all_triples_filtered.json
"""

import json
import os
from typing import List, Dict, Any

INPUT_FILE  = "data/mintaka_dev_with_gpt_triples.json"
OUTPUT_FILE = "data/mintaka_dev_with_all_triples_filtered.json"


# ========================================================================== #
#  判断函数
# ========================================================================== #

def is_entity_triples_empty(entity_triples: Dict) -> bool:
    """判断 entity_triples 字典是否实质上为空。

    entity_triples 结构：
      {entity_id: {"label": ..., "triple_count": ..., "triples": [...]}, ...}
    """
    if not entity_triples:
        return True
    for info in entity_triples.values():
        triples = info.get("triples", [])
        if triples:          # 有至少一条有效三元组即不为空
            return False
    return True


def is_gpt_triples_empty(gpt_triples: List) -> bool:
    """判断 gpt_triples 列表是否实质上为空。

    gpt_triples 结构：
      [{"head": ..., "relation": ..., "tail": ..., ...}, ...]
    """
    if not gpt_triples:
        return True
    for t in gpt_triples:
        if isinstance(t, dict):
            if t.get("head") or t.get("relation") or t.get("tail"):
                return False
    return True


# ========================================================================== #
#  主过滤逻辑
# ========================================================================== #

def filter_empty_records(data: List[Dict]) -> tuple:
    """过滤 entity_triples 或 gpt_triples 任意一方为空的记录。

    两者均不为空才保留；任意一方为空即丢弃。
    返回: (filtered_data, stats)
    """
    stats = {
        "total": len(data),
        "kept": 0,
        "removed": 0,
        "both_empty": 0,            # 两者均为空
        "only_entity_empty": 0,     # 仅 entity_triples 为空
        "only_gpt_empty": 0,        # 仅 gpt_triples 为空
        "both_non_empty": 0,        # 两者均不为空 → 保留
    }

    filtered = []
    for record in data:
        ent_empty = is_entity_triples_empty(record.get("entity_triples", {}))
        gpt_empty = is_gpt_triples_empty(record.get("gpt_triples", []))

        if not ent_empty and not gpt_empty:
            filtered.append(record)
            stats["kept"] += 1
            stats["both_non_empty"] += 1
        else:
            stats["removed"] += 1
            if ent_empty and gpt_empty:
                stats["both_empty"] += 1
            elif ent_empty:
                stats["only_entity_empty"] += 1
            else:
                stats["only_gpt_empty"] += 1

    return filtered, stats


# ========================================================================== #
#  入口
# ========================================================================== #

def main():
    print("=" * 70)
    print("筛选工具：entity_triples 或 gpt_triples 任意为空则移除记录")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 输入文件不存在: {INPUT_FILE}")
        return

    print(f"\n读取文件: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始数据: {len(data)} 条")

    print("\n开始过滤...")
    filtered, stats = filter_empty_records(data)

    total = stats["total"]
    print(f"\n{'=' * 70}")
    print(f"过滤统计:")
    print(f"  总记录数:                     {total}")
    print(f"  保留记录（两者均不为空）:     {stats['kept']}  "
          f"({stats['kept'] / max(total, 1) * 100:.2f}%)")
    print(f"  移除记录:                     {stats['removed']}  "
          f"({stats['removed'] / max(total, 1) * 100:.2f}%)")
    print(f"\n  移除原因细分:")
    print(f"  - 两者均为空:                 {stats['both_empty']}")
    print(f"  - 仅 entity_triples 为空:     {stats['only_entity_empty']}")
    print(f"  - 仅 gpt_triples 为空:        {stats['only_gpt_empty']}")
    print(f"{'=' * 70}")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n保存成功: {OUTPUT_FILE}")
    print(f"最终数据: {len(filtered)} 条")
    print(f"\n{'=' * 70}")
    print("过滤完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
