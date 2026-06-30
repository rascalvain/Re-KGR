"""
Mintaka 数据集处理 — 第 5 步：清理异常三元组

针对两类三元组字段分别做清理：

  entity_triples（Wikidata 子图）
      结构：{entity_id: {"label":..., "triple_count":..., "triples":[{head,relation,tail}]}}
      清理规则：移除 head / relation / tail 任意一项为空字符串的三元组；
               同步更新 triple_count；移除清理后 triples 列表为空的 entity。

  gpt_triples（GPT 提取三元组）
      结构：[{"head":..., "relation":..., "tail":..., ...}]
      清理规则：移除 head / relation / tail 任意一项为空字符串的三元组；
               同步更新 gpt_triple_count / supported_triple_count。

输入：第 4 步输出 mintaka_dev_with_all_triples_filtered.json
输出：mintaka_dev_with_all_triples_cleaned.json
"""

import json
import os
from typing import Dict, List, Tuple, Any

INPUT_FILE  = "data/mintaka_dev_with_all_triples_filtered.json"
OUTPUT_FILE = "data/mintaka_dev_with_all_triples_cleaned.json"

# 三元组字段长度上下限（字符数）
MIN_FIELD_LEN = 1
MAX_FIELD_LEN = 500


# ========================================================================== #
#  单条三元组校验
# ========================================================================== #

def validate_triple_dict(t: Dict) -> Tuple[bool, str]:
    """校验结构化三元组字典 {head, relation, tail}。

    返回 (is_valid, reason)。
    """
    if not isinstance(t, dict):
        return False, "非字典类型"

    for field in ("head", "relation", "tail"):
        val = t.get(field, "")
        if not isinstance(val, str):
            return False, f"{field} 非字符串"
        val = val.strip()
        if len(val) < MIN_FIELD_LEN:
            return False, f"{field} 为空"
        if len(val) > MAX_FIELD_LEN:
            return False, f"{field} 过长({len(val)}字符)"

    return True, "正常"


# ========================================================================== #
#  字段级清理
# ========================================================================== #

def clean_entity_triples(entity_triples: Dict, field_stats: Dict,
                          examples: List, item_idx: int) -> Dict:
    """清理 entity_triples 字典，返回清理后的字典。"""
    cleaned_map = {}
    for eid, info in entity_triples.items():
        raw_triples = info.get("triples", [])
        kept = []
        for t in raw_triples:
            field_stats["original"] += 1
            valid, reason = validate_triple_dict(t)
            if valid:
                kept.append(t)
                field_stats["kept"] += 1
            else:
                field_stats["removed"] += 1
                field_stats["reasons"][reason] = field_stats["reasons"].get(reason, 0) + 1
                if len(examples) < 10:
                    examples.append({
                        "item_idx": item_idx,
                        "field": "entity_triples",
                        "entity_id": eid,
                        "triple": t,
                        "reason": reason,
                    })

        if kept:
            cleaned_map[eid] = {
                "label": info.get("label", eid),
                "triple_count": len(kept),
                "triples": kept,
            }
    return cleaned_map


def clean_gpt_triples(gpt_triples: List, field_stats: Dict,
                       examples: List, item_idx: int) -> List:
    """清理 gpt_triples 列表，返回清理后的列表。"""
    kept = []
    for t in gpt_triples:
        field_stats["original"] += 1
        valid, reason = validate_triple_dict(t)
        if valid:
            kept.append(t)
            field_stats["kept"] += 1
        else:
            field_stats["removed"] += 1
            field_stats["reasons"][reason] = field_stats["reasons"].get(reason, 0) + 1
            if len(examples) < 10:
                examples.append({
                    "item_idx": item_idx,
                    "field": "gpt_triples",
                    "triple": t,
                    "reason": reason,
                })
    return kept


# ========================================================================== #
#  主清理逻辑
# ========================================================================== #

def _empty_stats() -> Dict:
    return {"original": 0, "kept": 0, "removed": 0, "reasons": {}}


def clean_all(data: List[Dict]) -> Tuple[List[Dict], Dict, List]:
    stats = {
        "entity_triples": _empty_stats(),
        "gpt_triples":    _empty_stats(),
    }
    examples: List[Dict] = []

    for idx, item in enumerate(data):
        # --- entity_triples ---
        if "entity_triples" in item:
            item["entity_triples"] = clean_entity_triples(
                item["entity_triples"], stats["entity_triples"], examples, idx)
            # 同步 total_triple_count
            item["total_triple_count"] = sum(
                len(v.get("triples", []))
                for v in item["entity_triples"].values()
            )

        # --- gpt_triples ---
        if "gpt_triples" in item:
            cleaned_gpt = clean_gpt_triples(
                item["gpt_triples"], stats["gpt_triples"], examples, idx)
            item["gpt_triples"] = cleaned_gpt
            item["gpt_triple_count"] = len(cleaned_gpt)
            item["supported_triple_count"] = sum(
                1 for t in cleaned_gpt if t.get("has_wikidata_support", False))

    return data, stats, examples


# ========================================================================== #
#  入口
# ========================================================================== #

def main():
    print("=" * 70)
    print("清理工具：移除 entity_triples 与 gpt_triples 中的异常三元组")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 输入文件不存在: {INPUT_FILE}")
        return

    print(f"\n读取文件: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"数据量: {len(data)} 条")

    print("\n开始清理...")
    data, stats, examples = clean_all(data)

    # 打印统计
    print(f"\n{'=' * 70}")
    print("清理统计:")
    total_orig = total_kept = total_removed = 0
    for field_name, fs in stats.items():
        orig, kept, removed = fs["original"], fs["kept"], fs["removed"]
        total_orig    += orig
        total_kept    += kept
        total_removed += removed
        rate = removed / max(orig, 1) * 100
        print(f"\n  [{field_name}]")
        print(f"    原始三元组: {orig}")
        print(f"    保留三元组: {kept}")
        print(f"    移除三元组: {removed}  ({rate:.2f}%)")
        if fs["reasons"]:
            print(f"    移除原因:")
            for reason, cnt in sorted(fs["reasons"].items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"      - {reason}: {cnt}")

    print(f"\n  [总计]")
    print(f"    原始: {total_orig}  保留: {total_kept}  "
          f"移除: {total_removed}  ({total_removed / max(total_orig, 1) * 100:.2f}%)")

    if examples:
        print(f"\n  [异常三元组示例（最多10条）]")
        for i, ex in enumerate(examples, 1):
            t = ex.get("triple", {})
            t_str = f"({t.get('head','')}, {t.get('relation','')}, {t.get('tail','')})" \
                if isinstance(t, dict) else str(t)
            print(f"\n  示例 {i}: 数据[{ex['item_idx']}].{ex['field']}")
            print(f"    内容: {t_str[:120]}{'...' if len(t_str) > 120 else ''}")
            print(f"    原因: {ex['reason']}")

    print(f"\n{'=' * 70}")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n保存成功: {OUTPUT_FILE}")
    print(f"最终数据: {len(data)} 条")
    print(f"\n{'=' * 70}")
    print("清理完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
