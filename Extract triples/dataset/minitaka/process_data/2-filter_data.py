"""
Mintaka 数据集处理 — 第 2 步：过滤无效 GPT 答案

过滤规则：
  1. gpt_final_answer 为 None 或空字符串
  2. gpt_sentence（推理文本）为 None 或空字符串
  3. gpt_final_answer 匹配过滤词（partial 模式：包含即过滤）

内置过滤词（可在 FILTER_WORDS 中扩展）：
  "i don't know", "i do not know", "unknown", "n/a", "not available",
  "cannot determine", "unable to determine", "no information" 等

输入：第 1 步输出 mintaka_dev_with_answers.json
输出：mintaka_dev_with_answers_filtered.json
"""

import json
import os

# ========================================================================== #
#  配置
# ========================================================================== #

INPUT_FILE  = "data/mintaka_dev_with_gpt_answers.json"
OUTPUT_FILE = "data/mintaka_dev_with_answers_filtered.json"

# 匹配模式："exact"（精确匹配）或 "partial"（部分匹配，包含即过滤）
MATCH_MODE = "partial"

# 内置过滤词列表（均小写，partial 模式下只要答案包含其中一个即被过滤）
FILTER_WORDS = [
    "i don't know",
    "i do not know",
    "i cannot determine",
    "i can't determine",
    "i am unable",
    "i'm unable",
    "cannot be determined",
    "cannot determine",
    "unable to determine",
    "not possible to determine",
    "unknown",
    "not known",
    "no information",
    "no data",
    "not available",
    "n/a",
    "none",
    "insufficient information",
    "insufficient data",
    "not enough information",
    "not mentioned",
    "not specified",
    "not stated",
    "not provided",
    "cannot answer",
    "cannot be answered",
    "there is no",
    "there are no",
    "does not exist",
    "do not exist",
]


# ========================================================================== #
#  工具函数
# ========================================================================== #

def normalize(text: str) -> str:
    return text.lower().strip() if text else ""


def should_filter(answer: str, mode: str = "partial") -> tuple:
    """判断答案是否应被过滤。返回 (bool, reason)"""
    if answer is None:
        return True, "(None)"
    normed = normalize(answer)
    if not normed:
        return True, "(空字符串)"

    for fw in FILTER_WORDS:
        if mode == "exact":
            if normed == fw:
                return True, fw
        else:
            if fw in normed:
                return True, fw

    return False, ""


# ========================================================================== #
#  主过滤逻辑
# ========================================================================== #

def filter_records(data):
    filtered = []
    removed = []
    reason_counts = {}

    for item in data:
        gpt_answer   = item.get("gpt_final_answer", None)
        gpt_sentence = item.get("gpt_sentence", None)

        # 先判断 gpt_sentence 是否为空
        if not gpt_sentence or not str(gpt_sentence).strip():
            reason = "(gpt_sentence 为空)"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            removed.append({"item": item, "reason": reason})
            continue

        # 再判断 gpt_final_answer
        do_filter, reason = should_filter(str(gpt_answer) if gpt_answer is not None else None, MATCH_MODE)
        if do_filter:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            removed.append({"item": item, "reason": reason})
        else:
            filtered.append(item)

    return filtered, removed, reason_counts


# ========================================================================== #
#  入口
# ========================================================================== #

def main():
    print("=" * 70)
    print("Mintaka 数据过滤工具")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 输入文件不存在: {INPUT_FILE}")
        return

    print(f"\n读取文件: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)
    print(f"原始数据: {total} 条")
    print(f"匹配模式: {MATCH_MODE}")
    print(f"过滤词数量: {len(FILTER_WORDS)}")

    print("\n开始过滤...")
    filtered, removed, reason_counts = filter_records(data)

    kept = len(filtered)
    removed_cnt = len(removed)

    print(f"\n{'=' * 70}")
    print(f"过滤统计:")
    print(f"  原始数据:  {total} 条")
    print(f"  保留数据:  {kept}  ({kept / max(total, 1) * 100:.2f}%)")
    print(f"  移除数据:  {removed_cnt}  ({removed_cnt / max(total, 1) * 100:.2f}%)")

    if reason_counts:
        print(f"\n  移除原因统计（前 20）:")
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (reason, cnt) in enumerate(sorted_reasons[:20], 1):
            print(f"    {i:2d}. {reason}: {cnt} 条")

    # 显示移除示例
    examples = removed[:10]
    if examples:
        print(f"\n  移除示例（前 {len(examples)} 条）:")
        for i, ex in enumerate(examples, 1):
            q = ex["item"].get("question", "N/A")
            a = ex["item"].get("gpt_final_answer", "N/A")
            print(f"    {i}. 问题: {str(q)[:60]}")
            print(f"       答案: [{a}]")
            print(f"       原因: {ex['reason']}")

    # 标签分布
    correct_cnt      = sum(1 for it in filtered if it.get("generation_label") == "correct")
    hallucination_cnt = sum(1 for it in filtered if it.get("generation_label") == "hallucination")
    print(f"\n  过滤后标签分布:")
    print(f"    correct:      {correct_cnt}  ({correct_cnt / max(kept, 1) * 100:.1f}%)")
    print(f"    hallucination:{hallucination_cnt}  ({hallucination_cnt / max(kept, 1) * 100:.1f}%)")

    print(f"\n{'=' * 70}")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n保存成功: {OUTPUT_FILE}")
    print(f"最终数据: {kept} 条")
    print(f"\n{'=' * 70}")
    print("过滤完成！")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
