import json

# 配置路径
input_file = "./hotpot_dev_with_gpt_answers_new.json"
filter_file = "filter.txt"
output_file = "hotpot_dev_with_gpt_answers_filtered.json"

# 配置匹配模式
MATCH_MODE = "partial"  # 可选: "exact" (精确匹配) 或 "partial" (部分匹配)

print("正在加载过滤列表...")
with open(filter_file, 'r', encoding='utf-8') as f:
    filter_list = [line.strip() for line in f if line.strip()]

print(f"已加载 {len(filter_list)} 个过滤词")
print(f"匹配模式: {MATCH_MODE}")
print("过滤词列表:")
for i, word in enumerate(filter_list, 1):
    print(f"  {i}. {word}")


def normalize(text):
    """标准化文本用于匹配"""
    if text is None:
        return ""
    return text.lower().strip()


# 创建标准化的过滤集合
normalized_filters = {normalize(word) for word in filter_list}


def should_filter(answer, match_mode="exact"):
    """判断答案是否应该被过滤"""
    if answer is None:
        return True

    normalized_answer = normalize(answer)

    if not normalized_answer:
        return True

    if match_mode == "exact":
        # 精确匹配
        return normalized_answer in normalized_filters
    else:
        # 部分匹配：答案中包含任何过滤词
        for filter_word in normalized_filters:
            if filter_word in normalized_answer:
                return True
        return False


print("\n正在加载JSON文件...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"已加载 {len(data)} 条数据")

# 筛选数据
filtered_data = []
removed_count = 0
removed_by_category = {}
removed_examples = []

for item in data:
    answer = item.get('gpt_final_answer')

    if should_filter(answer, MATCH_MODE):
        removed_count += 1

        # 统计被过滤的原因
        normalized_ans = normalize(answer)
        matched_filter = None

        if answer is None:
            matched_filter = "(None/Null)"
        elif not normalized_ans:
            matched_filter = "(Empty)"
        else:
            for filter_word in filter_list:
                if MATCH_MODE == "exact":
                    if normalize(filter_word) == normalized_ans:
                        matched_filter = filter_word
                        break
                else:
                    if normalize(filter_word) in normalized_ans:
                        matched_filter = filter_word
                        break

        if matched_filter:
            removed_by_category[matched_filter] = removed_by_category.get(matched_filter, 0) + 1

        # 保存前10个被移除的例子
        if len(removed_examples) < 10:
            removed_examples.append({
                'question': item.get('question', 'N/A'),
                'answer': answer,
                'matched': matched_filter
            })
    else:
        filtered_data.append(item)

print(f"\n筛选结果:")
print(f"  - 原始数据: {len(data)} 条")
print(f"  - 被过滤: {removed_count} 条 ({removed_count / len(data) * 100:.2f}%)")
print(f"  - 保留数据: {len(filtered_data)} 条 ({len(filtered_data) / len(data) * 100:.2f}%)")

# 显示按过滤词分类的统计
if removed_by_category:
    print(f"\n被过滤数据的分类统计（前20个）:")
    sorted_categories = sorted(removed_by_category.items(), key=lambda x: x[1], reverse=True)
    for i, (category, count) in enumerate(sorted_categories[:20], 1):
        print(f"  {i}. {category}: {count} 条")

if removed_examples:
    print(f"\n被移除的数据示例（前{len(removed_examples)}条）:")
    for i, example in enumerate(removed_examples, 1):
        print(f"  {i}. 问题: {example['question'][:50]}...")
        print(f"     答案: [{example['answer']}]")
        print(f"     匹配: {example['matched']}")

# 保存筛选后的数据
print(f"\n正在保存筛选后的数据到: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 完成！")
print(f"   原始文件: {input_file}")
print(f"   筛选后文件: {output_file}")
print(f"   保留 {len(filtered_data)} 条有效数据")

# 统计筛选后的标签分布
correct_count = sum(1 for item in filtered_data if item.get('generation_label') == 'correct')
hallucination_count = sum(1 for item in filtered_data if item.get('generation_label') == 'hallucination')

if len(filtered_data) > 0:
    print(f"\n筛选后的标签分布:")
    print(f"  - 正确答案: {correct_count} ({correct_count / len(filtered_data) * 100:.1f}%)")
    print(f"  - 幻觉答案: {hallucination_count} ({hallucination_count / len(filtered_data) * 100:.1f}%)")