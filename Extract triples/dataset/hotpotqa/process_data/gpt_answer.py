import json

# 配置路径
input_file = "./hotpot_dev_with_gpt_answers_new.json"
output_file = "unique_gpt_final_answers.txt"

print("正在加载JSON文件...")
# 读取JSON文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"已加载 {len(data)} 条数据")

# 提取所有gpt_final_answer并去重
unique_answers = set()
total_answers = 0
none_count = 0

for item in data:
    total_answers += 1
    answer = item.get('gpt_final_answer')
    if answer is not None and answer.strip():  # 只添加非空答案
        unique_answers.add(answer.strip())
    else:
        none_count += 1

print(f"\n统计信息:")
print(f"  - 总答案数: {total_answers}")
print(f"  - 空答案数: {none_count}")
print(f"  - 去重后唯一答案数: {len(unique_answers)}")

# 排序（可选）
sorted_answers = sorted(unique_answers)

# 写入txt文件
print(f"\n正在写入文件: {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    for answer in sorted_answers:
        f.write(answer + '\n')

print(f"✅ 完成！已将 {len(unique_answers)} 个唯一答案写入文件")
print(f"   文件位置: {output_file}")

# 显示前10个示例
print(f"\n前10个唯一答案示例:")
for i, answer in enumerate(sorted_answers[:10], 1):
    print(f"  {i}. {answer}")