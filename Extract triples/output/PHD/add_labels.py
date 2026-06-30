import json

# 读取两个JSON文件
print("正在读取文件...")
with open('processed_PHD_graph_data.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

with open('triple_consistency_scores.json', 'r', encoding='utf-8') as f:
    score_data = json.load(f)

print(f"原始数据数量: {len(original_data)}")
print(f"得分数据数量: {len(score_data)}")

# 检查数据是否一一对应
if len(original_data) != len(score_data):
    print("警告：两个文件的数据数量不一致！")
else:
    print("数据数量一致，开始处理...")

# 将label添加到score_data中
matched_count = 0
mismatched_entities = []

for i, (orig, score) in enumerate(zip(original_data, score_data)):
    # 验证entity是否匹配
    if orig['entity'] == score['entity']:
        score['label'] = orig['label']
        matched_count += 1
    else:
        mismatched_entities.append({
            'index': i,
            'original_entity': orig['entity'],
            'score_entity': score['entity']
        })

print(f"\n成功匹配: {matched_count} 条数据")

if mismatched_entities:
    print(f"警告：发现 {len(mismatched_entities)} 条不匹配的数据")
    print("前5条不匹配示例:")
    for item in mismatched_entities[:5]:
        print(f"  索引 {item['index']}: 原始={item['original_entity']}, 得分={item['score_entity']}")

# 保存添加了label的数据
output_file = 'triple_consistency_scores_with_labels.json'
print(f"\n正在保存到 {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(score_data, f, ensure_ascii=False, indent=2)

print(f"完成！已保存到 {output_file}")

# 统计label分布
label_counts = {}
for item in score_data:
    label = item.get('label', 'unknown')
    label_counts[label] = label_counts.get(label, 0) + 1

print("\n标签分布:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")















