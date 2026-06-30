"""
诊断为什么图转换会失败
"""

import json
from config_mintaka_rgat import Config

def diagnose_data_quality():
    """诊断原始数据质量"""
    print("=" * 60)
    print("诊断 Mintaka 数据质量")
    print("=" * 60)

    # 加载原始数据
    print(f"\n[1] 加载数据文件...")
    print(f"    {Config.DATA_PATH}")

    with open(Config.DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"    总样本数: {len(data)}")

    # 统计三元组情况
    print(f"\n[2] 统计三元组情况（前100个样本）...")

    stats = {
        'total': 0,
        'has_entity_triples': 0,
        'has_gpt_triples': 0,
        'has_both': 0,
        'entity_empty': 0,
        'gpt_empty': 0,
        'both_empty': 0,
        'no_label': 0,
        'hallucination': 0,
        'correct': 0
    }

    entity_triple_counts = []
    gpt_triple_counts = []

    max_check = min(100, len(data))

    for i, item in enumerate(data[:max_check]):
        stats['total'] += 1

        entity_triples = item.get('entity_triples', [])
        gpt_triples = item.get('gpt_triples', [])
        label = item.get('generation_label', '')

        # 统计标签
        if label == 'hallucination':
            stats['hallucination'] += 1
        elif label == 'correct':
            stats['correct'] += 1
        else:
            stats['no_label'] += 1

        # 统计三元组
        has_entity = entity_triples and len(entity_triples) > 0
        has_gpt = gpt_triples and len(gpt_triples) > 0

        if has_entity:
            stats['has_entity_triples'] += 1
            entity_triple_counts.append(len(entity_triples))
        else:
            stats['entity_empty'] += 1

        if has_gpt:
            stats['has_gpt_triples'] += 1
            gpt_triple_counts.append(len(gpt_triples))
        else:
            stats['gpt_empty'] += 1

        if has_entity and has_gpt:
            stats['has_both'] += 1

        if not has_entity and not has_gpt:
            stats['both_empty'] += 1

        # 打印前5个有问题的样本
        if i < 5 and (not has_entity or not has_gpt):
            print(f"\n    样本 {i}:")
            print(f"      ID: {item.get('id', 'N/A')}")
            print(f"      标签: {label}")
            print(f"      entity_triples数量: {len(entity_triples)}")
            print(f"      gpt_triples数量: {len(gpt_triples)}")

    print(f"\n[3] 统计结果:")
    print(f"    总样本: {stats['total']}")
    print(f"    有entity_triples: {stats['has_entity_triples']} ({stats['has_entity_triples']/stats['total']*100:.1f}%)")
    print(f"    有gpt_triples: {stats['has_gpt_triples']} ({stats['has_gpt_triples']/stats['total']*100:.1f}%)")
    print(f"    两者都有: {stats['has_both']} ({stats['has_both']/stats['total']*100:.1f}%)")
    print(f"    entity为空: {stats['entity_empty']}")
    print(f"    gpt为空: {stats['gpt_empty']}")
    print(f"    两者都空: {stats['both_empty']}")

    if entity_triple_counts:
        print(f"\n    entity_triples统计:")
        print(f"      平均: {sum(entity_triple_counts)/len(entity_triple_counts):.1f}")
        print(f"      最小: {min(entity_triple_counts)}")
        print(f"      最大: {max(entity_triple_counts)}")

    if gpt_triple_counts:
        print(f"\n    gpt_triples统计:")
        print(f"      平均: {sum(gpt_triple_counts)/len(gpt_triple_counts):.1f}")
        print(f"      最小: {min(gpt_triple_counts)}")
        print(f"      最大: {max(gpt_triple_counts)}")

    print(f"\n    标签分布:")
    print(f"      幻觉: {stats['hallucination']} ({stats['hallucination']/stats['total']*100:.1f}%)")
    print(f"      正确: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
    print(f"      无标签: {stats['no_label']}")

    # 检查三元组格式
    print(f"\n[4] 检查三元组格式（第一个有效样本）...")
    for item in data[:10]:
        entity_triples = item.get('entity_triples', [])
        gpt_triples = item.get('gpt_triples', [])

        if entity_triples and len(entity_triples) > 0:
            print(f"\n    entity_triples示例:")
            print(f"      类型: {type(entity_triples)}")
            print(f"      第一个元素类型: {type(entity_triples[0])}")
            print(f"      第一个元素: {entity_triples[0]}")

            if isinstance(entity_triples[0], dict):
                print(f"      包含的键: {list(entity_triples[0].keys())}")
            break

    for item in data[:10]:
        gpt_triples = item.get('gpt_triples', [])

        if gpt_triples and len(gpt_triples) > 0:
            print(f"\n    gpt_triples示例:")
            print(f"      类型: {type(gpt_triples)}")
            print(f"      第一个元素类型: {type(gpt_triples[0])}")
            print(f"      第一个元素: {gpt_triples[0]}")

            if isinstance(gpt_triples[0], dict):
                print(f"      包含的键: {list(gpt_triples[0].keys())}")
            break

    # 问题诊断
    print(f"\n[5] 问题诊断:")

    if stats['has_both'] < stats['total'] * 0.5:
        print(f"    ❌ 严重问题：超过50%的样本缺少三元组")
        print(f"       两者都有的样本只有 {stats['has_both']}/{stats['total']}")
        return False

    if stats['entity_empty'] > 0 or stats['gpt_empty'] > 0:
        print(f"    ⚠️  警告：有样本的三元组为空")
        print(f"       entity为空: {stats['entity_empty']}")
        print(f"       gpt为空: {stats['gpt_empty']}")
        print(f"       这些样本会在图转换时失败")

    if stats['has_both'] > 0:
        print(f"    ✓ 有 {stats['has_both']} 个样本可以用于训练")
        return True

    return False


def diagnose_graph_conversion():
    """诊断图转换过程"""
    print("\n" + "=" * 60)
    print("诊断图转换过程")
    print("=" * 60)

    from data_loader_mintaka import MintakaGraphDataset
    import pickle

    # 创建数据集（只加载少量样本）
    print(f"\n[1] 创建数据集（前50个样本）...")
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=50
    )

    print(f"    有效样本数: {len(dataset)}")

    # 检查实体和关系映射
    print(f"\n[2] 检查实体和关系映射...")
    with open(Config.ENTITY2IDX_PATH, 'rb') as f:
        entity2id = pickle.load(f)
    with open(Config.RELATION2IDX_PATH, 'rb') as f:
        relation2id = pickle.load(f)

    print(f"    实体词表大小: {len(entity2id)}")
    print(f"    关系词表大小: {len(relation2id)}")

    # 测试图转换
    print(f"\n[3] 测试图转换（前10个样本）...")

    conversion_stats = {
        'total': 0,
        'both_success': 0,
        'entity_failed': 0,
        'gpt_failed': 0,
        'both_failed': 0
    }

    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        conversion_stats['total'] += 1

        if sample is None:
            conversion_stats['both_failed'] += 1
            print(f"    样本 {i}: 完全失败 ❌")
        elif isinstance(sample, tuple) and len(sample) == 4:
            entity_graph, gpt_graph, label, metadata = sample

            if entity_graph is not None and gpt_graph is not None:
                conversion_stats['both_success'] += 1
                print(f"    样本 {i}: 成功 ✓ (entity节点={entity_graph.num_nodes}, gpt节点={gpt_graph.num_nodes})")
            elif entity_graph is None and gpt_graph is None:
                conversion_stats['both_failed'] += 1
                print(f"    样本 {i}: 两个图都失败 ❌")
            elif entity_graph is None:
                conversion_stats['entity_failed'] += 1
                print(f"    样本 {i}: entity图失败 ⚠️")
            elif gpt_graph is None:
                conversion_stats['gpt_failed'] += 1
                print(f"    样本 {i}: gpt图失败 ⚠️")

    print(f"\n[4] 转换统计:")
    print(f"    总样本: {conversion_stats['total']}")
    print(f"    完全成功: {conversion_stats['both_success']} ({conversion_stats['both_success']/conversion_stats['total']*100:.1f}%)")
    print(f"    entity图失败: {conversion_stats['entity_failed']}")
    print(f"    gpt图失败: {conversion_stats['gpt_failed']}")
    print(f"    两个都失败: {conversion_stats['both_failed']}")

    success_rate = conversion_stats['both_success'] / conversion_stats['total'] * 100

    print(f"\n[5] 结论:")
    if success_rate >= 80:
        print(f"    ✓ 图转换成功率: {success_rate:.1f}% (良好)")
        print(f"    可以正常训练")
        return True
    elif success_rate >= 50:
        print(f"    ⚠️  图转换成功率: {success_rate:.1f}% (一般)")
        print(f"    训练可能受影响，建议检查数据质量")
        return True
    else:
        print(f"    ❌ 图转换成功率: {success_rate:.1f}% (差)")
        print(f"    无法正常训练，需要修复数据")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Mintaka 数据集诊断工具")
    print("=" * 80)

    # 诊断数据质量
    data_ok = diagnose_data_quality()

    # 诊断图转换
    if data_ok:
        graph_ok = diagnose_graph_conversion()

        if graph_ok:
            print("\n" + "=" * 80)
            print("✓ 诊断完成：数据集可以用于训练")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ 诊断完成：图转换失败率过高")
            print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 诊断完成：数据质量有严重问题")
        print("=" * 80)
