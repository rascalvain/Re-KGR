"""
验证数据加载器修复是否有效
"""

import torch
from config_mintaka_rgat import Config
from data_loader_mintaka import MintakaGraphDataset

def verify_fix():
    print("=" * 60)
    print("验证数据加载器修复")
    print("=" * 60)

    # 加载数据集
    print("\n[1] 加载数据集（前100个样本）...")
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=100
    )

    print(f"    数据集大小: {len(dataset)}")

    # 测试直接访问
    print("\n[2] 测试直接访问前10个样本...")
    none_count = 0
    valid_count = 0
    invalid_graph_count = 0

    for i in range(min(10, len(dataset))):
        sample = dataset[i]

        if sample is None:
            none_count += 1
            print(f"    样本 {i}: 完全None ❌")
        elif isinstance(sample, tuple) and len(sample) == 4:
            entity_graph, gpt_graph, label, metadata = sample

            if entity_graph is None or gpt_graph is None:
                invalid_graph_count += 1
                print(f"    样本 {i}: label={label} (图无效但标签可用) ⚠️")
            else:
                valid_count += 1
                print(f"    样本 {i}: label={label} (完全有效) ✓")
        else:
            print(f"    样本 {i}: 格式错误 ❌")

    print(f"\n    统计:")
    print(f"      完全有效: {valid_count}")
    print(f"      图无效但有标签: {invalid_graph_count}")
    print(f"      完全None: {none_count}")

    # 测试 random_split
    print("\n[3] 测试 random_split...")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"    训练集: {len(train_dataset)}")

    # 扫描训练集标签
    print("\n[4] 扫描训练集标签...")
    hall_count = 0
    correct_count = 0
    none_count = 0
    invalid_graph_count = 0

    for i in range(len(train_dataset)):
        sample = train_dataset[i]

        if sample is None:
            none_count += 1
        elif isinstance(sample, tuple) and len(sample) == 4:
            entity_graph, gpt_graph, label, metadata = sample

            if label == 0:
                hall_count += 1
            elif label == 1:
                correct_count += 1

            if entity_graph is None or gpt_graph is None:
                invalid_graph_count += 1

    print(f"    幻觉样本: {hall_count}")
    print(f"    正确样本: {correct_count}")
    print(f"    完全None: {none_count}")
    print(f"    图无效但有标签: {invalid_graph_count}")

    # 判断修复是否成功
    print("\n" + "=" * 60)
    if hall_count > 0 and correct_count > 0:
        print("✓ 修复成功！采样器现在可以提取标签了")
        print(f"  - 可用于训练的样本: {hall_count + correct_count - invalid_graph_count}")
        print(f"  - 会被collate_fn过滤的样本: {invalid_graph_count}")
        return True
    else:
        print("❌ 修复失败：仍然无法提取标签")
        return False


if __name__ == '__main__':
    success = verify_fix()
    exit(0 if success else 1)
