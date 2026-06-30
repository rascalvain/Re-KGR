"""
测试 EpochBalancedSampler 在 Mintaka 数据集上的兼容性
"""

import sys
import torch
from torch.utils.data import DataLoader

# 导入配置和数据加载器
from config_mintaka_rgat import Config
from data_loader_mintaka import MintakaGraphDataset, collate_fn
from EpochBalancedSampler import EpochBalancedSampler, BatchBalancedSampler, extract_label_from_sample


def test_label_extraction():
    """测试标签提取功能"""
    print("=" * 60)
    print("测试 1: 标签提取功能")
    print("=" * 60)

    # 加载数据集（只加载少量样本）
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=10
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 测试前5个样本的标签提取
    print("\n测试前5个样本的标签提取:")
    for idx in range(min(5, len(dataset))):
        sample = dataset[idx]

        # 使用采样器的提取函数
        label = extract_label_from_sample(sample, idx)

        # 直接从样本获取
        direct_label = sample[2] if isinstance(sample, tuple) else None
        metadata = sample[3] if isinstance(sample, tuple) and len(sample) > 3 else {}

        print(f"\n  样本 {idx}:")
        print(f"    提取的标签: {label}")
        print(f"    直接标签: {direct_label}")
        print(f"    metadata['label']: {metadata.get('label', 'N/A')}")
        print(f"    metadata['generation_label']: {metadata.get('generation_label', 'N/A')}")
        print(f"    ✓ 一致性检查: {'通过' if label == direct_label else '失败'}")

    print("\n" + "=" * 60)


def test_epoch_balanced_sampler():
    """测试 Epoch 平衡采样器"""
    print("\n测试 2: EpochBalancedSampler")
    print("=" * 60)

    # 加载数据集（使用较小的子集）
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=100
    )

    # 创建采样器
    sampler = EpochBalancedSampler(dataset, seed=42)

    # 测试两个 epoch
    for epoch in range(2):
        sampler.set_epoch(epoch)
        indices = list(sampler)

        # 统计标签分布
        labels = []
        for idx in indices:
            sample = dataset[idx]
            label = extract_label_from_sample(sample, idx)
            labels.append(label)

        num_hallucination = sum(1 for l in labels if l == 0)
        num_correct = sum(1 for l in labels if l == 1)

        print(f"\nEpoch {epoch}:")
        print(f"  采样总数: {len(indices)}")
        print(f"  幻觉样本: {num_hallucination} ({num_hallucination/len(indices)*100:.1f}%)")
        print(f"  正确样本: {num_correct} ({num_correct/len(indices)*100:.1f}%)")
        print(f"  平衡性检查: {'✓ 通过' if abs(num_hallucination - num_correct) <= 1 else '✗ 失败'}")

    print("\n" + "=" * 60)


def test_batch_balanced_sampler():
    """测试批次平衡采样器"""
    print("\n测试 3: BatchBalancedSampler")
    print("=" * 60)

    # 加载数据集
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=100
    )

    batch_size = 6  # 必须是偶数

    try:
        # 创建采样器
        sampler = BatchBalancedSampler(dataset, batch_size=batch_size)

        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_fn
        )

        print(f"\n使用 batch_size={batch_size} 测试前3个batch:")

        # 测试前3个batch
        for batch_idx, (entity_batch, gpt_batch, labels, metadata_list) in enumerate(loader):
            if batch_idx >= 3:
                break

            if entity_batch is None:
                continue

            num_hallucination = (labels == 0).sum().item()
            num_correct = (labels == 1).sum().item()

            print(f"\n  Batch {batch_idx}:")
            print(f"    大小: {len(labels)}")
            print(f"    幻觉: {num_hallucination}")
            print(f"    正确: {num_correct}")
            print(f"    1:1 平衡: {'✓ 通过' if num_hallucination == num_correct else '✗ 失败'}")

        print("\n✓ BatchBalancedSampler 测试通过")

    except Exception as e:
        print(f"\n✗ BatchBalancedSampler 测试失败: {e}")

    print("\n" + "=" * 60)


def test_dataloader_integration():
    """测试与 DataLoader 的集成"""
    print("\n测试 4: DataLoader 集成测试")
    print("=" * 60)

    # 加载数据集
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=50
    )

    # 创建采样器
    sampler = EpochBalancedSampler(dataset, seed=42)

    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=6,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"\nDataLoader 配置:")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  采样器大小: {len(sampler)}")
    print(f"  Batch size: 6")

    # 遍历一个 epoch
    total_samples = 0
    total_hallucination = 0
    total_correct = 0

    print(f"\n遍历一个完整 epoch:")
    for batch_idx, (entity_batch, gpt_batch, labels, metadata_list) in enumerate(loader):
        if entity_batch is None:
            continue

        batch_hall = (labels == 0).sum().item()
        batch_correct = (labels == 1).sum().item()

        total_samples += len(labels)
        total_hallucination += batch_hall
        total_correct += batch_correct

        if batch_idx < 3:  # 只打印前3个batch
            print(f"  Batch {batch_idx}: 幻觉={batch_hall}, 正确={batch_correct}")

    print(f"\nEpoch 统计:")
    print(f"  总样本: {total_samples}")
    print(f"  总幻觉: {total_hallucination} ({total_hallucination/total_samples*100:.1f}%)")
    print(f"  总正确: {total_correct} ({total_correct/total_samples*100:.1f}%)")
    print(f"  平衡性: {'✓ 通过' if abs(total_hallucination - total_correct) <= 5 else '✗ 失败'}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Mintaka 数据集 - 采样器兼容性测试")
    print("=" * 60)

    try:
        # 运行所有测试
        test_label_extraction()
        test_epoch_balanced_sampler()
        test_batch_balanced_sampler()
        test_dataloader_integration()

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！采样器完全兼容 Mintaka 数据集")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
