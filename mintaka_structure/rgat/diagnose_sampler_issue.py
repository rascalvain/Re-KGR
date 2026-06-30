"""
诊断采样器问题
"""

import torch
from config_mintaka_rgat import Config
from data_loader_mintaka import MintakaGraphDataset

def diagnose():
    print("=" * 60)
    print("诊断采样器问题")
    print("=" * 60)

    # 加载数据集
    print("\n[1] 加载原始数据集...")
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=100  # 只测试100个样本
    )

    print(f"    数据集大小: {len(dataset)}")

    # 测试直接访问
    print("\n[2] 测试直接访问原始数据集...")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        if sample is None:
            print(f"    样本 {i}: None ❌")
        else:
            label = sample[2] if isinstance(sample, tuple) and len(sample) > 2 else None
            print(f"    样本 {i}: label={label} ✓")

    # 划分数据集
    print("\n[3] 使用 random_split 划分数据集...")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"    训练集: {len(train_dataset)}")
    print(f"    验证集: {len(val_dataset)}")
    print(f"    测试集: {len(test_dataset)}")

    # 测试访问 Subset
    print("\n[4] 测试访问训练集 (Subset)...")
    print(f"    训练集类型: {type(train_dataset)}")

    none_count = 0
    valid_count = 0
    error_count = 0

    for i in range(min(10, len(train_dataset))):
        try:
            sample = train_dataset[i]
            if sample is None:
                none_count += 1
                print(f"    样本 {i}: None ❌")
            else:
                valid_count += 1
                label = sample[2] if isinstance(sample, tuple) and len(sample) > 2 else None
                if i < 5:  # 只打印前5个
                    print(f"    样本 {i}: label={label} ✓")
        except Exception as e:
            error_count += 1
            print(f"    样本 {i}: 错误 - {e} ❌")

    print(f"\n    统计:")
    print(f"      有效样本: {valid_count}")
    print(f"      None样本: {none_count}")
    print(f"      错误样本: {error_count}")

    # 扫描完整训练集
    print(f"\n[5] 扫描完整训练集...")
    hall_count = 0
    correct_count = 0
    none_count = 0
    error_count = 0

    for i in range(len(train_dataset)):
        try:
            sample = train_dataset[i]
            if sample is None:
                none_count += 1
            elif isinstance(sample, tuple) and len(sample) > 2:
                label = sample[2]
                if label == 0:
                    hall_count += 1
                elif label == 1:
                    correct_count += 1
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f"      错误 {i}: {e}")

    print(f"    幻觉样本: {hall_count}")
    print(f"    正确样本: {correct_count}")
    print(f"    None样本: {none_count}")
    print(f"    错误样本: {error_count}")

    if hall_count == 0 and correct_count == 0:
        print("\n❌ 问题确认：无法从训练集提取标签")
        print("   可能原因：")
        print("   1. 数据集 __getitem__ 返回 None")
        print("   2. random_split 的 Subset 访问有问题")
        print("   3. 标签格式不正确")

        # 进一步诊断
        print("\n[6] 深入诊断第一个样本...")
        try:
            sample = train_dataset[0]
            print(f"    类型: {type(sample)}")
            if sample is not None:
                print(f"    长度: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
                if isinstance(sample, tuple):
                    for i, item in enumerate(sample):
                        print(f"    sample[{i}]: {type(item)}")
                        if i == 2:  # label位置
                            print(f"             值: {item}")
        except Exception as e:
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\n✓ 可以正常提取标签")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    diagnose()
