"""
修复版平衡采样器 - 支持 Subset 对象
"""

import torch
import numpy as np
from torch.utils.data import Sampler, Subset
import random


def extract_label_from_sample(sample, idx):
    """
    从样本中提取标签（支持多种格式）
    """
    if sample is None:
        return None

    try:
        # 格式1: Tuple/List - (graph1, graph2, label, metadata)
        if isinstance(sample, (tuple, list)):
            if len(sample) >= 3:
                # 位置2应该是label（数值）
                label_value = sample[2]

                # 直接是数值
                if isinstance(label_value, (int, float)):
                    return int(label_value)
                elif isinstance(label_value, torch.Tensor):
                    return int(label_value.item())

            # Fallback: 尝试从metadata（位置3）提取
            if len(sample) >= 4 and isinstance(sample[3], dict):
                metadata = sample[3]

                # 优先使用数值label
                if 'label' in metadata:
                    label = metadata['label']
                    if isinstance(label, (int, float)):
                        return int(label)
                    elif isinstance(label, torch.Tensor):
                        return int(label.item())

                # 使用generation_label字符串
                if 'generation_label' in metadata:
                    gen_label = metadata['generation_label']
                    if isinstance(gen_label, str):
                        if gen_label.lower() == 'hallucination':
                            return 0
                        elif gen_label.lower() in ['correct', 'factual']:
                            return 1

        # 格式2: Dict
        elif isinstance(sample, dict):
            if 'label' in sample:
                label = sample['label']
                if isinstance(label, (int, float)):
                    return int(label)
                elif isinstance(label, torch.Tensor):
                    return int(label.item())

            if 'generation_label' in sample:
                gen_label = sample['generation_label']
                if isinstance(gen_label, str):
                    if gen_label.lower() == 'hallucination':
                        return 0
                    elif gen_label.lower() in ['correct', 'factual']:
                        return 1

        return None

    except Exception as e:
        # print(f"  ⚠️ 警告: 提取样本 {idx} 的标签时出错: {e}")
        return None


def scan_dataset_labels(dataset, max_samples=None, verbose=True):
    """
    扫描数据集并提取所有有效样本的标签

    Args:
        dataset: 数据集（可以是Subset或普通Dataset）
        max_samples: 最大扫描样本数（用于调试）
        verbose: 是否打印详细信息

    Returns:
        hallucination_indices: 幻觉样本的索引列表
        correct_indices: 正确样本的索引列表
    """
    hallucination_indices = []
    correct_indices = []
    failed_count = 0
    none_count = 0

    dataset_len = len(dataset)
    scan_size = min(dataset_len, max_samples) if max_samples else dataset_len

    if verbose:
        print(f"   正在扫描数据集（共 {dataset_len} 个样本）...")

    for idx in range(scan_size):
        try:
            sample = dataset[idx]

            # 处理返回None的情况
            if sample is None:
                none_count += 1
                continue

            # 提取标签
            label = extract_label_from_sample(sample, idx)

            if label == 0:
                hallucination_indices.append(idx)
            elif label == 1:
                correct_indices.append(idx)
            else:
                failed_count += 1
                # 调试：打印前几个失败样本
                if failed_count <= 3 and verbose:
                    print(f"   ⚠️ 样本 {idx} 标签提取失败: label={label}, sample_type={type(sample)}")
                    if isinstance(sample, (tuple, list)) and len(sample) >= 3:
                        print(f"       sample[2]={sample[2]}, type={type(sample[2])}")

        except Exception as e:
            failed_count += 1
            if failed_count <= 3 and verbose:
                print(f"   ⚠️ 样本 {idx} 访问失败: {e}")

    if verbose:
        print(f"\n   📈 扫描结果:")
        print(f"      - 幻觉样本 (label=0): {len(hallucination_indices)}")
        print(f"      - 正确样本 (label=1): {len(correct_indices)}")
        if none_count > 0:
            print(f"      - None样本: {none_count}")
        if failed_count > 0:
            print(f"      - 失败样本: {failed_count}")

    return hallucination_indices, correct_indices


class BatchBalancedSampler(Sampler):
    """
    批次级别的平衡采样器：保证每个batch内都是幻觉:事实 = 1:1
    修复版：支持Subset对象
    """

    def __init__(self, dataset, batch_size):
        if batch_size % 2 != 0:
            raise ValueError(f"❌ batch_size必须是偶数，当前={batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

        # 扫描数据集
        self.hallucination_indices, self.correct_indices = scan_dataset_labels(
            dataset, verbose=True
        )

        print(f"📊 批次平衡采样器统计:")
        print(f"  幻觉样本: {len(self.hallucination_indices)}")
        print(f"  事实样本: {len(self.correct_indices)}")
        print(f"  batch_size: {batch_size} (每batch: 幻觉{self.half_batch} + 事实{self.half_batch})")

        if len(self.hallucination_indices) == 0 or len(self.correct_indices) == 0:
            raise ValueError("❌ 错误：必须同时有幻觉和事实样本")

        # 计算batch数量
        self.num_batches = min(
            len(self.hallucination_indices) // self.half_batch,
            len(self.correct_indices) // self.half_batch
        )

        if self.num_batches == 0:
            raise ValueError(
                f"❌ 样本数量不足\n"
                f"   需要：每类至少{self.half_batch}个样本\n"
                f"   当前：幻觉{len(self.hallucination_indices)}，事实{len(self.correct_indices)}"
            )

        # 统计
        used_hall = self.num_batches * self.half_batch
        used_correct = self.num_batches * self.half_batch
        dropped_hall = len(self.hallucination_indices) - used_hall
        dropped_correct = len(self.correct_indices) - used_correct

        print(f"  每epoch的batch数: {self.num_batches}")
        print(f"  每epoch的样本数: {self.num_batches * batch_size}")
        if dropped_hall > 0 or dropped_correct > 0:
            print(f"  ⚠️ 丢弃样本: 幻觉{dropped_hall}, 事实{dropped_correct}")

    def __iter__(self):
        hall_shuffled = self.hallucination_indices.copy()
        correct_shuffled = self.correct_indices.copy()
        random.shuffle(hall_shuffled)
        random.shuffle(correct_shuffled)

        for i in range(self.num_batches):
            batch_indices = []
            start = i * self.half_batch
            batch_indices.extend(hall_shuffled[start:start + self.half_batch])
            batch_indices.extend(correct_shuffled[start:start + self.half_batch])
            random.shuffle(batch_indices)

            for idx in batch_indices:
                yield idx

    def __len__(self):
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


class EpochBalancedSampler(Sampler):
    """Epoch级别的平衡采样器（修复版）"""

    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        print("\n📊 初始化 Epoch平衡采样器...")

        # 扫描数据集
        self.hallucination_indices, self.correct_indices = scan_dataset_labels(
            dataset, verbose=True
        )

        self.num_hallucination = len(self.hallucination_indices)
        self.num_correct = len(self.correct_indices)

        # 验证
        if self.num_hallucination == 0 and self.num_correct == 0:
            raise ValueError("❌ 数据集中没有找到任何有效标签")

        if self.num_hallucination == 0:
            raise ValueError("❌ 没有找到幻觉样本")

        if self.num_correct == 0:
            raise ValueError("❌ 没有找到正确样本")

        # 计算采样数量
        self.samples_per_class = min(self.num_hallucination, self.num_correct)
        self.total_samples = self.samples_per_class * 2

        print(f"\n   ✓ 采样策略:")
        print(f"      每个epoch采样 {self.samples_per_class} 个幻觉样本")
        print(f"      每个epoch采样 {self.samples_per_class} 个正确样本")
        print(f"      每个epoch总样本数: {self.total_samples}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        sampled_hallucination = rng.choice(
            self.hallucination_indices,
            size=self.samples_per_class,
            replace=False
        )
        sampled_correct = rng.choice(
            self.correct_indices,
            size=self.samples_per_class,
            replace=False
        )

        indices = np.concatenate([sampled_hallucination, sampled_correct])
        rng.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class FixedBalancedSampler(Sampler):
    """固定平衡采样器（修复版）"""

    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed

        print("\n📊 初始化 固定平衡采样器（验证/测试集）...")

        # 扫描数据集
        self.hallucination_indices, self.correct_indices = scan_dataset_labels(
            dataset, verbose=True
        )

        self.num_hallucination = len(self.hallucination_indices)
        self.num_correct = len(self.correct_indices)

        if self.num_hallucination == 0 or self.num_correct == 0:
            raise ValueError("❌ 无法进行平衡采样")

        # 计算采样数量
        self.samples_per_class = min(self.num_hallucination, self.num_correct)
        self.total_samples = self.samples_per_class * 2

        # 一次性采样
        rng = np.random.default_rng(self.seed)

        sampled_hallucination = rng.choice(
            self.hallucination_indices,
            size=self.samples_per_class,
            replace=False
        )
        sampled_correct = rng.choice(
            self.correct_indices,
            size=self.samples_per_class,
            replace=False
        )

        self.fixed_indices = np.concatenate([sampled_hallucination, sampled_correct])
        rng.shuffle(self.fixed_indices)
        self.fixed_indices = self.fixed_indices.tolist()

        print(f"\n   ✓ 固定采样: {self.total_samples}个样本 (50:50)")

    def __iter__(self):
        return iter(self.fixed_indices)

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        pass


# 其他采样器（简化版）
class SequentialBalancedSampler(EpochBalancedSampler):
    pass


class StratifiedBalancedSampler(EpochBalancedSampler):
    pass


class BalancedSampler(EpochBalancedSampler):
    pass
