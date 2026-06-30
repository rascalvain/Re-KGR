"""
平衡采样器 - 用于处理类别不平衡问题
支持 HotpotQA 数据集的 tuple 格式: (context_graph, gpt_graph, label, metadata)
"""

import torch
import numpy as np
from torch.utils.data import Sampler
import random


def extract_label_from_sample(sample, idx):
    """
    从样本中提取标签（支持多种格式）

    Args:
        sample: 数据集样本，可能是tuple、dict等
        idx: 样本索引（用于调试）

    Returns:
        int: 0 (幻觉) 或 1 (正确)，如果无法提取则返回 None
    """
    try:
        # 格式1: Tuple/List - HotpotQA格式 (context_graph, gpt_graph, label, metadata)
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

        # 格式2: Dict（直接包含label字段）
        elif isinstance(sample, dict):
            # 优先使用数值标签
            if 'label' in sample:
                label = sample['label']
                if isinstance(label, (int, float)):
                    return int(label)
                elif isinstance(label, torch.Tensor):
                    return int(label.item())

            # 使用generation_label字符串
            if 'generation_label' in sample:
                gen_label = sample['generation_label']
                if isinstance(gen_label, str):
                    if gen_label.lower() == 'hallucination':
                        return 0
                    elif gen_label.lower() in ['correct', 'factual']:
                        return 1

        # 无法识别格式
        return None

    except Exception as e:
        print(f"  ⚠️ 错误: 提取样本 {idx} 的标签时出错: {e}")
        return None


class BatchBalancedSampler(Sampler):
    """
    批次级别的平衡采样器：保证每个batch内都是幻觉:事实 = 1:1

    特点：
    - 每个batch固定包含 batch_size//2 个幻觉样本 + batch_size//2 个事实样本
    - 每轮训练对两类样本分别shuffle
    - 不会有单类batch出现

    限制：
    - batch_size 必须是偶数
    - 会丢弃少量不能整除的样本
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小（必须是偶数）
        """
        if batch_size % 2 != 0:
            raise ValueError(f"❌ batch_size必须是偶数才能保证1:1平衡，当前={batch_size}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

        # 分离幻觉和事实样本的索引
        self.hallucination_indices = []
        self.correct_indices = []

        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                # ✅ 修正：传递两个参数
                label = extract_label_from_sample(sample, idx)

                if label == 0:  # 幻觉
                    self.hallucination_indices.append(idx)
                elif label == 1:  # 事实
                    self.correct_indices.append(idx)
            except Exception as e:
                print(f"⚠️ 样本 {idx} 标签提取失败: {e}")
                continue

        print(f"📊 批次平衡采样器统计:")
        print(f"  幻觉样本: {len(self.hallucination_indices)}")
        print(f"  事实样本: {len(self.correct_indices)}")
        print(f"  batch_size: {batch_size} (每batch: 幻觉{self.half_batch} + 事实{self.half_batch})")

        if len(self.hallucination_indices) == 0 or len(self.correct_indices) == 0:
            raise ValueError("❌ 错误：必须同时有幻觉和事实样本")

        # 计算epoch中batch的数量（取决于较少的那一类）
        self.num_batches = min(
            len(self.hallucination_indices) // self.half_batch,
            len(self.correct_indices) // self.half_batch
        )

        if self.num_batches == 0:
            raise ValueError(
                f"❌ 错误：样本数量不足以组成一个batch\n"
                f"   batch_size={batch_size}, 需要每类至少{self.half_batch}个样本\n"
                f"   当前：幻觉{len(self.hallucination_indices)}，事实{len(self.correct_indices)}"
            )

        # 计算使用的样本数和丢弃的样本数
        used_hall = self.num_batches * self.half_batch
        used_correct = self.num_batches * self.half_batch
        dropped_hall = len(self.hallucination_indices) - used_hall
        dropped_correct = len(self.correct_indices) - used_correct

        print(f"  每epoch的batch数: {self.num_batches}")
        print(f"  每epoch的样本数: {self.num_batches * batch_size}")
        if dropped_hall > 0 or dropped_correct > 0:
            print(f"  ⚠️ 丢弃样本: 幻觉{dropped_hall}, 事实{dropped_correct}")

    def __iter__(self):
        # 每次迭代时打乱两类样本的索引（实现每轮不同的采样）
        hall_shuffled = self.hallucination_indices.copy()
        correct_shuffled = self.correct_indices.copy()
        random.shuffle(hall_shuffled)
        random.shuffle(correct_shuffled)

        # 构建平衡的batch
        for i in range(self.num_batches):
            batch_indices = []

            # 每个batch取 half_batch 个幻觉样本和 half_batch 个事实样本
            start = i * self.half_batch
            batch_indices.extend(hall_shuffled[start:start + self.half_batch])
            batch_indices.extend(correct_shuffled[start:start + self.half_batch])

            # 打乱batch内的样本顺序（避免总是前一半幻觉、后一半事实）
            random.shuffle(batch_indices)

            # yield batch内的每个索引
            for idx in batch_indices:
                yield idx

    def __len__(self):
        return self.num_batches * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

class EpochBalancedSampler(Sampler):
    """
    Epoch级别的平衡采样器
    每个epoch随机采样，确保幻觉和正确样本比例为50:50
    """

    def __init__(self, dataset, seed=42):
        """
        Args:
            dataset: PyTorch Dataset对象
            seed: 随机种子
        """
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        print("\n📊 初始化 Epoch平衡采样器...")

        # 扫描数据集，统计标签分布
        self.hallucination_indices = []
        self.correct_indices = []

        print(f"   正在扫描 {len(dataset)} 个样本...")
        failed_count = 0
        failed_samples = []

        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]

                # 使用提取函数获取标签
                label = extract_label_from_sample(sample, idx)

                if label == 0:
                    self.hallucination_indices.append(idx)
                elif label == 1:
                    self.correct_indices.append(idx)
                else:
                    failed_count += 1
                    if failed_count <= 5:  # 只记录前5个失败样本
                        failed_samples.append(idx)

            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"   样本 {idx} 处理失败: {e}")

        self.num_hallucination = len(self.hallucination_indices)
        self.num_correct = len(self.correct_indices)

        print(f"\n   📈 标签分布统计:")
        print(f"      - 幻觉样本 (label=0): {self.num_hallucination}")
        print(f"      - 正确样本 (label=1): {self.num_correct}")
        if failed_count > 0:
            print(f"      - 无效样本: {failed_count}")
            if failed_samples:
                print(f"        失败索引示例: {failed_samples}")

        # 验证是否有有效样本
        if self.num_hallucination == 0 and self.num_correct == 0:
            raise ValueError(
                f"\n❌ 错误：数据集中没有找到任何有效标签！\n"
                f"   幻觉样本数 = {self.num_hallucination}\n"
                f"   正确样本数 = {self.num_correct}\n"
                f"   失败样本数 = {failed_count}\n"
                f"\n   请检查:\n"
                f"   1. 数据集格式是否为: (context_graph, gpt_graph, label, metadata)\n"
                f"   2. label (位置2) 是否为 0 或 1\n"
                f"   3. metadata['generation_label'] 是否为 'hallucination' 或 'correct'"
            )

        if self.num_hallucination == 0:
            raise ValueError(
                f"\n❌ 错误：没有找到幻觉样本 (label=0)！\n"
                f"   正确样本数: {self.num_correct}\n"
                f"   请检查数据集中是否包含 generation_label='hallucination' 的样本"
            )

        if self.num_correct == 0:
            raise ValueError(
                f"\n❌ 错误：没有找到正确样本 (label=1)！\n"
                f"   幻觉样本数: {self.num_hallucination}\n"
                f"   请检查数据集中是否包含 generation_label='correct' 的样本"
            )

        # 计算每个类别的采样数量（取较小值，确保50:50）
        self.samples_per_class = min(self.num_hallucination, self.num_correct)
        self.total_samples = self.samples_per_class * 2

        print(f"\n   ✓ 采样策略:")
        print(f"      每个epoch采样 {self.samples_per_class} 个幻觉样本")
        print(f"      每个epoch采样 {self.samples_per_class} 个正确样本")
        print(f"      每个epoch总样本数: {self.total_samples}")
        print(f"      平衡比例: 50% 幻觉 / 50% 正确")

    def __iter__(self):
        # 使用epoch作为随机种子，确保每个epoch不同但可复现
        rng = np.random.default_rng(self.seed + self.epoch)

        # 从每个类别随机采样
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

        # 合并并打乱
        indices = np.concatenate([sampled_hallucination, sampled_correct])
        rng.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        """设置当前epoch（用于改变随机采样）"""
        self.epoch = epoch


class SequentialBalancedSampler(Sampler):
    """
    顺序平衡采样器
    将数据分成多个epoch，每个epoch内交替采样幻觉和正确样本
    """

    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

        print("\n📊 初始化 顺序平衡采样器...")

        # 扫描数据集
        self.hallucination_indices = []
        self.correct_indices = []

        print(f"   正在扫描 {len(dataset)} 个样本...")

        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                label = extract_label_from_sample(sample, idx)

                if label == 0:
                    self.hallucination_indices.append(idx)
                elif label == 1:
                    self.correct_indices.append(idx)
            except Exception as e:
                continue

        self.num_hallucination = len(self.hallucination_indices)
        self.num_correct = len(self.correct_indices)

        print(f"   幻觉样本: {self.num_hallucination}")
        print(f"   正确样本: {self.num_correct}")

        # 验证
        if self.num_hallucination == 0 or self.num_correct == 0:
            raise ValueError("没有足够的样本进行平衡采样")

        # 计算分组数
        self.samples_per_class = min(self.num_hallucination, self.num_correct)
        self.num_groups = max(self.num_hallucination, self.num_correct) // self.samples_per_class

        print(f"   每组样本数: {self.samples_per_class * 2}")
        print(f"   总组数: {self.num_groups}")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # 打乱两个类别的索引
        hallucination_shuffled = self.hallucination_indices.copy()
        correct_shuffled = self.correct_indices.copy()
        rng.shuffle(hallucination_shuffled)
        rng.shuffle(correct_shuffled)

        # 选择当前epoch对应的样本
        group_idx = self.epoch % self.num_groups
        start_idx = group_idx * self.samples_per_class
        end_idx = start_idx + self.samples_per_class

        hallucination_group = hallucination_shuffled[start_idx:end_idx]
        correct_group = correct_shuffled[start_idx:end_idx]

        # 合并并打乱
        indices = hallucination_group + correct_group
        rng.shuffle(indices)

        return iter(indices)

    def __len__(self):
        return self.samples_per_class * 2

    def set_epoch(self, epoch):
        self.epoch = epoch


class StratifiedBalancedSampler(Sampler):
    """
    分层平衡采样器
    保证每个batch中幻觉和正确样本比例接近50:50
    """

    def __init__(self, dataset, batch_size=32, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        print("\n📊 初始化 分层平衡采样器...")

        # 扫描数据集
        self.hallucination_indices = []
        self.correct_indices = []

        print(f"   正在扫描 {len(dataset)} 个样本...")

        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                label = extract_label_from_sample(sample, idx)

                if label == 0:
                    self.hallucination_indices.append(idx)
                elif label == 1:
                    self.correct_indices.append(idx)
            except Exception as e:
                continue

        self.num_hallucination = len(self.hallucination_indices)
        self.num_correct = len(self.correct_indices)

        print(f"   幻觉样本: {self.num_hallucination}")
        print(f"   正确样本: {self.num_correct}")

        # 验证
        if self.num_hallucination == 0 or self.num_correct == 0:
            raise ValueError("没有足够的样本进行平衡采样")

        # 计算采样数量
        self.samples_per_class = min(self.num_hallucination, self.num_correct)
        self.total_samples = self.samples_per_class * 2

        print(f"   每个epoch总样本数: {self.total_samples}")
        print(f"   平衡比例: 50:50")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # 从每个类别随机采样
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

        # 交织两个类别（确保batch内平衡）
        indices = []
        for i in range(self.samples_per_class):
            indices.append(sampled_hallucination[i])
            indices.append(sampled_correct[i])

        # 轻微打乱（保持局部平衡）
        indices = np.array(indices)
        chunk_size = self.batch_size
        for i in range(0, len(indices), chunk_size):
            chunk = indices[i:i+chunk_size]
            rng.shuffle(chunk)
            indices[i:i+chunk_size] = chunk

        return iter(indices.tolist())

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# 兼容性：默认使用Epoch平衡采样器
class BalancedSampler(EpochBalancedSampler):
    """默认平衡采样器（使用Epoch策略）"""
    pass


class FixedBalancedSampler(Sampler):
    """
    固定平衡采样器
    只在初始化时采样一次，之后每个epoch都使用相同的样本
    适用于验证集和测试集，保证评估的一致性
    """

    def __init__(self, dataset, seed=42):
        """
        Args:
            dataset: PyTorch Dataset对象
            seed: 随机种子（用于首次采样）
        """
        self.dataset = dataset
        self.seed = seed

        print("\n📊 初始化 固定平衡采样器（验证/测试集）...")

        # 扫描数据集，统计标签分布
        self.hallucination_indices = []
        self.correct_indices = []

        print(f"   正在扫描 {len(dataset)} 个样本...")
        failed_count = 0

        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                label = extract_label_from_sample(sample, idx)

                if label == 0:
                    self.hallucination_indices.append(idx)
                elif label == 1:
                    self.correct_indices.append(idx)
                else:
                    failed_count += 1

            except Exception as e:
                failed_count += 1

        self.num_hallucination = len(self.hallucination_indices)
        self.num_correct = len(self.correct_indices)

        print(f"\n   📈 标签分布统计:")
        print(f"      - 幻觉样本 (label=0): {self.num_hallucination}")
        print(f"      - 正确样本 (label=1): {self.num_correct}")
        if failed_count > 0:
            print(f"      - 无效样本: {failed_count}")

        # 验证是否有有效样本
        if self.num_hallucination == 0 or self.num_correct == 0:
            raise ValueError(
                f"\n❌ 错误：无法进行平衡采样\n"
                f"   幻觉样本数 = {self.num_hallucination}\n"
                f"   正确样本数 = {self.num_correct}"
            )

        # 计算每个类别的采样数量
        self.samples_per_class = min(self.num_hallucination, self.num_correct)
        self.total_samples = self.samples_per_class * 2

        # 🔥 一次性采样，之后固定使用
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

        # 合并并打乱（固定顺序）
        self.fixed_indices = np.concatenate([sampled_hallucination, sampled_correct])
        rng.shuffle(self.fixed_indices)
        self.fixed_indices = self.fixed_indices.tolist()

        print(f"\n   ✓ 固定采样策略:")
        print(f"      采样 {self.samples_per_class} 个幻觉样本")
        print(f"      采样 {self.samples_per_class} 个正确样本")
        print(f"      总样本数: {self.total_samples} (固定)")
        print(f"      平衡比例: 50% 幻觉 / 50% 正确")
        print(f"      ⚠️  每个epoch使用相同样本（用于一致性评估）")

    def __iter__(self):
        """每次迭代都返回相同的固定索引"""
        return iter(self.fixed_indices)

    def __len__(self):
        return self.total_samples

    def set_epoch(self, epoch):
        """
        空实现，保持接口一致
        注意：调用此方法不会改变采样结果
        """
        pass

if __name__ == '__main__':
    """测试采样器"""
    print("="*60)
    print("测试平衡采样器")
    print("="*60)

    # 创建模拟数据集
    class MockDataset:
        def __init__(self, num_samples=100, ratio=0.3):
            self.samples = []
            for i in range(num_samples):
                # ratio比例的幻觉样本
                label = 0 if i < int(num_samples * ratio) else 1
                # 模拟HotpotQA格式: (context_graph, gpt_graph, label, metadata)
                metadata = {
                    '_id': f'sample_{i}',
                    'label': label,
                    'generation_label': 'hallucination' if label == 0 else 'correct'
                }
                self.samples.append((None, None, label, metadata))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    # 测试
    dataset = MockDataset(num_samples=100, ratio=0.3)  # 30% 幻觉

    print("\n测试 EpochBalancedSampler:")
    sampler = EpochBalancedSampler(dataset, seed=42)

    # 测试两个epoch
    for epoch in range(2):
        sampler.set_epoch(epoch)
        indices = list(sampler)

        # 统计标签分布
        labels = [dataset[i][2] for i in indices]
        num_hallucination = sum(1 for l in labels if l == 0)
        num_correct = sum(1 for l in labels if l == 1)

        print(f"\nEpoch {epoch}:")
        print(f"  采样总数: {len(indices)}")
        print(f"  幻觉: {num_hallucination} ({num_hallucination/len(indices)*100:.1f}%)")
        print(f"  正确: {num_correct} ({num_correct/len(indices)*100:.1f}%)")

    print("\n✓ 测试完成！")