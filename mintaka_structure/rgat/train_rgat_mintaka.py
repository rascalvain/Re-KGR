"""
Mintaka 数据集的 RGAT 训练脚本
使用孪生网络架构训练图注意力嵌入模型
核心改进：使用 R-GAT 替代 R-GCN，引入多头注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# 导入RGAT专用配置
from config_mintaka_rgat import Config, create_directories

# 导入Mintaka数据加载器
from data_loader_mintaka import MintakaGraphDataset, collate_fn

# 导入RGAT模型
from siamese_rgat_improved import SiameseRGATWithEmbedding, ImprovedContrastiveLoss


# ==================== 监督对比学习损失函数 ====================
class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失 (Supervised Contrastive Learning)

    核心思想：
    - 利用标签信息：同类样本（都是幻觉或都是事实）拉近
    - 异类样本（一个幻觉一个事实）推远

    相比无监督对比学习的优势：
    - 明确知道哪些样本应该接近，哪些应该远离
    - 对于不平衡数据更有效
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, embedding_dim] - 图嵌入向量
            labels: [batch_size] - 标签 (0=幻觉, 1=事实)
        Returns:
            loss: scalar
            loss_dict: 包含额外信息的字典
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            # batch size太小，返回0损失
            return torch.tensor(0.0, device=device), {
                'contrastive_loss': 0.0,
                'avg_similarity': 0.0,
                'num_positives': 0
            }

        # 标准化特征（使其在单位超球面上）
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵 [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 为数值稳定性，减去每行的最大值
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # 创建标签掩码
        # mask[i, j] = 1 表示样本i和样本j属于同一类
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除对角线（自己和自己不计算）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算exp(logits)
        exp_logits = torch.exp(logits) * logits_mask

        # 计算log_prob = log(exp(s_ij) / sum(exp(s_ik)))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 对每个样本，计算其与所有正样本对的平均log-likelihood
        # mask.sum(1)是每个样本的正样本对数量
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 损失：负的平均log-likelihood
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # 统计信息
        avg_similarity = similarity_matrix[mask.bool()].mean().item() if mask.sum() > 0 else 0.0
        num_positives = mask.sum().item()

        loss_dict = {
            'contrastive_loss': loss.item(),
            'avg_similarity': avg_similarity,
            'num_positives': num_positives
        }

        return loss, loss_dict


class TripletContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0, anti_collapse_weight=5.0):  # 🔥 增大权重
        super().__init__()
        self.margin = margin
        self.anti_collapse_weight = anti_collapse_weight

    def forward(self, context_emb, gpt_emb, labels):
        # 🔥 不要归一化！
        # context_emb = F.normalize(context_emb, dim=1)  # 删除
        # gpt_emb = F.normalize(gpt_emb, dim=1)          # 删除

        # 计算相似度（点积，不是余弦）
        similarity = (context_emb * gpt_emb).sum(dim=1)

        factual_mask = (labels == 1)
        hallucination_mask = (labels == 0)

        loss = 0.0

        # 事实样本：相似度应该 > margin
        if factual_mask.sum() > 0:
            factual_sim = similarity[factual_mask]
            factual_loss = torch.relu(self.margin - factual_sim).mean()
            loss += factual_loss

        # 幻觉样本：相似度应该 < -margin
        if hallucination_mask.sum() > 0:
            hallucination_sim = similarity[hallucination_mask]
            hallucination_loss = torch.relu(hallucination_sim + self.margin).mean()
            loss += hallucination_loss

        # 🔥 强化的范数惩罚
        context_norm = context_emb.norm(dim=1).mean()
        gpt_norm = gpt_emb.norm(dim=1).mean()

        # 目标范数：鼓励嵌入达到较大的范数
        target_norm = 2.0  # 🔥 提高目标范数（从1.0→2.0）

        # L2损失：惩罚范数偏离目标
        norm_loss = (context_norm - target_norm) ** 2 + (gpt_norm - target_norm) ** 2

        # 🔥 方差惩罚（防止所有嵌入聚在一起）
        context_var = context_emb.var(dim=0).mean()
        gpt_var = gpt_emb.var(dim=0).mean()

        target_var = 1.0  # 鼓励高方差
        var_loss = torch.relu(target_var - context_var) + torch.relu(target_var - gpt_var)

        # 🔥 总损失（增大反坍塌权重）
        total_loss = loss + self.anti_collapse_weight * (norm_loss + var_loss)

        # 统计信息
        loss_dict = {
            'triplet_loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'norm_loss': norm_loss.item(),
            'var_loss': var_loss.item(),
            'total_loss': total_loss.item(),
            'avg_factual_sim': similarity[factual_mask].mean().item() if factual_mask.sum() > 0 else 0.0,
            'avg_hallucination_sim': similarity[
                hallucination_mask].mean().item() if hallucination_mask.sum() > 0 else 0.0,
            'sim_gap': (similarity[factual_mask].mean() - similarity[
                hallucination_mask].mean()).item() if factual_mask.sum() > 0 and hallucination_mask.sum() > 0 else 0.0,
            'context_norm': context_norm.item(),
            'gpt_norm': gpt_norm.item(),
            'context_var': context_var.item(),
            'gpt_var': gpt_var.item()
        }

        return total_loss, loss_dict

class RGATTrainer:
    """RGAT训练器"""
    
    def __init__(self, config_dict, use_balanced_sampling=True, sampler_type='batch_balanced'):
        """
        初始化训练器
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict
        self.use_balanced_sampling = use_balanced_sampling

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        create_directories()
        
        # 检查嵌入文件
        self._check_embedding_files()
        
        # 加载数据集
        print("\n" + "="*60)
        print("加载数据集")
        print("="*60)
        self.dataset = MintakaGraphDataset(
            config_dict['data_path'],
            config_dict['entity_mapping_path'],
            config_dict['relation_mapping_path'],
            max_samples=config_dict.get('max_samples')
        )
        
        # 划分训练集、验证集、测试集
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                self.dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config_dict.get('seed', 42))
            )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        print(f"  测试集: {len(self.test_dataset)} 样本")
        
        # 数据加载器
        # 数据加载器
        if use_balanced_sampling:
            print(f"\n" + "=" * 60)
            print("使用平衡采样")
            print("=" * 60)

            # 导入采样器
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))

            # 🔥 训练集：根据 sampler_type 选择采样策略
            if sampler_type == 'batch_balanced':
                # ⭐ 新增：批次级别1:1平衡采样
                from EpochBalancedSampler import BatchBalancedSampler
                print("训练集采样器: 批次级1:1平衡采样 ⭐")
                print("  → 保证每个batch内幻觉:事实 = 1:1")

                batch_size = config_dict['batch_size']
                if batch_size % 2 != 0:
                    print(f"⚠️ 警告: batch_size={batch_size}不是偶数，自动调整为{batch_size - 1}")
                    batch_size = batch_size - 1
                    config_dict['batch_size'] = batch_size

                train_sampler = BatchBalancedSampler(
                    self.train_dataset,
                    batch_size=batch_size
                )

            elif sampler_type == 'random':
                from EpochBalancedSampler import EpochBalancedSampler
                print("训练集采样器: Epoch随机平衡采样")
                print("  → 整个epoch平衡，batch内不保证")
                train_sampler = EpochBalancedSampler(self.train_dataset, seed=config_dict.get('seed', 42))

            elif sampler_type == 'sequential':
                from EpochBalancedSampler import SequentialBalancedSampler
                print("训练集采样器: 顺序平衡采样")
                print("  → 整个epoch平衡，batch内不保证")
                train_sampler = SequentialBalancedSampler(self.train_dataset, seed=config_dict.get('seed', 42))

            else:  # stratified
                from EpochBalancedSampler import StratifiedBalancedSampler
                print("训练集采样器: 分层平衡采样")
                print("  → 整个epoch平衡，batch内尽量平衡")
                train_sampler = StratifiedBalancedSampler(
                    self.train_dataset,
                    batch_size=config_dict['batch_size'],
                    seed=config_dict.get('seed', 42)
                )

            # 🔥 验证集和测试集：固定采样（保持一致性）
            from EpochBalancedSampler import FixedBalancedSampler
            print("验证/测试集采样器: 固定平衡采样（保持评估一致性）")

            val_sampler = FixedBalancedSampler(self.val_dataset, seed=42)
            test_sampler = FixedBalancedSampler(self.test_dataset, seed=43)  # 不同种子

            self.train_sampler = train_sampler  # 🔥 保存引用，用于set_epoch

            # 数据加载器（使用采样器）
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config_dict['batch_size'],
                sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=config_dict.get('num_workers', 0)
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config_dict['batch_size'],
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=config_dict.get('num_workers', 0)
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config_dict['batch_size'],
                sampler=test_sampler,
                collate_fn=collate_fn,
                num_workers=config_dict.get('num_workers', 0)
            )

        else:
            # 原始的数据加载器（使用shuffle）
            print("\n使用标准采样（shuffle=True）")
            self.train_sampler = None  # 🔥 添加这个

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=config_dict.get('num_workers', 0)
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=config_dict.get('num_workers', 0)
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=config_dict.get('num_workers', 0)
            )
        
        # 初始化模型（使用RGAT）
        print("\n" + "="*60)
        print("初始化RGAT模型")
        print("="*60)
        self.model = SiameseRGATWithEmbedding(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            hidden_channels=config_dict['hidden_channels'],
            out_channels=config_dict['out_channels'],
            num_layers=config_dict['num_layers'],
            freeze_embeddings=config_dict.get('freeze_embeddings', True),
            dropout=config_dict.get('dropout', 0.3),
            num_heads=config_dict.get('num_heads', 4)  # 🔹 RGAT特有参数
        ).to(self.device)
        
        print(f"\n模型参数:")
        print(f"  隐藏层维度: {config_dict['hidden_channels']}")
        print(f"  输出维度: {config_dict['out_channels']}")
        print(f"  层数: {config_dict['num_layers']}")
        print(f"  注意力头数: {config_dict.get('num_heads', 4)} 🔹")
        print(f"  Dropout: {config_dict.get('dropout', 0.3)}")
        
        # 损失函数
        # ==================== 配置损失函数 ====================
        use_triplet = config_dict.get('use_triplet_contrastive', False)
        use_supervised = config_dict.get('use_supervised_contrastive', False)

        print(f"\n" + "=" * 60)
        print("配置损失函数")
        print("=" * 60)

        if use_triplet:
            self.criterion = TripletContrastiveLoss(
                margin=config_dict.get('triplet_margin', 2.0),
                anti_collapse_weight=config_dict.get('anti_collapse_weight', 5.0)  # ✅ 改为这个
            )
            self.loss_type = 'triplet'

            print(f"损失函数: 三元组对比学习 ⭐")
            print(f"  Margin: {config_dict.get('triplet_margin', 2.0)}")
            print(f"  反坍塌权重: {config_dict.get('anti_collapse_weight', 5.0)}")
            print(f"  说明: 事实样本GPT图接近context图，幻觉样本远离")
            print(f"  优势: 直接利用GPT-Context关系学习区分性特征")

        elif use_supervised:
            # 方案：监督对比学习
            self.criterion = SupervisedContrastiveLoss(
                temperature=config_dict.get('temperature', 0.07)
            )
            self.loss_type = 'supervised'
            print(f"损失函数: 监督对比学习")

        else:
            # 方案：无监督对比学习
            self.criterion = ImprovedContrastiveLoss(
                margin=config_dict['margin'],
                temperature=config_dict['temperature'],
                alpha=config_dict['alpha']
            )
            self.loss_type = 'unsupervised'
            print(f"损失函数: 无监督对比学习")

        print("=" * 60)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )
        if self.config.get('scheduler_type') == 'cosine':
            print("\n使用CosineAnnealingWarmRestarts")
        # 学习率调度器
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config_dict.get('t_0', 10),
                T_mult=config_dict.get('t_mult', 2),
                eta_min=config_dict.get('eta_min', 1e-6)
            )
        elif self.config.get('scheduler_type') == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # 监控验证loss（越小越好）
                factor=0.5,  # 每次降低50%
                patience=5,  # 5个epoch不改善就降低
                verbose=True,
                min_lr=1e-6
            )
            self.scheduler_type = 'reduce_on_plateau'
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _check_embedding_files(self):
        """检查嵌入文件是否存在"""
        print("\n检查嵌入文件...")
        
        entity_emb_path = self.config['entity_embedding_path']
        relation_emb_path = self.config['relation_embedding_path']
        
        if not os.path.exists(entity_emb_path):
            print(f"\n❌ 错误: RGCN实体嵌入文件不存在")
            print(f"期望路径: {entity_emb_path}")
            print(f"\n这个文件应该由 prepare_embeddings.py 生成")
            print(f"\n请按以下步骤操作:")
            print(f"  1. 确保已生成混合嵌入")
            print(f"     python generate_hybrid_embeddings.py")
            print(f"  2. 运行嵌入准备脚本")
            print(f"     python prepare_embeddings.py")
            exit(1)
        
        if not os.path.exists(relation_emb_path):
            print(f"\n❌ 错误: RGCN关系映射文件不存在")
            print(f"期望路径: {relation_emb_path}")
            print(f"\n请先运行: python prepare_embeddings.py")
            exit(1)
        
        print("✓ 嵌入文件检查通过")
        print(f"  - 实体嵌入: {os.path.basename(entity_emb_path)}")
        print(f"  - 关系映射: {os.path.basename(relation_emb_path)}")

    def train_epoch(self, epoch):
        """训练一个epoch（支持梯度累积）"""
        self.model.train()
        total_loss = 0
        batch_count = 0

        # 🔥 梯度累积配置
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)

        # 🔥 设置采样器的epoch（用于动态采样）
        if hasattr(self, 'train_sampler') and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')

        # 🔥 在epoch开始时清零梯度
        self.optimizer.zero_grad()

        for batch_idx, (context_batch, gpt_batch, labels, metadata_list) in enumerate(pbar):
            if context_batch is None:
                continue

            # 移到设备
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            # 🔥 根据损失函数类型选择不同的前向传播方式
            if self.loss_type == 'triplet':
                # 三元组对比学习：需要context和gpt两个嵌入 + 标签
                context_emb, gpt_emb = self.model(context_batch, gpt_batch)
                loss, loss_dict = self.criterion(context_emb, gpt_emb, labels)

            elif self.loss_type == 'supervised':
                # 监督对比学习：只需要GPT图的嵌入 + 标签
                _, gpt_emb = self.model(context_batch, gpt_batch)
                loss, loss_dict = self.criterion(gpt_emb, labels)

            else:  # 'unsupervised'
                # 无监督对比学习：需要两个图的嵌入
                context_emb, gpt_emb = self.model(context_batch, gpt_batch)
                # 假设所有样本对都是负样本对（context vs gpt）
                similarity_labels = torch.zeros(context_emb.size(0)).to(self.device)
                loss, loss_dict = self.criterion(context_emb, gpt_emb, similarity_labels)

            # 🔥 梯度累积：loss除以累积步数
            loss = loss / accumulation_steps

            # 反向传播（累积梯度）
            loss.backward()

            # 🔥 每accumulation_steps步才更新一次参数
            if (batch_idx + 1) % accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('gradient_clip_norm', 1.0)
                )

                # 更新参数
                self.optimizer.step()

                # 清零梯度（为下一次累积做准备）
                self.optimizer.zero_grad()

            # 🔥 统计loss时恢复原始值
            total_loss += loss.item() * accumulation_steps
            batch_count += 1

            # 🔥 更新进度条（根据不同损失函数显示不同信息）
            if self.loss_type == 'triplet':
                # 三元组损失：显示事实/幻觉相似度差距
                pbar.set_postfix({
                    'loss': f'{(loss.item() * accumulation_steps):.4f}',  # 显示真实loss
                    'fact_sim': f'{loss_dict.get("avg_factual_sim", 0):.2f}',
                    'hall_sim': f'{loss_dict.get("avg_hallucination_sim", 0):.2f}',
                    'gap': f'{loss_dict.get("sim_gap", 0):.2f}',  # ⭐ 关键指标
                    'accum': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'  # 🔥 显示累积进度
                })
            elif self.loss_type == 'supervised':
                # 监督对比学习：显示相似度和正样本对数
                pbar.set_postfix({
                    'loss': f'{(loss.item() * accumulation_steps):.4f}',
                    'sim': f'{loss_dict.get("avg_similarity", 0):.3f}',
                    'pos': loss_dict.get("num_positives", 0),
                    'accum': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'
                })
            else:
                # 无监督对比学习：显示基本信息
                pbar.set_postfix({
                    'loss': f'{(loss.item() * accumulation_steps):.4f}',
                    'sim': f'{loss_dict.get("avg_similarity", 0):.3f}',
                    'accum': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'
                })

        # 🔥 处理最后不足accumulation_steps的剩余batch
        # 如果还有未更新的梯度，在epoch结束时更新
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip_norm', 1.0)
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        batch_count = 0

        for context_batch, gpt_batch, labels, metadata_list in self.val_loader:
            if context_batch is None:
                continue

            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            # 🔥 根据损失函数类型选择不同的前向传播方式
            if self.loss_type == 'triplet':
                # 三元组对比学习：需要context和gpt两个嵌入 + 标签
                context_emb, gpt_emb = self.model(context_batch, gpt_batch)
                loss, loss_dict = self.criterion(context_emb, gpt_emb, labels)

            elif self.loss_type == 'supervised':
                # 监督对比学习：只需要GPT图的嵌入 + 标签
                _, gpt_emb = self.model(context_batch, gpt_batch)
                loss, loss_dict = self.criterion(gpt_emb, labels)

            else:  # 'unsupervised'
                # 无监督对比学习：需要两个图的嵌入
                context_emb, gpt_emb = self.model(context_batch, gpt_batch)
                # 假设所有样本对都是负样本对
                similarity_labels = torch.zeros(context_emb.size(0)).to(self.device)
                loss, loss_dict = self.criterion(context_emb, gpt_emb, similarity_labels)

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss
    
    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练 RGAT")
        print("=" * 60)

        # 🔥 诊断前几个batch的标签分布
        print("\n诊断前10个batch的标签分布...")
        for i, (_, _, labels, _) in enumerate(self.train_loader):
            if i >= 10:
                break
            num_hall = (labels == 0).sum().item()
            num_fact = (labels == 1).sum().item()
            print(f"  Batch {i + 1}: 幻觉={num_hall}, 事实={num_fact}")
        print("="*60)
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")
        print(f"早停耐心: {self.config['early_stopping_patience']}")
        
        for epoch in range(self.config['num_epochs']):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # 更新学习率
            if self.scheduler_type == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 打印信息
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_rgat_model.pth', epoch, val_loss)
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\n早停触发 (patience={self.config['early_stopping_patience']})")
                break
            
            # 定期保存
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss)
        
        # 训练结束
        print("\n" + "="*60)
        print("训练完成")
        print("="*60)
        print(f"最佳验证损失: {self.best_val_loss:.4f}")

        # 🔥 诊断最终编码器质量
        print("\n加载最佳模型进行诊断...")

        # 🔥 在加载前清理显存
        del self.optimizer  # 删除优化器（占用大量显存）
        if hasattr(self, 'scheduler'):
            del self.scheduler
        torch.cuda.empty_cache()  # 清理显存碎片
        import gc
        gc.collect()  # 垃圾回收

        print("✓ 显存已清理")

        best_checkpoint = torch.load(
            os.path.join(self.config['checkpoint_dir'], 'best_rgat_model.pth'),
            map_location=self.device
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        self.diagnose_encoder_quality()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 保存训练历史
        self.save_training_history()
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, checkpoint_path)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RGAT Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(self.config['output_dir'], 'rgat_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {plot_path}")
        plt.close()
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        history_path = os.path.join(self.config['output_dir'], 'rgat_training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {history_path}")


    def diagnose_encoder_quality(self):
        """诊断编码器学习的区分性特征质量"""
        print("\n" + "=" * 60)
        print("🔍 编码器质量诊断")
        print("=" * 60)

        self.model.eval()
        hallucination_embeddings = []
        factual_embeddings = []

        with torch.no_grad():
            # 从训练集采样一些样本
            sample_count = 0
            max_samples = 200  # 采样200个样本

            for context_batch, gpt_batch, labels, _ in self.train_loader:
                if sample_count >= max_samples:
                    break

                if context_batch is None:
                    continue

                gpt_batch = gpt_batch.to(self.device)
                labels = labels.numpy()

                # 获取GPT图的嵌入
                _, gpt_emb = self.model(context_batch.to(self.device), gpt_batch)
                gpt_emb = gpt_emb.cpu().numpy()

                # 按标签分类收集
                for i, label in enumerate(labels):
                    if label == 0:
                        hallucination_embeddings.append(gpt_emb[i])
                    else:
                        factual_embeddings.append(gpt_emb[i])
                    sample_count += 1

                    if sample_count >= max_samples:
                        break

        if len(hallucination_embeddings) > 0 and len(factual_embeddings) > 0:
            hall_emb = np.array(hallucination_embeddings)
            fact_emb = np.array(factual_embeddings)

            # 计算每类的均值嵌入
            hall_mean = hall_emb.mean(axis=0)
            fact_mean = fact_emb.mean(axis=0)

            # 计算类内方差
            hall_var = ((hall_emb - hall_mean) ** 2).mean()
            fact_var = ((fact_emb - fact_mean) ** 2).mean()

            # 计算类间距离
            inter_class_distance = np.linalg.norm(hall_mean - fact_mean)

            # 计算类内距离
            intra_class_distance = (hall_var + fact_var) / 2

            # Fisher判别比（类间距离 / 类内距离）
            fisher_ratio = inter_class_distance / (np.sqrt(intra_class_distance) + 1e-6)

            print(f"\n嵌入空间分析:")
            print(f"  样本数: 幻觉={len(hallucination_embeddings)}, 事实={len(factual_embeddings)}")
            print(f"  幻觉嵌入均值范数: {np.linalg.norm(hall_mean):.4f}")
            print(f"  事实嵌入均值范数: {np.linalg.norm(fact_mean):.4f}")
            print(f"\n区分性分析:")
            print(f"  类间距离: {inter_class_distance:.4f}")
            print(f"  类内方差: {intra_class_distance:.4f}")
            print(f"  Fisher判别比: {fisher_ratio:.4f}")

            if fisher_ratio < 1.0:
                print(f"  ⚠️ 警告: Fisher比 < 1.0，编码器区分能力弱")
                print(f"  建议: 继续训练或调整超参数")
            elif fisher_ratio < 2.0:
                print(f"  ⚙️ 一般: Fisher比在1-2之间，编码器有一定区分能力")
            else:
                print(f"  ✓ 良好: Fisher比 > 2.0，编码器具有良好区分能力")
        else:
            print(f"  ⚠️ 样本不足，无法进行分析")

        print("=" * 60)
        self.model.train()

def main():
    """主函数"""
    # 打印配置
    print("\n" + "="*60)
    print("RGAT训练配置 - Mintaka")
    print("="*60)
    Config.print_config()

    # 获取配置字典
    config_dict = Config.get_config_dict()

    # 确认使用RGAT
    print(f"\n使用模型: R-GAT (关系图注意力网络)")
    print(f"数据集: Mintaka")
    print(f"注意力头数: {config_dict.get('num_heads', 4)}")

    # 创建训练器
    trainer = RGATTrainer(config_dict)

    # 开始训练
    trainer.train()

    print("\n✓ RGAT训练完成！")
    print(f"\n生成的文件:")
    print(f"  最佳模型: {Config.BEST_MODEL_PATH}")
    print(f"  训练曲线: {os.path.join(Config.OUTPUT_DIR, 'rgat_training_curves.png')}")
    print(f"  训练历史: {os.path.join(Config.OUTPUT_DIR, 'rgat_training_history.json')}")


if __name__ == '__main__':
    main()
