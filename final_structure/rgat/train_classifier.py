"""
RGAT分类器训练脚本（使用预训练RGAT编码器）
核心改进：
1. 使用预训练的Siamese RGAT编码器
2. 集成批次级平衡采样器（确保每个批次内类别平衡）
3. 支持冻结/微调编码器
"""
from turtle import config_dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 导入配置和模型
from config_hotpotqa_classifier import Config, create_directories
from classifier_with_pretrained_rgat import HallucinationClassifierWithPretrainedRGAT

# 导入数据加载器（复用RGCN的）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rgcn'))
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn

# 🔥 导入平衡采样器（包括BatchBalancedSampler）
sys.path.insert(0, os.path.dirname(__file__))
from EpochBalancedSampler import (
    EpochBalancedSampler,
    SequentialBalancedSampler,
    StratifiedBalancedSampler,
    FixedBalancedSampler,
    BatchBalancedSampler  # 🔥 新增
)


class RGATClassifierTrainer:
    """RGAT分类器训练器（带批次级平衡采样）"""

    def __init__(self, config_dict, pretrained_rgat_path,
                 use_balanced_sampling=True,
                 sampler_type='batch_balanced'):  # 🔥 默认使用batch_balanced
        """
        Args:
            config_dict: 配置字典
            pretrained_rgat_path: 预训练RGAT模型路径
            freeze_encoder: 是否冻结编码器
            use_balanced_sampling: 是否使用平衡采样
            sampler_type: 采样器类型 ('epoch', 'sequential', 'stratified', 'batch_balanced')
        """
        self.config = config_dict
        self.pretrained_rgat_path = pretrained_rgat_path
        self.freeze_encoder = config_dict['freeze_encoder']
        self.use_balanced_sampling = use_balanced_sampling
        self.sampler_type = sampler_type

        # 🔥 验证批次大小必须是偶数（BatchBalancedSampler要求）
        if sampler_type == 'batch_balanced' and config_dict['batch_size'] % 2 != 0:
            raise ValueError(
                f"❌ 使用BatchBalancedSampler时，batch_size必须是偶数！\n"
                f"   当前: {config_dict['batch_size']}\n"
                f"   建议: {config_dict['batch_size'] + 1}"
            )

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载数据集
        print("\n" + "="*60)
        print("加载HotpotQA数据集")
        print("="*60)
        self.dataset = HotpotQAGraphDataset(
            config_dict['data_path'],
            config_dict['entity_mapping_path'],
            config_dict['relation_mapping_path'],
            max_samples=config_dict.get('max_samples')
        )

        # 划分数据集
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

        # 🔥 使用平衡采样器
        if use_balanced_sampling:
            print(f"\n{'='*60}")
            print(f"使用平衡采样器")
            print(f"{'='*60}")
            print(f"采样器类型: {sampler_type}")

            # 训练集采样器
            if sampler_type == 'batch_balanced':
                # 🔥 批次级平衡采样（确保每个批次内1:1）
                train_sampler = BatchBalancedSampler(
                    self.train_dataset,
                    batch_size=config_dict['batch_size']
                )
                print("📊 训练集: 批次级平衡采样 (每批次内1:1)")
                print(f"   - 批次大小: {config_dict['batch_size']}")
                print(f"   - 每批次: {config_dict['batch_size']//2} 幻觉 + {config_dict['batch_size']//2} 事实")

            elif sampler_type == 'epoch':
                train_sampler = EpochBalancedSampler(
                    self.train_dataset,
                    seed=config_dict.get('seed', 42)
                )
                print("📊 训练集: Epoch随机平衡采样")

            elif sampler_type == 'sequential':
                train_sampler = SequentialBalancedSampler(
                    self.train_dataset,
                    seed=config_dict.get('seed', 42)
                )
                print("📊 训练集: 顺序平衡采样")

            elif sampler_type == 'stratified':
                train_sampler = StratifiedBalancedSampler(
                    self.train_dataset,
                    batch_size=config_dict['batch_size'],
                    seed=config_dict.get('seed', 42)
                )
                print("📊 训练集: 分层平衡采样")
            else:
                raise ValueError(f"未知采样器类型: {sampler_type}")

            # 验证集和测试集采样器（固定采样）
            val_sampler = FixedBalancedSampler(self.val_dataset, seed=42)
            test_sampler = FixedBalancedSampler(self.test_dataset, seed=42)
            print("📊 验证集/测试集: 固定平衡采样")

            # 数据加载器
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config_dict['batch_size'],
                sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=0
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config_dict['batch_size'],
                sampler=val_sampler,
                collate_fn=collate_fn,
                num_workers=0
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config_dict['batch_size'],
                sampler=test_sampler,
                collate_fn=collate_fn,
                num_workers=0
            )

            self.train_sampler = train_sampler
            self.val_sampler = val_sampler
            self.test_sampler = test_sampler

        else:
            print("\n不使用平衡采样（原始shuffle）")

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0
            )

            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=config_dict['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )

            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

        # 初始化模型
        print("\n" + "="*60)
        print("初始化RGAT分类器")
        print("="*60)
        print(f"预训练模型: {pretrained_rgat_path}")
        print(f"编码器策略: {'冻结' if self.freeze_encoder else '微调'}")

        self.model = HallucinationClassifierWithPretrainedRGAT(
            pretrained_model_path=pretrained_rgat_path,
            freeze_encoder=self.freeze_encoder,
            ffn_hidden_dim=config_dict.get('ffn_hidden_dim', 128),
            dropout=config_dict.get('dropout', 0.3)
        ).to(self.device)

        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

        # 损失函数（交叉熵）
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )

        # 学习率调度器
        if config_dict.get('scheduler_type') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
            self.scheduler_type = 'plateau'
        elif config_dict.get('scheduler_type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            self.scheduler_type = 'cosine'
        else:
            self.scheduler = None
            self.scheduler_type = None

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 🔥 设置采样器的epoch（用于动态采样）
        if self.train_sampler is not None and hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [训练]")
        for batch_idx, (context_batch, gpt_batch, labels, _) in enumerate(pbar):
            # 数据转移到设备
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            # 🔥 统计批次内的类别分布（用于监控）
            num_hallucination = (labels == 0).sum().item()
            num_factual = (labels == 1).sum().item()

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(context_batch, gpt_batch)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip_norm']
                )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # 🔥 更新进度条（显示批次内类别分布）
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total*100:.1f}%',
                'h/f': f'{num_hallucination}/{num_factual}'  # 幻觉/事实
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, data_loader, desc="验证"):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"[{desc}]")
            for context_batch, gpt_batch, labels, _ in pbar:
                context_batch = context_batch.to(self.device)
                gpt_batch = gpt_batch.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(context_batch, gpt_batch)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 🔥 实时显示准确率
                current_acc = np.mean(
                    np.array(all_predictions) == np.array(all_labels)
                )
                pbar.set_postfix({'acc': f'{current_acc*100:.1f}%'})

        avg_loss = total_loss / len(data_loader)

        # 计算指标
        f1 = f1_score(all_labels, all_predictions, average='binary')
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        return avg_loss, accuracy, f1, all_predictions, all_labels

    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练 RGAT 分类器")
        print("="*60)
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")
        print(f"早停耐心: {self.config.get('early_stopping_patience', 10)}")

        for epoch in range(self.config['num_epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_acc, val_f1, _, _ = self.validate(self.val_loader, "验证")
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)

            # 打印信息
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f}")

            # 更新学习率
            if self.scheduler is not None:
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()
                print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                self.save_checkpoint('best_rgat_classifier.pth', epoch, val_f1)
                print(f"  ✓ 保存最佳模型 (F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1

            # 早停
            early_stopping_patience = self.config.get('early_stopping_patience', 10)
            if self.patience_counter >= early_stopping_patience:
                print(f"\n早停触发 (patience={early_stopping_patience})")
                break

        # 训练结束
        print("\n" + "="*60)
        print("训练完成")
        print("="*60)
        print(f"最佳验证F1: {self.best_val_f1:.4f}")

        # 测试集评估
        self.evaluate_test_set()

        # 绘制训练曲线
        self.plot_training_curves()

        # 保存训练历史
        self.save_training_history()

    def evaluate_test_set(self):
        """在测试集上评估"""
        print("\n" + "="*60)
        print("测试集评估")
        print("="*60)

        # 加载最佳模型
        best_model_path = os.path.join(
            self.config['checkpoint_dir'],
            'best_rgat_classifier.pth'
        )
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 评估
        test_loss, test_acc, test_f1, predictions, labels = \
            self.validate(self.test_loader, "测试")

        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc*100:.2f}%")
        print(f"  F1 Score: {test_f1:.4f}")

        # 详细报告
        print("\n分类报告:")
        print(classification_report(
            labels, predictions,
            target_names=['幻觉', '事实'],
            digits=4
        ))

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        print("\n混淆矩阵:")
        print(f"              预测幻觉  预测事实")
        print(f"真实幻觉:     {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"真实事实:     {cm[1][0]:6d}    {cm[1][1]:6d}")

        # 保存测试结果
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                labels, predictions,
                target_names=['幻觉', '事实'],
                output_dict=True
            )
        }

        result_path = os.path.join(
            self.config['output_dir'],
            'rgat_classifier_test_results.json'
        )
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n测试结果已保存到: {result_path}")

    def save_checkpoint(self, filename, epoch, val_f1):
        """保存模型检查点"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_f1': val_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'config': self.config,
            'pretrained_rgat_path': self.pretrained_rgat_path,
            'freeze_encoder': self.freeze_encoder
        }, checkpoint_path)

    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('RGAT Classifier Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1曲线
        axes[1].plot(self.val_f1_scores, label='Val F1', linewidth=2, color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('RGAT Classifier Validation F1')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plot_path = os.path.join(
            self.config['output_dir'],
            'rgat_classifier_training_curves.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n训练曲线已保存到: {plot_path}")
        plt.close()

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'use_balanced_sampling': self.use_balanced_sampling,
            'sampler_type': self.sampler_type,
            'freeze_encoder': self.freeze_encoder
        }

        history_path = os.path.join(
            self.config['output_dir'],
            'rgat_classifier_training_history.json'
        )
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"训练历史已保存到: {history_path}")


def main():
    """主函数"""
    # 打印配置
    print("\n" + "="*60)
    print("RGAT分类器训练配置")
    print("="*60)
    Config.print_config()

    # 获取配置字典
    config_dict = Config.get_config_dict()

    # 🔥 验证batch_size是偶数
    if config_dict['batch_size'] % 2 != 0:
        print(f"\n⚠️  警告: batch_size={config_dict['batch_size']} 不是偶数")
        print(f"   建议修改 config_hotpotqa_rgat.py:")
        print(f"   BATCH_SIZE = {config_dict['batch_size'] + 1}")
        print(f"\n使用BatchBalancedSampler需要偶数批次大小以实现1:1平衡")
        return

    # 🔥 预训练RGAT模型路径
    pretrained_rgat_path = Config.BEST_MODEL_PATH  # best_rgat_model.pth

    if not os.path.exists(pretrained_rgat_path):
        print(f"\n❌ 错误: 未找到预训练RGAT模型")
        print(f"   路径: {pretrained_rgat_path}")
        print(f"\n请先训练Siamese RGAT模型:")
        print(f"   python train_rgat_hotpotqa.py")
        return

    # 创建训练器
    print(f"\n🔹 使用预训练RGAT编码器")
    print(f"🔹 模型路径: {pretrained_rgat_path}")
    print(f"🔹 编码器策略: 微调 (freeze_encoder=False)")
    print(f"🔹 使用平衡采样: ✓")
    print(f"🔹 采样器类型: batch_balanced (每批次内1:1)")

    trainer = RGATClassifierTrainer(
        config_dict,
        pretrained_rgat_path=pretrained_rgat_path,
        use_balanced_sampling=True,  # 🔥 使用平衡采样
        sampler_type='batch_balanced'  # 🔥 批次级平衡采样
    )

    # 开始训练
    trainer.train()

    print("\n✓ RGAT分类器训练完成！")
    print(f"\n生成的文件:")
    print(f"  最佳模型: {os.path.join(Config.CHECKPOINT_DIR, 'best_rgat_classifier.pth')}")
    print(f"  训练曲线: {os.path.join(Config.OUTPUT_DIR, 'rgat_classifier_training_curves.png')}")
    print(f"  训练历史: {os.path.join(Config.OUTPUT_DIR, 'rgat_classifier_training_history.json')}")
    print(f"  测试结果: {os.path.join(Config.OUTPUT_DIR, 'rgat_classifier_test_results.json')}")


if __name__ == '__main__':
    main()