"""
Mintaka 数据集的分类器训练脚本
使用预训练的 RGAT 编码器训练幻觉检测分类器
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 导入配置
from config_mintaka_rgat import Config, create_directories

# 导入数据加载器
from data_loader_mintaka import MintakaGraphDataset, collate_fn

# 导入分类器
from classifier_with_pretrained_rgat import HallucinationClassifierWithPretrainedRGAT


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ClassifierTrainer:
    """分类器训练器"""

    def __init__(self, config_dict, pretrained_model_path):
        """
        初始化训练器
        Args:
            config_dict: 配置字典
            pretrained_model_path: 预训练RGAT模型路径
        """
        self.config = config_dict
        self.pretrained_model_path = pretrained_model_path

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 创建输出目录
        create_directories()

        # 加载数据集
        print("\n" + "=" * 60)
        print("加载数据集")
        print("=" * 60)
        self.dataset = MintakaGraphDataset(
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

        # 数据加载器
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

        # 初始化分类器
        print("\n" + "=" * 60)
        print("初始化分类器")
        print("=" * 60)
        self.model = HallucinationClassifierWithPretrainedRGAT(
            pretrained_model_path=pretrained_model_path,
            freeze_encoder=config_dict.get('freeze_encoder', False),
            ffn_hidden_dim=config_dict.get('ffn_hidden_dim', 128),
            dropout=config_dict.get('ffn_dropout', 0.3)
        ).to(self.device)

        # 损失函数
        loss_type = config_dict.get('loss_type', 'focal')
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=config_dict.get('focal_alpha', 0.75),
                gamma=config_dict.get('focal_gamma', 2.0)
            )
            print(f"损失函数: Focal Loss (alpha={config_dict.get('focal_alpha', 0.75)}, gamma={config_dict.get('focal_gamma', 2.0)})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print(f"损失函数: Cross Entropy Loss")

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 监控准确率（越高越好）
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )

        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}')

        for entity_batch, gpt_batch, labels, metadata_list in pbar:
            if entity_batch is None:
                continue

            # 移到设备
            entity_batch = entity_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(gpt_batch, entity_batch)  # 注意顺序：response, reference
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for entity_batch, gpt_batch, labels, metadata_list in self.val_loader:
            if entity_batch is None:
                continue

            entity_batch = entity_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(gpt_batch, entity_batch)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        return avg_loss, acc, precision, recall, f1

    @torch.no_grad()
    def test(self):
        """测试"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for entity_batch, gpt_batch, labels, metadata_list in self.test_loader:
            if entity_batch is None:
                continue

            entity_batch = entity_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)

            logits = self.model(gpt_batch, entity_batch)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\n" + "=" * 60)
        print("测试集结果")
        print("=" * 60)
        print(f"  准确率: {acc:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"\n混淆矩阵:")
        print(f"  {cm}")

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练分类器")
        print("=" * 60)
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")

        for epoch in range(self.config['num_epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # 更新学习率
            self.scheduler.step(val_acc)

            # 打印信息
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"  验证F1: {val_f1:.4f}")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint('best_classifier.pth', epoch, val_acc)
                print(f"  ✓ 保存最佳模型 (val_acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.config.get('early_stopping_patience', 10):
                print(f"\n早停触发")
                break

        # 测试
        print("\n" + "=" * 60)
        print("加载最佳模型进行测试")
        print("=" * 60)
        best_checkpoint = torch.load(
            os.path.join(self.config['checkpoint_dir'], 'best_classifier.pth'),
            map_location=self.device
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        test_results = self.test()

        # 保存结果
        self.save_results(test_results)

    def save_checkpoint(self, filename, epoch, val_acc):
        """保存检查点"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'pretrained_model_path': self.pretrained_model_path
        }, checkpoint_path)

    def save_results(self, test_results):
        """保存结果"""
        results = {
            'test_results': test_results,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'config': self.config
        }

        result_path = os.path.join(self.config['output_dir'], 'classifier_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {result_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='训练幻觉检测分类器')
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='预训练RGAT模型路径')
    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 60)
    print("分类器训练配置 - Mintaka")
    print("=" * 60)
    Config.print_config()

    # 获取配置字典
    config_dict = Config.get_config_dict()

    # 创建训练器
    trainer = ClassifierTrainer(config_dict, args.pretrained_model)

    # 开始训练
    trainer.train()

    print("\n✓ 分类器训练完成！")


if __name__ == '__main__':
    main()
