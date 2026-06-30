# 创建新文件：train_classifier_end2end.py
"""
端到端训练RGAT分类器（不使用预训练）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 导入配置和数据加载器
from config_hotpotqa_classifier import Config, create_directories

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rgcn'))
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn

# 导入平衡采样器
from EpochBalancedSampler import BatchBalancedSampler, FixedBalancedSampler

# 导入RGAT编码器
from siamese_rgat_improved import ImprovedRGATEncoderWithEmbedding


class End2EndHallucinationClassifier(nn.Module):
    """端到端RGAT分类器（从头训练）"""

    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels=128, out_channels=64,
                 num_layers=2, num_heads=4, dropout=0.2):
        super().__init__()

        print(f"\n初始化端到端分类器（从头训练）")
        print(f"  编码器配置:")
        print(f"    - 隐藏层: {hidden_channels}")
        print(f"    - 输出: {out_channels}")
        print(f"    - 层数: {num_layers}")
        print(f"    - 注意力头数: {num_heads}")
        print(f"    - Dropout: {dropout}")

        # RGAT编码器（不加载预训练权重）
        self.encoder = ImprovedRGATEncoderWithEmbedding(
            entity_embedding_path=entity_embedding_path,
            relation_embedding_path=relation_embedding_path,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            freeze_embeddings=True,
            dropout=dropout
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(2 * out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

        print(f"  分类头: {2 * out_channels} → 256 → 128 → 64 → 2")

    def forward(self, response_graph, reference_graph):
        h_response = self.encoder(response_graph)
        h_reference = self.encoder(reference_graph)
        h_concat = torch.cat([h_response, h_reference], dim=-1)
        logits = self.classifier(h_concat)
        return logits


class End2EndTrainer:
    """端到端训练器"""

    def __init__(self, config_dict):
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'=' * 60}")
        print("端到端训练 RGAT 分类器（从头训练）")
        print(f"{'=' * 60}")
        print(f"使用设备: {self.device}")

        # 加载数据集
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
                generator=torch.Generator().manual_seed(42)
            )

        print(f"\n数据集: 训练={len(self.train_dataset)}, "
              f"验证={len(self.val_dataset)}, 测试={len(self.test_dataset)}")

        # 使用平衡采样器
        train_sampler = BatchBalancedSampler(
            self.train_dataset,
            batch_size=config_dict['batch_size']
        )
        val_sampler = FixedBalancedSampler(self.val_dataset, seed=42)
        test_sampler = FixedBalancedSampler(self.test_dataset, seed=42)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config_dict['batch_size'],
            sampler=train_sampler, collate_fn=collate_fn, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config_dict['batch_size'],
            sampler=val_sampler, collate_fn=collate_fn, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config_dict['batch_size'],
            sampler=test_sampler, collate_fn=collate_fn, num_workers=0
        )

        # 初始化模型
        self.model = End2EndHallucinationClassifier(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            hidden_channels=config_dict['hidden_channels'],
            out_channels=config_dict['out_channels'],
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            dropout=config_dict['dropout']
        ).to(self.device)

        # 打印参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.best_val_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pred_count = {0: 0, 1: 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [训练]")
        for context_batch, gpt_batch, labels, _ in pbar:
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(context_batch, gpt_batch)
            loss = self.criterion(logits, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)

            for pred in predictions.cpu().numpy():
                pred_count[pred] += 1

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / total * 100:.1f}%',
                'pred_h/f': f'{pred_count[0]}/{pred_count[1]}'
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        print(f"  训练预测分布: 幻觉={pred_count[0]}, 事实={pred_count[1]}")

        return avg_loss, accuracy

    def validate(self, data_loader, desc="验证"):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for context_batch, gpt_batch, labels, _ in tqdm(data_loader, desc=f"[{desc}]"):
                context_batch = context_batch.to(self.device)
                gpt_batch = gpt_batch.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(context_batch, gpt_batch)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_labels, all_predictions, average='binary')
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

        return avg_loss, accuracy, f1, all_predictions, all_labels

    def train(self):
        print(f"\n开始训练...")
        print(f"  Epochs: {self.config['num_epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  学习率: {self.config['learning_rate']}")

        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss, val_acc, val_f1, _, _ = self.validate(self.val_loader, "验证")
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)

            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%, F1: {val_f1:.4f}")

            self.scheduler.step(val_f1)
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0

                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    'best_end2end_classifier.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': val_f1,
                    'config': self.config
                }, checkpoint_path)
                print(f"  ✓ 保存最佳模型 (F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= 20:
                print(f"\n早停触发")
                break

        print(f"\n训练完成！最佳F1: {self.best_val_f1:.4f}")

        # 测试集评估
        self.evaluate_test()

    def evaluate_test(self):
        print(f"\n{'=' * 60}")
        print("测试集评估")
        print(f"{'=' * 60}")

        # 加载最佳模型
        checkpoint = torch.load(
            os.path.join(self.config['checkpoint_dir'], 'best_end2end_classifier.pth')
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_f1, predictions, labels = \
            self.validate(self.test_loader, "测试")

        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc * 100:.2f}%")
        print(f"  F1 Score: {test_f1:.4f}")

        print("\n分类报告:")
        print(classification_report(labels, predictions, target_names=['幻觉', '事实'], digits=4))

        cm = confusion_matrix(labels, predictions)
        print("\n混淆矩阵:")
        print(f"              预测幻觉  预测事实")
        print(f"真实幻觉:     {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"真实事实:     {cm[1][0]:6d}    {cm[1][1]:6d}")


def main():
    Config.print_config()
    create_directories()

    config_dict = Config.get_config_dict()

    # 调整配置
    config_dict['num_layers'] = 2  # 减少层数，避免过拟合
    config_dict['hidden_channels'] = 128
    config_dict['out_channels'] = 64
    config_dict['num_heads'] = 4
    config_dict['dropout'] = 0.2
    config_dict['batch_size'] = 32
    config_dict['learning_rate'] = 1e-3  # 从头训练，使用较大学习率
    config_dict['num_epochs'] = 200

    print(f"\n调整后的配置:")
    print(f"  层数: {config_dict['num_layers']}")
    print(f"  学习率: {config_dict['learning_rate']}")
    print(f"  Batch size: {config_dict['batch_size']}")

    trainer = End2EndTrainer(config_dict)
    trainer.train()


if __name__ == '__main__':
    main()