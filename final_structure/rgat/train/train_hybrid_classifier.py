"""
训练图+文本混合分类器
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 导入配置
from config.config_hotpotqa_classifier import Config, create_directories

# 导入数据加载器和模型
from dataloader.data_loader_with_text import HotpotQAGraphTextDataset, collate_fn_with_text
from framework.hybrid_graph_text_classifier import HybridGraphTextClassifier
from framework.EpochBalancedSampler import BatchBalancedSampler, FixedBalancedSampler


class HybridClassifierTrainer:
    """图+文本混合分类器训练器"""

    def __init__(self, config_dict):
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 🔥 创建带时间戳的输出文件夹
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"hybrid_{timestamp}"

        # 创建本次训练的专属目录
        self.run_dir = os.path.join(config_dict['output_dir'], self.run_name)
        self.run_checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print("训练图+文本混合分类器")
        print(f"{'=' * 60}")
        print(f"设备: {self.device}")
        print(f"🔥 本次训练ID: {self.run_name}")
        print(f"🔥 输出目录: {self.run_dir}")

        # 加载数据集
        print("\n加载数据集（图+文本）...")
        self.dataset = HotpotQAGraphTextDataset(
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

        print(f"数据集划分:")
        print(f"  训练集: {len(self.train_dataset)}")
        print(f"  验证集: {len(self.val_dataset)}")
        print(f"  测试集: {len(self.test_dataset)}")

        # 数据加载器
        train_sampler = BatchBalancedSampler(self.train_dataset, batch_size=config_dict['batch_size'])
        val_sampler = FixedBalancedSampler(self.val_dataset, seed=42)
        test_sampler = FixedBalancedSampler(self.test_dataset, seed=42)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config_dict['batch_size'],
            sampler=train_sampler, collate_fn=collate_fn_with_text, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config_dict['batch_size'],
            sampler=val_sampler, collate_fn=collate_fn_with_text, num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config_dict['batch_size'],
            sampler=test_sampler, collate_fn=collate_fn_with_text, num_workers=0
        )

        # 初始化模型
        self.model = HybridGraphTextClassifier(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            sbert_model_path=config_dict['sbert_model_path'],
            hidden_channels=config_dict['hidden_channels'],
            out_channels=config_dict['out_channels'],
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            dropout=config_dict['dropout'],
            freeze_text_encoder=True  # 冻结文本编码器
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )

        # 学习率调度器
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        # )

        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # 预热阶段：前3个epoch线性增加学习率
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,  # 从10%开始
            total_iters=3  # 3个epoch预热
        )

        # 主训练阶段：余弦退火
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=47,  # 50 - 3 = 47个epoch
            eta_min=1e-6
        )

        # 组合调度器
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[3]
        )

        print("✓ 使用预热+余弦退火学习率策略")

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_accuracies = []
        self.val_balanced_accuracies = []  
        self.best_val_f1 = 0.0
        self.best_val_acc = 0.0
        self.best_balanced_acc = 0.0  # 而不是 best_val_acc 或 best_val_f1
        self.patience_counter = 0

        config_save_path = os.path.join(self.run_dir, 'config.json')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            # 将不可序列化的对象转换为字符串
            config_to_save = {}
            for k, v in config_dict.items():
                if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    config_to_save[k] = v
                else:
                    config_to_save[k] = str(v)
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        print(f"✓ 配置已保存到: {config_save_path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        pred_count = {0: 0, 1: 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [训练]")
        for batch_data in pbar:
            if batch_data[0] is None:
                continue

            context_batch, gpt_batch, labels, gpt_texts, _ = batch_data
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(context_batch, gpt_batch, gpt_texts)
            loss = self.criterion(logits, labels)
            loss.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
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
            for batch_data in tqdm(data_loader, desc=f"[{desc}]"):
                if batch_data[0] is None:
                    continue

                context_batch, gpt_batch, labels, gpt_texts, _ = batch_data
                context_batch = context_batch.to(self.device)
                gpt_batch = gpt_batch.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(context_batch, gpt_batch, gpt_texts)
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

            # self.scheduler.step(val_f1)
            self.scheduler.step()  # ✅ 每个epoch都step，不管指标
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            # if val_acc > self.best_val_acc:
            #     self.best_val_acc = val_acc
            #     self.patience_counter = 0
            val_loss, val_acc, val_f1, predictions, labels = self.validate(self.val_loader)

            # 🔥 计算更多指标
            from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score

            precision, recall, f1_per_class, _ = precision_recall_fscore_support(
                labels, predictions, average=None, labels=[0, 1]
            )
            balanced_acc = balanced_accuracy_score(labels, predictions)

            # 记录所有指标
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            self.val_balanced_accuracies.append(balanced_acc)  # 新增

            print(f"\nEpoch {epoch + 1}")
            print(f"  验证 - Loss: {val_loss:.4f}")
            print(f"  准确率: {val_acc * 100:.2f}%")
            print(f"  平衡准确率: {balanced_acc * 100:.2f}%")  # 🔥 更好的指标
            print(f"  F1 (宏平均): {val_f1:.4f}")
            print(f"  幻觉类 - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1_per_class[0]:.3f}")
            print(f"  事实类 - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1_per_class[1]:.3f}")

            # 🔥 保存模型的策略：根据任务选择
            # 选项1：平衡准确率（推荐用于不平衡数据）
            val_loss, val_acc, val_f1, predictions, labels = self.validate(self.val_loader)

            # 🔥 计算更多指标
            from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score

            precision, recall, f1_per_class, _ = precision_recall_fscore_support(
                labels, predictions, average=None, labels=[0, 1]
            )
            balanced_acc = balanced_accuracy_score(labels, predictions)

            # 记录所有指标
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            self.val_balanced_accuracies.append(balanced_acc)  # 新增

            print(f"\nEpoch {epoch + 1}")
            print(f"  验证 - Loss: {val_loss:.4f}")
            print(f"  准确率: {val_acc * 100:.2f}%")
            print(f"  平衡准确率: {balanced_acc * 100:.2f}%")  # 🔥 更好的指标
            print(f"  F1 (宏平均): {val_f1:.4f}")
            print(f"  幻觉类 - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1_per_class[0]:.3f}")
            print(f"  事实类 - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1_per_class[1]:.3f}")

            # 🔥 保存模型的策略：根据任务选择
            # 选项1：平衡准确率（推荐用于不平衡数据）
            if balanced_acc > self.best_balanced_acc:
                self.best_balanced_acc = balanced_acc

                checkpoint_path = os.path.join(
                    self.run_checkpoint_dir,  # 🔥 修改：使用run_checkpoint_dir
                    'best_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'config': self.config,
                    'run_name': self.run_name  # 🔥 新增：保存运行ID
                }, checkpoint_path)
                print(f"  ✓ 保存最佳模型 (Acc: {val_acc * 100:.2f}%, F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1

                # 早停
            if self.patience_counter >= 25:  # 使用建议的patience=10
                print(f"\n早停触发 (patience=25)")
                break

        print(f"\n训练完成！最佳验证准确率: {self.best_val_acc * 100:.2f}%")

        self.save_training_history()

        # 测试集评估
        self.evaluate_test()

        # 绘制曲线
        self.plot_curves()

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'run_name': self.run_name,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'total_epochs': len(self.train_losses),
            'config_summary': {
                'batch_size': self.config['batch_size'],
                'learning_rate': self.config['learning_rate'],
                'num_layers': self.config['num_layers'],
                'hidden_channels': self.config['hidden_channels'],
                'dropout': self.config['dropout']
            }
        }

        history_path = os.path.join(self.run_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 训练历史已保存到: {history_path}")

    def evaluate_test(self):
        """在测试集上评估"""
        print(f"\n{'=' * 60}")
        print("测试集评估")
        print(f"{'=' * 60}")

        # 加载最佳模型
        checkpoint_path = os.path.join(self.run_checkpoint_dir, 'best_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 评估
        test_loss, test_acc, test_f1, predictions, labels = \
            self.validate(self.test_loader, "测试")

        # 打印结果
        print(f"\n测试集结果:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc * 100:.2f}%")
        print(f"  F1 Score: {test_f1:.4f}")

        # 🔥 生成分类报告字符串（先定义再使用）
        report_str = classification_report(
            labels, predictions,
            target_names=['幻觉', '事实'],
            digits=4
        )
        print("\n分类报告:")
        print(report_str)

        # 生成混淆矩阵
        cm = confusion_matrix(labels, predictions)
        print("\n混淆矩阵:")
        print(f"              预测幻觉  预测事实")
        print(f"真实幻觉:     {cm[0][0]:6d}    {cm[0][1]:6d}")
        print(f"真实事实:     {cm[1][0]:6d}    {cm[1][1]:6d}")

        # 🔥 保存测试结果
        test_results = {
            'run_name': self.run_name,
            'test_loss': float(test_loss),  # 转换为Python float
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                labels, predictions,
                target_names=['幻觉', '事实'],
                output_dict=True
            ),
            'classification_report_str': report_str  # 现在report_str已经定义了
        }

        result_path = os.path.join(self.run_dir, 'test_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 测试结果已保存到: {result_path}")

    def plot_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Hybrid Classifier Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.val_f1_scores, label='Val F1', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Hybrid Classifier Validation F1')
        axes[1].legend()
        axes[1].grid(True)

        plt.savefig(os.path.join(self.config['output_dir'], 'hybrid_training_curves.png'), dpi=300)
        # 🔥 修改：保存到时间戳文件夹
        plot_path = os.path.join(self.run_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 训练曲线已保存到: {plot_path}")
        plt.close()

def print_summary(self):
    """打印本次训练的总结信息"""
    print(f"\n{'='*60}")
    print("训练总结")
    print(f"{'='*60}")
    print(f"运行ID: {self.run_name}")
    print(f"输出目录: {self.run_dir}")
    print(f"\n保存的文件:")
    print(f"  ├─ config.json              (训练配置)")
    print(f"  ├─ training_history.json    (训练历史)")
    print(f"  ├─ test_results.json        (测试结果)")
    print(f"  ├─ training_curves.png      (训练曲线)")
    print(f"  └─ checkpoints/")
    print(f"      └─ best_model.pth       (最佳模型)")
    print(f"\n最佳结果:")
    print(f"  验证准确率: {self.best_val_acc*100:.2f}%")
    print(f"  训练轮数: {len(self.train_losses)}")
    print(f"{'='*60}\n")

def main():
    Config.print_config()
    create_directories()

    config_dict = Config.get_config_dict()

    # 调整配置
    # config_dict['num_layers'] = 2
    # config_dict['hidden_channels'] = 128
    # config_dict['out_channels'] = 64
    # config_dict['num_heads'] = 4
    # config_dict['dropout'] = 0.2
    # config_dict['batch_size'] = 16
    # config_dict['learning_rate'] = 5e-4
    # config_dict['num_epochs'] = 100
    config_dict['sbert_model_path'] = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/sentence-bert'

    print(f"\n混合分类器配置:")
    print(f"  Batch size: {config_dict['batch_size']}")
    print(f"  学习率: {config_dict['learning_rate']}")
    print(f"  RGAT层数: {config_dict['num_layers']}")

    trainer = HybridClassifierTrainer(config_dict)
    trainer.train()


if __name__ == '__main__':
    main()