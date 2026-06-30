"""
使用预训练Siamese RGCN训练FFN分类器
迁移学习方案
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from config_hotpotqa import Config, create_directories
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn
from classifier_with_pretrained import HallucinationClassifierWithPretrainedEncoder


class PretrainedClassifierTrainer:
    """使用预训练编码器的分类器训练器"""
    
    def __init__(self, config_dict, pretrained_model_path, freeze_encoder=False):
        """
        Args:
            config_dict: 配置字典
            pretrained_model_path: 预训练Siamese RGCN模型路径
            freeze_encoder: 是否冻结编码器
        """
        self.config = config_dict
        self.freeze_encoder = freeze_encoder
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        create_directories()
        
        # 加载数据集
        print("\n" + "="*60)
        print("加载数据集")
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
        
        # 数据加载器
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
        
        # 初始化模型（使用预训练编码器）
        print("\n" + "="*60)
        print("初始化模型（预训练编码器 + FFN）")
        print("="*60)
        print(f"策略: {'冻结编码器' if freeze_encoder else '微调编码器'}")
        
        self.model = HallucinationClassifierWithPretrainedEncoder(
            pretrained_model_path=pretrained_model_path,
            freeze_encoder=freeze_encoder,
            ffn_hidden_dim=config_dict.get('ffn_hidden_dim', 128),
            dropout=config_dict.get('dropout', 0.3)
        ).to(self.device)
        
        # 打印参数统计
        params = self.model.get_trainable_params()
        print(f"\n参数统计:")
        print(f"  总参数: {params['total']:,}")
        print(f"  可训练参数: {params['trainable']:,} ({params['trainable']/params['total']*100:.1f}%)")
        print(f"  编码器: {params['encoder']:,} (可训练: {params['encoder_trainable']:,})")
        print(f"  分类器: {params['classifier']:,} (可训练: {params['classifier_trainable']:,})")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器（根据是否冻结编码器选择不同的学习率）
        if freeze_encoder:
            # 只优化分类器
            self.optimizer = optim.Adam(
                self.model.classifier.parameters(),
                lr=config_dict['learning_rate'],
                weight_decay=config_dict.get('weight_decay', 1e-5)
            )
            print(f"\n优化器: 仅优化FFN分类器")
        else:
            # 使用不同学习率（编码器较小，分类器较大）
            encoder_lr = config_dict['learning_rate'] * 0.1  # 编码器用1/10学习率
            classifier_lr = config_dict['learning_rate']
            
            self.optimizer = optim.Adam([
                {'params': self.model.encoder.parameters(), 'lr': encoder_lr},
                {'params': self.model.classifier.parameters(), 'lr': classifier_lr}
            ], weight_decay=config_dict.get('weight_decay', 1e-5))
            
            print(f"\n优化器: 分层学习率")
            print(f"  编码器学习率: {encoder_lr:.6f}")
            print(f"  分类器学习率: {classifier_lr:.6f}")
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config_dict.get('t_0', 10),
            T_mult=config_dict.get('t_mult', 2),
            eta_min=config_dict.get('eta_min', 1e-6)
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for context_batch, gpt_batch, labels, metadata_list in pbar:
            if context_batch is None:
                continue
            
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            logits = self.model(gpt_batch, context_batch)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip_norm', 1.0)
            )
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            current_acc = correct / total if total > 0 else 0
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for context_batch, gpt_batch, labels, metadata_list in self.val_loader:
            if context_batch is None:
                continue
            
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(gpt_batch, context_batch)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  训练: loss={train_loss:.4f}, acc={train_acc:.4f}")
            print(f"  验证: loss={val_loss:.4f}, acc={val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                model_name = 'best_pretrained_classifier_frozen.pth' if self.freeze_encoder else 'best_pretrained_classifier_finetuned.pth'
                self.save_checkpoint(model_name, epoch, val_loss, val_acc)
                print(f"  ✓ 保存最佳模型 (acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"\n早停触发")
                break
        
        print("\n训练完成！")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        
        self.plot_training_curves()
    
    def save_checkpoint(self, filename, epoch, val_loss, val_acc):
        """保存检查点"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'freeze_encoder': self.freeze_encoder,
            'config': self.config
        }, checkpoint_path)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.val_losses, label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.train_accs, label='Train')
        ax2.plot(self.val_accs, label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        suffix = 'frozen' if self.freeze_encoder else 'finetuned'
        plot_path = os.path.join(
            self.config['output_dir'], 
            f'pretrained_classifier_{suffix}_curves.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存: {plot_path}")
        plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用预训练Siamese RGCN训练FFN分类器')
    parser.add_argument('--pretrained_model', type=str,
                       default='../rgcn_output/checkpoints/best_model.pth',
                       help='预训练Siamese RGCN模型路径')
    parser.add_argument('--freeze_encoder', action='store_true',default=True,
                       help='是否冻结编码器（默认：微调编码器）')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数（默认30，因为有预训练）')
    
    args = parser.parse_args()
    
    # 检查预训练模型
    pretrained_path = os.path.join(os.path.dirname(__file__), args.pretrained_model)
    if not os.path.exists(pretrained_path):
        print(f"\n❌ 错误: 预训练模型不存在: {pretrained_path}")
        print(f"\n请先运行: python train_rgcn_hotpotqa.py")
        return
    
    print("="*60)
    print("使用预训练Siamese RGCN训练FFN分类器")
    print("="*60)
    print(f"预训练模型: {pretrained_path}")
    print(f"策略: {'冻结编码器' if args.freeze_encoder else '微调编码器'}")
    
    # 获取配置
    config_dict = Config.get_config_dict()
    config_dict['ffn_hidden_dim'] = 128
    config_dict['num_epochs'] = args.epochs  # 使用更少的epoch
    
    # 创建训练器
    trainer = PretrainedClassifierTrainer(
        config_dict=config_dict,
        pretrained_model_path=pretrained_path,
        freeze_encoder=args.freeze_encoder
    )
    
    # 训练
    trainer.train()
    
    print("\n✓ 完成！")
    
    model_name = 'frozen' if args.freeze_encoder else 'finetuned'
    print(f"\n生成的文件:")
    print(f"  模型: rgcn_output/checkpoints/best_pretrained_classifier_{model_name}.pth")
    print(f"  曲线: rgcn_output/pretrained_classifier_{model_name}_curves.png")


if __name__ == '__main__':
    main()

