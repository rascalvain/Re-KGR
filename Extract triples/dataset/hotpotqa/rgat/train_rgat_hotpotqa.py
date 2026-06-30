"""
HotpotQA 数据集的 RGAT 训练脚本
使用孪生网络架构训练图注意力嵌入模型
核心改进：使用 R-GAT 替代 R-GCN，引入多头注意力机制
"""

import torch
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
from config_hotpotqa_rgat import Config, create_directories

# 添加rgcn目录到路径（复用数据加载器）
rgcn_dir = os.path.join(os.path.dirname(__file__), '..', 'rgcn')
sys.path.insert(0, os.path.abspath(rgcn_dir))

from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn

# 导入RGAT模型
from siamese_rgat_improved import SiameseRGATWithEmbedding, ImprovedContrastiveLoss


class RGATTrainer:
    """RGAT训练器"""
    
    def __init__(self, config_dict):
        """
        初始化训练器
        Args:
            config_dict: 配置字典
        """
        self.config = config_dict
        
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
        self.dataset = HotpotQAGraphDataset(
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
        self.criterion = ImprovedContrastiveLoss(
            margin=config_dict['margin'],
            temperature=config_dict['temperature'],
            alpha=config_dict['alpha']
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )
        
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
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for context_batch, gpt_batch, labels, metadata_list in pbar:
            if context_batch is None:
                continue
            
            # 移到设备
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            context_emb, gpt_emb = self.model(context_batch, gpt_batch)
            
            # 计算损失（返回元组：loss, loss_dict）
            loss, loss_dict = self.criterion(context_emb, gpt_emb, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip_norm', 1.0)
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 更新进度条（显示更多信息）
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sim': f'{loss_dict["avg_similarity"]:.3f}'
            })
        
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
            
            context_emb, gpt_emb = self.model(context_batch, gpt_batch)
            
            # 计算损失（返回元组：loss, loss_dict）
            loss, loss_dict = self.criterion(context_emb, gpt_emb, labels)
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始训练 RGAT")
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


def main():
    """主函数"""
    # 打印配置
    print("\n" + "="*60)
    print("RGAT训练配置")
    print("="*60)
    Config.print_config()
    
    # 获取配置字典
    config_dict = Config.get_config_dict()
    
    # 确认使用RGAT
    print(f"\n🔹 使用模型: R-GAT (关系图注意力网络)")
    print(f"🔹 注意力头数: {config_dict.get('num_heads', 4)}")
    
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
