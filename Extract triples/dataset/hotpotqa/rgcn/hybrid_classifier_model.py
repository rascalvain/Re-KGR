"""
混合模型：Siamese RGCN + FFN分类器
利用预训练的Siamese RGCN编码器，添加FFN分类头

优势：
1. 利用Siamese RGCN的对比学习表示
2. FFN学习非线性分类边界
3. 支持编码器冻结/微调
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 导入原有的Siamese RGCN
from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding, SiameseRGCNWithEmbedding


class HybridRGCNClassifier(nn.Module):
    """
    混合模型：Siamese RGCN Encoder + FFN Classifier
    
    架构：
    1. 使用Siamese RGCN的编码器提取图特征
    2. 拼接两个图的特征
    3. FFN分类头输出二分类概率
    """
    
    def __init__(self, 
                 entity_embedding_path,
                 relation_embedding_path,
                 hidden_channels=128,
                 out_channels=64,
                 num_layers=3,
                 num_bases=-1,
                 dropout=0.3,
                 ffn_hidden_dim=128,
                 freeze_encoder=False):
        """
        Args:
            entity_embedding_path: 实体嵌入路径
            relation_embedding_path: 关系映射路径
            hidden_channels: RGCN隐藏层维度
            out_channels: RGCN输出维度
            num_layers: RGCN层数
            num_bases: RGCN基数（-1表示不使用）
            dropout: Dropout率
            ffn_hidden_dim: FFN隐藏层维度
            freeze_encoder: 是否冻结编码器（只训练分类头）
        """
        super(HybridRGCNClassifier, self).__init__()
        
        self.freeze_encoder = freeze_encoder
        self.out_channels = out_channels
        
        # 使用原有的RGCN编码器
        print(f"\n初始化Siamese RGCN编码器...")
        self.encoder = ImprovedRGCNEncoderWithEmbedding(
            entity_embedding_path=entity_embedding_path,
            relation_embedding_path=relation_embedding_path,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_bases=num_bases,
            dropout=dropout,
            use_attention=True  # 使用注意力池化
        )
        
        # 冻结编码器（如果需要）
        if freeze_encoder:
            print(f"  ⚠️ 编码器已冻结，只训练FFN分类头")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print(f"  ✓ 编码器可训练（端到端微调）")
        
        # FFN分类头
        # 输入：concat[h_response, h_reference] = 2 * out_channels
        # 输出：2分类logits
        self.classifier = nn.Sequential(
            nn.Linear(2 * out_channels, ffn_hidden_dim),
            nn.BatchNorm1d(ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(ffn_hidden_dim, ffn_hidden_dim // 2),
            nn.BatchNorm1d(ffn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(ffn_hidden_dim // 2, 2)
        )
        
        print(f"\nFFN分类头:")
        print(f"  输入维度: {2 * out_channels}")
        print(f"  隐藏层: {ffn_hidden_dim} → {ffn_hidden_dim // 2}")
        print(f"  输出: 2 (二分类)")
        print(f"  Dropout: {dropout}")
    
    def load_pretrained_encoder(self, siamese_model_path):
        """
        加载预训练的Siamese RGCN编码器权重
        
        Args:
            siamese_model_path: Siamese RGCN模型检查点路径
        """
        print(f"\n加载预训练的Siamese RGCN编码器: {siamese_model_path}")
        
        if not os.path.exists(siamese_model_path):
            print(f"  ⚠️ 模型文件不存在，跳过加载")
            return False
        
        try:
            checkpoint = torch.load(siamese_model_path, map_location='cpu')
            
            # 提取编码器的权重
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 过滤出编码器的权重
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # 从 "encoder.xxx" 或 "context_encoder.xxx" 中提取
                if key.startswith('encoder.') or key.startswith('context_encoder.'):
                    new_key = key.replace('context_encoder.', '').replace('encoder.', '')
                    encoder_state_dict[new_key] = value
            
            if encoder_state_dict:
                # 加载权重（允许部分匹配）
                self.encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"  ✓ 成功加载 {len(encoder_state_dict)} 个编码器参数")
                
                if 'epoch' in checkpoint:
                    print(f"  预训练轮数: {checkpoint['epoch']}")
                if 'val_loss' in checkpoint:
                    print(f"  验证损失: {checkpoint['val_loss']:.4f}")
                
                return True
            else:
                print(f"  ⚠️ 未找到编码器权重")
                return False
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            return False
    
    def forward(self, response_graph, reference_graph):
        """
        前向传播
        
        Args:
            response_graph: 响应图（GPT）
            reference_graph: 参考图（Context）
            
        Returns:
            logits: [batch_size, 2]
        """
        # 1. RGCN编码（使用注意力池化）
        h_response = self.encoder(response_graph)    # [batch_size, out_channels]
        h_reference = self.encoder(reference_graph)  # [batch_size, out_channels]
        
        # 2. 拼接特征
        h_concat = torch.cat([h_response, h_reference], dim=1)  # [batch_size, 2*out_channels]
        
        # 3. FFN分类
        logits = self.classifier(h_concat)  # [batch_size, 2]
        
        return logits
    
    def predict(self, response_graph, reference_graph):
        """
        预测（推理模式）
        
        Returns:
            predictions: [batch_size] (0=幻觉, 1=非幻觉)
            probabilities: [batch_size, 2]
        """
        logits = self.forward(response_graph, reference_graph)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    def get_graph_embeddings(self, response_graph, reference_graph):
        """
        获取图嵌入（用于分析）
        
        Returns:
            h_response: [batch_size, out_channels]
            h_reference: [batch_size, out_channels]
        """
        with torch.no_grad():
            h_response = self.encoder(response_graph)
            h_reference = self.encoder(reference_graph)
        
        return h_response, h_reference
    
    def unfreeze_encoder(self):
        """解冻编码器，允许微调"""
        print("解冻编码器，允许端到端微调...")
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def freeze_encoder_fn(self):
        """冻结编码器，只训练分类头"""
        print("冻结编码器，只训练FFN分类头...")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True


def load_hybrid_model(model_path, config_dict, device='cuda'):
    """
    便捷函数：加载混合模型
    
    Args:
        model_path: 模型检查点路径
        config_dict: 配置字典
        device: 设备
        
    Returns:
        model: 加载好的模型
        checkpoint: 检查点信息
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = HybridRGCNClassifier(
        entity_embedding_path=config_dict['entity_embedding_path'],
        relation_embedding_path=config_dict['relation_embedding_path'],
        hidden_channels=config_dict['hidden_channels'],
        out_channels=config_dict['out_channels'],
        num_layers=config_dict['num_layers'],
        num_bases=config_dict.get('num_bases', -1),
        dropout=config_dict.get('dropout', 0.3),
        ffn_hidden_dim=config_dict.get('ffn_hidden_dim', 128),
        freeze_encoder=False
    ).to(device)
    
    # 加载权重
    print(f"\n加载混合模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"  验证准确率: {checkpoint.get('val_acc', 0):.4f}")
    
    return model, checkpoint


if __name__ == "__main__":
    print("混合模型：Siamese RGCN + FFN分类器")
    print("="*60)
    
    print("\n使用方式:")
    print("\n1. 从零开始训练:")
    print("   python train_hybrid_classifier.py")
    
    print("\n2. 加载预训练的Siamese RGCN:")
    print("   python train_hybrid_classifier.py --pretrained_encoder best_model.pth")
    
    print("\n3. 冻结编码器，只训练分类头:")
    print("   python train_hybrid_classifier.py --pretrained_encoder best_model.pth --freeze_encoder")
    
    print("\n优势:")
    print("  ✓ 利用Siamese RGCN的对比学习表示")
    print("  ✓ FFN学习非线性分类边界")
    print("  ✓ 支持迁移学习（冻结/微调编码器）")
    print("  ✓ 灵活的训练策略")

