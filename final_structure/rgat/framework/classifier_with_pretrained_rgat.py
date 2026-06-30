"""
使用预训练Siamese RGAT + FFN分类器
迁移学习方案：
1. 加载预训练的Siamese RGAT编码器（对比学习训练）
2. 冻结或微调编码器
3. 训练FFN分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import pickle
import os


class HallucinationClassifierWithPretrainedRGAT(nn.Module):
    """
    使用预训练Siamese RGAT编码器的幻觉检测分类器
    
    架构:
    1. 预训练的RGAT编码器 (from Siamese RGAT)
    2. 全局平均池化
    3. FFN分类头
    """
    
    def __init__(self, pretrained_model_path, 
                 freeze_encoder=False,
                 ffn_hidden_dim=128, 
                 dropout=0.3):
        """
        Args:
            pretrained_model_path: 预训练Siamese RGAT模型路径
            freeze_encoder: 是否冻结编码器权重
            ffn_hidden_dim: FFN隐藏层维度
            dropout: Dropout率
        """
        super(HallucinationClassifierWithPretrainedRGAT, self).__init__()
        
        self.freeze_encoder = freeze_encoder
        
        # 1. 加载预训练的Siamese RGAT编码器
        print(f"\n加载预训练RGAT模型: {pretrained_model_path}")
        self.encoder, self.out_channels = self._load_pretrained_encoder(pretrained_model_path)
        
        print(f"  编码器输出维度: {self.out_channels}")
        
        # 2. 冻结编码器（如果需要）
        if freeze_encoder:
            print(f"  冻结编码器权重")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print(f"  允许微调编码器")
        
        # 3. FFN分类头
        # 输入: concat[h_response, h_reference] → 2 * out_channels
        # 输出: 2 (幻觉 vs 非幻觉)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.out_channels, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, ffn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim // 2, 2)
        )
        
        print(f"\nFFN分类头:")
        print(f"  输入维度: {2 * self.out_channels}")
        print(f"  隐藏层: {ffn_hidden_dim} → {ffn_hidden_dim // 2}")
        print(f"  输出: 2 (二分类)")
    
    def _load_pretrained_encoder(self, model_path):
        """加载预训练的Siamese RGAT编码器"""
        
        # 导入Siamese RGAT模型
        import sys
        
        # 添加rgat路径
        rgat_path = os.path.dirname(__file__)
        sys.path.insert(0, os.path.abspath(rgat_path))
        
        from siamese_rgat_improved import ImprovedRGATEncoderWithEmbedding
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"  预训练模型信息:")
        print(f"    训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"    验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # 获取配置
        config = checkpoint.get('config', {})
        
        # 重建编码器
        encoder = ImprovedRGATEncoderWithEmbedding(
            entity_embedding_path=config['entity_embedding_path'],
            relation_embedding_path=config['relation_embedding_path'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            freeze_embeddings=True,  # 始终冻结嵌入
            dropout=config.get('dropout', 0.3),
            num_heads=config.get('num_heads', 4)  # 🔹 RGAT特有参数
        )
        
        # 加载编码器权重（从Siamese模型中提取）
        model_state = checkpoint['model_state_dict']
        encoder_state = {k.replace('encoder.', ''): v 
                        for k, v in model_state.items() 
                        if k.startswith('encoder.')}
        
        encoder.load_state_dict(encoder_state)
        print(f"  ✓ 编码器权重加载成功")
        
        return encoder, config['out_channels']
    
    def forward(self, response_graph, reference_graph):
        """
        前向传播
        Args:
            response_graph: LLM回复的图（PyG Data对象）
            reference_graph: 参考知识图谱（PyG Data对象）
        Returns:
            logits: [batch_size, 2]
        """
        # 编码两个图
        h_response = self.encoder(response_graph)  # [batch_size, out_channels]
        h_reference = self.encoder(reference_graph)  # [batch_size, out_channels]
        
        # 拼接
        h_concat = torch.cat([h_response, h_reference], dim=-1)  # [batch_size, 2*out_channels]
        
        # 分类
        logits = self.classifier(h_concat)  # [batch_size, 2]
        
        return logits
    
    def predict(self, response_graph, reference_graph):
        """
        预测幻觉标签
        Args:
            response_graph: LLM回复的图
            reference_graph: 参考知识图谱
        Returns:
            predictions: [batch_size] (0: 幻觉, 1: 事实)
            probabilities: [batch_size, 2]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(response_graph, reference_graph)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        
        return predictions, probabilities


def load_classifier(checkpoint_path, device='cpu'):
    """
    加载训练好的分类器
    Args:
        checkpoint_path: 分类器检查点路径
        device: 设备
    Returns:
        model: 分类器模型
        config: 训练配置
    """
    print(f"加载分类器: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    config = checkpoint['config']
    
    # 重建模型
    model = HallucinationClassifierWithPretrainedRGAT(
        pretrained_model_path=config['pretrained_model_path'],
        freeze_encoder=config.get('freeze_encoder', False),
        ffn_hidden_dim=config.get('ffn_hidden_dim', 128),
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 分类器加载成功")
    
    return model, config


if __name__ == "__main__":
    print("RGAT分类器模块")
    print("\n使用示例:")
    print("""
    # 1. 初始化分类器（使用预训练RGAT）
    classifier = HallucinationClassifierWithPretrainedRGAT(
        pretrained_model_path='path/to/best_rgat_model.pth',
        freeze_encoder=False,  # 允许微调
        ffn_hidden_dim=128
    )
    
    # 2. 训练分类器
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for response_graph, reference_graph, labels in train_loader:
        logits = classifier(response_graph, reference_graph)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    
    # 3. 推理
    predictions, probs = classifier.predict(response_graph, reference_graph)
    """)

