"""
图+文本混合分类器
结合RGAT图编码器和SentenceBERT文本编码器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import os
import sys

# 导入RGAT编码器
from framework.siamese_rgat_improved import ImprovedRGATEncoderWithEmbedding


class HybridGraphTextClassifier(nn.Module):
    """
    图+文本混合分类器

    架构:
    1. 图编码器 (RGAT): 编码context图和gpt图
    2. 文本编码器 (SentenceBERT): 编码gpt文本
    3. 融合层: 将图特征和文本特征融合后分类
    """

    def __init__(self,
                 entity_embedding_path,
                 relation_embedding_path,
                 sbert_model_path,
                 hidden_channels=128,
                 out_channels=64,
                 num_layers=2,
                 num_heads=4,
                 dropout=0.2,
                 fusion_hidden_dim=128,  # 🔥 新增：融合层隐藏维度
                 fusion_dropout=0.65,    # 🔥 新增：融合层dropout
                 freeze_text_encoder=True):
        """
        Args:
            entity_embedding_path: 实体嵌入路径
            relation_embedding_path: 关系映射路径
            sbert_model_path: SentenceBERT模型路径
            hidden_channels: RGAT隐藏层维度
            out_channels: RGAT输出维度
            num_layers: RGAT层数
            num_heads: RGAT注意力头数
            dropout: RGAT的Dropout率
            fusion_hidden_dim: 融合分类器的隐藏层维度
            fusion_dropout: 融合分类器的Dropout率
            freeze_text_encoder: 是否冻结文本编码器
        """
        super().__init__()

        print(f"\n{'=' * 60}")
        print("初始化图+文本混合分类器")
        print(f"{'=' * 60}")

        # 1. 图编码器（RGAT）
        print(f"\n【图编码器】RGAT")
        print(f"  - 隐藏层: {hidden_channels}")
        print(f"  - 输出: {out_channels}")
        print(f"  - 层数: {num_layers}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - Dropout: {dropout}")

        self.graph_encoder = ImprovedRGATEncoderWithEmbedding(
            entity_embedding_path=entity_embedding_path,
            relation_embedding_path=relation_embedding_path,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            freeze_embeddings=True,
            dropout=dropout
        )

        # 2. 文本编码器（SentenceBERT）
        print(f"\n【文本编码器】SentenceBERT")
        print(f"  - 模型路径: {sbert_model_path}")
        print(f"  - 冻结: {freeze_text_encoder}")

        from sentence_transformers import SentenceTransformer
        self.text_encoder = SentenceTransformer(sbert_model_path)

        # 获取文本编码器的输出维度
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension()
        print(f"  - 输出维度: {self.text_dim}")

        # 冻结文本编码器（可选）
        if freeze_text_encoder:
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # 3. 融合分类器
        graph_feature_dim = 2 * out_channels  # context图 + gpt图
        total_feature_dim = graph_feature_dim + self.text_dim

        print(f"\n【融合分类器】")
        print(f"  - 图特征: {graph_feature_dim} (context {out_channels} + gpt {out_channels})")
        print(f"  - 文本特征: {self.text_dim}")
        print(f"  - 总特征: {total_feature_dim}")
        print(f"  - 架构: {total_feature_dim} → {fusion_hidden_dim} → 2")
        print(f"  - Dropout: {fusion_dropout}")

        # 🔥 使用可配置参数构建融合分类器
        self.fusion_classifier = nn.Sequential(
            # 第一层：降维到fusion_hidden_dim
            nn.Linear(total_feature_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),  # 🔥 使用参数

            # 输出层
            nn.Linear(fusion_hidden_dim, 2)
        )

        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n【参数统计】")
        print(f"  - 总参数: {total_params:,}")
        print(f"  - 可训练: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")
        print(f"{'=' * 60}\n")

    def forward(self, context_graphs, gpt_graphs, gpt_texts):
        """
        前向传播
        Args:
            context_graphs: Context图（PyG Batch）
            gpt_graphs: GPT生成的图（PyG Batch）
            gpt_texts: GPT生成的文本（字符串列表）
        Returns:
            logits: [batch_size, 2]
        """
        # 1. 编码图特征
        h_context = self.graph_encoder(context_graphs)  # [B, out_channels]
        h_gpt_graph = self.graph_encoder(gpt_graphs)  # [B, out_channels]
        graph_features = torch.cat([h_context, h_gpt_graph], dim=-1)  # [B, 2*out_channels]

        # 2. 编码文本特征
        with torch.no_grad():  # 文本编码器不需要梯度
            text_embeddings = self.text_encoder.encode(
                gpt_texts,
                convert_to_tensor=True,
                device=graph_features.device,
                show_progress_bar=False
            )  # [B, text_dim]

        # 3. 融合特征
        combined_features = torch.cat([graph_features, text_embeddings], dim=-1)  # [B, total_dim]

        # 4. 分类
        logits = self.fusion_classifier(combined_features)  # [B, 2]

        return logits

    def predict(self, context_graphs, gpt_graphs, gpt_texts):
        """
        预测
        Returns:
            predictions: [batch_size] (0: 幻觉, 1: 事实)
            probabilities: [batch_size, 2]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(context_graphs, gpt_graphs, gpt_texts)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities


def load_hybrid_classifier(checkpoint_path, device='cpu'):
    """
    加载训练好的混合分类器
    """
    print(f"加载混合分类器: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    model = HybridGraphTextClassifier(
        entity_embedding_path=config['entity_embedding_path'],
        relation_embedding_path=config['relation_embedding_path'],
        sbert_model_path=config['sbert_model_path'],
        hidden_channels=config['hidden_channels'],
        out_channels=config['out_channels'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        fusion_hidden_dim=config.get('fusion_hidden_dim', 128),  # 🔥 新增
        fusion_dropout=config.get('fusion_dropout', 0.65),       # 🔥 新增
        freeze_text_encoder=config.get('freeze_text_encoder', True)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 混合分类器加载成功")

    return model, config