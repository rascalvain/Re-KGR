"""
改进的孪生R-GAT模型 - 使用预构建嵌入
核心改进：将RGCNConv替换为RGATConv，引入注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pickle
import numpy as np


class ImprovedRGATEncoderWithEmbedding(nn.Module):
    """
    改进的R-GAT编码器 - 使用预构建嵌入层和关系注意力机制
    """

    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels, out_channels, num_layers=3,
                 freeze_embeddings=True, dropout=0.3, num_heads=4):
        super(ImprovedRGATEncoderWithEmbedding, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads

        # 🔹 加载实体嵌入矩阵
        print(f"  加载实体嵌入: {entity_embedding_path}")
        with open(entity_embedding_path, 'rb') as f:
            entity_data = pickle.load(f)

        entity_embeddings = entity_data['embeddings']  # numpy array
        num_entities = entity_data['num_entities']
        self.embedding_dim = entity_data['embedding_dim']

        print(f"    实体数: {num_entities}, 嵌入维度: {self.embedding_dim}")

        # 创建实体嵌入层
        self.entity_embedding = nn.Embedding(num_entities, self.embedding_dim)
        self.entity_embedding.weight.data.copy_(torch.from_numpy(entity_embeddings))

        if freeze_embeddings:
            self.entity_embedding.weight.requires_grad = False
            print(f"    实体嵌入已冻结")

        # 🔹 加载关系嵌入（用于获取关系数量）
        print(f"  加载关系映射: {relation_embedding_path}")
        with open(relation_embedding_path, 'rb') as f:
            relation_data = pickle.load(f)

        self.num_relations = len(relation_data['relation2id'])
        print(f"    关系数: {self.num_relations}")

        # 🔹 构建R-GAT层（关键改动：使用RGATConv）
        self.convs = nn.ModuleList()

        # 第一层
        self.convs.append(
            RGATConv(
                self.embedding_dim,
                hidden_channels // num_heads,  # 每个head的输出维度
                num_relations=self.num_relations,
                heads=num_heads,
                concat=True,  # 拼接多头输出
                dropout=dropout
            )
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    num_relations=self.num_relations,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout
                )
            )

        # 最后一层（不拼接，取平均）
        if num_layers > 1:
            self.convs.append(
                RGATConv(
                    hidden_channels,
                    out_channels,
                    num_relations=self.num_relations,
                    heads=num_heads,
                    concat=False,  # 最后一层取平均
                    dropout=dropout
                )
            )
        else:
            self.convs[0] = RGATConv(
                self.embedding_dim,
                out_channels,
                num_relations=self.num_relations,
                heads=num_heads,
                concat=False,
                dropout=dropout
            )

        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels if i < num_layers - 1 else out_channels)
            for i in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 注意力池化
        self.attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.Tanh(),
            nn.Linear(out_channels // 2, 1)
        )

    def forward(self, data, return_node_embeddings=False):
        """
        前向传播
        Args:
            data: PyG Data对象，包含:
                - node_ids: 节点ID (LongTensor)
                - edge_index: 边索引
                - edge_type: 边类型（关系类型）
                - batch: 批次索引
        """
        # 🔹 通过ID查询嵌入（零拷贝，极快）
        x = self.entity_embedding(data.node_ids)
        edge_index = data.edge_index
        edge_type = data.edge_type

        # 🔹 R-GAT层传播（关键：使用注意力机制）
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.batch_norms[i](x)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        if return_node_embeddings:
            return x

        # 🔹 注意力池化（图级别表示）
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=0)

        # 加权求和
        graph_embedding = global_mean_pool(x * attn_weights, data.batch)

        return graph_embedding


class SiameseRGATWithEmbedding(nn.Module):
    """
    孪生R-GAT网络 - 用于图对比学习
    """

    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels, out_channels, num_layers=3,
                 freeze_embeddings=True, dropout=0.3, num_heads=4):
        super(SiameseRGATWithEmbedding, self).__init__()

        # 🔹 共享的R-GAT编码器（带预构建嵌入层）
        self.encoder = ImprovedRGATEncoderWithEmbedding(
            entity_embedding_path,
            relation_embedding_path,
            hidden_channels,
            out_channels,
            num_layers,
            freeze_embeddings,
            dropout,
            num_heads
        )

        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels // 2)
        )

    def forward(self, gen_data, wiki_data, return_node_embeddings=False):
        """
        前向传播
        Args:
            gen_data: 生成图数据（包含node_ids）
            wiki_data: 维基图数据（包含node_ids）
        """
        if return_node_embeddings:
            gen_node_emb = self.encoder(gen_data, return_node_embeddings=True)
            wiki_node_emb = self.encoder(wiki_data, return_node_embeddings=True)
            return gen_node_emb, wiki_node_emb

        # 编码
        gen_embedding = self.encoder(gen_data)
        wiki_embedding = self.encoder(wiki_data)

        # 投影
        gen_proj = self.projection_head(gen_embedding)
        wiki_proj = self.projection_head(wiki_embedding)

        return gen_proj, wiki_proj

    def encode(self, data):
        """单个图编码（用于推理）"""
        return self.encoder(data)


class ImprovedContrastiveLoss(nn.Module):
    """
    改进的对比损失函数（与RGCN版本保持一致）
    """
    def __init__(self, margin=0.5, temperature=0.1, alpha=0.7):
        super(ImprovedContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, gen_emb, wiki_emb, labels):
        """
        计算对比损失
        Args:
            gen_emb: 生成图嵌入 [batch_size, embed_dim]
            wiki_emb: wiki图嵌入 [batch_size, embed_dim]
            labels: 标签 [batch_size] (1 for factual, 0 for hallucination)
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情字典
        """
        # 归一化嵌入
        gen_emb = F.normalize(gen_emb, p=2, dim=1)
        wiki_emb = F.normalize(wiki_emb, p=2, dim=1)

        # 计算余弦相似度
        similarity = F.cosine_similarity(gen_emb, wiki_emb, dim=1)

        # 掩码
        factual_mask = labels.bool()
        hallucination_mask = ~factual_mask

        # 1. 对比损失
        if factual_mask.any():
            factual_loss = (1 - similarity[factual_mask]).mean()
        else:
            factual_loss = torch.tensor(0.0, device=gen_emb.device)

        if hallucination_mask.any():
            hallucination_similarity = similarity[hallucination_mask]
            hallucination_loss = F.relu(hallucination_similarity - (-self.margin)).mean()
        else:
            hallucination_loss = torch.tensor(0.0, device=gen_emb.device)

        contrastive_loss = factual_loss + hallucination_loss

        # 2. InfoNCE损失
        batch_size = gen_emb.size(0)
        if batch_size > 1:
            sim_matrix = torch.mm(gen_emb, wiki_emb.t()) / self.temperature
            targets = torch.arange(batch_size, device=gen_emb.device)

            if factual_mask.any():
                infonce_loss = F.cross_entropy(
                    sim_matrix[factual_mask],
                    targets[factual_mask]
                )
            else:
                infonce_loss = torch.tensor(0.0, device=gen_emb.device)
        else:
            infonce_loss = torch.tensor(0.0, device=gen_emb.device)

        # 总损失
        total_loss = self.alpha * contrastive_loss + (1 - self.alpha) * infonce_loss

        # 统计信息
        loss_dict = {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'factual_loss': factual_loss.item(),
            'hallucination_loss': hallucination_loss.item(),
            'infonce_loss': infonce_loss.item(),
            'avg_similarity': similarity.mean().item(),
            'factual_similarity': similarity[factual_mask].mean().item() if factual_mask.any() else 0.0,
            'hallucination_similarity': similarity[hallucination_mask].mean().item() if hallucination_mask.any() else 0.0
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    print("测试孪生R-GAT模型（使用预构建嵌入）...")
    print("\n使用前请确保已经生成嵌入文件：")
    print("  1. 运行 extract_entities_relations.py 提取实体和关系")
    print("  2. 运行 build_embedding_weights.py 构建嵌入矩阵")
    print("  3. 使用 data_loader_with_ids.py 加载数据")