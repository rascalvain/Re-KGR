"""
改进的孪生R-GCN模型 - 使用预构建嵌入
核心优势：
1. 从pkl文件加载预构建的实体和关系嵌入
2. 使用 nn.Embedding 层，通过ID查询嵌入（零拷贝，极快）
3. 大幅降低GPU内存使用和计算开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pickle
import numpy as np


class ImprovedRGCNEncoderWithEmbedding(nn.Module):
    """
    改进的R-GCN编码器 - 使用预构建嵌入层
    """
    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels, out_channels, num_layers=3,
                 freeze_embeddings=True, dropout=0.3):
        super(ImprovedRGCNEncoderWithEmbedding, self).__init__()

        self.num_layers = num_layers

        # 🔥 加载实体嵌入矩阵
        print(f"加载实体嵌入: {entity_embedding_path}")
        with open(entity_embedding_path, 'rb') as f:
            entity_data = pickle.load(f)
            entity_embeddings = torch.FloatTensor(entity_data['embeddings'])
            self.num_entities = entity_data['num_entities']
            self.embedding_dim = entity_embeddings.shape[1]
            print(f"  实体数: {self.num_entities}, 嵌入维度: {self.embedding_dim}")

        # 🔥 加载关系映射（获取关系数量）
        print(f"加载关系映射: {relation_embedding_path}")
        with open(relation_embedding_path, 'rb') as f:
            relation_data = pickle.load(f)
            self.num_relations = relation_data['num_relations']
            print(f"  关系数: {self.num_relations}")

        # 🔥 创建实体嵌入层（从预训练权重初始化）
        self.entity_embedding = nn.Embedding.from_pretrained(
            entity_embeddings,
            freeze=freeze_embeddings  # 是否冻结（不训练嵌入）
        )
        print(f"  嵌入层冻结: {freeze_embeddings}")

        # R-GCN卷积层
        self.convs = nn.ModuleList()

        # 第一层
        self.convs.append(RGCNConv(self.embedding_dim, hidden_channels, self.num_relations))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, self.num_relations))

        # 最后一层
        if num_layers > 1:
            self.convs.append(RGCNConv(hidden_channels, out_channels, self.num_relations))
        else:
            self.convs[0] = RGCNConv(self.embedding_dim, out_channels, self.num_relations)

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
            data: PyTorch Geometric Data对象
                - node_ids: 节点的全局ID [num_nodes]  🔥 关键：输入是ID而非特征
                - edge_index: 边索引 [2, num_edges]
                - edge_type: 边类型ID [num_edges]
                - batch: 批次索引 [num_nodes]
        Returns:
            如果return_node_embeddings=True: (graph_embedding, node_embeddings)
            否则: graph_embedding
        """
        # 🔥 步骤1: 通过node_ids从嵌入层获取节点特征（零拷贝，极快）
        x = self.entity_embedding(data.node_ids)  # [num_nodes, embedding_dim]

        edge_index = data.edge_index
        edge_type = data.edge_type
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 步骤2: 图卷积层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.batch_norms[i](x)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # 保存节点嵌入（用于三元组验证）
        node_embeddings = x

        # 步骤3: 图级别池化（使用注意力机制）
        attention_weights = F.softmax(self.attention(x), dim=0)
        graph_embedding = global_mean_pool(x * attention_weights, batch)

        if return_node_embeddings:
            return graph_embedding, node_embeddings
        return graph_embedding


class SiameseRGCNWithEmbedding(nn.Module):
    """
    孪生R-GCN模型 - 使用预构建嵌入
    """
    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels, out_channels, num_layers=3,
                 freeze_embeddings=True, dropout=0.3):
        super(SiameseRGCNWithEmbedding, self).__init__()

        # 🔥 共享的R-GCN编码器（带预构建嵌入层）
        self.encoder = ImprovedRGCNEncoderWithEmbedding(
            entity_embedding_path,
            relation_embedding_path,
            hidden_channels,
            out_channels,
            num_layers,
            freeze_embeddings,
            dropout
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
            wiki_data: 参考图数据（包含node_ids）
            return_node_embeddings: 是否返回节点嵌入
        Returns:
            如果return_node_embeddings=True:
                (gen_graph_emb, wiki_graph_emb, gen_node_emb, wiki_node_emb)
            否则:
                (gen_graph_emb, wiki_graph_emb)
        """
        if return_node_embeddings:
            gen_graph_emb, gen_node_emb = self.encoder(gen_data, return_node_embeddings=True)
            wiki_graph_emb, wiki_node_emb = self.encoder(wiki_data, return_node_embeddings=True)

            # 应用投影头
            gen_graph_emb_proj = self.projection_head(gen_graph_emb)
            wiki_graph_emb_proj = self.projection_head(wiki_graph_emb)

            return gen_graph_emb_proj, wiki_graph_emb_proj, gen_node_emb, wiki_node_emb
        else:
            gen_graph_emb = self.encoder(gen_data, return_node_embeddings=False)
            wiki_graph_emb = self.encoder(wiki_data, return_node_embeddings=False)

            # 应用投影头
            gen_graph_emb_proj = self.projection_head(gen_graph_emb)
            wiki_graph_emb_proj = self.projection_head(wiki_graph_emb)

            return gen_graph_emb_proj, wiki_graph_emb_proj


class ImprovedContrastiveLoss(nn.Module):
    """
    改进的对比损失函数
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
    print("测试孪生R-GCN模型（使用预构建嵌入）...")

    # 注意：需要先运行构建嵌入的脚本
    # 这里仅作为示例
    print("\n使用前请确保已经生成嵌入文件：")
    print("  1. 运行 extract_entities_relations.py 提取实体和关系")
    print("  2. 运行 build_embedding_weights.py 构建嵌入矩阵")
    print("  3. 使用 data_loader_with_ids.py 加载数据")