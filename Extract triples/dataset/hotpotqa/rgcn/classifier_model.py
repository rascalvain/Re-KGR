"""
带FFN分类头的RGCN模型
实现论文中的二元分类架构：
1. RGCN编码 → 节点特征
2. 全局平均池化 → 图特征 h
3. 拼接两个图特征 → concat[h_response, h_reference]
4. FFN分类 → logit → [幻觉概率 | 非幻觉概率]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data, Batch
import pickle
import numpy as np


class RGCNEncoderWithPooling(nn.Module):
    """
    RGCN编码器 + 全局平均池化
    输出图级别的特征向量
    """
    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels, out_channels, num_layers=3,
                 freeze_embeddings=True, dropout=0.3):
        super(RGCNEncoderWithPooling, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 🔥 加载实体嵌入矩阵
        print(f"加载实体嵌入: {entity_embedding_path}")
        with open(entity_embedding_path, 'rb') as f:
            entity_data = pickle.load(f)
            entity_embeddings = torch.FloatTensor(entity_data['embeddings'])
            self.num_entities = entity_data['num_entities']
            self.embedding_dim = entity_embeddings.shape[1]
            print(f"  实体数: {self.num_entities}, 嵌入维度: {self.embedding_dim}")
        
        # 🔥 加载关系映射
        print(f"加载关系映射: {relation_embedding_path}")
        with open(relation_embedding_path, 'rb') as f:
            relation_data = pickle.load(f)
            self.num_relations = relation_data['num_relations']
            print(f"  关系数: {self.num_relations}")
        
        # 🔥 创建实体嵌入层
        self.entity_embedding = nn.Embedding.from_pretrained(
            entity_embeddings,
            freeze=freeze_embeddings
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
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        前向传播
        Args:
            data: PyG Data对象，包含 node_ids, edge_index, edge_type, batch
        Returns:
            graph_embedding: 图级别特征向量 [batch_size, out_channels]
        """
        node_ids = data.node_ids
        edge_index = data.edge_index
        edge_type = data.edge_type
        batch = data.batch if hasattr(data, 'batch') else None
        
        # 获取节点嵌入
        x = self.entity_embedding(node_ids)
        
        # R-GCN层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        # 🔥 全局平均池化：h = 1/|V| * Σ(v_n^L)
        if batch is not None:
            # 批处理模式
            graph_embedding = self._global_mean_pool(x, batch)
        else:
            # 单图模式
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        return graph_embedding
    
    def _global_mean_pool(self, x, batch):
        """
        全局平均池化
        Args:
            x: 节点特征 [num_nodes, out_channels]
            batch: 批索引 [num_nodes]
        Returns:
            pooled: 图特征 [batch_size, out_channels]
        """
        batch_size = batch.max().item() + 1
        out_channels = x.size(1)
        
        pooled = torch.zeros(batch_size, out_channels, device=x.device)
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.any():
                pooled[i] = x[mask].mean(dim=0)
        
        return pooled


class HallucinationClassifier(nn.Module):
    """
    幻觉检测分类器
    架构: RGCN Encoder → Global Pooling → FFN → Binary Classification
    
    ŷ = FFNs(concat[h_response, h_reference])
    """
    def __init__(self, entity_embedding_path, relation_embedding_path,
                 hidden_channels=128, out_channels=64, num_layers=3,
                 freeze_embeddings=True, dropout=0.3,
                 ffn_hidden_dim=128):
        """
        Args:
            entity_embedding_path: 实体嵌入文件路径
            relation_embedding_path: 关系映射文件路径
            hidden_channels: RGCN隐藏层维度
            out_channels: RGCN输出维度（图特征维度）
            num_layers: RGCN层数
            freeze_embeddings: 是否冻结嵌入
            dropout: Dropout率
            ffn_hidden_dim: FFN隐藏层维度
        """
        super(HallucinationClassifier, self).__init__()
        
        # RGCN编码器（共享权重）
        self.encoder = RGCNEncoderWithPooling(
            entity_embedding_path=entity_embedding_path,
            relation_embedding_path=relation_embedding_path,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            freeze_embeddings=freeze_embeddings,
            dropout=dropout
        )
        
        # 🔥 FFN分类头
        # 输入: concat[h_response, h_reference] → 2 * out_channels
        # 输出: 2 (幻觉 vs 非幻觉)
        self.classifier = nn.Sequential(
            nn.Linear(2 * out_channels, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, ffn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim // 2, 2)  # 2分类
        )
        
        print(f"\n分类器架构:")
        print(f"  输入维度: {2 * out_channels} (concat[h_response, h_reference])")
        print(f"  FFN隐藏层: {ffn_hidden_dim} → {ffn_hidden_dim // 2}")
        print(f"  输出维度: 2 (幻觉 vs 非幻觉)")
    
    def forward(self, response_graph, reference_graph):
        """
        前向传播
        Args:
            response_graph: 响应图（GPT生成）
            reference_graph: 参考图（KB）
        Returns:
            logits: 分类logits [batch_size, 2]
        """
        # 1. RGCN编码 + 全局平均池化
        h_response = self.encoder(response_graph)      # [batch_size, out_channels]
        h_reference = self.encoder(reference_graph)    # [batch_size, out_channels]
        
        # 2. 拼接特征
        # concat[h_response, h_reference]
        h_concat = torch.cat([h_response, h_reference], dim=1)  # [batch_size, 2*out_channels]
        
        # 3. FFN分类
        # ŷ = FFNs(concat[h_response, h_reference])
        logits = self.classifier(h_concat)  # [batch_size, 2]
        
        return logits
    
    def predict(self, response_graph, reference_graph):
        """
        预测（推理模式）
        Returns:
            predictions: 预测类别 [batch_size] (0=幻觉, 1=非幻觉)
            probabilities: 预测概率 [batch_size, 2]
        """
        logits = self.forward(response_graph, reference_graph)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities


if __name__ == "__main__":
    print("测试幻觉检测分类器...")
    
    # 示例：需要先准备嵌入文件
    print("\n使用前请确保:")
    print("  1. 运行 generate_hybrid_embeddings.py 生成混合嵌入")
    print("  2. 运行 prepare_embeddings.py 准备RGCN嵌入")
    print("  3. 准备图数据（PyG Data对象）")
    
    print("\n模型架构:")
    print("  输入: G_response, G_reference")
    print("  RGCN编码 → 节点特征")
    print("  全局平均池化 → 图特征 h")
    print("  拼接 → concat[h_response, h_reference]")
    print("  FFN → logits")
    print("  Softmax → [P(幻觉), P(非幻觉)]")











