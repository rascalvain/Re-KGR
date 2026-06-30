"""
从链接预测RGCN中提取节点嵌入
直接从数据集的context_triples构建子图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import pickle
import numpy as np
import json
from tqdm import tqdm
import os


# ============================================================================
# 模型定义（与训练时完全相同）
# ============================================================================

class RGCNEncoderLinkPred(nn.Module):
    """RGCN编码器 - 链接预测版本"""

    def __init__(self, entity_embedding_path, relation_embedding_path,
                 num_layers=2, dropout=0.3):
        super(RGCNEncoderLinkPred, self).__init__()

        with open(entity_embedding_path, 'rb') as f:
            entity_data = pickle.load(f)
            entity_embeddings = torch.FloatTensor(entity_data['embeddings'])
            self.num_entities = entity_data['num_entities']
            self.embedding_dim = entity_embeddings.shape[1]

        with open(relation_embedding_path, 'rb') as f:
            relation_data = pickle.load(f)
            self.num_relations = relation_data['num_relations']

        self.entity_embedding = nn.Embedding.from_pretrained(
            entity_embeddings, freeze=True
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(self.embedding_dim, self.embedding_dim, self.num_relations)
            )

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.embedding_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, node_ids, edge_index, edge_type):
        x = self.entity_embedding(node_ids)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            x = self.batch_norms[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x


class DistMultScorer(nn.Module):
    """DistMult评分函数"""

    def __init__(self, embedding_dim, num_relations):
        super(DistMultScorer, self).__init__()
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, head_emb, relation_ids, tail_emb):
        rel_emb = self.relation_embeddings(relation_ids)
        scores = torch.sum(head_emb * rel_emb * tail_emb, dim=1)
        return scores


class LinkPredictionRGCN(nn.Module):
    """链接预测RGCN模型"""

    def __init__(self, entity_embedding_path, relation_embedding_path,
                 num_layers=2, dropout=0.3):
        super(LinkPredictionRGCN, self).__init__()

        self.encoder = RGCNEncoderLinkPred(
            entity_embedding_path, relation_embedding_path,
            num_layers, dropout
        )

        self.scorer = DistMultScorer(
            self.encoder.embedding_dim,
            self.encoder.num_relations
        )

        self.embedding_dim = self.encoder.embedding_dim

    def forward(self, node_ids, edge_index, edge_type,
                head_indices, relation_ids, tail_indices):
        node_embeddings = self.encoder(node_ids, edge_index, edge_type)
        head_emb = node_embeddings[head_indices]
        tail_emb = node_embeddings[tail_indices]
        scores = self.scorer.score(head_emb, relation_ids, tail_emb)
        return scores, node_embeddings


# ============================================================================
# 节点嵌入提取器
# ============================================================================

class LinkPredRGCNNodeEncoder:
    """从链接预测RGCN提取节点嵌入"""

    def __init__(self, checkpoint_path, entity2idx, relation2idx, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.entity2idx = entity2idx
        self.relation2idx = relation2idx
        self.num_entities = len(entity2idx)

        print(f"使用设备: {self.device}")

        # 加载checkpoint
        print(f"\n[1/3] 加载checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        print(f"  ✓ Checkpoint加载成功")

        # 读取配置
        print(f"\n[2/3] 读取配置")
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint中没有config信息")

        self.config = checkpoint['config']
        print(f"  嵌入维度: {checkpoint.get('embedding_dim', 'N/A')}")
        print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")

        # 重建模型
        print(f"\n[3/3] 重建链接预测RGCN模型")

        self.model = LinkPredictionRGCN(
            entity_embedding_path=self.config['entity_embedding_path'],
            relation_embedding_path=self.config['relation_embedding_path'],
            num_layers=self.config.get('num_layers', 2),
            dropout=0.0
        ).to(self.device)

        # 加载权重
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ 权重加载成功")
        except RuntimeError as e:
            print(f"  ❌ 加载失败: {e}")
            raise

        self.model.eval()

        print(f"\n模型准备完成:")
        print(f"  实体数: {self.num_entities}")
        print(f"  嵌入维度: {self.model.embedding_dim}")

    def _parse_triple(self, triple_str):
        """解析三元组（与训练代码相同）"""
        triple_str = triple_str.strip()
        if triple_str.startswith('(') and triple_str.endswith(')'):
            triple_str = triple_str[1:-1]

        parts = []
        current = ''
        paren_count = 0

        for char in triple_str:
            if char == ',' and paren_count == 0:
                parts.append(current.strip())
                current = ''
            else:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                current += char

        if current:
            parts.append(current.strip())

        return tuple(parts) if len(parts) == 3 else (None, None, None)

    def _build_subgraph_from_triples(self, triples_list):
        """
        从三元组列表构建子图（与训练代码相同）

        Returns:
            dict with:
                - entities: [全局实体ID列表]
                - edge_index: [[源节点局部索引], [目标节点局部索引]]
                - edge_types: [边类型列表]
                - global_to_local: {全局ID: 局部ID}
        """
        entities_set = set()
        triples_parsed = []

        for triple_obj in triples_list:
            if 'triple' in triple_obj:
                h, r, t = self._parse_triple(triple_obj['triple'])
                if h and r and t:
                    h_id = self.entity2idx.get(h, 0)
                    r_id = self.relation2idx.get(r, 0)
                    t_id = self.entity2idx.get(t, 0)

                    entities_set.add(h_id)
                    entities_set.add(t_id)
                    triples_parsed.append((h_id, r_id, t_id))

        if not entities_set or not triples_parsed:
            return None

        # 创建全局到局部的映射
        entities_list = sorted(list(entities_set))
        global_to_local = {ent: i for i, ent in enumerate(entities_list)}

        # 构建边索引和边类型
        edge_index = [[], []]
        edge_types = []

        for h_global, r, t_global in triples_parsed:
            if h_global in global_to_local and t_global in global_to_local:
                h_local = global_to_local[h_global]
                t_local = global_to_local[t_global]
                edge_index[0].append(h_local)
                edge_index[1].append(t_local)
                edge_types.append(r)

        return {
            'entities': entities_list,
            'edge_index': edge_index,
            'edge_types': edge_types,
            'global_to_local': global_to_local
        }

    @torch.no_grad()
    def encode_subgraph(self, subgraph_data):
        """
        对子图进行编码

        Args:
            subgraph_data: dict from _build_subgraph_from_triples

        Returns:
            node_embeddings: numpy array [num_nodes, embedding_dim]
            global_node_ids: list of global node IDs
        """
        if subgraph_data is None:
            return None, None

        entities = subgraph_data['entities']
        edge_index = subgraph_data['edge_index']
        edge_types = subgraph_data['edge_types']

        # 转换为tensor
        node_ids_t = torch.LongTensor(entities).to(self.device)
        edge_index_t = torch.LongTensor(edge_index).to(self.device)
        edge_type_t = torch.LongTensor(edge_types).to(self.device)

        # 通过编码器获取节点表示
        node_emb = self.model.encoder(node_ids_t, edge_index_t, edge_type_t)

        return node_emb.cpu().numpy(), entities

    def extract_from_json_dataset(self, data_path):
        """
        从JSON数据集提取节点表示

        Args:
            data_path: hotpot_dev_with_triples_aligned.json 路径

        Returns:
            node_representations: dict {node_id: [list of embeddings]}
        """
        print(f"\n从数据集提取节点表示...")
        print(f"数据文件: {data_path}")

        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"样本数: {len(data)}")

        node_representations = {}
        skipped = 0

        for item in tqdm(data, desc="处理样本"):
            # 检查是否有context_triples字段
            if 'context_triples' not in item:
                skipped += 1
                continue

            # 从三元组构建子图
            subgraph = self._build_subgraph_from_triples(item['context_triples'])

            if subgraph is None:
                skipped += 1
                continue

            # 编码子图
            try:
                node_embs, global_node_ids = self.encode_subgraph(subgraph)

                if node_embs is not None:
                    # 存储每个节点的表示
                    for nid, emb in zip(global_node_ids, node_embs):
                        nid = int(nid)
                        if nid not in node_representations:
                            node_representations[nid] = []
                        node_representations[nid].append(emb)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\n⚠️ GPU OOM，跳过此样本")
                    torch.cuda.empty_cache()
                    skipped += 1
                    continue
                else:
                    raise e

        print(f"\n处理完成:")
        print(f"  成功处理: {len(data) - skipped} 个样本")
        print(f"  跳过: {skipped} 个样本")
        print(f"  提取节点数: {len(node_representations)}")

        return node_representations

    def aggregate_and_save(self, node_representations, output_path):
        """聚合并保存节点表示（按entity2idx顺序）"""
        print(f"\n聚合节点表示...")

        # 统计
        num_appearances = {nid: len(embs) for nid, embs in node_representations.items()}
        print(f"  节点出现次数统计:")
        print(f"    最少: {min(num_appearances.values())} 次")
        print(f"    最多: {max(num_appearances.values())} 次")
        print(f"    平均: {np.mean(list(num_appearances.values())):.2f} 次")

        # 平均聚合
        aggregated = {}
        for nid, emb_list in tqdm(node_representations.items(), desc="聚合"):
            aggregated[nid] = np.mean(emb_list, axis=0)

        # 按entity2idx顺序排列
        embedding_dim = list(aggregated.values())[0].shape[0]
        embeddings_matrix = np.zeros((self.num_entities, embedding_dim), dtype=np.float32)

        coverage = 0
        for entity_name, entity_id in self.entity2idx.items():
            if entity_id in aggregated:
                embeddings_matrix[entity_id] = aggregated[entity_id]
                coverage += 1

        # 保存
        from datetime import datetime

        result = {
            'embeddings': embeddings_matrix,
            'num_entities': self.num_entities,
            'embedding_dim': embedding_dim,
            'entity2id': self.entity2idx,
            'source': 'link_prediction_rgcn',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'coverage': coverage,
            'statistics': {
                'num_appearances': num_appearances,
                'total_covered': coverage
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

        print(f"\n✓ 节点表示已保存")
        print(f"  输出: {output_path}")
        print(f"  形状: {embeddings_matrix.shape}")
        print(f"  覆盖率: {coverage}/{self.num_entities} ({100*coverage/self.num_entities:.1f}%)")

        return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    import sys
    sys.path.insert(0, '.')

    from config_hotpotqa import Config

    print("="*70)
    print("  从链接预测RGCN提取节点嵌入")
    print("="*70)

    # 配置
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model_subgraph.pth')
    output_dir = Config.HYBRID_EMBEDDINGS_DIR
    data_path = Config.HOTPOTQA_DATA_PATH

    # 检查文件
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint不存在: {checkpoint_path}")
        print("\n可能的checkpoint文件名:")
        print("  - best_model_subgraph.pth")
        print("  - best_model.pth")
        return

    if not os.path.exists(data_path):
        print(f"\n❌ 数据文件不存在: {data_path}")
        return

    # 加载映射
    print(f"\n加载实体和关系映射...")
    with open(Config.ENTITY2IDX_PATH, 'rb') as f:
        entity2idx = pickle.load(f)

    with open(Config.RELATION2IDX_PATH, 'rb') as f:
        relation2idx = pickle.load(f)

    print(f"  实体数: {len(entity2idx)}")
    print(f"  关系数: {len(relation2idx)}")

    # 创建编码器
    encoder = LinkPredRGCNNodeEncoder(
        checkpoint_path,
        entity2idx,
        relation2idx
    )

    # 从JSON数据集提取节点表示
    node_representations = encoder.extract_from_json_dataset(data_path)

    # 聚合并保存
    output_path = os.path.join(output_dir, 'node_embeddings_linkpred_rgcn.pkl')
    result = encoder.aggregate_and_save(node_representations, output_path)

    print("\n" + "="*70)
    print("  提取完成！")
    print("="*70)
    print("\n使用示例:")
    print("""
import pickle

# 加载提取的节点嵌入
with open('node_embeddings_linkpred_rgcn.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']  # [num_entities, 384]
entity2id = data['entity2id']    # {entity_name: index}

# 查询特定实体的嵌入
entity_name = "Paris"
if entity_name in entity2id:
    idx = entity2id[entity_name]
    entity_emb = embeddings[idx]
    print(f"{entity_name} 的RGCN嵌入: {entity_emb[:5]}...")
    """)


if __name__ == '__main__':
    main()