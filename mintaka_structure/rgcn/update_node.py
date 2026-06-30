"""
从链接预测RGCN中提取更新后的节点嵌入 - Mintaka版本
纯PyTorch实现，不依赖torch_geometric/torch_scatter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import json
from tqdm import tqdm
import os


# ============================================================================
# 模型定义（与训练时完全相同）
# ============================================================================

class RGCNLayer(nn.Module):
    """纯PyTorch实现的RGCN层（basis decomposition）"""

    def __init__(self, in_dim, out_dim, num_relations, num_bases=30):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        self.bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, x, edge_index, edge_type):
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # 只计算当前子图中出现的关系类型的权重
        unique_rels = torch.unique(edge_type)
        W_subset = torch.einsum('rb,bio->rio', self.coefficients[unique_rels], self.bases)

        rel_map = torch.zeros(self.num_relations, dtype=torch.long, device=x.device)
        rel_map[unique_rels] = torch.arange(len(unique_rels), device=x.device)
        local_edge_type = rel_map[edge_type]

        src_features = x[src]
        edge_W = W_subset[local_edge_type]
        messages = torch.bmm(src_features.unsqueeze(1), edge_W).squeeze(1)

        out = torch.zeros(num_nodes, self.out_dim, device=x.device)
        count = torch.zeros(num_nodes, 1, device=x.device)

        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=x.device))

        count = count.clamp(min=1)
        out = out / count

        out = out + self.self_loop(x) + self.bias

        return out


class RGCNEncoderLinkPred(nn.Module):
    """RGCN编码器 - 链接预测版本（纯PyTorch）"""

    def __init__(self, entity_embedding_path, relation_embedding_path,
                 num_layers=2, dropout=0.3, num_bases=30):
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
                RGCNLayer(self.embedding_dim, self.embedding_dim,
                          self.num_relations, num_bases=num_bases)
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
                 num_layers=2, dropout=0.3, num_bases=30):
        super(LinkPredictionRGCN, self).__init__()

        self.encoder = RGCNEncoderLinkPred(
            entity_embedding_path, relation_embedding_path,
            num_layers, dropout, num_bases
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
    """从链接预测RGCN提取更新后的节点嵌入"""

    def __init__(self, checkpoint_path, entity2idx, relation2idx, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.entity2idx = entity2idx
        self.relation2idx = relation2idx
        self.num_entities = len(entity2idx)

        print(f"使用设备: {self.device}")

        print(f"\n[1/3] 加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        print(f"\n[2/3] 读取配置")
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint中没有config信息")

        self.config = checkpoint['config']
        print(f"  嵌入维度: {checkpoint.get('embedding_dim', 'N/A')}")
        print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")

        print(f"\n[3/3] 重建链接预测RGCN模型")
        self.model = LinkPredictionRGCN(
            entity_embedding_path=self.config['entity_embedding_path'],
            relation_embedding_path=self.config['relation_embedding_path'],
            num_layers=self.config.get('num_layers', 2),
            dropout=0.0,
            num_bases=self.config.get('num_bases', 30)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"\n模型准备完成:")
        print(f"  实体数: {self.num_entities}")
        print(f"  嵌入维度: {self.model.embedding_dim}")

    def _flatten_entity_triples(self, entity_triples):
        """将Mintaka的entity_triples嵌套字典展平为三元组列表"""
        all_triples = []
        for entity_qid, entity_info in entity_triples.items():
            if 'triples' in entity_info:
                all_triples.extend(entity_info['triples'])
        return all_triples

    def _build_subgraph_from_triples(self, triples_list):
        """从Mintaka格式三元组列表构建子图"""
        entities_set = set()
        triples_parsed = []

        for triple in triples_list:
            h = triple.get('head', '')
            r = triple.get('relation', '')
            t = triple.get('tail', '')

            if not h or not r or not t:
                continue

            h_id = self.entity2idx.get(h)
            r_id = self.relation2idx.get(r)
            t_id = self.entity2idx.get(t)

            if h_id is None or r_id is None or t_id is None:
                continue

            entities_set.add(h_id)
            entities_set.add(t_id)
            triples_parsed.append((h_id, r_id, t_id))

        if not entities_set or not triples_parsed:
            return None

        entities_list = sorted(list(entities_set))
        global_to_local = {ent: i for i, ent in enumerate(entities_list)}

        edge_index = [[], []]
        edge_types = []

        for h_global, r, t_global in triples_parsed:
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
        """对子图进行编码，返回节点嵌入"""
        if subgraph_data is None:
            return None, None

        entities = subgraph_data['entities']
        edge_index = subgraph_data['edge_index']
        edge_types = subgraph_data['edge_types']

        node_ids_t = torch.LongTensor(entities).to(self.device)
        edge_index_t = torch.LongTensor(edge_index).to(self.device)
        edge_type_t = torch.LongTensor(edge_types).to(self.device)

        node_emb = self.model.encoder(node_ids_t, edge_index_t, edge_type_t)

        return node_emb.cpu().numpy(), entities

    def extract_from_json_dataset(self, data_path):
        """从Mintaka JSON数据集提取节点表示"""
        print(f"\n从数据集提取节点表示...")
        print(f"数据文件: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"样本数: {len(data)}")

        node_representations = {}
        skipped = 0

        for item in tqdm(data, desc="处理样本"):
            if 'entity_triples' not in item:
                skipped += 1
                continue

            triples_list = self._flatten_entity_triples(item['entity_triples'])

            if not triples_list:
                skipped += 1
                continue

            subgraph = self._build_subgraph_from_triples(triples_list)

            if subgraph is None:
                skipped += 1
                continue

            try:
                node_embs, global_node_ids = self.encode_subgraph(subgraph)

                if node_embs is not None:
                    for nid, emb in zip(global_node_ids, node_embs):
                        nid = int(nid)
                        if nid not in node_representations:
                            node_representations[nid] = []
                        node_representations[nid].append(emb)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\n警告: GPU OOM，跳过此样本")
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
        """聚合并保存节点表示（平均聚合）"""
        print(f"\n聚合节点表示...")

        num_appearances = {nid: len(embs) for nid, embs in node_representations.items()}
        print(f"  节点出现次数统计:")
        print(f"    最少: {min(num_appearances.values())} 次")
        print(f"    最多: {max(num_appearances.values())} 次")
        print(f"    平均: {np.mean(list(num_appearances.values())):.2f} 次")

        aggregated = {}
        for nid, emb_list in tqdm(node_representations.items(), desc="聚合"):
            aggregated[nid] = np.mean(emb_list, axis=0)

        embedding_dim = list(aggregated.values())[0].shape[0]
        embeddings_matrix = np.zeros((self.num_entities, embedding_dim), dtype=np.float32)

        coverage = 0
        for entity_name, entity_id in self.entity2idx.items():
            if entity_id in aggregated:
                embeddings_matrix[entity_id] = aggregated[entity_id]
                coverage += 1

        from datetime import datetime

        result = {
            'embeddings': embeddings_matrix,
            'num_entities': self.num_entities,
            'embedding_dim': embedding_dim,
            'entity2id': self.entity2idx,
            'source': 'link_prediction_rgcn_mintaka',
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

        print(f"\n节点表示已保存")
        print(f"  输出: {output_path}")
        print(f"  形状: {embeddings_matrix.shape}")
        print(f"  覆盖率: {coverage}/{self.num_entities} ({100 * coverage / self.num_entities:.1f}%)")

        return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    import sys
    sys.path.insert(0, '.')

    from config_mintaka import Config

    print("=" * 70)
    print("  从链接预测RGCN提取节点嵌入 - Mintaka")
    print("=" * 70)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model_subgraph.pth')
    output_dir = Config.HYBRID_EMBEDDINGS_DIR
    data_path = Config.DATA_PATH

    if not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint不存在: {checkpoint_path}")
        print("请先运行 train_rgcn_linkpred.py 完成训练")
        return

    if not os.path.exists(data_path):
        print(f"\n数据文件不存在: {data_path}")
        return

    print(f"\n加载实体和关系映射...")
    with open(Config.ENTITY2IDX_PATH, 'rb') as f:
        entity2idx = pickle.load(f)

    with open(Config.RELATION2IDX_PATH, 'rb') as f:
        relation2idx = pickle.load(f)

    print(f"  实体数: {len(entity2idx)}")
    print(f"  关系数: {len(relation2idx)}")

    encoder = LinkPredRGCNNodeEncoder(
        checkpoint_path,
        entity2idx,
        relation2idx
    )

    node_representations = encoder.extract_from_json_dataset(data_path)

    output_path = os.path.join(output_dir, 'node_embeddings_linkpred_rgcn.pkl')
    result = encoder.aggregate_and_save(node_representations, output_path)

    print("\n" + "=" * 70)
    print("  提取完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
