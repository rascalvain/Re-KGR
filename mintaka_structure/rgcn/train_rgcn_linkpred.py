"""
RGCN链接预测训练 - Mintaka版本
纯PyTorch实现，不依赖torch_geometric/torch_scatter
每个样本的entity_triples构成子图，使用链接预测目标训练RGCN更新节点表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import pickle
import random
from tqdm import tqdm

from config_mintaka import Config, create_directories


class RGCNLayer(nn.Module):
    """纯PyTorch实现的RGCN层（basis decomposition）"""

    def __init__(self, in_dim, out_dim, num_relations, num_bases=30):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis matrices: num_bases个 (in_dim, out_dim) 矩阵
        self.bases = nn.Parameter(torch.Tensor(self.num_bases, in_dim, out_dim))
        # 每个关系的组合系数
        self.coefficients = nn.Parameter(torch.Tensor(num_relations, self.num_bases))
        # 自环变换
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: [num_nodes, in_dim]
            edge_index: [2, num_edges] (src, dst)
            edge_type: [num_edges]
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # 只计算当前子图中出现的关系类型的权重
        # 避免展开全部num_relations个关系的权重矩阵（5044*868*868会OOM）
        unique_rels = torch.unique(edge_type)
        # W_subset: [K, in_dim, out_dim]  K=子图中关系类型数（通常<30）
        W_subset = torch.einsum('rb,bio->rio', self.coefficients[unique_rels], self.bases)

        # 建立 global_rel_id -> local_index 映射
        rel_map = torch.zeros(self.num_relations, dtype=torch.long, device=x.device)
        rel_map[unique_rels] = torch.arange(len(unique_rels), device=x.device)
        local_edge_type = rel_map[edge_type]

        # 消息计算
        src_features = x[src]  # [num_edges, in_dim]
        edge_W = W_subset[local_edge_type]  # [num_edges, in_dim, out_dim]
        messages = torch.bmm(src_features.unsqueeze(1), edge_W).squeeze(1)  # [num_edges, out_dim]

        # 聚合到目标节点（mean aggregation）
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

        print(f"  RGCN编码器: {num_layers}层, {self.embedding_dim}维, "
              f"{self.num_entities}实体, {self.num_relations}关系, "
              f"num_bases={num_bases}")

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


class SubgraphLinkPredictionTrainer:
    """
    基于Mintaka数据集子图的链接预测训练器
    使用每个样本的entity_triples构建子图
    """

    def __init__(self, config_dict):
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        create_directories()
        self._check_embedding_files()

        with open(self.config['entity_mapping_path'], 'rb') as f:
            self.entity2idx = pickle.load(f)

        with open(self.config['relation_mapping_path'], 'rb') as f:
            self.relation2idx = pickle.load(f)

        self._load_subgraph_data()

        print("\n" + "=" * 60)
        print("初始化RGCN模型（链接预测）")
        print("=" * 60)
        self.model = LinkPredictionRGCN(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            num_layers=config_dict.get('num_layers', 2),
            dropout=config_dict.get('dropout', 0.3),
            num_bases=config_dict.get('num_bases', 30)
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config_dict['learning_rate'],
            weight_decay=config_dict.get('weight_decay', 1e-5)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config_dict.get('t_0', 10),
            T_mult=config_dict.get('t_mult', 2),
            eta_min=config_dict.get('eta_min', 1e-6)
        )

        self.train_losses = []
        self.val_metrics = []
        self.best_val_metric = 0.0
        self.patience_counter = 0

    def _check_embedding_files(self):
        print("\n检查嵌入文件...")
        entity_emb_path = self.config['entity_embedding_path']
        relation_emb_path = self.config['relation_embedding_path']

        if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
            print(f"\n错误: 嵌入文件不存在，请先运行 prepare_embeddings.py")
            exit(1)
        print("嵌入文件检查通过")

    def _flatten_entity_triples(self, entity_triples):
        """
        将Mintaka的entity_triples嵌套字典展平为三元组列表
        """
        all_triples = []
        for entity_qid, entity_info in entity_triples.items():
            if 'triples' in entity_info:
                all_triples.extend(entity_info['triples'])
        return all_triples

    def _build_subgraph_from_triples(self, triples_list):
        """
        从Mintaka格式的三元组列表构建子图
        """
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
            'global_to_local': global_to_local,
            'triples': triples_parsed
        }

    def _load_subgraph_data(self):
        """加载Mintaka数据集并为每个样本构建子图"""
        print("\n" + "=" * 60)
        print("加载数据集子图")
        print("=" * 60)

        with open(self.config['data_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self.config.get('max_samples'):
            data = data[:self.config['max_samples']]

        print(f"加载 {len(data)} 个样本...")

        self.subgraphs = []

        for item in tqdm(data, desc="构建子图"):
            if 'entity_triples' not in item:
                continue

            triples_list = self._flatten_entity_triples(item['entity_triples'])

            if not triples_list:
                continue

            max_triples = self.config.get('max_triples_per_subgraph', 32)
            if len(triples_list) > max_triples:
                triples_list = random.sample(triples_list, max_triples)

            subgraph = self._build_subgraph_from_triples(triples_list)

            if subgraph and len(subgraph['triples']) > 0:
                self.subgraphs.append(subgraph)

        print(f"  有效子图数: {len(self.subgraphs)}")

        if not self.subgraphs:
            print("错误: 没有有效子图，请检查数据和映射文件")
            exit(1)

        node_counts = [len(sg['entities']) for sg in self.subgraphs]
        edge_counts = [len(sg['edge_types']) for sg in self.subgraphs]

        print(f"  子图节点数: 平均{sum(node_counts)/len(node_counts):.1f}, "
              f"最小{min(node_counts)}, 最大{max(node_counts)}")
        print(f"  子图边数: 平均{sum(edge_counts)/len(edge_counts):.1f}, "
              f"最小{min(edge_counts)}, 最大{max(edge_counts)}")

        random.seed(self.config.get('seed', 42))
        random.shuffle(self.subgraphs)

        train_size = int(0.7 * len(self.subgraphs))
        val_size = int(0.15 * len(self.subgraphs))

        self.train_subgraphs = self.subgraphs[:train_size]
        self.val_subgraphs = self.subgraphs[train_size:train_size + val_size]
        self.test_subgraphs = self.subgraphs[train_size + val_size:]

        print(f"\n数据集划分:")
        print(f"  训练子图: {len(self.train_subgraphs)}")
        print(f"  验证子图: {len(self.val_subgraphs)}")
        print(f"  测试子图: {len(self.test_subgraphs)}")

    def _generate_negative_samples(self, positive_triples, all_entities):
        """生成负样本：随机替换head或tail"""
        negative_triples = []
        for h, r, t in positive_triples:
            if random.random() < 0.5:
                neg_h = random.choice(all_entities)
                negative_triples.append((neg_h, r, t))
            else:
                neg_t = random.choice(all_entities)
                negative_triples.append((h, r, neg_t))
        return negative_triples

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_size = self.config['batch_size']

        random.shuffle(self.train_subgraphs)

        pbar = tqdm(range(0, len(self.train_subgraphs), batch_size),
                    desc=f'Epoch {epoch + 1}')

        for i in pbar:
            batch_subgraphs = self.train_subgraphs[i:i + batch_size]
            batch_loss = 0
            valid_batches = 0

            for subgraph in batch_subgraphs:
                try:
                    entities = subgraph['entities']
                    edge_index = subgraph['edge_index']
                    edge_types = subgraph['edge_types']
                    global_to_local = subgraph['global_to_local']
                    triples = subgraph['triples']

                    if len(triples) == 0:
                        continue

                    num_samples = min(len(triples), 32)
                    pos_triples = random.sample(triples, num_samples)
                    neg_triples = self._generate_negative_samples(pos_triples, entities)

                    pos_local = [(global_to_local[h], r, global_to_local[t])
                                 for h, r, t in pos_triples
                                 if h in global_to_local and t in global_to_local]

                    neg_local = [(global_to_local[h], r, global_to_local[t])
                                 for h, r, t in neg_triples
                                 if h in global_to_local and t in global_to_local]

                    if not pos_local or not neg_local:
                        continue

                    node_ids = torch.LongTensor(entities).to(self.device)
                    edge_index_t = torch.LongTensor(edge_index).to(self.device)
                    edge_type_t = torch.LongTensor(edge_types).to(self.device)

                    pos_h = torch.LongTensor([h for h, r, t in pos_local]).to(self.device)
                    pos_r = torch.LongTensor([r for h, r, t in pos_local]).to(self.device)
                    pos_t = torch.LongTensor([t for h, r, t in pos_local]).to(self.device)

                    neg_h = torch.LongTensor([h for h, r, t in neg_local]).to(self.device)
                    neg_r = torch.LongTensor([r for h, r, t in neg_local]).to(self.device)
                    neg_t = torch.LongTensor([t for h, r, t in neg_local]).to(self.device)

                    pos_scores, _ = self.model(
                        node_ids, edge_index_t, edge_type_t,
                        pos_h, pos_r, pos_t
                    )

                    neg_scores, _ = self.model(
                        node_ids, edge_index_t, edge_type_t,
                        neg_h, neg_r, neg_t
                    )

                    margin = self.config.get('margin', 1.0)
                    loss = F.relu(margin - pos_scores + neg_scores).mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    batch_loss += loss.item()
                    valid_batches += 1

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"\n警告: GPU OOM，跳过此子图")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            if valid_batches > 0:
                avg_batch_loss = batch_loss / valid_batches
                total_loss += avg_batch_loss
                pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})

        avg_loss = total_loss / max(1, len(self.train_subgraphs) // batch_size)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """验证：计算MRR"""
        self.model.eval()

        ranks = []
        sample_size = min(50, len(self.val_subgraphs))

        for subgraph in tqdm(self.val_subgraphs[:sample_size], desc="验证", leave=False):
            try:
                entities = subgraph['entities']
                edge_index = subgraph['edge_index']
                edge_types = subgraph['edge_types']
                global_to_local = subgraph['global_to_local']
                triples = subgraph['triples']

                if len(triples) == 0:
                    continue

                h_global, r, t_global = random.choice(triples)

                if h_global not in global_to_local or t_global not in global_to_local:
                    continue

                h_local = global_to_local[h_global]
                t_local = global_to_local[t_global]

                node_ids = torch.LongTensor(entities).to(self.device)
                edge_index_t = torch.LongTensor(edge_index).to(self.device)
                edge_type_t = torch.LongTensor(edge_types).to(self.device)

                num_entities = len(entities)
                heads = torch.LongTensor([h_local] * num_entities).to(self.device)
                rels = torch.LongTensor([r] * num_entities).to(self.device)
                tails = torch.LongTensor(list(range(num_entities))).to(self.device)

                scores, _ = self.model(node_ids, edge_index_t, edge_type_t,
                                       heads, rels, tails)
                sorted_indices = torch.argsort(scores, descending=True)
                rank = (sorted_indices == t_local).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(1.0 / rank)

            except RuntimeError:
                torch.cuda.empty_cache()
                continue

        mrr = sum(ranks) / len(ranks) if ranks else 0.0
        return mrr

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练（链接预测）")
        print("=" * 60)
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_mrr = self.validate()
            self.val_metrics.append(val_mrr)

            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证MRR: {val_mrr:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_mrr > self.best_val_metric:
                self.best_val_metric = val_mrr
                self.patience_counter = 0
                self.save_checkpoint('best_model_subgraph.pth', epoch, val_mrr)
                print(f"  保存最佳模型 (MRR: {val_mrr:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.get('early_stopping_patience', 20):
                print(f"\n早停触发 (patience={self.patience_counter})")
                break

            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', epoch, val_mrr)

            torch.cuda.empty_cache()

        print("\n训练完成")
        print(f"最佳验证MRR: {self.best_val_metric:.4f}")

    def save_checkpoint(self, filename, epoch, val_metric):
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metric': val_metric,
            'config': self.config,
            'embedding_dim': self.model.embedding_dim
        }, checkpoint_path)


def main():
    Config.print_config()
    config_dict = Config.get_config_dict()

    trainer = SubgraphLinkPredictionTrainer(config_dict)
    trainer.train()

    print(f"\n训练完成！模型输出维度: {trainer.model.embedding_dim}维")
    print(f"下一步: python update_node.py")


if __name__ == '__main__':
    main()
