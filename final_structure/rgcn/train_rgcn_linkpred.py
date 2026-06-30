"""
RGCN链接预测训练 - 直接使用数据集中的子图
每个样本的context_triples就是一个子图，无需采样
大幅降低显存占用，提高训练效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
import json
import os
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from config_hotpotqa import Config, create_directories


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

        print(f"  RGCN编码器: {num_layers}层, {self.embedding_dim}维")

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


class SubgraphLinkPredictionTrainer:
    """
    基于数据集子图的训练器
    直接使用每个样本的context_triples作为子图
    """

    def __init__(self, config_dict):
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        create_directories()
        self._check_embedding_files()

        # 加载映射
        with open(self.config['entity_mapping_path'], 'rb') as f:
            self.entity2idx = pickle.load(f)

        with open(self.config['relation_mapping_path'], 'rb') as f:
            self.relation2idx = pickle.load(f)

        # 加载数据集并构建子图训练数据
        self._load_subgraph_data()

        # 创建模型
        print("\n" + "="*60)
        print("初始化RGCN模型（子图模式）")
        print("="*60)
        self.model = LinkPredictionRGCN(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            num_layers=config_dict.get('num_layers', 2),
            dropout=config_dict.get('dropout', 0.3)
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
        """检查嵌入文件"""
        print("\n检查嵌入文件...")
        entity_emb_path = self.config['entity_embedding_path']
        relation_emb_path = self.config['relation_embedding_path']

        if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
            print(f"\n❌ 错误: 嵌入文件不存在，请先运行 prepare_embeddings.py")
            exit(1)
        print("✓ 嵌入文件检查通过")

    def _parse_triple(self, triple_str):
        """解析三元组"""
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
        从三元组列表构建子图

        Returns:
            subgraph_data: {
                'entities': [全局实体ID列表],
                'edge_index': [[源节点局部索引], [目标节点局部索引]],
                'edge_types': [边类型列表],
                'global_to_local': {全局ID: 局部ID},
                'triples': [(h_global, r, t_global), ...]  # 原始三元组
            }
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
            'global_to_local': global_to_local,
            'triples': triples_parsed
        }

    def _load_subgraph_data(self):
        """
        加载数据集并为每个样本构建子图
        """
        print("\n" + "="*60)
        print("加载数据集子图")
        print("="*60)

        with open(self.config['data_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self.config.get('max_samples'):
            data = data[:self.config['max_samples']]

        print(f"加载 {len(data)} 个样本...")

        # 为每个样本构建子图
        self.subgraphs = []

        for item in tqdm(data, desc="构建子图"):
            if 'context_triples' not in item:
                continue

            subgraph = self._build_subgraph_from_triples(item['context_triples'])

            if subgraph and len(subgraph['triples']) > 0:
                self.subgraphs.append(subgraph)

        print(f"  有效子图数: {len(self.subgraphs)}")

        # 统计子图大小
        node_counts = [len(sg['entities']) for sg in self.subgraphs]
        edge_counts = [len(sg['edge_types']) for sg in self.subgraphs]

        print(f"  子图节点数: 平均{sum(node_counts)/len(node_counts):.1f}, "
              f"最小{min(node_counts)}, 最大{max(node_counts)}")
        print(f"  子图边数: 平均{sum(edge_counts)/len(edge_counts):.1f}, "
              f"最小{min(edge_counts)}, 最大{max(edge_counts)}")

        # 划分训练/验证/测试集
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
        """生成负样本"""
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

        pbar = tqdm(range(0, len(self.train_subgraphs), batch_size), desc=f'Epoch {epoch+1}')

        for i in pbar:
            batch_subgraphs = self.train_subgraphs[i:i+batch_size]
            batch_loss = 0
            valid_batches = 0

            # 🔥 关键：每个子图独立处理
            for subgraph in batch_subgraphs:
                try:
                    # 准备子图数据
                    entities = subgraph['entities']
                    edge_index = subgraph['edge_index']
                    edge_types = subgraph['edge_types']
                    global_to_local = subgraph['global_to_local']
                    triples = subgraph['triples']

                    if len(triples) == 0:
                        continue

                    # 采样正负样本
                    num_samples = min(len(triples), 32)  # 每个子图最多32个三元组
                    pos_triples = random.sample(triples, num_samples)
                    neg_triples = self._generate_negative_samples(pos_triples, entities)

                    # 转换为局部索引
                    pos_local = [(global_to_local[h], r, global_to_local[t])
                                for h, r, t in pos_triples
                                if h in global_to_local and t in global_to_local]

                    neg_local = [(global_to_local[h], r, global_to_local[t])
                                for h, r, t in neg_triples
                                if h in global_to_local and t in global_to_local]

                    if not pos_local or not neg_local:
                        continue

                    # 转为tensor
                    node_ids = torch.LongTensor(entities).to(self.device)
                    edge_index_t = torch.LongTensor(edge_index).to(self.device)
                    edge_type_t = torch.LongTensor(edge_types).to(self.device)

                    pos_h = torch.LongTensor([h for h, r, t in pos_local]).to(self.device)
                    pos_r = torch.LongTensor([r for h, r, t in pos_local]).to(self.device)
                    pos_t = torch.LongTensor([t for h, r, t in pos_local]).to(self.device)

                    neg_h = torch.LongTensor([h for h, r, t in neg_local]).to(self.device)
                    neg_r = torch.LongTensor([r for h, r, t in neg_local]).to(self.device)
                    neg_t = torch.LongTensor([t for h, r, t in neg_local]).to(self.device)

                    # 前向传播
                    pos_scores, _ = self.model(
                        node_ids, edge_index_t, edge_type_t,
                        pos_h, pos_r, pos_t
                    )

                    neg_scores, _ = self.model(
                        node_ids, edge_index_t, edge_type_t,
                        neg_h, neg_r, neg_t
                    )

                    # Margin loss
                    loss = F.relu(self.config.get('margin', 1.0) - pos_scores + neg_scores).mean()

                    # 反向传播
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
        """验证"""
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

                # 随机选一个三元组验证
                h_global, r, t_global = random.choice(triples)

                if h_global not in global_to_local or t_global not in global_to_local:
                    continue

                h_local = global_to_local[h_global]
                t_local = global_to_local[t_global]

                # 转为tensor
                node_ids = torch.LongTensor(entities).to(self.device)
                edge_index_t = torch.LongTensor(edge_index).to(self.device)
                edge_type_t = torch.LongTensor(edge_types).to(self.device)

                num_entities = len(entities)
                heads = torch.LongTensor([h_local] * num_entities).to(self.device)
                rels = torch.LongTensor([r] * num_entities).to(self.device)
                tails = torch.LongTensor(list(range(num_entities))).to(self.device)

                scores, _ = self.model(node_ids, edge_index_t, edge_type_t, heads, rels, tails)
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
        print("\n" + "="*60)
        print("开始训练（子图模式）")
        print("="*60)
        print(f"训练轮数: {self.config['num_epochs']}")
        print(f"批大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_mrr = self.validate()
            self.val_metrics.append(val_mrr)

            self.scheduler.step()

            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证MRR: {val_mrr:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            if val_mrr > self.best_val_metric:
                self.best_val_metric = val_mrr
                self.patience_counter = 0
                self.save_checkpoint('best_model_subgraph.pth', epoch, val_mrr)
                print(f"  ✓ 保存最佳模型 (MRR: {val_mrr:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.get('early_stopping_patience', 20):
                print(f"\n早停触发")
                break

            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_mrr)

            torch.cuda.empty_cache()

        print("\n训练完成")
        print(f"最佳验证MRR: {self.best_val_metric:.4f}")

    def save_checkpoint(self, filename, epoch, val_metric):
        """保存检查点"""
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
    """主函数"""
    Config.print_config()

    config_dict = Config.get_config_dict()

    # 优化配置
    config_dict['num_layers'] = 2  # 2层足够
    config_dict['batch_size'] = 4  # 每批处理4个子图

    trainer = SubgraphLinkPredictionTrainer(config_dict)
    trainer.train()

    print("\n✓ 训练完成！")
    print(f"模型输出维度: {trainer.model.embedding_dim}维")


if __name__ == '__main__':
    main()