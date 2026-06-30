"""
Mintaka 数据加载器 - 使用混合嵌入
将 entity_triples 和 gpt_triples 转换为图数据
"""

import json
import torch
import pickle
import numpy as np
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from tqdm import tqdm


class MintakaGraphDataset(Dataset):
    """Mintaka 图数据集加载器"""

    def __init__(self, data_path, entity_mapping_path, relation_mapping_path, max_samples=None):
        """
        初始化数据集
        Args:
            data_path: Mintaka JSON 文件路径
            entity_mapping_path: 实体映射 pkl 文件路径
            relation_mapping_path: 关系映射 pkl 文件路径
            max_samples: 最大样本数
        """
        # 加载数据
        print(f"加载Mintaka数据集: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        if max_samples:
            self.data = self.data[:max_samples]

        # 加载实体映射
        print(f"加载实体映射: {entity_mapping_path}")
        with open(entity_mapping_path, 'rb') as f:
            self.entity2id = pickle.load(f)

        # 加载关系映射
        print(f"加载关系映射: {relation_mapping_path}")
        with open(relation_mapping_path, 'rb') as f:
            self.relation2id = pickle.load(f)

        # 设置 UNK ID（如果实体/关系不在映射中，使用ID 0）
        self.unk_entity_id = 0
        self.unk_relation_id = 0

        # 过滤有效数据
        self.valid_data = []
        print("过滤有效样本...")
        for item in tqdm(self.data):
            # 检查是否同时存在 entity_triples、gpt_triples 和 generation_label
            if (item.get('entity_triples') and
                item.get('gpt_triples') and
                len(item['entity_triples']) > 0 and
                len(item['gpt_triples']) > 0 and
                item.get('generation_label') in ['hallucination', 'correct']):
                self.valid_data.append(item)

        # 统计标签分布
        hallucination_count = sum(1 for item in self.valid_data
                                  if item.get('generation_label') == 'hallucination')
        correct_count = sum(1 for item in self.valid_data
                           if item.get('generation_label') == 'correct')

        print(f"\n数据集统计:")
        print(f"  总样本: {len(self.data)}")
        print(f"  有效样本: {len(self.valid_data)}")
        print(f"  - 幻觉样本 (label=0): {hallucination_count} ({hallucination_count/len(self.valid_data)*100:.1f}%)")
        print(f"  - 正确样本 (label=1): {correct_count} ({correct_count/len(self.valid_data)*100:.1f}%)")
        print(f"  实体词表大小: {len(self.entity2id)}")
        print(f"  关系词表大小: {len(self.relation2id)}")

    def __len__(self):
        return len(self.valid_data)

    def _triples_to_pyg_data(self, triples_list, graph_label='entity'):
        """
        将三元组列表转换为 PyG Data 对象
        Args:
            triples_list: 三元组列表，每个元素是 {"head": ..., "relation": ..., "tail": ...}
            graph_label: 图标签（用于调试）
        Returns:
            Data 对象，包含 node_ids, edge_index, edge_type
        """
        if not triples_list:
            return None

        try:
            # 收集所有节点（实体）
            entities_in_graph = set()
            parsed_triples = []

            for triple_obj in triples_list:
                # Mintaka 格式：{"head": ..., "relation": ..., "tail": ...}
                if 'head' in triple_obj and 'relation' in triple_obj and 'tail' in triple_obj:
                    head = triple_obj['head']
                    relation = triple_obj['relation']
                    tail = triple_obj['tail']

                    if head and relation and tail:
                        entities_in_graph.add(head)
                        entities_in_graph.add(tail)
                        parsed_triples.append((head, relation, tail))

            if not entities_in_graph or not parsed_triples:
                return None

            # 创建局部节点映射（图内ID -> 全局ID）
            entities_list = sorted(list(entities_in_graph))
            local2global_entity = {}  # 局部ID -> 全局ID
            node_ids = []  # 全局实体ID列表

            for local_id, entity in enumerate(entities_list):
                global_id = self.entity2id.get(entity, self.unk_entity_id)
                local2global_entity[entity] = local_id
                node_ids.append(global_id)

            # 构建边
            edge_index = [[], []]  # [source_nodes, target_nodes]
            edge_types = []  # 边的关系类型

            for head, relation, tail in parsed_triples:
                if head in local2global_entity and tail in local2global_entity:
                    source_id = local2global_entity[head]
                    target_id = local2global_entity[tail]
                    relation_id = self.relation2id.get(relation, self.unk_relation_id)

                    edge_index[0].append(source_id)
                    edge_index[1].append(target_id)
                    edge_types.append(relation_id)

            if not edge_index[0]:
                return None

            # 转换为张量
            node_ids_tensor = torch.LongTensor(node_ids)
            edge_index_tensor = torch.LongTensor(edge_index)
            edge_type_tensor = torch.LongTensor(edge_types)

            # 创建 PyG Data 对象
            data = Data(
                node_ids=node_ids_tensor,
                edge_index=edge_index_tensor,
                edge_type=edge_type_tensor,
                num_nodes=len(node_ids)
            )

            return data

        except Exception as e:
            print(f"转换图 {graph_label} 时出错: {e}")
            return None

    def _extract_entity_triples_list(self, entity_triples_dict):
        """
        从嵌套字典中提取三元组列表

        Args:
            entity_triples_dict: 格式为 {"entity_id": {"triples": [...]}} 的字典

        Returns:
            list: 三元组列表
        """
        if not entity_triples_dict:
            return []

        # 如果已经是列表，直接返回
        if isinstance(entity_triples_dict, list):
            return entity_triples_dict

        # 如果是字典，提取所有实体的三元组
        all_triples = []
        if isinstance(entity_triples_dict, dict):
            for entity_id, entity_data in entity_triples_dict.items():
                if isinstance(entity_data, dict) and 'triples' in entity_data:
                    triples = entity_data['triples']
                    if isinstance(triples, list):
                        all_triples.extend(triples)

        return all_triples

    def __getitem__(self, idx):
        """
        获取一个样本
        Returns:
            tuple: (entity_graph, gpt_graph, label, metadata)
                - entity_graph: 实体图（参考图/KB图）
                - gpt_graph: GPT生成图（响应图）
                - label: 标签（0=幻觉，1=正常）
                - metadata: 元数据字典
        """
        item = self.valid_data[idx]

        # 先提取标签（确保采样器可以访问）
        generation_label = item.get('generation_label', None)

        if generation_label == 'hallucination':
            label = 0  # 幻觉
        elif generation_label == 'correct':
            label = 1  # 正确/非幻觉
        else:
            # 如果没有标签，跳过该样本
            print(f"警告: 样本 {item.get('id')} 没有有效的 generation_label，跳过")
            return None

        # 提取三元组列表
        entity_triples_raw = item.get('entity_triples', {})
        entity_triples_list = self._extract_entity_triples_list(entity_triples_raw)

        gpt_triples_list = item.get('gpt_triples', [])

        # 元数据
        metadata = {
            'id': item.get('id', ''),
            'question': item.get('question', ''),
            'answer': item.get('answer', {}),
            'num_entity_triples': len(entity_triples_list),
            'num_gpt_triples': len(gpt_triples_list),
            'generation_label': generation_label,
            'label': label
        }

        # 转换图（可能失败）
        entity_graph = self._triples_to_pyg_data(
            entity_triples_list,
            graph_label=f'entity_{idx}'
        )

        gpt_graph = self._triples_to_pyg_data(
            gpt_triples_list,
            graph_label=f'gpt_{idx}'
        )

        # 🔥 关键修改：即使图转换失败，也返回带标签的元组
        # 这样采样器可以提取标签，collate_fn 会过滤掉无效图
        if entity_graph is None or gpt_graph is None:
            # 返回 None 作为图，但保留标签和元数据
            return None, None, label, metadata

        return entity_graph, gpt_graph, label, metadata


def collate_fn(batch):
    """
    自定义批处理函数
    Args:
        batch: 样本列表
    Returns:
        (entity_batch, gpt_batch, labels, metadata_list)
    """
    # 过滤 None 样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None, None, None, None

    entity_graphs = []
    gpt_graphs = []
    labels = []
    metadata_list = []

    for entity_graph, gpt_graph, label, metadata in batch:
        if entity_graph is not None and gpt_graph is not None:
            entity_graphs.append(entity_graph)
            gpt_graphs.append(gpt_graph)
            labels.append(label)
            metadata_list.append(metadata)

    if len(entity_graphs) == 0:
        return None, None, None, None

    # 批处理图数据
    entity_batch = Batch.from_data_list(entity_graphs)
    gpt_batch = Batch.from_data_list(gpt_graphs)
    labels_tensor = torch.LongTensor(labels)

    return entity_batch, gpt_batch, labels_tensor, metadata_list


if __name__ == '__main__':
    # 测试数据加载器
    from config_mintaka_rgat import Config

    print("测试Mintaka数据加载器")
    print("=" * 60)

    # 检查文件是否存在
    import os
    if not os.path.exists(Config.DATA_PATH):
        print(f"错误: 数据文件不存在 - {Config.DATA_PATH}")
        exit(1)

    if not os.path.exists(Config.ENTITY2IDX_PATH):
        print(f"错误: 实体映射文件不存在 - {Config.ENTITY2IDX_PATH}")
        print("请先运行: python generate_hybrid_embeddings.py")
        exit(1)

    if not os.path.exists(Config.RELATION2IDX_PATH):
        print(f"错误: 关系映射文件不存在 - {Config.RELATION2IDX_PATH}")
        print("请先运行: python generate_hybrid_embeddings.py")
        exit(1)

    # 创建数据集
    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=10
    )

    print(f"\n数据集大小: {len(dataset)}")

    # 测试获取一个样本
    entity_graph, gpt_graph, label, metadata = dataset[0]

    print(f"\n样本 0:")
    print(f"  Entity图: {entity_graph.num_nodes} 节点, {entity_graph.edge_index.size(1)} 边")
    print(f"  GPT图: {gpt_graph.num_nodes} 节点, {gpt_graph.edge_index.size(1)} 边")
    print(f"  标签: {label} ({'幻觉' if label == 0 else '正确'})")
    print(f"  问题: {metadata['question'][:100]}...")

    # 测试批处理
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for entity_batch, gpt_batch, labels, metadata_list in loader:
        if entity_batch is None:
            continue
        print(f"\nBatch:")
        print(f"  Entity batch: {entity_batch.num_graphs} 图, {entity_batch.num_nodes} 总节点")
        print(f"  GPT batch: {gpt_batch.num_graphs} 图, {gpt_batch.num_nodes} 总节点")
        print(f"  Labels: {labels}")
        break

    print("\n测试完成！")
