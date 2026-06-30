"""
生成完整的RGCN更新后节点嵌入矩阵
保持与entity2id相同的索引顺序，输出为pkl格式
"""

import torch
import pickle
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from config_hotpotqa import Config, create_directories
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn
from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding


class GlobalNodeEmbeddingGenerator:
    """
    全局节点嵌入生成器
    为entity2id中的所有实体生成RGCN更新后的嵌入矩阵
    """

    def __init__(self, checkpoint_path, entity_embedding_path, relation_embedding_path,
                 entity2idx_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化生成器

        Args:
            checkpoint_path: 训练好的模型检查点路径
            entity_embedding_path: 实体嵌入文件路径
            relation_embedding_path: 关系映射文件路径
            entity2idx_path: entity2id映射文件路径
            device: 运行设备
        """
        self.device = device
        print(f"使用设备: {self.device}")

        # 加载entity2id映射
        print(f"\n加载entity2id映射: {entity2idx_path}")
        with open(entity2idx_path, 'rb') as f:
            self.entity2idx = pickle.load(f)
        self.num_entities = len(self.entity2idx)
        print(f"  实体总数: {self.num_entities}")

        # 创建反向映射 idx -> entity
        self.idx2entity = {idx: entity for entity, idx in self.entity2idx.items()}

        # 加载检查点
        print(f"\n正在加载模型检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 从检查点中获取模型配置
        model_config = checkpoint.get('config', {})
        self.hidden_channels = model_config.get('hidden_channels', 128)
        self.out_channels = model_config.get('out_channels', 64)
        num_layers = model_config.get('num_layers', 3)
        dropout = model_config.get('dropout', 0.3)

        print(f"模型配置:")
        print(f"  隐藏层维度: {self.hidden_channels}")
        print(f"  输出维度: {self.out_channels}")
        print(f"  层数: {num_layers}")

        # 创建RGCN编码器
        self.encoder = ImprovedRGCNEncoderWithEmbedding(
            entity_embedding_path=entity_embedding_path,
            relation_embedding_path=relation_embedding_path,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=num_layers,
            freeze_embeddings=True,
            dropout=dropout
        ).to(device)

        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    new_key = key[8:]
                    encoder_state_dict[new_key] = value

            self.encoder.load_state_dict(encoder_state_dict)
            print("✓ 成功加载编码器权重")
        else:
            raise ValueError("检查点中未找到model_state_dict")

        # 设置为评估模式
        self.encoder.eval()
        print("✓ 模型已设置为评估模式")

        # 初始化全局嵌入矩阵（用于累积）
        # 策略：对每个实体，收集其在所有图中的嵌入，然后取平均
        self.entity_embeddings_sum = defaultdict(lambda: np.zeros(self.out_channels))
        self.entity_counts = defaultdict(int)

    def process_dataloader(self, dataloader, dataset_name='dataset',
                          use_context=True, use_gpt=False):
        """
        处理dataloader中的所有图，累积节点嵌入

        Args:
            dataloader: PyTorch DataLoader对象
            dataset_name: 数据集名称（用于日志）
            use_context: 是否使用context图（参考图）
            use_gpt: 是否使用gpt图（生成图）
        """
        print(f"\n处理 {dataset_name} ...")
        print(f"  使用context图: {'是' if use_context else '否'}")
        print(f"  使用gpt图: {'是' if use_gpt else '否'}")

        processed_entities = set()

        with torch.no_grad():
            for context_batch, gpt_batch, labels, metadata_list in tqdm(dataloader, desc=f'处理{dataset_name}'):
                if context_batch is None:
                    continue

                # 处理context图
                if use_context:
                    context_batch = context_batch.to(self.device)
                    _, context_node_emb = self.encoder(
                        context_batch, return_node_embeddings=True
                    )
                    context_node_emb = context_node_emb.cpu().numpy()
                    context_node_ids = context_batch.node_ids.cpu().numpy()

                    # 累积每个节点的嵌入
                    for node_id, node_emb in zip(context_node_ids, context_node_emb):
                        node_id = int(node_id)
                        self.entity_embeddings_sum[node_id] += node_emb
                        self.entity_counts[node_id] += 1
                        processed_entities.add(node_id)

                # 处理gpt图
                if use_gpt:
                    gpt_batch = gpt_batch.to(self.device)
                    _, gpt_node_emb = self.encoder(
                        gpt_batch, return_node_embeddings=True
                    )
                    gpt_node_emb = gpt_node_emb.cpu().numpy()
                    gpt_node_ids = gpt_batch.node_ids.cpu().numpy()

                    for node_id, node_emb in zip(gpt_node_ids, gpt_node_emb):
                        node_id = int(node_id)
                        self.entity_embeddings_sum[node_id] += node_emb
                        self.entity_counts[node_id] += 1
                        processed_entities.add(node_id)

        print(f"✓ {dataset_name} 处理完成")
        print(f"  处理的唯一实体数: {len(processed_entities)}")
        return processed_entities

    def generate_global_embedding_matrix(self, fallback_strategy='original'):
        """
        生成全局嵌入矩阵

        Args:
            fallback_strategy: 对于未在数据集中出现的实体的处理策略
                - 'original': 使用原始嵌入（从encoder的embedding层获取）
                - 'zero': 使用零向量
                - 'mean': 使用所有已处理实体的平均嵌入

        Returns:
            embedding_matrix: numpy数组，shape为[num_entities, out_channels]
        """
        print("\n" + "="*60)
        print("生成全局嵌入矩阵")
        print("="*60)

        # 初始化嵌入矩阵
        embedding_matrix = np.zeros((self.num_entities, self.out_channels), dtype=np.float32)

        # 计算每个实体的平均嵌入
        processed_count = 0
        for entity_id in tqdm(range(self.num_entities), desc="计算平均嵌入"):
            if entity_id in self.entity_counts:
                # 已处理的实体：取平均
                count = self.entity_counts[entity_id]
                embedding_matrix[entity_id] = self.entity_embeddings_sum[entity_id] / count
                processed_count += 1

        # 处理未出现的实体
        unprocessed_ids = [i for i in range(self.num_entities) if i not in self.entity_counts]

        if len(unprocessed_ids) > 0:
            print(f"\n处理 {len(unprocessed_ids)} 个未在数据集中出现的实体...")

            if fallback_strategy == 'original':
                print("  策略: 使用原始嵌入（通过RGCN处理单个节点）")
                # 对于未出现的实体，通过RGCN处理其原始嵌入
                with torch.no_grad():
                    # 批量处理以提高效率
                    batch_size = 1000
                    for i in tqdm(range(0, len(unprocessed_ids), batch_size), desc="处理未出现实体"):
                        batch_ids = unprocessed_ids[i:i+batch_size]

                        # 创建只包含单个节点的虚拟图（无边）
                        for entity_id in batch_ids:
                            # 获取原始嵌入并通过第一层RGCN（由于无边，相当于简单变换）
                            node_ids_tensor = torch.LongTensor([entity_id]).to(self.device)
                            initial_emb = self.encoder.entity_embedding(node_ids_tensor)

                            # 通过所有RGCN层（无边情况下，等同于MLP变换）
                            x = initial_emb
                            for i_layer, conv in enumerate(self.encoder.convs):
                                # 对于无边的情况，使用self-loop
                                edge_index = torch.LongTensor([[0], [0]]).to(self.device)
                                edge_type = torch.LongTensor([0]).to(self.device)

                                x = conv(x, edge_index, edge_type)
                                x = self.encoder.batch_norms[i_layer](x)

                                if i_layer < len(self.encoder.convs) - 1:
                                    x = torch.relu(x)

                            embedding_matrix[entity_id] = x.cpu().numpy()[0]

            elif fallback_strategy == 'zero':
                print("  策略: 使用零向量")
                # 已经初始化为零，无需额外操作
                pass

            elif fallback_strategy == 'mean':
                print("  策略: 使用已处理实体的平均嵌入")
                # 计算所有已处理实体的平均嵌入
                processed_embeddings = [embedding_matrix[i] for i in range(self.num_entities)
                                       if i in self.entity_counts]
                mean_embedding = np.mean(processed_embeddings, axis=0)

                for entity_id in unprocessed_ids:
                    embedding_matrix[entity_id] = mean_embedding

            else:
                raise ValueError(f"未知的fallback策略: {fallback_strategy}")

        print(f"\n✓ 全局嵌入矩阵生成完成")
        print(f"  矩阵形状: {embedding_matrix.shape}")
        print(f"  已处理实体: {processed_count} ({processed_count/self.num_entities*100:.2f}%)")
        print(f"  未处理实体: {len(unprocessed_ids)} ({len(unprocessed_ids)/self.num_entities*100:.2f}%)")

        return embedding_matrix

    def save_embeddings(self, embedding_matrix, output_path, save_formats=['pkl', 'npy']):
        """
        保存嵌入矩阵

        Args:
            embedding_matrix: 嵌入矩阵 [num_entities, out_channels]
            output_path: 输出路径（不含扩展名）
            save_formats: 保存格式列表，可选 'pkl', 'npy', 'pt'
        """
        print("\n" + "="*60)
        print("保存嵌入矩阵")
        print("="*60)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存为pickle格式（推荐，包含元数据）
        if 'pkl' in save_formats:
            pkl_path = output_path + '.pkl'
            embedding_data = {
                'embeddings': embedding_matrix,
                'num_entities': self.num_entities,
                'embedding_dim': self.out_channels,
                'entity2idx': self.entity2idx,  # 保存映射以便验证
                'description': 'RGCN更新后的节点嵌入矩阵，索引对应entity2id'
            }
            with open(pkl_path, 'wb') as f:
                pickle.dump(embedding_data, f)
            print(f"✓ PKL格式已保存: {pkl_path}")

        # 保存为numpy格式
        if 'npy' in save_formats:
            npy_path = output_path + '.npy'
            np.save(npy_path, embedding_matrix)
            print(f"✓ NPY格式已保存: {npy_path}")

        # 保存为PyTorch格式
        if 'pt' in save_formats:
            pt_path = output_path + '.pt'
            torch.save(torch.FloatTensor(embedding_matrix), pt_path)
            print(f"✓ PT格式已保存: {pt_path}")

        # 保存统计信息
        stats_path = output_path + '_stats.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("RGCN更新后节点嵌入统计\n")
            f.write("="*60 + "\n")
            f.write(f"实体总数: {self.num_entities}\n")
            f.write(f"嵌入维度: {self.out_channels}\n")
            f.write(f"矩阵形状: {embedding_matrix.shape}\n")
            f.write(f"数据类型: {embedding_matrix.dtype}\n")
            f.write(f"\n嵌入统计:\n")
            f.write(f"  最小值: {embedding_matrix.min():.6f}\n")
            f.write(f"  最大值: {embedding_matrix.max():.6f}\n")
            f.write(f"  均值: {embedding_matrix.mean():.6f}\n")
            f.write(f"  标准差: {embedding_matrix.std():.6f}\n")
            f.write(f"\n已处理实体数: {len(self.entity_counts)}\n")
            f.write(f"未处理实体数: {self.num_entities - len(self.entity_counts)}\n")
        print(f"✓ 统计信息已保存: {stats_path}")


def main():
    """主函数"""
    print("="*60)
    print("RGCN全局节点嵌入生成器")
    print("="*60)

    # 配置
    config_dict = Config.get_config_dict()

    # 检查模型文件
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 错误: 模型文件不存在")
        print(f"期望路径: {checkpoint_path}")
        print(f"\n请先训练模型: python train_rgcn_hotpotqa.py")
        return

    # 1. 加载数据集
    print("\n[步骤 1] 加载数据集")
    print("="*60)
    dataset = HotpotQAGraphDataset(
        data_path=config_dict['data_path'],
        entity_mapping_path=config_dict['entity_mapping_path'],
        relation_mapping_path=config_dict['relation_mapping_path'],
        max_samples=config_dict.get('max_samples')
    )

    # 2. 创建DataLoader（使用全部数据）
    print("\n[步骤 2] 创建DataLoader")
    print("="*60)
    dataloader = DataLoader(
        dataset,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config_dict.get('num_workers', 0)
    )
    print(f"总样本数: {len(dataset)}")

    # 3. 初始化生成器
    print("\n[步骤 3] 初始化嵌入生成器")
    print("="*60)
    generator = GlobalNodeEmbeddingGenerator(
        checkpoint_path=checkpoint_path,
        entity_embedding_path=config_dict['entity_embedding_path'],
        relation_embedding_path=config_dict['relation_embedding_path'],
        entity2idx_path=config_dict['entity_mapping_path']
    )

    # 4. 处理数据集，累积节点嵌入
    print("\n[步骤 4] 处理数据集")
    print("="*60)
    processed_entities = generator.process_dataloader(
        dataloader,
        dataset_name='完整数据集',
        use_context=True,   # 使用context图（参考图）
        use_gpt=False       # 可选：是否也使用gpt图
    )

    # 5. 生成全局嵌入矩阵
    print("\n[步骤 5] 生成全局嵌入矩阵")
    print("="*60)
    embedding_matrix = generator.generate_global_embedding_matrix(
        fallback_strategy='original'  # 对未出现的实体使用原始嵌入
        # 可选: 'zero', 'mean'
    )

    # 6. 保存嵌入矩阵
    print("\n[步骤 6] 保存嵌入矩阵")
    print("="*60)
    output_path = os.path.join(
        Config.HYBRID_EMBEDDINGS_DIR,
        'entity_embeddings_rgcn_updated'
    )
    generator.save_embeddings(
        embedding_matrix,
        output_path,
        save_formats=['pkl', 'npy']  # 保存为pkl和npy格式
    )

    # 7. 完成
    print("\n" + "="*60)
    print("✓ 全局节点嵌入生成完成！")
    print("="*60)
    print(f"\n输出文件:")
    print(f"  {output_path}.pkl")
    print(f"  {output_path}.npy")
    print(f"  {output_path}_stats.txt")

    # 8. 使用说明
    print("\n" + "="*60)
    print("如何使用更新后的嵌入:")
    print("="*60)
    print("""
# 方法1: 加载PKL格式（推荐，包含完整信息）
import pickle

with open('hybrid_embeddings/entity_embeddings_rgcn_updated.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings']  # [num_entities, out_channels]
entity2idx = data['entity2idx']  # 实体到索引的映射

# 查询特定实体的嵌入
entity_name = "Barack Obama"
if entity_name in entity2idx:
    entity_id = entity2idx[entity_name]
    entity_embedding = embeddings[entity_id]
    print(f"{entity_name}的嵌入: {entity_embedding.shape}")

# 方法2: 加载NPY格式（只有矩阵数据）
import numpy as np

embeddings = np.load('hybrid_embeddings/entity_embeddings_rgcn_updated.npy')
print(f"嵌入矩阵形状: {embeddings.shape}")

# 注意: 使用NPY格式时，需要单独加载entity2idx映射
with open('hybrid_embeddings/entity2idx.pkl', 'rb') as f:
    entity2idx = pickle.load(f)
    """)

    print("\n✓ 可以直接使用相同的entity2id文件进行索引！")


if __name__ == '__main__':
    main()