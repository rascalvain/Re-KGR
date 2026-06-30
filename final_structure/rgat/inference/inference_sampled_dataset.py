"""
随机抽样50条数据并添加幻觉检测结果
1. hallucination_prediction: 整体检测结果（0=事实，1=幻觉）
2. triple_hallucination_labels: 每个三元组的幻觉等级（0=事实，1=可能幻觉，2=幻觉）
"""
import torch
import json
import numpy as np
from tqdm import tqdm
import os
import pickle
from datetime import datetime
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

# 导入模型
from framework.hybrid_graph_text_classifier import load_hybrid_classifier
from dataloader.data_loader_hotpotqa import HotpotQAGraphDataset


class DatasetEnhancer:
    """数据集增强器：添加幻觉检测结果"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        初始化
        Args:
            checkpoint_path: 训练好的模型路径
            device: 设备
        """
        self.device = device

        # 加载模型
        print(f"📦 加载混合分类器...")
        self.model, self.config = load_hybrid_classifier(checkpoint_path, device=device)
        self.model.eval()
        self.graph_encoder = self.model.graph_encoder
        print(f"✓ 模型加载成功\n")

        # 三元组分类阈值（根据幻觉分数）
        self.triple_thresholds = {
            'hallucination': 0.6,  # >= 0.6: 幻觉 (标签2)
            'possible_hallucination': 0.4  # 0.4-0.6: 可能幻觉 (标签1)
            # < 0.4: 事实 (标签0)
        }

    def parse_triple(self, triple_str):
        """解析三元组字符串"""
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

        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        else:
            return None, None, None

    def triple_to_graph(self, triple_obj, entity2id, relation2id):
        """将单个三元组转换为图"""
        if 'triple' not in triple_obj:
            return None

        head, relation, tail = self.parse_triple(triple_obj['triple'])
        if not head or not relation or not tail:
            return None

        head_id = entity2id.get(head, 0)
        tail_id = entity2id.get(tail, 0)
        relation_id = relation2id.get(relation, 0)

        node_ids = torch.LongTensor([head_id, tail_id])
        edge_index = torch.LongTensor([[0], [1]])
        edge_type = torch.LongTensor([relation_id])

        data = Data(
            node_ids=node_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=2
        )

        return data

    def compute_triple_similarity(self, context_graph, triple_graph):
        """计算三元组与context图的相似度"""
        with torch.no_grad():
            context_batch = Batch.from_data_list([context_graph]).to(self.device)
            h_context = self.graph_encoder(context_batch)

            triple_batch = Batch.from_data_list([triple_graph]).to(self.device)
            h_triple = self.graph_encoder(triple_batch)

            similarity = F.cosine_similarity(h_context, h_triple, dim=-1).item()

        return similarity

    def classify_triple(self, similarity):
        """
        根据相似度对三元组分类
        Returns:
            label: 0=事实, 1=可能幻觉, 2=幻觉
        """
        hallucination_score = (1 - similarity) / 2

        if hallucination_score >= self.triple_thresholds['hallucination']:
            return 2  # 幻觉
        elif hallucination_score >= self.triple_thresholds['possible_hallucination']:
            return 1  # 可能幻觉
        else:
            return 0  # 事实

    def predict_sample(self, context_graph, gpt_graph, gpt_text):
        """
        预测样本的整体幻觉标签
        Returns:
            prediction: 0=事实, 1=幻觉
            confidence: 置信度
        """
        with torch.no_grad():
            context_batch = Batch.from_data_list([context_graph]).to(self.device)
            gpt_batch = Batch.from_data_list([gpt_graph]).to(self.device)

            logits = self.model(context_batch, gpt_batch, [gpt_text])
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, prediction].item()

        return prediction, confidence


def enhance_sampled_dataset(checkpoint_path,
                            input_json_path,
                            output_json_path,
                            entity_mapping_path,
                            relation_mapping_path,
                            num_samples=50,
                            hallucination_ratio=0.6,
                            device='cuda'):
    """
    随机抽样并增强数据集

    Args:
        checkpoint_path: 模型路径
        input_json_path: 输入JSON路径
        output_json_path: 输出JSON路径
        entity_mapping_path: 实体映射路径
        relation_mapping_path: 关系映射路径
        num_samples: 抽样数量
        device: 设备
    """
    """
    随机抽样并增强数据集（按generation_label分层抽样）

    Args:
        checkpoint_path: 模型路径
        input_json_path: 输入JSON路径
        output_json_path: 输出JSON路径
        entity_mapping_path: 实体映射路径
        relation_mapping_path: 关系映射路径
        num_samples: 抽样数量
        hallucination_ratio: 幻觉样本比例（默认0.6，即6:4）
        device: 设备
    """
    print(f"\n{'=' * 80}")
    print(f"分层抽样 {num_samples} 条数据并添加幻觉检测结果")
    print(f"  - 幻觉样本比例: {hallucination_ratio * 100:.0f}%")
    print(f"  - 事实样本比例: {(1 - hallucination_ratio) * 100:.0f}%")
    print(f"{'=' * 80}\n")

    # 1. 初始化增强器
    enhancer = DatasetEnhancer(checkpoint_path, device=device)

    # 2. 加载原始数据
    print(f"📊 加载原始数据...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    print(f"✓ 加载完成: {len(full_data)} 条记录\n")

    # 3. 🔥 按generation_label分层抽样
    print(f"🎲 按 generation_label 分层抽样...")

    # 3.1 分组：hallucination 和 correct
    hallucination_indices = []
    correct_indices = []

    for idx, item in enumerate(full_data):
        label = item.get('generation_label', '')
        if label == 'hallucination':
            hallucination_indices.append(idx)
        elif label == 'correct':
            correct_indices.append(idx)

    print(f"  数据分布:")
    print(f"    - hallucination: {len(hallucination_indices)} 条")
    print(f"    - correct: {len(correct_indices)} 条")

    # 3.2 计算每组抽样数量
    num_hallucination = int(num_samples * hallucination_ratio)
    num_correct = num_samples - num_hallucination

    print(f"\n  抽样目标:")
    print(f"    - hallucination: {num_hallucination} 条")
    print(f"    - correct: {num_correct} 条")

    # 3.3 检查是否有足够样本
    if len(hallucination_indices) < num_hallucination:
        print(f"  ⚠️ 警告: hallucination样本不足，调整为 {len(hallucination_indices)} 条")
        num_hallucination = len(hallucination_indices)
        num_correct = min(num_samples - num_hallucination, len(correct_indices))

    if len(correct_indices) < num_correct:
        print(f"  ⚠️ 警告: correct样本不足，调整为 {len(correct_indices)} 条")
        num_correct = len(correct_indices)
        num_hallucination = min(num_samples - num_correct, len(hallucination_indices))

    # 3.4 随机抽样
    np.random.seed(42)  # 设置随机种子以便复现

    sampled_hall_indices = np.random.choice(
        hallucination_indices,
        num_hallucination,
        replace=False
    ).tolist()

    sampled_corr_indices = np.random.choice(
        correct_indices,
        num_correct,
        replace=False
    ).tolist()

    # 3.5 合并并排序
    sampled_indices = sorted(sampled_hall_indices + sampled_corr_indices)
    sampled_data = [full_data[i] for i in sampled_indices]

    print(f"\n✓ 抽样完成，共 {len(sampled_data)} 条记录")
    print(f"    - hallucination: {num_hallucination} 条")
    print(f"    - correct: {num_correct} 条")
    print(
        f"    - 实际比例: {num_hallucination / len(sampled_data) * 100:.1f}% : {num_correct / len(sampled_data) * 100:.1f}%")
    print(f"  前10个索引: {sampled_indices[:10]}...")
    print()

    # 4. 加载映射
    print(f"📊 加载实体和关系映射...")
    with open(entity_mapping_path, 'rb') as f:
        entity2id = pickle.load(f)
    with open(relation_mapping_path, 'rb') as f:
        relation2id = pickle.load(f)
    print(f"✓ 映射加载完成\n")

    # 5. 加载图数据集
    print(f"📊 加载图数据集...")
    dataset = HotpotQAGraphDataset(
        data_path=input_json_path,
        entity_mapping_path=entity_mapping_path,
        relation_mapping_path=relation_mapping_path
    )
    print(f"✓ 图数据集加载完成\n")

    # 6. 创建ID到索引的映射
    id_to_dataset_idx = {}
    for idx in range(len(dataset)):
        _, _, _, metadata = dataset[idx]
        sample_id = metadata.get('_id', '')
        if sample_id:
            id_to_dataset_idx[sample_id] = idx

    # 7. 处理抽样的记录
    print(f"{'=' * 80}")
    print("开始处理抽样数据...")
    print(f"{'=' * 80}\n")

    enhanced_data = []
    processed_count = 0
    skipped_count = 0

    for item in tqdm(sampled_data, desc="处理进度"):
        try:
            # 获取样本ID
            sample_id = item.get('_id', '')

            # 检查是否有必要的字段
            if not sample_id or sample_id not in id_to_dataset_idx:
                # 无法处理，标记为未处理
                enhanced_item = item.copy()
                enhanced_item['hallucination_prediction'] = -1
                enhanced_item['triple_hallucination_labels'] = []
                enhanced_data.append(enhanced_item)
                skipped_count += 1
                continue

            # 从数据集获取图数据
            dataset_idx = id_to_dataset_idx[sample_id]
            context_graph, gpt_graph, true_label, metadata = dataset[dataset_idx]

            # 获取GPT生成的文本
            gpt_text = item.get('gpt_sentence', '') or item.get('answer', '') or ''

            # 获取GPT三元组
            gpt_triples = item.get('gpt_sentence_triples', [])

            # 1️⃣ 预测整体幻觉标签
            if gpt_text and context_graph is not None and gpt_graph is not None:
                prediction, confidence = enhancer.predict_sample(
                    context_graph, gpt_graph, gpt_text
                )
            else:
                prediction = -1
                confidence = 0.0

            # 2️⃣ 预测每个三元组的幻觉标签
            triple_labels = []

            if context_graph is not None and gpt_triples:
                for triple_obj in gpt_triples:
                    triple_graph = enhancer.triple_to_graph(
                        triple_obj, entity2id, relation2id
                    )

                    if triple_graph is None:
                        triple_labels.append(-1)
                        continue

                    similarity = enhancer.compute_triple_similarity(
                        context_graph.to(device), triple_graph
                    )
                    label = enhancer.classify_triple(similarity)
                    triple_labels.append(label)

            # 创建增强后的记录
            enhanced_item = item.copy()
            enhanced_item['hallucination_prediction'] = prediction
            enhanced_item['triple_hallucination_labels'] = triple_labels

            # 可选：添加统计信息
            enhanced_item['detection_metadata'] = {
                'prediction_confidence': round(confidence, 4) if confidence > 0 else None,
                'num_triples': len(triple_labels),
                'triple_stats': {
                    'factual': triple_labels.count(0),
                    'possible_hallucination': triple_labels.count(1),
                    'hallucination': triple_labels.count(2),
                    'unknown': triple_labels.count(-1)
                }
            }

            enhanced_data.append(enhanced_item)
            processed_count += 1

        except Exception as e:
            print(f"\n⚠️ 处理样本 {item.get('_id', 'unknown')} 时出错: {e}")
            enhanced_item = item.copy()
            enhanced_item['hallucination_prediction'] = -1
            enhanced_item['triple_hallucination_labels'] = []
            enhanced_data.append(enhanced_item)
            skipped_count += 1

    # 8. 保存结果
    print(f"\n{'=' * 80}")
    print("保存增强后的数据...")
    print(f"{'=' * 80}\n")

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    # 9. 统计
    print(f"✓ 处理完成！")
    print(f"\n统计信息:")
    print(f"  输出记录数: {len(enhanced_data)}")
    print(f"  成功处理: {processed_count}")
    print(f"  跳过/失败: {skipped_count}")

    # 整体预测统计
    predictions = [item['hallucination_prediction'] for item in enhanced_data]
    pred_factual = predictions.count(0)
    pred_hallucination = predictions.count(1)
    pred_unknown = predictions.count(-1)

    print(f"\n整体检测结果分布:")
    print(f"  ✅ 事实 (0): {pred_factual} ({pred_factual / len(enhanced_data) * 100:.1f}%)")
    print(f"  ❌ 幻觉 (1): {pred_hallucination} ({pred_hallucination / len(enhanced_data) * 100:.1f}%)")
    print(f"  ❓ 未检测 (-1): {pred_unknown} ({pred_unknown / len(enhanced_data) * 100:.1f}%)")

    # 三元组统计
    all_triple_labels = []
    for item in enhanced_data:
        all_triple_labels.extend(item.get('triple_hallucination_labels', []))

    if all_triple_labels:
        total_triples = len(all_triple_labels)
        print(f"\n三元组检测结果分布:")
        print(f"  总三元组数: {total_triples}")
        print(f"  ✅ 事实 (0): {all_triple_labels.count(0)} ({all_triple_labels.count(0) / total_triples * 100:.1f}%)")
        print(
            f"  ⚠️  可能幻觉 (1): {all_triple_labels.count(1)} ({all_triple_labels.count(1) / total_triples * 100:.1f}%)")
        print(f"  ❌ 幻觉 (2): {all_triple_labels.count(2)} ({all_triple_labels.count(2) / total_triples * 100:.1f}%)")
        if all_triple_labels.count(-1) > 0:
            print(f"  ❓ 未检测 (-1): {all_triple_labels.count(-1)}")

    # 显示几个示例
    print(f"\n{'=' * 80}")
    print("样本示例（前3个）")
    print(f"{'=' * 80}\n")

    for i, item in enumerate(enhanced_data[:3], 1):
        pred_label = {0: '事实', 1: '幻觉', -1: '未知'}
        triple_label = {0: '事实', 1: '可能幻觉', 2: '幻觉', -1: '未知'}

        print(f"【样本 {i}】")
        print(f"ID: {item.get('_id', 'N/A')}")
        print(f"问题: {item.get('question', 'N/A')[:80]}...")
        print(f"整体预测: {pred_label.get(item['hallucination_prediction'], '未知')}")

        triple_labels = item.get('triple_hallucination_labels', [])
        if triple_labels:
            print(f"三元组标签: {triple_labels}")
            print(f"三元组分布: ", end='')
            print(f"事实={triple_labels.count(0)}, ", end='')
            print(f"可能幻觉={triple_labels.count(1)}, ", end='')
            print(f"幻觉={triple_labels.count(2)}")
        print()

    print(f"{'=' * 80}")
    print(f"✓ 输出文件: {output_json_path}")
    print(f"{'=' * 80}\n")

    return enhanced_data


def main():
    """主函数"""

    # 🔥 配置路径
    checkpoint_path = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/rgat_output/hybrid_20260119_145940(目前去最高)/checkpoints/best_model.pth'

    input_json_path = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/hotpot_dev_merged_triples_clustered.json'

    # 输出路径
    output_dir = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/rgat_output'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_json_path = os.path.join(
        output_dir,
        f'sampled_50_with_detection_{timestamp}.json'
    )

    # 映射路径
    embeddings_dir = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/final_hybrid_embeddings'
    entity_mapping_path = os.path.join(embeddings_dir, 'entity2idx.pkl')
    relation_mapping_path = os.path.join(embeddings_dir, 'relation2idx.pkl')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")

    # 🎲 运行抽样和增强（只输出50条）
    enhanced_data = enhance_sampled_dataset(
        checkpoint_path=checkpoint_path,
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        entity_mapping_path=entity_mapping_path,
        relation_mapping_path=relation_mapping_path,
        num_samples=50,  # 🔥 抽样50条
        device=device
    )

    print(f"\n🎉 完成！输出文件包含 {len(enhanced_data)} 条记录")


if __name__ == '__main__':
    main()