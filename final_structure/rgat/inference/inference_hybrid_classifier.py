"""
使用训练好的混合分类器进行幻觉检测推理
"""
import torch
import json
import numpy as np
from tqdm import tqdm
import os
import sys
from datetime import datetime

# 导入模型和数据加载器
from framework.hybrid_graph_text_classifier import load_hybrid_classifier
from dataloader.data_loader_hotpotqa import HotpotQAGraphDataset  # 修正类名


def inference_on_samples(checkpoint_path,
                         data_path,
                         entity_mapping_path,
                         relation_mapping_path,
                         num_samples=50,
                         output_path=None,
                         device='cuda'):
    """
    对数据集中的样本进行幻觉检测推理

    Args:
        checkpoint_path: 训练好的模型checkpoint路径
        data_path: 数据集路径
        entity_mapping_path: 实体映射路径
        relation_mapping_path: 关系映射路径
        num_samples: 推理的样本数量
        output_path: 结果保存路径
        device: 设备
    """
    print(f"\n{'='*80}")
    print("混合分类器推理 - 幻觉检测")
    print(f"{'='*80}\n")

    # 1. 加载模型
    print(f"📦 加载模型...")
    model, config = load_hybrid_classifier(checkpoint_path, device=device)
    model.eval()
    print(f"✓ 模型加载成功\n")

    # 2. 加载原始JSON数据（用于获取文本）
    print(f"📊 加载原始数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"✓ 原始数据加载成功: {len(raw_data)} 条记录\n")

    # 3. 加载图数据集
    print(f"📊 加载图数据集...")
    dataset = HotpotQAGraphDataset(
        data_path=data_path,  # 修正参数名
        entity_mapping_path=entity_mapping_path,
        relation_mapping_path=relation_mapping_path
    )
    print(f"✓ 图数据集加载成功: {len(dataset)} 条样本\n")

    # 4. 创建数据索引映射（原始数据 _id -> 文本）
    print(f"🔗 创建数据映射...")
    id_to_text = {}
    for item in raw_data:
        _id = item.get('_id', '')
        # 获取 GPT 生成的文本
        gpt_text = item.get('gpt_sentence', '') or item.get('answer', '') or ''
        id_to_text[_id] = gpt_text
    print(f"✓ 映射创建成功: {len(id_to_text)} 条记录\n")

    # 5. 随机选择样本
    if num_samples > len(dataset):
        num_samples = len(dataset)

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    print(f"🎲 随机选择 {num_samples} 条样本进行推理\n")

    # 6. 推理
    results = []
    label_names = {0: '幻觉', 1: '事实'}

    print(f"{'='*80}")
    print("开始推理...")
    print(f"{'='*80}\n")

    skipped = 0
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="推理进度")):
            try:
                # 获取样本
                context_graph, gpt_graph, label, metadata = dataset[idx]

                # 获取对应的文本
                sample_id = metadata.get('_id', '')
                gpt_text = id_to_text.get(sample_id, '')

                # 如果没有文本，跳过
                if not gpt_text:
                    skipped += 1
                    continue

                # 将图数据移到设备
                context_graph = context_graph.to(device)
                gpt_graph = gpt_graph.to(device)

                # 预测
                from torch_geometric.data import Batch
                context_batch = Batch.from_data_list([context_graph])
                gpt_batch = Batch.from_data_list([gpt_graph])

                logits = model(context_batch, gpt_batch, [gpt_text])
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()

                # 获取概率
                prob_hallucination = probabilities[0, 0].item()
                prob_factual = probabilities[0, 1].item()

                # 记录结果
                result = {
                    'sample_id': sample_id,
                    'dataset_index': int(idx),
                    'question': metadata.get('question', 'N/A'),
                    'gpt_response': gpt_text[:300] + '...' if len(gpt_text) > 300 else gpt_text,
                    'true_label': label_names[label],
                    'predicted_label': label_names[prediction],
                    'is_correct': (prediction == label),
                    'confidence': max(prob_hallucination, prob_factual),
                    'probabilities': {
                        '幻觉': round(prob_hallucination, 4),
                        '事实': round(prob_factual, 4)
                    },
                    'graph_info': {
                        'context_nodes': context_graph.num_nodes,
                        'context_edges': context_graph.edge_index.shape[1],
                        'gpt_nodes': gpt_graph.num_nodes,
                        'gpt_edges': gpt_graph.edge_index.shape[1]
                    }
                }
                results.append(result)

            except Exception as e:
                print(f"\n⚠️ 样本 {idx} 处理失败: {e}")
                skipped += 1
                continue

    if skipped > 0:
        print(f"\n⚠️ 跳过了 {skipped} 个样本")

    if len(results) == 0:
        print("\n❌ 没有成功处理的样本！")
        return []

    # 7. 统计结果
    print(f"\n{'='*80}")
    print("推理结果统计")
    print(f"{'='*80}\n")

    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total * 100

    true_hallucinations = sum(1 for r in results if r['true_label'] == '幻觉')
    true_factuals = sum(1 for r in results if r['true_label'] == '事实')

    pred_hallucinations = sum(1 for r in results if r['predicted_label'] == '幻觉')
    pred_factuals = sum(1 for r in results if r['predicted_label'] == '事实')

    # 计算混淆矩阵
    tp = sum(1 for r in results if r['true_label'] == '事实' and r['predicted_label'] == '事实')
    tn = sum(1 for r in results if r['true_label'] == '幻觉' and r['predicted_label'] == '幻觉')
    fp = sum(1 for r in results if r['true_label'] == '幻觉' and r['predicted_label'] == '事实')
    fn = sum(1 for r in results if r['true_label'] == '事实' and r['predicted_label'] == '幻觉')

    print(f"总样本数: {total}")
    print(f"预测正确: {correct} ({accuracy:.2f}%)")
    print(f"\n真实分布:")
    print(f"  - 幻觉: {true_hallucinations} ({true_hallucinations/total*100:.1f}%)")
    print(f"  - 事实: {true_factuals} ({true_factuals/total*100:.1f}%)")
    print(f"\n预测分布:")
    print(f"  - 幻觉: {pred_hallucinations} ({pred_hallucinations/total*100:.1f}%)")
    print(f"  - 事实: {pred_factuals} ({pred_factuals/total*100:.1f}%)")
    print(f"\n混淆矩阵:")
    print(f"              预测幻觉  预测事实")
    print(f"真实幻觉:      {tn:4d}      {fp:4d}")
    print(f"真实事实:      {fn:4d}      {tp:4d}")

    # 计算更多指标
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"\n精确率 (Precision): {precision:.2%}")
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"召回率 (Recall): {recall:.2%}")
    if tp + fp > 0 and tp + fn > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1分数: {f1:.4f}")

    # 8. 显示部分样本
    print(f"\n{'='*80}")
    print("样本展示（前10个）")
    print(f"{'='*80}\n")

    for i, result in enumerate(results[:10], 1):
        correct_mark = '✓' if result['is_correct'] else '✗'
        print(f"【样本 {i}】{correct_mark}")
        print(f"问题: {result['question'][:80]}...")
        print(f"GPT回答: {result['gpt_response'][:120]}...")
        print(f"真实标签: {result['true_label']} | 预测: {result['predicted_label']}")
        print(f"置信度: {result['confidence']:.2%} | "
              f"概率 [幻觉={result['probabilities']['幻觉']:.2%}, "
              f"事实={result['probabilities']['事实']:.2%}]")
        print(f"图信息: Context({result['graph_info']['context_nodes']}节点, "
              f"{result['graph_info']['context_edges']}边) | "
              f"GPT({result['graph_info']['gpt_nodes']}节点, "
              f"{result['graph_info']['gpt_edges']}边)")
        print()

    # 9. 保存结果
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'inference_results_{timestamp}.json'

    # 添加统计信息
    output_data = {
        'metadata': {
            'checkpoint': checkpoint_path,
            'num_samples': total,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        },
        'statistics': {
            'true_distribution': {
                '幻觉': true_hallucinations,
                '事实': true_factuals
            },
            'predicted_distribution': {
                '幻觉': pred_hallucinations,
                '事实': pred_factuals
            },
            'confusion_matrix': {
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
            }
        },
        'samples': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"{'='*80}")
    print(f"✓ 结果已保存到: {output_path}")
    print(f"{'='*80}\n")

    return results


def main():
    """主函数"""
    # 配置参数

    # 🔥 修改这里：使用你最新训练的模型路径
    checkpoint_path = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/rgat_output/hybrid_20260119_145940(目前去最高)/checkpoints/best_model.pth'

    # 数据路径
    data_path = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/hotpot_dev_merged_triples_clustered.json'

    # 映射路径
    embeddings_dir = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/final_hybrid_embeddings'
    entity_mapping_path = os.path.join(embeddings_dir, 'entity2idx.pkl')
    relation_mapping_path = os.path.join(embeddings_dir, 'relation2idx.pkl')

    # 输出路径
    output_dir = '/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/rgat_output'
    os.makedirs(output_dir, exist_ok=True)

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 推理
    results = inference_on_samples(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        entity_mapping_path=entity_mapping_path,
        relation_mapping_path=relation_mapping_path,
        num_samples=50,
        output_path=os.path.join(output_dir, 'inference_results.json'),
        device=device
    )

    print(f"\n✓ 推理完成！共处理 {len(results)} 个样本")


if __name__ == '__main__':
    main()