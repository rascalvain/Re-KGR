"""
简单的推理示例 - 检测单条数据
"""

import torch
from config_hotpotqa import Config
from data_loader_hotpotqa import HotpotQAGraphDataset
from inference_hotpotqa import HallucinationDetector


def detect_single_sample():
    """检测单个样本"""
    print("="*60)
    print("单样本幻觉检测示例")
    print("="*60)
    
    # 配置
    config_dict = Config.get_config_dict()
    model_path = './rgcn_output/checkpoints/best_model.pth'
    
    # 加载数据
    dataset = HotpotQAGraphDataset(
        config_dict['data_path'],
        config_dict['entity_mapping_path'],
        config_dict['relation_mapping_path'],
        max_samples=10  # 只加载10个样本做演示
    )
    
    # 创建检测器
    detector = HallucinationDetector(model_path, config_dict)
    
    # 检测第一个样本
    context_graph, gpt_graph, label, metadata = dataset[0]
    
    similarity = detector.predict_similarity(context_graph, gpt_graph)
    
    # 判断（阈值0.7）
    threshold = 0.7
    is_hallucination = similarity < threshold
    
    print(f"\n样本信息:")
    print(f"  ID: {metadata['_id']}")
    print(f"  问题: {metadata['question'][:80]}...")
    print(f"  答案: {metadata['answer']}")
    print(f"\n图统计:")
    print(f"  Context三元组数: {metadata['num_context_triples']}")
    print(f"  GPT三元组数: {metadata['num_gpt_triples']}")
    print(f"  Context图节点数: {context_graph.num_nodes}")
    print(f"  GPT图节点数: {gpt_graph.num_nodes}")
    print(f"\n检测结果:")
    print(f"  相似度: {similarity:.4f}")
    print(f"  阈值: {threshold:.4f}")
    print(f"  判断: {'🔴 幻觉' if is_hallucination else '✅ 非幻觉'}")
    print(f"  置信度: {abs(similarity - threshold):.4f}")


def detect_batch():
    """批量检测示例"""
    print("\n" + "="*60)
    print("批量幻觉检测示例")
    print("="*60)
    
    from torch.utils.data import DataLoader
    from data_loader_hotpotqa import collate_fn
    
    # 配置
    config_dict = Config.get_config_dict()
    model_path = './rgcn_output/checkpoints/best_model.pth'
    
    # 加载数据
    dataset = HotpotQAGraphDataset(
        config_dict['data_path'],
        config_dict['entity_mapping_path'],
        config_dict['relation_mapping_path'],
        max_samples=10
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 创建检测器
    detector = HallucinationDetector(model_path, config_dict)
    
    # 批量预测
    predictions, similarities, metadata_list = detector.predict_batch(
        dataloader, 
        threshold=0.7
    )
    
    print(f"\n批量检测结果:")
    print(f"{'序号':<4} {'相似度':<10} {'判断':<15} {'问题'}")
    print("-" * 80)
    for i, (pred, sim, meta) in enumerate(zip(predictions, similarities, metadata_list)):
        label = '✅ 非幻觉' if pred == 1 else '🔴 幻觉'
        question = meta['question'][:50] + '...' if len(meta['question']) > 50 else meta['question']
        print(f"{i+1:<4} {sim:<10.4f} {label:<15} {question}")


if __name__ == '__main__':
    # 单样本检测
    detect_single_sample()
    
    # 批量检测
    detect_batch()











