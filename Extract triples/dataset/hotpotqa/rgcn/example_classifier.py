"""
FFN分类器简单示例
演示如何使用分类器进行幻觉检测
"""

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(__file__))

from classifier_model import HallucinationClassifier
from data_loader_hotpotqa import HotpotQAGraphDataset
from config_hotpotqa import Config


def example_single_prediction():
    """单样本预测示例"""
    print("="*60)
    print("示例 1: 单样本预测")
    print("="*60)
    
    # 配置
    config_dict = Config.get_config_dict()
    config_dict['ffn_hidden_dim'] = 128
    
    # 加载模型
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print(f"请先运行: python train_classifier.py")
        return
    
    print(f"\n加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = HallucinationClassifier(
        entity_embedding_path=config_dict['entity_embedding_path'],
        relation_embedding_path=config_dict['relation_embedding_path'],
        hidden_channels=config_dict['hidden_channels'],
        out_channels=config_dict['out_channels'],
        num_layers=config_dict['num_layers'],
        freeze_embeddings=True,
        dropout=0.3,
        ffn_hidden_dim=128
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ 模型加载完成")
    print(f"  验证准确率: {checkpoint.get('val_acc', 0):.4f}")
    
    # 加载数据
    print(f"\n加载数据...")
    dataset = HotpotQAGraphDataset(
        config_dict['data_path'],
        config_dict['entity_mapping_path'],
        config_dict['relation_mapping_path'],
        max_samples=10  # 只加载10个样本
    )
    
    # 获取第一个样本
    context_graph, gpt_graph, label, metadata = dataset[0]
    
    print(f"\n样本信息:")
    print(f"  ID: {metadata.get('_id', 'N/A')}")
    print(f"  问题: {metadata.get('question', 'N/A')[:80]}...")
    print(f"  答案: {metadata.get('answer', 'N/A')}")
    print(f"  Context三元组数: {metadata.get('num_context_triples', 0)}")
    print(f"  GPT三元组数: {metadata.get('num_gpt_triples', 0)}")
    
    # 预测
    print(f"\n进行预测...")
    with torch.no_grad():
        context_graph = context_graph.to(device)
        gpt_graph = gpt_graph.to(device)
        
        # 前向传播
        logits = model(gpt_graph, context_graph)  # (response, reference)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        # 提取概率
        prob_hall = probabilities[0, 0].item()      # P(幻觉)
        prob_non_hall = probabilities[0, 1].item()  # P(非幻觉)
        pred_label = prediction.item()
    
    # 显示结果
    print(f"\n检测结果:")
    print(f"  预测: {'✅ 非幻觉' if pred_label == 1 else '⚠️ 幻觉'}")
    print(f"  P(幻觉): {prob_hall:.4f}")
    print(f"  P(非幻觉): {prob_non_hall:.4f}")
    print(f"  置信度: {max(prob_hall, prob_non_hall):.4f}")
    
    # 解释
    print(f"\n解释:")
    if pred_label == 1:
        print(f"  模型认为GPT生成的响应与KB一致，不是幻觉。")
    else:
        print(f"  模型认为GPT生成的响应与KB不一致，可能是幻觉。")


def example_batch_prediction():
    """批量预测示例"""
    print("\n" + "="*60)
    print("示例 2: 批量预测")
    print("="*60)
    
    from torch.utils.data import DataLoader
    from data_loader_hotpotqa import collate_fn
    from inference_classifier import ClassifierInference
    
    # 配置
    config_dict = Config.get_config_dict()
    config_dict['ffn_hidden_dim'] = 128
    
    # 模型路径
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载数据
    print(f"\n加载数据...")
    dataset = HotpotQAGraphDataset(
        config_dict['data_path'],
        config_dict['entity_mapping_path'],
        config_dict['relation_mapping_path'],
        max_samples=20  # 只加载20个样本
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"数据集大小: {len(dataset)} 样本")
    
    # 初始化推理器
    print(f"\n加载模型...")
    inference = ClassifierInference(model_path, config_dict)
    
    # 批量预测
    print(f"\n开始批量预测...")
    predictions, probabilities, ground_truth, metadata = \
        inference.predict_batch(dataloader)
    
    # 显示结果
    print(f"\n预测结果 (前10个样本):")
    print("-" * 80)
    for i in range(min(10, len(predictions))):
        pred = predictions[i]
        prob = probabilities[i]
        meta = metadata[i]
        
        pred_label = "非幻觉" if pred == 1 else "幻觉"
        confidence = prob[pred]
        
        print(f"\n样本 {i+1}:")
        print(f"  ID: {meta.get('_id', 'N/A')}")
        print(f"  问题: {meta.get('question', 'N/A')[:60]}...")
        print(f"  预测: {pred_label} (置信度: {confidence:.4f})")
        print(f"  P(幻觉)={prob[0]:.4f}, P(非幻觉)={prob[1]:.4f}")
    
    # 统计
    n_hall = (predictions == 0).sum()
    n_non_hall = (predictions == 1).sum()
    
    print(f"\n" + "="*60)
    print(f"预测统计:")
    print(f"  幻觉: {n_hall} ({n_hall/len(predictions)*100:.1f}%)")
    print(f"  非幻觉: {n_non_hall} ({n_non_hall/len(predictions)*100:.1f}%)")
    
    # 评估
    if ground_truth is not None and len(ground_truth) > 0:
        metrics = inference.evaluate(predictions, ground_truth)
        print(f"\n评估指标:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")


def example_compare_methods():
    """对比FFN分类器和相似度方法"""
    print("\n" + "="*60)
    print("示例 3: 对比FFN分类器和相似度方法")
    print("="*60)
    
    print("\n特性对比:")
    print("-" * 80)
    
    comparison = [
        ("判断方式", "相似度阈值", "端到端FFN分类"),
        ("输出", "相似度分数 [0,1]", "二分类概率 [P(hall), P(non-hall)]"),
        ("决策边界", "固定阈值（需手动调）", "自动学习"),
        ("表达能力", "线性", "非线性（多层FFN）"),
        ("训练方式", "无监督（仅编码器）", "有监督（端到端）"),
        ("可解释性", "高（相似度直观）", "中（概率分布）"),
        ("准确率", "较低", "较高"),
        ("适用场景", "快速原型、可解释性", "生产环境、高准确率")
    ]
    
    print(f"{'特性':<15} | {'相似度方法':<30} | {'FFN分类器':<30}")
    print("-" * 80)
    for feature, method1, method2 in comparison:
        print(f"{feature:<15} | {method1:<30} | {method2:<30}")
    
    print("\n推荐:")
    print("  - 如需快速验证和高可解释性 → 使用相似度方法")
    print("  - 如需高准确率和生产部署 → 使用FFN分类器")
    print("  - 最佳实践：两种方法都运行，对比结果")


def main():
    """主函数"""
    print("\n🚀 FFN分类器使用示例\n")
    
    # 示例1: 单样本预测
    try:
        example_single_prediction()
    except Exception as e:
        print(f"\n示例1出错: {e}")
    
    # 示例2: 批量预测
    try:
        example_batch_prediction()
    except Exception as e:
        print(f"\n示例2出错: {e}")
    
    # 示例3: 方法对比
    example_compare_methods()
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("\n详细使用指南:")
    print("  - 查看 FFN_CLASSIFIER_GUIDE.md")
    print("\n运行完整训练和推理:")
    print("  - run_classifier.bat")


if __name__ == '__main__':
    main()











