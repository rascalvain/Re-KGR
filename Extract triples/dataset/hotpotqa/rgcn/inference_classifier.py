"""
使用FFN分类器进行幻觉检测推理
直接输出二分类标签：0=幻觉, 1=非幻觉
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from config_hotpotqa import Config
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn
from classifier_model import HallucinationClassifier


class ClassifierInference:
    """FFN分类器推理器"""
    
    def __init__(self, model_path, config_dict):
        """
        初始化推理器
        Args:
            model_path: 训练好的模型路径
            config_dict: 配置字典
        """
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"\n加载模型: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print("✓ 模型加载完成")
    
    def _load_model(self, model_path):
        """加载模型"""
        # 初始化模型
        model = HallucinationClassifier(
            entity_embedding_path=self.config['entity_embedding_path'],
            relation_embedding_path=self.config['relation_embedding_path'],
            hidden_channels=self.config['hidden_channels'],
            out_channels=self.config['out_channels'],
            num_layers=self.config['num_layers'],
            freeze_embeddings=self.config.get('freeze_embeddings', True),
            dropout=self.config.get('dropout', 0.3),
            ffn_hidden_dim=self.config.get('ffn_hidden_dim', 128)
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"  验证准确率: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"  验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        return model
    
    @torch.no_grad()
    def predict_batch(self, dataloader):
        """
        批量预测
        Args:
            dataloader: 数据加载器
        Returns:
            predictions: 预测标签 [0=幻觉, 1=非幻觉]
            probabilities: 预测概率 [P(幻觉), P(非幻觉)]
            ground_truth: 真实标签
            metadata: 元数据
        """
        all_predictions = []
        all_probabilities = []
        all_ground_truth = []
        all_metadata = []
        
        for context_batch, gpt_batch, labels, metadata_list in tqdm(dataloader, desc='推理中'):
            if context_batch is None:
                continue
            
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            
            # 前向传播
            logits = self.model(gpt_batch, context_batch)
            probabilities = F.softmax(logits, dim=1)  # [batch_size, 2]
            predictions = torch.argmax(probabilities, dim=1)  # [batch_size]
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_ground_truth.extend(labels.numpy())
            all_metadata.extend(metadata_list)
        
        return (
            np.array(all_predictions),
            np.array(all_probabilities),
            np.array(all_ground_truth),
            all_metadata
        )
    
    @torch.no_grad()
    def predict_single(self, response_graph, reference_graph):
        """
        单样本预测
        Args:
            response_graph: 响应图
            reference_graph: 参考图
        Returns:
            prediction: 预测标签 (0=幻觉, 1=非幻觉)
            probabilities: [P(幻觉), P(非幻觉)]
        """
        response_graph = response_graph.to(self.device)
        reference_graph = reference_graph.to(self.device)
        
        logits = self.model(response_graph, reference_graph)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        
        return prediction.item(), probabilities[0].cpu().numpy()
    
    def evaluate(self, predictions, ground_truth):
        """
        评估模型性能
        Args:
            predictions: 预测标签
            ground_truth: 真实标签
        Returns:
            metrics: 评估指标字典
        """
        metrics = {
            'accuracy': accuracy_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions, zero_division=0),
            'recall': recall_score(ground_truth, predictions, zero_division=0),
            'f1': f1_score(ground_truth, predictions, zero_division=0)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, ground_truth, predictions, save_path):
        """绘制混淆矩阵"""
        cm = confusion_matrix(ground_truth, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hallucination', 'Non-Hallucination'],
            yticklabels=['Hallucination', 'Non-Hallucination']
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")
        plt.close()
    
    def plot_probability_distribution(self, probabilities, ground_truth, predictions, save_path):
        """绘制概率分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 非幻觉概率 (索引1)
        probs_non_hall = probabilities[:, 1]
        
        # 按真实标签分布
        ax = axes[0]
        for label in [0, 1]:
            mask = ground_truth == label
            label_name = 'Hallucination' if label == 0 else 'Non-Hallucination'
            ax.hist(probs_non_hall[mask], bins=20, alpha=0.6, label=f'True {label_name}')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('P(Non-Hallucination)')
        ax.set_ylabel('Count')
        ax.set_title('Probability Distribution by True Label')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 按预测标签分布
        ax = axes[1]
        for pred in [0, 1]:
            mask = predictions == pred
            pred_name = 'Hallucination' if pred == 0 else 'Non-Hallucination'
            ax.hist(probs_non_hall[mask], bins=20, alpha=0.6, label=f'Predicted {pred_name}')
        ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('P(Non-Hallucination)')
        ax.set_ylabel('Count')
        ax.set_title('Probability Distribution by Predicted Label')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"概率分布已保存: {save_path}")
        plt.close()
    
    def save_predictions(self, predictions, probabilities, metadata, save_path):
        """保存预测结果"""
        results = []
        
        for i, meta in enumerate(metadata):
            pred = int(predictions[i])
            prob = probabilities[i]  # [P(hall), P(non-hall)]
            
            result = {
                '_id': meta.get('_id', f'sample_{i}'),
                'question': meta.get('question', ''),
                'answer': meta.get('answer', ''),
                'prediction': pred,
                'label': 'Non-Hallucination' if pred == 1 else 'Hallucination',
                'prob_hallucination': float(prob[0]),
                'prob_non_hallucination': float(prob[1]),
                'confidence': float(max(prob)),
                'num_context_triples': meta.get('num_context_triples', 0),
                'num_gpt_triples': meta.get('num_gpt_triples', 0)
            }
            
            results.append(result)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"预测结果已保存: {save_path}")
        return results


def main():
    """主函数"""
    print("="*60)
    print("FFN分类器幻觉检测推理")
    print("="*60)
    
    # 获取配置
    config_dict = Config.get_config_dict()
    config_dict['ffn_hidden_dim'] = 128
    
    # 模型路径
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_classifier.pth')
    
    if not os.path.exists(model_path):
        print(f"\n❌ 错误: 模型文件不存在: {model_path}")
        print(f"\n请先运行训练: python train_classifier.py")
        return
    
    # 加载数据集
    print("\n加载测试数据集...")
    dataset = HotpotQAGraphDataset(
        config_dict['data_path'],
        config_dict['entity_mapping_path'],
        config_dict['relation_mapping_path'],
        max_samples=config_dict.get('max_samples')
    )
    
    # 使用全部数据进行推理（也可以只用测试集）
    dataloader = DataLoader(
        dataset,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"数据集大小: {len(dataset)} 样本")
    
    # 初始化推理器
    inference = ClassifierInference(model_path, config_dict)
    
    # 批量预测
    print("\n开始推理...")
    predictions, probabilities, ground_truth, metadata = inference.predict_batch(dataloader)
    
    # 评估
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    metrics = inference.evaluate(predictions, ground_truth)
    
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(
        ground_truth, predictions,
        target_names=['Hallucination', 'Non-Hallucination'],
        digits=4
    ))
    
    # 保存结果
    output_dir = Config.OUTPUT_DIR
    
    # 保存预测结果
    predictions_path = os.path.join(output_dir, 'classifier_predictions.json')
    inference.save_predictions(predictions, probabilities, metadata, predictions_path)
    
    # 保存评估指标
    metrics_path = os.path.join(output_dir, 'classifier_evaluation_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"评估指标已保存: {metrics_path}")
    
    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, 'classifier_confusion_matrix.png')
    inference.plot_confusion_matrix(ground_truth, predictions, cm_path)
    
    # 绘制概率分布
    prob_dist_path = os.path.join(output_dir, 'classifier_probability_distribution.png')
    inference.plot_probability_distribution(probabilities, ground_truth, predictions, prob_dist_path)
    
    # 统计
    print("\n" + "="*60)
    print("预测统计")
    print("="*60)
    n_hallucination = (predictions == 0).sum()
    n_non_hallucination = (predictions == 1).sum()
    print(f"预测为幻觉: {n_hallucination} ({n_hallucination/len(predictions)*100:.2f}%)")
    print(f"预测为非幻觉: {n_non_hallucination} ({n_non_hallucination/len(predictions)*100:.2f}%)")
    
    print("\n✓ 推理完成！")
    print(f"\n生成的文件:")
    print(f"  预测结果: {predictions_path}")
    print(f"  评估指标: {metrics_path}")
    print(f"  混淆矩阵: {cm_path}")
    print(f"  概率分布: {prob_dist_path}")


if __name__ == '__main__':
    main()











