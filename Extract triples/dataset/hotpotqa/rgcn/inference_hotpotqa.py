"""
HotpotQA 幻觉检测推理脚本
使用训练好的RGCN模型对数据进行幻觉检测（二分类）
"""

import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config_hotpotqa import Config
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn
from siamese_rgcn_improved import SiameseRGCNWithEmbedding
from torch.utils.data import DataLoader


class HallucinationDetector:
    """幻觉检测器"""
    
    def __init__(self, model_path, config_dict, device='cuda'):
        """
        初始化检测器
        Args:
            model_path: 训练好的模型路径
            config_dict: 配置字典
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config_dict
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = SiameseRGCNWithEmbedding(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            hidden_channels=config_dict['hidden_channels'],
            out_channels=config_dict['out_channels'],
            num_layers=config_dict['num_layers'],
            freeze_embeddings=config_dict.get('freeze_embeddings', True),
            dropout=config_dict.get('dropout', 0.3)
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ 模型已加载到 {self.device}")
    
    @torch.no_grad()
    def predict_similarity(self, context_graph, gpt_graph):
        """
        预测两个图的相似度
        Args:
            context_graph: KB图（参考图）
            gpt_graph: GPT生成图（响应图）
        Returns:
            similarity: 余弦相似度 [0, 1]
        """
        context_graph = context_graph.to(self.device)
        gpt_graph = gpt_graph.to(self.device)
        
        # 获取图嵌入
        context_emb, gpt_emb = self.model(context_graph, gpt_graph)
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            context_emb.unsqueeze(0), 
            gpt_emb.unsqueeze(0)
        ).item()
        
        return similarity
    
    @torch.no_grad()
    def predict_batch(self, dataloader, threshold=0.7):
        """
        批量预测
        Args:
            dataloader: 数据加载器
            threshold: 相似度阈值（高于threshold判断为非幻觉，低于判断为幻觉）
        Returns:
            predictions: 预测标签列表 (0=幻觉, 1=非幻觉)
            similarities: 相似度列表
            metadata_list: 元数据列表
        """
        predictions = []
        similarities = []
        all_metadata = []
        
        for context_batch, gpt_batch, labels, metadata_list in tqdm(dataloader, desc="推理中"):
            if context_batch is None:
                continue
            
            context_batch = context_batch.to(self.device)
            gpt_batch = gpt_batch.to(self.device)
            
            # 获取图嵌入
            context_emb, gpt_emb = self.model(context_batch, gpt_batch)
            
            # 计算余弦相似度
            batch_similarities = F.cosine_similarity(context_emb, gpt_emb, dim=1)
            
            # 根据阈值判断
            batch_predictions = (batch_similarities >= threshold).long()
            
            predictions.extend(batch_predictions.cpu().numpy())
            similarities.extend(batch_similarities.cpu().numpy())
            all_metadata.extend(metadata_list)
        
        return predictions, similarities, all_metadata
    
    def find_optimal_threshold(self, dataloader, true_labels=None):
        """
        寻找最优阈值（如果有真实标签）
        Args:
            dataloader: 数据加载器
            true_labels: 真实标签列表
        Returns:
            best_threshold: 最优阈值
            best_f1: 最优F1分数
        """
        if true_labels is None:
            print("警告: 没有真实标签，无法计算最优阈值")
            return 0.7, None
        
        # 获取所有相似度
        _, similarities, _ = self.predict_batch(dataloader, threshold=0.5)
        
        # 尝试不同阈值
        thresholds = np.arange(0.3, 0.95, 0.05)
        best_threshold = 0.7
        best_f1 = 0
        
        results = []
        for threshold in thresholds:
            predictions = [1 if sim >= threshold else 0 for sim in similarities]
            _, _, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            results.append((threshold, f1))
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # 绘制阈值-F1曲线
        thresholds_list, f1_scores = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_list, f1_scores, marker='o')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('Threshold vs F1 Score')
        plt.axvline(x=best_threshold, color='r', linestyle='--', 
                   label=f'Best Threshold: {best_threshold:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(Config.OUTPUT_DIR, 'threshold_f1_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"阈值-F1曲线已保存: {plot_path}")
        plt.close()
        
        return best_threshold, best_f1
    
    def evaluate(self, dataloader, true_labels, threshold=0.7):
        """
        评估模型性能
        Args:
            dataloader: 数据加载器
            true_labels: 真实标签
            threshold: 相似度阈值
        Returns:
            metrics: 评估指标字典
        """
        predictions, similarities, metadata_list = self.predict_batch(dataloader, threshold)
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        # 分类报告
        report = classification_report(
            true_labels, predictions,
            target_names=['Hallucination', 'Non-Hallucination'],
            zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': threshold,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'avg_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities)
        }
        
        # 打印结果
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        print(f"阈值: {threshold:.3f}")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"\n平均相似度: {metrics['avg_similarity']:.4f} ± {metrics['std_similarity']:.4f}")
        print(f"\n分类报告:")
        print(report)
        print(f"\n混淆矩阵:")
        print(f"                预测")
        print(f"              幻觉  非幻觉")
        print(f"真实 幻觉    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"    非幻觉  {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, ['Hallucination', 'Non-Hallucination'])
        
        # 绘制相似度分布
        self.plot_similarity_distribution(similarities, predictions, true_labels, threshold)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        plot_path = os.path.join(Config.OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存: {plot_path}")
        plt.close()
    
    def plot_similarity_distribution(self, similarities, predictions, true_labels, threshold):
        """绘制相似度分布"""
        similarities = np.array(similarities)
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        plt.figure(figsize=(12, 5))
        
        # 子图1: 按真实标签分布
        plt.subplot(1, 2, 1)
        plt.hist(similarities[true_labels == 0], bins=20, alpha=0.5, 
                label='Hallucination', color='red')
        plt.hist(similarities[true_labels == 1], bins=20, alpha=0.5, 
                label='Non-Hallucination', color='green')
        plt.axvline(x=threshold, color='blue', linestyle='--', 
                   label=f'Threshold: {threshold:.2f}')
        plt.xlabel('Similarity')
        plt.ylabel('Count')
        plt.title('Similarity Distribution by True Label')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 按预测标签分布
        plt.subplot(1, 2, 2)
        plt.hist(similarities[predictions == 0], bins=20, alpha=0.5, 
                label='Predicted Hallucination', color='orange')
        plt.hist(similarities[predictions == 1], bins=20, alpha=0.5, 
                label='Predicted Non-Hallucination', color='cyan')
        plt.axvline(x=threshold, color='blue', linestyle='--', 
                   label=f'Threshold: {threshold:.2f}')
        plt.xlabel('Similarity')
        plt.ylabel('Count')
        plt.title('Similarity Distribution by Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(Config.OUTPUT_DIR, 'similarity_distribution.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"相似度分布已保存: {plot_path}")
        plt.close()
    
    def predict_and_save(self, dataloader, output_file, threshold=0.7):
        """
        预测并保存结果到JSON文件
        Args:
            dataloader: 数据加载器
            output_file: 输出文件路径
            threshold: 相似度阈值
        """
        predictions, similarities, metadata_list = self.predict_batch(dataloader, threshold)
        
        # 构建结果
        results = []
        for pred, sim, metadata in zip(predictions, similarities, metadata_list):
            result = {
                '_id': metadata['_id'],
                'question': metadata['question'],
                'answer': metadata['answer'],
                'similarity': float(sim),
                'prediction': int(pred),
                'label': 'Non-Hallucination' if pred == 1 else 'Hallucination',
                'confidence': float(abs(sim - threshold)),
                'num_context_triples': metadata['num_context_triples'],
                'num_gpt_triples': metadata['num_gpt_triples']
            }
            results.append(result)
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n预测结果已保存到: {output_file}")
        print(f"总样本数: {len(results)}")
        print(f"幻觉样本: {sum(1 for r in results if r['prediction'] == 0)}")
        print(f"非幻觉样本: {sum(1 for r in results if r['prediction'] == 1)}")
        
        return results


def main():
    """主函数"""
    print("="*60)
    print("HotpotQA 幻觉检测推理")
    print("="*60)
    
    # 配置
    config_dict = Config.get_config_dict()
    
    # 检查模型文件
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        print("请先训练模型: python train_rgcn_hotpotqa.py")
        return
    
    # 加载数据集
    print("\n加载数据集...")
    dataset = HotpotQAGraphDataset(
        config_dict['data_path'],
        config_dict['entity_mapping_path'],
        config_dict['relation_mapping_path'],
        max_samples=config_dict.get('max_samples')
    )
    
    # 创建数据加载器（使用全部数据进行推理）
    dataloader = DataLoader(
        dataset,
        batch_size=config_dict['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建检测器
    detector = HallucinationDetector(model_path, config_dict)
    
    # 获取真实标签（HotpotQA默认都是非幻觉，这里模拟）
    # 实际使用时需要有标注的数据
    true_labels = [1] * len(dataset)  # 1 = 非幻觉
    
    # 寻找最优阈值
    print("\n寻找最优阈值...")
    best_threshold, best_f1 = detector.find_optimal_threshold(dataloader, true_labels)
    print(f"最优阈值: {best_threshold:.3f}, F1: {best_f1:.4f}")
    
    # 评估
    print("\n评估模型...")
    metrics = detector.evaluate(dataloader, true_labels, threshold=best_threshold)
    
    # 保存预测结果
    output_file = os.path.join(Config.OUTPUT_DIR, 'hallucination_predictions.json')
    results = detector.predict_and_save(dataloader, output_file, threshold=best_threshold)
    
    # 保存评估指标
    metrics_file = os.path.join(Config.OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存: {metrics_file}")
    
    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  - {output_file}")
    print(f"  - {metrics_file}")
    print(f"  - {Config.OUTPUT_DIR}/confusion_matrix.png")
    print(f"  - {Config.OUTPUT_DIR}/similarity_distribution.png")
    print(f"  - {Config.OUTPUT_DIR}/threshold_f1_curve.png")


if __name__ == '__main__':
    main()











