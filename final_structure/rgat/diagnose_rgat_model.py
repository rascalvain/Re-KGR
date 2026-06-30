"""
RGAT模型诊断工具
独立运行，用于分析已训练好的RGAT编码器质量

使用方法:
    python diagnose_rgat_model.py --model_path checkpoints/best_rgat_model.pth
    python diagnose_rgat_model.py --model_path checkpoints/best_rgat_model.pth --num_samples 500
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader, Subset

# 导入配置
from config_hotpotqa_rgat import Config

# 添加rgcn目录到路径
rgcn_dir = os.path.join(os.path.dirname(__file__), '..', 'rgcn')
sys.path.insert(0, os.path.abspath(rgcn_dir))
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn

# 导入RGAT模型
from siamese_rgat_improved import SiameseRGATWithEmbedding


class RGATDiagnosticTool:
    """RGAT模型诊断工具"""

    def __init__(self, model_path, device='cuda'):
        """
        初始化诊断工具
        Args:
            model_path: 模型checkpoint路径
            device: 运行设备
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print("\n" + "=" * 60)
        print("RGAT模型诊断工具")
        print("=" * 60)
        print(f"模型路径: {model_path}")
        print(f"使用设备: {self.device}")

        # 加载checkpoint
        self.checkpoint = self._load_checkpoint()

        # 重建模型
        self.model = self._rebuild_model()

        # 加载数据集
        self.dataset = self._load_dataset()

    def _load_checkpoint(self):
        """加载checkpoint"""
        print("\n加载checkpoint...")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 在CPU上加载（避免显存问题）
        checkpoint = torch.load(self.model_path, map_location='cpu')

        print(f"✓ Checkpoint加载成功")
        print(f"  训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"  验证损失: {checkpoint.get('val_loss', 'N/A')}")

        return checkpoint

    def _rebuild_model(self):
        """重建模型"""
        print("\n重建模型...")

        # 从checkpoint获取配置
        if 'config' in self.checkpoint and self.checkpoint['config'] is not None:
            config = self.checkpoint['config']
        else:
            # 使用默认配置
            config = Config.get_config_dict()
            print("  ⚠️ Checkpoint中无config，使用默认配置")

        # 打印模型配置
        print(f"  模型配置:")
        print(f"    隐藏层维度: {config.get('hidden_channels', 128)}")
        print(f"    输出维度: {config.get('out_channels', 64)}")
        print(f"    层数: {config.get('num_layers', 3)}")
        print(f"    注意力头数: {config.get('num_heads', 4)}")

        # 创建模型
        model = SiameseRGATWithEmbedding(
            entity_embedding_path=config['entity_embedding_path'],
            relation_embedding_path=config['relation_embedding_path'],
            hidden_channels=config['hidden_channels'],
            out_channels=config['out_channels'],
            num_layers=config['num_layers'],
            freeze_embeddings=False,
            dropout=config.get('dropout', 0.3),
            num_heads=config.get('num_heads', 4)
        )

        # 加载权重
        model_state = self.checkpoint['model_state_dict']
        model.load_state_dict(model_state)

        # 移到设备
        model.to(self.device)
        model.eval()

        print(f"✓ 模型重建成功")

        return model

    def _load_dataset(self):
        """加载数据集"""
        print("\n加载数据集...")

        config = self.checkpoint.get('config', Config.get_config_dict())

        dataset = HotpotQAGraphDataset(
            config['data_path'],
            config['entity_mapping_path'],
            config['relation_mapping_path'],
            max_samples=None
        )

        print(f"✓ 数据集加载成功")
        print(f"  总样本数: {len(dataset)}")

        return dataset

    def diagnose_encoder_quality(self, num_samples=300):
        """
        诊断编码器质量（参考训练脚本实现）
        Args:
            num_samples: 用于诊断的样本数量
        """
        print("\n" + "=" * 60)
        print("🔍 编码器质量诊断")
        print("=" * 60)

        self.model.eval()
        hallucination_embeddings = []
        factual_embeddings = []

        # 🔥 创建临时的DataLoader（与训练时相同的方式）
        from torch.utils.data import DataLoader, Subset

        # 随机采样索引
        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, indices)

        # 创建DataLoader
        temp_loader = DataLoader(
            subset,
            batch_size=16,  # 使用较小的batch size
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        print(f"\n从 {len(self.dataset)} 个样本中采样 {len(indices)} 个...")
        print(f"使用 {len(temp_loader)} 个批次进行处理...")

        sample_count = 0

        with torch.no_grad():
            for context_batch, gpt_batch, labels, _ in tqdm(temp_loader, desc="提取嵌入"):
                if context_batch is None:
                    continue

                # 移到设备
                context_batch = context_batch.to(self.device)
                gpt_batch = gpt_batch.to(self.device)
                labels = labels.cpu().numpy()  # 转换为numpy

                try:
                    # 🔥 使用与训练时相同的方式获取嵌入
                    _, gpt_emb = self.model(context_batch, gpt_batch)
                    gpt_emb = gpt_emb.cpu().numpy()

                    # 按标签分类收集
                    for i, label in enumerate(labels):
                        if label == 0:
                            hallucination_embeddings.append(gpt_emb[i])
                        elif label == 1:
                            factual_embeddings.append(gpt_emb[i])
                        sample_count += 1

                except Exception as e:
                    print(f"\n⚠️ 批次处理出错: {e}")
                    continue

        print(f"\n成功提取:")
        print(f"  幻觉样本: {len(hallucination_embeddings)}")
        print(f"  事实样本: {len(factual_embeddings)}")

        # 检查是否有足够的数据
        if len(hallucination_embeddings) == 0 or len(factual_embeddings) == 0:
            print("\n❌ 错误: 无法提取足够的嵌入")
            print(f"   幻觉样本: {len(hallucination_embeddings)}")
            print(f"   事实样本: {len(factual_embeddings)}")
            return None

        # 转换为numpy数组
        hall_emb = np.array(hallucination_embeddings)
        fact_emb = np.array(factual_embeddings)

        # 🔥 使用与训练脚本相同的统计方法
        # 计算每类的均值嵌入
        hall_mean = hall_emb.mean(axis=0)
        fact_mean = fact_emb.mean(axis=0)

        # 计算类内方差
        hall_var = ((hall_emb - hall_mean) ** 2).mean()
        fact_var = ((fact_emb - fact_mean) ** 2).mean()

        # 计算类间距离
        inter_class_distance = np.linalg.norm(hall_mean - fact_mean)

        # 计算类内距离
        intra_class_distance = (hall_var + fact_var) / 2

        # Fisher判别比（类间距离 / 类内距离）
        fisher_ratio = inter_class_distance / (np.sqrt(intra_class_distance) + 1e-6)

        # 构建结果字典
        results = {
            'num_hallucination': len(hallucination_embeddings),
            'num_factual': len(factual_embeddings),
            'hall_mean_norm': float(np.linalg.norm(hall_mean)),
            'fact_mean_norm': float(np.linalg.norm(fact_mean)),
            'inter_class_distance': float(inter_class_distance),
            'intra_class_variance': float(intra_class_distance),
            'fisher_ratio': float(fisher_ratio),
            'hall_var': float(hall_var),
            'fact_var': float(fact_var)
        }

        # 打印结果
        print(f"\n嵌入空间分析:")
        print(f"  样本数: 幻觉={results['num_hallucination']}, 事实={results['num_factual']}")
        print(f"  幻觉嵌入均值范数: {results['hall_mean_norm']:.4f}")
        print(f"  事实嵌入均值范数: {results['fact_mean_norm']:.4f}")

        print(f"\n区分性分析:")
        print(f"  类间距离: {results['inter_class_distance']:.4f}")
        print(f"  类内方差: {results['intra_class_variance']:.4f}")
        print(f"  Fisher判别比: {results['fisher_ratio']:.4f}")

        # 质量评估
        fisher = results['fisher_ratio']
        if fisher < 1.0:
            print(f"  ⚠️ 警告: Fisher比 < 1.0，编码器区分能力弱")
            print(f"  建议: 继续训练或调整超参数")
        elif fisher < 2.0:
            print(f"  ⚙️ 一般: Fisher比在1-2之间，编码器有一定区分能力")
        else:
            print(f"  ✓ 良好: Fisher比 > 2.0，编码器具有良好区分能力")

        print("=" * 60)

        # 可视化
        self._visualize_embeddings(hall_emb, fact_emb)

        # 保存结果
        self._save_results(results)

        return results

    def _compute_statistics(self, hall_emb, fact_emb):
        """计算统计量"""

        # 计算均值
        hall_mean = hall_emb.mean(axis=0)
        fact_mean = fact_emb.mean(axis=0)

        # 计算范数
        hall_norm = np.linalg.norm(hall_mean)
        fact_norm = np.linalg.norm(fact_mean)

        # 计算类内方差
        hall_var = ((hall_emb - hall_mean) ** 2).mean()
        fact_var = ((fact_emb - fact_mean) ** 2).mean()
        intra_class_var = (hall_var + fact_var) / 2

        # 计算类间距离
        inter_class_distance = np.linalg.norm(hall_mean - fact_mean)

        # Fisher判别比
        fisher_ratio = inter_class_distance / (np.sqrt(intra_class_var) + 1e-6)

        # 余弦相似度
        cosine_sim = np.dot(hall_mean, fact_mean) / (hall_norm * fact_norm + 1e-6)

        # 每个维度的平均激活
        hall_activation = np.abs(hall_emb).mean(axis=0)
        fact_activation = np.abs(fact_emb).mean(axis=0)

        # 稀疏度（接近0的维度比例）
        threshold = 0.01
        hall_sparsity = (hall_activation < threshold).sum() / len(hall_activation)
        fact_sparsity = (fact_activation < threshold).sum() / len(fact_activation)

        return {
            'num_hallucination': len(hall_emb),
            'num_factual': len(fact_emb),
            'hall_mean_norm': float(hall_norm),
            'fact_mean_norm': float(fact_norm),
            'inter_class_distance': float(inter_class_distance),
            'intra_class_variance': float(intra_class_var),
            'fisher_ratio': float(fisher_ratio),
            'cosine_similarity': float(cosine_sim),
            'hall_sparsity': float(hall_sparsity),
            'fact_sparsity': float(fact_sparsity),
            'hall_var': float(hall_var),
            'fact_var': float(fact_var)
        }

    def _print_results(self, results):
        """打印诊断结果"""
        print(f"\n嵌入空间分析:")
        print(f"  样本数: 幻觉={results['num_hallucination']}, 事实={results['num_factual']}")
        print(f"  幻觉嵌入均值范数: {results['hall_mean_norm']:.4f}")
        print(f"  事实嵌入均值范数: {results['fact_mean_norm']:.4f}")

        print(f"\n区分性分析:")
        print(f"  类间距离: {results['inter_class_distance']:.4f}")
        print(f"  类内方差: {results['intra_class_variance']:.4f}")
        print(f"  Fisher判别比: {results['fisher_ratio']:.4f}")
        print(f"  余弦相似度: {results['cosine_similarity']:.4f}")

        print(f"\n稀疏性分析:")
        print(f"  幻觉嵌入稀疏度: {results['hall_sparsity'] * 100:.1f}%")
        print(f"  事实嵌入稀疏度: {results['fact_sparsity'] * 100:.1f}%")

        # 质量评估
        fisher = results['fisher_ratio']
        if fisher < 1.0:
            status = "⚠️ 警告: Fisher比 < 1.0，编码器区分能力弱"
            suggestion = "建议: 增大输出维度或继续训练"
        elif fisher < 2.0:
            status = "⚙️ 一般: Fisher比在1-2之间，编码器有一定区分能力"
            suggestion = "建议: 可以尝试微调超参数"
        elif fisher < 3.5:
            status = "✓ 良好: Fisher比在2-3.5之间，编码器具有良好区分能力"
            suggestion = "建议: 可以用于下游任务"
        else:
            status = "✓✓ 优秀: Fisher比 > 3.5，编码器具有优秀的区分能力"
            suggestion = "建议: 模型已达到很好的效果"

        print(f"\n质量评估:")
        print(f"  {status}")
        print(f"  {suggestion}")

    def _visualize_embeddings(self, hall_emb, fact_emb):
        """可视化嵌入空间（使用t-SNE降维）"""
        print(f"\n生成t-SNE可视化...")

        # 合并数据
        all_emb = np.vstack([hall_emb, fact_emb])
        labels = np.array([0] * len(hall_emb) + [1] * len(fact_emb))

        # t-SNE降维
        print(f"  执行t-SNE降维（可能需要几分钟）...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_emb) // 4))
        emb_2d = tsne.fit_transform(all_emb)

        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 子图1：散点图
        ax = axes[0]
        scatter = ax.scatter(
            emb_2d[labels == 0, 0], emb_2d[labels == 0, 1],
            c='red', label='幻觉', alpha=0.6, s=30
        )
        scatter = ax.scatter(
            emb_2d[labels == 1, 0], emb_2d[labels == 1, 1],
            c='blue', label='事实', alpha=0.6, s=30
        )
        ax.set_xlabel('t-SNE维度 1', fontsize=12)
        ax.set_ylabel('t-SNE维度 2', fontsize=12)
        ax.set_title('RGAT嵌入空间可视化 (t-SNE)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 子图2：密度图
        ax = axes[1]
        sns.kdeplot(
            x=emb_2d[labels == 0, 0], y=emb_2d[labels == 0, 1],
            cmap='Reds', shade=True, alpha=0.5, ax=ax, label='幻觉'
        )
        sns.kdeplot(
            x=emb_2d[labels == 1, 0], y=emb_2d[labels == 1, 1],
            cmap='Blues', shade=True, alpha=0.5, ax=ax, label='事实'
        )
        ax.set_xlabel('t-SNE维度 1', fontsize=12)
        ax.set_ylabel('t-SNE维度 2', fontsize=12)
        ax.set_title('嵌入密度分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)

        # 保存
        output_dir = os.path.dirname(self.model_path)
        vis_path = os.path.join(output_dir, 'embedding_visualization.png')
        plt.tight_layout()
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 可视化已保存: {vis_path}")
        plt.close()

    def _save_results(self, results):
        """保存诊断结果"""
        output_dir = os.path.dirname(self.model_path)
        result_path = os.path.join(output_dir, 'diagnostic_report.json')

        # 添加元信息
        report = {
            'model_path': self.model_path,
            'diagnostics': results,
            'checkpoint_info': {
                'epoch': self.checkpoint.get('epoch', 'N/A'),
                'val_loss': float(self.checkpoint.get('val_loss', 0)) if self.checkpoint.get('val_loss') else 'N/A'
            }
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 诊断报告已保存: {result_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RGAT模型诊断工具')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=300,
        help='用于诊断的样本数量（默认300）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='运行设备（cuda或cpu）'
    )

    args = parser.parse_args()

    # 如果没有指定模型路径，使用默认路径
    if args.model_path is None:
        args.model_path = Config.BEST_MODEL_PATH
        print(f"未指定模型路径，使用默认: {args.model_path}")

    # 创建诊断工具
    tool = RGATDiagnosticTool(args.model_path, args.device)

    # 执行诊断
    results = tool.diagnose_encoder_quality(num_samples=args.num_samples)

    print("\n" + "=" * 60)
    print("✓ 诊断完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    output_dir = os.path.dirname(args.model_path)
    print(f"  诊断报告: {os.path.join(output_dir, 'diagnostic_report.json')}")
    print(f"  可视化图: {os.path.join(output_dir, 'embedding_visualization.png')}")


if __name__ == '__main__':
    main()