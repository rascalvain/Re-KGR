#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用sentence-transformer对关系词进行对齐、聚类和去重
"""

import os
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import torch

class RelationDeduplicator:
    """关系词去重和对齐工具"""
    
    def __init__(self, model_path=None, similarity_threshold=0.85, device='cuda'):
        """
        初始化
        
        Args:
            model_path: sentence-transformer模型路径，如果为None则使用本地模型
            similarity_threshold: 相似度阈值，超过此阈值的关系词将被视为重复
            device: 计算设备 ('cuda' 或 'cpu')
        """
        # 确定模型路径
        if model_path is None:
            # 使用项目中的本地模型
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            model_path = os.path.join(base_dir, 'sentence-bert')
            if not os.path.exists(model_path):
                # 如果本地模型不存在，使用在线模型
                print(f"本地模型不存在，使用在线模型: all-MiniLM-L6-v2")
                model_path = 'all-MiniLM-L6-v2'
        
        print(f"加载sentence-transformer模型: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self.similarity_threshold = similarity_threshold
        self.device = device
        print(f"相似度阈值: {similarity_threshold}")
        print(f"使用设备: {device}")
    
    def load_relations(self, relation_file):
        """
        加载关系词文件
        
        Args:
            relation_file: relation2id.txt文件路径
            
        Returns:
            list: [(关系词, id), ...]
        """
        relations = []
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    relation = parts[0]
                    rel_id = int(parts[1])
                    relations.append((relation, rel_id))
        print(f"加载了 {len(relations)} 个关系词")
        return relations
    
    def compute_embeddings(self, relations, batch_size=32):
        """
        计算关系词的嵌入向量
        
        Args:
            relations: 关系词列表
            batch_size: 批处理大小
            
        Returns:
            numpy.ndarray: 嵌入向量矩阵
        """
        relation_texts = [rel[0] for rel in relations]
        print(f"开始计算嵌入向量，批大小: {batch_size}")
        embeddings = self.model.encode(
            relation_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"嵌入向量维度: {embeddings.shape}")
        return embeddings
    
    def find_duplicates(self, relations, embeddings):
        """
        找到重复的关系词
        
        Args:
            relations: 关系词列表 [(关系词, id), ...]
            embeddings: 嵌入向量矩阵
            
        Returns:
            dict: {主关系词: [重复关系词列表], ...}
            dict: 关系词到主关系词的映射 {关系词: 主关系词, ...}
        """
        print("开始计算相似度矩阵...")
        # 计算余弦相似度矩阵
        similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
        
        # 找到相似的关系词对
        duplicate_groups = defaultdict(list)
        relation_to_main = {}
        processed = set()
        
        print(f"开始查找重复关系词（阈值: {self.similarity_threshold}）...")
        for i in tqdm(range(len(relations)), desc="处理关系词"):
            if i in processed:
                continue
            
            main_relation, main_id = relations[i]
            duplicate_groups[main_relation].append((main_relation, main_id))
            relation_to_main[main_relation] = main_relation
            
            # 查找与当前关系词相似的其他关系词
            similar_indices = np.where(similarity_matrix[i] >= self.similarity_threshold)[0]
            
            for j in similar_indices:
                if i != j and j not in processed:
                    similar_relation, similar_id = relations[j]
                    # 避免完全相同的字符串（已经在同一组）
                    if similar_relation != main_relation:
                        duplicate_groups[main_relation].append((similar_relation, similar_id))
                        relation_to_main[similar_relation] = main_relation
                        processed.add(j)
            
            processed.add(i)
        
        print(f"找到 {len(duplicate_groups)} 个关系词组（包含重复）")
        return duplicate_groups, relation_to_main
    
    def deduplicate_relations(self, relations, duplicate_groups, relation_to_main):
        """
        生成去重后的关系词列表
        
        Args:
            relations: 原始关系词列表
            duplicate_groups: 重复关系词组
            relation_to_main: 关系词到主关系词的映射
            
        Returns:
            list: 去重后的关系词列表 [(主关系词, 新id), ...]
            dict: 原始ID到新ID的映射 {旧id: 新id, ...}
            dict: 关系词对齐信息 {主关系词: [所有对齐的关系词], ...}
        """
        # 只保留主关系词
        main_relations = list(duplicate_groups.keys())
        main_relations.sort()  # 按字母顺序排序
        
        # 创建新的ID映射
        new_relation2id = {}
        old_id_to_new_id = {}
        alignment_info = {}
        
        for new_id, main_relation in enumerate(main_relations):
            new_relation2id[main_relation] = new_id
            
            # 记录该组中所有关系词的旧ID到新ID的映射
            group = duplicate_groups[main_relation]
            for rel_text, old_id in group:
                old_id_to_new_id[old_id] = new_id
                if main_relation not in alignment_info:
                    alignment_info[main_relation] = []
                if rel_text not in alignment_info[main_relation]:
                    alignment_info[main_relation].append(rel_text)
        
        print(f"去重前: {len(relations)} 个关系词")
        print(f"去重后: {len(new_relation2id)} 个关系词")
        print(f"减少了: {len(relations) - len(new_relation2id)} 个关系词 ({100*(len(relations) - len(new_relation2id))/len(relations):.2f}%)")
        
        return new_relation2id, old_id_to_new_id, alignment_info
    
    def save_results(self, new_relation2id, old_id_to_new_id, alignment_info, 
                     output_dir='.'):
        """
        保存结果
        
        Args:
            new_relation2id: 新的关系词到ID的映射
            old_id_to_new_id: 旧ID到新ID的映射
            alignment_info: 对齐信息
            output_dir: 输出目录
        """
        # 保存新的relation2id.txt
        output_file = os.path.join(output_dir, 'relation2id_deduplicated.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for relation, rel_id in sorted(new_relation2id.items(), key=lambda x: x[1]):
                f.write(f"{relation}\t{rel_id}\n")
        print(f"已保存去重后的关系词文件: {output_file}")
        
        # 保存ID映射文件
        id_mapping_file = os.path.join(output_dir, 'id_mapping.json')
        with open(id_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(old_id_to_new_id, f, ensure_ascii=False, indent=2)
        print(f"已保存ID映射文件: {id_mapping_file}")
        
        # 保存对齐信息
        alignment_file = os.path.join(output_dir, 'relation_alignment.json')
        with open(alignment_file, 'w', encoding='utf-8') as f:
            json.dump(alignment_info, f, ensure_ascii=False, indent=2)
        print(f"已保存对齐信息文件: {alignment_file}")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, 'deduplication_stats.json')
        stats = {
            'original_count': len(old_id_to_new_id),
            'deduplicated_count': len(new_relation2id),
            'reduced_count': len(old_id_to_new_id) - len(new_relation2id),
            'reduction_rate': f"{(len(old_id_to_new_id) - len(new_relation2id)) / len(old_id_to_new_id) * 100:.2f}%",
            'similarity_threshold': self.similarity_threshold,
            'groups_with_duplicates': sum(1 for v in alignment_info.values() if len(v) > 1)
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"已保存统计信息文件: {stats_file}")
        print(f"\n统计信息:")
        print(f"  原始关系词数: {stats['original_count']}")
        print(f"  去重后关系词数: {stats['deduplicated_count']}")
        print(f"  减少数量: {stats['reduced_count']}")
        print(f"  减少比例: {stats['reduction_rate']}")
        print(f"  包含重复的组数: {stats['groups_with_duplicates']}")
    
    def process(self, relation_file, output_dir='.', similarity_threshold=None):
        """
        完整的处理流程
        
        Args:
            relation_file: 输入的关系词文件路径
            output_dir: 输出目录
            similarity_threshold: 相似度阈值（如果提供，会覆盖初始化时的阈值）
        """
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        
        # 1. 加载关系词
        relations = self.load_relations(relation_file)
        
        # 2. 计算嵌入向量
        embeddings = self.compute_embeddings(relations)
        
        # 3. 查找重复
        duplicate_groups, relation_to_main = self.find_duplicates(relations, embeddings)
        
        # 4. 去重
        new_relation2id, old_id_to_new_id, alignment_info = self.deduplicate_relations(
            relations, duplicate_groups, relation_to_main
        )
        
        # 5. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        self.save_results(new_relation2id, old_id_to_new_id, alignment_info, output_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='关系词去重和对齐工具')
    parser.add_argument('--input', '-i', type=str, default='relation2id.txt',
                        help='输入的关系词文件路径')
    parser.add_argument('--output', '-o', type=str, default='.',
                        help='输出目录')
    parser.add_argument('--threshold', '-t', type=float, default=0.85,
                        help='相似度阈值 (0-1)，默认0.85')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='sentence-transformer模型路径，默认使用项目中的本地模型')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='计算设备，默认cuda')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    # 创建去重器
    deduplicator = RelationDeduplicator(
        model_path=args.model,
        similarity_threshold=args.threshold,
        device=args.device
    )
    
    # 处理
    deduplicator.process(args.input, args.output)


if __name__ == '__main__':
    main()






