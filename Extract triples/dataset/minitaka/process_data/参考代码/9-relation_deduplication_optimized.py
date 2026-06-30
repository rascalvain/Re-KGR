#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用sentence-transformer对关系词进行对齐、聚类和去重（优化版本）
使用更高效的算法处理大量关系词
"""

import os
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import torch
from sklearn.cluster import DBSCAN
import pickle

class RelationDeduplicatorOptimized:
    """关系词去重和对齐工具（优化版本）"""
    
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
        try:
            self.model = SentenceTransformer(model_path, device=device)
        except Exception as e:
            print(f"加载本地模型失败: {e}")
            print("尝试使用在线模型: all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        self.similarity_threshold = similarity_threshold
        self.device = device
        print(f"相似度阈值: {similarity_threshold}")
        print(f"使用设备: {device}")
    
    def load_relations(self, relation_file):
        """加载关系词文件"""
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
    
    def compute_embeddings(self, relations, batch_size=64, cache_file=None):
        """
        计算关系词的嵌入向量（支持缓存）
        
        Args:
            relations: 关系词列表
            batch_size: 批处理大小
            cache_file: 缓存文件路径，如果存在则直接加载
        """
        if cache_file and os.path.exists(cache_file):
            print(f"从缓存加载嵌入向量: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        relation_texts = [rel[0] for rel in relations]
        print(f"开始计算嵌入向量，批大小: {batch_size}")
        embeddings = self.model.encode(
            relation_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        if cache_file:
            print(f"保存嵌入向量到缓存: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        
        print(f"嵌入向量维度: {embeddings.shape}")
        return embeddings
    
    def find_duplicates_with_clustering(self, relations, embeddings):
        """
        使用DBSCAN聚类找到重复的关系词（更高效）
        
        Args:
            relations: 关系词列表
            embeddings: 嵌入向量矩阵
            
        Returns:
            dict: 关系词到主关系词的映射
            dict: 聚类信息
        """
        print("使用DBSCAN进行聚类...")
        
        # 将相似度阈值转换为距离阈值
        # cosine similarity = 1 - cosine distance
        # 所以 distance = 1 - similarity
        eps = 1 - self.similarity_threshold
        
        # 使用DBSCAN聚类
        # min_samples=1 表示每个点都可以是一个簇
        clustering = DBSCAN(eps=eps, min_samples=1, metric='cosine', n_jobs=-1)
        cluster_labels = clustering.fit_predict(embeddings)
        
        # 组织聚类结果
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(idx)
        
        print(f"找到 {len(clusters)} 个聚类")
        
        # 为每个聚类选择主关系词（选择最短的或第一个）
        relation_to_main = {}
        cluster_info = {}
        
        for cluster_id, indices in clusters.items():
            if len(indices) == 1:
                # 单独的关系词
                idx = indices[0]
                relation, _ = relations[idx]
                relation_to_main[relation] = relation
                cluster_info[relation] = [relation]
            else:
                # 有多个关系词的聚类，选择最短的作为主关系词
                cluster_relations = [(relations[idx][0], idx) for idx in indices]
                cluster_relations.sort(key=lambda x: (len(x[0]), x[0]))  # 先按长度，再按字母顺序
                main_relation = cluster_relations[0][0]
                
                for relation, _ in cluster_relations:
                    relation_to_main[relation] = main_relation
                
                cluster_info[main_relation] = [rel[0] for rel in cluster_relations]
        
        return relation_to_main, cluster_info
    
    def find_duplicates_with_similarity(self, relations, embeddings, batch_size=1000):
        """
        使用分批相似度计算找到重复的关系词（内存友好）
        
        Args:
            relations: 关系词列表
            embeddings: 嵌入向量矩阵
            batch_size: 批处理大小
            
        Returns:
            dict: 关系词到主关系词的映射
            dict: 聚类信息
        """
        print("使用分批相似度计算查找重复关系词...")
        
        n = len(relations)
        relation_to_main = {}
        processed = set()
        cluster_info = defaultdict(list)
        
        # 归一化嵌入向量以便计算余弦相似度
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        for i in tqdm(range(n), desc="处理关系词"):
            if i in processed:
                continue
            
            main_relation, main_id = relations[i]
            relation_to_main[main_relation] = main_relation
            cluster_info[main_relation].append(main_relation)
            
            # 分批计算相似度
            main_emb = embeddings_norm[i:i+1]
            
            # 只检查未处理的关系词
            remaining_indices = [j for j in range(i+1, n) if j not in processed]
            
            if remaining_indices:
                remaining_embs = embeddings_norm[remaining_indices]
                
                # 计算相似度
                similarities = np.dot(remaining_embs, main_emb.T).flatten()
                
                # 找到相似的关系词
                similar_mask = similarities >= self.similarity_threshold
                similar_indices = [remaining_indices[j] for j in range(len(remaining_indices)) if similar_mask[j]]
                
                for j in similar_indices:
                    similar_relation, _ = relations[j]
                    if similar_relation != main_relation:
                        relation_to_main[similar_relation] = main_relation
                        cluster_info[main_relation].append(similar_relation)
                        processed.add(j)
            
            processed.add(i)
        
        return relation_to_main, dict(cluster_info)
    
    def deduplicate_relations(self, relations, relation_to_main, cluster_info):
        """生成去重后的关系词列表"""
        # 获取所有主关系词
        main_relations = set(relation_to_main.values())
        main_relations = sorted(list(main_relations))
        
        # 创建新的ID映射
        new_relation2id = {}
        old_id_to_new_id = {}
        
        for new_id, main_relation in enumerate(main_relations):
            new_relation2id[main_relation] = new_id
            
            # 找到所有映射到该主关系词的旧ID
            for relation, old_id in relations:
                if relation_to_main.get(relation) == main_relation:
                    old_id_to_new_id[old_id] = new_id
        
        print(f"去重前: {len(relations)} 个关系词")
        print(f"去重后: {len(new_relation2id)} 个关系词")
        print(f"减少了: {len(relations) - len(new_relation2id)} 个关系词 ({100*(len(relations) - len(new_relation2id))/len(relations):.2f}%)")
        
        return new_relation2id, old_id_to_new_id, cluster_info
    
    def save_results(self, new_relation2id, old_id_to_new_id, alignment_info, output_dir='.'):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
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
        groups_with_duplicates = sum(1 for v in alignment_info.values() if len(v) > 1)
        stats = {
            'original_count': len(old_id_to_new_id),
            'deduplicated_count': len(new_relation2id),
            'reduced_count': len(old_id_to_new_id) - len(new_relation2id),
            'reduction_rate': f"{(len(old_id_to_new_id) - len(new_relation2id)) / len(old_id_to_new_id) * 100:.2f}%",
            'similarity_threshold': self.similarity_threshold,
            'groups_with_duplicates': groups_with_duplicates
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
    
    def process(self, relation_file, output_dir='.', similarity_threshold=None, 
                method='similarity', use_cache=True):
        """
        完整的处理流程
        
        Args:
            relation_file: 输入的关系词文件路径
            output_dir: 输出目录
            similarity_threshold: 相似度阈值
            method: 去重方法 ('clustering' 或 'similarity')
            use_cache: 是否使用嵌入向量缓存
        """
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        
        # 1. 加载关系词
        relations = self.load_relations(relation_file)
        
        # 2. 计算嵌入向量（支持缓存）
        cache_file = os.path.join(output_dir, 'embeddings_cache.pkl') if use_cache else None
        embeddings = self.compute_embeddings(relations, cache_file=cache_file)
        
        # 3. 查找重复
        if method == 'clustering':
            relation_to_main, cluster_info = self.find_duplicates_with_clustering(relations, embeddings)
        else:
            relation_to_main, cluster_info = self.find_duplicates_with_similarity(relations, embeddings)
        
        # 4. 去重
        new_relation2id, old_id_to_new_id, alignment_info = self.deduplicate_relations(
            relations, relation_to_main, cluster_info
        )
        
        # 5. 保存结果
        self.save_results(new_relation2id, old_id_to_new_id, alignment_info, output_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='关系词去重和对齐工具（优化版本）')
    parser.add_argument('--input', '-i', type=str, default='./data/relation2id.txt',
                        help='输入的关系词文件路径')
    parser.add_argument('--output', '-o', type=str, default='./data',
                        help='输出目录')
    parser.add_argument('--threshold', '-t', type=float, default=0.85,
                        help='相似度阈值 (0-1)，默认0.85')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='sentence-transformer模型路径')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='计算设备，默认cuda')
    parser.add_argument('--method', type=str, default='similarity',
                        choices=['clustering', 'similarity'],
                        help='去重方法: clustering(DBSCAN) 或 similarity(分批相似度)')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用嵌入向量缓存')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    # 创建去重器
    deduplicator = RelationDeduplicatorOptimized(
        model_path=args.model,
        similarity_threshold=args.threshold,
        device=args.device
    )
    
    # 处理
    deduplicator.process(
        args.input, 
        args.output,
        method=args.method,
        use_cache=not args.no_cache
    )


if __name__ == '__main__':
    main()






