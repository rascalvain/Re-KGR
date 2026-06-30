"""
使用混合嵌入的示例代码
"""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_hybrid_embeddings(embedding_dir='./hybrid_embeddings'):
    """
    加载混合嵌入
    
    Returns:
        entity_embeddings: dict {entity_name: embedding}
        relation_embeddings: dict {relation_name: embedding}
        stats: dict with statistics
    """
    print("正在加载混合嵌入...")
    
    entity_embeddings = pickle.load(
        open(f'{embedding_dir}/entity_hybrid_embeddings.pkl', 'rb')
    )
    relation_embeddings = pickle.load(
        open(f'{embedding_dir}/relation_hybrid_embeddings.pkl', 'rb')
    )
    stats = pickle.load(
        open(f'{embedding_dir}/embedding_stats.pkl', 'rb')
    )
    
    print(f"  实体数: {len(entity_embeddings)}")
    print(f"  关系数: {len(relation_embeddings)}")
    print(f"  嵌入维度: {stats['hybrid_dim']}")
    
    return entity_embeddings, relation_embeddings, stats


def get_entity_embedding(entity_name, entity_embeddings):
    """获取特定实体的嵌入"""
    if entity_name in entity_embeddings:
        return entity_embeddings[entity_name]
    else:
        print(f"实体 '{entity_name}' 不存在")
        return None


def get_relation_embedding(relation_name, relation_embeddings):
    """获取特定关系的嵌入"""
    if relation_name in relation_embeddings:
        return relation_embeddings[relation_name]
    else:
        print(f"关系 '{relation_name}' 不存在")
        return None


def find_similar_entities(entity_name, entity_embeddings, top_k=10):
    """找到最相似的实体"""
    if entity_name not in entity_embeddings:
        print(f"实体 '{entity_name}' 不存在")
        return []
    
    target_emb = entity_embeddings[entity_name].reshape(1, -1)
    
    # 计算与所有实体的相似度
    similarities = []
    for name, emb in entity_embeddings.items():
        if name == entity_name:
            continue
        sim = cosine_similarity(target_emb, emb.reshape(1, -1))[0][0]
        similarities.append((name, sim))
    
    # 排序并返回top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def compute_triple_score(head, relation, tail, entity_embeddings, relation_embeddings):
    """
    计算三元组的得分（简单版本：h + r - t 的距离）
    """
    if head not in entity_embeddings:
        print(f"头实体 '{head}' 不存在")
        return None
    if relation not in relation_embeddings:
        print(f"关系 '{relation}' 不存在")
        return None
    if tail not in entity_embeddings:
        print(f"尾实体 '{tail}' 不存在")
        return None
    
    h_emb = entity_embeddings[head]
    r_emb = relation_embeddings[relation]
    t_emb = entity_embeddings[tail]
    
    # TransE 得分：||h + r - t||
    score = np.linalg.norm(h_emb + r_emb - t_emb)
    
    return score


def predict_tail_entity(head, relation, entity_embeddings, relation_embeddings, top_k=10):
    """
    根据头实体和关系预测尾实体
    """
    if head not in entity_embeddings:
        print(f"头实体 '{head}' 不存在")
        return []
    if relation not in relation_embeddings:
        print(f"关系 '{relation}' 不存在")
        return []
    
    h_emb = entity_embeddings[head]
    r_emb = relation_embeddings[relation]
    
    # 预测：t ≈ h + r
    predicted = h_emb + r_emb
    
    # 找最接近的实体
    distances = []
    for name, emb in entity_embeddings.items():
        if name == head:  # 排除头实体本身
            continue
        dist = np.linalg.norm(predicted - emb)
        distances.append((name, dist))
    
    # 排序并返回top-k
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]


def main():
    print("="*60)
    print("混合嵌入使用示例")
    print("="*60)
    
    # 加载嵌入
    entity_embeddings, relation_embeddings, stats = load_hybrid_embeddings()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    print(f"实体总数: {stats['total_entities']}")
    print(f"  - KB中: {stats['kb_entities']} ({stats['kb_entities']/stats['total_entities']*100:.2f}%)")
    print(f"  - OOV: {stats['oov_entities']} ({stats['oov_entities']/stats['total_entities']*100:.2f}%)")
    print(f"关系总数: {stats['total_relations']}")
    print(f"  - KB中: {stats['kb_relations']} ({stats['kb_relations']/stats['total_relations']*100:.2f}%)")
    print(f"  - OOV: {stats['oov_relations']} ({stats['oov_relations']/stats['total_relations']*100:.2f}%)")
    print(f"\n嵌入维度: {stats['hybrid_dim']}")
    print(f"  - TransE: {stats['transe_dim']}")
    print(f"  - SentenceTransformer: {stats['sentence_dim']}")
    
    # 示例1: 获取特定实体的嵌入
    print("\n" + "="*60)
    print("示例 1: 获取特定实体的嵌入")
    print("="*60)
    
    if len(entity_embeddings) > 0:
        sample_entity = list(entity_embeddings.keys())[0]
        embedding = get_entity_embedding(sample_entity, entity_embeddings)
        if embedding is not None:
            print(f"\n实体: {sample_entity}")
            print(f"嵌入维度: {embedding.shape}")
            print(f"前10维: {embedding[:10]}")
    
    # 示例2: 查找相似实体
    print("\n" + "="*60)
    print("示例 2: 查找相似实体")
    print("="*60)
    
    if len(entity_embeddings) > 10:
        sample_entity = list(entity_embeddings.keys())[5]
        print(f"\n查找与 '{sample_entity}' 最相似的10个实体:")
        similar = find_similar_entities(sample_entity, entity_embeddings, top_k=10)
        for i, (name, sim) in enumerate(similar, 1):
            print(f"  {i}. {name[:60]} (相似度: {sim:.4f})")
    
    # 示例3: 三元组预测
    print("\n" + "="*60)
    print("示例 3: 根据头实体和关系预测尾实体")
    print("="*60)
    
    if len(entity_embeddings) > 0 and len(relation_embeddings) > 0:
        sample_head = list(entity_embeddings.keys())[0]
        sample_relation = list(relation_embeddings.keys())[0]
        
        print(f"\n给定:")
        print(f"  头实体: {sample_head}")
        print(f"  关系: {sample_relation}")
        print(f"\n预测的尾实体 (Top 10):")
        
        predictions = predict_tail_entity(
            sample_head, sample_relation, 
            entity_embeddings, relation_embeddings, 
            top_k=10
        )
        
        for i, (name, dist) in enumerate(predictions, 1):
            print(f"  {i}. {name[:60]} (距离: {dist:.4f})")
    
    # 示例4: 三元组评分
    print("\n" + "="*60)
    print("示例 4: 三元组评分")
    print("="*60)
    
    if len(entity_embeddings) >= 2 and len(relation_embeddings) > 0:
        entities_list = list(entity_embeddings.keys())
        head = entities_list[0]
        tail = entities_list[1]
        relation = list(relation_embeddings.keys())[0]
        
        score = compute_triple_score(head, relation, tail, 
                                     entity_embeddings, relation_embeddings)
        
        if score is not None:
            print(f"\n三元组: ({head}, {relation}, {tail})")
            print(f"得分: {score:.4f} (越小表示越可能为真)")
    
    print("\n" + "="*60)
    print("示例完成")
    print("="*60)
    print("\n你可以使用这些嵌入进行:")
    print("  - 实体/关系分类")
    print("  - 知识图谱补全")
    print("  - 三元组验证")
    print("  - 相似度计算")
    print("  - 下游任务的特征表示")


if __name__ == '__main__':
    main()

