import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 加载和使用 TransE 嵌入的示例代码
# ==========================================

def load_embeddings(output_dir='./output', format='pkl'):
    """
    加载训练好的嵌入向量
    
    Args:
        output_dir: 输出目录
        format: 文件格式，'pkl' 或 'npy'
    """
    print("正在加载嵌入向量...")
    
    if format == 'pkl':
        entity_embeddings = pickle.load(open(os.path.join(output_dir, 'ent_embeddings.pkl'), 'rb'))
        relation_embeddings = pickle.load(open(os.path.join(output_dir, 'rel_embeddings.pkl'), 'rb'))
    else:  # npy
        entity_embeddings = np.load(os.path.join(output_dir, 'ent_embeddings.npy'))
        relation_embeddings = np.load(os.path.join(output_dir, 'rel_embeddings.npy'))
    
    print(f"实体嵌入形状: {entity_embeddings.shape}")
    print(f"关系嵌入形状: {relation_embeddings.shape}")
    
    return entity_embeddings, relation_embeddings

def load_id_mappings():
    """加载实体和关系的ID映射"""
    print("正在加载ID映射...")
    
    entity2id = {}
    with open('entity2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity2id[parts[0]] = int(parts[1])
    
    relation2id = {}
    with open('relation2id.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation2id[parts[0]] = int(parts[1])
    
    # 创建反向映射
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    
    print(f"实体总数: {len(entity2id)}")
    print(f"关系总数: {len(relation2id)}")
    
    return entity2id, relation2id, id2entity, id2relation

def get_entity_embedding(entity_name, entity2id, entity_embeddings):
    """获取指定实体的嵌入向量"""
    if entity_name in entity2id:
        entity_id = entity2id[entity_name]
        return entity_embeddings[entity_id]
    else:
        print(f"实体 '{entity_name}' 不存在")
        return None

def get_relation_embedding(relation_name, relation2id, relation_embeddings):
    """获取指定关系的嵌入向量"""
    if relation_name in relation2id:
        relation_id = relation2id[relation_name]
        return relation_embeddings[relation_id]
    else:
        print(f"关系 '{relation_name}' 不存在")
        return None

def find_similar_entities(entity_name, entity2id, id2entity, entity_embeddings, top_k=10):
    """找到与指定实体最相似的K个实体"""
    if entity_name not in entity2id:
        print(f"实体 '{entity_name}' 不存在")
        return []
    
    entity_id = entity2id[entity_name]
    target_embedding = entity_embeddings[entity_id].reshape(1, -1)
    
    # 计算余弦相似度
    similarities = cosine_similarity(target_embedding, entity_embeddings)[0]
    
    # 获取top_k个最相似的实体（排除自身）
    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    results = []
    for idx in similar_indices:
        results.append({
            'entity': id2entity[idx],
            'similarity': similarities[idx]
        })
    
    return results

def predict_tail_entity(head_entity, relation, entity2id, relation2id, id2entity, 
                        entity_embeddings, relation_embeddings, top_k=10):
    """
    根据头实体和关系预测尾实体
    TransE: h + r ≈ t
    """
    if head_entity not in entity2id:
        print(f"实体 '{head_entity}' 不存在")
        return []
    
    if relation not in relation2id:
        print(f"关系 '{relation}' 不存在")
        return []
    
    head_id = entity2id[head_entity]
    relation_id = relation2id[relation]
    
    # TransE: t ≈ h + r
    predicted_tail = entity_embeddings[head_id] + relation_embeddings[relation_id]
    predicted_tail = predicted_tail.reshape(1, -1)
    
    # 找到最接近的实体
    distances = np.linalg.norm(entity_embeddings - predicted_tail, axis=1)
    nearest_indices = np.argsort(distances)[:top_k]
    
    results = []
    for idx in nearest_indices:
        results.append({
            'entity': id2entity[idx],
            'distance': distances[idx]
        })
    
    return results

def main():
    print("="*60)
    print("TransE 嵌入使用示例")
    print("="*60)
    
    # 加载数据（默认从 ./output 目录加载 pkl 格式）
    # 如果要使用 npy 格式，可以改为：load_embeddings(format='npy')
    entity_embeddings, relation_embeddings = load_embeddings(output_dir='./output', format='pkl')
    entity2id, relation2id, id2entity, id2relation = load_id_mappings()
    
    # ==========================================
    # 示例 1: 获取特定实体的嵌入
    # ==========================================
    print("\n" + "="*60)
    print("示例 1: 获取特定实体的嵌入向量")
    print("="*60)
    
    # 获取第一个实体作为示例
    if len(entity2id) > 0:
        sample_entity = list(entity2id.keys())[0]
        embedding = get_entity_embedding(sample_entity, entity2id, entity_embeddings)
        if embedding is not None:
            print(f"\n实体: {sample_entity}")
            print(f"嵌入维度: {embedding.shape}")
            print(f"嵌入向量（前10维）: {embedding[:10]}")
    
    # ==========================================
    # 示例 2: 找相似实体
    # ==========================================
    print("\n" + "="*60)
    print("示例 2: 查找相似实体")
    print("="*60)
    
    if len(entity2id) > 0:
        sample_entity = list(entity2id.keys())[0]
        print(f"\n查找与 '{sample_entity}' 最相似的10个实体:")
        similar_entities = find_similar_entities(
            sample_entity, entity2id, id2entity, entity_embeddings, top_k=10
        )
        
        for i, result in enumerate(similar_entities, 1):
            print(f"  {i}. {result['entity'][:60]} (相似度: {result['similarity']:.4f})")
    
    # ==========================================
    # 示例 3: 三元组预测
    # ==========================================
    print("\n" + "="*60)
    print("示例 3: 基于头实体和关系预测尾实体")
    print("="*60)
    
    if len(entity2id) > 0 and len(relation2id) > 0:
        sample_head = list(entity2id.keys())[0]
        sample_relation = list(relation2id.keys())[0]
        
        print(f"\n给定:")
        print(f"  头实体: {sample_head}")
        print(f"  关系: {sample_relation}")
        print(f"\n预测的尾实体（Top 10）:")
        
        predictions = predict_tail_entity(
            sample_head, sample_relation, 
            entity2id, relation2id, id2entity,
            entity_embeddings, relation_embeddings, 
            top_k=10
        )
        
        for i, result in enumerate(predictions, 1):
            print(f"  {i}. {result['entity'][:60]} (距离: {result['distance']:.4f})")
    
    # ==========================================
    # 示例 4: 保存特定实体的嵌入到文本文件
    # ==========================================
    print("\n" + "="*60)
    print("示例 4: 导出可读格式的嵌入")
    print("="*60)
    
    output_file = './output/sample_embeddings.txt'
    print(f"\n正在导出前10个实体的嵌入到: {output_file}")
    
    os.makedirs('./output', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(min(10, len(entity_embeddings))):
            entity_name = id2entity[i]
            embedding = entity_embeddings[i]
            f.write(f"{entity_name}\t{' '.join(map(str, embedding))}\n")
    
    print("导出完成！")
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("\n你可以根据需要修改此代码来:")
    print("  - 查找特定实体的相似实体")
    print("  - 进行知识图谱补全（链接预测）")
    print("  - 计算实体或关系之间的相似度")
    print("  - 将嵌入用于下游任务（如分类、聚类等）")

if __name__ == '__main__':
    main()

