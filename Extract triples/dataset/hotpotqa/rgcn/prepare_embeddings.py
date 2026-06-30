"""
准备RGCN所需的嵌入文件
将混合嵌入转换为RGCN需要的格式
"""

import pickle
import numpy as np
import os
from config_hotpotqa import Config


def prepare_embeddings_for_rgcn():
    """
    将混合嵌入转换为RGCN需要的格式
    RGCN需要的格式：
    {
        'embeddings': numpy array (num_entities, embedding_dim),
        'num_entities': int,
        'embedding_dim': int
    }
    """
    print("="*60)
    print("准备RGCN嵌入文件")
    print("="*60)
    
    # 检查输入文件
    if not os.path.exists(Config.ENTITY_HYBRID_EMBEDDING_PATH):
        print(f"错误: 实体混合嵌入文件不存在")
        print(f"路径: {Config.ENTITY_HYBRID_EMBEDDING_PATH}")
        print("\n请先运行: python generate_hybrid_embeddings.py")
        return False
    
    if not os.path.exists(Config.RELATION_HYBRID_EMBEDDING_PATH):
        print(f"错误: 关系混合嵌入文件不存在")
        print(f"路径: {Config.RELATION_HYBRID_EMBEDDING_PATH}")
        print("\n请先运行: python generate_hybrid_embeddings.py")
        return False
    
    if not os.path.exists(Config.ENTITY2IDX_PATH):
        print(f"错误: 实体映射文件不存在")
        print(f"路径: {Config.ENTITY2IDX_PATH}")
        print("\n请先运行: python generate_hybrid_embeddings.py")
        return False
    
    if not os.path.exists(Config.RELATION2IDX_PATH):
        print(f"错误: 关系映射文件不存在")
        print(f"路径: {Config.RELATION2IDX_PATH}")
        print("\n请先运行: python generate_hybrid_embeddings.py")
        return False
    
    # 1. 处理实体嵌入
    print("\n[1] 处理实体嵌入...")
    
    # 加载混合嵌入（字典格式）
    entity_hybrid_emb = pickle.load(open(Config.ENTITY_HYBRID_EMBEDDING_PATH, 'rb'))
    print(f"  已加载 {len(entity_hybrid_emb)} 个实体的混合嵌入")
    
    # 加载实体映射
    entity2idx = pickle.load(open(Config.ENTITY2IDX_PATH, 'rb'))
    print(f"  已加载实体映射: {len(entity2idx)} 个实体")
    
    # 转换为矩阵格式（按ID顺序）
    num_entities = len(entity2idx)
    embedding_dim = list(entity_hybrid_emb.values())[0].shape[0]
    
    entity_embeddings_matrix = np.zeros((num_entities, embedding_dim), dtype=np.float32)
    
    for entity, idx in entity2idx.items():
        if entity in entity_hybrid_emb:
            entity_embeddings_matrix[idx] = entity_hybrid_emb[entity]
        else:
            # 如果实体不在混合嵌入中（不应该发生），使用零向量
            print(f"  警告: 实体 '{entity}' 不在混合嵌入中")
    
    # 保存实体嵌入
    entity_emb_data = {
        'embeddings': entity_embeddings_matrix,
        'num_entities': num_entities,
        'embedding_dim': embedding_dim,
        'entity2id': entity2idx  # 保留映射以便查询
    }
    
    output_entity_path = os.path.join(Config.HYBRID_EMBEDDINGS_DIR, 'entity_embeddings_rgcn.pkl')
    pickle.dump(entity_emb_data, open(output_entity_path, 'wb'))
    print(f"  实体嵌入已保存到: {output_entity_path}")
    print(f"  形状: {entity_embeddings_matrix.shape}")
    
    # 2. 处理关系嵌入
    print("\n[2] 处理关系嵌入...")
    
    # 加载混合嵌入（字典格式）
    relation_hybrid_emb = pickle.load(open(Config.RELATION_HYBRID_EMBEDDING_PATH, 'rb'))
    print(f"  已加载 {len(relation_hybrid_emb)} 个关系的混合嵌入")
    
    # 加载关系映射
    relation2idx = pickle.load(open(Config.RELATION2IDX_PATH, 'rb'))
    print(f"  已加载关系映射: {len(relation2idx)} 个关系")
    
    # 保存关系映射（RGCN只需要关系数量）
    relation_data = {
        'num_relations': len(relation2idx),
        'relation2id': relation2idx  # 保留映射以便查询
    }
    
    output_relation_path = os.path.join(Config.HYBRID_EMBEDDINGS_DIR, 'relation_mappings_rgcn.pkl')
    pickle.dump(relation_data, open(output_relation_path, 'wb'))
    print(f"  关系映射已保存到: {output_relation_path}")
    print(f"  关系数: {len(relation2idx)}")
    
    # 3. 统计信息
    print("\n" + "="*60)
    print("准备完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  实体嵌入: {output_entity_path}")
    print(f"    - 实体数: {num_entities}")
    print(f"    - 嵌入维度: {embedding_dim}")
    print(f"  关系映射: {output_relation_path}")
    print(f"    - 关系数: {len(relation2idx)}")
    
    # 配置信息
    print(f"\n✅ 嵌入文件已准备好，供RGCN训练使用")
    print(f"\n配置文件中已自动使用这些路径:")
    print(f"  ENTITY_EMBEDDING_RGCN_PATH = '{output_entity_path}'")
    print(f"  RELATION_MAPPING_RGCN_PATH = '{output_relation_path}'")
    
    return True


if __name__ == '__main__':
    success = prepare_embeddings_for_rgcn()
    if success:
        print("\n✓ 嵌入文件准备成功！")
        print("\n下一步: 运行训练脚本")
        print("  python train_rgcn_hotpotqa.py")
    else:
        print("\n✗ 嵌入文件准备失败")

