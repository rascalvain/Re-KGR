"""准备RGAT所需的嵌入文件 - Mintaka版本"""

import pickle
import numpy as np
import os

from config_mintaka_rgat import Config


def prepare_embeddings_for_rgat():
    print("=" * 60)
    print("准备RGAT嵌入文件 - Mintaka")
    print("=" * 60)

    # 检查源文件是否存在
    for path, desc in [
        (Config.ENTITY_HYBRID_EMBEDDING_PATH, "实体混合嵌入"),
        (Config.RELATION_HYBRID_EMBEDDING_PATH, "关系混合嵌入"),
        (Config.ENTITY2IDX_PATH, "实体映射"),
        (Config.RELATION2IDX_PATH, "关系映射"),
    ]:
        if not os.path.exists(path):
            print(f"错误: {desc}文件不存在: {path}")
            print("请先运行: python generate_hybrid_embeddings.py")
            return False

    # 1. 处理实体嵌入
    print("\n[1] 处理实体嵌入...")

    entity_hybrid_emb = pickle.load(open(Config.ENTITY_HYBRID_EMBEDDING_PATH, 'rb'))
    entity2idx = pickle.load(open(Config.ENTITY2IDX_PATH, 'rb'))
    print(f"  已加载 {len(entity_hybrid_emb)} 个实体的混合嵌入")
    print(f"  已加载实体映射: {len(entity2idx)} 个实体")

    num_entities = len(entity2idx)
    embedding_dim = list(entity_hybrid_emb.values())[0].shape[0]

    entity_embeddings_matrix = np.zeros((num_entities, embedding_dim), dtype=np.float32)

    for entity, idx in entity2idx.items():
        if entity in entity_hybrid_emb:
            entity_embeddings_matrix[idx] = entity_hybrid_emb[entity]
        else:
            print(f"  警告: 实体 '{entity}' 不在混合嵌入中")

    entity_emb_data = {
        'embeddings': entity_embeddings_matrix,
        'num_entities': num_entities,
        'embedding_dim': embedding_dim,
        'entity2id': entity2idx
    }

    os.makedirs(Config.HYBRID_EMBEDDINGS_DIR, exist_ok=True)
    output_entity_path = Config.ENTITY_EMBEDDING_RGAT_PATH
    pickle.dump(entity_emb_data, open(output_entity_path, 'wb'))
    print(f"  实体嵌入已保存: {output_entity_path}")
    print(f"  形状: {entity_embeddings_matrix.shape}")

    # 2. 处理关系映射
    print("\n[2] 处理关系映射...")

    relation2idx = pickle.load(open(Config.RELATION2IDX_PATH, 'rb'))
    print(f"  已加载关系映射: {len(relation2idx)} 个关系")

    relation_data = {
        'num_relations': len(relation2idx),
        'relation2id': relation2idx
    }

    output_relation_path = Config.RELATION_MAPPING_RGAT_PATH
    pickle.dump(relation_data, open(output_relation_path, 'wb'))
    print(f"  关系映射已保存: {output_relation_path}")
    print(f"  关系数: {len(relation2idx)}")

    # 3. 统计
    print("\n" + "=" * 60)
    print("准备完成！")
    print("=" * 60)
    print(f"  实体数: {num_entities}")
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  关系数: {len(relation2idx)}")
    print(f"\n生成的文件:")
    print(f"  {output_entity_path}")
    print(f"  {output_relation_path}")

    return True


if __name__ == '__main__':
    success = prepare_embeddings_for_rgat()
    if success:
        print("\n嵌入文件准备成功！")
        print("\n下一步:")
        print("  1. 训练RGAT: python train_rgat_mintaka.py")
        print("  2. 训练分类器: python train_classifier.py --pretrained_model <path>")
    else:
        print("\n嵌入文件准备失败")
