"""
生成混合嵌入：TransE + SentenceTransformer
处理 OOV（Out-Of-Vocabulary）问题

对于KB中存在的实体/关系：embedding = Concat(TransE_Vector, SentenceTransformer_Vector)
对于KB中不存在的实体/关系（OOV）：embedding = Concat(Zero_Vector, SentenceTransformer_Vector)
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse


def load_kb_mappings(entity_file='./data/entity2id.txt', relation_file='./data/relation2id.txt'):
    """
    加载KB中的实体和关系映射

    Returns:
        entity2id: {entity_name: id}
        relation2id: {relation_name: id}
    """
    print("正在加载KB映射...")

    entity2id = {}
    if os.path.exists(entity_file):
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    entity2id[parts[0]] = int(parts[1])
        print(f"  KB中实体数: {len(entity2id)}")

    relation2id = {}
    if os.path.exists(relation_file):
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    relation2id[parts[0]] = int(parts[1])
        print(f"  KB中关系数: {len(relation2id)}")

    return entity2id, relation2id


def load_transe_embeddings(transe_dir='./output', format='pkl'):
    """
    加载 TransE 训练得到的嵌入

    Returns:
        ent_embeddings: numpy array (num_entities, transe_dim)
        rel_embeddings: numpy array (num_relations, transe_dim)
    """
    print("\n正在加载 TransE 嵌入...")

    if format == 'pkl':
        ent_file = os.path.join(transe_dir, 'ent_embeddings.pkl')
        rel_file = os.path.join(transe_dir, 'rel_embeddings.pkl')

        ent_embeddings = pickle.load(open(ent_file, 'rb'))
        rel_embeddings = pickle.load(open(rel_file, 'rb'))
    else:  # npy
        ent_file = os.path.join(transe_dir, 'ent_embeddings.npy')
        rel_file = os.path.join(transe_dir, 'rel_embeddings.npy')

        ent_embeddings = np.load(ent_file)
        rel_embeddings = np.load(rel_file)

    print(f"  实体 TransE 嵌入: {ent_embeddings.shape}")
    print(f"  关系 TransE 嵌入: {rel_embeddings.shape}")

    return ent_embeddings, rel_embeddings


def load_all_entities_relations_from_file(file_path):
    """
    从文件中加载所有需要嵌入的实体和关系
    支持多种格式：
    - 每行一个实体/关系
    - 三元组格式：head \t tail \t relation

    Returns:
        all_entities: set of entity names
        all_relations: set of relation names
    """
    print(f"\n正在从 {file_path} 加载实体和关系...")

    all_entities = set()
    all_relations = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')

            if len(parts) == 3:
                # 三元组格式：head \t tail \t relation
                head, tail, relation = parts
                all_entities.add(head)
                all_entities.add(tail)
                all_relations.add(relation)
            elif len(parts) == 1:
                # 单独的实体或关系（需要额外指定类型）
                all_entities.add(parts[0])

    print(f"  总实体数: {len(all_entities)}")
    print(f"  总关系数: {len(all_relations)}")

    return all_entities, all_relations


def load_all_entities_relations_from_lists(entity_file=None, relation_file=None):
    """
    从单独的实体和关系列表文件加载

    Args:
        entity_file: 实体列表文件，每行一个实体
        relation_file: 关系列表文件，每行一个关系

    Returns:
        all_entities: set of entity names
        all_relations: set of relation names
    """
    all_entities = set()
    all_relations = set()

    if entity_file and os.path.exists(entity_file):
        print(f"\n正在从 {entity_file} 加载实体...")
        with open(entity_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if entity:
                    all_entities.add(entity)
        print(f"  加载实体数: {len(all_entities)}")

    if relation_file and os.path.exists(relation_file):
        print(f"\n正在从 {relation_file} 加载关系...")
        with open(relation_file, 'r', encoding='utf-8') as f:
            for line in f:
                relation = line.strip()
                if relation:
                    all_relations.add(relation)
        print(f"  加载关系数: {len(all_relations)}")

    return all_entities, all_relations


def generate_sentence_embeddings(texts, model_name='/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/sentence-bert',
                                 batch_size=32):
    """
    使用 SentenceTransformer 生成文本嵌入

    Args:
        texts: list of strings
        model_name: SentenceTransformer 模型名称或本地路径
        batch_size: batch size

    Returns:
        embeddings: numpy array (num_texts, embedding_dim)
    """
    print(f"\n正在加载 SentenceTransformer 模型: {model_name}")

    # 检查是否是本地路径
    if os.path.exists(model_name):
        print(f"  使用本地模型: {os.path.abspath(model_name)}")

    model = SentenceTransformer(model_name)

    print(f"正在生成 {len(texts)} 个文本的嵌入...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                             convert_to_numpy=True)

    print(f"  SentenceTransformer 嵌入维度: {embeddings.shape[1]}")

    return embeddings


def generate_hybrid_embeddings(all_entities, all_relations,
                               entity2id, relation2id,
                               transe_ent_embeddings, transe_rel_embeddings,
                               sentence_model='sentence-transformers/all-MiniLM-L6-v2',
                               batch_size=32):
    """
    生成混合嵌入

    Returns:
        entity_hybrid_embeddings: dict {entity_name: hybrid_embedding}
        relation_hybrid_embeddings: dict {relation_name: hybrid_embedding}
        stats: dict with statistics
    """
    print("\n" + "="*60)
    print("生成混合嵌入")
    print("="*60)

    # 1. 生成 SentenceTransformer 嵌入
    print("\n[步骤 1] 生成 SentenceTransformer 嵌入")
    entities_list = sorted(list(all_entities))
    relations_list = sorted(list(all_relations))

    entity_sent_embeddings = generate_sentence_embeddings(
        entities_list, model_name=sentence_model, batch_size=batch_size
    )

    relation_sent_embeddings = generate_sentence_embeddings(
        relations_list, model_name=sentence_model, batch_size=batch_size
    )

    sent_dim = entity_sent_embeddings.shape[1]
    transe_dim = transe_ent_embeddings.shape[1] if len(transe_ent_embeddings) > 0 else 100
    hybrid_dim = transe_dim + sent_dim

    print(f"\nTransE 维度: {transe_dim}")
    print(f"SentenceTransformer 维度: {sent_dim}")
    print(f"混合嵌入维度: {hybrid_dim}")

    # 2. 生成实体混合嵌入
    print("\n[步骤 2] 生成实体混合嵌入")
    entity_hybrid_embeddings = {}
    kb_entities = 0
    oov_entities = 0

    zero_vector = np.zeros(transe_dim)

    for idx, entity in enumerate(tqdm(entities_list, desc="处理实体")):
        sent_emb = entity_sent_embeddings[idx]

        if entity in entity2id:
            # 在KB中：TransE + SentenceTransformer
            entity_id = entity2id[entity]
            transe_emb = transe_ent_embeddings[entity_id]
            kb_entities += 1
        else:
            # OOV：Zero + SentenceTransformer
            transe_emb = zero_vector
            oov_entities += 1

        # 拼接
        hybrid_emb = np.concatenate([transe_emb, sent_emb])
        entity_hybrid_embeddings[entity] = hybrid_emb

    print(f"  KB实体: {kb_entities}")
    print(f"  OOV实体: {oov_entities}")

    # 3. 生成关系混合嵌入
    print("\n[步骤 3] 生成关系混合嵌入")
    relation_hybrid_embeddings = {}
    kb_relations = 0
    oov_relations = 0

    for idx, relation in enumerate(tqdm(relations_list, desc="处理关系")):
        sent_emb = relation_sent_embeddings[idx]

        if relation in relation2id:
            # 在KB中：TransE + SentenceTransformer
            relation_id = relation2id[relation]
            transe_emb = transe_rel_embeddings[relation_id]
            kb_relations += 1
        else:
            # OOV：Zero + SentenceTransformer
            transe_emb = zero_vector
            oov_relations += 1

        # 拼接
        hybrid_emb = np.concatenate([transe_emb, sent_emb])
        relation_hybrid_embeddings[relation] = hybrid_emb

    print(f"  KB关系: {kb_relations}")
    print(f"  OOV关系: {oov_relations}")

    # 统计信息
    stats = {
        'total_entities': len(all_entities),
        'kb_entities': kb_entities,
        'oov_entities': oov_entities,
        'total_relations': len(all_relations),
        'kb_relations': kb_relations,
        'oov_relations': oov_relations,
        'transe_dim': transe_dim,
        'sentence_dim': sent_dim,
        'hybrid_dim': hybrid_dim
    }

    return entity_hybrid_embeddings, relation_hybrid_embeddings, stats


def save_hybrid_embeddings(entity_embeddings, relation_embeddings, stats, output_dir='./hybrid_embeddings'):
    """
    保存混合嵌入
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("保存混合嵌入")
    print("="*60)

    # 保存为 pickle 格式
    entity_file = os.path.join(output_dir, 'entity_hybrid_embeddings.pkl')
    relation_file = os.path.join(output_dir, 'relation_hybrid_embeddings.pkl')
    stats_file = os.path.join(output_dir, 'embedding_stats.pkl')

    pickle.dump(entity_embeddings, open(entity_file, 'wb'))
    pickle.dump(relation_embeddings, open(relation_file, 'wb'))
    pickle.dump(stats, open(stats_file, 'wb'))

    print(f"实体混合嵌入已保存: {entity_file}")
    print(f"关系混合嵌入已保存: {relation_file}")
    print(f"统计信息已保存: {stats_file}")

    # 同时保存为 numpy 格式（矩阵形式，带索引映射）
    entities_list = sorted(list(entity_embeddings.keys()))
    relations_list = sorted(list(relation_embeddings.keys()))

    entity_matrix = np.array([entity_embeddings[e] for e in entities_list])
    relation_matrix = np.array([relation_embeddings[r] for r in relations_list])

    entity2idx = {e: idx for idx, e in enumerate(entities_list)}
    relation2idx = {r: idx for idx, r in enumerate(relations_list)}

    np.save(os.path.join(output_dir, 'entity_hybrid_embeddings.npy'), entity_matrix)
    np.save(os.path.join(output_dir, 'relation_hybrid_embeddings.npy'), relation_matrix)

    pickle.dump(entity2idx, open(os.path.join(output_dir, 'entity2idx.pkl'), 'wb'))
    pickle.dump(relation2idx, open(os.path.join(output_dir, 'relation2idx.pkl'), 'wb'))

    print(f"实体嵌入矩阵已保存: {os.path.join(output_dir, 'entity_hybrid_embeddings.npy')}")
    print(f"关系嵌入矩阵已保存: {os.path.join(output_dir, 'relation_hybrid_embeddings.npy')}")

    # 保存文本格式的映射
    with open(os.path.join(output_dir, 'entity2idx.txt'), 'w', encoding='utf-8') as f:
        for entity, idx in entity2idx.items():
            f.write(f"{entity}\t{idx}\n")

    with open(os.path.join(output_dir, 'relation2idx.txt'), 'w', encoding='utf-8') as f:
        for relation, idx in relation2idx.items():
            f.write(f"{relation}\t{idx}\n")

    print(f"实体映射已保存: {os.path.join(output_dir, 'entity2idx.txt')}")
    print(f"关系映射已保存: {os.path.join(output_dir, 'relation2idx.txt')}")


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    print(f"实体:")
    print(f"  总数: {stats['total_entities']}")
    print(f"  KB中: {stats['kb_entities']} ({stats['kb_entities']/stats['total_entities']*100:.2f}%)")
    print(f"  OOV: {stats['oov_entities']} ({stats['oov_entities']/stats['total_entities']*100:.2f}%)")
    print(f"\n关系:")
    print(f"  总数: {stats['total_relations']}")
    print(f"  KB中: {stats['kb_relations']} ({stats['kb_relations']/stats['total_relations']*100:.2f}%)")
    print(f"  OOV: {stats['oov_relations']} ({stats['oov_relations']/stats['total_relations']*100:.2f}%)")
    print(f"\n嵌入维度:")
    print(f"  TransE: {stats['transe_dim']}")
    print(f"  SentenceTransformer: {stats['sentence_dim']}")
    print(f"  混合嵌入: {stats['hybrid_dim']}")


def main():
    parser = argparse.ArgumentParser(description='生成混合嵌入 (TransE + SentenceTransformer)')

    # 输入文件
    parser.add_argument('--kb_entity_file', type=str, default='entity2id.txt',
                       help='KB实体映射文件')
    parser.add_argument('--kb_relation_file', type=str, default='relation2id.txt',
                       help='KB关系映射文件')
    parser.add_argument('--transe_dir', type=str, default='./output',
                       help='TransE嵌入目录')
    parser.add_argument('--transe_format', type=str, default='pkl', choices=['pkl', 'npy'],
                       help='TransE嵌入格式')

    # 需要嵌入的实体和关系来源
    parser.add_argument('--triple_file', type=str, default='triples.txt',
                       help='三元组文件（格式：head\\ttail\\trelation）')
    parser.add_argument('--entity_file', type=str, default='',
                       help='额外的实体列表文件（可选）')
    parser.add_argument('--relation_file', type=str, default='',
                       help='额外的关系列表文件（可选）')

    # SentenceTransformer 配置
    parser.add_argument('--sentence_model', type=str,
                       default='/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/sentence-bert',
                       help='SentenceTransformer模型名称或本地路径（默认使用本地sentence-bert）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for SentenceTransformer')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='./hybrid_embeddings',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("混合嵌入生成器 (TransE + SentenceTransformer)")
    print("="*60)
    
    # 1. 加载 KB 映射
    entity2id, relation2id = load_kb_mappings(args.kb_entity_file, args.kb_relation_file)
    
    # 2. 加载 TransE 嵌入
    transe_ent_embeddings, transe_rel_embeddings = load_transe_embeddings(
        args.transe_dir, args.transe_format
    )
    
    # 3. 加载所有需要嵌入的实体和关系
    all_entities, all_relations = set(), set()
    
    # 从三元组文件加载
    if os.path.exists(args.triple_file):
        entities, relations = load_all_entities_relations_from_file(args.triple_file)
        all_entities.update(entities)
        all_relations.update(relations)
    
    # 从单独的列表文件加载（如果提供）
    if args.entity_file or args.relation_file:
        entities, relations = load_all_entities_relations_from_lists(
            args.entity_file, args.relation_file
        )
        all_entities.update(entities)
        all_relations.update(relations)
    
    # 如果没有提供额外文件，使用KB中的所有实体和关系
    if not all_entities and not all_relations:
        print("\n未指定实体和关系来源，使用KB中的所有实体和关系...")
        all_entities = set(entity2id.keys())
        all_relations = set(relation2id.keys())
        print(f"  使用 {len(all_entities)} 个实体")
        print(f"  使用 {len(all_relations)} 个关系")
    
    # 4. 生成混合嵌入
    entity_hybrid_embeddings, relation_hybrid_embeddings, stats = generate_hybrid_embeddings(
        all_entities, all_relations,
        entity2id, relation2id,
        transe_ent_embeddings, transe_rel_embeddings,
        sentence_model=args.sentence_model,
        batch_size=args.batch_size
    )
    
    # 5. 保存结果
    save_hybrid_embeddings(entity_hybrid_embeddings, relation_hybrid_embeddings, 
                          stats, args.output_dir)
    
    # 6. 打印统计信息
    print_statistics(stats)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print(f"\n生成的文件位于: {args.output_dir}/")
    print(f"  - entity_hybrid_embeddings.pkl  (字典格式)")
    print(f"  - relation_hybrid_embeddings.pkl  (字典格式)")
    print(f"  - entity_hybrid_embeddings.npy  (矩阵格式)")
    print(f"  - relation_hybrid_embeddings.npy  (矩阵格式)")
    print(f"  - entity2idx.txt / entity2idx.pkl  (实体映射)")
    print(f"  - relation2idx.txt / relation2idx.pkl  (关系映射)")
    print(f"  - embedding_stats.pkl  (统计信息)")


if __name__ == '__main__':
    main()

