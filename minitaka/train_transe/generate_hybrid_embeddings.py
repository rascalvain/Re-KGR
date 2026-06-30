#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成混合嵌入：TransE + SentenceTransformer

策略：
  KB 内实体/关系 → Concat(TransE_Vector, SentenceTransformer_Vector)
  OOV 实体/关系  → Concat(Zero_Vector,   SentenceTransformer_Vector)
"""

import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def load_id_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_transe_embeddings(transe_dir, fmt="pkl"):
    if fmt == "pkl":
        ent = pickle.load(open(os.path.join(transe_dir, "ent_embeddings.pkl"), "rb"))
        rel = pickle.load(open(os.path.join(transe_dir, "rel_embeddings.pkl"), "rb"))
    else:
        ent = np.load(os.path.join(transe_dir, "ent_embeddings.npy"))
        rel = np.load(os.path.join(transe_dir, "rel_embeddings.npy"))
    print(f"TransE 实体嵌入: {ent.shape}  关系嵌入: {rel.shape}")
    return ent, rel


def load_all_from_triples(path):
    entities, relations = set(), set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx == 0 and line.lower().startswith("head"):
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                entities.add(parts[0].strip())
                entities.add(parts[1].strip())
                relations.add(parts[2].strip())
    return entities, relations


def generate_hybrid(all_names, name2id, transe_emb, sent_emb_map, transe_dim):
    """为一组名称生成混合嵌入，返回 {name: hybrid_vector}。"""
    zero = np.zeros(transe_dim)
    result = {}
    kb, oov = 0, 0
    for name in all_names:
        s_emb = sent_emb_map[name]
        if name in name2id:
            t_emb = transe_emb[name2id[name]]
            kb += 1
        else:
            t_emb = zero
            oov += 1
        result[name] = np.concatenate([t_emb, s_emb])
    return result, kb, oov


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    preprocess_data = os.path.join("/root/autodl-fs/gca/mintaka/preprocess_data", "data")

    parser = argparse.ArgumentParser(
        description="生成混合嵌入 (TransE + SentenceTransformer)"
    )
    parser.add_argument("--entity2id",
                        default=os.path.join(preprocess_data, "entity2id.txt"))
    parser.add_argument("--relation2id",
                        default=os.path.join(preprocess_data, "relation2id_deduplicated.txt"))
    parser.add_argument("--triples",
                        default=os.path.join(preprocess_data, "triples.txt"))
    parser.add_argument("--transe_dir",
                        default=os.path.join(base, "output"))
    parser.add_argument("--transe_fmt", default="pkl", choices=["pkl", "npy"])
    parser.add_argument("--sentence_model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer 模型名或本地路径")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir",
                        default=os.path.join(base, "hybrid_embeddings"))
    args = parser.parse_args()

    print("=" * 60)
    print("混合嵌入生成器 (TransE + SentenceTransformer)")
    print("=" * 60)

    entity2id = load_id_mapping(args.entity2id)
    relation2id = load_id_mapping(args.relation2id)
    transe_ent, transe_rel = load_transe_embeddings(args.transe_dir, args.transe_fmt)
    transe_dim = transe_ent.shape[1]

    all_entities, all_relations = set(), set()
    if os.path.exists(args.triples):
        ents, rels = load_all_from_triples(args.triples)
        all_entities.update(ents)
        all_relations.update(rels)
    if not all_entities:
        all_entities = set(entity2id.keys())
        all_relations = set(relation2id.keys())

    print(f"待嵌入实体: {len(all_entities)}  关系: {len(all_relations)}")

    # SentenceTransformer 编码
    print(f"\n加载 SentenceTransformer: {args.sentence_model}")

    # 兼容旧版 transformers：对 position_embeddings 尺寸不匹配的情况做 patch
    from transformers import AutoModel as _AutoModel
    _orig_from_pretrained = _AutoModel.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, *args_inner, **kwargs_inner):
        kwargs_inner.setdefault("ignore_mismatched_sizes", True)
        return _orig_from_pretrained(cls, *args_inner, **kwargs_inner)

    _AutoModel.from_pretrained = _patched_from_pretrained

    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(args.sentence_model)

    ent_list = sorted(all_entities)
    rel_list = sorted(all_relations)
    ent_sent = st_model.encode(ent_list, batch_size=args.batch_size, show_progress_bar=True)
    rel_sent = st_model.encode(rel_list, batch_size=args.batch_size, show_progress_bar=True)

    ent_sent_map = dict(zip(ent_list, ent_sent))
    rel_sent_map = dict(zip(rel_list, rel_sent))

    sent_dim = ent_sent.shape[1]
    hybrid_dim = transe_dim + sent_dim
    print(f"TransE dim={transe_dim}  Sentence dim={sent_dim}  Hybrid dim={hybrid_dim}")

    ent_hybrid, ent_kb, ent_oov = generate_hybrid(
        all_entities, entity2id, transe_ent, ent_sent_map, transe_dim
    )
    print(f"实体: KB={ent_kb}  OOV={ent_oov}")

    rel_hybrid, rel_kb, rel_oov = generate_hybrid(
        all_relations, relation2id, transe_rel, rel_sent_map, transe_dim
    )
    print(f"关系: KB={rel_kb}  OOV={rel_oov}")

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)

    pickle.dump(ent_hybrid, open(os.path.join(args.output_dir, "entity_hybrid_embeddings.pkl"), "wb"))
    pickle.dump(rel_hybrid, open(os.path.join(args.output_dir, "relation_hybrid_embeddings.pkl"), "wb"))

    ent_matrix = np.array([ent_hybrid[e] for e in ent_list])
    rel_matrix = np.array([rel_hybrid[r] for r in rel_list])
    np.save(os.path.join(args.output_dir, "entity_hybrid_embeddings.npy"), ent_matrix)
    np.save(os.path.join(args.output_dir, "relation_hybrid_embeddings.npy"), rel_matrix)

    ent2idx = {e: i for i, e in enumerate(ent_list)}
    rel2idx = {r: i for i, r in enumerate(rel_list)}
    pickle.dump(ent2idx, open(os.path.join(args.output_dir, "entity2idx.pkl"), "wb"))
    pickle.dump(rel2idx, open(os.path.join(args.output_dir, "relation2idx.pkl"), "wb"))

    with open(os.path.join(args.output_dir, "entity2idx.txt"), "w", encoding="utf-8") as f:
        for e, i in ent2idx.items():
            f.write(f"{e}\t{i}\n")
    with open(os.path.join(args.output_dir, "relation2idx.txt"), "w", encoding="utf-8") as f:
        for r, i in rel2idx.items():
            f.write(f"{r}\t{i}\n")

    stats = {
        "total_entities": len(all_entities), "kb_entities": ent_kb, "oov_entities": ent_oov,
        "total_relations": len(all_relations), "kb_relations": rel_kb, "oov_relations": rel_oov,
        "transe_dim": transe_dim, "sentence_dim": sent_dim, "hybrid_dim": hybrid_dim,
    }
    pickle.dump(stats, open(os.path.join(args.output_dir, "embedding_stats.pkl"), "wb"))

    print(f"\n混合嵌入已保存到: {args.output_dir}/")
    print(f"  entity_hybrid_embeddings .pkl / .npy")
    print(f"  relation_hybrid_embeddings .pkl / .npy")
    print(f"  entity2idx / relation2idx .txt / .pkl")
    print("完成！")


if __name__ == "__main__":
    main()
