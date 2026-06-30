#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载和使用 TransE 嵌入的示例代码

功能：
  - 加载训练好的嵌入
  - 查找相似实体
  - 链接预测（给定 head + relation，预测 tail）
  - 三元组评分
  - 导出可读格式
"""

import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(output_dir="./output", fmt="pkl"):
    if fmt == "pkl":
        ent = pickle.load(open(os.path.join(output_dir, "ent_embeddings.pkl"), "rb"))
        rel = pickle.load(open(os.path.join(output_dir, "rel_embeddings.pkl"), "rb"))
    else:
        ent = np.load(os.path.join(output_dir, "ent_embeddings.npy"))
        rel = np.load(os.path.join(output_dir, "rel_embeddings.npy"))
    print(f"实体嵌入: {ent.shape}  关系嵌入: {rel.shape}")
    return ent, rel


def load_id_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def find_similar_entities(entity_name, entity2id, id2entity, ent_emb, top_k=10):
    if entity_name not in entity2id:
        print(f"实体 '{entity_name}' 不存在")
        return []
    eid = entity2id[entity_name]
    sims = cosine_similarity(ent_emb[eid].reshape(1, -1), ent_emb)[0]
    indices = np.argsort(sims)[::-1][1 : top_k + 1]
    return [(id2entity[i], sims[i]) for i in indices]


def predict_tail(head, relation, entity2id, relation2id, id2entity, ent_emb, rel_emb, top_k=10):
    if head not in entity2id or relation not in relation2id:
        print(f"head='{head}' 或 relation='{relation}' 不存在")
        return []
    predicted = ent_emb[entity2id[head]] + rel_emb[relation2id[relation]]
    dists = np.linalg.norm(ent_emb - predicted, axis=1)
    indices = np.argsort(dists)[:top_k]
    return [(id2entity[i], dists[i]) for i in indices]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TransE 嵌入使用示例")
    parser.add_argument("--outdir", default="./output")
    parser.add_argument("--entity2id", default="/root/autodl-fs/gca/mintaka/preprocess_data/data/entity2id.txt")
    parser.add_argument("--relation2id", default="/root/autodl-fs/gca/mintaka/preprocess_data/data/relation2id_deduplicated.txt")
    parser.add_argument("--fmt", default="pkl", choices=["pkl", "npy"])
    args = parser.parse_args()

    ent_emb, rel_emb = load_embeddings(args.outdir, args.fmt)
    entity2id = load_id_mapping(args.entity2id)
    relation2id = load_id_mapping(args.relation2id)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # 示例 1：获取嵌入
    print("\n" + "=" * 60)
    print("示例 1：获取特定实体的嵌入向量")
    print("=" * 60)
    if entity2id:
        sample = list(entity2id.keys())[0]
        emb = ent_emb[entity2id[sample]]
        print(f"  实体: {sample}")
        print(f"  维度: {emb.shape}")
        print(f"  前 10 维: {emb[:10]}")

    # 示例 2：相似实体
    print("\n" + "=" * 60)
    print("示例 2：查找相似实体")
    print("=" * 60)
    if entity2id:
        sample = list(entity2id.keys())[0]
        print(f"  与 '{sample}' 最相似的 10 个实体:")
        for i, (name, sim) in enumerate(
            find_similar_entities(sample, entity2id, id2entity, ent_emb), 1
        ):
            print(f"    {i}. {name[:60]}  (cos={sim:.4f})")

    # 示例 3：链接预测
    print("\n" + "=" * 60)
    print("示例 3：链接预测（h + r → t）")
    print("=" * 60)
    if entity2id and relation2id:
        head = list(entity2id.keys())[0]
        rel = list(relation2id.keys())[0]
        print(f"  head: {head}")
        print(f"  relation: {rel}")
        print(f"  预测 tail Top 10:")
        for i, (name, dist) in enumerate(
            predict_tail(head, rel, entity2id, relation2id, id2entity, ent_emb, rel_emb), 1
        ):
            print(f"    {i}. {name[:60]}  (dist={dist:.4f})")

    # 示例 4：导出可读格式
    print("\n" + "=" * 60)
    print("示例 4：导出前 10 个实体嵌入到文本文件")
    print("=" * 60)
    out_file = os.path.join(args.outdir, "sample_embeddings.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(min(10, len(ent_emb))):
            name = id2entity.get(i, f"Entity_{i}")
            f.write(f"{name}\t{' '.join(map(str, ent_emb[i]))}\n")
    print(f"  已导出到: {out_file}")

    print("\n完成！")


if __name__ == "__main__":
    main()
