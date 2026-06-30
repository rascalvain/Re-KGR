#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mintaka TransE 训练脚本

读取 preprocess_data 产出的数据，转换为 OpenKE 格式后训练 TransE，
并导出实体/关系嵌入向量。

输入文件（来自 preprocess_data/data/）：
  - entity2id.txt                 name\tid（无首行 count）
  - relation2id_deduplicated.txt  name\tid（无首行 count）
  - triples.txt                   head\ttail\trelation（含表头行）

输出文件：
  openke_data/                    OpenKE 格式数据
  output/                         模型与嵌入
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from openke.config import Trainer, Tester
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.loss import MarginLoss
from openke.module.model import TransE
from openke.module.strategy import NegativeSampling


# ================================================================
# 1. 数据读取
# ================================================================

def load_id_mapping(path):
    """读取 name\\tid 映射文件（无首行 count），返回 {name: int(id)}。"""
    mapping = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                mapping[parts[0].strip()] = int(parts[1].strip())
    return mapping


def load_triples(path, entity2id, relation2id):
    """
    读取 triples.txt（head\\ttail\\trelation），跳过表头行，
    返回 [(head_id, tail_id, relation_id), ...]。
    """
    triples = []
    skipped = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx == 0 and line.lower().startswith("head"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            head, tail, relation = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if head in entity2id and tail in entity2id and relation in relation2id:
                triples.append((entity2id[head], entity2id[tail], relation2id[relation]))
            else:
                skipped += 1
                if skipped <= 5:
                    print(f"  跳过三元组 (行{idx + 1}): {head} | {tail} | {relation}")
    if skipped > 0:
        print(f"共跳过 {skipped} 个无法映射的三元组")
    return triples


# ================================================================
# 2. 数据格式转换（→ OpenKE）
# ================================================================

def write_openke_entity2id(entity2id, path):
    """写入 OpenKE entity2id.txt：首行 count，之后 name\\tid。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(entity2id)}\n")
        for name, eid in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{eid}\n")


def write_openke_relation2id(relation2id, path):
    """写入 OpenKE relation2id.txt：首行 count，之后 name\\tid。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(relation2id)}\n")
        for name, rid in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{rid}\n")


def write_openke_train2id(triples, path):
    """写入 OpenKE train2id.txt：首行 count，之后 h\\tt\\tr（tab 分隔）。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(triples)}\n")
        for h, t, r in triples:
            f.write(f"{h}\t{t}\t{r}\n")


def write_empty_split(path):
    """写入空的 valid2id.txt / test2id.txt 占位文件。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("0\n")


def prepare_openke_data(entity2id, relation2id, triples, datadir):
    """将原始数据转换为 OpenKE 五件套，写入 datadir。"""
    os.makedirs(datadir, exist_ok=True)
    write_openke_entity2id(entity2id, os.path.join(datadir, "entity2id.txt"))
    write_openke_relation2id(relation2id, os.path.join(datadir, "relation2id.txt"))
    write_openke_train2id(triples, os.path.join(datadir, "train2id.txt"))
    write_empty_split(os.path.join(datadir, "valid2id.txt"))
    write_empty_split(os.path.join(datadir, "test2id.txt"))

    print(f"OpenKE 数据已写入 {datadir}/")
    print(f"  entity2id.txt   : {len(entity2id)} 个实体")
    print(f"  relation2id.txt : {len(relation2id)} 个关系")
    print(f"  train2id.txt    : {len(triples)} 条三元组")
    print(f"  valid2id.txt    : 0（占位）")
    print(f"  test2id.txt     : 0（占位）")


# ================================================================
# 3. TransE 训练
# ================================================================

def train(args, datadir, outdir):
    """加载 OpenKE 格式数据 → 训练 TransE → 返回模型。"""
    print("\n" + "=" * 60)
    print("加载数据 & 初始化 TransE")
    print("=" * 60)

    train_dataloader = TrainDataLoader(
        in_path=datadir + "/",
        batch_size=args.batch_size,
        threads=args.threads,
        sampling_mode="normal",
        bern_flag=args.bern_flag,
        filter_flag=1,
        neg_ent=args.neg_ent,
        neg_rel=0,
    )

    print(f"实体总数:     {train_dataloader.get_ent_tot()}")
    print(f"关系总数:     {train_dataloader.get_rel_tot()}")
    print(f"训练三元组数: {train_dataloader.get_triple_tot()}")

    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        p_norm=1,
        norm_flag=True,
    )

    if args.model_path and os.path.exists(args.model_path):
        print(f"加载预训练模型: {args.model_path}")
        transe.load_state_dict(
            torch.load(args.model_path, map_location=torch.device("cpu"))
        )

    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=args.margin),
        batch_size=train_dataloader.get_batch_size(),
    )

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)
    print(f"  dim={args.dim}  epochs={args.epoch}  lr={args.lr}")
    print(f"  margin={args.margin}  neg_ent={args.neg_ent}  bern={args.bern_flag}")
    print(f"  batch_size={args.batch_size}  save_steps={args.save_steps}")
    print(f"  GPU: {'是' if torch.cuda.is_available() else '否（CPU）'}")

    os.makedirs(outdir, exist_ok=True)

    try:
        trainer = Trainer(
            model=model,
            data_loader=train_dataloader,
            train_times=args.epoch,
            alpha=args.lr,
            use_gpu=torch.cuda.is_available(),
            save_steps=args.save_steps,
            checkpoint_dir=outdir,
        )
    except TypeError:
        print("检测到旧版 OpenKE，使用基础参数...")
        trainer = Trainer(
            model=model,
            data_loader=train_dataloader,
            train_times=args.epoch,
            alpha=args.lr,
            use_gpu=torch.cuda.is_available(),
        )

    trainer.run()

    ckpt_path = os.path.join(outdir, "transe_final.ckpt")
    transe.save_checkpoint(ckpt_path)
    print(f"\n模型已保存: {ckpt_path}")

    if args.test:
        print("\n" + "=" * 60)
        print("链接预测评估")
        print("=" * 60)
        test_dataloader = TestDataLoader(
            in_path=datadir + "/", sampling_mode="link"
        )
        transe.load_checkpoint(ckpt_path)
        tester = Tester(
            model=transe,
            data_loader=test_dataloader,
            use_gpu=torch.cuda.is_available(),
        )
        tester.run_link_prediction(type_constrain=False)

    return transe


# ================================================================
# 4. 嵌入导出
# ================================================================

def save_embeddings(transe, outdir):
    """从训练好的 TransE 导出实体/关系嵌入，同时保存 pkl 和 npy。"""
    print("\n" + "=" * 60)
    print("提取并保存嵌入向量")
    print("=" * 60)

    ent_emb = transe.ent_embeddings.weight.data.cpu().detach().numpy()
    rel_emb = transe.rel_embeddings.weight.data.cpu().detach().numpy()

    print(f"实体嵌入 shape: {ent_emb.shape}")
    print(f"关系嵌入 shape: {rel_emb.shape}")

    for name, arr in [("ent_embeddings", ent_emb), ("rel_embeddings", rel_emb)]:
        npy_path = os.path.join(outdir, f"{name}.npy")
        pkl_path = os.path.join(outdir, f"{name}.pkl")
        np.save(npy_path, arr)
        with open(pkl_path, "wb") as f:
            pickle.dump(arr, f)
        print(f"  {name}: {npy_path}, {pkl_path}")

    return ent_emb, rel_emb


# ================================================================
# 5. 主流程
# ================================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    preprocess_data = os.path.join("/root/autodl-fs/gca/mintaka/preprocess_data", "data")

    parser = argparse.ArgumentParser(description="Mintaka TransE 训练")

    # 输入（默认指向 preprocess_data/data/）
    parser.add_argument("--entity2id",
                        default=os.path.join(preprocess_data, "entity2id.txt"),
                        help="实体映射文件（name\\tid，无首行 count）")
    parser.add_argument("--relation2id",
                        default=os.path.join(preprocess_data, "relation2id_deduplicated.txt"),
                        help="关系映射文件（name\\tid，无首行 count）")
    parser.add_argument("--triples",
                        default=os.path.join(preprocess_data, "triples.txt"),
                        help="三元组文件（head\\ttail\\trelation）")

    # 目录
    parser.add_argument("--datadir",
                        default=os.path.join(base, "openke_data"),
                        help="OpenKE 格式数据输出目录")
    parser.add_argument("--outdir",
                        default=os.path.join(base, "output"),
                        help="模型与嵌入输出目录")

    # 模型超参
    parser.add_argument("--dim",         type=int,   default=100,  help="嵌入维度")
    parser.add_argument("--epoch",       type=int,   default=1000, help="训练轮数")
    parser.add_argument("--batch_size",  type=int,   default=300,  help="batch size")
    parser.add_argument("--lr",          type=float, default=1.0,  help="学习率")
    parser.add_argument("--margin",      type=float, default=5.0,  help="margin 损失")
    parser.add_argument("--neg_ent",     type=int,   default=64,   help="负采样实体数量")
    parser.add_argument("--bern_flag",   type=int,   default=0,    help="1=伯努利负采样")
    parser.add_argument("--save_steps",  type=int,   default=100,  help="保存间隔 epoch")
    parser.add_argument("--threads",     type=int,   default=8,    help="数据加载线程数")

    # 可选
    parser.add_argument("--model_path",  default="",
                        help="预训练模型路径（可选，用于继续训练）")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="跳过数据准备（openke_data/ 已存在时使用）")
    parser.add_argument("--test",        action="store_true",
                        help="训练后运行链接预测评估")

    args = parser.parse_args()

    # ── 数据准备 ──
    if not args.skip_prepare:
        print("=" * 60)
        print("步骤 1：读取 preprocess_data 输出 → 转换为 OpenKE 格式")
        print("=" * 60)

        for p in [args.entity2id, args.relation2id, args.triples]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"输入文件不存在：{p}")

        entity2id = load_id_mapping(args.entity2id)
        relation2id = load_id_mapping(args.relation2id)
        triples = load_triples(args.triples, entity2id, relation2id)

        print(f"实体数: {len(entity2id)}  关系数: {len(relation2id)}  三元组数: {len(triples)}")

        prepare_openke_data(entity2id, relation2id, triples, args.datadir)

        stats = {
            "entity2id_file": args.entity2id,
            "relation2id_file": args.relation2id,
            "triples_file": args.triples,
            "entities": len(entity2id),
            "relations": len(relation2id),
            "triples": len(triples),
        }
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "data_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── 训练 ──
    transe = train(args, args.datadir, args.outdir)

    # ── 嵌入导出 ──
    ent_emb, rel_emb = save_embeddings(transe, args.outdir)

    # ── 示例输出 ──
    if not args.skip_prepare:
        id2entity = {v: k for k, v in entity2id.items()}
        id2relation = {v: k for k, v in relation2id.items()}
        print("\n前 5 个实体嵌入（前 10 维）:")
        for i in range(min(5, len(ent_emb))):
            name = id2entity.get(i, f"Entity_{i}")
            print(f"  [{i}] {name[:50]}: {ent_emb[i][:10]}")
        print("\n前 5 个关系嵌入（前 10 维）:")
        for i in range(min(5, len(rel_emb))):
            name = id2relation.get(i, f"Relation_{i}")
            print(f"  [{i}] {name[:50]}: {rel_emb[i][:10]}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"输出目录: {args.outdir}")
    print(f"  transe_final.ckpt")
    print(f"  ent_embeddings.pkl / .npy")
    print(f"  rel_embeddings.pkl / .npy")


if __name__ == "__main__":
    main()
