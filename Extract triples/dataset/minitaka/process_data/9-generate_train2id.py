#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mintaka 第 9 步：根据 triples.txt + entity2id/relation2id 生成 train2id.txt

参考：14-generate_train2id.py
输出格式（OpenKE 常用）：
- 第一行：训练三元组数量
- 后续每行：head_id tail_id relation_id
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def load_id_mapping(path: str) -> Dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            idx = parts[1].strip()
            if name:
                mapping[name] = idx
    return mapping


def maybe_skip_header(first_line: str) -> bool:
    header = first_line.strip().lower()
    return header == "head\ttail\trelation"


def parse_triples_txt(path: str) -> List[Tuple[str, str, str]]:
    triples = []
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline()
        if first and not maybe_skip_header(first):
            line = first.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 3:
                    triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return triples


def convert_to_id_triples(
    triples: List[Tuple[str, str, str]],
    entity2id: Dict[str, str],
    relation2id: Dict[str, str],
    strict: bool = False,
    deduplicate: bool = True,
):
    stats = defaultdict(int)
    missing_entity = defaultdict(int)
    missing_relation = defaultdict(int)
    id_triples = []
    seen = set()

    for h, t, r in triples:
        stats["raw_triples"] += 1

        if h not in entity2id:
            missing_entity[h] += 1
            stats["skipped_missing_head"] += 1
            if strict:
                raise KeyError(f"head 实体未找到：{h}")
            continue
        if t not in entity2id:
            missing_entity[t] += 1
            stats["skipped_missing_tail"] += 1
            if strict:
                raise KeyError(f"tail 实体未找到：{t}")
            continue
        if r not in relation2id:
            missing_relation[r] += 1
            stats["skipped_missing_relation"] += 1
            if strict:
                raise KeyError(f"relation 未找到：{r}")
            continue

        row = (entity2id[h], entity2id[t], relation2id[r])
        if deduplicate:
            if row in seen:
                stats["duplicate_id_triples"] += 1
                continue
            seen.add(row)
        id_triples.append(row)
        stats["kept_triples"] += 1

    return id_triples, stats, missing_entity, missing_relation


def save_train2id(path: str, id_triples: List[Tuple[str, str, str]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(id_triples)}\n")
        for h, t, r in id_triples:
            f.write(f"{h} {t} {r}\n")


def main():
    parser = argparse.ArgumentParser(description="生成 Mintaka TransE 训练文件 train2id.txt")
    parser.add_argument(
        "--triples", "-t",
        default="data/triples.txt",
        help="输入 triples.txt（默认 data/triples.txt）",
    )
    parser.add_argument(
        "--entity2id", "-e",
        default="data/entity2id.txt",
        help="实体映射文件（默认 data/entity2id.txt）",
    )
    parser.add_argument(
        "--relation2id", "-r",
        default="data/relation2id_deduplicated.txt",
        help="关系映射文件（默认 data/relation2id_deduplicated.txt）",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/train2id.txt",
        help="输出 train2id.txt（默认 data/train2id.txt）",
    )
    parser.add_argument(
        "--stats",
        default="data/train2id_stats.json",
        help="统计信息输出 JSON（默认 data/train2id_stats.json）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="发现缺失映射时立即报错退出（默认跳过并统计）",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="保留重复 ID 三元组（默认去重）",
    )
    args = parser.parse_args()

    for p in [args.triples, args.entity2id, args.relation2id]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"文件不存在：{p}")

    triples = parse_triples_txt(args.triples)
    entity2id = load_id_mapping(args.entity2id)
    relation2id = load_id_mapping(args.relation2id)

    id_triples, stats, missing_entity, missing_relation = convert_to_id_triples(
        triples=triples,
        entity2id=entity2id,
        relation2id=relation2id,
        strict=args.strict,
        deduplicate=(not args.keep_duplicates),
    )
    save_train2id(args.output, id_triples)

    stats_payload = {
        "triples_file": args.triples,
        "entity2id_file": args.entity2id,
        "relation2id_file": args.relation2id,
        "output_file": args.output,
        "strict_mode": args.strict,
        "deduplicate": not args.keep_duplicates,
        "entity_count": len(entity2id),
        "relation_count": len(relation2id),
        **stats,
        "missing_entity_top20": dict(
            sorted(missing_entity.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
        "missing_relation_top20": dict(
            sorted(missing_relation.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("Mintaka train2id.txt 生成完成")
    print("=" * 68)
    print(f"triples 输入: {args.triples}")
    print(f"entity2id: {args.entity2id}  ({len(entity2id)} 项)")
    print(f"relation2id: {args.relation2id}  ({len(relation2id)} 项)")
    print(f"raw triples: {stats.get('raw_triples', 0)}")
    print(f"kept triples: {stats.get('kept_triples', 0)}")
    print(f"缺失 head: {stats.get('skipped_missing_head', 0)}")
    print(f"缺失 tail: {stats.get('skipped_missing_tail', 0)}")
    print(f"缺失 relation: {stats.get('skipped_missing_relation', 0)}")
    print(f"ID 重复: {stats.get('duplicate_id_triples', 0)}")
    print(f"输出文件: {args.output}")
    print(f"统计文件: {args.stats}")
    print("=" * 68)


if __name__ == "__main__":
    main()
