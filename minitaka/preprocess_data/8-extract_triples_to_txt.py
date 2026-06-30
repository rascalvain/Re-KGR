#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mintaka 第 8 步：从标准化后的 JSON 抽取三元组到 triples.txt

参考：13-extract_triples_to_txt.py
适配点：
- Mintaka 三元组为结构化字典：
  1) entity_triples.*.triples[]: {"head","relation","tail"}
  2) gpt_triples[]: {"head","relation","tail"}
- 支持选择来源（entity / gpt / both）
- 默认去重（按 head, tail, relation 三元组）

输出格式与参考脚本一致：
head<TAB>tail<TAB>relation
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_text(v) -> str:
    if not isinstance(v, str):
        return ""
    return v.strip()


def iter_entity_triples(item: dict) -> Iterable[dict]:
    entity_triples = item.get("entity_triples", {})
    if not isinstance(entity_triples, dict):
        return
    for info in entity_triples.values():
        if not isinstance(info, dict):
            continue
        triples = info.get("triples", [])
        if not isinstance(triples, list):
            continue
        for t in triples:
            if isinstance(t, dict):
                yield t


def iter_gpt_triples(item: dict) -> Iterable[dict]:
    gpt_triples = item.get("gpt_triples", [])
    if not isinstance(gpt_triples, list):
        return
    for t in gpt_triples:
        if isinstance(t, dict):
            yield t


def extract_triples(data: List[dict], source: str, deduplicate: bool = True):
    stats: Dict[str, int] = defaultdict(int)
    seen = set()
    rows: List[Tuple[str, str, str]] = []

    for item in data:
        stats["records"] += 1
        candidates = []

        if source in ("entity", "both"):
            candidates.extend(iter_entity_triples(item))
        if source in ("gpt", "both"):
            candidates.extend(iter_gpt_triples(item))

        for t in candidates:
            stats["raw_triples"] += 1
            h = normalize_text(t.get("head"))
            r = normalize_text(t.get("relation"))
            tail = normalize_text(t.get("tail"))

            if not h or not r or not tail:
                stats["invalid_triples"] += 1
                continue

            key = (h, tail, r)
            if deduplicate:
                if key in seen:
                    stats["duplicate_triples"] += 1
                    continue
                seen.add(key)

            rows.append(key)
            stats["kept_triples"] += 1

    return rows, stats


def save_triples_txt(rows: List[Tuple[str, str, str]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("head\ttail\trelation\n")
        for h, t, r in rows:
            f.write(f"{h}\t{t}\t{r}\n")


def main():
    parser = argparse.ArgumentParser(description="提取 Mintaka 三元组到 triples.txt（用于 TransE）")
    parser.add_argument(
        "--input", "-i",
        default="data/mintaka_dev_normalized_final.json",
        help="输入 JSON（默认 data/mintaka_dev_normalized_final.json）",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/triples.txt",
        help="输出 triples.txt（默认 data/triples.txt）",
    )
    parser.add_argument(
        "--source",
        choices=["entity", "gpt", "both"],
        default="both",
        help="三元组来源（默认 both）",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="保留重复三元组（默认去重）",
    )
    parser.add_argument(
        "--stats",
        default="data/triples_extract_stats.json",
        help="统计信息输出 JSON（默认 data/triples_extract_stats.json）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在：{args.input}")

    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list")

    rows, stats = extract_triples(
        data=data,
        source=args.source,
        deduplicate=(not args.keep_duplicates),
    )
    save_triples_txt(rows, args.output)

    stats_payload = {
        "input_file": args.input,
        "output_file": args.output,
        "source": args.source,
        "deduplicate": not args.keep_duplicates,
        **stats,
    }
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("Mintaka triples.txt 生成完成")
    print("=" * 68)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"来源: {args.source}")
    print(f"记录数: {stats.get('records', 0)}")
    print(f"原始三元组: {stats.get('raw_triples', 0)}")
    print(f"保留三元组: {stats.get('kept_triples', 0)}")
    print(f"无效三元组: {stats.get('invalid_triples', 0)}")
    print(f"重复三元组: {stats.get('duplicate_triples', 0)}")
    print(f"统计文件: {args.stats}")
    print("=" * 68)


if __name__ == "__main__":
    main()
