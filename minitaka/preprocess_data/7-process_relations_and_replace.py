#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mintaka 数据集处理 - 第 7 步：关系词替换

用途：
1. 读取 relation_alignment.json（聚类代表关系 -> 关系列表）
2. 反向构建关系映射（原关系 -> 聚类代表关系）
3. 在数据集中执行关系替换，默认处理：
   - entity_triples.*.triples[].relation
   - gpt_triples[].relation
4. 兼容处理 Hotpot 风格字段（若存在）：
   - context_triples[].triple  (字符串 "(h, r, t)")

示例：
python 7-process_relations_and_replace.py ^
  --input data/mintaka_dev_stage1_canonicalized.json ^
  --alignment data/relation_alignment.json ^
  --output data/mintaka_dev_relations_replaced.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple


def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_mapping_from_alignment(alignment_file: str) -> Dict[str, str]:
    """支持两类 JSON 格式：
    1) canonical -> [relation1, relation2, ...]
    2) relation  -> canonical
    """
    raw = load_json(alignment_file)
    if not isinstance(raw, dict):
        raise ValueError(f"alignment 文件格式错误（应为 dict）：{alignment_file}")

    mapping: Dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            canonical = str(k).strip()
            if canonical:
                mapping[canonical] = canonical
            for rel in v:
                if isinstance(rel, str):
                    rel_name = rel.strip()
                    if rel_name:
                        mapping[rel_name] = canonical
        elif isinstance(v, str):
            rel_name = str(k).strip()
            canonical = v.strip()
            if rel_name and canonical:
                mapping[rel_name] = canonical
        else:
            continue

    if not mapping:
        raise ValueError(f"alignment 文件未解析到有效关系映射：{alignment_file}")
    return mapping


def build_identity_mapping_from_relation2id(relation2id_file: str) -> Dict[str, str]:
    """退化映射：relation -> relation（不会产生去重替换，只做字段覆盖）"""
    mapping: Dict[str, str] = {}
    with open(relation2id_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            rel = parts[0].strip()
            if rel:
                mapping[rel] = rel
    return mapping


def parse_triple_string(triple_text: str) -> Optional[Tuple[str, str, str]]:
    """解析 '(head, relation, tail)'，仅按前两个逗号分割。"""
    if not isinstance(triple_text, str):
        return None
    s = triple_text.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1].strip()
    parts = [p.strip() for p in inner.split(",", 2)]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


class RelationReplacer:
    def __init__(self, relation_mapping: Dict[str, str]):
        self.relation_mapping = relation_mapping
        self.stats = {
            "total_records": 0,
            "total_triples": 0,
            "replaced_triples": 0,
            "field_triple_counts": defaultdict(int),
            "missing_relations": defaultdict(int),
        }

    def _replace_relation(self, relation: str) -> Tuple[str, bool]:
        rel = relation.strip() if isinstance(relation, str) else ""
        if not rel:
            return relation, False
        if rel not in self.relation_mapping:
            self.stats["missing_relations"][rel] += 1
            return relation, False

        new_rel = self.relation_mapping[rel]
        changed = new_rel != relation
        if changed:
            self.stats["replaced_triples"] += 1
        return new_rel, changed

    def _process_entity_triples(self, entity_triples: dict):
        if not isinstance(entity_triples, dict):
            return
        for info in entity_triples.values():
            if not isinstance(info, dict):
                continue
            triples = info.get("triples", [])
            if not isinstance(triples, list):
                continue
            for t in triples:
                if not isinstance(t, dict) or "relation" not in t:
                    continue
                self.stats["total_triples"] += 1
                self.stats["field_triple_counts"]["entity_triples"] += 1
                new_rel, _ = self._replace_relation(str(t.get("relation", "")))
                t["relation"] = new_rel

    def _process_gpt_triples(self, gpt_triples: list):
        if not isinstance(gpt_triples, list):
            return
        for t in gpt_triples:
            if not isinstance(t, dict) or "relation" not in t:
                continue
            self.stats["total_triples"] += 1
            self.stats["field_triple_counts"]["gpt_triples"] += 1
            new_rel, _ = self._replace_relation(str(t.get("relation", "")))
            t["relation"] = new_rel

    def _process_context_triples(self, context_triples: list):
        """兼容参考脚本中的字符串三元组字段。"""
        if not isinstance(context_triples, list):
            return
        for idx, t in enumerate(context_triples):
            if isinstance(t, dict) and isinstance(t.get("triple"), str):
                parsed = parse_triple_string(t["triple"])
                if parsed is None:
                    continue
                h, r, tail = parsed
                self.stats["total_triples"] += 1
                self.stats["field_triple_counts"]["context_triples"] += 1
                new_rel, _ = self._replace_relation(r)
                t["triple"] = f"({h}, {new_rel}, {tail})"
            elif isinstance(t, str):
                parsed = parse_triple_string(t)
                if parsed is None:
                    continue
                h, r, tail = parsed
                self.stats["total_triples"] += 1
                self.stats["field_triple_counts"]["context_triples"] += 1
                new_rel, _ = self._replace_relation(r)
                context_triples[idx] = f"({h}, {new_rel}, {tail})"

    def process_record(self, item: dict):
        if not isinstance(item, dict):
            return
        self._process_entity_triples(item.get("entity_triples"))
        self._process_gpt_triples(item.get("gpt_triples"))
        self._process_context_triples(item.get("context_triples"))

    def process_dataset(self, data: list):
        if not isinstance(data, list):
            raise ValueError("输入 JSON 顶层必须是 list")
        self.stats["total_records"] = len(data)
        for item in data:
            self.process_record(item)

    def save_stats(self, stats_file: str):
        payload = {
            "total_records": self.stats["total_records"],
            "total_triples": self.stats["total_triples"],
            "replaced_triples": self.stats["replaced_triples"],
            "not_replaced_triples": self.stats["total_triples"] - self.stats["replaced_triples"],
            "field_triple_counts": dict(self.stats["field_triple_counts"]),
            "missing_relations": dict(
                sorted(
                    self.stats["missing_relations"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        }
        save_json(payload, stats_file)


def auto_find_file(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="关系去重映射回写（类似 12-process_relations_and_replace.py）")
    parser.add_argument("--input", "-i", required=True, help="输入 JSON 文件")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 文件（默认自动追加后缀）")
    parser.add_argument("--alignment", "-a", default=None, help="relation_alignment.json 路径")
    parser.add_argument("--relation2id", "-r", default=None, help="relation2id_deduplicated.txt（fallback）")
    parser.add_argument("--stats", default=None, help="统计输出 JSON（默认 output 同目录 replacement_stats.json）")
    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    base_name, ext = os.path.splitext(input_file)
    output_file = args.output or f"{base_name}_relations_replaced{ext or '.json'}"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.dirname(os.path.abspath(input_file))

    alignment_file = auto_find_file([
        args.alignment,
        os.path.join(input_dir, "relation_alignment.json"),
        os.path.join(input_dir, "data", "relation_alignment.json"),
        os.path.join(script_dir, "data", "relation_alignment.json"),
    ])
    relation2id_file = auto_find_file([
        args.relation2id,
        os.path.join(input_dir, "relation2id_deduplicated.txt"),
        os.path.join(input_dir, "data", "relation2id_deduplicated.txt"),
        os.path.join(script_dir, "data", "relation2id_deduplicated.txt"),
    ])

    if alignment_file:
        relation_mapping = build_mapping_from_alignment(alignment_file)
        print(f"[INFO] 使用 alignment 映射：{alignment_file}")
        print(f"[INFO] 映射关系数：{len(relation_mapping)}")
    elif relation2id_file:
        relation_mapping = build_identity_mapping_from_relation2id(relation2id_file)
        print(f"[WARN] 未找到 alignment，使用 relation2id fallback：{relation2id_file}")
        print(f"[INFO] 加载关系数：{len(relation_mapping)}")
    else:
        raise FileNotFoundError(
            "未找到映射文件。请提供 --alignment，或确保 data/relation_alignment.json 存在。"
        )

    data = load_json(input_file)
    replacer = RelationReplacer(relation_mapping)
    replacer.process_dataset(data)
    save_json(data, output_file)

    stats_file = args.stats or os.path.join(os.path.dirname(output_file) or ".", "replacement_stats.json")
    replacer.save_stats(stats_file)

    print("\n================ Result ================")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"统计文件: {stats_file}")
    print(f"记录数: {replacer.stats['total_records']}")
    print(f"三元组总数: {replacer.stats['total_triples']}")
    print(f"替换数: {replacer.stats['replaced_triples']}")
    print(f"未命中关系数: {len(replacer.stats['missing_relations'])}")
    print("========================================")


if __name__ == "__main__":
    main()
