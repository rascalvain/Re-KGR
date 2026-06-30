"""
Mintaka 数据集处理 — 第 2.7 步：替换 entity_triples 中的 relation URI 为自然语言标签

问题：2-retrieve_wikidata_triples.py 的 SPARQL 查询中，relation 字段存储的是
Wikidata 属性的直接谓词 URI（如 http://www.wikidata.org/prop/direct/P50），
而非属性的自然语言标签（如 "author"）。

解决方案：
  1. 扫描数据中所有 entity_triples.*.triples[].relation 字段
  2. 从 URI 中提取 P-ID（如 P50）
  3. 批量调用 Wikidata wbgetentities API 获取属性的英文标签
  4. 将 URI 原地替换为自然语言标签
  5. 保存结果

输入：mintaka_dev_with_wikidata_triples_pruned.json（或任意含 entity_triples 的文件）
输出：同名文件（原地覆盖）或指定输出文件
"""

import json
import os
import re
import time
import requests
from collections import defaultdict

# ========================================================================== #
#  配置
# ========================================================================== #

INPUT_FILE  = "data/mintaka_dev_with_all_triples_pruned.json"
OUTPUT_FILE = "data/mintaka_dev_with_all_triples_replaced.json"  # 默认原地覆盖

# Wikidata API
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
API_BATCH_SIZE = 50          # wbgetentities 每次最多 50 个 ID
REQUEST_DELAY  = 1.0         # 两次请求之间的间隔（秒）
MAX_RETRIES    = 5

# Wikidata 要求携带合法 User-Agent，否则返回 403
HEADERS = {
    "User-Agent": (
        "MintakaRelationMapper/1.0 "
        "(Wikidata property label lookup; https://github.com/wikimedia) "
        "python-requests/2.28"
    )
}

# 属性 URI 前缀（均视为 P-ID 来查询）
URI_PREFIXES = [
    "http://www.wikidata.org/prop/direct/",
    "http://www.wikidata.org/prop/",
    "https://www.wikidata.org/prop/direct/",
    "https://www.wikidata.org/prop/",
]

# ========================================================================== #
#  工具函数
# ========================================================================== #

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_pid(relation_str: str):
    """从 URI 或原始字符串中提取 Wikidata 属性 ID（P-数字）。

    示例：
      "http://www.wikidata.org/prop/direct/P50" → "P50"
      "P50"                                      → "P50"
      "author"                                   → None（已是自然语言，不处理）
      "0" / ""                                   → None（无效，跳过）
    """
    if not relation_str or not relation_str.strip():
        return None
    for prefix in URI_PREFIXES:
        if relation_str.startswith(prefix):
            tail = relation_str[len(prefix):]
            m = re.match(r"^(P\d+)$", tail.strip())
            return m.group(1) if m else None
    # 不带 URI 前缀，但本身就是 P-ID（严格匹配，避免误识别数字字符串）
    m = re.match(r"^(P\d+)$", relation_str.strip())
    return m.group(1) if m else None


LABEL_CACHE_FILE = "data/property_label_cache.json"


def load_label_cache() -> dict:
    """加载已缓存的属性标签，支持断点续传。"""
    if os.path.exists(LABEL_CACHE_FILE):
        with open(LABEL_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  载入缓存: {len(cache)} 条属性标签")
        return cache
    return {}


def save_label_cache(cache: dict):
    os.makedirs(os.path.dirname(LABEL_CACHE_FILE) or ".", exist_ok=True)
    with open(LABEL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def fetch_property_labels(pids: list) -> dict:
    """批量查询 Wikidata API，返回 {P-ID: 英文标签} 映射。

    每批最多 50 个，超出自动分批。支持断点续传（结果缓存到本地）。
    """
    label_map = load_label_cache()

    # 过滤已缓存的，只查询未知 P-ID
    remaining = [p for p in pids if p not in label_map]
    total = len(remaining)
    if total == 0:
        print("  所有属性标签均已缓存，跳过 API 查询。")
        return label_map

    print(f"  需查询: {total} 个（已缓存 {len(pids) - total} 个）")

    for batch_start in range(0, total, API_BATCH_SIZE):
        batch = remaining[batch_start: batch_start + API_BATCH_SIZE]
        ids_str = "|".join(batch)

        params = {
            "action":    "wbgetentities",
            "ids":       ids_str,
            "props":     "labels",
            "languages": "en",
            "format":    "json",
        }

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(WIKIDATA_API, params=params,
                                    headers=HEADERS, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                entities = data.get("entities", {})
                for pid, info in entities.items():
                    en_label = info.get("labels", {}).get("en", {}).get("value", "")
                    label_map[pid] = en_label if en_label else pid
                success = True
                break

            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                print(f"    API 请求失败 (attempt {attempt + 1}/{MAX_RETRIES}): {e}，等待 {wait}s")
                time.sleep(wait)

        if not success:
            print(f"    ⚠️  批次 {batch_start}-{batch_start+len(batch)} 失败，用 P-ID 占位")
            for pid in batch:
                label_map[pid] = pid

        done = min(batch_start + API_BATCH_SIZE, total)
        print(f"  进度: {done}/{total} 个属性查询完毕")

        # 每批保存一次缓存
        save_label_cache(label_map)
        time.sleep(REQUEST_DELAY)

    return label_map


# ========================================================================== #
#  扫描 + 替换
# ========================================================================== #

def collect_relation_uris(data: list) -> set:
    """遍历所有 entity_triples，收集需要替换的 relation 字段值。"""
    uris = set()
    for item in data:
        for eid, info in item.get("entity_triples", {}).items():
            for triple in info.get("triples", []):
                rel = triple.get("relation", "")
                if rel:
                    uris.add(rel)
    return uris


def replace_relations(data: list, label_map: dict) -> tuple:
    """用 label_map 将 entity_triples 中的 relation URI 替换为自然语言标签。

    返回 (data, replace_count, skip_count)
    """
    replace_count = 0
    skip_count = 0

    for item in data:
        for eid, info in item.get("entity_triples", {}).items():
            for triple in info.get("triples", []):
                rel = triple.get("relation", "")
                if rel in label_map:
                    triple["relation"] = label_map[rel]
                    replace_count += 1
                else:
                    skip_count += 1

    return data, replace_count, skip_count


# ========================================================================== #
#  主流程
# ========================================================================== #

def main():
    print("=" * 70)
    print("替换 entity_triples.relation URI → 自然语言属性标签")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 输入文件不存在: {INPUT_FILE}")
        return

    print(f"\n读取文件: {INPUT_FILE}")
    data = load_data(INPUT_FILE)
    print(f"数据条数: {len(data)}")

    # ---- 第一步：收集所有 relation 值 ----
    print("\n[1/3] 扫描 relation 字段...")
    all_rels = collect_relation_uris(data)
    print(f"  共发现 {len(all_rels)} 种不同 relation 值")

    # 区分 URI（需要查询）和已是自然语言（跳过）
    pid_to_uri = {}    # {P50: "http://...P50"}
    plain_rels = []    # 已是自然语言的
    for rel in all_rels:
        pid = extract_pid(rel)
        if pid:
            pid_to_uri[pid] = rel
        else:
            plain_rels.append(rel)

    print(f"  需要查询的属性 URI: {len(pid_to_uri)} 个")
    if plain_rels:
        print(f"  已是自然语言，跳过: {len(plain_rels)} 个（示例: {plain_rels[:3]}）")

    if not pid_to_uri:
        print("\n✅ 所有 relation 字段均已是自然语言，无需替换。")
        return

    # ---- 第二步：批量查询 Wikidata 属性标签 ----
    print(f"\n[2/3] 批量查询 Wikidata 属性标签（每批 {API_BATCH_SIZE} 个）...")
    pid_labels = fetch_property_labels(list(pid_to_uri.keys()))

    # 构建 URI → 标签 映射
    uri_to_label = {}
    for pid, uri in pid_to_uri.items():
        label = pid_labels.get(pid, pid)   # 查不到标签则保留 P-ID
        uri_to_label[uri] = label

    # 打印部分样例
    sample = list(uri_to_label.items())[:10]
    print(f"\n  替换样例（前 10 条）:")
    for uri, label in sample:
        print(f"    {uri}  →  {label}")

    # ---- 第三步：原地替换 ----
    print(f"\n[3/3] 替换数据中的 relation 字段...")
    data, replace_count, skip_count = replace_relations(data, uri_to_label)

    # ---- 保存 ----
    save_data(data, OUTPUT_FILE)

    print(f"\n{'=' * 70}")
    print(f"完成！")
    print(f"  替换三元组数:  {replace_count}")
    print(f"  未替换（已是标签或未匹配）: {skip_count}")
    print(f"  输出文件:      {OUTPUT_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
