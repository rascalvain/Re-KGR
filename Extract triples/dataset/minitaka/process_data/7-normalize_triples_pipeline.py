"""
Mintaka 数据集处理 — 第 6 步：三元组规范化 Pipeline

整合以下四阶段工作于单一脚本（参考 HotpotQA 的 6/7/9/10 号脚本）：

  Stage 1  LLM 规范化  (参考 6-triples_align.py)
           对 gpt_triples 中的实体名称和关系词用 LLM 进行标准化，
           映射到 Wikidata 风格的规范表达；entity_triples 已经是
           Wikidata 标准标签，默认跳过（可通过配置开启）。

  Stage 2  实体 & 关系提取  (参考 7-extract_entities_relations.py)
           从 entity_triples 和 gpt_triples 两个字段中提取所有
           唯一实体和关系，输出 entity2id.txt / relation2id.txt。

  Stage 3  关系去重  (参考 9-relation_deduplication_optimized.py)
           用 SBERT 对关系词编码，DBSCAN 聚类找到语义重复关系，
           每组选最短表达作为代表词，输出去重后的 relation2id 和
           对齐映射文件。

  Stage 4  回写映射  (参考 10-cluster_relations.py)
           将关系去重映射应用回 JSON 数据中两个三元组字段，
           输出最终规范化数据集。

Mintaka 与 HotpotQA 的差异适配：
  - 三元组格式：{"head": ..., "relation": ..., "tail": ...}  字典
    （而非 HotpotQA 的 {"triple": "(h, r, t)"} 字符串）
  - entity_triples：{"Q-ID": {"label":..., "triples":[...]}, ...}  嵌套字典
  - gpt_triples：[{"head":..., "relation":..., "tail":...}, ...]    平铺列表

输入：mintaka_dev_with_all_triples_cleaned.json
输出：
  data/mintaka_dev_stage1_canonicalized.json  Stage 1 规范化结果
  data/entity2id.txt                          Stage 2 实体词表
  data/relation2id.txt                        Stage 2 关系词表（原始）
  data/relation_alignment.json                Stage 3 关系对齐映射
  data/relation2id_deduplicated.txt           Stage 3 去重后关系词表
  data/mintaka_dev_normalized_final.json      Stage 4 最终输出
"""

import json
import os
import re
import time
import pickle
import numpy as np
from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple, Optional

import torch
from openai import OpenAI
import openai
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# ========================================================================== #
#  全局配置
# ========================================================================== #

# ---- 输入 / 输出 ----
INPUT_FILE           = "data/mintaka_dev_with_all_triples_cleaned.json"
STAGE1_OUTPUT        = "data/mintaka_dev_stage1_canonicalized.json"
STAGE1_CHECKPOINT    = "data/stage1_checkpoint.json"
ENTITY2ID_FILE       = "data/entity2id.txt"
RELATION2ID_FILE     = "data/relation2id.txt"
ALIGNMENT_FILE       = "data/relation_alignment.json"
RELATION_DEDUP_FILE  = "data/relation2id_deduplicated.txt"
FINAL_OUTPUT         = "data/mintaka_dev_normalized_final.json"

# ---- 阶段开关（True = 执行该阶段）----
RUN_STAGE1 = True   # LLM 规范化
RUN_STAGE2 = True   # 实体 & 关系提取
RUN_STAGE3 = True   # 关系去重
RUN_STAGE4 = True   # 回写映射

# ---- Stage 1 参数 ----
CANONICALIZE_GPT_TRIPLES    = True   # 对 gpt_triples 做 LLM 规范化
CANONICALIZE_ENTITY_TRIPLES = False  # entity_triples 已是 Wikidata 标准标签，默认跳过
BATCH_SIZE    = 15    # 每批送入 LLM 的三元组数
API_DELAY     = 0.5   # API 调用间隔（秒）
MAX_RETRIES   = 3
RETRY_DELAY   = 2
SAVE_EVERY    = 20    # 每处理 N 条数据保存一次

# ---- Stage 3 参数 ----
SBERT_MODEL_PATH     = "/home/shu1004/lyx/mintaka_preprocess/process_data/all-mpnet-base-v2"   # 本地路径或 HF 模型名
SBERT_DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.85   # 余弦相似度阈值，超过则视为重复
EMBEDDING_CACHE_FILE = "data/relation_embeddings_cache.pkl"

# ---- OpenAI API ----
client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)
LLM_MODEL = "gpt-3.5-turbo"

# ========================================================================== #
#  LLM 规范化 Prompt
# ========================================================================== #

CANONICALIZATION_PROMPT = """You are a Knowledge Graph Canonicalization Engine. Transform raw triples into standardized, Wikidata-aligned triples.

Given a batch of raw triples as [Subject, Relation, Object], you must:
1. Standardize entity names to official Wikidata labels (full names, proper capitalization)
2. Normalize relations to standard Wikidata property labels or clear English predicates
3. Correct logical direction when necessary (e.g., swap subject/object to match canonical relation direction)

Common relation mappings:
- "started", "created", "established" → "founded by" or "founder"
- "is", "is a", "is an" → "instance of"
- "born on", "birth date" → "date of birth"
- "directed by" → "director"
- "wife", "husband" → "spouse"
- "attended", "studied at" → "educated at"
- "works for", "employed by" → "employer"
- "made by", "manufactured by" → "manufacturer"

Direction correction examples:
- ["Steve Jobs", "founded", "Apple"] → ["Apple Inc.", "founded by", "Steve Jobs"]
- ["Wachowskis", "directed", "The Matrix"] → ["The Matrix", "director", "Wachowskis"]

Output ONLY a valid JSON array of arrays, each inner array has exactly 3 strings.
NO explanatory text. NO markdown. NO Wikidata QIDs/PIDs.

Input:
{batch_data}

Output (JSON only):"""


# ========================================================================== #
#  通用工具
# ========================================================================== #

def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def triple_to_list(t: dict) -> list:
    """{"head":h,"relation":r,"tail":t} → [h, r, t]"""
    return [t.get("head", ""), t.get("relation", ""), t.get("tail", "")]


def list_to_triple(lst: list) -> dict:
    """[h, r, t] → {"head":h, "relation":r, "tail":t}"""
    return {"head": lst[0], "relation": lst[1], "tail": lst[2]}


# ========================================================================== #
#  Stage 1：LLM 规范化
# ========================================================================== #

def call_llm(prompt: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            params = {
                "model":    LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "n": 1,
            }
            try:
                params["max_tokens"] = 1024
                resp = client.chat.completions.create(**params)
            except (TypeError, openai.BadRequestError):
                params.pop("max_tokens", None)
                resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"      LLM 错误 (attempt {attempt + 1}): {str(e)[:80]}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return "[]"


def clean_llm_response(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*",     "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",     "", text)
    return text.strip()


def validate_triple_list(item) -> Optional[list]:
    """验证 LLM 返回的单条三元组 [h, r, t]"""
    if not isinstance(item, (list, tuple)) or len(item) != 3:
        return None
    parts = []
    for p in item:
        if not isinstance(p, str):
            return None
        p = p.strip()
        if not p or len(p) > 500:
            return None
        parts.append(p)
    return parts


def canonicalize_batch(raw_triples: List[dict]) -> List[dict]:
    """对一批三元组调用 LLM 规范化，失败时返回原始数据。"""
    if not raw_triples:
        return raw_triples

    raw_lists = [triple_to_list(t) for t in raw_triples]
    prompt = CANONICALIZATION_PROMPT.format(
        batch_data=json.dumps(raw_lists, ensure_ascii=False, indent=2)
    )

    response = call_llm(prompt)
    time.sleep(API_DELAY)

    try:
        cleaned = clean_llm_response(response)
        result = json.loads(cleaned)
        if not isinstance(result, list):
            return raw_triples

        valid = []
        for idx, item in enumerate(result):
            v = validate_triple_list(item)
            valid.append(list_to_triple(v) if v else raw_triples[idx] if idx < len(raw_triples) else None)

        # 数量补齐：结果不足时补原始数据
        while len(valid) < len(raw_triples):
            valid.append(raw_triples[len(valid)])

        return valid[:len(raw_triples)]

    except (json.JSONDecodeError, Exception) as e:
        print(f"      解析失败 ({str(e)[:60]})，保留原始数据")
        return raw_triples


def canonicalize_gpt_triples(gpt_triples: List[dict]) -> List[dict]:
    """对 gpt_triples 平铺列表分批规范化。"""
    if not gpt_triples:
        return gpt_triples

    result = []
    total_batches = (len(gpt_triples) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(0, len(gpt_triples), BATCH_SIZE):
        batch = gpt_triples[batch_idx: batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"        gpt_triples 批 {batch_num}/{total_batches} ({len(batch)} 条)...", end=" ")
        aligned = canonicalize_batch(batch)
        result.extend(aligned)
        print("✓")
    return result


def canonicalize_entity_triples(entity_triples: dict) -> dict:
    """对 entity_triples 嵌套结构分批规范化。"""
    if not entity_triples:
        return entity_triples

    # 展开：[(eid, idx, triple_dict), ...]
    flat = []
    for eid, info in entity_triples.items():
        for tidx, t in enumerate(info.get("triples", [])):
            flat.append((eid, tidx, t))

    aligned_flat = []
    total_batches = (len(flat) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(0, len(flat), BATCH_SIZE):
        batch_items = flat[batch_idx: batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        raw = [item[2] for item in batch_items]
        print(f"        entity_triples 批 {batch_num}/{total_batches} ({len(raw)} 条)...", end=" ")
        aligned = canonicalize_batch(raw)
        for i, (eid, tidx, _) in enumerate(batch_items):
            aligned_flat.append((eid, tidx, aligned[i]))
        print("✓")

    # 重建嵌套结构
    new_map: Dict[str, list] = {eid: [] for eid in entity_triples}
    for eid, tidx, t in aligned_flat:
        new_map[eid].append((tidx, t))

    new_entity_triples = {}
    for eid, info in entity_triples.items():
        triples_sorted = sorted(new_map[eid], key=lambda x: x[0])
        triples = [t for _, t in triples_sorted]
        new_entity_triples[eid] = {
            "label":        info.get("label", eid),
            "triple_count": len(triples),
            "triples":      triples,
        }
    return new_entity_triples


def load_stage1_checkpoint() -> Tuple[list, int]:
    """加载 Stage 1 断点，返回 (已处理数据, 起始索引)。"""
    if os.path.exists(STAGE1_OUTPUT) and os.path.exists(STAGE1_CHECKPOINT):
        try:
            with open(STAGE1_CHECKPOINT, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            last_idx = ckpt.get("last_processed_index", -1)
            data = load_json(STAGE1_OUTPUT)
            print(f"  断点续传：已处理 {last_idx + 1} 条，从第 {last_idx + 2} 条继续")
            return data, last_idx + 1
        except Exception as e:
            print(f"  断点加载失败 ({e})，从头开始")
    return None, 0


def save_stage1_checkpoint(data: list, idx: int):
    save_json(data, STAGE1_OUTPUT)
    ckpt = {"last_processed_index": idx,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(data)}
    save_json(ckpt, STAGE1_CHECKPOINT)


def run_stage1(data: list) -> list:
    """Stage 1：LLM 规范化 gpt_triples（可选 entity_triples）。"""
    print("\n" + "=" * 70)
    print("Stage 1: LLM 三元组规范化")
    print(f"  规范化 gpt_triples:    {CANONICALIZE_GPT_TRIPLES}")
    print(f"  规范化 entity_triples: {CANONICALIZE_ENTITY_TRIPLES}")
    print("=" * 70)

    loaded, start_idx = load_stage1_checkpoint()
    if loaded is not None:
        processed = loaded
    else:
        processed = [None] * len(data)
        # 将未处理的数据先填充原始值（便于保存）
        for i, item in enumerate(data):
            processed[i] = item

    total = len(data)

    try:
        for idx in range(start_idx, total):
            item = data[idx]
            record_id = item.get("id", f"idx-{idx}")
            print(f"\n[{idx + 1}/{total}] ID: {record_id}")

            if CANONICALIZE_GPT_TRIPLES and item.get("gpt_triples"):
                print(f"    gpt_triples ({len(item['gpt_triples'])} 条)")
                item["gpt_triples"] = canonicalize_gpt_triples(item["gpt_triples"])
                item["gpt_triple_count"] = len(item["gpt_triples"])

            if CANONICALIZE_ENTITY_TRIPLES and item.get("entity_triples"):
                total_ent = sum(
                    len(v.get("triples", []))
                    for v in item["entity_triples"].values()
                )
                print(f"    entity_triples ({total_ent} 条)")
                item["entity_triples"] = canonicalize_entity_triples(item["entity_triples"])

            processed[idx] = item

            if (idx + 1) % SAVE_EVERY == 0 or idx == total - 1:
                save_stage1_checkpoint(processed, idx)
                print(f"  💾 进度已保存 ({idx + 1}/{total})")

    except KeyboardInterrupt:
        print("\n用户中断，保存已完成部分...")
        save_stage1_checkpoint(processed, idx)
        print(f"  进度已保存至第 {idx + 1} 条")
        raise

    # 完成后删除 checkpoint
    if os.path.exists(STAGE1_CHECKPOINT):
        os.remove(STAGE1_CHECKPOINT)
    save_json(processed, STAGE1_OUTPUT)
    print(f"\n✅ Stage 1 完成，已保存: {STAGE1_OUTPUT}")
    return processed


# ========================================================================== #
#  Stage 2：实体 & 关系提取
# ========================================================================== #

def extract_from_entity_triples(entity_triples: dict, entities: OrderedDict, relations: OrderedDict):
    """从 entity_triples 嵌套结构中提取实体和关系。"""
    for eid, info in entity_triples.items():
        for t in info.get("triples", []):
            h, r, o = t.get("head", ""), t.get("relation", ""), t.get("tail", "")
            if h: entities[h] = None
            if o: entities[o] = None
            if r: relations[r] = None


def extract_from_gpt_triples(gpt_triples: list, entities: OrderedDict, relations: OrderedDict):
    """从 gpt_triples 平铺列表中提取实体和关系。"""
    for t in gpt_triples:
        h, r, o = t.get("head", ""), t.get("relation", ""), t.get("tail", "")
        if h: entities[h] = None
        if o: entities[o] = None
        if r: relations[r] = None


def save_id_file(items: list, path: str):
    """保存 entity2id / relation2id 格式文件：item<TAB>id"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            f.write(f"{item}\t{idx}\n")
    print(f"  已保存 {len(items)} 条 → {path}")


def run_stage2(data: list):
    """Stage 2：提取全部唯一实体 & 关系，保存 entity2id / relation2id。"""
    print("\n" + "=" * 70)
    print("Stage 2: 实体 & 关系提取")
    print("=" * 70)

    entities  = OrderedDict()
    relations = OrderedDict()

    for idx, item in enumerate(data):
        if (idx + 1) % 500 == 0:
            print(f"  已扫描 {idx + 1}/{len(data)} 条...")
        if item.get("entity_triples"):
            extract_from_entity_triples(item["entity_triples"], entities, relations)
        if item.get("gpt_triples"):
            extract_from_gpt_triples(item["gpt_triples"], entities, relations)

    print(f"\n  发现唯一实体:  {len(entities)}")
    print(f"  发现唯一关系:  {len(relations)}")

    save_id_file(list(entities.keys()),  ENTITY2ID_FILE)
    save_id_file(list(relations.keys()), RELATION2ID_FILE)
    print(f"✅ Stage 2 完成")


# ========================================================================== #
#  Stage 3：关系去重（SBERT + DBSCAN）
# ========================================================================== #

def load_relation_file(path: str) -> List[Tuple[str, int]]:
    """加载 relation2id.txt → [(relation, id), ...]"""
    rels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rels.append((parts[0], int(parts[1])))
    return rels


def load_sbert_model():
    """加载 SBERT 模型（自动 fallback）。"""
    model_path = SBERT_MODEL_PATH
    print(f"  加载 SBERT 模型: {model_path}  设备: {SBERT_DEVICE}")
    try:
        from transformers import AutoModel as _AutoModel
        _orig = _AutoModel.from_pretrained
        _AutoModel.from_pretrained = lambda *a, **kw: _orig(
            *a, **{**kw, "ignore_mismatched_sizes": True}
        )
        model = SentenceTransformer(model_path, device=SBERT_DEVICE)
        _AutoModel.from_pretrained = _orig
    except Exception as e:
        print(f"  本地模型加载失败 ({e})，尝试在线模型 all-MiniLM-L6-v2")
        model = SentenceTransformer("all-MiniLM-L6-v2", device=SBERT_DEVICE)
    return model


def compute_embeddings(model, texts: list) -> np.ndarray:
    """计算文本嵌入（支持磁盘缓存）。"""
    if os.path.exists(EMBEDDING_CACHE_FILE):
        print(f"  从缓存加载嵌入向量: {EMBEDDING_CACHE_FILE}")
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if len(cached) == len(texts):
            return cached
        print(f"  缓存条数 {len(cached)} ≠ 当前 {len(texts)}，重新计算")

    print(f"  计算 {len(texts)} 条关系词的嵌入...")
    embs = model.encode(
        texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True
    )
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_FILE) or ".", exist_ok=True)
    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump(embs, f)
    print(f"  嵌入向量已缓存: {EMBEDDING_CACHE_FILE}")
    return embs


def dbscan_cluster(relations: List[Tuple[str, int]], embeddings: np.ndarray):
    """DBSCAN 聚类找到语义相近的关系词组。"""
    eps = 1.0 - SIMILARITY_THRESHOLD
    print(f"  DBSCAN 聚类 (eps={eps:.2f}, 相似度阈值={SIMILARITY_THRESHOLD})...")
    clustering = DBSCAN(eps=eps, min_samples=1, metric="cosine", n_jobs=-1)
    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(idx)

    print(f"  发现 {len(clusters)} 个聚类（原始 {len(relations)} 条）")

    # 每组选最短关系词作为代表
    relation_to_canonical = {}
    canonical_to_group = {}
    for cluster_id, indices in clusters.items():
        cluster_rels = [(relations[i][0], i) for i in indices]
        cluster_rels.sort(key=lambda x: (len(x[0]), x[0]))
        canonical = cluster_rels[0][0]
        for rel, _ in cluster_rels:
            relation_to_canonical[rel] = canonical
        canonical_to_group[canonical] = [r for r, _ in cluster_rels]

    return relation_to_canonical, canonical_to_group


def run_stage3():
    """Stage 3：SBERT + DBSCAN 关系去重。"""
    print("\n" + "=" * 70)
    print("Stage 3: 关系去重（SBERT + DBSCAN）")
    print("=" * 70)

    if not os.path.exists(RELATION2ID_FILE):
        print(f"  ❌ 找不到 {RELATION2ID_FILE}，请先运行 Stage 2")
        return

    relations = load_relation_file(RELATION2ID_FILE)
    print(f"  加载关系词: {len(relations)} 条")

    model = load_sbert_model()
    texts = [r[0] for r in relations]
    embeddings = compute_embeddings(model, texts)

    relation_to_canonical, canonical_to_group = dbscan_cluster(relations, embeddings)

    # 统计
    deduped_rels = sorted(set(relation_to_canonical.values()))
    reduced = len(relations) - len(deduped_rels)
    print(f"\n  去重前: {len(relations)}  去重后: {len(deduped_rels)}"
          f"  减少: {reduced} ({reduced / max(len(relations), 1) * 100:.1f}%)")

    # 保存 relation_alignment.json
    os.makedirs(os.path.dirname(ALIGNMENT_FILE) or ".", exist_ok=True)
    with open(ALIGNMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(canonical_to_group, f, ensure_ascii=False, indent=2)
    print(f"  已保存对齐映射: {ALIGNMENT_FILE}")

    # 保存去重后的 relation2id
    with open(RELATION_DEDUP_FILE, "w", encoding="utf-8") as f:
        for new_id, rel in enumerate(deduped_rels):
            f.write(f"{rel}\t{new_id}\n")
    print(f"  已保存去重关系词表: {RELATION_DEDUP_FILE}")

    print(f"✅ Stage 3 完成")
    return relation_to_canonical


# ========================================================================== #
#  Stage 4：回写去重映射
# ========================================================================== #

def apply_dedup_to_entity_triples(entity_triples: dict,
                                   mapping: Dict[str, str]) -> dict:
    """将关系映射应用到 entity_triples 嵌套结构。"""
    for eid, info in entity_triples.items():
        for t in info.get("triples", []):
            r = t.get("relation", "")
            if r in mapping:
                t["relation"] = mapping[r]
    return entity_triples


def apply_dedup_to_gpt_triples(gpt_triples: list,
                                mapping: Dict[str, str]) -> list:
    """将关系映射应用到 gpt_triples 平铺列表。"""
    for t in gpt_triples:
        r = t.get("relation", "")
        if r in mapping:
            t["relation"] = mapping[r]
    return gpt_triples


def load_alignment(path: str) -> Dict[str, str]:
    """加载 relation_alignment.json，转换为 {original: canonical}。"""
    with open(path, "r", encoding="utf-8") as f:
        canonical_to_group = json.load(f)
    mapping = {}
    for canonical, group in canonical_to_group.items():
        for rel in group:
            mapping[rel] = canonical
    return mapping


def run_stage4(data: list, relation_to_canonical: Optional[Dict] = None):
    """Stage 4：将关系去重映射回写到 JSON 数据集。"""
    print("\n" + "=" * 70)
    print("Stage 4: 回写关系去重映射")
    print("=" * 70)

    if relation_to_canonical is None:
        if not os.path.exists(ALIGNMENT_FILE):
            print(f"  ❌ 找不到 {ALIGNMENT_FILE}，请先运行 Stage 3")
            return data
        relation_to_canonical = load_alignment(ALIGNMENT_FILE)

    replace_count = 0
    for idx, item in enumerate(data):
        if item.get("entity_triples"):
            apply_dedup_to_entity_triples(item["entity_triples"], relation_to_canonical)
        if item.get("gpt_triples"):
            apply_dedup_to_gpt_triples(item["gpt_triples"], relation_to_canonical)
        replace_count += 1

        if (idx + 1) % 500 == 0:
            print(f"  已处理 {idx + 1}/{len(data)} 条...")

    save_json(data, FINAL_OUTPUT)
    print(f"\n  处理 {replace_count} 条数据")
    print(f"  已保存最终结果: {FINAL_OUTPUT}")
    print(f"✅ Stage 4 完成")
    return data


# ========================================================================== #
#  主入口
# ========================================================================== #

def main():
    print("=" * 70)
    print("Mintaka 三元组规范化 Pipeline")
    print("  Stage 1: LLM 规范化")
    print("  Stage 2: 实体 & 关系提取")
    print("  Stage 3: SBERT + DBSCAN 关系去重")
    print("  Stage 4: 回写去重映射")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ 输入文件不存在: {INPUT_FILE}")
        return

    os.makedirs("data", exist_ok=True)

    # 读取原始输入
    print(f"\n读取输入: {INPUT_FILE}")
    data = load_json(INPUT_FILE)
    print(f"数据条数: {len(data)}")

    # ---- Stage 1 ----
    if RUN_STAGE1:
        data = run_stage1(data)
    else:
        print(f"\nStage 1 已跳过，尝试加载: {STAGE1_OUTPUT}")
        if os.path.exists(STAGE1_OUTPUT):
            data = load_json(STAGE1_OUTPUT)
            print(f"  加载 {len(data)} 条")
        else:
            print(f"  文件不存在，使用原始数据")

    # ---- Stage 2 ----
    if RUN_STAGE2:
        run_stage2(data)

    # ---- Stage 3 ----
    relation_to_canonical = None
    if RUN_STAGE3:
        relation_to_canonical = run_stage3()

    # ---- Stage 4 ----
    if RUN_STAGE4:
        data = run_stage4(data, relation_to_canonical)

    print("\n" + "=" * 70)
    print("Pipeline 完成！输出文件：")
    for f in [STAGE1_OUTPUT, ENTITY2ID_FILE, RELATION2ID_FILE,
              ALIGNMENT_FILE, RELATION_DEDUP_FILE, FINAL_OUTPUT]:
        status = "✅" if os.path.exists(f) else "⬜"
        print(f"  {status}  {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
