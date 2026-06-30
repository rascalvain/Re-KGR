"""
Mintaka 数据集处理 — 第 2 步：从 Wikidata 检索知识图谱三元组

对应论文 KGR 中的 Entity Detection + KG Retrieval 阶段。

Mintaka 数据集自带 Wikidata ID（questionEntity.name / answer[].name），
本脚本直接利用这些 ID 查询 Wikidata SPARQL 端点，获取每个实体的
本地子图三元组（即该实体作为主语或宾语的所有关系）。

同时用 LLM 从 GPT 推理文本中提取中间推理涉及的额外实体，
并查询其 Wikidata 三元组作为补充（对应论文中"验证推理过程中用到
的、不出现在原始问题中的事实"这一核心思想）。

流程：
    1. 收集每条数据中所有已知的 Wikidata ID（问题实体 + 答案实体 + 支持实体）
    2. 用 LLM 从 GPT 推理文本中提取额外实体名称 → Wikidata 搜索 → 获取 ID
    3. 对所有实体 ID 查询 Wikidata SPARQL，获取本地子图三元组
    4. 将三元组存入数据中
"""

import json
import time
import requests
import os
from openai import OpenAI
import openai

# ========================================================================== #
#  API 配置
# ========================================================================== #

client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)

LLM_MODEL = "gpt-3.5-turbo"

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"

# 每个实体最多检索的三元组数量（避免热门实体返回数千条）
MAX_TRIPLES_PER_ENTITY = 200

# ========================================================================== #
#  Prompt
# ========================================================================== #

ENTITY_EXTRACTION_PROMPT = """Given a question and a reasoning response, extract all important named entities mentioned in the reasoning that could be verified against a knowledge graph.

Question: {question}

Reasoning:
{reasoning}

Extract all specific named entities (people, places, organizations, works, events, dates, etc.) that appear in the reasoning process. Return ONLY a JSON list of entity names, nothing else.

Example output: ["Nicolas Chopin", "April 15, 1771", "Warsaw"]

Entities:"""


# ========================================================================== #
#  工具函数
# ========================================================================== #

def load_progress(save_path):
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_data(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def request_llm(prompt, temperature=0.0, max_tokens=300):
    """调用 LLM API"""
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            time.sleep(2 ** retry_count)
            retry_count += 1
        except Exception as e:
            print(f"    LLM API 错误: {e}")
            time.sleep(1)
            retry_count += 1
    return None


# ========================================================================== #
#  Wikidata 查询
# ========================================================================== #

_triple_cache = {}


def query_wikidata_sparql(sparql_query, retries=3):
    """执行 SPARQL 查询，返回 JSON 结果"""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "MintakaKGRetrieval/1.0 (research project)"
    }
    for attempt in range(retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": sparql_query, "format": "json"},
                headers=headers,
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    SPARQL 限速，等待 {wait} 秒...")
                time.sleep(wait)
            else:
                print(f"    SPARQL 状态码 {resp.status_code}: {resp.text[:200]}")
                time.sleep(1)
        except requests.exceptions.Timeout:
            print(f"    SPARQL 超时，重试 ({attempt + 1}/{retries})...")
            time.sleep(2)
        except Exception as e:
            print(f"    SPARQL 错误: {e}")
            time.sleep(1)
    return None


def get_entity_triples(entity_id):
    """获取一个 Wikidata 实体的本地子图三元组

    查询该实体作为主语(subject)的所有三元组，
    返回 [(head_label, relation_label, tail_label), ...] 格式。
    """
    if entity_id in _triple_cache:
        return _triple_cache[entity_id]

    # 作为主语的三元组：(entity) --relation--> (value)
    sparql = f"""
    SELECT ?propLabel ?valueLabel WHERE {{
      wd:{entity_id} ?prop ?value .
      ?property wikibase:directClaim ?prop .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {MAX_TRIPLES_PER_ENTITY}
    """

    result = query_wikidata_sparql(sparql)
    triples = []
    entity_label = get_entity_label(entity_id)

    if result and "results" in result:
        for binding in result["results"]["bindings"]:
            prop_label = binding.get("propLabel", {}).get("value", "")
            value_label = binding.get("valueLabel", {}).get("value", "")
            if prop_label and value_label:
                triples.append((entity_label, prop_label, value_label))

    _triple_cache[entity_id] = triples
    return triples


_label_cache = {}


def get_entity_label(entity_id):
    """获取 Wikidata 实体的英文标签"""
    if entity_id in _label_cache:
        return _label_cache[entity_id]

    sparql = f"""
    SELECT ?label WHERE {{
      wd:{entity_id} rdfs:label ?label .
      FILTER(LANG(?label) = "en")
    }}
    LIMIT 1
    """
    result = query_wikidata_sparql(sparql)
    label = entity_id
    if result and result.get("results", {}).get("bindings"):
        label = result["results"]["bindings"][0].get("label", {}).get("value", entity_id)

    _label_cache[entity_id] = label
    return label


def search_wikidata_entity(entity_name):
    """通过名称搜索 Wikidata 实体，返回最匹配的实体 ID"""
    params = {
        "action": "wbsearchentities",
        "search": entity_name,
        "language": "en",
        "format": "json",
        "limit": 1
    }
    try:
        resp = requests.get(WIKIDATA_SEARCH_URL, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("search", [])
            if results:
                return results[0]["id"]
    except Exception as e:
        print(f"    Wikidata 搜索 \"{entity_name}\" 失败: {e}")
    return None


# ========================================================================== #
#  从数据集字段中收集已知 Wikidata ID
# ========================================================================== #

def collect_known_entity_ids(item):
    """从 Mintaka 数据条目中收集所有已知的 Wikidata ID

    来源：questionEntity、answer、answer.supportingEnt
    返回 {entity_id: label} 字典
    """
    entities = {}

    # 1. questionEntity
    for ent in (item.get("questionEntity") or []):
        if ent.get("entityType") == "entity":
            qid = ent.get("name", "")
            if isinstance(qid, str) and qid.startswith("Q"):
                entities[qid] = ent.get("label", qid)

    # 2. answer 实体
    answer_obj = item.get("answer") or {}
    if answer_obj.get("answerType") == "entity":
        for ans in (answer_obj.get("answer") or []):
            if isinstance(ans, dict):
                qid = ans.get("name", "")
                if isinstance(qid, str) and qid.startswith("Q"):
                    label = qid
                    lbl = ans.get("label") or {}
                    if isinstance(lbl, dict):
                        label = lbl.get("en", qid) or qid
                    elif isinstance(lbl, str):
                        label = lbl
                    entities[qid] = label

    # 3. supportingEnt
    for ent in (answer_obj.get("supportingEnt") or []):
        qid = ent.get("name", "")
        if isinstance(qid, str) and qid.startswith("Q"):
            label = qid
            lbl = ent.get("label") or {}
            if isinstance(lbl, dict):
                label = lbl.get("en", qid) or qid
            elif isinstance(lbl, str):
                label = lbl
            entities[qid] = label

    return entities


# ========================================================================== #
#  从 GPT 推理文本中提取额外实体（论文核心：验证推理中的事实）
# ========================================================================== #

def extract_entities_from_reasoning(question, reasoning):
    """用 LLM 从推理文本中提取实体名称列表"""
    if not reasoning:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(question=question, reasoning=reasoning)
    response = request_llm(prompt, temperature=0.0, max_tokens=400)

    if not response:
        return []

    # 解析 JSON 列表
    try:
        # 尝试直接解析
        entities = json.loads(response)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if e]
    except json.JSONDecodeError:
        pass

    # 回退：尝试提取 [...] 部分
    start = response.find("[")
    end = response.rfind("]")
    if start != -1 and end != -1:
        try:
            entities = json.loads(response[start:end + 1])
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if e]
        except json.JSONDecodeError:
            pass

    return []


def resolve_entity_names_to_ids(entity_names, known_entities):
    """将实体名称列表解析为 Wikidata ID

    先检查是否已在 known_entities 中（按标签匹配），
    否则调用 Wikidata 搜索 API。

    返回 {entity_id: label} 字典（仅新增的）
    """
    known_labels_lower = {v.lower(): k for k, v in known_entities.items()}
    new_entities = {}

    for name in entity_names:
        name_lower = name.lower().strip()
        if not name_lower or len(name_lower) < 2:
            continue
        # 已知实体跳过
        if name_lower in known_labels_lower:
            continue
        # 检查是否已在本轮新增中
        if any(name_lower == v.lower() for v in new_entities.values()):
            continue

        qid = search_wikidata_entity(name)
        if qid and qid not in known_entities and qid not in new_entities:
            new_entities[qid] = name
            time.sleep(0.2)

    return new_entities


# ========================================================================== #
#  主处理流程
# ========================================================================== #

def process_mintaka_wikidata(input_path, output_path, target_samples=None):
    """为每条数据检索 Wikidata 三元组

    流程：
    1. 收集已知 Wikidata ID（来自数据集字段）
    2. 从 GPT 推理文本中提取额外实体 → 搜索 Wikidata ID
    3. 对所有实体查询 Wikidata 三元组
    4. 保存结果
    """
    print(f"正在加载数据文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    if target_samples and target_samples < len(data):
        data = data[:target_samples]
    total = len(data)

    # 断点续传
    processed_data = load_progress(output_path)
    processed_ids = {it.get("id") for it in processed_data}
    start_index = len(processed_data)

    if processed_data:
        print(f"已有进度: {start_index} 条，继续处理...")

    print(f"总数据量: {total} 条\n")

    start_time = time.time()

    for idx in range(total):
        item = data[idx]
        item_id = item.get("id", str(idx))

        if item_id in processed_ids:
            continue

        result = item.copy()
        question = item.get("question", "")
        gpt_sentence = item.get("gpt_sentence", "")
        label = item.get("generation_label", "")

        print(f"{'=' * 80}")
        print(f"[{len(processed_data) + 1}/{total}]  ID: {item_id}  标签: {label}")
        print(f"问题: {question}")

        # ---- 步骤 1：收集已知 Wikidata ID ----
        known_entities = collect_known_entity_ids(item)
        print(f"  已知实体 ({len(known_entities)}): "
              + ", ".join(f"{v}({k})" for k, v in list(known_entities.items())[:5])
              + ("..." if len(known_entities) > 5 else ""))

        # ---- 步骤 2：从 GPT 推理文本中提取额外实体 ----
        print(f"  正在从 GPT 推理文本中提取额外实体...")
        extra_names = extract_entities_from_reasoning(question, gpt_sentence)
        print(f"  提取到 {len(extra_names)} 个实体名: {extra_names[:5]}{'...' if len(extra_names) > 5 else ''}")

        extra_entities = resolve_entity_names_to_ids(extra_names, known_entities)
        print(f"  解析到 {len(extra_entities)} 个新 Wikidata ID: "
              + ", ".join(f"{v}({k})" for k, v in list(extra_entities.items())[:5]))

        all_entities = {**known_entities, **extra_entities}

        # ---- 步骤 3：查询 Wikidata 三元组 ----
        all_triples = []
        entity_triple_map = {}

        for qid, qlabel in all_entities.items():
            print(f"    查询 {qlabel} ({qid}) 的三元组...")
            triples = get_entity_triples(qid)
            entity_triple_map[qid] = {
                "label": qlabel,
                "triple_count": len(triples),
                "triples": [
                    {"head": t[0], "relation": t[1], "tail": t[2]}
                    for t in triples
                ]
            }
            all_triples.extend(triples)
            print(f"      → {len(triples)} 条三元组")
            time.sleep(0.5)

        # ---- 保存结果 ----
        result["entity_ids"] = {k: v for k, v in all_entities.items()}
        result["known_entity_count"] = len(known_entities)
        result["extra_entity_count"] = len(extra_entities)
        result["entity_triples"] = entity_triple_map
        result["total_triple_count"] = len(all_triples)

        processed_data.append(result)

        print(f"  合计: {len(all_entities)} 个实体, {len(all_triples)} 条三元组")

        # 每 5 条保存一次
        if len(processed_data) % 5 == 0:
            elapsed = time.time() - start_time
            done = len(processed_data) - start_index
            avg = elapsed / max(done, 1)
            remaining = (total - len(processed_data)) * avg
            print(f"\n💾 保存进度 ({len(processed_data)}/{total})  "
                  f"平均 {avg:.1f}s/条  预计剩余 {remaining / 60:.1f} 分钟")
            save_data(processed_data, output_path)

    # 最终保存
    save_data(processed_data, output_path)
    total_time = time.time() - start_time

    total_triples = sum(it.get("total_triple_count", 0) for it in processed_data)
    avg_triples = total_triples / max(len(processed_data), 1)

    print(f"\n{'=' * 80}")
    print(f"处理完成！结果已保存到: {output_path}")
    print(f"\n📊 最终统计:")
    print(f"  - 总数据量:     {len(processed_data)}")
    print(f"  - 三元组总数:   {total_triples}")
    print(f"  - 平均三元组/条: {avg_triples:.1f}")
    print(f"  - 总耗时:       {total_time / 60:.1f} 分钟")

    return processed_data


# ========================================================================== #
#  入口
# ========================================================================== #

if __name__ == "__main__":
    SPLIT = "dev"
    input_file = f"data/mintaka_{SPLIT}_with_answers_filtered.json"
    output_file = f"data/mintaka_{SPLIT}_with_wikidata_triples.json"

    TARGET_SAMPLES = None

    os.makedirs("data", exist_ok=True)

    print(f"""
{'=' * 80}
Mintaka Wikidata 三元组检索脚本
{'=' * 80}
对应论文 KGR 的 Entity Detection + KG Retrieval 阶段

配置:
  - 输入: {input_file}
  - 输出: {output_file}
  - LLM:  {LLM_MODEL}（用于从推理文本提取额外实体）
  - 每实体最大三元组数: {MAX_TRIPLES_PER_ENTITY}
  - 目标样本数: {'全部' if TARGET_SAMPLES is None else TARGET_SAMPLES}

流程:
  1. 收集数据集中已有的 Wikidata ID（问题实体 + 答案实体 + 支持实体）
  2. 用 LLM 从 GPT 推理文本中提取额外实体 → Wikidata 搜索获取 ID
  3. 对所有实体查询 Wikidata SPARQL 获取本地子图三元组
  4. 三元组以结构化格式保存到输出文件
{'=' * 80}
    """)

    try:
        result = process_mintaka_wikidata(
            input_file, output_file,
            target_samples=TARGET_SAMPLES
        )
        print(f"\n✅ 成功处理 {len(result)} 条数据")
    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
