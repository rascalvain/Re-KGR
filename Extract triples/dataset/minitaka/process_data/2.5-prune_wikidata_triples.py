"""
Mintaka 数据集处理 — 第 2.5 步：Wikidata 三元组剪枝

参考论文 PoG (Paths-over-Graph, WWW'25) 的 Fuzzy + Precise Path
Selection 策略 (Section 4.3, Algorithm 2)，采用两阶段剪枝将每条数据的
Wikidata 三元组缩减到 W_MAX 以内：

  阶段 1  Fuzzy Selection
      用 SBERT 将 "问题 + GPT推理文本" 编码为查询向量，
      将每条三元组编码为候选向量，按余弦相似度降序排列，
      保留 Top-W1 条候选三元组。

  阶段 2  Precise Selection
      将阶段 1 的 W1 条候选三元组分批送入 LLM，由 LLM
      对每条三元组评分（1-5），综合 SBERT 分数与 LLM 评分
      的加权得分，最终保留 Top-W_MAX 条最相关三元组。

输入：第 2 步输出 mintaka_*_with_wikidata_triples.json
输出：剪枝后的 mintaka_*_with_wikidata_triples_pruned.json
"""

import json
import time
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import openai

# ========================================================================== #
#  配置
# ========================================================================== #

# 两阶段剪枝参数
W1 = 200            # Fuzzy Selection 保留的候选数（宽松筛选）
W_MAX = 100          # Precise Selection 最终保留的三元组数
LLM_BATCH_SIZE = 40  # 每批送入 LLM 精选的三元组数

# 加权融合参数: final_score = α * sbert_score_norm + (1-α) * llm_score_norm
ALPHA = 0.4

# 断点续传：每隔多少条保存一次进度
SAVE_INTERVAL = 50

# API 配置
client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key="sk-IQ8vi7XzSOgnTAW805DchQy2YVSOA8q6WYb7vUZRYOHKN6vN",
)
LLM_MODEL = "gpt-3.5-turbo"

# 加载语义模型
SEMANTIC_MODEL_NAME = "all-mpnet-base-v2"
print(f"正在加载语义模型: {SEMANTIC_MODEL_NAME} ...")
from transformers import AutoModel
_orig = AutoModel.from_pretrained
AutoModel.from_pretrained = lambda *a, **kw: _orig(*a, **{**kw, "ignore_mismatched_sizes": True})
_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
AutoModel.from_pretrained = _orig
print("语义模型加载完成。")

# Precise Selection 的 prompt
PRECISE_SELECTION_PROMPT = """You are a knowledge graph expert. Given a question and its reasoning text, score each knowledge triple for how relevant and useful it is to verify or answer the question.

Question: {question}
Reasoning: {reasoning}

Score each triple from 1 to 5:
  5 = Directly supports or contradicts a key claim in the reasoning
  4 = Highly relevant to the question's core entities or relationships
  3 = Moderately relevant, provides useful context
  2 = Weakly relevant, only tangentially related
  1 = Irrelevant or too generic to be useful

Candidate triples:
{triples_text}

Output ONLY a JSON list of scores in the same order as the triples above, e.g.:
[4, 2, 5, 1, 3, ...]"""


# ========================================================================== #
#  工具函数
# ========================================================================== #

def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_progress(save_path):
    """加载已有进度，用于断点续传。文件不存在或为空时返回 []"""
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            out = json.load(f)
        return out if isinstance(out, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def triple_to_text(t):
    """三元组字典 → 自然语言句子，用于语义编码"""
    return f"{t['head']} {t['relation']} {t['tail']}"


def request_llm(prompt, temperature=0.0, max_tokens=800):
    """调用 LLM API，带重试"""
    for attempt in range(5):
        try:
            params = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "n": 1,
            }
            try:
                params["max_tokens"] = max_tokens
                resp = client.chat.completions.create(**params)
            except (TypeError, openai.BadRequestError):
                params.pop("max_tokens", None)
                params["max_completion_tokens"] = max_tokens * 2
                resp = client.chat.completions.create(**params)

            return resp.choices[0].message.content.strip()

        except openai.RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    LLM 错误 (attempt {attempt + 1}): {e}")
            time.sleep(1)
    return None


def parse_llm_scores(response_text, expected_count):
    """从 LLM 返回的文本中解析评分列表

    容错策略：优先解析 JSON 列表，失败则逐行提取数字。
    """
    if not response_text:
        return [3.0] * expected_count

    # 尝试直接解析 JSON 列表
    try:
        match = re.search(r'\[[\s\S]*?\]', response_text)
        if match:
            scores = json.loads(match.group())
            scores = [max(1, min(5, float(s))) for s in scores]
            if len(scores) == expected_count:
                return scores
            if len(scores) > expected_count:
                return scores[:expected_count]
            return scores + [3.0] * (expected_count - len(scores))
    except (json.JSONDecodeError, ValueError):
        pass

    # 逐行提取数字
    numbers = re.findall(r'\b([1-5])\b', response_text)
    scores = [float(n) for n in numbers[:expected_count]]
    if len(scores) < expected_count:
        scores += [3.0] * (expected_count - len(scores))
    return scores


# ========================================================================== #
#  两阶段剪枝核心
# ========================================================================== #

def fuzzy_selection(candidates, question, gpt_sentence, top_k):
    """阶段 1: Fuzzy Selection（SBERT 语义排序）

    将问题+推理文本作为查询，对所有候选三元组编码并按余弦相似度排序。
    返回 (selected_candidates, sbert_scores)，每个元素保持原始结构。
    """
    query_text = f"{question} {gpt_sentence}"
    candidate_texts = [triple_to_text(x["triple"]) for x in candidates]

    query_emb = _model.encode([query_text], convert_to_tensor=True)
    cand_embs = _model.encode(candidate_texts, convert_to_tensor=True,
                              batch_size=256, show_progress_bar=False)

    scores = util.cos_sim(query_emb, cand_embs)[0].cpu().numpy()

    top_indices = np.argsort(scores)[::-1][:top_k]
    selected = [candidates[i] for i in top_indices]
    selected_scores = [float(scores[i]) for i in top_indices]

    return selected, selected_scores


def precise_selection(candidates, sbert_scores, question, gpt_sentence, top_k):
    """阶段 2: Precise Selection（LLM 精选）

    将 Fuzzy Selection 的候选三元组分批送入 LLM 评分，
    结合 SBERT 分数加权融合，取 Top-W_MAX。

    返回 (final_candidates, final_scores)。
    """
    n = len(candidates)
    llm_scores = [3.0] * n
    reasoning_short = gpt_sentence[:500] if len(gpt_sentence) > 500 else gpt_sentence

    for batch_start in range(0, n, LLM_BATCH_SIZE):
        batch_end = min(batch_start + LLM_BATCH_SIZE, n)
        batch = candidates[batch_start:batch_end]

        triples_text = "\n".join(
            f"  {i + 1}. ({t['triple']['head']}, {t['triple']['relation']}, {t['triple']['tail']})"
            for i, t in enumerate(batch)
        )

        prompt = PRECISE_SELECTION_PROMPT.format(
            question=question,
            reasoning=reasoning_short,
            triples_text=triples_text
        )

        response = request_llm(prompt, temperature=0.0, max_tokens=400)
        batch_scores = parse_llm_scores(response, len(batch))
        llm_scores[batch_start:batch_end] = batch_scores

    # 归一化 SBERT 分数到 [0, 1]
    sbert_arr = np.array(sbert_scores)
    sbert_min, sbert_max = sbert_arr.min(), sbert_arr.max()
    if sbert_max > sbert_min:
        sbert_norm = (sbert_arr - sbert_min) / (sbert_max - sbert_min)
    else:
        sbert_norm = np.ones_like(sbert_arr) * 0.5

    # 归一化 LLM 评分到 [0, 1]（原始范围 1-5）
    llm_arr = np.array(llm_scores)
    llm_norm = (llm_arr - 1.0) / 4.0

    # 加权融合
    final_scores = ALPHA * sbert_norm + (1 - ALPHA) * llm_norm

    top_indices = np.argsort(final_scores)[::-1][:top_k]
    selected = [candidates[i] for i in top_indices]
    selected_scores = [float(final_scores[i]) for i in top_indices]

    return selected, selected_scores


def prune_triples_for_item(item):
    """对单条数据执行两阶段剪枝 (Fuzzy + Precise Selection)

    返回 (pruned_entity_triples, stats_dict)
    """
    entity_triples = item.get("entity_triples", {})
    question = item.get("question", "")
    gpt_sentence = item.get("gpt_sentence", "")

    all_triples = []
    for eid, info in entity_triples.items():
        for t in info.get("triples", []):
            all_triples.append({"triple": t, "entity_id": eid})

    original_count = len(all_triples)
    if original_count == 0:
        return entity_triples, {
            "original": 0, "after_fuzzy": 0,
            "after_precise": 0, "llm_calls": 0
        }

    # 三元组数已 <= W_MAX，无需剪枝
    if original_count <= W_MAX:
        return entity_triples, {
            "original": original_count, "after_fuzzy": original_count,
            "after_precise": original_count, "llm_calls": 0
        }

    # ---- 阶段 1：Fuzzy Selection ----
    fuzzy_top = min(W1, original_count)
    fuzzy_selected, sbert_scores = fuzzy_selection(
        all_triples, question, gpt_sentence, fuzzy_top)
    after_fuzzy = len(fuzzy_selected)

    # Fuzzy 后已 <= W_MAX，无需 Precise Selection
    if after_fuzzy <= W_MAX:
        pruned = _rebuild_entity_triples(entity_triples, fuzzy_selected)
        return pruned, {
            "original": original_count, "after_fuzzy": after_fuzzy,
            "after_precise": after_fuzzy, "llm_calls": 0
        }

    # ---- 阶段 2：Precise Selection（LLM 精选） ----
    llm_calls = (after_fuzzy + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
    precise_selected, _ = precise_selection(
        fuzzy_selected, sbert_scores, question, gpt_sentence, W_MAX)
    after_precise = len(precise_selected)

    pruned = _rebuild_entity_triples(entity_triples, precise_selected)
    return pruned, {
        "original": original_count, "after_fuzzy": after_fuzzy,
        "after_precise": after_precise, "llm_calls": llm_calls
    }


def _rebuild_entity_triples(original_map, selected_items):
    """根据保留的三元组列表重建 entity_triples 结构"""
    new_map = {}
    for x in selected_items:
        eid = x["entity_id"]
        if eid not in new_map:
            orig = original_map.get(eid, {})
            new_map[eid] = {
                "label": orig.get("label", eid),
                "triple_count": 0,
                "triples": []
            }
        new_map[eid]["triples"].append(x["triple"])
        new_map[eid]["triple_count"] = len(new_map[eid]["triples"])
    return new_map


# ========================================================================== #
#  主处理
# ========================================================================== #

def prune_all(input_path, output_path):
    data = load_data(input_path)
    total = len(data)

    # 断点续传：加载已有进度
    processed_data = load_progress(output_path)
    start_index = len(processed_data)
    if start_index > 0:
        print(f"📂 已有进度: {start_index} 条，从第 {start_index + 1} 条继续...\n")

    print(f"总数据量: {total}")
    print(f"Fuzzy Selection 上限 W1: {W1}")
    print(f"Precise Selection 最终上限 W_MAX: {W_MAX}")
    print(f"SBERT/LLM 加权系数 α: {ALPHA}\n")

    start_time = time.time()
    stats_agg = {
        "original": 0, "after_fuzzy": 0,
        "after_precise": 0, "llm_calls": 0,
        "items_used_precise": 0,
    }

    # 累积已有进度的统计
    for item in processed_data:
        s = item.get("prune_stats", {})
        stats_agg["original"] += s.get("original", 0)
        stats_agg["after_fuzzy"] += s.get("after_fuzzy", 0)
        stats_agg["after_precise"] += s.get("after_precise", 0)
        stats_agg["llm_calls"] += s.get("llm_calls", 0)
        if s.get("llm_calls", 0) > 0:
            stats_agg["items_used_precise"] += 1

    # 从 start_index 起处理剩余数据
    done_in_run = 0
    try:
        for idx in range(start_index, total):
            item = data[idx]
            pruned_triples, stats = prune_triples_for_item(item)

            item["entity_triples"] = pruned_triples
            item["total_triple_count"] = stats["after_precise"]
            item["prune_stats"] = stats

            processed_data.append(item)
            done_in_run += 1

            stats_agg["original"] += stats["original"]
            stats_agg["after_fuzzy"] += stats["after_fuzzy"]
            stats_agg["after_precise"] += stats["after_precise"]
            stats_agg["llm_calls"] += stats["llm_calls"]
            if stats["llm_calls"] > 0:
                stats_agg["items_used_precise"] += 1

            if done_in_run % 20 == 0 or idx == total - 1:
                elapsed = time.time() - start_time
                avg = elapsed / max(done_in_run, 1)
                remaining = (total - idx - 1) * avg
                stage_info = (f"原始 {stats['original']} "
                              f"→ Fuzzy {stats['after_fuzzy']} "
                              f"→ Precise {stats['after_precise']}")
                if stats["llm_calls"] > 0:
                    stage_info += f" (LLM×{stats['llm_calls']})"
                print(f"[{len(processed_data)}/{total}]  {stage_info}  "
                      f"(累计 {elapsed:.0f}s, 剩余约 {remaining / 60:.1f}min)")

            if done_in_run % 50 == 0 or idx == total - 1:
                save_data(processed_data, output_path)
                print(f"  💾 中间保存完成 ({len(processed_data)}/{total})")
    except KeyboardInterrupt:
        save_data(processed_data, output_path)
        print(f"\n⏸️  用户中断，进度已保存 ({len(processed_data)}/{total})，可重新运行以断点续传")
        raise

    save_data(processed_data, output_path)
    total_time = time.time() - start_time

    orig = stats_agg["original"]
    af = stats_agg["after_fuzzy"]
    ap = stats_agg["after_precise"]

    print(f"\n{'=' * 80}")
    print(f"剪枝完成！结果已保存到: {output_path}")
    print(f"\n📊 两阶段剪枝统计:")
    print(f"  阶段 1 — Fuzzy Selection (SBERT Top-{W1}):")
    print(f"    {orig} → {af}  (减少 {orig - af}, "
          f"-{(orig - af) / max(orig, 1) * 100:.1f}%)")
    print(f"  阶段 2 — Precise Selection (LLM Top-{W_MAX}):")
    print(f"    {af} → {ap}  (减少 {af - ap}, "
          f"-{(af - ap) / max(af, 1) * 100:.1f}%)")
    print(f"\n  总压缩: {orig} → {ap}  "
          f"(保留 {ap / max(orig, 1) * 100:.1f}%)")
    print(f"  平均三元组/条: {ap / max(total, 1):.1f}")
    print(f"  使用 Precise Selection 的数据条数: {stats_agg['items_used_precise']}")
    print(f"  LLM 总调用次数: {stats_agg['llm_calls']}")
    print(f"  总耗时: {total_time:.1f} 秒 ({total_time / 60:.1f} 分钟)")


# ========================================================================== #
#  入口
# ========================================================================== #

if __name__ == "__main__":
    SPLIT = "dev"
    input_file = f"data/mintaka_{SPLIT}_with_wikidata_triples.json"
    output_file = f"data/mintaka_{SPLIT}_with_wikidata_triples_pruned.json"

    os.makedirs("data", exist_ok=True)

    print(f"""
{'=' * 80}
Wikidata 三元组剪枝（参考 PoG Fuzzy + Precise Path Selection）
{'=' * 80}

配置:
  输入:                {input_file}
  输出:                {output_file}
  语义模型:            {SEMANTIC_MODEL_NAME}
  LLM:                 {LLM_MODEL}
  Fuzzy Selection W1:  {W1}
  Precise Selection:   {W_MAX}
  融合权重 α(SBERT):   {ALPHA}  (LLM: {1 - ALPHA})
  LLM 批次大小:        {LLM_BATCH_SIZE}

两阶段剪枝流程:
  阶段 1  Fuzzy Selection (SBERT)
          编码问题+推理文本 vs 三元组，余弦相似度取 Top-{W1}
  阶段 2  Precise Selection (LLM)
          LLM 为每条三元组评分(1-5)，加权融合 SBERT 分数，
          最终保留 Top-{W_MAX}

注: 若三元组数 ≤ {W_MAX}，直接保留不剪枝；
    若阶段 1 后三元组数 ≤ {W_MAX}，跳过阶段 2（节省 LLM 调用）。
{'=' * 80}
    """)

    try:
        prune_all(input_file, output_file)
        print("\n✅ 剪枝完成")
    except KeyboardInterrupt:
        print("\n⏸️  用户中断，进度已保存，可重新运行以断点续传")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
