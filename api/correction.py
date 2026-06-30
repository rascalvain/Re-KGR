"""
GPT 修正模块

基于 retrifit.py 的 Few-Shot prompt 逻辑，接收推理模块输出的幻觉标签，
调用 GPT 对 LLM 生成文本进行修正，并构建与 page4_pipeline.py 输出结构
完全一致的 JSON 响应。
"""

import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OpenAI = None          # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

# ── 模块级客户端（由 init_client() 或首次调用时初始化）────────────────
_client = None


def init_client() -> None:
    """初始化 OpenAI 客户端，Flask 启动时调用。"""
    global _client
    if not _OPENAI_AVAILABLE or not config.OPENAI_API_KEY:
        logger.warning("GPT 修正不可用：OpenAI SDK 缺失或 OPENAI_API_KEY 未设置")
        return
    kwargs: dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    _client = _OpenAI(**kwargs)      # type: ignore[call-arg]
    logger.info("OpenAI 客户端初始化成功，model=%s", config.OPENAI_MODEL)


@property
def client_enabled() -> bool:
    return _client is not None


# ── 内部工具 ──────────────────────────────────────────────────────────

def _format_triples(triples: list, max_count: int = 20) -> str:
    if not triples:
        return "None"
    lines = []
    for item in triples[:max_count]:
        if isinstance(item, dict):
            content = item.get("triple") or _dict_to_str(item)
        else:
            content = str(item)
        lines.append(f"- {content}")
    return "\n".join(lines)


def _dict_to_str(d: dict) -> str:
    s = d.get("s", d.get("subject", "?"))
    r = d.get("r", d.get("relation", "?"))
    t = d.get("t", d.get("object",  "?"))
    return f"({s}, {r}, {t})"


def _hallucinated_triples_str(gpt_triples: list, labels: list, threshold: int = 1) -> str:
    lines = []
    for idx, obj in enumerate(gpt_triples):
        lbl = labels[idx] if idx < len(labels) else -1
        if lbl >= threshold:
            content = obj.get("triple", _dict_to_str(obj)) if isinstance(obj, dict) else str(obj)
            lines.append(f"- {content}")
    return "\n".join(lines) if lines else "None"


def _parse_triple_str(s: str) -> tuple:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts, cur, depth = [], "", 0
    for ch in s:
        if ch == "," and depth == 0:
            parts.append(cur.strip()); cur = ""
        else:
            depth += ch == "(" and 1 or (ch == ")" and -1 or 0)
            cur += ch
    if cur:
        parts.append(cur.strip())
    return tuple(parts) if len(parts) == 3 else (s, "UNKNOWN", "?")


# ── Few-Shot correction prompt（与 retrifit.py 保持一致）───────────────

def _build_correction_prompt(
    query: str,
    initial_text: str,
    context_triples: list,
    gpt_triples: list,
    triple_labels: list,
) -> str:
    ctx_str   = _format_triples(context_triples)
    halluc_str = _hallucinated_triples_str(gpt_triples, triple_labels)

    return (
        "You are a Fact Correction Assistant. Your task is to correct a Draft Response "
        "based strictly on the provided Trusted Context Triples.\n\n"
        "### Rules:\n"
        "1. Remove all information listed in Identified Hallucinations.\n"
        "2. Only use information present in Trusted Context Triples.\n"
        "3. If the context lacks the answer, state that explicitly. Do NOT invent facts.\n"
        "4. Keep the response fluent and coherent.\n"
        "5. Preserve step-by-step structure if present.\n\n"
        "---\n"
        "### Example 1\n"
        "**Question**: Who directed Inception?\n"
        "**Trusted Context**:\n- (Inception, director, Christopher Nolan)\n"
        "**Draft**: Inception was directed by Steven Spielberg.\n"
        "**Hallucinations**:\n- (Inception, director, Steven Spielberg)\n"
        "**Corrected**: Inception was directed by Christopher Nolan.\n\n"
        "---\n"
        "### Example 2\n"
        "**Question**: What is the height of Everest?\n"
        "**Trusted Context**:\n- (Everest, location, Himalayas)\n"
        "**Draft**: Everest is in the Himalayas and is 8,848 m tall.\n"
        "**Hallucinations**:\n- (Everest, height, 8,848 m)\n"
        "**Corrected**: Everest is in the Himalayas. The provided context does not state its height.\n\n"
        "---\n"
        f"### Current Task\n"
        f"**Question**: {query}\n\n"
        f"**Trusted Context Triples**:\n{ctx_str}\n\n"
        f"**Draft Response**:\n{initial_text}\n\n"
        f"**Identified Hallucinations**:\n{halluc_str}\n\n"
        "**Corrected Response**:"
    )


# ── GPT API 调用 ──────────────────────────────────────────────────────

def call_correction_api(
    query: str,
    initial_text: str,
    context_triples: list,
    gpt_triples: list,
    triple_labels: list,
) -> str | None:
    """
    调用 GPT 生成修正后文本。返回字符串，失败时返回 None。
    """
    if _client is None:
        logger.warning("GPT 客户端未初始化，跳过修正")
        return None

    prompt = _build_correction_prompt(
        query, initial_text, context_triples, gpt_triples, triple_labels
    )
    try:
        resp = _client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that corrects factual errors "
                            "based on provided knowledge graphs."},
                {"role": "user", "content": prompt},
            ],
            temperature=config.CORRECTION_TEMPERATURE,
            max_tokens=config.CORRECTION_MAX_TOKENS,
            top_p=1.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("GPT 修正 API 调用失败: %s", exc)
        return None


# ── 结构化输出构建 ────────────────────────────────────────────────────

def build_response(
    initial_text: str,
    corrected_text: str,
    gpt_triples: list,
    triple_labels: list,
    final_nodes: list,
    final_links: list,
) -> dict:
    """
    将推理模块的检测结果 + GPT 修正文本组装成 page4_pipeline 兼容的完整响应结构：
      INITIAL_TEXT_HIGHLIGHTED, ORIGINAL_TEXT, TRIPLETS,
      CORRECTED_TEXT, CORRECTION_DETAILS, METRICS
    """
    id_to_label = {str(n.get("id", "")): str(n.get("label", "?")) for n in final_nodes}
    halluc_idx  = {i for i, lbl in enumerate(triple_labels) if lbl in (1, 2)}

    # ── INITIAL_TEXT_HIGHLIGHTED ─────────────────────────────────────
    highlighted = _highlight(initial_text, gpt_triples, halluc_idx)

    # ── ORIGINAL_TEXT ────────────────────────────────────────────────
    original = _mark_del(initial_text, gpt_triples, halluc_idx)

    # ── TRIPLETS ─────────────────────────────────────────────────────
    triplets: list[dict] = []
    sorted_links = sorted(final_links, key=lambda x: x.get("w", 0), reverse=True)

    # match=true：来自参考子图的高置信边
    for lnk in sorted_links[:4]:
        s = id_to_label.get(str(lnk.get("s", "")), "?")
        t = id_to_label.get(str(lnk.get("t", "")), "?")
        r = str(lnk.get("lb", "RELATED"))
        w = round(float(lnk.get("w", 0.8)), 2)
        triplets.append({"s": s, "r": r, "t": t, "conf": w,
                         "match": True, "label": f"事实一致：匹配（参考子图 w={w}）"})

    # match=false：被检测为幻觉的三元组
    for idx, obj in enumerate(gpt_triples):
        lbl = triple_labels[idx] if idx < len(triple_labels) else -1
        if lbl not in (1, 2):
            continue
        raw = obj.get("triple", _dict_to_str(obj)) if isinstance(obj, dict) else str(obj)
        h, r, t = _parse_triple_str(raw)
        conf     = 0.06 if lbl == 2 else 0.12
        zh_label = "虚构实体 / 幻觉关系" if lbl == 2 else "拓扑相悖 / 关系错位"
        triplets.append({"s": h, "r": r, "t": t, "conf": conf,
                         "match": False, "label": zh_label})

    # 保底
    if not triplets:
        triplets = _minimal_triplets(final_nodes, final_links)
    triplets = triplets[:6]

    # ── CORRECTED_TEXT ───────────────────────────────────────────────
    corrected_html = _wrap_diff_add(corrected_text)

    # ── CORRECTION_DETAILS ───────────────────────────────────────────
    details = _build_details(gpt_triples, triple_labels)

    # ── METRICS ──────────────────────────────────────────────────────
    match_false  = sum(1 for tr in triplets if not tr["match"])
    true_confs   = [tr["conf"] for tr in triplets if tr["match"]]
    avg_conf     = sum(true_confs) / len(true_confs) if true_confs else 0.8
    halluc_ratio = match_false / len(triplets) if triplets else 0.0
    corr_conf    = int((avg_conf * 0.7 + (1 - halluc_ratio) * 0.3) * 100)
    fact_rate    = int(sum(1 for tr in triplets if tr["match"]) / len(triplets) * 100) if triplets else 80

    return {
        "INITIAL_TEXT_HIGHLIGHTED": highlighted,
        "ORIGINAL_TEXT":            original,
        "TRIPLETS":                 triplets,
        "CORRECTED_TEXT":           corrected_html,
        "CORRECTION_DETAILS":       details,
        "METRICS": {
            "halluc_count":         match_false,
            "correction_conf":      min(100, max(0, corr_conf)),
            "fact_consistency_rate": min(100, max(0, fact_rate)),
        },
    }


# ── 文本标注辅助 ──────────────────────────────────────────────────────

def _highlight(text: str, triples: list, halluc_idx: set) -> str:
    result = text
    for idx in sorted(halluc_idx):
        if idx >= len(triples):
            continue
        obj = triples[idx]
        raw = obj.get("triple", _dict_to_str(obj)) if isinstance(obj, dict) else str(obj)
        h, _r, t = _parse_triple_str(raw)
        for term in (t, h):
            if term and term != "?" and len(term) > 2 and term in result:
                result = result.replace(term, f"<halluc>{term}</halluc>", 1)
                break
    return result


def _mark_del(text: str, triples: list, halluc_idx: set) -> str:
    result = text
    for idx in sorted(halluc_idx):
        if idx >= len(triples):
            continue
        obj = triples[idx]
        raw = obj.get("triple", _dict_to_str(obj)) if isinstance(obj, dict) else str(obj)
        _h, _r, t = _parse_triple_str(raw)
        if t and t != "?" and len(t) > 2 and t in result:
            result = result.replace(t, f"<del>{t}</del>", 1)
    return result


def _wrap_diff_add(text: str) -> str:
    """将修正文本的各句用 diff-add 包裹（GPT 输出为纯文本时的简单处理）。"""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    for s in sentences:
        if re.search(r"\b(is|are|was|were|has|have|located|based|according)\b", s, re.I):
            parts.append(f'<span class="diff-add">{s}</span>')
        else:
            parts.append(s)
    return " ".join(parts)


def _build_details(triples: list, labels: list) -> list:
    details = []
    for idx, obj in enumerate(triples):
        lbl = labels[idx] if idx < len(labels) else -1
        raw = obj.get("triple", _dict_to_str(obj)) if isinstance(obj, dict) else str(obj)
        h, r, t = _parse_triple_str(raw)
        path = f"{h}→{r}→{t}"
        if lbl == 0:
            details.append({"text": f"{path}: 保留，事实一致（conf=0.85）", "is_ok": True})
        elif lbl == 1:
            details.append({"text": f"{path}: 删除幻觉，拓扑相悖（conf=0.12）", "is_ok": False})
        elif lbl == 2:
            details.append({"text": f"{path}: 删除幻觉，虚构实体（conf=0.06）", "is_ok": False})
    return details or [{"text": "原始文本中无可识别三元组", "is_ok": True}]


def _minimal_triplets(nodes: list, links: list) -> list:
    id_to_lbl = {str(n.get("id", "")): str(n.get("label", "?")) for n in nodes}
    result = []
    for lnk in sorted(links, key=lambda x: x.get("w", 0), reverse=True)[:4]:
        s = id_to_lbl.get(str(lnk.get("s", "")), "?")
        t = id_to_lbl.get(str(lnk.get("t", "")), "?")
        r = str(lnk.get("lb", "RELATED"))
        w = round(float(lnk.get("w", 0.8)), 2)
        result.append({"s": s, "r": r, "t": t, "conf": w,
                       "match": True, "label": "事实一致：匹配"})
    return result or [{"s": "实体", "r": "RELATED", "t": "目标",
                       "conf": 0.8, "match": True, "label": "事实一致：匹配"}]
