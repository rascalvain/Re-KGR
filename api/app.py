"""
Flask API 入口

提供三个端点：
  POST /detect    — RGAT 幻觉检测（整体判定 + 逐三元组标签）
  POST /correct   — 检测 + GPT 修正，返回与 page4_pipeline 对齐的完整结构
  GET  /health    — 健康检查

启动方式：
    cd D:\\模型\\GCA-main\\api
    python app.py
"""

import logging
import os
import sys

# 确保 api/ 目录在路径中（支持从任意工作目录启动）
_API_DIR = os.path.dirname(os.path.abspath(__file__))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from flask import Flask, jsonify, request

import config
import inference
import correction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ─── 路由 ───────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """
    RGAT 幻觉检测端点。

    请求体（JSON）：
    {
        "initial_text":     "LLM 生成的待检测文本",
        "context_triples":  [{"triple": "(h, r, t)"} | {"s":..,"r":..,"t":..}, ...],
        "gpt_triples":      [{"triple": "(h, r, t)"} | {"s":..,"r":..,"t":..}, ...]
    }

    返回（JSON）：
    {
        "hallucination_prediction":    0 | 1,
        "prediction_confidence":       float,
        "triple_hallucination_labels": [0|1|2|-1, ...],
        "triple_stats": {
            "factual": int, "possible_hallucination": int,
            "hallucination": int, "unknown": int
        }
    }
    """
    if not request.is_json:
        return jsonify({"error": "请求必须为 JSON 格式"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "无法解析 JSON 请求体"}), 400

    initial_text = data.get("initial_text", "")
    if not isinstance(initial_text, str) or not initial_text.strip():
        return jsonify({"error": "'initial_text' 字段必须为非空字符串"}), 400

    context_triples = data.get("context_triples", [])
    gpt_triples     = data.get("gpt_triples", [])

    if not isinstance(context_triples, list):
        return jsonify({"error": "'context_triples' 必须为列表"}), 400
    if not isinstance(gpt_triples, list):
        return jsonify({"error": "'gpt_triples' 必须为列表"}), 400

    try:
        sample_result = inference.detect_sample(initial_text, context_triples, gpt_triples)
        triple_result = inference.classify_triples(context_triples, gpt_triples)
    except RuntimeError as e:
        return jsonify({"error": f"服务内部错误（模型未初始化？）: {e}"}), 500
    except Exception as e:
        logger.error("检测异常: %s", e, exc_info=True)
        return jsonify({"error": "检测失败，请检查输入"}), 500

    return jsonify({**sample_result, **triple_result}), 200


@app.route("/correct", methods=["POST"])
def correct():
    """
    完整幻觉检测 + GPT 修正端点。

    请求体（JSON）：
    {
        "query":            "推理查询字符串",
        "initial_text":     "LLM 生成的初始推理文本（可能含幻觉）",
        "context_triples":  [{"triple": "(h, r, t)"}, ...],
        "gpt_triples":      [{"triple": "(h, r, t)"}, ...],
        "final_nodes":      [{"id": "n1", "label": "实体A", "type": "actor"}, ...],
        "final_links":      [{"s": "n1", "t": "n2", "lb": "ATTACKED", "w": 0.95}, ...]
    }

    返回（JSON）：与 page4_pipeline.Page4PipelineService.run_correction() 完全相同的结构：
    {
        "INITIAL_TEXT_HIGHLIGHTED": str,
        "ORIGINAL_TEXT":            str,
        "TRIPLETS":                 [...],
        "CORRECTED_TEXT":           str,
        "CORRECTION_DETAILS":       [...],
        "METRICS":                  {...},
        "meta":                     {...}
    }
    """
    if not request.is_json:
        return jsonify({"error": "请求必须为 JSON 格式"}), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "无法解析 JSON 请求体"}), 400

    # ── 字段校验 ─────────────────────────────────────────────────────
    query = str(data.get("query", "")).strip() or "推理目标行为校正"

    initial_text = data.get("initial_text", "")
    if not isinstance(initial_text, str) or not initial_text.strip():
        return jsonify({"error": "'initial_text' 字段必须为非空字符串"}), 400

    context_triples = data.get("context_triples", [])
    gpt_triples     = data.get("gpt_triples", [])
    final_nodes     = data.get("final_nodes", [])
    final_links     = data.get("final_links", [])

    for field_name, field_val in [
        ("context_triples", context_triples),
        ("gpt_triples",     gpt_triples),
        ("final_nodes",     final_nodes),
        ("final_links",     final_links),
    ]:
        if not isinstance(field_val, list):
            return jsonify({"error": f"'{field_name}' 必须为列表"}), 400

    # ── Step 1: RGAT 检测 ────────────────────────────────────────────
    try:
        sample_det = inference.detect_sample(initial_text, context_triples, gpt_triples)
        triple_det = inference.classify_triples(context_triples, gpt_triples)
    except RuntimeError as e:
        return jsonify({"error": f"模型未初始化: {e}"}), 500
    except Exception as e:
        logger.error("检测异常: %s", e, exc_info=True)
        return jsonify({"error": "幻觉检测失败"}), 500

    triple_labels = triple_det["triple_hallucination_labels"]

    # ── Step 2: GPT 修正 ─────────────────────────────────────────────
    corrected_text_raw = correction.call_correction_api(
        query=query,
        initial_text=initial_text,
        context_triples=context_triples,
        gpt_triples=gpt_triples,
        triple_labels=triple_labels,
    )

    gpt_succeeded = corrected_text_raw is not None
    if not gpt_succeeded:
        # GPT 不可用时，用检测结果标注后的原文作为修正文
        corrected_text_raw = initial_text

    # ── Step 3: 构建完整响应 ──────────────────────────────────────────
    try:
        result = correction.build_response(
            initial_text=initial_text,
            corrected_text=corrected_text_raw,
            gpt_triples=gpt_triples,
            triple_labels=triple_labels,
            final_nodes=final_nodes,
            final_links=final_links,
        )
    except Exception as e:
        logger.error("响应构建异常: %s", e, exc_info=True)
        return jsonify({"error": "响应构建失败"}), 500

    result["meta"] = {
        "source":                    "model+gpt" if gpt_succeeded else "model-only",
        "hallucination_prediction":  sample_det["hallucination_prediction"],
        "prediction_confidence":     sample_det["prediction_confidence"],
        "triple_stats":              triple_det["triple_stats"],
        "gpt_correction_succeeded":  gpt_succeeded,
    }

    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    """健康检查端点。"""
    model_loaded = inference._model is not None
    gpt_ready    = correction._client is not None
    return jsonify({
        "status":       "ok" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "gpt_ready":    gpt_ready,
    }), 200 if model_loaded else 503


# ─── 启动 ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("正在加载 RGAT 模型，请稍候...")
    try:
        inference.load_model()
        logger.info("RGAT 模型加载完毕")
    except Exception as e:
        logger.error("模型加载失败: %s", e, exc_info=True)
        raise

    logger.info("初始化 GPT 修正客户端...")
    correction.init_client()

    logger.info("服务就绪，监听 %s:%d", config.FLASK_HOST, config.FLASK_PORT)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        use_reloader=False,   # 禁止热重载，避免模型二次加载
    )
