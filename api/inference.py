"""
模型推理模块

负责一次性加载 HybridGraphTextClassifier（RGAT + SentenceBERT），
并提供两个推理接口：
  - detect_sample()    : 对单个样本做整体幻觉判定
  - classify_triples() : 对三元组列表逐条打幻觉标签
"""

import logging
import os
import pickle
import sys

import torch
import torch.nn.functional as F

import config

logger = logging.getLogger(__name__)

# ── RGAT 包路径注入 ───────────────────────────────────────────────────
if config.RGAT_ROOT not in sys.path:
    sys.path.insert(0, config.RGAT_ROOT)

# ── 模块级延迟变量（由 load_model() 初始化）──────────────────────────
_model        = None
_graph_encoder = None
_entity2id:   dict = {}
_relation2id: dict = {}
_device       = None


def load_model() -> None:
    """
    一次性加载所有推理所需资源，Flask 启动时调用一次。

    加载流程：
    1. 注入 RGAT 包路径
    2. 加载 HybridGraphTextClassifier checkpoint
    3. 加载 entity2idx.pkl / relation2idx.pkl
    """
    global _model, _graph_encoder, _entity2id, _relation2id, _device

    _device = torch.device(config.DEVICE)
    logger.info("推理设备: %s", _device)

    # ── 1. 导入模型类 ─────────────────────────────────────────────────
    from framework.hybrid_graph_text_classifier import load_hybrid_classifier

    # ── 2. 加载 checkpoint ───────────────────────────────────────────
    if not os.path.isfile(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint 不存在: {config.CHECKPOINT_PATH}")

    logger.info("加载 HybridGraphTextClassifier: %s", config.CHECKPOINT_PATH)
    _model, model_cfg = load_hybrid_classifier(config.CHECKPOINT_PATH, device=str(_device))
    _model.eval()
    _graph_encoder = _model.graph_encoder
    logger.info("模型加载成功，配置: %s", model_cfg)

    # ── 3. 加载实体 / 关系映射 ────────────────────────────────────────
    if not os.path.isfile(config.ENTITY_MAPPING_PATH):
        raise FileNotFoundError(f"实体映射不存在: {config.ENTITY_MAPPING_PATH}")
    if not os.path.isfile(config.RELATION_MAPPING_PATH):
        raise FileNotFoundError(f"关系映射不存在: {config.RELATION_MAPPING_PATH}")

    with open(config.ENTITY_MAPPING_PATH, "rb") as f:
        _entity2id = pickle.load(f)
    with open(config.RELATION_MAPPING_PATH, "rb") as f:
        _relation2id = pickle.load(f)

    logger.info(
        "映射加载成功: %d 实体, %d 关系",
        len(_entity2id), len(_relation2id),
    )


# ── 内部图构建工具 ────────────────────────────────────────────────────

def _parse_triple_str(s: str) -> tuple:
    """解析 '(head, relation, tail)' 字符串，返回 (h, r, t) 或 (None,None,None)。"""
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
    return tuple(parts) if len(parts) == 3 else (None, None, None)


def _triple_obj_to_parts(obj: dict) -> tuple:
    """从三元组 dict 中提取 (h, r, t)，支持 triple 字符串和 s/r/t 键。"""
    if "triple" in obj:
        return _parse_triple_str(str(obj["triple"]))
    h = str(obj.get("s", obj.get("subject", ""))).strip()
    r = str(obj.get("r", obj.get("relation", ""))).strip()
    t = str(obj.get("t", obj.get("object",  ""))).strip()
    return (h, r, t) if h and r and t else (None, None, None)


def _build_graph(triples: list) -> "torch_geometric.data.Data | None":
    """将三元组列表转换为 PyG Data 对象（与 data_loader_hotpotqa.py 逻辑对齐）。"""
    from torch_geometric.data import Data

    entity_set: dict[str, int] = {}
    edge_src, edge_dst, edge_types = [], [], []

    def _eid(name: str) -> int:
        if name not in entity_set:
            entity_set[name] = len(entity_set)
        return entity_set[name]

    for obj in triples:
        h, r, t = _triple_obj_to_parts(obj) if isinstance(obj, dict) else _parse_triple_str(str(obj))
        if not h or not r or not t:
            continue
        edge_src.append(_eid(h))
        edge_dst.append(_eid(t))
        edge_types.append(_relation2id.get(r, 0))

    if not entity_set:
        return None

    node_ids = torch.LongTensor([_entity2id.get(n, 0) for n in entity_set])

    if not edge_src:
        n = len(entity_set)
        edge_index = torch.LongTensor([list(range(n)), list(range(n))])
        edge_type  = torch.LongTensor([0] * n)
    else:
        edge_index = torch.LongTensor([edge_src, edge_dst])
        edge_type  = torch.LongTensor(edge_types)

    return Data(node_ids=node_ids, edge_index=edge_index,
                edge_type=edge_type, num_nodes=len(entity_set))


def _single_triple_graph(obj: dict) -> "torch_geometric.data.Data | None":
    """将单条三元组 dict 转为 2 节点 PyG Data 对象。"""
    from torch_geometric.data import Data

    h, r, t = _triple_obj_to_parts(obj)
    if not h or not r or not t:
        return None
    return Data(
        node_ids   = torch.LongTensor([_entity2id.get(h, 0), _entity2id.get(t, 0)]),
        edge_index = torch.LongTensor([[0], [1]]),
        edge_type  = torch.LongTensor([_relation2id.get(r, 0)]),
        num_nodes  = 2,
    )


def _cosine_sim(g1, g2) -> float:
    from torch_geometric.data import Batch
    with torch.no_grad():
        h1 = _graph_encoder(Batch.from_data_list([g1]).to(_device))
        h2 = _graph_encoder(Batch.from_data_list([g2]).to(_device))
        return float(F.cosine_similarity(h1, h2, dim=-1).item())


def _classify_by_sim(sim: float) -> int:
    """余弦相似度 → 幻觉标签（0/1/2）。"""
    score = (1.0 - sim) / 2.0
    if score >= config.HALLUCINATION_THRESH:
        return 2
    if score >= config.POSSIBLE_THRESH:
        return 1
    return 0


# ── 公开推理接口 ──────────────────────────────────────────────────────

def detect_sample(
    gpt_text: str,
    context_triples: list,
    gpt_triples: list,
) -> dict:
    """
    对单个样本做整体幻觉判定（0=事实 / 1=幻觉）。

    Parameters
    ----------
    gpt_text        : LLM 生成的文本
    context_triples : 参考知识图谱三元组列表（每项为 dict 或字符串）
    gpt_triples     : 从 gpt_text 中提取的三元组列表

    Returns
    -------
    {
        "hallucination_prediction": 0 | 1,
        "prediction_confidence":    float,
    }
    """
    if _model is None:
        raise RuntimeError("模型未初始化，请先调用 load_model()")

    from torch_geometric.data import Batch

    ctx_graph = _build_graph(context_triples)
    gpt_graph = _build_graph(gpt_triples)

    # 退化兜底：单节点自环图
    if ctx_graph is None:
        from torch_geometric.data import Data
        ctx_graph = Data(node_ids=torch.LongTensor([0]),
                         edge_index=torch.LongTensor([[0],[0]]),
                         edge_type=torch.LongTensor([0]), num_nodes=1)
    if gpt_graph is None:
        from torch_geometric.data import Data
        gpt_graph = Data(node_ids=torch.LongTensor([0]),
                         edge_index=torch.LongTensor([[0],[0]]),
                         edge_type=torch.LongTensor([0]), num_nodes=1)

    with torch.no_grad():
        logits = _model(
            Batch.from_data_list([ctx_graph]).to(_device),
            Batch.from_data_list([gpt_graph]).to(_device),
            [gpt_text],
        )
        probs      = torch.softmax(logits, dim=-1)
        prediction = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, prediction].item())

    return {
        "hallucination_prediction": prediction,
        "prediction_confidence":    round(confidence, 4),
    }


def classify_triples(
    context_triples: list,
    gpt_triples: list,
) -> dict:
    """
    对 gpt_triples 中每条三元组独立打幻觉标签。

    Returns
    -------
    {
        "triple_hallucination_labels": [0|1|2|-1, ...],
        "triple_stats": {
            "factual": int,
            "possible_hallucination": int,
            "hallucination": int,
            "unknown": int,
        },
    }
    """
    if _model is None:
        raise RuntimeError("模型未初始化，请先调用 load_model()")

    ctx_graph = _build_graph(context_triples)
    if ctx_graph is None:
        labels = [-1] * len(gpt_triples)
        return {
            "triple_hallucination_labels": labels,
            "triple_stats": {"factual": 0, "possible_hallucination": 0,
                             "hallucination": 0, "unknown": len(gpt_triples)},
        }

    labels = []
    for obj in gpt_triples:
        t_graph = _single_triple_graph(obj) if isinstance(obj, dict) else None
        if t_graph is None:
            labels.append(-1)
            continue
        sim   = _cosine_sim(ctx_graph, t_graph)
        labels.append(_classify_by_sim(sim))

    return {
        "triple_hallucination_labels": labels,
        "triple_stats": {
            "factual":              labels.count(0),
            "possible_hallucination": labels.count(1),
            "hallucination":        labels.count(2),
            "unknown":              labels.count(-1),
        },
    }
