"""
API 配置中心

所有路径和运行参数在此处统一管理，其他模块只从此处导入。
部署时按实际环境填写各路径。

启动：
    cd D:\模型\GCA-main\api
    python app.py
"""

import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── 项目根目录 ────────────────────────────────────────────────────────
# api/ 的上级即为 GCA-main 根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RGAT 模块所在目录（含 framework / dataloader 子包）
RGAT_ROOT = os.path.join(PROJECT_ROOT, "final_structure", "rgat")

# ── 模型路径（部署时填写） ────────────────────────────────────────────
# 训练好的 checkpoint（含 best_model.pth）
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "hotpotqa", "data", "final_data", "rgat_output",
    "hybrid_20260119_145940(目前去最高)", "checkpoints", "best_model.pth"
)

# TransE/混合嵌入目录（含 entity2idx.pkl / relation2idx.pkl）
EMBEDDINGS_DIR = os.path.join(
    PROJECT_ROOT,
    "hotpotqa", "data", "final_data", "final_hybrid_embeddings"
)
ENTITY_MAPPING_PATH  = os.path.join(EMBEDDINGS_DIR, "entity2idx.pkl")
RELATION_MAPPING_PATH = os.path.join(EMBEDDINGS_DIR, "relation2idx.pkl")

# ── 检测阈值 ──────────────────────────────────────────────────────────
# hallucination_score = (1 - cosine_similarity) / 2
# >= HALLUCINATION_THRESH  → label 2（幻觉）
# >= POSSIBLE_THRESH       → label 1（可能幻觉）
# <  POSSIBLE_THRESH       → label 0（事实）
HALLUCINATION_THRESH: float = 0.6
POSSIBLE_THRESH:      float = 0.4

# ── GPT 修正参数 ──────────────────────────────────────────────────────
# 从环境变量读取，不在此处硬编码
OPENAI_API_KEY:  str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL:    str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# GPT 修正生成参数
CORRECTION_TEMPERATURE: float = 0.0
CORRECTION_MAX_TOKENS:  int   = 600

# ── 推理设备 ────────────────────────────────────────────────────────
if _TORCH_AVAILABLE:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

# ── Flask 服务参数 ────────────────────────────────────────────────────
FLASK_HOST:  str  = "0.0.0.0"
FLASK_PORT:  int  = 5002          # 与后端其他服务错开端口
FLASK_DEBUG: bool = False
