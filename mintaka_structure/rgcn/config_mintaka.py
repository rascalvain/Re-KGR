"""
Mintaka 数据集的 RGCN 链接预测配置文件
"""

import os


class Config:
    """配置类"""

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 服务器数据根目录
    DATA_ROOT = "/root/autodl-fs/gca/mintaka"

    # 预处理后的数据文件（含 entity_triples / gpt_triples）
    DATA_PATH = os.path.join(DATA_ROOT, "preprocess_data", "data",
                             "mintaka_dev_stage1_canonicalized.json")

    # 混合嵌入目录（由 train_transe/generate_hybrid_embeddings.py 生成）
    HYBRID_EMBEDDINGS_DIR = os.path.join(DATA_ROOT, "train_transe", "hybrid_embeddings")

    # RGCN 专用嵌入文件（由 prepare_embeddings.py 生成）
    ENTITY_EMBEDDING_RGCN_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "entity_embeddings_rgcn.pkl")
    RELATION_MAPPING_RGCN_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "relation_mappings_rgcn.pkl")

    # 原始混合嵌入文件
    ENTITY_HYBRID_EMBEDDING_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "entity_hybrid_embeddings.pkl")
    RELATION_HYBRID_EMBEDDING_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "relation_hybrid_embeddings.pkl")
    ENTITY2IDX_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "entity2idx.pkl")
    RELATION2IDX_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "relation2idx.pkl")

    # 输出路径
    OUTPUT_DIR = os.path.join(CURRENT_DIR, "rgcn_output")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # ==================== 模型配置 ====================
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_BASES = 30  # RGCNConv basis decomposition，大幅降低显存占用

    # ==================== 训练配置 ====================
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MARGIN = 1.0

    # 学习率调度器
    T_0 = 10
    T_MULT = 2
    ETA_MIN = 1e-6

    # 训练策略
    EARLY_STOPPING_PATIENCE = 20
    SAVE_INTERVAL = 10
    MAX_TRIPLES_PER_SUBGRAPH = 32

    # 数据
    MAX_SAMPLES = None
    SEED = 42

    @classmethod
    def get_config_dict(cls):
        return {
            "data_path": cls.DATA_PATH,
            "entity_embedding_path": cls.ENTITY_EMBEDDING_RGCN_PATH,
            "relation_embedding_path": cls.RELATION_MAPPING_RGCN_PATH,
            "entity_mapping_path": cls.ENTITY2IDX_PATH,
            "relation_mapping_path": cls.RELATION2IDX_PATH,
            "num_layers": cls.NUM_LAYERS,
            "num_bases": cls.NUM_BASES,
            "dropout": cls.DROPOUT,
            "batch_size": cls.BATCH_SIZE,
            "num_epochs": cls.NUM_EPOCHS,
            "learning_rate": cls.LEARNING_RATE,
            "weight_decay": cls.WEIGHT_DECAY,
            "margin": cls.MARGIN,
            "t_0": cls.T_0,
            "t_mult": cls.T_MULT,
            "eta_min": cls.ETA_MIN,
            "early_stopping_patience": cls.EARLY_STOPPING_PATIENCE,
            "save_interval": cls.SAVE_INTERVAL,
            "max_triples_per_subgraph": cls.MAX_TRIPLES_PER_SUBGRAPH,
            "max_samples": cls.MAX_SAMPLES,
            "seed": cls.SEED,
            "checkpoint_dir": cls.CHECKPOINT_DIR,
            "output_dir": cls.OUTPUT_DIR,
            "log_dir": cls.LOG_DIR,
        }

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("RGCN 链接预测配置 - Mintaka")
        print("=" * 60)
        print(f"  数据文件: {cls.DATA_PATH}")
        print(f"  实体嵌入: {cls.ENTITY_EMBEDDING_RGCN_PATH}")
        print(f"  关系映射: {cls.RELATION_MAPPING_RGCN_PATH}")
        print(f"  RGCN 层数: {cls.NUM_LAYERS}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  Margin: {cls.MARGIN}")
        print(f"  输出目录: {cls.OUTPUT_DIR}")
        print("=" * 60)


def create_directories():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)


if __name__ == "__main__":
    Config.print_config()
    create_directories()
