"""
Mintaka 数据集的 RGAT 配置文件
"""

import os


class Config:
    """配置类"""

    # ==================== 损失函数配置 ====================
    USE_SUPERVISED_CONTRASTIVE = False  # 禁用监督对比学习
    USE_TRIPLET_CONTRASTIVE = True      # 启用三元组对比学习
    ANTI_COLLAPSE_WEIGHT = 5.0

    # 三元组对比学习参数
    TRIPLET_MARGIN = 2.0      # margin值，控制分离程度
    TEMPERATURE = 0.1         # 温度参数

    # ==================== 损失函数配置 ====================
    LOSS_TYPE = 'focal'       # 'focal' | 'weighted_ce' | 'ce'
    FOCAL_ALPHA = 0.75        # Focal Loss的alpha参数（幻觉类权重，推荐0.7-0.8）
    FOCAL_GAMMA = 2.0         # Focal Loss的gamma参数（聚焦参数，推荐2.0）

    # ==================== 分类器配置 ====================
    FFN_HIDDEN_DIM = 256      # FFN隐藏层维度（推荐128-512）
    FFN_DROPOUT = 0.5         # FFN的Dropout率
    FREEZE_ENCODER = False    # 是否冻结预训练RGAT编码器

    # ==================== 路径配置 ====================
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 服务器数据根目录
    DATA_ROOT = "/root/autodl-fs/gca/mintaka"

    # 预处理后的数据文件（使用最终处理好的版本）
    DATA_PATH = os.path.join(DATA_ROOT, "preprocess_data", "data",
                             "mintaka_dev_relations_replaced.json")

    # 混合嵌入目录（与RGCN共用）
    HYBRID_EMBEDDINGS_DIR = os.path.join(DATA_ROOT, "OpenKE", "hybrid_embeddings")

    # RGAT使用相同的嵌入文件格式
    ENTITY_EMBEDDING_RGAT_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "entity_embeddings_rgcn.pkl")
    RELATION_MAPPING_RGAT_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "relation_mappings_rgcn.pkl")

    # 原始混合嵌入文件
    ENTITY_HYBRID_EMBEDDING_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "entity_hybrid_embeddings.pkl")
    RELATION_HYBRID_EMBEDDING_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "relation_hybrid_embeddings.pkl")
    ENTITY2IDX_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "entity2idx.pkl")
    RELATION2IDX_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, "relation2idx.pkl")

    # 输出路径（RGAT专用，与RGCN区分）
    OUTPUT_DIR = os.path.join(CURRENT_DIR, "rgat_output")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_rgat_model.pth")
    RESULT_PATH = os.path.join(OUTPUT_DIR, "rgat_results.json")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # ==================== 模型配置 ====================
    # R-GAT架构（优化显存版本）
    HIDDEN_CHANNELS = 128     # 隐藏层维度（从256降低到128）
    OUT_CHANNELS = 64         # 输出维度（从128降低到64）
    NUM_LAYERS = 2            # R-GAT层数（从3降低到2）
    NUM_HEADS = 4             # 注意力头数（从8降低到4）
    DROPOUT = 0.5             # Dropout率

    # 孪生网络
    USE_PROJECTION_HEAD = True  # 是否使用投影头
    PROJECTION_DIM = 64         # 投影维度（从128降低到64）

    # ==================== 训练配置 ====================
    # 基础训练参数（优化显存版本）
    BATCH_SIZE = 2                    # Mintaka数据集batch size（从6降低到2）
    GRADIENT_ACCUMULATION_STEPS = 6   # 梯度累积步数（从3增加到6，保持有效batch=12）
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # 损失函数参数
    MARGIN = 0.5               # 对比学习的margin
    ALPHA = 0.7                # 对比损失和InfoNCE的权重平衡

    # 学习率调度器
    SCHEDULER_TYPE = 'reduce_on_plateau'  # 'cosine', 'step', 'reduce_on_plateau'
    REDUCE_LR_FACTOR = 0.5     # 学习率衰减因子
    REDUCE_LR_PATIENCE = 5     # 验证损失不下降的耐心值
    REDUCE_LR_MIN_LR = 1e-6    # 最小学习率
    T_0 = 10                   # CosineAnnealingWarmRestarts的初始周期
    T_MULT = 2                 # 周期倍增因子
    ETA_MIN = 1e-6             # 最小学习率

    # 训练策略
    EARLY_STOPPING_PATIENCE = 25  # 早停的patience
    GRADIENT_CLIP_NORM = 2.0      # 梯度裁剪
    SAVE_INTERVAL = 10            # 保存检查点的间隔

    # 数据相关
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    MAX_SAMPLES = None         # 最大样本数，None表示使用全部数据
    NUM_WORKERS = 0            # DataLoader的工作进程数
    SEED = 42                  # 随机种子

    # ==================== 嵌入配置 ====================
    FREEZE_EMBEDDINGS = True   # 是否冻结嵌入层（不更新混合嵌入）

    # ==================== 验证配置 ====================
    # 相似度阈值
    HIGH_SIMILARITY_THRESHOLD = 0.85    # 高相似度阈值
    MEDIUM_SIMILARITY_THRESHOLD = 0.70  # 中等相似度阈值
    LOW_SIMILARITY_THRESHOLD = 0.55     # 低相似度阈值

    # ==================== 日志配置 ====================
    LOG_INTERVAL = 10          # 日志打印间隔（每多少个batch）
    SAVE_PLOTS = True          # 是否保存训练曲线图

    # ==================== 采样器配置 ====================
    USE_BALANCED_SAMPLING = True

    @classmethod
    def get_config_dict(cls):
        """返回配置字典"""
        return {
            # 数据
            'data_path': cls.DATA_PATH,
            'entity_embedding_path': cls.ENTITY_EMBEDDING_RGAT_PATH,
            'relation_embedding_path': cls.RELATION_MAPPING_RGAT_PATH,
            'entity_mapping_path': cls.ENTITY2IDX_PATH,
            'relation_mapping_path': cls.RELATION2IDX_PATH,

            # 模型
            'hidden_channels': cls.HIDDEN_CHANNELS,
            'out_channels': cls.OUT_CHANNELS,
            'num_layers': cls.NUM_LAYERS,
            'num_heads': cls.NUM_HEADS,
            'dropout': cls.DROPOUT,
            'use_projection_head': cls.USE_PROJECTION_HEAD,
            'projection_dim': cls.PROJECTION_DIM,
            'freeze_embeddings': cls.FREEZE_EMBEDDINGS,

            # 分类器配置
            'ffn_hidden_dim': cls.FFN_HIDDEN_DIM,
            'ffn_dropout': cls.FFN_DROPOUT,
            'freeze_encoder': cls.FREEZE_ENCODER,

            # 训练
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'margin': cls.MARGIN,
            'alpha': cls.ALPHA,
            'scheduler_type': cls.SCHEDULER_TYPE,
            't_0': cls.T_0,
            't_mult': cls.T_MULT,
            'eta_min': cls.ETA_MIN,
            'early_stopping_patience': cls.EARLY_STOPPING_PATIENCE,
            'gradient_clip_norm': cls.GRADIENT_CLIP_NORM,
            'save_interval': cls.SAVE_INTERVAL,

            # 数据
            'max_samples': cls.MAX_SAMPLES,
            'num_workers': cls.NUM_WORKERS,
            'seed': cls.SEED,

            # 输出
            'checkpoint_dir': cls.CHECKPOINT_DIR,
            'output_dir': cls.OUTPUT_DIR,
            'log_dir': cls.LOG_DIR,

            # 采样器
            'use_balanced_sampling': cls.USE_BALANCED_SAMPLING,
            'reduce_lr_factor': cls.REDUCE_LR_FACTOR,
            'reduce_lr_patience': cls.REDUCE_LR_PATIENCE,
            'reduce_lr_min_lr': cls.REDUCE_LR_MIN_LR,

            # 损失函数配置
            'loss_type': cls.LOSS_TYPE,
            'focal_alpha': cls.FOCAL_ALPHA,
            'focal_gamma': cls.FOCAL_GAMMA,
            'use_supervised_contrastive': cls.USE_SUPERVISED_CONTRASTIVE,
            'temperature': cls.TEMPERATURE,
            'triplet_margin': cls.TRIPLET_MARGIN,
            'use_triplet_contrastive': cls.USE_TRIPLET_CONTRASTIVE,
            'gradient_accumulation_steps': cls.GRADIENT_ACCUMULATION_STEPS,
            'anti_collapse_weight': cls.ANTI_COLLAPSE_WEIGHT,
        }

    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("RGAT配置信息 - Mintaka")
        print("=" * 60)
        print(f"\n数据:")
        print(f"  数据文件: {cls.DATA_PATH}")
        print(f"  实体嵌入(RGAT): {cls.ENTITY_EMBEDDING_RGAT_PATH}")
        print(f"  关系映射(RGAT): {cls.RELATION_MAPPING_RGAT_PATH}")
        print(f"\n模型:")
        print(f"  隐藏层维度: {cls.HIDDEN_CHANNELS}")
        print(f"  输出维度: {cls.OUT_CHANNELS}")
        print(f"  层数: {cls.NUM_LAYERS}")
        print(f"  注意力头数: {cls.NUM_HEADS}")
        print(f"  Dropout: {cls.DROPOUT}")
        print(f"\n训练:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  冻结嵌入: {cls.FREEZE_EMBEDDINGS}")
        print(f"\n输出:")
        print(f"  检查点目录: {cls.CHECKPOINT_DIR}")
        print(f"  结果文件: {cls.RESULT_PATH}")
        print("=" * 60)


def create_directories():
    """创建必要的目录"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    print(f"已创建输出目录: {Config.OUTPUT_DIR}")


if __name__ == '__main__':
    Config.print_config()
    create_directories()
