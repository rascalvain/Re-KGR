"""
HotpotQA 数据集的 RGAT 配置文件
"""

import os

class Config:
    """配置类"""
    # ==================== 损失函数配置 ====================
    USE_SUPERVISED_CONTRASTIVE = False  # 🔥 禁用监督对比学习
    USE_TRIPLET_CONTRASTIVE = True  # 🔥 启用三元组对比学习
    ANTI_COLLAPSE_WEIGHT = 5.0

    # 三元组对比学习参数
    TRIPLET_MARGIN = 2.0  # margin值，控制分离程度
    TEMPERATURE = 0.1  # 温度参数
    # ==================== 监督对比学习配置 ====================
    # USE_SUPERVISED_CONTRASTIVE = True  # 是否使用监督对比学习（推荐）
    # TEMPERATURE = 0.07  # 温度参数（监督对比学习推荐0.05-0.1）
    # ==================== 损失函数配置 ====================
    LOSS_TYPE = 'focal'  # 'focal' | 'weighted_ce' | 'ce'
    FOCAL_ALPHA = 0.75  # Focal Loss的alpha参数（幻觉类权重，推荐0.7-0.8）
    FOCAL_GAMMA = 2.0  # Focal Loss的gamma参数（聚焦参数，推荐2.0）
    # ==================== 分类器配置 ====================
    # FFN分类头参数（用于幻觉检测分类器）
    FFN_HIDDEN_DIM = 256  # FFN隐藏层维度（推荐128-512）
    FFN_DROPOUT = 0.5  # FFN的Dropout率（可以与RGAT的dropout不同）
    FREEZE_ENCODER = False  # 是否冻结预训练RGAT编码器
    # ==================== 路径配置 ====================
    # 当前工作目录
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
    
    # 数据路径
    # DATA_DIR = os.path.dirname(CURRENT_DIR)  # hotpotqa 目录
    DATA_DIR = "/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data"
    HOTPOTQA_DATA_PATH = os.path.join(DATA_DIR, 'hotpot_dev_merged_triples_clustered.json')
    
    # 嵌入路径（与RGCN共用相同的嵌入）
    HYBRID_EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'final_hybrid_embeddings')
    
    # RGAT使用相同的嵌入文件格式
    ENTITY_EMBEDDING_RGAT_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, 'node_embeddings_linkpred_rgcn.pkl')
    RELATION_MAPPING_RGAT_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, 'relation_mappings_rgcn.pkl')
    
    # 原始混合嵌入文件（供 prepare_embeddings.py 使用）
    ENTITY_HYBRID_EMBEDDING_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, 'entity_hybrid_embeddings.pkl')
    RELATION_HYBRID_EMBEDDING_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, 'relation_hybrid_embeddings.pkl')
    ENTITY2IDX_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, 'entity2idx.pkl')
    RELATION2IDX_PATH = os.path.join(HYBRID_EMBEDDINGS_DIR, 'relation2idx.pkl')
    
    # sentence-bert 模型路径
    SBERT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'sentence-bert')
    
    # 输出路径（RGAT专用，与RGCN区分）
    OUTPUT_DIR = os.path.join(DATA_DIR, 'rgat_output')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_rgat_model.pth')
    RESULT_PATH = os.path.join(OUTPUT_DIR, 'rgat_results.json')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # ==================== 模型配置 ====================
    # R-GAT架构
    HIDDEN_CHANNELS = 256      # 隐藏层维度
    OUT_CHANNELS = 128          # 输出维度
    NUM_LAYERS = 3             # R-GAT层数
    NUM_HEADS = 8              # 注意力头数（RGAT特有）⭐
    DROPOUT = 0.5              # Dropout率
    
    # 孪生网络
    USE_PROJECTION_HEAD = True  # 是否使用投影头
    PROJECTION_DIM = 128         # 投影维度
    
    # ==================== 训练配置 ====================
    # 基础训练参数
    BATCH_SIZE = 6             # HotpotQA数据较大，使用较小的batch size
    GRADIENT_ACCUMULATION_STEPS = 3
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # 损失函数参数
    MARGIN = 0.5               # 对比学习的margin
    # TEMPERATURE = 0.1          # InfoNCE温度参数
    ALPHA = 0.7                # 对比损失和InfoNCE的权重平衡
    
    # 学习率调度器
    SCHEDULER_TYPE = 'reduce_on_plateau'  # 'cosine', 'step', 'reduce_on_plateau'
    REDUCE_LR_FACTOR = 0.5  # 新增：学习率衰减因子
    REDUCE_LR_PATIENCE = 5  # 新增：验证损失不下降的耐心值
    REDUCE_LR_MIN_LR = 1e-6  # 新增：最小学习率
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

    #======================采样器配置====================
    use_balanced_sampling = True
    
    @classmethod
    def get_config_dict(cls):
        """返回配置字典"""
        return {
            # 数据
            'data_path': cls.HOTPOTQA_DATA_PATH,
            'entity_embedding_path': cls.ENTITY_EMBEDDING_RGAT_PATH,  # 使用RGAT路径
            'relation_embedding_path': cls.RELATION_MAPPING_RGAT_PATH,  # 使用RGAT路径
            'entity_mapping_path': cls.ENTITY2IDX_PATH,
            'relation_mapping_path': cls.RELATION2IDX_PATH,
            
            # 模型
            'hidden_channels': cls.HIDDEN_CHANNELS,
            'out_channels': cls.OUT_CHANNELS,
            'num_layers': cls.NUM_LAYERS,
            'num_heads': cls.NUM_HEADS,  # ⭐ RGAT特有参数
            'dropout': cls.DROPOUT,
            'use_projection_head': cls.USE_PROJECTION_HEAD,
            'projection_dim': cls.PROJECTION_DIM,
            'freeze_embeddings': cls.FREEZE_EMBEDDINGS,

            # 🔥 分类器配置（新增）
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

            #采样器
            'use_balanced_sampling' : cls.use_balanced_sampling,
            'reduce_lr_factor' : cls.REDUCE_LR_FACTOR,
            'reduce_lr_patience' : cls.REDUCE_LR_PATIENCE,
            'reduce_lr_min_lr' : cls.REDUCE_LR_MIN_LR,

            # 损失函数配置
            'loss_type': Config.LOSS_TYPE,
            'focal_alpha': Config.FOCAL_ALPHA,
            'focal_gamma': Config.FOCAL_GAMMA,

            'use_supervised_contrastive': cls.USE_SUPERVISED_CONTRASTIVE,
            'temperature': cls.TEMPERATURE,  # 确保这个参数存在
            'triplet_margin':cls.TRIPLET_MARGIN,

            'use_triplet_contrastive':cls.USE_TRIPLET_CONTRASTIVE,
            'gradient_accumulation_steps': cls.GRADIENT_ACCUMULATION_STEPS,
            'anti_collapse_weight': cls.ANTI_COLLAPSE_WEIGHT ,
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("="*60)
        print("RGAT配置信息 - HotpotQA")
        print("="*60)
        print(f"\n数据:")
        print(f"  数据文件: {cls.HOTPOTQA_DATA_PATH}")
        print(f"  实体嵌入(RGAT): {cls.ENTITY_EMBEDDING_RGAT_PATH}")
        print(f"  关系映射(RGAT): {cls.RELATION_MAPPING_RGAT_PATH}")
        print(f"\n模型:")
        print(f"  隐藏层维度: {cls.HIDDEN_CHANNELS}")
        print(f"  输出维度: {cls.OUT_CHANNELS}")
        print(f"  层数: {cls.NUM_LAYERS}")
        print(f"  注意力头数: {cls.NUM_HEADS} ⭐")
        print(f"  Dropout: {cls.DROPOUT}")
        print(f"\n训练:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  学习率: {cls.LEARNING_RATE}")
        print(f"  冻结嵌入: {cls.FREEZE_EMBEDDINGS}")
        print(f"\n输出:")
        print(f"  检查点目录: {cls.CHECKPOINT_DIR}")
        print(f"  结果文件: {cls.RESULT_PATH}")
        print("="*60)


# 创建必要的目录
def create_directories():
    """创建必要的目录"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    print(f"已创建输出目录: {Config.OUTPUT_DIR}")


if __name__ == '__main__':
    # 测试配置
    Config.print_config()
    create_directories()
