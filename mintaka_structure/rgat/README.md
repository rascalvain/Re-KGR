# Mintaka RGAT 幻觉检测模型

本目录包含了基于关系图注意力网络（R-GAT）的幻觉检测模型实现，针对 Mintaka 数据集进行了适配。

## 目录结构

```
mintaka_structure/rgat/
├── config_mintaka_rgat.py           # 配置文件
├── data_loader_mintaka.py           # 数据加载器
├── siamese_rgat_improved.py         # RGAT 模型定义
├── classifier_with_pretrained_rgat.py  # 分类器定义
├── train_rgat_mintaka.py            # RGAT 训练脚本
├── train_classifier.py              # 分类器训练脚本
├── prepare_embeddings.py            # 嵌入文件准备脚本
├── run_train.sh                     # 完整训练流程脚本
└── README.md                        # 本文件
```

## 模型架构

### 1. RGAT 编码器（对比学习阶段）
- **输入**: entity_triples（知识图谱三元组）和 gpt_triples（GPT生成的三元组）
- **架构**:
  - 多层 R-GAT（关系图注意力网络）
  - 多头注意力机制（默认8个头）
  - 注意力池化
- **训练目标**: 三元组对比学习，学习区分幻觉和事实的图表示

### 2. 幻觉检测分类器（分类阶段）
- **输入**: GPT图和参考图的编码表示
- **架构**:
  - 预训练的 RGAT 编码器（可冻结或微调）
  - FFN 分类头
- **输出**: 二分类（0=幻觉，1=事实）

## 训练流程

### 前置条件

确保已经完成以下步骤：
1. 完成 Mintaka 数据预处理（生成 `mintaka_dev_stage1_canonicalized.json`）
2. 训练 TransE 并生成混合嵌入（参考 `mintaka_structure/rgcn` 目录）

### 方式一：使用一键脚本（推荐）

```bash
cd mintaka_structure/rgat
bash run_train.sh
```

这个脚本会自动完成以下步骤：
1. 准备RGAT嵌入文件
2. 训练RGAT模型
3. 训练幻觉检测分类器

### 方式二：分步执行

#### 步骤 1: 准备嵌入文件

```bash
python prepare_embeddings.py
```

这将从混合嵌入中提取RGAT所需的格式：
- `entity_embeddings_rgcn.pkl` - 实体嵌入矩阵
- `relation_mappings_rgcn.pkl` - 关系映射

#### 步骤 2: 训练RGAT模型

```bash
python train_rgat_mintaka.py
```

训练参数（在 `config_mintaka_rgat.py` 中配置）：
- Batch size: 6
- 梯度累积步数: 3（有效batch size = 18）
- Epochs: 500（带早停）
- 学习率: 1e-4
- 注意力头数: 8
- 损失函数: 三元组对比学习

输出文件：
- `rgat_output/checkpoints/best_rgat_model.pth` - 最佳RGAT模型
- `rgat_output/rgat_training_curves.png` - 训练曲线
- `rgat_output/rgat_training_history.json` - 训练历史

#### 步骤 3: 训练幻觉检测分类器

```bash
python train_classifier.py --pretrained_model rgat_output/checkpoints/best_rgat_model.pth
```

训练参数：
- Freeze encoder: False（允许微调）
- FFN hidden dim: 256
- 损失函数: Focal Loss (alpha=0.75, gamma=2.0)

输出文件：
- `rgat_output/checkpoints/best_classifier.pth` - 最佳分类器
- `rgat_output/classifier_results.json` - 测试结果

## 配置说明

主要配置项（在 `config_mintaka_rgat.py` 中）：

### 数据路径
```python
DATA_ROOT = "/root/autodl-fs/gca/mintaka"  # 服务器数据根目录
DATA_PATH = ".../ mintaka_dev_stage1_canonicalized.json"  # 数据文件
```

### 模型参数
```python
HIDDEN_CHANNELS = 256      # 隐藏层维度
OUT_CHANNELS = 128         # 输出维度
NUM_LAYERS = 3             # RGAT层数
NUM_HEADS = 8              # 注意力头数
DROPOUT = 0.5              # Dropout率
```

### 训练参数
```python
BATCH_SIZE = 6                     # 批大小
GRADIENT_ACCUMULATION_STEPS = 3    # 梯度累积
NUM_EPOCHS = 500                   # 最大轮数
LEARNING_RATE = 1e-4               # 学习率
```

### 损失函数
```python
USE_TRIPLET_CONTRASTIVE = True     # 使用三元组对比学习
TRIPLET_MARGIN = 2.0               # Margin值
ANTI_COLLAPSE_WEIGHT = 5.0         # 反坍塌权重
```

## 数据格式

### 输入数据格式（Mintaka）
```json
{
  "id": "...",
  "question": "...",
  "entity_triples": [
    {"head": "entity1", "relation": "rel", "tail": "entity2"},
    ...
  ],
  "gpt_triples": [
    {"head": "entity1", "relation": "rel", "tail": "entity2"},
    ...
  ],
  "generation_label": "hallucination"  // 或 "correct"
}
```

### 关键差异（vs HotpotQA）
- Mintaka 使用字典格式的三元组：`{"head": ..., "relation": ..., "tail": ...}`
- HotpotQA 使用字符串格式：`"(head, relation, tail)"`
- 字段名：`entity_triples` vs `context_triples`

## 评估指标

分类器在测试集上的指标：
- **准确率 (Accuracy)**: 整体分类正确率
- **精确率 (Precision)**: 幻觉检测的精确度
- **召回率 (Recall)**: 幻觉检测的覆盖率
- **F1分数**: 精确率和召回率的调和平均
- **混淆矩阵**: 详细分类结果

## 与 RGCN 版本的区别

| 特性 | RGCN | RGAT |
|------|------|------|
| 消息传递 | 简单求和 | 多头注意力 |
| 计算复杂度 | 较低 | 较高 |
| 表达能力 | 一般 | 更强 |
| 参数量 | 较少 | 较多 |
| 适用场景 | 大规模图 | 中小规模图 |

## 常见问题

### 1. 显存不足
- 减小 `BATCH_SIZE`（如改为4或2）
- 减小 `HIDDEN_CHANNELS`（如改为128）
- 减少 `NUM_HEADS`（如改为4）

### 2. 训练不收敛
- 调整学习率（增大或减小）
- 调整 `TRIPLET_MARGIN`
- 增加 `ANTI_COLLAPSE_WEIGHT`

### 3. 过拟合
- 增大 `DROPOUT`
- 使用数据增强
- 减少模型层数

## 参考

- **论文**: Graph Attention Networks (Veličković et al., ICLR 2018)
- **框架**: PyTorch Geometric
- **基础版本**: final_structure/rgat (HotpotQA版本)

## 联系方式

如有问题，请联系项目维护者。
