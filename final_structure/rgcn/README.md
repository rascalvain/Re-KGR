# HotpotQA RGCN 模型说明

基于孪生 R-GCN 的图嵌入模型，用于处理 HotpotQA 数据集。

## 📋 概述

本模型使用关系图卷积网络（R-GCN）来学习知识图谱的嵌入表示，并通过孪生网络架构对比 `context_triples`（KB三元组）和 `gpt_sentence_triples`（响应三元组）之间的相似性。

### 核心特点

- ✅ **使用混合嵌入**: TransE + SentenceTransformer的结合
- ✅ **孪生网络**: 对比学习架构
- ✅ **R-GCN编码**: 多关系图卷积神经网络
- ✅ **批处理训练**: 支持GPU加速
- ✅ **早停机制**: 自动停止训练防止过拟合

## 📂 文件结构

```
rgcn/
├── config_hotpotqa.py           # 配置文件
├── data_loader_hotpotqa.py      # 数据加载器
├── siamese_rgcn_improved.py     # RGCN模型
├── prepare_embeddings.py        # 嵌入准备脚本
├── train_rgcn_hotpotqa.py       # 训练脚本
├── test_data_loader.py          # 测试脚本
├── run_all.bat                  # 一键运行脚本
└── README.md                    # 本文件
```

## 🚀 快速开始

### 前置要求

1. **已生成混合嵌入**: 运行过 `generate_hybrid_embeddings.py`
2. **Python 包**: 
   ```bash
   pip install torch torch-geometric numpy tqdm matplotlib scikit-learn
   ```

### 步骤 1: 准备嵌入文件

将混合嵌入转换为RGCN所需格式：

```bash
cd rgcn
python prepare_embeddings.py
```

这会生成：
- `entity_embeddings_rgcn.pkl` - 实体嵌入矩阵
- `relation_mappings_rgcn.pkl` - 关系映射

### 步骤 2: 测试数据加载器（可选）

```bash
python test_data_loader.py
```

验证数据是否正确加载。

### 步骤 3: 训练模型

```bash
python train_rgcn_hotpotqa.py
```

或使用一键脚本：
```bash
run_all.bat
```

## ⚙️ 配置说明

在 `config_hotpotqa.py` 中可以调整以下参数：

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `HIDDEN_CHANNELS` | 128 | 隐藏层维度 |
| `OUT_CHANNELS` | 64 | 输出维度 |
| `NUM_LAYERS` | 3 | R-GCN层数 |
| `DROPOUT` | 0.3 | Dropout率 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 8 | 批大小 |
| `NUM_EPOCHS` | 100 | 训练轮数 |
| `LEARNING_RATE` | 1e-3 | 学习率 |
| `MARGIN` | 0.5 | 对比损失的margin |
| `ALPHA` | 0.7 | 损失平衡参数 |
| `EARLY_STOPPING_PATIENCE` | 20 | 早停耐心值 |

## 📊 输出文件

训练完成后，在 `rgcn_output/` 目录下会生成：

```
rgcn_output/
├── checkpoints/
│   ├── best_model.pth              # 最佳模型
│   └── checkpoint_epoch_*.pth      # 定期检查点
├── training_curves.png             # 训练曲线图
└── training_history.json           # 训练历史
```

## 💡 使用示例

### 加载训练好的模型

```python
import torch
from siamese_rgcn_improved import SiameseRGCNWithEmbedding
from config_hotpotqa import Config

# 加载模型
model = SiameseRGCNWithEmbedding(
    entity_embedding_path=Config.get_config_dict()['entity_embedding_path'],
    relation_embedding_path=Config.get_config_dict()['relation_embedding_path'],
    hidden_channels=128,
    out_channels=64,
    num_layers=3
)

# 加载权重
checkpoint = torch.load('rgcn_output/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 使用模型
with torch.no_grad():
    context_emb, gpt_emb = model(context_graph, gpt_graph)
    similarity = torch.cosine_similarity(context_emb, gpt_emb)
```

### 计算图相似度

```python
# 准备图数据
from data_loader_hotpotqa import HotpotQAGraphDataset

dataset = HotpotQAGraphDataset(
    data_path='../hotpot_dev_with_triples_aligned.json',
    entity_mapping_path='../hybrid_embeddings/entity2idx.pkl',
    relation_mapping_path='../hybrid_embeddings/relation2idx.pkl'
)

# 获取一个样本
context_graph, gpt_graph, label, metadata = dataset[0]

# 计算相似度
context_emb, gpt_emb = model(context_graph.to(device), gpt_graph.to(device))
similarity = torch.cosine_similarity(context_emb.unsqueeze(0), gpt_emb.unsqueeze(0))

print(f"图相似度: {similarity.item():.4f}")
```

## 🎯 数据格式

### 输入数据格式

`hotpot_dev_with_triples_aligned.json`:
```json
[
  {
    "_id": "...",
    "question": "...",
    "answer": "...",
    "context_triples": [
      {"triple": "(head, relation, tail)"},
      ...
    ],
    "gpt_sentence_triples": [
      {"triple": "(head, relation, tail)"},
      ...
    ]
  },
  ...
]
```

### 图数据格式

PyG Data 对象：
- `node_ids`: 节点的全局ID（对应嵌入矩阵的索引）
- `edge_index`: 边的索引 [2, num_edges]
- `edge_type`: 边的关系类型
- `num_nodes`: 节点数量

## 🔧 调试和测试

### 测试数据加载

```bash
python test_data_loader.py
```

### 查看配置

```python
from config_hotpotqa import Config
Config.print_config()
```

### 检查嵌入

```python
import pickle

# 检查实体嵌入
emb_data = pickle.load(open('../hybrid_embeddings/entity_embeddings_rgcn.pkl', 'rb'))
print(f"实体数: {emb_data['num_entities']}")
print(f"嵌入维度: {emb_data['embedding_dim']}")
print(f"嵌入形状: {emb_data['embeddings'].shape}")
```

## ⚠️ 常见问题

### Q1: 嵌入文件不存在

**错误**: `错误: 实体嵌入文件不存在`

**解决**:
```bash
# 先生成混合嵌入
cd ..
python generate_hybrid_embeddings.py

# 再准备RGCN嵌入
cd rgcn
python prepare_embeddings.py
```

### Q2: CUDA out of memory

**解决**: 减小 `BATCH_SIZE` 或 `HIDDEN_CHANNELS`

```python
# 在 config_hotpotqa.py 中修改
BATCH_SIZE = 4  # 减小batch size
HIDDEN_CHANNELS = 64  # 减小隐藏层维度
```

### Q3: 训练损失不下降

**可能原因和解决**:
1. 学习率太大 → 减小 `LEARNING_RATE` 到 1e-4
2. 数据问题 → 检查数据加载器
3. 模型太复杂 → 减少 `NUM_LAYERS`

### Q4: 数据集太小

**说明**: HotpotQA只有110个样本，建议：
- 增加训练轮数
- 使用数据增强
- 调整早停耐心值

## 📈 性能优化

### 加速训练

1. **使用GPU**: 自动检测，无需配置
2. **增大批大小**: 如果内存允许
3. **减少层数**: 对小数据集足够
4. **冻结嵌入**: `FREEZE_EMBEDDINGS = True`（默认）

### 提高效果

1. **调整学习率**: 尝试 1e-4 到 1e-2
2. **增加训练轮数**: `NUM_EPOCHS = 200`
3. **调整margin**: `MARGIN = 0.3 到 1.0`
4. **使用不同的嵌入**: 尝试更大的TransE维度

## 📚 相关文档

- **混合嵌入生成**: `../README_Hybrid_Embeddings.md`
- **TransE训练**: `../README_TransE_Updated.md`
- **完整工作流程**: `../WORKFLOW_SUMMARY.md`
- **原始RGCN实现**: `../../Graph-based Contextual Consistency Comparison/new_rgcn/`

## 🔗 参考

- [R-GCN论文](https://arxiv.org/abs/1703.06103)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [HotpotQA数据集](https://hotpotqa.github.io/)

## 📅 更新日期

2024-12-04

## ✅ 完整工作流程

```bash
# 1. 提取实体和关系
cd ..
python extract_entities_relations.py

# 2. 提取三元组
python extract_triples.py

# 3. 训练TransE
python train_transe.py --prepare_data

# 4. 生成混合嵌入
python generate_hybrid_embeddings.py

# 5. 准备RGCN嵌入
cd rgcn
python prepare_embeddings.py

# 6. 训练RGCN
python train_rgcn_hotpotqa.py
```

现在可以开始训练RGCN模型了！🎉

