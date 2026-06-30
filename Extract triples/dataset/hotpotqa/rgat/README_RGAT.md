# RGAT（关系图注意力网络）实现指南

## 📋 概述

本目录包含从RGCN迁移到RGAT的完整实现，核心改进是引入**多头注意力机制**来捕获关系依赖。

## 🔄 核心改动总结

| 组件 | RGCN | RGAT |
|------|------|------|
| **核心层** | `RGCNConv` | `RGATConv` |
| **注意力机制** | 无 | 多头注意力 |
| **新增参数** | - | `num_heads`, `concat` |
| **输出维度** | `out_channels` | `out_channels * heads`（concat=True时） |
| **优势** | 简单高效 | 捕获关系依赖的注意力权重 |

## 📁 文件结构

```
rgat/
├── siamese_rgat_improved.py          # RGAT模型定义
├── train_rgat_hotpotqa.py           # 训练脚本
├── classifier_with_pretrained_rgat.py  # 分类器（使用预训练RGAT）
└── README_RGAT.md                    # 本文档
```

## 🚀 使用流程

### 1️⃣ 准备数据和嵌入

**RGAT与RGCN完全兼容**，直接使用RGCN的嵌入文件即可。

```bash
# 如果还没有生成嵌入，先运行：
cd ../rgcn
python prepare_embeddings.py
```

这将生成：
- `hybrid_embeddings/entity_embeddings_rgcn.pkl` - 实体嵌入
- `hybrid_embeddings/relation_mappings_rgcn.pkl` - 关系映射

### 2️⃣ 配置参数

RGAT使用与RGCN相同的配置文件 `../rgcn/config_hotpotqa.py`，已包含RGAT特有参数：

```python
# ==================== 模型配置 ====================
# R-GAT架构（新增）
HIDDEN_CHANNELS = 128      # 隐藏层维度
OUT_CHANNELS = 64          # 输出维度
NUM_LAYERS = 3             # R-GAT层数
NUM_HEADS = 4              # 注意力头数（RGAT特有）⭐
DROPOUT = 0.3              # Dropout率
```

### 3️⃣ 训练RGAT模型

```bash
cd rgat
python train_rgat_hotpotqa.py
```

**训练过程：**
- 自动加载HotpotQA数据集
- 使用孪生网络架构进行对比学习
- 保存最佳模型到 `rgat_output/checkpoints/best_rgat_model.pth`
- 生成训练曲线图

**输出文件：**
```
rgat_output/
├── checkpoints/
│   └── best_rgat_model.pth        # 最佳模型
├── rgat_training_curves.png       # 训练曲线
└── rgat_training_history.json     # 训练历史
```

### 4️⃣ 使用预训练RGAT进行分类

```python
from classifier_with_pretrained_rgat import HallucinationClassifierWithPretrainedRGAT

# 加载分类器
classifier = HallucinationClassifierWithPretrainedRGAT(
    pretrained_model_path='rgat_output/checkpoints/best_rgat_model.pth',
    freeze_encoder=False,  # 允许微调
    ffn_hidden_dim=128
)

# 推理
predictions, probs = classifier.predict(response_graph, reference_graph)
```

## 🔍 关键代码对比

### RGCN vs RGAT - 模型定义

**RGCN (使用RGCNConv):**
```python
from torch_geometric.nn import RGCNConv

self.convs.append(
    RGCNConv(
        in_channels,
        out_channels,
        num_relations=self.num_relations
    )
)
```

**RGAT (使用RGATConv):**
```python
from torch_geometric.nn import RGATConv

self.convs.append(
    RGATConv(
        in_channels,
        out_channels // num_heads,  # 每个head的输出维度
        num_relations=self.num_relations,
        heads=num_heads,            # 🔹 多头注意力
        concat=True,                # 🔹 拼接多头输出
        dropout=dropout             # 🔹 注意力dropout
    )
)
```

### 模型初始化对比

**RGCN:**
```python
model = SiameseRGCNWithEmbedding(
    entity_embedding_path=...,
    relation_embedding_path=...,
    hidden_channels=128,
    out_channels=64,
    num_layers=3,
    dropout=0.3
)
```

**RGAT:**
```python
model = SiameseRGATWithEmbedding(
    entity_embedding_path=...,
    relation_embedding_path=...,
    hidden_channels=128,
    out_channels=64,
    num_layers=3,
    dropout=0.3,
    num_heads=4  # 🔹 新增参数
)
```

## ⚙️ 参数说明

### RGAT特有参数

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `num_heads` | 注意力头数 | 4, 8 |
| `concat` | 是否拼接多头输出 | 中间层: True<br>最后一层: False |
| `dropout` | 注意力dropout率 | 0.3 |

### 注意力头数选择建议

- **小规模图** (节点 < 1000): `num_heads = 2, 4`
- **中等规模图** (节点 1000-10000): `num_heads = 4, 8`
- **大规模图** (节点 > 10000): `num_heads = 8, 16`

## 📊 性能对比

### 优势
✅ **更强的表达能力**: 多头注意力可以捕获不同类型的关系依赖  
✅ **更好的泛化性**: 注意力机制有助于关注重要关系  
✅ **可解释性**: 可以可视化注意力权重  

### 权衡
⚠️ **计算开销**: 比RGCN略高（取决于注意力头数）  
⚠️ **参数量**: 随注意力头数增加  

## 🔧 常见问题

### Q1: RGAT和RGCN可以共用嵌入文件吗？
**是的**！RGAT和RGCN使用完全相同的嵌入格式，可以直接复用。

### Q2: 如何调整注意力头数？
修改 `../rgcn/config_hotpotqa.py` 中的 `NUM_HEADS` 参数即可。

### Q3: 训练RGAT比RGCN慢多少？
取决于 `num_heads` 的值。一般情况下：
- `num_heads=4`: 约慢 20-30%
- `num_heads=8`: 约慢 40-60%

### Q4: 如何从RGCN模型迁移到RGAT？
两个模型的检查点不兼容，需要**重新训练**。但可以使用相同的数据和嵌入。

## 📝 示例：完整训练流程

```bash
# 1. 确保嵌入文件存在
cd ../rgcn
python prepare_embeddings.py

# 2. 返回RGAT目录
cd ../rgat

# 3. 训练RGAT模型
python train_rgat_hotpotqa.py

# 4. 查看结果
ls rgat_output/checkpoints/
# 输出: best_rgat_model.pth
```

## 🎯 下一步

1. **调参优化**: 尝试不同的 `num_heads` 值（2, 4, 8）
2. **性能对比**: 与RGCN模型对比验证集损失
3. **可视化**: 提取注意力权重进行可视化分析
4. **下游任务**: 将预训练RGAT用于幻觉检测分类

## 📚 参考资料

- **PyTorch Geometric RGATConv**: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGATConv.html
- **原始论文**: "Graph Attention Networks" (Veličković et al., ICLR 2018)

---

**🎉 完成！现在你已经拥有一个完整的RGAT实现，可以直接开始训练和使用了！**

