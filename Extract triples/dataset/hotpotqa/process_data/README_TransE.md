# TransE 模型训练说明

本文档说明如何使用 OpenKE 训练 TransE 模型并获取实体和关系的图谱嵌入表示。

## 📋 前置要求

### 1. 安装 OpenKE

```bash
# 克隆 OpenKE 仓库
git clone https://github.com/thunlp/OpenKE
cd OpenKE

# 编译 C++ 模块
bash make.sh

# 安装
python setup.py install
```

### 2. 所需的 Python 包

```bash
pip install numpy torch scikit-learn
```

**注意**: OpenKE 需要 PyTorch。如果你有 GPU，请安装 CUDA 版本的 PyTorch 以加速训练。

## 📁 数据文件

确保你有以下三个文件（已由前面的步骤生成）：

1. **entity2id.txt** - 实体到ID的映射（5,478个实体）
   - 格式: `实体名称\tID`
   
2. **relation2id.txt** - 关系到ID的映射（1,856个关系）
   - 格式: `关系名称\tID`
   
3. **triples.txt** - 三元组数据（3,975个三元组）
   - 格式: `头实体\t尾实体\t关系`

## 🚀 使用步骤

### 步骤 1: 训练 TransE 模型

运行训练脚本：

```bash
python train_transe.py
```

**训练过程说明**:

1. **读取数据**: 读取 entity2id.txt, relation2id.txt, triples.txt
2. **格式转换**: 转换为 OpenKE 所需的格式，保存到 `openke_data/` 目录
3. **模型训练**: 训练 TransE 模型（默认500个epoch）
4. **保存结果**: 
   - 嵌入向量保存到 `embeddings/` 目录
   - 模型检查点保存到 `checkpoint/` 目录

**重要参数**（可在 train_transe.py 中调整）:

```python
# TransE 模型参数
dim = 100              # 嵌入维度（可选：50, 100, 200, 300）
p_norm = 1             # 范数类型（1=L1, 2=L2）
margin = 5.0           # margin损失的margin值

# 训练参数
train_times = 500      # 训练轮数（epoch）
alpha = 0.5            # 学习率
nbatches = 100         # batch数量
neg_ent = 25           # 负采样数量
```

**训练时间估计**:
- CPU: 约 10-30 分钟（取决于数据量和epoch数）
- GPU: 约 2-5 分钟

### 步骤 2: 使用嵌入向量

训练完成后，运行示例代码：

```bash
python use_embeddings.py
```

这个脚本展示了如何：
- 加载训练好的嵌入向量
- 获取特定实体/关系的嵌入
- 查找相似实体
- 进行三元组预测（知识图谱补全）

## 📊 输出文件说明

训练完成后，会生成以下文件和目录：

```
./
├── openke_data/              # OpenKE 格式的数据文件
│   ├── entity2id.txt        # 实体映射（OpenKE格式）
│   ├── relation2id.txt      # 关系映射（OpenKE格式）
│   └── train2id.txt         # 训练三元组（h t r 格式）
│
├── embeddings/               # 嵌入向量
│   ├── entity_embeddings.npy    # 实体嵌入 (numpy格式)
│   └── relation_embeddings.npy  # 关系嵌入 (numpy格式)
│
└── checkpoint/               # 模型检查点
    └── transe.ckpt          # 模型权重
```

## 💡 使用嵌入的示例

### 1. 加载嵌入

```python
import numpy as np

# 加载嵌入
entity_embeddings = np.load('./embeddings/entity_embeddings.npy')
relation_embeddings = np.load('./embeddings/relation_embeddings.npy')

# 查看形状
print(f"实体嵌入: {entity_embeddings.shape}")  # (5478, 100)
print(f"关系嵌入: {relation_embeddings.shape}") # (1856, 100)
```

### 2. 获取特定实体的嵌入

```python
# 加载 entity2id 映射
entity2id = {}
with open('entity2id.txt', 'r', encoding='utf-8') as f:
    for line in f:
        entity, eid = line.strip().split('\t')
        entity2id[entity] = int(eid)

# 获取某个实体的嵌入
entity_name = "Paris"
if entity_name in entity2id:
    entity_id = entity2id[entity_name]
    embedding = entity_embeddings[entity_id]
    print(f"{entity_name} 的嵌入: {embedding}")
```

### 3. 计算实体相似度

```python
from sklearn.metrics.pairwise import cosine_similarity

# 计算两个实体的余弦相似度
entity1_emb = entity_embeddings[entity2id["Paris"]].reshape(1, -1)
entity2_emb = entity_embeddings[entity2id["France"]].reshape(1, -1)

similarity = cosine_similarity(entity1_emb, entity2_emb)[0][0]
print(f"相似度: {similarity}")
```

### 4. 知识图谱补全

```python
# TransE: h + r ≈ t
# 给定头实体和关系，预测尾实体

head_id = entity2id["Paris"]
relation_id = relation2id["capital_of"]

# 预测尾实体
predicted_tail = entity_embeddings[head_id] + relation_embeddings[relation_id]

# 找到最接近的实体
distances = np.linalg.norm(entity_embeddings - predicted_tail, axis=1)
nearest_entity_id = np.argmin(distances)

print(f"预测的尾实体ID: {nearest_entity_id}")
```

## ⚙️ 常见问题

### Q1: 训练很慢怎么办？

**方案**:
1. 减少 `train_times`（epoch数）到 100-200
2. 增加 `nbatches`（batch数）
3. 使用 GPU 训练（安装 CUDA 版本的 PyTorch）
4. 减少 `neg_ent`（负采样数）

### Q2: 内存不足怎么办？

**方案**:
1. 减少 `dim`（嵌入维度）到 50
2. 增加 `nbatches`（减小batch size）
3. 使用更少的训练数据

### Q3: 如何评估模型质量？

OpenKE 支持链接预测任务的评估指标：
- Mean Rank (MR)
- Mean Reciprocal Rank (MRR)
- Hits@1, Hits@3, Hits@10

需要准备测试集（test2id.txt）和验证集（valid2id.txt）。

### Q4: ID必须连续吗？

**是的！** OpenKE 要求实体ID和关系ID必须从0开始连续编号。本代码已经处理了这个问题。

## 📚 进一步使用

训练好的嵌入可以用于：

1. **实体分类**: 将实体嵌入作为特征进行分类
2. **实体聚类**: 基于嵌入向量进行聚类分析
3. **关系抽取**: 使用嵌入辅助关系抽取任务
4. **问答系统**: 利用嵌入进行语义匹配
5. **推荐系统**: 基于实体嵌入的相似度推荐

## 🔗 参考资料

- [OpenKE GitHub](https://github.com/thunlp/OpenKE)
- [TransE 论文](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [OpenKE 文档](https://openke.readthedocs.io/)

## 📝 注意事项

1. **数据格式**: OpenKE 对输入格式要求严格，确保：
   - 文件第一行是数据总数
   - train2id.txt 的格式是 `h t r`（不是 `h r t`）
   - 使用 tab 分隔符

2. **ID连续性**: 实体和关系的ID必须从0开始连续编号

3. **GPU使用**: 如果有GPU，确保安装了正确版本的PyTorch和CUDA

4. **参数调优**: 不同的数据集可能需要不同的超参数，建议多尝试几组参数

