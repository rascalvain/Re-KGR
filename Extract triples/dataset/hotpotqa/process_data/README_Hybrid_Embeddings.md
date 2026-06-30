# 混合嵌入生成说明（TransE + SentenceTransformer）

处理 OOV（Out-Of-Vocabulary）问题的完整解决方案。

## 🎯 方案说明

### 核心思路

对于知识图谱中的实体和关系：

1. **在KB中存在**：
   ```
   embedding = Concat(TransE_Vector, SentenceTransformer_Vector)
   ```

2. **不在KB中（OOV）**：
   ```
   embedding = Concat(Zero_Vector, SentenceTransformer_Vector)
   ```

### 优势

- ✅ **结构化信息**：TransE捕获图谱结构关系
- ✅ **语义信息**：SentenceTransformer捕获文本语义
- ✅ **OOV处理**：零向量 + 语义向量确保所有实体/关系都有嵌入
- ✅ **维度一致**：所有嵌入维度相同，便于下游使用

## 📋 前置要求

### 0. 本地 sentence-bert 模型

✅ **已配置**: 代码默认使用本地 sentence-bert 模型

模型位置：`g:\小论文\第三章\GCA-main\sentence-bert\`

测试模型：
```bash
python test_sentence_bert.py
```

详细说明请参考：`LOCAL_MODEL_USAGE.md`

### 1. 已完成 TransE 训练

确保已经运行过 TransE 训练，并在 `./output/` 目录有以下文件：
- `ent_embeddings.pkl`
- `rel_embeddings.pkl`

如果还没有，先运行：
```bash
python train_transe.py --prepare_data
```

### 2. 安装依赖

```bash
pip install sentence-transformers scikit-learn tqdm
```

## 🚀 快速开始

### ⚠️ 首先测试本地模型

在开始之前，建议先测试本地 sentence-bert 模型是否可以正常加载：

```bash
python test_sentence_bert.py
```

### 方式 1: 一键运行（推荐）

**Windows:**
```bash
run_generate_hybrid.bat
```

这会自动使用本地 sentence-bert 模型（位于 `../../../sentence-bert`）。

### 方式 2: 命令行运行

**基础用法**（使用 triples.txt 中的所有实体和关系，默认使用本地模型）:
```bash
python generate_hybrid_embeddings.py
```

**自定义参数**:
```bash
python generate_hybrid_embeddings.py \
    --kb_entity_file entity2id.txt \
    --kb_relation_file relation2id.txt \
    --transe_dir ./output \
    --triple_file triples.txt \
    --sentence_model ../../../sentence-bert \
    --output_dir ./hybrid_embeddings
```

**注意**: 代码已默认使用本地 sentence-bert 模型，无需手动指定。

## ⚙️ 命令行参数

### KB 相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--kb_entity_file` | entity2id.txt | KB中的实体映射文件 |
| `--kb_relation_file` | relation2id.txt | KB中的关系映射文件 |
| `--transe_dir` | ./output | TransE嵌入所在目录 |
| `--transe_format` | pkl | TransE嵌入格式（pkl/npy） |

### 输入数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--triple_file` | triples.txt | 三元组文件（head\ttail\trelation） |
| `--entity_file` | "" | 额外的实体列表（可选） |
| `--relation_file` | "" | 额外的关系列表（可选） |

### SentenceTransformer 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sentence_model` | ../../../sentence-bert | 模型路径（默认使用本地模型）⭐ |
| `--batch_size` | 32 | Batch size |

**注意**: 代码默认使用本地 sentence-bert 模型。如需使用其他模型，可指定不同的路径或在线模型名称。

### 输出参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | ./hybrid_embeddings | 输出目录 |

## 📊 输出文件

生成的文件位于 `./hybrid_embeddings/` 目录：

```
hybrid_embeddings/
├── entity_hybrid_embeddings.pkl      # 实体嵌入（字典格式）
├── relation_hybrid_embeddings.pkl    # 关系嵌入（字典格式）
├── entity_hybrid_embeddings.npy      # 实体嵌入（矩阵格式）
├── relation_hybrid_embeddings.npy    # 关系嵌入（矩阵格式）
├── entity2idx.pkl / entity2idx.txt   # 实体到索引的映射
├── relation2idx.pkl / relation2idx.txt # 关系到索引的映射
└── embedding_stats.pkl               # 统计信息
```

### 文件格式说明

1. **字典格式（.pkl）**:
   ```python
   {
       "Paris": array([...]),  # 维度: transe_dim + sentence_dim
       "France": array([...]),
       ...
   }
   ```

2. **矩阵格式（.npy）**:
   ```python
   array([[...],  # 实体0的嵌入
          [...],  # 实体1的嵌入
          ...])   # 形状: (num_entities, transe_dim + sentence_dim)
   ```

## 💡 使用嵌入

### 方式 1: 加载字典格式（推荐）

```python
import pickle

# 加载嵌入
entity_embeddings = pickle.load(open('./hybrid_embeddings/entity_hybrid_embeddings.pkl', 'rb'))
relation_embeddings = pickle.load(open('./hybrid_embeddings/relation_hybrid_embeddings.pkl', 'rb'))

# 获取特定实体的嵌入
entity_name = "Paris"
if entity_name in entity_embeddings:
    embedding = entity_embeddings[entity_name]
    print(f"{entity_name} 的嵌入维度: {embedding.shape}")
```

### 方式 2: 加载矩阵格式

```python
import pickle
import numpy as np

# 加载嵌入矩阵
entity_matrix = np.load('./hybrid_embeddings/entity_hybrid_embeddings.npy')
entity2idx = pickle.load(open('./hybrid_embeddings/entity2idx.pkl', 'rb'))

# 获取特定实体的嵌入
entity_name = "Paris"
if entity_name in entity2idx:
    idx = entity2idx[entity_name]
    embedding = entity_matrix[idx]
    print(f"{entity_name} 的嵌入: {embedding}")
```

### 方式 3: 使用示例脚本

```bash
python use_hybrid_embeddings.py
```

这会展示：
- 如何加载嵌入
- 如何查找相似实体
- 如何进行三元组预测
- 如何计算三元组得分

## 🎯 应用场景

### 1. 响应图谱嵌入

如果你有响应图谱的三元组文件（如 `response_triples.txt`）：

```bash
python generate_hybrid_embeddings.py \
    --triple_file response_triples.txt \
    --output_dir ./response_embeddings
```

### 2. 特定实体/关系列表

如果你有单独的实体和关系列表：

```bash
python generate_hybrid_embeddings.py \
    --entity_file my_entities.txt \
    --relation_file my_relations.txt \
    --output_dir ./custom_embeddings
```

### 3. 使用不同的 SentenceTransformer 模型

```bash
# 默认使用本地模型（推荐）
python generate_hybrid_embeddings.py

# 显式指定本地模型
python generate_hybrid_embeddings.py \
    --sentence_model ../../../sentence-bert

# 使用其他本地模型
python generate_hybrid_embeddings.py \
    --sentence_model /path/to/other/model

# 使用在线模型（需要网络，会自动下载）
python generate_hybrid_embeddings.py \
    --sentence_model sentence-transformers/all-mpnet-base-v2

# 使用多语言模型（需要网络）
python generate_hybrid_embeddings.py \
    --sentence_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 📈 嵌入维度

假设：
- TransE 维度 = 100
- SentenceTransformer 维度 = 384 (本地 sentence-bert 模型)

则：
- **混合嵌入维度 = 100 + 384 = 484**

### 本地模型 vs 在线模型

**本地 sentence-bert 模型**（默认，推荐）:
- ✅ 无需网络，离线可用
- ✅ 加载速度快
- ✅ 结果可重现
- ✅ 384 维，速度快

**其他在线模型**（需手动指定）:
- `all-mpnet-base-v2`: 768 维（更好的质量，需下载）
- `all-MiniLM-L12-v2`: 384 维（需下载）
- `paraphrase-multilingual-MiniLM-L12-v2`: 384 维（多语言，需下载）

## 📊 统计信息示例

运行后会显示：

```
统计信息
========================================
实体:
  总数: 6000
  KB中: 5478 (91.30%)
  OOV: 522 (8.70%)

关系:
  总数: 2000
  KB中: 1856 (92.80%)
  OOV: 144 (7.20%)

嵌入维度:
  TransE: 100
  SentenceTransformer: 384
  混合嵌入: 484
```

这说明：
- 91.30% 的实体在KB中，使用 TransE + Sentence 嵌入
- 8.70% 的实体是 OOV，使用 Zero + Sentence 嵌入

## 🔬 验证嵌入质量

### 检查嵌入维度

```python
import pickle

entity_emb = pickle.load(open('./hybrid_embeddings/entity_hybrid_embeddings.pkl', 'rb'))
stats = pickle.load(open('./hybrid_embeddings/embedding_stats.pkl', 'rb'))

print(f"期望维度: {stats['hybrid_dim']}")
print(f"实际维度: {list(entity_emb.values())[0].shape[0]}")
```

### 检查 OOV 处理

```python
# 加载统计信息
stats = pickle.load(open('./hybrid_embeddings/embedding_stats.pkl', 'rb'))

print(f"OOV 实体数: {stats['oov_entities']}")
print(f"OOV 关系数: {stats['oov_relations']}")

# OOV 实体的嵌入前半部分应该是零向量
entity_emb = pickle.load(open('./hybrid_embeddings/entity_hybrid_embeddings.pkl', 'rb'))
oov_entity = "some_oov_entity"  # 替换为实际的 OOV 实体

if oov_entity in entity_emb:
    transe_part = entity_emb[oov_entity][:stats['transe_dim']]
    print(f"TransE 部分全零: {np.allclose(transe_part, 0)}")
```

## ⚠️ 常见问题

### Q1: 如何确定哪些是 OOV 实体？

查看输出的统计信息，或者：
```python
entity2id = {}
with open('entity2id.txt', 'r', encoding='utf-8') as f:
    for line in f:
        entity, eid = line.strip().split('\t')
        entity2id[entity] = int(eid)

# 检查某个实体是否在KB中
entity = "Paris"
if entity in entity2id:
    print(f"{entity} 在KB中")
else:
    print(f"{entity} 是 OOV")
```

### Q2: 生成速度慢怎么办？

1. 增加 `--batch_size`（如 64, 128）
2. 使用更小的 SentenceTransformer 模型
3. 使用 GPU（SentenceTransformer 会自动检测）

### Q3: 内存不足怎么办？

1. 减少 `--batch_size`
2. 分批处理实体和关系
3. 使用更小的 SentenceTransformer 模型

### Q4: 如何使用不同的 TransE 维度？

TransE 维度在训练时指定（`train_transe.py --dim`），混合嵌入会自动适配。

## 🔗 相关文档

- **TransE 训练**: `README_TransE_Updated.md`
- **快速开始**: `QUICKSTART.md`
- **问题修复**: `FIX_NOTES.md`

## 📝 完整工作流程

```bash
# 1. 提取实体和关系
python extract_entities_relations.py

# 2. 提取三元组
python extract_triples.py

# 3. 训练 TransE
python train_transe.py --prepare_data

# 4. 生成混合嵌入
python generate_hybrid_embeddings.py

# 5. 使用嵌入
python use_hybrid_embeddings.py
```

## 🎓 技术细节

### 嵌入拼接方式

```python
# 对于 KB 中的实体
transe_emb = transe_ent_embeddings[entity_id]  # shape: (100,)
sent_emb = sentence_model.encode(entity_name)   # shape: (384,)
hybrid_emb = np.concatenate([transe_emb, sent_emb])  # shape: (484,)

# 对于 OOV 实体
zero_emb = np.zeros(100)  # shape: (100,)
sent_emb = sentence_model.encode(entity_name)  # shape: (384,)
hybrid_emb = np.concatenate([zero_emb, sent_emb])  # shape: (484,)
```

### 为什么使用零向量？

1. **维度一致性**：确保所有嵌入维度相同
2. **语义保留**：SentenceTransformer 部分仍包含完整的语义信息
3. **下游兼容**：便于统一处理，无需特殊逻辑

## 📅 更新日期

2024-12-04

