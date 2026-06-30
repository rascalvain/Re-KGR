# 完整工作流程总结

从数据提取到混合嵌入生成的完整流程。

## 📋 工作流程概览

```
原始数据 (JSON)
    ↓
[1] 提取实体和关系
    ↓
entity2id.txt + relation2id.txt
    ↓
[2] 提取三元组
    ↓
triples.txt
    ↓
[3] 训练 TransE
    ↓
TransE 嵌入 (output/)
    ↓
[4] 生成混合嵌入
    ↓
混合嵌入 (hybrid_embeddings/)
    ↓
[5] 下游应用
```

## 🔢 步骤详解

### 步骤 1: 提取实体和关系

**目的**: 从 JSON 文件中提取所有唯一的实体和关系

**输入**: `hotpot_dev_with_triples_aligned.json`

**输出**: 
- `entity2id.txt` (5,478 个实体)
- `relation2id.txt` (1,856 个关系)

**命令**:
```bash
python extract_entities_relations.py
```

**文件说明**:
- 自动去重
- 按 ID 顺序保存（从 0 开始）
- 格式：实体/关系名称 \t ID

---

### 步骤 2: 提取三元组

**目的**: 从 JSON 文件中提取所有 context_triples

**输入**: 
- `hotpot_dev_with_triples_aligned.json`
- `entity2id.txt`
- `relation2id.txt`

**输出**: 
- `triples.txt` (3,975 个三元组)

**命令**:
```bash
python extract_triples.py
```

**文件格式**: `head \t tail \t relation`

---

### 步骤 3: 训练 TransE

**目的**: 在知识图谱（KB）上训练 TransE 模型，获取结构化嵌入

**输入**:
- `entity2id.txt`
- `relation2id.txt`
- `triples.txt`

**输出**: `output/`
- `ent_embeddings.pkl` (5,478 × 100)
- `rel_embeddings.pkl` (1,856 × 100)
- `transe_final.ckpt`

**命令**:
```bash
# Windows 一键运行
run_train.bat

# 或手动运行
python train_transe.py --prepare_data

# 快速测试
python train_transe.py --prepare_data --dim 50 --epoch 100
```

**关键参数**:
- `--dim`: 嵌入维度（默认 100）
- `--epoch`: 训练轮数（默认 1000）
- `--lr`: 学习率（默认 1.0）

**说明**:
- 自动检测 GPU
- 嵌入按 ID 顺序保存
- 同时保存 pkl 和 npy 格式

---

### 步骤 4: 生成混合嵌入

**目的**: 结合 TransE（结构）和 SentenceTransformer（语义），处理 OOV 问题

**输入**:
- `entity2id.txt` + `relation2id.txt` (KB映射)
- `output/ent_embeddings.pkl` + `rel_embeddings.pkl` (TransE嵌入)
- `triples.txt` (需要嵌入的实体/关系)

**输出**: `hybrid_embeddings/`
- `entity_hybrid_embeddings.pkl` (字典格式)
- `relation_hybrid_embeddings.pkl` (字典格式)
- `entity_hybrid_embeddings.npy` (矩阵格式)
- `relation_hybrid_embeddings.npy` (矩阵格式)
- `entity2idx.txt` / `relation2idx.txt` (映射)
- `embedding_stats.pkl` (统计信息)

**命令**:
```bash
# Windows 一键运行
run_generate_hybrid.bat

# 或手动运行
python generate_hybrid_embeddings.py

# 使用自定义数据
python generate_hybrid_embeddings.py --triple_file response_triples.txt
```

**嵌入规则**:
- **在 KB 中**: `embedding = Concat(TransE_Vector, Sentence_Vector)`
- **不在 KB (OOV)**: `embedding = Concat(Zero_Vector, Sentence_Vector)`

**嵌入维度**: TransE维度 (100) + Sentence维度 (384) = 484

---

### 步骤 5: 使用嵌入

**目的**: 在下游任务中使用混合嵌入

**命令**:
```bash
python use_hybrid_embeddings.py
```

**示例应用**:
- 实体分类
- 知识图谱补全
- 三元组验证
- 相似度计算

---

## 📊 数据统计

| 项目 | 数量 | 文件 |
|------|------|------|
| JSON 记录 | 110 | hotpot_dev_with_triples_aligned.json |
| 实体总数 | 5,478 | entity2id.txt |
| 关系总数 | 1,856 | relation2id.txt |
| 三元组总数 | 3,975 | triples.txt |
| TransE 实体嵌入 | 5,478 × 100 | output/ent_embeddings.pkl |
| TransE 关系嵌入 | 1,856 × 100 | output/rel_embeddings.pkl |
| 混合实体嵌入 | N × 484 | hybrid_embeddings/entity_hybrid_embeddings.pkl |
| 混合关系嵌入 | M × 484 | hybrid_embeddings/relation_hybrid_embeddings.pkl |

N 和 M 取决于输入数据（可能包含 OOV 实体/关系）

---

## 🚀 快速开始（完整流程）

### 方式 1: 一步一步执行

```bash
# 1. 提取实体和关系
python extract_entities_relations.py

# 2. 提取三元组
python extract_triples.py

# 3. 训练 TransE（快速测试）
python train_transe.py --prepare_data --dim 50 --epoch 100

# 4. 生成混合嵌入
python generate_hybrid_embeddings.py

# 5. 使用嵌入
python use_hybrid_embeddings.py
```

### 方式 2: 使用批处理脚本（Windows）

```bash
# 训练 TransE
run_train.bat

# 生成混合嵌入
run_generate_hybrid.bat
```

---

## 📁 最终文件结构

```
hotpotqa/
│
├── 数据文件
│   ├── hotpot_dev_with_triples_aligned.json  # 原始数据
│   ├── entity2id.txt                         # 实体映射
│   ├── relation2id.txt                       # 关系映射
│   └── triples.txt                           # 三元组
│
├── TransE 嵌入
│   └── output/
│       ├── ent_embeddings.pkl                # 实体 TransE 嵌入
│       ├── rel_embeddings.pkl                # 关系 TransE 嵌入
│       ├── ent_embeddings.npy
│       ├── rel_embeddings.npy
│       └── transe_final.ckpt                 # 模型检查点
│
├── 混合嵌入
│   └── hybrid_embeddings/
│       ├── entity_hybrid_embeddings.pkl      # 实体混合嵌入（字典）
│       ├── relation_hybrid_embeddings.pkl    # 关系混合嵌入（字典）
│       ├── entity_hybrid_embeddings.npy      # 实体混合嵌入（矩阵）
│       ├── relation_hybrid_embeddings.npy    # 关系混合嵌入（矩阵）
│       ├── entity2idx.txt                    # 实体索引映射
│       ├── relation2idx.txt                  # 关系索引映射
│       └── embedding_stats.pkl               # 统计信息
│
└── 代码和文档
    ├── extract_entities_relations.py         # 提取实体和关系
    ├── extract_triples.py                    # 提取三元组
    ├── train_transe.py                       # 训练 TransE
    ├── generate_hybrid_embeddings.py         # 生成混合嵌入
    ├── use_embeddings.py                     # 使用 TransE 嵌入
    ├── use_hybrid_embeddings.py              # 使用混合嵌入
    ├── run_train.bat                         # TransE 训练脚本
    ├── run_generate_hybrid.bat               # 混合嵌入生成脚本
    ├── README_TransE_Updated.md              # TransE 详细文档
    ├── README_Hybrid_Embeddings.md           # 混合嵌入详细文档
    ├── QUICKSTART.md                         # 快速开始
    └── WORKFLOW_SUMMARY.md                   # 本文档
```

---

## 🎯 常见使用场景

### 场景 1: 仅使用 KB 中的实体和关系

```bash
# 步骤 1-3: 提取数据并训练 TransE
python extract_entities_relations.py
python extract_triples.py
python train_transe.py --prepare_data

# 使用 TransE 嵌入（无 OOV）
python use_embeddings.py
```

### 场景 2: 包含 OOV 实体/关系（响应图谱）

```bash
# 步骤 1-3: 同上

# 生成混合嵌入（处理 OOV）
python generate_hybrid_embeddings.py \
    --triple_file response_triples.txt \
    --output_dir ./response_embeddings

# 使用混合嵌入
python use_hybrid_embeddings.py
```

### 场景 3: 自定义 SentenceTransformer 模型

```bash
# 使用更大的模型（更好的语义表示）
python generate_hybrid_embeddings.py \
    --sentence_model sentence-transformers/all-mpnet-base-v2
```

---

## 💡 最佳实践

### TransE 训练

1. **快速测试**: `--dim 50 --epoch 100` (~5分钟)
2. **标准训练**: `--dim 100 --epoch 1000` (~30分钟)
3. **高质量**: `--dim 200 --epoch 2000` (~2小时)

### 混合嵌入生成

1. **内存充足**: 使用大 `--batch_size` (64, 128)
2. **GPU 可用**: 自动加速 SentenceTransformer
3. **追求质量**: 使用 `all-mpnet-base-v2` 模型

### 嵌入使用

1. **小数据集**: 使用字典格式 (.pkl)
2. **大数据集**: 使用矩阵格式 (.npy)
3. **需要映射**: 使用 entity2idx.txt

---

## ⚠️ 常见问题

### Q1: TransE 训练报错 `unexpected keyword argument 'patient'`

**解决**: 已修复，代码会自动适配不同版本的 OpenKE。

### Q2: 混合嵌入生成很慢

**解决**: 
1. 增加 `--batch_size`
2. 使用 GPU
3. 使用更小的 SentenceTransformer 模型

### Q3: 如何确认嵌入是否正确生成？

**验证**:
```python
import pickle
import numpy as np

# 检查维度
entity_emb = pickle.load(open('./hybrid_embeddings/entity_hybrid_embeddings.pkl', 'rb'))
print(f"实体数: {len(entity_emb)}")
print(f"嵌入维度: {list(entity_emb.values())[0].shape}")

# 检查统计
stats = pickle.load(open('./hybrid_embeddings/embedding_stats.pkl', 'rb'))
print(f"KB实体: {stats['kb_entities']}")
print(f"OOV实体: {stats['oov_entities']}")
```

---

## 📚 相关文档

- **TransE 训练**: `README_TransE_Updated.md`
- **混合嵌入**: `README_Hybrid_Embeddings.md`
- **快速开始**: `QUICKSTART.md`
- **问题修复**: `FIX_NOTES.md`

---

## 🔗 参考资料

- [OpenKE](https://github.com/thunlp/OpenKE)
- [SentenceTransformers](https://www.sbert.net/)
- [TransE 论文](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)

---

## 📅 更新记录

- **2024-12-04**: 创建完整工作流程文档
- **2024-12-04**: 添加混合嵌入生成功能
- **2024-12-04**: 修复 OpenKE 版本兼容性问题

---

## ✅ 检查清单

完成所有步骤后，确认以下文件存在：

- [ ] `entity2id.txt`
- [ ] `relation2id.txt`
- [ ] `triples.txt`
- [ ] `output/ent_embeddings.pkl`
- [ ] `output/rel_embeddings.pkl`
- [ ] `hybrid_embeddings/entity_hybrid_embeddings.pkl`
- [ ] `hybrid_embeddings/relation_hybrid_embeddings.pkl`
- [ ] `hybrid_embeddings/embedding_stats.pkl`

全部存在 = 成功完成！🎉

