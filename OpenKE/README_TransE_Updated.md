# TransE 模型训练说明（优化版）

本文档说明如何使用优化后的训练脚本，通过 OpenKE 训练 TransE 模型并获取实体和关系的图谱嵌入表示。

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
2. **relation2id.txt** - 关系到ID的映射（1,856个关系）
3. **triples.txt** - 三元组数据（3,975个三元组）

## 🚀 快速开始

### 方法 1: 使用脚本文件（推荐）

**Windows:**
```bash
run_train.bat
```

**Linux/Mac:**
```bash
bash run_train.sh
```

### 方法 2: 使用命令行参数

**基础训练**（包含数据准备）:

```bash
python train_transe.py \
    --prepare_data \
    --dim 100 \
    --epoch 1000 \
    --lr 1.0 \
    --margin 5.0 \
    --batch_size 300 \
    --neg_ent 64 \
    --save_steps 10 \
    --datadir ./openke_data \
    --outdir ./output
```

**如果已经准备好 OpenKE 格式的数据**:

```bash
python train_transe.py \
    --dim 100 \
    --epoch 1000 \
    --lr 1.0 \
    --margin 5.0 \
    --batch_size 300 \
    --neg_ent 64 \
    --datadir ./openke_data \
    --outdir ./output
```

**带测试评估的训练**:

```bash
python train_transe.py \
    --prepare_data \
    --test \
    --dim 100 \
    --epoch 1000 \
    --datadir ./openke_data \
    --outdir ./output
```

## ⚙️ 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dim` | int | 100 | 嵌入维度（推荐：50, 100, 200） |
| `--epoch` | int | 1000 | 训练轮数 |
| `--lr` | float | 1.0 | 学习率 |
| `--margin` | float | 5.0 | Margin损失的margin值 |
| `--batch_size` | int | 300 | Batch size |
| `--neg_ent` | int | 64 | 负采样实体数量 |
| `--bern_flag` | int | 0 | 是否使用伯努利负采样（0/1） |
| `--save_steps` | int | 10 | 每隔多少epoch保存一次 |
| `--patient` | int | -1 | 早停耐心值（-1表示不使用） |
| `--datadir` | str | ./openke_data | OpenKE格式数据目录 |
| `--outdir` | str | ./output | 输出目录 |
| `--model_path` | str | "" | 预训练模型路径（可选） |
| `--prepare_data` | flag | False | 是否需要准备数据 |
| `--test` | flag | False | 是否运行测试评估 |

## 📊 输出文件

训练完成后，在 `output/` 目录下会生成：

```
output/
├── transe_final.ckpt          # 最终模型检查点
├── ent_embeddings.pkl         # 实体嵌入（pickle格式）
├── rel_embeddings.pkl         # 关系嵌入（pickle格式）
├── ent_embeddings.npy         # 实体嵌入（numpy格式）
└── rel_embeddings.npy         # 关系嵌入（numpy格式）
```

**说明**:
- `.pkl` 文件使用 pickle 格式保存，可以直接用 `pickle.load()` 加载
- `.npy` 文件使用 numpy 格式保存，可以用 `np.load()` 加载
- 嵌入按照 ID 顺序保存（ID=0的实体/关系在第0行）

## 💡 使用嵌入

### 加载嵌入向量

```python
import pickle
import numpy as np

# 方法1: 加载 pickle 格式
ent_embeddings = pickle.load(open('./output/ent_embeddings.pkl', 'rb'))
rel_embeddings = pickle.load(open('./output/rel_embeddings.pkl', 'rb'))

# 方法2: 加载 numpy 格式
ent_embeddings = np.load('./output/ent_embeddings.npy')
rel_embeddings = np.load('./output/rel_embeddings.npy')

print(f"实体嵌入形状: {ent_embeddings.shape}")  # (5478, 100)
print(f"关系嵌入形状: {rel_embeddings.shape}")  # (1856, 100)
```

### 使用示例脚本

```bash
python use_embeddings.py
```

这个脚本会展示：
- 如何加载嵌入向量
- 如何查找相似实体
- 如何进行三元组预测
- 如何导出可读格式

### 获取特定实体的嵌入

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
    embedding = ent_embeddings[entity_id]
    print(f"{entity_name} 的嵌入: {embedding}")
```

### 计算实体相似度

```python
from sklearn.metrics.pairwise import cosine_similarity

# 计算两个实体的余弦相似度
entity1_id = entity2id["Paris"]
entity2_id = entity2id["France"]

entity1_emb = ent_embeddings[entity1_id].reshape(1, -1)
entity2_emb = ent_embeddings[entity2_id].reshape(1, -1)

similarity = cosine_similarity(entity1_emb, entity2_emb)[0][0]
print(f"相似度: {similarity:.4f}")
```

### 知识图谱补全

```python
# TransE: h + r ≈ t
# 给定头实体和关系，预测尾实体

head_id = entity2id["Paris"]
relation_id = relation2id["capital_of"]

# 预测尾实体
predicted_tail = ent_embeddings[head_id] + rel_embeddings[relation_id]

# 找到最接近的实体
distances = np.linalg.norm(ent_embeddings - predicted_tail, axis=1)
nearest_entity_id = np.argmin(distances)

print(f"预测的尾实体ID: {nearest_entity_id}")
```

## 🎯 不同场景的推荐配置

### 快速测试（~5分钟）
```bash
python train_transe.py --prepare_data --dim 50 --epoch 100 --outdir ./output_test
```

### 标准训练（~30分钟）
```bash
python train_transe.py --prepare_data --dim 100 --epoch 1000 --outdir ./output
```

### 高质量训练（~2小时）
```bash
python train_transe.py --prepare_data --dim 200 --epoch 2000 --lr 0.5 --outdir ./output_high
```

### 使用GPU加速
确保安装了 CUDA 版本的 PyTorch，脚本会自动检测并使用 GPU。

## ⚠️ 常见问题

### Q1: 训练很慢怎么办？

**方案**:
1. 减少 `--epoch`（如改为 100-500）
2. 减少 `--dim`（如改为 50）
3. 增加 `--batch_size`
4. 使用 GPU（安装 CUDA 版本的 PyTorch）
5. 减少 `--neg_ent`（如改为 25）

### Q2: 内存不足怎么办？

**方案**:
1. 减少 `--dim`（嵌入维度）
2. 减少 `--batch_size`
3. 减少 `--neg_ent`

### Q3: 如何从中断处继续训练？

使用 `--model_path` 参数：

```bash
python train_transe.py \
    --model_path ./output/transe_final.ckpt \
    --epoch 500 \
    --datadir ./openke_data \
    --outdir ./output_continue
```

### Q4: 如何评估模型质量？

添加 `--test` 参数：

```bash
python train_transe.py --prepare_data --test --datadir ./openke_data --outdir ./output
```

这会在训练后评估链接预测的性能（MR, MRR, Hits@10等）。

**注意**: 需要在 `datadir` 中准备 `test2id.txt` 文件。

### Q5: pickle 和 numpy 格式有什么区别？

- **pickle 格式** (`.pkl`): Python 原生序列化，保留完整的 numpy 数组信息
- **numpy 格式** (`.npy`): numpy 专用格式，文件更小，加载速度更快

两种格式内容相同，按需选择即可。

## 📚 进一步使用

训练好的嵌入可以用于：

1. **实体分类**: 将实体嵌入作为特征进行分类
2. **实体聚类**: 基于嵌入向量进行聚类分析
3. **关系抽取**: 使用嵌入辅助关系抽取任务
4. **问答系统**: 利用嵌入进行语义匹配
5. **推荐系统**: 基于实体嵌入的相似度推荐
6. **图神经网络**: 作为初始节点特征

## 🔗 参考资料

- [OpenKE GitHub](https://github.com/thunlp/OpenKE)
- [TransE 论文](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
- [OpenKE 文档](https://openke.readthedocs.io/)

## 📝 更新日志

### v2.0 (优化版)
- ✅ 添加命令行参数支持
- ✅ 添加测试评估功能
- ✅ 支持从预训练模型继续训练
- ✅ 同时保存 pickle 和 numpy 格式
- ✅ 添加早停机制
- ✅ 优化训练流程
- ✅ 添加训练脚本示例

### v1.0 (初始版)
- 基础 TransE 训练功能
- numpy 格式保存

