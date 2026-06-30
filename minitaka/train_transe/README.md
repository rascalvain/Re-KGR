# Mintaka TransE 训练

基于 `preprocess_data` 管线产出的数据，使用 OpenKE 训练 TransE 图嵌入。

## 目录结构

```
train_transe/
├── train_transe.py                # 主训练脚本（数据转换 + 训练 + 嵌入导出）
├── use_embeddings.py              # 嵌入使用示例（相似度、链接预测等）
├── generate_hybrid_embeddings.py  # TransE + SentenceTransformer 混合嵌入
├── run_train.sh                   # Linux 一键训练
├── run_train.bat                  # Windows 一键训练
└── README.md
```

## 前置要求

### 1. 安装 OpenKE

```bash
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
cd OpenKE/openke
bash make.sh
cd ..
python setup.py install
```

### 2. Python 依赖

```bash
pip install numpy torch scikit-learn
# 生成混合嵌入时还需要：
pip install sentence-transformers tqdm
```

### 3. 数据准备

确保服务器上 `/root/autodl-fs/gca/mintaka/preprocess_data/data/` 下已通过步骤 1-9 生成以下文件：

| 文件 | 说明 |
|------|------|
| `entity2id.txt` | 实体名 → ID 映射（`name\tid`，无首行 count） |
| `relation2id_deduplicated.txt` | 关系名 → ID 映射（同上） |
| `triples.txt` | 三元组（`head\ttail\trelation`，首行为表头） |

脚本默认从 `/root/autodl-fs/gca/mintaka/preprocess_data/data/` 读取，无需手动复制。

## 快速开始

### Windows

```bash
run_train.bat
```

### Linux / Mac

```bash
bash run_train.sh
```

### 命令行（自定义参数）

```bash
# 标准训练（约 30 分钟，CPU）
python train_transe.py --dim 100 --epoch 1000

# 快速测试（约 5 分钟）
python train_transe.py --dim 50 --epoch 100

# 高维度（与 SentenceBERT 拼接时推荐）
python train_transe.py --dim 384 --epoch 1000

# 跳过数据准备（openke_data/ 已存在时）
python train_transe.py --skip_prepare --epoch 500

# 从检查点继续训练
python train_transe.py --skip_prepare --model_path ./output/transe_final.ckpt --epoch 500
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--entity2id` | `/root/autodl-fs/gca/mintaka/preprocess_data/data/entity2id.txt` | 实体映射文件 |
| `--relation2id` | `/root/autodl-fs/gca/mintaka/preprocess_data/data/relation2id_deduplicated.txt` | 关系映射文件 |
| `--triples` | `/root/autodl-fs/gca/mintaka/preprocess_data/data/triples.txt` | 三元组文件 |
| `--datadir` | `./openke_data` | OpenKE 格式数据输出目录 |
| `--outdir` | `./output` | 模型与嵌入输出目录 |
| `--dim` | 100 | 嵌入维度 |
| `--epoch` | 1000 | 训练轮数 |
| `--batch_size` | 300 | batch size |
| `--lr` | 1.0 | 学习率 |
| `--margin` | 5.0 | MarginLoss margin |
| `--neg_ent` | 64 | 负采样实体数 |
| `--bern_flag` | 0 | 1=伯努利负采样 |
| `--save_steps` | 100 | checkpoint 保存间隔 |
| `--threads` | 8 | 数据加载线程数 |
| `--model_path` | | 预训练模型路径（继续训练用） |
| `--skip_prepare` | false | 跳过数据格式转换 |
| `--test` | false | 训练后运行链接预测评估 |

## 输出文件

训练完成后产出：

```
output/
├── transe_final.ckpt      # PyTorch checkpoint
├── ent_embeddings.pkl      # 实体嵌入 (N_entity, dim)  pickle 格式
├── ent_embeddings.npy      # 实体嵌入 numpy 格式
├── rel_embeddings.pkl      # 关系嵌入 (N_relation, dim)
├── rel_embeddings.npy
└── data_stats.json         # 数据统计信息

openke_data/                # OpenKE 格式中间文件
├── entity2id.txt           # 首行 count + name\tid
├── relation2id.txt
├── train2id.txt            # 首行 count + h\tt\tr
├── valid2id.txt            # 占位 (0)
└── test2id.txt             # 占位 (0)
```

## 数据流

```
/root/autodl-fs/gca/mintaka/
  preprocess_data/data/                    train_transe/
    entity2id.txt          ─┐
    relation2id_deduplicated.txt ──┼──→ train_transe.py
    triples.txt            ─┘        │
                                     ├──→ openke_data/  (OpenKE 格式)
                                     └──→ output/       (模型 + 嵌入)
```

`train_transe.py` 自动完成以下转换：
1. 为 entity2id / relation2id 添加首行 count（OpenKE 要求）
2. 将 triples.txt 转为 train2id.txt（名称 → ID，tab 分隔）
3. 生成空的 valid2id.txt / test2id.txt 占位文件

## 使用嵌入

### 方式 1：Python 直接加载

```python
import pickle
import numpy as np

# pickle 格式
ent_emb = pickle.load(open("output/ent_embeddings.pkl", "rb"))
rel_emb = pickle.load(open("output/rel_embeddings.pkl", "rb"))

# numpy 格式
ent_emb = np.load("output/ent_embeddings.npy")
rel_emb = np.load("output/rel_embeddings.npy")
```

### 方式 2：示例脚本

```bash
python use_embeddings.py
```

展示：相似实体查找、链接预测、文本导出等。

### 方式 3：生成混合嵌入

```bash
python generate_hybrid_embeddings.py \
    --sentence_model sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir ./hybrid_embeddings
```

对 KB 内实体拼接 TransE + SentenceBERT 向量，OOV 实体用零向量补位。

## 常见问题

**Q: ModuleNotFoundError: No module named 'openke'**
安装 OpenKE：`cd OpenKE && python setup.py install`

**Q: 训练很慢**
减少 `--epoch`、`--dim`，或使用 GPU（安装 CUDA 版 PyTorch，脚本自动检测）。

**Q: 内存不足**
减小 `--batch_size` 和 `--neg_ent`。

**Q: pickle 和 numpy 格式区别？**
内容完全一致。numpy 文件更小、加载更快；pickle 保留完整 Python 对象信息。
