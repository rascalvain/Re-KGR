# Mintaka RGCN 链接预测

基于 Mintaka 数据集的 entity_triples，使用 RGCN + 链接预测目标训练图神经网络，更新节点表示。

## 流程概览

```
hybrid_embeddings/          →  prepare_embeddings.py  →  RGCN格式嵌入
mintaka_dev_stage1_*.json   →  train_rgcn_linkpred.py →  训练RGCN模型
训练好的模型 + 数据         →  update_node.py         →  更新后的节点嵌入
```

## 前置要求

1. 已完成 TransE 训练和混合嵌入生成（`train_transe/` 目录）
2. 服务器路径 `/root/autodl-fs/gca/mintaka/` 下存在：
   - `train_transe/hybrid_embeddings/entity_hybrid_embeddings.pkl`
   - `train_transe/hybrid_embeddings/relation_hybrid_embeddings.pkl`
   - `train_transe/hybrid_embeddings/entity2idx.pkl`
   - `train_transe/hybrid_embeddings/relation2idx.pkl`
   - `preprocess_data/data/mintaka_dev_stage1_canonicalized.json`

3. Python 依赖：
```bash
pip install torch torch_geometric numpy tqdm
```

## 使用方法

### 一键运行

```bash
bash run_train.sh
```

### 分步运行

```bash
# 1. 将混合嵌入转为RGCN矩阵格式
python prepare_embeddings.py

# 2. 训练RGCN（链接预测目标）
python train_rgcn_linkpred.py

# 3. 提取更新后的节点嵌入
python update_node.py
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config_mintaka.py` | 配置文件（路径、超参数） |
| `prepare_embeddings.py` | 将dict格式混合嵌入转为RGCN矩阵格式 |
| `train_rgcn_linkpred.py` | RGCN链接预测训练（核心） |
| `update_node.py` | 从训练好的RGCN提取更新后的节点嵌入 |
| `run_train.sh` | 一键训练脚本 |

## 模型架构

- **编码器**: 2层 RGCNConv + BatchNorm，输入输出维度相同（保持嵌入维度不变）
- **评分函数**: DistMult（head * relation * tail 的内积）
- **训练目标**: Margin-based loss，正样本得分高于负样本
- **负采样**: 随机替换 head 或 tail

## 数据格式

Mintaka 的 `entity_triples` 格式：
```json
{
  "Q123": {
    "label": "Paris",
    "triple_count": 5,
    "triples": [
      {"head": "Paris", "relation": "http://www.wikidata.org/prop/direct/P17", "tail": "France"},
      ...
    ]
  }
}
```

训练时将每个样本的所有 entity_triples 展平为一个子图。

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| NUM_LAYERS | 2 | RGCN层数 |
| DROPOUT | 0.3 | Dropout率 |
| BATCH_SIZE | 4 | 每批子图数 |
| NUM_EPOCHS | 100 | 最大训练轮数 |
| LEARNING_RATE | 1e-3 | 学习率 |
| MARGIN | 1.0 | Margin loss的margin |
| EARLY_STOPPING_PATIENCE | 20 | 早停耐心值 |
| MAX_TRIPLES_PER_SUBGRAPH | 32 | 每个子图最大三元组数 |

## 输出

```
rgcn_output/
├── checkpoints/
│   ├── best_model_subgraph.pth    # 最佳模型
│   └── checkpoint_epoch_*.pth     # 定期保存
└── logs/

hybrid_embeddings/
└── node_embeddings_linkpred_rgcn.pkl  # 更新后的节点嵌入
```

`node_embeddings_linkpred_rgcn.pkl` 内容：
```python
{
    'embeddings': np.ndarray,   # [num_entities, embedding_dim]
    'num_entities': int,
    'embedding_dim': int,
    'entity2id': dict,          # {entity_name: index}
    'coverage': int,            # 有RGCN表示的实体数
    'source': 'link_prediction_rgcn_mintaka'
}
```
