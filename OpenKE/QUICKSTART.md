# TransE 训练快速开始指南

## 📦 一键安装

```bash
# 1. 安装 OpenKE
git clone https://github.com/thunlp/OpenKE
cd OpenKE
bash make.sh
python setup.py install

# 2. 安装依赖
pip install numpy torch scikit-learn
```

## 🚀 一键训练

### Windows 用户

```bash
# 双击或在命令行运行
run_train.bat
```

### Linux/Mac 用户

```bash
bash run_train.sh
```

## 📝 命令行训练（自定义参数）

### 最简单的训练命令

```bash
python train_transe.py --prepare_data
```

这会使用默认参数训练模型（1000个epoch，维度100）。

### 快速测试（5分钟）

```bash
python train_transe.py --prepare_data --dim 50 --epoch 100
```

### 推荐配置（30分钟）

```bash
python train_transe.py --prepare_data --dim 100 --epoch 1000 --lr 1.0 --margin 5.0
```

### 高质量训练（2小时）

```bash
python train_transe.py --prepare_data --dim 200 --epoch 2000 --lr 0.5
```

## 📊 查看结果

训练完成后，在 `output/` 目录下找到：

- `ent_embeddings.pkl` - 实体嵌入（5478个实体 × 100维）
- `rel_embeddings.pkl` - 关系嵌入（1856个关系 × 100维）
- `transe_final.ckpt` - 模型检查点

## 💡 使用嵌入

```python
import pickle

# 加载嵌入
ent_emb = pickle.load(open('./output/ent_embeddings.pkl', 'rb'))
rel_emb = pickle.load(open('./output/rel_embeddings.pkl', 'rb'))

print(f"实体嵌入: {ent_emb.shape}")  # (5478, 100)
print(f"关系嵌入: {rel_emb.shape}")  # (1856, 100)
```

或运行示例脚本：

```bash
python use_embeddings.py
```

## 🎯 常用参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--dim` | 100 | 嵌入维度，越大效果越好但越慢 |
| `--epoch` | 1000 | 训练轮数，越多效果越好但越慢 |
| `--lr` | 1.0 | 学习率，太大不收敛，太小训练慢 |
| `--batch_size` | 300 | batch大小，越大训练越快但占内存 |
| `--neg_ent` | 64 | 负采样数，越多效果越好但越慢 |

## ❓ 问题排查

### 问题1: ModuleNotFoundError: No module named 'openke'

**解决**: 安装 OpenKE
```bash
cd OpenKE
python setup.py install
```

### 问题2: 训练太慢

**解决**: 减少参数
```bash
python train_transe.py --prepare_data --dim 50 --epoch 100
```

### 问题3: 内存不足

**解决**: 减少 batch_size
```bash
python train_transe.py --prepare_data --batch_size 100
```

## 📚 完整文档

- 详细使用说明：`README_TransE_Updated.md`
- 代码实现：`train_transe.py`
- 使用示例：`use_embeddings.py`

## ⏱️ 预计时间

- **数据准备**: < 1 分钟
- **训练（CPU）**: 
  - 100 epoch: ~5 分钟
  - 1000 epoch: ~30 分钟
  - 2000 epoch: ~60 分钟
- **训练（GPU）**: 约为 CPU 的 1/10

## ✅ 训练成功标志

看到以下输出表示成功：

```
========================================
训练完成！
========================================

生成的文件:
  - 模型检查点: ./output/transe_final.ckpt
  - 实体嵌入(pkl): ./output/ent_embeddings.pkl
  - 关系嵌入(pkl): ./output/rel_embeddings.pkl
  - 实体嵌入(npy): ./output/ent_embeddings.npy
  - 关系嵌入(npy): ./output/rel_embeddings.npy
```

现在可以使用嵌入向量了！

