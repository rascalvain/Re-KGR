# 故障排除指南

## ❌ KeyError: 'embeddings'

### 错误信息
```
KeyError: 'embeddings'
File "siamese_rgcn_improved.py", line 33, in __init__
    entity_embeddings = torch.FloatTensor(entity_data['embeddings'])
```

### 问题原因

RGCN模型期望的嵌入文件格式与混合嵌入生成的格式不同。

**混合嵌入格式** (由 `generate_hybrid_embeddings.py` 生成):
```python
{
    "entity1": numpy_array([...]),  # 直接的字典映射
    "entity2": numpy_array([...]),
    ...
}
```

**RGCN需要的格式** (由 `prepare_embeddings.py` 生成):
```python
{
    'embeddings': numpy_array([[...], [...]]),  # 矩阵形式
    'num_entities': int,
    'embedding_dim': int,
    'entity2id': {...}
}
```

### 解决方案

**步骤 1: 检查是否有混合嵌入**

```bash
dir ..\hybrid_embeddings\entity_hybrid_embeddings.pkl
dir ..\hybrid_embeddings\relation_hybrid_embeddings.pkl
```

如果不存在，先生成混合嵌入：
```bash
cd ..
python generate_hybrid_embeddings.py
cd rgcn
```

**步骤 2: 运行嵌入准备脚本**

```bash
python prepare_embeddings.py
```

这会生成：
- `entity_embeddings_rgcn.pkl` ← RGCN需要的实体嵌入
- `relation_mappings_rgcn.pkl` ← RGCN需要的关系映射

**步骤 3: 验证文件生成**

```bash
dir ..\hybrid_embeddings\entity_embeddings_rgcn.pkl
dir ..\hybrid_embeddings\relation_mappings_rgcn.pkl
```

两个文件都应该存在。

**步骤 4: 重新训练**

```bash
python train_rgcn_hotpotqa.py
```

## ❌ 文件不存在错误

### 错误: entity2idx.pkl 不存在

**原因**: 未生成混合嵌入

**解决**:
```bash
cd ..
python generate_hybrid_embeddings.py
cd rgcn
```

### 错误: entity_embeddings_rgcn.pkl 不存在

**原因**: 未运行嵌入准备脚本

**解决**:
```bash
python prepare_embeddings.py
```

## ❌ AttributeError: 'tuple' object has no attribute 'backward'

### 错误信息
```
AttributeError: 'tuple' object has no attribute 'backward'
File "train_rgcn_hotpotqa.py", line 188, in train_epoch
    loss.backward()
```

### 问题原因

损失函数 `ImprovedContrastiveLoss` 返回一个元组 `(loss, loss_dict)`，而不是单个张量。代码试图对整个元组调用 `.backward()`。

### 解决方案

**已修复！** 最新版本的 `train_rgcn_hotpotqa.py` 已经正确处理了这个问题。

如果你仍然遇到这个错误，请更新代码：

```python
# 错误的写法
loss = self.criterion(context_emb, gpt_emb, labels)
loss.backward()

# 正确的写法
loss, loss_dict = self.criterion(context_emb, gpt_emb, labels)
loss.backward()
```

或者直接重新获取最新的代码文件。

## ❌ CUDA out of memory

### 错误信息
```
RuntimeError: CUDA out of memory
```

### 解决方案

**方案 1: 减小batch size**

编辑 `config_hotpotqa.py`:
```python
BATCH_SIZE = 4  # 从8减小到4
```

**方案 2: 减小模型大小**

编辑 `config_hotpotqa.py`:
```python
HIDDEN_CHANNELS = 64   # 从128减小到64
OUT_CHANNELS = 32      # 从64减小到32
```

**方案 3: 使用CPU**

RGCN会自动检测GPU，如果内存不足会使用CPU（较慢）。

## ❌ 数据加载失败

### 错误: No valid samples

**原因**: 数据集没有有效的三元组对

**解决**: 运行测试脚本检查数据
```bash
python test_data_loader.py
```

查看输出，确认：
- 有效样本数 > 0
- Context图和GPT图都有节点和边

## ❌ 训练损失不下降

### 可能原因和解决

**1. 学习率太大**

编辑 `config_hotpotqa.py`:
```python
LEARNING_RATE = 1e-4  # 从1e-3减小到1e-4
```

**2. 数据问题**

检查数据：
```bash
python test_data_loader.py
```

**3. 模型太复杂**

减少层数：
```python
NUM_LAYERS = 2  # 从3减小到2
```

## 🔧 完整检查清单

运行以下命令，逐步检查：

```bash
# 1. 检查混合嵌入
cd ..
python -c "import os; print('混合嵌入存在:' if os.path.exists('hybrid_embeddings/entity_hybrid_embeddings.pkl') else '混合嵌入不存在')"

# 2. 如果不存在，生成混合嵌入
python generate_hybrid_embeddings.py

# 3. 进入rgcn目录
cd rgcn

# 4. 准备RGCN嵌入
python prepare_embeddings.py

# 5. 测试数据加载
python test_data_loader.py

# 6. 开始训练
python train_rgcn_hotpotqa.py
```

## 📋 文件清单

训练前确保以下文件存在：

### 混合嵌入文件（由 generate_hybrid_embeddings.py 生成）
```
../hybrid_embeddings/
├── entity_hybrid_embeddings.pkl     ✓ 必须
├── relation_hybrid_embeddings.pkl   ✓ 必须
├── entity2idx.pkl                   ✓ 必须
└── relation2idx.pkl                 ✓ 必须
```

### RGCN嵌入文件（由 prepare_embeddings.py 生成）
```
../hybrid_embeddings/
├── entity_embeddings_rgcn.pkl       ✓ 必须
└── relation_mappings_rgcn.pkl       ✓ 必须
```

### 数据文件
```
../hotpot_dev_with_triples_aligned.json  ✓ 必须
```

## 🆘 仍然有问题？

### 方案 1: 运行一键脚本

```bash
run_all.bat
```

这会自动完成所有步骤。

### 方案 2: 清理重新开始

```bash
# 删除旧的RGCN嵌入
del ..\hybrid_embeddings\entity_embeddings_rgcn.pkl
del ..\hybrid_embeddings\relation_mappings_rgcn.pkl

# 重新准备
python prepare_embeddings.py

# 重新训练
python train_rgcn_hotpotqa.py
```

### 方案 3: 检查Python环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
python -c "import pickle; print('pickle: OK')"
```

确保所有包都已安装。

## 📞 调试输出

如果需要更详细的错误信息，在Python脚本中添加：

```python
import traceback

try:
    # 你的代码
    pass
except Exception as e:
    print(f"错误: {e}")
    traceback.print_exc()
```

## ✅ 成功标志

看到以下输出表示一切正常：

**prepare_embeddings.py**:
```
✅ 嵌入文件已准备好，供RGCN训练使用
实体嵌入已保存到: ../hybrid_embeddings/entity_embeddings_rgcn.pkl
  形状: (5478, 484)
```

**test_data_loader.py**:
```
✓ 数据加载器测试通过！
```

**train_rgcn_hotpotqa.py**:
```
✓ 嵌入文件检查通过
  - 实体嵌入: entity_embeddings_rgcn.pkl
  - 关系映射: relation_mappings_rgcn.pkl

开始训练...
```

## 📅 更新日期

2024-12-04

