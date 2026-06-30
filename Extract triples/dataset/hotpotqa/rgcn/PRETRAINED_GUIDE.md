# 🔄 预训练编码器 + FFN分类器 使用指南

## 📋 方案概述

这是一个**迁移学习方案**，结合两种方法的优势：

1. **阶段1**: 使用Siamese RGCN进行对比学习（无监督/自监督）
2. **阶段2**: 加载预训练编码器，训练FFN分类头（监督学习）

### 优势

✅ **利用无标签数据**: Siamese RGCN通过对比学习从大量无标签数据学习表示  
✅ **快速训练**: FFN分类头参数少，训练快速  
✅ **更好的初始化**: 预训练编码器提供更好的特征表示  
✅ **灵活策略**: 可以选择冻结或微调编码器

---

## 🎯 三种训练策略对比

| 策略 | 编码器初始化 | 编码器训练 | FFN训练 | 训练时间 | 准确率 | 适用场景 |
|------|------------|-----------|---------|----------|--------|----------|
| **从头训练** | 随机初始化 | ✓ | ✓ | 长 | 中 | 有大量标注数据 |
| **预训练+冻结** | 预训练 | ✗ | ✓ | 短 | 中高 | 标注数据少，快速验证 |
| **预训练+微调** ⭐ | 预训练 | ✓ (小lr) | ✓ (大lr) | 中 | 高 | 推荐！平衡性能和速度 |

---

## 🚀 完整工作流程

### Step 1: 训练Siamese RGCN（对比学习）

```bash
# 使用对比学习训练编码器
python train_rgcn_hotpotqa.py

# 输出: rgcn_output/checkpoints/best_model.pth
```

**目的**: 学习良好的图表示（无需标注）

### Step 2: 训练FFN分类器（有监督）

#### 方式A: 冻结编码器（快速）

```bash
# 只训练FFN，编码器权重固定
python train_pretrained_classifier.py --freeze_encoder

# 输出: best_pretrained_classifier_frozen.pth
```

**特点**:
- ⚡ 训练快（只优化FFN）
- 💾 显存占用少
- 🎯 适合快速验证

#### 方式B: 微调编码器（推荐）⭐

```bash
# 编码器和FFN一起训练（编码器用较小学习率）
python train_pretrained_classifier.py

# 输出: best_pretrained_classifier_finetuned.pth
```

**特点**:
- 🎯 准确率更高
- 🔧 编码器适应分类任务
- ⚖️ 平衡性能和训练时间

---

## 📊 详细对比

### 1. 从头训练 FFN分类器

```bash
python train_classifier.py
```

**流程**:
```
随机初始化RGCN → 端到端训练 → 分类器
```

**参数量**: 全部参数从头学习

**优点**:
- ✅ 不依赖预训练
- ✅ 直接优化分类目标

**缺点**:
- ❌ 训练时间长
- ❌ 需要更多数据
- ❌ 可能过拟合

**建议**: 数据量大（10k+样本）且有充足时间

---

### 2. 预训练 + 冻结编码器

```bash
# 先训练Siamese RGCN
python train_rgcn_hotpotqa.py

# 然后冻结编码器，只训练FFN
python train_pretrained_classifier.py --freeze_encoder
```

**流程**:
```
预训练RGCN (对比学习) → 冻结 → 训练FFN → 分类器
```

**参数量**: 
- 冻结: ~500k (编码器)
- 训练: ~100k (仅FFN)

**优点**:
- ✅ 训练快速（10-15 epochs）
- ✅ 显存占用少
- ✅ 利用无标签数据
- ✅ 不会破坏预训练表示

**缺点**:
- ❌ 编码器不适应分类任务
- ❌ 准确率可能略低

**建议**: 
- 快速原型验证
- 标注数据很少（<1k样本）
- 计算资源受限

---

### 3. 预训练 + 微调编码器 ⭐ (推荐)

```bash
# 先训练Siamese RGCN
python train_rgcn_hotpotqa.py

# 然后微调整个模型
python train_pretrained_classifier.py
```

**流程**:
```
预训练RGCN (对比学习) → 微调编码器(小lr) + 训练FFN(大lr) → 分类器
```

**参数量**: 全部参数可训练
- 编码器: 学习率 × 0.1
- FFN: 学习率 × 1.0

**优点**:
- ✅ 准确率最高
- ✅ 编码器适应分类任务
- ✅ 利用预训练初始化
- ✅ 训练比从头快

**缺点**:
- ⚠️ 训练时间中等（20-30 epochs）
- ⚠️ 需要调整学习率

**建议**: 
- **生产环境部署** ⭐
- 追求最高准确率
- 有适量标注数据（1k-10k样本）

---

## 📈 预期性能对比

| 方法 | 训练时间 | 验证准确率 | F1分数 | 显存占用 |
|------|----------|-----------|--------|----------|
| 从头训练 | 3-4小时 | 85-87% | 0.84-0.86 | 高 |
| 预训练+冻结 | 30分钟-1小时 | 83-85% | 0.82-0.84 | 低 |
| 预训练+微调 ⭐ | 1.5-2.5小时 | 87-90% | 0.86-0.89 | 中 |

*基于HotpotQA数据集，GPU: RTX 3090*

---

## 💡 使用建议

### 何时使用预训练+冻结

✅ **适用场景**:
- 快速原型和概念验证
- 标注数据非常少（<500样本）
- 计算资源受限
- 需要快速迭代

❌ **不适用**:
- 追求最高准确率
- 有充足标注数据
- 编码器表示与分类任务差异大

### 何时使用预训练+微调 ⭐

✅ **适用场景**:
- **生产环境部署** (推荐)
- 追求准确率和速度的平衡
- 有适量标注数据（1k+样本）
- 希望利用无标签数据

❌ **不适用**:
- 极度资源受限
- 只想快速验证

---

## 🔧 高级用法

### 1. 调整学习率比例

```python
# 在 train_pretrained_classifier.py 中
encoder_lr = config['learning_rate'] * 0.1  # 编码器: 10%学习率
classifier_lr = config['learning_rate']     # FFN: 100%学习率

# 更激进的微调（编码器学习更多）
encoder_lr = config['learning_rate'] * 0.5  # 50%

# 更保守的微调（编码器几乎冻结）
encoder_lr = config['learning_rate'] * 0.01  # 1%
```

### 2. 分阶段训练

```bash
# 阶段1: 冻结编码器，快速训练FFN (10 epochs)
python train_pretrained_classifier.py --freeze_encoder --epochs 10

# 阶段2: 微调整个模型 (20 epochs)
# 加载阶段1的模型，继续训练
python train_pretrained_classifier.py --epochs 20
```

### 3. 渐进式解冻

```python
# 先冻结全部编码器
for param in model.encoder.parameters():
    param.requires_grad = False

# 训练5个epoch后，解冻最后一层
for param in model.encoder.convs[-1].parameters():
    param.requires_grad = True

# 再训练5个epoch后，解冻全部
for param in model.encoder.parameters():
    param.requires_grad = True
```

---

## 📝 完整示例

### 示例1: 快速验证（冻结编码器）

```bash
# 1. 准备嵌入
python prepare_embeddings.py

# 2. 训练Siamese RGCN (对比学习)
python train_rgcn_hotpotqa.py

# 3. 冻结编码器，训练FFN
python train_pretrained_classifier.py --freeze_encoder --epochs 15

# 4. 推理
python inference_classifier.py \
  --model_path rgcn_output/checkpoints/best_pretrained_classifier_frozen.pth

# 总时间: ~2小时
```

### 示例2: 最佳性能（微调编码器）⭐

```bash
# 1. 准备嵌入
python prepare_embeddings.py

# 2. 训练Siamese RGCN (对比学习)
python train_rgcn_hotpotqa.py

# 3. 微调整个模型
python train_pretrained_classifier.py --epochs 30

# 4. 推理
python inference_classifier.py \
  --model_path rgcn_output/checkpoints/best_pretrained_classifier_finetuned.pth

# 总时间: ~3-4小时
```

### 示例3: 对比所有方法

```bash
# 方法1: 从头训练
python train_classifier.py
mv rgcn_output/checkpoints/best_classifier.pth \
   rgcn_output/checkpoints/best_classifier_scratch.pth

# 方法2: 预训练+冻结
python train_rgcn_hotpotqa.py
python train_pretrained_classifier.py --freeze_encoder

# 方法3: 预训练+微调
python train_pretrained_classifier.py

# 对比三个模型的性能
```

---

## 🎯 推荐流程 ⭐

**最佳实践**:

```bash
# 步骤1: 使用对比学习训练编码器（利用所有数据）
python train_rgcn_hotpotqa.py

# 步骤2: 微调模型（使用标注数据）
python train_pretrained_classifier.py

# 步骤3: 评估
python inference_classifier.py

# 步骤4 (可选): 如果性能不够，尝试从头训练对比
python train_classifier.py
```

**理由**:
1. Siamese RGCN可以利用所有数据（无需标注）
2. 微调在预训练基础上快速收敛
3. 通常能达到最佳准确率
4. 训练时间适中

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `classifier_with_pretrained.py` | 预训练编码器+FFN模型 |
| `train_pretrained_classifier.py` | 训练脚本 |
| `train_rgcn_hotpotqa.py` | Siamese RGCN训练（预训练阶段） |
| `train_classifier.py` | 从头训练（对比用） |
| `inference_classifier.py` | 统一推理脚本 |

---

## ⚙️ 参数调优建议

### 冻结编码器模式

```python
learning_rate = 0.001    # FFN学习率可以大一些
batch_size = 16          # 显存占用小，可以用大batch
num_epochs = 15-20       # 收敛快
```

### 微调编码器模式

```python
encoder_lr = 0.0001      # 编码器学习率要小
classifier_lr = 0.001    # FFN学习率正常
batch_size = 8           # 显存占用大，用小batch
num_epochs = 25-35       # 需要更多epoch
```

---

## 🎉 总结

| 需求 | 推荐方案 |
|------|---------|
| 快速验证 | 预训练+冻结 |
| 最高准确率 | 预训练+微调 ⭐ |
| 资源受限 | 预训练+冻结 |
| 生产部署 | 预训练+微调 ⭐ |
| 数据量大 | 从头训练 或 预训练+微调 |
| 数据量小 | 预训练+冻结 |

**综合推荐**: **预训练+微调** ⭐

这个方案结合了无监督预训练和有监督微调的优势，通常能达到最佳性能！

开始训练吧！🚀











