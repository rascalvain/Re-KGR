# 🔄 两种幻觉检测方法对比

## 📋 方法概览

本项目实现了两种幻觉检测方法：

### 方法 1: 相似度阈值方法

**架构**:
```
RGCN编码 → 图嵌入 → 余弦相似度 → 阈值判断 → 二分类
```

**核心思想**: 
- 计算两个图嵌入的余弦相似度
- 如果相似度 >= 阈值 → 非幻觉
- 如果相似度 < 阈值 → 幻觉

**文件**: `inference_hotpotqa.py`

---

### 方法 2: FFN分类器方法 ⭐ (推荐)

**架构**:
```
RGCN编码 → 全局平均池化 → 拼接特征 → FFN → Softmax → 二分类概率
```

**核心思想**:
- 使用前馈神经网络学习最优决策边界
- 端到端训练，直接优化分类损失
- 输出二分类概率分布

**文件**: `train_classifier.py`, `inference_classifier.py`

---

## 📊 详细对比

| 维度 | 方法1: 相似度阈值 | 方法2: FFN分类器 ⭐ |
|------|-------------------|-------------------|
| **架构** | RGCN + 相似度 | RGCN + 池化 + FFN |
| **决策方式** | 固定阈值（如0.7） | 学习的非线性分类器 |
| **训练目标** | 对比学习（InfoNCE） | 交叉熵损失（端到端） |
| **输出** | 相似度分数 [0, 1] | 概率分布 [P(hall), P(non-hall)] |
| **决策边界** | 线性（阈值） | 非线性（FFN学习） |
| **参数调优** | 需手动调阈值 | 自动学习最优边界 |
| **表达能力** | 有限（线性） | 强大（多层非线性） |
| **准确率** | 中等 | 较高 |
| **可解释性** | 高（相似度直观） | 中（概率值） |
| **训练复杂度** | 低（仅编码器） | 中（编码器+分类器） |
| **推理速度** | 快 | 较快 |
| **适用场景** | 快速原型、探索性分析 | 生产环境、追求准确率 |

---

## 🔍 技术细节对比

### 1. 模型架构

#### 方法1: 相似度阈值

```python
# 1. RGCN编码（Siamese架构）
h_context = encoder(context_graph)    # [batch, 64]
h_gpt = encoder(gpt_graph)            # [batch, 64]

# 2. 余弦相似度
similarity = cosine_similarity(h_context, h_gpt)  # [batch]

# 3. 阈值判断
is_hallucination = similarity < threshold  # threshold = 0.7

# 问题：
# - 阈值是超参数，需要手动调优
# - 线性决策边界，表达能力有限
# - 无法学习复杂的分类模式
```

#### 方法2: FFN分类器

```python
# 1. RGCN编码 + 全局平均池化
h_context = encoder(context_graph)    # [batch, 64]
h_gpt = encoder(gpt_graph)            # [batch, 64]

# 2. 特征拼接
h_concat = torch.cat([h_context, h_gpt], dim=1)  # [batch, 128]

# 3. FFN分类
logits = FFN(h_concat)  # [batch, 2]
# FFN: Linear(128→128) → ReLU → Dropout
#      Linear(128→64) → ReLU → Dropout
#      Linear(64→2)

# 4. Softmax
probs = softmax(logits)  # [P(幻觉), P(非幻觉)]

# 优势：
# - 端到端学习最优决策边界
# - 非线性分类器，表达能力强
# - 直接优化分类目标
```

### 2. 训练目标

#### 方法1: 对比学习

```python
# 损失函数: InfoNCE
# 目标: 拉近正样本，推远负样本

loss = -log(exp(sim(h1, h2) / τ) / Σ exp(sim(h1, h_neg) / τ))

# 特点:
# - 无监督/自监督学习
# - 学习相似度表示
# - 不直接优化分类目标
```

#### 方法2: 监督分类

```python
# 损失函数: 交叉熵
# 目标: 最大化正确类别的概率

loss = CrossEntropyLoss(logits, labels)

# 特点:
# - 有监督学习
# - 直接优化分类准确率
# - 端到端训练
```

### 3. 决策边界

#### 方法1: 线性边界

```
相似度空间: [0, 1]
          ↓
      阈值=0.7
          ↓
    [0, 0.7) → 幻觉
    [0.7, 1] → 非幻觉

限制: 固定的线性切分
```

#### 方法2: 非线性边界

```
特征空间: R^128 (concat[h_context, h_gpt])
          ↓
        FFN学习
          ↓
    复杂的非线性决策边界

优势: 可以学习任意复杂的分类模式
```

---

## 📈 性能对比

### 预期性能

| 指标 | 方法1 | 方法2 |
|------|-------|-------|
| **准确率** | 75-80% | 85-90% |
| **F1分数** | 0.75-0.78 | 0.84-0.88 |
| **训练时间** | 2-3小时 | 3-4小时 |
| **推理速度** | 快 | 快 |

### 实际测试

运行两种方法并对比：

```bash
# 方法1
python train_rgcn_hotpotqa.py
python inference_hotpotqa.py

# 方法2
python train_classifier.py
python inference_classifier.py

# 对比结果文件
# - rgcn_output/hallucination_predictions.json (方法1)
# - rgcn_output/classifier_predictions.json (方法2)
```

---

## 💡 使用建议

### 何时使用方法1（相似度阈值）

✅ **适用场景**:
- 快速原型和概念验证
- 需要高可解释性（相似度分数直观）
- 探索性数据分析
- 无标注数据（自监督学习）
- 资源受限（训练更快）

❌ **不适用**:
- 追求最高准确率
- 生产环境部署
- 数据分布复杂

### 何时使用方法2（FFN分类器）⭐

✅ **适用场景**:
- 生产环境部署
- 追求高准确率和F1分数
- 有标注数据
- 数据分布复杂
- 可以接受稍长的训练时间

❌ **不适用**:
- 需要极高可解释性
- 无标注数据
- 资源非常受限

---

## 🔧 如何选择

### 决策流程

```
有标注的幻觉数据吗？
├─ 是 → 使用方法2（FFN分类器）✓
└─ 否 → 
    ├─ 只是快速验证概念？
    │   └─ 是 → 使用方法1（相似度）
    └─ 需要生产部署？
        └─ 是 → 先标注数据，然后用方法2
```

### 推荐策略 ⭐

**最佳实践**: 两种方法都试试！

```bash
# Step 1: 快速验证（方法1）
python train_rgcn_hotpotqa.py
python inference_hotpotqa.py
# → 快速了解数据，评估基线性能

# Step 2: 优化准确率（方法2）
python train_classifier.py
python inference_classifier.py
# → 在方法1的基础上提升性能

# Step 3: 对比分析
# 比较两种方法的预测结果
# 分析差异，理解模型行为
```

---

## 📝 代码示例

### 使用方法1

```python
from inference_hotpotqa import HallucinationDetector

# 初始化
detector = HallucinationDetector(model_path, config_dict)

# 预测
similarity = detector.predict_similarity(context_graph, gpt_graph)

# 判断（手动设置阈值）
threshold = 0.7
is_hallucination = similarity < threshold

print(f"相似度: {similarity:.4f}")
print(f"判断: {'幻觉' if is_hallucination else '非幻觉'}")
```

### 使用方法2

```python
from inference_classifier import ClassifierInference

# 初始化
inference = ClassifierInference(model_path, config_dict)

# 预测
prediction, probabilities = inference.predict_single(
    response_graph, reference_graph
)

print(f"预测: {'幻觉' if prediction == 0 else '非幻觉'}")
print(f"P(幻觉): {probabilities[0]:.4f}")
print(f"P(非幻觉): {probabilities[1]:.4f}")
```

---

## 🎯 总结

| 方面 | 推荐 |
|------|------|
| **快速开始** | 方法1 |
| **最高准确率** | 方法2 ⭐ |
| **可解释性** | 方法1 |
| **生产部署** | 方法2 ⭐ |
| **无标注数据** | 方法1 |
| **有标注数据** | 方法2 ⭐ |

**综合推荐**: 
- 🚀 **研究阶段**: 先用方法1快速验证
- 🏭 **生产部署**: 用方法2追求最佳性能
- 🎯 **最佳实践**: 两种方法都运行，对比分析

---

## 📚 相关文件

### 方法1文件
- `siamese_rgcn_improved.py` - Siamese RGCN模型
- `train_rgcn_hotpotqa.py` - 训练脚本
- `inference_hotpotqa.py` - 推理脚本

### 方法2文件
- `classifier_model.py` - FFN分类器模型
- `train_classifier.py` - 训练脚本
- `inference_classifier.py` - 推理脚本
- `FFN_CLASSIFIER_GUIDE.md` - 详细指南

### 通用文件
- `config_hotpotqa.py` - 配置文件
- `data_loader_hotpotqa.py` - 数据加载器
- `prepare_embeddings.py` - 嵌入准备

---

## 🚀 快速上手

```bash
# 方法1（相似度）
python train_rgcn_hotpotqa.py
python inference_hotpotqa.py

# 方法2（FFN分类器）⭐
python train_classifier.py
python inference_classifier.py

# 或使用一键脚本
run_all.bat          # 方法1
run_classifier.bat   # 方法2
```

选择适合你的方法，开始检测幻觉吧！🎉











