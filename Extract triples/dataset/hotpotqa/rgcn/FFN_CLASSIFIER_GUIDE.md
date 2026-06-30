# 📚 FFN分类器使用指南

## 🎯 架构说明

实现论文中的二元分类架构：

```
输入: G_response (响应图), G_reference (参考图)
    ↓
RGCN编码 (3层关系图卷积)
    ├─> Context节点特征: v_1^L, v_2^L, ..., v_n^L
    └─> GPT节点特征: u_1^L, u_2^L, ..., u_m^L
    ↓
全局平均池化
    ├─> h_reference = 1/|V| * Σ v_n^L
    └─> h_response = 1/|V| * Σ u_n^L
    ↓
特征拼接
    concat[h_response, h_reference]  # 维度: 2 * out_channels
    ↓
FFN分类头 (前馈神经网络)
    Linear(2*out_channels → 128) → ReLU → Dropout
    → Linear(128 → 64) → ReLU → Dropout
    → Linear(64 → 2)
    ↓
Softmax
    [P(幻觉), P(非幻觉)]
    ↓
输出
    预测标签: argmax → 0 (幻觉) 或 1 (非幻觉)
```

## 🚀 快速开始

### 方式 1: 一键运行（推荐）

```bash
run_classifier.bat
```

### 方式 2: 分步运行

```bash
# 1. 训练分类器
python train_classifier.py

# 2. 运行推理
python inference_classifier.py
```

## 📁 核心文件

| 文件 | 说明 |
|------|------|
| `classifier_model.py` | FFN分类器模型定义 |
| `train_classifier.py` | 训练脚本 |
| `inference_classifier.py` | 推理脚本 |
| `run_classifier.bat` | 一键运行脚本 |

## 🔧 模型架构详解

### 1. RGCN编码器

```python
# 输入: 图数据 (节点ID, 边索引, 边类型)
# 输出: 节点特征 [num_nodes, out_channels]

# 第1层: embedding_dim → hidden_channels (128)
RGCNConv(embedding_dim, 128, num_relations)

# 第2层: 128 → 128
RGCNConv(128, 128, num_relations)

# 第3层: 128 → out_channels (64)
RGCNConv(128, 64, num_relations)
```

### 2. 全局平均池化

```python
# 对每个图的所有节点特征取平均
h = 1/|V| * Σ(v_n^L)

# 输入: 节点特征 [num_nodes, 64]
# 输出: 图特征 [batch_size, 64]
```

### 3. FFN分类头

```python
# 输入: concat[h_response, h_reference] [batch_size, 128]

Linear(128 → 128) + ReLU + Dropout(0.3)
Linear(128 → 64) + ReLU + Dropout(0.3)
Linear(64 → 2)  # 2分类

# 输出: logits [batch_size, 2]
# Softmax后得到概率 [P(幻觉), P(非幻觉)]
```

## ⚙️ 关键参数

### 训练参数 (config_hotpotqa.py)

```python
# RGCN参数
'hidden_channels': 128,      # RGCN隐藏层维度
'out_channels': 64,          # RGCN输出维度（图特征维度）
'num_layers': 3,             # RGCN层数

# FFN参数
'ffn_hidden_dim': 128,       # FFN隐藏层维度

# 训练参数
'batch_size': 8,             # 批大小
'learning_rate': 0.001,      # 学习率
'num_epochs': 50,            # 训练轮数
'early_stopping_patience': 10,  # 早停耐心

# 正则化
'dropout': 0.3,              # Dropout率
'weight_decay': 1e-5,        # L2正则化
'gradient_clip_norm': 1.0,   # 梯度裁剪
```

### 优化器和调度器

```python
# 优化器: Adam
optimizer = Adam(lr=0.001, weight_decay=1e-5)

# 学习率调度: CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(
    T_0=10,      # 首次重启周期
    T_mult=2,    # 周期倍增因子
    eta_min=1e-6 # 最小学习率
)
```

## 📊 输出文件

### 训练输出

```
rgcn_output/
├── checkpoints/
│   ├── best_classifier.pth           # 最佳模型（基于验证准确率）
│   └── classifier_epoch_*.pth        # 定期保存的检查点
├── classifier_training_curves.png    # 训练曲线（损失和准确率）
└── classifier_training_history.json  # 训练历史
```

### 推理输出

```
rgcn_output/
├── classifier_predictions.json            # 预测结果（每条数据的标签和概率）
├── classifier_evaluation_metrics.json     # 评估指标
├── classifier_confusion_matrix.png        # 混淆矩阵
└── classifier_probability_distribution.png # 概率分布
```

## 📄 预测结果格式

`classifier_predictions.json`:

```json
[
  {
    "_id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood...",
    "answer": "yes",
    "prediction": 1,                    # 0=幻觉, 1=非幻觉
    "label": "Non-Hallucination",
    "prob_hallucination": 0.1234,       # P(幻觉)
    "prob_non_hallucination": 0.8766,   # P(非幻觉)
    "confidence": 0.8766,               # max(prob)
    "num_context_triples": 36,
    "num_gpt_triples": 28
  }
]
```

## 📈 评估指标

```json
{
  "accuracy": 0.8545,   # 准确率
  "precision": 0.8800,  # 精确率
  "recall": 0.8182,     # 召回率
  "f1": 0.8480          # F1分数
}
```

## 💡 使用示例

### 1. 训练后查看结果

```python
import json

# 加载预测结果
with open('rgcn_output/classifier_predictions.json', 'r', encoding='utf-8') as f:
    predictions = json.load(f)

# 查看第一个样本
sample = predictions[0]
print(f"问题: {sample['question']}")
print(f"答案: {sample['answer']}")
print(f"预测: {sample['label']}")
print(f"置信度: {sample['confidence']:.4f}")
```

### 2. 使用Python API

```python
from classifier_model import HallucinationClassifier
import torch

# 加载模型
model = HallucinationClassifier(
    entity_embedding_path='../hybrid_embeddings/entity_embeddings_rgcn.pkl',
    relation_embedding_path='../hybrid_embeddings/relation_mappings_rgcn.pkl',
    hidden_channels=128,
    out_channels=64,
    num_layers=3,
    ffn_hidden_dim=128
)

# 加载权重
checkpoint = torch.load('rgcn_output/checkpoints/best_classifier.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测
with torch.no_grad():
    logits = model(response_graph, reference_graph)
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    
    print(f"预测标签: {prediction.item()}")  # 0 or 1
    print(f"P(幻觉): {probabilities[0, 0]:.4f}")
    print(f"P(非幻觉): {probabilities[0, 1]:.4f}")
```

### 3. 批量预测

```python
from inference_classifier import ClassifierInference
from config_hotpotqa import Config

# 初始化推理器
config_dict = Config.get_config_dict()
config_dict['ffn_hidden_dim'] = 128

inference = ClassifierInference(
    model_path='rgcn_output/checkpoints/best_classifier.pth',
    config_dict=config_dict
)

# 批量预测
predictions, probabilities, ground_truth, metadata = \
    inference.predict_batch(dataloader)

# 查看结果
for i in range(len(predictions)):
    pred = "非幻觉" if predictions[i] == 1 else "幻觉"
    conf = probabilities[i, predictions[i]]
    print(f"样本{i}: {pred} (置信度: {conf:.4f})")
```

## 🎨 可视化

### 1. 训练曲线

- 左图: 训练和验证损失曲线
- 右图: 训练和验证准确率曲线

### 2. 混淆矩阵

展示预测结果与真实标签的对应关系：

```
              Predicted
              Hall  Non-Hall
Actual Hall    TP      FN
       Non-H   FP      TN
```

### 3. 概率分布

- 左图: 按真实标签的概率分布
- 右图: 按预测标签的概率分布
- 红色虚线: 决策边界 (0.5)

## 🔍 与相似度方法的对比

| 特性 | 相似度方法 (旧) | FFN分类器 (新) |
|------|----------------|---------------|
| **判断方式** | 阈值判断 | 端到端学习 |
| **输出** | 相似度分数 | 二分类概率 |
| **优化** | 需手动调阈值 | 自动优化决策边界 |
| **表达能力** | 线性 | 非线性（FFN） |
| **准确率** | 较低 | 较高 |
| **可解释性** | 高（相似度直观） | 中（概率值） |

## ⚠️ 注意事项

### 1. 数据标注

当前使用的是模拟标签（基于三元组数量），实际使用时需要：
- 人工标注真实的幻觉标签
- 或使用已标注的幻觉检测数据集

### 2. 类别不平衡

如果数据不平衡，可以：

```python
# 计算类权重
from torch.nn import CrossEntropyLoss
pos_weight = num_neg / num_pos
criterion = CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight]))
```

### 3. 过拟合

防止过拟合的方法：
- ✓ Dropout (0.3)
- ✓ 权重衰减 (1e-5)
- ✓ 早停 (patience=10)
- ✓ 批归一化
- ✓ 梯度裁剪

### 4. 模型调优

提高准确率的方法：
- 增加训练数据
- 调整FFN隐藏层维度
- 调整学习率和批大小
- 尝试不同的RGCN层数
- 使用更好的嵌入

## 🆚 对比测试

建议同时运行两种方法并对比：

```bash
# 方法1: 相似度阈值
python inference_hotpotqa.py

# 方法2: FFN分类器
python inference_classifier.py

# 对比结果
# - classifier_predictions.json (FFN)
# - hallucination_predictions.json (相似度)
```

## 📚 相关文档

- **模型架构**: `classifier_model.py`
- **训练脚本**: `train_classifier.py`
- **推理脚本**: `inference_classifier.py`
- **配置文件**: `config_hotpotqa.py`
- **数据加载**: `data_loader_hotpotqa.py`

## 🎉 快速验证

```bash
# 1. 准备嵌入（如果还没做）
cd ..
python generate_hybrid_embeddings.py
cd rgcn
python prepare_embeddings.py

# 2. 训练和推理
run_classifier.bat

# 3. 查看结果
# - rgcn_output/classifier_predictions.json
# - rgcn_output/classifier_confusion_matrix.png
```

完成！现在你有了基于FFN的端到端幻觉检测分类器！🚀











