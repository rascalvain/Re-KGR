# 幻觉检测推理指南

基于训练好的RGCN模型进行幻觉检测，输出二分类标签。

## 🎯 核心思路

根据你提供的架构图：

```
Graphing Module (图嵌入)
    ├─> G_response (响应图)
    └─> G_reference (参考图/KB图)
         ↓
    RGAT Layers (图卷积编码)
         ↓
    FFNs (前馈网络)
         ↓
    Logit → [0.08|0.92] → Hallucinated/Non-Hallucinated
```

我们实现了：
1. **图嵌入**: 使用RGCN编码图
2. **相似度计算**: 余弦相似度
3. **二分类判断**: 基于阈值分类

## 📊 使用流程

### 步骤 1: 训练模型（如果还没训练）

```bash
python train_rgcn_hotpotqa.py
```

### 步骤 2: 运行推理

**方式 1: 完整推理和评估**
```bash
python inference_hotpotqa.py
```

**方式 2: 简单示例**
```bash
python example_inference.py
```

## 🔧 推理脚本功能

### inference_hotpotqa.py

完整的推理和评估流程：

1. **加载训练好的模型**
2. **批量预测** - 对所有数据进行预测
3. **阈值优化** - 寻找最优判断阈值
4. **性能评估** - 计算准确率、F1等指标
5. **可视化** - 生成混淆矩阵和相似度分布图
6. **保存结果** - 输出JSON格式的预测结果

### 输出文件

```
rgcn_output/
├── hallucination_predictions.json    # 每条数据的预测结果
├── evaluation_metrics.json           # 评估指标
├── confusion_matrix.png              # 混淆矩阵
├── similarity_distribution.png       # 相似度分布
└── threshold_f1_curve.png            # 阈值-F1曲线
```

## 📄 预测结果格式

`hallucination_predictions.json`:

```json
[
  {
    "_id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "similarity": 0.8245,
    "prediction": 1,
    "label": "Non-Hallucination",
    "confidence": 0.1245,
    "num_context_triples": 36,
    "num_gpt_triples": 28
  },
  ...
]
```

字段说明：
- `similarity`: 图相似度 [0, 1]
- `prediction`: 0=幻觉, 1=非幻觉
- `label`: 文本标签
- `confidence`: 置信度（与阈值的距离）

## 🎯 核心代码

### 1. 加载检测器

```python
from inference_hotpotqa import HallucinationDetector
from config_hotpotqa import Config

config_dict = Config.get_config_dict()
detector = HallucinationDetector(
    model_path='./rgcn_output/checkpoints/best_model.pth',
    config_dict=config_dict
)
```

### 2. 单样本检测

```python
# 获取样本
context_graph, gpt_graph, _, metadata = dataset[0]

# 预测相似度
similarity = detector.predict_similarity(context_graph, gpt_graph)

# 判断（基于阈值）
threshold = 0.7
is_hallucination = similarity < threshold

print(f"相似度: {similarity:.4f}")
print(f"判断: {'幻觉' if is_hallucination else '非幻觉'}")
```

### 3. 批量检测

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# 批量预测
predictions, similarities, metadata_list = detector.predict_batch(
    dataloader, 
    threshold=0.7
)

# predictions: [0, 1, 1, 0, ...] (0=幻觉, 1=非幻觉)
# similarities: [0.65, 0.82, 0.91, 0.48, ...]
```

### 4. 评估性能（如果有真实标签）

```python
true_labels = [...]  # 真实标签

metrics = detector.evaluate(
    dataloader, 
    true_labels, 
    threshold=0.7
)

print(f"准确率: {metrics['accuracy']:.4f}")
print(f"F1分数: {metrics['f1']:.4f}")
```

### 5. 寻找最优阈值

```python
best_threshold, best_f1 = detector.find_optimal_threshold(
    dataloader, 
    true_labels
)

print(f"最优阈值: {best_threshold:.3f}")
print(f"最优F1: {best_f1:.4f}")
```

## ⚙️ 阈值调整

阈值决定了分类的边界：

- **高阈值 (0.8-0.9)**: 更严格，倾向于判断为幻觉
  - 高精确率，低召回率
  - 宁可错杀，不可放过

- **低阈值 (0.5-0.6)**: 更宽松，倾向于判断为非幻觉
  - 低精确率，高召回率
  - 宁可放过，不可错杀

- **平衡阈值 (0.7)**: 默认值，平衡精确率和召回率

```python
# 自定义阈值
predictions, _, _ = detector.predict_batch(dataloader, threshold=0.75)
```

## 📈 评估指标

推理脚本会计算：

1. **准确率 (Accuracy)**: 正确分类的比例
2. **精确率 (Precision)**: 预测为非幻觉中实际为非幻觉的比例
3. **召回率 (Recall)**: 实际为非幻觉中被正确识别的比例
4. **F1分数**: 精确率和召回率的调和平均
5. **混淆矩阵**: 详细的分类结果

示例输出：
```
评估结果
============================================================
阈值: 0.700
准确率: 0.8545
精确率: 0.8800
召回率: 0.8182
F1分数: 0.8480

平均相似度: 0.7245 ± 0.1234

分类报告:
                    precision    recall  f1-score   support
    Hallucination       0.82      0.89      0.85        45
Non-Hallucination       0.88      0.82      0.85        65
         accuracy                           0.85       110
```

## 🎨 可视化

### 1. 混淆矩阵

显示真实标签与预测标签的对应关系。

### 2. 相似度分布

- 按真实标签的相似度分布
- 按预测标签的相似度分布
- 阈值线

### 3. 阈值-F1曲线

不同阈值下的F1分数，帮助选择最优阈值。

## 💡 实际应用示例

### 场景 1: 批量检测新数据

```python
# 加载新数据
new_dataset = HotpotQAGraphDataset(
    'new_data.json',
    entity_mapping_path,
    relation_mapping_path
)

new_dataloader = DataLoader(new_dataset, batch_size=8, collate_fn=collate_fn)

# 预测
detector.predict_and_save(
    new_dataloader,
    output_file='new_predictions.json',
    threshold=0.7
)
```

### 场景 2: 实时检测单条

```python
# 获取单条数据
sample = dataset[i]

# 检测
similarity = detector.predict_similarity(sample[0], sample[1])
is_hallucination = similarity < 0.7

# 返回结果
result = {
    'similarity': similarity,
    'is_hallucination': is_hallucination,
    'confidence': abs(similarity - 0.7)
}
```

### 场景 3: API服务

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
detector = HallucinationDetector(model_path, config_dict)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    # 处理数据...
    context_graph, gpt_graph = process_data(data)
    
    similarity = detector.predict_similarity(context_graph, gpt_graph)
    is_hallucination = similarity < 0.7
    
    return jsonify({
        'is_hallucination': bool(is_hallucination),
        'similarity': float(similarity),
        'confidence': float(abs(similarity - 0.7))
    })
```

## ⚠️ 注意事项

### 1. 真实标签

HotpotQA原始数据没有幻觉标注，需要：
- 人工标注部分数据作为测试集
- 或使用其他有标注的幻觉检测数据集

### 2. 阈值选择

- 如果有标注数据，使用 `find_optimal_threshold()` 自动寻找
- 如果没有标注数据，根据业务需求选择（默认0.7）

### 3. 模型更新

每次重新训练后，需要重新评估阈值。

## 🔍 常见问题

### Q1: 如何提高检测准确率？

1. 增加训练数据
2. 调整模型超参数
3. 使用更好的图嵌入
4. 集成多个模型

### Q2: 相似度都很高怎么办？

- 检查数据质量
- 调整阈值
- 使用更多的特征

### Q3: 如何处理边界情况？

相似度接近阈值的样本：
- 使用 `confidence` 字段识别低置信度样本
- 人工复核这些样本
- 使用集成方法

## 📚 相关文档

- **训练指南**: `README.md`
- **配置说明**: `config_hotpotqa.py`
- **故障排除**: `TROUBLESHOOTING.md`

## 🎉 快速开始

```bash
# 1. 训练模型（如果还没训练）
python train_rgcn_hotpotqa.py

# 2. 运行推理
python inference_hotpotqa.py

# 3. 查看结果
cat rgcn_output/hallucination_predictions.json
```

就这么简单！🚀











