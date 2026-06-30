# 🎉 完整幻觉检测系统 - 使用指南

## 📊 系统架构

参考你提供的架构图，我们实现了完整的幻觉检测流程：

```
数据输入
    ↓
Graphing Module (图构建)
    ├─> G_response (响应图 - GPT生成的三元组)
    └─> G_reference (参考图 - KB中的三元组)
    ↓
RGCN Layers (关系图卷积编码)
    ├─> 实体嵌入 (TransE + SentenceTransformer)
    ├─> 多关系图卷积 (3层)
    └─> 注意力池化
    ↓
相似度计算 (余弦相似度)
    ↓
二分类判断 (基于阈值)
    ↓
输出: Hallucination / Non-Hallucination
```

## 🚀 完整工作流程

### 1️⃣ 数据准备

```bash
cd "g:\小论文\第三章\GCA-main\Extract triples\dataset\hotpotqa"

# 提取实体和关系
python extract_entities_relations.py

# 提取三元组
python extract_triples.py
```

### 2️⃣ 训练TransE

```bash
# 训练TransE获取结构化嵌入
python train_transe.py --prepare_data
```

### 3️⃣ 生成混合嵌入

```bash
# 生成 TransE + SentenceTransformer 混合嵌入
python generate_hybrid_embeddings.py
```

### 4️⃣ 训练RGCN

```bash
cd rgcn

# 准备RGCN嵌入
python prepare_embeddings.py

# 训练RGCN模型
python train_rgcn_hotpotqa.py
```

### 5️⃣ 幻觉检测推理 ⭐

```bash
# 方式1: 一键推理
run_inference.bat

# 方式2: 命令行
python inference_hotpotqa.py

# 方式3: 简单示例
python example_inference.py
```

## 📁 核心文件

### 推理相关

| 文件 | 说明 | 用途 |
|------|------|------|
| `inference_hotpotqa.py` | 完整推理脚本 | 批量检测+评估+可视化 |
| `example_inference.py` | 简单示例 | 单样本检测演示 |
| `run_inference.bat` | 一键运行 | Windows快速推理 |
| `INFERENCE_GUIDE.md` | 推理指南 | 详细使用说明 |

### 输出文件

```
rgcn_output/
├── checkpoints/
│   └── best_model.pth                     # 训练好的模型
├── hallucination_predictions.json         # 预测结果 ⭐
├── evaluation_metrics.json                # 评估指标
├── confusion_matrix.png                   # 混淆矩阵
├── similarity_distribution.png            # 相似度分布
└── threshold_f1_curve.png                 # 阈值-F1曲线
```

## 🎯 使用方式

### 方式 1: 完整推理（推荐）

```bash
python inference_hotpotqa.py
```

**功能**:
- ✅ 批量预测所有数据
- ✅ 自动寻找最优阈值
- ✅ 计算评估指标
- ✅ 生成可视化图表
- ✅ 保存预测结果

**输出示例**:
```
评估结果
============================================================
阈值: 0.700
准确率: 0.8545
精确率: 0.8800
召回率: 0.8182
F1分数: 0.8480

分类报告:
                    precision    recall  f1-score   support
    Hallucination       0.82      0.89      0.85        45
Non-Hallucination       0.88      0.82      0.85        65
```

### 方式 2: 简单示例

```bash
python example_inference.py
```

**功能**:
- ✅ 单样本检测演示
- ✅ 批量检测示例
- ✅ 快速验证

**输出示例**:
```
样本信息:
  ID: 5a8b57f25542995d1e6f1371
  问题: Were Scott Derrickson and Ed Wood of the same nationality?
  答案: yes

检测结果:
  相似度: 0.8245
  阈值: 0.7000
  判断: ✅ 非幻觉
  置信度: 0.1245
```

### 方式 3: Python API

```python
from inference_hotpotqa import HallucinationDetector
from config_hotpotqa import Config

# 初始化检测器
config_dict = Config.get_config_dict()
detector = HallucinationDetector(
    model_path='./rgcn_output/checkpoints/best_model.pth',
    config_dict=config_dict
)

# 检测单个样本
similarity = detector.predict_similarity(context_graph, gpt_graph)
is_hallucination = similarity < 0.7

# 批量检测
predictions, similarities, metadata = detector.predict_batch(
    dataloader, 
    threshold=0.7
)
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
  }
]
```

**字段说明**:
- `similarity`: 图相似度 [0, 1]
- `prediction`: 0=幻觉, 1=非幻觉
- `label`: "Hallucination" 或 "Non-Hallucination"
- `confidence`: 置信度（与阈值的距离）

## 🎨 可视化输出

### 1. 混淆矩阵 (`confusion_matrix.png`)

显示预测结果与真实标签的对应关系。

### 2. 相似度分布 (`similarity_distribution.png`)

- 左图: 按真实标签的相似度分布
- 右图: 按预测标签的相似度分布
- 蓝色虚线: 判断阈值

### 3. 阈值-F1曲线 (`threshold_f1_curve.png`)

展示不同阈值下的F1分数，帮助选择最优阈值。

## ⚙️ 核心参数

### 相似度阈值

决定分类的边界：

```python
# 严格模式（高精确率）
threshold = 0.8

# 平衡模式（默认）
threshold = 0.7

# 宽松模式（高召回率）
threshold = 0.6
```

### 批大小

```python
# 在 config_hotpotqa.py 中调整
BATCH_SIZE = 8  # 根据内存调整
```

## 📊 评估指标

| 指标 | 说明 | 应用场景 |
|------|------|----------|
| **准确率** | 整体正确率 | 平衡的数据集 |
| **精确率** | 预测为非幻觉中的正确率 | 追求质量 |
| **召回率** | 实际非幻觉被识别的比例 | 追求覆盖 |
| **F1分数** | 精确率和召回率的调和平均 | 综合评价 |

## 💡 实际应用场景

### 场景 1: 批量检测历史数据

```bash
# 修改配置加载全部数据
# 在 config_hotpotqa.py 中设置
MAX_SAMPLES = None

# 运行推理
python inference_hotpotqa.py
```

### 场景 2: 在线服务API

```python
from flask import Flask, request, jsonify
from inference_hotpotqa import HallucinationDetector

app = Flask(__name__)
detector = HallucinationDetector(model_path, config_dict)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    # 构建图数据
    context_graph, gpt_graph = build_graphs(data)
    
    # 检测
    similarity = detector.predict_similarity(context_graph, gpt_graph)
    is_hallucination = similarity < 0.7
    
    return jsonify({
        'is_hallucination': bool(is_hallucination),
        'similarity': float(similarity),
        'confidence': float(abs(similarity - 0.7))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 场景 3: 集成到QA系统

```python
def answer_with_hallucination_check(question, answer, triples):
    """回答问题并检测幻觉"""
    
    # 1. 从KB检索相关三元组
    kb_triples = retrieve_from_kb(question)
    
    # 2. 构建图
    context_graph = build_graph(kb_triples)
    gpt_graph = build_graph(triples)
    
    # 3. 检测幻觉
    similarity = detector.predict_similarity(context_graph, gpt_graph)
    is_hallucination = similarity < 0.7
    
    # 4. 返回结果
    return {
        'answer': answer,
        'is_reliable': not is_hallucination,
        'confidence': similarity
    }
```

## 🔧 调优建议

### 提高准确率

1. **增加训练数据** - 更多样本，更好的泛化
2. **调整模型参数** - 尝试不同的隐藏层维度
3. **优化阈值** - 使用验证集寻找最优阈值
4. **特征工程** - 添加更多特征（如答案一致性）

### 处理不平衡数据

```python
# 在训练时使用类权重
from torch.nn import BCEWithLogitsLoss

# 计算类权重
pos_weight = torch.tensor([num_neg / num_pos])
criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
```

### 集成多个模型

```python
def ensemble_predict(models, context_graph, gpt_graph):
    """集成多个模型的预测"""
    similarities = []
    for model in models:
        sim = model.predict_similarity(context_graph, gpt_graph)
        similarities.append(sim)
    
    # 平均相似度
    avg_similarity = np.mean(similarities)
    return avg_similarity
```

## ⚠️ 注意事项

### 1. 数据标注

HotpotQA原始数据没有幻觉标注，需要：
- 人工标注部分数据
- 或使用已标注的幻觉检测数据集
- 推理脚本中使用的是模拟标签

### 2. 阈值选择

- 有标注数据时：使用 `find_optimal_threshold()` 自动优化
- 无标注数据时：根据业务需求手动设置

### 3. 模型更新

每次重新训练模型后：
- 重新评估阈值
- 更新评估指标
- 检查性能变化

## 📚 文档导航

- **推理详细指南**: `INFERENCE_GUIDE.md` ⭐
- **训练指南**: `README.md`
- **快速开始**: `QUICKSTART.md`
- **故障排除**: `TROUBLESHOOTING.md`

## 🎉 快速开始

```bash
# 1. 确保模型已训练
python train_rgcn_hotpotqa.py

# 2. 运行推理
python inference_hotpotqa.py

# 3. 查看结果
# - hallucination_predictions.json (预测结果)
# - confusion_matrix.png (混淆矩阵)
# - similarity_distribution.png (相似度分布)
```

就这么简单！🚀

现在你有了一个完整的幻觉检测系统，可以：
- ✅ 批量检测数据
- ✅ 获取二分类标签
- ✅ 评估模型性能
- ✅ 可视化分析结果
- ✅ 集成到实际应用

开始检测吧！











