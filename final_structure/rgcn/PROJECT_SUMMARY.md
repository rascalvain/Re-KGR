# HotpotQA RGCN 项目总结

## 📦 项目概述

基于 new_rgcn 代码实现的 HotpotQA 数据集适配版本，使用孪生 R-GCN 学习知识图谱嵌入。

## 🎯 核心功能

1. **数据适配**: 将 HotpotQA 的 `context_triples` 和 `gpt_sentence_triples` 转换为图数据
2. **混合嵌入**: 使用 TransE + SentenceTransformer 的混合嵌入作为初始节点特征
3. **R-GCN编码**: 多关系图卷积网络学习图表示
4. **孪生架构**: 对比学习框架，学习图之间的相似性

## 📂 文件说明

### 配置文件
- **config_hotpotqa.py** (222行)
  - 所有超参数配置
  - 路径配置
  - 默认参数设置

### 数据处理
- **data_loader_hotpotqa.py** (331行)
  - HotpotQA数据加载器
  - 三元组解析
  - 图数据转换
  - PyG Data对象构建

### 嵌入准备
- **prepare_embeddings.py** (139行)
  - 转换混合嵌入格式
  - 生成RGCN所需的嵌入文件
  - 实体嵌入矩阵构建
  - 关系映射准备

### 模型
- **siamese_rgcn_improved.py** (274行)
  - 孪生R-GCN模型
  - 多层R-GCN卷积
  - 注意力池化
  - 对比学习损失

### 训练
- **train_rgcn_hotpotqa.py** (308行)
  - 完整训练流程
  - 早停机制
  - 学习率调度
  - 模型保存和加载
  - 训练曲线绘制

### 测试
- **test_data_loader.py** (164行)
  - 数据加载器测试
  - 样本检查
  - 批处理测试
  - 统计信息

### 文档
- **README.md** - 完整文档
- **QUICKSTART.md** - 快速开始指南
- **PROJECT_SUMMARY.md** - 本文件

### 脚本
- **run_all.bat** - 一键运行脚本

## 🔄 完整工作流程

```
1. 数据准备
   └─> hotpot_dev_with_triples_aligned.json

2. 实体和关系提取
   └─> entity2id.txt + relation2id.txt

3. 三元组提取
   └─> triples.txt

4. TransE训练
   └─> output/ent_embeddings.pkl + rel_embeddings.pkl

5. 混合嵌入生成
   └─> hybrid_embeddings/entity_hybrid_embeddings.pkl
       hybrid_embeddings/relation_hybrid_embeddings.pkl
       hybrid_embeddings/entity2idx.pkl
       hybrid_embeddings/relation2idx.pkl

6. RGCN嵌入准备
   └─> hybrid_embeddings/entity_embeddings_rgcn.pkl
       hybrid_embeddings/relation_mappings_rgcn.pkl

7. RGCN训练
   └─> rgcn_output/checkpoints/best_model.pth
       rgcn_output/training_curves.png
       rgcn_output/training_history.json
```

## 📊 数据统计

### HotpotQA数据集
- **样本数**: 110
- **有效样本**: ~110 (具有context_triples和gpt_sentence_triples)
- **平均Context节点数**: ~40
- **平均GPT节点数**: ~30

### 嵌入信息
- **实体数**: 5,478
- **关系数**: 1,856
- **嵌入维度**: 484 (100 TransE + 384 SentenceTransformer)

### 模型参数
- **隐藏层维度**: 128
- **输出维度**: 64
- **R-GCN层数**: 3
- **总参数量**: ~500K

## 🎮 使用指南

### 快速开始

```bash
# 一键运行
cd rgcn
run_all.bat
```

### 分步运行

```bash
# 1. 准备嵌入
python prepare_embeddings.py

# 2. 测试数据
python test_data_loader.py

# 3. 训练模型
python train_rgcn_hotpotqa.py
```

### 使用训练好的模型

```python
import torch
from siamese_rgcn_improved import SiameseRGCNWithEmbedding

# 加载模型
model = SiameseRGCNWithEmbedding(...)
checkpoint = torch.load('rgcn_output/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
context_emb, gpt_emb = model(context_graph, gpt_graph)
similarity = torch.cosine_similarity(context_emb, gpt_emb)
```

## ⚙️ 配置选项

### 模型配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| HIDDEN_CHANNELS | 128 | 隐藏层维度 |
| OUT_CHANNELS | 64 | 输出维度 |
| NUM_LAYERS | 3 | R-GCN层数 |
| DROPOUT | 0.3 | Dropout率 |

### 训练配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| BATCH_SIZE | 8 | 批大小 |
| NUM_EPOCHS | 100 | 训练轮数 |
| LEARNING_RATE | 1e-3 | 学习率 |
| EARLY_STOPPING_PATIENCE | 20 | 早停耐心 |

## 📈 性能

### 训练时间
- **CPU**: 10-30分钟 (100 epochs)
- **GPU**: 2-5分钟 (100 epochs)

### 内存使用
- **CPU**: ~2-4 GB
- **GPU**: ~1-2 GB

### 模型大小
- **检查点**: ~2 MB
- **嵌入文件**: ~20 MB

## 🔧 优化建议

### 提速
1. ✅ 使用GPU
2. ✅ 增大batch size
3. ✅ 减少层数
4. ✅ 冻结嵌入（默认）

### 提升效果
1. ✅ 增加训练轮数
2. ✅ 调整学习率
3. ✅ 尝试不同的margin值
4. ✅ 使用数据增强

### 节省内存
1. ✅ 减小batch size
2. ✅ 减小隐藏层维度
3. ✅ 减少层数

## ⚠️ 注意事项

### 数据依赖
- 必须先运行 `generate_hybrid_embeddings.py`
- 嵌入文件必须存在
- 数据文件格式必须正确

### 训练注意
- 数据集较小（110样本），容易过拟合
- 建议使用较大的dropout
- 早停很重要
- 监控验证损失

### GPU使用
- 自动检测GPU
- 如果CUDA out of memory，减小batch size
- CPU训练也可以，只是慢一些

## 🐛 常见问题

### Q1: 嵌入文件不存在
```bash
cd ..
python generate_hybrid_embeddings.py
cd rgcn
python prepare_embeddings.py
```

### Q2: 数据加载失败
```bash
python test_data_loader.py  # 检查问题
```

### Q3: 训练不收敛
- 减小学习率
- 检查数据
- 增加训练轮数

### Q4: 内存不足
- 减小BATCH_SIZE
- 减小HIDDEN_CHANNELS
- 使用CPU

## 📚 参考资料

### 原始实现
- **new_rgcn**: `../../Graph-based Contextual Consistency Comparison/new_rgcn/`

### 相关文档
- **混合嵌入**: `../README_Hybrid_Embeddings.md`
- **TransE训练**: `../README_TransE_Updated.md`
- **完整流程**: `../WORKFLOW_SUMMARY.md`

### 论文
- [R-GCN: Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
- [TransE: Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)

## ✅ 验证清单

训练前：
- [ ] 混合嵌入已生成
- [ ] RGCN嵌入已准备
- [ ] 数据加载器测试通过

训练后：
- [ ] 训练曲线已生成
- [ ] 最佳模型已保存
- [ ] 验证损失合理

使用前：
- [ ] 模型可以加载
- [ ] 可以进行推理
- [ ] 结果符合预期

## 🎓 技术亮点

1. **混合嵌入**: TransE捕获结构 + SentenceTransformer捕获语义
2. **OOV处理**: Zero vector + Sentence embedding确保所有实体都有嵌入
3. **R-GCN**: 多关系图卷积，自然处理知识图谱
4. **孪生架构**: 对比学习，学习图相似性
5. **端到端**: 从数据到模型的完整流程

## 📅 项目信息

- **创建日期**: 2024-12-04
- **版本**: 1.0
- **基于**: new_rgcn 实现
- **适配**: HotpotQA 数据集

## 🎉 总结

现在你有了：
- ✅ 完整的数据处理流程
- ✅ 可训练的RGCN模型
- ✅ 详细的文档和示例
- ✅ 一键运行脚本
- ✅ 测试和验证工具

可以开始训练和使用RGCN模型了！🚀

