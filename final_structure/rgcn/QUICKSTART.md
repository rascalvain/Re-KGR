# HotpotQA RGCN 快速开始

## 🎯 5分钟上手

### 前提条件

✅ 已运行 `generate_hybrid_embeddings.py` 生成混合嵌入

### 方式 1: 一键运行（最简单）

```bash
cd rgcn
run_all.bat
```

这会自动完成：
1. 准备嵌入文件
2. 测试数据加载器
3. 训练RGCN模型

### 方式 2: 分步运行

```bash
cd rgcn

# 步骤1: 准备嵌入
python prepare_embeddings.py

# 步骤2: 测试数据（可选）
python test_data_loader.py

# 步骤3: 训练模型
python train_rgcn_hotpotqa.py
```

## 📊 预期输出

### 准备嵌入

```
实体嵌入已保存到: ../hybrid_embeddings/entity_embeddings_rgcn.pkl
  形状: (5478, 484)
关系映射已保存到: ../hybrid_embeddings/relation_mappings_rgcn.pkl
  关系数: 1856
```

### 训练输出

```
Epoch 1/100
  训练损失: 0.3456
  验证损失: 0.2987
  学习率: 0.001000
  ✓ 保存最佳模型 (val_loss: 0.2987)

...

训练完成
最佳验证损失: 0.1234
```

### 生成文件

```
rgcn_output/
├── checkpoints/
│   └── best_model.pth              # 最佳模型 ⭐
├── training_curves.png             # 训练曲线
└── training_history.json           # 训练历史
```

## ⚙️ 调整参数

编辑 `config_hotpotqa.py`:

```python
# 快速测试
BATCH_SIZE = 4
NUM_EPOCHS = 10

# 标准训练
BATCH_SIZE = 8
NUM_EPOCHS = 100

# 高质量训练
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
```

## 🔧 常见问题

### Q1: "嵌入文件不存在"

**解决**:
```bash
cd ..
python generate_hybrid_embeddings.py
cd rgcn
python prepare_embeddings.py
```

### Q2: CUDA out of memory

**解决**: 编辑 `config_hotpotqa.py`
```python
BATCH_SIZE = 4  # 或更小
HIDDEN_CHANNELS = 64  # 减小
```

### Q3: 数据加载失败

**检查**:
```bash
python test_data_loader.py
```

## 📈 下一步

训练完成后：

1. **查看训练曲线**: `rgcn_output/training_curves.png`
2. **加载模型使用**: 参考 `README.md` 中的使用示例
3. **调整超参数**: 优化模型性能

## 📚 完整文档

详细说明请参考：
- **完整文档**: `README.md`
- **配置说明**: `config_hotpotqa.py`
- **使用示例**: `README.md` 的使用示例部分

## ⏱️ 预计时间

- **准备嵌入**: < 1 分钟
- **测试数据**: < 1 分钟
- **训练模型**: 
  - CPU: 10-30 分钟
  - GPU: 2-5 分钟

## ✅ 成功标志

看到以下输出表示成功：

```
✓ 训练完成！

生成的文件:
  最佳模型: rgcn_output/checkpoints/best_model.pth
  训练曲线: rgcn_output/training_curves.png
  训练历史: rgcn_output/training_history.json
```

现在可以使用训练好的RGCN模型了！🎉

