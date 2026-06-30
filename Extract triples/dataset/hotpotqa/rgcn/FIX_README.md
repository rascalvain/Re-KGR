# 🔧 错误修复说明

## 问题描述

运行 `train_rgcn_hotpotqa.py` 时遇到错误：
```
KeyError: 'embeddings'
```

## 问题原因

RGCN模型需要特定格式的嵌入文件，而不能直接使用混合嵌入文件。需要先运行 `prepare_embeddings.py` 将混合嵌入转换为RGCN需要的格式。

## ✅ 解决方案

### 快速修复（推荐）

直接运行一键脚本，会自动完成所有步骤：

```bash
cd rgcn
run_all.bat
```

### 手动修复

按以下步骤操作：

#### 步骤 1: 检查混合嵌入是否存在

```bash
# Windows
dir ..\hybrid_embeddings\entity_hybrid_embeddings.pkl

# Linux/Mac
ls -la ../hybrid_embeddings/entity_hybrid_embeddings.pkl
```

如果不存在，先生成：
```bash
cd ..
python generate_hybrid_embeddings.py
cd rgcn
```

#### 步骤 2: 准备RGCN嵌入

**这是关键步骤！**

```bash
python prepare_embeddings.py
```

预期输出：
```
准备RGCN嵌入文件
========================================
[1] 处理实体嵌入...
  已加载 5478 个实体的混合嵌入
  实体嵌入已保存到: ../hybrid_embeddings/entity_embeddings_rgcn.pkl
  形状: (5478, 484)

[2] 处理关系嵌入...
  关系映射已保存到: ../hybrid_embeddings/relation_mappings_rgcn.pkl
  关系数: 1856

✅ 嵌入文件已准备好，供RGCN训练使用
```

#### 步骤 3: 验证文件生成

```bash
# Windows
dir ..\hybrid_embeddings\entity_embeddings_rgcn.pkl
dir ..\hybrid_embeddings\relation_mappings_rgcn.pkl

# Linux/Mac
ls -la ../hybrid_embeddings/entity_embeddings_rgcn.pkl
ls -la ../hybrid_embeddings/relation_mappings_rgcn.pkl
```

两个文件都应该存在。

#### 步骤 4: 测试数据加载（可选但推荐）

```bash
python test_data_loader.py
```

应该看到：
```
✓ 数据加载器测试通过！
```

#### 步骤 5: 开始训练

```bash
python train_rgcn_hotpotqa.py
```

应该看到：
```
✓ 嵌入文件检查通过
  - 实体嵌入: entity_embeddings_rgcn.pkl
  - 关系映射: relation_mappings_rgcn.pkl

开始训练...
Epoch 1/100
  训练损失: 0.3456
  验证损失: 0.2987
  ...
```

## 📋 文件格式说明

### 混合嵌入格式（generate_hybrid_embeddings.py 生成）

```python
# entity_hybrid_embeddings.pkl
{
    "entity_name_1": numpy.array([...]),  # 484维向量
    "entity_name_2": numpy.array([...]),
    ...
}
```

### RGCN嵌入格式（prepare_embeddings.py 生成）

```python
# entity_embeddings_rgcn.pkl
{
    'embeddings': numpy.array([[...], [...]]),  # (num_entities, 484)矩阵
    'num_entities': 5478,
    'embedding_dim': 484,
    'entity2id': {...}
}
```

这就是为什么需要转换！

## 🔄 完整流程图

```
1. generate_hybrid_embeddings.py
   └─> entity_hybrid_embeddings.pkl (字典格式)
       relation_hybrid_embeddings.pkl
       entity2idx.pkl
       relation2idx.pkl

2. prepare_embeddings.py (转换格式)  ← ⚠️ 必须运行这一步！
   └─> entity_embeddings_rgcn.pkl (矩阵格式)
       relation_mappings_rgcn.pkl

3. train_rgcn_hotpotqa.py (训练)
   └─> 读取RGCN格式的嵌入文件
       训练模型
```

## ⚠️ 注意事项

1. **必须先运行 prepare_embeddings.py**
   - 这不是可选的，是必需的！
   - 它将字典格式转换为矩阵格式

2. **文件命名很重要**
   - `entity_embeddings_rgcn.pkl` ← RGCN训练用
   - `entity_hybrid_embeddings.pkl` ← 原始混合嵌入

3. **配置文件已更新**
   - 现在使用 `ENTITY_EMBEDDING_RGCN_PATH`
   - 指向正确的文件

## 🆘 还是不行？

查看详细的故障排除指南：
```bash
# 打开文档
TROUBLESHOOTING.md
```

或者运行完整的检查：
```bash
# 清理并重新开始
python prepare_embeddings.py
python test_data_loader.py
python train_rgcn_hotpotqa.py
```

## ✅ 成功标志

当你看到以下输出时，说明问题已解决：

```
加载实体嵌入: ../hybrid_embeddings/entity_embeddings_rgcn.pkl
  实体数: 5478, 嵌入维度: 484
加载关系映射: ../hybrid_embeddings/relation_mappings_rgcn.pkl
  关系数: 1856
初始化孪生R-GCN模型...
  嵌入层冻结: True

开始训练...
```

## 📅 更新日期

2024-12-04

## 💡 记住

**每次生成新的混合嵌入后，都要重新运行 `prepare_embeddings.py`！**











