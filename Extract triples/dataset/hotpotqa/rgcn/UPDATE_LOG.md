# 更新日志

## 2024-12-04 - 版本 1.1

### 🐛 Bug修复

#### 1. 修复 `AttributeError: 'tuple' object has no attribute 'backward'`

**问题**: 
训练时报错，损失函数返回元组但代码试图直接调用 `.backward()`

**原因**: 
`ImprovedContrastiveLoss.forward()` 返回 `(loss, loss_dict)` 元组，包含损失值和详细信息字典。

**修复**:
- 更新 `train_epoch()` 方法，正确解包损失元组
- 更新 `validate()` 方法，正确解包损失元组
- 改进进度条显示，添加相似度信息

**修改文件**:
- `train_rgcn_hotpotqa.py`

**修复前**:
```python
loss = self.criterion(context_emb, gpt_emb, labels)
loss.backward()
```

**修复后**:
```python
loss, loss_dict = self.criterion(context_emb, gpt_emb, labels)
loss.backward()
```

#### 2. 修复 `KeyError: 'embeddings'`

**问题**: 
模型加载嵌入时找不到 'embeddings' 键

**原因**: 
混合嵌入格式（字典）与RGCN需要的格式（矩阵）不匹配

**修复**:
- 添加 `prepare_embeddings.py` 转换脚本
- 更新 `config_hotpotqa.py` 使用正确的文件路径
- 添加详细的错误提示

**修改文件**:
- `config_hotpotqa.py`
- `train_rgcn_hotpotqa.py`
- `prepare_embeddings.py`

### 📚 文档更新

#### 新增文档
- `TROUBLESHOOTING.md` - 详细的故障排除指南
- `FIX_README.md` - 快速修复说明
- `UPDATE_LOG.md` (本文件) - 更新日志

#### 更新文档
- `README.md` - 更新使用说明
- `QUICKSTART.md` - 更新快速开始步骤

### ✨ 改进

1. **更好的错误提示**
   - 添加详细的错误信息
   - 提供具体的修复步骤
   - 指向相关文档

2. **改进的训练输出**
   - 进度条显示损失和相似度
   - 更详细的日志信息
   - 更清晰的文件检查提示

3. **完善的文档**
   - 添加故障排除指南
   - 添加快速修复说明
   - 更新使用流程

### 📋 已知问题

无

### 🔄 迁移指南

如果你使用的是旧版本，请按以下步骤更新：

1. **备份你的工作**（如果有修改）

2. **获取最新代码**
   - 更新 `train_rgcn_hotpotqa.py`
   - 更新 `config_hotpotqa.py`

3. **重新准备嵌入**（如果之前没运行过）
   ```bash
   python prepare_embeddings.py
   ```

4. **重新训练**
   ```bash
   python train_rgcn_hotpotqa.py
   ```

### ✅ 测试

所有功能已测试：
- ✓ 嵌入文件加载
- ✓ 数据加载器
- ✓ 模型初始化
- ✓ 训练循环
- ✓ 验证循环
- ✓ 检查点保存

### 🙏 致谢

感谢用户报告的问题和反馈！

---

## 2024-12-04 - 版本 1.0

### 🎉 初始发布

- 实现基于 new_rgcn 的 HotpotQA 适配
- 数据加载器
- RGCN 模型
- 训练脚本
- 完整文档

### 核心功能

1. **数据适配**
   - HotpotQA JSON 解析
   - 三元组提取
   - 图数据转换

2. **混合嵌入**
   - TransE + SentenceTransformer
   - OOV 处理
   - 484维嵌入

3. **R-GCN 模型**
   - 多关系图卷积
   - 孪生网络架构
   - 对比学习损失

4. **训练框架**
   - 批处理训练
   - 早停机制
   - 学习率调度
   - GPU 支持

### 文档

- README.md - 完整文档
- QUICKSTART.md - 快速开始
- PROJECT_SUMMARY.md - 项目总结

---

## 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| 1.1 | 2024-12-04 | 修复损失函数和嵌入加载错误 |
| 1.0 | 2024-12-04 | 初始发布 |











