# 更新说明 - 使用本地 sentence-bert 模型

## 📅 更新日期
2024-12-04

## 🎯 更新内容

已将 `generate_hybrid_embeddings.py` 修改为默认使用本地 sentence-bert 模型。

## ✅ 修改文件

### 1. generate_hybrid_embeddings.py
- ✅ 默认模型路径改为：`../../../sentence-bert`
- ✅ 添加本地路径检测逻辑
- ✅ 显示模型绝对路径（方便调试）

### 2. run_generate_hybrid.bat
- ✅ 更新为使用本地模型路径

### 3. 新增文件

#### test_sentence_bert.py
测试本地 sentence-bert 模型的工具，检查：
- 模型路径是否存在
- 模型文件是否完整
- 能否成功加载
- 编码功能是否正常

#### LOCAL_MODEL_USAGE.md
详细的本地模型使用文档，包含：
- 模型位置和路径说明
- 验证步骤
- 使用示例
- 常见问题解答

#### UPDATES.md（本文件）
更新说明文档

### 4. 更新文件

#### README_Hybrid_Embeddings.md
- ✅ 添加本地模型测试步骤
- ✅ 更新默认参数说明
- ✅ 添加本地 vs 在线模型对比

## 🚀 快速使用

### 步骤 1: 测试本地模型（推荐）

```bash
python test_sentence_bert.py
```

**预期输出**:
```
========================================
测试本地 sentence-bert 模型
========================================

[测试 1] 检查模型路径
  相对路径: ../../../sentence-bert
  绝对路径: g:\小论文\第三章\GCA-main\sentence-bert
  ✓ 模型路径存在

[测试 2] 检查模型文件
  ✓ config.json (612 B)
  ✓ pytorch_model.bin (87.0 MB)
  ✓ vocab.txt (226.0 KB)
  ✓ tokenizer_config.json (350 B)
  ✓ modules.json (349 B)

[测试 3] 加载 SentenceTransformer 模型
  正在加载模型...
  ✓ 模型加载成功

[测试 4] 测试编码功能
  ✓ 编码成功
  输入文本数: 2
  嵌入形状: (2, 384)
  嵌入维度: 384

========================================
测试完成
========================================
✓ 本地 sentence-bert 模型可以正常使用！
```

### 步骤 2: 生成混合嵌入

**方式 1: 一键运行**
```bash
run_generate_hybrid.bat
```

**方式 2: 命令行运行**
```bash
python generate_hybrid_embeddings.py
```

**预期输出**:
```
正在加载 SentenceTransformer 模型: ../../../sentence-bert
  使用本地模型: g:\小论文\第三章\GCA-main\sentence-bert
正在生成 5478 个文本的嵌入...
  SentenceTransformer 嵌入维度: 384
```

## 📊 模型信息

### 本地模型路径
- **绝对路径**: `g:\小论文\第三章\GCA-main\sentence-bert\`
- **相对路径**: `../../../sentence-bert`（从 hotpotqa/ 目录）

### 模型规格
- **大小**: 约 87 MB（主模型文件）
- **嵌入维度**: 384
- **词汇表**: 30,523 个词
- **类型**: 基于 BERT 的句子嵌入模型

### 混合嵌入维度
- TransE: 100 维
- SentenceTransformer: 384 维
- **总计: 484 维**

## 🎯 优势

### 使用本地模型的好处

1. ✅ **无需网络**: 完全离线可用
2. ✅ **加载快速**: 无需下载，立即使用
3. ✅ **结果稳定**: 避免网络问题
4. ✅ **版本固定**: 确保可重现性
5. ✅ **节省时间**: 首次运行无需等待下载

## 🔧 配置说明

### 默认配置
```python
--sentence_model ../../../sentence-bert  # 默认使用本地模型
--batch_size 32                          # 可根据内存调整
```

### 使用其他模型
```bash
# 使用其他本地模型
python generate_hybrid_embeddings.py --sentence_model /path/to/model

# 使用在线模型（需要网络）
python generate_hybrid_embeddings.py --sentence_model sentence-transformers/all-mpnet-base-v2
```

## ⚠️ 注意事项

### 1. 工作目录
确保在正确的目录下运行：
```bash
cd "g:\小论文\第三章\GCA-main\Extract triples\dataset\hotpotqa"
```

### 2. 模型完整性
如果测试失败，检查以下文件：
- `pytorch_model.bin` (87 MB) - 必须存在
- `config.json` - 必须存在
- `vocab.txt` (226 KB) - 必须存在
- `tokenizer_config.json` - 必须存在
- `modules.json` - 必须存在

### 3. 路径问题
如果相对路径不work，使用绝对路径：
```bash
python generate_hybrid_embeddings.py --sentence_model "g:\小论文\第三章\GCA-main\sentence-bert"
```

## 📚 相关文档

1. **本地模型使用**: `LOCAL_MODEL_USAGE.md` ⭐
2. **混合嵌入详解**: `README_Hybrid_Embeddings.md`
3. **完整工作流程**: `WORKFLOW_SUMMARY.md`
4. **快速开始**: `QUICKSTART.md`

## 🔄 回退到在线模型

如果需要使用在线模型：

```bash
python generate_hybrid_embeddings.py \
    --sentence_model sentence-transformers/all-MiniLM-L6-v2
```

## ✅ 验证更新

运行以下命令验证更新是否成功：

```bash
# 1. 测试本地模型
python test_sentence_bert.py

# 2. 查看默认参数
python generate_hybrid_embeddings.py --help | grep sentence_model

# 预期输出：
# --sentence_model ../../../sentence-bert
```

## 📞 问题反馈

如果遇到问题：

1. 运行测试脚本：`python test_sentence_bert.py`
2. 查看详细文档：`LOCAL_MODEL_USAGE.md`
3. 检查模型文件是否完整
4. 尝试使用绝对路径

## 🎉 总结

现在你可以：
- ✅ 直接运行 `python generate_hybrid_embeddings.py`
- ✅ 无需网络连接
- ✅ 快速生成混合嵌入
- ✅ 结果稳定可重现

开始使用：
```bash
python test_sentence_bert.py  # 测试模型
python generate_hybrid_embeddings.py  # 生成嵌入
python use_hybrid_embeddings.py  # 使用嵌入
```

