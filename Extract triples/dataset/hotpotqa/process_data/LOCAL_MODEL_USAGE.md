# 使用本地 sentence-bert 模型说明

## 📍 模型位置

本地 sentence-bert 模型位于：
```
g:\小论文\第三章\GCA-main\sentence-bert\
```

从当前工作目录（`hotpotqa/`）的相对路径：
```
../../../sentence-bert
```

## ✅ 验证模型

在生成混合嵌入之前，建议先验证模型是否可以正常加载：

```bash
python test_sentence_bert.py
```

这会检查：
- ✓ 模型路径是否存在
- ✓ 必要的模型文件是否完整
- ✓ 能否成功加载模型
- ✓ 编码功能是否正常

## 🚀 使用本地模型

### 方式 1: 使用默认配置（推荐）

代码已经默认使用本地模型，直接运行即可：

```bash
python generate_hybrid_embeddings.py
```

或使用批处理脚本：
```bash
run_generate_hybrid.bat
```

### 方式 2: 显式指定模型路径

```bash
python generate_hybrid_embeddings.py --sentence_model ../../../sentence-bert
```

### 方式 3: 使用绝对路径

```bash
python generate_hybrid_embeddings.py --sentence_model "g:\小论文\第三章\GCA-main\sentence-bert"
```

### 方式 4: 使用在线模型（如果需要）

如果需要使用不同的模型：

```bash
# 使用其他在线模型
python generate_hybrid_embeddings.py --sentence_model sentence-transformers/all-mpnet-base-v2
```

## 📊 本地模型信息

### 模型文件

```
sentence-bert/
├── config.json                           # 模型配置
├── pytorch_model.bin                     # 模型权重 (87MB)
├── vocab.txt                             # 词汇表 (226KB, 30523词)
├── tokenizer_config.json                 # 分词器配置
├── tokenizer.json                        # 分词器 (455KB)
├── special_tokens_map.json               # 特殊标记
├── sentence_bert_config.json             # SentenceBERT配置
├── config_sentence_transformers.json     # SentenceTransformers配置
├── modules.json                          # 模块配置
└── 1_Pooling/
    └── config.json                       # Pooling层配置
```

### 模型特点

- **基础模型**: 基于 BERT 的句子嵌入模型
- **嵌入维度**: 384 维（通常）
- **速度**: 快速编码，适合大规模文本
- **质量**: 在语义相似度任务上表现良好

## 🔧 配置说明

### generate_hybrid_embeddings.py 中的相关配置

```python
# 默认模型路径（相对路径）
--sentence_model ../../../sentence-bert

# 或使用绝对路径
--sentence_model g:\小论文\第三章\GCA-main\sentence-bert

# Batch size（根据内存调整）
--batch_size 32  # 可以增大到 64, 128 等
```

### 嵌入维度计算

假设 TransE 维度为 100，本地 sentence-bert 维度为 384：

```
混合嵌入维度 = 100 + 384 = 484
```

## ⚠️ 常见问题

### Q1: 模型加载失败

**错误**: `OSError: ../../../sentence-bert does not appear to be a valid model`

**解决**:
1. 检查路径是否正确：
   ```bash
   python test_sentence_bert.py
   ```

2. 确认当前工作目录：
   ```bash
   cd "g:\小论文\第三章\GCA-main\Extract triples\dataset\hotpotqa"
   ```

3. 使用绝对路径：
   ```bash
   python generate_hybrid_embeddings.py --sentence_model "g:\小论文\第三章\GCA-main\sentence-bert"
   ```

### Q2: 找不到 sentence-transformers 模块

**错误**: `ModuleNotFoundError: No module named 'sentence_transformers'`

**解决**:
```bash
pip install sentence-transformers
```

### Q3: 模型文件不完整

**解决**:
1. 检查 `pytorch_model.bin` 是否存在且大小约 87MB
2. 检查 `vocab.txt` 是否存在且大小约 226KB
3. 如果文件损坏，重新下载模型

### Q4: 内存不足

**解决**:
```bash
# 减小 batch size
python generate_hybrid_embeddings.py --batch_size 16
```

## 📝 完整示例

### 示例 1: 使用本地模型生成混合嵌入

```bash
# 1. 测试本地模型
python test_sentence_bert.py

# 2. 生成混合嵌入（使用默认本地模型）
python generate_hybrid_embeddings.py

# 输出：
# 正在加载 SentenceTransformer 模型: ../../../sentence-bert
#   使用本地模型: g:\小论文\第三章\GCA-main\sentence-bert
# 正在生成 5478 个文本的嵌入...
#   SentenceTransformer 嵌入维度: 384
```

### 示例 2: 为响应图谱生成混合嵌入

```bash
python generate_hybrid_embeddings.py \
    --triple_file response_triples.txt \
    --output_dir ./response_embeddings \
    --sentence_model ../../../sentence-bert
```

### 示例 3: 使用更大的 batch size（如果内存充足）

```bash
python generate_hybrid_embeddings.py --batch_size 64
```

## 🎯 优势

使用本地模型的优势：

1. ✅ **无需网络**: 离线也能运行
2. ✅ **速度快**: 不需要下载，立即使用
3. ✅ **稳定性**: 避免网络问题导致的中断
4. ✅ **版本固定**: 确保结果可重现

## 🔄 切换模型

如果需要使用其他模型，只需修改 `--sentence_model` 参数：

```bash
# 使用其他本地模型
python generate_hybrid_embeddings.py --sentence_model /path/to/other/model

# 使用在线模型
python generate_hybrid_embeddings.py --sentence_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 📚 相关文档

- **混合嵌入生成**: `README_Hybrid_Embeddings.md`
- **完整工作流程**: `WORKFLOW_SUMMARY.md`
- **快速开始**: `QUICKSTART.md`

## 📅 更新日期

2024-12-04

