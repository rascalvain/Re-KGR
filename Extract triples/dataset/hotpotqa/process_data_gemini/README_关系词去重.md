# 关系词去重和对齐工具使用说明

## 功能说明

使用sentence-transformer对关系词进行语义对齐、聚类和去重，去除冗余的关系词。

## 主要功能

1. **语义嵌入**: 使用sentence-transformer将关系词编码为向量
2. **相似度计算**: 计算关系词之间的语义相似度
3. **去重对齐**: 将相似的关系词合并，保留主关系词
4. **ID映射**: 生成旧ID到新ID的映射文件
5. **结果保存**: 保存去重后的关系词文件、对齐信息和统计信息

## 文件说明

- `relation_deduplication.py`: 基础版本的去重工具
- `relation_deduplication_optimized.py`: 优化版本，支持大量关系词的高效处理

## 使用方法

### 基础使用

```bash
# 使用默认参数（相似度阈值0.85）
python relation_deduplication_optimized.py --input relation2id.txt

# 指定输出目录
python relation_deduplication_optimized.py --input relation2id.txt --output ./output

# 调整相似度阈值（0-1之间，值越大越严格）
python relation_deduplication_optimized.py --input relation2id.txt --threshold 0.9

# 使用CPU（如果没有GPU）
python relation_deduplication_optimized.py --input relation2id.txt --device cpu
```

### 高级选项

```bash
# 使用DBSCAN聚类方法（适合大量关系词）
python relation_deduplication_optimized.py --input relation2id.txt --method clustering

# 使用分批相似度计算方法（内存友好）
python relation_deduplication_optimized.py --input relation2id.txt --method similarity

# 不使用缓存（重新计算嵌入向量）
python relation_deduplication_optimized.py --input relation2id.txt --no-cache

# 指定自定义模型路径
python relation_deduplication_optimized.py --input relation2id.txt --model /path/to/model
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 输入的关系词文件路径 | `relation2id.txt` |
| `--output` | `-o` | 输出目录 | `.` (当前目录) |
| `--threshold` | `-t` | 相似度阈值 (0-1) | `0.85` |
| `--model` | `-m` | sentence-transformer模型路径 | 自动检测 |
| `--device` | `-d` | 计算设备 (`cuda` 或 `cpu`) | `cuda` |
| `--method` | | 去重方法 (`clustering` 或 `similarity`) | `similarity` |
| `--no-cache` | | 不使用嵌入向量缓存 | `False` |

## 输出文件

运行后会生成以下文件：

1. **relation2id_deduplicated.txt**: 去重后的关系词文件（格式与原文件相同）
2. **id_mapping.json**: 旧ID到新ID的映射关系
3. **relation_alignment.json**: 关系词对齐信息，显示哪些关系词被合并到一起
4. **deduplication_stats.json**: 统计信息（去重前后的数量、减少比例等）
5. **embeddings_cache.pkl**: 嵌入向量缓存（如果使用缓存）

## 相似度阈值建议

- **0.90-0.95**: 非常严格，只合并几乎完全相同的关系词
- **0.85-0.90**: 推荐值，平衡去重效果和准确性
- **0.80-0.85**: 较宽松，会合并更多相似的关系词
- **0.75-0.80**: 很宽松，可能合并语义相关但不完全相同的关系词

## 示例

### 示例1: 基本去重

```bash
python relation_deduplication_optimized.py \
    --input relation2id.txt \
    --output ./deduplicated \
    --threshold 0.85
```

### 示例2: 使用聚类方法处理大量关系词

```bash
python relation_deduplication_optimized.py \
    --input relation2id.txt \
    --output ./deduplicated \
    --method clustering \
    --threshold 0.85
```

### 示例3: 使用CPU和自定义阈值

```bash
python relation_deduplication_optimized.py \
    --input relation2id.txt \
    --output ./deduplicated \
    --device cpu \
    --threshold 0.88
```

## 性能说明

- **嵌入向量计算**: 对于2万个关系词，使用GPU大约需要5-10分钟
- **相似度计算**: 
  - `similarity`方法：分批计算，内存友好，适合大文件
  - `clustering`方法：使用DBSCAN，速度快但内存占用较大
- **缓存机制**: 首次运行会计算并保存嵌入向量，后续运行可直接加载，大幅提升速度

## 注意事项

1. 首次运行需要下载或加载sentence-transformer模型，可能需要一些时间
2. 如果使用GPU，确保已安装CUDA版本的PyTorch
3. 相似度阈值需要根据实际数据调整，建议先用默认值测试
4. 去重后的关系词ID会重新编号（从0开始）

## 结果解读

### relation_alignment.json 示例

```json
{
  "director": ["director", "co-director"],
  "writer": ["writer", "author"],
  "producer": ["producer"]
}
```

表示：
- `director` 和 `co-director` 被合并，主关系词是 `director`
- `writer` 和 `author` 被合并，主关系词是 `writer`
- `producer` 单独存在，没有重复

### id_mapping.json 示例

```json
{
  "0": 0,
  "1": 1,
  "6": 2,
  "7": 3,
  ...
}
```

表示旧ID到新ID的映射关系。

## 故障排除

1. **ModuleNotFoundError**: 确保已安装 `sentence-transformers` 和 `scikit-learn`
   ```bash
   pip install sentence-transformers scikit-learn
   ```

2. **CUDA错误**: 如果GPU不可用，使用 `--device cpu`

3. **内存不足**: 
   - 使用 `--method similarity` 方法（分批处理）
   - 降低 `batch_size`（需要修改代码）

4. **模型加载失败**: 检查模型路径，或使用在线模型（会自动下载）

## 联系与支持

如有问题，请检查：
1. Python版本（建议3.7+）
2. 依赖包版本
3. 输入文件格式是否正确






