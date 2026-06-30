# 数据范围配置说明

## 功能概述
`generate_answer.py` 现在支持灵活的数据范围配置，可以自定义处理数据集的任意区间。

## 配置参数

### 1. 基本范围参数

#### `START_INDEX`
- **说明**: 起始索引（从0开始，包含该索引）
- **类型**: 整数
- **默认值**: 0
- **示例**: `START_INDEX = 100` 从第101条数据开始处理

#### `END_INDEX`
- **说明**: 结束索引（不包含该索引）
- **类型**: 整数 或 None
- **默认值**: None（处理到末尾）
- **示例**: `END_INDEX = 200` 处理到第200条（不包含第200条）

#### `TARGET_SAMPLES`
- **说明**: 从起始位置开始处理的样本数量
- **类型**: 整数 或 None
- **默认值**: None（处理全部）
- **优先级**: 低于 END_INDEX
- **示例**: `TARGET_SAMPLES = 50` 从起始位置处理50条

### 2. 断点续传参数

#### `RESUME_MODE`
- **说明**: 是否从上次中断处继续处理
- **类型**: 布尔值
- **默认值**: False
- **注意**: 启用时会忽略 START_INDEX 参数
- **示例**: `RESUME_MODE = True` 从上次保存的位置继续

## 使用场景示例

### 场景1: 处理前100条数据（测试）
```python
START_INDEX = 0
END_INDEX = 100
TARGET_SAMPLES = None
RESUME_MODE = False
```
或者：
```python
START_INDEX = 0
END_INDEX = None
TARGET_SAMPLES = 100
RESUME_MODE = False
```

### 场景2: 处理100-200条数据（分片处理）
```python
START_INDEX = 100
END_INDEX = 200
TARGET_SAMPLES = None
RESUME_MODE = False
```

### 场景3: 处理200条之后的所有数据
```python
START_INDEX = 200
END_INDEX = None
TARGET_SAMPLES = None
RESUME_MODE = False
```

### 场景4: 断点续传（中断后继续）
```python
START_INDEX = 0  # 会被忽略
END_INDEX = None
TARGET_SAMPLES = None
RESUME_MODE = True  # 从上次保存的位置自动继续
```

### 场景5: 处理中间的50条数据
```python
START_INDEX = 300
END_INDEX = None
TARGET_SAMPLES = 50  # 从第300条开始处理50条，即300-349
RESUME_MODE = False
```

### 场景6: 分批处理大数据集
```python
# 第一批：0-1000
START_INDEX = 0
END_INDEX = 1000

# 第二批：1000-2000
START_INDEX = 1000
END_INDEX = 2000

# 第三批：2000-3000
START_INDEX = 2000
END_INDEX = 3000

# ... 以此类推
```

## 参数优先级

1. **RESUME_MODE = True** 时：
   - 自动从上次保存位置继续
   - START_INDEX 被忽略
   - END_INDEX 和 TARGET_SAMPLES 仍然有效

2. **指定了 END_INDEX** 时：
   - 使用 [START_INDEX, END_INDEX) 作为处理范围
   - TARGET_SAMPLES 被忽略

3. **指定了 TARGET_SAMPLES** 时：
   - 使用 [START_INDEX, START_INDEX + TARGET_SAMPLES) 作为处理范围

4. **都未指定** 时：
   - 从 START_INDEX 处理到数据集末尾

## 统计信息说明

程序会显示两类统计信息：

### 本批次统计
- 仅统计当前运行处理的数据
- 包括：处理数量、正确率、幻觉率、耗时等

### 全部数据统计
- 统计输出文件中所有数据（包括之前处理的）
- 仅在存在历史数据时显示

## 注意事项

1. **索引范围**: 索引从0开始，END_INDEX 不包含在处理范围内（左闭右开区间）

2. **数据覆盖**: 
   - RESUME_MODE = False 时，会创建新的输出文件或覆盖已有文件
   - RESUME_MODE = True 时，会追加到已有文件

3. **范围验证**: 程序会自动验证范围的有效性，避免越界

4. **保存频率**: 每处理5条数据自动保存一次，防止数据丢失

## 实际应用建议

### 测试阶段
```python
START_INDEX = 0
TARGET_SAMPLES = 10  # 只处理10条测试
```

### 正式处理（单机）
```python
START_INDEX = 0
END_INDEX = None
RESUME_MODE = True  # 支持中断后继续
```

### 多机并行处理
```python
# 机器1
START_INDEX = 0
END_INDEX = 2500

# 机器2
START_INDEX = 2500
END_INDEX = 5000

# 机器3
START_INDEX = 5000
END_INDEX = 7500

# 机器4
START_INDEX = 7500
END_INDEX = None
```

## 常见问题

**Q: 如何知道数据集总共有多少条？**
A: 运行程序后会显示 "数据集总量: X 条"

**Q: 如果中断了怎么办？**
A: 设置 RESUME_MODE = True 重新运行即可

**Q: 可以跳过某些数据吗？**
A: 可以，通过设置不同的 START_INDEX 和 END_INDEX

**Q: 多次运行会覆盖之前的结果吗？**
A: RESUME_MODE = False 时会覆盖，True 时会追加

