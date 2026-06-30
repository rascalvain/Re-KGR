# RGAT 训练命令指南

本指南提供了在服务器上训练 RGAT 模型的详细命令。

## 快速开始

### 方式一：前台训练（推荐用于测试）

```bash
cd /root/autodl-fs/gca/mintaka_structure/rgat

# 训练 RGAT（输出同时显示在终端和保存到日志）
bash train_rgat.sh
```

日志将保存到：`rgat_output/logs/train_rgat_YYYYMMDD_HHMMSS.log`

---

### 方式二：后台训练（推荐用于长时间训练）

```bash
cd /root/autodl-fs/gca/mintaka_structure/rgat

# 启动后台训练
bash train_rgat_background.sh
```

**后台训练的管理命令：**

```bash
# 查看实时日志
tail -f rgat_output/logs/train_rgat_*.log

# 或使用监控脚本
bash monitor_training.sh

# 查看进程状态
ps -p $(cat rgat_output/logs/train_rgat.pid)

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 停止训练
kill $(cat rgat_output/logs/train_rgat.pid)
```

---

### 方式三：完整流程（嵌入准备 + RGAT训练 + 分类器训练）

```bash
cd /root/autodl-fs/gca/mintaka_structure/rgat

# 运行完整流程
bash run_full_pipeline.sh
```

这将自动完成：
1. 准备嵌入文件
2. 训练RGAT模型
3. 训练分类器

所有日志都会保存到单独的文件中。

---

## 分步训练

### 步骤 1: 准备嵌入文件

```bash
python prepare_embeddings.py
```

### 步骤 2: 训练 RGAT 模型

**前台训练：**
```bash
bash train_rgat.sh
```

**或直接运行Python脚本并保存日志：**
```bash
# 方法1: 只保存到日志文件（不显示在终端）
python train_rgat_mintaka.py > rgat_output/logs/train_rgat.log 2>&1

# 方法2: 同时显示在终端和保存到日志
python train_rgat_mintaka.py 2>&1 | tee rgat_output/logs/train_rgat.log

# 方法3: 后台运行
nohup python -u train_rgat_mintaka.py > rgat_output/logs/train_rgat.log 2>&1 &
```

### 步骤 3: 训练分类器

```bash
bash train_classifier.sh rgat_output/checkpoints/best_rgat_model.pth
```

**或直接运行：**
```bash
python train_classifier.py \
    --pretrained_model rgat_output/checkpoints/best_rgat_model.pth \
    2>&1 | tee rgat_output/logs/train_classifier.log
```

---

## 常用命令说明

### 日志命令详解

| 命令 | 说明 |
|------|------|
| `> file.log 2>&1` | 将标准输出和错误输出都重定向到文件（覆盖） |
| `>> file.log 2>&1` | 追加模式（不覆盖原有内容） |
| `2>&1 \| tee file.log` | 同时显示在终端和保存到文件 |
| `nohup ... &` | 后台运行，断开SSH连接后继续运行 |
| `python -u` | 禁用Python输出缓冲（实时写入日志） |

### 查看日志

```bash
# 查看完整日志
cat rgat_output/logs/train_rgat_20240526_120000.log

# 实时查看最新日志（自动滚动）
tail -f rgat_output/logs/train_rgat_20240526_120000.log

# 查看最后100行
tail -n 100 rgat_output/logs/train_rgat_20240526_120000.log

# 搜索关键词
grep "Epoch" rgat_output/logs/train_rgat_20240526_120000.log
grep "val_loss" rgat_output/logs/train_rgat_20240526_120000.log

# 查看最新的日志文件
ls -t rgat_output/logs/train_rgat_*.log | head -1
```

---

## 监控训练进度

### 实时监控日志

```bash
# 使用监控脚本（自动找到最新日志）
bash monitor_training.sh

# 或手动指定日志文件
tail -f rgat_output/logs/train_rgat_20240526_120000.log
```

### 监控GPU使用

```bash
# 实时监控GPU（每1秒刷新）
watch -n 1 nvidia-smi

# 或使用简化命令
nvidia-smi -l 1

# 查看特定GPU
nvidia-smi -i 0 -l 1
```

### 监控系统资源

```bash
# 查看CPU和内存使用
htop

# 或
top
```

---

## 训练输出文件

训练完成后，会生成以下文件：

### RGAT 训练输出

```
rgat_output/
├── checkpoints/
│   ├── best_rgat_model.pth              # 最佳RGAT模型
│   └── checkpoint_epoch_*.pth            # 定期保存的检查点
├── logs/
│   ├── train_rgat_YYYYMMDD_HHMMSS.log   # 训练日志
│   └── train_rgat.pid                    # 后台进程ID（如果使用后台模式）
├── rgat_training_curves.png              # 训练曲线图
└── rgat_training_history.json            # 训练历史数据
```

### 分类器训练输出

```
rgat_output/
├── checkpoints/
│   └── best_classifier.pth               # 最佳分类器
├── logs/
│   └── train_classifier_YYYYMMDD_HHMMSS.log  # 训练日志
└── classifier_results.json               # 测试结果（准确率、F1等）
```

---

## 常见问题

### 1. 如何查看训练进度？

```bash
# 实时查看日志
tail -f rgat_output/logs/train_rgat_*.log

# 搜索验证损失
grep "验证损失" rgat_output/logs/train_rgat_*.log
```

### 2. 如何从检查点恢复训练？

目前脚本会自动保存检查点，但不支持自动恢复。如需恢复，需要修改训练脚本加载检查点。

### 3. 训练中断了怎么办？

如果是后台训练：
```bash
# 检查进程是否还在运行
ps -p $(cat rgat_output/logs/train_rgat.pid)

# 如果已停止，重新启动
bash train_rgat_background.sh
```

### 4. 如何调整训练参数？

编辑配置文件：
```bash
vim config_mintaka_rgat.py
```

常用参数：
- `BATCH_SIZE`: 批大小
- `NUM_EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率
- `NUM_HEADS`: 注意力头数

### 5. 显存不足怎么办？

```bash
# 编辑配置文件，减小batch size
vim config_mintaka_rgat.py

# 修改以下参数：
# BATCH_SIZE = 4  # 从6改为4
# HIDDEN_CHANNELS = 128  # 从256改为128
# NUM_HEADS = 4  # 从8改为4
```

---

## 示例：完整训练流程

```bash
# 1. 进入目录
cd /root/autodl-fs/gca/mintaka_structure/rgat

# 2. 确保脚本有执行权限
chmod +x *.sh

# 3. 启动后台训练
bash train_rgat_background.sh

# 4. 查看训练开始情况
tail -n 50 rgat_output/logs/train_rgat_*.log

# 5. 实时监控（新开一个终端）
bash monitor_training.sh

# 6. 监控GPU（再开一个终端）
watch -n 1 nvidia-smi

# 7. 等待训练完成（可以关闭SSH，训练会继续）

# 8. 训练完成后，训练分类器
bash train_classifier.sh rgat_output/checkpoints/best_rgat_model.pth

# 9. 查看结果
cat rgat_output/classifier_results.json
```

---

## 时间估计

基于 Mintaka dev 数据集（约2000样本）：

- **RGAT训练**: 约2-4小时（取决于GPU）
- **分类器训练**: 约30分钟-1小时
- **总计**: 约3-5小时

*实际时间取决于GPU性能、批大小和早停触发时机*

---

## 联系与支持

训练过程中如遇到问题，请检查：
1. 日志文件中的错误信息
2. GPU显存是否足够
3. 数据文件路径是否正确
4. 嵌入文件是否已准备
