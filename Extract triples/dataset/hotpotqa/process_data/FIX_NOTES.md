# 修复说明

## 问题描述

错误信息：
```
TypeError: __init__() got an unexpected keyword argument 'patient'
```

## 原因

不同版本的 OpenKE 支持的参数不同。旧版本的 `Trainer` 类不支持 `patient`、`save_steps`、`checkpoint_dir` 等参数。

## 解决方案

已更新 `train_transe.py`，代码会自动检测 OpenKE 版本并使用兼容的参数。

### 修改内容

1. **自动兼容性处理**: 使用 try-except 来适配不同版本
2. **移除不兼容参数**: 对于旧版本，只使用基础参数
3. **保留核心功能**: 训练和嵌入保存功能不受影响

## 使用方法

### 1. 测试安装（推荐先运行）

```bash
python test_openke.py
```

这会检查：
- OpenKE 是否正确安装
- 支持哪些参数
- PyTorch 和 CUDA 状态
- 其他依赖是否齐全

### 2. 开始训练

**最简单的方式**:
```bash
python train_transe.py --prepare_data
```

**自定义参数**:
```bash
python train_transe.py --prepare_data --dim 100 --epoch 1000 --lr 1.0
```

**Windows 一键运行**:
```bash
run_train.bat
```

## 兼容性说明

### 新版本 OpenKE
- 支持 `save_steps`、`checkpoint_dir` 等参数
- 可以在训练过程中定期保存模型
- 训练完成后会自动保存最终模型

### 旧版本 OpenKE
- 只使用基础参数：`model`, `data_loader`, `train_times`, `alpha`, `use_gpu`
- 训练完成后手动保存最终模型
- 功能完全正常，只是缺少训练过程中的自动保存

## 核心功能保证

无论使用哪个版本的 OpenKE，以下功能都能正常工作：

✓ 数据准备和格式转换
✓ TransE 模型训练
✓ 最终模型保存
✓ 嵌入向量提取
✓ 保存为 pickle 和 numpy 格式
✓ GPU 自动检测和使用

## 输出文件

训练完成后，在 `output/` 目录会有：

```
output/
├── transe_final.ckpt          # 最终模型
├── ent_embeddings.pkl         # 实体嵌入（pickle）
├── rel_embeddings.pkl         # 关系嵌入（pickle）
├── ent_embeddings.npy         # 实体嵌入（numpy）
└── rel_embeddings.npy         # 关系嵌入（numpy）
```

## 常见问题

### Q1: 还是报错怎么办？

先运行测试脚本：
```bash
python test_openke.py
```

查看具体哪个环节出问题。

### Q2: 如何确认训练成功？

看到以下输出表示成功：
```
训练完成！
实体嵌入已保存到: ./output/ent_embeddings.pkl
关系嵌入已保存到: ./output/rel_embeddings.pkl
```

### Q3: 训练太慢怎么办？

减少参数快速测试：
```bash
python train_transe.py --prepare_data --dim 50 --epoch 100
```

### Q4: 需要升级 OpenKE 吗？

不需要！代码已经兼容新旧版本。

## 验证结果

训练完成后，运行：
```bash
python use_embeddings.py
```

这会展示如何使用嵌入向量。

## 技术细节

### 版本检测逻辑

```python
try:
    # 尝试使用新版本参数
    trainer = Trainer(
        model = model,
        data_loader = train_dataloader,
        train_times = args.epoch,
        alpha = args.lr,
        use_gpu = torch.cuda.is_available(),
        save_steps = args.save_steps,
        checkpoint_dir = args.outdir
    )
except TypeError:
    # 回退到旧版本参数
    trainer = Trainer(
        model = model,
        data_loader = train_dataloader,
        train_times = args.epoch,
        alpha = args.lr,
        use_gpu = torch.cuda.is_available()
    )
```

这确保了无论使用哪个版本的 OpenKE，代码都能正常运行。

## 更新日期

2024-12-04

## 联系支持

如果遇到其他问题，请提供：
1. `python test_openke.py` 的完整输出
2. 错误信息的完整堆栈跟踪
3. OpenKE 和 PyTorch 的版本信息

