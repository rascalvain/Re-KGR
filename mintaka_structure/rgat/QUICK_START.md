# 快速开始 - RGAT 训练命令

## 最简单的方式（推荐）

```bash
cd /root/autodl-fs/gca/mintaka_structure/rgat

# 一键完整训练（嵌入准备 + RGAT + 分类器）
bash run_full_pipeline.sh
```

所有日志自动保存到 `rgat_output/logs/` 目录。

---

## 分步训练

### 1. 仅训练 RGAT

```bash
# 前台训练（显示在终端 + 保存日志）
bash train_rgat.sh

# 或后台训练（适合长时间训练）
bash train_rgat_background.sh
```

### 2. 仅训练分类器

```bash
bash train_classifier.sh rgat_output/checkpoints/best_rgat_model.pth
```

---

## 监控训练

```bash
# 实时查看日志
bash monitor_training.sh

# 或手动查看
tail -f rgat_output/logs/train_rgat_*.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

---

## 原始Python命令（带日志）

如果你更喜欢直接运行Python脚本：

### RGAT训练

```bash
# 方法1: 同时显示在终端和保存日志（推荐）
python train_rgat_mintaka.py 2>&1 | tee rgat_output/logs/train_rgat.log

# 方法2: 仅保存到日志
python train_rgat_mintaka.py > rgat_output/logs/train_rgat.log 2>&1

# 方法3: 后台运行
nohup python -u train_rgat_mintaka.py > rgat_output/logs/train_rgat.log 2>&1 &
```

### 分类器训练

```bash
python train_classifier.py \
    --pretrained_model rgat_output/checkpoints/best_rgat_model.pth \
    2>&1 | tee rgat_output/logs/train_classifier.log
```

---

## 输出文件位置

```
rgat_output/
├── checkpoints/
│   ├── best_rgat_model.pth      ← RGAT模型
│   └── best_classifier.pth      ← 分类器
├── logs/
│   ├── train_rgat_*.log         ← RGAT训练日志
│   └── train_classifier_*.log   ← 分类器日志
├── rgat_training_curves.png     ← 训练曲线
└── classifier_results.json      ← 测试结果
```

---

## 常用监控命令

| 命令 | 说明 |
|------|------|
| `bash monitor_training.sh` | 实时查看训练日志 |
| `watch -n 1 nvidia-smi` | 实时查看GPU使用 |
| `tail -f rgat_output/logs/train_rgat_*.log` | 手动查看日志 |
| `grep "Epoch" rgat_output/logs/train_rgat_*.log` | 搜索训练轮次 |
| `cat rgat_output/classifier_results.json` | 查看测试结果 |

---

详细文档请查看 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
