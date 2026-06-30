#!/bin/bash
# 分类器训练脚本 - 带日志记录

# 检查参数
if [ $# -eq 0 ]; then
    echo "用法: bash train_classifier.sh <预训练RGAT模型路径>"
    echo "示例: bash train_classifier.sh rgat_output/checkpoints/best_rgat_model.pth"
    exit 1
fi

PRETRAINED_MODEL=$1

# 检查模型文件是否存在
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 找不到预训练模型文件: $PRETRAINED_MODEL"
    exit 1
fi

# 创建日志目录
mkdir -p rgat_output/logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="rgat_output/logs/train_classifier_${TIMESTAMP}.log"

echo "=================================="
echo "开始训练分类器 - Mintaka"
echo "=================================="
echo "预训练模型: $PRETRAINED_MODEL"
echo "日志文件: $LOG_FILE"
echo ""

# 使用 tee 同时输出到终端和日志文件
python train_classifier.py --pretrained_model "$PRETRAINED_MODEL" 2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "训练完成！"
    echo "=================================="
    echo "日志已保存到: $LOG_FILE"
    echo "模型已保存到: rgat_output/checkpoints/best_classifier.pth"
    echo "结果已保存到: rgat_output/classifier_results.json"
else
    echo ""
    echo "=================================="
    echo "训练失败！请检查日志文件"
    echo "=================================="
    echo "日志文件: $LOG_FILE"
    exit 1
fi
