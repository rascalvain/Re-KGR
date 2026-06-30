#!/bin/bash
# RGAT 训练脚本 - 带日志记录

# 创建日志目录
mkdir -p rgat_output/logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="rgat_output/logs/train_rgat_${TIMESTAMP}.log"

echo "=================================="
echo "开始训练 RGAT 模型 - Mintaka"
echo "=================================="
echo "日志文件: $LOG_FILE"
echo ""

# 使用 tee 同时输出到终端和日志文件
# -a 表示追加模式（如果文件已存在）
python train_rgat_mintaka.py 2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "训练完成！"
    echo "=================================="
    echo "日志已保存到: $LOG_FILE"
    echo "模型已保存到: rgat_output/checkpoints/best_rgat_model.pth"
else
    echo ""
    echo "=================================="
    echo "训练失败！请检查日志文件"
    echo "=================================="
    echo "日志文件: $LOG_FILE"
    exit 1
fi
