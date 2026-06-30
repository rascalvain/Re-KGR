#!/bin/bash
# RGAT 后台训练脚本 - 适用于长时间训练

# 创建日志目录
mkdir -p rgat_output/logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="rgat_output/logs/train_rgat_${TIMESTAMP}.log"
PID_FILE="rgat_output/logs/train_rgat.pid"

echo "=================================="
echo "启动 RGAT 后台训练 - Mintaka"
echo "=================================="
echo "日志文件: $LOG_FILE"
echo "PID文件: $PID_FILE"
echo ""

# 使用 nohup 后台运行
nohup python -u train_rgat_mintaka.py > "$LOG_FILE" 2>&1 &

# 保存进程ID
echo $! > "$PID_FILE"

echo "训练已在后台启动！"
echo "进程ID: $(cat $PID_FILE)"
echo ""
echo "查看实时日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看进程状态:"
echo "  ps -p $(cat $PID_FILE)"
echo ""
echo "停止训练:"
echo "  kill $(cat $PID_FILE)"
echo ""
echo "查看GPU使用:"
echo "  watch -n 1 nvidia-smi"
