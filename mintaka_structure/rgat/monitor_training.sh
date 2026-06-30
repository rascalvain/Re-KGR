#!/bin/bash
# 训练监控脚本

# 查找最新的日志文件
LATEST_LOG=$(ls -t rgat_output/logs/train_rgat_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "未找到训练日志文件"
    echo "日志目录: rgat_output/logs/"
    exit 1
fi

echo "=================================="
echo "RGAT 训练监控"
echo "=================================="
echo "日志文件: $LATEST_LOG"
echo ""
echo "按 Ctrl+C 退出监控"
echo ""
echo "=================================="
echo ""

# 实时显示最新的日志
tail -f "$LATEST_LOG"
