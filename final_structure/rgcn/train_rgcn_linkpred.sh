#!/bin/bash

##############################################
# RGCN链接预测训练脚本（后台运行版本）
# 使用nohup后台运行，输出重定向到日志文件
##############################################

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="train_rgcn_linkpred.py"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_link_prediction_${TIMESTAMP}.log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 打印信息
echo "=========================================="
echo "RGCN链接预测训练 - 后台运行"
echo "=========================================="
echo "脚本目录: $SCRIPT_DIR"
echo "Python脚本: $PYTHON_SCRIPT"
echo "日志文件: $LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# 检查Python脚本是否存在
if [ ! -f "$SCRIPT_DIR/$PYTHON_SCRIPT" ]; then
    echo "错误: 未找到 $PYTHON_SCRIPT"
    exit 1
fi

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查必需文件
ENTITY_EMB="/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/data/final_data/final_hybrid_embeddings/entity_embeddings_rgcn.pkl"
if [ ! -f "$ENTITY_EMB" ]; then
    echo "错误: 未找到嵌入文件 $ENTITY_EMB"
    echo "请先运行: python prepare_embeddings.py"
    exit 1
fi

echo "✓ 所有检查通过"
echo ""

# 使用nohup后台运行Python脚本
# 2>&1 将标准错误重定向到标准输出
# & 表示后台运行
echo "启动训练进程（后台运行）..."
nohup python -u "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!

# 保存PID到文件
PID_FILE="$LOG_DIR/train_link_prediction.pid"
echo $PID > "$PID_FILE"

echo ""
echo "=========================================="
echo "训练已在后台启动"
echo "=========================================="
echo "进程ID (PID): $PID"
echo "PID文件: $PID_FILE"
echo "日志文件: $LOG_FILE"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f $LOG_FILE"
echo "  实时监控: watch -n 2 tail -20 $LOG_FILE"
echo "  检查进程: ps -p $PID"
echo "  停止训练: kill $PID"
echo ""
echo "训练完成后会在 rgcn_output/checkpoints/ 生成模型文件"
echo "=========================================="

# 等待1秒，检查进程是否正常启动
sleep 1
if ps -p $PID > /dev/null; then
    echo "✓ 训练进程运行正常"
    echo ""
    echo "实时查看日志输出（按Ctrl+C退出监控，不影响训练）:"
    echo "----------------------------------------"
    tail -f "$LOG_FILE"
else
    echo "✗ 训练进程启动失败，请检查日志文件"
    exit 1
fi