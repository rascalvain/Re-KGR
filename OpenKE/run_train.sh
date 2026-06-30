#!/bin/bash

# ================= 配置区域 =================
PROJECT_ROOT="/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/gpt3.5/final_structure"
DATA_DIR="${PROJECT_ROOT}/data/graph_data"
OUT_DIR="${PROJECT_ROOT}/transe_output"

# 定义日志文件 (包含时间戳)
LOG_FILE="train_transe_$(date +%Y%m%d_%H%M%S).log"

# 确保输出目录存在
mkdir -p "$OUT_DIR"

# ================= 核心命令 =================
# 将具体的 python 命令定义为一个字符串或函数
CMD="python train_transe.py \
    --prepare_data \
    --dim 384 \
    --epoch 5000 \
    --lr 1.0 \
    --margin 5.0 \
    --batch_size 300 \
    --neg_ent 64 \
    --save_steps 10 \
    --datadir $DATA_DIR \
    --outdir $OUT_DIR"

# ================= 后台执行 =================
echo "正在后台启动训练任务..."
echo "日志文件: $LOG_FILE"

# 使用 nohup 执行 CMD，将 stdout(1) 重定向到日志，stderr(2) 重定向到 stdout
nohup $CMD > "$LOG_FILE" 2>&1 &

# 获取并打印 PID
PID=$!
echo "任务已启动，PID: $PID"
echo "你可以使用 'tail -f $LOG_FILE' 查看实时日志"