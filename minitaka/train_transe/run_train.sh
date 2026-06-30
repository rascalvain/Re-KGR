#!/bin/bash

# ================= 配置 =================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/train_transe_$(date +%Y%m%d_%H%M%S).log"

# 默认嵌入维度 100；如需与 SentenceBERT(384 维) 拼接，可改为 384
DIM=100
EPOCH=1000
LR=1.0
MARGIN=5.0
BATCH=300
NEG_ENT=64
SAVE_STEPS=100

# ================= 训练 =================
CMD="python ${SCRIPT_DIR}/train_transe.py \
    --dim ${DIM} \
    --epoch ${EPOCH} \
    --lr ${LR} \
    --margin ${MARGIN} \
    --batch_size ${BATCH} \
    --neg_ent ${NEG_ENT} \
    --save_steps ${SAVE_STEPS}"

echo "启动 TransE 训练..."
echo "日志: ${LOG_FILE}"
nohup ${CMD} > "${LOG_FILE}" 2>&1 &
PID=$!
echo "PID: ${PID}"
echo "tail -f ${LOG_FILE}  查看实时日志"
