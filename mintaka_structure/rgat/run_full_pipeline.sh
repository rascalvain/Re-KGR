#!/bin/bash
# 完整训练流程 - 带日志记录

set -e  # 遇到错误时退出

# 创建日志目录
mkdir -p rgat_output/logs

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="rgat_output/logs/full_pipeline_${TIMESTAMP}.log"

echo "=================================="
echo "Mintaka RGAT 完整训练流程"
echo "=================================="
echo "主日志文件: $MAIN_LOG"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 函数：记录并显示消息
log_message() {
    echo "$1" | tee -a "$MAIN_LOG"
}

# ==================== 步骤 1: 准备嵌入 ====================
log_message ""
log_message "[步骤 1/3] 准备RGAT嵌入文件..."
log_message "$(date '+%Y-%m-%d %H:%M:%S')"

python prepare_embeddings.py 2>&1 | tee -a "$MAIN_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "错误: 嵌入文件准备失败"
    exit 1
fi

# ==================== 步骤 2: 训练 RGAT ====================
log_message ""
log_message "[步骤 2/3] 训练RGAT模型..."
log_message "$(date '+%Y-%m-%d %H:%M:%S')"

RGAT_LOG="rgat_output/logs/train_rgat_${TIMESTAMP}.log"
python train_rgat_mintaka.py 2>&1 | tee "$RGAT_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "错误: RGAT训练失败"
    log_message "详细日志: $RGAT_LOG"
    exit 1
fi

# 追加RGAT日志摘要到主日志
log_message "RGAT训练完成，详细日志: $RGAT_LOG"

# ==================== 步骤 3: 训练分类器 ====================
log_message ""
log_message "[步骤 3/3] 训练幻觉检测分类器..."
log_message "$(date '+%Y-%m-%d %H:%M:%S')"

RGAT_MODEL="rgat_output/checkpoints/best_rgat_model.pth"

if [ ! -f "$RGAT_MODEL" ]; then
    log_message "错误: 找不到预训练RGAT模型: $RGAT_MODEL"
    exit 1
fi

CLASSIFIER_LOG="rgat_output/logs/train_classifier_${TIMESTAMP}.log"
python train_classifier.py --pretrained_model "$RGAT_MODEL" 2>&1 | tee "$CLASSIFIER_LOG"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "错误: 分类器训练失败"
    log_message "详细日志: $CLASSIFIER_LOG"
    exit 1
fi

# 追加分类器日志摘要到主日志
log_message "分类器训练完成，详细日志: $CLASSIFIER_LOG"

# ==================== 完成 ====================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

log_message ""
log_message "=================================="
log_message "训练完成！"
log_message "=================================="
log_message "总耗时: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
log_message ""
log_message "生成的文件:"
log_message "  RGAT模型: rgat_output/checkpoints/best_rgat_model.pth"
log_message "  分类器: rgat_output/checkpoints/best_classifier.pth"
log_message "  分类结果: rgat_output/classifier_results.json"
log_message ""
log_message "日志文件:"
log_message "  主日志: $MAIN_LOG"
log_message "  RGAT日志: $RGAT_LOG"
log_message "  分类器日志: $CLASSIFIER_LOG"
log_message ""
log_message "训练曲线:"
log_message "  rgat_output/rgat_training_curves.png"
