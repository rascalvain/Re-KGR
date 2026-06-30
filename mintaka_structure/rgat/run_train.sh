#!/bin/bash
# Mintaka RGAT 训练流程脚本

set -e  # 遇到错误时退出

echo "=================================="
echo "Mintaka RGAT 训练流程"
echo "=================================="

# 1. 准备嵌入文件
echo ""
echo "[步骤 1/3] 准备RGAT嵌入文件..."
python prepare_embeddings.py

if [ $? -ne 0 ]; then
    echo "错误: 嵌入文件准备失败"
    exit 1
fi

# 2. 训练RGAT
echo ""
echo "[步骤 2/3] 训练RGAT模型..."
python train_rgat_mintaka.py

if [ $? -ne 0 ]; then
    echo "错误: RGAT训练失败"
    exit 1
fi

# 3. 训练分类器
echo ""
echo "[步骤 3/3] 训练幻觉检测分类器..."
RGAT_MODEL="rgat_output/checkpoints/best_rgat_model.pth"

if [ ! -f "$RGAT_MODEL" ]; then
    echo "错误: 找不到预训练RGAT模型: $RGAT_MODEL"
    exit 1
fi

python train_classifier.py --pretrained_model "$RGAT_MODEL"

if [ $? -ne 0 ]; then
    echo "错误: 分类器训练失败"
    exit 1
fi

echo ""
echo "=================================="
echo "训练完成！"
echo "=================================="
echo "生成的文件:"
echo "  RGAT模型: rgat_output/checkpoints/best_rgat_model.pth"
echo "  分类器: rgat_output/checkpoints/best_classifier.pth"
echo "  结果: rgat_output/classifier_results.json"
