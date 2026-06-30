#!/bin/bash
# Mintaka RGCN 链接预测训练流程

set -e

echo "============================================"
echo "  Mintaka RGCN 链接预测训练"
echo "============================================"

cd "$(dirname "$0")"

# Step 1: 准备嵌入
echo ""
echo "[Step 1] 准备RGCN嵌入文件..."
python prepare_embeddings.py

# Step 2: 训练RGCN
echo ""
echo "[Step 2] 训练RGCN（链接预测）..."
python train_rgcn_linkpred.py

# Step 3: 提取节点嵌入
echo ""
echo "[Step 3] 提取更新后的节点嵌入..."
python update_node.py

echo ""
echo "============================================"
echo "  全部完成！"
echo "============================================"
