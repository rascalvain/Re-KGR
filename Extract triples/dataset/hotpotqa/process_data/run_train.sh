#!/bin/bash
# TransE 训练脚本示例

# 基础训练（需要先准备数据）
python train_transe.py \
    --prepare_data \
    --dim 100 \
    --epoch 1000 \
    --lr 1.0 \
    --margin 5.0 \
    --batch_size 300 \
    --neg_ent 64 \
    --save_steps 10 \
    --datadir ./openke_data \
    --outdir ./output

# 如果已经准备好数据，可以直接训练
# python train_transe.py \
#     --dim 100 \
#     --epoch 1000 \
#     --lr 1.0 \
#     --margin 5.0 \
#     --batch_size 300 \
#     --neg_ent 64 \
#     --save_steps 10 \
#     --datadir ./openke_data \
#     --outdir ./output

# 带测试的训练
# python train_transe.py \
#     --prepare_data \
#     --test \
#     --dim 100 \
#     --epoch 1000 \
#     --lr 1.0 \
#     --margin 5.0 \
#     --batch_size 300 \
#     --neg_ent 64 \
#     --save_steps 10 \
#     --datadir ./openke_data \
#     --outdir ./output

# 从预训练模型继续训练
# python train_transe.py \
#     --model_path ./output/transe_final.ckpt \
#     --dim 100 \
#     --epoch 500 \
#     --lr 0.5 \
#     --margin 5.0 \
#     --batch_size 300 \
#     --neg_ent 64 \
#     --save_steps 10 \
#     --datadir ./openke_data \
#     --outdir ./output_continue

