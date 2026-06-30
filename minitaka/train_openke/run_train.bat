@echo off
REM TransE 训练脚本示例 (Windows)

echo ========================================
echo TransE 模型训练
echo ========================================

REM 基础训练（需要先准备数据）
python train_transe.py ^
    --prepare_data ^
    --dim 100 ^
    --epoch 1000 ^
    --lr 1.0 ^
    --margin 5.0 ^
    --batch_size 300 ^
    --neg_ent 64 ^
    --save_steps 10 ^
    --datadir ./openke_data ^
    --outdir ./output

echo.
echo ========================================
echo 训练完成！
echo ========================================
echo.
echo 生成的文件：
echo   - ./output/transe_final.ckpt
echo   - ./output/ent_embeddings.pkl
echo   - ./output/rel_embeddings.pkl
echo   - ./output/ent_embeddings.npy
echo   - ./output/rel_embeddings.npy
echo.

pause

