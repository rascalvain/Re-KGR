@echo off
chcp 65001 >nul

echo ========================================
echo  Mintaka TransE 训练
echo ========================================

REM 默认嵌入维度 100；如需与 SentenceBERT(384) 拼接，改为 384
set DIM=100
set EPOCH=1000
set LR=1.0
set MARGIN=5.0
set BATCH=300
set NEG_ENT=64
set SAVE_STEPS=100

python train_transe.py ^
    --dim %DIM% ^
    --epoch %EPOCH% ^
    --lr %LR% ^
    --margin %MARGIN% ^
    --batch_size %BATCH% ^
    --neg_ent %NEG_ENT% ^
    --save_steps %SAVE_STEPS%

echo.
echo ========================================
echo  训练完成！
echo ========================================
echo  输出文件：
echo    output\transe_final.ckpt
echo    output\ent_embeddings.pkl / .npy
echo    output\rel_embeddings.pkl / .npy
echo ========================================

pause
