@echo off
chcp 65001 > nul
echo ========================================
echo 预训练编码器 + FFN分类器 完整流程
echo ========================================
echo.

set /p MODE="选择模式 (1=冻结编码器, 2=微调编码器[推荐]): "

if "%MODE%"=="1" (
    set FREEZE_FLAG=--freeze_encoder
    set MODE_NAME=冻结
) else (
    set FREEZE_FLAG=
    set MODE_NAME=微调
)

echo.
echo 已选择: %MODE_NAME%编码器模式
echo.

echo [1/4] 检查嵌入文件...
if not exist "..\hybrid_embeddings\entity_embeddings_rgcn.pkl" (
    echo ❌ 错误: RGCN嵌入文件不存在
    echo.
    echo 请先运行:
    echo   cd ..
    echo   python generate_hybrid_embeddings.py
    echo   cd rgcn
    echo   python prepare_embeddings.py
    pause
    exit /b 1
)
echo ✓ 嵌入文件存在
echo.

echo [2/4] 训练Siamese RGCN (预训练阶段)...
if not exist "rgcn_output\checkpoints\best_model.pth" (
    echo 预训练模型不存在，开始训练...
    python train_rgcn_hotpotqa.py
    if errorlevel 1 (
        echo ❌ Siamese RGCN训练失败
        pause
        exit /b 1
    )
) else (
    echo ✓ 预训练模型已存在，跳过
)
echo.

echo [3/4] 训练FFN分类器 (%MODE_NAME%模式)...
python train_pretrained_classifier.py %FREEZE_FLAG%
if errorlevel 1 (
    echo ❌ 分类器训练失败
    pause
    exit /b 1
)
echo ✓ 分类器训练完成
echo.

echo [4/4] 运行推理...
if "%MODE%"=="1" (
    set MODEL_PATH=rgcn_output\checkpoints\best_pretrained_classifier_frozen.pth
) else (
    set MODEL_PATH=rgcn_output\checkpoints\best_pretrained_classifier_finetuned.pth
)

python inference_classifier.py --model_path %MODEL_PATH%
if errorlevel 1 (
    echo ❌ 推理失败
    pause
    exit /b 1
)
echo ✓ 推理完成
echo.

echo ========================================
echo 全部完成！
echo ========================================
echo.
echo 生成的文件:
if "%MODE%"=="1" (
    echo   - rgcn_output\checkpoints\best_pretrained_classifier_frozen.pth
    echo   - rgcn_output\pretrained_classifier_frozen_curves.png
) else (
    echo   - rgcn_output\checkpoints\best_pretrained_classifier_finetuned.pth
    echo   - rgcn_output\pretrained_classifier_finetuned_curves.png
)
echo   - rgcn_output\classifier_predictions.json
echo   - rgcn_output\classifier_confusion_matrix.png
echo.

pause











