@echo off
REM 幻觉检测推理脚本

echo ========================================
echo HotpotQA 幻觉检测推理
echo ========================================
echo.

REM 检查模型是否存在
if not exist "rgcn_output\checkpoints\best_model.pth" (
    echo ❌ 错误: 模型文件不存在
    echo.
    echo 请先训练模型:
    echo   python train_rgcn_hotpotqa.py
    echo.
    pause
    exit /b 1
)

echo ✓ 模型文件检查通过
echo.

REM 运行推理
echo [1/1] 运行幻觉检测推理...
echo.
python inference_hotpotqa.py

if %errorlevel% neq 0 (
    echo.
    echo ✗ 推理失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
echo 生成的文件:
echo   - rgcn_output/hallucination_predictions.json
echo   - rgcn_output/evaluation_metrics.json
echo   - rgcn_output/confusion_matrix.png
echo   - rgcn_output/similarity_distribution.png
echo   - rgcn_output/threshold_f1_curve.png
echo.

pause











