@echo off
REM HotpotQA RGCN 一键运行脚本

echo ========================================
echo HotpotQA RGCN 完整流程
echo ========================================
echo.

REM 步骤1: 准备嵌入
echo [步骤 1/3] 准备RGCN嵌入文件...
echo.
python prepare_embeddings.py
if %errorlevel% neq 0 (
    echo.
    echo ✗ 嵌入准备失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo.

REM 步骤2: 测试数据加载器
echo [步骤 2/3] 测试数据加载器...
echo.
python test_data_loader.py
if %errorlevel% neq 0 (
    echo.
    echo ✗ 数据加载器测试失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo.

REM 步骤3: 训练RGCN
echo [步骤 3/3] 训练RGCN模型...
echo.
python train_rgcn_hotpotqa.py
if %errorlevel% neq 0 (
    echo.
    echo ✗ 训练失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 完成！
echo ========================================
echo.
echo 生成的文件:
echo   - rgcn_output/checkpoints/best_model.pth
echo   - rgcn_output/training_curves.png
echo   - rgcn_output/training_history.json
echo.

pause

