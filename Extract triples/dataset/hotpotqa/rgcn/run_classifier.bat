@echo off
chcp 65001 > nul
echo ========================================
echo FFN分类器 - 完整训练和推理流程
echo ========================================
echo.

echo [1/3] 检查嵌入文件...
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

echo [2/3] 训练FFN分类器...
python train_classifier.py
if errorlevel 1 (
    echo ❌ 训练失败
    pause
    exit /b 1
)
echo ✓ 训练完成
echo.

echo [3/3] 运行推理...
python inference_classifier.py
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
echo 查看结果文件:
echo   - rgcn_output\best_classifier.pth (最佳模型)
echo   - rgcn_output\classifier_predictions.json (预测结果)
echo   - rgcn_output\classifier_confusion_matrix.png (混淆矩阵)
echo   - rgcn_output\classifier_probability_distribution.png (概率分布)
echo.

pause











