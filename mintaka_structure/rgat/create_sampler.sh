#!/bin/bash
# 在服务器上创建 EpochBalancedSampler.py

cd /root/autodl-fs/gca/mintaka/mintaka_structure/rgat

echo "创建 EpochBalancedSampler.py..."

# 从 hotpotqa 版本复制（如果存在）
if [ -f "/root/autodl-fs/gca/mintaka/final_structure/rgat/EpochBalancedSampler.py" ]; then
    echo "从 final_structure 复制..."
    cp /root/autodl-fs/gca/mintaka/final_structure/rgat/EpochBalancedSampler.py ./
    echo "✓ 文件已复制"
else
    echo "错误: 找不到源文件，请手动上传 EpochBalancedSampler.py"
    exit 1
fi

# 验证文件
if [ -f "EpochBalancedSampler.py" ]; then
    echo "✓ EpochBalancedSampler.py 创建成功"
    ls -lh EpochBalancedSampler.py
else
    echo "✗ 创建失败"
    exit 1
fi

echo ""
echo "现在可以重新运行训练："
echo "  bash train_rgat_background.sh"
