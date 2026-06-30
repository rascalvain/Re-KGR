#!/bin/bash
# 修复Windows行尾符问题

echo "=================================="
echo "修复Shell脚本行尾符"
echo "=================================="

cd "$(dirname "$0")"

# 检查是否有dos2unix
if command -v dos2unix &> /dev/null; then
    echo "使用 dos2unix 转换..."
    dos2unix *.sh
else
    echo "使用 sed 转换..."
    for file in *.sh; do
        if [ -f "$file" ]; then
            sed -i 's/\r$//' "$file"
            echo "已转换: $file"
        fi
    done
fi

# 赋予执行权限
echo ""
echo "设置执行权限..."
chmod +x *.sh

echo ""
echo "=================================="
echo "修复完成！"
echo "=================================="
echo "现在可以运行脚本了，例如："
echo "  bash train_rgat.sh"
echo "  或"
echo "  ./train_rgat.sh"
