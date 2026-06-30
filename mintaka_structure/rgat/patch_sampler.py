"""
临时补丁：修复 EpochBalancedSampler 处理 Subset 的问题
在服务器上运行此脚本，自动修补采样器
"""

import os
import shutil

def patch_sampler():
    """修补采样器文件"""

    current_file = "EpochBalancedSampler.py"
    backup_file = "EpochBalancedSampler.py.backup"
    fixed_file = "EpochBalancedSampler_fixed.py"

    print("=" * 60)
    print("修补 EpochBalancedSampler")
    print("=" * 60)

    # 检查文件
    if not os.path.exists(current_file):
        print(f"❌ 错误: 找不到 {current_file}")
        return False

    if not os.path.exists(fixed_file):
        print(f"❌ 错误: 找不到修复文件 {fixed_file}")
        print("   请先上传 EpochBalancedSampler_fixed.py")
        return False

    # 备份原文件
    if not os.path.exists(backup_file):
        print(f"\n备份原文件...")
        shutil.copy2(current_file, backup_file)
        print(f"✓ 已备份到: {backup_file}")
    else:
        print(f"\n备份文件已存在: {backup_file}")

    # 替换文件
    print(f"\n替换文件...")
    shutil.copy2(fixed_file, current_file)
    print(f"✓ 已将 {fixed_file} 复制为 {current_file}")

    # 验证
    print(f"\n验证文件...")
    with open(current_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'scan_dataset_labels' in content:
            print("✓ 修复成功：检测到新函数 scan_dataset_labels")
        else:
            print("❌ 警告：未检测到预期的修复")
            return False

    print("\n" + "=" * 60)
    print("✓ 修补完成！")
    print("=" * 60)
    print("\n现在可以重新运行训练：")
    print("  bash train_rgat_background.sh")
    print("\n如需恢复原文件：")
    print(f"  cp {backup_file} {current_file}")

    return True


if __name__ == '__main__':
    success = patch_sampler()
    exit(0 if success else 1)
