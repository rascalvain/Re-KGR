"""
RGAT快速启动脚本
用于检查环境和依赖是否准备就绪
"""

import os
import sys


def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_python_packages():
    """检查Python包"""
    print_header("步骤1: 检查Python包")
    
    required_packages = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name:25s} 已安装")
        except ImportError:
            print(f"✗ {name:25s} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少以下包: {', '.join(missing_packages)}")
        print("\n安装命令:")
        print("  pip install torch torch-geometric numpy tqdm matplotlib")
        return False
    
    print("\n✅ 所有必需的包已安装")
    return True


def check_config_file():
    """检查配置文件"""
    print_header("步骤2: 检查配置文件")
    
    config_file = os.path.join(os.path.dirname(__file__), 'config_hotpotqa_rgat.py')
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    print(f"✓ 配置文件: config_hotpotqa_rgat.py")
    
    # 导入配置
    try:
        from config_hotpotqa_rgat import Config
        print(f"✓ 配置导入成功")
        
        # 显示关键配置
        print(f"\n关键配置:")
        print(f"  数据路径: {Config.DATA_DIR}")
        print(f"  输出目录: {Config.OUTPUT_DIR}")
        print(f"  注意力头数: {Config.NUM_HEADS}")
        
        return True
    except Exception as e:
        print(f"❌ 配置导入失败: {e}")
        return False


def check_embedding_files():
    """检查嵌入文件"""
    print_header("步骤3: 检查嵌入文件")
    
    from config_hotpotqa_rgat import Config
    
    files_to_check = {
        '实体嵌入': Config.ENTITY_EMBEDDING_RGAT_PATH,
        '关系映射': Config.RELATION_MAPPING_RGAT_PATH,
        '实体索引': Config.ENTITY2IDX_PATH,
        '关系索引': Config.RELATION2IDX_PATH
    }
    
    all_exist = True
    
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ {name:12s} ({size_mb:.2f} MB)")
        else:
            print(f"✗ {name:12s} 不存在")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 部分嵌入文件缺失")
        print("\n生成嵌入文件:")
        print("  cd ../rgcn")
        print("  python prepare_embeddings.py")
        return False
    
    print("\n✅ 所有嵌入文件已准备就绪")
    return True


def check_data_file():
    """检查数据文件"""
    print_header("步骤4: 检查数据文件")
    
    from config_hotpotqa_rgat import Config
    
    if not os.path.exists(Config.HOTPOTQA_DATA_PATH):
        print(f"❌ 数据文件不存在:")
        print(f"   {Config.HOTPOTQA_DATA_PATH}")
        return False
    
    size_mb = os.path.getsize(Config.HOTPOTQA_DATA_PATH) / (1024 * 1024)
    print(f"✓ 数据文件: {os.path.basename(Config.HOTPOTQA_DATA_PATH)}")
    print(f"  大小: {size_mb:.2f} MB")
    
    # 尝试加载数据
    try:
        import json
        with open(Config.HOTPOTQA_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"  样本数: {len(data)}")
        elif isinstance(data, dict):
            print(f"  键数: {len(data)}")
        
        print("\n✅ 数据文件有效")
        return True
    except Exception as e:
        print(f"\n❌ 数据文件读取失败: {e}")
        return False


def check_model_files():
    """检查模型文件"""
    print_header("步骤5: 检查模型文件")
    
    model_files = [
        'siamese_rgat_improved.py',
        'train_rgat_hotpotqa.py',
        'classifier_with_pretrained_rgat.py',
        'test_rgat_model.py'
    ]
    
    all_exist = True
    
    for filename in model_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            print(f"✓ {filename}")
        else:
            print(f"✗ {filename} 不存在")
            all_exist = False
    
    if not all_exist:
        print("\n❌ 部分模型文件缺失")
        return False
    
    print("\n✅ 所有模型文件已准备就绪")
    return True


def check_data_loader():
    """检查数据加载器"""
    print_header("步骤6: 检查数据加载器")
    
    rgcn_dir = os.path.join(os.path.dirname(__file__), '..', 'rgcn')
    data_loader_path = os.path.join(rgcn_dir, 'data_loader_hotpotqa.py')
    
    if not os.path.exists(data_loader_path):
        print(f"❌ 数据加载器不存在:")
        print(f"   {data_loader_path}")
        return False
    
    print(f"✓ 数据加载器: ../rgcn/data_loader_hotpotqa.py")
    
    # 尝试导入
    try:
        sys.path.insert(0, os.path.abspath(rgcn_dir))
        from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn
        print(f"✓ 数据加载器导入成功")
        print("\n✅ 数据加载器可用")
        return True
    except Exception as e:
        print(f"\n❌ 数据加载器导入失败: {e}")
        return False


def print_summary(results):
    """打印总结"""
    print_header("检查总结")
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    
    if all_passed:
        print("\n🎉 所有检查通过！环境已准备就绪。")
        print("\n📝 下一步:")
        print("  1. 运行测试: python test_rgat_model.py")
        print("  2. 开始训练: python train_rgat_hotpotqa.py")
        print("\n📚 更多信息: 查看 README_RGAT.md")
    else:
        print("\n⚠️ 部分检查失败，请根据上述提示解决问题。")
    
    print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("  RGAT 环境检查工具")
    print("="*70)
    
    results = []
    
    # 执行所有检查
    results.append(("Python包", check_python_packages()))
    results.append(("配置文件", check_config_file()))
    results.append(("嵌入文件", check_embedding_files()))
    results.append(("数据文件", check_data_file()))
    results.append(("模型文件", check_model_files()))
    results.append(("数据加载器", check_data_loader()))
    
    # 打印总结
    print_summary(results)


if __name__ == '__main__':
    main()

