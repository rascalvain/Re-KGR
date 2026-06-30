"""
测试 OpenKE 安装和版本兼容性
"""
import sys

def test_openke_installation():
    """测试 OpenKE 是否正确安装"""
    print("="*60)
    print("测试 OpenKE 安装")
    print("="*60)
    
    # 测试 1: 导入 OpenKE
    print("\n[测试 1] 导入 OpenKE 模块...")
    try:
        import openke
        print("✓ OpenKE 模块导入成功")
    except ImportError as e:
        print(f"✗ OpenKE 模块导入失败: {e}")
        print("\n请按照以下步骤安装 OpenKE:")
        print("  1. git clone https://github.com/thunlp/OpenKE")
        print("  2. cd OpenKE")
        print("  3. bash make.sh")
        print("  4. python setup.py install")
        return False
    
    # 测试 2: 导入主要组件
    print("\n[测试 2] 导入 OpenKE 组件...")
    try:
        from openke.config import Trainer, Tester
        from openke.module.model import TransE
        from openke.module.loss import MarginLoss
        from openke.module.strategy import NegativeSampling
        from openke.data import TrainDataLoader
        print("✓ OpenKE 组件导入成功")
    except ImportError as e:
        print(f"✗ OpenKE 组件导入失败: {e}")
        return False
    
    # 测试 3: 检查 Trainer 支持的参数
    print("\n[测试 3] 检查 Trainer 支持的参数...")
    import inspect
    try:
        sig = inspect.signature(Trainer.__init__)
        params = list(sig.parameters.keys())
        print(f"✓ Trainer 支持的参数: {params}")
        
        # 检查常用参数
        optional_params = ['save_steps', 'checkpoint_dir', 'patient', 'save_num']
        supported = []
        not_supported = []
        
        for param in optional_params:
            if param in params:
                supported.append(param)
            else:
                not_supported.append(param)
        
        if supported:
            print(f"  支持的可选参数: {supported}")
        if not_supported:
            print(f"  不支持的参数: {not_supported}")
            print(f"  注意: 训练脚本会自动适配您的 OpenKE 版本")
            
    except Exception as e:
        print(f"✗ 无法检查参数: {e}")
    
    # 测试 4: 检查 PyTorch
    print("\n[测试 4] 检查 PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用 (GPU训练)")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA 不可用 (将使用 CPU 训练)")
    except ImportError:
        print("✗ PyTorch 未安装")
        print("  请安装: pip install torch")
        return False
    
    # 测试 5: 检查其他依赖
    print("\n[测试 5] 检查其他依赖...")
    dependencies = {
        'numpy': 'numpy',
        'pickle': 'pickle (内置)',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} 未安装")
            all_ok = False
    
    # 总结
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    
    if all_ok:
        print("✓ 所有测试通过！可以开始训练了。")
        print("\n运行训练:")
        print("  python train_transe.py --prepare_data")
        return True
    else:
        print("✗ 部分测试失败，请检查安装。")
        return False

if __name__ == '__main__':
    success = test_openke_installation()
    sys.exit(0 if success else 1)

