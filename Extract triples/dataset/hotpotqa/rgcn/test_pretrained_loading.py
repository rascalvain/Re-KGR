"""
测试预训练编码器加载（带详细诊断）
"""

import torch
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(__file__))

from classifier_with_pretrained import HallucinationClassifierWithPretrainedEncoder


def test_load_pretrained():
    """测试加载预训练模型"""
    
    print("="*60)
    print("测试预训练编码器加载")
    print("="*60)
    
    # 预训练模型路径
    pretrained_path = os.path.join(
        os.path.dirname(__file__),
        'rgcn_output',
        'checkpoints',
        'best_model.pth'
    )
    
    if not os.path.exists(pretrained_path):
        print(f"\n❌ 预训练模型不存在: {pretrained_path}")
        print(f"\n请先运行: python train_rgcn_hotpotqa.py")
        return False
    
    # 检查checkpoint内容
    print(f"\n1. 检查checkpoint内容...")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'config' in checkpoint:
        print(f"\n  ✓ Config存在:")
        config = checkpoint['config']
        if config:
            print(f"    hidden_channels: {config.get('hidden_channels', 'N/A')}")
            print(f"    out_channels: {config.get('out_channels', 'N/A')}")
            print(f"    num_layers: {config.get('num_layers', 'N/A')}")
            print(f"    num_relations: {config.get('num_relations', 'N/A')}")
        else:
            print(f"    ⚠ Config is None")
    else:
        print(f"  ⚠ Config不存在，将使用推断方法")
    
    # 检查模型state_dict中的层
    print(f"\n2. 检查模型结构...")
    model_state = checkpoint['model_state_dict']
    
    # 找所有conv层
    conv_layers = set()
    for key in model_state.keys():
        if 'encoder.convs.' in key and '.weight' in key:
            layer_idx = int(key.split('.')[2])
            conv_layers.add(layer_idx)
            shape = model_state[key].shape
            print(f"    {key}: {shape}")
    
    print(f"\n  发现 {len(conv_layers)} 个卷积层: {sorted(conv_layers)}")
    
    # 测试加载
    print(f"\n3. 尝试加载模型...")
    try:
        model = HallucinationClassifierWithPretrainedEncoder(
            pretrained_model_path=pretrained_path,
            freeze_encoder=True,
            ffn_hidden_dim=128,
            dropout=0.3
        )
        
        print(f"\n✓ 模型加载成功！")
        
        # 打印参数统计
        params = model.get_trainable_params()
        print(f"\n参数统计:")
        print(f"  总参数: {params['total']:,}")
        print(f"  可训练参数: {params['trainable']:,}")
        print(f"  编码器: {params['encoder']:,} (可训练: {params['encoder_trainable']:,})")
        print(f"  分类器: {params['classifier']:,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 模型加载失败:")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_load_pretrained()
    
    if success:
        print("\n" + "="*60)
        print("✓ 测试通过！可以使用预训练编码器")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ 测试失败！请检查上面的错误信息")
        print("="*60)











