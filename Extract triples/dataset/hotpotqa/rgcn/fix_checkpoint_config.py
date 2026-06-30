"""
自动修复缺少config的checkpoint
"""

import torch
import os
import pickle


def fix_checkpoint_config():
    """自动修复checkpoint的config"""
    
    print("="*60)
    print("自动修复Checkpoint Config")
    print("="*60)
    
    # Checkpoint路径
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        'rgcn_output',
        'checkpoints',
        'best_model.pth'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint不存在: {checkpoint_path}")
        return False
    
    # 加载checkpoint
    print(f"\n1. 加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    
    # 检查是否已有config
    if 'config' in checkpoint and checkpoint['config'] is not None:
        config = checkpoint['config']
        if all(k in config for k in ['hidden_channels', 'out_channels', 'num_layers']):
            print(f"\n✓ Config完整，无需修复:")
            print(f"  hidden_channels: {config['hidden_channels']}")
            print(f"  out_channels: {config['out_channels']}")
            print(f"  num_layers: {config['num_layers']}")
            return True
    
    print(f"\n⚠ Config缺失或不完整，开始修复...")
    
    # 从model_state_dict推断配置
    print(f"\n2. 从模型结构推断配置...")
    model_state = checkpoint['model_state_dict']
    
    # 找所有conv层
    conv_layers = []
    for key in model_state.keys():
        if 'encoder.convs.' in key and '.weight' in key:
            layer_idx = int(key.split('.')[2])
            shape = model_state[key].shape
            conv_layers.append((layer_idx, key, shape))
    
    conv_layers.sort(key=lambda x: x[0])
    
    if not conv_layers:
        print(f"  ❌ 无法找到卷积层")
        return False
    
    print(f"\n  发现 {len(conv_layers)} 个卷积层:")
    for idx, key, shape in conv_layers:
        print(f"    Layer {idx}: {key} → {shape}")
    
    # 推断配置
    # convs.*.weight shape: [num_relations, in_features, out_features]
    num_layers = len(conv_layers)
    num_relations = conv_layers[0][2][0]  # 第一层的shape[0]
    hidden_channels = conv_layers[0][2][2]  # 第一层的输出维度shape[2]
    out_channels = conv_layers[-1][2][2]  # 最后一层的输出维度shape[2]
    
    print(f"\n  推断的配置:")
    print(f"    num_layers: {num_layers}")
    print(f"    num_relations: {num_relations}")
    print(f"    hidden_channels: {hidden_channels}")
    print(f"    out_channels: {out_channels}")
    
    # 从config_hotpotqa.py获取其他配置（如果可能）
    try:
        from config_hotpotqa import Config
        config_dict = Config.get_config_dict()
        dropout = config_dict.get('dropout', 0.3)
        freeze_embeddings = config_dict.get('freeze_embeddings', True)
        print(f"    dropout: {dropout}")
        print(f"    freeze_embeddings: {freeze_embeddings}")
    except:
        dropout = 0.3
        freeze_embeddings = True
        print(f"    dropout: {dropout} (默认)")
        print(f"    freeze_embeddings: {freeze_embeddings} (默认)")
    
    # 创建config
    new_config = {
        'hidden_channels': int(hidden_channels),
        'out_channels': int(out_channels),
        'num_layers': int(num_layers),
        'num_relations': int(num_relations),
        'dropout': dropout,
        'freeze_embeddings': freeze_embeddings,
    }
    
    # 备份原checkpoint
    backup_path = checkpoint_path + '.backup'
    if not os.path.exists(backup_path):
        print(f"\n3. 备份原checkpoint...")
        torch.save(checkpoint, backup_path)
        print(f"  ✓ 备份到: {backup_path}")
    
    # 添加config到checkpoint
    print(f"\n4. 添加config到checkpoint...")
    checkpoint['config'] = new_config
    
    # 保存修复后的checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"  ✓ 已保存修复后的checkpoint")
    
    # 验证
    print(f"\n5. 验证修复...")
    checkpoint_new = torch.load(checkpoint_path, map_location='cpu')
    if 'config' in checkpoint_new and checkpoint_new['config'] is not None:
        print(f"  ✓ Config已成功添加:")
        for key, value in checkpoint_new['config'].items():
            print(f"    {key}: {value}")
        return True
    else:
        print(f"  ❌ 验证失败")
        return False


def main():
    """主函数"""
    print("\n此脚本会自动修复缺少config的checkpoint")
    print("如果checkpoint已有完整config，则无需修复\n")
    
    input("按Enter继续...")
    
    success = fix_checkpoint_config()
    
    if success:
        print("\n" + "="*60)
        print("✓ 修复成功！")
        print("="*60)
        print("\n现在可以运行:")
        print("  python test_pretrained_loading.py")
        print("  或")
        print("  python train_pretrained_classifier.py")
    else:
        print("\n" + "="*60)
        print("✗ 修复失败！")
        print("="*60)
        print("\n请查看上面的错误信息")
        print("或参考: PRETRAINED_LOADING_TROUBLESHOOTING.md")


if __name__ == '__main__':
    main()











