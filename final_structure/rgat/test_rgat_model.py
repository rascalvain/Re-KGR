"""
RGAT模型测试脚本
用于验证模型是否能正常初始化和运行
"""

import torch
import sys
import os

# 导入RGAT专用配置
from config_hotpotqa_rgat import Config
from siamese_rgat_improved import SiameseRGATWithEmbedding, ImprovedContrastiveLoss
from torch_geometric.data import Data


def test_rgat_initialization():
    """测试RGAT模型初始化"""
    print("\n" + "="*60)
    print("测试1: RGAT模型初始化")
    print("="*60)
    
    try:
        config_dict = Config.get_config_dict()
        
        # 检查嵌入文件
        if not os.path.exists(config_dict['entity_embedding_path']):
            print(f"❌ 实体嵌入文件不存在: {config_dict['entity_embedding_path']}")
            print("\n请先运行:")
            print("  cd ../rgcn")
            print("  python prepare_embeddings.py")
            return False
        
        if not os.path.exists(config_dict['relation_embedding_path']):
            print(f"❌ 关系映射文件不存在: {config_dict['relation_embedding_path']}")
            print("\n请先运行:")
            print("  cd ../rgcn")
            print("  python prepare_embeddings.py")
            return False
        
        # 初始化模型
        print("\n正在初始化RGAT模型...")
        model = SiameseRGATWithEmbedding(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            hidden_channels=config_dict['hidden_channels'],
            out_channels=config_dict['out_channels'],
            num_layers=config_dict['num_layers'],
            freeze_embeddings=config_dict.get('freeze_embeddings', True),
            dropout=config_dict.get('dropout', 0.3),
            num_heads=config_dict.get('num_heads', 4)
        )
        
        print(f"\n✓ RGAT模型初始化成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  冻结参数: {total_params - trainable_params:,}")
        
        return True, model, config_dict
        
    except Exception as e:
        print(f"\n❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_rgat_forward():
    """测试RGAT前向传播"""
    print("\n" + "="*60)
    print("测试2: RGAT前向传播")
    print("="*60)
    
    success, model, config_dict = test_rgat_initialization()
    if not success:
        return False
    
    try:
        # 创建虚拟图数据
        print("\n创建测试数据...")
        
        # 图1: 5个节点，7条边
        node_ids_1 = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        edge_index_1 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 4],
            [1, 0, 2, 1, 3, 2, 3]
        ], dtype=torch.long)
        edge_type_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.long)
        batch_1 = torch.zeros(5, dtype=torch.long)
        
        # 图2: 6个节点，8条边
        node_ids_2 = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.long)
        edge_index_2 = torch.tensor([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3]
        ], dtype=torch.long)
        edge_type_2 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        batch_2 = torch.zeros(6, dtype=torch.long)
        
        data_1 = Data(
            node_ids=node_ids_1,
            edge_index=edge_index_1,
            edge_type=edge_type_1,
            batch=batch_1
        )
        
        data_2 = Data(
            node_ids=node_ids_2,
            edge_index=edge_index_2,
            edge_type=edge_type_2,
            batch=batch_2
        )
        
        print("  图1: 5个节点, 7条边")
        print("  图2: 6个节点, 8条边")
        
        # 前向传播
        print("\n执行前向传播...")
        model.eval()
        with torch.no_grad():
            emb_1, emb_2 = model(data_1, data_2)
        
        print(f"\n✓ 前向传播成功")
        print(f"  图1嵌入形状: {emb_1.shape}")
        print(f"  图2嵌入形状: {emb_2.shape}")
        print(f"  嵌入范围: [{emb_1.min().item():.4f}, {emb_1.max().item():.4f}]")
        
        # 测试损失函数
        print("\n测试损失函数...")
        criterion = ImprovedContrastiveLoss(
            margin=config_dict['margin'],
            temperature=config_dict['temperature'],
            alpha=config_dict['alpha']
        )
        
        labels = torch.tensor([1], dtype=torch.long)  # 1表示事实
        loss, loss_dict = criterion(emb_1, emb_2, labels)
        
        print(f"✓ 损失计算成功")
        print(f"  总损失: {loss.item():.4f}")
        print(f"  对比损失: {loss_dict['contrastive_loss']:.4f}")
        print(f"  相似度: {loss_dict['avg_similarity']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_saving():
    """测试模型保存和加载"""
    print("\n" + "="*60)
    print("测试3: 模型保存和加载")
    print("="*60)
    
    success, model, config_dict = test_rgat_initialization()
    if not success:
        return False
    
    try:
        import tempfile
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        print(f"\n保存模型到临时文件: {tmp_path}")
        
        # 保存模型
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config_dict
        }
        torch.save(checkpoint, tmp_path)
        print("✓ 模型保存成功")
        
        # 加载模型
        print("\n加载模型...")
        checkpoint = torch.load(tmp_path, map_location='cpu')
        
        new_model = SiameseRGATWithEmbedding(
            entity_embedding_path=config_dict['entity_embedding_path'],
            relation_embedding_path=config_dict['relation_embedding_path'],
            hidden_channels=config_dict['hidden_channels'],
            out_channels=config_dict['out_channels'],
            num_layers=config_dict['num_layers'],
            freeze_embeddings=config_dict.get('freeze_embeddings', True),
            dropout=config_dict.get('dropout', 0.3),
            num_heads=config_dict.get('num_heads', 4)
        )
        
        new_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ 模型加载成功")
        
        # 清理临时文件
        os.remove(tmp_path)
        print(f"✓ 清理临时文件")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 模型保存/加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*70)
    print("RGAT模型测试套件")
    print("="*70)
    
    results = []
    
    # 测试1: 模型初始化
    results.append(("模型初始化", test_rgat_initialization()[0] if isinstance(test_rgat_initialization(), tuple) else test_rgat_initialization()))
    
    # 测试2: 前向传播
    results.append(("前向传播", test_rgat_forward()))
    
    # 测试3: 模型保存和加载
    results.append(("模型保存和加载", test_model_saving()))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！RGAT模型可以正常使用。")
        print("\n下一步:")
        print("  python train_rgat_hotpotqa.py  # 开始训练")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")
    
    print("="*70)


if __name__ == '__main__':
    main()

