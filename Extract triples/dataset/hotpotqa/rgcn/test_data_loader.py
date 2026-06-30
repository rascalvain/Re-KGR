"""
测试数据加载器
"""

from config_hotpotqa import Config
from data_loader_hotpotqa import HotpotQAGraphDataset, collate_fn
from torch.utils.data import DataLoader


def test_data_loader():
    """测试数据加载器"""
    print("="*60)
    print("测试 HotpotQA 数据加载器")
    print("="*60)
    
    # 检查文件
    import os
    files_to_check = {
        '数据文件': Config.HOTPOTQA_DATA_PATH,
        '实体映射': Config.ENTITY2IDX_PATH,
        '关系映射': Config.RELATION2IDX_PATH
    }
    
    print("\n[1] 检查文件:")
    all_exist = True
    for name, path in files_to_check.items():
        if os.path.exists(path):
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: {path} (不存在)")
            all_exist = False
    
    if not all_exist:
        print("\n错误: 部分文件不存在，请先运行相关脚本")
        return False
    
    # 创建数据集
    print("\n[2] 加载数据集 (前10个样本):")
    try:
        dataset = HotpotQAGraphDataset(
            Config.HOTPOTQA_DATA_PATH,
            Config.ENTITY2IDX_PATH,
            Config.RELATION2IDX_PATH,
            max_samples=10
        )
    except Exception as e:
        print(f"  ✗ 数据集加载失败: {e}")
        return False
    
    if len(dataset) == 0:
        print("  ✗ 没有有效样本")
        return False
    
    print(f"  ✓ 数据集大小: {len(dataset)}")
    
    # 测试获取样本
    print("\n[3] 测试获取样本:")
    try:
        context_graph, gpt_graph, label, metadata = dataset[0]
        
        print(f"  Context 图:")
        print(f"    - 节点数: {context_graph.num_nodes}")
        print(f"    - 边数: {context_graph.edge_index.shape[1]}")
        print(f"    - 节点ID范围: [{context_graph.node_ids.min()}, {context_graph.node_ids.max()}]")
        print(f"    - 边类型数: {len(context_graph.edge_type.unique())}")
        
        print(f"  GPT 图:")
        print(f"    - 节点数: {gpt_graph.num_nodes}")
        print(f"    - 边数: {gpt_graph.edge_index.shape[1]}")
        print(f"    - 节点ID范围: [{gpt_graph.node_ids.min()}, {gpt_graph.node_ids.max()}]")
        print(f"    - 边类型数: {len(gpt_graph.edge_type.unique())}")
        
        print(f"  标签: {label}")
        print(f"  问题: {metadata['question'][:80]}...")
        print(f"  Context三元组数: {metadata['num_context_triples']}")
        print(f"  GPT三元组数: {metadata['num_gpt_triples']}")
        
    except Exception as e:
        print(f"  ✗ 获取样本失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试批处理
    print("\n[4] 测试批处理:")
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        for context_batch, gpt_batch, labels, metadata_list in dataloader:
            if context_batch is not None:
                print(f"  ✓ Batch size: {len(labels)}")
                print(f"  Context batch:")
                print(f"    - 图数: {context_batch.num_graphs}")
                print(f"    - 总节点数: {context_batch.num_nodes}")
                print(f"    - 总边数: {context_batch.edge_index.shape[1]}")
                print(f"  GPT batch:")
                print(f"    - 图数: {gpt_batch.num_graphs}")
                print(f"    - 总节点数: {gpt_batch.num_nodes}")
                print(f"    - 总边数: {gpt_batch.edge_index.shape[1]}")
                break
        else:
            print("  ✗ 没有有效batch")
            return False
            
    except Exception as e:
        print(f"  ✗ 批处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试完整数据集
    print("\n[5] 测试完整数据集:")
    try:
        full_dataset = HotpotQAGraphDataset(
            Config.HOTPOTQA_DATA_PATH,
            Config.ENTITY2IDX_PATH,
            Config.RELATION2IDX_PATH,
            max_samples=None  # 加载全部
        )
        print(f"  ✓ 完整数据集大小: {len(full_dataset)}")
        
        # 统计
        total_context_nodes = 0
        total_gpt_nodes = 0
        total_context_edges = 0
        total_gpt_edges = 0
        
        for i in range(len(full_dataset)):
            context_graph, gpt_graph, _, _ = full_dataset[i]
            total_context_nodes += context_graph.num_nodes
            total_gpt_nodes += gpt_graph.num_nodes
            total_context_edges += context_graph.edge_index.shape[1]
            total_gpt_edges += gpt_graph.edge_index.shape[1]
        
        print(f"  统计信息:")
        print(f"    - Context 平均节点数: {total_context_nodes / len(full_dataset):.1f}")
        print(f"    - GPT 平均节点数: {total_gpt_nodes / len(full_dataset):.1f}")
        print(f"    - Context 平均边数: {total_context_edges / len(full_dataset):.1f}")
        print(f"    - GPT 平均边数: {total_gpt_edges / len(full_dataset):.1f}")
        
    except Exception as e:
        print(f"  ✗ 完整数据集测试失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ 数据加载器测试通过！")
    print("="*60)
    return True


if __name__ == '__main__':
    import sys
    success = test_data_loader()
    sys.exit(0 if success else 1)

