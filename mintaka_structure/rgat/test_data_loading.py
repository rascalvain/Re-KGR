"""
测试修复后的数据加载
"""

from config_mintaka_rgat import Config
from data_loader_mintaka import MintakaGraphDataset

def test_data_loading():
    print("=" * 60)
    print("测试数据加载")
    print("=" * 60)

    # 加载数据集（只测试前20个样本）
    print(f"\n[1] 加载数据集...")
    print(f"    数据文件: {Config.DATA_PATH}")

    dataset = MintakaGraphDataset(
        Config.DATA_PATH,
        Config.ENTITY2IDX_PATH,
        Config.RELATION2IDX_PATH,
        max_samples=20
    )

    print(f"    有效样本数: {len(dataset)}")

    # 测试前10个样本
    print(f"\n[2] 测试前10个样本...")

    success_count = 0
    graph_invalid_count = 0
    complete_fail_count = 0

    for i in range(min(10, len(dataset))):
        sample = dataset[i]

        if sample is None:
            complete_fail_count += 1
            print(f"    样本 {i}: 完全失败 ❌")
        elif isinstance(sample, tuple) and len(sample) == 4:
            entity_graph, gpt_graph, label, metadata = sample

            if entity_graph is not None and gpt_graph is not None:
                success_count += 1
                print(f"    样本 {i}: 成功 ✓")
                print(f"             entity图: {entity_graph.num_nodes}节点, {entity_graph.edge_index.size(1)}边")
                print(f"             gpt图: {gpt_graph.num_nodes}节点, {gpt_graph.edge_index.size(1)}边")
                print(f"             标签: {label} ({'幻觉' if label == 0 else '正确'})")
            else:
                graph_invalid_count += 1
                print(f"    样本 {i}: 图无效但有标签 ⚠️")
                print(f"             entity图: {'None' if entity_graph is None else 'OK'}")
                print(f"             gpt图: {'None' if gpt_graph is None else 'OK'}")
                print(f"             标签: {label}")
                print(f"             entity三元组数: {metadata['num_entity_triples']}")
                print(f"             gpt三元组数: {metadata['num_gpt_triples']}")

    print(f"\n[3] 统计:")
    print(f"    完全成功: {success_count}/10")
    print(f"    图无效: {graph_invalid_count}/10")
    print(f"    完全失败: {complete_fail_count}/10")

    if success_count >= 8:
        print(f"\n✓ 数据加载成功！可以正常训练")
        return True
    elif success_count >= 5:
        print(f"\n⚠️  数据加载部分成功，训练可能受影响")
        return True
    else:
        print(f"\n❌ 数据加载失败，无法训练")
        return False


if __name__ == '__main__':
    success = test_data_loading()
    exit(0 if success else 1)
