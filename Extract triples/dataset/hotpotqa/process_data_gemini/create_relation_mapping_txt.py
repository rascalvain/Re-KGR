import json
import os

def create_relation_mapping_txt_from_json(json_file, output_txt_file):
    """
    从JSON映射文件创建TXT格式的关系词映射
    格式: original_relation \t cluster_representative
    """
    print(f"正在加载JSON映射文件: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        cluster_mapping = json.load(f)
    
    print(f"找到 {len(cluster_mapping)} 个聚类簇")
    
    # 创建反向映射：从每个关系词到其代表词
    relation_to_cluster = {}
    for cluster_representative, relations in cluster_mapping.items():
        for rel in relations:
            relation_to_cluster[rel] = cluster_representative
    
    print(f"共有 {len(relation_to_cluster)} 个关系词")
    
    # 写入txt文件
    print(f"正在保存到: {output_txt_file}")
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        # 按关系词排序输出
        for relation in sorted(relation_to_cluster.keys()):
            cluster_rep = relation_to_cluster[relation]
            f.write(f"{relation}\t{cluster_rep}\n")
    
    print(f"\n✓ 映射文件已保存!")
    print(f"✓ 总共 {len(relation_to_cluster)} 个关系词映射")
    
    # 显示一些示例
    print(f"\n示例映射（前10个）:")
    for i, (relation, cluster_rep) in enumerate(sorted(relation_to_cluster.items())[:10]):
        if relation == cluster_rep:
            print(f"  {relation} -> {cluster_rep} (代表词)")
        else:
            print(f"  {relation} -> {cluster_rep}")
    
    return relation_to_cluster


def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 文件路径
    json_file = os.path.join(script_dir, 'data/relation_cluster_mapping.json')
    output_txt_file = os.path.join(script_dir, 'data/relation_to_cluster_mapping.txt')
    
    print("=" * 70)
    print("从JSON映射创建TXT格式的关系词映射")
    print("=" * 70)
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_file):
        print(f"\n错误: 找不到文件 {json_file}")
        print("请先运行聚类脚本 cluster_relations.py 生成映射文件")
        return
    
    # 创建映射
    relation_to_cluster = create_relation_mapping_txt_from_json(json_file, output_txt_file)
    
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

