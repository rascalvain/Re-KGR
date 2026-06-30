"""
生成 TransE 训练所需的 train2id.txt 文件
"""
import os


def load_entity2id(entity_file):
    """加载实体到ID的映射"""
    entity2id = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式：entity_name\tid
            parts = line.split('\t')
            if len(parts) == 2:
                entity_name = parts[0].strip()
                entity_id = parts[1].strip()
                entity2id[entity_name] = entity_id
    return entity2id


def load_relation2id(relation_file):
    """加载关系到ID的映射"""
    relation2id = {}
    with open(relation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式：relation_name\tid
            parts = line.split('\t')
            if len(parts) == 2:
                relation_name = parts[0].strip()
                relation_id = parts[1].strip()
                relation2id[relation_name] = relation_id
    return relation2id


def generate_train2id(entity2id_file, relation2id_file, triples_file, output_file):
    """生成 train2id.txt 文件"""

    print("🔄 加载 entity2id 映射...")
    entity2id = load_entity2id(entity2id_file)
    print(f"✓ 加载完成: {len(entity2id)} 个实体\n")

    print("🔄 加载 relation2id 映射...")
    relation2id = load_relation2id(relation2id_file)
    print(f"✓ 加载完成: {len(relation2id)} 个关系\n")

    print("🔄 读取 triples.txt 并转换为 ID 格式...")
    train_triples = []
    skipped_count = 0

    with open(triples_file, 'r', encoding='utf-8') as f:
        # 跳过表头
        header = f.readline()
        print(f"表头: {header.strip()}\n")

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue

            # 格式：head\ttail\trelation
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"⚠️  行 {line_num}: 格式错误，跳过")
                skipped_count += 1
                continue

            head, tail, relation = parts[0].strip(), parts[1].strip(), parts[2].strip()

            # 查找对应的ID
            if head not in entity2id:
                print(f"⚠️  行 {line_num}: 头实体 '{head}' 未找到，跳过")
                skipped_count += 1
                continue

            if tail not in entity2id:
                print(f"⚠️  行 {line_num}: 尾实体 '{tail}' 未找到，跳过")
                skipped_count += 1
                continue

            if relation not in relation2id:
                print(f"⚠️  行 {line_num}: 关系 '{relation}' 未找到，跳过")
                skipped_count += 1
                continue

            # 转换为ID格式
            head_id = entity2id[head]
            tail_id = entity2id[tail]
            relation_id = relation2id[relation]

            train_triples.append((head_id, tail_id, relation_id))

    print(f"\n✓ 转换完成:")
    print(f"   - 成功: {len(train_triples)} 条三元组")
    print(f"   - 跳过: {skipped_count} 条\n")

    # 写入 train2id.txt
    print(f"💾 写入 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 第一行：三元组总数
        f.write(f"{len(train_triples)}\n")

        # 后续每行：head_id tail_id relation_id
        for head_id, tail_id, relation_id in train_triples:
            f.write(f"{head_id} {tail_id} {relation_id}\n")

    print(f"✓ 写入完成: {len(train_triples)} 条三元组\n")

    # 显示前几条示例
    print("=" * 60)
    print("前 5 条三元组示例:")
    print("=" * 60)
    print("格式: head_id tail_id relation_id\n")
    for i, (h, t, r) in enumerate(train_triples[:5], 1):
        print(f"{i}. {h} {t} {r}")
    print()


def main():
    """主函数"""
    # 文件路径配置
    data_dir = r"/media/shu1004/pytorch/projects/lyx/GCA/GCA-main/hotpotqa/gpt3.5/final_structure/data/graph_data"

    entity2id_file = os.path.join(data_dir, "entity2id.txt")
    relation2id_file = os.path.join(data_dir, "relation2id.txt")
    triples_file = os.path.join(data_dir, "triples.txt")
    output_file = os.path.join(data_dir, "train2id.txt")

    # 检查文件是否存在
    for file_path in [entity2id_file, relation2id_file, triples_file]:
        if not os.path.exists(file_path):
            print(f"❌ 错误: 文件不存在 - {file_path}")
            return

    print("=" * 60)
    print("生成 TransE 训练文件: train2id.txt")
    print("=" * 60)
    print()

    # 生成 train2id.txt
    generate_train2id(entity2id_file, relation2id_file, triples_file, output_file)

    print("=" * 60)
    print("✅ 完成！train2id.txt 已生成")
    print("=" * 60)
    print(f"\n输出文件: {output_file}")


if __name__ == '__main__':
    main()