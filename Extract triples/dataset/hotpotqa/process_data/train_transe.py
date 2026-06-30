import os
import argparse
import numpy as np
import pickle
import torch
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# ==========================================
# 1. 读取现有数据文件
# ==========================================

def load_entity2id(file_path):
    """读取 entity2id.txt，返回 {entity_name: id} 字典"""
    entity2id = {}
    print(f"正在读取 {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, eid = parts[0], int(parts[1])
                entity2id[entity] = eid
    print(f"已读取 {len(entity2id)} 个实体")
    return entity2id

def load_relation2id(file_path):
    """读取 relation2id.txt，返回 {relation_name: id} 字典"""
    relation2id = {}
    print(f"正在读取 {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, rid = parts[0], int(parts[1])
                relation2id[relation] = rid
    print(f"已读取 {len(relation2id)} 个关系")
    return relation2id

def load_triples(file_path, entity2id, relation2id):
    """
    读取 triples.txt，格式为: head \t tail \t relation
    转换为 (head_id, tail_id, relation_id) 的列表
    """
    triples = []
    print(f"正在读取 {file_path}...")
    skipped = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, tail, relation = parts[0], parts[1], parts[2]
                
                # 转换为ID
                if head in entity2id and tail in entity2id and relation in relation2id:
                    head_id = entity2id[head]
                    tail_id = entity2id[tail]
                    relation_id = relation2id[relation]
                    triples.append((head_id, tail_id, relation_id))
                else:
                    skipped += 1
                    if skipped <= 5:  # 只打印前5个跳过的
                        print(f"  警告: 跳过三元组 (行{idx}): {head} | {tail} | {relation}")
    
    print(f"已读取 {len(triples)} 个三元组")
    if skipped > 0:
        print(f"跳过了 {skipped} 个无法映射的三元组")
    
    return triples

# ==========================================
# 2. 将数据转换为 OpenKE 需要的文件格式
# ==========================================

def write_openke_entity2id(entity2id, file_path):
    """
    写入 OpenKE 格式的 entity2id.txt
    格式: 第一行是实体总数，之后每行是: entity_name \t entity_id
    """
    print(f"正在写入 {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(len(entity2id)) + "\n")
        for entity, eid in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{entity}\t{eid}\n")
    print(f"已写入 {len(entity2id)} 个实体")

def write_openke_relation2id(relation2id, file_path):
    """
    写入 OpenKE 格式的 relation2id.txt
    格式: 第一行是关系总数，之后每行是: relation_name \t relation_id
    """
    print(f"正在写入 {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(len(relation2id)) + "\n")
        for relation, rid in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f"{relation}\t{rid}\n")
    print(f"已写入 {len(relation2id)} 个关系")

def write_openke_train2id(triples, file_path):
    """
    写入 OpenKE 格式的 train2id.txt
    格式: 第一行是三元组总数，之后每行是: head_id \t tail_id \t relation_id
    注意：OpenKE要求的顺序是 h t r（不是 h r t）
    """
    print(f"正在写入 {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(str(len(triples)) + "\n")
        for head_id, tail_id, relation_id in triples:
            f.write(f"{head_id}\t{tail_id}\t{relation_id}\n")
    print(f"已写入 {len(triples)} 个三元组")

# ==========================================
# 3. 主流程
# ==========================================

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练 TransE 模型获取知识图谱嵌入')
    parser.add_argument("--dim", type=int, default=100, help="嵌入维度")
    parser.add_argument("--epoch", type=int, default=1000, help="训练轮数")
    parser.add_argument("--save_steps", type=int, default=10, help="每隔多少epoch保存一次模型")
    parser.add_argument("--batch_size", type=int, default=300, help="batch size")
    parser.add_argument("--patient", type=int, default=-1, help="早停耐心值，-1表示不使用早停")
    parser.add_argument("--lr", type=float, default=1.0, help="学习率")
    parser.add_argument("--margin", type=float, default=5.0, help="margin损失的margin值")
    parser.add_argument("--outdir", type=str, default="./output", help="输出目录")
    parser.add_argument("--datadir", type=str, default="./openke_data", help="OpenKE格式数据目录")
    parser.add_argument("--model_path", type=str, default="", help="预训练模型路径（可选）")
    parser.add_argument("--neg_ent", type=int, default=64, help="负采样实体数量")
    parser.add_argument("--bern_flag", type=int, default=0, help="是否使用伯努利负采样")
    parser.add_argument("--test", action="store_true", help="是否运行测试评估")
    parser.add_argument("--prepare_data", action="store_true", help="是否需要准备数据（从triples.txt等转换）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)
    
    # 如果需要准备数据
    if args.prepare_data:
        print("\n" + "="*60)
        print("步骤 1: 读取原始数据文件并转换为 OpenKE 格式")
        print("="*60)
        
        entity_file = 'entity2id.txt'
        relation_file = 'relation2id.txt'
        triples_file = 'triples.txt'
        
        entity2id = load_entity2id(entity_file)
        relation2id = load_relation2id(relation_file)
        triples = load_triples(triples_file, entity2id, relation2id)
        
        # 创建 OpenKE 数据目录
        os.makedirs(args.datadir, exist_ok=True)
        
        # 转换为 OpenKE 格式
        write_openke_entity2id(entity2id, os.path.join(args.datadir, "entity2id.txt"))
        write_openke_relation2id(relation2id, os.path.join(args.datadir, "relation2id.txt"))
        write_openke_train2id(triples, os.path.join(args.datadir, "train2id.txt"))
        
        print(f"\nOpenKE 格式数据已保存到: {args.datadir}/")
    
    # 训练 TransE
    print("\n" + "="*60)
    print("步骤 2: 加载数据并配置 TransE 模型")
    print("="*60)
    
    # 加载训练数据
    print("\n正在加载训练数据...")
    train_dataloader = TrainDataLoader(
        in_path = args.datadir + "/",
        batch_size = args.batch_size,
        threads = 8,
        sampling_mode = "normal",
        bern_flag = args.bern_flag,
        filter_flag = 1,
        neg_ent = args.neg_ent,
        neg_rel = 0
    )
    
    print(f"实体总数: {train_dataloader.get_ent_tot()}")
    print(f"关系总数: {train_dataloader.get_rel_tot()}")
    print(f"训练三元组数: {train_dataloader.get_triple_tot()}")
    print(f"Batch size: {train_dataloader.get_batch_size()}")
    
    # 加载测试数据（如果需要测试）
    test_dataloader = None
    if args.test:
        print("\n正在加载测试数据...")
        test_dataloader = TestDataLoader(
            in_path = args.datadir + "/",
            sampling_mode = "link"
        )
    
    # 定义模型
    print(f"\n正在初始化 TransE 模型（维度={args.dim}）...")
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = args.dim,
        p_norm = 1,
        norm_flag = True
    )
    
    # 加载预训练模型（如果提供）
    if args.model_path:
        print(f'加载预训练模型: {args.model_path}')
        transe.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
    # 定义损失函数
    model = NegativeSampling(
        model = transe,
        loss = MarginLoss(margin = args.margin),
        batch_size = train_dataloader.get_batch_size()
    )
    
    # 训练模型
    print("\n" + "="*60)
    print("步骤 3: 开始训练")
    print("="*60)
    print(f"训练参数:")
    print(f"  - Epoch数: {args.epoch}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - Margin: {args.margin}")
    print(f"  - 负采样数: {args.neg_ent}")
    print(f"  - 保存间隔: {args.save_steps} epochs")
    print(f"  - GPU: {'是' if torch.cuda.is_available() else '否'}")
    print()
    
    # 创建 Trainer（使用兼容的参数）
    try:
        # 尝试使用完整参数（新版本 OpenKE）
        trainer = Trainer(
            model = model,
            data_loader = train_dataloader,
            train_times = args.epoch,
            alpha = args.lr,
            use_gpu = torch.cuda.is_available(),
            save_steps = args.save_steps,
            checkpoint_dir = args.outdir
        )
    except TypeError:
        # 如果失败，使用基础参数（旧版本 OpenKE）
        print("检测到旧版本 OpenKE，使用基础参数...")
        trainer = Trainer(
            model = model,
            data_loader = train_dataloader,
            train_times = args.epoch,
            alpha = args.lr,
            use_gpu = torch.cuda.is_available()
        )
    
    trainer.run()
    
    # 保存最终模型
    print("\n" + "="*60)
    print("保存最终模型...")
    print("="*60)
    model_save_path = os.path.join(args.outdir, 'transe_final.ckpt')
    transe.save_checkpoint(model_save_path)
    print(f"最终模型已保存到: {model_save_path}")
    
    # 测试模型（如果需要）
    if args.test and test_dataloader is not None:
        print("\n" + "="*60)
        print("步骤 4: 测试模型")
        print("="*60)
        transe.load_checkpoint(model_save_path)
        tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = torch.cuda.is_available())
        tester.run_link_prediction(type_constrain = False)
    
    # 提取并保存嵌入
    print("\n" + "="*60)
    print("步骤 5: 提取并保存嵌入向量")
    print("="*60)
    
    # 获取嵌入（按照ID顺序）
    ent_embeddings = transe.ent_embeddings.weight.data.cpu().detach().numpy()
    rel_embeddings = transe.rel_embeddings.weight.data.cpu().detach().numpy()
    
    print(f"实体嵌入形状: {ent_embeddings.shape}")
    print(f"关系嵌入形状: {rel_embeddings.shape}")
    
    # 保存为 pickle 格式（按ID映射顺序）
    ent_emb_pkl = os.path.join(args.outdir, "ent_embeddings.pkl")
    rel_emb_pkl = os.path.join(args.outdir, "rel_embeddings.pkl")
    
    pickle.dump(ent_embeddings, open(ent_emb_pkl, "wb"))
    pickle.dump(rel_embeddings, open(rel_emb_pkl, "wb"))
    
    print(f"实体嵌入已保存到: {ent_emb_pkl}")
    print(f"关系嵌入已保存到: {rel_emb_pkl}")
    
    # 同时保存为 numpy 格式（方便其他工具使用）
    ent_emb_npy = os.path.join(args.outdir, "ent_embeddings.npy")
    rel_emb_npy = os.path.join(args.outdir, "rel_embeddings.npy")
    
    np.save(ent_emb_npy, ent_embeddings)
    np.save(rel_emb_npy, rel_embeddings)
    
    print(f"实体嵌入已保存到: {ent_emb_npy}")
    print(f"关系嵌入已保存到: {rel_emb_npy}")
    
    # 显示一些示例
    if args.prepare_data:
        print("\n" + "="*60)
        print("嵌入向量示例")
        print("="*60)
        
        # 加载ID映射
        entity2id = load_entity2id('entity2id.txt')
        relation2id = load_relation2id('relation2id.txt')
        
        id2entity = {v: k for k, v in entity2id.items()}
        id2relation = {v: k for k, v in relation2id.items()}
        
        # 显示前5个实体的嵌入
        print("\n前5个实体的嵌入向量（前10维）:")
        for i in range(min(5, len(ent_embeddings))):
            entity_name = id2entity.get(i, f"Entity_{i}")
            print(f"  [{i}] {entity_name[:50]}: {ent_embeddings[i][:10]}")
        
        # 显示前5个关系的嵌入
        print("\n前5个关系的嵌入向量（前10维）:")
        for i in range(min(5, len(rel_embeddings))):
            relation_name = id2relation.get(i, f"Relation_{i}")
            print(f"  [{i}] {relation_name[:50]}: {rel_embeddings[i][:10]}")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  - 模型检查点: {model_save_path}")
    print(f"  - 实体嵌入(pkl): {ent_emb_pkl}")
    print(f"  - 关系嵌入(pkl): {rel_emb_pkl}")
    print(f"  - 实体嵌入(npy): {ent_emb_npy}")
    print(f"  - 关系嵌入(npy): {rel_emb_npy}")
    
    return ent_embeddings, rel_embeddings

if __name__ == '__main__':
    main()

