"""
使用预训练Siamese RGCN + FFN分类器
迁移学习方案：
1. 加载预训练的Siamese RGCN编码器（对比学习训练）
2. 冻结或微调编码器
3. 训练FFN分类头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import pickle
import os


class HallucinationClassifierWithPretrainedEncoder(nn.Module):
    """
    使用预训练Siamese RGCN编码器的幻觉检测分类器
    
    架构:
    1. 预训练的RGCN编码器 (from Siamese RGCN)
    2. 全局平均池化
    3. FFN分类头
    """
    
    def __init__(self, pretrained_model_path, 
                 freeze_encoder=False,
                 ffn_hidden_dim=128, 
                 dropout=0.3):
        """
        Args:
            pretrained_model_path: 预训练Siamese RGCN模型路径
            freeze_encoder: 是否冻结编码器权重
            ffn_hidden_dim: FFN隐藏层维度
            dropout: Dropout率
        """
        super(HallucinationClassifierWithPretrainedEncoder, self).__init__()
        
        self.freeze_encoder = freeze_encoder
        
        # 1. 加载预训练的Siamese RGCN编码器
        print(f"\n加载预训练模型: {pretrained_model_path}")
        self.encoder, self.out_channels = self._load_pretrained_encoder(pretrained_model_path)
        
        print(f"  编码器输出维度: {self.out_channels}")
        
        # 2. 冻结编码器（如果需要）
        if freeze_encoder:
            print(f"  冻结编码器权重")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print(f"  允许微调编码器")
        
        # 3. FFN分类头
        # 输入: concat[h_response, h_reference] → 2 * out_channels
        # 输出: 2 (幻觉 vs 非幻觉)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.out_channels, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, ffn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim // 2, 2)
        )
        
        print(f"\nFFN分类头:")
        print(f"  输入维度: {2 * self.out_channels}")
        print(f"  隐藏层: {ffn_hidden_dim} → {ffn_hidden_dim // 2}")
        print(f"  输出: 2 (二分类)")
    
    def _load_pretrained_encoder(self, model_path):
        """加载预训练的Siamese RGCN编码器"""
        
        # 导入Siamese RGCN模型
        import sys
        import os
        
        # 添加new_rgcn路径
        new_rgcn_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', '..', 
            'Graph-based Contextual Consistency Comparison', 
            'new_rgcn'
        )
        sys.path.insert(0, os.path.abspath(new_rgcn_path))
        
        try:
            from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding
        except ImportError:
            # 如果导入失败，尝试从当前目录
            from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"  预训练模型信息:")
        print(f"    训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"    验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}" 
              if 'val_loss' in checkpoint else "    验证损失: N/A")
        
        # 🔥 优先从checkpoint的config读取配置
        if 'config' in checkpoint and checkpoint['config'] is not None:
            config = checkpoint['config']
            print(f"\n  ✓ 从checkpoint.config读取配置")
            
            # 直接使用保存的配置重新创建模型
            encoder = self._create_encoder_from_config(checkpoint, config)
            out_channels = config.get('out_channels', 64)
            return encoder, out_channels
        
        # 如果没有config，则推断配置
        print(f"\n  ⚠ checkpoint中无config，尝试推断配置...")
        model_state = checkpoint['model_state_dict']
        
        # 推断配置
        entity_emb_key = 'encoder.entity_embedding.weight'
        
        # 获取嵌入配置
        entity_emb_shape = model_state[entity_emb_key].shape
        num_entities = entity_emb_shape[0]
        embedding_dim = entity_emb_shape[1]
        
        # 🔥 正确推断RGCN配置
        # 1. 推断层数：找有多少个convs层
        conv_layer_indices = set()
        for key in model_state.keys():
            if key.startswith('encoder.convs.') and '.weight' in key:
                layer_idx = int(key.split('.')[2])
                conv_layer_indices.add(layer_idx)
        num_layers = len(conv_layer_indices)
        
        # 2. 推断关系数和维度
        # convs.0.weight shape: [num_relations, in_features, out_features]
        if 'encoder.convs.0.weight' in model_state:
            conv0_shape = model_state['encoder.convs.0.weight'].shape
            num_relations = conv0_shape[0]
            # hidden_channels是第一层的输出维度
            hidden_channels = conv0_shape[2]
        else:
            num_relations = 50
            hidden_channels = 128
        
        # 3. 推断输出维度（从最后一层）
        last_layer_idx = num_layers - 1
        if f'encoder.convs.{last_layer_idx}.weight' in model_state:
            # 最后一层的输出维度
            last_conv_shape = model_state[f'encoder.convs.{last_layer_idx}.weight'].shape
            out_channels = last_conv_shape[2]
        elif 'encoder.attention.0.bias' in model_state:
            # 从attention层推断
            attention_hidden = model_state['encoder.attention.0.bias'].shape[0]
            out_channels = attention_hidden * 2  # attention输入是out_channels的一半
        else:
            out_channels = 64
        
        print(f"  编码器配置 (从checkpoint推断):")
        print(f"    实体数: {num_entities}")
        print(f"    嵌入维度: {embedding_dim}")
        print(f"    关系数: {num_relations}")
        print(f"    RGCN层数: {num_layers}")
        print(f"    隐藏维度: {hidden_channels}")
        print(f"    输出维度: {out_channels}")
        
        # 🔥 调试信息：打印实际的shape
        print(f"\n  调试信息 - 实际层形状:")
        for i in range(num_layers):
            if f'encoder.convs.{i}.weight' in model_state:
                shape = model_state[f'encoder.convs.{i}.weight'].shape
                print(f"    convs.{i}.weight: {shape} → [num_rel, in_feat, out_feat]")
        
        # 创建编码器（需要提供临时的嵌入文件）
        # 注意：这里我们从checkpoint中提取嵌入
        encoder = self._create_encoder_from_checkpoint(
            checkpoint,
            num_entities,
            embedding_dim,
            num_relations,
            hidden_channels,
            out_channels,
            num_layers
        )
        
        return encoder, out_channels
    
    def _create_encoder_from_config(self, checkpoint, config):
        """
        从保存的config创建编码器（优化版）
        
        🔥 直接使用config中保存的嵌入文件路径，无需临时文件
        """
        import sys
        import os
        
        # 添加路径
        new_rgcn_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', '..', 
            'Graph-based Contextual Consistency Comparison', 
            'new_rgcn'
        )
        sys.path.insert(0, os.path.abspath(new_rgcn_path))
        from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding
        
        print(f"  使用配置:")
        print(f"    hidden_channels: {config.get('hidden_channels', 128)}")
        print(f"    out_channels: {config.get('out_channels', 64)}")
        print(f"    num_layers: {config.get('num_layers', 3)}")
        print(f"    num_relations: {config.get('num_relations', 50)}")
        
        # 🔥 关键优化：直接使用config中保存的嵌入文件路径
        # 这些路径在训练时就存在，无需创建临时文件
        entity_emb_path = config.get('entity_embedding_path')
        relation_emb_path = config.get('relation_embedding_path')
        
        if not entity_emb_path or not relation_emb_path:
            print(f"  ⚠ Config中缺少嵌入路径，使用默认路径")
            # 使用默认的RGCN嵌入路径
            from config_hotpotqa import Config as HotpotConfig
            entity_emb_path = HotpotConfig.ENTITY_EMBEDDING_RGCN_PATH
            relation_emb_path = HotpotConfig.RELATION_MAPPING_RGCN_PATH
        
        print(f"  嵌入文件路径:")
        print(f"    entity: {os.path.basename(entity_emb_path)}")
        print(f"    relation: {os.path.basename(relation_emb_path)}")
        
        # 直接创建编码器（使用训练时的相同路径）
        encoder = ImprovedRGCNEncoderWithEmbedding(
            entity_embedding_path=entity_emb_path,
            relation_embedding_path=relation_emb_path,
            hidden_channels=config.get('hidden_channels', 128),
            out_channels=config.get('out_channels', 64),
            num_layers=config.get('num_layers', 3),
            freeze_embeddings=False,
            dropout=config.get('dropout', 0.3)
        )
        
        # 🔥 加载checkpoint的权重（覆盖从文件加载的初始嵌入）
        model_state = checkpoint['model_state_dict']
        encoder_state_dict = {}
        for key, value in model_state.items():
            if key.startswith('encoder.'):
                new_key = key[8:]  # 移除'encoder.'前缀
                encoder_state_dict[new_key] = value
        
        encoder.load_state_dict(encoder_state_dict)
        
        print(f"  ✓ 编码器加载完成（直接使用config路径，无临时文件）")
        
        return encoder
    
    def _create_encoder_from_checkpoint(self, checkpoint, num_entities, 
                                       embedding_dim, num_relations,
                                       hidden_channels, out_channels, num_layers):
        """
        从检查点创建编码器（备用方法，当config不存在时）
        
        🔥 优化：直接使用默认的嵌入文件路径，无需临时文件
        """
        # 导入编码器类
        import sys
        import os
        new_rgcn_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', '..', 
            'Graph-based Contextual Consistency Comparison', 
            'new_rgcn'
        )
        sys.path.insert(0, os.path.abspath(new_rgcn_path))
        
        from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding
        
        # 🔥 直接使用默认的RGCN嵌入路径（训练时生成的）
        from config_hotpotqa import Config as HotpotConfig
        entity_emb_path = HotpotConfig.ENTITY_EMBEDDING_RGCN_PATH
        relation_emb_path = HotpotConfig.RELATION_MAPPING_RGCN_PATH
        
        print(f"  使用默认嵌入路径:")
        print(f"    entity: {os.path.basename(entity_emb_path)}")
        print(f"    relation: {os.path.basename(relation_emb_path)}")
        
        # 检查文件是否存在
        if not os.path.exists(entity_emb_path) or not os.path.exists(relation_emb_path):
            print(f"\n  ⚠ 警告: 嵌入文件不存在，请先运行:")
            print(f"    python prepare_embeddings.py")
            raise FileNotFoundError(f"嵌入文件不存在: {entity_emb_path}")
        
        # 直接创建编码器（使用默认路径）
        encoder = ImprovedRGCNEncoderWithEmbedding(
            entity_embedding_path=entity_emb_path,
            relation_embedding_path=relation_emb_path,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            freeze_embeddings=False,
            dropout=0.3
        )
        
        # 🔥 加载checkpoint的权重（覆盖从文件加载的初始嵌入）
        encoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('encoder.'):
                # 移除'encoder.'前缀
                new_key = key[8:]
                encoder_state_dict[new_key] = value
        
        encoder.load_state_dict(encoder_state_dict)
        
        print(f"  ✓ 编码器加载完成（使用默认路径，无临时文件）")
        
        return encoder
    
    def forward(self, response_graph, reference_graph):
        """
        前向传播
        Args:
            response_graph: 响应图（GPT生成）
            reference_graph: 参考图（KB）
        Returns:
            logits: 分类logits [batch_size, 2]
        """
        # 1. RGCN编码
        with torch.set_grad_enabled(not self.freeze_encoder):
            h_response = self.encoder(response_graph)      # [batch_size, out_channels]
            h_reference = self.encoder(reference_graph)    # [batch_size, out_channels]
        
        # 2. 全局平均池化（如果编码器没有池化）
        if len(h_response.shape) == 1:
            h_response = h_response.unsqueeze(0)
        if len(h_reference.shape) == 1:
            h_reference = h_reference.unsqueeze(0)
        
        # 3. 拼接特征
        h_concat = torch.cat([h_response, h_reference], dim=1)
        
        # 4. FFN分类
        logits = self.classifier(h_concat)
        
        return logits
    
    def predict(self, response_graph, reference_graph):
        """预测"""
        logits = self.forward(response_graph, reference_graph)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities
    
    def get_trainable_params(self):
        """获取可训练参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'classifier': classifier_params,
            'encoder_trainable': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            'classifier_trainable': sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        }


def test_model():
    """测试模型加载"""
    print("="*60)
    print("测试预训练编码器 + FFN分类器")
    print("="*60)
    
    # 预训练模型路径（需要先训练Siamese RGCN）
    pretrained_path = os.path.join(
        os.path.dirname(__file__),
        'rgcn_output',
        'checkpoints',
        'best_model.pth'
    )
    
    if not os.path.exists(pretrained_path):
        print(f"\n❌ 预训练模型不存在: {pretrained_path}")
        print(f"\n请先运行: python train_rgcn_hotpotqa.py")
        return
    
    # 方式1: 冻结编码器
    print("\n" + "="*60)
    print("方式1: 冻结编码器（只训练FFN）")
    print("="*60)
    model1 = HallucinationClassifierWithPretrainedEncoder(
        pretrained_model_path=pretrained_path,
        freeze_encoder=True,
        ffn_hidden_dim=128,
        dropout=0.3
    )
    
    params1 = model1.get_trainable_params()
    print(f"\n参数统计:")
    print(f"  总参数: {params1['total']:,}")
    print(f"  可训练参数: {params1['trainable']:,}")
    print(f"  编码器参数: {params1['encoder']:,} (可训练: {params1['encoder_trainable']:,})")
    print(f"  分类器参数: {params1['classifier']:,} (可训练: {params1['classifier_trainable']:,})")
    
    # 方式2: 微调编码器
    print("\n" + "="*60)
    print("方式2: 微调编码器（编码器+FFN一起训练）")
    print("="*60)
    model2 = HallucinationClassifierWithPretrainedEncoder(
        pretrained_model_path=pretrained_path,
        freeze_encoder=False,
        ffn_hidden_dim=128,
        dropout=0.3
    )
    
    params2 = model2.get_trainable_params()
    print(f"\n参数统计:")
    print(f"  总参数: {params2['total']:,}")
    print(f"  可训练参数: {params2['trainable']:,}")
    print(f"  编码器参数: {params2['encoder']:,} (可训练: {params2['encoder_trainable']:,})")
    print(f"  分类器参数: {params2['classifier']:,} (可训练: {params2['classifier_trainable']:,})")
    
    print("\n✓ 模型加载成功！")


if __name__ == '__main__':
    test_model()

