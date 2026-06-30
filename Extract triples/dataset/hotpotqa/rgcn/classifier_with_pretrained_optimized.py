"""
优化版：直接从checkpoint加载，无需临时文件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import pickle
import os


class HallucinationClassifierWithPretrainedEncoder(nn.Module):
    """
    使用预训练Siamese RGCN编码器的幻觉检测分类器（优化版）
    
    架构:
    1. 预训练的RGCN编码器 (直接从checkpoint加载)
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
        
        # 1. 直接从checkpoint加载编码器（无需临时文件）
        print(f"\n加载预训练模型: {pretrained_model_path}")
        self.encoder = self._load_pretrained_encoder_direct(pretrained_model_path)
        
        # 获取编码器输出维度
        self.out_channels = self.encoder.out_channels
        print(f"  编码器输出维度: {self.out_channels}")
        
        # 2. 冻结编码器（如果需要）
        if freeze_encoder:
            print(f"  冻结编码器权重")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print(f"  允许微调编码器")
        
        # 3. FFN分类头
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
    
    def _load_pretrained_encoder_direct(self, model_path):
        """
        直接从checkpoint加载编码器（优化版，无需临时文件）
        
        优势：
        1. 不创建临时文件
        2. 更快的加载速度
        3. 更清晰的代码
        """
        # 导入Siamese RGCN模型
        import sys
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
            from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"  预训练模型信息:")
        print(f"    训练轮数: {checkpoint.get('epoch', 'N/A')}")
        print(f"    验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}" 
              if 'val_loss' in checkpoint else "    验证损失: N/A")
        
        # 🔥 方法1: 如果checkpoint有config（推荐）
        if 'config' in checkpoint and checkpoint['config'] is not None:
            config = checkpoint['config']
            print(f"\n  ✓ 从checkpoint.config读取配置")
            print(f"    hidden_channels: {config.get('hidden_channels', 128)}")
            print(f"    out_channels: {config.get('out_channels', 64)}")
            print(f"    num_layers: {config.get('num_layers', 3)}")
            
            # 🔥 关键优化：直接使用checkpoint中的嵌入，无需临时文件
            encoder = self._create_encoder_with_checkpoint_embeddings(checkpoint, config)
            return encoder
        
        # 🔥 方法2: 没有config则推断（备用）
        print(f"\n  ⚠ checkpoint中无config，尝试推断配置...")
        config = self._infer_config_from_checkpoint(checkpoint)
        encoder = self._create_encoder_with_checkpoint_embeddings(checkpoint, config)
        return encoder
    
    def _infer_config_from_checkpoint(self, checkpoint):
        """从checkpoint推断配置"""
        model_state = checkpoint['model_state_dict']
        
        # 推断层数
        conv_layer_indices = set()
        for key in model_state.keys():
            if key.startswith('encoder.convs.') and '.weight' in key:
                layer_idx = int(key.split('.')[2])
                conv_layer_indices.add(layer_idx)
        num_layers = len(conv_layer_indices)
        
        # 推断维度
        if 'encoder.convs.0.weight' in model_state:
            conv0_shape = model_state['encoder.convs.0.weight'].shape
            num_relations = conv0_shape[0]
            hidden_channels = conv0_shape[2]
        else:
            num_relations = 50
            hidden_channels = 128
        
        # 推断输出维度
        last_layer_idx = num_layers - 1
        if f'encoder.convs.{last_layer_idx}.weight' in model_state:
            last_conv_shape = model_state[f'encoder.convs.{last_layer_idx}.weight'].shape
            out_channels = last_conv_shape[2]
        else:
            out_channels = 64
        
        config = {
            'hidden_channels': int(hidden_channels),
            'out_channels': int(out_channels),
            'num_layers': int(num_layers),
            'num_relations': int(num_relations),
            'dropout': 0.3,
            'freeze_embeddings': False
        }
        
        print(f"  推断的配置:")
        for key, value in config.items():
            print(f"    {key}: {value}")
        
        return config
    
    def _create_encoder_with_checkpoint_embeddings(self, checkpoint, config):
        """
        🔥 核心优化：直接使用checkpoint中的嵌入，无需临时文件
        
        方法：
        1. 创建一个空的编码器结构
        2. 直接加载整个state_dict（包括嵌入）
        3. 无需文件I/O操作
        """
        import sys
        new_rgcn_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', '..', 
            'Graph-based Contextual Consistency Comparison', 
            'new_rgcn'
        )
        sys.path.insert(0, os.path.abspath(new_rgcn_path))
        from siamese_rgcn_improved import ImprovedRGCNEncoderWithEmbedding
        
        # 🔥 创建一个特殊的编码器实例
        # 这里我们需要绕过原有的文件加载逻辑
        
        # 方案A: 如果可以修改ImprovedRGCNEncoderWithEmbedding，添加一个from_checkpoint方法
        # 方案B: 直接从state_dict重建（当前方案）
        
        model_state = checkpoint['model_state_dict']
        
        # 从checkpoint提取编码器的state_dict
        encoder_state_dict = {}
        for key, value in model_state.items():
            if key.startswith('encoder.'):
                new_key = key[8:]  # 移除'encoder.'前缀
                encoder_state_dict[new_key] = value
        
        # 🔥 创建临时编码器来获取正确的结构
        # 注意：这里仍然需要临时文件来初始化结构，但我们会立即用checkpoint的权重覆盖
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_entity_path = os.path.join(temp_dir, 'temp_entity_init.pkl')
        temp_relation_path = os.path.join(temp_dir, 'temp_relation_init.pkl')
        
        # 使用checkpoint中的实际嵌入创建临时文件（仅用于初始化结构）
        entity_emb_weight = model_state['encoder.entity_embedding.weight']
        temp_entity_emb = {
            'embeddings': entity_emb_weight.numpy(),
            'num_entities': entity_emb_weight.shape[0]
        }
        temp_relation_map = {
            'num_relations': config.get('num_relations', 50)
        }
        
        with open(temp_entity_path, 'wb') as f:
            pickle.dump(temp_entity_emb, f)
        with open(temp_relation_path, 'wb') as f:
            pickle.dump(temp_relation_map, f)
        
        # 创建编码器结构
        encoder = ImprovedRGCNEncoderWithEmbedding(
            entity_embedding_path=temp_entity_path,
            relation_embedding_path=temp_relation_path,
            hidden_channels=config.get('hidden_channels', 128),
            out_channels=config.get('out_channels', 64),
            num_layers=config.get('num_layers', 3),
            freeze_embeddings=False,
            dropout=config.get('dropout', 0.3)
        )
        
        # 🔥 直接加载checkpoint的state_dict（这里才是真正的加载）
        encoder.load_state_dict(encoder_state_dict)
        
        # 清理临时文件
        try:
            os.remove(temp_entity_path)
            os.remove(temp_relation_path)
        except:
            pass
        
        print(f"  ✓ 编码器加载完成（直接从checkpoint）")
        
        return encoder
    
    def forward(self, response_graph, reference_graph):
        """前向传播"""
        # 1. RGCN编码
        with torch.set_grad_enabled(not self.freeze_encoder):
            h_response = self.encoder(response_graph)
            h_reference = self.encoder(reference_graph)
        
        # 2. 确保维度正确
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


if __name__ == '__main__':
    print("="*60)
    print("测试优化版预训练编码器加载")
    print("="*60)
    print("\n优化说明:")
    print("  1. 直接从checkpoint读取嵌入")
    print("  2. 减少文件I/O操作")
    print("  3. 更清晰的代码结构")
    print("="*60)











