# 🔧 预训练编码器加载故障排除

## 问题：配置推断错误

### 错误症状

```
RuntimeError: Error(s) in loading state_dict for ImprovedRGCNEncoderWithEmbedding:
    size mismatch for convs.0.weight: copying a param with shape torch.Size([1458, 768, 128])
```

### 根本原因

1. **Config未保存**：旧版本的训练脚本可能没有保存config到checkpoint
2. **Config不完整**：保存的config缺少必要参数（num_relations等）
3. **推断逻辑错误**：从state_dict推断配置时出错

---

## 解决方案

### 方案1: 重新训练模型（推荐）⭐

确保使用最新的训练脚本，它会保存完整config：

```bash
# 删除旧模型
rm rgcn_output/checkpoints/best_model.pth

# 重新训练（会保存完整config）
python train_rgcn_hotpotqa.py
```

**检查训练脚本是否保存config**:

```python
# 在 train_rgcn_hotpotqa.py 中确认有这段代码
torch.save({
    'epoch': epoch,
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'val_loss': val_loss,
    'config': self.config  # ⭐ 必须保存config
}, checkpoint_path)
```

---

### 方案2: 手动修复checkpoint

如果不想重新训练，可以手动添加config到checkpoint：

```python
import torch

# 加载旧checkpoint
checkpoint_path = 'rgcn_output/checkpoints/best_model.pth'
checkpoint = torch.load(checkpoint_path)

# 手动添加正确的config
# 🔥 根据你的实际训练参数填写
checkpoint['config'] = {
    'hidden_channels': 128,     # RGCN隐藏层维度
    'out_channels': 64,         # RGCN输出维度
    'num_layers': 3,            # RGCN层数
    'num_relations': 1458,      # 关系数（从你的数据）
    'dropout': 0.3,             # Dropout率
    'freeze_embeddings': True,  # 是否冻结嵌入
}

# 保存修复后的checkpoint
torch.save(checkpoint, checkpoint_path)
print("✓ Checkpoint已修复")
```

**如何确定正确的参数值**:

1. **hidden_channels**: 查看 `config_hotpotqa.py` 中的 `HIDDEN_CHANNELS`
2. **out_channels**: 查看 `config_hotpotqa.py` 中的 `OUT_CHANNELS`
3. **num_layers**: 查看 `config_hotpotqa.py` 中的 `NUM_LAYERS`
4. **num_relations**: 运行以下代码获取：

```python
import pickle
with open('../hybrid_embeddings/relation2idx.pkl', 'rb') as f:
    relation2idx = pickle.load(f)
num_relations = len(relation2idx)
print(f"关系数: {num_relations}")
```

---

### 方案3: 使用诊断工具

运行诊断脚本查看详细信息：

```bash
python test_pretrained_loading.py
```

这会显示：
- Checkpoint中是否有config
- 实际的模型结构
- 推断出的配置值
- 详细的错误信息

---

### 方案4: 修改加载逻辑（高级）

如果以上方案都不行，可以在 `classifier_with_pretrained.py` 中添加手动配置选项：

```python
# 修改 HallucinationClassifierWithPretrainedEncoder 的 __init__
def __init__(self, pretrained_model_path, 
             freeze_encoder=False,
             ffn_hidden_dim=128, 
             dropout=0.3,
             manual_config=None):  # ⭐ 新增参数
    """
    Args:
        manual_config: 手动指定的配置字典（可选）
                      {'hidden_channels': 128, 'out_channels': 64, ...}
    """
    # ...
    if manual_config:
        # 使用手动配置
        encoder = self._create_encoder_from_config(checkpoint, manual_config)
    else:
        # 原有逻辑
        encoder = self._load_pretrained_encoder(pretrained_model_path)
```

使用示例：

```python
model = HallucinationClassifierWithPretrainedEncoder(
    pretrained_model_path=pretrained_path,
    freeze_encoder=True,
    manual_config={
        'hidden_channels': 128,
        'out_channels': 64,
        'num_layers': 3,
        'num_relations': 1458,
        'dropout': 0.3
    }
)
```

---

## 🔍 诊断步骤

### 1. 检查checkpoint内容

```python
import torch

checkpoint = torch.load('rgcn_output/checkpoints/best_model.pth')

# 检查是否有config
print("Keys:", checkpoint.keys())

if 'config' in checkpoint:
    print("Config:", checkpoint['config'])
else:
    print("⚠ 没有config!")

# 检查模型结构
model_state = checkpoint['model_state_dict']
conv_keys = [k for k in model_state.keys() if 'encoder.convs' in k and 'weight' in k]
print("\nConv层:")
for key in sorted(conv_keys):
    print(f"  {key}: {model_state[key].shape}")
```

### 2. 运行测试脚本

```bash
python test_pretrained_loading.py
```

### 3. 查看config_hotpotqa.py中的配置

```bash
cat config_hotpotqa.py | grep -E "(HIDDEN_CHANNELS|OUT_CHANNELS|NUM_LAYERS)"
```

---

## ✅ 验证修复

修复后，运行以下命令验证：

```bash
python test_pretrained_loading.py
```

预期输出：

```
============================================================
测试预训练编码器加载
============================================================

1. 检查checkpoint内容...
  Checkpoint keys: ['epoch', 'model_state_dict', 'optimizer_state_dict', ...]
  
  ✓ Config存在:
    hidden_channels: 128
    out_channels: 64
    num_layers: 3
    num_relations: 1458

2. 检查模型结构...
    encoder.convs.0.weight: torch.Size([1458, 768, 128])
    encoder.convs.1.weight: torch.Size([1458, 128, 128])
    encoder.convs.2.weight: torch.Size([1458, 128, 64])
  
  发现 3 个卷积层: [0, 1, 2]

3. 尝试加载模型...
  ✓ 从checkpoint.config读取配置
  
✓ 模型加载成功！

参数统计:
  总参数: 5,234,567
  可训练参数: 123,456
  ...

============================================================
✓ 测试通过！可以使用预训练编码器
============================================================
```

---

## 📋 快速检查清单

- [ ] 运行 `python test_pretrained_loading.py` 诊断问题
- [ ] 检查checkpoint中是否有config
- [ ] 如果没有config，选择解决方案：
  - [ ] 方案1: 重新训练（推荐）
  - [ ] 方案2: 手动修复checkpoint
  - [ ] 方案3: 使用manual_config参数
- [ ] 验证修复是否成功

---

## 🆘 仍然失败？

如果以上方案都不行，请提供以下信息：

1. 运行 `python test_pretrained_loading.py` 的完整输出
2. `config_hotpotqa.py` 中的配置值
3. 训练时使用的命令
4. PyTorch和PyTorch Geometric版本

---

## 💡 预防措施

为了避免此问题，在训练新模型时确保：

1. **使用最新的训练脚本**
2. **保存完整的config**：
   ```python
   torch.save({
       'config': self.config,  # ⭐ 必须
       # ... other keys
   }, checkpoint_path)
   ```

3. **训练完成后立即测试加载**：
   ```bash
   python test_pretrained_loading.py
   ```

这样可以早期发现问题！











