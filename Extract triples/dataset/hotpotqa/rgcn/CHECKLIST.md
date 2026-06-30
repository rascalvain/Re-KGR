# ✅ 使用前检查清单

## 环境准备

### 必需的Python包

```bash
# 安装PyTorch和相关包
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 安装其他依赖
pip install sentence-transformers
pip install scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install numpy pandas
```

### OpenKE安装（用于TransE训练）

```bash
# 克隆OpenKE
git clone https://github.com/thunlp/OpenKE
cd OpenKE
bash make.sh
pip install -e .
```

---

## 📋 完整检查清单

### 阶段1: 数据准备 ✓

- [ ] 已有 `hotpot_dev_with_triples_aligned.json`
- [ ] 运行 `extract_entities_relations.py` → 生成 `entity2id.txt`, `relation2id.txt`
- [ ] 运行 `extract_triples.py` → 生成 `triples.txt`

### 阶段2: TransE训练 ✓

- [ ] 运行 `train_transe.py` → 生成TransE嵌入
- [ ] 输出文件存在:
  - `transe_output/entity_embeddings.pkl`
  - `transe_output/relation_embeddings.pkl`

### 阶段3: 混合嵌入生成 ✓

- [ ] 本地Sentence-BERT模型存在: `sentence-bert/`
- [ ] 运行 `generate_hybrid_embeddings.py` → 生成混合嵌入
- [ ] 输出文件存在:
  - `hybrid_embeddings/entity_hybrid_embeddings.pkl`
  - `hybrid_embeddings/relation_hybrid_embeddings.pkl`
  - `hybrid_embeddings/entity2idx.pkl`
  - `hybrid_embeddings/relation2idx.pkl`

### 阶段4: RGCN嵌入准备 ✓

- [ ] 进入 `rgcn/` 目录
- [ ] 运行 `prepare_embeddings.py` → 转换嵌入格式
- [ ] 输出文件存在:
  - `../hybrid_embeddings/entity_embeddings_rgcn.pkl`
  - `../hybrid_embeddings/relation_mappings_rgcn.pkl`

---

## 🚀 三种方法运行清单

### 方法1: 相似度阈值

- [ ] 文件存在: `train_rgcn_hotpotqa.py`, `inference_hotpotqa.py`
- [ ] 运行: `python train_rgcn_hotpotqa.py`
- [ ] 检查输出: `rgcn_output/checkpoints/best_model.pth`
- [ ] 运行: `python inference_hotpotqa.py`
- [ ] 检查结果: `rgcn_output/hallucination_predictions.json`

**一键脚本**: `run_all.bat` ✓

### 方法2: 从头训练FFN

- [ ] 文件存在: `classifier_model.py`, `train_classifier.py`
- [ ] 运行: `python train_classifier.py`
- [ ] 检查输出: `rgcn_output/checkpoints/best_classifier.pth`
- [ ] 运行: `python inference_classifier.py`
- [ ] 检查结果: `rgcn_output/classifier_predictions.json`

**一键脚本**: `run_classifier.bat` ✓

### 方法3: 预训练+FFN ⭐

- [ ] 文件存在: `classifier_with_pretrained.py`, `train_pretrained_classifier.py`
- [ ] 预训练模型存在: `rgcn_output/checkpoints/best_model.pth` (Siamese RGCN)
- [ ] 运行冻结模式: `python train_pretrained_classifier.py --freeze_encoder`
  - 或运行微调模式: `python train_pretrained_classifier.py`
- [ ] 检查输出: 
  - 冻结: `best_pretrained_classifier_frozen.pth`
  - 微调: `best_pretrained_classifier_finetuned.pth`
- [ ] 运行: `python inference_classifier.py`
- [ ] 检查结果: `rgcn_output/classifier_predictions.json`

**一键脚本**: `run_pretrained_classifier.bat` ✓

---

## 📊 预期输出文件

### 训练输出

```
rgcn_output/
├── checkpoints/
│   ├── best_model.pth                         # Siamese RGCN
│   ├── best_classifier.pth                    # FFN（从头）
│   ├── best_pretrained_classifier_frozen.pth  # FFN（冻结）
│   └── best_pretrained_classifier_finetuned.pth # FFN（微调）⭐
```

### 推理输出

```
rgcn_output/
├── hallucination_predictions.json             # 方法1结果
├── classifier_predictions.json                # 方法2,3结果
├── evaluation_metrics.json                    # 评估指标
├── confusion_matrix.png                       # 混淆矩阵
└── *_distribution.png                         # 分布图
```

---

## 🔍 故障排查

### 问题1: 找不到嵌入文件

**错误**: `FileNotFoundError: entity_embeddings_rgcn.pkl`

**解决**:
```bash
cd "g:\小论文\第三章\GCA-main\Extract triples\dataset\hotpotqa"
python generate_hybrid_embeddings.py
cd rgcn
python prepare_embeddings.py
```

### 问题2: CUDA内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决**:
- 减小batch size: 在 `config_hotpotqa.py` 中设置 `BATCH_SIZE = 4`
- 或使用CPU: 设置 `device = 'cpu'`

### 问题3: 预训练模型不存在

**错误**: `❌ 预训练模型不存在`

**解决**:
```bash
# 先训练Siamese RGCN
python train_rgcn_hotpotqa.py
```

### 问题4: OpenKE导入失败

**错误**: `ModuleNotFoundError: No module named 'openke'`

**解决**:
```bash
# 重新安装OpenKE
cd /path/to/OpenKE
bash make.sh
pip install -e .
```

---

## 🎯 快速验证

运行以下命令快速验证环境：

```bash
# 1. 检查Python包
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "import sentence_transformers; print('SentenceTransformers OK')"

# 2. 检查文件
cd "g:\小论文\第三章\GCA-main\Extract triples\dataset\hotpotqa\rgcn"
dir ..\hybrid_embeddings\*.pkl

# 3. 测试模型加载
python -c "from config_hotpotqa import Config; Config.print_config()"
```

---

## 📝 运行日志模板

记录你的运行过程：

```
日期: ____________________

方法: □ 方法1  □ 方法2  □ 方法3-冻结  □ 方法3-微调

预处理:
  □ 数据提取完成
  □ TransE训练完成
  □ 混合嵌入生成完成
  □ RGCN嵌入准备完成

训练:
  开始时间: ____________________
  结束时间: ____________________
  训练时长: ____________________
  最佳验证准确率: __________

推理:
  推理时间: ____________________
  准确率: __________
  精确率: __________
  召回率: __________
  F1分数: __________

问题记录:
  _____________________________________________
  _____________________________________________
  _____________________________________________

备注:
  _____________________________________________
  _____________________________________________
```

---

## ✅ 最终确认

在运行前，确保：

- [ ] 所有Python依赖已安装
- [ ] CUDA可用（或准备用CPU训练）
- [ ] 数据文件已准备
- [ ] 嵌入文件已生成
- [ ] 有足够的磁盘空间（至少10GB）
- [ ] 有足够的时间（2-4小时）

---

## 🎉 准备就绪！

如果所有检查都通过，你可以开始了：

```bash
cd "g:\小论文\第三章\GCA-main\Extract triples\dataset\hotpotqa\rgcn"

# 推荐：运行方法3-微调
run_pretrained_classifier.bat
# 选择: 2
```

祝你训练顺利！🚀











