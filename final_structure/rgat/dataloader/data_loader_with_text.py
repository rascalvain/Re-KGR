"""
HotpotQA 数据加载器（支持图+文本）
"""
import json
import torch
import pickle
import numpy as np
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import os

# 导入原始数据加载器
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rgcn'))
from dataloader.data_loader_hotpotqa import HotpotQAGraphDataset as BaseDataset


class HotpotQAGraphTextDataset(BaseDataset):
    """HotpotQA 图+文本数据集（继承原始数据加载器）"""

    def __getitem__(self, idx):
        """
        获取一个样本（图+文本）
        Returns:
            tuple: (context_graph, gpt_graph, label, gpt_text, metadata)
        """
        # 调用父类方法获取图数据
        result = super().__getitem__(idx)

        if result is None:
            return None

        context_graph, gpt_graph, label, metadata = result

        # 🔥 获取GPT生成的原始文本
        item = self.valid_data[idx]
        gpt_text = item.get('gpt_response', '')  # GPT生成的文本

        # 如果没有gpt_response字段，尝试其他字段
        if not gpt_text:
            gpt_text = item.get('gpt_sentence', '')

        # 如果仍然没有文本，使用answer作为备选
        if not gpt_text:
            gpt_text = item.get('answer', '')

        # 清理文本
        gpt_text = str(gpt_text).strip()

        # 添加到metadata
        metadata['gpt_text'] = gpt_text

        return context_graph, gpt_graph, label, gpt_text, metadata


def collate_fn_with_text(batch):
    """
    自定义批处理函数（支持文本）
    Args:
        batch: 样本列表
    Returns:
        (context_batch, gpt_batch, labels, gpt_texts, metadata_list)
    """
    # 过滤 None 样本
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None, None, None, None, None

    context_graphs = []
    gpt_graphs = []
    labels = []
    gpt_texts = []  # 🔥 新增：文本列表
    metadata_list = []

    for context_graph, gpt_graph, label, gpt_text, metadata in batch:
        if context_graph is not None and gpt_graph is not None:
            context_graphs.append(context_graph)
            gpt_graphs.append(gpt_graph)
            labels.append(label)
            gpt_texts.append(gpt_text)
            metadata_list.append(metadata)

    if len(context_graphs) == 0:
        return None, None, None, None, None

    # 批处理图
    context_batch = Batch.from_data_list(context_graphs)
    gpt_batch = Batch.from_data_list(gpt_graphs)
    labels = torch.LongTensor(labels)

    # 🔥 文本保持为列表（不需要转换为tensor）

    return context_batch, gpt_batch, labels, gpt_texts, metadata_list