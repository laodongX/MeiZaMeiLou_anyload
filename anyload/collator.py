"""
universal_collator.py — 统一批整理器
处理: 动态填充 / CLM标签移位 / 多模态特征对齐 / 混合模态批处理
"""
import torch
from dataclasses import dataclass
import torch
from typing import Dict, List, Any,Optional
from .config import UniversalDataConfig, TaskType, Modality
@dataclass
class UniversalCollator:
    """
    通用批整理器 — 一个类覆盖所有任务和模态的 collation 逻辑

    功能:
    1. 动态填充到 batch 内最长序列 (或固定 max_length)
    2. CLM 标签自动移位 (若 Dataset 未做)
    3. 多模态特征对齐 (pixel_values / input_features 堆叠)
    4. attention_mask 生成
    5. ignore_index 统一为 -100
    """
    task:TaskType =TaskType.CLM
    modality:Modality = Modality.TEXT
    pad_token_id:int =0
    max_length:Optional[int] = -100 # None=动态填充到batch最长
    dynamic_padding:bool = True # True=最长填充, False=固定max_length
    ignore_index:int = -100
    def __call__(self, batch:List[Dict[str,Any]]) -> Dict[str,torch.Tensor]:
        """主入口"""
        # 区分: batch 来自 DynamicDataset (已有 input_ids/labels) 还是 StreamingDataset
        first = batch[0]

        if "input_ids" in first:
            return self._collated_tokenized(batch)
        else:
            raise   ValueError("请使用 DynamicDataset 或 StreamingDataset 预先分词")

    def _collated_tokenized(self,batch:List[Dict[str,Any]]) -> Dict[str,torch.Tensor]:
        """整理已分词的 batch"""
        # ── 基础字段: input_ids, attention_mask, labels ──
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [ item['labels'] for item in batch]
        attention_mask_list = [item.get("attention_mask") for item in batch]

        #确保是tensor
        input_ids_list = [t if isinstance(t,torch.Tensor) else torch.tensor(t) for t in input_ids_list]
        labels_list = [t if isinstance(t,torch.Tensor) else torch.tensor(t) for t in labels_list]

        #动态填充
        if self.dynamic_padding:
            max_len = self.max_length or max(t.size(0) for t in input_ids_list)
        else:
            max_len = self.max_length or 512

        input_ids = self._pad_sequence(input_ids_list,max_len,self.pad_token_id)
        labels = self._pad_sequence(labels_list,max_len,self.ignore_index)

        # attention_mask
        if attention_mask_list[0] is not None:
            attention_mask_list = [t if isinstance(t,torch.Tensor) else torch.tensor(t) for t in attention_mask_list]
            attention_mask = self._pad_sequence(attention_mask_list,max_len,0)

        else:
            attention_mask = (input_ids != self.pad_token_id)

        result = {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels":labels
        }

        if "pixel_values" in batch[0] and batch[0]['pixel_values'] is not None:
            pixel_values = torch.stack([
                item['pixel_values'] if isinstance(item['pixel_values'],torch.Tensor) else torch.tensor(item['pixel_values'])for item in batch
            ],dim=0)
            result['pixel_values'] = pixel_values

        if "input_features" in batch[0] and batch[0]['input_features'] is not None:
            input_features = torch.stack([
                item['input_features'] if isinstance(item['input_features'],torch.Tensor)
                else torch.tensor(item['input_features']) for item in batch
            ])
            result['input_features'] = input_features

        if 'image_grid_thw' in batch[0] and batch[0]['image_grid_thw'] is not None:
            result['image_grid_thw'] = torch.stack([
                item['image_grid_thw'] if isinstance(item['image_grid_thw'],torch.Tensor)
                else torch.tensor(item['image_grid_thw']) for item in batch
            ])

        return result

    def _pad_sequence(self,sequences:List[torch.Tensor],max_len:int,pad_value:int):
        """将变长序列填充到统一长度"""
        padded = [ ]
        for seq in sequences:
            if seq.dim() ==0:
                seq = seq.unsqueeze(0)
            curr_len = seq.size(0)
            if curr_len >= max_len:
                padded.append(seq[:max_len])
            else:
                padd_size = max_len - curr_len
                padding = torch.full((padd_size,),pad_value,dtype=seq.dtype)
                padded.append(torch.cat([seq,padding],dim=0))

        return torch.stack(padded,dim=0)









