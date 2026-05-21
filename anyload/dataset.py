"""
universal_dataset.py — 三种加载策略统一接口
┌───────────────┐ ┌───────────────┐ ┌──────────────────────┐
│ DynamicDataset │ │ MemmapDataset │ │ StreamingDataset     │
│ (Arrow缓存)    │ │ (mmap .bin)   │ │ (TB级流式)           │
│ 推荐 ✓         │ │ 极致IO ✓      │ │ 超大数据 ✓           │
└───────────────┘ └───────────────┘ └──────────────────────┘
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,IterableDataset
from typing import Dict,Any,Optional,List
from .config import UniversalDataConfig,LoadMode,TaskType,Modality
from .processor import UniversalProcessor,ProcessedSample

# ============================================================
# 策略一: DynamicDataset (datasets库自动Arrow缓存) — 推荐
# ============================================================
class DynamicDataset(Dataset):
    """
    动态分词数据集 — HuggingFace datasets 库驱动
    优势: 自动Arrow缓存, 参数改了自动重处理, 多模态原生支持
    """
    def __init__(self,config:UniversalDataConfig,processor:UniversalProcessor,split:str="train"):
        super(DynamicDataset, self).__init__()
        self.config = config
        self.processor =processor
        self.split = split

        #加载数据集
        data_path = config.train_path if split == "train" else config.val_path
        if data_path is None:
            raise ValueError(f"数据没有找到路径{split}模式")
        self._load_and_process(data_path)

    def _load_and_process(self,data_path:str):
        """ 加载 + 分词 + 自动缓存 """
        from datasets import load_dataset,DatasetDict,Image as HFImage, Audio as HFAudio
        cfg = self.config

        #1.加载原始数据
        dataset = load_dataset('json',data_files=data_path,split='train',cache_dir=cfg.cache_dir)

        #2.多模态列类型转换
        if cfg.image_col and cfg.image_col in dataset.column_names:
            dataset = dataset.cast_column(cfg.image_col,HFImage())

        if cfg.audio_col and cfg.audio_col in dataset.column_names:
            dataset = dataset.cast_column(cfg.audio_col,HFAudio(sampling_rate=cfg.audio_sampling_rate))

        #3质量过滤 如果有
        if cfg.filetr_config:
            dataset = self._apply_filters(dataset,cfg.filetr_config)


        #4 分词 利用datasets.map自动缓存
        columns_to_remove = [c for c in dataset.column_names if c not in ['input_ids','attention_mask','labels']]
        if cfg.is_multimodal and self.processor._mode == "multimodal_processor":
            # 多模态: 逐条处理 (processor 不支持简单 batched)
            def process_fn(example):
                result = self.processor.encode(sample=example)
                out = {
                    "input_ids":result.input_ids.tolist(),
                    "attention_mask":result.attention_mask.tolist(),
                    "labels":result.labels.tolist()
                }
                if result.pixel_values:
                    out['pixel_values'] = result.pixel_values.numpy().tolist()
                if result.input_features:
                    out['input_features'] = result.input_features.numpy().tolist()

                return out

            self.dataset = dataset.map(
                process_fn,
                remove_columns = columns_to_remove,
                desc=f"DynamicDataset处理{self.split}"
            )

        else:
            #纯文本则批量化处理 快速度
            self.dataset = dataset.map(
                self.processor.batch_encode,
                batched=True,
                remove_columns = columns_to_remove,
                desc = f"DynamicDataset处理{self.split}"
            )

        #5设定格式为pytorch
        tensor_columns = ['input_ids','attention_mask','labels']
        if "pixel_values" in self.dataset.column_names:
            tensor_columns.append("pixel_values")

        if "input_features" in self.dataset.column_names:
            tensor_columns.append("input_features")

        self.dataset.set_format(
            type="torch",columns=tensor_columns
        )
        print(f"[DynamicDataset] {self.split}: {len(self.dataset):,} 样本已就绪")

    def _apply_filters(self,dataset,filter_cfg:Dict):
        """应用质量过滤 (对应 prepare_cci40.py 的逻辑)"""
        def filter_fn(example):
            text = example.get(self.config.text_col,"")
            meta = example.get("metadata",{})

            #长度过滤
            if filter_cfg.get("min_chars") and len(text) < filter_cfg['min_chars']:
                return False
            if filter_cfg.get("max_chars") and len(text) > filter_cfg['max_chars']:
                return False
            #质量指标过滤
            if filter_cfg.get("loss_max") and isinstance(meta,dict):
                if meta.get("loss",99) > filter_cfg['loss_max']:
                    return False

            if filter_cfg.get("toxicity_max") and isinstance(meta,dict):
                if meta.get("toxicity",1) > filter_cfg['toxicity_max']:
                    return False
            return True

        before = len(dataset)
        dataset = dataset.filter(filter_fn,desc="质量过滤")
        after = len(dataset)
        print(f"[DynamicDataset] 质量过滤: {before:,} → {after:,} ({after / before * 100:.1f}%)")
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


# ============================================================
# 策略二: MemmapDataset (mmap直读预分词.bin) — 极致IO
# ============================================================
class MemmapDataset(Dataset):
    """
        内存映射数据集 — 读取预分词的 .bin 文件
        优势: 零拷贝, 极致IO, 适合TB级固定数据
        要求: .bin 文件 dtype=np.int32, shape=[N, seq_len]
        """
    def __init__(self,config:UniversalDataConfig,processor:UniversalProcessor):
        super(MemmapDataset, self).__init__()
        cfg = config
        self.seq_len = cfg.max_length
        self.pad_id = processor.pad_token_id

        bin_path = cfg.memap_bin_path
        if bin_path is None:
            raise ValueError("memmapDataset需要memmap_bin_path")

        #mmap加载
        self.data = np.memmap(bin_path,dtype=np.int32,mode='r')
        assert self.data.size % self.seq_len == 0 ,(f"bin 文件大小 {self.data.size} 不是 seq_len={self.seq_len} 的整数倍")
        self.num_samples = self.data.size // self.seq_len

        #stride 子采样
        stride = cfg.stride or 1
        self.indices = np.arange(0,self.num_samples,stride)
        print(f"[MemmapDataset] {bin_path}: {self.num_samples:,} 样本, "
              f"stride={stride}, 使用 {len(self.indices):,}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        start = self.indices[item] * self.seq_len
        end = start +self.seq_len
        ids = self.data[start:end].copy()
        input_ids = torch.from_numpy(ids).long()

        #CLM Labels : 右移 + padding_ignore
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        labels[labels == self.pad_id] =-100

        attention_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids":input_ids,
            "attention_mask":attention_mask,
            "labels" : labels
        }

# ============================================================
# 策略三: StreamingDataset (TB级流式加载) — 超大数据
# ============================================================
class StreamingDataset(IterableDataset):
    def __init__(self,config:UniversalDataConfig,processor:UniversalProcessor,split:str = "train"):
        super(StreamingDataset, self).__init__()
        self.config = config
        self.processor = processor
        self.split = split
        self.data_path = config.train_path if split == "train" else config.val_path

    def __iter__(self):
        """逐行读取 + 实时分词"""
        import random
        rng = random.Random(self.config.seed)

        #收集偏移量（二进制模式！ 修复源代码的文本模式bug）
        offsets = []
        with open(self.data_path,'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        #shuffle
        rng.shuffle(offsets)

        #流式逐条处理
        for offset in offsets:
            with open(self.data_path,'rb') as f:
                f.seek(offset)
                line = f.readline()

            try:
                sample = json.loads(line.decode('utf-8'))
            except json.JSONDecodeError:
                continue

            #质量过滤
            if not self._pass_filter(sample):
                continue

            result = self.processor.encode(sample)
            yield  {
                "inputs_ids":result.input_ids,
                "attention_mask":result.attention_mask,
                "labels":result.labels,
                "pixel_values":result.pixel_values,
                "input_features":result.input_features,
            }

    def _pass_filter(self,sample:Dict) -> bool:
            cfg =self.config
            if cfg.filetr_config is None:
                return True
            text =sample.get(cfg.text_col,'')
            meta = sample.get('metadata',{})
            fc = cfg.filetr_config

            if fc.get('min_chars') and len('text') < fc['min_chars']:
                return False

            if fc.get("max_chars") and len(text) > fc['max_chars']:
                return False

            if isinstance(meta,dict):
                if fc.get("loss_max") and meta.get("loss",99) > fc['loss_max']:
                    return False

                if fc.get('toxicity_max') and meta.get("toxicity",1) > fc['toxicity_max']:
                    return False

            return  True


# ============================================================
# 工厂: 根据配置自动选择策略
# ============================================================
class UniversalDatasetFactory:
    """根据 LoadMode 自动创建对应 Dataset"""
    # @staticmethod
    def create(config:UniversalDataConfig,processor:UniversalProcessor,split:str = "train"):
        if config.load_mode == LoadMode.DYNAMIC:
            return DynamicDataset(config,processor,split)
        elif config.load_mode == LoadMode.MEMAP:
            return MemmapDataset(config,processor)
        elif config.load_mode == LoadMode.STREAMING:
            return StreamingDataset(config,processor,split)
        else:
            raise ValueError(f"未知load_mode{config.load_mode}")

