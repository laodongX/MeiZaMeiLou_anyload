"""
universal_data_loader.py — 一行代码创建训练/验证 DataLoader
"""
import json

import torch
from torch.utils.data import DataLoader
from typing import Optional,Tuple,Dict

from .config import UniversalDataConfig,LoadMode
from .processor import UniversalProcessor
from .dataset import UniversalDatasetFactory,StreamingDataset
from .collator import UniversalCollator

class UniversalDataLoader:
    """
    工业级数据加载器 — 创建即用

    用法:
        config = UniversalDataConfig(
            model_id="Qwen/Qwen2.5-0.5B",
            train_path="pretrain_train.jsonl",
            val_path="pretrain_val.jsonl",
        )
        udl = UniversalDataLoader(config)

        for batch in udl.train_dataloader():
            loss = model(batch["input_ids"], labels=batch["labels"])
    """
    def __init__(self,config:UniversalDataConfig):
        self.config =config
        self.processor = UniversalProcessor(config=config)
        self.collator = UniversalCollator(
            task=config.task,
            modality=config.modality,
            pad_token_id=self.processor.pad_token_id,
            max_length=config.max_length,
            dynamic_padding=(config.padding == "longest")
        )
        self._train_dataset =None
        self._val_dataset =None
        self._test_dataset = None

    @property
    def vocab_size(self):
        """供外部使用"""
        return self.processor.vocab_size

    @property
    def tokenizer(self):
        """供外部使用 如 decode验证"""
        return self.processor.tokenizer

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self._train_dataset = UniversalDatasetFactory.create(config=self.config,
                                                                 processor=self.processor,
                                                                 split="train")
        #StreamingDataset 需要特殊处理

        is_streaming = isinstance(self._train_dataset,StreamingDataset)
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train and not is_streaming,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True#训练集丢弃不完善的batch
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.config.val_path is None:
            return None

        if self._val_dataset is None:
            self._val_dataset = UniversalDatasetFactory.create(
                self.config,self.processor,split="val"
            )

        is_streaming = isinstance(self._val_dataset,StreamingDataset)
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.effective_eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collator,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=False
        )

    def test_dataloader(self) -> Optional[DataLoader]:
         if self.config.test_path is None:
             return None

         if self._test_dataset is None:
             self._test_dataset = UniversalDatasetFactory.create(self.config,self.processor,split='test')

         return DataLoader(
             self._test_dataset,
             batch_size=self.config.effective_eval_batch_size,
             num_workers=self.config.num_workers,
             shuffle=False,
             collate_fn=self.collator,
             pin_memory=self.config.pin_memory

         )

    # ============================================================
    # 便捷方法: 预分词生成 .bin (离线模式用)
    # ============================================================
    def tokenize_to_bin(self,jsonl_path:str,bin_path:str,seql_len:Optional[int] = None):
        """
        将 JSONL 预分词为 .bin 文件 (供 MemmapDataset 使用)
        修复了原 tokenize_and_save.py 的 int16 溢出问题
        """
        seql_len =seql_len or self.config.max_length
        pad_id = self.processor.pad_token_id
        CHUNK = 500000

        def chunked_save(ids_list):
            import numpy as np
            arr = np.full((len(ids_list),seql_len),pad_id,dtype=np.int32) #int32
            for i, ids in enumerate(ids_list):
                t = ids[:seql_len]
                arr[i,:len(t)] = t
            with open(bin_path,'ab') as f:
                f.write(arr.tobytes())

            return len(arr)

        import numpy as np
        count = 0
        with open(jsonl_path,'r',encoding='utf-8') as f:
            buf = []
            for line in f:
                text = json.loads(line).get(self.config.text_col,"")
                ids = self.processor.tokenizer.encode(text)
                buf.append(ids)
                if len(buf) > CHUNK:
                    count += chunked_save(buf)
                    print(f"chunk done ,tottal",count)
                    buf = []

            if buf:#防止最后不够50000没保存到count
                count+=chunked_save(buf)
                buf = []

        print(f"✅ 预分词完成: {count:,} 样本 → {bin_path}")

        #验证
        data = np.memmap(bin_path,dtype=np.int32,mode='r')

        print(f"   验证: min={data.min()}, max={data.max()}, samples={data.size // seql_len}")
        return count
