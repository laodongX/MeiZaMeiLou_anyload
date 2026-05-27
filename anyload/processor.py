"""
universal_processor.py — 统一多模态处理器
自动选择 AutoProcessor / AutoTokenizer, 屏蔽模型差异
"""
import torch
import numpy as np
from typing import Optional,Dict,List,Union,Any
from PIL import Image
from dataclasses import dataclass

from .config import UniversalDataConfig,TaskType,Modality

@dataclass
class ProcessedSample:
    """处理器统一输出格式"""
    input_ids :torch.Tensor  # [seq_len]
    attention_mask:torch.Tensor  # [seq_len]
    labels:torch.Tensor  # [seq_len]

    # 多模态附加 (可选)
    pixel_values:Optional[torch.FloatTensor] = None # [C, H, W]
    input_features:Optional[torch.FloatTensor] = None # [freq, time]
    image_grid_thw:Optional[torch.LongTensor] = None # Qwen2-VL 专用


@dataclass
class UniversalProcessor:
    """
    工业级统一处理器 — 一个类兼容所有模型
    ┌──────────────────────────────────────────────────────┐
    │  Qwen2.5  → AutoTokenizer  → input_ids              │
    │  Qwen2-VL → AutoProcessor → input_ids + pixel_values│
    │  LLaVA    → AutoProcessor → input_ids + pixel_values│
    │  Whisper  → AutoProcessor → input_features + labels │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self,config:UniversalDataConfig):
        self.config = config
        self._setup_processor()
        self._set_up_special_tokens()

    def _setup_processor(self):
        """自动选择 Processor 或 Tokenizer"""
        cfg = self.config

        if cfg.is_multimodal:
            #多模态：优先尝试AutoProcessor
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(self.config.model_id,trust_remote_code=self.config.trust_remote_code)
                self.tokenizer = self.processor.tokenizer
                self._mode = "multimodal_processor"
                print(f"[Processor] 多模态 Processor: {cfg.model_id}")
                return
            except Exception:
                print(f"[Processor] AutoProcessor 加载失败, 回退到 Tokenizer")
        #纯文本活多模态回退
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id,trust_remote_code=self.config.trust_remote_code)
        self.processor = None
        self._mode = "text_tokenizer"
        print(f"[Processor] 文本 Tokenizer: {cfg.model_id}, vocab={self.tokenizer.vocab_size}")

    def _set_up_special_tokens(self):
        """确保 PAD token 存在 (CLM训练必须)"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[Processor] 设置 pad_token = eos_token ({self.tokenizer.eos_token_id})")

    @property
    def vocab_size(self) ->int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id or 0

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    # ============================================================
    # 核心方法: 编码单个样本
    # ============================================================

    def encode(self,sample:Dict[str,Any]) -> ProcessedSample:
        """
            统一编码入口 — 输入原始样本字典, 输出 ProcessedSample
            自动根据配置的任务类型和模态选择编码路径
        """
        cfg = self.config
        if self._mode == "multimodal_processor" and self.processor is not None:
            return self._encode_multimodal(sample)
        else:
            return self._encode_text(sample)

    def _encode_text(self,sample:Dict[str,Any]) -> ProcessedSample:
        """纯文本编码 (支持 CLM / S2S / MLM)"""
        cfg = self.config

        #提取文本
        if cfg.task == TaskType.S2S and cfg.source_col and cfg.target_col:
            source = sample.get(cfg.source_col,"")
            target = sample.get(cfg.target_col,"")
            return self._encode_s2s(source,target)
        else:
            text = sample.get(cfg.text_col,"")
            return self._encode_clm(text)

    def _encode_clm(self,text:str) -> ProcessedSample:
        """CLM 编码: input_ids + shifted labels"""
        cfg = self.config
        encoded = self.tokenizer(
            text,
            truncation = cfg.truncation,
            max_length = cfg.max_length,
            padding=cfg.padding,
            return_tensors = None #返回LIST 然后统一再转 tensor（pt）
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        # CLM: labels = input_ids 右移一位, 末位 pad
        labels = input_ids[1:] + [self.pad_token_id]

        #padding 位置的 label 设为 -100 (ignore_index)
        labels = [
            label if mask ==1 else -100
            for label,mask in zip(labels,attention_mask)
        ]

        return ProcessedSample(
            input_ids=torch.tensor(input_ids,dtype=torch.long),
            labels=torch.tensor(labels,dtype=torch.long),
            attention_mask=torch.tensor(attention_mask,dtype=torch.long)
        )

    def _encode_s2s(self,source:str,target:str) -> ProcessedSample:
        """S2S 编码: source → encoder, target → decoder_labels"""
        cfg = self.config
        model_inputs = self.tokenizer(
            source,
            truncation=cfg.truncation,
            max_length = cfg.max_length,
            padding = cfg.padding,
            return_tensors=None
        )

        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                truncation = cfg.truncation,
                max_length = cfg.max_length,
                padding = cfg.padding,
                return_tensors = None
            )

        #decoder Labels:padding -> -100
        labels = [
            t if t !=self.pad_token_id else -100
            for t in targets['input_ids']
        ]

        return ProcessedSample(
            input_ids=torch.tensor(model_inputs['input_ids'],dtype=torch.long),
            attention_mask=torch.tensor(model_inputs['attention_mask'],dtype=torch.long),
            labels=torch.tensor(model_inputs['labels'],dtype=torch.long)
        )

    def _encode_multimodal(self,sample:Dict[str,Any]) -> ProcessedSample:
        """多模态编码: 统一调用 processor"""
        cfg = self.config

        #构建处理器输入
        processor_inputs ={}

        #文本
        text = sample.get(cfg.text_col,"")
        if text:
            processor_inputs['text'] = text

        #图像
        if cfg.image_col and sample.get(cfg.image_col):
            image_path = sample[cfg.image_col]
            try:
                if isinstance(image_path,str):
                    image = Image.open(image_path).convert("RGB")
                else:
                    image = image_path
                processor_inputs["image"] = image

            except Exception as e:
                print(f"[Processor] 图像加载失败: {e}, 降级为纯文本")

        #音频
        if cfg.audio_col and sample.get(cfg.audio_col):
            try:
                import torchaudio
                audio_path = sample[cfg.audio_col]
                waveform,sr = torchaudio.load(audio_path)
                if sr != cfg.audio_sampling_rate:
                    resampler = torchaudio.transforms.Resample(sr,cfg.audio_sampling_rate)
                    waveform = resampler(waveform)
                processor_inputs['audio'] = waveform.squeeze(0).numpy()

            except Exception as e:
                print(f"[Processor] 音频加载失败: {e}, 降级为纯文本")

        #调用pprocessor
        try:
            encoded = self.processor(
                **processor_inputs,
                truncation = cfg.truncation,
                max_length = cfg.max_length,
                padding=cfg.padding,
                return_tensors="pt"
            )
        except Exception as e:
            print(f"[Processor] 多模态编码失败: {e}, 降级为纯文本")
            return self._encode_clm(text)

        #提取结果
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        #CLM labels
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.pad_token_id
        #padding位置ignore
        labels[attention_mask == 0 ] =-100

        result = ProcessedSample(
            input_ids=torch.tensor(input_ids,dtype=torch.long),
            attention_mask=torch.tensor(attention_mask,dtype=torch.long),
            labels = torch.tensor(labels,dtype=torch.long)
        )

        #附加多模态特征pixel_values,input_features(ct时间步），image_grid_thw
        if "pixel_values" in encoded:
            result.pixel_values = encoded['pixel_values'].squeeze(0)

        if "input_features" in encoded:
            result.input_features = encoded["input_features"].squeeze(0)

        if "image_grid_thw" in encoded:
            result.image_grid_thw = encoded['image_grid_thw'].squeeze(0)

        return result
    # ============================================================
    # 批量编码 (用于 datasets.map)
    # ============================================================
    def batch_encode(self,batch:Dict[str,List]) -> Dict[str,List]:
        """
        datasets.map 专用 — 输入批量化原始样本, 输出批量化 IDs
        返回 Python list (非 tensor), 方便 Arrow 缓存
        """
        cfg = self.config
        texts = batch.get(cfg.text_col,[])

        #S2S 任务专编
        if cfg.task == TaskType.S2S and cfg.source_col:
            sources = batch.get(cfg.source_col,[])
            targets = batch.get(cfg.target_col,[])
            return self._batch_encode_s2s(sources,targets)

        #CLM,VLM,ALM统一走文本编码
        encoded = self.tokenizer(
            texts,
            truncation=cfg.truncation,
            max_length = cfg.max_length,
            padding = cfg.padding

        )

        #CLM Labels:右移+padding ignore
        all_labels = []
        for ids,mask in zip(encoded['input_ids'],encoded['attention_mask']):
            labels = ids[1:] +[self.pad_token_id]
            labels = [l if m ==1 else -100 for l,m in zip(labels,mask)]
            all_labels.append(labels)

        return {
            "input_ids":encoded['input_ids'],
            'attention_mask':encoded['attention_mask'],
            'labels':all_labels
        }

    def _batch_encode_s2s(self,sources,targets):
        cfg = self.config
        model_inputs = self.tokenizer(
            sources,
            truncation=True,
            max_length=cfg.max_length,
            padding = cfg.padding
        )
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                targets,
                truncation=True,
                max_length=cfg.max_length,
                padding=cfg.padding
            )

        labels = [
            [t if t !=self.pad_token_id else -100 for t in ids]
            for ids in target_enc['input_ids']
        ]
        return {
            "input_ids":model_inputs['input_ids'],
            "attention_mask":model_inputs['attention_mask'],
            "labels":labels
        }


