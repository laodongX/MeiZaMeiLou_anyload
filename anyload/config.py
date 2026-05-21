"""
universal_data_config.py — 工业级数据配置中心
支持: 文本 / 图文 / 音频 / 视频多模态, CLM/S2S任务, 多种加载策略
"""
from dataclasses import dataclass
from typing import Optional,List,Dict,Union
from enum import Enum

class TaskType(Enum):
    CLM = "clm" # 因果语言模型 (GPT/Qwen/LLaMA)
    S2S = "s2s" # 序列到序列 (T5/BART)
    MLM = 'mlm' # 掩码语言模型 (BERT)
    VLM = "vlm" # 视觉语言模型 (Qwen2-VL/LLaVA)
    ALM = "alm" # 音频语言模型 (Whisper/Qwen-Audio)

class LoadMode(Enum):
    DYNAMIC = "dynamic" # 动态分词 (datasets库自动缓存Arrow)
    MEMAP = "memap" # 离线预分词 (mmap直读.bin)
    STREAMING = "streaming" # 流式加载 (TB级数据)

class Modality(Enum):
    TEXT = "text"
    IMAGE_TEXT = "image_text"
    AUDIO_TEXT = "audio_text"
    VIDEO_TEXT = "video_text"
    MULTI = "multi" #全模态混合

@dataclass
class UniversalDataConfig:
    """工业级通用数据配置 — 一个dataclass覆盖所有场景"""

    # ── 数据源 ──
    train_path: str = "pretrain_train.jsonl"
    val_path:Optional[str] = None
    test_path:Optional[str] = None

    ## 多数据集混合 (name:weight 格式, 支持领域加权采样)
    mixed_datasets:Optional[Dict[str,float]] = None #{"jsonl_path" : 0.7, "another":0.3}

    # ── 模型 ──
    model_id :str = "Qwen/Qwen2.5-0.5B"
    trust_remote_code:bool = True

    # ── 任务 & 模态 ──
    task:TaskType = TaskType.CLM
    modality:Modality = Modality.TEXT

    ## ── 序列策略 ──
    max_length :int = 512
    truncation:bool = True
    padding:str = "max_length"  # "max_length" | "longest"
    stride:Optional[int] = None # 滑窗步长 (None=不滑窗)

    # ── 加载策略 ──
    load_mode:LoadMode = LoadMode.DYNAMIC
    memap_bin_path :Optional[str] =None # LoadMode.MEMMAP时指定.bin路径

    # ── 多模态列映射 ──
    text_col:str = "text"
    image_col:Optional[str] = None
    audio_col:Optional[str] = None
    video_col:Optional[str] =None

    #DATALOADER
    batch_size:int = 2
    eval_batch_size:Optional[int] = None
    num_workers:int = 0
    pin_memory:bool = True
    prefetch_factor:Optional[int] = None
    shuffle_train:bool = True

    #高级选项
    filetr_config:Optional[Dict] = None   # {"loss_max": 3.0, "toxicity_max": 0.1, ...}
    cache_dir:Optional[str] = None   # datasets缓存目录
    streaming_buffer_size:int = 1000 # 流式加载缓冲
    seed:int =42

    #S2S 专用
    source_col:Optional[str] = None
    target_col:Optional[str] =None

    #VLM 专用
    image_resolution:int =224
    vision_model_id :Optional[str] = None # 视觉编码器 (默认同model_id)

    #audio 专用
    audio_sampling_rate = 16000

    @property
    def effective_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size


    @property
    def is_multimodal(self) ->bool  :
        return self.modality != Modality.TEXT

    def __post_init__(self):
        #类型兼容 ： 允许字符串输入
        if isinstance(self.task,str):
            self.task = TaskType(self.task)

        if isinstance(self.modality,str):
            self.modality = Modality(self.modality)

        if isinstance(self.load_mode,str):
            self.load_mode = LoadMode(self.load_mode)

            

