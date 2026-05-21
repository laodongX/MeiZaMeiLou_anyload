# MeiZaMeiLou_anyload 🚀

**Universal, multi-modal, config‑driven data loading pipeline for LLM / VLM / ALM training.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features
- 🧠 **Config‑driven** – One `dataclass` controls everything: dataset, model, modality, sequence strategy, loading mode.
- 🌐 **Multi‑modal unified** – Text, images, audio, video processed through a single `UniversalProcessor` (AutoProcessor / AutoTokenizer).
- ⚡ **Three loading strategies** –  
  - **Dynamic** (default): HuggingFace `datasets` with Arrow caching, auto‑reprocess on config change.  
  - **Memmap**: zero‑copy memory‑mapped `.bin` files for extreme I/O.  
  - **Streaming**: constant memory, line‑by‑line processing for TB‑scale data.
- 🏷️ **Auto CLM/S2S labelling** – `UniversalCollator` handles label shifting, padding, and ignore index automatically.
- 🧹 **Built‑in quality filtering** – Length, loss, toxicity filters included.
- 🐛 **Bug‑free by design** – Binary mode offsets, `int32` dtypes, proper pad token handling – all common pitfalls are fixed.
- 🔌 **Drop‑in replacement** – Returns standard `input_ids, attention_mask, labels` dicts, compatible with most training loops.

## 📦 Installation
```bash
git clone https://github.com/yourusername/anyload.git
cd anyload
pip install -e .
🚀 Quick Start
1. Pure text CLM (e.g. Qwen2.5)
from anyload import UniversalDataConfig, UniversalDataLoader, TaskType, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    train_path="data/train.jsonl",
    val_path="data/val.jsonl",
    task=TaskType.CLM,
    load_mode=LoadMode.DYNAMIC,
    max_length=512,
    batch_size=4,
    filter_config={"min_chars": 100, "loss_max": 3.0, "toxicity_max": 0.1},
)

loader = UniversalDataLoader(config)
vocab_size = loader.vocab_size

for batch in loader.train_dataloader():
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    # ... training step ...


2. Multimodal VLM (e.g. Qwen2-VL)
config = UniversalDataConfig(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    train_path="multimodal.jsonl",
    task=TaskType.VLM,
    modality=Modality.IMAGE_TEXT,
    image_col="image",
    max_length=1024,
    batch_size=1,
)

loader = UniversalDataLoader(config)
for batch in loader.train_dataloader():
    input_ids = batch["input_ids"].cuda()
    pixel_values = batch["pixel_values"].cuda()
    labels = batch["labels"].cuda()
    # ... VLM training ...