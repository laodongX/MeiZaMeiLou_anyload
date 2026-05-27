
### 🚀 MeiZaMeiLou_anyload

![Python 3.8+](https://www.python.org/downloads/)
![License: MIT](https://opensource.org/licenses/MIT)

**Universal, multi-modal, config-driven data loading pipeline for LLM / VLM / ALM training.**

> **Semantic fidelity-first data pipeline.**  
> Every "translation" layer from disk to GPU loses semantics—we help you see where it leaks.

---

### ❓ Why You Need This

Have you faced these pain points during model training?
- **IO Bottleneck**: Data flows from Parquet/MySQL → Python → Tensor → GPU, slower than GPU computation.
- **Maintenance Hell**: Changing tokenizer or sequence length requires rewriting data scripts.
- **Information Leakage**: A layer "looks fine" but silently loses critical semantic information.

**`anyload` solves this**: Make every step from disk to GPU configurable, measurable, and optimizable.

---

### ✨ Core Features

- **🧠 Semantic Fidelity First**: `Memnode` interface reserved for future "fidelity-driven pruning".
- **🌐 Unified Multi-Modal**: Text / Image / Audio processed through single `UniversalProcessor`.
- **⚡ Three Loading Modes**:
  - **Dynamic** (Default): HuggingFace `datasets` + Arrow caching, auto-reprocess on config change.
  - **Memmap**: Zero-copy `.bin` files for extreme IO performance.
  - **Streaming**: Constant memory, line-by-line processing for TB-scale data.
- **🧼 Built-in Quality Filters**: Length, Loss, Toxicity filtering with one config.
- **🔌 Zero-Intrusion Integration**: Outputs standard `input_ids, attention_mask, labels` compatible with existing training loops.

---

### 🚀 Quick Start

#### 1. Installation

```bash
git clone https://github.com/laodongX/MeiZaMeiLou_anyload.git
cd MeiZaMeiLou_anyload
pip install -e .
```

#### 2. Text CLM Training (e.g., Qwen2.5)

```python
from anyload import UniversalDataConfig, UniversalDataLoader, TaskType, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    train_path="pretrain_train.jsonl",
    val_path="pretrain_val.jsonl",
    task=TaskType.CLM,
    load_mode=LoadMode.DYNAMIC,
    max_length=512,
    batch_size=2,
    filter_config={"min_chars": 100, "loss_max": 3.0},
)

loader = UniversalDataLoader(config)
for batch in loader.train_dataloader():
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    # ... your training step ...
```

#### 3. Multi-Modal VLM Training (e.g., Qwen2-VL)

```python
from anyload import UniversalDataConfig, UniversalDataLoader, TaskType, Modality

config = UniversalDataConfig(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    train_path="multimodal_train.jsonl",
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
```

#### 4. Memmap Extreme IO Mode

```python
from anyload import UniversalDataConfig, UniversalDataLoader, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    load_mode=LoadMode.MEMMAP,
    memmap_bin_path="pretrain_train_tokens.bin",
    max_length=512,
    batch_size=4,
    stride=1,
)

loader = UniversalDataLoader(config)
for batch in loader.train_dataloader():
    # Zero-copy read
    pass
```

---

### 🧪 Semantic Fidelity Integration

Pair with `fidelity-metrics` to scan semantic leakage across model layers.

```python
from fidelity_metrics import SemanticFidelityProbe

probe = SemanticFidelityProbe(dim=768)
# Scan fidelity across model layers on validation set
report = probe(z_layer4, z_layer8)
print(report) 
# Output: {'structural': 0.85, 'distributional': 0.78, 'combined': 0.815}

# If combined < 0.6 at any layer, critical information loss occurs—adjust strategy.
```

---

### 🗺️ Roadmap

- [x] Unified multi-modal data pipeline
- [x] Dynamic / Memmap / Streaming modes
- [ ] Integrated `fidelity-metrics` probes
- [ ] Arrow / Iceberg zero-copy GPU direct access
- [ ] Edge-device optimization (NPU-friendly ops, quantization)

---

### 📄 License

MIT License

---

### 🚀 MeiZaMeiLou_anyload

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Universal, multi-modal, config-driven data loading pipeline for LLM / VLM / ALM training.**

> **语义保真度驱动的通用数据管线。**
> 从磁盘到 GPU，每一层“翻译”都在丢失语义——我们帮你看到丢在了哪里。

---

### ❓ 为什么需要这个库？

训练大模型时，你是否遇到过这些痛点？
- **IO 瓶颈**：数据从 Parquet/MySQL → Python → Tensor → GPU，速度比 GPU 还慢。
- **维护困难**：换个分词器或序列长度，就要重写一遍数据脚本。
- **信息断流**：模型某层“看起来没问题”，但信息其实已经悄悄丢失。

**`anyload` 的目标**：让数据从磁盘到 GPU 的每一步都可配置、可度量、可优化。

---

### ✨ 核心特性

- **🧠 语义保真度优先**：预留 `Memnode` 接口，为后续“保真度驱动淘汰”铺路。
- **🌐 多模态统一**：文本 / 图像 / 音频，统一通过 `UniversalProcessor` 处理。
- **⚡ 三种加载模式**：
  - **Dynamic** (默认)：基于 HuggingFace `datasets` + Arrow 缓存，配置更改自动重处理。
  - **Memmap**：零拷贝 `.bin` 文件，极致 IO 性能。
  - **Streaming**：恒定内存，逐行处理 TB 级数据。
- **🧼 内置质量过滤**：支持长度、Loss、Toxicity 一键过滤。
- **🔌 0 侵入接入**：输出标准 `input_ids, attention_mask, labels`，兼容现有训练循环。

---

### 🚀 快速开始

#### 1. 安装

```bash
git clone https://github.com/laodongX/MeiZaMeiLou_anyload.git
cd MeiZaMeiLou_anyload
pip install -e .
```

#### 2. 纯文本 CLM 训练 (如 Qwen2.5)

```python
from anyload import UniversalDataConfig, UniversalDataLoader, TaskType, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    train_path="pretrain_train.jsonl",
    val_path="pretrain_val.jsonl",
    task=TaskType.CLM,
    load_mode=LoadMode.DYNAMIC,
    max_length=512,
    batch_size=2,
    filter_config={"min_chars": 100, "loss_max": 3.0},
)

loader = UniversalDataLoader(config)
for batch in loader.train_dataloader():
    input_ids = batch["input_ids"].cuda()
    labels = batch["labels"].cuda()
    # ... 你的训练步骤 ...
```

#### 3. 多模态 VLM 训练 (如 Qwen2-VL)

```python
from anyload import UniversalDataConfig, UniversalDataLoader, TaskType, Modality

config = UniversalDataConfig(
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    train_path="multimodal_train.jsonl",
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
    # ... VLM 训练 ...
```

#### 4. Memmap 极致 IO 模式

```python
from anyload import UniversalDataConfig, UniversalDataLoader, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    load_mode=LoadMode.MEMMAP,
    memmap_bin_path="pretrain_train_tokens.bin",
    max_length=512,
    batch_size=4,
    stride=1,
)

loader = UniversalDataLoader(config)
for batch in loader.train_dataloader():
    # 零拷贝读取
    pass
```

---

## 🧪 与语义保真度联动

配合 [Fidelity-metrics](https://github.com/laodongX/Fidelity-metrics)，你可以诊断模型各层的信息保真度 ：

```python
from fidelity_metrics import SemanticFidelityProbe

probe = SemanticFidelityProbe(dim=768)
report = probe(z_layer4, z_layer8)
# {'structural': 0.85, 'distributional': 0.78, 'combined': 0.815}
\`\`\`

一旦某层 combined < 0.6，说明信息在这里"断流"了。
```

---

### 🗺️ 路线图

- [x] 多模态统一数据管线
- [x] Dynamic / Memmap / Streaming 三模式
- [ ] 集成 `fidelity-metrics` 探针
- [ ] Arrow / Iceberg 零拷贝直通 GPU
- [ ] 为端侧场景优化 (NPU 友好算子、量化支持)

---

### 📄 许可证

MIT License