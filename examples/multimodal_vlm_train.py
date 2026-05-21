from anyload.config import UniversalDataConfig, TaskType, Modality
from anyload.loader import UniversalDataLoader
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
    # batch 包含 input_ids, labels, pixel_values 等
    print(batch.keys())
    break