
from anyload.config import UniversalDataConfig,TaskType,LoadMode
from anyload.loader import UniversalDataLoader

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    train_path="fake_train.jsonl",
    val_path="fake_val.jsonl",
    task=TaskType.CLM,
    load_mode=LoadMode.DYNAMIC,
    max_length=512,
    batch_size=2,
    filetr_config={"min_chars":100,"loss_max":3.0}
)
loader = UniversalDataLoader(config)
print("Vocab size:", loader.vocab_size)

for batch in loader.train_dataloader():
    # 训练循环
    print(batch.keys())
    break