from anyload import UniversalDataConfig, UniversalDataLoader, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    train_path="huge_data.jsonl",
    load_mode=LoadMode.STREAMING,
    max_length=512,
    batch_size=8,
)
loader = UniversalDataLoader(config)
for i, batch in enumerate(loader.train_dataloader()):
    if i > 5: break
    print(batch["input_ids"].shape)