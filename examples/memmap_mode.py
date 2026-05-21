from anyload import UniversalDataConfig, UniversalDataLoader, LoadMode

config = UniversalDataConfig(
    model_id="Qwen/Qwen2.5-0.5B",
    load_mode=LoadMode.MEMAP,
    memap_bin_path="pretrain_train_tokens.bin",
    max_length=512,
    batch_size=4,
    stride=1,
)
loader = UniversalDataLoader(config)
print("Total samples:", len(loader._train_dataset))