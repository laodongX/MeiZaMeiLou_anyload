from anyload import UniversalDataConfig, TaskType

def test_config_creation():
    cfg = UniversalDataConfig(model_id="test", train_path="dummy.jsonl")
    assert cfg.task == TaskType.CLM