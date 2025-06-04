from dataclasses import dataclass
from typing import Any
from types import SimpleNamespace

import os
import yaml

@dataclass
class BaseConfig:
    model_name: str
    batch_size: int
    epochs: int
    lr: float
    valid_freq: int
    save_path: str
    params: Any

    @classmethod
    def load_config(cls, model_name: str) -> 'BaseConfig':
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
        cfg = configs[model_name]
        cfg['params'] = SimpleNamespace(**cfg['params'])
        return cls(**cfg)