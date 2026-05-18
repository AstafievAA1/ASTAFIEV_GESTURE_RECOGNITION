import os
import yaml
import torch
from typing import Dict, Optional

def load_config(config_path: str = "config.yaml") -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")
    return device

def find_latest_model(models_path: str) -> Optional[str]:
    if not os.path.exists(models_path):
        return None
    models = [f for f in os.listdir(models_path) if f.endswith('.pth')]
    if not models:
        return None
    models_with_time = [
        (f, os.path.getmtime(os.path.join(models_path, f)))
        for f in models
    ]
    models_with_time.sort(key=lambda x: x[1], reverse=True)
    return os.path.join(models_path, models_with_time[0][0])

