import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# sets random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# loads configuration from yaml file
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# configures python logging
def setup_logging(log_dir="logs/", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / "logicheck.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

# gets the active device for pytorch
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.getLogger(__name__).info(f"GPU: {name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).warning("No GPU detected -- training on CPU (slow).")
    return device

# counts trainable model parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M"
