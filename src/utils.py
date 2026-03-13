# src/utils.py
import logging
import sys
import os
import json
from pathlib import Path

import yaml


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = 'logs', level: str = 'INFO') -> None:
    """
    Sets up logging to both stdout and a rotating file.
    Call this once at the start of main.py.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt = '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{log_dir}/app.log', encoding='utf-8'),
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Loads YAML config. Raises FileNotFoundError if not found."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Config not found: {config_path}\n'
            f'Create one from the template in configs/config.yaml'
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Model Metadata ────────────────────────────────────────────────────────────

def save_metadata(
    class_names: list,
    img_size: int,
    architecture: str = 'EfficientNetV2S',
    path: str = 'models/metadata.json'
) -> None:
    """
    Saves class names, image size, and architecture to a JSON file.
    This file is required by the Streamlit app at inference time.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'class_names':  class_names,
        'num_classes':  len(class_names),
        'img_size':     img_size,
        'architecture': architecture,
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    logging.getLogger(__name__).info(f'Metadata saved to {path}')


def load_metadata(path: str = 'models/metadata.json') -> dict:
    """Loads the metadata JSON saved by save_metadata()."""
    with open(path) as f:
        return json.load(f)


# ── Environment Variables ─────────────────────────────────────────────────────

def get_env(key: str, default=None):
    """
    Reads an environment variable. Raises EnvironmentError if
    the key is required (no default) and not set.

    Usage:
        MODEL_DIR = get_env('MODEL_DIR', 'models/saved/final_model')
        SECRET_KEY = get_env('SECRET_KEY')  # raises if missing
    """
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.\n"
            f"Copy .env.example to .env and fill in the value."
        )
    return val
