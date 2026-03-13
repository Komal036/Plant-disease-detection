# tests/test_preprocessing.py
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ── Helper ────────────────────────────────────────────────────────────────────

def create_dummy_dataset(root: Path, class_names: list,
                         n_per_class: int = 20, img_size: int = 64):
    """Creates a fake image dataset with random RGB images."""
    for cls in class_names:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f'{i}.jpg')


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_datasets_returns_four_values():
    from src.data_preprocessing import build_datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        create_dummy_dataset(root, ['Healthy', 'EarlyBlight', 'LateBlight'],
                             n_per_class=20)
        result = build_datasets(str(root), img_size=64, batch_size=8)
        assert len(result) == 4, 'Expected (train_ds, val_ds, test_ds, class_names)'


def test_class_names_are_sorted():
    from src.data_preprocessing import build_datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        classes = ['Tomato_Blight', 'Healthy', 'Potato_Rot']
        create_dummy_dataset(root, classes, n_per_class=15)
        _, _, _, class_names = build_datasets(str(root), img_size=32, batch_size=4)
        assert class_names == sorted(classes)


def test_image_shapes_and_normalisation():
    from src.data_preprocessing import build_datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        create_dummy_dataset(root, ['A', 'B', 'C'], n_per_class=20)
        train_ds, _, _, _ = build_datasets(
            str(root), img_size=64, batch_size=8)
        for imgs, labels in train_ds.take(1):
            assert imgs.shape[1:] == (64, 64, 3), 'Wrong image shape'
            assert imgs.numpy().max() <= 1.0, 'Images not normalised (max > 1)'
            assert imgs.numpy().min() >= 0.0, 'Images not normalised (min < 0)'


def test_no_data_leakage():
    """
    All samples must appear in exactly one split.
    Train + Val + Test sizes must sum to total images.
    """
    from src.data_preprocessing import build_datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        n_per_class = 30
        classes = ['A', 'B']
        total = n_per_class * len(classes)
        create_dummy_dataset(root, classes, n_per_class=n_per_class)

        train_ds, val_ds, test_ds, _ = build_datasets(
            str(root), img_size=32, batch_size=4,
            val_split=0.15, test_split=0.10)

        count = lambda ds: sum(1 for _ in ds.unbatch())
        train_n = count(train_ds)
        val_n   = count(val_ds)
        test_n  = count(test_ds)

        assert train_n + val_n + test_n == total, (
            f'Data leakage or loss: {train_n}+{val_n}+{test_n} != {total}'
        )
        assert train_n > val_n,  'Train set should be larger than val set'
        assert train_n > test_n, 'Train set should be larger than test set'


def test_missing_data_dir_raises():
    from src.data_preprocessing import build_datasets
    with pytest.raises(FileNotFoundError):
        build_datasets('/nonexistent/path/to/data')
