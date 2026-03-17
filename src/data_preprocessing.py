# src/data_preprocessing.py

import os
import random
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int = 42):
    """Ensure fully reproducible training runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f'Global seed set to {seed}')


def build_datasets(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
    **kwargs   # absorbs any extra config keys safely
):
    """
    Loads images from a directory structured as:
        data_dir/
            ClassName1/
                img1.jpg ...
            ClassName2/
                ...

    Returns:
        train_ds, val_ds, test_ds  (tf.data.Dataset)
        class_names                (list of str, sorted)
        y_train                    (list of int, for class-weight computation)

    Key design decisions:
    - Stratified split prevents class imbalance in any fold
    - EfficientNetV2S preprocess_input applied instead of /255 normalisation
    - Augmentation applied ONLY to training set (no leakage)
    - val and test sets are cached in memory after first epoch
    - Fixed seed guarantees identical splits across runs
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Dataset directory not found: {data_dir}')

    # ── Collect all image paths and integer labels ───────────────────────────
    all_paths, all_labels = [], []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    if len(class_names) == 0:
        raise ValueError(f'No subdirectories (classes) found in {data_dir}')

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    for cls in class_names:
        cls_dir = data_dir / cls
        imgs = [p for p in cls_dir.iterdir() if p.suffix in VALID_EXTS]
        if len(imgs) == 0:
            logger.warning(f'No images found for class: {cls}')
        for img_path in imgs:
            all_paths.append(str(img_path))
            all_labels.append(class_to_idx[cls])

    logger.info(f'Found {len(all_paths)} images across {len(class_names)} classes')

    # ── Stratified 3-way split ───────────────────────────────────────────────
    # Step 1: carve out test set
    X_tv, X_test, y_tv, y_test = train_test_split(
        all_paths, all_labels,
        test_size=test_split,
        stratify=all_labels,
        random_state=seed
    )
    # Step 2: split remainder into train / val
    adjusted_val = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=adjusted_val,
        stratify=y_tv,
        random_state=seed
    )

    logger.info(
        f'Split sizes — Train: {len(X_train)}, '
        f'Val: {len(X_val)}, Test: {len(X_test)}'
    )

    # ── TF dataset helpers ───────────────────────────────────────────────────
    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        # FIX: EfficientNetV2S was pretrained with preprocess_input,
        # NOT simple /255 normalisation. Using the correct preprocessing
        # function is critical for transfer learning accuracy.
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        # NOTE: do NOT clip to [0,1] here — preprocess_input output range
        # is [-1, 1] (or model-specific). Clipping would corrupt the values.
        return img, label

    def make_ds(paths, labels, augment_flag=False, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if augment_flag:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            # Reduced buffer (256) balances RAM usage vs randomness
            ds = ds.shuffle(buffer_size=256, seed=seed)
        if not augment_flag:
            # Cache val and test sets after first read — they never change,
            # so there is no reason to re-read from disk every epoch.
            ds = ds.cache()
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(X_train, y_train, augment_flag=True,  shuffle=True)
    val_ds   = make_ds(X_val,   y_val,   augment_flag=False, shuffle=False)
    test_ds  = make_ds(X_test,  y_test,  augment_flag=False, shuffle=False)

    # Return y_train so main.py can compute class weights without
    # iterating over the entire dataset a second time.
    return train_ds, val_ds, test_ds, class_names, y_train
