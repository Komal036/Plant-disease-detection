# tests/test_model.py
import numpy as np
import pytest
import tensorflow as tf

from src.model_training import build_model, compute_class_weights


# ── build_model tests ─────────────────────────────────────────────────────────

def test_output_shape_matches_num_classes():
    num_classes = 15
    model = build_model(num_classes, img_size=224)
    dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (2, num_classes), (
        f'Expected shape (2, {num_classes}), got {preds.shape}'
    )


def test_output_is_valid_probability_distribution():
    """Softmax output must sum to 1 and be in [0, 1]."""
    model = build_model(5, img_size=64)
    dummy = np.random.rand(3, 64, 64, 3).astype(np.float32)
    preds = model.predict(dummy, verbose=0)

    for i, row in enumerate(preds):
        assert abs(row.sum() - 1.0) < 1e-5, \
            f'Row {i} does not sum to 1: {row.sum()}'
        assert row.min() >= 0.0, f'Row {i} has negative probability'
        assert row.max() <= 1.0, f'Row {i} has probability > 1'


def test_model_accepts_batch_of_one():
    model = build_model(10, img_size=128)
    single = np.random.rand(1, 128, 128, 3).astype(np.float32)
    preds = model.predict(single, verbose=0)
    assert preds.shape == (1, 10)


def test_base_is_frozen_initially():
    """Phase 1: the EfficientNet base should be non-trainable by default."""
    model = build_model(5, img_size=64)
    base_layer = model.layers[1]   # EfficientNetV2S
    assert not base_layer.trainable, (
        'Base model should be frozen (trainable=False) after build_model()'
    )


def test_model_is_keras_model():
    model = build_model(3, img_size=64)
    assert isinstance(model, tf.keras.Model)


# ── compute_class_weights tests ───────────────────────────────────────────────

def test_class_weights_keys_match_classes():
    labels = [0, 0, 0, 1, 1, 2]
    weights = compute_class_weights(labels)
    assert set(weights.keys()) == {0, 1, 2}


def test_minority_class_gets_higher_weight():
    """Class 2 appears only once — it should have the highest weight."""
    labels = [0] * 10 + [1] * 10 + [2] * 2
    weights = compute_class_weights(labels)
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]
