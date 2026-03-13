# src/model_training.py
import logging
import tensorflow as tf
from pathlib import Path

logger = logging.getLogger(__name__)


def build_model(
    num_classes: int,
    img_size: int = 224,
    dropout_rate: float = 0.3,
    dense_units: int = 256,
) -> tf.keras.Model:
    """
    Builds an EfficientNetV2S model with a custom classification head.

    Architecture:
        EfficientNetV2S (frozen, ImageNet weights)
        → GlobalAveragePooling2D
        → BatchNormalization
        → Dropout(dropout_rate)
        → Dense(dense_units, relu)
        → Dropout(dropout_rate / 2)
        → Dense(num_classes, softmax)

    The base is initially frozen for Phase-1 training.
    Call unfreeze_top_layers() before Phase-2 fine-tuning.
    """
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = False   # Phase 1: train head only

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))

    # Pass training=False so BatchNorm layers inside the base stay frozen
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name='plant_disease_classifier')
    logger.info(f'Model built: {model.count_params():,} total parameters')
    return model


def get_callbacks(save_dir: str) -> list:
    """Standard callbacks used in both training phases."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(save_dir) / 'best_checkpoint.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/',
            histogram_freq=1
        ),
    ]


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config: dict,
    save_dir: str
):
    """
    Two-phase training:
        Phase 1 — Train only the classification head (base frozen).
        Phase 2 — Unfreeze the top N layers of the base for fine-tuning
                  with a much lower learning rate.

    Args:
        model:     Model returned by build_model()
        train_ds:  Training tf.data.Dataset
        val_ds:    Validation tf.data.Dataset
        config:    Dict with keys:
                       epochs_head, epochs_finetune,
                       lr_head, lr_finetune,
                       unfreeze_layers, label_smoothing
                       class_weight (optional)
        save_dir:  Directory to save the final model

    Returns:
        history1, history2  (Keras History objects)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    callbacks = get_callbacks(str(save_dir))

    # ── Phase 1: Train classification head ───────────────────────────────────
    logger.info('=== Phase 1: Training classification head ===')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr_head']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        ),
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs_head'],
        callbacks=callbacks,
        class_weight=config.get('class_weight'),
        verbose=1
    )

    # ── Phase 2: Fine-tune top layers of the base ─────────────────────────────
    logger.info('=== Phase 2: Fine-tuning top layers ===')

    base_model = model.layers[1]          # EfficientNetV2S is layer index 1
    base_model.trainable = True

    # Freeze all layers EXCEPT the last `unfreeze_layers` layers
    n_unfreeze = config.get('unfreeze_layers', 40)
    for layer in base_model.layers[:-n_unfreeze]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    logger.info(
        f'Fine-tuning {trainable_count} / {len(base_model.layers)} '
        f'base layers'
    )

    # Use a much smaller LR to avoid destroying pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr_finetune']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs_finetune'],
        callbacks=callbacks,
        class_weight=config.get('class_weight'),
        verbose=1
    )

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = str(save_dir / 'final_model')
    model.save(final_path)
    logger.info(f'Final model saved to: {final_path}')

    return history1, history2


def compute_class_weights(labels: list) -> dict:
    """
    Computes balanced class weights to handle dataset imbalance.
    Pass the returned dict to model.fit(class_weight=...).
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    logger.info(f'Class weights computed: {weight_dict}')
    return weight_dict
