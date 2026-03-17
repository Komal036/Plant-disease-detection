# src/model_evaluation.py
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)


def evaluate_full(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list,
    output_dir: str = 'results/'
) -> dict:
    """
    Runs full evaluation on the hold-out test set and saves:
        - classification_report.txt
        - confusion_matrix.png  (normalised)
        - f1_scores.png         (per-class bar chart)
        - per_class_accuracy.png

    Returns a dict with accuracy, macro_f1, weighted_f1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect predictions ───────────────────────────────────────────────────
    y_true, y_pred_probs = [], []
    for batch_imgs, batch_labels in test_ds:
        probs = model.predict(batch_imgs, verbose=0)
        y_pred_probs.extend(probs)
        y_true.extend(batch_labels.numpy())

    y_true       = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    # ── 1. Classification report ─────────────────────────────────────────────
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )
    logger.info(f'\nClassification Report:\n{report}')
    (output_dir / 'classification_report.txt').write_text(report)

    # ── 2. Normalised confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True, fmt='.2f',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title('Normalised Confusion Matrix', fontsize=14, pad=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    logger.info('Confusion matrix saved.')

    # ── 3. Per-class F1 bar chart ─────────────────────────────────────────────
    f1_scores = f1_score(y_true, y_pred, average=None)
    fig, ax = plt.subplots(figsize=(14, 5))
    bar_colors = ['#4CAF50' if s >= 0.9 else '#FF9800' if s >= 0.75 else '#F44336'
                  for s in f1_scores]
    ax.bar(class_names, f1_scores, color=bar_colors)
    ax.axhline(0.90, color='red',    linestyle='--', linewidth=1.2, label='0.90 target')
    ax.axhline(0.75, color='orange', linestyle='--', linewidth=1.0, label='0.75 baseline')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class F1 Score on Test Set', fontsize=13)
    ax.set_ylabel('F1 Score')
    ax.legend()
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / 'f1_scores.png', dpi=150)
    plt.close(fig)
    logger.info('F1 score chart saved.')

    # ── 4. Per-class accuracy bar chart ──────────────────────────────────────
    per_class_acc = []
    for i in range(len(class_names)):
        mask = (y_true == i)
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(float(np.mean(y_pred[mask] == i)))

    fig, ax = plt.subplots(figsize=(14, 5))
    acc_colors = ['#4CAF50' if a >= 0.9 else '#FF9800' if a >= 0.75 else '#F44336'
                  for a in per_class_acc]
    ax.bar(class_names, per_class_acc, color=acc_colors)
    ax.axhline(0.90, color='red',    linestyle='--', linewidth=1.2, label='0.90 target')
    ax.axhline(0.75, color='orange', linestyle='--', linewidth=1.0, label='0.75 baseline')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class Accuracy on Test Set', fontsize=13)
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / 'per_class_accuracy.png', dpi=150)
    plt.close(fig)
    logger.info('Per-class accuracy chart saved.')

    # ── 5. Summary metrics ────────────────────────────────────────────────────
    metrics = {
        'accuracy':    float(np.mean(y_true == y_pred)),
        'macro_f1':    float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
    }
    logger.info(f'Test Metrics: {metrics}')
    return metrics


def plot_training_history(history1, history2=None, output_dir: str = 'results/'):
    """
    Plots accuracy and loss curves.
    Pass history2 when two-phase training was used.
    Handles the case where history1 is None (Phase 1 was skipped on resume).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def combine(key):
        vals = history1.history[key] if history1 else []
        if history2:
            vals = vals + history2.history[key]
        return vals

    acc     = combine('accuracy')
    val_acc = combine('val_accuracy')
    loss    = combine('loss')
    val_loss= combine('val_loss')
    epochs  = range(1, len(acc) + 1)

    if len(acc) == 0:
        logger.warning('No training history to plot.')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, acc,     'g-',  label='Train Accuracy')
    ax1.plot(epochs, val_acc, 'b--', label='Val Accuracy')
    if history1 and history2:
        split = len(history1.history['accuracy'])
        ax1.axvline(split, color='gray', linestyle=':', label='Fine-tune start')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, loss,     'r-',  label='Train Loss')
    ax2.plot(epochs, val_loss, 'b--', label='Val Loss')
    if history1 and history2:
        ax2.axvline(split, color='gray', linestyle=':', label='Fine-tune start')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'training_history.png', dpi=150)
    plt.close(fig)
    logger.info('Training history plot saved.')
