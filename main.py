# main.py  —  Unified CLI entrypoint
#
# Usage:
#   python main.py --mode train
#   python main.py --mode evaluate
#   python main.py --mode export
#   python main.py --mode train --config configs/config.yaml

import argparse
import logging

from src.utils import setup_logging, load_config, save_metadata, load_metadata
from src.data_preprocessing import build_datasets, set_global_seeds
from src.model_training import build_model, train, compute_class_weights


logger = logging.getLogger(__name__)


def mode_train(cfg: dict):
    """Full training pipeline: data → model → train → save."""
    logger.info('──── MODE: TRAIN ────')

    # Build datasets
    # FIX: build_datasets now returns y_train as the 5th element so we
    # can compute class weights without iterating the dataset a second time.
    train_ds, val_ds, test_ds, class_names, y_train = build_datasets(
        data_dir   = cfg['data']['raw_dir'],
        img_size   = cfg['data']['img_size'],
        batch_size = cfg['data']['batch_size'],
        val_split  = cfg['data']['val_split'],
        test_split = cfg['data']['test_split'],
        seed       = cfg['data']['seed'],
    )

    # Save class metadata for inference
    save_metadata(
        class_names  = class_names,
        img_size     = cfg['data']['img_size'],
        architecture = cfg['model']['architecture'],
        path         = cfg['paths']['metadata_path'],
    )

    # Build model
    model = build_model(
        num_classes  = len(class_names),
        img_size     = cfg['data']['img_size'],
        dropout_rate = cfg['model']['dropout_rate'],
        dense_units  = cfg['model']['dense_units'],
    )
    model.summary()

    # Compute class weights directly from y_train (no second dataset pass)
    class_weight = compute_class_weights(y_train)

    train_cfg = dict(cfg['training'])
    train_cfg['class_weight'] = class_weight

    # Read resume epochs from config (default 0 = start fresh)
    initial_epoch_head     = train_cfg.pop('initial_epoch_head',     0)
    initial_epoch_finetune = train_cfg.pop('initial_epoch_finetune', 0)

    # If resuming mid-Phase-2, reload the best checkpoint first
    if initial_epoch_finetune > 0:
        import tensorflow as tf
        checkpoint_path = cfg['paths']['model_save_dir'] + '/best_checkpoint.keras'
        logger.info(f'Resuming Phase 2 — loading checkpoint: {checkpoint_path}')
        model = tf.keras.models.load_model(checkpoint_path)

    # Train
    h1, h2 = train(
        model                  = model,
        train_ds               = train_ds,
        val_ds                 = val_ds,
        config                 = train_cfg,
        save_dir               = cfg['paths']['model_save_dir'],
        initial_epoch_head     = initial_epoch_head,
        initial_epoch_finetune = initial_epoch_finetune,
    )

    # Plot training curves
    from src.model_evaluation import plot_training_history
    plot_training_history(h1, h2, output_dir=cfg['paths']['results_dir'])

    logger.info('Training complete.')


def mode_evaluate(cfg: dict):
    """Loads the saved model and runs full evaluation on the test set."""
    logger.info('──── MODE: EVALUATE ────')
    import tensorflow as tf
    from src.model_evaluation import evaluate_full

    model_path = cfg['paths']['model_save_dir'] + '/final_model'
    logger.info(f'Loading model from: {model_path}')
    model = tf.keras.models.load_model(model_path)

    meta = load_metadata(cfg['paths']['metadata_path'])

    # FIX: build_datasets now returns 5 values
    _, _, test_ds, _, _ = build_datasets(
        data_dir   = cfg['data']['raw_dir'],
        img_size   = meta['img_size'],
        batch_size = cfg['data']['batch_size'],
        val_split  = cfg['data']['val_split'],
        test_split = cfg['data']['test_split'],
        seed       = cfg['data']['seed'],
    )

    metrics = evaluate_full(
        model       = model,
        test_ds     = test_ds,
        class_names = meta['class_names'],
        output_dir  = cfg['paths']['results_dir'],
    )
    logger.info(f'Final test metrics: {metrics}')


def mode_export(cfg: dict):
    """Exports the SavedModel to TFLite FP16 for faster inference."""
    logger.info('──── MODE: EXPORT (TFLite FP16) ────')
    import tensorflow as tf
    from pathlib import Path

    model_path  = cfg['paths']['model_save_dir'] + '/final_model'
    output_path = cfg['inference']['tflite_model_path']

    logger.info(f'Converting: {model_path} → {output_path}')

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(tflite_model)

    size_mb = len(tflite_model) / 1_048_576
    logger.info(f'TFLite model saved: {size_mb:.1f} MB → {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plant Disease Detection — ML Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'export'],
        default='train',
        help='Pipeline mode to run (default: train)'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Path to config YAML (default: configs/config.yaml)'
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg['paths']['log_dir'])
    set_global_seeds(cfg['data']['seed'])

    logger.info(f'Config loaded from: {args.config}')
    logger.info(f'Running mode: {args.mode}')

    if args.mode == 'train':
        mode_train(cfg)
    elif args.mode == 'evaluate':
        mode_evaluate(cfg)
    elif args.mode == 'export':
        mode_export(cfg)


if __name__ == '__main__':
    main()
