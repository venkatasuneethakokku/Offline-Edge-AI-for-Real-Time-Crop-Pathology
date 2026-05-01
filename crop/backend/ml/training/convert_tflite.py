import logging
from pathlib import Path
from typing import Iterable

import tensorflow as tf

from backend.ml.training.model_builder import build_model

LOGGER = logging.getLogger(__name__)


def _representative_data_gen(dataset_root: Path, image_size: tuple[int, int] = (224, 224)) -> Iterable[list[tf.Tensor]]:
    rep_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        labels=None,
        image_size=image_size,
        batch_size=1,
        shuffle=True,
    ).take(100)

    for batch in rep_ds:
        # Model includes preprocess_input, so representative data should be raw float32 images.
        yield [tf.cast(batch, tf.float32)]


def convert_to_tflite_int8(
    model_path: Path,
    dataset_root: Path,
    class_names: list[str],
    output_path: Path,
) -> Path:
    model_path = Path(model_path)
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found for representative dataset: {dataset_root}")

    model, _ = build_model(num_classes=len(class_names))
    model.load_weights(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_data_gen(dataset_root)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    LOGGER.info("INT8 TFLite model saved to %s", output_path)
    return output_path
