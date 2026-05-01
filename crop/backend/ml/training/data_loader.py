import logging
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def get_dataset_size(dataset_root: Path) -> int:
    dataset_root = Path(dataset_root)
    return sum(1 for p in dataset_root.rglob("*") if _is_image(p))


def create_datasets(
    dataset_root: Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 16,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, tf.data.Dataset | list[str]]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    LOGGER.info("Loaded dataset from %s with %s classes", dataset_root, len(class_names))
    return {"train": train_ds, "val": val_ds, "class_names": class_names}
