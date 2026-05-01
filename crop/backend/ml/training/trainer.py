import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import tensorflow as tf

from backend.ml.training.model_builder import build_model

LOGGER = logging.getLogger(__name__)


def _plot_history(history: Dict[str, List[float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.get("loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.get("accuracy", []), label="train_acc")
    plt.plot(history.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _build_callbacks(models_dir: Path) -> List[tf.keras.callbacks.Callback]:
    models_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_dir / "best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_names: List[str],
    models_dir: Path,
    reports_dir: Path,
    initial_epochs: int = 15,
    fine_tune_epochs: int = 10,
) -> Dict[str, Any]:
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model, base_model = build_model(num_classes=len(class_names))
    callbacks = _build_callbacks(models_dir)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    LOGGER.info("Starting initial training for %s epochs", initial_epochs)
    initial_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - 20)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    total_epochs = initial_epochs + fine_tune_epochs
    LOGGER.info("Starting fine-tuning for %s epochs", fine_tune_epochs)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=initial_history.epoch[-1] + 1,
        callbacks=callbacks,
        verbose=1,
    )

    final_history = {}
    for key in set(initial_history.history) | set(fine_tune_history.history):
        final_history[key] = initial_history.history.get(key, []) + fine_tune_history.history.get(key, [])

    final_model_path = models_dir / "final_model.h5"
    model.save(final_model_path)
    with (models_dir / "class_names.json").open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    _plot_history(final_history, reports_dir / "training_history.png")
    with (reports_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(final_history, f, indent=2)

    LOGGER.info("Training complete. Model artifacts saved in %s", models_dir)
    final_val_acc = final_history.get("val_accuracy", [0.0])[-1] if final_history.get("val_accuracy") else 0.0
    return {
        "final_model_path": str(final_model_path),
        "best_model_path": str(models_dir / "best_model.h5"),
        "class_names_path": str(models_dir / "class_names.json"),
        "training_history_plot": str(reports_dir / "training_history.png"),
        "final_val_accuracy": float(final_val_acc),
    }
