import argparse
import logging
from pathlib import Path

import tensorflow as tf

from backend.ml.training.convert_tflite import convert_to_tflite_int8
from backend.ml.training.data_loader import create_datasets, get_dataset_size
from backend.ml.training.evaluator import evaluate_model
from backend.ml.training.trainer import train_model


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def enable_gpu_memory_growth() -> None:
    logger = logging.getLogger("run_training.gpu")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected. Training will run on CPU.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("Enabled memory growth for %s GPU(s)", len(gpus))
    except RuntimeError as exc:
        logger.warning("Could not set GPU memory growth: %s", exc)


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training, evaluation, and TFLite conversion pipeline.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to class-folder dataset root (e.g., Crop___Disease).",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("backend/ml/models"), help="Directory for model artifacts.")
    parser.add_argument("--reports-dir", type=Path, default=Path("backend/ml/reports"), help="Directory for reports.")
    parser.add_argument("--initial-epochs", type=int, default=15, help="Initial frozen-backbone training epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Fine-tuning epochs.")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    enable_gpu_memory_growth()
    args = parse_args()
    logger = logging.getLogger("run_training")

    try:
        dataset_size = get_dataset_size(args.dataset_root)
        logger.info("Dataset size (images): %s", dataset_size)

        datasets = create_datasets(
            dataset_root=args.dataset_root,
            image_size=(224, 224),
            batch_size=16,
            validation_split=0.2,
            seed=42,
        )
        class_names = datasets["class_names"]
        logger.info("Class names (%s): %s", len(class_names), class_names)

        training_outputs = train_model(
            train_ds=datasets["train"],
            val_ds=datasets["val"],
            class_names=class_names,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            initial_epochs=args.initial_epochs,
            fine_tune_epochs=args.fine_tune_epochs,
        )

        eval_outputs = evaluate_model(
            model_path=Path(training_outputs["best_model_path"]),
            eval_ds=datasets["val"],
            class_names=class_names,
            reports_dir=args.reports_dir,
        )

        tflite_path = convert_to_tflite_int8(
            model_path=Path(training_outputs["best_model_path"]),
            dataset_root=args.dataset_root,
            class_names=class_names,
            output_path=args.models_dir / "crop_model.tflite",
        )

        best_model_size = file_size_mb(Path(training_outputs["best_model_path"]))
        final_model_size = file_size_mb(Path(training_outputs["final_model_path"]))
        tflite_size = file_size_mb(Path(tflite_path))

        logger.info("Training pipeline complete")
        logger.info("Saved final model: %s", training_outputs["final_model_path"])
        logger.info("Saved best model: %s", training_outputs["best_model_path"])
        logger.info("Saved class names: %s", training_outputs["class_names_path"])
        logger.info("Saved training plot: %s", training_outputs["training_history_plot"])
        logger.info("Saved TFLite model: %s", tflite_path)
        logger.info("Saved evaluation artifacts: %s", eval_outputs)
        logger.info("Final validation accuracy: %.4f", training_outputs["final_val_accuracy"])
        logger.info(
            "Model sizes (MB) | best_model.h5=%s | final_model.h5=%s | crop_model.tflite=%s",
            best_model_size,
            final_model_size,
            tflite_size,
        )

    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
