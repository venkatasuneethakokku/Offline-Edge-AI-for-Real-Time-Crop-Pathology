import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from backend.ml.training.model_builder import build_model

LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model_path: Path,
    eval_ds: tf.data.Dataset,
    class_names: List[str],
    reports_dir: Path,
) -> Dict[str, str]:
    model_path = Path(model_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model, _ = build_model(num_classes=len(class_names))
    model.load_weights(model_path)
    y_true = np.concatenate([labels.numpy() for _, labels in eval_ds], axis=0)
    y_pred_probs = model.predict(eval_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    cm_img = reports_dir / "confusion_matrix.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_img, dpi=200)
    plt.close()

    report_json = reports_dir / "classification_report.json"
    report_txt = reports_dir / "classification_report.txt"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    with report_txt.open("w", encoding="utf-8") as f:
        f.write(report_text)

    LOGGER.info("Evaluation complete. Reports saved to %s", reports_dir)
    return {
        "confusion_matrix_path": str(cm_img),
        "classification_report_json": str(report_json),
        "classification_report_txt": str(report_txt),
    }
