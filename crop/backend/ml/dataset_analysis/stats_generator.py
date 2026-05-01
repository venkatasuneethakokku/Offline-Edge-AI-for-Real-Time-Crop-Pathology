import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Dict

from PIL import Image

from backend.ml.dataset_analysis.analyzer import analyze_dataset

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def generate_dataset_report(dataset_root: Path, report_path: Path) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    report_path = Path(report_path)
    summary = analyze_dataset(dataset_root)

    widths = []
    heights = []
    total_size_bytes = 0

    for image_path in dataset_root.rglob("*"):
        if not _is_image(image_path):
            continue
        total_size_bytes += image_path.stat().st_size
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except OSError as exc:
            LOGGER.warning("Skipping unreadable image while generating stats: %s | %s", image_path, exc)

    class_counts = summary["class_counts"]
    non_zero_counts = [c for c in class_counts.values() if c > 0]
    imbalance_ratio = (max(non_zero_counts) / min(non_zero_counts)) if non_zero_counts else 0.0

    report: Dict[str, Any] = {
        "dataset_root": summary["dataset_root"],
        "total_images": summary["total_images"],
        "total_classes": summary["total_classes"],
        "class_distribution": class_counts,
        "empty_folders": summary["empty_folders"],
        "imbalance_ratio": round(imbalance_ratio, 4),
        "average_image_resolution": {
            "width": round(mean(widths), 2) if widths else 0,
            "height": round(mean(heights), 2) if heights else 0,
        },
        "dataset_size_mb": round(total_size_bytes / (1024 * 1024), 2),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    LOGGER.info("Dataset report saved to %s", report_path)
    return report
