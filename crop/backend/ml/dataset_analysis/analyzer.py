import logging
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def analyze_dataset(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    class_counts: Dict[str, int] = {}
    empty_folders: List[str] = []
    total_images = 0

    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        count = sum(1 for p in class_dir.rglob("*") if _is_image(p))
        class_counts[class_dir.name] = count
        total_images += count
        if count == 0:
            empty_folders.append(class_dir.name)

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "total_classes": len(class_counts),
        "total_images": total_images,
        "class_counts": class_counts,
        "empty_folders": empty_folders,
    }
    LOGGER.info("Dataset analysis complete: %s images across %s classes", total_images, len(class_counts))
    return summary
