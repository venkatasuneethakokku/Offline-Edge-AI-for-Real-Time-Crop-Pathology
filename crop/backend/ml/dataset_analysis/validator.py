import logging
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def validate_and_clean_images(dataset_root: Path, quarantine_dir: Path | None = None) -> Dict[str, List[str] | int]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    quarantine_dir = (
        Path(quarantine_dir) if quarantine_dir else dataset_root.parent / f"{dataset_root.name}_corrupted_backup"
    )
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    log_path = quarantine_dir / "corrupted_files.log"

    corrupted_files: List[str] = []
    scanned = 0

    for image_path in dataset_root.rglob("*"):
        if not _is_image(image_path):
            continue
        scanned += 1
        try:
            with Image.open(image_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            LOGGER.warning("Corrupted image detected: %s | %s", image_path, exc)
            corrupted_files.append(str(image_path))
            relative = image_path.relative_to(dataset_root)
            destination = quarantine_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(destination))

    if corrupted_files:
        with log_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(corrupted_files))
        LOGGER.info("Moved %s corrupted images to %s", len(corrupted_files), quarantine_dir)
    else:
        LOGGER.info("No corrupted images found in %s", dataset_root)

    return {
        "total_scanned": scanned,
        "corrupted_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "log_path": str(log_path),
        "quarantine_dir": str(quarantine_dir),
    }
