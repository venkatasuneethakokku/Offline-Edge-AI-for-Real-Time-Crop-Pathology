import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _copy_files(items: List[Tuple[Path, str]], split_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    indices: Dict[str, int] = {}
    for src, class_name in items:
        target = split_dir / class_name
        target.mkdir(parents=True, exist_ok=True)
        idx = indices.get(class_name, 0)
        dst = target / src.name
        if dst.exists():
            dst = target / f"{src.stem}_{idx}{src.suffix}"
        shutil.copy2(src, dst)
        counts[class_name] = counts.get(class_name, 0) + 1
        indices[class_name] = idx + 1
    return counts


def _collect_samples(dataset_root: Path) -> tuple[List[Path], List[str], List[str]]:
    files: List[Path] = []
    labels: List[str] = []
    classes: List[str] = []
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        class_name = class_dir.name
        classes.append(class_name)
        class_files = sorted([p for p in class_dir.rglob("*") if _is_image(p)])
        files.extend(class_files)
        labels.extend([class_name] * len(class_files))
    return files, labels, classes


def _has_sufficient_samples_for_stratify(labels: List[str]) -> bool:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return bool(counts) and all(count >= 2 for count in counts.values())


def _per_class_fallback_split(
    dataset_root: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> Dict[str, Dict[str, int]]:
    LOGGER.warning("Using fallback split because some classes have <2 samples for stratification")
    split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        class_name = class_dir.name
        files = sorted([p for p in class_dir.rglob("*") if _is_image(p)])
        if len(files) < 3:
            train_files, val_files, test_files = files, [], []
        else:
            train_files, temp_files = train_test_split(
                files, train_size=train_ratio, random_state=random_state, shuffle=True
            )
            relative_val = val_ratio / (val_ratio + test_ratio)
            val_files, test_files = train_test_split(
                temp_files, train_size=relative_val, random_state=random_state, shuffle=True
            )
        _copy_files([(p, class_name) for p in train_files], output_root / "train")
        _copy_files([(p, class_name) for p in val_files], output_root / "val")
        _copy_files([(p, class_name) for p in test_files], output_root / "test")
        split_counts["train"][class_name] = len(train_files)
        split_counts["val"][class_name] = len(val_files)
        split_counts["test"][class_name] = len(test_files)
    return split_counts


def split_dataset(
    dataset_root: Path,
    output_root: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Dict[str, int]]:
    if round(train_ratio + val_ratio + test_ratio, 2) != 1.00:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    for split_name in ["train", "val", "test"]:
        (output_root / split_name).mkdir(parents=True, exist_ok=True)

    files, labels, classes = _collect_samples(dataset_root)
    if not files:
        raise ValueError(f"No images found in dataset root: {dataset_root}")

    if not _has_sufficient_samples_for_stratify(labels):
        return _per_class_fallback_split(dataset_root, output_root, train_ratio, val_ratio, test_ratio, random_state)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, train_size=train_ratio, random_state=random_state, shuffle=True, stratify=labels
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        train_size=relative_val,
        random_state=random_state,
        shuffle=True,
        stratify=temp_labels,
    )

    train_counts = _copy_files(list(zip(train_files, train_labels)), output_root / "train")
    val_counts = _copy_files(list(zip(val_files, val_labels)), output_root / "val")
    test_counts = _copy_files(list(zip(test_files, test_labels)), output_root / "test")

    split_counts: Dict[str, Dict[str, int]] = {
        "train": {c: train_counts.get(c, 0) for c in classes},
        "val": {c: val_counts.get(c, 0) for c in classes},
        "test": {c: test_counts.get(c, 0) for c in classes},
    }
    LOGGER.info("Stratified split complete. Output directory: %s", output_root)
    return split_counts
