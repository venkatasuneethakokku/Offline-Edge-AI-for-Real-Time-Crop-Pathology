import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def plot_class_distribution(class_counts: Dict[str, int], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(class_counts.keys())
    values = list(class_counts.values())

    plt.figure(figsize=(14, 7))
    bars = plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.title("Class Distribution")
    plt.tight_layout()

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)

    plt.savefig(output_path, dpi=200)
    plt.close()
    LOGGER.info("Class distribution chart saved to %s", output_path)
    return output_path
