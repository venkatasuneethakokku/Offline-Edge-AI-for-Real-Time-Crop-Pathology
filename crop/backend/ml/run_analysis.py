import argparse
import json
import logging
from pathlib import Path

from backend.ml.dataset_analysis.analyzer import analyze_dataset
from backend.ml.dataset_analysis.splitter import split_dataset
from backend.ml.dataset_analysis.stats_generator import generate_dataset_report
from backend.ml.dataset_analysis.validator import validate_and_clean_images
from backend.ml.dataset_analysis.visualize import plot_class_distribution


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset analysis and preparation pipeline.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to raw dataset root.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("dataset_processed"),
        help="Path to output processed dataset directory.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("backend/ml/reports"),
        help="Directory to save analysis reports.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    logger = logging.getLogger("run_analysis")

    try:
        analysis = analyze_dataset(args.dataset_root)
        validation = validate_and_clean_images(args.dataset_root)
        split_info = split_dataset(args.dataset_root, args.processed_root)
        report = generate_dataset_report(args.dataset_root, args.reports_dir / "dataset_report.json")
        chart_path = plot_class_distribution(report["class_distribution"], args.reports_dir / "class_distribution.png")

        logger.info("Analysis complete")
        logger.info("Total images: %s", analysis["total_images"])
        logger.info("Corrupted removed: %s", validation["corrupted_count"])
        logger.info("Processed split root: %s", args.processed_root)
        logger.info("Class chart: %s", chart_path)

        summary_path = args.reports_dir / "analysis_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "analysis": analysis,
                    "validation": validation,
                    "split_info": split_info,
                    "dataset_report_path": str(args.reports_dir / "dataset_report.json"),
                    "class_distribution_chart": str(chart_path),
                },
                f,
                indent=2,
            )
        logger.info("Summary saved to %s", summary_path)

    except Exception as exc:
        logger.exception("Dataset analysis pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
