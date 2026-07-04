"""
results_freq.py

Utility script that loads evaluation results produced by evaluate_freq.py
and displays them in the notebook using the same show_evaluation_outputs()
style used for the RGB and depth+frequency models.

Usage in notebook:
    from results_freq import show_freq_results
    show_freq_results("results/freq_1_1")
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Image


def show_freq_results(output_dir: str):
    """
    Display evaluation metrics and confusion matrices for a frequency model run.

    Args:
        output_dir: Path to the folder produced by evaluate_freq.py.
                    Must contain metrics.json, confusion_fake.png,
                    confusion_transform.png.

    This function is a drop-in equivalent of show_evaluation_outputs()
    already defined in the project notebook.
    """

    output_dir  = Path(output_dir)
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Build a flat table from the metrics dictionary.
    rows = []
    for metric_name, value in metrics.items():

        # Skip nested classification reports to keep the table readable.
        if metric_name in ("fake_classification_report", "transform_classification_report"):
            continue

        if isinstance(value, dict):
            for sub_name, sub_value in value.items():
                rows.append({
                    "Metric": metric_name,
                    "Group":  sub_name,
                    "Value":  sub_value,
                })
        else:
            rows.append({
                "Metric": metric_name,
                "Group":  "-",
                "Value":  value,
            })

    metrics_df = pd.DataFrame(rows)

    if not metrics_df.empty:
        metrics_df["Value"] = metrics_df["Value"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )

    print(f"Evaluation metrics — {output_dir.name}")
    display(metrics_df)

    fake_cm_path      = output_dir / "confusion_fake.png"
    transform_cm_path = output_dir / "confusion_transform.png"

    if fake_cm_path.exists():
        print("\nReal/Fake confusion matrix")
        display(Image(filename=str(fake_cm_path)))

    if transform_cm_path.exists():
        print("\nTransformation confusion matrix")
        display(Image(filename=str(transform_cm_path)))

    if not fake_cm_path.exists() and not transform_cm_path.exists():
        print("\nNo confusion matrix images found in this folder.")
