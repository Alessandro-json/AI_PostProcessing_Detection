import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from dataset_freq import RRFreqDatasetFromCSV
from model_freq import FreqMultiTaskModel


FAKE_LABEL_NAMES      = ["real", "ai"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]


def load_checkpoint(model, checkpoint_path, device):
    """
    Load trained weights into the model.

    Supports all three checkpoint formats used in this project:
        1. {"model_state_dict": ...}   <- saved by train_freq.py
        2. {"state_dict": ...}
        3. The state_dict directly
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def evaluate_model(model, dataloader, device):
    """
    Run the frequency model on the test set and collect predictions.

    Differences from evaluate_RGB.py:
        - The DataLoader also returns "freq_map" tensors.
        - model() receives both images and freq_map.

    Returns a DataFrame with one row per test image.
    """

    model.eval()
    rows = []

    with torch.no_grad():

        for batch in dataloader:

            # Move both modalities to the correct device.
            images           = batch["image"].to(device)
            freq_map         = batch["freq_map"].to(device)
            true_fake        = batch["fake_label"].to(device)
            true_transform   = batch["transform_label"].to(device)

            # Forward pass: RGB + FFT spectrum -> two task outputs.
            outputs = model(images=images, freq_map=freq_map)

            fake_logits      = outputs["fake_logits"]
            transform_logits = outputs["transform_logits"]

            pred_fake      = torch.argmax(fake_logits,      dim=1)
            pred_transform = torch.argmax(transform_logits, dim=1)

            batch_size = images.size(0)

            for i in range(batch_size):
                rows.append({
                    "image_path":    batch["image_path"][i],
                    "true_fake":     int(true_fake[i].cpu()),
                    "pred_fake":     int(pred_fake[i].cpu()),
                    "true_transform": int(true_transform[i].cpu()),
                    "pred_transform": int(pred_transform[i].cpu()),
                })

    return pd.DataFrame(rows)


def compute_metrics(predictions_df):
    """
    Compute evaluation metrics from the predictions DataFrame.

    Identical logic to evaluate_RGB.py so results are directly comparable:
        - global fake accuracy + macro F1
        - global transform accuracy + macro F1
        - fake accuracy broken down by transformation type
    """

    y_true_fake      = predictions_df["true_fake"]
    y_pred_fake      = predictions_df["pred_fake"]
    y_true_transform = predictions_df["true_transform"]
    y_pred_transform = predictions_df["pred_transform"]

    metrics = {
        "fake_accuracy":   accuracy_score(y_true_fake, y_pred_fake),
        "fake_f1_macro":   f1_score(y_true_fake, y_pred_fake, average="macro"),

        "transform_accuracy": accuracy_score(y_true_transform, y_pred_transform),
        "transform_f1_macro": f1_score(
            y_true_transform,
            y_pred_transform,
            average="macro",
        ),

        "fake_accuracy_by_transform": {},
    }

    # Per-transformation real/fake accuracy.
    # Lets us see whether FFT helps more on original, transfer, or redigital.
    for transform_id, transform_name in enumerate(TRANSFORM_LABEL_NAMES):

        subset = predictions_df[predictions_df["true_transform"] == transform_id]

        if len(subset) == 0:
            metrics["fake_accuracy_by_transform"][transform_name] = None
        else:
            metrics["fake_accuracy_by_transform"][transform_name] = accuracy_score(
                subset["true_fake"],
                subset["pred_fake"],
            )

    return metrics


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """
    Create and save a confusion matrix plot.
    Identical to evaluate_RGB.py.
    """

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_results(predictions_df, metrics, output_dir):
    """
    Save all evaluation outputs to disk:
        - predictions.csv
        - metrics.json
        - confusion_fake.png
        - confusion_transform.png

    The output format is identical to evaluate_RGB.py so that
    show_evaluation_outputs() in the notebook works without any change.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    save_confusion_matrix(
        y_true=predictions_df["true_fake"],
        y_pred=predictions_df["pred_fake"],
        labels=FAKE_LABEL_NAMES,
        title="Real/Fake Confusion Matrix (Freq)",
        output_path=output_dir / "confusion_fake.png",
    )

    save_confusion_matrix(
        y_true=predictions_df["true_transform"],
        y_pred=predictions_df["pred_transform"],
        labels=TRANSFORM_LABEL_NAMES,
        title="Transformation Confusion Matrix (Freq)",
        output_path=output_dir / "confusion_transform.png",
    )


def print_metrics(metrics):
    """
    Print a summary of the main metrics in the terminal.
    """

    print("\nEvaluation results — Frequency model")
    print("=" * 50)
    print(f"Fake accuracy:        {metrics['fake_accuracy']:.4f}")
    print(f"Fake F1 macro:        {metrics['fake_f1_macro']:.4f}")
    print(f"Transform accuracy:   {metrics['transform_accuracy']:.4f}")
    print(f"Transform F1 macro:   {metrics['transform_f1_macro']:.4f}")

    print("\nFake accuracy by transformation:")
    for name, value in metrics["fake_accuracy_by_transform"].items():
        if value is None:
            print(f"  {name}: not available")
        else:
            print(f"  {name}: {value:.4f}")


def parse_args():
    """
    Command-line arguments mirror evaluate_RGB.py exactly,
    so the notebook can call this script with the same variable names.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate the RGB + FFT frequency multi-task model."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/splits/test_balanced.csv",
        help="Path to test CSV file.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="data/raw/RRDataset_subset",
        help="Root folder containing subset images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_freq_multitask_1_1.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/freq_multitask_1_1",
        help="Folder where evaluation results are saved.",
    )
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--image_size",   type=int, default=224)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Set to 0 on Colab to avoid DataLoader issues.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use if the checkpoint was trained without ImageNet weights.",
    )
    parser.add_argument(
        "--freq_features",
        type=int,
        default=128,
        help="Must match the value used during training.",
    )
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Use if the model was trained without sigmoid gating.",
    )

    return parser.parse_args()


def main():
    """
    Main evaluation pipeline for the frequency model.

    Steps mirror evaluate_RGB.py exactly:
        1. Parse arguments.
        2. Select device.
        3. Build dataset and dataloader.
        4. Build model and load checkpoint.
        5. Run inference.
        6. Compute metrics.
        7. Save and print results.
    """

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test dataset: no augmentation, deterministic order.
    dataset = RRFreqDatasetFromCSV(
        csv_path=args.csv_path,
        image_root=args.image_root,
        image_size=args.image_size,
        train=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Build model with the same options used during training.
    model = FreqMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        freq_features=args.freq_features,
        use_attention=not args.no_attention,
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model = model.to(device)

    # Run inference on the full test set.
    predictions_df = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    metrics = compute_metrics(predictions_df)

    output_dir = Path(args.output_dir)
    save_results(predictions_df, metrics, output_dir)
    print_metrics(metrics)

    print(f"\nSaved results in: {output_dir}")


if __name__ == "__main__":
    main()
