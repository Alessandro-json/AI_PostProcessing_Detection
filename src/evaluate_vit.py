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

from dataset_vit import RRViTDatasetFromCSV
from model_vit import ViTMultiTaskModel


FAKE_LABEL_NAMES      = ["real", "ai"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]


def load_checkpoint(model, checkpoint_path, device):
    """
    Load trained weights into the model.
    Supports all checkpoint formats used in this project.
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
    Run the ViT model on the test set and collect predictions.
    Identical logic to evaluate_freq.py, minus the freq_map input.
    """

    model.eval()
    rows = []

    with torch.no_grad():

        for batch in dataloader:

            images         = batch["image"].to(device)
            true_fake      = batch["fake_label"].to(device)
            true_transform = batch["transform_label"].to(device)

            outputs = model(images)

            fake_logits      = outputs["fake_logits"]
            transform_logits = outputs["transform_logits"]

            pred_fake      = torch.argmax(fake_logits,      dim=1)
            pred_transform = torch.argmax(transform_logits, dim=1)

            batch_size = images.size(0)

            for i in range(batch_size):
                rows.append({
                    "image_path":     batch["image_path"][i],
                    "true_fake":      int(true_fake[i].cpu()),
                    "pred_fake":      int(pred_fake[i].cpu()),
                    "true_transform": int(true_transform[i].cpu()),
                    "pred_transform": int(pred_transform[i].cpu()),
                })

    return pd.DataFrame(rows)


def compute_metrics(predictions_df):
    """
    Compute evaluation metrics from the predictions DataFrame.
    Identical logic to evaluate_freq.py / evaluate_RGB.py so all
    three result tables are directly comparable.
    """

    y_true_fake      = predictions_df["true_fake"]
    y_pred_fake      = predictions_df["pred_fake"]
    y_true_transform = predictions_df["true_transform"]
    y_pred_transform = predictions_df["pred_transform"]

    metrics = {
        "fake_accuracy": accuracy_score(y_true_fake, y_pred_fake),
        "fake_f1_macro": f1_score(y_true_fake, y_pred_fake, average="macro"),

        "transform_accuracy": accuracy_score(y_true_transform, y_pred_transform),
        "transform_f1_macro": f1_score(
            y_true_transform,
            y_pred_transform,
            average="macro",
        ),

        "fake_accuracy_by_transform": {},
    }

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
    Save all evaluation outputs to disk, same format as evaluate_freq.py
    so show_evaluation_outputs() in the notebook works unchanged.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    save_confusion_matrix(
        y_true=predictions_df["true_fake"],
        y_pred=predictions_df["pred_fake"],
        labels=FAKE_LABEL_NAMES,
        title="Real/Fake Confusion Matrix (ViT)",
        output_path=output_dir / "confusion_fake.png",
    )

    save_confusion_matrix(
        y_true=predictions_df["true_transform"],
        y_pred=predictions_df["pred_transform"],
        labels=TRANSFORM_LABEL_NAMES,
        title="Transformation Confusion Matrix (ViT)",
        output_path=output_dir / "confusion_transform.png",
    )


def print_metrics(metrics):
    """
    Print a summary of the main metrics in the terminal.
    """

    print("\nEvaluation results — ViT model")
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
    Command-line arguments mirror evaluate_freq.py / evaluate_RGB.py.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate the ViT-Small multi-task model (RGB only)."
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
        default="checkpoints/best_vit_multitask_1_1.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/vit_multitask_1_1",
        help="Folder where evaluation results are saved.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Set to 0 on Colab to avoid DataLoader issues.",
    )
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="vit_small_patch16_224",
        help="Must match the model used during training.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use if the checkpoint was trained without ImageNet weights.",
    )

    return parser.parse_args()


def main():
    """
    Main evaluation pipeline for the ViT model.
    Steps mirror evaluate_freq.py / evaluate_RGB.py exactly.
    """

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RRViTDatasetFromCSV(
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

    model = ViTMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        vit_model_name=args.vit_model_name,
    )

    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    model = model.to(device)

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
