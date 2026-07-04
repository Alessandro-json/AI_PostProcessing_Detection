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

from dataset import RRDatasetFromCSV, build_eval_transform
from model_vit_RGB import ViTRGBMultiTaskModel


FAKE_LABEL_NAMES      = ["real", "ai"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


def evaluate_model(model, dataloader, device, task):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            images  = batch["image"].to(device)
            outputs = model(images)

            if task in ["fake", "multitask"]:
                true_fake   = batch["fake_label"].to(device)
                pred_fake   = torch.argmax(outputs["fake_logits"], dim=1)

            if task in ["transform", "multitask"]:
                true_transform = batch["transform_label"].to(device)
                pred_transform = torch.argmax(outputs["transform_logits"], dim=1)

            for i in range(images.size(0)):
                row = {"image_path": batch["image_path"][i]}

                if task in ["fake", "multitask"]:
                    row["true_fake"] = int(true_fake[i].cpu())
                    row["pred_fake"] = int(pred_fake[i].cpu())

                if task in ["transform", "multitask"]:
                    row["true_transform"] = int(true_transform[i].cpu())
                    row["pred_transform"] = int(pred_transform[i].cpu())

                # Needed to compute fake accuracy by transformation even in fake-only mode.
                if task == "fake":
                    row["true_transform"] = int(batch["transform_label"][i])

                rows.append(row)

    return pd.DataFrame(rows)


def compute_metrics(predictions_df, task):
    metrics = {}

    if task in ["fake", "multitask"]:
        y_true = predictions_df["true_fake"]
        y_pred = predictions_df["pred_fake"]

        metrics["fake_accuracy"] = accuracy_score(y_true, y_pred)
        metrics["fake_f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

        metrics["fake_accuracy_by_transform"] = {}
        for tid, tname in enumerate(TRANSFORM_LABEL_NAMES):
            subset = predictions_df[predictions_df["true_transform"] == tid]
            if len(subset) == 0:
                metrics["fake_accuracy_by_transform"][tname] = None
            else:
                metrics["fake_accuracy_by_transform"][tname] = accuracy_score(
                    subset["true_fake"], subset["pred_fake"],
                )

    if task in ["transform", "multitask"]:
        y_true = predictions_df["true_transform"]
        y_pred = predictions_df["pred_transform"]

        metrics["transform_accuracy"] = accuracy_score(y_true, y_pred)
        metrics["transform_f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return metrics


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def save_results(predictions_df, metrics, output_dir, task):
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    if task in ["fake", "multitask"]:
        save_confusion_matrix(
            predictions_df["true_fake"], predictions_df["pred_fake"],
            FAKE_LABEL_NAMES, "Real/Fake Confusion Matrix (ViT-RGB)",
            output_dir / "confusion_fake.png",
        )

    if task in ["transform", "multitask"]:
        save_confusion_matrix(
            predictions_df["true_transform"], predictions_df["pred_transform"],
            TRANSFORM_LABEL_NAMES, "Transformation Confusion Matrix (ViT-RGB)",
            output_dir / "confusion_transform.png",
        )


def print_metrics(metrics, task):
    print("\nEvaluation results — ViT-RGB model")
    print("=" * 50)

    if task in ["fake", "multitask"]:
        print(f"Fake accuracy:        {metrics['fake_accuracy']:.4f}")
        print(f"Fake F1 macro:        {metrics['fake_f1_macro']:.4f}")
        print("\nFake accuracy by transformation:")
        for name, value in metrics["fake_accuracy_by_transform"].items():
            print(f"  {name}: {value:.4f}" if value is not None else f"  {name}: not available")

    if task in ["transform", "multitask"]:
        print(f"Transform accuracy:   {metrics['transform_accuracy']:.4f}")
        print(f"Transform F1 macro:   {metrics['transform_f1_macro']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT-Small RGB multi-task model.")

    parser.add_argument("--task",        type=str, default="multitask",
                        choices=["fake", "transform", "multitask"])
    parser.add_argument("--csv_path",    type=str, default="data/splits/test_balanced.csv")
    parser.add_argument("--image_root",  type=str, default="data/raw/RRDataset_subset")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_vit_rgb.pt")
    parser.add_argument("--output_dir",  type=str, default="results/vit_rgb")
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_pretrained",  action="store_true")
    parser.add_argument("--vit_model_name", type=str, default="vit_small_patch16_224",
                        help="Must match the value used during training.")

    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected task: {args.task}")
    print(f"Using device:  {device}")

    transform = build_eval_transform(image_size=args.image_size)
    dataset   = RRDatasetFromCSV(csv_path=args.csv_path, image_root=args.image_root,
                                 transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = ViTRGBMultiTaskModel(
        task=args.task,
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        vit_model_name=args.vit_model_name,
    )
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    predictions_df = evaluate_model(model, dataloader, device, args.task)
    metrics        = compute_metrics(predictions_df, args.task)

    output_dir = Path(args.output_dir)
    save_results(predictions_df, metrics, output_dir, args.task)
    print_metrics(metrics, args.task)
    print(f"\nSaved results in: {output_dir}")


if __name__ == "__main__":
    main()
