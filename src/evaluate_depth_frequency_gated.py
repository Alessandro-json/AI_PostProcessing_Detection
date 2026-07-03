import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_depth_frequency import RRDepthFrequencyDatasetFromCSV
from model_depth_frequency_gated import RGBDepthFrequencyMultiTaskModel


FAKE_LABEL_NAMES = ["real", "fake"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB+Depth+Frequency multi-task model."
    )

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")

    return parser.parse_args()


def load_model(checkpoint_path, device, use_attention=True, pretrained=True):
    model = RGBDepthFrequencyMultiTaskModel(
        num_transform_classes=3,
        pretrained=pretrained,
        use_attention=use_attention,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove possible "module." prefix if the model was trained with DataParallel.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key.replace("module.", "", 1)] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def save_confusion_matrix(cm, labels, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def evaluate(model, dataloader, device):
    all_image_paths = []

    all_fake_labels = []
    all_fake_preds = []

    all_transform_labels = []
    all_transform_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating RGB+Depth+Frequency"):
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)
            frequency = batch["frequency"].to(device)

            fake_labels = batch["fake_label"].to(device)
            transform_labels = batch["transform_label"].to(device)

            outputs = model(
                images=images,
                depth=depth,
                frequency=frequency,
            )

            fake_logits = outputs["fake_logits"]
            transform_logits = outputs["transform_logits"]

            fake_preds = torch.argmax(fake_logits, dim=1)
            transform_preds = torch.argmax(transform_logits, dim=1)

            all_fake_labels.extend(fake_labels.cpu().numpy().tolist())
            all_fake_preds.extend(fake_preds.cpu().numpy().tolist())

            all_transform_labels.extend(transform_labels.cpu().numpy().tolist())
            all_transform_preds.extend(transform_preds.cpu().numpy().tolist())

            all_image_paths.extend(batch["image_path"])

    return {
        "image_paths": all_image_paths,
        "fake_labels": all_fake_labels,
        "fake_preds": all_fake_preds,
        "transform_labels": all_transform_labels,
        "transform_preds": all_transform_preds,
    }


def compute_metrics(results):
    fake_labels = results["fake_labels"]
    fake_preds = results["fake_preds"]

    transform_labels = results["transform_labels"]
    transform_preds = results["transform_preds"]

    fake_acc = accuracy_score(fake_labels, fake_preds)
    fake_f1_macro = f1_score(fake_labels, fake_preds, average="macro")

    transform_acc = accuracy_score(transform_labels, transform_preds)
    transform_f1_macro = f1_score(transform_labels, transform_preds, average="macro")

    fake_accuracy_by_transform = {}

    for transform_id, transform_name in enumerate(TRANSFORM_LABEL_NAMES):
        indices = [
            i for i, label in enumerate(transform_labels)
            if label == transform_id
        ]

        if len(indices) == 0:
            fake_accuracy_by_transform[transform_name] = None
        else:
            subset_true = [fake_labels[i] for i in indices]
            subset_pred = [fake_preds[i] for i in indices]
            fake_accuracy_by_transform[transform_name] = accuracy_score(
                subset_true,
                subset_pred,
            )

    fake_report = classification_report(
        fake_labels,
        fake_preds,
        target_names=FAKE_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    transform_report = classification_report(
        transform_labels,
        transform_preds,
        target_names=TRANSFORM_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "fake_accuracy": fake_acc,
        "fake_f1_macro": fake_f1_macro,
        "fake_accuracy_by_transform": fake_accuracy_by_transform,
        "transform_accuracy": transform_acc,
        "transform_f1_macro": transform_f1_macro,
        "fake_classification_report": fake_report,
        "transform_classification_report": transform_report,
    }

    return metrics


def save_predictions(results, output_path):
    import pandas as pd

    rows = []

    for i in range(len(results["image_paths"])):
        rows.append(
            {
                "image_path": results["image_paths"][i],
                "fake_label": results["fake_labels"][i],
                "fake_pred": results["fake_preds"][i],
                "transform_label": results["transform_labels"][i],
                "transform_pred": results["transform_preds"][i],
                "fake_correct": results["fake_labels"][i] == results["fake_preds"][i],
                "transform_correct": results["transform_labels"][i]
                == results["transform_preds"][i],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RRDepthFrequencyDatasetFromCSV(
        csv_path=args.csv_path,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        use_attention=not args.no_attention,
        pretrained=not args.no_pretrained,
    )

    results = evaluate(model, dataloader, device)
    metrics = compute_metrics(results)

    save_predictions(
        results,
        output_dir / "predictions.csv",
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    fake_cm = confusion_matrix(
        results["fake_labels"],
        results["fake_preds"],
        labels=[0, 1],
    )

    transform_cm = confusion_matrix(
        results["transform_labels"],
        results["transform_preds"],
        labels=[0, 1, 2],
    )

    save_confusion_matrix(
        fake_cm,
        FAKE_LABEL_NAMES,
        "Fake classification confusion matrix",
        output_dir / "confusion_fake.png",
    )

    save_confusion_matrix(
        transform_cm,
        TRANSFORM_LABEL_NAMES,
        "Transformation classification confusion matrix",
        output_dir / "confusion_transform.png",
    )

    print()
    print("Evaluation results")
    print("------------------")
    print(f"Fake accuracy:        {metrics['fake_accuracy']:.4f}")
    print(f"Fake F1 macro:        {metrics['fake_f1_macro']:.4f}")

    print("Fake accuracy by transformation:")
    for name, value in metrics["fake_accuracy_by_transform"].items():
        if value is None:
            print(f"  {name}: None")
        else:
            print(f"  {name}: {value:.4f}")

    print(f"Transform accuracy:   {metrics['transform_accuracy']:.4f}")
    print(f"Transform F1 macro:   {metrics['transform_f1_macro']:.4f}")

    print()
    print(f"Saved results in: {output_dir}")


if __name__ == "__main__":
    main()