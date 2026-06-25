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

from dataset_depth import RRGeometricDatasetFromCSV
from model_depth import GeometricMultiTaskModel


FAKE_LABEL_NAMES = ["real", "ai"]
TRANSFORM_LABEL_NAMES = ["original", "transfer", "redigital"]


def load_checkpoint(model, checkpoint_path, device):
    """
    Load trained weights into the depth-based model.

    Different training scripts may save checkpoints in different formats:
    1. Only the model state_dict.
    2. A dictionary containing "model_state_dict".
    3. A dictionary containing "state_dict".

    This function supports all three common cases.
    """

    # Load checkpoint on the selected device.
    # map_location avoids errors when loading a GPU checkpoint on CPU.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Case 1: checkpoint saved as {"model_state_dict": ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    # Case 2: checkpoint saved as {"state_dict": ...}
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])

    # Case 3: checkpoint is directly the model state_dict.
    else:
        model.load_state_dict(checkpoint)

    return model


def evaluate_model(model, dataloader, device, task, use_edge=True):
    """
    Run the depth-based model on the test set and collect predictions.

    The model receives RGB images and precomputed depth maps.
    If use_edge is True, the edge-consistency map is also passed to the model.
    Logits are converted into predicted classes using argmax.
    """

    # Set the model to evaluation mode.
    model.eval()

    # This list will contain one dictionary per image.
    rows = []

    # Disable gradient computation during evaluation.
    # This saves memory and makes inference faster.
    with torch.no_grad():

        # Loop over test batches.
        for batch in dataloader:

            # Move RGB images and depth maps to GPU if available, otherwise CPU.
            images = batch["image"].to(device)
            depth = batch["depth"].to(device)

            # The edge-consistency map is needed only for checkpoints trained with edge.
            edge_consistency = None
            if use_edge:
                if "edge_consistency" not in batch:
                    raise KeyError(
                        "edge_consistency is missing from the batch. "
                        "Use --no_edge if the checkpoint was trained without edge consistency."
                    )
                edge_consistency = batch["edge_consistency"].to(device)

            # Forward pass through the depth-based model.
            outputs = model(
                images=images,
                depth=depth,
                edge_consistency=edge_consistency,
            )

            # Real/fake predictions.
            if task in ["fake", "multitask"]:
                true_fake = batch["fake_label"].to(device)
                fake_logits = outputs["fake_logits"]
                pred_fake = torch.argmax(fake_logits, dim=1)

            # Transformation predictions.
            if task in ["transform", "multitask"]:
                true_transform = batch["transform_label"].to(device)
                transform_logits = outputs["transform_logits"]
                pred_transform = torch.argmax(transform_logits, dim=1)

            # Store predictions image by image.
            batch_size = images.size(0)

            for i in range(batch_size):
                row = {
                    "image_path": batch["image_path"][i],
                }

                # Save real/fake prediction fields only when this task is active.
                if task in ["fake", "multitask"]:
                    row["true_fake"] = int(true_fake[i].cpu())
                    row["pred_fake"] = int(pred_fake[i].cpu())

                # Save transformation prediction fields only when this task is active.
                if task in ["transform", "multitask"]:
                    row["true_transform"] = int(true_transform[i].cpu())
                    row["pred_transform"] = int(pred_transform[i].cpu())

                # For fake-only evaluation, we still store the true transformation label.
                # This allows us to compute real/fake accuracy separately for:
                # original, transmitted, and re-digitized images.
                if task == "fake":
                    row["true_transform"] = int(batch["transform_label"][i])

                rows.append(row)

    # Convert collected rows into a DataFrame.
    # This makes metric computation and CSV export easier.
    return pd.DataFrame(rows)


def compute_metrics(predictions_df, task):
    """
    Compute evaluation metrics from the predictions DataFrame.
    """

    metrics = {}

    # Metrics for the real/fake task.
    if task in ["fake", "multitask"]:

        y_true_fake = predictions_df["true_fake"]
        y_pred_fake = predictions_df["pred_fake"]

        metrics["fake_accuracy"] = accuracy_score(
            y_true_fake,
            y_pred_fake,
        )

        metrics["fake_f1_macro"] = f1_score(
            y_true_fake,
            y_pred_fake,
            average="macro",
            zero_division=0,
        )

        # Real/fake accuracy separately for each transformation class.
        metrics["fake_accuracy_by_transform"] = {}

        for transform_id, transform_name in enumerate(TRANSFORM_LABEL_NAMES):

            # Select images belonging to one transformation category.
            subset = predictions_df[predictions_df["true_transform"] == transform_id]

            if len(subset) == 0:
                metrics["fake_accuracy_by_transform"][transform_name] = None
            else:
                metrics["fake_accuracy_by_transform"][transform_name] = accuracy_score(
                    subset["true_fake"],
                    subset["pred_fake"],
                )

    # Metrics for the transformation task.
    if task in ["transform", "multitask"]:

        y_true_transform = predictions_df["true_transform"]
        y_pred_transform = predictions_df["pred_transform"]

        metrics["transform_accuracy"] = accuracy_score(
            y_true_transform,
            y_pred_transform,
        )

        metrics["transform_f1_macro"] = f1_score(
            y_true_transform,
            y_pred_transform,
            average="macro",
            zero_division=0,
        )

    return metrics


def save_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """
    Create and save a confusion matrix plot.
    """

    # Build numeric confusion matrix.
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    # Create a sklearn display object.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    # Create the figure.
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot the confusion matrix.
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    # Add title and save to file.
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)

    # Close the figure to avoid memory issues in repeated evaluations.
    plt.close(fig)


def save_results(predictions_df, metrics, output_dir, task):
    """
    Save all evaluation outputs.

    The function creates:
    - predictions.csv
    - metrics.json
    - confusion_fake.png and/or confusion_transform.png
    """

    # Create output folder if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file paths.
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"

    # Save one row per evaluated image.
    predictions_df.to_csv(predictions_path, index=False)

    # Save global metrics in JSON format.
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save confusion matrix for real/fake prediction only when available.
    if task in ["fake", "multitask"]:
        save_confusion_matrix(
            y_true=predictions_df["true_fake"],
            y_pred=predictions_df["pred_fake"],
            labels=FAKE_LABEL_NAMES,
            title="Real/Fake Confusion Matrix",
            output_path=output_dir / "confusion_fake.png",
        )

    # Save confusion matrix for transformation prediction only when available.
    if task in ["transform", "multitask"]:
        save_confusion_matrix(
            y_true=predictions_df["true_transform"],
            y_pred=predictions_df["pred_transform"],
            labels=TRANSFORM_LABEL_NAMES,
            title="Transformation Confusion Matrix",
            output_path=output_dir / "confusion_transform.png",
        )


def print_metrics(metrics, task):
    """
    Print the main metrics in the terminal.
    """

    print("\nEvaluation results")
    print("=" * 50)

    # Print real/fake metrics only when available.
    if task in ["fake", "multitask"]:
        print(f"Fake accuracy:        {metrics['fake_accuracy']:.4f}")
        print(f"Fake F1 macro:        {metrics['fake_f1_macro']:.4f}")

        print("\nFake accuracy by transformation:")

        for name, value in metrics["fake_accuracy_by_transform"].items():
            if value is None:
                print(f"  {name}: not available")
            else:
                print(f"  {name}: {value:.4f}")

    # Print transformation metrics only when available.
    if task in ["transform", "multitask"]:
        print(f"Transform accuracy:   {metrics['transform_accuracy']:.4f}")
        print(f"Transform F1 macro:   {metrics['transform_f1_macro']:.4f}")


def parse_args():
    """
    Read command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate RGB + depth multi-task model."
    )

    # Evaluation task.
    parser.add_argument(
        "--task",
        type=str,
        default="multitask",
        choices=["fake", "transform", "multitask"],
        help=(
            "Evaluation task. "
            "'fake' evaluates only the real/fake head. "
            "'transform' evaluates only the transformation head. "
            "'multitask' evaluates both heads."
        ),
    )

    # CSV containing test image paths and labels.
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/splits/test_balanced.csv",
        help="Path to test CSV file.",
    )

    # Root folder containing the RGB image files.
    parser.add_argument(
        "--image_root",
        type=str,
        default="data/raw/RRDataset_subset",
        help="Root folder containing subset RGB images.",
    )

    # Root folder containing the precomputed depth maps.
    parser.add_argument(
        "--depth_root",
        type=str,
        required=True,
        help="Root folder containing precomputed .npy depth maps.",
    )

    # Trained model checkpoint.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_depth_only.pt",
        help="Path to model checkpoint.",
    )

    # Folder where results will be saved.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/depth_model",
        help="Folder where evaluation results are saved.",
    )

    # Number of images processed at once.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    # Input size used by the model.
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size.",
    )

    # Number of subprocesses used by the DataLoader.
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    # Use this flag only if the model was trained from scratch.
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use this if the checkpoint was trained without ImageNet pretrained weights.",
    )

    # Use this flag when evaluating a checkpoint trained without edge consistency.
    parser.add_argument(
        "--no_edge",
        action="store_true",
        help="Use this if the checkpoint was trained without the edge-consistency branch.",
    )

    # Use this flag when evaluating a checkpoint trained without attention.
    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Use this if the checkpoint was trained without the attention block.",
    )

    return parser.parse_args()


def main():
    """
    Main evaluation pipeline.

    Steps:
    1. Read arguments.
    2. Select device.
    3. Build depth dataset and dataloader.
    4. Build depth-based model.
    5. Load checkpoint.
    6. Run evaluation.
    7. Compute and save results.
    """

    args = parse_args()
    print(f"Selected task: {args.task}")

    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the test dataset from CSV.
    # The dataset returns RGB images, depth maps, labels, and optionally edge consistency.
    dataset = RRGeometricDatasetFromCSV(
        csv_path=args.csv_path,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=False,
    )

    # DataLoader creates batches from the dataset.
    # shuffle=False because we want deterministic evaluation order.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Build the RGB + depth multi-task architecture.
    model = GeometricMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        use_edge=not args.no_edge,
        use_attention=not args.no_attention,
    )

    # Load the trained weights into the model.
    model = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # Move model to GPU/CPU.
    model = model.to(device)

    # Run inference on the full test set.
    predictions_df = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        task=args.task,
        use_edge=not args.no_edge,
    )

    # Compute all metrics from saved predictions.
    metrics = compute_metrics(
        predictions_df,
        task=args.task,
    )

    # Save CSV, JSON, and confusion matrices.
    output_dir = Path(args.output_dir)
    save_results(
        predictions_df=predictions_df,
        metrics=metrics,
        output_dir=output_dir,
        task=args.task,
    )

    # Print summary in terminal.
    print_metrics(
        metrics=metrics,
        task=args.task,
    )

    print("\nSaved results in:")
    print(output_dir)


if __name__ == "__main__":
    main()
