# train_geometric.py

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_depth import RRGeometricDatasetFromCSV
from model_depth import GeometricMultiTaskModel


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
):
    """
    Train the geometric multi-task model for one epoch.

    Args:
        model: The model we want to train.
        loader: DataLoader that provides RGB images, depth maps, edge maps, and labels.
        optimizer: Optimizer used to update model weights.
        device: "cuda" if GPU is available, otherwise "cpu".
        lambda_fake: Weight of the real/fake classification loss.
        lambda_transform: Weight of the transformation classification loss.

    Returns:
        Dictionary with training loss and accuracies.
    """

    model.train()

    # Classification losses for the two Project 2 tasks.
    fake_criterion = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    for batch in tqdm(loader, desc="Training geometric"):

        # Move batch tensors to GPU/CPU.
        images = batch["image"].to(device)
        depth = batch["depth"].to(device)
        edge_consistency = batch["edge_consistency"].to(device)

        fake_labels = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # Forward pass through RGB + depth + edge model.
        outputs = model(
            images=images,
            depth=depth,
            edge_consistency=edge_consistency,
        )

        # Loss for real/fake classification.
        fake_loss = fake_criterion(
            outputs["fake_logits"],
            fake_labels,
        )

        # Loss for transformation classification.
        transform_loss = transform_criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        # Multi-task loss, same logic as the RGB baseline.
        loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        # Convert logits into predicted labels.
        fake_pred = outputs["fake_logits"].argmax(dim=1)
        transform_pred = outputs["transform_logits"].argmax(dim=1)

        correct_fake += (fake_pred == fake_labels).sum().item()
        correct_transform += (transform_pred == transform_labels).sum().item()

        total += batch_size

    return {
        "loss": total_loss / total,
        "fake_acc": correct_fake / total,
        "transform_acc": correct_transform / total,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
):
    """
    Evaluate the geometric multi-task model on validation data.

    Args:
        model: The trained model.
        loader: Validation DataLoader.
        device: "cuda" or "cpu".
        lambda_fake: Weight of the real/fake loss.
        lambda_transform: Weight of the transformation loss.

    Returns:
        Dictionary with validation loss and accuracies.
    """

    model.eval()

    fake_criterion = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    for batch in tqdm(loader, desc="Validation geometric"):

        images = batch["image"].to(device)
        depth = batch["depth"].to(device)
        edge_consistency = batch["edge_consistency"].to(device)

        fake_labels = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # Forward pass only. No gradients are stored during validation.
        outputs = model(
            images=images,
            depth=depth,
            edge_consistency=edge_consistency,
        )

        fake_loss = fake_criterion(
            outputs["fake_logits"],
            fake_labels,
        )

        transform_loss = transform_criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        fake_pred = outputs["fake_logits"].argmax(dim=1)
        transform_pred = outputs["transform_logits"].argmax(dim=1)

        correct_fake += (fake_pred == fake_labels).sum().item()
        correct_transform += (transform_pred == transform_labels).sum().item()

        total += batch_size

    return {
        "loss": total_loss / total,
        "fake_acc": correct_fake / total,
        "transform_acc": correct_transform / total,
    }


def main():
    """
    Main training function for the geometric model.

    This follows the same structure as train.py:
        1. Read command-line arguments.
        2. Create datasets and dataloaders.
        3. Create the model.
        4. Create optimizer and scheduler.
        5. Train and validate.
        6. Save the best checkpoint.
    """

    parser = argparse.ArgumentParser()

    # Same arguments as the RGB baseline.
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    # New argument for geometric training:
    # folder containing precomputed depth maps.
    parser.add_argument("--depth_root", type=str, required=True)

    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--checkpoint_name", type=str, default="best_geometric.pt")

    # Multi-task loss weights.
    parser.add_argument("--lambda_fake", type=float, default=1.0)
    parser.add_argument("--lambda_transform", type=float, default=1.0)

    # Checkpoint folder.
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Optional switches for ablation studies.
    parser.add_argument(
        "--no_edge",
        action="store_true",
        help="Disable edge-depth consistency branch and use only RGB + depth.",
    )

    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Disable attention/gating after feature fusion.",
    )

    args = parser.parse_args()

    # Select GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint folder if needed.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training dataset:
    # reads the same CSV as the RGB baseline, but also loads depth maps.
    train_dataset = RRGeometricDatasetFromCSV(
        csv_path=args.train_csv,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=True,
    )

    # Validation dataset.
    val_dataset = RRGeometricDatasetFromCSV(
        csv_path=args.val_csv,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=False,
    )

    # DataLoaders create mini-batches.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Create the geometric multi-task model.
    model = GeometricMultiTaskModel(
        num_transform_classes=3,
        pretrained=True,
        use_edge=not args.no_edge,
        use_attention=not args.no_attention,
    )

    model = model.to(device)

    # AdamW optimizer, same style as the current training script.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Reduce LR when validation loss stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_score = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
        )

        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        # Combined validation score for the two tasks.
        val_score = 0.5 * val_metrics["fake_acc"] + 0.5 * val_metrics["transform_acc"]

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Val score: {val_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

        # Save best checkpoint according to the combined validation score.
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / args.checkpoint_name

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "val_score": val_score,
                    "args": vars(args),
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint to {checkpoint_path}")

        else:
            epochs_without_improvement += 1

            print(
                f"No improvement for {epochs_without_improvement}/"
                f"{args.patience} epochs"
            )

            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()