# train_geometric.py

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_depth import RRGeometricDatasetFromCSV
from model_depth import GeometricMultiTaskModel
from uncertainty_loss import UncertaintyWeightedLoss


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake=1.0,
    lambda_transform=1.0,
    uncertainty_loss=None,
):
	
    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    weight_fake_sum = 0.0
    weight_transform_sum = 0.0

    for batch in tqdm(loader, desc="Training geometric"):
        images = batch["image"].to(device)
        depth = batch["depth"].to(device)
        fake_labels = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        edge_consistency = batch.get("edge_consistency")
        if edge_consistency is not None:
            edge_consistency = edge_consistency.to(device)

        outputs = model(
            images=images,
            depth=depth,
            edge_consistency=edge_consistency,
        )

        fake_loss = criterion(outputs["fake_logits"], fake_labels)
        transform_loss = criterion(outputs["transform_logits"], transform_labels)

        if uncertainty_loss is not None:
            loss, loss_info = uncertainty_loss(fake_loss, transform_loss)
            weight_fake_sum += loss_info["weight_fake"]
            weight_transform_sum += loss_info["weight_transform"]
        else:
            loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        fake_preds = outputs["fake_logits"].argmax(dim=1)
        transform_preds = outputs["transform_logits"].argmax(dim=1)

        correct_fake += (fake_preds == fake_labels).sum().item()
        correct_transform += (transform_preds == transform_labels).sum().item()
        total += batch_size

    metrics = {
        "loss": total_loss / total,
        "fake_acc": correct_fake / total,
        "transform_acc": correct_transform / total,
    }

    if uncertainty_loss is not None:
        metrics["weight_fake"] = weight_fake_sum / len(loader)
        metrics["weight_transform"] = weight_transform_sum / len(loader)

    return metrics

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
	uncertainty_loss=None
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
		# Use the same loss combination strategy during validation.
		# This affects only the validation loss, not the accuracy computation.
        if uncertainty_loss is not None:
            loss, loss_info = uncertainty_loss(fake_loss, transform_loss)
        else:
            loss = lambda_fake * fake_loss + lambda_transform * transform_loss
        #loss = lambda_fake * fake_loss + lambda_transform * transform_loss

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
	#use uncertainty weighting
    parser.add_argument(
    "--use_uncertainty_weighting",
    action="store_true",
    help="Use learnable uncertainty weighting between fake and transform losses.",
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

	# Create the uncertainty loss only if requested.
	# Otherwise, the training keeps using fixed lambda weights.
    if args.use_uncertainty_weighting:
    	uncertainty_loss = UncertaintyWeightedLoss().to(device)
    else:
    	uncertainty_loss = None

    model = model.to(device)

    # If uncertainty weighting is enabled, the optimizer must also update
	# the learnable log-variance parameters of the loss function.
    if uncertainty_loss is not None:
        optimizer = torch.optim.AdamW(
			list(model.parameters()) + list(uncertainty_loss.parameters()),
			lr=args.lr,
			weight_decay=args.weight_decay,
		)
    else:
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
			uncertainty_loss=uncertainty_loss,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
			uncertainty_loss=uncertainty_loss,
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