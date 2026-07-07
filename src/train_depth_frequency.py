import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_depth_frequency import RRDepthFrequencyDatasetFromCSV
from model_depth_frequency import RGBDepthFrequencyMultiTaskModel

try:
    from loss import UncertaintyWeightedLoss
except ImportError:
    UncertaintyWeightedLoss = None


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake=1.0,
    lambda_transform=1.0,
    uncertainty_loss=None,
):
    """
    Train the RGB + depth + frequency model for one epoch.
    """

    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    weight_fake_sum = 0.0
    weight_transform_sum = 0.0

    for batch in tqdm(loader, desc="Training RGB+Depth+Frequency"):
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

        fake_loss = criterion(outputs["fake_logits"], fake_labels)
        transform_loss = criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        # If uncertainty weighting is enabled, the task weights are learned.
        # Otherwise, fixed lambda values are used.
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
        correct_transform += (
            transform_preds == transform_labels
        ).sum().item()

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
    lambda_fake=1.0,
    lambda_transform=1.0,
    uncertainty_loss=None,
):
    """
    Evaluate the RGB + depth + frequency model on the validation set.
    """

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    weight_fake_sum = 0.0
    weight_transform_sum = 0.0

    for batch in tqdm(loader, desc="Validation RGB+Depth+Frequency"):
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

        fake_loss = criterion(outputs["fake_logits"], fake_labels)
        transform_loss = criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        # Use the same loss combination strategy during validation.
        if uncertainty_loss is not None:
            loss, loss_info = uncertainty_loss(fake_loss, transform_loss)
            weight_fake_sum += loss_info["weight_fake"]
            weight_transform_sum += loss_info["weight_transform"]
        else:
            loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        fake_preds = outputs["fake_logits"].argmax(dim=1)
        transform_preds = outputs["transform_logits"].argmax(dim=1)

        correct_fake += (fake_preds == fake_labels).sum().item()
        correct_transform += (
            transform_preds == transform_labels
        ).sum().item()

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


def parse_args():
    """
    Read command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Train RGB + depth + frequency multi-task model."
    )

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Folder where checkpoints are saved.",
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_depth_frequency.pt",
        help="Name of the best checkpoint file.",
    )

    parser.add_argument(
        "--lambda_fake",
        type=float,
        default=1.0,
        help="Fixed weight for the real/fake loss.",
    )

    parser.add_argument(
        "--lambda_transform",
        type=float,
        default=1.0,
        help="Fixed weight for the transformation loss.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Number of epochs without validation improvement before early stopping.",
    )

    parser.add_argument(
        "--use_uncertainty_weighting",
        action="store_true",
        help="Use learnable uncertainty weighting between the two task losses.",
    )

    parser.add_argument(
        "--no_attention",
        action="store_true",
        help="Disable the shared attention gate after fusion.",
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use this flag if you do not want ImageNet pretrained ResNet weights.",
    )

    return parser.parse_args()


def main():
    """
    Main training pipeline.
    """

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RRDepthFrequencyDatasetFromCSV(
        csv_path=args.train_csv,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=True,
    )

    val_dataset = RRDepthFrequencyDatasetFromCSV(
        csv_path=args.val_csv,
        image_root=args.image_root,
        depth_root=args.depth_root,
        image_size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = RGBDepthFrequencyMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        use_attention=not args.no_attention,
    ).to(device)

    # Create the uncertainty loss only if requested.
    # Otherwise, the training keeps using fixed lambda weights.
    if args.use_uncertainty_weighting:
        if UncertaintyWeightedLoss is None:
            raise ImportError(
                "uncertainty_loss.py was not found. "
                "Put it inside src/ or disable --use_uncertainty_weighting."
            )

        uncertainty_loss = UncertaintyWeightedLoss().to(device)
    else:
        uncertainty_loss = None

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_score = -1.0
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

        # Validation score used for model selection.
        # Both tasks are equally important.
        val_score = (
            0.5 * val_metrics["fake_acc"]
            + 0.5 * val_metrics["transform_acc"]
        )

        scheduler.step(val_score)

        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")
        print(f"Val score: {val_score:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / args.checkpoint_name

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_score": best_val_score,
                "args": vars(args),
            }

            if uncertainty_loss is not None:
                checkpoint[
                    "uncertainty_loss_state_dict"
                ] = uncertainty_loss.state_dict()

            torch.save(checkpoint, checkpoint_path)

            print(f"Saved best checkpoint to {checkpoint_path}")

        else:
            epochs_without_improvement += 1

            print(
                f"No improvement for "
                f"{epochs_without_improvement}/{args.patience} epochs"
            )

            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()