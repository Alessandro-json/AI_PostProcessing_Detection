import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_freq import RRFreqDatasetFromCSV
from model_freq import FreqMultiTaskModel


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
):
    """
    Train the frequency multi-task model for one epoch.

    Args:
        model:            The model to train.
        loader:           DataLoader providing RGB images, FFT maps, and labels.
        optimizer:        Optimizer used to update model weights.
        device:           "cuda" or "cpu".
        lambda_fake:      Weight of the real/fake classification loss.
        lambda_transform: Weight of the transformation classification loss.

    Returns:
        Dictionary with training loss and accuracies.
    """

    model.train()

    fake_criterion      = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    correct_fake      = 0
    correct_transform = 0
    total             = 0

    for batch in tqdm(loader, desc="Training freq"):

        # Move all tensors to GPU/CPU.
        images          = batch["image"].to(device)
        freq_map        = batch["freq_map"].to(device)
        fake_labels     = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # Forward pass: RGB + FFT map -> two predictions.
        outputs = model(images=images, freq_map=freq_map)

        # Loss for real/fake task.
        fake_loss = fake_criterion(
            outputs["fake_logits"],
            fake_labels,
        )

        # Loss for transformation task.
        transform_loss = transform_criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        # Multi-task loss: weighted sum.
        loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        # Convert logits to predicted class ids.
        fake_pred      = outputs["fake_logits"].argmax(dim=1)
        transform_pred = outputs["transform_logits"].argmax(dim=1)

        correct_fake      += (fake_pred      == fake_labels).sum().item()
        correct_transform += (transform_pred == transform_labels).sum().item()
        total             += batch_size

    return {
        "loss":          total_loss / total,
        "fake_acc":      correct_fake / total,
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
    Evaluate the frequency multi-task model on validation data.

    Args:
        model:            The trained model.
        loader:           Validation DataLoader.
        device:           "cuda" or "cpu".
        lambda_fake:      Weight of the real/fake loss.
        lambda_transform: Weight of the transformation loss.

    Returns:
        Dictionary with validation loss and accuracies.
    """

    model.eval()

    fake_criterion      = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    correct_fake      = 0
    correct_transform = 0
    total             = 0

    for batch in tqdm(loader, desc="Validation freq"):

        images           = batch["image"].to(device)
        freq_map         = batch["freq_map"].to(device)
        fake_labels      = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # Forward pass only. No gradients stored.
        outputs = model(images=images, freq_map=freq_map)

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

        fake_pred      = outputs["fake_logits"].argmax(dim=1)
        transform_pred = outputs["transform_logits"].argmax(dim=1)

        correct_fake      += (fake_pred      == fake_labels).sum().item()
        correct_transform += (transform_pred == transform_labels).sum().item()
        total             += batch_size

    return {
        "loss":          total_loss / total,
        "fake_acc":      correct_fake / total,
        "transform_acc": correct_transform / total,
    }


def main():
    """
    Main training function for the frequency multi-task model.

    Structure mirrors train.py and train_depth.py exactly:
        1. Parse arguments.
        2. Build datasets and dataloaders.
        3. Build model.
        4. Build optimizer and scheduler.
        5. Train and validate epoch by epoch.
        6. Save the best checkpoint.

    Example usage:
        python src/train_freq.py \
            --train_csv data/splits/train_balanced.csv \
            --val_csv   data/splits/val_balanced.csv \
            --image_root data/raw/RRDataset_subset \
            --epochs 10 \
            --batch_size 32 \
            --checkpoint_name best_freq.pt
    """

    parser = argparse.ArgumentParser(
        description="Train the RGB + FFT frequency multi-task model."
    )

    # --- Paths ---
    parser.add_argument("--train_csv",   type=str, required=True,
                        help="Path to training CSV.")
    parser.add_argument("--val_csv",     type=str, required=True,
                        help="Path to validation CSV.")
    parser.add_argument("--image_root",  type=str, required=True,
                        help="Root folder containing subset images.")

    # --- Training hyperparameters ---
    # Same defaults as train.py for easy comparison.
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--image_size",   type=int,   default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=0,
                        help="Set to 0 on Colab to avoid DataLoader issues.")
    parser.add_argument("--patience",     type=int,   default=4,
                        help="Early stopping patience in epochs.")

    # --- Multi-task loss weights ---
    parser.add_argument("--lambda_fake",      type=float, default=1.0,
                        help="Weight of the real/fake loss.")
    parser.add_argument("--lambda_transform", type=float, default=1.0,
                        help="Weight of the transformation loss.")

    # --- Model options ---
    parser.add_argument("--freq_features", type=int, default=128,
                        help="Output size of the FreqEncoder.")
    parser.add_argument("--no_attention",  action="store_true",
                        help="Disable sigmoid gating after fusion.")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train RGB backbone from scratch (no ImageNet weights).")

    # --- Checkpoint ---
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_freq.pt")

    args = parser.parse_args()

    # Select GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint folder.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets ---
    train_dataset = RRFreqDatasetFromCSV(
        csv_path=args.train_csv,
        image_root=args.image_root,
        image_size=args.image_size,
        train=True,
    )

    val_dataset = RRFreqDatasetFromCSV(
        csv_path=args.val_csv,
        image_root=args.image_root,
        image_size=args.image_size,
        train=False,
    )

    # --- DataLoaders ---
    # num_workers=0 is the safe default for Colab (avoids multiprocessing issues).
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

    # --- Model ---
    model = FreqMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        freq_features=args.freq_features,
        use_attention=not args.no_attention,
    )
    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Scheduler ---
    # Reduces learning rate by 0.5 when validation loss stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_score         = 0.0
    epochs_without_improvement = 0

    # --- Training loop ---
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

        # Combined score: average of the two task accuracies.
        # Same formula used in train.py and train_depth.py.
        val_score = 0.5 * val_metrics["fake_acc"] + 0.5 * val_metrics["transform_acc"]

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Val score: {val_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint only when the combined score improves.
        if val_score > best_val_score:
            best_val_score             = val_score
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / args.checkpoint_name

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch,
                    "val_metrics":      val_metrics,
                    "val_score":        val_score,
                    "args":             vars(args),
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
