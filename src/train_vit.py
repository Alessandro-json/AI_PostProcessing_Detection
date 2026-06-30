import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_vit import RRViTDatasetFromCSV
from model_vit import ViTMultiTaskModel


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
):
    """
    Train the ViT multi-task model for one epoch.
    Same structure as train_one_epoch() in train.py / train_freq.py.
    """

    model.train()

    fake_criterion      = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    correct_fake      = 0
    correct_transform = 0
    total             = 0

    for batch in tqdm(loader, desc="Training ViT"):

        images           = batch["image"].to(device)
        fake_labels      = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # ViT forward pass — RGB only, no extra modality.
        outputs = model(images)

        fake_loss = fake_criterion(
            outputs["fake_logits"],
            fake_labels,
        )
        transform_loss = transform_criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
):
    """
    Evaluate the ViT multi-task model on validation data.
    """

    model.eval()

    fake_criterion      = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    correct_fake      = 0
    correct_transform = 0
    total             = 0

    for batch in tqdm(loader, desc="Validation ViT"):

        images           = batch["image"].to(device)
        fake_labels      = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        outputs = model(images)

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
    Main training function for the ViT multi-task model.

    Structure mirrors train.py / train_freq.py exactly.

    Example usage:
        python src/train_vit.py \\
            --train_csv data/splits/train_balanced.csv \\
            --val_csv   data/splits/val_balanced.csv \\
            --image_root data/raw/RRDataset_subset \\
            --epochs 10 \\
            --batch_size 16 \\
            --checkpoint_name best_vit_multitask_1_1.pt

    Note on batch_size: ViT-Small uses more GPU memory than ResNet18
    for the same batch size. If you hit an out-of-memory error on
    Colab free, lower --batch_size to 16 (or 8).
    """

    parser = argparse.ArgumentParser(
        description="Train the ViT-Small multi-task model (RGB only)."
    )

    # --- Paths ---
    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--val_csv",     type=str, required=True)
    parser.add_argument("--image_root",  type=str, required=True)

    # --- Training hyperparameters ---
    # batch_size default lower than train.py/train_freq.py (32) because
    # ViT-Small needs more memory per sample than ResNet18.
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--image_size",   type=int,   default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=0,
                        help="Set to 0 on Colab to avoid DataLoader issues.")
    parser.add_argument("--patience",     type=int,   default=4,
                        help="Early stopping patience in epochs.")

    # --- Multi-task loss weights ---
    parser.add_argument("--lambda_fake",      type=float, default=1.0)
    parser.add_argument("--lambda_transform", type=float, default=1.0)

    # --- Model options ---
    parser.add_argument(
        "--vit_model_name",
        type=str,
        default="vit_small_patch16_224",
        help=(
            "timm model name. Default vit_small_patch16_224 is the "
            "recommended choice for Colab free. Use vit_tiny_patch16_224 "
            "if you run out of memory, or vit_base_patch16_224 if you "
            "have more GPU memory available."
        ),
    )
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train ViT from scratch (no ImageNet weights). Not recommended.")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze the ViT backbone, train only the two heads. "
                             "Faster and uses less memory, lower accuracy ceiling.")

    # --- Checkpoint ---
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_vit.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"ViT model:    {args.vit_model_name}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Datasets ---
    train_dataset = RRViTDatasetFromCSV(
        csv_path=args.train_csv,
        image_root=args.image_root,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = RRViTDatasetFromCSV(
        csv_path=args.val_csv,
        image_root=args.image_root,
        image_size=args.image_size,
        train=False,
    )

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
    model = ViTMultiTaskModel(
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        vit_model_name=args.vit_model_name,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # --- Optimizer ---
    # Only optimize parameters that require gradients (matters if
    # --freeze_backbone is set).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Scheduler ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_score             = 0.0
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

        val_score = 0.5 * val_metrics["fake_acc"] + 0.5 * val_metrics["transform_acc"]

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Val score: {val_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

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
