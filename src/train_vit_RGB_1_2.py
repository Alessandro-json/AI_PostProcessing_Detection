import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RRDatasetFromCSV, build_train_transform, build_eval_transform
from model_vit_RGB import ViTRGBMultiTaskModel
from loss import UncertaintyWeightedLoss


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
# Identical to compute_loss() in train_RGB.py.

def compute_loss(outputs, batch, task, criterion, lambda_fake, lambda_transform, device, adaptive_loss=None):
    loss_info = {}

    if task in ["fake", "multitask"]:
        fake_labels = batch["fake_label"].to(device)
        fake_loss = criterion(outputs["fake_logits"], fake_labels)
        loss_info["fake_loss"] = fake_loss.item()

    if task in ["transform", "multitask"]:
        transform_labels = batch["transform_label"].to(device)
        transform_loss = criterion(outputs["transform_logits"], transform_labels)
        loss_info["transform_loss"] = transform_loss.item()

    if task == "fake":
        loss = fake_loss
    elif task == "transform":
        loss = transform_loss
    elif task == "multitask":
        if adaptive_loss is None:
            loss = lambda_fake * fake_loss + lambda_transform * transform_loss
            loss_info["weight_fake"]      = lambda_fake
            loss_info["weight_transform"] = lambda_transform
        else:
            loss, learned_info = adaptive_loss(
                fake_loss=fake_loss,
                transform_loss=transform_loss,
            )
            loss_info.update(learned_info)
    else:
        raise ValueError(f"Unknown task: {task}")

    return loss, loss_info


def compute_validation_score(val_metrics, task):
    if task == "fake":
        return val_metrics["fake_acc"]
    if task == "transform":
        return val_metrics["transform_acc"]
    if task == "multitask":
        return 0.5 * val_metrics["fake_acc"] + 0.5 * val_metrics["transform_acc"]
    raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, task,
                    lambda_fake=1.0, lambda_transform=2.0, adaptive_loss=None):
    """
    Train the ViT-RGB model for one epoch.
    Identical to train_one_epoch() in train_RGB.py — only the model class changes.
    """

    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    correct_fake      = 0
    correct_transform = 0
    total_samples     = 0
    weight_fake_sum      = 0.0
    weight_transform_sum = 0.0
    weight_count         = 0

    for batch in tqdm(loader, desc="Training ViT-RGB 1-2"):

        images = batch["image"].to(device)
        outputs = model(images)

        loss, loss_info = compute_loss(
            outputs=outputs, batch=batch, task=task, criterion=criterion,
            lambda_fake=lambda_fake, lambda_transform=lambda_transform,
            device=device, adaptive_loss=adaptive_loss,
        )

        if adaptive_loss is not None:
            weight_fake_sum      += loss_info["weight_fake"]
            weight_transform_sum += loss_info["weight_transform"]
            weight_count         += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

        if task in ["fake", "multitask"]:
            fake_labels = batch["fake_label"].to(device)
            fake_pred   = outputs["fake_logits"].argmax(dim=1)
            correct_fake += (fake_pred == fake_labels).sum().item()

        if task in ["transform", "multitask"]:
            transform_labels = batch["transform_label"].to(device)
            transform_pred   = outputs["transform_logits"].argmax(dim=1)
            correct_transform += (transform_pred == transform_labels).sum().item()

    metrics = {"loss": total_loss / total_samples}
    if task in ["fake",      "multitask"]: metrics["fake_acc"]      = correct_fake      / total_samples
    if task in ["transform", "multitask"]: metrics["transform_acc"] = correct_transform / total_samples
    if adaptive_loss is not None and weight_count > 0:
        metrics["weight_fake"]      = weight_fake_sum      / weight_count
        metrics["weight_transform"] = weight_transform_sum / weight_count

    return metrics


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, task,
             lambda_fake=1.0, lambda_transform=2.0, adaptive_loss=None):
    """
    Evaluate the ViT-RGB model on validation data.
    Identical to evaluate() in train_RGB.py.
    """

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    total_samples     = 0
    correct_fake      = 0
    correct_transform = 0
    weight_fake_sum      = 0.0
    weight_transform_sum = 0.0
    weight_count         = 0

    for batch in tqdm(loader, desc="Validation ViT-RGB 1-2"):

        images = batch["image"].to(device)
        outputs = model(images)

        loss, loss_info = compute_loss(
            outputs=outputs, batch=batch, task=task, criterion=criterion,
            lambda_fake=lambda_fake, lambda_transform=lambda_transform,
            device=device, adaptive_loss=adaptive_loss,
        )

        if adaptive_loss is not None:
            weight_fake_sum      += loss_info["weight_fake"]
            weight_transform_sum += loss_info["weight_transform"]
            weight_count         += 1

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        total_samples += batch_size

        if task in ["fake", "multitask"]:
            fake_labels = batch["fake_label"].to(device)
            fake_pred   = outputs["fake_logits"].argmax(dim=1)
            correct_fake += (fake_pred == fake_labels).sum().item()

        if task in ["transform", "multitask"]:
            transform_labels = batch["transform_label"].to(device)
            transform_pred   = outputs["transform_logits"].argmax(dim=1)
            correct_transform += (transform_pred == transform_labels).sum().item()

    metrics = {"loss": total_loss / total_samples}
    if task in ["fake",      "multitask"]: metrics["fake_acc"]      = correct_fake      / total_samples
    if task in ["transform", "multitask"]: metrics["transform_acc"] = correct_transform / total_samples
    if adaptive_loss is not None and weight_count > 0:
        metrics["weight_fake"]      = weight_fake_sum      / weight_count
        metrics["weight_transform"] = weight_transform_sum / weight_count

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Train the ViT-Small RGB multi-task model with lambda_fake=1.0, lambda_transform=2.0.

    This script is the direct ViT alternative to:
        python src/train_RGB.py --task multitask --lambda_fake 1.0 --lambda_transform 2.0

    The loss weighting 1.0/2.0 gives twice the importance to the transformation
    task compared to the fake/real task. This is useful when the transformation
    head is harder to train (which is common with 3 classes vs 2).

    Example usage:
        python src/train_vit_RGB_1_2.py \\
            --train_csv  data/splits/train_balanced.csv \\
            --val_csv    data/splits/val_balanced.csv \\
            --image_root data/raw/RRDataset_subset \\
            --epochs 10 \\
            --batch_size 16 \\
            --num_workers 0 \\
            --checkpoint_name best_vit_rgb_multitask_1_2.pt
    """

    parser = argparse.ArgumentParser(
        description="Train ViT-Small RGB — multitask, lambda_fake=1.0, lambda_transform=2.0."
    )

    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--val_csv",     type=str, required=True)
    parser.add_argument("--image_root",  type=str, required=True)

    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Default 16: ViT-Small needs more GPU memory than ResNet18.")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--image_size",   type=int,   default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--patience",     type=int,   default=4)

    # Loss weights — fixed at 1.0/2.0 as defaults to match the experiment.
    # Can still be overridden from command line if needed.
    parser.add_argument("--lambda_fake",      type=float, default=1.0,
                        help="Weight of the real/fake loss (default 1.0).")
    parser.add_argument("--lambda_transform", type=float, default=2.0,
                        help="Weight of the transformation loss (default 2.0).")
    parser.add_argument("--loss_weighting",   type=str,   default="manual",
                        choices=["manual", "learned"])

    parser.add_argument("--vit_model_name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_pretrained",   action="store_true")

    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_vit_rgb_multitask_1_2.pt")

    args = parser.parse_args()

    # Task is always multitask for this script.
    task = "multitask"

    print(f"Task:           {task}")
    print(f"ViT model:      {args.vit_model_name}")
    print(f"lambda_fake:    {args.lambda_fake}")
    print(f"lambda_transform: {args.lambda_transform}")
    print(f"Loss weighting: {args.loss_weighting}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:   {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RRDatasetFromCSV(
        csv_path=args.train_csv,
        image_root=args.image_root,
        transform=build_train_transform(args.image_size),
    )
    val_dataset = RRDatasetFromCSV(
        csv_path=args.val_csv,
        image_root=args.image_root,
        transform=build_eval_transform(args.image_size),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = ViTRGBMultiTaskModel(
        task=task,
        num_transform_classes=3,
        pretrained=not args.no_pretrained,
        vit_model_name=args.vit_model_name,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    adaptive_loss = None
    if args.loss_weighting == "learned":
        adaptive_loss = UncertaintyWeightedLoss().to(device)

    parameters = list(model.parameters())
    if adaptive_loss is not None:
        parameters += list(adaptive_loss.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    best_val_score             = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, task=task,
            lambda_fake=args.lambda_fake, lambda_transform=args.lambda_transform,
            adaptive_loss=adaptive_loss,
        )
        val_metrics = evaluate(
            model=model, loader=val_loader, device=device, task=task,
            lambda_fake=args.lambda_fake, lambda_transform=args.lambda_transform,
            adaptive_loss=adaptive_loss,
        )

        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        if adaptive_loss is not None:
            print(f"Learned weights (train): fake={train_metrics['weight_fake']:.4f}, "
                  f"transform={train_metrics['weight_transform']:.4f}")
            print(f"Learned weights (val):   fake={val_metrics['weight_fake']:.4f}, "
                  f"transform={val_metrics['weight_transform']:.4f}")

        val_score = compute_validation_score(val_metrics=val_metrics, task=task)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Val score:     {val_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

        if val_score > best_val_score:
            best_val_score             = val_score
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / args.checkpoint_name
            torch.save({
                "model_state_dict": model.state_dict(),
                "adaptive_loss_state_dict": (
                    adaptive_loss.state_dict() if adaptive_loss is not None else None
                ),
                "epoch":       epoch,
                "val_metrics": val_metrics,
                "val_score":   val_score,
                "task":        task,
                "args":        vars(args),
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{args.patience} epochs")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
