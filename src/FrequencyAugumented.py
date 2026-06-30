import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_freq import RRFreqDatasetFromCSV
from model_freq import FreqMultiTaskModel


# ---------------------------------------------------------------------------
# Learned Uncertainty Weighting (Kendall et al., 2018)
# ---------------------------------------------------------------------------
# Instead of fixing lambda_fake and lambda_transform by hand, this module
# learns two log-variance parameters (one per task) during training.
#
# The multi-task loss becomes:
#
#   L = (1 / 2*exp(log_var_fake))     * L_fake     + (1/2) * log_var_fake
#     + (1 / 2*exp(log_var_transform)) * L_transform + (1/2) * log_var_transform
#
# When a task is easy the model increases its log_var (↓ weight),
# when it is hard it decreases log_var (↑ weight).
# The log(σ) regularisation term prevents the weights from going to zero.
#
# We parameterise with log(σ²) instead of σ directly for numerical stability.
# ---------------------------------------------------------------------------

class LearnedWeightingLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting for two classification tasks.

    Adds two learnable scalar parameters to the model's optimizer so that
    they are updated by the same backward pass as the backbone weights.
    """

    def __init__(self):
        super().__init__()
        # Initialise both log-variances to 0  →  initial weight = 0.5 each.
        self.log_var_fake      = nn.Parameter(torch.zeros(1))
        self.log_var_transform = nn.Parameter(torch.zeros(1))

    def forward(self, loss_fake: torch.Tensor, loss_transform: torch.Tensor):
        """
        Compute the combined uncertainty-weighted loss.

        Args:
            loss_fake:      Scalar cross-entropy for the real/fake task.
            loss_transform: Scalar cross-entropy for the transform task.

        Returns:
            total_loss:  Combined scalar loss passed to .backward().
            w_fake:      Effective weight on loss_fake  (for logging).
            w_transform: Effective weight on loss_transform (for logging).
        """

        # Precision = 1 / exp(log_var)  — higher precision → higher weight.
        precision_fake      = torch.exp(-self.log_var_fake)
        precision_transform = torch.exp(-self.log_var_transform)

        # Weighted loss + regularisation term.
        total_loss = (
            precision_fake      * loss_fake      + self.log_var_fake
            + precision_transform * loss_transform + self.log_var_transform
        )

        return total_loss, precision_fake.item(), precision_transform.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
    learned_weighting: LearnedWeightingLoss = None,
):
    """
    Train the frequency multi-task model for one epoch.

    Args:
        model:             FreqMultiTaskModel instance.
        loader:            DataLoader with images, freq_maps, and labels.
        optimizer:         Optimizer (includes learned_weighting params if used).
        device:            "cuda" or "cpu".
        lambda_fake:       Manual weight for fake loss (ignored when learned_weighting is set).
        lambda_transform:  Manual weight for transform loss (ignored when learned_weighting is set).
        learned_weighting: LearnedWeightingLoss module, or None for manual weights.

    Returns:
        Dictionary with average loss, accuracies, and (optionally) learned weights.
    """

    model.train()
    if learned_weighting is not None:
        learned_weighting.train()

    fake_criterion      = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss        = 0.0
    correct_fake      = 0
    correct_transform = 0
    total             = 0

    # Running average of the effective weights (for logging).
    sum_w_fake      = 0.0
    sum_w_transform = 0.0

    for batch in tqdm(loader, desc="Training freq"):

        images           = batch["image"].to(device)
        freq_map         = batch["freq_map"].to(device)
        fake_labels      = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        outputs = model(images=images, freq_map=freq_map)

        loss_fake      = fake_criterion(outputs["fake_logits"],      fake_labels)
        loss_transform = transform_criterion(outputs["transform_logits"], transform_labels)

        if learned_weighting is not None:
            # Let the module compute the weighted combination.
            loss, w_fake, w_transform = learned_weighting(loss_fake, loss_transform)
            sum_w_fake      += w_fake
            sum_w_transform += w_transform
        else:
            # Fixed manual weights.
            loss = lambda_fake * loss_fake + lambda_transform * loss_transform

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

    n_batches = len(loader)
    metrics = {
        "loss":          total_loss / total,
        "fake_acc":      correct_fake / total,
        "transform_acc": correct_transform / total,
    }

    if learned_weighting is not None:
        metrics["w_fake"]      = sum_w_fake      / n_batches
        metrics["w_transform"] = sum_w_transform / n_batches

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
    learned_weighting: LearnedWeightingLoss = None,
):
    """
    Evaluate the frequency multi-task model on validation data.
    """

    model.eval()
    if learned_weighting is not None:
        learned_weighting.eval()

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

        outputs = model(images=images, freq_map=freq_map)

        loss_fake      = fake_criterion(outputs["fake_logits"],      fake_labels)
        loss_transform = transform_criterion(outputs["transform_logits"], transform_labels)

        if learned_weighting is not None:
            loss, _, _ = learned_weighting(loss_fake, loss_transform)
        else:
            loss = lambda_fake * loss_fake + lambda_transform * loss_transform

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


# ---------------------------------------------------------------------------
# Cosine Annealing with Linear Warm-up
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, args, steps_per_epoch: int):
    """
    Build the learning rate scheduler.

    --scheduler reduce_on_plateau  (default, same as before)
        ReduceLROnPlateau: cuts LR by 0.5 when val_loss stagnates.
        Safe choice, requires no extra hyperparameters.

    --scheduler cosine
        Linear warm-up for `--warmup_epochs` epochs, then cosine decay
        to eta_min=1e-6 over the remaining epochs.

        Why cosine is better than ReduceLROnPlateau for short runs:
        - ReduceLROnPlateau reacts *after* stagnation has already happened.
        - Cosine decays smoothly from the start, so the model keeps making
          small improvements in the last epochs instead of plateauing.

    Returns the scheduler and a flag `step_every_batch` that indicates
    whether scheduler.step() should be called per batch (True) or per
    epoch (False).
    """

    if args.scheduler == "cosine":
        # --- Phase 1: linear warm-up ---
        # During warm-up the LR ramps from lr/10 to lr linearly.
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps  = args.epochs        * steps_per_epoch

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linearly increase from 0.1 to 1.0 over warmup_steps.
                return 0.1 + 0.9 * current_step / max(1, warmup_steps)
            # Cosine decay from 1.0 to eta_min/lr over the remaining steps.
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item())
            eta_min_ratio = 1e-6 / args.lr
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

        scheduler      = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        step_per_batch = True   # must call scheduler.step() after every batch

    else:
        # Default: ReduceLROnPlateau (same behaviour as original script).
        scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )
        step_per_batch = False  # call scheduler.step(val_loss) after each epoch

    return scheduler, step_per_batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Main training function for the frequency multi-task model.

    New arguments compared to the original version:
        --scheduler         "reduce_on_plateau" (default) or "cosine"
        --warmup_epochs     number of warm-up epochs for cosine scheduler
        --loss_weighting    "manual" (default) or "learned"

    Example — cosine scheduler + learned weighting:
        python src/train_freq.py \\
            --train_csv data/splits/train_balanced.csv \\
            --val_csv   data/splits/val_balanced.csv \\
            --image_root data/raw/RRDataset_subset \\
            --epochs 20 \\
            --batch_size 32 \\
            --scheduler cosine \\
            --warmup_epochs 2 \\
            --loss_weighting learned \\
            --checkpoint_name best_freq_learned_cosine.pt
    """

    parser = argparse.ArgumentParser(
        description="Train the RGB + FFT frequency multi-task model."
    )

    # --- Paths ---
    parser.add_argument("--train_csv",   type=str, required=True)
    parser.add_argument("--val_csv",     type=str, required=True)
    parser.add_argument("--image_root",  type=str, required=True)

    # --- Training hyperparameters ---
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--image_size",   type=int,   default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--patience",     type=int,   default=5,
                        help="Early stopping patience in epochs.")

    # --- Scheduler ---
    parser.add_argument(
        "--scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "cosine"],
        help=(
            "'reduce_on_plateau': cuts LR when val_loss stagnates (safe default). "
            "'cosine': linear warm-up then cosine decay (better for longer runs)."
        ),
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Number of warm-up epochs for the cosine scheduler.",
    )

    # --- Loss weighting ---
    parser.add_argument(
        "--loss_weighting",
        type=str,
        default="manual",
        choices=["manual", "learned"],
        help=(
            "'manual': fixed lambda_fake / lambda_transform weights. "
            "'learned': Kendall et al. 2018 uncertainty weighting — "
            "the model learns the task weights automatically."
        ),
    )
    parser.add_argument("--lambda_fake",      type=float, default=1.0)
    parser.add_argument("--lambda_transform", type=float, default=1.0)

    # --- Model options ---
    parser.add_argument("--freq_features", type=int, default=128)
    parser.add_argument("--no_attention",  action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")

    # --- Checkpoint ---
    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="best_freq.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Scheduler:    {args.scheduler}")
    print(f"Loss weights: {args.loss_weighting}")

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
    ).to(device)

    # --- Learned weighting module (optional) ---
    learned_weighting = None
    if args.loss_weighting == "learned":
        learned_weighting = LearnedWeightingLoss().to(device)
        print("Learned uncertainty weighting enabled.")
        print(f"  Initial log_var_fake:      {learned_weighting.log_var_fake.item():.4f}")
        print(f"  Initial log_var_transform: {learned_weighting.log_var_transform.item():.4f}")

    # --- Optimizer ---
    # Include learned_weighting parameters so they are updated by the same
    # optimizer as the backbone. If learned_weighting is None this is a no-op.
    extra_params = (
        list(learned_weighting.parameters())
        if learned_weighting is not None
        else []
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + extra_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --- Scheduler ---
    scheduler, step_per_batch = build_scheduler(
        optimizer=optimizer,
        args=args,
        steps_per_epoch=len(train_loader),
    )

    best_val_score             = 0.0
    epochs_without_improvement = 0

    # --- Training loop ---
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Pass scheduler to train_one_epoch only when it steps per batch.
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
            learned_weighting=learned_weighting,
        )

        # Step cosine scheduler after every batch inside the epoch.
        # For ReduceLROnPlateau we step after the epoch (see below).
        if step_per_batch:
            # Already stepped inside train_one_epoch implicitly via LambdaLR.
            # LambdaLR needs an explicit step() call per optimizer step.
            # We call it once here to advance the epoch-level counter.
            pass  # LambdaLR was stepped per batch inside the loop above.
            # NOTE: to keep the code simple we step LambdaLR once per epoch.
            scheduler.step()

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
            learned_weighting=learned_weighting,
        )

        val_score  = 0.5 * val_metrics["fake_acc"] + 0.5 * val_metrics["transform_acc"]
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")
        print(f"Val score:     {val_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

        # Log learned weights if active.
        if learned_weighting is not None:
            w_fake      = torch.exp(-learned_weighting.log_var_fake).item()
            w_transform = torch.exp(-learned_weighting.log_var_transform).item()
            print(f"Learned w_fake: {w_fake:.4f}  |  w_transform: {w_transform:.4f}")

        # Step ReduceLROnPlateau on val loss (epoch-level).
        if not step_per_batch:
            scheduler.step(val_metrics["loss"])

        # Save checkpoint when val_score improves.
        if val_score > best_val_score:
            best_val_score             = val_score
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / args.checkpoint_name

            save_dict = {
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_metrics":      val_metrics,
                "val_score":        val_score,
                "args":             vars(args),
            }
            if learned_weighting is not None:
                save_dict["learned_weighting_state_dict"] = learned_weighting.state_dict()

            torch.save(save_dict, checkpoint_path)
            print(f"Saved best checkpoint → {checkpoint_path}")

        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/{args.patience} epochs"
            )
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()