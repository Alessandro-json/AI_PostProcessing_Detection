import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RRDatasetFromCSV, build_train_transform, build_eval_transform
from model_RGB import RGBMultiTaskModel
from loss import UncertaintyWeightedLoss


def compute_loss(outputs, batch, task, criterion, lambda_fake, lambda_transform, device, adaptive_loss=None):
    """
    Compute the training or validation loss depending on the selected task.
    """

    # Start from empty and add only the losses required by the selected task.
    loss_info = {}

    # Real/fake classification loss.
    if task in ["fake", "multitask"]:
        fake_labels = batch["fake_label"].to(device)
        fake_loss = criterion(
            outputs["fake_logits"],
            fake_labels,
        )

        loss_info["fake_loss"] = fake_loss.item()

    # Transformation classification loss.
    if task in ["transform", "multitask"]:
        transform_labels = batch["transform_label"].to(device)
        transform_loss = criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        loss_info["transform_loss"] = transform_loss.item()
 
    # Single-task real/fake baseline.
    if task == "fake":
        loss = fake_loss

    # Single-task transformation baseline.
    elif task == "transform":
        loss = transform_loss

    # Multi-task baseline.
    elif task == "multitask":
        # Manual fixed loss weighting.
        if adaptive_loss is None:
            loss = lambda_fake * fake_loss + lambda_transform * transform_loss

            loss_info["weight_fake"] = lambda_fake
            loss_info["weight_transform"] = lambda_transform
        
        # Learned uncertainty weighting.
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
    """
    Compute the score.

    For single-task experiments:
        the score is the validation accuracy of that task.

    For multi-task experiments:
        the score is the average between real/fake accuracy and transformation accuracy.
    """

    if task == "fake":
        return val_metrics["fake_acc"]

    if task == "transform":
        return val_metrics["transform_acc"]

    if task == "multitask":
        return 0.5 * val_metrics["fake_acc"] + 0.5 * val_metrics["transform_acc"]

    raise ValueError(f"Unknown task: {task}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    task,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
    adaptive_loss=None,
):
    """
    Train the model for one epoch.

    Args:
        model: The neural network we want to train.
        loader: PyTorch DataLoader that provides batches of training data.
        optimizer: Algorithm that updates the model weights.
        device: "cuda" if GPU is available, otherwise "cpu".
        task: "fake", "transform" or "multitask".
        lambda_fake: Weight of the real/fake classification loss.
        lambda_transform: Weight of the transformation classification loss.

    Returns:
        A dictionary with training loss and accuracies.
    """

    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total_samples = 0

    # These variables are used only when learned uncertainty weighting is active.
    # They store the average learned weights over the epoch.
    weight_fake_sum = 0.0
    weight_transform_sum = 0.0
    weight_count = 0

    for batch in tqdm(loader, desc="Training"):

        # Move images and labels to the selected device.
        # If we have a GPU, this sends the tensors to the GPU.
        images = batch["image"].to(device)

        # Forward pass:
        # send images through the model and get predictions.
        outputs = model(images)

        # Compute the correct loss for the selected task.
        loss, loss_info = compute_loss(
            outputs=outputs,
            batch=batch,
            task=task,
            criterion=criterion,
            lambda_fake=lambda_fake,
            lambda_transform=lambda_transform,
            device=device,
            adaptive_loss=adaptive_loss,
        )

        # Store learned weights for epoch-level logging.
        # This is useful only when uncertainty weighting is active.
        if adaptive_loss is not None:
            weight_fake_sum += loss_info["weight_fake"]
            weight_transform_sum += loss_info["weight_transform"]
            weight_count += 1

        # Reset old gradients.
        optimizer.zero_grad()

        # Backpropagation:
        # compute gradients of the loss with respect to model parameters.
        loss.backward()

        # Update model weights using the computed gradients.
        optimizer.step()

        # Number of images in the current batch.
        batch_size = images.size(0)

        # Accumulate weighted loss.
        # Multiplying by batch_size allows us to compute the average loss correctly.
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Compute real/fake accuracy when this task is active.
        if task in ["fake", "multitask"]:
            fake_labels = batch["fake_label"].to(device)
            fake_pred = outputs["fake_logits"].argmax(dim=1)

            correct_fake += (fake_pred == fake_labels).sum().item()

        # Compute transformation accuracy when this task is active.
        if task in ["transform", "multitask"]:
            transform_labels = batch["transform_label"].to(device)
            transform_pred = outputs["transform_logits"].argmax(dim=1)

            correct_transform += (transform_pred == transform_labels).sum().item()

    # Build metrics dictionary.
    metrics = {
        "loss": total_loss / total_samples,
    }

    if task in ["fake", "multitask"]:
        metrics["fake_acc"] = correct_fake / total_samples

    if task in ["transform", "multitask"]:
        metrics["transform_acc"] = correct_transform / total_samples

    # Add average learned weights to the metrics dictionary.
    if adaptive_loss is not None and weight_count > 0:
        metrics["weight_fake"] = weight_fake_sum / weight_count
        metrics["weight_transform"] = weight_transform_sum / weight_count
    
    return metrics



@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    task,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
    adaptive_loss=None,
):
    """
    Evaluate the model on validation data.

    Args:
        model: The trained model.
        loader: DataLoader for validation data.
        device: "cuda" or "cpu".
        task: "fake", "transform" or "multitask".
        lambda_fake: Weight of the real/fake loss.
        lambda_transform: Weight of the transformation loss.
        adaptive_loss: "None" if the weight are added manually.

    Returns:
        A dictionary with validation loss and accuracies.
    """

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    correct_fake = 0
    correct_transform = 0

    # These variables are used only when learned uncertainty weighting is active.
    weight_fake_sum = 0.0
    weight_transform_sum = 0.0
    weight_count = 0

    for batch in tqdm(loader, desc="Validation"):
        # Move input images to the selected device.
        images = batch["image"].to(device)

        # Forward pass without gradient computation.
        outputs = model(images)

        # Compute validation loss for the selected task.
        loss, loss_info = compute_loss(
            outputs=outputs,
            batch=batch,
            task=task,
            criterion=criterion,
            lambda_fake=lambda_fake,
            lambda_transform=lambda_transform,
            device=device,
            adaptive_loss=adaptive_loss,
        )
        
        # Store learned weights for validation logging.
        if adaptive_loss is not None:
            weight_fake_sum += loss_info["weight_fake"]
            weight_transform_sum += loss_info["weight_transform"]
            weight_count += 1

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Real/fake validation accuracy.
        if task in ["fake", "multitask"]:
            fake_labels = batch["fake_label"].to(device)
            fake_pred = outputs["fake_logits"].argmax(dim=1)

            correct_fake += (fake_pred == fake_labels).sum().item()

        # Transformation validation accuracy.
        if task in ["transform", "multitask"]:
            transform_labels = batch["transform_label"].to(device)
            transform_pred = outputs["transform_logits"].argmax(dim=1)

            correct_transform += (transform_pred == transform_labels).sum().item()

    # Build metrics dictionary.
    metrics = {
        "loss": total_loss / total_samples,
    }

    if task in ["fake", "multitask"]:
        metrics["fake_acc"] = correct_fake / total_samples

    if task in ["transform", "multitask"]:
        metrics["transform_acc"] = correct_transform / total_samples

    # Add average learned weights to the metrics dictionary.
    if adaptive_loss is not None and weight_count > 0:
        metrics["weight_fake"] = weight_fake_sum / weight_count
        metrics["weight_transform"] = weight_transform_sum / weight_count

    return metrics


def parse_args():
    """
    Read command-line arguments.

    This script supports three experiments with the same code:
        1. fake-only baseline
        2. transformation-only baseline
        3. joint multi-task baseline
    """

    parser = argparse.ArgumentParser(
        description="Train RGB baselines for Project 2."
    )

    # Task selection.
    parser.add_argument(
        "--task",
        type=str,
        default="multitask",
        choices=["fake", "transform", "multitask"],
        help=(
            "Training task. "
            "'fake' trains only the real/fake head. "
            "'transform' trains only the transformation head. "
            "'multitask' trains both heads jointly."
        ),
    )

    # Dataset paths.
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to the training CSV file.",
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to the validation CSV file.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root folder containing the images.",
    )

    # Training hyperparameters.
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images per batch.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image size.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay used by AdamW.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience.",
    )

    # Loss weights.
    # These are both used only when task == "multitask".
    # In single-task training, only the relevant one affects the loss.
    parser.add_argument(
        "--lambda_fake",
        type=float,
        default=1.0,
        help="Loss weight for the real/fake task.",
    )

    parser.add_argument(
        "--lambda_transform",
        type=float,
        default=1.0,
        help="Loss weight for the transformation task.",
    )

    parser.add_argument(
        "--loss_weighting",
        type=str,
        default="manual",
        choices=["manual", "learned"],
        help=(
            "Loss weighting strategy for multi-task training. "
            "'manual' uses lambda_fake and lambda_transform. "
            "'learned' uses uncertainty-based learnable weighting."
        ),
    )

    # Checkpoint settings.
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Folder where checkpoints are saved.",
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_rgb_baseline.pt",
        help="Checkpoint filename.",
    )

    # Pretraining option.
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Train the backbone from scratch instead of using ImageNet weights.",
    )

    return parser.parse_args()


def main():
    """
    Main training function.

    This function:
        1. Read command-line arguments.
        2. Select GPU or CPU.
        3. Create datasets.
        4. Create DataLoaders.
        5. Build the model.
        6. Train and validate.
        7. Save the best checkpoint.
    """

    args = parse_args()
    print(f"Selected task: {args.task}")

    # Select GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint folder if it does not exist.
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create the training dataset.
    train_dataset = RRDatasetFromCSV(
        csv_path=args.train_csv,
        image_root=args.image_root,
        transform=build_train_transform(args.image_size),
    )

    # Create the validation dataset.
    val_dataset = RRDatasetFromCSV(
        csv_path=args.val_csv,
        image_root=args.image_root,
        transform=build_eval_transform(args.image_size),
    )

    # DataLoader creates batches from the dataset.
    #
    # shuffle=True for training:
    #   images are presented in random order at every epoch.
    #
    # num_workers:
    #   number of subprocesses used to load data.
    #
    # pin_memory:
    #   can speed up CPU-to-GPU transfer when using CUDA.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Validation loader does not need shuffling.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Create the RGB multi-task baseline model.
    model = RGBMultiTaskModel(
        task=args.task,
        num_transform_classes=3,
        pretrained=True,
    )

    # Move model to GPU or CPU.
    model = model.to(device)

    # Optional adaptive loss weighting.
    adaptive_loss = None

    if args.loss_weighting == "learned":
        if args.task != "multitask":
            raise ValueError(
                "Learned loss weighting can only be used with task='multitask'."
            )

        adaptive_loss = UncertaintyWeightedLoss().to(device)

    # AdamW optimizer updates the model weights.
    parameters = list(model.parameters())

    if adaptive_loss is not None:
        parameters += list(adaptive_loss.parameters())

    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler to lower the learning rate if validation loss doesn't improve. 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_score = 0.0
    epochs_without_improvement = 0

    # Main training loop.
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train for one epoch.
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            task=args.task,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
            adaptive_loss=adaptive_loss,
        )

        # Validate after the epoch.
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            task=args.task,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
            adaptive_loss=adaptive_loss,
        )

        # Print results.
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        # Print learned loss weights when uncertainty weighting is active.
        if adaptive_loss is not None:
            print(
                "Learned loss weights "
                f"(train): fake={train_metrics['weight_fake']:.4f}, "
                f"transform={train_metrics['weight_transform']:.4f}"
            )

            print(
                "Learned loss weights "
                f"(val):   fake={val_metrics['weight_fake']:.4f}, "
                f"transform={val_metrics['weight_transform']:.4f}"
            )

        # Select the correct validation score depending on the task.
        val_score = compute_validation_score(
            val_metrics=val_metrics,
            task=args.task,
        )

        # Update the scheduler using validation loss.
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        print(f"Val score: {val_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")

        # Save the model only if the combined validation score improved.
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_without_improvement = 0

            checkpoint_path = checkpoint_dir / args.checkpoint_name

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "adaptive_loss_state_dict": (
                        adaptive_loss.state_dict() if adaptive_loss is not None else None
                    ),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "val_score": val_score,
                    "task": args.task,
                    "args": vars(args),
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint to {checkpoint_path}")
        else:
             # Count epochs without improvement for early stopping.
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