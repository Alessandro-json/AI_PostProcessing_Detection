import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RRDatasetFromCSV, build_train_transform, build_eval_transform
from model import RGBMultiTaskModel


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_fake: float = 1.0,
    lambda_transform: float = 1.0,
):
    """
    Train the model for one epoch.

    Args:
        model: The neural network we want to train.
        loader: PyTorch DataLoader that provides batches of training data.
        optimizer: Algorithm that updates the model weights.
        device: "cuda" if GPU is available, otherwise "cpu".
        lambda_fake: Weight of the real/fake classification loss.
        lambda_transform: Weight of the transformation classification loss.

    Returns:
        A dictionary with training loss and accuracies.
    """

    model.train()

    # CrossEntropyLoss is used for classification tasks.
    #
    # fake_criterion:
    #   task 1: real vs AI-generated
    #
    # transform_criterion:
    #   task 2: original vs transmitted vs re-digitized
    fake_criterion = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    for batch in tqdm(loader, desc="Training"):

        # Move images and labels to the selected device.
        # If we have a GPU, this sends the tensors to the GPU.
        images = batch["image"].to(device)
        fake_labels = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # Forward pass:
        # send images through the model and get predictions.
        outputs = model(images)

        # Compute the loss for the real/fake task.
        fake_loss = fake_criterion(
            outputs["fake_logits"],
            fake_labels,
        )

        # Compute the loss for the transformation task.
        transform_loss = transform_criterion(
            outputs["transform_logits"],
            transform_labels,
        )

        # Multi-task loss:
        # weighted sum of the two losses.
        loss = lambda_fake * fake_loss + lambda_transform * transform_loss

        # Reset old gradients.
        optimizer.zero_grad()

        # Backpropagation:
        # compute gradients of the loss with respect to model parameters.
        loss.backward()

        # Update model weights using the computed gradients.
        optimizer.step()

        # Number of images in the current batch.
        batch_size = images.size(0)

        # Accumulate total loss.
        total_loss += loss.item() * batch_size

        # Convert logits into predicted classes.
        fake_pred = outputs["fake_logits"].argmax(dim=1)
        transform_pred = outputs["transform_logits"].argmax(dim=1)

        # Count correct predictions for both tasks.
        correct_fake += (fake_pred == fake_labels).sum().item()
        correct_transform += (transform_pred == transform_labels).sum().item()

        # Count total number of processed images.
        total += batch_size

    # Return average loss and accuracy values.
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
    Evaluate the model on validation data.

    Args:
        model: The trained model.
        loader: DataLoader for validation data.
        device: "cuda" or "cpu".
        lambda_fake: Weight of the real/fake loss.
        lambda_transform: Weight of the transformation loss.

    Returns:
        A dictionary with validation loss and accuracies.
    """

    model.eval()

    fake_criterion = nn.CrossEntropyLoss()
    transform_criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_fake = 0
    correct_transform = 0
    total = 0

    for batch in tqdm(loader, desc="Validation"):

        images = batch["image"].to(device)
        fake_labels = batch["fake_label"].to(device)
        transform_labels = batch["transform_label"].to(device)

        # Forward pass only.
        # Because of @torch.no_grad(), PyTorch does not store gradients here.
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
    Main training function.

    This function:
        1. Reads command-line arguments.
        2. Creates datasets and dataloaders.
        3. Creates the model.
        4. Creates the optimizer.
        5. Runs training and validation.
        6. Saves the best checkpoint.
    """

    # argparse allows us to pass parameters from the terminal.
    #
    # Example:
    # python src/train.py --train_csv data/splits/train.csv --val_csv data/splits/val.csv ...
    parser = argparse.ArgumentParser()

    # Paths to CSV files and image folder.
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)

    # Multi-task loss weights.
    parser.add_argument("--lambda_fake", type=float, default=1.0)
    parser.add_argument("--lambda_transform", type=float, default=1.0)

    # Folder where checkpoints will be saved.
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Read all arguments from the command line.
    args = parser.parse_args()

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
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Validation loader does not need shuffling.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Create the RGB multi-task baseline model.
    model = RGBMultiTaskModel(
        num_transform_classes=3,
        pretrained=True,
    )

    # Move model to GPU or CPU.
    model = model.to(device)

    # AdamW optimizer updates the model weights.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # We save the checkpoint with the best validation fake accuracy.
    best_val_fake_acc = 0.0

    # Main training loop.
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train for one epoch.
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
        )

        # Validate after the epoch.
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lambda_fake=args.lambda_fake,
            lambda_transform=args.lambda_transform,
        )

        # Print results.
        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        # Save the model only if fake accuracy improved.
        if val_metrics["fake_acc"] > best_val_fake_acc:
            best_val_fake_acc = val_metrics["fake_acc"]

            checkpoint_path = checkpoint_dir / "best_rgb_baseline.pt"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()