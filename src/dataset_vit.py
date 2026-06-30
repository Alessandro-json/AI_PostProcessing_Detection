from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_train_transform(image_size: int = 224):
    """
    Transformations used during training to prepare images for the ViT.

    Identical to build_train_transform() in dataset.py: ViT-Small pretrained
    on ImageNet-1k via timm expects the same normalization as torchvision's
    ImageNet-pretrained CNNs (mean/std below), so no special handling
    is needed here compared to the RGB baseline.

    Image size must stay at 224 (or another multiple of the 16x16 patch
    size) for vit_small_patch16_224.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_eval_transform(image_size: int = 224):
    """
    Transformations used during validation and testing.
    No augmentation, same normalization as training.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class RRViTDatasetFromCSV(Dataset):
    """
    Dataset for the ViT multi-task model.

    Uses the exact same CSV format as RRDatasetFromCSV in dataset.py:

        CSV columns required:
            image_path      — relative path from image_root
            fake_label      — 0 = real, 1 = AI-generated
            transform_label — 0 = original, 1 = internet-transmitted, 2 = re-digitized

    Returns per sample:
        {
            "image":           RGB tensor [3, H, W] — ImageNet-normalized
            "fake_label":      LongTensor scalar
            "transform_label": LongTensor scalar
            "image_path":      str — full path, useful for debugging
        }
    """

    def __init__(
        self,
        csv_path,
        image_root,
        image_size: int = 224,
        train: bool = True,
    ):
        """
        Args:
            csv_path:   Path to the CSV file.
            image_root: Root folder where images are stored.
            image_size: Size to resize images to (square). Keep at 224
                        for vit_small_patch16_224.
            train:      If True, apply training transform (with augmentation).
                        If False, apply eval transform (no augmentation).
        """
        self.csv_path   = Path(csv_path)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.train      = train

        self.data = pd.read_csv(self.csv_path)

        required_columns = {"image_path", "fake_label", "transform_label"}
        missing_columns  = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in CSV: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )

        # Build the appropriate transform pipeline once, not per sample.
        if self.train:
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_eval_transform(image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Load one sample: image + both labels.
        """

        row = self.data.iloc[index]

        image_path = self.image_root / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        fake_label      = torch.tensor(int(row["fake_label"]),      dtype=torch.long)
        transform_label = torch.tensor(int(row["transform_label"]), dtype=torch.long)

        return {
            "image":           image_tensor,
            "fake_label":      fake_label,
            "transform_label": transform_label,
            "image_path":      str(image_path),
        }
