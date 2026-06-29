from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# ImageNet normalization for the RGB backbone (pretrained on ImageNet).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def compute_fft_map(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the log-magnitude FFT spectrum of an RGB image.

    Steps:
        1. Convert RGB to grayscale.
        2. Apply 2D FFT.
        3. Shift the zero-frequency component to the center.
        4. Compute the log of the magnitude: log(1 + |FFT|).
        5. Normalize to [0, 1].

    Why log-magnitude?
        AI-generated images often contain periodic artifacts in the
        frequency domain. The log-magnitude spectrum makes these
        artifacts visible because it compresses the large dynamic
        range of the FFT output (which spans several orders of magnitude).

    Args:
        rgb_tensor: Tensor [3, H, W] with values in [0, 1].

    Returns:
        Tensor [1, H, W] — normalized log-magnitude FFT spectrum.
    """

    # Step 1: Convert RGB to grayscale using standard luminance weights.
    # Output shape: [H, W]
    gray = (
        0.299 * rgb_tensor[0]
        + 0.587 * rgb_tensor[1]
        + 0.114 * rgb_tensor[2]
    )

    # Step 2: Compute the 2D FFT.
    # torch.fft.fft2 operates on the last two dimensions.
    fft = torch.fft.fft2(gray)

    # Step 3: Shift zero frequency to the center of the spectrum.
    # Without fftshift, low frequencies are at the corners.
    fft_shifted = torch.fft.fftshift(fft)

    # Step 4: Log-magnitude. Adding 1 avoids log(0).
    log_magnitude = torch.log(1.0 + torch.abs(fft_shifted))

    # Step 5: Normalize to [0, 1].
    min_val = log_magnitude.min()
    max_val = log_magnitude.max()

    if max_val - min_val > 1e-8:
        log_magnitude = (log_magnitude - min_val) / (max_val - min_val)
    else:
        log_magnitude = torch.zeros_like(log_magnitude)

    # Add channel dimension: [1, H, W]
    return log_magnitude.unsqueeze(0)


class RRFreqDatasetFromCSV(Dataset):
    """
    Dataset for RGB + FFT log-magnitude spectrum.

    Uses the same CSV format as RRDatasetFromCSV and RRGeometricDatasetFromCSV:

        CSV columns required:
            image_path      — relative path from image_root
            fake_label      — 0 = real, 1 = AI-generated
            transform_label — 0 = original, 1 = internet-transmitted, 2 = re-digitized

    The FFT map is computed on-the-fly during loading.
    No precomputation step is needed (unlike depth maps).

    Returns per sample:
        {
            "image":           RGB tensor [3, H, W]   — ImageNet-normalized
            "freq_map":        FFT tensor [1, H, W]   — log-magnitude, [0, 1]
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
            image_size: Size to resize images to (square).
            train:      If True, apply random horizontal flip augmentation.
        """
        self.csv_path   = Path(csv_path)
        self.image_root = Path(image_root)
        self.image_size = image_size
        self.train      = train

        # Load CSV exactly like the other datasets in this project.
        self.data = pd.read_csv(self.csv_path)

        required_columns = {"image_path", "fake_label", "transform_label"}
        missing_columns  = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in CSV: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Load one sample.

        Returns a dictionary containing the RGB tensor, the FFT map,
        both labels, and the image path for debugging.
        """

        row = self.data.iloc[index]

        # Build the full image path from root + relative path in CSV.
        image_path = self.image_root / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load RGB image.
        image = Image.open(image_path).convert("RGB")

        # Resize to the target size.
        image = TF.resize(image, [self.image_size, self.image_size])

        # Random horizontal flip during training.
        # The same flip is applied to both RGB and the FFT map
        # because FFT is computed after flipping.
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)

        # Convert PIL image to tensor with values in [0, 1].
        # Shape: [3, H, W]
        rgb_raw = TF.to_tensor(image)

        # Compute the FFT log-magnitude spectrum from the raw RGB tensor.
        # Shape: [1, H, W]
        freq_map = compute_fft_map(rgb_raw)

        # Normalize RGB for the pretrained ResNet backbone.
        image_tensor = (rgb_raw - IMAGENET_MEAN) / IMAGENET_STD

        # Convert CSV labels to PyTorch tensors.
        fake_label      = torch.tensor(int(row["fake_label"]),      dtype=torch.long)
        transform_label = torch.tensor(int(row["transform_label"]), dtype=torch.long)

        return {
            "image":           image_tensor,
            "freq_map":        freq_map,
            "fake_label":      fake_label,
            "transform_label": transform_label,
            "image_path":      str(image_path),
        }
