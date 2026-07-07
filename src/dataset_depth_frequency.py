import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_01(array):

    array = array.astype(np.float32)
    min_value = array.min()
    max_value = array.max()

    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    return (array - min_value) / (max_value - min_value)


def find_depth_path(depth_root, image_rel_path):
    """
    Find the depth map corresponding to an RGB image.

    The expected structure is:
        RGB image:
            image_root/original/real/example.jpg

        Depth map:
            depth_root/original/real/example.npy
    """

    depth_path = Path(depth_root) / Path(image_rel_path).with_suffix(".npy")

    if not depth_path.exists():
        raise FileNotFoundError(
            f"Depth map not found: {depth_path}\n"
            "The depth map must have the same relative path as the RGB image, "
            "but with .npy extension."
        )

    return depth_path


def rgb_to_grayscale(rgb_tensor):
    """
    Convert an RGB tensor in [0, 1] to a grayscale tensor.
    """

    red = rgb_tensor[0:1]
    green = rgb_tensor[1:2]
    blue = rgb_tensor[2:3]

    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    return gray


def compute_frequency_map(rgb_tensor):
    """
    Compute a simple frequency magnitude map from an RGB image.

    The input RGB tensor must be in [0, 1].
    The output is a single-channel map in [0, 1].

    This map is designed to capture high-frequency patterns, compression traces,
    and possible forensic artifacts.
    """

    gray = rgb_to_grayscale(rgb_tensor).squeeze(0)

    fft = torch.fft.fft2(gray)

    fft_shifted = torch.fft.fftshift(fft)

    # Use log magnitude for numerical stability and better visualization.
    magnitude = torch.log1p(torch.abs(fft_shifted))

    min_value = magnitude.min()
    max_value = magnitude.max()

    if max_value - min_value < 1e-8:
        frequency_map = torch.zeros_like(magnitude)
    else:
        frequency_map = (magnitude - min_value) / (max_value - min_value)

    return frequency_map.unsqueeze(0).float()


class RRDepthFrequencyDatasetFromCSV(Dataset):
    """
    Dataset for RGB + depth + frequency training.

    Each sample contains:
    1. RGB image.
    2. Precomputed depth map.
    3. Frequency map computed from the RGB image.
    4. Real/fake label.
    5. Transformation label.
    """

    def __init__(
        self,
        csv_path,
        image_root,
        depth_root,
        image_size=224,
        train=True,
    ):
        self.data = pd.read_csv(csv_path)

        self.image_root = Path(image_root)
        self.depth_root = Path(depth_root)
        self.image_size = image_size
        self.train = train

        required_columns = {"image_path", "fake_label", "transform_label"}

        missing_columns = required_columns - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"CSV file is missing required columns: {missing_columns}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Load one sample from the dataset.

        The frequency map is computed after resizing and augmentation,
        so it stays aligned with the RGB image.
        """

        row = self.data.iloc[index]

        image_rel_path = Path(str(row["image_path"]))
        image_path = self.image_root / image_rel_path
        depth_path = find_depth_path(self.depth_root, image_rel_path)

        if not image_path.exists():
            raise FileNotFoundError(f"RGB image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        depth_array = np.load(depth_path)
        depth_array = normalize_01(depth_array)

        depth_image = Image.fromarray(
            (depth_array * 255).astype(np.uint8),
            mode="L",
        )

        # Resize RGB and depth using the same spatial size.
        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.BILINEAR,
        )

        depth_image = depth_image.resize(
            (self.image_size, self.image_size),
            resample=Image.BILINEAR,
        )

        # Apply the same horizontal flip to RGB and depth during training.
        if self.train and random.random() < 0.5:
            image = ImageOps.mirror(image)
            depth_image = ImageOps.mirror(depth_image)

        rgb_raw = TF.to_tensor(image)

        frequency = compute_frequency_map(rgb_raw)

        # Normalize for ResNet.
        image_tensor = TF.normalize(
            rgb_raw,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        # Convert depth image to tensor in [0, 1].
        depth_tensor = TF.to_tensor(depth_image)

        fake_label = int(row["fake_label"])
        transform_label = int(row["transform_label"])

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "frequency": frequency,
            "fake_label": torch.tensor(fake_label, dtype=torch.long),
            "transform_label": torch.tensor(transform_label, dtype=torch.long),
            "image_path": str(image_rel_path),
        }