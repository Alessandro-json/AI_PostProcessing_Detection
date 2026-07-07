from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_01(array: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array to the [0, 1] range.

    Different depth maps estimators may output
    different value ranges.
    """
    array = array.astype(np.float32)

    min_val = array.min()
    max_val = array.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    return (array - min_val) / (max_val - min_val)


def find_depth_path(depth_root: Path, image_rel_path: str) -> Path:
    """
    Find the depth map corresponding to an image.
    """
    rel_path = Path(image_rel_path)
    depth_path = depth_root / rel_path.with_suffix(".npy")

    if not depth_path.exists():
        raise FileNotFoundError(
            f"Depth map not found: {depth_path}\n"
            f"The depth map must have the same relative path as the RGB image, "
            f"but with .npy extension."
        )

    return depth_path


def rgb_to_grayscale(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image tensor to grayscale.
    """
    r = rgb_tensor[0:1]
    g = rgb_tensor[1:2]
    b = rgb_tensor[2:3]

    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_edges(single_channel_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute Sobel edge magnitude for a single-channel tensor.
    """
    if single_channel_tensor.dim() != 3 or single_channel_tensor.shape[0] != 1:
        raise ValueError("sobel_edges expects a tensor with shape [1, H, W].")

    x = single_channel_tensor.unsqueeze(0)

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0],
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)

    sobel_x = sobel_x.to(x.device)
    sobel_y = sobel_y.to(x.device)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    edges = edges.squeeze(0)

    max_val = edges.max()

    if max_val > 0:
        edges = edges / max_val

    return edges


class RRGeometricDatasetFromCSV(Dataset):
    """
    Dataset for RGB + depth + edge-depth consistency.

    Expected columns:
        image_path
        fake_label
        transform_label

    Returns:
        {
            "image": RGB tensor [3, H, W],
            "depth": depth tensor [1, H, W],
            "edge_consistency": edge-depth inconsistency tensor [1, H, W],
            "fake_label": real/fake label,
            "transform_label": transformation label,
            "image_path": full image path
        }
    """

    def __init__(
        self,
        csv_path,
        image_root,
        depth_root,
        image_size: int = 224,
        train: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.depth_root = Path(depth_root)
        self.image_size = image_size
        self.train = train

        self.data = pd.read_csv(self.csv_path)

        required_columns = {"image_path", "fake_label", "transform_label"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in CSV: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # Build RGB image path from image_root and CSV relative path.
        image_rel_path = row["image_path"]
        image_path = self.image_root / image_rel_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Build corresponding depth map path.
        depth_path = find_depth_path(self.depth_root, image_rel_path)

        image = Image.open(image_path).convert("RGB")

        depth_array = np.load(depth_path)

        # In case the saved depth has an extra singleton dimension, remove it.
        if depth_array.ndim == 3:
            depth_array = depth_array.squeeze()

        # Normalize depth maps
        depth_array = normalize_01(depth_array)

        # Convert depth map to PIL image so that resize/flips are easy.
        depth_uint8 = (depth_array * 255).astype(np.uint8)
        depth_image = Image.fromarray(depth_uint8, mode="L")
        image = TF.resize(image, [self.image_size, self.image_size])
        depth_image = TF.resize(depth_image, [self.image_size, self.image_size])

        # Apply the same random horizontal flip to RGB and depth during training.
        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            depth_image = TF.hflip(depth_image)

        rgb_raw = TF.to_tensor(image)
        depth_tensor = TF.to_tensor(depth_image)

        # Compute edge maps from RGB grayscale and depth.
        gray_rgb = rgb_to_grayscale(rgb_raw)
        rgb_edges = sobel_edges(gray_rgb)
        depth_edges = sobel_edges(depth_tensor)

        # high values mean RGB edges and depth edges disagree.
        edge_consistency = torch.abs(rgb_edges - depth_edges)
        image_tensor = (rgb_raw - IMAGENET_MEAN) / IMAGENET_STD

        fake_label = torch.tensor(int(row["fake_label"]), dtype=torch.long)
        transform_label = torch.tensor(int(row["transform_label"]), dtype=torch.long)

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "edge_consistency": edge_consistency,
            "fake_label": fake_label,
            "transform_label": transform_label,
            "image_path": str(image_path),
        }