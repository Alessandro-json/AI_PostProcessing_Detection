from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def build_train_transform(image_size: int = 224):
    """
    Transformations used during training to prepare the images for the neural network.
    """
    return transforms.Compose([
        # All images must have the same spatial size before entering the model.
        transforms.Resize((image_size, image_size)),

        # With probability 0.5, the image is flipped horizontally to generalize better.
        transforms.RandomHorizontalFlip(p=0.5),

        # Convert PIL image with pixel values in [0, 255]
        # to a PyTorch tensor with values in [0, 1].
        # Output shape becomes [channels, height, width].
        transforms.ToTensor(),

        # Normalize RGB channels using ImageNet mean and standard deviation.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_eval_transform(image_size: int = 224):
    """
    Transformations used during validation and testing.
    """
    return transforms.Compose([
        # Resize image to the same size used during training.
        transforms.Resize((image_size, image_size)),

        # Convert the image to a PyTorch tensor.
        transforms.ToTensor(),

        # Apply the same ImageNet normalization used during training.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class RRDatasetFromCSV(Dataset):
    """
    PyTorch Dataset for RRDataset using a CSV file.

    The CSV file must contain one row per image and must have these columns:

        image_path,fake_label,transform_label

    Label meaning:

        fake_label:
            0 = real image
            1 = AI-generated image

        transform_label:
            0 = original image
            1 = internet-transmitted image
            2 = re-digitized image

    The dataset returns a dictionary containing:
        - image tensor
        - real/fake label
        - transformation label
        - image path, useful for debugging
    """

    def __init__(self, csv_path, image_root, transform=None):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to the CSV file containing image paths and labels.
            image_root: Root folder where images are stored.
            transform: Transformations applied to each image.
                
        """

        # Convert paths to Path objects.
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)

        # Load CSV file into a pandas DataFrame.
        self.data = pd.read_csv(self.csv_path)

        # Store transformations.
        self.transform = transform

        # Check that the CSV has the columns we need.
        required_columns = {"image_path", "fake_label", "transform_label"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in CSV: {missing_columns}. "
                f"Expected columns are: {required_columns}"
            )

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Load one sample from the dataset.

        Returns:
            A dictionary with:
                image: Tensor of shape [3, image_size, image_size]
                fake_label: Tensor containing 0 or 1
                transform_label: Tensor containing 0, 1, or 2
                image_path: Full image path as a string
        """

        # Select one row from the CSV.
        row = self.data.iloc[index]

        # Build the full path of the image.
        image_path = self.image_root / row["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Open the image and force RGB format.
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing / augmentation transformations.
        if self.transform is not None:
            image = self.transform(image)

        # Convert labels from CSV values to PyTorch tensors.
        fake_label = torch.tensor(int(row["fake_label"]), dtype=torch.long)
        transform_label = torch.tensor(int(row["transform_label"]), dtype=torch.long)

        # Return everything as a dictionary.
        return {
            "image": image,
            "fake_label": fake_label,
            "transform_label": transform_label,
            "image_path": str(image_path),
        }