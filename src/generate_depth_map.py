# generate_depth_maps.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F


def load_midas_model(model_type: str, device):
    """
    Load a pretrained MiDaS model from PyTorch Hub.

    This model is used only to estimate depth maps before training.
    We do not train MiDaS.
    """

    # Trust only the official repositories needed by MiDaS.
    trust_torch_hub_repositories()

    # Load pretrained MiDaS model without interactive prompts.
    midas = torch.hub.load(
        "intel-isl/MiDaS",
        model_type,
        trust_repo=True,
    )

    midas.to(device)
    midas.eval()

    # Load the official MiDaS transforms without interactive prompts.
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
    )

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform

def trust_torch_hub_repositories():
    """
    Add the official Torch Hub repositories used by MiDaS to the trusted list.

    This avoids interactive trust prompts in Colab for:
    1. intel-isl/MiDaS
    2. rwightman/gen-efficientnet-pytorch

    Only these known repositories are trusted.
    """

    hub_dir = Path(torch.hub.get_dir())
    hub_dir.mkdir(parents=True, exist_ok=True)

    trusted_list_path = hub_dir / "trusted_list"

    repositories_to_trust = [
        "intel-isl_MiDaS",
        "rwightman_gen-efficientnet-pytorch",
    ]

    existing_repositories = set()

    if trusted_list_path.exists():
        with open(trusted_list_path, "r", encoding="utf-8") as file:
            existing_repositories = {
                line.strip()
                for line in file
                if line.strip()
            }

    with open(trusted_list_path, "a", encoding="utf-8") as file:
        for repository in repositories_to_trust:
            if repository not in existing_repositories:
                file.write(repository + "\n")


def compute_depth_for_image(image_path: Path, midas, transform, device):
    """
    Compute a depth map for a single RGB image.

    Args:
        image_path: path to the RGB image.
        midas: pretrained MiDaS model.
        transform: MiDaS preprocessing transform.
        device: cuda or cpu.

    Returns:
        depth_map: numpy array with shape [H, W].
    """

    # Load image and convert to RGB.
    image = Image.open(image_path).convert("RGB")

    # Convert PIL image to numpy array because MiDaS transforms expect numpy RGB.
    image_np = np.array(image)

    # Apply MiDaS preprocessing.
    input_batch = transform(image_np).to(device)

    with torch.no_grad():
        # Predict relative depth.
        prediction = midas(input_batch)

        # Resize prediction back to the original image size.
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=image_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Move result to CPU and convert to numpy.
    depth_map = prediction.cpu().numpy().astype(np.float32)

    return depth_map


def collect_unique_image_paths(csv_paths, image_root: Path):
    """
    Read one or more CSV files and collect all unique image paths.

    The CSV files must contain the column:
        image_path

    This follows the same logic as train.py:
    image_root + image_path gives the full RGB image path.
    """

    all_relative_paths = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if "image_path" not in df.columns:
            raise ValueError(f"CSV {csv_path} does not contain 'image_path' column.")

        all_relative_paths.extend(df["image_path"].tolist())

    # Remove duplicates while preserving order.
    unique_relative_paths = list(dict.fromkeys(all_relative_paths))

    full_paths = []

    for rel_path in unique_relative_paths:
        image_path = image_root / rel_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        full_paths.append((rel_path, image_path))

    return full_paths


def main():
    """
    Main function.

    This script:
        1. Reads CSV files.
        2. Finds all RGB images.
        3. Loads a pretrained MiDaS depth model.
        4. Computes one depth map for each RGB image.
        5. Saves each depth map as .npy using the same relative path.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSV files, e.g. train_balanced.csv val_balanced.csv.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root folder containing RGB images.",
    )

    parser.add_argument(
        "--depth_root",
        type=str,
        required=True,
        help="Folder where computed depth maps will be saved.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="MiDaS_small",
        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
        help="MiDaS model type. MiDaS_small is fastest and recommended first.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit for debugging. Example: --max_images 20",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, recompute depth maps even if they already exist.",
    )

    args = parser.parse_args()

    image_root = Path(args.image_root)
    depth_root = Path(args.depth_root)
    depth_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Collecting image paths from CSV files...")
    image_items = collect_unique_image_paths(args.csv_paths, image_root)

    if args.max_images is not None:
        image_items = image_items[: args.max_images]

    print(f"Number of images to process: {len(image_items)}")

    print(f"Loading pretrained MiDaS model: {args.model_type}")
    midas, transform = load_midas_model(args.model_type, device)

    for rel_path, image_path in tqdm(image_items, desc="Generating depth maps"):
        rel_path = Path(rel_path)

        # Save depth with the same relative path as RGB image, but .npy extension.
        depth_path = depth_root / rel_path.with_suffix(".npy")
        depth_path.parent.mkdir(parents=True, exist_ok=True)

        if depth_path.exists() and not args.overwrite:
            continue

        depth_map = compute_depth_for_image(
            image_path=image_path,
            midas=midas,
            transform=transform,
            device=device,
        )

        np.save(depth_path, depth_map)

    print("Depth map generation completed.")
    print(f"Depth maps saved in: {depth_root}")


if __name__ == "__main__":
    main()