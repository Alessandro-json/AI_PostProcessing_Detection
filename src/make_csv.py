from pathlib import Path

import pandas as pd


def collect_split(dataset_root: Path, split: str):
    """
    Collect all images for one split.

    Args:
        dataset_root: Root folder of the extracted dataset.

        split:
            Dataset split name:
                train
                val

    Returns:
        A list of dictionaries.
        Each dictionary corresponds to one image and contains:
            - image_path
            - fake_label
            - transform_label
    """

    rows = []

    # Folder containing AI-generated images.
    ai_dir = dataset_root / split / "ai"

    # Folder containing real images.
    real_dir = dataset_root / split / "real"

    # Collect AI-generated images.
    for image_path in sorted(ai_dir.glob("*.png")):
        rows.append(
            {
                "image_path": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
                "fake_label": 1,
                "transform_label": 0,
            }
        )

    # Collect real images.
    for image_path in sorted(real_dir.glob("*.jpg")):
        rows.append(
            {
                "image_path": str(image_path.relative_to(dataset_root)).replace("\\", "/"),
                "fake_label": 0,
                "transform_label": 0,
            }
        )

    return rows


def main():
    """
    Create train.csv and val.csv from the extracted RRDataset folder.
    """

    dataset_root = Path("data/raw/RRDataset_original_train_val")
    output_dir = Path("data/splits")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        rows = collect_split(dataset_root, split)
        df = pd.DataFrame(rows)

        output_path = output_dir / f"{split}.csv"
        df.to_csv(output_path, index=False)

        print(f"Saved {output_path}")
        print(f"Number of images: {len(df)}")
        print(df["fake_label"].value_counts().sort_index())
        print()


if __name__ == "__main__":
    main()