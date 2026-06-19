from pathlib import Path
import shutil

import pandas as pd


SEED = 42
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Folder name -> numeric label used by the model
TRANSFORM_LABELS = {
    "original": 0,
    "transfer": 1,
    "redigital": 2,
}

FAKE_LABELS = {
    "real": 0,
    "ai": 1,
}


def collect_images(folder: Path):
    """
    Return all image files inside a folder.
    """
    images = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)


def get_ai_category(image_path: Path):
    """
    Extract category from AI filename.
    """

    stem = image_path.stem

    # Remove transformation prefix if it exists
    if stem.startswith("transfer_"):
        stem = stem.replace("transfer_", "", 1)

    if stem.startswith("redigital_"):
        stem = stem.replace("redigital_", "", 1)

    # Remove final numeric id
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]

    category = "_".join(parts)

    return category if category else "unknown"


def sample_ai(images, n_total):
    """
    Sample AI images approximately balanced by category.
    """

    df = pd.DataFrame({
        "source_path": images,
        "category": [get_ai_category(p) for p in images],
    })

    # If there are fewer available images than requested,
    # we take all of them.
    if len(df) <= n_total:
        return df.sample(frac=1.0, random_state=SEED)

    categories = sorted(df["category"].unique())
    quota = n_total // len(categories)
    remainder = n_total % len(categories)

    sampled = []

    for i, category in enumerate(categories):

        # Select all images belonging to this category
        category_df = df[df["category"] == category]

        # Give one extra image to the first "remainder" categories
        n_category = quota + (1 if i < remainder else 0)
        
        n_category = min(n_category, len(category_df))

        sampled.append(category_df.sample(n=n_category, random_state=SEED))

    sampled_df = pd.concat(sampled)

    # If some category had fewer images, fill missing samples randomly
    missing = n_total - len(sampled_df)

    if missing > 0:

        # Remove images already selected.
        remaining = df.drop(sampled_df.index)

        # Sample extra images from the remaining pool.
        extra = remaining.sample(
            n=min(missing, len(remaining)),
            random_state=SEED,
        )
        sampled_df = pd.concat([sampled_df, extra])

    return sampled_df.sample(frac=1.0, random_state=SEED)


def sample_real(images, n_total):
    """
    Sample real images randomly.
    """

    df = pd.DataFrame({
        "source_path": images,
        "category": "unknown",
    })

    return df.sample(n=min(n_total, len(df)), random_state=SEED)


def split_group(df):
    """
    Split one group into train/val/test.

    Doing this inside each group keeps all splits balanced.
    """

    # Shuffle the group before splitting.
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def copy_images(df, subset_root: Path):
    """
    Copy selected images into RRDataset_subset and create image_path.
    """

    rows = []

    for _, row in df.iterrows():
        source_path = Path(row["source_path"])

        relative_path = (
            Path(row["transform_name"])
            / row["fake_name"]
            / source_path.name
        )

        destination_path = subset_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, destination_path)

        new_row = row.to_dict()
        new_row["image_path"] = str(relative_path).replace("\\", "/")
        rows.append(new_row)

    return pd.DataFrame(rows)


def save_csv(df, output_dir: Path, split_name: str):
    """
    Save one split CSV.
    """

    columns = [
        "image_path",
        "fake_label",
        "transform_label",
        "category",
        "transform_name",
        "fake_name",
    ]

    output_path = output_dir / f"{split_name}_balanced.csv"
    df[columns].to_csv(output_path, index=False)

    print(f"Saved {output_path}")


def main():
    print("=" * 60)
    print("Preparing balanced RRDataset subset")
    print("=" * 60)

    full_dataset_root = Path("data/raw/RRDataset_final")
    subset_root = Path("data/raw/RRDataset_subset")
    output_dir = Path("data/splits")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove old subset to avoid mixing old and new images
    if subset_root.exists():
        shutil.rmtree(subset_root)
    
    n_per_group = 500
    train_parts = []
    val_parts = []
    test_parts = []

    for transform_name, transform_label in TRANSFORM_LABELS.items():
        for fake_name, fake_label in FAKE_LABELS.items():

            folder = full_dataset_root / transform_name / fake_name

            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder}")

            images = collect_images(folder)
            print(f"{transform_name}/{fake_name}: found {len(images)} images")

            # Select images from this group.
            n_select = min(n_per_group, len(images))

            if fake_name == "ai":
                selected_df = sample_ai(images, n_select)
            else:
                selected_df = sample_real(images, n_select)

            print(
                f"{transform_name}/{fake_name}: selected "
                f"{len(selected_df)} images"
            )

            if fake_name == "ai":
                print("AI category distribution:")
                print(selected_df["category"].value_counts().sort_index())
                
            selected_df["transform_name"] = transform_name
            selected_df["fake_name"] = fake_name
            selected_df["fake_label"] = fake_label
            selected_df["transform_label"] = transform_label
            
            # Split this single group into train/val/test.
            group_train, group_val, group_test = split_group(selected_df)

            train_parts.append(group_train)
            val_parts.append(group_val)
            test_parts.append(group_test)

    # Merge all groups into final train/val/test tables.
    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts)
    test_df = pd.concat(test_parts)

    # Copy selected images into RRDataset_subset
    all_df = pd.concat([train_df, val_df, test_df])
    print("\nCopying selected images into subset folder...")
    all_df = copy_images(all_df, subset_root)

    train_final = all_df[all_df["split"] == "train"]
    val_final = all_df[all_df["split"] == "val"]
    test_final = all_df[all_df["split"] == "test"]

    save_csv(train_final, output_dir, "train")
    save_csv(val_final, output_dir, "val")
    save_csv(test_final, output_dir, "test")

    print("\nBalance check:")
    print(all_df.groupby(["transform_name", "fake_name", "split"]).size())

    print("\nSubset ready in:")
    print(subset_root)


if __name__ == "__main__":
    main()