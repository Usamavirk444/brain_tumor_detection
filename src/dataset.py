"""
dataset.py — Brain Tumor Dataset Loading & Preprocessing
==========================================================
Handles all data-related operations:
  1. Loading images from /yes and /no folders
  2. Splitting into train / validation / test sets (70% / 15% / 15%)
  3. Applying data augmentation to the training set only
  4. Returning PyTorch DataLoader objects ready for training

Two loading modes are supported:

  A) On-the-fly mode (original)
     Reads from data/brain_mri/{yes,no}/ and applies augmentation in RAM.
     Use: get_dataloaders(data_dir="data/brain_mri", ...)

  B) Pre-split mode (recommended after running prepare_dataset.py)
     Reads from pre-split data/processed/train|val|test/{yes,no}/ folders.
     Augmentation was already applied to disk — only normalise at runtime.
     Use: get_dataloaders_presplit(processed_dir="data/processed", ...)

Course Concepts Covered (Chapter 2 & 7):
  ✔ Data preprocessing and normalisation
  ✔ Data augmentation (reduces overfitting, covered in Chapter 7)
  ✔ Train/val/test split principles

Usage:
    from dataset import get_dataloaders, get_dataloaders_presplit

    # Mode A — on-the-fly augmentation
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir="data/brain_mri", batch_size=32
    )

    # Mode B — pre-split (faster, augmented images already on disk)
    train_loader, val_loader, test_loader, class_names = get_dataloaders_presplit(
        processed_dir="data/processed", batch_size=32
    )
"""

import os
import random
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# ImageNet normalisation constants (used because our model learns RGB images
# and these values standardise pixel intensities to ~N(0,1))
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def get_train_transforms() -> transforms.Compose:
    """
    Data augmentation pipeline for the TRAINING set.

    Augmentation prevents overfitting by generating slight variations of
    each image, effectively increasing dataset size.  (Chapter 7)

    Operations applied:
      - RandomHorizontalFlip   : Mirror left-right (50% chance)
      - RandomRotation(15)     : Rotate up to ±15 degrees
      - ColorJitter            : Vary brightness/contrast (simulate scanner diffs)
      - RandomResizedCrop(224) : Random zoom/crop (scale 80%–100% of original)
      - ToTensor               : Convert PIL Image [0-255] → Tensor [0-1]
      - Normalize              : Standardise to ImageNet mean/std
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),                              # Slightly larger before crop
        transforms.RandomHorizontalFlip(p=0.5),                    # Mirror left-right
        transforms.RandomRotation(degrees=15),                     # Rotate ±15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Scanner variation
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Random zoom crop
        transforms.ToTensor(),                                      # [H,W,C] → [C,H,W], values 0-1
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),# Normalise pixel values
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Deterministic preprocessing pipeline for VALIDATION and TEST sets.

    No augmentation — we always want the same evaluation to compare models.
    Only resize + centre crop + normalise.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),            # Deterministic centre crop
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Custom Dataset class
# ---------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset that loads brain MRI images from a folder with two subfolders:
        data_dir/
            yes/   ← images WITH tumor    → label = 1
            no/    ← images WITHOUT tumor → label = 0

    Args:
        data_dir  : Path to the root folder containing /yes and /no subfolders
        transform : torchvision transform pipeline to apply to each image
    """

    # Fixed class ordering: class_to_idx = {"no": 0, "yes": 1}
    CLASSES = ["no", "yes"]

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # Build list of (image_path, label) tuples
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Walk through /yes and /no subdirectories and collect all image paths."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for class_name in self.CLASSES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"  ⚠ Warning: folder not found → {class_dir}")
                continue

            label = self.class_to_idx[class_name]
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in valid_extensions:
                    self.samples.append((img_file, label))

        print(f"  Dataset loaded: {len(self.samples)} images total")
        for cls in self.CLASSES:
            count = sum(1 for _, lbl in self.samples if lbl == self.class_to_idx[cls])
            print(f"    Class '{cls}': {count} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns a single (image_tensor, label) pair.

        Args:
            idx : Index of the sample

        Returns:
            image : Tensor of shape (3, 224, 224)
            label : Integer class index (0 = no tumor, 1 = tumor)
        """
        img_path, label = self.samples[idx]

        # Load image as RGB (ensures 3 channels regardless of source format)
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (augmentation for train, resize/normalise for eval)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @property
    def class_names(self) -> List[str]:
        return self.CLASSES


# ---------------------------------------------------------------------------
# DataLoader factory function
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.70,
    val_split: float = 0.15,
    num_workers: int = 0,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train, validation, and test DataLoaders from a single directory.

    Split ratios: 80% train / 10% validation / 10% test

    Args:
        data_dir     : Path to the root folder (containing /yes and /no)
        batch_size   : Number of images per mini-batch
        train_split  : Fraction of data used for training (default 0.70)
        val_split    : Fraction of data used for validation (default 0.15)
        num_workers  : Parallel data loading workers (set 0 on macOS to avoid issues)
        random_seed  : Seed for reproducible splits

    Returns:
        train_loader : DataLoader with augmentation
        val_loader   : DataLoader without augmentation
        test_loader  : DataLoader without augmentation
        class_names  : ["no", "yes"]
    """
    # Seed for reproducibility
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Step 1: Load full dataset (no transform yet — we apply per-split below)
    full_dataset_train = BrainTumorDataset(data_dir, transform=get_train_transforms())
    full_dataset_eval  = BrainTumorDataset(data_dir, transform=get_eval_transforms())

    n = len(full_dataset_train)
    indices = list(range(n))
    random.shuffle(indices)

    # Step 2: Calculate split sizes
    n_train = int(train_split * n)
    n_val   = int(val_split * n)
    # Everything leftover goes to test
    n_test  = n - n_train - n_val

    train_indices = indices[:n_train]
    val_indices   = indices[n_train : n_train + n_val]
    test_indices  = indices[n_train + n_val :]

    print(f"\n  Data splits:")
    print(f"    Train      : {len(train_indices)} images")
    print(f"    Validation : {len(val_indices)} images")
    print(f"    Test       : {len(test_indices)} images")

    # Step 3: Build subset datasets
    # Training set uses augmentation transforms, eval sets use deterministic transforms
    train_subset = Subset(full_dataset_train, train_indices)
    val_subset   = Subset(full_dataset_eval,  val_indices)
    test_subset  = Subset(full_dataset_eval,  test_indices)

    # Step 4: Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=False,      # Keep False for MPS compatibility
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,         # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, BrainTumorDataset.CLASSES


# ---------------------------------------------------------------------------
# Pre-split DataLoader factory (use after running prepare_dataset.py)
# ---------------------------------------------------------------------------

def get_normalise_only_transforms() -> transforms.Compose:
    """
    Minimal transform for pre-augmented images already saved to disk.
    Images were already resized to 224×224 by prepare_dataset.py,
    so we only need ToTensor + Normalize here.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),       # Safety resize (images already 224)
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_dataloaders_presplit(
    processed_dir: str = "data/processed",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """
    Build DataLoaders from a pre-split folder created by prepare_dataset.py.

    Expected folder structure:
        processed_dir/
        ├── train/yes/   train/no/   ← originals + augmented copies
        ├── val/yes/     val/no/     ← originals only
        └── test/yes/    test/no/    ← originals only

    HYBRID AUGMENTATION STRATEGY:
        Training split  → get_train_transforms() applied at runtime
                          Each epoch the model sees the 2,288 pre-split images
                          with FRESH random transforms (flip, rotate, jitter, crop).
                          This combines the larger dataset size (2,288 vs 177) with
                          the variety of on-the-fly augmentation to prevent memorisation.
        Val / Test split → normalise only (deterministic, no augmentation)

    Args:
        processed_dir : Root of the pre-split dataset (default: data/processed)
        batch_size    : Number of images per mini-batch
        num_workers   : Parallel workers (use 0 on macOS)

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    base = Path(processed_dir)

    datasets = {}
    for split in ("train", "val", "test"):
        split_dir = base / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Pre-split folder not found: {split_dir}\n"
                "Run:  python src/prepare_dataset.py  first."
            )
        # Training: use full augmentation pipeline (fresh random transforms each epoch)
        # Val/Test: normalise only — always deterministic for fair comparison
        transform = get_train_transforms() if split == "train" else get_normalise_only_transforms()
        print(f"\n  Loading '{split}' split from {split_dir}")
        datasets[split] = BrainTumorDataset(str(split_dir), transform=transform)

    print(f"\n  Split summary (pre-split mode):")
    for split in ("train", "val", "test"):
        print(f"    {split:5s} : {len(datasets[split])} images")

    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader, BrainTumorDataset.CLASSES


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalisation for visualisation purposes.

    Converts a normalised tensor back to [0, 1] range for display.

    Args:
        tensor : Normalised image tensor of shape (3, H, W)

    Returns:
        Denormalised tensor, clipped to [0, 1]
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/brain_mri"

    print(f"Testing dataset loading from: {data_dir}")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir, batch_size=8
    )

    print(f"\n  Class names : {class_names}")
    print(f"  Train batches      : {len(train_loader)}")
    print(f"  Validation batches : {len(val_loader)}")
    print(f"  Test batches       : {len(test_loader)}")

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\n  Sample batch:")
    print(f"    Image tensor shape : {images.shape}")
    print(f"    Labels             : {labels.tolist()}")
    print(f"    Pixel min / max    : {images.min():.3f} / {images.max():.3f}")
    print("\n  ✅ Dataset loading successful!")
