"""
prepare_dataset.py — Offline Dataset Preparation & Augmentation
================================================================
Reads raw MRI images from data/brain_mri/{yes,no}/
Splits them into train / val / test with a fixed random seed,
then saves augmented copies to disk under data/processed/.

Folder structure created:
    data/processed/
    ├── train/
    │   ├── yes/        ← original + N augmented copies per image
    │   └── no/
    ├── val/
    │   ├── yes/        ← original images only (no augmentation)
    │   └── no/
    └── test/
        ├── yes/        ← original images only (no augmentation)
        └── no/

Why offline augmentation?
  • Augmented images are generated once and reused every epoch → faster training
  • You can inspect exactly what the model sees during training
  • No RAM spent computing transforms on-the-fly each epoch

Usage:
    cd brain_tumor_detection
    python src/prepare_dataset.py
    python src/prepare_dataset.py --src_dir data/brain_mri --out_dir data/processed --aug_factor 8

Options:
    --src_dir       Raw dataset root (default: data/brain_mri)
    --out_dir       Output root      (default: data/processed)
    --aug_factor    Augmented copies per training image (default: 8)
    --seed          Random seed      (default: 42)
    --val_ratio     Fraction for val (default: 0.15)
    --test_ratio    Fraction for test(default: 0.15)
    --overwrite     Delete and rebuild out_dir if it already exists
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Augmentation pipeline  (training only — mirrors what was in dataset.py)
# ---------------------------------------------------------------------------
TRAIN_AUG = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomGrayscale(p=0.05),
])

# For the original training copy we keep (no aug, just resize to 224)
RESIZE_ONLY = transforms.Resize((224, 224))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(class_dir: Path) -> list[Path]:
    """Return sorted list of image paths inside a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
    imgs.sort()
    return imgs


def save_pil(img: Image.Image, dest: Path) -> None:
    """Save a PIL image as JPEG, creating parent dirs if needed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(dest, format="JPEG", quality=95)


def copy_images(paths: list[Path], dest_dir: Path, label: str) -> int:
    """Copy (resize to 224) images to dest_dir. Returns count."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = RESIZE_ONLY(img)
        save_pil(img, dest_dir / p.name)
    n = len(paths)
    print(f"    {label:5s} → {dest_dir}  ({n} images)")
    return n


def augment_images(paths: list[Path], dest_dir: Path, aug_factor: int,
                   label: str, seed: int) -> int:
    """
    Save the original resized image + aug_factor augmented copies per image.
    Files are named:  {stem}_orig.jpg, {stem}_aug1.jpg … {stem}_augN.jpg
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    rng = random.Random(seed)

    for p in paths:
        img = Image.open(p).convert("RGB")

        # 1. Save resized original
        orig = RESIZE_ONLY(img)
        save_pil(orig, dest_dir / f"{p.stem}_orig.jpg")
        total += 1

        # 2. Save N augmented copies
        for i in range(1, aug_factor + 1):
            # Seed each augmentation deterministically so results are reproducible
            random.seed(rng.randint(0, 2**31))
            aug_img = TRAIN_AUG(img)
            save_pil(aug_img, dest_dir / f"{p.stem}_aug{i:02d}.jpg")
            total += 1

    print(f"    {label:5s} → {dest_dir}  ({total} images, {len(paths)} orig × {aug_factor + 1})")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    src_dir: str = "data/brain_mri",
    out_dir: str = "data/processed",
    aug_factor: int = 8,
    seed: int = 42,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    overwrite: bool = False,
) -> None:

    src = Path(src_dir)
    out = Path(out_dir)

    # Validate source
    for cls in ("yes", "no"):
        cls_dir = src / cls
        if not cls_dir.is_dir():
            print(f"  ❌  Source folder not found: {cls_dir}")
            sys.exit(1)

    # Handle existing output
    if out.exists():
        if overwrite:
            print(f"  🗑️  Removing existing {out} ...")
            shutil.rmtree(out)
        else:
            print(f"  ⚠️   Output folder already exists: {out}")
            print("       Use --overwrite to rebuild it from scratch.")
            sys.exit(1)

    random.seed(seed)
    print()
    print("=" * 60)
    print("  Brain Tumor Dataset — Offline Preparation")
    print("=" * 60)
    print(f"  Source  : {src}")
    print(f"  Output  : {out}")
    print(f"  Split   : train={1-val_ratio-test_ratio:.0%} / val={val_ratio:.0%} / test={test_ratio:.0%}")
    print(f"  Aug ×   : {aug_factor + 1}  (1 original + {aug_factor} augmented per training image)")
    print(f"  Seed    : {seed}")
    print()

    grand_total = {"train": 0, "val": 0, "test": 0}

    for cls in ("yes", "no"):
        imgs = collect_images(src / cls)
        n = len(imgs)
        random.shuffle(imgs)

        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        test_imgs  = imgs[:n_test]
        val_imgs   = imgs[n_test : n_test + n_val]
        train_imgs = imgs[n_test + n_val:]

        print(f"  Class '{cls}'  ({n} total → train {len(train_imgs)} / val {len(val_imgs)} / test {len(test_imgs)})")

        # Test — no augmentation
        grand_total["test"]  += copy_images(test_imgs,  out / "test"  / cls, "test")
        # Val — no augmentation
        grand_total["val"]   += copy_images(val_imgs,   out / "val"   / cls, "val")
        # Train — original + augmented copies
        grand_total["train"] += augment_images(train_imgs, out / "train" / cls, aug_factor, "train", seed)

    print()
    print("  ✅  Done!")
    print(f"     train : {grand_total['train']} images")
    print(f"     val   : {grand_total['val']} images")
    print(f"     test  : {grand_total['test']} images")
    print(f"     TOTAL : {sum(grand_total.values())} images")
    print()
    print(f"  Load in train.py with:  --data_dir {out}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-generate augmented brain MRI dataset on disk",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--src_dir",    type=str,   default="data/brain_mri",
                        help="Raw dataset root folder (default: data/brain_mri)")
    parser.add_argument("--out_dir",    type=str,   default="data/processed",
                        help="Output root folder (default: data/processed)")
    parser.add_argument("--aug_factor", type=int,   default=8,
                        help="Number of augmented copies per training image (default: 8)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--val_ratio",  type=float, default=0.15,
                        help="Fraction of data for validation (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Fraction of data for test (default: 0.15)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Delete and rebuild output folder if it already exists")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        src_dir    = args.src_dir,
        out_dir    = args.out_dir,
        aug_factor = args.aug_factor,
        seed       = args.seed,
        val_ratio  = args.val_ratio,
        test_ratio = args.test_ratio,
        overwrite  = args.overwrite,
    )
