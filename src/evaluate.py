"""
evaluate.py — Model Evaluation & Metrics
=========================================
Loads the saved best model and evaluates it on the test set.

Outputs:
  - Accuracy, Precision, Recall, F1-Score (printed to console)
  - Confusion matrix plot (saved to results/)
  - Sample prediction grid (saved to results/)

Course Concepts Covered (Chapter 2 & 5):
  ✔ Evaluation metrics: accuracy, precision, recall, F1
  ✔ Confusion matrix analysis
  ✔ Model inference in eval mode (no gradients)

Usage:
    cd brain_tumor_detection
    python src/evaluate.py --data_dir data/brain_mri --model_path results/best_model.pth
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model   import BrainTumorCNN
from dataset import get_dataloaders, denormalize


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run inference on all batches in loader and collect predictions.

    Returns:
        all_preds  : List of predicted class indices
        all_labels : List of true class indices
        all_probs  : List of softmax probability arrays (for each image)
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    softmax = nn.Softmax(dim=1)

    for images, labels in tqdm(loader, desc="  Evaluating", unit="batch"):
        images = images.to(device)
        outputs = model(images)                 # Raw logits
        probs   = softmax(outputs).cpu().numpy() # Convert to probabilities
        preds   = np.argmax(probs, axis=1)       # Predicted class

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    return all_preds, all_labels, all_probs


# ---------------------------------------------------------------------------
# Print metrics
# ---------------------------------------------------------------------------

def print_metrics(
    preds: list,
    labels: list,
    class_names: list,
) -> dict:
    """
    Compute and print all classification metrics.

    Metrics explained:
      Accuracy  : Overall fraction correctly classified
      Precision : Of all predicted positives, how many are actually positive?
                  (Important when false positives are costly)
      Recall    : Of all actual positives, how many did we detect?
                  (Important when false negatives are costly — critical in medicine!)
      F1-Score  : Harmonic mean of precision and recall (balanced metric)

    In medical diagnosis, HIGH RECALL is especially important because
    missing a real tumour (false negative) is more dangerous than a
    false alarm (false positive).
    """
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="binary", pos_label=1, zero_division=0)
    rec  = recall_score(labels, preds,    average="binary", pos_label=1, zero_division=0)
    f1   = f1_score(labels, preds,        average="binary", pos_label=1, zero_division=0)

    print("\n" + "=" * 55)
    print("  MODEL EVALUATION RESULTS — Test Set")
    print("=" * 55)
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print()
    print("  Detailed Classification Report:")
    print("-" * 55)
    print(classification_report(
        labels, preds,
        target_names=[f"Class '{c}'" for c in class_names],
        digits=4
    ))
    print("=" * 55)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    preds: list,
    labels: list,
    class_names: list,
    save_dir: str = "results",
) -> None:
    """
    Generate and save a confusion matrix heatmap.

    Confusion matrix layout (binary):
             Predicted
             no    yes
    Actual no   TN    FP
           yes  FN    TP

    TN = True Negatives  (correctly predicted "no tumor")
    TP = True Positives  (correctly predicted "tumor")
    FP = False Positives (said "tumor" but no tumor — false alarm)
    FN = False Negatives (missed real tumour — most dangerous error!)
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Predicted: {c}" for c in class_names],
        yticklabels=[f"Actual: {c}"    for c in class_names],
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Sample prediction grid
# ---------------------------------------------------------------------------

def save_sample_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list,
    device: torch.device,
    save_dir: str = "results",
    n_samples: int = 12,
) -> None:
    """
    Save a grid of test images with predicted vs actual labels.

    Green title = correct prediction
    Red title   = incorrect prediction
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    softmax = nn.Softmax(dim=1)

    images_shown = []
    preds_shown  = []
    labels_shown = []
    confs_shown  = []

    with torch.no_grad():
        for images, labels in loader:
            imgs_cpu = images  # Keep on CPU for plotting
            images = images.to(device)
            outputs = model(images)
            probs = softmax(outputs).cpu()
            preds = torch.argmax(probs, dim=1)

            for i in range(len(labels)):
                if len(images_shown) >= n_samples:
                    break
                images_shown.append(imgs_cpu[i])
                preds_shown.append(preds[i].item())
                labels_shown.append(labels[i].item())
                confs_shown.append(probs[i][preds[i].item()].item())

            if len(images_shown) >= n_samples:
                break

    # Plot grid
    cols = 4
    rows = (len(images_shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten()

    for i, (img_tensor, pred, label, conf) in enumerate(
        zip(images_shown, preds_shown, labels_shown, confs_shown)
    ):
        # Denormalize for display
        img_disp = denormalize(img_tensor).permute(1, 2, 0).numpy()
        axes[i].imshow(img_disp)
        axes[i].axis("off")

        title_color = "green" if pred == label else "red"
        title = (
            f"Pred: {class_names[pred]} ({conf*100:.1f}%)\n"
            f"True: {class_names[label]}"
        )
        axes[i].set_title(title, color=title_color, fontsize=8)

    # Hide any unused subplot panels
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sample Predictions (Green = Correct, Red = Wrong)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Sample predictions saved → {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(args):
    print("\n" + "=" * 60)
    print("  Brain Tumor CNN — Evaluation on Test Set")
    print("=" * 60)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # Load data — directly from data/processed/test/{yes,no}/
    print("\n  Loading dataset...")
    from dataset import BrainTumorDataset, get_normalise_only_transforms
    from torch.utils.data import DataLoader as _DataLoader
    test_dataset = BrainTumorDataset(
        os.path.join(args.data_dir, "test"),
        transform=get_normalise_only_transforms()
    )
    test_loader  = _DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names  = BrainTumorDataset.CLASSES

    # Load model
    print(f"\n  Loading model from: {args.model_path}")
    model = BrainTumorCNN(num_classes=2, dropout_rate=0.4, l2_lambda=5e-5).to(device)

    if not os.path.exists(args.model_path):
        print(f"\n  ❌ Model checkpoint not found: {args.model_path}")
        print("     Please run train.py first.\n")
        sys.exit(1)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✅ Model loaded (trained for {checkpoint.get('epoch', '?')} epochs, "
          f"best_val_f1 = {checkpoint.get('best_val_f1', 0):.4f})")

    # Run evaluation
    print("\n  Running inference on test set...")
    preds, labels, probs = run_evaluation(model, test_loader, device)

    # Print metrics
    metrics = print_metrics(preds, labels, class_names)

    # Save confusion matrix
    save_confusion_matrix(preds, labels, class_names, save_dir=args.save_dir)

    # Save sample predictions
    save_sample_predictions(
        model, test_loader, class_names, device,
        save_dir=args.save_dir, n_samples=12
    )

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor CNN on test set")
    parser.add_argument("--data_dir",   type=str, default="data/processed",
                        help="Path to processed dataset directory (expects a test/ subfolder)")
    parser.add_argument("--model_path", type=str, default="results/best_brain_tumor_model.pth",
                        help="Path to saved model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--save_dir",   type=str, default="results",
                        help="Directory to save evaluation outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
