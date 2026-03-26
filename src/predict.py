"""
predict.py — Single Image Brain Tumor Prediction
=================================================
Loads the trained model and predicts whether a single MRI image
contains a brain tumor or not.

Output:
  - Diagnosis: "TUMOR DETECTED" or "NO TUMOR DETECTED"
  - Confidence percentage for both classes
  - Optional: save a visualisation of the prediction

Course Concepts Covered (Chapter 5):
  ✔ Model inference / deployment
  ✔ Softmax probabilities for classification output

Usage:
    cd brain_tumor_detection
    python src/predict.py --image path/to/mri.jpg --model_path results/best_model.pth

Example:
    python src/predict.py --image data/brain_mri/yes/Y1.jpg
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
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model   import BrainTumorCNN
from dataset import get_eval_transforms, denormalize


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict_image(
    image_path: str,
    model_path: str = "results/best_model.pth",
    device: torch.device = None,
    save_result: bool = True,
    save_dir: str = "results",
) -> dict:
    """
    Predict whether a single MRI image has a brain tumor.

    Pipeline:
      1. Load and preprocess the image (same as test set transforms)
      2. Load trained model
      3. Run forward pass → get logits
      4. Apply Softmax → convert to probabilities
      5. Return prediction and confidence

    Args:
        image_path  : Path to the input MRI image (JPG/PNG)
        model_path  : Path to the saved model checkpoint
        device      : torch.device to use (auto-detects if None)
        save_result : If True, save a visualisation image
        save_dir    : Folder to save visualisation

    Returns:
        result dict with keys:
            'predicted_class' : 0 (no tumor) or 1 (tumor)
            'predicted_label' : "no" or "yes"
            'diagnosis'       : Human-readable diagnosis string
            'confidence'      : Confidence % for predicted class
            'prob_no_tumor'   : Probability of no tumor (%)
            'prob_tumor'      : Probability of tumor (%)
    """
    CLASS_NAMES = ["no", "yes"]

    # ------------------------------------------------------------------
    # 1. Detect device
    # ------------------------------------------------------------------
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"\n  ❌ Model not found: {model_path}")
        print("     Please train the model first:  python src/train.py")
        sys.exit(1)

    model = BrainTumorCNN(num_classes=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ------------------------------------------------------------------
    # 3. Load and preprocess image
    # ------------------------------------------------------------------
    if not os.path.exists(image_path):
        print(f"\n  ❌ Image not found: {image_path}")
        sys.exit(1)

    image_pil = Image.open(image_path).convert("RGB")

    # Apply the same preprocessing as the test set (resize + normalise)
    transform = get_eval_transforms()
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Add batch dim: (1, 3, 224, 224)

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits = model(image_tensor)                      # Raw scores (1, 2)
        probs  = torch.softmax(logits, dim=1).cpu()[0]   # Probabilities (2,)

    # ------------------------------------------------------------------
    # 5. Extract result
    # ------------------------------------------------------------------
    predicted_class = int(torch.argmax(probs).item())
    predicted_label = CLASS_NAMES[predicted_class]
    confidence      = float(probs[predicted_class].item()) * 100
    prob_no_tumor   = float(probs[0].item()) * 100
    prob_tumor      = float(probs[1].item()) * 100

    if predicted_class == 1:
        diagnosis = "⚠️  TUMOR DETECTED"
        diag_color = "red"
    else:
        diagnosis = "✅  NO TUMOR DETECTED"
        diag_color = "green"

    result = {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "diagnosis":        diagnosis,
        "confidence":       confidence,
        "prob_no_tumor":    prob_no_tumor,
        "prob_tumor":       prob_tumor,
    }

    # ------------------------------------------------------------------
    # 6. Print result to console
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  BRAIN TUMOR DETECTION — PREDICTION RESULT")
    print("=" * 50)
    print(f"  Image     : {image_path}")
    print(f"  Diagnosis : {diagnosis}")
    print(f"  Confidence: {confidence:.2f}%")
    print()
    print("  Class Probabilities:")
    print(f"    No Tumor : {prob_no_tumor:.2f}%")
    print(f"    Tumor    : {prob_tumor:.2f}%")
    print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # 7. Save visualisation
    # ------------------------------------------------------------------
    if save_result:
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Brain Tumor Detection Result", fontsize=13, fontweight="bold")

        # Original image (resized for display)
        display_img = image_pil.resize((224, 224))
        axes[0].imshow(display_img)
        axes[0].set_title(diagnosis, color=diag_color, fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Probability bar chart
        bar_colors = ["steelblue", "tomato"]
        bars = axes[1].bar(
            ["No Tumor", "Tumor"],
            [prob_no_tumor, prob_tumor],
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        axes[1].set_ylim(0, 110)
        axes[1].set_ylabel("Probability (%)", fontsize=10)
        axes[1].set_title("Class Probabilities", fontsize=11)

        # Add percentage labels on bars
        for bar, prob in zip(bars, [prob_no_tumor, prob_tumor]):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{prob:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        axes[1].grid(axis="y", alpha=0.3)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        img_name = Path(image_path).stem
        save_path = os.path.join(save_dir, f"prediction_{img_name}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Visualisation saved → {save_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict brain tumor from a single MRI image",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input MRI image (JPG or PNG)\nExample: data/brain_mri/yes/Y10.jpg"
    )
    parser.add_argument(
        "--model_path", type=str, default="results/best_model.pth",
        help="Path to trained model checkpoint (default: results/best_model.pth)"
    )
    parser.add_argument(
        "--save_dir", type=str, default="results",
        help="Directory to save prediction visualisation"
    )
    parser.add_argument(
        "--no_save", action="store_true",
        help="Do not save prediction visualisation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image(
        image_path=args.image,
        model_path=args.model_path,
        save_result=not args.no_save,
        save_dir=args.save_dir,
    )
