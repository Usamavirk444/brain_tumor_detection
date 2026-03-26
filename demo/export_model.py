"""
export_model.py — Export PyTorch model to ONNX format
======================================================
Run this ONCE on your Mac before opening the demo.

Usage:
    cd brain_tumor_detection
    python demo/export_model.py

Output:
    demo/brain_tumor_model.onnx   ← load this in index.html
"""

import sys
import torch
from pathlib import Path

# Add src/ to path so we can import model.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import BrainTumorCNN

MODEL_PATH = Path(__file__).parent.parent / "results" / "best_brain_tumor_model.pth"
OUTPUT_PATH = Path(__file__).parent / "brain_tumor_model.onnx"

def export():
    print("\n" + "="*50)
    print("  BrainTumorCNN — ONNX Export")
    print("="*50)

    if not MODEL_PATH.exists():
        print(f"\n  ❌  Model not found: {MODEL_PATH}")
        print("     Run train.py first.\n")
        sys.exit(1)

    # Load model
    device = torch.device("cpu")
    model = BrainTumorCNN(num_classes=2, dropout_rate=0.4, l2_lambda=5e-5)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"\n  ✅  Model loaded from: {MODEL_PATH}")

    # Export using legacy exporter (compatible with ONNX Runtime Web)
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False   # Force legacy exporter — compatible with ONNX Runtime Web
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✅  ONNX model saved: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print("\n  Now open demo/index.html in Chrome.")
    print("="*50 + "\n")

if __name__ == "__main__":
    export()
