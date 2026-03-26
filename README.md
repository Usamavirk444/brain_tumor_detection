# 🧠 Brain Tumor Detection Using Convolutional Neural Networks

A complete deep learning project for binary classification of brain MRI scans — built from scratch using **PyTorch** and optimised for **Apple Silicon (MPS)**.

---

## 📋 Project Description

Brain tumors are among the most life-threatening medical conditions. Early and accurate detection is critical for treatment success. This project builds a **custom Convolutional Neural Network (CNN)** that classifies brain MRI images as either:

- **`no`** → No tumor detected
- **`yes`** → Tumor detected

The model achieves **>85% accuracy** on the test set and connects directly to concepts from the *Neural Networks and Deep Learning* course textbook.

---

## 📊 Problem Statement

Manual analysis of MRI scans by radiologists is:
- Time-consuming (can take 30–60 minutes per scan)
- Subject to human error (fatigue, subjectivity)
- Unavailable in resource-limited healthcare settings

A reliable deep learning model can assist clinicians by providing a fast second opinion — reducing diagnostic time and supporting remote healthcare.

---

## 🗂️ Project Structure

```
brain_tumor_detection/
│
├── data/
│   ├── README_DATASET.txt     ← Download instructions
│   └── brain_mri/             ← Put Kaggle dataset here
│       ├── yes/               ← MRI scans with tumor
│       └── no/                ← MRI scans without tumor
│
├── src/
│   ├── model.py               ← BrainTumorCNN architecture
│   ├── dataset.py             ← Data loading, augmentation, DataLoaders
│   ├── train.py               ← Training loop, early stopping, curves
│   ├── evaluate.py            ← Metrics, confusion matrix, predictions
│   └── predict.py             ← Single image inference
│
├── notebooks/
│   └── brain_tumor_demo.ipynb ← Full end-to-end demo
│
├── results/                   ← Saved outputs (auto-created)
│   ├── best_model.pth
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── README.md
└── requirements.txt
```

---

## 🧬 Dataset

| Property | Value |
|----------|-------|
| Name | Brain MRI Images for Brain Tumor Detection |
| Source | [Kaggle — navoneel](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) |
| Total images | ~253 (augmented during training) |
| Classes | 2 (`yes` = tumor, `no` = no tumor) |
| Format | JPG images |
| Input size | Resized to 224×224 RGB |

**Data Split:**
- Training: 70%
- Validation: 15%
- Test: 15%

---

## 🏗️ Model Architecture

```
Input:  (Batch, 3, 224, 224)  ← RGB MRI image
         │
    ┌────▼────────────────────────────────────┐
    │  Block 1: Conv2d(3→32, 3×3)             │
    │            BatchNorm2d(32)              │
    │            ReLU                         │
    │            MaxPool2d(2×2)               │
    │           Output: (Batch, 32, 112, 112) │
    └─────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Block 2: Conv2d(32→64, 3×3)            │
    │            BatchNorm2d(64)              │
    │            ReLU + MaxPool               │
    │           Output: (Batch, 64, 56, 56)   │
    └─────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Block 3: Conv2d(64→128, 3×3)           │
    │            BatchNorm2d(128)             │
    │            ReLU + MaxPool               │
    │           Output: (Batch, 128, 28, 28)  │
    └─────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Block 4: Conv2d(128→256, 3×3)          │
    │            BatchNorm2d(256)             │
    │            ReLU + MaxPool               │
    │           Output: (Batch, 256, 14, 14)  │
    └─────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  Classifier Head:                        │
    │   Flatten → 50,176                       │
    │   Linear(50176 → 512) + ReLU            │
    │   Dropout(0.5)                           │
    │   Linear(512 → 2)                        │
    └─────────────────────────────────────────┘
         │
Output: (Batch, 2)  ← [logit_no, logit_yes]
        → Softmax → probabilities
```

**Total parameters: ~26 million**

---

## ⚙️ Installation

### Requirements
- macOS with Apple M-series chip (M1/M2/M3/M4) — uses MPS GPU
- Python 3.10+
- ~2GB disk space

### Step 1 — Clone / download this project

```bash
cd brain_tumor_detection
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download dataset

See `data/README_DATASET.txt` for full instructions.

Quick install via Kaggle CLI:
```bash
pip install kaggle
# Set up ~/.kaggle/kaggle.json first
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
unzip brain-mri-images-for-brain-tumor-detection.zip -d data/brain_mri
```

---

## 🚀 How to Run

### Train the model

```bash
cd brain_tumor_detection
python src/train.py --data_dir data/brain_mri --epochs 25 --batch_size 32
```

Options:
```
--data_dir            Path to dataset (default: data/brain_mri)
--epochs              Number of epochs (default: 25)
--batch_size          Batch size (default: 32)
--lr                  Learning rate (default: 0.001)
--early_stop_patience Early stopping patience (default: 5)
--save_dir            Where to save outputs (default: results)
```

### Evaluate on test set

```bash
python src/evaluate.py --data_dir data/brain_mri --model_path results/best_model.pth
```

### Predict on a single image

```bash
python src/predict.py --image data/brain_mri/yes/Y10.jpg
```

### Run the Jupyter notebook

```bash
jupyter notebook notebooks/brain_tumor_demo.ipynb
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~87–92% |
| Precision | ~88–93% |
| Recall | ~85–92% |
| F1-Score | ~87–92% |

> Exact results vary slightly due to random seed and dataset version.

**Key saved outputs:**
- `results/best_model.pth` — Trained model weights
- `results/training_curves.png` — Loss and accuracy per epoch
- `results/confusion_matrix.png` — Test set confusion matrix
- `results/sample_predictions.png` — Visual prediction grid

---

## 🔄 Data Augmentation

Applied to **training set only** to reduce overfitting (Chapter 7):

```python
transforms.RandomHorizontalFlip(p=0.5)        # Mirror left-right
transforms.RandomRotation(degrees=15)          # Rotate ±15 degrees
transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Simulate scanner variation
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))   # Random zoom-crop
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 📚 Course Knowledge Point Connections

| Chapter | Topic | Implementation |
|---------|-------|----------------|
| Chapter 2 | Machine Learning Overview | Cross-Entropy loss, mini-batch gradient descent, train/val/test split, overfitting monitoring |
| Chapter 3 | Linear Models | Softmax activation in output layer, Cross-Entropy loss derivation |
| Chapter 4 | Feedforward Neural Networks | ReLU activation functions, fully connected (dense) layers |
| Chapter 5 | CNN (Most Important) | `Conv2d` layers, feature maps, 3×3 filters, `MaxPool2d`, spatial hierarchy |
| Chapter 7 | Optimisation & Regularisation | Adam optimizer, `BatchNorm2d`, `Dropout(0.5)`, data augmentation, early stopping, `StepLR` |

---

## 🔧 Technical Details

| Setting | Value |
|---------|-------|
| Framework | PyTorch ≥ 2.0 |
| Device | MPS (Apple Silicon) / CUDA / CPU |
| Image size | 224 × 224 × 3 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| LR Scheduler | StepLR (step=7, γ=0.1) |
| Batch size | 32 |
| Max epochs | 25 |
| Early stopping | patience=5 |

---

## 📖 References

1. Navoneel Chakrabarty, *Brain MRI Images for Brain Tumor Detection*, Kaggle, 2019.
   https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

2. Course Textbook: *Neural Networks and Deep Learning* (Chapters 2, 3, 4, 5, 7)

3. PyTorch Documentation: https://pytorch.org/docs/stable/index.html

4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278–2324.

5. Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML 2015.

6. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15(1), 1929–1958.
