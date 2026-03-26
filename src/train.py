import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Import your model from model.py
from model import BrainTumorCNN

# ------------------------------------------------------------------------------
# FIX 1: Move ImageDataset to TOP LEVEL (avoids pickle error on Mac)
# ------------------------------------------------------------------------------
class ImageDataset(torch.utils.data.Dataset):
    """Standalone Dataset class (unpicklable if nested)"""
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

# ------------------------------------------------------------------------------
# Auto Dataset Loader (Matches your ../data/processed folder structure)
# ------------------------------------------------------------------------------
class BrainTumorDataset:
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        # Load all image paths and labels (0 = no tumor, 1 = yes tumor)
        self.train_paths, self.train_labels = self._load_folder(os.path.join(data_dir, "train"))
        self.val_paths, self.val_labels = self._load_folder(os.path.join(data_dir, "val"))
        self.test_paths, self.test_labels = self._load_folder(os.path.join(data_dir, "test"))

    def _load_folder(self, folder):
        """Load images from yes/no subfolders"""
        paths = []
        labels = []
        # Map "no" → 0, "yes" → 1 (matches your classification)
        for label, cls in enumerate(["no", "yes"]):
            cls_folder = os.path.join(folder, cls)
            if not os.path.exists(cls_folder):
                raise ValueError(f"Folder missing: {cls_folder} (check your dataset structure!)")
            # Load all image files (png/jpg/jpeg)
            for file in os.listdir(cls_folder):
                if file.lower().endswith(("png", "jpg", "jpeg")):
                    paths.append(os.path.join(cls_folder, file))
                    labels.append(label)
        print(f"Loaded {len(paths)} images from {folder} (yes: {labels.count(1)}, no: {labels.count(0)})")
        return paths, labels

    def get_dataloaders(self, batch_size=16):
        """Return train/val/test DataLoaders (FIX 2: num_workers=0 for Mac)"""
        # Train dataset (with mild augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),  # Mild flip (you have 12x aug already)
            transforms.RandomRotation(3),            # Small rotation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Val/Test dataset (NO augmentation)
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create DataLoaders (num_workers=0 to fix Mac multiprocessing error)
        train_loader = DataLoader(
            ImageDataset(self.train_paths, self.train_labels, train_transform),
            batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            ImageDataset(self.val_paths, self.val_labels, val_test_transform),
            batch_size=8, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            ImageDataset(self.test_paths, self.test_labels, val_test_transform),
            batch_size=8, shuffle=False, num_workers=0
        )
        return train_loader, val_loader, test_loader

# ------------------------------------------------------------------------------
# Training Function (customized for your data)
# ------------------------------------------------------------------------------
def train_model(data_dir="../data/processed", epochs=30):
    # Step 1: Set device (GPU/CPU/MPS)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nStarting training on device: {device}")

    # Step 2: Initialize model
    model = BrainTumorCNN(
        num_classes=2,
        dropout_rate=0.4,
        l2_lambda=5e-5
    ).to(device)
    model.summary()

    # Step 3: Load data (auto from ../data/processed folder)
    dataset = BrainTumorDataset(data_dir, transform=None)
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=16)

    # Step 4: Loss + Optimizer (fix 1.6:1 class imbalance)
    class_weights = torch.tensor([1404/884, 1.0]).to(device)  # Weight "no" more heavily
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,          # Slow LR for small val/test sets
        weight_decay=5e-5 # Match model's L2 lambda
    )
    # Step 5: LR scheduler (FIX 3: Removed verbose=True for older PyTorch)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.6, patience=4, min_lr=1e-6
    )
    print("✅ LR Scheduler initialized (will reduce LR if val F1 plateaus for 4 epochs)")

    # Step 6: Training loop
    best_val_f1 = 0.0
    best_test_metrics = None

    for epoch in range(epochs):
        # --------------------------
        # Train Phase
        # --------------------------
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) + model.get_l2_loss()
            
            # Backward pass + optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # --------------------------
        # Validation Phase
        # --------------------------
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # --------------------------
        # Calculate Metrics
        # --------------------------
        # Train metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_f1 = f1_score(train_labels, train_preds, average="weighted")
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))

        # Val metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        # Update LR scheduler
        scheduler.step(val_f1)

        # --------------------------
        # Save Best Model (by Val F1)
        # --------------------------
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Evaluate on test set
            test_preds, test_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            
            # Test metrics
            test_f1 = f1_score(test_labels, test_preds, average="weighted")
            test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
            best_test_metrics = {
                "f1": test_f1,
                "acc": test_acc,
                "confusion_matrix": confusion_matrix(test_labels, test_preds),
                "report": classification_report(test_labels, test_preds, target_names=["No Tumor", "Yes Tumor"])
            }

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_f1": best_val_f1,
                "best_test_f1": test_f1
            }, "../results/best_brain_tumor_model.pth")

            print(f"\n📌 EPOCH {epoch+1} — NEW BEST MODEL")
            print(f"Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")

        # --------------------------
        # Print Progress
        # --------------------------
        print(f"\nEPOCH {epoch+1}/{epochs}")
        print(f"Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print("-"*60)

    # --------------------------
    # Final Results
    # --------------------------
    print("\n" + "="*70)
    print("FINAL BEST RESULTS (Test Set)")
    print("="*70)
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Best Test F1: {best_test_metrics['f1']:.4f}")
    print(f"Best Test Accuracy: {best_test_metrics['acc']:.4f}")
    print("\nConfusion Matrix:")
    print(best_test_metrics["confusion_matrix"])
    print("\nClassification Report:")
    print(best_test_metrics["report"])
    print("="*70)

# ------------------------------------------------------------------------------
# Run Training (main entry point)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Correct path for your folder structure (src/ → ../data/processed)
    train_model(data_dir="../data/processed", epochs=30)