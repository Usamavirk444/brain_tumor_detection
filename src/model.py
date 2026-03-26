import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block to prevent vanishing gradients (MRI feature extraction)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (adjust channels if needed)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class BrainTumorCNN(nn.Module):
    """Optimized CNN for brain tumor binary classification (Yes/No)"""
    def __init__(self, num_classes=2, dropout_rate=0.4, l2_lambda=5e-5):
        super().__init__()
        self.l2_lambda = l2_lambda

        # Convolutional Backbone (Residual Blocks)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
            ResidualBlock(32, 64, stride=2),        # 112 → 56
            ResidualBlock(64, 128, stride=2),       # 56 → 28
            ResidualBlock(128, 256, stride=2),      # 28 → 14
            nn.Dropout2d(p=0.1)                     # Light spatial dropout
        )

        # Classifier Head (small for your val/test size)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),           # 256×14×14 → 256×1×1
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

        # Weight initialization (critical for MRI)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_l2_loss(self):
        """L2 regularization (reduced for your augmented data)"""
        return self.l2_lambda * sum(torch.norm(p, 2) for p in self.parameters())

    def summary(self):
        """Simple model summary"""
        print("="*60)
        print("BrainTumorCNN (Optimized for Your Dataset)")
        print("="*60)
        print(f"Input: (Batch, 3, 224, 224) | Device: {next(self.parameters()).device}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Params: {total_params:,} | Trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("="*60)

# Quick test (run model.py alone to verify)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = BrainTumorCNN().to(device)
    model.summary()
    # Test forward pass with dummy data
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\n✅ Model test passed! Output shape: {output.shape}")