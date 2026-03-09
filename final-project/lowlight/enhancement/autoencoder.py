"""
Lightweight convolutional autoencoder for low-light enhancement. MSE loss, modest number of epochs.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Optional PyTorch; fail gracefully if not installed
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None
    torch = None


if HAS_TORCH:

    class ConvBlock(nn.Module):
        def __init__(self, in_c: int, out_c: int):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class LightweightAE(nn.Module):
        """
        Simple conv autoencoder. Encoder: 3 -> 32 -> 64 -> 128. Decoder: 128 -> 64 -> 32 -> 3.
        """

        def __init__(self, in_channels: int = 3):
            super().__init__()
            # Encoder: 3 conv layers
            self.enc1 = ConvBlock(in_channels, 32)   # -> 32
            self.enc2 = ConvBlock(32, 64)             # -> 64
            self.enc3 = ConvBlock(64, 128)            # -> 128
            self.pool = nn.MaxPool2d(2, 2)           # optional; we keep spatial with stride 1

            self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
            self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
            self.dec3 = nn.Conv2d(32, in_channels, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)

        def encode(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            return e3

        def decode(self, z):
            d1 = self.relu(self.dec1(z))
            d2 = self.relu(self.dec2(d1))
            out = self.dec3(d2)
            return torch.sigmoid(out)

        def forward(self, x):
            return self.decode(self.encode(x))

    def build_autoencoder(in_channels: int = 3) -> "nn.Module":
        """Build the lightweight AE model."""
        return LightweightAE(in_channels=in_channels)

    def train_autoencoder(
        low_imgs: np.ndarray,
        normal_imgs: np.ndarray,
        epochs: int = 25,
        lr: float = 1e-3,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> Tuple["nn.Module", list]:
        """
        Train AE to map low -> normal. low_imgs, normal_imgs: (N,H,W,C) float [0,1].
        Returns: trained model, list of per-epoch MSE losses.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        model = build_autoencoder(in_channels=low_imgs.shape[-1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        # (N,H,W,C) -> (N,C,H,W)
        low_t = torch.from_numpy(np.ascontiguousarray(low_imgs.transpose(0, 3, 1, 2))).float().to(device)
        normal_t = torch.from_numpy(np.ascontiguousarray(normal_imgs.transpose(0, 3, 1, 2))).float().to(device)
        losses = []
        n = len(low_t)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x = low_t[idx]
                y = normal_t[idx]
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, (n + batch_size - 1) // batch_size))
        return model, losses

    def predict_autoencoder(model: "nn.Module", low_imgs: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """Enhance images with trained AE. low_imgs (N,H,W,C) float [0,1] -> (N,H,W,C) float [0,1]."""
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        x = torch.from_numpy(np.ascontiguousarray(low_imgs.transpose(0, 3, 1, 2))).float().to(device)
        with torch.no_grad():
            out = model(x)
        out = out.cpu().numpy().transpose(0, 2, 3, 1)
        return np.clip(out, 0, 1).astype(np.float64)

    def save_autoencoder(model: "nn.Module", path: str) -> None:
        """Save model state dict."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    def load_autoencoder(path: str, in_channels: int = 3, device: Optional[str] = None) -> "nn.Module":
        """Load model from state dict."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_autoencoder(in_channels=in_channels).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model

else:

    def build_autoencoder(in_channels: int = 3):
        raise RuntimeError("PyTorch is required for the autoencoder. Install with: pip install torch")

    def train_autoencoder(*args, **kwargs):
        raise RuntimeError("PyTorch is required for the autoencoder. Install with: pip install torch")

    def predict_autoencoder(*args, **kwargs):
        raise RuntimeError("PyTorch is required for the autoencoder. Install with: pip install torch")

    def save_autoencoder(*args, **kwargs):
        raise RuntimeError("PyTorch is required for the autoencoder. Install with: pip install torch")

    def load_autoencoder(*args, **kwargs):
        raise RuntimeError("PyTorch is required for the autoencoder. Install with: pip install torch")
