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
        Conv autoencoder with skip connections (U-Net style) for better reconstruction.
        Encoder: 3 -> 32 -> 64 -> 128. Decoder receives skip from encoder for higher PSNR/SSIM.
        """

        def __init__(self, in_channels: int = 3):
            super().__init__()
            self.enc1 = ConvBlock(in_channels, 32)   # -> 32
            self.enc2 = ConvBlock(32, 64)             # -> 64
            self.enc3 = ConvBlock(64, 128)           # -> 128

            # Decoder with skip: dec1 in = 128 + 64 (skip), dec2 in = 64 + 32 (skip), dec3 in = 32
            self.dec1 = nn.Conv2d(128 + 64, 64, 3, padding=1)
            self.dec2 = nn.Conv2d(64 + 32, 32, 3, padding=1)
            self.dec3 = nn.Conv2d(32, in_channels, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)

        def encode(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            return e1, e2, e3

        def decode(self, z, e2, e1):
            # z is e3 (128 ch). Concatenate e2 as skip and decode to 64
            d1 = self.relu(self.dec1(torch.cat([z, e2], dim=1)))
            d2 = self.relu(self.dec2(torch.cat([d1, e1], dim=1)))
            out = self.dec3(d2)
            return torch.sigmoid(out)

        def forward(self, x):
            e1, e2, e3 = self.encode(x)
            return self.decode(e3, e2, e1)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )
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
            n_batches = 0
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
                n_batches += 1
            mean_loss = epoch_loss / max(1, n_batches)
            losses.append(mean_loss)
            scheduler.step(mean_loss)
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
