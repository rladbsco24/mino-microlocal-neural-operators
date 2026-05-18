from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import SpectralResidualBlock
from .wavepacket import LocalFourierWavepacketTokenizer


class ConvStack(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FNOStyleBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = SpectralResidualBlock(width)
        self.local = ConvStack(width, width, width)
        self.project = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        lifted = self.lift(x)
        hidden = lifted + self.spectral(lifted) + self.local(lifted)
        return self.project(hidden)


class WNOStyleBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.low = ConvStack(in_channels, width, width)
        self.high = ConvStack(width, width, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        coarse = self.down(x)
        coarse = self.low(coarse)
        coarse = self.up(coarse)
        coarse = coarse[..., : x.shape[-2], : x.shape[-1]]
        return self.high(coarse)


class PDNOStyleBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, patch_size: int = 16, stride: int = 8, max_modes: int = 12) -> None:
        super().__init__()
        self.tokenizer = LocalFourierWavepacketTokenizer(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
            max_modes=max_modes,
        )
        feature_dim = in_channels * 2
        self.symbol = nn.Sequential(
            nn.Linear(feature_dim + 5, 64),
            nn.GELU(),
            nn.Linear(64, out_channels * 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        encoding = self.tokenizer(x)
        metadata = encoding.metadata.unsqueeze(0).expand(encoding.features.shape[0], -1, -1)
        coeffs = self.symbol(torch.cat([encoding.features, metadata], dim=-1))
        return self.tokenizer.synthesize(coeffs, encoding)


class LocalKernelBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.stack = ConvStack(in_channels, width, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.stack(x)


class UNetStyleBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, width: int = 24) -> None:
        super().__init__()
        self.enc1 = ConvStack(in_channels, width, width)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvStack(width, width * 2, width * 2)
        self.bottleneck = ConvStack(width * 2, width * 2, width * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = ConvStack(width * 3, width, width)
        self.project = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(e2)
        up = self.up(b)[..., : e1.shape[-2], : e1.shape[-1]]
        fused = torch.cat([up, e1], dim=1)
        return self.project(self.dec(fused))


def build_baseline(name: str, in_channels: int = 1, out_channels: int = 1) -> nn.Module:
    normalized = name.lower()
    if normalized == "fnostyle":
        return FNOStyleBaseline(in_channels=in_channels, out_channels=out_channels)
    if normalized == "wnostyle":
        return WNOStyleBaseline(in_channels=in_channels, out_channels=out_channels)
    if normalized == "pdnostyle":
        return PDNOStyleBaseline(in_channels=in_channels, out_channels=out_channels)
    if normalized == "localkernel":
        return LocalKernelBaseline(in_channels=in_channels, out_channels=out_channels)
    if normalized == "unetstyle":
        return UNetStyleBaseline(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unknown baseline: {name}")
