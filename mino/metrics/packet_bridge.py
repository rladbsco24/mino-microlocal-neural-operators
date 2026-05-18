from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .wavefront import relative_l2


def spectral_energy_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_energy = torch.fft.rfft2(prediction, norm="ortho").abs().square()
    target_energy = torch.fft.rfft2(target, norm="ortho").abs().square()
    numerator = (pred_energy - target_energy).abs().flatten(1).sum(dim=-1)
    denominator = target_energy.flatten(1).sum(dim=-1).clamp_min(eps)
    return numerator / denominator


def high_frequency_energy_drift(prediction: Tensor, target: Tensor, cutoff_fraction: float = 0.35, eps: float = 1e-8) -> Tensor:
    height = int(prediction.shape[-2])
    width = int(prediction.shape[-1])
    ky = torch.fft.fftfreq(height, device=prediction.device).view(-1, 1)
    kx = torch.fft.rfftfreq(width, device=prediction.device).view(1, -1)
    radius = torch.sqrt(kx.square() + ky.square())
    mask = radius >= float(cutoff_fraction) * radius.max().clamp_min(eps)
    pred_energy = torch.fft.rfft2(prediction, norm="ortho").abs().square()
    target_energy = torch.fft.rfft2(target, norm="ortho").abs().square()
    pred_high = pred_energy[..., mask].sum(dim=-1)
    target_high = target_energy[..., mask].sum(dim=-1)
    return (pred_high - target_high).abs() / target_high.clamp_min(eps)


def enstrophy_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_enstrophy = 0.5 * prediction.square().flatten(1).mean(dim=-1)
    target_enstrophy = 0.5 * target.square().flatten(1).mean(dim=-1)
    return (pred_enstrophy - target_enstrophy).abs() / target_enstrophy.clamp_min(eps)


def packet_energy_cascade_error(prediction: Tensor, target: Tensor, bins: int = 8, eps: float = 1e-8) -> Tensor:
    height = int(prediction.shape[-2])
    width = int(prediction.shape[-1])
    ky = torch.fft.fftfreq(height, device=prediction.device).view(-1, 1)
    kx = torch.fft.rfftfreq(width, device=prediction.device).view(1, -1)
    radius = torch.sqrt(kx.square() + ky.square())
    normalized_radius = radius / radius.max().clamp_min(eps)
    pred_energy = torch.fft.rfft2(prediction, norm="ortho").abs().square()
    target_energy = torch.fft.rfft2(target, norm="ortho").abs().square()
    errors: list[Tensor] = []
    for index in range(bins):
        left = float(index) / float(bins)
        right = float(index + 1) / float(bins)
        mask = (normalized_radius >= left) & (normalized_radius < right if index < bins - 1 else normalized_radius <= right)
        if not bool(mask.any()):
            continue
        pred_bin = pred_energy[..., mask].sum(dim=-1)
        target_bin = target_energy[..., mask].sum(dim=-1)
        errors.append((pred_bin - target_bin).abs() / target_bin.clamp_min(eps))
    if not errors:
        return torch.zeros(prediction.shape[0], device=prediction.device, dtype=prediction.dtype)
    return torch.stack(errors, dim=0).mean(dim=0)


def packet_bridge_metrics(
    model: nn.Module,
    diagnostics: dict[str, object],
    target: Tensor,
    *,
    proxy_temperature: float = 0.05,
) -> dict[str, float]:
    prediction = diagnostics["prediction"]
    if not isinstance(prediction, Tensor):
        raise ValueError("diagnostics['prediction'] must be a tensor.")
    residual_budget = relative_l2(prediction, target).mean()
    transport_budget = torch.tensor(math.nan, device=target.device)
    symbol_budget = torch.tensor(math.nan, device=target.device)
    if hasattr(model, "proxy_losses_from_diagnostics"):
        proxy_losses = model.proxy_losses_from_diagnostics(
            diagnostics,
            target,
            proxy_temperature=proxy_temperature,
        )
        transport_budget = proxy_losses["transport_proxy"].detach()
        symbol_budget = proxy_losses["symbol_proxy"].detach()
    cascade = packet_energy_cascade_error(prediction, target).mean()
    return {
        "transport_budget": float(transport_budget.detach().cpu().item()),
        "symbol_budget": float(symbol_budget.detach().cpu().item()),
        "residual_budget": float(residual_budget.detach().cpu().item()),
        "packet_trajectory_consistency": float(torch.sqrt(transport_budget.clamp_min(0.0)).detach().cpu().item())
        if torch.isfinite(transport_budget)
        else math.nan,
        "packet_amplitude_error": float(symbol_budget.detach().cpu().item()) if torch.isfinite(symbol_budget) else math.nan,
        "packet_residual_energy": float(residual_budget.detach().cpu().item()),
        "packet_energy_cascade_error": float(cascade.detach().cpu().item()),
    }
