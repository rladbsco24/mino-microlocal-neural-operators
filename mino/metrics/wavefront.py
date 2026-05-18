from __future__ import annotations

import torch
from torch import Tensor, nn


def relative_l2(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    numerator = torch.linalg.vector_norm((prediction - target).reshape(prediction.shape[0], -1), dim=-1)
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=-1).clamp_min(eps)
    return numerator / denominator


def _complex_pair_view(field: Tensor) -> Tensor | None:
    """View channel pairs as finite complex fields when the data provides them."""

    channels = field.shape[1]
    if channels < 2 or channels % 2 != 0:
        return None
    batch, _, height, width = field.shape
    return field.reshape(batch, channels // 2, 2, height, width)


def complex_relative_l2(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative L2 in genuine real/imaginary channel pairs, with real fallback.

    High-frequency Helmholtz diagnostics should not manufacture an imaginary
    channel from a scalar field.  If the dataset has no real/imaginary channel
    pairs, this falls back to the ordinary real-valued relative L2.
    """

    pred_pairs = _complex_pair_view(prediction)
    target_pairs = _complex_pair_view(target)
    if pred_pairs is None or target_pairs is None:
        return relative_l2(prediction, target, eps=eps)
    diff = pred_pairs - target_pairs
    numerator = diff.square().sum(dim=2).flatten(1).sum(dim=1).sqrt()
    denominator = target_pairs.square().sum(dim=2).flatten(1).sum(dim=1).sqrt().clamp_min(eps)
    return numerator / denominator


def amplitude_relative_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative error of the scalar or complex amplitude envelope."""

    pred_pairs = _complex_pair_view(prediction)
    target_pairs = _complex_pair_view(target)
    if pred_pairs is not None and target_pairs is not None:
        pred_amp = pred_pairs.square().sum(dim=2).sqrt()
        target_amp = target_pairs.square().sum(dim=2).sqrt()
    else:
        pred_amp = prediction.abs()
        target_amp = target.abs()
    numerator = torch.linalg.vector_norm((pred_amp - target_amp).reshape(prediction.shape[0], -1), dim=-1)
    denominator = torch.linalg.vector_norm(target_amp.reshape(target.shape[0], -1), dim=-1).clamp_min(eps)
    return numerator / denominator


def boundary_trace_relative_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative error on the boundary receiver trace.

    This is a cheap finite proxy for receiver/far-field mismatch in outgoing
    Helmholtz rows.  It is not a Sommerfeld residual certificate; it records
    whether the predicted field agrees with the target on the domain boundary.
    """

    pred_trace = torch.cat(
        (
            prediction[..., 0, :],
            prediction[..., -1, :],
            prediction[..., 1:-1, 0],
            prediction[..., 1:-1, -1],
        ),
        dim=-1,
    )
    target_trace = torch.cat(
        (
            target[..., 0, :],
            target[..., -1, :],
            target[..., 1:-1, 0],
            target[..., 1:-1, -1],
        ),
        dim=-1,
    )
    numerator = torch.linalg.vector_norm((pred_trace - target_trace).reshape(prediction.shape[0], -1), dim=-1)
    denominator = torch.linalg.vector_norm(target_trace.reshape(target.shape[0], -1), dim=-1).clamp_min(eps)
    return numerator / denominator


def phase_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_spec = torch.fft.rfft2(prediction, norm="ortho")
    target_spec = torch.fft.rfft2(target, norm="ortho")
    pred_phase = torch.angle(pred_spec + eps)
    target_phase = torch.angle(target_spec + eps)
    wrapped = torch.atan2(torch.sin(pred_phase - target_phase), torch.cos(pred_phase - target_phase))
    return wrapped.abs().mean(dim=(-2, -1, -3))


def packet_consistency(prediction: Tensor, target: Tensor, patch_size: int = 16, stride: int = 8) -> Tensor:
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    pred_patches = unfold(prediction).transpose(1, 2)
    target_patches = unfold(target).transpose(1, 2)
    pred_energy = torch.linalg.vector_norm(pred_patches, dim=-1)
    target_energy = torch.linalg.vector_norm(target_patches, dim=-1).clamp_min(1e-8)
    return ((pred_energy - target_energy).abs() / target_energy).mean(dim=-1)


def _frequency_grids(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=device, dtype=dtype).view(1, width // 2 + 1)
    return fy, fx


def spectral_centroid(field: Tensor, eps: float = 1e-8) -> Tensor:
    """Return a global frequency centroid used as a cheap wavefront proxy."""
    _, _, height, width = field.shape
    spec = torch.fft.rfft2(field, norm="ortho")
    power = spec.abs().square().sum(dim=1)
    fy, fx = _frequency_grids(height, width, device=field.device, dtype=field.dtype)
    mass = power.sum(dim=(-2, -1)).clamp_min(eps)
    cy = (power * fy).sum(dim=(-2, -1)) / mass
    cx = (power * fx).sum(dim=(-2, -1)) / mass
    return torch.stack((cy, cx), dim=-1)


def high_frequency_fraction(field: Tensor, cutoff: float = 0.25, eps: float = 1e-8) -> Tensor:
    _, _, height, width = field.shape
    spec = torch.fft.rfft2(field, norm="ortho")
    power = spec.abs().square().sum(dim=1)
    fy, fx = _frequency_grids(height, width, device=field.device, dtype=field.dtype)
    radius = torch.sqrt(fy.square() + fx.square())
    high = power[..., radius >= cutoff].sum(dim=-1)
    total = power.sum(dim=(-2, -1)).clamp_min(eps)
    return high / total


def wavefront_transport_proxy(prediction: Tensor, target: Tensor) -> Tensor:
    """Frequency-centroid and high-frequency-energy mismatch.

    This is not a wavefront-set estimator. It is a deterministic, cheap proxy
    that reacts to transport or high-frequency damping errors that plain
    relative L2 can hide.
    """
    centroid_error = torch.linalg.vector_norm(
        spectral_centroid(prediction) - spectral_centroid(target),
        dim=-1,
    )
    high_error = (high_frequency_fraction(prediction) - high_frequency_fraction(target)).abs()
    return centroid_error + high_error


def packet_threshold_wavefront_localization_error(
    prediction: Tensor,
    target: Tensor,
    *,
    patch_size: int = 16,
    stride: int = 8,
    threshold_ratio: float = 0.10,
    eps: float = 1e-8,
) -> Tensor:
    """Hausdorff-style error between active packet-support proxies.

    This is a finite packet-threshold proxy for Corollary 4.10 in the paper,
    not a classical wavefront-set estimator.  It measures whether high-energy
    packet locations move to the same phase-space support after thresholding.
    """
    batch, channels, height, width = prediction.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    pred_patches = unfold(prediction).transpose(1, 2).reshape(batch, -1, channels, patch_size, patch_size)
    target_patches = unfold(target).transpose(1, 2).reshape(batch, -1, channels, patch_size, patch_size)
    pred_energy = torch.linalg.vector_norm(pred_patches.reshape(batch, pred_patches.shape[1], -1), dim=-1)
    target_energy = torch.linalg.vector_norm(target_patches.reshape(batch, target_patches.shape[1], -1), dim=-1)

    ys = torch.arange(0, height - patch_size + 1, stride, device=prediction.device, dtype=prediction.dtype)
    xs = torch.arange(0, width - patch_size + 1, stride, device=prediction.device, dtype=prediction.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers_y = (yy.reshape(-1) + 0.5 * patch_size) / max(height - 1, 1)
    centers_x = (xx.reshape(-1) + 0.5 * patch_size) / max(width - 1, 1)
    centers = torch.stack((centers_y, centers_x), dim=-1)

    def _local_centroid(patches: Tensor) -> Tensor:
        spectrum = torch.fft.rfft2(patches, norm="ortho")
        power = spectrum.abs().square().sum(dim=2)
        fy, fx = _frequency_grids(patch_size, patch_size, device=patches.device, dtype=patches.dtype)
        mass = power.sum(dim=(-2, -1), keepdim=False).clamp_min(eps)
        cy = (power * fy).sum(dim=(-2, -1)) / mass
        cx = (power * fx).sum(dim=(-2, -1)) / mass
        return torch.stack((cy, cx), dim=-1)

    pred_freq = _local_centroid(pred_patches)
    target_freq = _local_centroid(target_patches)
    pred_points = torch.cat((centers.unsqueeze(0).expand(batch, -1, -1), pred_freq), dim=-1)
    target_points = torch.cat((centers.unsqueeze(0).expand(batch, -1, -1), target_freq), dim=-1)

    errors: list[Tensor] = []
    for sample in range(batch):
        pred_threshold = threshold_ratio * pred_energy[sample].amax().clamp_min(eps)
        target_threshold = threshold_ratio * target_energy[sample].amax().clamp_min(eps)
        pred_active = pred_energy[sample] >= pred_threshold
        target_active = target_energy[sample] >= target_threshold
        if not bool(pred_active.any()) or not bool(target_active.any()):
            errors.append(prediction.new_tensor(0.0))
            continue
        distances = torch.cdist(pred_points[sample, pred_active], target_points[sample, target_active], p=2.0)
        directed_pred = distances.min(dim=1).values.mean()
        directed_target = distances.min(dim=0).values.mean()
        mass_error = (pred_active.to(prediction.dtype).mean() - target_active.to(prediction.dtype).mean()).abs()
        errors.append(0.5 * (directed_pred + directed_target) + mass_error)
    return torch.stack(errors)


def sobolev_relative_error(prediction: Tensor, target: Tensor, order: float = 1.0, eps: float = 1e-8) -> Tensor:
    _, _, height, width = prediction.shape
    fy, fx = _frequency_grids(height, width, device=prediction.device, dtype=prediction.dtype)
    weight = torch.pow(1.0 + fy.square() + fx.square(), order)
    pred_spec = torch.fft.rfft2(prediction, norm="ortho")
    target_spec = torch.fft.rfft2(target, norm="ortho")
    diff_norm = (weight * (pred_spec - target_spec).abs().square().sum(dim=1)).sum(dim=(-2, -1)).sqrt()
    target_norm = (weight * target_spec.abs().square().sum(dim=1)).sum(dim=(-2, -1)).sqrt().clamp_min(eps)
    return diff_norm / target_norm


def sobolev_h1_relative_error(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    return sobolev_relative_error(prediction, target, order=1.0, eps=eps)


def high_frequency_relative_error(prediction: Tensor, target: Tensor, cutoff: float = 0.25, eps: float = 1e-8) -> Tensor:
    _, _, height, width = prediction.shape
    pred_spec = torch.fft.rfft2(prediction, norm="ortho")
    target_spec = torch.fft.rfft2(target, norm="ortho")
    fy, fx = _frequency_grids(height, width, device=prediction.device, dtype=prediction.dtype)
    radius = torch.sqrt(fy.square() + fx.square())
    mask = radius >= cutoff
    diff_norm = ((pred_spec - target_spec).abs().square().sum(dim=1)[..., mask]).sum(dim=-1).sqrt()
    target_norm = (target_spec.abs().square().sum(dim=1)[..., mask]).sum(dim=-1).sqrt().clamp_min(eps)
    return diff_norm / target_norm


def spectral_order_proxy(field: Tensor, eps: float = 1e-8) -> Tensor:
    _, _, height, width = field.shape
    spectrum = torch.fft.rfft2(field, norm="ortho")
    power = spectrum.abs().square().sum(dim=1).clamp_min(eps)
    fy, fx = _frequency_grids(height, width, device=field.device, dtype=field.dtype)
    radius = torch.sqrt(fy.square() + fx.square())
    active = radius > 0.0
    log_radius = torch.log(radius[active].clamp_min(eps))
    log_power = torch.log(power[..., active])
    centered_radius = log_radius - log_radius.mean()
    centered_power = log_power - log_power.mean(dim=-1, keepdim=True)
    denominator = centered_radius.square().mean().clamp_min(eps)
    return (centered_power * centered_radius).mean(dim=-1) / denominator


def symbol_order_scaling_error(prediction: Tensor, target: Tensor) -> Tensor:
    return (spectral_order_proxy(prediction) - spectral_order_proxy(target)).abs()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
