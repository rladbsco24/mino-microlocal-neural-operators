from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .baselines import (
    FNOStyleBaseline,
    HybridSpectralUNetCorrector,
    LocalKernelBaseline,
    PDNOStyleBaseline,
    UNetStyleBaseline,
    WNOStyleBaseline,
)
from .layers import (
    CanonicalPropagationLayer,
    DissipativePseudodifferentialBranch,
    IdentityPseudodifferentialBranch,
    SpectralResidualBlock,
    SymbolModulationLayer,
    _metadata_batch,
)
from .official_wrappers import build_official_model
from .wavepacket import (
    AnisotropicGaborWavepacketTokenizer,
    LocalFourierWavepacketTokenizer,
    MultiScaleWavepacketTokenizer,
    _build_rect_window,
    packet_transport_symbol_proxy,
)


def _normalize_tuple(value: tuple[int, ...] | list[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _build_tokenizer(
    *,
    in_channels: int,
    patch_size: int,
    stride: int,
    max_modes: int,
    window_type: str,
    mode_strategy: str,
    frame_type: str = "local_fft",
    frame_patch_sizes: tuple[int, ...] | list[int] | None = None,
    frame_strides: tuple[int, ...] | list[int] | None = None,
    frame_max_modes: tuple[int, ...] | list[int] | None = None,
) -> nn.Module:
    patch_sizes = _normalize_tuple(frame_patch_sizes)
    if frame_type == "gabor_gaussian":
        window_type = "gaussian"
        mode_strategy = "shell_balanced"
    elif frame_type == "multiscale_gabor":
        window_type = "gaussian"
        mode_strategy = "shell_balanced"
        patch_sizes = patch_sizes or (8, 16, 32)
        frame_strides = frame_strides or (4, 8, 16)
        frame_max_modes = frame_max_modes or (8, 16, max_modes)
    elif frame_type == "anisotropic_gabor":
        window_type = "gaussian"
        mode_strategy = "shell_balanced"
        short = max(4, patch_size // 2)
        return AnisotropicGaborWavepacketTokenizer(
            in_channels=in_channels,
            patch_shapes=((short, patch_size), (patch_size, short), (patch_size, patch_size)),
            stride_shapes=((max(1, short // 2), max(1, patch_size // 2)), (max(1, patch_size // 2), max(1, short // 2)), (max(1, patch_size // 2), max(1, patch_size // 2))),
            max_modes=(min(8, max_modes), min(8, max_modes), max_modes),
            window_type=window_type,
            mode_strategy=mode_strategy,
        )
    elif frame_type == "directional_gabor":
        window_type = "gaussian"
        mode_strategy = "shell_balanced"
        short = max(4, patch_size // 2)
        long = max(patch_size, patch_size * 2)
        return AnisotropicGaborWavepacketTokenizer(
            in_channels=in_channels,
            patch_shapes=(
                (short, patch_size),
                (patch_size, short),
                (short, long),
                (long, short),
                (patch_size, patch_size),
            ),
            stride_shapes=(
                (max(1, short // 2), max(1, patch_size // 2)),
                (max(1, patch_size // 2), max(1, short // 2)),
                (max(1, short // 2), max(1, long // 2)),
                (max(1, long // 2), max(1, short // 2)),
                (max(1, patch_size // 2), max(1, patch_size // 2)),
            ),
            max_modes=(
                min(8, max_modes),
                min(8, max_modes),
                min(12, max_modes),
                min(12, max_modes),
                max_modes,
            ),
            window_type=window_type,
            mode_strategy=mode_strategy,
        )
    elif frame_type != "local_fft":
        raise ValueError(f"Unsupported frame_type: {frame_type}")
    if not patch_sizes:
        return LocalFourierWavepacketTokenizer(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
            max_modes=max_modes,
            window_type=window_type,
            mode_strategy=mode_strategy,
        )
    strides = _normalize_tuple(frame_strides) or tuple(max(1, size // 2) for size in patch_sizes)
    max_modes_tuple = _normalize_tuple(frame_max_modes) or tuple(max_modes for _ in patch_sizes)
    if len(patch_sizes) == 1:
        return LocalFourierWavepacketTokenizer(
            in_channels=in_channels,
            patch_size=patch_sizes[0],
            stride=strides[0],
            max_modes=max_modes_tuple[0],
            window_type=window_type,
            mode_strategy=mode_strategy,
        )
    return MultiScaleWavepacketTokenizer(
        in_channels=in_channels,
        patch_sizes=patch_sizes,
        strides=strides,
        max_modes=max_modes_tuple,
        window_type=window_type,
        mode_strategy=mode_strategy,
    )


def _tokenizer_diagnostics(tokenizer: nn.Module, x: Tensor, encoding: object) -> dict[str, Tensor]:
    reconstructed = tokenizer.synthesize(encoding.features, encoding)
    input_energy = x.square().mean().clamp_min(1e-8)
    reconstruction_error = (reconstructed - x).square().mean() / input_energy
    feature_energy = torch.linalg.vector_norm(encoding.features, dim=-1)
    active_fraction = (feature_energy > 1e-6).to(x.dtype).mean()
    metadata = encoding.metadata
    covering_radius = metadata[..., 4].mean() if metadata.numel() else x.new_tensor(0.0)
    if metadata.shape[0] > 1:
        span = metadata[..., :2].amax(dim=0) - metadata[..., :2].amin(dim=0)
        phase_window_diameter = torch.linalg.vector_norm(span, dim=0)
    else:
        phase_window_diameter = x.new_tensor(0.0)
    return {
        "tokenizer_reconstruction_error": reconstruction_error,
        "tokenizer_active_fraction": active_fraction,
        "tokenizer_covering_radius": covering_radius,
        "tokenizer_phase_window_diameter": phase_window_diameter,
    }


def _frequency_order_proxy(metadata: Tensor, token_values: Tensor, eps: float = 1e-8) -> Tensor:
    if metadata.shape[-2] < 2:
        return token_values.new_tensor(0.0)
    metadata_batch = _metadata_batch(metadata.to(device=token_values.device, dtype=token_values.dtype), token_values)
    freq_norm = torch.linalg.vector_norm(metadata_batch[..., 2:4], dim=-1).clamp_min(eps)
    value_norm = torch.linalg.vector_norm(token_values, dim=-1).clamp_min(eps)
    log_freq = torch.log1p(freq_norm).reshape(-1)
    log_value = torch.log(value_norm).reshape(-1)
    centered_freq = log_freq - log_freq.mean()
    centered_value = log_value - log_value.mean()
    denominator = centered_freq.square().mean().clamp_min(eps)
    return (centered_freq * centered_value).mean() / denominator


def _finite_symbol_seminorm_proxy(
    metadata: Tensor,
    token_values: Tensor,
    *,
    max_tokens: int = 64,
    eps: float = 1e-6,
) -> Tensor:
    """Finite first-seminorm proxy for a packet symbol action.

    This is not a proof of membership in S^m_{1,0}.  It is an executable
    compact-window smoothness diagnostic: nearby packet metadata should not
    produce arbitrarily rough symbol actions.
    """

    if token_values.shape[-2] < 2:
        return token_values.new_tensor(0.0)
    metadata_batch = _metadata_batch(metadata.to(device=token_values.device, dtype=token_values.dtype), token_values)
    num_tokens = int(token_values.shape[-2])
    sample_count = min(max_tokens, num_tokens)
    if sample_count < 2:
        return token_values.new_tensor(0.0)
    sample_idx = torch.linspace(
        0,
        num_tokens - 1,
        sample_count,
        device=token_values.device,
    ).round().long().unique()
    coords = metadata_batch[:, sample_idx, :4]
    values = token_values[:, sample_idx, :]
    coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    value_diff = values.unsqueeze(2) - values.unsqueeze(1)
    coord_dist = torch.linalg.vector_norm(coord_diff, dim=-1)
    value_dist = torch.linalg.vector_norm(value_diff, dim=-1)
    off_diag = coord_dist > eps
    if not bool(off_diag.any()):
        return token_values.new_tensor(0.0)
    ratios = value_dist[off_diag] / coord_dist[off_diag].clamp_min(eps)
    return ratios.mean()


def _lowpass_field(field: Tensor, cutoff: float) -> Tensor:
    if cutoff <= 0.0:
        return field
    height, width = field.shape[-2:]
    fy = torch.fft.fftfreq(height, device=field.device, dtype=field.dtype).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=field.device, dtype=field.dtype).view(1, width // 2 + 1)
    radius = torch.sqrt(fy.square() + fx.square())
    mask = (radius <= cutoff).to(field.dtype).view(1, 1, height, width // 2 + 1)
    spectrum = torch.fft.rfft2(field, norm="ortho")
    return torch.fft.irfft2(spectrum * mask, s=(height, width), norm="ortho")


def _highpass_field(field: Tensor, cutoff: float) -> Tensor:
    if cutoff <= 0.0:
        return field
    return field - _lowpass_field(field, cutoff)


def _field_norm(field: Tensor) -> Tensor:
    return torch.linalg.vector_norm(field.reshape(field.shape[0], -1), dim=-1).mean()


def _transported_synthesis_warp(
    field: Tensor,
    token_features: Tensor,
    encoding: object,
    final_metadata: Tensor,
    *,
    scale: float = 1.0,
) -> Tensor:
    """Warp synthesized packets using the learned spatial metadata shift.

    The packet layers update phase-space metadata, but the local FFT/Gabor
    synthesizer folds coefficients back onto the original analysis grid.  This
    warp is a finite implementation of the transported-synthesis realization
    used by the theorem-facing model: packet-center motion must affect the
    field, not only the internal message-passing graph.
    """
    if scale == 0.0 or token_features.numel() == 0:
        return field
    batch, _, height, width = field.shape
    device = field.device
    dtype = field.dtype
    displacement = torch.zeros(batch, 2, height, width, device=device, dtype=dtype)
    normalizer = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
    token_slices = getattr(encoding, "token_slices", ()) or ((0, token_features.shape[1]),)
    patch_counts = getattr(encoding, "group_patch_counts", ()) or (getattr(encoding, "patch_count"),)
    mode_counts = getattr(encoding, "group_mode_counts", ()) or (getattr(encoding, "mode_count"),)
    patch_shapes = getattr(encoding, "group_patch_shapes", ()) or (getattr(encoding, "patch_shape"),)
    stride_shapes = getattr(encoding, "group_stride_shapes", ()) or tuple(
        (max(1, int(shape[0]) // 2), max(1, int(shape[1]) // 2))
        for shape in patch_shapes
    )
    initial_metadata = getattr(encoding, "metadata")
    for (start, stop), patch_count, mode_count, patch_shape, stride_shape in zip(
        token_slices,
        patch_counts,
        mode_counts,
        patch_shapes,
        stride_shapes,
        strict=True,
    ):
        if patch_count <= 0 or mode_count <= 0:
            continue
        patch_height, patch_width = int(patch_shape[0]), int(patch_shape[1])
        if height < patch_height or width < patch_width:
            continue
        group_features = token_features[:, start:stop, :]
        group_initial = initial_metadata[start:stop].to(device=device, dtype=dtype)
        group_initial = group_initial.unsqueeze(0).expand(batch, -1, -1)
        if final_metadata.dim() == 2:
            group_final = final_metadata[start:stop].to(device=device, dtype=dtype)
            group_final = group_final.unsqueeze(0).expand(batch, -1, -1)
        elif final_metadata.dim() == 3:
            group_final = final_metadata[:, start:stop, :].to(device=device, dtype=dtype)
            if group_final.shape[0] != batch:
                raise ValueError("Batched final metadata must match the field batch dimension.")
        else:
            raise ValueError("final_metadata must have shape (tokens, dim) or (batch, tokens, dim).")
        if group_features.shape[1] != patch_count * mode_count:
            continue
        energy = torch.linalg.vector_norm(group_features.reshape(batch, patch_count, mode_count, -1), dim=-1)
        weights = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        token_shift = (group_final[..., :2] - group_initial[..., :2]).reshape(batch, patch_count, mode_count, 2)
        patch_shift = (weights.unsqueeze(-1) * token_shift).sum(dim=2)
        patch_weight = energy.mean(dim=-1).clamp_min(1e-8)

        values = torch.cat(
            [
                patch_shift[:, :, 0:1] * patch_weight.unsqueeze(-1),
                patch_shift[:, :, 1:2] * patch_weight.unsqueeze(-1),
                patch_weight.unsqueeze(-1),
            ],
            dim=-1,
        )
        values = values.transpose(1, 2).reshape(batch, 3, patch_count)
        fold_values = values.repeat_interleave(patch_height * patch_width, dim=1)
        stride_y = max(1, (height - patch_height) // max(int(round(patch_count**0.5)) - 1, 1))
        grid_cols = max(1, (width - patch_width) // max(stride_y, 1) + 1)
        grid_rows = max(1, patch_count // grid_cols)
        if grid_rows * grid_cols != patch_count:
            # Fall back to the standard half-overlap convention used by the
            # tokenizer families when the rectangular grid is ambiguous.
            stride_y = max(1, patch_height // 2)
            stride_x = max(1, patch_width // 2)
        else:
            stride_x = max(1, (width - patch_width) // max(grid_cols - 1, 1))
        try:
            folded = F.fold(
                fold_values,
                output_size=(height, width),
                kernel_size=(patch_height, patch_width),
                stride=(int(stride_shape[0]), int(stride_shape[1])),
            )
        except RuntimeError:
            stride_y = max(1, patch_height // 2)
            stride_x = max(1, patch_width // 2)
            folded = F.fold(
                fold_values,
                output_size=(height, width),
                kernel_size=(patch_height, patch_width),
                stride=(stride_y, stride_x),
            )
        displacement[:, 0:1] = displacement[:, 0:1] + folded[:, 0:1]
        displacement[:, 1:2] = displacement[:, 1:2] + folded[:, 1:2]
        normalizer = normalizer + folded[:, 2:3]
    shift = displacement / normalizer.clamp_min(1e-8)
    if not torch.isfinite(shift).all():
        return field
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    sample_y = (yy.unsqueeze(0) - scale * shift[:, 0]).clamp(0.0, 1.0)
    sample_x = (xx.unsqueeze(0) - scale * shift[:, 1]).clamp(0.0, 1.0)
    grid = torch.stack((2.0 * sample_x - 1.0, 2.0 * sample_y - 1.0), dim=-1)
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _transported_synthesis_atom_splat(
    field: Tensor,
    token_features: Tensor,
    encoding: object,
    final_metadata: Tensor,
    *,
    scale: float = 1.0,
    chunk_size: int = 128,
) -> Tensor:
    """Synthesize moved Gabor-like packet atoms from final metadata.

    The default executable path warps an already reconstructed field.  This
    path instead places local complex Fourier/Gabor atoms at their transported
    centers and transported frequencies.  It is still an executable
    approximation to the ideal frame synthesis, but it tests whether the main
    bottleneck is the previous field-level warp rather than learned transport.
    """
    if scale == 0.0 or token_features.numel() == 0:
        return field
    if token_features.shape[-1] % 2 != 0:
        raise ValueError("Token feature dimension must pack real/imaginary channels.")
    batch, _, height, width = field.shape
    channels = token_features.shape[-1] // 2
    if channels != field.shape[1]:
        return field
    device = field.device
    dtype = field.dtype
    metadata = final_metadata
    if metadata.dim() == 2:
        metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
    elif metadata.dim() == 3:
        if metadata.shape[0] != batch:
            raise ValueError("Batched final metadata must match the field batch dimension.")
    else:
        raise ValueError("final_metadata must have shape (tokens, dim) or (batch, tokens, dim).")
    metadata = metadata.to(device=device, dtype=dtype)
    if metadata.shape[1] != token_features.shape[1]:
        return field

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    yy = yy.view(1, 1, height, width)
    xx = xx.view(1, 1, height, width)

    out = torch.zeros(batch, channels, height, width, device=device, dtype=dtype)
    normalizer = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)
    token_slices = getattr(encoding, "token_slices", ()) or ((0, token_features.shape[1]),)
    patch_shapes = getattr(encoding, "group_patch_shapes", ()) or (getattr(encoding, "patch_shape"),)
    mode_counts = getattr(encoding, "group_mode_counts", ()) or (getattr(encoding, "mode_count"),)
    eps = torch.finfo(dtype).eps
    for (start, stop), patch_shape, mode_count in zip(token_slices, patch_shapes, mode_counts, strict=True):
        patch_height, patch_width = int(patch_shape[0]), int(patch_shape[1])
        patch_area_sqrt = math.sqrt(max(patch_height * patch_width, 1))
        group_features = token_features[:, start:stop, :].to(dtype=dtype)
        group_metadata = metadata[:, start:stop, :]
        for chunk_start in range(0, group_features.shape[1], chunk_size):
            chunk_stop = min(chunk_start + chunk_size, group_features.shape[1])
            coeff = group_features[:, chunk_start:chunk_stop, :].reshape(batch, -1, channels, 2)
            meta = group_metadata[:, chunk_start:chunk_stop, :]
            center_y = meta[..., 0].view(batch, -1, 1, 1)
            center_x = meta[..., 1].view(batch, -1, 1, 1)
            freq_y = meta[..., 2].view(batch, -1, 1, 1)
            freq_x = meta[..., 3].view(batch, -1, 1, 1)
            packet_scale = meta[..., 4].abs().clamp_min(1.0 / float(max(height, width))).view(batch, -1, 1, 1)
            sigma = (0.35 * packet_scale).clamp_min(1.0 / float(max(height, width)))
            dy_norm = yy - center_y
            dx_norm = xx - center_x
            dy_pix = dy_norm * float(max(height - 1, 1))
            dx_pix = dx_norm * float(max(width - 1, 1))
            window = torch.exp(-0.5 * (dy_norm.square() + dx_norm.square()) / sigma.square().clamp_min(eps))
            phase = 2.0 * math.pi * scale * (freq_y * dy_pix + freq_x * dx_pix)
            cos_phase = torch.cos(phase)
            sin_phase = torch.sin(phase)
            real = coeff[..., 0].permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            imag = coeff[..., 1].permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            nonzero_freq = ((freq_y.abs() + freq_x.abs()) > 1e-7).to(dtype)
            conjugate_factor = 1.0 + nonzero_freq
            atom = conjugate_factor.unsqueeze(1) * (real * cos_phase.unsqueeze(1) - imag * sin_phase.unsqueeze(1))
            atom = atom * window.unsqueeze(1) / patch_area_sqrt
            out = out + atom.sum(dim=2)
            normalizer = normalizer + window.square().sum(dim=1, keepdim=True) / float(max(int(mode_count), 1))
    return out / normalizer.clamp_min(1e-4)


def _group_patch_textures(
    token_features: Tensor,
    encoding: object,
    token_start: int,
    token_stop: int,
    patch_count: int,
    mode_count: int,
    patch_shape: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Reconstruct local iFFT patch textures for one packet group."""
    batch = token_features.shape[0]
    patch_height, patch_width = int(patch_shape[0]), int(patch_shape[1])
    group_features = token_features[:, token_start:token_stop, :]
    if group_features.shape[1] != patch_count * mode_count:
        raise ValueError("Group feature count does not match patch/mode metadata.")
    if group_features.shape[-1] % 2 != 0:
        raise ValueError("Token feature dimension must pack real/imaginary channels.")
    channels = group_features.shape[-1] // 2
    coeffs = group_features.reshape(batch, patch_count, mode_count, channels, 2)
    coeffs = coeffs.permute(0, 3, 1, 2, 4).contiguous()
    complex_coeffs = torch.complex(coeffs[..., 0], coeffs[..., 1])

    initial_metadata = getattr(encoding, "metadata")[token_start:token_stop].to(
        device=token_features.device,
        dtype=token_features.dtype,
    )
    mode_metadata = initial_metadata.reshape(patch_count, mode_count, -1)[0, :, :]
    signed_ky = torch.round(mode_metadata[:, 2] * float(patch_height)).to(torch.long)
    signed_kx = torch.round(mode_metadata[:, 3] * float(patch_width)).to(torch.long)
    ky = torch.where(signed_ky < 0, signed_ky + patch_height, signed_ky).clamp(0, patch_height - 1)
    kx = signed_kx.clamp(0, patch_width // 2)

    spectrum = torch.zeros(
        batch,
        channels,
        patch_count,
        patch_height,
        patch_width // 2 + 1,
        dtype=complex_coeffs.dtype,
        device=complex_coeffs.device,
    )
    spectrum[..., ky, kx] = complex_coeffs
    patches = torch.fft.irfft2(spectrum, s=(patch_height, patch_width), norm="ortho")
    window = _build_rect_window((patch_height, patch_width), token_features.device, token_features.dtype, "gaussian")
    patches = patches * window.view(1, 1, 1, patch_height, patch_width)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    return patches, window.square()


def _transported_synthesis_patch_fold(
    field: Tensor,
    token_features: Tensor,
    encoding: object,
    final_metadata: Tensor,
    *,
    scale: float = 1.0,
    chunk_size: int = 24,
) -> Tensor:
    """Move reconstructed local iFFT patches and fold them at final centers."""
    if scale == 0.0 or token_features.numel() == 0:
        return field
    batch, channels, height, width = field.shape
    if token_features.shape[-1] // 2 != channels:
        return field
    device = field.device
    dtype = field.dtype
    metadata = final_metadata
    if metadata.dim() == 2:
        metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
    elif metadata.dim() == 3:
        if metadata.shape[0] != batch:
            raise ValueError("Batched final metadata must match the field batch dimension.")
    else:
        raise ValueError("final_metadata must have shape (tokens, dim) or (batch, tokens, dim).")
    metadata = metadata.to(device=device, dtype=dtype)
    initial_metadata = getattr(encoding, "metadata").to(device=device, dtype=dtype)

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    yy = yy.view(1, height, width)
    xx = xx.view(1, height, width)
    output = torch.zeros(batch, channels, height, width, device=device, dtype=dtype)
    normalizer = torch.zeros(batch, 1, height, width, device=device, dtype=dtype)

    token_slices = getattr(encoding, "token_slices", ()) or ((0, token_features.shape[1]),)
    patch_counts = getattr(encoding, "group_patch_counts", ()) or (getattr(encoding, "patch_count"),)
    mode_counts = getattr(encoding, "group_mode_counts", ()) or (getattr(encoding, "mode_count"),)
    patch_shapes = getattr(encoding, "group_patch_shapes", ()) or (getattr(encoding, "patch_shape"),)
    for (start, stop), patch_count, mode_count, patch_shape in zip(
        token_slices,
        patch_counts,
        mode_counts,
        patch_shapes,
        strict=True,
    ):
        patches, window_sq = _group_patch_textures(token_features, encoding, start, stop, patch_count, mode_count, patch_shape)
        patch_height, patch_width = int(patch_shape[0]), int(patch_shape[1])
        group_features = token_features[:, start:stop, :].reshape(batch, patch_count, mode_count, -1)
        energy = torch.linalg.vector_norm(group_features, dim=-1).clamp_min(1e-8)
        weights = energy / energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        group_initial = initial_metadata[start:stop].reshape(patch_count, mode_count, -1)
        group_final = metadata[:, start:stop, :].reshape(batch, patch_count, mode_count, -1)
        initial_centers = group_initial[:, 0, :2].unsqueeze(0).expand(batch, -1, -1)
        final_centers = (weights.unsqueeze(-1) * group_final[..., :2]).sum(dim=2)
        centers = initial_centers + scale * (final_centers - initial_centers)
        for chunk_start in range(0, patch_count, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, patch_count)
            chunk_patches = patches[:, chunk_start:chunk_stop, :, :, :].reshape(-1, channels, patch_height, patch_width)
            chunk_centers = centers[:, chunk_start:chunk_stop, :].reshape(-1, 2)
            local_y = (yy - chunk_centers[:, 0:1].view(-1, 1, 1)) * float(max(height - 1, 1)) + 0.5 * float(patch_height - 1)
            local_x = (xx - chunk_centers[:, 1:2].view(-1, 1, 1)) * float(max(width - 1, 1)) + 0.5 * float(patch_width - 1)
            grid_y = 2.0 * local_y / float(max(patch_height - 1, 1)) - 1.0
            grid_x = 2.0 * local_x / float(max(patch_width - 1, 1)) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1)
            sampled = F.grid_sample(chunk_patches, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = sampled.reshape(batch, chunk_stop - chunk_start, channels, height, width)
            output = output + sampled.sum(dim=1)

            window_patch = window_sq.view(1, 1, patch_height, patch_width).expand(
                chunk_patches.shape[0],
                -1,
                -1,
                -1,
            )
            sampled_norm = F.grid_sample(window_patch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled_norm = sampled_norm.reshape(batch, chunk_stop - chunk_start, 1, height, width)
            normalizer = normalizer + sampled_norm.sum(dim=1)
    return output / normalizer.clamp_min(1e-4)


def _transported_synthesis(
    field: Tensor,
    token_features: Tensor,
    encoding: object,
    final_metadata: Tensor,
    *,
    scale: float,
    mode: str,
) -> Tensor:
    if mode == "warp":
        return _transported_synthesis_warp(field, token_features, encoding, final_metadata, scale=scale)
    if mode == "atom_splat":
        return _transported_synthesis_atom_splat(field, token_features, encoding, final_metadata, scale=scale)
    if mode == "patch_fold":
        return _transported_synthesis_patch_fold(field, token_features, encoding, final_metadata, scale=scale)
    raise ValueError(f"Unsupported transported_synthesis_mode: {mode}")


def _zero_like_field(reference: Tensor) -> Tensor:
    return reference.new_zeros(reference.shape)


def _transported_input_carrier(
    skip_field: Tensor,
    encoding: object,
    final_metadata: Tensor | None,
    *,
    transported_synthesis_scale: float,
    transported_input_scale: float,
    highpass_cutoff: float,
    transported_synthesis_mode: str = "warp",
) -> Tensor:
    """Move the input packet cloud through the learned metadata.

    This is deliberately stricter than warping the learned output coefficients:
    the high-frequency carrier comes from the input analysis coefficients, so a
    wrong or disabled transport branch leaves the carrier at the wrong place.
    """
    if transported_input_scale == 0.0 or final_metadata is None:
        return skip_field.new_zeros(skip_field.shape)
    transported = _transported_synthesis(
        skip_field,
        getattr(encoding, "features"),
        encoding,
        final_metadata,
        scale=transported_synthesis_scale,
        mode=transported_synthesis_mode,
    )
    return transported_input_scale * _highpass_field(transported, highpass_cutoff)


class ScaleAwareTokenNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.scale_proj = nn.Linear(1, width)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)

    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        scale = _metadata_batch(metadata, features)[..., 4:5]
        return self.norm(features) + self.scale_proj(scale)


def _make_branch_weights(
    *,
    features: Tensor,
    metadata: Tensor,
    num_branches: int,
    branch_routing: str,
    router: nn.Module | None,
    branch_prior_strength: float = 0.0,
) -> Tensor:
    if num_branches <= 1:
        return features.new_ones((*features.shape[:2], 1))
    if branch_routing == "single":
        weights = features.new_zeros((*features.shape[:2], num_branches))
        weights[..., 0] = 1.0
        return weights
    if branch_routing == "uniform":
        return features.new_full((*features.shape[:2], num_branches), 1.0 / float(num_branches))
    metadata_batch = _metadata_batch(metadata, features)
    if branch_routing in {"frequency_softmax", "metadata_frequency_softmax"}:
        freq = metadata_batch[..., 2:4]
        freq_norm = torch.linalg.vector_norm(freq, dim=-1, keepdim=True).clamp_min(1e-8)
        angle = torch.atan2(freq[..., :1], freq[..., 1:2])
        centers = torch.linspace(
            -math.pi,
            math.pi,
            steps=num_branches + 1,
            device=features.device,
            dtype=features.dtype,
        )[:-1].view(1, 1, num_branches)
        directional_alignment = torch.cos(angle - centers)
        high_frequency_gate = freq_norm / (freq_norm + 0.05)
        prior_logits = float(branch_prior_strength) * high_frequency_gate * directional_alignment
        if branch_routing == "frequency_softmax":
            return torch.softmax(prior_logits, dim=-1)
    else:
        prior_logits = features.new_zeros((*features.shape[:2], num_branches))
    if branch_routing not in {"metadata_softmax", "metadata_frequency_softmax"}:
        raise ValueError(f"Unsupported branch_routing: {branch_routing}")
    if router is None:
        raise ValueError(f"{branch_routing} branch routing requires a router module.")
    learned_logits = router(torch.cat([features, metadata_batch], dim=-1))
    return torch.softmax(learned_logits + prior_logits, dim=-1)


def _branch_diagnostics(weights: Tensor, branch_metadata: Tensor) -> dict[str, Tensor]:
    eps = torch.finfo(weights.dtype).eps
    entropy = -(weights.clamp_min(eps) * weights.clamp_min(eps).log()).sum(dim=-1).mean()
    usage = weights.mean(dim=(0, 1))
    diversity = 1.0 - usage.square().sum()
    usage_max = usage.max()
    if branch_metadata.shape[2] > 1:
        centers = branch_metadata[..., :2]
        pairwise = torch.cdist(
            centers.reshape(-1, centers.shape[2], 2),
            centers.reshape(-1, centers.shape[2], 2),
        )
        spread = pairwise.mean()
    else:
        spread = weights.new_tensor(0.0)
    return {
        "branch_entropy": entropy,
        "branch_diversity": diversity,
        "branch_usage_max": usage_max,
        "branch_spread": spread,
    }


class MiNOCoreBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        metadata_dim: int = 5,
        hidden_dim: int = 128,
        transport_scale: float = 0.03,
        transport_stencil: int = 4,
        transport_parameterization: str = "mlp_displacement",
        sparse_topk: bool = False,
        wavefront_confidence_scale: float = 0.0,
        symbol_order: float = 0.0,
        symbol_parameterization: str = "mlp",
        helmholtz_shell_radius: float = 0.0,
        helmholtz_refractive_index: float = 1.0,
        helmholtz_absorption: float = 0.05,
        helmholtz_resolvent_cap: float = 6.0,
        num_canonical_branches: int = 1,
        branch_routing: str = "metadata_softmax",
        branch_prior_strength: float = 0.0,
        branch_synthesis: str = "sum",
        edge_symbol_parameterization: str = "none",
        edge_symbol_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_canonical_branches = max(int(num_canonical_branches), 1)
        self.branch_routing = branch_routing
        self.branch_prior_strength = float(branch_prior_strength)
        self.branch_synthesis = branch_synthesis
        if self.branch_synthesis != "sum":
            raise ValueError(f"Unsupported branch_synthesis: {branch_synthesis}")
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.propagation = CanonicalPropagationLayer(
            width,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            transport_scale=transport_scale,
            stencil_size=transport_stencil,
            confidence_gating=False,
            transport_parameterization=transport_parameterization,
            sparse_topk=sparse_topk,
            wavefront_confidence_scale=wavefront_confidence_scale,
            edge_symbol_parameterization=edge_symbol_parameterization,
            edge_symbol_strength=edge_symbol_strength,
        )
        self.symbol = SymbolModulationLayer(
            width,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            symbol_order=symbol_order,
            symbol_parameterization=symbol_parameterization,
            helmholtz_shell_radius=helmholtz_shell_radius,
            helmholtz_refractive_index=helmholtz_refractive_index,
            helmholtz_absorption=helmholtz_absorption,
            helmholtz_resolvent_cap=helmholtz_resolvent_cap,
        )
        if self.num_canonical_branches > 1:
            self.branch_propagations = nn.ModuleList(
                [
                    self.propagation,
                    *[
                        CanonicalPropagationLayer(
                            width,
                            metadata_dim=metadata_dim,
                            hidden_dim=hidden_dim,
                            transport_scale=transport_scale,
                            stencil_size=transport_stencil,
                            confidence_gating=False,
                            transport_parameterization=transport_parameterization,
                            sparse_topk=sparse_topk,
                            wavefront_confidence_scale=wavefront_confidence_scale,
                            edge_symbol_parameterization=edge_symbol_parameterization,
                            edge_symbol_strength=edge_symbol_strength,
                        )
                        for _ in range(self.num_canonical_branches - 1)
                    ],
                ]
            )
            self.branch_symbols = nn.ModuleList(
                [
                    self.symbol,
                    *[
                        SymbolModulationLayer(
                            width,
                            metadata_dim=metadata_dim,
                            hidden_dim=hidden_dim,
                            symbol_order=symbol_order,
                            symbol_parameterization=symbol_parameterization,
                            helmholtz_shell_radius=helmholtz_shell_radius,
                            helmholtz_refractive_index=helmholtz_refractive_index,
                            helmholtz_absorption=helmholtz_absorption,
                            helmholtz_resolvent_cap=helmholtz_resolvent_cap,
                        )
                        for _ in range(self.num_canonical_branches - 1)
                    ],
                ]
            )
            self.branch_router = nn.Linear(width + metadata_dim, self.num_canonical_branches)
        else:
            self.branch_propagations = nn.ModuleList()
            self.branch_symbols = nn.ModuleList()
            self.branch_router = None
        self.last_branch_weights: Tensor | None = None
        self.last_branch_metadata: Tensor | None = None

    def forward(
        self,
        features: Tensor,
        metadata: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        metadata_before = _metadata_batch(metadata, features)
        if self.num_canonical_branches <= 1:
            propagated, metadata = self.propagation(self.norm1(features), metadata)
            features = features + propagated
            symbol = self.symbol(
                self.norm2(features),
                metadata,
                getattr(self.propagation, "last_local_tube_coordinate", None),
            )
            features = features + symbol
            branch_weights = features.new_ones((*features.shape[:2], 1))
            branch_metadata = metadata.unsqueeze(2)
        else:
            branch_weights = _make_branch_weights(
                features=features,
                metadata=metadata,
                num_branches=self.num_canonical_branches,
                branch_routing=self.branch_routing,
                router=self.branch_router,
                branch_prior_strength=self.branch_prior_strength,
            )
            branch_input = self.norm1(features)
            propagated_items: list[Tensor] = []
            symbol_items: list[Tensor] = []
            metadata_items: list[Tensor] = []
            for propagation, symbol_layer in zip(self.branch_propagations, self.branch_symbols, strict=True):
                branch_propagated, branch_metadata_item = propagation(branch_input, metadata)
                branch_features = features + branch_propagated
                branch_symbol = symbol_layer(
                    self.norm2(branch_features),
                    branch_metadata_item,
                    getattr(propagation, "last_local_tube_coordinate", None),
                )
                propagated_items.append(branch_propagated)
                symbol_items.append(branch_symbol)
                metadata_items.append(_metadata_batch(branch_metadata_item, features))
            propagated_stack = torch.stack(propagated_items, dim=2)
            symbol_stack = torch.stack(symbol_items, dim=2)
            branch_metadata = torch.stack(metadata_items, dim=2)
            propagated = (branch_weights.unsqueeze(-1) * propagated_stack).sum(dim=2)
            symbol = (branch_weights.unsqueeze(-1) * symbol_stack).sum(dim=2)
            metadata = (branch_weights.unsqueeze(-1) * branch_metadata).sum(dim=2)
            features = features + propagated + symbol
        self.last_branch_weights = branch_weights.detach()
        self.last_branch_metadata = branch_metadata.detach()
        if return_diagnostics:
            diagnostics = {
                "transport_norm": torch.linalg.vector_norm(propagated, dim=-1).mean(),
                "symbol_norm": torch.linalg.vector_norm(symbol, dim=-1).mean(),
                "symbol_order_proxy": _frequency_order_proxy(metadata, symbol),
                "symbol_seminorm_proxy": _finite_symbol_seminorm_proxy(metadata, symbol),
                "edge_symbol_deviation_proxy": getattr(
                    self.propagation,
                    "last_edge_symbol_deviation_proxy",
                    symbol.new_tensor(0.0),
                ),
                "canonical_defect_proxy": getattr(
                    self.propagation,
                    "last_canonical_defect_proxy",
                    symbol.new_tensor(0.0),
                ),
                "wavefront_confidence_proxy": getattr(
                    self.propagation,
                    "last_wavefront_confidence_proxy",
                    symbol.new_tensor(0.0),
                ),
                "metadata_shift": torch.linalg.vector_norm(metadata[..., :4] - metadata_before[..., :4], dim=-1).mean(),
                "local_tube_coordinate_norm": torch.linalg.vector_norm(
                    getattr(self.propagation, "last_local_tube_coordinate", symbol.new_zeros(*symbol.shape[:2], 4)),
                    dim=-1,
                ).mean(),
                **_branch_diagnostics(branch_weights, branch_metadata),
            }
            if getattr(self.symbol, "symbol_parameterization", "") == "helmholtz_resolvent":
                diagnostics.update(self.symbol.helmholtz_resolvent_diagnostics(metadata, symbol))
            return features, metadata, diagnostics
        return features, metadata


class MiNOPlusBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        metadata_dim: int = 5,
        hidden_dim: int = 128,
        transport_scale: float = 0.03,
        transport_stencil: int = 8,
        transport_parameterization: str = "mlp_displacement",
        sparse_topk: bool = False,
        wavefront_confidence_scale: float = 0.0,
        pdo_symbol_scale: float = 0.0,
        pdo_symbol_order: float = -2.0,
        dissipative_symbol_scale: float = 0.0,
        dissipative_time_step: float = 1.0,
        symbol_order: float = 0.0,
        symbol_parameterization: str = "mlp",
        helmholtz_shell_radius: float = 0.0,
        helmholtz_refractive_index: float = 1.0,
        helmholtz_absorption: float = 0.05,
        helmholtz_resolvent_cap: float = 6.0,
        token_refine_scale: float = 1.0,
        num_canonical_branches: int = 1,
        branch_routing: str = "metadata_softmax",
        branch_prior_strength: float = 0.0,
        branch_synthesis: str = "sum",
        edge_symbol_parameterization: str = "none",
        edge_symbol_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_canonical_branches = max(int(num_canonical_branches), 1)
        self.branch_routing = branch_routing
        self.branch_prior_strength = float(branch_prior_strength)
        self.branch_synthesis = branch_synthesis
        if self.branch_synthesis != "sum":
            raise ValueError(f"Unsupported branch_synthesis: {branch_synthesis}")
        self.pdo_symbol_scale = float(pdo_symbol_scale)
        self.dissipative_symbol_scale = float(dissipative_symbol_scale)
        self.token_refine_scale = float(token_refine_scale)
        self.norm1 = ScaleAwareTokenNorm(width)
        self.norm2 = ScaleAwareTokenNorm(width)
        self.propagation = CanonicalPropagationLayer(
            width,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            transport_scale=transport_scale,
            stencil_size=transport_stencil,
            confidence_gating=True,
            transport_parameterization=transport_parameterization,
            sparse_topk=sparse_topk,
            wavefront_confidence_scale=wavefront_confidence_scale,
            edge_symbol_parameterization=edge_symbol_parameterization,
            edge_symbol_strength=edge_symbol_strength,
        )
        self.symbol = SymbolModulationLayer(
            width,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            symbol_order=symbol_order,
            symbol_parameterization=symbol_parameterization,
            helmholtz_shell_radius=helmholtz_shell_radius,
            helmholtz_refractive_index=helmholtz_refractive_index,
            helmholtz_absorption=helmholtz_absorption,
            helmholtz_resolvent_cap=helmholtz_resolvent_cap,
        )
        if self.num_canonical_branches > 1:
            self.branch_propagations = nn.ModuleList(
                [
                    self.propagation,
                    *[
                        CanonicalPropagationLayer(
                            width,
                            metadata_dim=metadata_dim,
                            hidden_dim=hidden_dim,
                            transport_scale=transport_scale,
                            stencil_size=transport_stencil,
                            confidence_gating=True,
                            transport_parameterization=transport_parameterization,
                            sparse_topk=sparse_topk,
                            wavefront_confidence_scale=wavefront_confidence_scale,
                            edge_symbol_parameterization=edge_symbol_parameterization,
                            edge_symbol_strength=edge_symbol_strength,
                        )
                        for _ in range(self.num_canonical_branches - 1)
                    ],
                ]
            )
            self.branch_symbols = nn.ModuleList(
                [
                    self.symbol,
                    *[
                        SymbolModulationLayer(
                            width,
                            metadata_dim=metadata_dim,
                            hidden_dim=hidden_dim,
                            symbol_order=symbol_order,
                            symbol_parameterization=symbol_parameterization,
                            helmholtz_shell_radius=helmholtz_shell_radius,
                            helmholtz_refractive_index=helmholtz_refractive_index,
                            helmholtz_absorption=helmholtz_absorption,
                            helmholtz_resolvent_cap=helmholtz_resolvent_cap,
                        )
                        for _ in range(self.num_canonical_branches - 1)
                    ],
                ]
            )
            self.branch_router = nn.Linear(width + metadata_dim, self.num_canonical_branches)
        else:
            self.branch_propagations = nn.ModuleList()
            self.branch_symbols = nn.ModuleList()
            self.branch_router = None
        self.norm_pdo = ScaleAwareTokenNorm(width) if self.pdo_symbol_scale != 0.0 else None
        self.pdo_identity = (
            IdentityPseudodifferentialBranch(
                width,
                metadata_dim=metadata_dim,
                hidden_dim=hidden_dim,
                symbol_order=pdo_symbol_order,
                symbol_parameterization=symbol_parameterization,
                helmholtz_shell_radius=helmholtz_shell_radius,
                helmholtz_refractive_index=helmholtz_refractive_index,
                helmholtz_absorption=helmholtz_absorption,
                helmholtz_resolvent_cap=helmholtz_resolvent_cap,
            )
            if self.pdo_symbol_scale != 0.0
            else None
        )
        self.norm_dissipative = ScaleAwareTokenNorm(width) if self.dissipative_symbol_scale != 0.0 else None
        self.dissipative_symbol = (
            DissipativePseudodifferentialBranch(
                width,
                metadata_dim=metadata_dim,
                hidden_dim=hidden_dim,
                time_step=dissipative_time_step,
            )
            if self.dissipative_symbol_scale != 0.0
            else None
        )
        self.refine = nn.Sequential(
            nn.Linear(width + metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, width),
        )
        self.refine_gate = nn.Linear(width, width)
        self.last_branch_weights: Tensor | None = None
        self.last_branch_metadata: Tensor | None = None

    def forward(
        self,
        features: Tensor,
        metadata: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        metadata_before = _metadata_batch(metadata, features)
        if self.num_canonical_branches <= 1:
            propagated, metadata = self.propagation(self.norm1(features, metadata), metadata)
            features = features + propagated
            symbol = self.symbol(
                self.norm2(features, metadata),
                metadata,
                getattr(self.propagation, "last_local_tube_coordinate", None),
            )
            branch_weights = features.new_ones((*features.shape[:2], 1))
            branch_metadata = metadata.unsqueeze(2)
        else:
            branch_weights = _make_branch_weights(
                features=features,
                metadata=metadata,
                num_branches=self.num_canonical_branches,
                branch_routing=self.branch_routing,
                router=self.branch_router,
                branch_prior_strength=self.branch_prior_strength,
            )
            branch_input = self.norm1(features, metadata)
            propagated_items: list[Tensor] = []
            symbol_items: list[Tensor] = []
            metadata_items: list[Tensor] = []
            for propagation, symbol_layer in zip(self.branch_propagations, self.branch_symbols, strict=True):
                branch_propagated, branch_metadata_item = propagation(branch_input, metadata)
                branch_features = features + branch_propagated
                branch_symbol = symbol_layer(
                    self.norm2(branch_features, branch_metadata_item),
                    branch_metadata_item,
                    getattr(propagation, "last_local_tube_coordinate", None),
                )
                propagated_items.append(branch_propagated)
                symbol_items.append(branch_symbol)
                metadata_items.append(_metadata_batch(branch_metadata_item, features))
            propagated_stack = torch.stack(propagated_items, dim=2)
            symbol_stack = torch.stack(symbol_items, dim=2)
            branch_metadata = torch.stack(metadata_items, dim=2)
            propagated = (branch_weights.unsqueeze(-1) * propagated_stack).sum(dim=2)
            symbol = (branch_weights.unsqueeze(-1) * symbol_stack).sum(dim=2)
            metadata = (branch_weights.unsqueeze(-1) * branch_metadata).sum(dim=2)
            features = features + propagated
        self.last_branch_weights = branch_weights.detach()
        self.last_branch_metadata = branch_metadata.detach()
        if self.pdo_identity is None or self.pdo_symbol_scale == 0.0:
            pdo_symbol = torch.zeros_like(features)
        else:
            assert self.norm_pdo is not None
            pdo_symbol = self.pdo_symbol_scale * self.pdo_identity(self.norm_pdo(features, metadata), metadata)
        if self.dissipative_symbol is None or self.dissipative_symbol_scale == 0.0:
            dissipative_symbol = torch.zeros_like(features)
            dissipative_multiplier = torch.ones_like(features)
        else:
            assert self.norm_dissipative is not None
            dissipative_raw, dissipative_multiplier = self.dissipative_symbol(
                self.norm_dissipative(features, metadata),
                metadata,
            )
            dissipative_symbol = self.dissipative_symbol_scale * dissipative_raw
        metadata_batch = _metadata_batch(metadata, features)
        refine = self.refine(torch.cat([features, metadata_batch], dim=-1))
        gate = torch.sigmoid(self.refine_gate(features))
        token_refine = self.token_refine_scale * gate * refine
        features = features + symbol + pdo_symbol + dissipative_symbol + token_refine
        if return_diagnostics:
            diagnostics = {
                "transport_norm": torch.linalg.vector_norm(propagated, dim=-1).mean(),
                "symbol_norm": torch.linalg.vector_norm(symbol, dim=-1).mean(),
                "symbol_order_proxy": _frequency_order_proxy(metadata, symbol),
                "symbol_seminorm_proxy": _finite_symbol_seminorm_proxy(metadata, symbol),
                "edge_symbol_deviation_proxy": getattr(
                    self.propagation,
                    "last_edge_symbol_deviation_proxy",
                    symbol.new_tensor(0.0),
                ),
                "canonical_defect_proxy": getattr(
                    self.propagation,
                    "last_canonical_defect_proxy",
                    symbol.new_tensor(0.0),
                ),
                "wavefront_confidence_proxy": getattr(
                    self.propagation,
                    "last_wavefront_confidence_proxy",
                    symbol.new_tensor(0.0),
                ),
                "pdo_identity_norm": torch.linalg.vector_norm(pdo_symbol, dim=-1).mean(),
                "pdo_identity_order_proxy": _frequency_order_proxy(metadata, pdo_symbol),
                "pdo_identity_seminorm_proxy": _finite_symbol_seminorm_proxy(metadata, pdo_symbol),
                "dissipative_symbol_norm": torch.linalg.vector_norm(dissipative_symbol, dim=-1).mean(),
                "dissipative_multiplier_mean": dissipative_multiplier.mean(),
                "refine_norm": torch.linalg.vector_norm(token_refine, dim=-1).mean(),
                "metadata_shift": torch.linalg.vector_norm(metadata[..., :4] - metadata_before[..., :4], dim=-1).mean(),
                "local_tube_coordinate_norm": torch.linalg.vector_norm(
                    getattr(self.propagation, "last_local_tube_coordinate", symbol.new_zeros(*symbol.shape[:2], 4)),
                    dim=-1,
                ).mean(),
                **_branch_diagnostics(branch_weights, branch_metadata),
            }
            if getattr(self.symbol, "symbol_parameterization", "") == "helmholtz_resolvent":
                diagnostics.update(self.symbol.helmholtz_resolvent_diagnostics(metadata, symbol))
            return features, metadata, diagnostics
        return features, metadata


class MicrolocalNeuralOperatorCore(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        depth: int = 4,
        patch_size: int = 16,
        stride: int = 8,
        max_modes: int = 12,
        window_type: str = "hann",
        mode_strategy: str = "radial",
        low_frequency_scale: float = 0.1,
        transport_scale: float = 0.03,
        transport_stencil: int = 4,
        transport_parameterization: str = "mlp_displacement",
        sparse_topk: bool = False,
        wavefront_confidence_scale: float = 0.0,
        symbol_order: float = 0.0,
        symbol_parameterization: str = "mlp",
        helmholtz_shell_radius: float = 0.0,
        helmholtz_refractive_index: float = 1.0,
        helmholtz_absorption: float = 0.05,
        helmholtz_resolvent_cap: float = 6.0,
        frame_type: str = "local_fft",
        frame_patch_sizes: tuple[int, ...] | list[int] | None = None,
        frame_strides: tuple[int, ...] | list[int] | None = None,
        frame_max_modes: tuple[int, ...] | list[int] | None = None,
        skip_lowpass_cutoff: float = 0.0,
        transported_synthesis_scale: float = 0.0,
        transported_input_scale: float = 0.0,
        transported_synthesis_mode: str = "warp",
        transported_decoder_channels: int = 0,
        transported_decoder_scale: float = 0.0,
        transported_decoder_transport_gate: bool = True,
        num_canonical_branches: int = 1,
        branch_routing: str = "metadata_softmax",
        branch_prior_strength: float = 0.0,
        branch_entropy_weight: float = 0.0,
        branch_diversity_weight: float = 0.0,
        branch_synthesis: str = "sum",
        edge_symbol_parameterization: str = "none",
        edge_symbol_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_frequency_scale = low_frequency_scale
        self.skip_lowpass_cutoff = float(skip_lowpass_cutoff)
        self.transported_synthesis_scale = float(transported_synthesis_scale)
        self.transported_input_scale = float(transported_input_scale)
        if transported_synthesis_mode not in {"warp", "atom_splat", "patch_fold"}:
            raise ValueError(f"Unsupported transported_synthesis_mode: {transported_synthesis_mode}")
        self.transported_synthesis_mode = transported_synthesis_mode
        self.transported_decoder_scale = float(transported_decoder_scale)
        self.transported_decoder_transport_gate = bool(transported_decoder_transport_gate)
        self.num_canonical_branches = max(int(num_canonical_branches), 1)
        self.branch_prior_strength = float(branch_prior_strength)
        self.branch_entropy_weight = float(branch_entropy_weight)
        self.branch_diversity_weight = float(branch_diversity_weight)
        self.branch_synthesis = branch_synthesis
        self._last_branch_metadata: Tensor | None = None
        self._last_branch_weights: Tensor | None = None
        self.transported_decoder: nn.Module | None = None
        if self.transported_decoder_scale != 0.0 and transported_decoder_channels > 0:
            self.transported_decoder = nn.Sequential(
                nn.Conv2d(out_channels * 3, transported_decoder_channels, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(transported_decoder_channels, transported_decoder_channels, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(transported_decoder_channels, out_channels, kernel_size=3, padding=1, bias=False),
            )
            last = self.transported_decoder[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
        self.tokenizer = _build_tokenizer(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
            max_modes=max_modes,
            window_type=window_type,
            mode_strategy=mode_strategy,
            frame_type=frame_type,
            frame_patch_sizes=frame_patch_sizes,
            frame_strides=frame_strides,
            frame_max_modes=frame_max_modes,
        )
        self.input_proj = nn.Linear(in_channels * 2, width)
        self.blocks = nn.ModuleList(
            [
                MiNOCoreBlock(
                    width,
                    transport_scale=transport_scale,
                    transport_stencil=transport_stencil,
                    transport_parameterization=transport_parameterization,
                    sparse_topk=sparse_topk,
                    wavefront_confidence_scale=wavefront_confidence_scale,
                    symbol_order=symbol_order,
                    symbol_parameterization=symbol_parameterization,
                    helmholtz_shell_radius=helmholtz_shell_radius,
                    helmholtz_refractive_index=helmholtz_refractive_index,
                    helmholtz_absorption=helmholtz_absorption,
                    helmholtz_resolvent_cap=helmholtz_resolvent_cap,
                    num_canonical_branches=self.num_canonical_branches,
                    branch_routing=branch_routing,
                    branch_prior_strength=self.branch_prior_strength,
                    branch_synthesis=branch_synthesis,
                    edge_symbol_parameterization=edge_symbol_parameterization,
                    edge_symbol_strength=edge_symbol_strength,
                )
                for _ in range(depth)
            ]
        )
        self.output_proj = nn.Linear(width, out_channels * 2)
        self.low_frequency_residual = SpectralResidualBlock(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self._last_final_metadata: Tensor | None = None
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.dirac_(self.skip.weight)
        if self.skip.bias is not None:
            nn.init.zeros_(self.skip.bias)

    def transported_landing_correction(
        self,
        transported_reconstructed: Tensor,
        transported_input: Tensor,
        lifted: Tensor,
    ) -> Tensor:
        if self.transported_decoder is None or self.transported_decoder_scale == 0.0:
            return _zero_like_field(transported_reconstructed)
        decoder_input = torch.cat([transported_reconstructed, transported_input, lifted], dim=1)
        correction = self.transported_decoder(decoder_input)
        if self.transported_decoder_transport_gate:
            transported_energy = (
                transported_reconstructed.square().mean(dim=(1, 2, 3), keepdim=True)
                + transported_input.square().mean(dim=(1, 2, 3), keepdim=True)
            ).sqrt()
            lifted_energy = lifted.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
            gate = transported_energy / (transported_energy + lifted_energy + 1e-8)
            correction = correction * gate
        return self.transported_decoder_scale * correction

    def transported_landing_gate(
        self,
        transported_reconstructed: Tensor,
        transported_input: Tensor,
        lifted: Tensor,
    ) -> Tensor:
        if not self.transported_decoder_transport_gate:
            return transported_reconstructed.new_tensor(1.0)
        transported_energy = (
            transported_reconstructed.square().mean(dim=(1, 2, 3), keepdim=True)
            + transported_input.square().mean(dim=(1, 2, 3), keepdim=True)
        ).sqrt()
        lifted_energy = lifted.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return (transported_energy / (transported_energy + lifted_energy + 1e-8)).mean()

    def _branch_transport_synthesis(
        self,
        field: Tensor,
        token_features: Tensor,
        encoding: object,
        final_metadata: Tensor | None,
        branch_metadata: Tensor | None,
        branch_weights: Tensor | None,
        *,
        scale: float,
        mode: str,
    ) -> Tensor:
        if final_metadata is None:
            return field
        if (
            branch_metadata is None
            or branch_weights is None
            or branch_metadata.dim() != 4
            or branch_weights.shape[-1] <= 1
        ):
            return _transported_synthesis(field, token_features, encoding, final_metadata, scale=scale, mode=mode)
        transported = field.new_zeros(field.shape)
        weight_sum = branch_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_weights = branch_weights / weight_sum
        for branch_index in range(branch_weights.shape[-1]):
            weights = normalized_weights[..., branch_index : branch_index + 1]
            branch_features = token_features * weights
            branch_field = self.tokenizer.synthesize(branch_features, encoding)
            transported = transported + _transported_synthesis(
                branch_field,
                branch_features,
                encoding,
                branch_metadata[:, :, branch_index, :],
                scale=scale,
                mode=mode,
            )
        return transported

    def _branch_transport_input_carrier(
        self,
        skip_field: Tensor,
        encoding: object,
        final_metadata: Tensor | None,
        branch_metadata: Tensor | None,
        branch_weights: Tensor | None,
    ) -> Tensor:
        if self.transported_input_scale == 0.0 or final_metadata is None:
            return skip_field.new_zeros(skip_field.shape)
        if (
            branch_metadata is None
            or branch_weights is None
            or branch_metadata.dim() != 4
            or branch_weights.shape[-1] <= 1
        ):
            return _transported_input_carrier(
                skip_field,
                encoding,
                final_metadata,
                transported_synthesis_scale=self.transported_synthesis_scale,
                transported_input_scale=self.transported_input_scale,
                highpass_cutoff=self.skip_lowpass_cutoff,
                transported_synthesis_mode=self.transported_synthesis_mode,
            )
        transported = skip_field.new_zeros(skip_field.shape)
        input_features = getattr(encoding, "features")
        if input_features.shape[-1] // 2 != skip_field.shape[1]:
            return _transported_input_carrier(
                skip_field,
                encoding,
                final_metadata,
                transported_synthesis_scale=self.transported_synthesis_scale,
                transported_input_scale=self.transported_input_scale,
                highpass_cutoff=self.skip_lowpass_cutoff,
                transported_synthesis_mode=self.transported_synthesis_mode,
            )
        weight_sum = branch_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized_weights = branch_weights / weight_sum
        for branch_index in range(branch_weights.shape[-1]):
            weights = normalized_weights[..., branch_index : branch_index + 1]
            branch_features = input_features * weights
            branch_field = self.tokenizer.synthesize(branch_features, encoding)
            transported = transported + _transported_synthesis(
                branch_field,
                branch_features,
                encoding,
                branch_metadata[:, :, branch_index, :],
                scale=self.transported_synthesis_scale,
                mode=self.transported_synthesis_mode,
            )
        return self.transported_input_scale * _highpass_field(transported, self.skip_lowpass_cutoff)

    def forward_tokens(
        self,
        x: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, object] | tuple[Tensor, object, dict[str, object]]:
        encoding = self.tokenizer(x)
        metadata = encoding.metadata
        features = self.input_proj(encoding.features)
        block_diagnostics: list[dict[str, Tensor]] = []
        for block in self.blocks:
            if return_diagnostics:
                features, metadata, diagnostics = block(features, metadata, return_diagnostics=True)
                block_diagnostics.append(diagnostics)
            else:
                features, metadata = block(features, metadata)
        token_output = self.output_proj(features)
        self._last_final_metadata = metadata
        if self.blocks:
            last_block = self.blocks[-1]
            self._last_branch_metadata = getattr(last_block, "last_branch_metadata", None)
            self._last_branch_weights = getattr(last_block, "last_branch_weights", None)
        else:
            self._last_branch_metadata = None
            self._last_branch_weights = None
        if return_diagnostics:
            return token_output, encoding, {
                "final_metadata": metadata,
                "branch_final_metadata": self._last_branch_metadata,
                "branch_weights": self._last_branch_weights,
                "block_diagnostics": block_diagnostics,
            }
        return token_output, encoding

    def forward(self, x: Tensor) -> Tensor:
        token_output, encoding = self.forward_tokens(x)
        reconstructed = self.tokenizer.synthesize(token_output, encoding)
        final_metadata = self._last_final_metadata
        if final_metadata is not None:
            reconstructed = self._branch_transport_synthesis(
                reconstructed,
                token_output,
                encoding,
                final_metadata,
                self._last_branch_metadata,
                self._last_branch_weights,
                scale=self.transported_synthesis_scale,
                mode=self.transported_synthesis_mode,
            )
        skip_field = self.skip(x)
        transported_input = self._branch_transport_input_carrier(
            skip_field,
            encoding,
            final_metadata,
            self._last_branch_metadata,
            self._last_branch_weights,
        )
        lifted = _lowpass_field(skip_field, self.skip_lowpass_cutoff)
        landing = self.transported_landing_correction(reconstructed, transported_input, lifted)
        return reconstructed + transported_input + landing + lifted + self.low_frequency_scale * self.low_frequency_residual(lifted)

    def forward_with_diagnostics(self, x: Tensor) -> dict[str, object]:
        token_output, encoding, diagnostics = self.forward_tokens(x, return_diagnostics=True)
        reconstructed = self.tokenizer.synthesize(token_output, encoding)
        transported_reconstructed = self._branch_transport_synthesis(
            reconstructed,
            token_output,
            encoding,
            diagnostics["final_metadata"],
            diagnostics.get("branch_final_metadata"),
            diagnostics.get("branch_weights"),
            scale=self.transported_synthesis_scale,
            mode=self.transported_synthesis_mode,
        )
        skip_field = self.skip(x)
        transported_input = self._branch_transport_input_carrier(
            skip_field,
            encoding,
            diagnostics["final_metadata"],
            diagnostics.get("branch_final_metadata"),
            diagnostics.get("branch_weights"),
        )
        lifted = _lowpass_field(skip_field, self.skip_lowpass_cutoff)
        landing = self.transported_landing_correction(transported_reconstructed, transported_input, lifted)
        landing_gate = self.transported_landing_gate(transported_reconstructed, transported_input, lifted)
        prediction = transported_reconstructed + transported_input + landing + lifted + self.low_frequency_scale * self.low_frequency_residual(lifted)
        return {
            "prediction": prediction,
            "token_output": token_output,
            "encoding": encoding,
            "reconstructed": reconstructed,
            "transported_reconstructed": transported_reconstructed,
            "transported_input": transported_input,
            "transported_landing_correction": landing,
            "transported_synthesis_shift_norm": _field_norm(transported_reconstructed - reconstructed),
            "transported_input_norm": _field_norm(transported_input),
            "transported_landing_norm": _field_norm(landing),
            "transported_landing_gate": landing_gate,
            "transported_input_shift_norm": _field_norm(
                transported_input - self.transported_input_scale * _highpass_field(skip_field, self.skip_lowpass_cutoff)
            ),
            "final_metadata": diagnostics["final_metadata"],
            "branch_final_metadata": diagnostics.get("branch_final_metadata"),
            "branch_weights": diagnostics.get("branch_weights"),
            "block_diagnostics": diagnostics["block_diagnostics"],
            **_tokenizer_diagnostics(self.tokenizer, x, encoding),
        }

    def proxy_losses_from_diagnostics(
        self,
        diagnostics: dict[str, object],
        target: Tensor,
        *,
        proxy_temperature: float = 0.05,
    ) -> dict[str, Tensor]:
        target_encoding = self.tokenizer(target)
        proxy = packet_transport_symbol_proxy(
            prediction_features=diagnostics["token_output"],
            prediction_metadata=diagnostics["final_metadata"],
            encoding=diagnostics["encoding"],
            target_features=target_encoding.features,
            target_encoding=target_encoding,
            temperature=proxy_temperature,
        )
        block_diagnostics = diagnostics.get("block_diagnostics", [])
        if isinstance(block_diagnostics, list) and block_diagnostics:
            canonical_terms = [
                item["canonical_defect_proxy"]
                for item in block_diagnostics
                if isinstance(item, dict) and isinstance(item.get("canonical_defect_proxy"), Tensor)
            ]
            if canonical_terms:
                proxy["canonical_consistency_proxy"] = torch.stack(canonical_terms).mean()
            symbol_order_terms = [
                item["symbol_order_proxy"]
                for item in block_diagnostics
                if isinstance(item, dict) and isinstance(item.get("symbol_order_proxy"), Tensor)
            ]
            if symbol_order_terms:
                proxy["symbol_order_proxy"] = torch.stack(symbol_order_terms).mean()
            symbol_seminorm_terms = [
                item["symbol_seminorm_proxy"]
                for item in block_diagnostics
                if isinstance(item, dict) and isinstance(item.get("symbol_seminorm_proxy"), Tensor)
            ]
            if symbol_seminorm_terms:
                proxy["symbol_seminorm_proxy"] = torch.stack(symbol_seminorm_terms).mean()
            branch_entropy_terms = [
                item["branch_entropy"]
                for item in block_diagnostics
                if isinstance(item, dict) and isinstance(item.get("branch_entropy"), Tensor)
            ]
            branch_usage_terms = [
                item["branch_usage_max"]
                for item in block_diagnostics
                if isinstance(item, dict) and isinstance(item.get("branch_usage_max"), Tensor)
            ]
            if branch_entropy_terms:
                proxy["branch_entropy_proxy"] = torch.stack(branch_entropy_terms).mean()
            if branch_usage_terms:
                proxy["branch_usage_max_proxy"] = torch.stack(branch_usage_terms).mean()
        if "canonical_consistency_proxy" not in proxy:
            proxy["canonical_consistency_proxy"] = target.new_tensor(0.0)
        if "symbol_order_proxy" not in proxy:
            proxy["symbol_order_proxy"] = target.new_tensor(0.0)
        if "symbol_seminorm_proxy" not in proxy:
            proxy["symbol_seminorm_proxy"] = target.new_tensor(0.0)
        if "branch_entropy_proxy" not in proxy:
            proxy["branch_entropy_proxy"] = target.new_tensor(0.0)
        if "branch_usage_max_proxy" not in proxy:
            proxy["branch_usage_max_proxy"] = target.new_tensor(1.0)
        if diagnostics["token_output"].shape == target_encoding.features.shape:
            proxy["packet_space_proxy"] = torch.nn.functional.mse_loss(
                diagnostics["token_output"],
                target_encoding.features,
            )
        else:
            proxy["packet_space_proxy"] = target.new_tensor(0.0)
        return proxy


class MicrolocalNeuralOperatorPlus(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        depth: int = 6,
        patch_size: int = 16,
        stride: int = 8,
        max_modes: int = 12,
        window_type: str = "hann",
        mode_strategy: str = "radial",
        low_frequency_scale: float = 0.05,
        transport_scale: float = 0.03,
        transport_stencil: int = 8,
        transport_parameterization: str = "mlp_displacement",
        sparse_topk: bool = False,
        wavefront_confidence_scale: float = 0.0,
        symbol_order: float = 0.0,
        symbol_parameterization: str = "mlp",
        helmholtz_shell_radius: float = 0.0,
        helmholtz_refractive_index: float = 1.0,
        helmholtz_absorption: float = 0.05,
        helmholtz_resolvent_cap: float = 6.0,
        frame_type: str = "local_fft",
        local_refine_channels: int = 32,
        local_refine_scale: float = 1.0,
        route_bias_init: float = -1.5,
        refine_lowpass_cutoff: float = 0.0,
        transport_highpass_cutoff: float = 0.0,
        pdo_symbol_scale: float = 0.0,
        pdo_symbol_order: float = -2.0,
        dissipative_symbol_scale: float = 0.0,
        dissipative_time_step: float = 1.0,
        frame_patch_sizes: tuple[int, ...] | list[int] | None = None,
        frame_strides: tuple[int, ...] | list[int] | None = None,
        frame_max_modes: tuple[int, ...] | list[int] | None = None,
        skip_lowpass_cutoff: float = 0.0,
        transported_synthesis_scale: float = 0.0,
        transported_input_scale: float = 0.0,
        transported_synthesis_mode: str = "warp",
        transported_decoder_channels: int = 0,
        transported_decoder_scale: float = 0.0,
        transported_decoder_transport_gate: bool = True,
        token_refine_scale: float = 1.0,
        num_canonical_branches: int = 1,
        branch_routing: str = "metadata_softmax",
        branch_prior_strength: float = 0.0,
        branch_entropy_weight: float = 0.0,
        branch_diversity_weight: float = 0.0,
        branch_synthesis: str = "sum",
        edge_symbol_parameterization: str = "none",
        edge_symbol_strength: float = 0.5,
        field_corrector: str = "none",
        field_corrector_scale: float = 0.0,
        field_corrector_width: int = 32,
        field_corrector_input_mode: str = "input_core_carrier",
    ) -> None:
        super().__init__()
        self.core = MicrolocalNeuralOperatorCore(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=0,
            patch_size=patch_size,
            stride=stride,
            max_modes=max_modes,
            window_type=window_type,
            mode_strategy=mode_strategy,
            low_frequency_scale=0.0,
            transport_scale=transport_scale,
            transport_stencil=max(1, transport_stencil),
            transport_parameterization=transport_parameterization,
            sparse_topk=sparse_topk,
            wavefront_confidence_scale=wavefront_confidence_scale,
            symbol_order=symbol_order,
            symbol_parameterization=symbol_parameterization,
            helmholtz_shell_radius=helmholtz_shell_radius,
            helmholtz_refractive_index=helmholtz_refractive_index,
            helmholtz_absorption=helmholtz_absorption,
            helmholtz_resolvent_cap=helmholtz_resolvent_cap,
            frame_type=frame_type,
            frame_patch_sizes=frame_patch_sizes,
            frame_strides=frame_strides,
            frame_max_modes=frame_max_modes,
            skip_lowpass_cutoff=skip_lowpass_cutoff,
            transported_synthesis_scale=transported_synthesis_scale,
            transported_input_scale=transported_input_scale,
            transported_synthesis_mode=transported_synthesis_mode,
            transported_decoder_channels=transported_decoder_channels,
            transported_decoder_scale=transported_decoder_scale,
            transported_decoder_transport_gate=transported_decoder_transport_gate,
            num_canonical_branches=num_canonical_branches,
            branch_routing=branch_routing,
            branch_prior_strength=branch_prior_strength,
            branch_entropy_weight=branch_entropy_weight,
            branch_diversity_weight=branch_diversity_weight,
            branch_synthesis=branch_synthesis,
            edge_symbol_parameterization=edge_symbol_parameterization,
            edge_symbol_strength=edge_symbol_strength,
        )
        self.core.blocks = nn.ModuleList(
            [
                MiNOPlusBlock(
                    width,
                    transport_scale=transport_scale,
                    transport_stencil=transport_stencil,
                    transport_parameterization=transport_parameterization,
                    sparse_topk=sparse_topk,
                    wavefront_confidence_scale=wavefront_confidence_scale,
                    symbol_order=symbol_order,
                    symbol_parameterization=symbol_parameterization,
                    helmholtz_shell_radius=helmholtz_shell_radius,
                    helmholtz_refractive_index=helmholtz_refractive_index,
                    helmholtz_absorption=helmholtz_absorption,
                    helmholtz_resolvent_cap=helmholtz_resolvent_cap,
                    pdo_symbol_scale=pdo_symbol_scale,
                    pdo_symbol_order=pdo_symbol_order,
                    dissipative_symbol_scale=dissipative_symbol_scale,
                    dissipative_time_step=dissipative_time_step,
                    token_refine_scale=token_refine_scale,
                    num_canonical_branches=num_canonical_branches,
                    branch_routing=branch_routing,
                    branch_prior_strength=branch_prior_strength,
                    branch_synthesis=branch_synthesis,
                    edge_symbol_parameterization=edge_symbol_parameterization,
                    edge_symbol_strength=edge_symbol_strength,
                )
                for _ in range(depth)
            ]
        )
        self.pdo_symbol_scale = float(pdo_symbol_scale)
        self.dissipative_symbol_scale = float(dissipative_symbol_scale)
        self.low_frequency_scale = low_frequency_scale
        self.local_refine_scale = float(local_refine_scale)
        self.refine_lowpass_cutoff = float(refine_lowpass_cutoff)
        self.transport_highpass_cutoff = float(transport_highpass_cutoff)
        self.field_corrector_scale = float(field_corrector_scale)
        self.field_corrector_kind = field_corrector
        self.field_corrector_input_mode = field_corrector_input_mode
        if field_corrector_input_mode == "input_only":
            field_corrector_channels = in_channels
        elif field_corrector_input_mode == "input_core":
            field_corrector_channels = in_channels + out_channels
        elif field_corrector_input_mode == "input_core_carrier":
            field_corrector_channels = in_channels + 3 * out_channels
        else:
            raise ValueError(f"Unsupported field_corrector_input_mode: {field_corrector_input_mode}")
        if field_corrector == "none" or self.field_corrector_scale == 0.0:
            self.field_corrector: nn.Module | None = None
        elif field_corrector == "spectral":
            self.field_corrector = FNOStyleBaseline(
                in_channels=field_corrector_channels,
                out_channels=out_channels,
                width=field_corrector_width,
            )
        elif field_corrector == "unet":
            self.field_corrector = UNetStyleBaseline(
                in_channels=field_corrector_channels,
                out_channels=out_channels,
                width=max(field_corrector_width // 2, 8),
            )
        elif field_corrector == "hybrid":
            self.field_corrector = HybridSpectralUNetCorrector(
                in_channels=field_corrector_channels,
                out_channels=out_channels,
                width=field_corrector_width,
            )
        else:
            raise ValueError(f"Unsupported field_corrector: {field_corrector}")
        self.local_refine = nn.Sequential(
            nn.Conv2d(out_channels * 2, local_refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_refine_channels, local_refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(local_refine_channels, out_channels, kernel_size=3, padding=1),
        )
        self.route = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        nn.init.zeros_(self.route.weight)
        if self.route.bias is not None:
            nn.init.constant_(self.route.bias, float(route_bias_init))

    def _field_correction(
        self,
        x: Tensor,
        core_prediction: Tensor,
        transported_input: Tensor,
        lifted: Tensor,
    ) -> Tensor:
        if self.field_corrector is None or self.field_corrector_scale == 0.0:
            return core_prediction.new_zeros(core_prediction.shape)
        if self.field_corrector_input_mode == "input_only":
            corrector_input = x
        elif self.field_corrector_input_mode == "input_core":
            corrector_input = torch.cat([x, core_prediction], dim=1)
        else:
            corrector_input = torch.cat([x, core_prediction, transported_input, lifted], dim=1)
        return self.field_corrector_scale * self.field_corrector(corrector_input)

    def _effective_refine_cutoff(self) -> float:
        cutoffs = [
            cutoff
            for cutoff in (self.refine_lowpass_cutoff, self.transport_highpass_cutoff)
            if cutoff > 0.0
        ]
        return min(cutoffs) if cutoffs else 0.0

    def forward(self, x: Tensor) -> Tensor:
        token_output, encoding = self.core.forward_tokens(x)
        reconstructed = self.core.tokenizer.synthesize(token_output, encoding)
        final_metadata = self.core._last_final_metadata
        if final_metadata is not None:
            reconstructed = self.core._branch_transport_synthesis(
                reconstructed,
                token_output,
                encoding,
                final_metadata,
                self.core._last_branch_metadata,
                self.core._last_branch_weights,
                scale=self.core.transported_synthesis_scale,
                mode=self.core.transported_synthesis_mode,
            )
        skip_field = self.core.skip(x)
        transported_input = self.core._branch_transport_input_carrier(
            skip_field,
            encoding,
            final_metadata,
            self.core._last_branch_metadata,
            self.core._last_branch_weights,
        )
        lifted = _lowpass_field(skip_field, self.core.skip_lowpass_cutoff)
        low = self.low_frequency_scale * self.core.low_frequency_residual(lifted)
        landing = self.core.transported_landing_correction(reconstructed, transported_input, lifted)
        core_prediction = reconstructed + transported_input + landing + lifted + low
        refine_in = torch.cat([core_prediction, lifted], dim=1)
        high = self.local_refine(refine_in)
        route = torch.sigmoid(self.route(refine_in))
        raw_refine_correction = self.local_refine_scale * route * high
        refine_correction = _lowpass_field(raw_refine_correction, self._effective_refine_cutoff())
        field_correction = self._field_correction(x, core_prediction, transported_input, lifted)
        return core_prediction + refine_correction + field_correction

    def forward_with_diagnostics(self, x: Tensor) -> dict[str, object]:
        token_output, encoding, core_diagnostics = self.core.forward_tokens(x, return_diagnostics=True)
        reconstructed = self.core.tokenizer.synthesize(token_output, encoding)
        transported_reconstructed = self.core._branch_transport_synthesis(
            reconstructed,
            token_output,
            encoding,
            core_diagnostics["final_metadata"],
            core_diagnostics.get("branch_final_metadata"),
            core_diagnostics.get("branch_weights"),
            scale=self.core.transported_synthesis_scale,
            mode=self.core.transported_synthesis_mode,
        )
        skip_field = self.core.skip(x)
        transported_input = self.core._branch_transport_input_carrier(
            skip_field,
            encoding,
            core_diagnostics["final_metadata"],
            core_diagnostics.get("branch_final_metadata"),
            core_diagnostics.get("branch_weights"),
        )
        lifted = _lowpass_field(skip_field, self.core.skip_lowpass_cutoff)
        low = self.low_frequency_scale * self.core.low_frequency_residual(lifted)
        landing = self.core.transported_landing_correction(transported_reconstructed, transported_input, lifted)
        landing_gate = self.core.transported_landing_gate(transported_reconstructed, transported_input, lifted)
        core_prediction = transported_reconstructed + transported_input + landing + lifted + low
        refine_in = torch.cat([core_prediction, lifted], dim=1)
        high = self.local_refine(refine_in)
        route = torch.sigmoid(self.route(refine_in))
        raw_refine_correction = self.local_refine_scale * route * high
        refine_correction = _lowpass_field(raw_refine_correction, self._effective_refine_cutoff())
        field_correction = self._field_correction(x, core_prediction, transported_input, lifted)
        prediction = core_prediction + refine_correction + field_correction
        removed_refine = raw_refine_correction - refine_correction
        highpass_cutoff = self.transport_highpass_cutoff if self.transport_highpass_cutoff > 0.0 else self._effective_refine_cutoff()
        return {
            "prediction": prediction,
            "core_prediction": core_prediction,
            "refine_correction": refine_correction,
            "raw_refine_correction": raw_refine_correction,
            "removed_refine_correction": removed_refine,
            "token_output": token_output,
            "encoding": encoding,
            "reconstructed": reconstructed,
            "transported_reconstructed": transported_reconstructed,
            "transported_input": transported_input,
            "transported_landing_correction": landing,
            "transported_synthesis_shift_norm": _field_norm(transported_reconstructed - reconstructed),
            "transported_input_norm": _field_norm(transported_input),
            "transported_landing_norm": _field_norm(landing),
            "transported_landing_gate": landing_gate,
            "transported_input_shift_norm": _field_norm(
                transported_input - self.core.transported_input_scale * _highpass_field(skip_field, self.core.skip_lowpass_cutoff)
            ),
            "final_metadata": core_diagnostics["final_metadata"],
            "branch_final_metadata": core_diagnostics.get("branch_final_metadata"),
            "branch_weights": core_diagnostics.get("branch_weights"),
            "block_diagnostics": core_diagnostics["block_diagnostics"],
            "route_mean": route.mean(),
            "route_abs_mean": route.abs().mean(),
            "refine_field_norm": _field_norm(high),
            "raw_refine_correction_norm": _field_norm(raw_refine_correction),
            "refine_correction_norm": _field_norm(refine_correction),
            "field_correction_norm": _field_norm(field_correction),
            "refine_lowpass_removed_norm": _field_norm(removed_refine),
            "core_high_frequency_norm": _field_norm(_highpass_field(core_prediction, highpass_cutoff)),
            "refine_high_frequency_norm": _field_norm(_highpass_field(refine_correction, highpass_cutoff)),
            **_tokenizer_diagnostics(self.core.tokenizer, x, encoding),
        }

    def proxy_losses_from_diagnostics(
        self,
        diagnostics: dict[str, object],
        target: Tensor,
        *,
        proxy_temperature: float = 0.05,
    ) -> dict[str, Tensor]:
        return self.core.proxy_losses_from_diagnostics(
            diagnostics,
            target,
            proxy_temperature=proxy_temperature,
        )


MicrolocalNeuralOperator = MicrolocalNeuralOperatorCore


def _carrier_bound_model_kwargs(model_kwargs: dict[str, object] | None = None) -> dict[str, object]:
    """Defaults for the paper-defined carrier-bound MiNO architecture.

    Direct class constructors keep conservative legacy defaults so old scripts can
    opt into exact historical settings. The public model builder maps the MiNO
    names used in the manuscript to the theorem-aligned carrier-bound profile.
    """

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("frame_type", "gabor_gaussian")
    kwargs.setdefault("transport_parameterization", "hamiltonian_verlet")
    kwargs.setdefault("symbol_parameterization", "spectral_order")
    kwargs.setdefault("sparse_topk", True)
    kwargs.setdefault("skip_lowpass_cutoff", 0.16)
    kwargs.setdefault("transported_synthesis_scale", 1.0)
    kwargs.setdefault("transported_input_scale", 1.0)
    return kwargs


def build_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    model_kwargs: dict[str, object] | None = None,
) -> nn.Module:
    model_kwargs = model_kwargs or {}
    normalized = name.lower().replace("_", "").replace(" ", "").replace("-", "")
    if normalized in {"mino", "minocore", "minocarrier", "minocorecarrier"}:
        kwargs = _carrier_bound_model_kwargs(model_kwargs)
        return MicrolocalNeuralOperatorCore(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if normalized in {"minoplus", "minopluscarrier"}:
        kwargs = _carrier_bound_model_kwargs(model_kwargs)
        return MicrolocalNeuralOperatorPlus(in_channels=in_channels, out_channels=out_channels, **kwargs)
    if normalized in {"minolegacy", "minocorelegacy"}:
        return MicrolocalNeuralOperatorCore(in_channels=in_channels, out_channels=out_channels, **model_kwargs)
    if normalized == "minopluslegacy":
        return MicrolocalNeuralOperatorPlus(in_channels=in_channels, out_channels=out_channels, **model_kwargs)
    if normalized in {"fno", "conv-fno", "convfno", "wno", "wnoofficial", "wno-official", "uno", "fno+learnedlocaltc", "fnolearnedlocaltc"}:
        return build_official_model(name, in_channels=in_channels, out_channels=out_channels, model_kwargs=model_kwargs)
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
    if normalized in {"hybridspectralunet", "hybridspectralunetcorrector"}:
        return HybridSpectralUNetCorrector(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unknown model: {name}")
