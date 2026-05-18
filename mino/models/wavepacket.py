from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class WavepacketEncoding:
    features: Tensor
    metadata: Tensor
    input_shape: tuple[int, int]
    patch_shape: tuple[int, int]
    patch_count: int
    mode_count: int
    channels: int
    token_slices: tuple[tuple[int, int], ...] = ()
    group_patch_counts: tuple[int, ...] = ()
    group_mode_counts: tuple[int, ...] = ()
    group_patch_shapes: tuple[tuple[int, int], ...] = ()
    group_stride_shapes: tuple[tuple[int, int], ...] = ()


def _normalize_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    item = int(value)
    return item, item


def _build_window(size: int, device: torch.device, dtype: torch.dtype, window_type: str) -> Tensor:
    return _build_rect_window((size, size), device, dtype, window_type)


def _build_rect_window(shape: tuple[int, int], device: torch.device, dtype: torch.dtype, window_type: str) -> Tensor:
    height, width = _normalize_pair(shape)
    if window_type == "boxcar":
        return torch.ones(height, width, device=device, dtype=dtype)
    if window_type == "gaussian":
        coords_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        coords_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
        sigma = 0.35
        window = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
        return window.clamp_min(1e-3)
    if window_type != "hann":
        raise ValueError(f"Unsupported window type: {window_type}")
    window_y = torch.hann_window(height, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    window_x = torch.hann_window(width, periodic=False, device=device, dtype=dtype).clamp_min(1e-3)
    return torch.outer(window_y, window_x)


class LocalFourierWavepacketTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int = 16,
        stride: int = 8,
        max_modes: Optional[int] = 12,
        window_type: str = "hann",
        mode_strategy: str = "radial",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.window_type = window_type
        self.mode_strategy = mode_strategy
        mode_indices = self._build_mode_indices(patch_size, max_modes, mode_strategy)
        self.register_buffer("mode_indices", mode_indices, persistent=False)

    @staticmethod
    def _build_mode_indices(patch_size: int, max_modes: Optional[int], mode_strategy: str) -> Tensor:
        candidates: list[tuple[int, int, int]] = []
        for ky in range(patch_size):
            signed_ky = ky if ky <= patch_size // 2 else ky - patch_size
            for kx in range(patch_size // 2 + 1):
                radius = signed_ky * signed_ky + kx * kx
                candidates.append((radius, ky, kx))
        candidates.sort(key=lambda item: (item[0], abs(item[1]), item[2]))
        if max_modes is None or max_modes >= len(candidates):
            selected = candidates
        elif mode_strategy == "radial":
            selected = candidates[:max_modes]
        elif mode_strategy == "shell_balanced":
            grouped: dict[int, list[tuple[int, int, int]]] = {}
            for item in candidates:
                grouped.setdefault(item[0], []).append(item)
            shell_order = sorted(grouped)
            alternating_shells: list[int] = []
            left = 0
            right = len(shell_order) - 1
            while left <= right:
                alternating_shells.append(shell_order[left])
                if left != right:
                    alternating_shells.append(shell_order[right])
                left += 1
                right -= 1
            shell_offsets = {shell: 0 for shell in alternating_shells}
            selected = []
            cursor = 0
            while len(selected) < max_modes:
                shell = alternating_shells[cursor % len(alternating_shells)]
                offset = shell_offsets[shell]
                bucket = grouped[shell]
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    shell_offsets[shell] = offset + 1
                cursor += 1
        else:
            raise ValueError(f"Unsupported mode strategy: {mode_strategy}")
        return torch.tensor([[ky, kx] for _, ky, kx in selected], dtype=torch.long)

    def _patchify(self, x: Tensor) -> tuple[Tensor, int, int]:
        batch, channels, height, width = x.shape
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        patch_area = self.patch_size * self.patch_size
        patch_count = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(batch, patch_count, channels, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        assert patch_area == self.patch_size * self.patch_size
        return patches, height, width

    def _build_metadata(self, height: int, width: int, patch_count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        patch_centers_y = torch.arange(0, height - self.patch_size + 1, self.stride, device=device, dtype=dtype)
        patch_centers_x = torch.arange(0, width - self.patch_size + 1, self.stride, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(patch_centers_y, patch_centers_x, indexing="ij")
        centers = torch.stack(
            [
                (grid_y.reshape(-1) + 0.5 * self.patch_size) / max(height - 1, 1),
                (grid_x.reshape(-1) + 0.5 * self.patch_size) / max(width - 1, 1),
            ],
            dim=-1,
        )
        if centers.shape[0] != patch_count:
            raise ValueError("Patch count and metadata grid disagree.")
        mode_count = self.mode_indices.shape[0]
        signed_ky = torch.where(
            self.mode_indices[:, 0] <= self.patch_size // 2,
            self.mode_indices[:, 0],
            self.mode_indices[:, 0] - self.patch_size,
        )
        signed_kx = self.mode_indices[:, 1]
        freqs = torch.stack(
            [
                signed_ky.to(dtype) / float(self.patch_size),
                signed_kx.to(dtype) / float(self.patch_size),
            ],
            dim=-1,
        )
        patch_meta = centers[:, None, :].expand(patch_count, mode_count, 2)
        freq_meta = freqs[None, :, :].expand(patch_count, mode_count, 2)
        scale_meta = torch.full((patch_count, mode_count, 1), float(self.patch_size) / float(max(height, width)), device=device, dtype=dtype)
        metadata = torch.cat([patch_meta, freq_meta, scale_meta], dim=-1)
        return metadata.reshape(patch_count * mode_count, 5)

    def forward(self, x: Tensor) -> WavepacketEncoding:
        batch, channels, _, _ = x.shape
        patches, height, width = self._patchify(x)
        window = _build_window(self.patch_size, x.device, x.dtype, self.window_type)
        windowed = patches * window.view(1, 1, 1, self.patch_size, self.patch_size)
        spectrum = torch.fft.rfft2(windowed, norm="ortho")
        ky = self.mode_indices[:, 0]
        kx = self.mode_indices[:, 1]
        coeffs = spectrum[..., ky, kx]
        coeffs = coeffs.permute(0, 2, 3, 1).contiguous()
        features = torch.cat([coeffs.real, coeffs.imag], dim=-1).reshape(batch, -1, channels * 2)
        metadata = self._build_metadata(height, width, patches.shape[2], x.device, x.dtype)
        return WavepacketEncoding(
            features=features,
            metadata=metadata,
            input_shape=(height, width),
            patch_shape=(self.patch_size, self.patch_size),
            patch_count=patches.shape[2],
            mode_count=self.mode_indices.shape[0],
            channels=channels,
            token_slices=((0, features.shape[1]),),
            group_patch_counts=(patches.shape[2],),
            group_mode_counts=(self.mode_indices.shape[0],),
            group_patch_shapes=((self.patch_size, self.patch_size),),
            group_stride_shapes=((self.stride, self.stride),),
        )

    def synthesize(self, features: Tensor, encoding: WavepacketEncoding) -> Tensor:
        batch = features.shape[0]
        patch_count = encoding.patch_count
        mode_count = encoding.mode_count
        height, width = encoding.input_shape
        patch_size = self.patch_size
        if features.shape[1] != patch_count * mode_count:
            raise ValueError("Feature token count does not match encoding.")
        if features.shape[-1] % 2 != 0:
            raise ValueError("Feature dimension must be even for real/imaginary packing.")
        channels = features.shape[-1] // 2
        coeffs = features.reshape(batch, patch_count, mode_count, channels * 2)
        coeffs = coeffs.reshape(batch, patch_count, mode_count, channels, 2).permute(0, 3, 1, 2, 4).contiguous()
        complex_coeffs = torch.complex(coeffs[..., 0], coeffs[..., 1])
        spectrum = torch.zeros(
            batch,
            channels,
            patch_count,
            patch_size,
            patch_size // 2 + 1,
            dtype=complex_coeffs.dtype,
            device=complex_coeffs.device,
        )
        ky = self.mode_indices[:, 0]
        kx = self.mode_indices[:, 1]
        spectrum[..., ky, kx] = complex_coeffs
        patches = torch.fft.irfft2(spectrum, s=(patch_size, patch_size), norm="ortho")
        window = _build_window(patch_size, features.device, features.dtype, self.window_type)
        patches = patches * window.view(1, 1, 1, patch_size, patch_size)
        flat = patches.permute(0, 2, 1, 3, 4).reshape(batch, patch_count, channels * patch_size * patch_size)
        flat = flat.transpose(1, 2).contiguous()
        output = F.fold(flat, output_size=(height, width), kernel_size=patch_size, stride=self.stride)

        ones = torch.ones((1, channels, height, width), device=features.device, dtype=features.dtype)
        ones_patches, _, _ = self._patchify(ones)
        ones_patches = ones_patches * window.view(1, 1, 1, patch_size, patch_size).pow(2)
        ones_flat = ones_patches.permute(0, 2, 1, 3, 4).reshape(1, patch_count, channels * patch_size * patch_size)
        ones_flat = ones_flat.transpose(1, 2).contiguous()
        normalizer = F.fold(ones_flat, output_size=(height, width), kernel_size=patch_size, stride=self.stride)
        return output / normalizer.clamp_min(1e-6)


class RectangularFourierWavepacketTokenizer(nn.Module):
    """Local Fourier tokenizer with rectangular packets for anisotropic probes.

    This is an anisotropic Gabor-style candidate, not a full curvelet/shearlet
    implementation: it exposes elongated packet windows while preserving the
    same finite packet interface used by the MiNO blocks.
    """

    def __init__(
        self,
        in_channels: int,
        patch_shape: tuple[int, int] = (16, 32),
        stride_shape: tuple[int, int] | None = None,
        max_modes: Optional[int] = 12,
        window_type: str = "gaussian",
        mode_strategy: str = "shell_balanced",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_shape = _normalize_pair(patch_shape)
        self.stride_shape = _normalize_pair(stride_shape or (max(1, self.patch_shape[0] // 2), max(1, self.patch_shape[1] // 2)))
        self.window_type = window_type
        self.mode_strategy = mode_strategy
        mode_indices = self._build_mode_indices(self.patch_shape, max_modes, mode_strategy)
        self.register_buffer("mode_indices", mode_indices, persistent=False)

    @staticmethod
    def _build_mode_indices(
        patch_shape: tuple[int, int],
        max_modes: Optional[int],
        mode_strategy: str,
    ) -> Tensor:
        patch_height, patch_width = patch_shape
        candidates: list[tuple[int, int, int]] = []
        for ky in range(patch_height):
            signed_ky = ky if ky <= patch_height // 2 else ky - patch_height
            for kx in range(patch_width // 2 + 1):
                radius = (signed_ky * patch_width) ** 2 + (kx * patch_height) ** 2
                candidates.append((radius, ky, kx))
        candidates.sort(key=lambda item: (item[0], abs(item[1]), item[2]))
        if max_modes is None or max_modes >= len(candidates):
            selected = candidates
        elif mode_strategy == "radial":
            selected = candidates[:max_modes]
        elif mode_strategy == "shell_balanced":
            grouped: dict[int, list[tuple[int, int, int]]] = {}
            for item in candidates:
                grouped.setdefault(item[0], []).append(item)
            selected = []
            shells = sorted(grouped)
            shell_offsets = {shell: 0 for shell in shells}
            cursor = 0
            while len(selected) < max_modes:
                shell = shells[cursor % len(shells)]
                offset = shell_offsets[shell]
                bucket = grouped[shell]
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    shell_offsets[shell] = offset + 1
                cursor += 1
        else:
            raise ValueError(f"Unsupported mode strategy: {mode_strategy}")
        return torch.tensor([[ky, kx] for _, ky, kx in selected], dtype=torch.long)

    def _patchify(self, x: Tensor) -> tuple[Tensor, int, int]:
        batch, channels, height, width = x.shape
        patch_height, patch_width = self.patch_shape
        patches = F.unfold(x, kernel_size=self.patch_shape, stride=self.stride_shape)
        patch_count = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(batch, patch_count, channels, patch_height, patch_width)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        return patches, height, width

    def _build_metadata(self, height: int, width: int, patch_count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        patch_height, patch_width = self.patch_shape
        stride_y, stride_x = self.stride_shape
        patch_centers_y = torch.arange(0, height - patch_height + 1, stride_y, device=device, dtype=dtype)
        patch_centers_x = torch.arange(0, width - patch_width + 1, stride_x, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(patch_centers_y, patch_centers_x, indexing="ij")
        centers = torch.stack(
            [
                (grid_y.reshape(-1) + 0.5 * patch_height) / max(height - 1, 1),
                (grid_x.reshape(-1) + 0.5 * patch_width) / max(width - 1, 1),
            ],
            dim=-1,
        )
        if centers.shape[0] != patch_count:
            raise ValueError("Patch count and rectangular metadata grid disagree.")
        mode_count = self.mode_indices.shape[0]
        signed_ky = torch.where(
            self.mode_indices[:, 0] <= patch_height // 2,
            self.mode_indices[:, 0],
            self.mode_indices[:, 0] - patch_height,
        )
        signed_kx = self.mode_indices[:, 1]
        freqs = torch.stack(
            [
                signed_ky.to(dtype) / float(patch_height),
                signed_kx.to(dtype) / float(patch_width),
            ],
            dim=-1,
        )
        patch_meta = centers[:, None, :].expand(patch_count, mode_count, 2)
        freq_meta = freqs[None, :, :].expand(patch_count, mode_count, 2)
        scale = math.sqrt(float(patch_height * patch_width)) / float(max(height, width))
        scale_meta = torch.full((patch_count, mode_count, 1), scale, device=device, dtype=dtype)
        metadata = torch.cat([patch_meta, freq_meta, scale_meta], dim=-1)
        return metadata.reshape(patch_count * mode_count, 5)

    def forward(self, x: Tensor) -> WavepacketEncoding:
        batch, channels, _, _ = x.shape
        patches, height, width = self._patchify(x)
        patch_height, patch_width = self.patch_shape
        window = _build_rect_window(self.patch_shape, x.device, x.dtype, self.window_type)
        windowed = patches * window.view(1, 1, 1, patch_height, patch_width)
        spectrum = torch.fft.rfft2(windowed, norm="ortho")
        ky = self.mode_indices[:, 0]
        kx = self.mode_indices[:, 1]
        coeffs = spectrum[..., ky, kx]
        coeffs = coeffs.permute(0, 2, 3, 1).contiguous()
        features = torch.cat([coeffs.real, coeffs.imag], dim=-1).reshape(batch, -1, channels * 2)
        metadata = self._build_metadata(height, width, patches.shape[2], x.device, x.dtype)
        return WavepacketEncoding(
            features=features,
            metadata=metadata,
            input_shape=(height, width),
            patch_shape=self.patch_shape,
            patch_count=patches.shape[2],
            mode_count=self.mode_indices.shape[0],
            channels=channels,
            token_slices=((0, features.shape[1]),),
            group_patch_counts=(patches.shape[2],),
            group_mode_counts=(self.mode_indices.shape[0],),
            group_patch_shapes=(self.patch_shape,),
            group_stride_shapes=(self.stride_shape,),
        )

    def synthesize(self, features: Tensor, encoding: WavepacketEncoding) -> Tensor:
        batch = features.shape[0]
        patch_count = encoding.patch_count
        mode_count = encoding.mode_count
        height, width = encoding.input_shape
        patch_height, patch_width = self.patch_shape
        if features.shape[1] != patch_count * mode_count:
            raise ValueError("Feature token count does not match rectangular encoding.")
        if features.shape[-1] % 2 != 0:
            raise ValueError("Feature dimension must be even for real/imaginary packing.")
        channels = features.shape[-1] // 2
        coeffs = features.reshape(batch, patch_count, mode_count, channels * 2)
        coeffs = coeffs.reshape(batch, patch_count, mode_count, channels, 2).permute(0, 3, 1, 2, 4).contiguous()
        complex_coeffs = torch.complex(coeffs[..., 0], coeffs[..., 1])
        spectrum = torch.zeros(
            batch,
            channels,
            patch_count,
            patch_height,
            patch_width // 2 + 1,
            dtype=complex_coeffs.dtype,
            device=complex_coeffs.device,
        )
        ky = self.mode_indices[:, 0]
        kx = self.mode_indices[:, 1]
        spectrum[..., ky, kx] = complex_coeffs
        patches = torch.fft.irfft2(spectrum, s=self.patch_shape, norm="ortho")
        window = _build_rect_window(self.patch_shape, features.device, features.dtype, self.window_type)
        patches = patches * window.view(1, 1, 1, patch_height, patch_width)
        flat = patches.permute(0, 2, 1, 3, 4).reshape(batch, patch_count, channels * patch_height * patch_width)
        flat = flat.transpose(1, 2).contiguous()
        output = F.fold(flat, output_size=(height, width), kernel_size=self.patch_shape, stride=self.stride_shape)

        ones = torch.ones((1, channels, height, width), device=features.device, dtype=features.dtype)
        ones_patches, _, _ = self._patchify(ones)
        ones_patches = ones_patches * window.view(1, 1, 1, patch_height, patch_width).pow(2)
        ones_flat = ones_patches.permute(0, 2, 1, 3, 4).reshape(1, patch_count, channels * patch_height * patch_width)
        ones_flat = ones_flat.transpose(1, 2).contiguous()
        normalizer = F.fold(ones_flat, output_size=(height, width), kernel_size=self.patch_shape, stride=self.stride_shape)
        return output / normalizer.clamp_min(1e-6)


class MultiScaleWavepacketTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_sizes: tuple[int, ...],
        strides: tuple[int, ...],
        max_modes: tuple[Optional[int], ...],
        window_type: str = "hann",
        mode_strategy: str = "radial",
    ) -> None:
        super().__init__()
        if not patch_sizes:
            raise ValueError("patch_sizes must be non-empty.")
        if len(patch_sizes) != len(strides) or len(patch_sizes) != len(max_modes):
            raise ValueError("patch_sizes, strides, and max_modes must have the same length.")
        self.tokenizers = nn.ModuleList(
            [
                LocalFourierWavepacketTokenizer(
                    in_channels=in_channels,
                    patch_size=patch_size,
                    stride=stride,
                    max_modes=mode_count,
                    window_type=window_type,
                    mode_strategy=mode_strategy,
                )
                for patch_size, stride, mode_count in zip(patch_sizes, strides, max_modes, strict=True)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.tokenizers), dtype=torch.float32))

    def forward(self, x: Tensor) -> WavepacketEncoding:
        group_encodings = [tokenizer(x) for tokenizer in self.tokenizers]
        features = torch.cat([encoding.features for encoding in group_encodings], dim=1)
        metadata = torch.cat([encoding.metadata for encoding in group_encodings], dim=0)
        token_slices: list[tuple[int, int]] = []
        start = 0
        for encoding in group_encodings:
            stop = start + encoding.features.shape[1]
            token_slices.append((start, stop))
            start = stop
        return WavepacketEncoding(
            features=features,
            metadata=metadata,
            input_shape=group_encodings[0].input_shape,
            patch_shape=group_encodings[0].patch_shape,
            patch_count=features.shape[1],
            mode_count=1,
            channels=group_encodings[0].channels,
            token_slices=tuple(token_slices),
            group_patch_counts=tuple(encoding.patch_count for encoding in group_encodings),
            group_mode_counts=tuple(encoding.mode_count for encoding in group_encodings),
            group_patch_shapes=tuple(encoding.patch_shape for encoding in group_encodings),
            group_stride_shapes=tuple(encoding.group_stride_shapes[0] for encoding in group_encodings),
        )

    def synthesize(self, features: Tensor, encoding: WavepacketEncoding) -> Tensor:
        if len(encoding.token_slices) != len(self.tokenizers):
            raise ValueError("Encoding groups do not match tokenizer groups.")
        reconstructions: list[Tensor] = []
        for tokenizer, (start, stop) in zip(self.tokenizers, encoding.token_slices, strict=True):
            sub_features = features[:, start:stop, :]
            patch_size = tokenizer.patch_size
            sub_patch_count = ((encoding.input_shape[0] - patch_size) // tokenizer.stride + 1) * (
                (encoding.input_shape[1] - patch_size) // tokenizer.stride + 1
            )
            sub_mode_count = tokenizer.mode_indices.shape[0]
            sub_encoding = WavepacketEncoding(
                features=sub_features,
                metadata=encoding.metadata[start:stop],
                input_shape=encoding.input_shape,
                patch_shape=(patch_size, patch_size),
                patch_count=sub_patch_count,
                mode_count=sub_mode_count,
                channels=encoding.channels,
                token_slices=((0, stop - start),),
                group_patch_counts=(sub_patch_count,),
                group_mode_counts=(sub_mode_count,),
                group_patch_shapes=((patch_size, patch_size),),
                group_stride_shapes=((tokenizer.stride, tokenizer.stride),),
            )
            reconstructions.append(tokenizer.synthesize(sub_features, sub_encoding))
        weights = torch.softmax(self.scale_logits, dim=0)
        output = torch.zeros_like(reconstructions[0])
        for weight, reconstruction in zip(weights, reconstructions, strict=True):
            output = output + weight.to(reconstruction.dtype) * reconstruction
        return output


class AnisotropicGaborWavepacketTokenizer(nn.Module):
    """Multi-orientation rectangular Gabor tokenizer.

    The packet family is intentionally modest: it gives MiNO an executable
    anisotropic frame candidate for mechanism and high-frequency stress tests.
    """

    def __init__(
        self,
        in_channels: int,
        patch_shapes: tuple[tuple[int, int], ...] = ((8, 16), (16, 8), (16, 32), (32, 16)),
        stride_shapes: tuple[tuple[int, int], ...] | None = None,
        max_modes: tuple[Optional[int], ...] | None = None,
        window_type: str = "gaussian",
        mode_strategy: str = "shell_balanced",
    ) -> None:
        super().__init__()
        if not patch_shapes:
            raise ValueError("patch_shapes must be non-empty.")
        stride_shapes = stride_shapes or tuple((max(1, h // 2), max(1, w // 2)) for h, w in patch_shapes)
        max_modes = max_modes or tuple(12 for _ in patch_shapes)
        if len(patch_shapes) != len(stride_shapes) or len(patch_shapes) != len(max_modes):
            raise ValueError("patch_shapes, stride_shapes, and max_modes must have the same length.")
        self.tokenizers = nn.ModuleList(
            [
                RectangularFourierWavepacketTokenizer(
                    in_channels=in_channels,
                    patch_shape=patch_shape,
                    stride_shape=stride_shape,
                    max_modes=mode_count,
                    window_type=window_type,
                    mode_strategy=mode_strategy,
                )
                for patch_shape, stride_shape, mode_count in zip(patch_shapes, stride_shapes, max_modes, strict=True)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.tokenizers), dtype=torch.float32))

    def forward(self, x: Tensor) -> WavepacketEncoding:
        group_encodings = [tokenizer(x) for tokenizer in self.tokenizers]
        features = torch.cat([encoding.features for encoding in group_encodings], dim=1)
        metadata = torch.cat([encoding.metadata for encoding in group_encodings], dim=0)
        token_slices: list[tuple[int, int]] = []
        start = 0
        for encoding in group_encodings:
            stop = start + encoding.features.shape[1]
            token_slices.append((start, stop))
            start = stop
        return WavepacketEncoding(
            features=features,
            metadata=metadata,
            input_shape=group_encodings[0].input_shape,
            patch_shape=group_encodings[0].patch_shape,
            patch_count=features.shape[1],
            mode_count=1,
            channels=group_encodings[0].channels,
            token_slices=tuple(token_slices),
            group_patch_counts=tuple(encoding.patch_count for encoding in group_encodings),
            group_mode_counts=tuple(encoding.mode_count for encoding in group_encodings),
            group_patch_shapes=tuple(encoding.patch_shape for encoding in group_encodings),
            group_stride_shapes=tuple(encoding.group_stride_shapes[0] for encoding in group_encodings),
        )

    def synthesize(self, features: Tensor, encoding: WavepacketEncoding) -> Tensor:
        if len(encoding.token_slices) != len(self.tokenizers):
            raise ValueError("Encoding groups do not match anisotropic tokenizer groups.")
        reconstructions: list[Tensor] = []
        for tokenizer, (start, stop), patch_count, mode_count in zip(
            self.tokenizers,
            encoding.token_slices,
            encoding.group_patch_counts,
            encoding.group_mode_counts,
            strict=True,
        ):
            sub_features = features[:, start:stop, :]
            sub_encoding = WavepacketEncoding(
                features=sub_features,
                metadata=encoding.metadata[start:stop],
                input_shape=encoding.input_shape,
                patch_shape=tokenizer.patch_shape,
                patch_count=patch_count,
                mode_count=mode_count,
                channels=encoding.channels,
                token_slices=((0, stop - start),),
                group_patch_counts=(patch_count,),
                group_mode_counts=(mode_count,),
                group_patch_shapes=(tokenizer.patch_shape,),
                group_stride_shapes=(tokenizer.stride_shape,),
            )
            reconstructions.append(tokenizer.synthesize(sub_features, sub_encoding))
        weights = torch.softmax(self.scale_logits, dim=0)
        output = torch.zeros_like(reconstructions[0])
        for weight, reconstruction in zip(weights, reconstructions, strict=True):
            output = output + weight.to(reconstruction.dtype) * reconstruction
        return output


def _encoding_groups(features: Tensor, metadata: Tensor, encoding: WavepacketEncoding) -> list[tuple[Tensor, Tensor, int, int]]:
    token_slices = encoding.token_slices or ((0, features.shape[1]),)
    patch_counts = encoding.group_patch_counts or (encoding.patch_count,)
    mode_counts = encoding.group_mode_counts or (encoding.mode_count,)
    groups: list[tuple[Tensor, Tensor, int, int]] = []
    for (start, stop), patch_count, mode_count in zip(token_slices, patch_counts, mode_counts, strict=True):
        if metadata.dim() == 2:
            token_metadata = metadata[start:stop]
        elif metadata.dim() == 3:
            token_metadata = metadata[:, start:stop, :]
        else:
            raise ValueError("metadata must have shape (tokens, dim) or (batch, tokens, dim).")
        groups.append((features[:, start:stop, :], token_metadata, patch_count, mode_count))
    return groups


def grouped_patch_statistics(features: Tensor, metadata: Tensor, encoding: WavepacketEncoding) -> list[dict[str, Tensor]]:
    stats: list[dict[str, Tensor]] = []
    for token_features, token_metadata, patch_count, mode_count in _encoding_groups(features, metadata, encoding):
        batch = token_features.shape[0]
        reshaped_features = token_features.reshape(batch, patch_count, mode_count, -1)
        mode_energy = torch.linalg.vector_norm(reshaped_features, dim=-1).clamp_min(1e-8)
        patch_energy = mode_energy.mean(dim=-1)
        mode_weights = mode_energy / mode_energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if token_metadata.dim() == 2:
            reshaped_metadata = token_metadata.reshape(patch_count, mode_count, -1)
            patch_centers = (mode_weights.unsqueeze(-1) * reshaped_metadata.unsqueeze(0)[..., :2]).sum(dim=2)
            reference_centers = reshaped_metadata[:, 0, :2]
        else:
            reshaped_metadata = token_metadata.reshape(batch, patch_count, mode_count, -1)
            patch_centers = (mode_weights.unsqueeze(-1) * reshaped_metadata[..., :2]).sum(dim=2)
            reference_centers = reshaped_metadata[:, :, 0, :2]
        stats.append(
            {
                "patch_energy": patch_energy,
                "patch_centers": patch_centers,
                "reference_centers": reference_centers,
            }
        )
    return stats


def packet_transport_symbol_proxy(
    prediction_features: Tensor,
    prediction_metadata: Tensor,
    encoding: WavepacketEncoding,
    target_features: Tensor,
    target_encoding: WavepacketEncoding,
    *,
    temperature: float = 0.05,
) -> dict[str, Tensor]:
    if prediction_features.shape[0] != target_features.shape[0]:
        raise ValueError("Prediction and target batches must agree.")
    pred_groups = grouped_patch_statistics(prediction_features, prediction_metadata, encoding)
    target_groups = grouped_patch_statistics(target_features, target_encoding.metadata, target_encoding)
    if len(pred_groups) != len(target_groups):
        raise ValueError("Prediction and target packet groups must agree.")
    transport_terms: list[Tensor] = []
    symbol_terms: list[Tensor] = []
    temperature = max(float(temperature), 1e-4)
    for pred_group, target_group in zip(pred_groups, target_groups, strict=True):
        pred_centers = pred_group["patch_centers"]
        pred_energy = pred_group["patch_energy"]
        target_reference_centers = target_group["reference_centers"]
        if target_reference_centers.dim() == 2:
            target_centers = target_reference_centers.unsqueeze(0).expand(pred_centers.shape[0], -1, -1)
        else:
            target_centers = target_reference_centers
        target_energy = target_group["patch_energy"].detach()
        normalized_energy = target_energy / target_energy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        distance_sq = torch.cdist(pred_centers, target_centers, p=2.0).square()
        logits = normalized_energy.clamp_min(1e-8).log().unsqueeze(1) - distance_sq / temperature
        assignment = torch.softmax(logits, dim=-1)
        aligned_energy = (assignment * target_energy.unsqueeze(1)).sum(dim=-1)
        transport_terms.append((assignment * distance_sq).sum(dim=-1).mean())
        symbol_terms.append(F.l1_loss(pred_energy, aligned_energy))
    return {
        "transport_proxy": torch.stack(transport_terms).mean(),
        "symbol_proxy": torch.stack(symbol_terms).mean(),
    }
