from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


def _grid(size: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    coords = torch.linspace(0.0, 1.0, size, device=device)
    return torch.meshgrid(coords, coords, indexing="ij")


def _smooth_random_field(size: int, generator: torch.Generator) -> Tensor:
    base = torch.randn(size, size, generator=generator)
    spectrum = torch.fft.rfft2(base, norm="ortho")
    ky = torch.fft.fftfreq(size).view(-1, 1)
    kx = torch.fft.rfftfreq(size).view(1, -1)
    decay = 1.0 / (1.0 + 20.0 * (kx * kx + ky * ky))
    return torch.fft.irfft2(spectrum * decay, s=(size, size), norm="ortho")


def _darcy_surrogate(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    coeff = torch.sigmoid(_smooth_random_field(size, generator) * 2.0)
    forcing = _smooth_random_field(size, generator)
    spectrum = torch.fft.rfft2(coeff * forcing, norm="ortho")
    ky = torch.fft.fftfreq(size).view(-1, 1)
    kx = torch.fft.rfftfreq(size).view(1, -1)
    green = 1.0 / (1e-2 + kx * kx + ky * ky)
    target = torch.fft.irfft2(spectrum * green, s=(size, size), norm="ortho")
    return coeff.unsqueeze(0).float(), target.unsqueeze(0).float()


def _navier_surrogate(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    vorticity = _smooth_random_field(size, generator)
    vx = float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator))
    vy = float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator))
    dt = 0.15
    y, x = _grid(size)
    shifted_x = (x - vx * dt).clamp(0.0, 1.0)
    shifted_y = (y - vy * dt).clamp(0.0, 1.0)
    grid = torch.stack([shifted_x * 2.0 - 1.0, shifted_y * 2.0 - 1.0], dim=-1).unsqueeze(0)
    advected = torch.nn.functional.grid_sample(
        vorticity.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    spectrum = torch.fft.rfft2(advected, norm="ortho")
    ky = torch.fft.fftfreq(size).view(-1, 1)
    kx = torch.fft.rfftfreq(size).view(1, -1)
    diffusion = torch.exp(-0.15 * (kx * kx + ky * ky))
    target = torch.fft.irfft2(spectrum * diffusion, s=(size, size), norm="ortho")
    return vorticity.unsqueeze(0).float(), target.unsqueeze(0).float()


def _packet_field(size: int, generator: torch.Generator, count: int = 3) -> Tensor:
    y, x = _grid(size)
    field = torch.zeros(size, size)
    for _ in range(count):
        center_x = float(torch.empty(1).uniform_(0.2, 0.8, generator=generator))
        center_y = float(torch.empty(1).uniform_(0.2, 0.8, generator=generator))
        freq_x = float(torch.empty(1).uniform_(4.0, 12.0, generator=generator))
        freq_y = float(torch.empty(1).uniform_(-12.0, 12.0, generator=generator))
        sigma = float(torch.empty(1).uniform_(0.05, 0.15, generator=generator))
        phase = 2.0 * math.pi * (freq_x * x + freq_y * y)
        window = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * sigma * sigma))
        amplitude = float(torch.empty(1).uniform_(0.5, 1.5, generator=generator))
        field = field + amplitude * window * torch.cos(phase)
    return field


def _gabor_packet(
    size: int,
    *,
    center_x: float,
    center_y: float,
    freq_x: float,
    freq_y: float,
    sigma: float,
    amplitude: float = 1.0,
    chirp: float = 0.0,
    phase0: float = 0.0,
) -> Tensor:
    y, x = _grid(size)
    dx = x - center_x
    dy = y - center_y
    phase = 2.0 * math.pi * (freq_x * x + freq_y * y + chirp * (dx * dx - dy * dy)) + phase0
    window = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    return amplitude * window * torch.cos(phase)


def _translate_field(field: Tensor, *, vx: float, vy: float, dt: float) -> Tensor:
    size = field.shape[-1]
    y, x = _grid(size)
    shifted_x = (x - vx * dt).clamp(0.0, 1.0)
    shifted_y = (y - vy * dt).clamp(0.0, 1.0)
    grid = torch.stack([shifted_x * 2.0 - 1.0, shifted_y * 2.0 - 1.0], dim=-1).unsqueeze(0)
    return torch.nn.functional.grid_sample(
        field.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(0)


def _helmholtz_surrogate(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    source = _packet_field(size, generator, count=4)
    medium = _smooth_random_field(size, generator)
    y, x = _grid(size)
    local_phase = 2.0 * math.pi * (0.4 * medium + 0.2 * x + 0.15 * y)
    envelope = 1.0 + 0.35 * medium
    target = envelope * (source * torch.cos(local_phase) - torch.sin(local_phase) * _smooth_random_field(size, generator))
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


def _wave_surrogate(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    source = _packet_field(size, generator, count=3)
    y, x = _grid(size)
    vx = 0.08 + 0.04 * _smooth_random_field(size, generator)
    vy = -0.06 + 0.04 * _smooth_random_field(size, generator)
    dt = 0.12
    shifted_x = (x - vx * dt).clamp(0.0, 1.0)
    shifted_y = (y - vy * dt).clamp(0.0, 1.0)
    grid = torch.stack([shifted_x * 2.0 - 1.0, shifted_y * 2.0 - 1.0], dim=-1).unsqueeze(0)
    target = torch.nn.functional.grid_sample(
        source.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    target = target + 0.1 * _packet_field(size, generator, count=1)
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


def _wave_bicharacteristic_control(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Controlled constant-flow wave packets with known translation geometry."""
    source = _packet_field(size, generator, count=3)
    y, x = _grid(size)
    vx = 0.10
    vy = -0.07
    dt = 0.15
    shifted_x = (x - vx * dt).clamp(0.0, 1.0)
    shifted_y = (y - vy * dt).clamp(0.0, 1.0)
    grid = torch.stack([shifted_x * 2.0 - 1.0, shifted_y * 2.0 - 1.0], dim=-1).unsqueeze(0)
    target = torch.nn.functional.grid_sample(
        source.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


def _wave_chirp_propagation(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Single controlled chirped packet with known short-time translation."""
    center_x = float(torch.empty(1).uniform_(0.28, 0.72, generator=generator))
    center_y = float(torch.empty(1).uniform_(0.28, 0.72, generator=generator))
    freq_x = float(torch.empty(1).uniform_(7.0, 13.0, generator=generator))
    freq_y = float(torch.empty(1).uniform_(-10.0, 10.0, generator=generator))
    chirp = float(torch.empty(1).uniform_(3.0, 7.0, generator=generator))
    sigma = float(torch.empty(1).uniform_(0.055, 0.095, generator=generator))
    source = _gabor_packet(
        size,
        center_x=center_x,
        center_y=center_y,
        freq_x=freq_x,
        freq_y=freq_y,
        sigma=sigma,
        amplitude=1.0,
        chirp=chirp,
    )
    target = _translate_field(source, vx=0.11, vy=-0.06, dt=0.14)
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


def _wave_two_packet_no_interaction(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Two separated packets that should translate without packet mixing."""
    phase0 = float(torch.empty(1).uniform_(0.0, 2.0 * math.pi, generator=generator))
    packet_a = _gabor_packet(
        size,
        center_x=0.30,
        center_y=0.34,
        freq_x=9.0,
        freq_y=5.0,
        sigma=0.070,
        amplitude=1.0,
        chirp=0.0,
        phase0=phase0,
    )
    packet_b = _gabor_packet(
        size,
        center_x=0.70,
        center_y=0.66,
        freq_x=-7.0,
        freq_y=11.0,
        sigma=0.075,
        amplitude=0.85,
        chirp=0.0,
        phase0=-phase0,
    )
    source = packet_a + packet_b
    target = _translate_field(source, vx=0.09, vy=-0.045, dt=0.16)
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


def _wave_variable_speed_nocaustic(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Smooth variable-speed short-time advection without the extra residual packet."""
    source = _packet_field(size, generator, count=3)
    y, x = _grid(size)
    medium = _smooth_random_field(size, generator)
    vx = 0.08 + 0.025 * torch.tanh(medium)
    vy = -0.05 + 0.025 * torch.tanh(_smooth_random_field(size, generator))
    dt = 0.10
    shifted_x = (x - vx * dt).clamp(0.0, 1.0)
    shifted_y = (y - vy * dt).clamp(0.0, 1.0)
    grid = torch.stack([shifted_x * 2.0 - 1.0, shifted_y * 2.0 - 1.0], dim=-1).unsqueeze(0)
    target = torch.nn.functional.grid_sample(
        source.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


def _helmholtz_local_window_control(size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Mild local-window Helmholtz surrogate with a single dominant phase branch."""
    source = _packet_field(size, generator, count=3)
    medium = 0.25 * _smooth_random_field(size, generator)
    y, x = _grid(size)
    phase = 2.0 * math.pi * (0.18 * x + 0.12 * y + medium)
    target = (1.0 + 0.10 * medium) * (source * torch.cos(phase))
    return source.unsqueeze(0).float(), target.unsqueeze(0).float()


SCENARIOS: dict[str, Callable[[int, torch.Generator], tuple[Tensor, Tensor]]] = {
    "darcy": _darcy_surrogate,
    "navier_stokes": _navier_surrogate,
    "helmholtz": _helmholtz_surrogate,
    "wave": _wave_surrogate,
    "wave_bicharacteristic_control": _wave_bicharacteristic_control,
    "wave_chirp_propagation": _wave_chirp_propagation,
    "wave_two_packet_no_interaction": _wave_two_packet_no_interaction,
    "wave_variable_speed_nocaustic": _wave_variable_speed_nocaustic,
    "helmholtz_local_window_control": _helmholtz_local_window_control,
}


class SyntheticOperatorDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, scenario: str, count: int = 128, size: int = 64, seed: int = 0) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.count = count
        self.size = size
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        self.samples = [SCENARIOS[scenario](size, generator) for _ in range(count)]

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.samples[index]


def build_dataloaders(
    scenario: str,
    batch_size: int = 8,
    size: int = 64,
    train_count: int = 96,
    val_count: int = 16,
    test_count: int = 16,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SyntheticOperatorDataset(
        scenario=scenario,
        count=train_count + val_count + test_count,
        size=size,
        seed=seed,
    )
    lengths = [train_count, val_count, test_count]
    train_set, val_set, test_set = random_split(
        dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
