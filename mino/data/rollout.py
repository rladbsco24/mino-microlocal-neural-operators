from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class SequenceScenarioSpec:
    name: str
    family: str
    regime: str
    kind: str
    source_kind: str
    source: str
    path: Path | None = None
    synthetic_size: int = 64
    train_count: int = 96
    val_count: int = 16
    test_count: int = 16
    input_steps: int = 1
    rollout_steps: int = 20
    resolution: tuple[int, int] | None = None
    in_channels: int = 1
    out_channels: int = 1
    dt: float = 1.0
    evaluation_only: bool = False


@dataclass(frozen=True)
class SequenceScenarioLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    in_channels: int
    out_channels: int
    spatial_shape: tuple[int, int]
    input_steps: int
    rollout_steps: int
    spec: SequenceScenarioSpec


class RolloutArrayDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sequence dataset returning one-step inputs and multi-step targets.

    Samples are normalized to shape (N, T, C, H, W). The current MiNO models are
    one-step field-to-field maps, so the input is the last frame in the input
    window and the target is the next rollout_steps frames.
    """

    def __init__(self, sequences: Tensor, *, input_steps: int = 1, rollout_steps: int = 20) -> None:
        if sequences.ndim != 5:
            raise ValueError("sequences must have shape (N, T, C, H, W).")
        if input_steps < 1:
            raise ValueError("input_steps must be positive.")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be positive.")
        if sequences.shape[1] < input_steps + rollout_steps:
            raise ValueError(
                "sequence horizon is too short for the requested input_steps and rollout_steps."
            )
        self.sequences = sequences.float().contiguous()
        self.input_steps = int(input_steps)
        self.rollout_steps = int(rollout_steps)

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sequence = self.sequences[index]
        input_frame = sequence[self.input_steps - 1]
        targets = sequence[self.input_steps : self.input_steps + self.rollout_steps]
        return input_frame, targets


def _grid(size: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    coords = torch.linspace(0.0, 1.0, size, device=device)
    return torch.meshgrid(coords, coords, indexing="ij")


def _smooth_random_field(size: int, generator: torch.Generator) -> Tensor:
    base = torch.randn(size, size, generator=generator)
    spectrum = torch.fft.rfft2(base, norm="ortho")
    ky = torch.fft.fftfreq(size).view(-1, 1)
    kx = torch.fft.rfftfreq(size).view(1, -1)
    decay = 1.0 / (1.0 + 28.0 * (kx * kx + ky * ky))
    field = torch.fft.irfft2(spectrum * decay, s=(size, size), norm="ortho")
    return (field - field.mean()) / field.std().clamp_min(1e-6)


def _spectral_velocity(vorticity: Tensor) -> tuple[Tensor, Tensor]:
    size = int(vorticity.shape[-1])
    omega_hat = torch.fft.rfft2(vorticity, norm="ortho")
    ky = (2.0 * math.pi * torch.fft.fftfreq(size, device=vorticity.device)).view(-1, 1)
    kx = (2.0 * math.pi * torch.fft.rfftfreq(size, device=vorticity.device)).view(1, -1)
    k2 = kx.square() + ky.square()
    stream_hat = torch.where(k2 > 0, -omega_hat / k2.clamp_min(1e-12), torch.zeros_like(omega_hat))
    vx = torch.fft.irfft2(1j * ky * stream_hat, s=(size, size), norm="ortho").real
    vy = torch.fft.irfft2(-1j * kx * stream_hat, s=(size, size), norm="ortho").real
    max_speed = torch.stack([vx.abs().max(), vy.abs().max()]).max().clamp_min(1e-6)
    return 0.18 * vx / max_speed, 0.18 * vy / max_speed


def _advect_periodic(vorticity: Tensor, vx: Tensor, vy: Tensor, dt: float) -> Tensor:
    size = int(vorticity.shape[-1])
    y, x = _grid(size, device=vorticity.device)
    shifted_x = torch.remainder(x - dt * vx, 1.0)
    shifted_y = torch.remainder(y - dt * vy, 1.0)
    grid = torch.stack([shifted_x * 2.0 - 1.0, shifted_y * 2.0 - 1.0], dim=-1).unsqueeze(0)
    return F.grid_sample(
        vorticity.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(0)


def _diffuse(vorticity: Tensor, viscosity: float, dt: float) -> Tensor:
    size = int(vorticity.shape[-1])
    spectrum = torch.fft.rfft2(vorticity, norm="ortho")
    ky = (2.0 * math.pi * torch.fft.fftfreq(size, device=vorticity.device)).view(-1, 1)
    kx = (2.0 * math.pi * torch.fft.rfftfreq(size, device=vorticity.device)).view(1, -1)
    damping = torch.exp(-viscosity * dt * (kx.square() + ky.square()))
    return torch.fft.irfft2(spectrum * damping, s=(size, size), norm="ortho").real


def _navier_rollout_sequence(
    *,
    size: int,
    steps: int,
    generator: torch.Generator,
    viscosity: float = 1e-3,
    dt: float = 0.25,
) -> Tensor:
    y, x = _grid(size)
    forcing = 0.02 * (torch.sin(2.0 * math.pi * (x + y)) + torch.cos(2.0 * math.pi * (x - y)))
    vorticity = _smooth_random_field(size, generator)
    frames = [vorticity.unsqueeze(0)]
    for _ in range(steps):
        vx, vy = _spectral_velocity(vorticity)
        vorticity = _advect_periodic(vorticity, vx, vy, dt)
        vorticity = _diffuse(vorticity, viscosity, dt)
        vorticity = vorticity + dt * forcing
        vorticity = vorticity - vorticity.mean()
        frames.append(vorticity.unsqueeze(0))
    return torch.stack(frames, dim=0)


class SyntheticNavierStokesRolloutDataset(RolloutArrayDataset):
    def __init__(
        self,
        *,
        count: int = 128,
        size: int = 64,
        seed: int = 0,
        input_steps: int = 1,
        rollout_steps: int = 20,
        dt: float = 0.25,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        horizon = input_steps + rollout_steps
        sequences = torch.stack(
            [
                _navier_rollout_sequence(size=size, steps=horizon, generator=generator, dt=dt)
                for _ in range(count)
            ],
            dim=0,
        )
        super().__init__(sequences, input_steps=input_steps, rollout_steps=rollout_steps)


def _normalize_sequence_array(array: np.ndarray) -> Tensor:
    if array.ndim == 4:
        # PINO-style files are often (T, H, W, N); most NPZ files are (N, T, H, W).
        if array.shape[0] <= 256 and array.shape[1] == array.shape[2] and array.shape[-1] > array.shape[0]:
            array = np.moveaxis(array, -1, 0)[:, :, None, :, :]
        else:
            array = array[:, :, None, :, :]
    elif array.ndim == 5:
        pass
    else:
        raise ValueError("External rollout arrays must have shape (N,T,H,W), (N,T,C,H,W), or (T,H,W,N).")
    return torch.from_numpy(np.asarray(array)).float()


def _load_mat_array(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            f"Loading {path.name} requires scipy. Convert the Navier-Stokes cache to NPZ or install scipy."
        ) from error
    payload = loadmat(path)
    for key in ("u", "a", "data", "vorticity"):
        value = payload.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 4:
            return value
    candidates = [value for key, value in payload.items() if not key.startswith("__") and isinstance(value, np.ndarray)]
    for value in candidates:
        if value.ndim >= 4:
            return value
    raise ValueError(f"No sequence array found in {path}.")


def load_external_rollout_tensor(path: Path) -> Tensor:
    if not path.exists():
        raise FileNotFoundError(
            f"Navier-Stokes rollout cache not found: {path}. "
            "Use navier_stokes_long_rollout_synth for smoke tests or pass --sequence-cache-root."
        )
    suffix = path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(path)
        for key in ("sequences", "u", "data", "vorticity", "arr_0"):
            if key in archive:
                return _normalize_sequence_array(archive[key])
        raise ValueError(f"No supported rollout array key found in {path}.")
    if suffix == ".npy":
        return _normalize_sequence_array(np.load(path))
    if suffix in {".pt", ".pth"}:
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, dict):
            for key in ("sequences", "u", "data", "vorticity"):
                if key in tensor:
                    tensor = tensor[key]
                    break
        if not isinstance(tensor, Tensor):
            raise ValueError(f"No tensor sequence found in {path}.")
        return tensor.float()
    if suffix == ".mat":
        return _normalize_sequence_array(_load_mat_array(path))
    raise ValueError(f"Unsupported rollout cache format: {path.suffix}")


def _default_rollout_cache_root() -> Path:
    env_root = os.environ.get("MINO_ROLLOUT_CACHE_ROOT")
    if env_root:
        return Path(env_root)
    return Path(__file__).resolve().parents[3] / "pino_data"


def default_sequence_scenario_specs(sequence_cache_root: Path | None = None) -> dict[str, SequenceScenarioSpec]:
    cache_root = sequence_cache_root or _default_rollout_cache_root()
    pino_path = cache_root / "nv_V1e-3_N5000_T50.npz"
    if not pino_path.exists():
        pino_path = cache_root / "nv_V1e-3_N5000_T50.mat"
    return {
        "navier_stokes_long_rollout_synth": SequenceScenarioSpec(
            name="navier_stokes_long_rollout_synth",
            family="navier_stokes",
            regime="synthetic_rollout",
            kind="synthetic_rollout",
            source_kind="synthetic",
            source="navier_stokes_rollout",
            synthetic_size=64,
            train_count=96,
            val_count=16,
            test_count=16,
            input_steps=1,
            rollout_steps=20,
            resolution=(64, 64),
            in_channels=1,
            out_channels=1,
            dt=0.25,
        ),
        "navier_stokes_long_rollout_pino": SequenceScenarioSpec(
            name="navier_stokes_long_rollout_pino",
            family="navier_stokes",
            regime="long_rollout",
            kind="external_rollout",
            source_kind="pino_cache",
            source="nv_V1e-3_N5000_T50",
            path=pino_path,
            train_count=4800,
            val_count=100,
            test_count=100,
            input_steps=1,
            rollout_steps=50,
            resolution=(64, 64),
            in_channels=1,
            out_channels=1,
            dt=1.0,
        ),
    }


ROLLOUT_PROFILE_SCENARIOS: dict[str, tuple[str, ...]] = {
    "rollout_smoke": ("navier_stokes_long_rollout_synth",),
    "stage3_rollout_diagnostic": (
        "navier_stokes_long_rollout_synth",
        "navier_stokes_long_rollout_pino",
    ),
}


def list_rollout_profile_scenarios(profile: str) -> tuple[str, ...]:
    if profile not in ROLLOUT_PROFILE_SCENARIOS:
        raise ValueError(f"Unknown rollout profile: {profile}")
    return ROLLOUT_PROFILE_SCENARIOS[profile]


def get_sequence_scenario_spec(
    scenario_name: str,
    sequence_cache_root: Path | None = None,
) -> SequenceScenarioSpec:
    specs = default_sequence_scenario_specs(sequence_cache_root=sequence_cache_root)
    if scenario_name not in specs:
        raise ValueError(f"Unknown rollout scenario: {scenario_name}")
    return specs[scenario_name]


def _limit_dataset(dataset: Dataset[tuple[Tensor, Tensor]], count: int) -> Dataset[tuple[Tensor, Tensor]]:
    if count <= 0 or count >= len(dataset):
        return dataset
    return Subset(dataset, range(count))


def _split_external_sequences(sequences: Tensor, spec: SequenceScenarioSpec) -> tuple[Dataset[tuple[Tensor, Tensor]], ...]:
    dataset = RolloutArrayDataset(sequences, input_steps=spec.input_steps, rollout_steps=spec.rollout_steps)
    train_end = min(spec.train_count, len(dataset))
    val_end = min(train_end + spec.val_count, len(dataset))
    test_end = min(val_end + spec.test_count, len(dataset))
    if test_end <= val_end:
        raise ValueError(f"External rollout cache {spec.path} does not contain enough samples for a test split.")
    return (
        Subset(dataset, range(0, train_end)),
        Subset(dataset, range(train_end, val_end)),
        Subset(dataset, range(val_end, test_end)),
    )


def build_sequence_loaders(
    scenario_name: str,
    *,
    batch_size: int = 2,
    seed: int = 0,
    sequence_cache_root: Path | None = None,
    rollout_steps: int | None = None,
    synthetic_size: int | None = None,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    max_test_samples: int = 0,
) -> SequenceScenarioLoaders:
    spec = get_sequence_scenario_spec(scenario_name, sequence_cache_root=sequence_cache_root)
    if rollout_steps is not None:
        spec = SequenceScenarioSpec(**{**spec.__dict__, "rollout_steps": int(rollout_steps)})
    if synthetic_size is not None:
        spec = SequenceScenarioSpec(
            **{**spec.__dict__, "synthetic_size": int(synthetic_size), "resolution": (int(synthetic_size), int(synthetic_size))}
        )
    if spec.kind == "synthetic_rollout":
        train_dataset = SyntheticNavierStokesRolloutDataset(
            count=spec.train_count,
            size=spec.synthetic_size,
            seed=seed,
            input_steps=spec.input_steps,
            rollout_steps=spec.rollout_steps,
            dt=spec.dt,
        )
        val_dataset = SyntheticNavierStokesRolloutDataset(
            count=spec.val_count,
            size=spec.synthetic_size,
            seed=seed + 1,
            input_steps=spec.input_steps,
            rollout_steps=spec.rollout_steps,
            dt=spec.dt,
        )
        test_dataset = SyntheticNavierStokesRolloutDataset(
            count=spec.test_count,
            size=spec.synthetic_size,
            seed=seed + 2,
            input_steps=spec.input_steps,
            rollout_steps=spec.rollout_steps,
            dt=spec.dt,
        )
    elif spec.kind == "external_rollout":
        if spec.path is None:
            raise ValueError(f"Scenario {spec.name} is missing an external path.")
        train_dataset, val_dataset, test_dataset = _split_external_sequences(load_external_rollout_tensor(spec.path), spec)
    else:
        raise ValueError(f"Unsupported sequence scenario kind: {spec.kind}")

    train_dataset = _limit_dataset(train_dataset, max_train_samples)
    val_dataset = _limit_dataset(val_dataset, max_val_samples)
    test_dataset = _limit_dataset(test_dataset, max_test_samples)
    sample_input, sample_target = train_dataset[0]
    return SequenceScenarioLoaders(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        in_channels=int(sample_input.shape[0]),
        out_channels=int(sample_target.shape[1]),
        spatial_shape=(int(sample_input.shape[-2]), int(sample_input.shape[-1])),
        input_steps=spec.input_steps,
        rollout_steps=spec.rollout_steps,
        spec=spec,
    )


def sequence_plan_rows(
    *,
    scenarios: list[str],
    models: list[str],
    seeds: list[int],
    rollout_steps: list[int],
    hardware_profile: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        spec = get_sequence_scenario_spec(scenario)
        for model in models:
            for seed in seeds:
                for horizon in rollout_steps:
                    rows.append(
                        {
                            "scenario": scenario,
                            "family": spec.family,
                            "regime": spec.regime,
                            "source_kind": spec.source_kind,
                            "model": model,
                            "seed": seed,
                            "rollout_steps": horizon,
                            "hardware_profile": hardware_profile,
                        }
                    )
    return rows
