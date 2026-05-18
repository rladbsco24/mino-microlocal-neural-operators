from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .synthetic import SyntheticOperatorDataset


@dataclass(frozen=True)
class ScenarioSpec:
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
    resolution: tuple[int, int] | None = None
    in_channels: int | None = None
    out_channels: int | None = None
    evaluation_only: bool = False


@dataclass(frozen=True)
class ScenarioLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    in_channels: int
    out_channels: int
    spatial_shape: tuple[int, int]
    spec: ScenarioSpec


class NPZOperatorDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Inputs and targets must have the same sample count.")
        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


def _default_tcno_cache_root() -> Path:
    return Path(__file__).resolve().parents[3] / "tcno_icml2026_draft" / "generated" / "trainable_cache"


def _first_matching_path(cache_root: Path, slug: str) -> Path:
    preferred = (
        f"{slug}_v3learnedlocal_train96_val24_test24.npz",
        f"{slug}_v3learnedlocal_train64_val16_test16.npz",
        f"{slug}_v3learnedlocal_train48_val12_test12.npz",
        f"{slug}_train48_val12_test12.npz",
        f"{slug}_v3learnedlocal_train32_val8_test8.npz",
        f"{slug}_train32_val8_test8.npz",
        f"{slug}_v3learnedlocal_train24_val8_test8.npz",
        f"{slug}_v3learnedlocal_train8_val4_test4.npz",
        f"{slug}_train8_val4_test4.npz",
        f"{slug}_v2local_train96_val24_test24.npz",
        f"{slug}_v2local_train32_val8_test8.npz",
    )
    for filename in preferred:
        candidate = cache_root / filename
        if candidate.exists():
            return candidate
    matches = sorted(cache_root.glob(f"{slug}_*.npz"))
    if matches:
        return matches[0]
    return cache_root / f"{slug}.npz"


def _synthetic_spec(name: str, source: str, family: str) -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        family=family,
        regime="synthetic",
        kind="synthetic",
        source_kind="synthetic",
        source=source,
        synthetic_size=64,
        train_count=96,
        val_count=16,
        test_count=16,
        resolution=(64, 64),
        in_channels=1,
        out_channels=1,
    )


def _tcno_spec(cache_root: Path, slug: str, family: str, regime: str, *, evaluation_only: bool = False) -> ScenarioSpec:
    return ScenarioSpec(
        name=slug,
        family=family,
        regime=regime,
        kind="tcno_npz",
        source_kind="tcno_cache",
        source=slug,
        path=_first_matching_path(cache_root, slug),
        resolution=(192, 192),
        in_channels=3,
        out_channels=1,
        evaluation_only=evaluation_only,
    )


def default_scenario_specs(tcno_cache_root: Path | None = None) -> dict[str, ScenarioSpec]:
    cache_root = tcno_cache_root or _default_tcno_cache_root()
    specs = {
        "darcy_synth": _synthetic_spec("darcy_synth", "darcy", "darcy"),
        "navier_stokes_synth": _synthetic_spec("navier_stokes_synth", "navier_stokes", "navier_stokes"),
        "wave_synth": _synthetic_spec("wave_synth", "wave", "wave"),
        "wave_bicharacteristic_control": _synthetic_spec(
            "wave_bicharacteristic_control",
            "wave_bicharacteristic_control",
            "wave",
        ),
        "wave_chirp_propagation": _synthetic_spec(
            "wave_chirp_propagation",
            "wave_chirp_propagation",
            "wave",
        ),
        "wave_two_packet_no_interaction": _synthetic_spec(
            "wave_two_packet_no_interaction",
            "wave_two_packet_no_interaction",
            "wave",
        ),
        "wave_variable_speed_nocaustic": _synthetic_spec(
            "wave_variable_speed_nocaustic",
            "wave_variable_speed_nocaustic",
            "wave",
        ),
        "helmholtz_local_window_control": _synthetic_spec(
            "helmholtz_local_window_control",
            "helmholtz_local_window_control",
            "helmholtz",
        ),
        "poisson_robin_control": _tcno_spec(cache_root, "poisson_robin_control", "poisson_robin", "control"),
        "poisson_robin_positive": _tcno_spec(cache_root, "poisson_robin_positive", "poisson_robin", "positive"),
        "diffusion_neumann_control": _tcno_spec(cache_root, "diffusion_neumann_control", "diffusion_neumann", "control"),
        "diffusion_neumann_positive": _tcno_spec(cache_root, "diffusion_neumann_positive", "diffusion_neumann", "positive"),
        "helmholtz_dirichlet_control": _tcno_spec(cache_root, "helmholtz_dirichlet_control", "helmholtz_dirichlet", "control"),
        "helmholtz_dirichlet_positive": _tcno_spec(cache_root, "helmholtz_dirichlet_positive", "helmholtz_dirichlet", "positive"),
        "helmholtz_impedance_control": _tcno_spec(cache_root, "helmholtz_impedance_control", "helmholtz_impedance", "control"),
        "helmholtz_impedance_positive": _tcno_spec(cache_root, "helmholtz_impedance_positive", "helmholtz_impedance", "positive"),
        "helmholtz_variable_control": _tcno_spec(cache_root, "helmholtz_variable_control", "helmholtz_variable", "control"),
        "helmholtz_variable_positive": _tcno_spec(cache_root, "helmholtz_variable_positive", "helmholtz_variable", "positive"),
        "helmholtz_highk_control": _tcno_spec(cache_root, "helmholtz_highk_control", "helmholtz_highk", "control"),
        "helmholtz_highk_positive": _tcno_spec(cache_root, "helmholtz_highk_positive", "helmholtz_highk", "positive"),
        "helmholtz_multimodal_control": _tcno_spec(cache_root, "helmholtz_multimodal_control", "helmholtz_multimodal", "control"),
        "helmholtz_multimodal_positive": _tcno_spec(cache_root, "helmholtz_multimodal_positive", "helmholtz_multimodal", "positive"),
        "helmholtz_partial_control": _tcno_spec(cache_root, "helmholtz_partial_control", "helmholtz_partial", "control"),
        "helmholtz_partial_positive": _tcno_spec(cache_root, "helmholtz_partial_positive", "helmholtz_partial", "positive"),
        "helmholtz_highk_ood_control": _tcno_spec(cache_root, "helmholtz_highk_ood_control", "helmholtz_highk_ood", "control", evaluation_only=True),
        "helmholtz_highk_ood_positive": _tcno_spec(cache_root, "helmholtz_highk_ood_positive", "helmholtz_highk_ood", "positive", evaluation_only=True),
    }
    return specs


PROFILE_SCENARIOS: dict[str, tuple[str, ...]] = {
    "mixed_smoke": (
        "darcy_synth",
        "navier_stokes_synth",
        "wave_synth",
        "helmholtz_dirichlet_control",
        "diffusion_neumann_control",
    ),
    "mixed_medium": (
        "darcy_synth",
        "navier_stokes_synth",
        "wave_synth",
        "helmholtz_dirichlet_control",
        "helmholtz_dirichlet_positive",
        "diffusion_neumann_control",
        "diffusion_neumann_positive",
    ),
    "stage0_smoke": (
        "wave_synth",
        "helmholtz_dirichlet_control",
        "diffusion_neumann_control",
    ),
    "stage1_breadth": (
        "darcy_synth",
        "navier_stokes_synth",
        "wave_synth",
        "poisson_robin_control",
        "poisson_robin_positive",
        "diffusion_neumann_control",
        "diffusion_neumann_positive",
        "helmholtz_dirichlet_control",
        "helmholtz_dirichlet_positive",
        "helmholtz_impedance_control",
        "helmholtz_impedance_positive",
        "helmholtz_variable_control",
        "helmholtz_variable_positive",
        "helmholtz_highk_control",
        "helmholtz_highk_positive",
        "helmholtz_multimodal_control",
        "helmholtz_multimodal_positive",
        "helmholtz_partial_control",
        "helmholtz_partial_positive",
    ),
    "stage2_shortlist": (
        "wave_synth",
        "diffusion_neumann_control",
        "diffusion_neumann_positive",
        "helmholtz_dirichlet_control",
        "helmholtz_dirichlet_positive",
        "helmholtz_variable_control",
        "helmholtz_variable_positive",
        "helmholtz_highk_positive",
    ),
    "stage3_flagship": (
        "wave_synth",
        "helmholtz_dirichlet_control",
        "helmholtz_dirichlet_positive",
        "helmholtz_variable_control",
        "helmholtz_variable_positive",
        "helmholtz_highk_control",
        "helmholtz_highk_positive",
        "helmholtz_multimodal_control",
        "helmholtz_multimodal_positive",
        "helmholtz_partial_control",
        "helmholtz_partial_positive",
    ),
    "branch_id": (
        "wave_bicharacteristic_control",
        "wave_chirp_propagation",
        "wave_two_packet_no_interaction",
        "wave_variable_speed_nocaustic",
        "wave_synth",
        "helmholtz_local_window_control",
        "helmholtz_variable_control",
    ),
}


def list_profile_scenarios(profile: str) -> tuple[str, ...]:
    if profile not in PROFILE_SCENARIOS:
        raise ValueError(f"Unknown benchmark profile: {profile}")
    return PROFILE_SCENARIOS[profile]


def get_scenario_spec(scenario_name: str, tcno_cache_root: Path | None = None) -> ScenarioSpec:
    specs = default_scenario_specs(tcno_cache_root=tcno_cache_root)
    if scenario_name not in specs:
        raise ValueError(f"Unknown benchmark scenario: {scenario_name}")
    return specs[scenario_name]


def _build_synthetic_loaders(spec: ScenarioSpec, batch_size: int, seed: int) -> ScenarioLoaders:
    train_dataset = SyntheticOperatorDataset(spec.source, count=spec.train_count, size=spec.synthetic_size, seed=seed)
    val_dataset = SyntheticOperatorDataset(spec.source, count=spec.val_count, size=spec.synthetic_size, seed=seed + 1)
    test_dataset = SyntheticOperatorDataset(spec.source, count=spec.test_count, size=spec.synthetic_size, seed=seed + 2)
    sample_input, sample_target = train_dataset[0]
    return ScenarioLoaders(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        in_channels=int(sample_input.shape[0]),
        out_channels=int(sample_target.shape[0]),
        spatial_shape=(int(sample_input.shape[-2]), int(sample_input.shape[-1])),
        spec=spec,
    )


def _build_npz_loaders(spec: ScenarioSpec, batch_size: int) -> ScenarioLoaders:
    if spec.path is None:
        raise ValueError(f"Scenario {spec.name} is missing an NPZ path.")
    if not spec.path.exists():
        raise FileNotFoundError(f"Scenario cache not found: {spec.path}")
    archive = np.load(spec.path)
    train_dataset = NPZOperatorDataset(archive["train_inputs"], archive["train_targets"])
    val_dataset = NPZOperatorDataset(archive["val_inputs"], archive["val_targets"])
    test_dataset = NPZOperatorDataset(archive["test_inputs"], archive["test_targets"])
    sample_input, sample_target = train_dataset[0]
    return ScenarioLoaders(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        in_channels=int(sample_input.shape[0]),
        out_channels=int(sample_target.shape[0]),
        spatial_shape=(int(sample_input.shape[-2]), int(sample_input.shape[-1])),
        spec=spec,
    )


def build_benchmark_loaders(
    scenario_name: str,
    batch_size: int = 8,
    seed: int = 0,
    tcno_cache_root: Path | None = None,
) -> ScenarioLoaders:
    spec = get_scenario_spec(scenario_name, tcno_cache_root=tcno_cache_root)
    if spec.kind == "synthetic":
        return _build_synthetic_loaders(spec, batch_size=batch_size, seed=seed)
    if spec.kind == "tcno_npz":
        return _build_npz_loaders(spec, batch_size=batch_size)
    raise ValueError(f"Unsupported scenario kind: {spec.kind}")
