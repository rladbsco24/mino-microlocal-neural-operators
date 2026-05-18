from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from mino.data.rollout import RolloutArrayDataset, SyntheticNavierStokesRolloutDataset, build_sequence_loaders
from mino.metrics.packet_bridge import packet_bridge_metrics, spectral_energy_error
from mino.models.mino import build_model
from mino.training.rollout import evaluate_rollout_model


def test_synthetic_rollout_dataset_shapes_are_deterministic() -> None:
    dataset_a = SyntheticNavierStokesRolloutDataset(count=2, size=16, seed=3, rollout_steps=4)
    dataset_b = SyntheticNavierStokesRolloutDataset(count=2, size=16, seed=3, rollout_steps=4)
    initial, targets = dataset_a[0]
    assert initial.shape == (1, 16, 16)
    assert targets.shape == (4, 1, 16, 16)
    assert torch.allclose(initial, dataset_b[0][0])
    assert torch.allclose(targets, dataset_b[0][1])


def test_sequence_loader_reports_rollout_metadata() -> None:
    loaders = build_sequence_loaders(
        "navier_stokes_long_rollout_synth",
        batch_size=1,
        seed=0,
        rollout_steps=3,
        synthetic_size=16,
        max_train_samples=2,
        max_val_samples=1,
        max_test_samples=1,
    )
    inputs, targets = next(iter(loaders.train_loader))
    assert loaders.rollout_steps == 3
    assert inputs.shape == (1, 1, 16, 16)
    assert targets.shape == (1, 3, 1, 16, 16)


def test_rollout_evaluator_is_finite_with_identity_model() -> None:
    base = torch.randn(2, 5, 1, 16, 16)
    dataset = RolloutArrayDataset(base, input_steps=1, rollout_steps=4)
    metrics = evaluate_rollout_model(nn.Identity(), DataLoader(dataset, batch_size=1), torch.device("cpu"))
    assert metrics.mean_relative_l2 >= 0.0
    assert metrics.final_relative_l2 >= 0.0
    assert metrics.rollout_stability >= 0.0
    assert metrics.spectral_energy_error >= 0.0


def test_packet_bridge_metrics_are_finite_for_mino_plus() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "width": 16,
            "depth": 1,
            "patch_size": 16,
            "stride": 16,
            "max_modes": 6,
            "window_type": "gaussian",
            "mode_strategy": "shell_balanced",
            "transport_stencil": 2,
            "local_refine_channels": 8,
        },
    )
    x = torch.randn(2, 1, 16, 16)
    target = torch.randn(2, 1, 16, 16)
    diagnostics = model.forward_with_diagnostics(x)
    metrics = packet_bridge_metrics(model, diagnostics, target, proxy_temperature=0.1)
    assert metrics["transport_budget"] >= 0.0
    assert metrics["symbol_budget"] >= 0.0
    assert metrics["residual_budget"] >= 0.0
    assert metrics["packet_energy_cascade_error"] >= 0.0


def test_spectral_energy_error_zero_on_identity() -> None:
    x = torch.randn(2, 1, 16, 16)
    assert torch.allclose(spectral_energy_error(x, x), torch.zeros(2), atol=1e-6)
