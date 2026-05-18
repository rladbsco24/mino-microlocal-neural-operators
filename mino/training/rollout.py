from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mino.metrics.packet_bridge import (
    enstrophy_error,
    high_frequency_energy_drift,
    packet_bridge_metrics,
    packet_energy_cascade_error,
    spectral_energy_error,
)
from mino.metrics.wavefront import relative_l2


@dataclass
class RolloutMetrics:
    loss: float
    mean_relative_l2: float
    final_relative_l2: float
    rollout_stability: float
    spectral_energy_error: float
    enstrophy_error: float
    high_frequency_energy_drift: float
    transport_budget: float
    symbol_budget: float
    residual_budget: float
    packet_trajectory_consistency: float
    packet_amplitude_error: float
    packet_residual_energy: float
    packet_energy_cascade_error: float


def autoregressive_rollout(model: nn.Module, initial: Tensor, steps: int) -> Tensor:
    current = initial
    predictions: list[Tensor] = []
    for _ in range(steps):
        current = model(current)
        predictions.append(current)
    return torch.stack(predictions, dim=1)


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if value == value]
    return _mean(finite)


def evaluate_rollout_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    proxy_temperature: float = 0.05,
    max_batches: int = 0,
) -> RolloutMetrics:
    model.eval()
    losses: list[float] = []
    mean_rels: list[float] = []
    final_rels: list[float] = []
    stability: list[float] = []
    spectral_errors: list[float] = []
    enstrophy_errors: list[float] = []
    high_frequency_errors: list[float] = []
    transport_budgets: list[float] = []
    symbol_budgets: list[float] = []
    residual_budgets: list[float] = []
    trajectory_errors: list[float] = []
    amplitude_errors: list[float] = []
    residual_energies: list[float] = []
    cascade_errors: list[float] = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_index, (initial, targets) in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            initial = initial.to(device)
            targets = targets.to(device)
            current = initial
            predictions: list[Tensor] = []
            for step in range(targets.shape[1]):
                if hasattr(model, "forward_with_diagnostics"):
                    diagnostics = model.forward_with_diagnostics(current)
                    prediction = diagnostics["prediction"]
                    if isinstance(prediction, Tensor):
                        bridge = packet_bridge_metrics(
                            model,
                            diagnostics,
                            targets[:, step],
                            proxy_temperature=proxy_temperature,
                        )
                        transport_budgets.append(bridge["transport_budget"])
                        symbol_budgets.append(bridge["symbol_budget"])
                        residual_budgets.append(bridge["residual_budget"])
                        trajectory_errors.append(bridge["packet_trajectory_consistency"])
                        amplitude_errors.append(bridge["packet_amplitude_error"])
                        residual_energies.append(bridge["packet_residual_energy"])
                        cascade_errors.append(bridge["packet_energy_cascade_error"])
                else:
                    prediction = model(current)
                predictions.append(prediction)
                current = prediction
            rollout = torch.stack(predictions, dim=1)
            losses.append(float(criterion(rollout, targets).item()))
            flat_prediction = rollout.reshape(rollout.shape[0] * rollout.shape[1], *rollout.shape[2:])
            flat_target = targets.reshape(targets.shape[0] * targets.shape[1], *targets.shape[2:])
            mean_rels.append(float(relative_l2(flat_prediction, flat_target).mean().item()))
            final_rels.append(float(relative_l2(rollout[:, -1], targets[:, -1]).mean().item()))
            initial_norm = torch.linalg.vector_norm(initial.reshape(initial.shape[0], -1), dim=-1).clamp_min(1e-8)
            rollout_norm = torch.linalg.vector_norm(rollout.reshape(rollout.shape[0], rollout.shape[1], -1), dim=-1)
            stability.append(float((rollout_norm.max(dim=1).values / initial_norm).mean().item()))
            spectral_errors.append(float(spectral_energy_error(flat_prediction, flat_target).mean().item()))
            enstrophy_errors.append(float(enstrophy_error(flat_prediction, flat_target).mean().item()))
            high_frequency_errors.append(float(high_frequency_energy_drift(flat_prediction, flat_target).mean().item()))
            if not cascade_errors:
                cascade_errors.append(float(packet_energy_cascade_error(flat_prediction, flat_target).mean().item()))
    return RolloutMetrics(
        loss=_mean(losses),
        mean_relative_l2=_mean(mean_rels),
        final_relative_l2=_mean(final_rels),
        rollout_stability=_mean(stability),
        spectral_energy_error=_mean(spectral_errors),
        enstrophy_error=_mean(enstrophy_errors),
        high_frequency_energy_drift=_mean(high_frequency_errors),
        transport_budget=_finite_mean(transport_budgets),
        symbol_budget=_finite_mean(symbol_budgets),
        residual_budget=_finite_mean(residual_budgets),
        packet_trajectory_consistency=_finite_mean(trajectory_errors),
        packet_amplitude_error=_finite_mean(amplitude_errors),
        packet_residual_energy=_finite_mean(residual_energies),
        packet_energy_cascade_error=_finite_mean(cascade_errors),
    )


def fit_one_step_rollout_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    restore_best: bool = True,
    transport_proxy_weight: float = 0.0,
    symbol_proxy_weight: float = 0.0,
    proxy_temperature: float = 0.05,
) -> dict[str, object]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    start = perf_counter()
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        transport_terms: list[float] = []
        symbol_terms: list[float] = []
        for initial, targets in train_loader:
            initial = initial.to(device)
            one_step_target = targets[:, 0].to(device)
            optimizer.zero_grad(set_to_none=True)
            if (transport_proxy_weight > 0.0 or symbol_proxy_weight > 0.0) and hasattr(model, "forward_with_diagnostics"):
                diagnostics = model.forward_with_diagnostics(initial)
                prediction = diagnostics["prediction"]
                loss = criterion(prediction, one_step_target)
                if hasattr(model, "proxy_losses_from_diagnostics"):
                    proxy_losses = model.proxy_losses_from_diagnostics(
                        diagnostics,
                        one_step_target,
                        proxy_temperature=proxy_temperature,
                    )
                    if transport_proxy_weight > 0.0:
                        transport_term = proxy_losses["transport_proxy"]
                        loss = loss + transport_proxy_weight * transport_term
                        transport_terms.append(float(transport_term.detach().item()))
                    if symbol_proxy_weight > 0.0:
                        symbol_term = proxy_losses["symbol_proxy"]
                        loss = loss + symbol_proxy_weight * symbol_term
                        symbol_terms.append(float(symbol_term.detach().item()))
            else:
                prediction = model(initial)
                loss = criterion(prediction, one_step_target)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate_rollout_model(model, val_loader, device, proxy_temperature=proxy_temperature, max_batches=1)
        if restore_best and validation.loss < best_val_loss:
            best_val_loss = validation.loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": _mean(losses),
                "val_loss": validation.loss,
                "val_mean_relative_l2": validation.mean_relative_l2,
                "val_final_relative_l2": validation.final_relative_l2,
                "train_transport_proxy": _mean(transport_terms),
                "train_symbol_proxy": _mean(symbol_terms),
            }
        )
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "runtime_seconds": perf_counter() - start}
