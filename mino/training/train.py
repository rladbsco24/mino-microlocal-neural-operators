from __future__ import annotations

import copy
from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mino.metrics.wavefront import packet_consistency, phase_error, relative_l2


def _finite_tensor(tensor: Tensor, *, cap: float | None = 1.0e6) -> Tensor:
    """Keep executable training probes finite without changing ordinary values."""

    if cap is None:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    cap = max(float(abs(cap)), 1e-6)
    return torch.nan_to_num(tensor, nan=0.0, posinf=cap, neginf=-cap).clamp(-cap, cap)


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


def _negative_laplacian(field: Tensor) -> Tensor:
    height, width = field.shape[-2:]
    fy = torch.fft.fftfreq(height, device=field.device, dtype=field.dtype).view(height, 1)
    fx = torch.fft.rfftfreq(width, device=field.device, dtype=field.dtype).view(1, width // 2 + 1)
    radius_sq = fy.square() + fx.square()
    spectrum = torch.fft.rfft2(field, norm="ortho")
    return torch.fft.irfft2((2.0 * torch.pi) ** 2 * radius_sq.view(1, 1, height, width // 2 + 1) * spectrum, s=(height, width), norm="ortho")


def _helmholtz_residual_proxy(prediction: Tensor, inputs: Tensor, *, wavenumber: float, refractive_index: float) -> Tensor:
    """Weak finite Helmholtz residual ``-Delta u - k^2 n u = f``.

    The TCNO-style Helmholtz inputs are heterogeneous, so this is deliberately
    a regularizer rather than a theorem-grade residual certificate.  Channel 0
    is used as a source proxy and channel 1, when present, as a bounded
    refractive-index modulation around ``refractive_index``.
    """
    prediction = _finite_tensor(prediction)
    inputs = _finite_tensor(inputs)
    source = inputs[:, :1]
    if inputs.shape[1] > 1:
        n_field = float(refractive_index) * (1.0 + 0.25 * torch.tanh(inputs[:, 1:2]))
    else:
        n_field = torch.full_like(prediction, float(refractive_index))
    k = torch.as_tensor(float(wavenumber), device=prediction.device, dtype=prediction.dtype)
    residual = _finite_tensor(_negative_laplacian(prediction) - k.square() * n_field * prediction - source)
    normalizer = source.square().mean().detach() + (k.square() * n_field * prediction).square().mean().detach() + 1e-8
    value = residual.square().mean() / normalizer.clamp_min(1e-8)
    return _finite_tensor(value, cap=1.0e6)


def _complex_pair_view(field: Tensor) -> Tensor | None:
    """Return ``(..., real/imag, H, W)`` pairs when channels encode complex fields.

    The hook is intentionally inert for real-valued PDE datasets.  When a
    benchmark provides real/imaginary output channels, the loss below treats
    them as a genuine finite complex field rather than manufacturing a
    quadrature channel from a scalar image.
    """

    channels = field.shape[1]
    if channels < 2 or channels % 2 != 0:
        return None
    batch, _, height, width = field.shape
    return field.reshape(batch, channels // 2, 2, height, width)


def _complex_pair_relative_loss(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_pairs = _complex_pair_view(prediction)
    target_pairs = _complex_pair_view(target)
    if pred_pairs is None or target_pairs is None:
        return prediction.new_tensor(0.0)
    pred_pairs = _finite_tensor(pred_pairs)
    target_pairs = _finite_tensor(target_pairs)
    diff = pred_pairs - target_pairs
    numerator = diff.square().sum(dim=2).mean()
    denominator = target_pairs.square().sum(dim=2).mean().detach().clamp_min(eps)
    return _finite_tensor(numerator / denominator, cap=1.0e6)


def _complex_pair_phase_loss(prediction: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    pred_pairs = _complex_pair_view(prediction)
    target_pairs = _complex_pair_view(target)
    if pred_pairs is None or target_pairs is None:
        return prediction.new_tensor(0.0)
    pred_pairs = _finite_tensor(pred_pairs)
    target_pairs = _finite_tensor(target_pairs)
    pred_real, pred_imag = pred_pairs[:, :, 0], pred_pairs[:, :, 1]
    target_real, target_imag = target_pairs[:, :, 0], target_pairs[:, :, 1]
    inner_real = (pred_real * target_real + pred_imag * target_imag).flatten(1).sum(dim=1)
    inner_imag = (pred_imag * target_real - pred_real * target_imag).flatten(1).sum(dim=1)
    inner_abs = torch.sqrt(inner_real.square() + inner_imag.square() + eps)
    pred_norm = torch.sqrt((pred_real.square() + pred_imag.square()).flatten(1).sum(dim=1) + eps)
    target_norm = torch.sqrt((target_real.square() + target_imag.square()).flatten(1).sum(dim=1) + eps)
    coherence = inner_abs / (pred_norm * target_norm).clamp_min(eps)
    return _finite_tensor((1.0 - coherence.clamp(0.0, 1.0)).mean(), cap=1.0e6)


@dataclass
class EpochMetrics:
    loss: float
    relative_l2: float
    phase_error: float
    packet_consistency: float


def _aggregate_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = _finite_tensor(prediction)
    target = _finite_tensor(target)
    return {
        "relative_l2": float(relative_l2(prediction, target).mean().item()),
        "phase_error": float(phase_error(prediction, target).mean().item()),
        "packet_consistency": float(packet_consistency(prediction, target).mean().item()),
    }


def _set_refinement_requires_grad(model: nn.Module, enabled: bool) -> None:
    for name, parameter in model.named_parameters():
        if "local_refine" in name or ".route." in name or name.startswith("route."):
            parameter.requires_grad_(enabled)


def _symbol_identity_regularizer(model: nn.Module, device: torch.device) -> Tensor:
    """Penalize non-identity local symbol corrections.

    This is intentionally a parameter-space regularizer rather than a claim
    that the learned symbol is small.  It keeps theorem-facing source symbols,
    local tube-coordinate corrections, and retained-edge packet kernels from
    absorbing the carrier/landing mechanism unless the data pays for it.
    """

    total = torch.zeros((), device=device)
    count = 0
    for module in model.modules():
        local_kernel = getattr(module, "local_kernel_proj", None)
        if isinstance(local_kernel, nn.Linear):
            total = total + local_kernel.weight.square().mean()
            count += 1
        edge_symbol = getattr(module, "edge_symbol", None)
        if edge_symbol is not None:
            for parameter in edge_symbol.parameters():
                total = total + parameter.square().mean()
                count += 1
    if count == 0:
        return total
    return total / float(count)


def _batch_metadata(metadata: Tensor, batch: int) -> Tensor:
    if metadata.dim() == 2:
        return metadata.unsqueeze(0).expand(batch, -1, -1)
    if metadata.dim() == 3:
        return metadata
    raise ValueError("metadata must have shape (tokens, dim) or (batch, tokens, dim).")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> EpochMetrics:
    model.eval()
    losses: list[float] = []
    rels: list[float] = []
    phases: list[float] = []
    packets: list[float] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = _finite_tensor(targets.to(device))
            prediction = _finite_tensor(model(inputs))
            loss = criterion(prediction, targets)
            metrics = _aggregate_metrics(prediction, targets)
            losses.append(float(loss.item()))
            rels.append(metrics["relative_l2"])
            phases.append(metrics["phase_error"])
            packets.append(metrics["packet_consistency"])
    return EpochMetrics(
        loss=sum(losses) / max(len(losses), 1),
        relative_l2=sum(rels) / max(len(rels), 1),
        phase_error=sum(phases) / max(len(phases), 1),
        packet_consistency=sum(packets) / max(len(packets), 1),
    )


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    restore_best: bool = True,
    transport_proxy_weight: float = 0.0,
    symbol_proxy_weight: float = 0.0,
    proxy_temperature: float = 0.05,
    core_field_weight: float = 0.0,
    residual_energy_weight: float = 0.0,
    route_l1_weight: float = 0.0,
    canonical_loss_weight: float = 0.0,
    symbol_order_loss_weight: float = 0.0,
    symbol_order_target: float = 0.0,
    symbol_seminorm_loss_weight: float = 0.0,
    symbol_seminorm_target: float = 0.0,
    packet_space_loss_weight: float = 0.0,
    highfreq_core_loss_weight: float = 0.0,
    highfreq_cutoff: float = 0.25,
    helmholtz_residual_loss_weight: float = 0.0,
    helmholtz_residual_wavenumber: float = 12.0,
    helmholtz_residual_refractive_index: float = 1.0,
    complex_pair_loss_weight: float = 0.0,
    complex_phase_loss_weight: float = 0.0,
    symbol_identity_loss_weight: float = 0.0,
    metadata_flow_loss_weight: float = 0.0,
    metadata_flow_delta: tuple[float, float] | None = None,
    branch_entropy_weight: float = 0.0,
    branch_diversity_weight: float = 0.0,
    core_warmup_epochs: int = 0,
    freeze_refinement_epochs: int = 0,
    progress_every_epochs: int = 0,
    progress_label: str = "",
) -> dict[str, object]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    model.to(device)
    start = perf_counter()
    best_val_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        _set_refinement_requires_grad(model, enabled=epoch >= freeze_refinement_epochs)
        epoch_losses: list[float] = []
        transport_proxy_terms: list[float] = []
        symbol_proxy_terms: list[float] = []
        core_field_terms: list[float] = []
        residual_energy_terms: list[float] = []
        route_l1_terms: list[float] = []
        canonical_terms: list[float] = []
        symbol_order_terms: list[float] = []
        symbol_seminorm_terms: list[float] = []
        packet_space_terms: list[float] = []
        highfreq_core_terms: list[float] = []
        helmholtz_residual_terms: list[float] = []
        complex_pair_terms: list[float] = []
        complex_phase_terms: list[float] = []
        symbol_identity_terms: list[float] = []
        metadata_flow_terms: list[float] = []
        branch_entropy_terms: list[float] = []
        branch_diversity_terms: list[float] = []
        core_warmup_terms: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = _finite_tensor(targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            prediction: Tensor
            loss = torch.tensor(0.0, device=device)
            needs_diagnostics = (
                transport_proxy_weight > 0.0
                or symbol_proxy_weight > 0.0
                or core_field_weight > 0.0
                or residual_energy_weight > 0.0
                or route_l1_weight > 0.0
                or canonical_loss_weight > 0.0
                or symbol_order_loss_weight > 0.0
                or symbol_seminorm_loss_weight > 0.0
                or packet_space_loss_weight > 0.0
                or highfreq_core_loss_weight > 0.0
                or helmholtz_residual_loss_weight > 0.0
                or complex_pair_loss_weight > 0.0
                or complex_phase_loss_weight > 0.0
                or symbol_identity_loss_weight > 0.0
                or metadata_flow_loss_weight > 0.0
                or branch_entropy_weight > 0.0
                or branch_diversity_weight > 0.0
            )
            if needs_diagnostics and hasattr(model, "forward_with_diagnostics"):
                diagnostics = model.forward_with_diagnostics(inputs)
                prediction = _finite_tensor(diagnostics["prediction"])
                if epoch < core_warmup_epochs and "core_prediction" in diagnostics:
                    core_warmup_term = criterion(diagnostics["core_prediction"], targets)
                    loss = core_warmup_term
                    core_warmup_terms.append(float(core_warmup_term.detach().item()))
                else:
                    loss = criterion(prediction, targets)
                if core_field_weight > 0.0 and "core_prediction" in diagnostics:
                    core_term = criterion(diagnostics["core_prediction"], targets)
                    loss = loss + core_field_weight * core_term
                    core_field_terms.append(float(core_term.detach().item()))
                if residual_energy_weight > 0.0 and "refine_correction" in diagnostics:
                    correction = diagnostics["refine_correction"]
                    target_energy = targets.square().mean().detach().clamp_min(1e-8)
                    residual_term = correction.square().mean() / target_energy
                    loss = loss + residual_energy_weight * residual_term
                    residual_energy_terms.append(float(residual_term.detach().item()))
                if highfreq_core_loss_weight > 0.0 and "core_prediction" in diagnostics:
                    highfreq_term = criterion(
                        _highpass_field(diagnostics["core_prediction"], highfreq_cutoff),
                        _highpass_field(targets, highfreq_cutoff),
                    )
                    loss = loss + highfreq_core_loss_weight * highfreq_term
                    highfreq_core_terms.append(float(highfreq_term.detach().item()))
                if helmholtz_residual_loss_weight > 0.0:
                    helmholtz_term = _helmholtz_residual_proxy(
                        prediction,
                        inputs,
                        wavenumber=helmholtz_residual_wavenumber,
                        refractive_index=helmholtz_residual_refractive_index,
                    )
                    loss = loss + helmholtz_residual_loss_weight * helmholtz_term
                    helmholtz_residual_terms.append(float(helmholtz_term.detach().item()))
                if complex_pair_loss_weight > 0.0:
                    complex_pair_term = _complex_pair_relative_loss(prediction, targets)
                    loss = loss + complex_pair_loss_weight * complex_pair_term
                    complex_pair_terms.append(float(complex_pair_term.detach().item()))
                if complex_phase_loss_weight > 0.0:
                    complex_phase_term = _complex_pair_phase_loss(prediction, targets)
                    loss = loss + complex_phase_loss_weight * complex_phase_term
                    complex_phase_terms.append(float(complex_phase_term.detach().item()))
                if symbol_identity_loss_weight > 0.0:
                    symbol_identity_term = _symbol_identity_regularizer(model, device)
                    loss = loss + symbol_identity_loss_weight * symbol_identity_term
                    symbol_identity_terms.append(float(symbol_identity_term.detach().item()))
                if (
                    metadata_flow_loss_weight > 0.0
                    and metadata_flow_delta is not None
                    and "final_metadata" in diagnostics
                    and "encoding" in diagnostics
                ):
                    final_metadata = diagnostics["final_metadata"]
                    initial_metadata = diagnostics["encoding"].metadata.to(
                        device=final_metadata.device,
                        dtype=final_metadata.dtype,
                    )
                    initial_batch = _batch_metadata(initial_metadata, final_metadata.shape[0])
                    target_delta = torch.as_tensor(
                        metadata_flow_delta,
                        device=final_metadata.device,
                        dtype=final_metadata.dtype,
                    ).view(1, 1, 2)
                    target_scale = target_delta.square().mean().detach().clamp_min(1e-6)
                    metadata_flow_term = (
                        (final_metadata[..., :2] - initial_batch[..., :2] - target_delta).square().mean()
                        / target_scale
                    )
                    loss = loss + metadata_flow_loss_weight * metadata_flow_term
                    metadata_flow_terms.append(float(metadata_flow_term.detach().item()))
                if route_l1_weight > 0.0 and "route_abs_mean" in diagnostics:
                    route_term = diagnostics["route_abs_mean"]
                    loss = loss + route_l1_weight * route_term
                    route_l1_terms.append(float(route_term.detach().item()))
                if hasattr(model, "proxy_losses_from_diagnostics"):
                    proxy_losses = model.proxy_losses_from_diagnostics(
                        diagnostics,
                        targets,
                        proxy_temperature=proxy_temperature,
                    )
                    if transport_proxy_weight > 0.0:
                        transport_term = proxy_losses["transport_proxy"]
                        loss = loss + transport_proxy_weight * transport_term
                        transport_proxy_terms.append(float(transport_term.detach().item()))
                    if symbol_proxy_weight > 0.0:
                        symbol_term = proxy_losses["symbol_proxy"]
                        loss = loss + symbol_proxy_weight * symbol_term
                        symbol_proxy_terms.append(float(symbol_term.detach().item()))
                    if canonical_loss_weight > 0.0:
                        canonical_term = proxy_losses.get(
                            "canonical_consistency_proxy",
                            torch.tensor(0.0, device=device),
                        )
                        loss = loss + canonical_loss_weight * canonical_term
                        canonical_terms.append(float(canonical_term.detach().item()))
                    if symbol_order_loss_weight > 0.0:
                        symbol_order_proxy = proxy_losses.get(
                            "symbol_order_proxy",
                            torch.tensor(0.0, device=device),
                        )
                        symbol_order_target_tensor = torch.as_tensor(
                            symbol_order_target,
                            device=device,
                            dtype=symbol_order_proxy.dtype,
                        )
                        symbol_order_term = (symbol_order_proxy - symbol_order_target_tensor).square()
                        loss = loss + symbol_order_loss_weight * symbol_order_term
                        symbol_order_terms.append(float(symbol_order_term.detach().item()))
                    if symbol_seminorm_loss_weight > 0.0:
                        symbol_seminorm_proxy = proxy_losses.get(
                            "symbol_seminorm_proxy",
                            torch.tensor(0.0, device=device),
                        )
                        symbol_seminorm_target_tensor = torch.as_tensor(
                            symbol_seminorm_target,
                            device=device,
                            dtype=symbol_seminorm_proxy.dtype,
                        )
                        symbol_seminorm_term = (symbol_seminorm_proxy - symbol_seminorm_target_tensor).square()
                        loss = loss + symbol_seminorm_loss_weight * symbol_seminorm_term
                        symbol_seminorm_terms.append(float(symbol_seminorm_term.detach().item()))
                    if packet_space_loss_weight > 0.0:
                        packet_space_term = proxy_losses.get(
                            "packet_space_proxy",
                            torch.tensor(0.0, device=device),
                        )
                        loss = loss + packet_space_loss_weight * packet_space_term
                        packet_space_terms.append(float(packet_space_term.detach().item()))
                    if branch_entropy_weight > 0.0:
                        entropy_proxy = proxy_losses.get(
                            "branch_entropy_proxy",
                            torch.tensor(0.0, device=device),
                        )
                        branch_count = 1
                        core = getattr(model, "core", None)
                        if core is not None:
                            branch_count = int(getattr(core, "num_canonical_branches", 1))
                        max_entropy = torch.log(torch.as_tensor(max(branch_count, 1), device=device, dtype=entropy_proxy.dtype)).clamp_min(1e-8)
                        entropy_gap = (max_entropy - entropy_proxy).clamp_min(0.0) / max_entropy
                        loss = loss + branch_entropy_weight * entropy_gap
                        branch_entropy_terms.append(float(entropy_gap.detach().item()))
                    if branch_diversity_weight > 0.0:
                        usage_proxy = proxy_losses.get(
                            "branch_usage_max_proxy",
                            torch.tensor(1.0, device=device),
                        )
                        branch_count = 1
                        core = getattr(model, "core", None)
                        if core is not None:
                            branch_count = int(getattr(core, "num_canonical_branches", 1))
                        target_usage = torch.as_tensor(1.0 / float(max(branch_count, 1)), device=device, dtype=usage_proxy.dtype)
                        diversity_term = (usage_proxy - target_usage).clamp_min(0.0).square()
                        loss = loss + branch_diversity_weight * diversity_term
                        branch_diversity_terms.append(float(diversity_term.detach().item()))
            else:
                prediction = _finite_tensor(model(inputs))
                loss = criterion(prediction, targets)
                if helmholtz_residual_loss_weight > 0.0:
                    helmholtz_term = _helmholtz_residual_proxy(
                        prediction,
                        inputs,
                        wavenumber=helmholtz_residual_wavenumber,
                        refractive_index=helmholtz_residual_refractive_index,
                    )
                    loss = loss + helmholtz_residual_loss_weight * helmholtz_term
                    helmholtz_residual_terms.append(float(helmholtz_term.detach().item()))
                if complex_pair_loss_weight > 0.0:
                    complex_pair_term = _complex_pair_relative_loss(prediction, targets)
                    loss = loss + complex_pair_loss_weight * complex_pair_term
                    complex_pair_terms.append(float(complex_pair_term.detach().item()))
                if complex_phase_loss_weight > 0.0:
                    complex_phase_term = _complex_pair_phase_loss(prediction, targets)
                    loss = loss + complex_phase_loss_weight * complex_phase_term
                    complex_phase_terms.append(float(complex_phase_term.detach().item()))
                if symbol_identity_loss_weight > 0.0:
                    symbol_identity_term = _symbol_identity_regularizer(model, device)
                    loss = loss + symbol_identity_loss_weight * symbol_identity_term
                    symbol_identity_terms.append(float(symbol_identity_term.detach().item()))
            loss = _finite_tensor(loss, cap=1.0e6)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        validation = evaluate_model(model, val_loader, device, criterion)
        if restore_best and validation.loss < best_val_loss:
            best_val_loss = validation.loss
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                "val_loss": validation.loss,
                "val_relative_l2": validation.relative_l2,
                "val_phase_error": validation.phase_error,
                "val_packet_consistency": validation.packet_consistency,
                "train_transport_proxy": sum(transport_proxy_terms) / max(len(transport_proxy_terms), 1),
                "train_symbol_proxy": sum(symbol_proxy_terms) / max(len(symbol_proxy_terms), 1),
                "train_core_field_loss": sum(core_field_terms) / max(len(core_field_terms), 1),
                "train_residual_energy": sum(residual_energy_terms) / max(len(residual_energy_terms), 1),
                "train_route_l1": sum(route_l1_terms) / max(len(route_l1_terms), 1),
                "train_canonical_consistency": sum(canonical_terms) / max(len(canonical_terms), 1),
                "train_symbol_order_loss": sum(symbol_order_terms) / max(len(symbol_order_terms), 1),
                "train_symbol_seminorm_loss": sum(symbol_seminorm_terms) / max(len(symbol_seminorm_terms), 1),
                "train_packet_space_loss": sum(packet_space_terms) / max(len(packet_space_terms), 1),
                "train_highfreq_core_loss": sum(highfreq_core_terms) / max(len(highfreq_core_terms), 1),
                "train_helmholtz_residual_loss": sum(helmholtz_residual_terms) / max(len(helmholtz_residual_terms), 1),
                "train_complex_pair_loss": sum(complex_pair_terms) / max(len(complex_pair_terms), 1),
                "train_complex_phase_loss": sum(complex_phase_terms) / max(len(complex_phase_terms), 1),
                "train_symbol_identity_loss": sum(symbol_identity_terms) / max(len(symbol_identity_terms), 1),
                "train_metadata_flow_loss": sum(metadata_flow_terms) / max(len(metadata_flow_terms), 1),
                "train_branch_entropy_loss": sum(branch_entropy_terms) / max(len(branch_entropy_terms), 1),
                "train_branch_diversity_loss": sum(branch_diversity_terms) / max(len(branch_diversity_terms), 1),
                "train_core_warmup_loss": sum(core_warmup_terms) / max(len(core_warmup_terms), 1),
            }
        )
        if progress_every_epochs > 0 and (
            epoch == 0 or epoch + 1 == epochs or (epoch + 1) % progress_every_epochs == 0
        ):
            label = f" {progress_label}" if progress_label else ""
            print(
                f"[epoch]{label} {epoch + 1}/{epochs} "
                f"train={history[-1]['train_loss']:.6g} "
                f"val={validation.loss:.6g} "
                f"rel_l2={validation.relative_l2:.6g} "
                f"phase={validation.phase_error:.6g}",
                flush=True,
            )
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    _set_refinement_requires_grad(model, enabled=True)
    runtime = perf_counter() - start
    return {"history": history, "runtime_seconds": runtime}
