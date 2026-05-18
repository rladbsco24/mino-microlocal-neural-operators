from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data import build_benchmark_loaders, get_scenario_spec  # noqa: E402
from mino.data.benchmark import ScenarioLoaders  # noqa: E402
from mino.metrics.wavefront import (  # noqa: E402
    amplitude_relative_error,
    boundary_trace_relative_error,
    complex_relative_l2,
    count_parameters,
    high_frequency_fraction,
    high_frequency_relative_error,
    packet_threshold_wavefront_localization_error,
    relative_l2,
    sobolev_h1_relative_error,
    sobolev_relative_error,
    symbol_order_scaling_error,
    wavefront_transport_proxy,
)
from mino.models.layers import _metadata_batch  # noqa: E402
from mino.models.mino import build_model  # noqa: E402
from mino.training.train import evaluate_model, fit_model  # noqa: E402


DEFAULT_SCENARIOS = (
    "wave_synth",
    "helmholtz_variable_control",
    "helmholtz_dirichlet_control",
    "helmholtz_highk_positive",
)
DEFAULT_ABLATIONS = (
    "full",
    "no_local_refine",
    "freeze_transport",
    "no_transport",
    "no_symbol",
    "no_lowfreq",
)
CORE_ABLATION_SCENARIOS = (
    "wave_synth",
    "helmholtz_variable_control",
    "helmholtz_highk_positive",
)
CORE_ABLATIONS = (
    "full",
    "no_local_refine",
    "identity_transport",
    "randomized_metadata",
    "no_transport",
    "no_symbol",
)
TRANSPORT_ID_SCENARIOS = (
    "wave_synth",
    "helmholtz_variable_control",
    "helmholtz_highk_positive",
)
TRANSPORT_ID_ABLATIONS = (
    "full",
    "identity_transport",
    "randomized_metadata",
    "no_transport",
    "no_symbol",
    "no_local_refine",
)
CALCULUS_ID_SCENARIOS = (
    "wave_synth",
    "helmholtz_variable_control",
    "poisson_robin_control",
    "diffusion_neumann_control",
    "navier_stokes_synth",
)
CALCULUS_ID_ABLATIONS = (
    "full",
    "no_transport",
    "no_symbol",
    "no_pdo_identity",
    "no_dissipative",
    "no_local_refine",
)
BRANCH_ID_SCENARIOS = (
    "wave_bicharacteristic_control",
    "wave_variable_speed_nocaustic",
    "wave_synth",
    "helmholtz_local_window_control",
    "helmholtz_variable_control",
)
BRANCH_ID_V2_SCENARIOS = (
    "wave_bicharacteristic_control",
    "wave_chirp_propagation",
    "wave_two_packet_no_interaction",
    "wave_variable_speed_nocaustic",
    "helmholtz_local_window_control",
    "wave_synth",
)
BRANCH_ID_V3_SCENARIOS = (
    "wave_bicharacteristic_control",
    "wave_chirp_propagation",
    "wave_two_packet_no_interaction",
    "wave_variable_speed_nocaustic",
    "wave_synth",
)
BRANCH_ID_ABLATIONS = (
    "full",
    "core_only",
    "no_local_refine",
    "residual_limited_refine",
    "no_transport",
    "identity_transport",
    "randomized_metadata",
    "no_symbol",
    "oracle_transport",
    "oracle_symbol",
    "fno_plus_same_refine",
    "uno_plus_same_refine",
    "wno_plus_same_refine",
)
BRANCH_ID_V2_ABLATIONS = BRANCH_ID_ABLATIONS
BRANCH_ID_V3_ABLATIONS = (
    "full",
    "core_only",
    "no_transport",
    "identity_transport",
    "randomized_metadata",
    "no_symbol",
    "oracle_transport",
    "residual_limited_refine",
    "uno_plus_same_refine",
    "wno_plus_same_refine",
)
BRANCH_ID_V3_CONTROL_ABLATIONS = (
    "full",
    "full_no_flow_supervision",
    "no_transport",
    "identity_transport",
    "no_transport_no_carrier",
    "randomized_metadata",
    "no_symbol",
)
HELMHOLTZ_BRANCHED_SCENARIOS = (
    "helmholtz_highk_control",
    "helmholtz_highk_positive",
    "helmholtz_variable_positive",
)
HELMHOLTZ_BRANCHED_ABLATIONS = (
    "full",
    "single_branch",
    "no_branch_routing",
    "no_transport",
    "no_transport_no_carrier",
    "no_symbol",
)
HELMHOLTZ_HIGHK_CAREFUL_SCENARIOS = (
    "helmholtz_highk_control",
    "helmholtz_highk_positive",
    "helmholtz_highk_ood_control",
    "helmholtz_highk_ood_positive",
)
HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS = (
    "full",
    "single_branch",
    "no_branch_routing",
    "no_transport",
    "no_transported_synthesis",
    "no_transported_input_carrier",
    "no_landing_decoder",
    "no_transport_no_carrier",
    "no_symbol",
    "source_only_symbol",
    "no_edge_symbol",
    "no_resolvent_phase",
)
HELMHOLTZ_HIGHK_FLAGSHIP_SCENARIOS = (
    "helmholtz_highk_control",
    "helmholtz_highk_positive",
    "helmholtz_highk_ood_control",
    "helmholtz_highk_ood_positive",
    "helmholtz_variable_positive",
)
HELMHOLTZ_HIGHK_FLAGSHIP_ABLATIONS = (
    "full",
    "single_branch",
    "no_branch_routing",
    "no_transport",
    "no_transported_synthesis",
    "no_transported_input_carrier",
    "no_landing_decoder",
    "no_transport_no_carrier",
    "no_symbol",
    "source_only_symbol",
    "no_edge_symbol",
    "no_resolvent_phase",
)
HELMHOLTZ_HIGHK_8GB_SCENARIOS = (
    "helmholtz_highk_positive",
    "helmholtz_highk_ood_positive",
    "helmholtz_variable_positive",
)
HELMHOLTZ_HIGHK_8GB_ABLATIONS = (
    "full",
    "single_branch",
    "no_branch_routing",
    "no_transport",
    "no_transported_input_carrier",
    "no_landing_decoder",
    "no_transport_no_carrier",
    "no_symbol",
    "source_only_symbol",
    "no_edge_symbol",
    "no_resolvent_phase",
)
CROSS_RESOLUTION_WAVE_SCENARIOS = (
    "wave_bicharacteristic_control",
    "wave_variable_speed_nocaustic",
    "wave_synth",
)
CROSS_RESOLUTION_PAIRS = (
    (64, 64),
    (64, 128),
    (96, 192),
    (128, 256),
)
CROSS_RESOLUTION_PACKET_POLICIES = ("fixed_packet_budget", "scaled_packet_budget")
CROSS_RESOLUTION_STENCIL_POLICIES = ("fixed_K", "scaled_K")
BASELINE_REFINE_ABLATIONS = {
    "fno_plus_same_refine": "FNO",
    "uno_plus_same_refine": "UNO",
    "wno_plus_same_refine": "WNO-style",
}
LOSS_CONFIGS = {
    "field_only": (0.0, 0.0),
    "proxy_0.02_0.02": (0.02, 0.02),
    "proxy_0.10_0.05": (0.10, 0.05),
    "proxy_0.20_0.05": (0.20, 0.05),
}
CAMPAIGN_EPOCHS = {
    "smoke": 2,
    "proxy_sweep": 16,
    "ablation": 16,
    "ablation_core": 16,
    "transport_id": 16,
    "calculus_id": 16,
    "branch_id": 16,
    "branch_id_v2": 16,
    "branch_id_v3": 24,
    "branch_id_v3_controls": 24,
    "helmholtz_branched_highk": 24,
    "helmholtz_highk_careful": 36,
    "helmholtz_highk_flagship": 72,
    "helmholtz_highk_8gb": 24,
    "cross_resolution_wave": 0,
    "stage3": 24,
}
CAMPAIGN_LOSS_CONFIGS = {
    "smoke": ("proxy_0.02_0.02",),
    "proxy_sweep": ("field_only", "proxy_0.02_0.02"),
    "ablation": ("field_only", "proxy_0.02_0.02"),
    "ablation_core": ("proxy_0.02_0.02",),
    "transport_id": ("proxy_0.10_0.05",),
    "calculus_id": ("proxy_0.10_0.05",),
    "branch_id": ("proxy_0.10_0.05",),
    "branch_id_v2": ("proxy_0.10_0.05",),
    "branch_id_v3": ("proxy_0.20_0.05",),
    "branch_id_v3_controls": ("proxy_0.20_0.05",),
    "helmholtz_branched_highk": ("proxy_0.20_0.05",),
    "helmholtz_highk_careful": ("proxy_0.20_0.05",),
    "helmholtz_highk_flagship": ("proxy_0.20_0.05",),
    "helmholtz_highk_8gb": ("proxy_0.20_0.05",),
    "cross_resolution_wave": ("proxy_0.10_0.05",),
    "stage3": ("proxy_0.02_0.02",),
}
METRIC_FIELDS = (
    "test_relative_l2",
    "test_phase_error",
    "test_packet_consistency",
    "test_complex_relative_l2_proxy",
    "test_amplitude_error_proxy",
    "test_boundary_trace_error_proxy",
    "test_core_relative_l2",
    "test_refine_relative_energy",
    "test_route_mean",
    "test_raw_refine_correction_norm",
    "test_refine_correction_norm",
    "test_field_correction_norm",
    "test_refine_lowpass_removed_norm",
    "test_refine_high_frequency_fraction",
    "test_pdo_identity_norm",
    "test_dissipative_symbol_norm",
    "test_dissipative_multiplier_mean",
    "test_transport_budget",
    "test_symbol_budget",
    "test_residual_energy",
    "test_refinement_energy",
    "test_canonical_flow_error",
    "test_metadata_shift_norm",
    "test_symplectic_defect_proxy",
    "test_wavefront_confidence_proxy",
    "test_symbol_order_proxy",
    "test_symbol_seminorm_proxy",
    "test_local_tube_coordinate_norm",
    "test_edge_symbol_deviation_proxy",
    "test_helmholtz_shell_distance_proxy",
    "test_helmholtz_resolvent_envelope_proxy",
    "test_helmholtz_outgoing_flux_proxy",
    "test_helmholtz_resolvent_real_proxy",
    "test_helmholtz_resolvent_imag_proxy",
    "test_helmholtz_outgoing_gate_proxy",
    "test_helmholtz_shell_center_proxy",
    "test_helmholtz_complex_latent_imag_energy_proxy",
    "test_pdo_identity_order_proxy",
    "test_wf_transport_error_proxy",
    "test_packet_wavefront_localization_error",
    "test_sobolev_h1_error_proxy",
    "test_sobolev_h2_error_proxy",
    "test_high_frequency_relative_error_proxy",
    "test_egorov_intertwining_proxy",
    "test_egorov_targeted_probe_proxy",
    "test_egorov_jacobian_self_proxy",
    "test_egorov_jacobian_target_proxy",
    "test_symbol_order_scaling_error_proxy",
    "test_symbol_error_proxy",
    "test_branch_entropy",
    "test_branch_diversity",
    "test_branch_usage_max",
    "test_branch_spread",
    "test_tokenizer_reconstruction_error",
    "test_tokenizer_active_fraction",
    "test_tokenizer_covering_radius",
    "test_tokenizer_phase_window_diameter",
    "test_transported_synthesis_shift_norm",
    "test_transported_input_shift_norm",
    "test_transported_input_norm",
    "test_transported_landing_norm",
    "test_transported_landing_gate",
    "runtime_seconds",
    "final_train_transport_proxy",
    "final_train_symbol_proxy",
    "final_train_core_field_loss",
    "final_train_residual_energy",
    "final_train_route_l1",
    "final_train_canonical_consistency",
    "final_train_symbol_order_loss",
    "final_train_symbol_seminorm_loss",
    "final_train_packet_space_loss",
    "final_train_highfreq_core_loss",
    "final_train_helmholtz_residual_loss",
    "final_train_symbol_identity_loss",
    "final_train_core_warmup_loss",
)


class ZeroPropagation(nn.Module):
    def forward(self, features: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(features), metadata


class ZeroSymbol(nn.Module):
    def forward(self, features: Tensor, metadata: Tensor, local_tube_coordinate: Tensor | None = None) -> Tensor:
        return torch.zeros_like(features)


def _iter_core_blocks(model: nn.Module):
    core = getattr(model, "core", None)
    if core is None:
        return ()
    return tuple(getattr(core, "blocks", ()))


def _iter_symbol_modules(model: nn.Module):
    for block in _iter_core_blocks(model):
        symbol = getattr(block, "symbol", None)
        if symbol is not None:
            yield symbol
        for symbol in getattr(block, "branch_symbols", ()):
            if symbol is not None:
                yield symbol
        pdo = getattr(block, "pdo_symbol", None)
        if pdo is not None:
            yield pdo


def _disable_edge_symbols(model: nn.Module) -> None:
    for block in _iter_core_blocks(model):
        propagation = getattr(block, "propagation", None)
        if propagation is not None and hasattr(propagation, "edge_symbol"):
            propagation.edge_symbol = None
            propagation.edge_symbol_parameterization = "none"
        for propagation in getattr(block, "branch_propagations", ()):
            if hasattr(propagation, "edge_symbol"):
                propagation.edge_symbol = None
                propagation.edge_symbol_parameterization = "none"


def _set_local_symbol_kernel_enabled(model: nn.Module, enabled: bool) -> None:
    for symbol in _iter_symbol_modules(model):
        if hasattr(symbol, "local_kernel_enabled"):
            symbol.local_kernel_enabled = bool(enabled)


def _set_helmholtz_symbol_mode(model: nn.Module, mode: str) -> None:
    for symbol in _iter_symbol_modules(model):
        if hasattr(symbol, "helmholtz_symbol_mode"):
            symbol.helmholtz_symbol_mode = mode


class MetadataScramblingPropagation(nn.Module):
    """Keep the propagation capacity but corrupt the packet geometry."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, features: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        token_dim = 1 if metadata.dim() == 3 else 0
        if metadata.shape[token_dim] <= 1:
            return self.inner(features, metadata)
        permutation = torch.randperm(metadata.shape[token_dim], device=metadata.device)
        if metadata.dim() == 3:
            return self.inner(features, metadata[:, permutation, :])
        return self.inner(features, metadata[permutation])


class OracleConstantFlowPropagation(nn.Module):
    """Inject the known packet-center drift for controlled-flow diagnostics."""

    SUPPORTED_DELTAS = {
        "wave_bicharacteristic_control": (-0.07 * 0.15, 0.10 * 0.15),
        "wave_chirp_propagation": (-0.06 * 0.14, 0.11 * 0.14),
        "wave_two_packet_no_interaction": (-0.045 * 0.16, 0.09 * 0.16),
    }
    SUPPORTED = set(SUPPORTED_DELTAS)

    def __init__(self, inner: nn.Module, scenario: str, *, steps: int = 1) -> None:
        super().__init__()
        if scenario not in self.SUPPORTED:
            raise ValueError(f"oracle_transport is only implemented for {sorted(self.SUPPORTED)}, got {scenario!r}.")
        self.inner = inner
        steps = max(int(steps), 1)
        delta_y, delta_x = self.SUPPORTED_DELTAS[scenario]
        self.delta_y = delta_y / steps
        self.delta_x = delta_x / steps

    def forward(self, features: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        batch, tokens, width = features.shape
        oracle_metadata = _metadata_batch(metadata, features).clone()
        oracle_metadata[..., 0] = (oracle_metadata[..., 0] + self.delta_y).clamp(0.0, 1.0)
        oracle_metadata[..., 1] = (oracle_metadata[..., 1] + self.delta_x).clamp(0.0, 1.0)
        metadata_batch = oracle_metadata
        scaled = metadata_batch[..., :4] * self.inner.length_scale.view(1, 1, 4)
        distances = torch.cdist(scaled, scaled, p=2.0)
        logits = -distances / self.inner.temperature
        use_topk = self.inner.stencil_size is not None and 0 < self.inner.stencil_size < tokens
        top_values: Tensor | None = None
        top_indices: Tensor | None = None
        if use_topk:
            top_values, top_indices = torch.topk(logits, k=self.inner.stencil_size, dim=-1)
        message_source = self.inner.message_mlp(torch.cat([features, metadata_batch], dim=-1))
        if use_topk and getattr(self.inner, "sparse_topk", False):
            assert top_values is not None and top_indices is not None
            weights = torch.softmax(top_values, dim=-1)
            gathered = torch.gather(
                message_source.unsqueeze(1).expand(-1, tokens, -1, -1),
                2,
                top_indices.unsqueeze(-1).expand(-1, -1, -1, width),
            )
            local_message = (weights.unsqueeze(-1) * gathered).sum(dim=2)
        else:
            if use_topk:
                assert top_values is not None and top_indices is not None
                masked_logits = torch.full_like(logits, float("-inf"))
                masked_logits.scatter_(-1, top_indices, top_values)
                logits = masked_logits
            weights = torch.softmax(logits, dim=-1)
            local_message = torch.einsum("bij,bjf->bif", weights, message_source)
        if getattr(self.inner, "confidence_head", None) is not None:
            confidence = torch.sigmoid(self.inner.confidence_head(torch.cat([features, metadata_batch], dim=-1)))
            local_message = confidence * local_message
        gated = torch.sigmoid(self.inner.output_gate(features))
        return gated * local_message, oracle_metadata


class OracleIdentitySymbol(nn.Module):
    """Use the principal unit-amplitude symbol on controlled short-time waves."""

    SUPPORTED = {
        "wave_bicharacteristic_control",
        "wave_chirp_propagation",
        "wave_two_packet_no_interaction",
        "wave_variable_speed_nocaustic",
    }

    def __init__(self, scenario: str) -> None:
        super().__init__()
        if scenario not in self.SUPPORTED:
            raise ValueError(f"oracle_symbol is only implemented for {sorted(self.SUPPORTED)}, got {scenario!r}.")

    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        return torch.zeros_like(features)


class ZeroField(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]))


class SameLocalRefineWrapper(nn.Module):
    """Attach the MiNO-Plus local refinement mechanism to a baseline.

    This is a fairness diagnostic, not a new headline baseline. If this wrapper
    explains most of MiNO-Plus's gain, the microlocal claim should be lowered.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        out_channels: int,
        local_refine_channels: int,
        local_refine_scale: float,
        refine_lowpass_cutoff: float,
        route_bias_init: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.local_refine_scale = float(local_refine_scale)
        self.refine_lowpass_cutoff = float(refine_lowpass_cutoff)
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

    def _lowpass_refinement(self, correction: Tensor) -> Tensor:
        if self.refine_lowpass_cutoff <= 0.0:
            return correction
        height, width = correction.shape[-2:]
        fy = torch.fft.fftfreq(height, device=correction.device, dtype=correction.dtype).view(height, 1)
        fx = torch.fft.rfftfreq(width, device=correction.device, dtype=correction.dtype).view(1, width // 2 + 1)
        radius = torch.sqrt(fy.square() + fx.square())
        mask = (radius <= self.refine_lowpass_cutoff).to(correction.dtype).view(1, 1, height, width // 2 + 1)
        spectrum = torch.fft.rfft2(correction, norm="ortho")
        return torch.fft.irfft2(spectrum * mask, s=(height, width), norm="ortho")

    def forward(self, x: Tensor) -> Tensor:
        base_prediction = self.base(x)
        refine_in = torch.cat([base_prediction, base_prediction], dim=1)
        high = self.local_refine(refine_in)
        route = torch.sigmoid(self.route(refine_in))
        raw_refine = self.local_refine_scale * route * high
        refine = self._lowpass_refinement(raw_refine)
        return base_prediction + refine

    def forward_with_diagnostics(self, x: Tensor) -> dict[str, object]:
        base_prediction = self.base(x)
        refine_in = torch.cat([base_prediction, base_prediction], dim=1)
        high = self.local_refine(refine_in)
        route = torch.sigmoid(self.route(refine_in))
        raw_refine_correction = self.local_refine_scale * route * high
        refine_correction = self._lowpass_refinement(raw_refine_correction)
        removed_refine = raw_refine_correction - refine_correction
        return {
            "prediction": base_prediction + refine_correction,
            "core_prediction": base_prediction,
            "refine_correction": refine_correction,
            "raw_refine_correction": raw_refine_correction,
            "removed_refine_correction": removed_refine,
            "route_mean": route.mean(),
            "route_abs_mean": route.abs().mean(),
            "raw_refine_correction_norm": torch.linalg.vector_norm(
                raw_refine_correction.reshape(raw_refine_correction.shape[0], -1),
                dim=-1,
            ).mean(),
            "refine_correction_norm": torch.linalg.vector_norm(
                refine_correction.reshape(refine_correction.shape[0], -1),
                dim=-1,
            ).mean(),
            "refine_lowpass_removed_norm": torch.linalg.vector_norm(
                removed_refine.reshape(removed_refine.shape[0], -1),
                dim=-1,
            ).mean(),
            "block_diagnostics": [],
        }


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_seeds(raw: str) -> list[int]:
    return [int(item) for item in parse_csv(raw)]


def parse_int_tuple(raw: str) -> tuple[int, ...] | None:
    values = parse_csv(raw)
    if not values:
        return None
    return tuple(int(item) for item in values)


def safe_slug(raw: str) -> str:
    return (
        raw.replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("+", "plus")
        .replace(".", "p")
        .replace(",", "-")
    )


def plus_model_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "width": args.plus_width,
        "depth": args.plus_depth,
        "patch_size": args.plus_patch_size,
        "stride": args.plus_stride,
        "max_modes": args.plus_max_modes,
        "window_type": args.plus_window_type,
        "mode_strategy": args.plus_mode_strategy,
        "low_frequency_scale": args.plus_low_frequency_scale,
        "transport_scale": args.plus_transport_scale,
        "transport_stencil": args.plus_transport_stencil,
        "local_refine_channels": args.plus_local_refine_channels,
        "local_refine_scale": args.plus_local_refine_scale,
        "route_bias_init": args.plus_route_bias_init,
        "pdo_symbol_scale": args.plus_pdo_symbol_scale,
        "transport_highpass_cutoff": args.transport_highpass_cutoff,
        "pdo_symbol_order": args.plus_pdo_symbol_order,
        "dissipative_symbol_scale": args.plus_dissipative_symbol_scale,
        "dissipative_time_step": args.plus_dissipative_time_step,
        "symbol_order": args.plus_symbol_order,
        "symbol_parameterization": args.symbol_parameterization,
        "helmholtz_shell_radius": args.helmholtz_shell_radius,
        "helmholtz_refractive_index": args.helmholtz_refractive_index,
        "helmholtz_absorption": args.helmholtz_absorption,
        "helmholtz_resolvent_cap": args.helmholtz_resolvent_cap,
        "wavefront_confidence_scale": args.plus_wavefront_confidence_scale,
        "refine_lowpass_cutoff": args.plus_refine_lowpass_cutoff,
        "frame_patch_sizes": parse_int_tuple(args.plus_frame_patch_sizes),
        "frame_strides": parse_int_tuple(args.plus_frame_strides),
        "frame_max_modes": parse_int_tuple(args.plus_frame_max_modes),
        "transport_parameterization": args.transport_parameterization,
        "sparse_topk": args.sparse_topk,
        "frame_type": args.frame_type,
        "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
        "transported_synthesis_scale": args.transported_synthesis_scale,
        "transported_input_scale": args.transported_input_scale,
        "transported_synthesis_mode": args.transported_synthesis_mode,
        "transported_decoder_channels": args.transported_decoder_channels,
        "transported_decoder_scale": args.transported_decoder_scale,
        "transported_decoder_transport_gate": args.transported_decoder_transport_gate,
        "token_refine_scale": args.token_refine_scale,
        "num_canonical_branches": args.num_canonical_branches,
        "branch_routing": args.branch_routing,
        "branch_prior_strength": args.branch_prior_strength,
        "branch_entropy_weight": args.branch_entropy_weight,
        "branch_diversity_weight": args.branch_diversity_weight,
        "branch_synthesis": args.branch_synthesis,
        "edge_symbol_parameterization": args.edge_symbol_parameterization,
        "edge_symbol_strength": args.edge_symbol_strength,
        "field_corrector": args.field_corrector,
        "field_corrector_scale": args.field_corrector_scale,
        "field_corrector_width": args.field_corrector_width,
        "field_corrector_input_mode": args.field_corrector_input_mode,
    }


def plus_model_variant(args: argparse.Namespace) -> str:
    return (
        "MiNO-Plus"
        f"_win{args.plus_window_type}"
        f"_mod{args.plus_mode_strategy}"
        f"_K{args.plus_transport_stencil}"
        f"_modes{args.plus_max_modes}"
        f"_patch{args.plus_patch_size}"
        f"_ref{args.plus_local_refine_scale:g}"
        f"_rlp{args.plus_refine_lowpass_cutoff:g}"
        f"_hfc{args.transport_highpass_cutoff:g}"
        f"_rb{args.plus_route_bias_init:g}"
        f"_pdo{args.plus_pdo_symbol_scale:g}"
        f"_diss{args.plus_dissipative_symbol_scale:g}"
        f"_symord{args.plus_symbol_order:g}"
        f"_sympar{args.symbol_parameterization}"
        f"_hrad{args.helmholtz_shell_radius:g}"
        f"_href{args.helmholtz_refractive_index:g}"
        f"_heta{args.helmholtz_absorption:g}"
        f"_hcap{args.helmholtz_resolvent_cap:g}"
        f"_hres{args.helmholtz_residual_loss_weight:g}"
        f"_hk{args.helmholtz_residual_wavenumber:g}"
        f"_cplx{args.complex_pair_loss_weight:g}"
        f"_cphase{args.complex_phase_loss_weight:g}"
        f"_symid{args.symbol_identity_loss_weight:g}"
        f"_wfconf{args.plus_wavefront_confidence_scale:g}"
        f"_tr{args.transport_parameterization}"
        f"_frame{args.frame_type}"
        f"_sparse{int(args.sparse_topk)}"
        f"_sklp{args.skip_lowpass_cutoff:g}"
        f"_tsyn{args.transported_synthesis_scale:g}"
        f"_tin{args.transported_input_scale:g}"
        f"_tsm{args.transported_synthesis_mode}"
        f"_tdec{args.transported_decoder_scale:g}x{args.transported_decoder_channels}"
        f"_tdgate{int(args.transported_decoder_transport_gate)}"
        f"_tref{args.token_refine_scale:g}"
        f"_br{args.num_canonical_branches}"
        f"_brroute{args.branch_routing}"
        f"_brprior{args.branch_prior_strength:g}"
        f"_brent{args.branch_entropy_weight:g}"
        f"_brdiv{args.branch_diversity_weight:g}"
        f"_edgesym{args.edge_symbol_parameterization}"
        f"_edgestr{args.edge_symbol_strength:g}"
        f"_fcorr{args.field_corrector}"
        f"_fcscale{args.field_corrector_scale:g}"
        f"_fcw{args.field_corrector_width}"
        f"_fcin{args.field_corrector_input_mode}"
    )


def effective_metadata_flow_loss_weight(args: argparse.Namespace, ablation: str) -> float:
    """Return the training-time metadata-flow supervision weight for this row."""

    if ablation == "full_no_flow_supervision":
        return 0.0
    return float(args.metadata_flow_loss_weight)


def maybe_limit_loaders(loaders: ScenarioLoaders, args: argparse.Namespace) -> ScenarioLoaders:
    train_limit = args.max_train_samples
    val_limit = args.max_val_samples
    test_limit = args.max_test_samples
    if train_limit <= 0 and val_limit <= 0 and test_limit <= 0:
        return loaders

    def _limit(loader: DataLoader, limit: int, *, shuffle: bool) -> DataLoader:
        if limit <= 0 or limit >= len(loader.dataset):
            return loader
        dataset = Subset(loader.dataset, range(limit))
        return DataLoader(dataset, batch_size=loader.batch_size, shuffle=shuffle)

    return ScenarioLoaders(
        train_loader=_limit(loaders.train_loader, train_limit, shuffle=True),
        val_loader=_limit(loaders.val_loader, val_limit, shuffle=False),
        test_loader=_limit(loaders.test_loader, test_limit, shuffle=False),
        in_channels=loaders.in_channels,
        out_channels=loaders.out_channels,
        spatial_shape=loaders.spatial_shape,
        spec=loaders.spec,
    )


def apply_ablation(model: nn.Module, ablation: str, out_channels: int, *, scenario: str) -> None:
    if ablation in {"full", "full_no_flow_supervision"}:
        return
    if not hasattr(model, "core"):
        raise ValueError(f"Ablation {ablation!r} requires a MiNO-Plus-style model with a core.")
    if ablation == "oracle_transport":
        block_count = max(len(model.core.blocks), 1)
        for block in model.core.blocks:
            block.propagation = OracleConstantFlowPropagation(block.propagation, scenario, steps=block_count)
            if hasattr(block, "branch_propagations"):
                for index, propagation in enumerate(block.branch_propagations):
                    block.branch_propagations[index] = OracleConstantFlowPropagation(propagation, scenario, steps=block_count)
        return
    if ablation == "oracle_symbol":
        for block in model.core.blocks:
            block.symbol = OracleIdentitySymbol(scenario)
        return
    if ablation == "no_transport":
        for block in model.core.blocks:
            block.propagation = ZeroPropagation()
            if hasattr(block, "branch_propagations"):
                for index in range(len(block.branch_propagations)):
                    block.branch_propagations[index] = ZeroPropagation()
        return
    if ablation == "no_transport_no_carrier":
        for block in model.core.blocks:
            block.propagation = ZeroPropagation()
            if hasattr(block, "branch_propagations"):
                for index in range(len(block.branch_propagations)):
                    block.branch_propagations[index] = ZeroPropagation()
        model.core.transported_synthesis_scale = 0.0
        model.core.transported_input_scale = 0.0
        model.core.transported_decoder_scale = 0.0
        return
    if ablation == "no_transported_synthesis":
        model.core.transported_synthesis_scale = 0.0
        return
    if ablation == "no_transported_input_carrier":
        model.core.transported_input_scale = 0.0
        return
    if ablation == "no_landing_decoder":
        model.core.transported_decoder_scale = 0.0
        return
    if ablation == "single_branch":
        for block in model.core.blocks:
            if hasattr(block, "branch_routing"):
                block.branch_routing = "single"
        if hasattr(model.core, "num_canonical_branches"):
            model.core.num_canonical_branches = 1
        return
    if ablation == "no_branch_routing":
        for block in model.core.blocks:
            if hasattr(block, "branch_routing"):
                block.branch_routing = "uniform"
        return
    if ablation in {"freeze_transport", "identity_transport"}:
        for block in model.core.blocks:
            if hasattr(block.propagation, "transport_scale"):
                block.propagation.transport_scale = 0.0
            if hasattr(block, "branch_propagations"):
                for propagation in block.branch_propagations:
                    if hasattr(propagation, "transport_scale"):
                        propagation.transport_scale = 0.0
        return
    if ablation == "randomized_metadata":
        for block in model.core.blocks:
            block.propagation = MetadataScramblingPropagation(block.propagation)
            if hasattr(block, "branch_propagations"):
                for index, propagation in enumerate(block.branch_propagations):
                    block.branch_propagations[index] = MetadataScramblingPropagation(propagation)
        return
    if ablation == "no_symbol":
        for block in model.core.blocks:
            block.symbol = ZeroSymbol()
            if hasattr(block, "branch_symbols"):
                for index in range(len(block.branch_symbols)):
                    block.branch_symbols[index] = ZeroSymbol()
        return
    if ablation == "source_only_symbol":
        # Keep the source packet multiplier but remove local tube-coordinate
        # corrections and retained-edge packet kernels.  This isolates whether
        # the executable s(z,nu;h) refinement helps beyond a source-only s(z).
        _set_local_symbol_kernel_enabled(model, False)
        _disable_edge_symbols(model)
        return
    if ablation == "no_edge_symbol":
        _disable_edge_symbols(model)
        return
    if ablation == "no_resolvent_phase":
        # Remove the signed imaginary part of the Helmholtz resolvent action
        # while preserving the real/envelope/outgoing multiplier path.
        _set_helmholtz_symbol_mode(model, "amplitude_only")
        return
    if ablation == "no_pdo_identity":
        for block in model.core.blocks:
            if hasattr(block, "pdo_symbol_scale"):
                block.pdo_symbol_scale = 0.0
        if hasattr(model, "pdo_symbol_scale"):
            model.pdo_symbol_scale = 0.0
        return
    if ablation == "no_dissipative":
        for block in model.core.blocks:
            if hasattr(block, "dissipative_symbol_scale"):
                block.dissipative_symbol_scale = 0.0
        if hasattr(model, "dissipative_symbol_scale"):
            model.dissipative_symbol_scale = 0.0
        return
    if ablation == "no_lowfreq":
        model.low_frequency_scale = 0.0
        return
    if ablation in {"no_local_refine", "core_only"}:
        model.local_refine = ZeroField(out_channels)
        return
    if ablation == "residual_limited_refine":
        model.local_refine_scale = min(float(getattr(model, "local_refine_scale", 1.0)), 0.15)
        if hasattr(model, "route") and getattr(model.route, "bias", None) is not None:
            nn.init.constant_(model.route.bias, -4.0)
        return
    raise ValueError(f"Unknown ablation: {ablation}")


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def ensure_finite(row: dict[str, object]) -> None:
    for field in ("test_loss", *METRIC_FIELDS):
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"Non-finite metric {field}={value} for {row.get('run_id')}")


def _spatial_probe_mask(
    height: int,
    width: int,
    *,
    center_y: float,
    center_x: float,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, width)
    radius2 = (ys - center_y).square() + (xs - center_x).square()
    return torch.exp(-0.5 * radius2 / max(sigma * sigma, 1e-8))


def finite_egorov_intertwining_proxy(
    model: nn.Module,
    inputs: Tensor,
    prediction: Tensor,
    *,
    metadata_flow_delta: tuple[float, float] | None,
    sigma: float = 0.18,
    eps: float = 1e-8,
) -> Tensor | None:
    """Finite controlled-flow proxy for the planned Egorov diagnostic.

    The proxy compares A_out M(u) with M(A_in u), where A_out is the spatial
    probe A_in shifted by the known packet-center drift.  This is deliberately
    weaker than a pseudodifferential conjugation norm; it is an executable
    intertwining sanity check for controlled-flow rows.
    """
    if metadata_flow_delta is None:
        return None
    _, _, height, width = inputs.shape
    delta_y, delta_x = metadata_flow_delta
    centers = ((0.35, 0.35), (0.50, 0.50), (0.65, 0.65))
    residuals: list[Tensor] = []
    for center_y, center_x in centers:
        in_mask = _spatial_probe_mask(
            height,
            width,
            center_y=center_y,
            center_x=center_x,
            sigma=sigma,
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, 1, height, width)
        out_mask = _spatial_probe_mask(
            height,
            width,
            center_y=min(max(center_y + delta_y, 0.0), 1.0),
            center_x=min(max(center_x + delta_x, 0.0), 1.0),
            sigma=sigma,
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, 1, height, width)
        probed_output = model(inputs * in_mask)
        if isinstance(probed_output, tuple):
            probed_output = probed_output[0]
        if not isinstance(probed_output, Tensor):
            continue
        transported_probe = prediction * out_mask
        numerator = torch.linalg.vector_norm((probed_output - transported_probe).reshape(inputs.shape[0], -1), dim=-1)
        denominator = (
            torch.linalg.vector_norm(probed_output.reshape(inputs.shape[0], -1), dim=-1)
            + torch.linalg.vector_norm(transported_probe.reshape(inputs.shape[0], -1), dim=-1)
        ).clamp_min(eps)
        residuals.append(numerator / denominator)
    if not residuals:
        return None
    return torch.stack(residuals, dim=0).mean(dim=0)


def finite_egorov_targeted_probe_proxy(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    metadata_flow_delta: tuple[float, float] | None,
    sigma: float = 0.18,
    eps: float = 1e-8,
) -> Tensor | None:
    """Target-calibrated finite probe for controlled-flow rows.

    This compares M(A_in u) against A_out v, where v is the supervised target
    and A_out is shifted by the known flow.  It is still only a finite packet
    probe, but unlike the self-intertwining residual it can detect a
    self-consistent model that transports to the wrong place.
    """
    if metadata_flow_delta is None:
        return None
    _, _, height, width = inputs.shape
    delta_y, delta_x = metadata_flow_delta
    centers = ((0.35, 0.35), (0.50, 0.50), (0.65, 0.65))
    residuals: list[Tensor] = []
    for center_y, center_x in centers:
        in_mask = _spatial_probe_mask(
            height,
            width,
            center_y=center_y,
            center_x=center_x,
            sigma=sigma,
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, 1, height, width)
        out_mask = _spatial_probe_mask(
            height,
            width,
            center_y=min(max(center_y + delta_y, 0.0), 1.0),
            center_x=min(max(center_x + delta_x, 0.0), 1.0),
            sigma=sigma,
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, 1, height, width)
        probed_output = model(inputs * in_mask)
        if isinstance(probed_output, tuple):
            probed_output = probed_output[0]
        if not isinstance(probed_output, Tensor):
            continue
        target_probe = targets * out_mask
        numerator = torch.linalg.vector_norm((probed_output - target_probe).reshape(inputs.shape[0], -1), dim=-1)
        denominator = torch.linalg.vector_norm(target_probe.reshape(inputs.shape[0], -1), dim=-1).clamp_min(eps)
        residuals.append(numerator / denominator)
    if not residuals:
        return None
    return torch.stack(residuals, dim=0).mean(dim=0)


def finite_difference_egorov_jacobian_proxies(
    model: nn.Module,
    inputs: Tensor,
    prediction: Tensor,
    targets: Tensor,
    *,
    metadata_flow_delta: tuple[float, float] | None,
    sigma: float = 0.18,
    step: float = 1e-2,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor] | None:
    """Finite-difference Jacobian probes for controlled-flow rows.

    The self proxy compares D M[u](A_in u) with A_out D M[u](u), while the
    target proxy compares D M[u](A_in u) with A_out v.  The latter is
    supervised and target-calibrated; it is the one that can detect an
    internally coherent but wrong learned flow.
    """
    if metadata_flow_delta is None:
        return None
    _, _, height, width = inputs.shape
    delta_y, delta_x = metadata_flow_delta
    centers = ((0.35, 0.35), (0.50, 0.50), (0.65, 0.65))
    base_prediction = prediction
    global_output = model(inputs * (1.0 + step))
    if isinstance(global_output, tuple):
        global_output = global_output[0]
    if not isinstance(global_output, Tensor):
        return None
    global_response = (global_output - base_prediction) / step
    self_residuals: list[Tensor] = []
    target_residuals: list[Tensor] = []
    for center_y, center_x in centers:
        in_mask = _spatial_probe_mask(
            height,
            width,
            center_y=center_y,
            center_x=center_x,
            sigma=sigma,
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, 1, height, width)
        out_mask = _spatial_probe_mask(
            height,
            width,
            center_y=min(max(center_y + delta_y, 0.0), 1.0),
            center_x=min(max(center_x + delta_x, 0.0), 1.0),
            sigma=sigma,
            device=inputs.device,
            dtype=inputs.dtype,
        ).view(1, 1, height, width)
        localized_output = model(inputs + step * inputs * in_mask)
        if isinstance(localized_output, tuple):
            localized_output = localized_output[0]
        if not isinstance(localized_output, Tensor):
            continue
        localized_response = (localized_output - base_prediction) / step
        self_target = global_response * out_mask
        self_num = torch.linalg.vector_norm((localized_response - self_target).reshape(inputs.shape[0], -1), dim=-1)
        self_den = (
            torch.linalg.vector_norm(localized_response.reshape(inputs.shape[0], -1), dim=-1)
            + torch.linalg.vector_norm(self_target.reshape(inputs.shape[0], -1), dim=-1)
        ).clamp_min(eps)
        self_residuals.append(self_num / self_den)

        supervised_target = targets * out_mask
        target_num = torch.linalg.vector_norm((localized_response - supervised_target).reshape(inputs.shape[0], -1), dim=-1)
        target_den = torch.linalg.vector_norm(supervised_target.reshape(inputs.shape[0], -1), dim=-1).clamp_min(eps)
        target_residuals.append(target_num / target_den)
    if not self_residuals or not target_residuals:
        return None
    return torch.stack(self_residuals, dim=0).mean(dim=0), torch.stack(target_residuals, dim=0).mean(dim=0)


def evaluate_branch_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    metadata_flow_delta: tuple[float, float] | None = None,
) -> dict[str, float]:
    if not hasattr(model, "forward_with_diagnostics"):
        return {
            "test_core_relative_l2": math.nan,
            "test_refine_relative_energy": math.nan,
            "test_route_mean": math.nan,
            "test_raw_refine_correction_norm": math.nan,
            "test_refine_correction_norm": math.nan,
            "test_refine_lowpass_removed_norm": math.nan,
            "test_refine_high_frequency_fraction": math.nan,
            "test_pdo_identity_norm": math.nan,
            "test_dissipative_symbol_norm": math.nan,
            "test_dissipative_multiplier_mean": math.nan,
            "test_transport_budget": math.nan,
            "test_symbol_budget": math.nan,
            "test_residual_energy": math.nan,
            "test_refinement_energy": math.nan,
            "test_canonical_flow_error": math.nan,
            "test_metadata_shift_norm": math.nan,
            "test_symplectic_defect_proxy": math.nan,
            "test_wavefront_confidence_proxy": math.nan,
            "test_symbol_order_proxy": math.nan,
            "test_symbol_seminorm_proxy": math.nan,
            "test_local_tube_coordinate_norm": math.nan,
            "test_edge_symbol_deviation_proxy": math.nan,
            "test_pdo_identity_order_proxy": math.nan,
            "test_wf_transport_error_proxy": math.nan,
            "test_packet_wavefront_localization_error": math.nan,
            "test_complex_relative_l2_proxy": math.nan,
            "test_amplitude_error_proxy": math.nan,
            "test_boundary_trace_error_proxy": math.nan,
            "test_sobolev_h1_error_proxy": math.nan,
            "test_sobolev_h2_error_proxy": math.nan,
            "test_high_frequency_relative_error_proxy": math.nan,
            "test_egorov_intertwining_proxy": math.nan,
            "test_egorov_targeted_probe_proxy": math.nan,
            "test_egorov_jacobian_self_proxy": math.nan,
            "test_egorov_jacobian_target_proxy": math.nan,
            "test_symbol_order_scaling_error_proxy": math.nan,
            "test_symbol_error_proxy": math.nan,
            "test_branch_entropy": math.nan,
            "test_branch_diversity": math.nan,
            "test_branch_usage_max": math.nan,
            "test_branch_spread": math.nan,
            "test_tokenizer_reconstruction_error": math.nan,
            "test_tokenizer_active_fraction": math.nan,
            "test_tokenizer_covering_radius": math.nan,
            "test_tokenizer_phase_window_diameter": math.nan,
            "test_transported_synthesis_shift_norm": math.nan,
            "test_transported_input_shift_norm": math.nan,
            "test_transported_input_norm": math.nan,
            "test_transported_landing_norm": math.nan,
            "test_transported_landing_gate": math.nan,
        }
    core_rels: list[float] = []
    refine_energies: list[float] = []
    route_means: list[float] = []
    raw_refine_norms: list[float] = []
    refine_norms: list[float] = []
    refine_removed_norms: list[float] = []
    refine_hf_fractions: list[float] = []
    pdo_norms: list[float] = []
    dissipative_norms: list[float] = []
    dissipative_means: list[float] = []
    transport_norms: list[float] = []
    symbol_norms: list[float] = []
    metadata_shifts: list[float] = []
    canonical_flow_errors: list[float] = []
    canonical_defects: list[float] = []
    wavefront_confidence_proxies: list[float] = []
    symbol_order_proxies: list[float] = []
    symbol_seminorm_proxies: list[float] = []
    local_tube_coordinate_norms: list[float] = []
    edge_symbol_deviation_proxies: list[float] = []
    pdo_order_proxies: list[float] = []
    wf_transport_errors: list[float] = []
    packet_wf_errors: list[float] = []
    complex_relative_errors: list[float] = []
    amplitude_errors: list[float] = []
    boundary_trace_errors: list[float] = []
    sobolev_h1_errors: list[float] = []
    sobolev_h2_errors: list[float] = []
    high_frequency_errors: list[float] = []
    egorov_intertwining_errors: list[float] = []
    egorov_targeted_errors: list[float] = []
    egorov_jacobian_self_errors: list[float] = []
    egorov_jacobian_target_errors: list[float] = []
    symbol_order_scaling_errors: list[float] = []
    helmholtz_shell_distance_proxies: list[float] = []
    helmholtz_resolvent_envelope_proxies: list[float] = []
    helmholtz_outgoing_flux_proxies: list[float] = []
    helmholtz_resolvent_real_proxies: list[float] = []
    helmholtz_resolvent_imag_proxies: list[float] = []
    helmholtz_outgoing_gate_proxies: list[float] = []
    helmholtz_shell_center_proxies: list[float] = []
    helmholtz_complex_latent_imag_energy_proxies: list[float] = []
    branch_entropies: list[float] = []
    branch_diversities: list[float] = []
    branch_usage_maxes: list[float] = []
    branch_spreads: list[float] = []
    tokenizer_reconstruction_errors: list[float] = []
    tokenizer_active_fractions: list[float] = []
    tokenizer_covering_radii: list[float] = []
    tokenizer_phase_window_diameters: list[float] = []
    transported_shift_norms: list[float] = []
    transported_input_shift_norms: list[float] = []
    transported_input_norms: list[float] = []
    transported_landing_norms: list[float] = []
    transported_landing_gates: list[float] = []
    field_correction_norms: list[float] = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            diagnostics = model.forward_with_diagnostics(inputs)
            prediction = diagnostics.get("prediction")
            if isinstance(prediction, Tensor):
                wf_transport_errors.append(float(wavefront_transport_proxy(prediction, targets).mean().item()))
                packet_wf_errors.append(float(packet_threshold_wavefront_localization_error(prediction, targets).mean().item()))
                complex_relative_errors.append(float(complex_relative_l2(prediction, targets).mean().item()))
                amplitude_errors.append(float(amplitude_relative_error(prediction, targets).mean().item()))
                boundary_trace_errors.append(float(boundary_trace_relative_error(prediction, targets).mean().item()))
                sobolev_h1_errors.append(float(sobolev_h1_relative_error(prediction, targets).mean().item()))
                sobolev_h2_errors.append(float(sobolev_relative_error(prediction, targets, order=2.0).mean().item()))
                high_frequency_errors.append(float(high_frequency_relative_error(prediction, targets).mean().item()))
                egorov_proxy = finite_egorov_intertwining_proxy(
                    model,
                    inputs,
                    prediction,
                    metadata_flow_delta=metadata_flow_delta,
                )
                if isinstance(egorov_proxy, Tensor):
                    egorov_intertwining_errors.append(float(egorov_proxy.mean().item()))
                egorov_targeted = finite_egorov_targeted_probe_proxy(
                    model,
                    inputs,
                    targets,
                    metadata_flow_delta=metadata_flow_delta,
                )
                if isinstance(egorov_targeted, Tensor):
                    egorov_targeted_errors.append(float(egorov_targeted.mean().item()))
                jacobian_proxies = finite_difference_egorov_jacobian_proxies(
                    model,
                    inputs,
                    prediction,
                    targets,
                    metadata_flow_delta=metadata_flow_delta,
                )
                if jacobian_proxies is not None:
                    jac_self, jac_target = jacobian_proxies
                    egorov_jacobian_self_errors.append(float(jac_self.mean().item()))
                    egorov_jacobian_target_errors.append(float(jac_target.mean().item()))
                symbol_order_scaling_errors.append(float(symbol_order_scaling_error(prediction, targets).mean().item()))
            core_prediction = diagnostics.get("core_prediction")
            refine_correction = diagnostics.get("refine_correction")
            if isinstance(core_prediction, Tensor):
                core_rels.append(float(relative_l2(core_prediction, targets).mean().item()))
            if isinstance(refine_correction, Tensor):
                target_energy = targets.square().mean().clamp_min(1e-8)
                refine_energies.append(float((refine_correction.square().mean() / target_energy).item()))
            route_mean = diagnostics.get("route_mean")
            if isinstance(route_mean, Tensor):
                route_means.append(float(route_mean.item()))
            refine_norm = diagnostics.get("refine_correction_norm")
            if isinstance(refine_norm, Tensor):
                refine_norms.append(float(refine_norm.item()))
            field_correction_norm = diagnostics.get("field_correction_norm")
            if isinstance(field_correction_norm, Tensor):
                field_correction_norms.append(float(field_correction_norm.item()))
            raw_refine_norm = diagnostics.get("raw_refine_correction_norm")
            if isinstance(raw_refine_norm, Tensor):
                raw_refine_norms.append(float(raw_refine_norm.item()))
            removed_refine_norm = diagnostics.get("refine_lowpass_removed_norm")
            if isinstance(removed_refine_norm, Tensor):
                refine_removed_norms.append(float(removed_refine_norm.item()))
            if isinstance(refine_correction, Tensor):
                refine_hf_fractions.append(float(high_frequency_fraction(refine_correction).mean().item()))
            final_metadata = diagnostics.get("final_metadata")
            token_output = diagnostics.get("token_output")
            encoding = diagnostics.get("encoding")
            if isinstance(final_metadata, Tensor) and isinstance(token_output, Tensor) and encoding is not None:
                initial_metadata = getattr(encoding, "metadata", None)
                if isinstance(initial_metadata, Tensor):
                    initial_batch = _metadata_batch(initial_metadata.to(device=final_metadata.device, dtype=final_metadata.dtype), token_output)
                    shift = final_metadata[..., :2] - initial_batch[..., :2]
                    metadata_shifts.append(float(torch.linalg.vector_norm(shift, dim=-1).mean().item()))
                    if metadata_flow_delta is not None:
                        target_delta = torch.tensor(metadata_flow_delta, device=shift.device, dtype=shift.dtype).view(1, 1, 2)
                        canonical_flow_errors.append(float(torch.linalg.vector_norm(shift - target_delta, dim=-1).mean().item()))
                    else:
                        canonical_flow_errors.append(float(torch.linalg.vector_norm(shift, dim=-1).mean().item()))
            for key, sink in (
                ("tokenizer_reconstruction_error", tokenizer_reconstruction_errors),
                ("tokenizer_active_fraction", tokenizer_active_fractions),
                ("tokenizer_covering_radius", tokenizer_covering_radii),
                ("tokenizer_phase_window_diameter", tokenizer_phase_window_diameters),
                ("transported_synthesis_shift_norm", transported_shift_norms),
                ("transported_input_shift_norm", transported_input_shift_norms),
                ("transported_input_norm", transported_input_norms),
                ("transported_landing_norm", transported_landing_norms),
                ("transported_landing_gate", transported_landing_gates),
            ):
                value = diagnostics.get(key)
                if isinstance(value, Tensor):
                    sink.append(float(value.item()))
            block_diagnostics = diagnostics.get("block_diagnostics", [])
            if isinstance(block_diagnostics, list):
                for block_diag in block_diagnostics:
                    if not isinstance(block_diag, dict):
                        continue
                    pdo_norm = block_diag.get("pdo_identity_norm")
                    if isinstance(pdo_norm, Tensor):
                        pdo_norms.append(float(pdo_norm.item()))
                    transport_norm = block_diag.get("transport_norm")
                    if isinstance(transport_norm, Tensor):
                        transport_norms.append(float(transport_norm.item()))
                    symbol_norm = block_diag.get("symbol_norm")
                    if isinstance(symbol_norm, Tensor):
                        symbol_norms.append(float(symbol_norm.item()))
                    canonical_defect = block_diag.get("canonical_defect_proxy")
                    if isinstance(canonical_defect, Tensor):
                        canonical_defects.append(float(canonical_defect.item()))
                    wavefront_confidence = block_diag.get("wavefront_confidence_proxy")
                    if isinstance(wavefront_confidence, Tensor):
                        wavefront_confidence_proxies.append(float(wavefront_confidence.item()))
                    symbol_order = block_diag.get("symbol_order_proxy")
                    if isinstance(symbol_order, Tensor):
                        symbol_order_proxies.append(float(symbol_order.item()))
                    symbol_seminorm = block_diag.get("symbol_seminorm_proxy")
                    if isinstance(symbol_seminorm, Tensor):
                        symbol_seminorm_proxies.append(float(symbol_seminorm.item()))
                    local_tube_coordinate_norm = block_diag.get("local_tube_coordinate_norm")
                    if isinstance(local_tube_coordinate_norm, Tensor):
                        local_tube_coordinate_norms.append(float(local_tube_coordinate_norm.item()))
                    edge_symbol_deviation = block_diag.get("edge_symbol_deviation_proxy")
                    if isinstance(edge_symbol_deviation, Tensor):
                        edge_symbol_deviation_proxies.append(float(edge_symbol_deviation.item()))
                    helmholtz_shell_distance = block_diag.get("helmholtz_shell_distance_proxy")
                    if isinstance(helmholtz_shell_distance, Tensor):
                        helmholtz_shell_distance_proxies.append(float(helmholtz_shell_distance.item()))
                    helmholtz_resolvent_envelope = block_diag.get("helmholtz_resolvent_envelope_proxy")
                    if isinstance(helmholtz_resolvent_envelope, Tensor):
                        helmholtz_resolvent_envelope_proxies.append(float(helmholtz_resolvent_envelope.item()))
                    helmholtz_outgoing_flux = block_diag.get("helmholtz_outgoing_flux_proxy")
                    if isinstance(helmholtz_outgoing_flux, Tensor):
                        helmholtz_outgoing_flux_proxies.append(float(helmholtz_outgoing_flux.item()))
                    helmholtz_resolvent_real = block_diag.get("helmholtz_resolvent_real_proxy")
                    if isinstance(helmholtz_resolvent_real, Tensor):
                        helmholtz_resolvent_real_proxies.append(float(helmholtz_resolvent_real.item()))
                    helmholtz_resolvent_imag = block_diag.get("helmholtz_resolvent_imag_proxy")
                    if isinstance(helmholtz_resolvent_imag, Tensor):
                        helmholtz_resolvent_imag_proxies.append(float(helmholtz_resolvent_imag.item()))
                    helmholtz_outgoing_gate = block_diag.get("helmholtz_outgoing_gate_proxy")
                    if isinstance(helmholtz_outgoing_gate, Tensor):
                        helmholtz_outgoing_gate_proxies.append(float(helmholtz_outgoing_gate.item()))
                    helmholtz_shell_center = block_diag.get("helmholtz_shell_center_proxy")
                    if isinstance(helmholtz_shell_center, Tensor):
                        helmholtz_shell_center_proxies.append(float(helmholtz_shell_center.item()))
                    helmholtz_complex_latent_imag_energy = block_diag.get("helmholtz_complex_latent_imag_energy_proxy")
                    if isinstance(helmholtz_complex_latent_imag_energy, Tensor):
                        helmholtz_complex_latent_imag_energy_proxies.append(float(helmholtz_complex_latent_imag_energy.item()))
                    pdo_order = block_diag.get("pdo_identity_order_proxy")
                    if isinstance(pdo_order, Tensor):
                        pdo_order_proxies.append(float(pdo_order.item()))
                    dissipative_norm = block_diag.get("dissipative_symbol_norm")
                    if isinstance(dissipative_norm, Tensor):
                        dissipative_norms.append(float(dissipative_norm.item()))
                    multiplier_mean = block_diag.get("dissipative_multiplier_mean")
                    if isinstance(multiplier_mean, Tensor):
                        dissipative_means.append(float(multiplier_mean.item()))
                    branch_entropy = block_diag.get("branch_entropy")
                    if isinstance(branch_entropy, Tensor):
                        branch_entropies.append(float(branch_entropy.item()))
                    branch_diversity = block_diag.get("branch_diversity")
                    if isinstance(branch_diversity, Tensor):
                        branch_diversities.append(float(branch_diversity.item()))
                    branch_usage_max = block_diag.get("branch_usage_max")
                    if isinstance(branch_usage_max, Tensor):
                        branch_usage_maxes.append(float(branch_usage_max.item()))
                    branch_spread = block_diag.get("branch_spread")
                    if isinstance(branch_spread, Tensor):
                        branch_spreads.append(float(branch_spread.item()))
    return {
        "test_core_relative_l2": sum(core_rels) / max(len(core_rels), 1),
        "test_refine_relative_energy": sum(refine_energies) / max(len(refine_energies), 1),
        "test_route_mean": sum(route_means) / max(len(route_means), 1),
        "test_raw_refine_correction_norm": sum(raw_refine_norms) / max(len(raw_refine_norms), 1),
        "test_refine_correction_norm": sum(refine_norms) / max(len(refine_norms), 1),
        "test_field_correction_norm": sum(field_correction_norms) / max(len(field_correction_norms), 1),
        "test_refine_lowpass_removed_norm": sum(refine_removed_norms) / max(len(refine_removed_norms), 1),
        "test_refine_high_frequency_fraction": sum(refine_hf_fractions) / max(len(refine_hf_fractions), 1),
        "test_pdo_identity_norm": sum(pdo_norms) / max(len(pdo_norms), 1),
        "test_dissipative_symbol_norm": sum(dissipative_norms) / max(len(dissipative_norms), 1),
        "test_dissipative_multiplier_mean": sum(dissipative_means) / max(len(dissipative_means), 1),
        "test_transport_budget": sum(transport_norms) / max(len(transport_norms), 1),
        "test_symbol_budget": sum(symbol_norms) / max(len(symbol_norms), 1),
        "test_residual_energy": sum(refine_energies) / max(len(refine_energies), 1),
        "test_refinement_energy": sum(refine_energies) / max(len(refine_energies), 1),
        "test_canonical_flow_error": sum(canonical_flow_errors) / max(len(canonical_flow_errors), 1),
        "test_metadata_shift_norm": sum(metadata_shifts) / max(len(metadata_shifts), 1),
        "test_symplectic_defect_proxy": sum(canonical_defects) / max(len(canonical_defects), 1),
        "test_wavefront_confidence_proxy": sum(wavefront_confidence_proxies) / max(len(wavefront_confidence_proxies), 1),
        "test_symbol_order_proxy": sum(symbol_order_proxies) / max(len(symbol_order_proxies), 1),
        "test_symbol_seminorm_proxy": sum(symbol_seminorm_proxies) / max(len(symbol_seminorm_proxies), 1),
        "test_local_tube_coordinate_norm": sum(local_tube_coordinate_norms) / max(len(local_tube_coordinate_norms), 1),
        "test_edge_symbol_deviation_proxy": sum(edge_symbol_deviation_proxies) / max(len(edge_symbol_deviation_proxies), 1),
        "test_helmholtz_shell_distance_proxy": sum(helmholtz_shell_distance_proxies) / max(len(helmholtz_shell_distance_proxies), 1),
        "test_helmholtz_resolvent_envelope_proxy": sum(helmholtz_resolvent_envelope_proxies) / max(len(helmholtz_resolvent_envelope_proxies), 1),
        "test_helmholtz_outgoing_flux_proxy": sum(helmholtz_outgoing_flux_proxies) / max(len(helmholtz_outgoing_flux_proxies), 1),
        "test_helmholtz_resolvent_real_proxy": sum(helmholtz_resolvent_real_proxies) / max(len(helmholtz_resolvent_real_proxies), 1),
        "test_helmholtz_resolvent_imag_proxy": sum(helmholtz_resolvent_imag_proxies) / max(len(helmholtz_resolvent_imag_proxies), 1),
        "test_helmholtz_outgoing_gate_proxy": sum(helmholtz_outgoing_gate_proxies) / max(len(helmholtz_outgoing_gate_proxies), 1),
        "test_helmholtz_shell_center_proxy": sum(helmholtz_shell_center_proxies) / max(len(helmholtz_shell_center_proxies), 1),
        "test_helmholtz_complex_latent_imag_energy_proxy": sum(helmholtz_complex_latent_imag_energy_proxies) / max(len(helmholtz_complex_latent_imag_energy_proxies), 1),
        "test_pdo_identity_order_proxy": sum(pdo_order_proxies) / max(len(pdo_order_proxies), 1),
        "test_wf_transport_error_proxy": sum(wf_transport_errors) / max(len(wf_transport_errors), 1),
        "test_packet_wavefront_localization_error": sum(packet_wf_errors) / max(len(packet_wf_errors), 1),
        "test_complex_relative_l2_proxy": sum(complex_relative_errors) / max(len(complex_relative_errors), 1),
        "test_amplitude_error_proxy": sum(amplitude_errors) / max(len(amplitude_errors), 1),
        "test_boundary_trace_error_proxy": sum(boundary_trace_errors) / max(len(boundary_trace_errors), 1),
        "test_sobolev_h1_error_proxy": sum(sobolev_h1_errors) / max(len(sobolev_h1_errors), 1),
        "test_sobolev_h2_error_proxy": sum(sobolev_h2_errors) / max(len(sobolev_h2_errors), 1),
        "test_high_frequency_relative_error_proxy": sum(high_frequency_errors) / max(len(high_frequency_errors), 1),
        "test_egorov_intertwining_proxy": sum(egorov_intertwining_errors) / max(len(egorov_intertwining_errors), 1),
        "test_egorov_targeted_probe_proxy": sum(egorov_targeted_errors) / max(len(egorov_targeted_errors), 1),
        "test_egorov_jacobian_self_proxy": sum(egorov_jacobian_self_errors) / max(len(egorov_jacobian_self_errors), 1),
        "test_egorov_jacobian_target_proxy": sum(egorov_jacobian_target_errors) / max(len(egorov_jacobian_target_errors), 1),
        "test_symbol_order_scaling_error_proxy": sum(symbol_order_scaling_errors) / max(len(symbol_order_scaling_errors), 1),
        "test_symbol_error_proxy": sum(symbol_norms) / max(len(symbol_norms), 1),
        "test_branch_entropy": sum(branch_entropies) / max(len(branch_entropies), 1),
        "test_branch_diversity": sum(branch_diversities) / max(len(branch_diversities), 1),
        "test_branch_usage_max": sum(branch_usage_maxes) / max(len(branch_usage_maxes), 1),
        "test_branch_spread": sum(branch_spreads) / max(len(branch_spreads), 1),
        "test_tokenizer_reconstruction_error": sum(tokenizer_reconstruction_errors) / max(len(tokenizer_reconstruction_errors), 1),
        "test_tokenizer_active_fraction": sum(tokenizer_active_fractions) / max(len(tokenizer_active_fractions), 1),
        "test_tokenizer_covering_radius": sum(tokenizer_covering_radii) / max(len(tokenizer_covering_radii), 1),
        "test_tokenizer_phase_window_diameter": sum(tokenizer_phase_window_diameters) / max(len(tokenizer_phase_window_diameters), 1),
        "test_transported_synthesis_shift_norm": sum(transported_shift_norms) / max(len(transported_shift_norms), 1),
        "test_transported_input_shift_norm": sum(transported_input_shift_norms) / max(len(transported_input_shift_norms), 1),
        "test_transported_input_norm": sum(transported_input_norms) / max(len(transported_input_norms), 1),
        "test_transported_landing_norm": sum(transported_landing_norms) / max(len(transported_landing_norms), 1),
        "test_transported_landing_gate": sum(transported_landing_gates) / max(len(transported_landing_gates), 1),
    }


def run_training_once(
    *,
    scenario: str,
    seed: int,
    batch_size: int,
    loss_config: str,
    ablation: str,
    device: torch.device,
    args: argparse.Namespace,
    oom_fallback: bool,
) -> dict[str, object]:
    transport_weight, symbol_weight = LOSS_CONFIGS[loss_config]
    tcno_cache_root = Path(args.tcno_cache_root) if args.tcno_cache_root else None
    loaders = build_benchmark_loaders(
        scenario_name=scenario,
        batch_size=batch_size,
        seed=seed,
        tcno_cache_root=tcno_cache_root,
    )
    loaders = maybe_limit_loaders(loaders, args)
    spec = loaders.spec

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model_label = "MiNO-Plus"
    if ablation in BASELINE_REFINE_ABLATIONS:
        base_name = BASELINE_REFINE_ABLATIONS[ablation]
        base_model = build_model(
            base_name,
            in_channels=loaders.in_channels,
            out_channels=loaders.out_channels,
            model_kwargs={},
        )
        model = SameLocalRefineWrapper(
            base_model,
            out_channels=loaders.out_channels,
            local_refine_channels=args.plus_local_refine_channels,
            local_refine_scale=min(float(args.plus_local_refine_scale), 0.15),
            refine_lowpass_cutoff=args.plus_refine_lowpass_cutoff,
            route_bias_init=-4.0 if args.plus_route_bias_init == -1.5 else args.plus_route_bias_init,
        )
        model_label = f"{base_name}+SameRefine"
    else:
        model = build_model(
            "MiNO-Plus",
            in_channels=loaders.in_channels,
            out_channels=loaders.out_channels,
            model_kwargs=plus_model_kwargs(args),
        )
        apply_ablation(model, ablation, loaders.out_channels, scenario=scenario)
    model = model.to(device)
    flow_loss_weight = effective_metadata_flow_loss_weight(args, ablation)

    history = fit_model(
        model,
        loaders.train_loader,
        loaders.val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=None if args.grad_clip_norm <= 0 else args.grad_clip_norm,
        restore_best=True,
        transport_proxy_weight=transport_weight,
        symbol_proxy_weight=symbol_weight,
        proxy_temperature=args.proxy_temperature,
        core_field_weight=args.core_field_weight,
        residual_energy_weight=args.residual_energy_weight,
        route_l1_weight=args.route_l1_weight,
        canonical_loss_weight=args.canonical_loss_weight,
        symbol_order_loss_weight=args.symbol_order_loss_weight,
        symbol_order_target=args.symbol_order_target,
        symbol_seminorm_loss_weight=args.symbol_seminorm_loss_weight,
        symbol_seminorm_target=args.symbol_seminorm_target,
        packet_space_loss_weight=args.packet_space_loss_weight,
        highfreq_core_loss_weight=args.highfreq_core_loss_weight,
        highfreq_cutoff=args.highfreq_cutoff,
        helmholtz_residual_loss_weight=args.helmholtz_residual_loss_weight,
        helmholtz_residual_wavenumber=args.helmholtz_residual_wavenumber,
        helmholtz_residual_refractive_index=args.helmholtz_refractive_index,
        complex_pair_loss_weight=args.complex_pair_loss_weight,
        complex_phase_loss_weight=args.complex_phase_loss_weight,
        symbol_identity_loss_weight=args.symbol_identity_loss_weight,
        metadata_flow_loss_weight=flow_loss_weight,
        metadata_flow_delta=OracleConstantFlowPropagation.SUPPORTED_DELTAS.get(scenario),
        branch_entropy_weight=args.branch_entropy_weight,
        branch_diversity_weight=args.branch_diversity_weight,
        core_warmup_epochs=args.core_warmup_epochs,
        freeze_refinement_epochs=args.freeze_refinement_epochs,
        progress_every_epochs=args.progress_every_epochs,
        progress_label=f"{scenario}/{ablation}/seed{seed}",
    )
    metrics = evaluate_model(model, loaders.test_loader, device=device, criterion=nn.MSELoss())
    branch_metrics = evaluate_branch_diagnostics(
        model,
        loaders.test_loader,
        device,
        metadata_flow_delta=OracleConstantFlowPropagation.SUPPORTED_DELTAS.get(scenario),
    )
    final_history = history["history"][-1] if history["history"] else {}
    row = {
        "row_type": "run",
        "campaign": args.campaign,
        "scenario": scenario,
        "family": spec.family,
        "regime": spec.regime,
        "source_kind": spec.source_kind,
        "model": model_label,
        "model_variant": (
            plus_model_variant(args)
            if model_label == "MiNO-Plus"
            else f"{safe_slug(model_label)}_same_ref{min(float(args.plus_local_refine_scale), 0.15):g}"
        ),
        "loss_config": loss_config,
        "ablation": ablation,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "requested_batch_size": args.batch_size,
        "helmholtz_profile": args.helmholtz_profile,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "oom_fallback": oom_fallback,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "transport_proxy_weight": transport_weight,
        "symbol_proxy_weight": symbol_weight,
        "proxy_temperature": args.proxy_temperature,
        "core_field_weight": args.core_field_weight,
        "residual_energy_weight": args.residual_energy_weight,
        "route_l1_weight": args.route_l1_weight,
        "canonical_loss_weight": args.canonical_loss_weight,
        "symbol_order_loss_weight": args.symbol_order_loss_weight,
        "symbol_order_target": args.symbol_order_target,
        "symbol_seminorm_loss_weight": args.symbol_seminorm_loss_weight,
        "symbol_seminorm_target": args.symbol_seminorm_target,
        "packet_space_loss_weight": args.packet_space_loss_weight,
        "highfreq_core_loss_weight": args.highfreq_core_loss_weight,
        "helmholtz_residual_loss_weight": args.helmholtz_residual_loss_weight,
        "helmholtz_residual_wavenumber": args.helmholtz_residual_wavenumber,
        "complex_pair_loss_weight": args.complex_pair_loss_weight,
        "complex_phase_loss_weight": args.complex_phase_loss_weight,
        "symbol_identity_loss_weight": args.symbol_identity_loss_weight,
        "metadata_flow_loss_weight": args.metadata_flow_loss_weight,
        "effective_metadata_flow_loss_weight": flow_loss_weight,
        "num_canonical_branches": args.num_canonical_branches,
        "branch_routing": args.branch_routing,
        "branch_prior_strength": args.branch_prior_strength,
        "branch_entropy_weight": args.branch_entropy_weight,
        "branch_diversity_weight": args.branch_diversity_weight,
        "branch_synthesis": args.branch_synthesis,
        "edge_symbol_parameterization": args.edge_symbol_parameterization,
        "edge_symbol_strength": args.edge_symbol_strength,
        "field_corrector": args.field_corrector,
        "field_corrector_scale": args.field_corrector_scale,
        "field_corrector_width": args.field_corrector_width,
        "highfreq_cutoff": args.highfreq_cutoff,
        "core_warmup_epochs": args.core_warmup_epochs,
        "freeze_refinement_epochs": args.freeze_refinement_epochs,
        "window_type": args.plus_window_type,
        "mode_strategy": args.plus_mode_strategy,
        "transport_parameterization": args.transport_parameterization,
        "sparse_topk": args.sparse_topk,
        "frame_type": args.frame_type,
        "transport_stencil": args.plus_transport_stencil,
        "max_modes": args.plus_max_modes,
        "patch_size": args.plus_patch_size,
        "stride": args.plus_stride,
        "local_refine_scale": args.plus_local_refine_scale,
        "refine_lowpass_cutoff": args.plus_refine_lowpass_cutoff,
        "transport_highpass_cutoff": args.transport_highpass_cutoff,
        "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
        "transported_synthesis_scale": args.transported_synthesis_scale,
        "transported_input_scale": args.transported_input_scale,
        "transported_synthesis_mode": args.transported_synthesis_mode,
        "transported_decoder_channels": args.transported_decoder_channels,
        "transported_decoder_scale": args.transported_decoder_scale,
        "transported_decoder_transport_gate": args.transported_decoder_transport_gate,
        "token_refine_scale": args.token_refine_scale,
        "route_bias_init": args.plus_route_bias_init,
        "pdo_symbol_scale": args.plus_pdo_symbol_scale,
        "pdo_symbol_order": args.plus_pdo_symbol_order,
        "dissipative_symbol_scale": args.plus_dissipative_symbol_scale,
        "dissipative_time_step": args.plus_dissipative_time_step,
        "symbol_order": args.plus_symbol_order,
        "helmholtz_shell_radius": args.helmholtz_shell_radius,
        "helmholtz_refractive_index": args.helmholtz_refractive_index,
        "helmholtz_absorption": args.helmholtz_absorption,
        "helmholtz_resolvent_cap": args.helmholtz_resolvent_cap,
        "wavefront_confidence_scale": args.plus_wavefront_confidence_scale,
        "in_channels": loaders.in_channels,
        "out_channels": loaders.out_channels,
        "height": loaders.spatial_shape[0],
        "width": loaders.spatial_shape[1],
        "parameters": count_parameters(model),
        "runtime_seconds": history["runtime_seconds"],
        "best_val_loss": min((item["val_loss"] for item in history["history"]), default=math.nan),
        "test_loss": metrics.loss,
        "test_relative_l2": metrics.relative_l2,
        "test_phase_error": metrics.phase_error,
        "test_packet_consistency": metrics.packet_consistency,
        **branch_metrics,
        "final_train_transport_proxy": final_history.get("train_transport_proxy", 0.0),
        "final_train_symbol_proxy": final_history.get("train_symbol_proxy", 0.0),
        "final_train_core_field_loss": final_history.get("train_core_field_loss", 0.0),
        "final_train_residual_energy": final_history.get("train_residual_energy", 0.0),
        "final_train_route_l1": final_history.get("train_route_l1", 0.0),
        "final_train_canonical_consistency": final_history.get("train_canonical_consistency", 0.0),
        "final_train_symbol_order_loss": final_history.get("train_symbol_order_loss", 0.0),
        "final_train_symbol_seminorm_loss": final_history.get("train_symbol_seminorm_loss", 0.0),
        "final_train_packet_space_loss": final_history.get("train_packet_space_loss", 0.0),
        "final_train_highfreq_core_loss": final_history.get("train_highfreq_core_loss", 0.0),
        "final_train_helmholtz_residual_loss": final_history.get("train_helmholtz_residual_loss", 0.0),
        "final_train_complex_pair_loss": final_history.get("train_complex_pair_loss", 0.0),
        "final_train_complex_phase_loss": final_history.get("train_complex_phase_loss", 0.0),
        "final_train_symbol_identity_loss": final_history.get("train_symbol_identity_loss", 0.0),
        "final_train_metadata_flow_loss": final_history.get("train_metadata_flow_loss", 0.0),
        "final_train_branch_entropy_loss": final_history.get("train_branch_entropy_loss", 0.0),
        "final_train_branch_diversity_loss": final_history.get("train_branch_diversity_loss", 0.0),
        "final_train_core_warmup_loss": final_history.get("train_core_warmup_loss", 0.0),
        "reference_source": None,
    }
    row["run_id"] = make_run_id(row)
    ensure_finite(row)
    return row


def configure_torch_runtime(device: torch.device) -> dict[str, object]:
    flags: dict[str, object] = {
        "device": str(device),
        "tf32": False,
        "cudnn_benchmark": False,
    }
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        flags["tf32"] = True
        flags["cudnn_benchmark"] = True
    return flags


def run_with_oom_fallback(
    run_fn: Callable[[int, bool], dict[str, object]],
    batch_size: int,
) -> dict[str, object]:
    try:
        return run_fn(batch_size, False)
    except RuntimeError as error:
        if batch_size <= 1 or not is_cuda_oom(error):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return run_fn(1, True)


def make_run_id(row: dict[str, object]) -> str:
    prefix = "_".join(
        safe_slug(str(row.get(key, "")))
        for key in ("campaign", "scenario", "loss_config", "ablation", "seed")
        if key in row
    )
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix[:96]}_{digest}"


def unsupported_ablation_reason(ablation: str, scenario: str) -> str | None:
    if ablation == "oracle_transport" and scenario not in OracleConstantFlowPropagation.SUPPORTED:
        return (
            "oracle_transport requires a known packet-center flow; "
            f"supported scenarios are {sorted(OracleConstantFlowPropagation.SUPPORTED)}."
        )
    if ablation == "oracle_symbol" and scenario not in OracleIdentitySymbol.SUPPORTED:
        return (
            "oracle_symbol requires a controlled principal-symbol diagnostic; "
            f"supported scenarios are {sorted(OracleIdentitySymbol.SUPPORTED)}."
        )
    return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("row_type") != "run":
            continue
        key = (
            str(row["scenario"]),
            str(row["loss_config"]),
            str(row["ablation"]),
            str(row["model_variant"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (scenario, loss_config, ablation, model_variant), group in sorted(grouped.items()):
        out = {
            "scenario": scenario,
            "family": group[0]["family"],
            "regime": group[0]["regime"],
            "loss_config": loss_config,
            "ablation": ablation,
            "model_variant": model_variant,
            "runs": len(group),
        }
        for field in METRIC_FIELDS:
            values = [float(row[field]) for row in group if row.get(field) is not None]
            mean, std = mean_std(values)
            out[f"mean_{field}"] = mean
            out[f"std_{field}"] = std
        summary.append(out)
    return summary


def make_proxy_deltas(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["scenario"]), str(row["ablation"])): row
        for row in summary
        if row["loss_config"] == "field_only"
    }
    deltas: list[dict[str, object]] = []
    for row in summary:
        if row["loss_config"] == "field_only":
            continue
        baseline = by_key.get((str(row["scenario"]), str(row["ablation"])))
        if baseline is None:
            continue
        deltas.append(
            {
                "delta_type": "proxy_vs_field",
                "scenario": row["scenario"],
                "ablation": row["ablation"],
                "comparison": f"{row['loss_config']} - field_only",
                "delta_relative_l2": float(row["mean_test_relative_l2"]) - float(baseline["mean_test_relative_l2"]),
                "delta_phase_error": float(row["mean_test_phase_error"]) - float(baseline["mean_test_phase_error"]),
                "delta_packet_consistency": float(row["mean_test_packet_consistency"]) - float(baseline["mean_test_packet_consistency"]),
                "candidate_relative_l2": row["mean_test_relative_l2"],
                "baseline_relative_l2": baseline["mean_test_relative_l2"],
            }
        )
    return deltas


def make_ablation_deltas(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    full_rows = {
        (str(row["scenario"]), str(row["loss_config"])): row
        for row in summary
        if row["ablation"] == "full"
    }
    deltas: list[dict[str, object]] = []
    for row in summary:
        if row["ablation"] == "full":
            continue
        baseline = full_rows.get((str(row["scenario"]), str(row["loss_config"])))
        if baseline is None:
            continue
        deltas.append(
            {
                "delta_type": "ablation_vs_full",
                "scenario": row["scenario"],
                "loss_config": row["loss_config"],
                "ablation": row["ablation"],
                "delta_relative_l2": float(row["mean_test_relative_l2"]) - float(baseline["mean_test_relative_l2"]),
                "delta_phase_error": float(row["mean_test_phase_error"]) - float(baseline["mean_test_phase_error"]),
                "delta_packet_consistency": float(row["mean_test_packet_consistency"]) - float(baseline["mean_test_packet_consistency"]),
                "candidate_relative_l2": row["mean_test_relative_l2"],
                "baseline_relative_l2": baseline["mean_test_relative_l2"],
            }
        )
    return deltas


def resolve_campaign_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.epochs <= 0:
        args.epochs = CAMPAIGN_EPOCHS[args.campaign]
    if not args.scenarios:
        if args.campaign == "transport_id":
            args.scenarios = ",".join(TRANSPORT_ID_SCENARIOS)
        elif args.campaign == "calculus_id":
            args.scenarios = ",".join(CALCULUS_ID_SCENARIOS)
        elif args.campaign == "branch_id":
            args.scenarios = ",".join(BRANCH_ID_SCENARIOS)
        elif args.campaign == "branch_id_v2":
            args.scenarios = ",".join(BRANCH_ID_V2_SCENARIOS)
        elif args.campaign in {"branch_id_v3", "branch_id_v3_controls"}:
            args.scenarios = ",".join(BRANCH_ID_V3_SCENARIOS)
        elif args.campaign == "helmholtz_branched_highk":
            args.scenarios = ",".join(HELMHOLTZ_BRANCHED_SCENARIOS)
        elif args.campaign == "helmholtz_highk_careful":
            args.scenarios = ",".join(HELMHOLTZ_HIGHK_CAREFUL_SCENARIOS)
        elif args.campaign == "helmholtz_highk_flagship":
            args.scenarios = ",".join(HELMHOLTZ_HIGHK_FLAGSHIP_SCENARIOS)
        elif args.campaign == "helmholtz_highk_8gb":
            args.scenarios = ",".join(HELMHOLTZ_HIGHK_8GB_SCENARIOS)
        elif args.campaign == "cross_resolution_wave":
            args.scenarios = ",".join(CROSS_RESOLUTION_WAVE_SCENARIOS)
        elif args.campaign == "ablation_core":
            args.scenarios = ",".join(CORE_ABLATION_SCENARIOS)
        else:
            args.scenarios = ",".join(DEFAULT_SCENARIOS)
    if not args.loss_configs:
        args.loss_configs = ",".join(CAMPAIGN_LOSS_CONFIGS[args.campaign])
    if args.campaign == "ablation_core" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(CORE_ABLATIONS)
    if args.campaign == "transport_id" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(TRANSPORT_ID_ABLATIONS)
    if args.campaign == "calculus_id" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(CALCULUS_ID_ABLATIONS)
    if args.campaign == "branch_id" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(BRANCH_ID_ABLATIONS)
    if args.campaign == "branch_id_v2" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(BRANCH_ID_V2_ABLATIONS)
    if args.campaign == "branch_id_v3" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(BRANCH_ID_V3_ABLATIONS)
    if args.campaign == "branch_id_v3_controls" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(BRANCH_ID_V3_CONTROL_ABLATIONS)
    if args.campaign == "helmholtz_branched_highk" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(HELMHOLTZ_BRANCHED_ABLATIONS)
    if args.campaign == "helmholtz_highk_careful" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS)
    if args.campaign == "helmholtz_highk_flagship" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(HELMHOLTZ_HIGHK_FLAGSHIP_ABLATIONS)
    if args.campaign == "helmholtz_highk_8gb" and args.ablations == ",".join(DEFAULT_ABLATIONS):
        args.ablations = ",".join(HELMHOLTZ_HIGHK_8GB_ABLATIONS)
    if args.campaign == "transport_id":
        if args.plus_local_refine_scale == 1.0:
            args.plus_local_refine_scale = 0.15
        if args.plus_route_bias_init == -1.5:
            args.plus_route_bias_init = -4.0
        if args.core_field_weight == 0.0:
            args.core_field_weight = 1.0
        if args.residual_energy_weight == 0.0:
            args.residual_energy_weight = 0.05
        if args.route_l1_weight == 0.0:
            args.route_l1_weight = 0.01
        if args.plus_refine_lowpass_cutoff == 0.0:
            args.plus_refine_lowpass_cutoff = 0.25
        if args.core_warmup_epochs == 0:
            args.core_warmup_epochs = max(1, args.epochs // 4)
        if args.freeze_refinement_epochs == 0:
            args.freeze_refinement_epochs = max(1, args.epochs // 4)
    if args.campaign == "calculus_id":
        if args.transport_parameterization == "mlp_displacement" and not args.preserve_transport_parameterization:
            args.transport_parameterization = "hamiltonian_verlet"
        if args.frame_type == "local_fft":
            args.frame_type = "gabor_gaussian"
        if args.symbol_parameterization == "mlp":
            args.symbol_parameterization = "spectral_order"
        args.sparse_topk = True
        args.plus_low_frequency_scale = 0.0
        if args.plus_local_refine_scale == 1.0:
            args.plus_local_refine_scale = 0.05
        if args.plus_route_bias_init == -1.5:
            args.plus_route_bias_init = -5.0
        if args.plus_pdo_symbol_scale == 0.0:
            args.plus_pdo_symbol_scale = 0.10
        if args.plus_dissipative_symbol_scale == 0.0:
            args.plus_dissipative_symbol_scale = 0.10
        if args.plus_wavefront_confidence_scale == 0.0:
            args.plus_wavefront_confidence_scale = 2.0
        if args.core_field_weight == 0.0:
            args.core_field_weight = 2.0
        if args.residual_energy_weight == 0.0:
            args.residual_energy_weight = 0.50
        if args.route_l1_weight == 0.0:
            args.route_l1_weight = 0.05
        if args.canonical_loss_weight == 0.0:
            args.canonical_loss_weight = 0.02
        if args.symbol_order_loss_weight == 0.0:
            args.symbol_order_loss_weight = 0.01
        if args.symbol_order_target == 0.0:
            args.symbol_order_target = args.plus_symbol_order
        if args.symbol_seminorm_loss_weight == 0.0:
            args.symbol_seminorm_loss_weight = 0.001
        if args.packet_space_loss_weight == 0.0:
            args.packet_space_loss_weight = 0.05
        if args.highfreq_core_loss_weight == 0.0:
            args.highfreq_core_loss_weight = 1.00
        if args.plus_refine_lowpass_cutoff == 0.0:
            args.plus_refine_lowpass_cutoff = 0.16
        if args.transport_highpass_cutoff == 0.0:
            args.transport_highpass_cutoff = 0.16
        if args.highfreq_cutoff == 0.0:
            args.highfreq_cutoff = args.transport_highpass_cutoff
        if args.skip_lowpass_cutoff == 0.0:
            args.skip_lowpass_cutoff = 0.16
        if args.transported_synthesis_scale == 0.0:
            args.transported_synthesis_scale = 1.0
        if args.transported_input_scale == 0.0:
            args.transported_input_scale = 1.0
        if args.transported_decoder_channels == 0:
            args.transported_decoder_channels = 16
        if args.transported_decoder_scale == 0.0:
            args.transported_decoder_scale = 1.0
        if args.token_refine_scale == 1.0:
            args.token_refine_scale = 0.0
        if args.core_warmup_epochs == 0:
            args.core_warmup_epochs = max(2, args.epochs // 2)
        if args.freeze_refinement_epochs == 0:
            args.freeze_refinement_epochs = max(2, args.epochs // 2)
    if args.campaign in {"branch_id", "branch_id_v2", "branch_id_v3", "branch_id_v3_controls"}:
        if args.plus_local_refine_scale == 1.0:
            args.plus_local_refine_scale = 0.15
        if args.plus_route_bias_init == -1.5:
            args.plus_route_bias_init = -4.0
        if args.core_field_weight == 0.0:
            args.core_field_weight = 1.0
        if args.residual_energy_weight == 0.0:
            args.residual_energy_weight = 0.05
        if args.route_l1_weight == 0.0:
            args.route_l1_weight = 0.01
        if args.campaign == "branch_id_v2":
            if args.transport_parameterization == "mlp_displacement" and not args.preserve_transport_parameterization:
                args.transport_parameterization = "hamiltonian_verlet"
            if args.frame_type == "local_fft":
                args.frame_type = "gabor_gaussian"
            if args.symbol_parameterization == "mlp":
                args.symbol_parameterization = "spectral_order"
            args.sparse_topk = True
            if args.canonical_loss_weight == 0.0:
                args.canonical_loss_weight = 0.01
            if args.plus_wavefront_confidence_scale == 0.0:
                args.plus_wavefront_confidence_scale = 1.0
            if args.symbol_order_loss_weight == 0.0:
                args.symbol_order_loss_weight = 0.005
            if args.symbol_order_target == 0.0:
                args.symbol_order_target = args.plus_symbol_order
            if args.plus_refine_lowpass_cutoff == 0.0:
                args.plus_refine_lowpass_cutoff = 0.25
            if args.transport_highpass_cutoff == 0.0:
                args.transport_highpass_cutoff = 0.25
            if args.packet_space_loss_weight == 0.0:
                args.packet_space_loss_weight = 0.01
            if args.highfreq_core_loss_weight == 0.0:
                args.highfreq_core_loss_weight = 0.05
            if args.highfreq_cutoff == 0.0:
                args.highfreq_cutoff = args.transport_highpass_cutoff
            if args.core_warmup_epochs == 0:
                args.core_warmup_epochs = max(1, args.epochs // 4)
            if args.freeze_refinement_epochs == 0:
                args.freeze_refinement_epochs = max(1, args.epochs // 4)
        if args.campaign in {"branch_id_v3", "branch_id_v3_controls"}:
            if args.transport_parameterization == "mlp_displacement" and not args.preserve_transport_parameterization:
                args.transport_parameterization = "hamiltonian_verlet"
            if args.frame_type == "local_fft":
                args.frame_type = "gabor_gaussian"
            if args.symbol_parameterization == "mlp":
                args.symbol_parameterization = "spectral_order"
            args.sparse_topk = True
            args.plus_low_frequency_scale = 0.0
            if args.plus_local_refine_scale == 0.15:
                args.plus_local_refine_scale = 0.05
            if args.plus_route_bias_init == -4.0:
                args.plus_route_bias_init = -5.0
            if args.plus_wavefront_confidence_scale == 0.0:
                args.plus_wavefront_confidence_scale = 2.0
            if args.plus_refine_lowpass_cutoff == 0.0:
                args.plus_refine_lowpass_cutoff = 0.16
            if args.transport_highpass_cutoff == 0.0:
                args.transport_highpass_cutoff = 0.16
            if args.highfreq_cutoff == 0.0:
                args.highfreq_cutoff = args.transport_highpass_cutoff
            if args.core_field_weight == 1.0:
                args.core_field_weight = 2.0
            if args.residual_energy_weight == 0.05:
                args.residual_energy_weight = 0.50
            if args.route_l1_weight == 0.01:
                args.route_l1_weight = 0.05
            if args.canonical_loss_weight == 0.0:
                args.canonical_loss_weight = 0.02
            if args.symbol_order_loss_weight == 0.0:
                args.symbol_order_loss_weight = 0.01
            if args.symbol_order_target == 0.0:
                args.symbol_order_target = args.plus_symbol_order
            if args.packet_space_loss_weight == 0.0:
                args.packet_space_loss_weight = 0.05
            if args.highfreq_core_loss_weight == 0.0:
                args.highfreq_core_loss_weight = 1.00
            if args.metadata_flow_loss_weight == 0.0:
                args.metadata_flow_loss_weight = 2.00
            if args.core_warmup_epochs == 0:
                args.core_warmup_epochs = max(2, args.epochs // 2)
            if args.freeze_refinement_epochs == 0:
                args.freeze_refinement_epochs = max(2, args.epochs // 2)
            if args.skip_lowpass_cutoff == 0.0:
                args.skip_lowpass_cutoff = 0.16
            if args.transported_synthesis_scale == 0.0:
                args.transported_synthesis_scale = 1.0
            if args.transported_input_scale == 0.0:
                args.transported_input_scale = 1.0
            if args.token_refine_scale == 1.0:
                args.token_refine_scale = 0.0
    if args.campaign == "helmholtz_branched_highk":
        if args.plus_width == 64:
            args.plus_width = 48
        if args.plus_depth == 6:
            args.plus_depth = 4
        if args.plus_stride == 8:
            args.plus_stride = 16
        if args.plus_max_modes == 16:
            args.plus_max_modes = 8
        if args.plus_transport_stencil == 12:
            args.plus_transport_stencil = 6
        if args.transport_parameterization == "mlp_displacement" and not args.preserve_transport_parameterization:
            args.transport_parameterization = "hamiltonian_verlet"
        if args.frame_type == "local_fft":
            args.frame_type = "directional_gabor"
        if args.branch_routing == "metadata_softmax" and args.branch_prior_strength == 0.0:
            args.branch_routing = "metadata_frequency_softmax"
            args.branch_prior_strength = 1.5
        if args.symbol_parameterization == "mlp":
            args.symbol_parameterization = "spectral_order"
        args.sparse_topk = True
        args.plus_low_frequency_scale = 0.0
        if args.plus_local_refine_scale == 1.0:
            args.plus_local_refine_scale = 0.05
        if args.plus_route_bias_init == -1.5:
            args.plus_route_bias_init = -5.0
        if args.plus_wavefront_confidence_scale == 0.0:
            args.plus_wavefront_confidence_scale = 2.0
        if args.plus_refine_lowpass_cutoff == 0.0:
            args.plus_refine_lowpass_cutoff = 0.16
        if args.transport_highpass_cutoff == 0.0:
            args.transport_highpass_cutoff = 0.16
        if args.skip_lowpass_cutoff == 0.0:
            args.skip_lowpass_cutoff = 0.16
        if args.transported_synthesis_scale == 0.0:
            args.transported_synthesis_scale = 1.0
        if args.transported_input_scale == 0.0:
            args.transported_input_scale = 1.0
        if args.transported_decoder_channels == 0:
            args.transported_decoder_channels = 16
        if args.transported_decoder_scale == 0.0:
            args.transported_decoder_scale = 1.0
        if args.token_refine_scale == 1.0:
            args.token_refine_scale = 0.0
        if args.num_canonical_branches == 1:
            args.num_canonical_branches = 3
        if args.branch_entropy_weight == 0.0:
            args.branch_entropy_weight = 0.01
        if args.branch_diversity_weight == 0.0:
            args.branch_diversity_weight = 0.01
        if args.core_field_weight == 0.0:
            args.core_field_weight = 2.0
        if args.residual_energy_weight == 0.0:
            args.residual_energy_weight = 0.50
        if args.route_l1_weight == 0.0:
            args.route_l1_weight = 0.05
        if args.canonical_loss_weight == 0.0:
            args.canonical_loss_weight = 0.02
        if args.symbol_order_loss_weight == 0.0:
            args.symbol_order_loss_weight = 0.01
        if args.symbol_order_target == 0.0:
            args.symbol_order_target = args.plus_symbol_order
        if args.packet_space_loss_weight == 0.0:
            args.packet_space_loss_weight = 0.05
        if args.highfreq_core_loss_weight == 0.0:
            args.highfreq_core_loss_weight = 1.00
        if args.highfreq_cutoff == 0.0:
            args.highfreq_cutoff = args.transport_highpass_cutoff
        if args.core_warmup_epochs == 0:
            args.core_warmup_epochs = max(2, args.epochs // 2)
        if args.freeze_refinement_epochs == 0:
            args.freeze_refinement_epochs = max(2, args.epochs // 2)
    if args.campaign in {"helmholtz_highk_careful", "helmholtz_highk_flagship", "helmholtz_highk_8gb"}:
        # This campaign is intentionally separate from the earlier sample-capped
        # branched diagnostic.  It keeps more packet capacity and isolates the
        # finite landing/carrier paths so high-k failures can be attributed.
        # The retained-edge local packet kernel is not auto-enabled: reduced
        # runs show it changes phase-sensitive behavior more reliably than
        # relative L2, so it is a controlled refinement/ablation knob rather
        # than the default high-k field-error path.
        if args.plus_width == 64:
            args.plus_width = 72
        if args.transport_parameterization == "mlp_displacement" and not args.preserve_transport_parameterization:
            args.transport_parameterization = "hamiltonian_verlet"
        if args.frame_type == "local_fft":
            args.frame_type = "directional_gabor"
        if args.branch_routing == "metadata_softmax" and args.branch_prior_strength == 0.0:
            args.branch_routing = "metadata_frequency_softmax"
            args.branch_prior_strength = 1.5
        if args.symbol_parameterization == "mlp":
            args.symbol_parameterization = "helmholtz_resolvent"
        if args.helmholtz_absorption == 0.05:
            args.helmholtz_absorption = 0.08
        if args.helmholtz_resolvent_cap == 6.0:
            args.helmholtz_resolvent_cap = 8.0
        if args.helmholtz_residual_loss_weight == 0.0:
            args.helmholtz_residual_loss_weight = 0.01
        args.sparse_topk = True
        args.plus_low_frequency_scale = 0.0
        if args.plus_local_refine_scale == 1.0:
            args.plus_local_refine_scale = 0.03
        if args.plus_route_bias_init == -1.5:
            args.plus_route_bias_init = -5.0
        if args.field_corrector == "none":
            args.field_corrector = "hybrid"
        if args.field_corrector_scale == 0.0:
            args.field_corrector_scale = 1.0
        if args.field_corrector_width == 32:
            args.field_corrector_width = 48
        if args.learning_rate == 1e-4:
            args.learning_rate = 2e-4
        if args.plus_wavefront_confidence_scale == 0.0:
            args.plus_wavefront_confidence_scale = 2.5
        if args.plus_refine_lowpass_cutoff == 0.0:
            args.plus_refine_lowpass_cutoff = 0.12
        if args.transport_highpass_cutoff == 0.0:
            args.transport_highpass_cutoff = 0.12
        if args.skip_lowpass_cutoff == 0.0:
            args.skip_lowpass_cutoff = 0.12
        if args.transported_synthesis_scale == 0.0:
            args.transported_synthesis_scale = 1.0
        if args.transported_input_scale == 0.0:
            args.transported_input_scale = 1.0
        if args.transported_decoder_channels == 0:
            args.transported_decoder_channels = 24
        if args.transported_decoder_scale == 0.0:
            args.transported_decoder_scale = 1.0
        if args.token_refine_scale == 1.0:
            args.token_refine_scale = 0.0
        if args.num_canonical_branches == 1:
            args.num_canonical_branches = 4
        if args.branch_entropy_weight == 0.0:
            args.branch_entropy_weight = 0.03
        if args.branch_diversity_weight == 0.0:
            args.branch_diversity_weight = 0.05
        if args.core_field_weight == 0.0:
            args.core_field_weight = 2.0
        if args.residual_energy_weight == 0.0:
            args.residual_energy_weight = 0.35
        if args.route_l1_weight == 0.0:
            args.route_l1_weight = 0.05
        if args.canonical_loss_weight == 0.0:
            args.canonical_loss_weight = 0.02
        if args.symbol_order_loss_weight == 0.0:
            args.symbol_order_loss_weight = 0.01
        if args.symbol_order_target == 0.0:
            args.symbol_order_target = args.plus_symbol_order
        if args.symbol_seminorm_loss_weight == 0.0:
            args.symbol_seminorm_loss_weight = 0.001
        if args.symbol_identity_loss_weight == 0.0:
            args.symbol_identity_loss_weight = 0.001
        if args.packet_space_loss_weight == 0.0:
            args.packet_space_loss_weight = 0.05
        if args.highfreq_core_loss_weight == 0.0:
            args.highfreq_core_loss_weight = 1.50
        if args.highfreq_cutoff == 0.0:
            args.highfreq_cutoff = args.transport_highpass_cutoff
        if args.core_warmup_epochs == 0:
            args.core_warmup_epochs = max(3, args.epochs // 2)
        if args.freeze_refinement_epochs == 0:
            args.freeze_refinement_epochs = max(3, args.epochs // 2)
        if args.campaign == "helmholtz_highk_flagship":
            if args.helmholtz_profile == "default":
                args.helmholtz_profile = "flagship"
            if args.helmholtz_residual_loss_weight == 0.01:
                args.helmholtz_residual_loss_weight = 0.02
            if args.complex_pair_loss_weight == 0.0:
                args.complex_pair_loss_weight = 0.05
            if args.complex_phase_loss_weight == 0.0:
                args.complex_phase_loss_weight = 0.02
            if args.symbol_identity_loss_weight == 0.001:
                args.symbol_identity_loss_weight = 0.002
            if args.plus_width == 72:
                args.plus_width = 96
            if args.plus_depth == 6:
                args.plus_depth = 4
            if args.plus_max_modes == 16:
                args.plus_max_modes = 12
            if args.plus_transport_stencil == 12:
                args.plus_transport_stencil = 6
            if args.transported_decoder_channels == 24:
                args.transported_decoder_channels = 16
            if args.branch_entropy_weight == 0.03:
                args.branch_entropy_weight = 0.04
            if args.branch_diversity_weight == 0.05:
                args.branch_diversity_weight = 0.08
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 6
        if args.campaign == "helmholtz_highk_8gb":
            if args.helmholtz_profile == "default":
                args.helmholtz_profile = "competitive_8gb"
            if args.batch_size == 2:
                args.batch_size = 1
            if args.field_corrector_width == 48:
                args.field_corrector_width = 32
            if args.helmholtz_residual_loss_weight == 0.01:
                args.helmholtz_residual_loss_weight = 0.015
            if args.complex_pair_loss_weight == 0.0:
                args.complex_pair_loss_weight = 0.03
            if args.complex_phase_loss_weight == 0.0:
                args.complex_phase_loss_weight = 0.01
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 4
        if args.helmholtz_profile == "local_minimal":
            args.num_canonical_branches = 2
            args.plus_width = 32
            args.plus_depth = 1
            args.plus_max_modes = 4
            args.plus_transport_stencil = 2
            args.transported_decoder_channels = 4
            if args.epochs == CAMPAIGN_EPOCHS["helmholtz_highk_careful"]:
                args.epochs = 8
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 2
        elif args.helmholtz_profile == "local_midscale":
            args.num_canonical_branches = 2
            args.plus_width = 48
            args.plus_depth = 1
            args.plus_max_modes = 6
            args.plus_transport_stencil = 3
            args.transported_decoder_channels = 8
            if args.epochs == CAMPAIGN_EPOCHS["helmholtz_highk_careful"]:
                args.epochs = 16
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 4
        elif args.helmholtz_profile == "local_8gb":
            # Memory-safe high-k profile for 8GB consumer GPUs.  It preserves
            # the theorem-facing ingredients (directional packets, resolvent
            # symbol, carrier-bound synthesis, learned landing, and two
            # canonical branches) while avoiding the flagship OOM footprint.
            args.num_canonical_branches = 2
            args.plus_width = 48
            args.plus_depth = 2
            args.plus_max_modes = 6
            args.plus_transport_stencil = 3
            args.transported_decoder_channels = 8
            args.helmholtz_resolvent_cap = min(float(args.helmholtz_resolvent_cap), 8.0)
            args.helmholtz_absorption = max(float(args.helmholtz_absorption), 0.08)
            if args.epochs == CAMPAIGN_EPOCHS["helmholtz_highk_8gb"]:
                args.epochs = 24
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 4
        elif args.helmholtz_profile == "competitive_8gb":
            # Field-accuracy profile for testing whether the MiNO carrier-bound
            # core plus an identity-canonical spectral/U-shaped resolvent
            # corrector can become competitive with FNO/UNO-style Helmholtz
            # baselines.  This is a benchmark profile, not the clean mechanism
            # profile used for branch attribution.
            args.loss_configs = "field_only" if args.loss_configs == ",".join(CAMPAIGN_LOSS_CONFIGS[args.campaign]) else args.loss_configs
            args.num_canonical_branches = 1
            args.plus_width = 24
            args.plus_depth = 1
            args.plus_max_modes = 4
            args.plus_transport_stencil = 2
            args.transported_decoder_channels = 4
            args.plus_local_refine_scale = 0.0
            args.field_corrector = "hybrid"
            args.field_corrector_scale = max(float(args.field_corrector_scale), 1.0)
            args.field_corrector_width = max(int(args.field_corrector_width), 64)
            if args.field_corrector_input_mode == "input_core_carrier":
                args.field_corrector_input_mode = "input_only"
            args.core_field_weight = 0.0
            args.residual_energy_weight = 0.0
            args.route_l1_weight = 0.0
            args.canonical_loss_weight = 0.0
            args.symbol_order_loss_weight = 0.0
            args.symbol_seminorm_loss_weight = 0.0
            args.symbol_identity_loss_weight = 0.0
            args.packet_space_loss_weight = 0.0
            args.highfreq_core_loss_weight = 0.0
            args.core_warmup_epochs = 0
            args.freeze_refinement_epochs = 0
            args.learning_rate = max(float(args.learning_rate), 3e-4)
            if args.epochs == CAMPAIGN_EPOCHS["helmholtz_highk_8gb"]:
                args.epochs = 24
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 4
        elif args.helmholtz_profile == "paper_full":
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 4
        elif args.helmholtz_profile == "flagship":
            # Registered high-k protocol for future execution: stronger
            # anisotropic/resolvent capacity, OOD k rows, and explicit
            # phase/radiation losses when real/imaginary target channels exist.
            args.num_canonical_branches = max(args.num_canonical_branches, 4)
            args.plus_width = max(args.plus_width, 96)
            args.plus_depth = max(args.plus_depth, 4)
            args.plus_max_modes = max(args.plus_max_modes, 12)
            args.plus_transport_stencil = max(args.plus_transport_stencil, 6)
            args.transported_decoder_channels = max(args.transported_decoder_channels, 16)
            args.helmholtz_resolvent_cap = max(float(args.helmholtz_resolvent_cap), 10.0)
            args.helmholtz_absorption = max(float(args.helmholtz_absorption), 0.10)
            args.helmholtz_residual_loss_weight = max(float(args.helmholtz_residual_loss_weight), 0.02)
            args.complex_pair_loss_weight = max(float(args.complex_pair_loss_weight), 0.05)
            args.complex_phase_loss_weight = max(float(args.complex_phase_loss_weight), 0.02)
            if args.progress_every_epochs == 0:
                args.progress_every_epochs = 6
        if args.field_corrector != "none" and args.field_corrector_scale > 0.0:
            # The field corrector is the high-k identity-canonical/resolvent
            # path.  The old mechanism profile warmed up only the core for half
            # the run, which prevented this path from learning soon enough to
            # compete with FNO/UNO-style field solvers.
            args.core_warmup_epochs = min(args.core_warmup_epochs, max(1, args.epochs // 6))
    if args.campaign not in {
        "ablation",
        "ablation_core",
        "transport_id",
        "calculus_id",
        "branch_id",
        "branch_id_v2",
        "branch_id_v3",
        "branch_id_v3_controls",
        "helmholtz_branched_highk",
        "helmholtz_highk_careful",
        "helmholtz_highk_flagship",
        "helmholtz_highk_8gb",
        "cross_resolution_wave",
    }:
        args.ablations = "full"
    if args.campaign == "cross_resolution_wave":
        args.ablations = "full"
        if args.transport_parameterization == "mlp_displacement" and not args.preserve_transport_parameterization:
            args.transport_parameterization = "hamiltonian_verlet"
        if args.frame_type == "local_fft":
            args.frame_type = "gabor_gaussian"
        if args.symbol_parameterization == "mlp":
            args.symbol_parameterization = "spectral_order"
        args.sparse_topk = True
        if args.canonical_loss_weight == 0.0:
            args.canonical_loss_weight = 0.01
        if args.plus_wavefront_confidence_scale == 0.0:
            args.plus_wavefront_confidence_scale = 1.0
        if args.symbol_order_loss_weight == 0.0:
            args.symbol_order_loss_weight = 0.005
        if args.symbol_order_target == 0.0:
            args.symbol_order_target = args.plus_symbol_order
        if args.plus_refine_lowpass_cutoff == 0.0:
            args.plus_refine_lowpass_cutoff = 0.25
        if args.transport_highpass_cutoff == 0.0:
            args.transport_highpass_cutoff = 0.25
        if args.packet_space_loss_weight == 0.0:
            args.packet_space_loss_weight = 0.01
        if args.highfreq_core_loss_weight == 0.0:
            args.highfreq_core_loss_weight = 0.05
        if args.highfreq_cutoff == 0.0:
            args.highfreq_cutoff = args.transport_highpass_cutoff
        if args.core_warmup_epochs == 0:
            args.core_warmup_epochs = max(1, args.epochs // 4)
        if args.freeze_refinement_epochs == 0:
            args.freeze_refinement_epochs = max(1, args.epochs // 4)
    if args.campaign == "smoke":
        if args.max_train_samples < 0:
            args.max_train_samples = 2
        if args.max_val_samples < 0:
            args.max_val_samples = 1
        if args.max_test_samples < 0:
            args.max_test_samples = 1
    else:
        if args.max_train_samples < 0:
            args.max_train_samples = 0
        if args.max_val_samples < 0:
            args.max_val_samples = 0
        if args.max_test_samples < 0:
            args.max_test_samples = 0
    if args.campaign == "helmholtz_highk_careful":
        if args.helmholtz_profile == "local_minimal":
            if args.max_train_samples == 0:
                args.max_train_samples = 4
            if args.max_val_samples == 0:
                args.max_val_samples = 2
            if args.max_test_samples == 0:
                args.max_test_samples = 2
        elif args.helmholtz_profile == "local_midscale":
            if args.max_train_samples == 0:
                args.max_train_samples = 16
            if args.max_val_samples == 0:
                args.max_val_samples = 4
            if args.max_test_samples == 0:
                args.max_test_samples = 4
    if args.campaign == "helmholtz_highk_8gb":
        if args.max_train_samples == 0:
            args.max_train_samples = 12
        if args.max_val_samples == 0:
            args.max_val_samples = 4
        if args.max_test_samples == 0:
            args.max_test_samples = 4
    if args.output == "":
        args.output = str(Path("generated") / "empirical_closure" / args.campaign)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiNO empirical-closure campaigns.")
    parser.add_argument(
        "--campaign",
        choices=[
            "smoke",
            "proxy_sweep",
            "ablation",
            "ablation_core",
            "transport_id",
            "calculus_id",
            "branch_id",
            "branch_id_v2",
            "branch_id_v3",
            "branch_id_v3_controls",
            "helmholtz_branched_highk",
            "helmholtz_highk_careful",
            "helmholtz_highk_flagship",
            "helmholtz_highk_8gb",
            "cross_resolution_wave",
            "stage3",
        ],
        required=True,
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-val-samples", type=int, default=-1)
    parser.add_argument("--max-test-samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hardware-profile", default="local_smoke", choices=["local_smoke", "paper_v100", "paper_4090", "paper_a100"])
    parser.add_argument(
        "--helmholtz-profile",
        default="default",
        choices=["default", "local_minimal", "local_midscale", "local_8gb", "competitive_8gb", "paper_full", "flagship"],
        help="Preset capacity/sample profile for Helmholtz campaigns; flagship is the high-k OOD protocol and is not run by default.",
    )
    parser.add_argument("--progress-every-epochs", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-configs", default="")
    parser.add_argument("--ablations", default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--tcno-cache-root", default="")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--proxy-temperature", type=float, default=0.05)
    parser.add_argument("--core-field-weight", type=float, default=0.0)
    parser.add_argument("--residual-energy-weight", type=float, default=0.0)
    parser.add_argument("--route-l1-weight", type=float, default=0.0)
    parser.add_argument("--canonical-loss-weight", type=float, default=0.0)
    parser.add_argument("--symbol-order-loss-weight", type=float, default=0.0)
    parser.add_argument("--symbol-order-target", type=float, default=0.0)
    parser.add_argument("--symbol-seminorm-loss-weight", type=float, default=0.0)
    parser.add_argument("--symbol-seminorm-target", type=float, default=0.0)
    parser.add_argument("--packet-space-loss-weight", type=float, default=0.0)
    parser.add_argument("--highfreq-core-loss-weight", type=float, default=0.0)
    parser.add_argument("--highfreq-cutoff", type=float, default=0.0)
    parser.add_argument("--helmholtz-residual-loss-weight", type=float, default=0.0)
    parser.add_argument("--helmholtz-residual-wavenumber", type=float, default=12.0)
    parser.add_argument(
        "--complex-pair-loss-weight",
        type=float,
        default=0.0,
        help="Relative loss on explicit real/imaginary output channel pairs. Inert for scalar real-valued datasets.",
    )
    parser.add_argument(
        "--complex-phase-loss-weight",
        type=float,
        default=0.0,
        help="Complex coherence/phase loss on explicit real/imaginary output channel pairs. Inert for scalar real-valued datasets.",
    )
    parser.add_argument(
        "--symbol-identity-loss-weight",
        type=float,
        default=0.0,
        help="Penalize non-identity local tube and retained-edge symbol corrections; useful for separating field-error gains from phase/radiation symbol diagnostics.",
    )
    parser.add_argument("--metadata-flow-loss-weight", type=float, default=0.0)
    parser.add_argument("--transport-highpass-cutoff", type=float, default=0.0)
    parser.add_argument("--core-warmup-epochs", type=int, default=0)
    parser.add_argument("--freeze-refinement-epochs", type=int, default=0)
    parser.add_argument(
        "--transport-parameterization",
        default="mlp_displacement",
        choices=["mlp_displacement", "hamiltonian_euler", "hamiltonian_verlet"],
    )
    parser.add_argument(
        "--preserve-transport-parameterization",
        action="store_true",
        help="Do not replace mlp_displacement with theorem-facing Verlet in theorem-aligned campaigns; intended for transport-parameterization probes.",
    )
    parser.add_argument(
        "--symbol-parameterization",
        default="mlp",
        choices=["mlp", "spectral_order", "helmholtz_resolvent"],
    )
    parser.add_argument("--helmholtz-shell-radius", type=float, default=0.0)
    parser.add_argument("--helmholtz-refractive-index", type=float, default=1.0)
    parser.add_argument("--helmholtz-absorption", type=float, default=0.05)
    parser.add_argument("--helmholtz-resolvent-cap", type=float, default=6.0)
    parser.add_argument(
        "--frame-type",
        default="local_fft",
        choices=["local_fft", "gabor_gaussian", "multiscale_gabor", "anisotropic_gabor", "directional_gabor"],
    )
    parser.add_argument("--sparse-topk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plus-width", type=int, default=64)
    parser.add_argument("--plus-depth", type=int, default=6)
    parser.add_argument("--plus-patch-size", type=int, default=16)
    parser.add_argument("--plus-stride", type=int, default=8)
    parser.add_argument("--plus-max-modes", type=int, default=16)
    parser.add_argument("--plus-window-type", default="gaussian", choices=["hann", "boxcar", "gaussian"])
    parser.add_argument("--plus-mode-strategy", default="shell_balanced", choices=["radial", "shell_balanced"])
    parser.add_argument("--plus-low-frequency-scale", type=float, default=0.05)
    parser.add_argument("--plus-transport-scale", type=float, default=0.03)
    parser.add_argument("--plus-transport-stencil", type=int, default=12)
    parser.add_argument("--plus-local-refine-channels", type=int, default=32)
    parser.add_argument("--plus-local-refine-scale", type=float, default=1.0)
    parser.add_argument("--plus-refine-lowpass-cutoff", type=float, default=0.0)
    parser.add_argument("--plus-route-bias-init", type=float, default=-1.5)
    parser.add_argument("--plus-pdo-symbol-scale", type=float, default=0.0)
    parser.add_argument("--plus-pdo-symbol-order", type=float, default=-2.0)
    parser.add_argument("--plus-symbol-order", type=float, default=0.0)
    parser.add_argument("--plus-wavefront-confidence-scale", type=float, default=0.0)
    parser.add_argument("--plus-dissipative-symbol-scale", type=float, default=0.0)
    parser.add_argument("--plus-dissipative-time-step", type=float, default=1.0)
    parser.add_argument("--plus-frame-patch-sizes", default="")
    parser.add_argument("--plus-frame-strides", default="")
    parser.add_argument("--plus-frame-max-modes", default="")
    parser.add_argument("--skip-lowpass-cutoff", type=float, default=0.0)
    parser.add_argument("--transported-synthesis-scale", type=float, default=0.0)
    parser.add_argument("--transported-input-scale", type=float, default=0.0)
    parser.add_argument("--transported-synthesis-mode", default="warp", choices=["warp", "atom_splat", "patch_fold"])
    parser.add_argument("--transported-decoder-channels", type=int, default=0)
    parser.add_argument("--transported-decoder-scale", type=float, default=0.0)
    parser.add_argument("--transported-decoder-transport-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-refine-scale", type=float, default=1.0)
    parser.add_argument("--num-canonical-branches", type=int, default=1)
    parser.add_argument(
        "--branch-routing",
        default="metadata_softmax",
        choices=["metadata_softmax", "metadata_frequency_softmax", "frequency_softmax", "uniform", "single"],
    )
    parser.add_argument(
        "--branch-prior-strength",
        type=float,
        default=0.0,
        help="Frequency-sector prior strength for metadata_frequency_softmax/frequency_softmax multibranch routing.",
    )
    parser.add_argument("--branch-entropy-weight", type=float, default=0.0)
    parser.add_argument("--branch-diversity-weight", type=float, default=0.0)
    parser.add_argument("--branch-synthesis", default="sum", choices=["sum"])
    parser.add_argument(
        "--edge-symbol-parameterization",
        default="none",
        choices=["none", "local_packet_kernel"],
        help="Apply a learned edge-local packet symbol on retained canonical stencil entries.",
    )
    parser.add_argument("--edge-symbol-strength", type=float, default=0.5)
    parser.add_argument(
        "--field-corrector",
        default="none",
        choices=["none", "spectral", "unet", "hybrid"],
        help=(
            "Optional field-level identity-canonical residual corrector. "
            "High-k Helmholtz profiles use this as a resolvent/stress-test path, "
            "not as part of the packet-transport mechanism theorem."
        ),
    )
    parser.add_argument("--field-corrector-scale", type=float, default=0.0)
    parser.add_argument("--field-corrector-width", type=int, default=32)
    parser.add_argument(
        "--field-corrector-input-mode",
        default="input_core_carrier",
        choices=["input_only", "input_core", "input_core_carrier"],
        help="Inputs supplied to the optional field corrector.",
    )
    return resolve_campaign_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    runtime_flags = configure_torch_runtime(device)
    scenarios = parse_csv(args.scenarios)
    seeds = parse_seeds(args.seeds)
    loss_configs = parse_csv(args.loss_configs)
    ablations = parse_csv(args.ablations)

    for loss_config in loss_configs:
        if loss_config not in LOSS_CONFIGS:
            raise ValueError(f"Unknown loss config: {loss_config}")
    for scenario in scenarios:
        get_scenario_spec(scenario, tcno_cache_root=Path(args.tcno_cache_root) if args.tcno_cache_root else None)

    planned_rows: list[dict[str, object]] = []
    if args.campaign == "cross_resolution_wave":
        for scenario in scenarios:
            for train_resolution, test_resolution in CROSS_RESOLUTION_PAIRS:
                for packet_policy in CROSS_RESOLUTION_PACKET_POLICIES:
                    for stencil_policy in CROSS_RESOLUTION_STENCIL_POLICIES:
                        for seed in seeds:
                            planned_rows.append(
                                {
                                    "row_type": "plan",
                                    "campaign": args.campaign,
                                    "scenario": scenario,
                                    "loss_config": loss_configs[0],
                                    "ablation": "full",
                                    "seed": seed,
                                    "epochs": args.epochs,
                                    "batch_size": args.batch_size,
                                    "hardware_profile": args.hardware_profile,
                                    "train_resolution": train_resolution,
                                    "test_resolution": test_resolution,
                                    "packet_budget_policy": packet_policy,
                                    "stencil_policy": stencil_policy,
                                    "transport_parameterization": args.transport_parameterization,
                                    "frame_type": args.frame_type,
                                    "symbol_parameterization": args.symbol_parameterization,
                                    "symbol_order": args.plus_symbol_order,
                                    "helmholtz_shell_radius": args.helmholtz_shell_radius,
                                    "helmholtz_absorption": args.helmholtz_absorption,
                                    "helmholtz_resolvent_cap": args.helmholtz_resolvent_cap,
                                    "helmholtz_residual_loss_weight": args.helmholtz_residual_loss_weight,
                                    "helmholtz_residual_wavenumber": args.helmholtz_residual_wavenumber,
                                    "complex_pair_loss_weight": args.complex_pair_loss_weight,
                                    "complex_phase_loss_weight": args.complex_phase_loss_weight,
                                    "wavefront_confidence_scale": args.plus_wavefront_confidence_scale,
                                    "refine_lowpass_cutoff": args.plus_refine_lowpass_cutoff,
                                    "sparse_topk": args.sparse_topk,
                                    "canonical_loss_weight": args.canonical_loss_weight,
                                    "symbol_order_loss_weight": args.symbol_order_loss_weight,
                                    "symbol_order_target": args.symbol_order_target,
                                    "symbol_seminorm_loss_weight": args.symbol_seminorm_loss_weight,
                                    "symbol_seminorm_target": args.symbol_seminorm_target,
                                    "symbol_identity_loss_weight": args.symbol_identity_loss_weight,
                                    "packet_space_loss_weight": args.packet_space_loss_weight,
                                    "highfreq_core_loss_weight": args.highfreq_core_loss_weight,
                                    "metadata_flow_loss_weight": args.metadata_flow_loss_weight,
                                    "effective_metadata_flow_loss_weight": args.metadata_flow_loss_weight,
                                    "highfreq_cutoff": args.highfreq_cutoff,
                                    "transport_highpass_cutoff": args.transport_highpass_cutoff,
                                    "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
                                    "transported_synthesis_scale": args.transported_synthesis_scale,
                                    "transported_input_scale": args.transported_input_scale,
                                    "transported_decoder_channels": args.transported_decoder_channels,
                                    "transported_decoder_scale": args.transported_decoder_scale,
                                    "token_refine_scale": args.token_refine_scale,
                                    "num_canonical_branches": args.num_canonical_branches,
                                    "branch_routing": args.branch_routing,
                                    "branch_prior_strength": args.branch_prior_strength,
                                    "branch_entropy_weight": args.branch_entropy_weight,
                                    "branch_diversity_weight": args.branch_diversity_weight,
                                    "edge_symbol_parameterization": args.edge_symbol_parameterization,
                                    "edge_symbol_strength": args.edge_symbol_strength,
                                    "field_corrector": args.field_corrector,
                                    "field_corrector_scale": args.field_corrector_scale,
                                    "field_corrector_width": args.field_corrector_width,
                                    "field_corrector_input_mode": args.field_corrector_input_mode,
                                    "plus_width": args.plus_width,
                                    "plus_depth": args.plus_depth,
                                    "plus_transport_stencil": args.plus_transport_stencil,
                                    "plus_max_modes": args.plus_max_modes,
                                    "plus_patch_size": args.plus_patch_size,
                                    "plus_stride": args.plus_stride,
                                    "core_warmup_epochs": args.core_warmup_epochs,
                                    "freeze_refinement_epochs": args.freeze_refinement_epochs,
                                }
                            )
    else:
        for scenario in scenarios:
            for loss_config in loss_configs:
                for ablation in ablations:
                    for seed in seeds:
                        planned_rows.append(
                            {
                                "row_type": "plan",
                                "campaign": args.campaign,
                                "scenario": scenario,
                                "loss_config": loss_config,
                                "ablation": ablation,
                                "seed": seed,
                                "epochs": args.epochs,
                                "batch_size": args.batch_size,
                                "hardware_profile": args.hardware_profile,
                                "transport_parameterization": args.transport_parameterization,
                                "frame_type": args.frame_type,
                                "symbol_parameterization": args.symbol_parameterization,
                                "symbol_order": args.plus_symbol_order,
                                "helmholtz_shell_radius": args.helmholtz_shell_radius,
                                "helmholtz_absorption": args.helmholtz_absorption,
                                "helmholtz_resolvent_cap": args.helmholtz_resolvent_cap,
                                "helmholtz_residual_loss_weight": args.helmholtz_residual_loss_weight,
                                "helmholtz_residual_wavenumber": args.helmholtz_residual_wavenumber,
                                "complex_pair_loss_weight": args.complex_pair_loss_weight,
                                "complex_phase_loss_weight": args.complex_phase_loss_weight,
                                "wavefront_confidence_scale": args.plus_wavefront_confidence_scale,
                                "refine_lowpass_cutoff": args.plus_refine_lowpass_cutoff,
                                "sparse_topk": args.sparse_topk,
                                "canonical_loss_weight": args.canonical_loss_weight,
                                "symbol_order_loss_weight": args.symbol_order_loss_weight,
                                "symbol_order_target": args.symbol_order_target,
                                "symbol_seminorm_loss_weight": args.symbol_seminorm_loss_weight,
                                "symbol_seminorm_target": args.symbol_seminorm_target,
                                "symbol_identity_loss_weight": args.symbol_identity_loss_weight,
                                "packet_space_loss_weight": args.packet_space_loss_weight,
                                "highfreq_core_loss_weight": args.highfreq_core_loss_weight,
                                "metadata_flow_loss_weight": args.metadata_flow_loss_weight,
                                "effective_metadata_flow_loss_weight": effective_metadata_flow_loss_weight(args, ablation),
                                "highfreq_cutoff": args.highfreq_cutoff,
                                "transport_highpass_cutoff": args.transport_highpass_cutoff,
                                "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
                                "transported_synthesis_scale": args.transported_synthesis_scale,
                                "transported_input_scale": args.transported_input_scale,
                                "transported_decoder_channels": args.transported_decoder_channels,
                                "transported_decoder_scale": args.transported_decoder_scale,
                                "token_refine_scale": args.token_refine_scale,
                                "num_canonical_branches": args.num_canonical_branches,
                                "branch_routing": args.branch_routing,
                                "branch_prior_strength": args.branch_prior_strength,
                                "branch_entropy_weight": args.branch_entropy_weight,
                                "branch_diversity_weight": args.branch_diversity_weight,
                                "edge_symbol_parameterization": args.edge_symbol_parameterization,
                                "edge_symbol_strength": args.edge_symbol_strength,
                                "field_corrector": args.field_corrector,
                                "field_corrector_scale": args.field_corrector_scale,
                                "field_corrector_width": args.field_corrector_width,
                                "field_corrector_input_mode": args.field_corrector_input_mode,
                                "plus_width": args.plus_width,
                                "plus_depth": args.plus_depth,
                                "plus_transport_stencil": args.plus_transport_stencil,
                                "plus_max_modes": args.plus_max_modes,
                                "plus_patch_size": args.plus_patch_size,
                                "plus_stride": args.plus_stride,
                                "core_warmup_epochs": args.core_warmup_epochs,
                                "freeze_refinement_epochs": args.freeze_refinement_epochs,
                            }
                        )
    if args.resume_from > 0:
        planned_rows = planned_rows[args.resume_from :]
    if args.max_rows > 0:
        planned_rows = planned_rows[: args.max_rows]
    if args.dry_run:
        write_csv(output_dir / "empirical_closure_plan.csv", planned_rows)
        write_json(
            output_dir / "manifest.json",
            {
                "campaign": args.campaign,
                "scenarios": scenarios,
                "seeds": seeds,
                "loss_configs": loss_configs,
                "ablations": ablations,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hardware_profile": args.hardware_profile,
                "transport_parameterization": args.transport_parameterization,
                "frame_type": args.frame_type,
                "symbol_parameterization": args.symbol_parameterization,
                "helmholtz_shell_radius": args.helmholtz_shell_radius,
                "helmholtz_refractive_index": args.helmholtz_refractive_index,
                "helmholtz_absorption": args.helmholtz_absorption,
                "helmholtz_resolvent_cap": args.helmholtz_resolvent_cap,
                "helmholtz_residual_loss_weight": args.helmholtz_residual_loss_weight,
                "helmholtz_residual_wavenumber": args.helmholtz_residual_wavenumber,
                "complex_pair_loss_weight": args.complex_pair_loss_weight,
                "complex_phase_loss_weight": args.complex_phase_loss_weight,
                "sparse_topk": args.sparse_topk,
                "canonical_loss_weight": args.canonical_loss_weight,
                "symbol_order_loss_weight": args.symbol_order_loss_weight,
                "symbol_order_target": args.symbol_order_target,
                "symbol_seminorm_loss_weight": args.symbol_seminorm_loss_weight,
                "symbol_seminorm_target": args.symbol_seminorm_target,
                "symbol_identity_loss_weight": args.symbol_identity_loss_weight,
                "packet_space_loss_weight": args.packet_space_loss_weight,
                "highfreq_core_loss_weight": args.highfreq_core_loss_weight,
                "metadata_flow_loss_weight": args.metadata_flow_loss_weight,
                "highfreq_cutoff": args.highfreq_cutoff,
                "wavefront_confidence_scale": args.plus_wavefront_confidence_scale,
                "refine_lowpass_cutoff": args.plus_refine_lowpass_cutoff,
                "transport_highpass_cutoff": args.transport_highpass_cutoff,
                "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
                "transported_synthesis_scale": args.transported_synthesis_scale,
                "transported_input_scale": args.transported_input_scale,
                "transported_decoder_channels": args.transported_decoder_channels,
                "transported_decoder_scale": args.transported_decoder_scale,
                "token_refine_scale": args.token_refine_scale,
                "num_canonical_branches": args.num_canonical_branches,
                "branch_routing": args.branch_routing,
                "branch_prior_strength": args.branch_prior_strength,
                "branch_entropy_weight": args.branch_entropy_weight,
                "branch_diversity_weight": args.branch_diversity_weight,
                "edge_symbol_parameterization": args.edge_symbol_parameterization,
                "edge_symbol_strength": args.edge_symbol_strength,
                "field_corrector": args.field_corrector,
                "field_corrector_scale": args.field_corrector_scale,
                "field_corrector_width": args.field_corrector_width,
                "field_corrector_input_mode": args.field_corrector_input_mode,
                "plus_width": args.plus_width,
                "plus_depth": args.plus_depth,
                "plus_transport_stencil": args.plus_transport_stencil,
                "plus_max_modes": args.plus_max_modes,
                "plus_patch_size": args.plus_patch_size,
                "plus_stride": args.plus_stride,
                "core_warmup_epochs": args.core_warmup_epochs,
                "freeze_refinement_epochs": args.freeze_refinement_epochs,
                "runtime_flags": runtime_flags,
                "dry_run": True,
                "planned_rows": len(planned_rows),
                "output_dir": str(output_dir),
            },
        )
        print(f"[dry-run] wrote {output_dir / 'empirical_closure_plan.csv'}")
        print(f"[dry-run] planned_rows={len(planned_rows)}")
        return
    if args.campaign == "cross_resolution_wave":
        raise RuntimeError("cross_resolution_wave is a dry-run planning campaign in this batch; use --dry-run.")

    rows: list[dict[str, object]] = []
    start = perf_counter()
    for run_stub in planned_rows:
        scenario = str(run_stub["scenario"])
        loss_config = str(run_stub["loss_config"])
        ablation = str(run_stub["ablation"])
        seed = int(run_stub["seed"])
        run_id = make_run_id(run_stub)
        result_path = output_dir / f"{run_id}.json"
        if args.skip_existing and result_path.exists():
            print(f"[skip] {run_id}", flush=True)
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        unsupported_reason = unsupported_ablation_reason(ablation, scenario)
        if unsupported_reason is not None:
            row = {
                **run_stub,
                "row_type": "skipped",
                "run_id": run_id,
                "skip_reason": unsupported_reason,
            }
            write_json(result_path, row)
            rows.append(row)
            print(f"[skip-unsupported] {run_id}: {unsupported_reason}", flush=True)
            write_csv(output_dir / "empirical_closure_results.csv", rows)
            summary = aggregate_rows(rows)
            write_csv(output_dir / "empirical_closure_summary.csv", summary)
            write_csv(output_dir / "empirical_closure_proxy_deltas.csv", make_proxy_deltas(summary))
            write_csv(output_dir / "empirical_closure_ablation_deltas.csv", make_ablation_deltas(summary))
            continue

        def _run(batch_size: int, oom_fallback: bool) -> dict[str, object]:
            print(
                f"[run] campaign={args.campaign} scenario={scenario} "
                f"loss={loss_config} ablation={ablation} seed={seed} "
                f"epochs={args.epochs} batch={batch_size}",
                flush=True,
            )
            return run_training_once(
                scenario=scenario,
                seed=seed,
                batch_size=batch_size,
                loss_config=loss_config,
                ablation=ablation,
                device=device,
                args=args,
                oom_fallback=oom_fallback,
            )

        row = run_with_oom_fallback(_run, args.batch_size)
        write_json(result_path, row)
        rows.append(row)
        print(
            f"[done] {run_id} rel_l2={row['test_relative_l2']:.6g} "
            f"phase={row['test_phase_error']:.6g} runtime={row['runtime_seconds']:.1f}s",
            flush=True,
        )

        write_csv(output_dir / "empirical_closure_results.csv", rows)
        summary = aggregate_rows(rows)
        write_csv(output_dir / "empirical_closure_summary.csv", summary)
        write_csv(output_dir / "empirical_closure_proxy_deltas.csv", make_proxy_deltas(summary))
        write_csv(output_dir / "empirical_closure_ablation_deltas.csv", make_ablation_deltas(summary))

    summary = aggregate_rows(rows)
    proxy_deltas = make_proxy_deltas(summary)
    ablation_deltas = make_ablation_deltas(summary)
    write_csv(output_dir / "empirical_closure_results.csv", rows)
    write_csv(output_dir / "empirical_closure_summary.csv", summary)
    write_csv(output_dir / "empirical_closure_proxy_deltas.csv", proxy_deltas)
    write_csv(output_dir / "empirical_closure_ablation_deltas.csv", ablation_deltas)
    manifest = {
        "campaign": args.campaign,
        "scenarios": scenarios,
        "seeds": seeds,
        "loss_configs": loss_configs,
        "ablations": ablations,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hardware_profile": args.hardware_profile,
        "helmholtz_profile": args.helmholtz_profile,
        "transport_parameterization": args.transport_parameterization,
        "frame_type": args.frame_type,
        "symbol_parameterization": args.symbol_parameterization,
        "helmholtz_shell_radius": args.helmholtz_shell_radius,
        "helmholtz_refractive_index": args.helmholtz_refractive_index,
        "helmholtz_absorption": args.helmholtz_absorption,
        "helmholtz_resolvent_cap": args.helmholtz_resolvent_cap,
        "helmholtz_residual_loss_weight": args.helmholtz_residual_loss_weight,
        "helmholtz_residual_wavenumber": args.helmholtz_residual_wavenumber,
        "complex_pair_loss_weight": args.complex_pair_loss_weight,
        "complex_phase_loss_weight": args.complex_phase_loss_weight,
        "sparse_topk": args.sparse_topk,
        "canonical_loss_weight": args.canonical_loss_weight,
        "symbol_order_loss_weight": args.symbol_order_loss_weight,
        "symbol_order_target": args.symbol_order_target,
        "symbol_seminorm_loss_weight": args.symbol_seminorm_loss_weight,
        "symbol_seminorm_target": args.symbol_seminorm_target,
        "packet_space_loss_weight": args.packet_space_loss_weight,
        "highfreq_core_loss_weight": args.highfreq_core_loss_weight,
        "metadata_flow_loss_weight": args.metadata_flow_loss_weight,
        "highfreq_cutoff": args.highfreq_cutoff,
        "wavefront_confidence_scale": args.plus_wavefront_confidence_scale,
        "refine_lowpass_cutoff": args.plus_refine_lowpass_cutoff,
        "transport_highpass_cutoff": args.transport_highpass_cutoff,
        "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
        "transported_synthesis_scale": args.transported_synthesis_scale,
        "transported_input_scale": args.transported_input_scale,
        "transported_decoder_channels": args.transported_decoder_channels,
        "transported_decoder_scale": args.transported_decoder_scale,
        "token_refine_scale": args.token_refine_scale,
        "num_canonical_branches": args.num_canonical_branches,
        "branch_routing": args.branch_routing,
        "branch_prior_strength": args.branch_prior_strength,
        "branch_entropy_weight": args.branch_entropy_weight,
        "branch_diversity_weight": args.branch_diversity_weight,
        "field_corrector": args.field_corrector,
        "field_corrector_scale": args.field_corrector_scale,
        "field_corrector_width": args.field_corrector_width,
        "field_corrector_input_mode": args.field_corrector_input_mode,
        "core_warmup_epochs": args.core_warmup_epochs,
        "freeze_refinement_epochs": args.freeze_refinement_epochs,
        "runtime_flags": runtime_flags,
        "device": str(device),
        "runtime_seconds": perf_counter() - start,
        "output_dir": str(output_dir),
        "rows": len(rows),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote {output_dir / 'empirical_closure_results.csv'}")
    print(f"Wrote {output_dir / 'empirical_closure_summary.csv'}")
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
