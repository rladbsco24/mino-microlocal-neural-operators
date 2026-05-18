from __future__ import annotations

import torch

from mino.models.mino import _finite_symbol_seminorm_proxy, _frequency_order_proxy, build_model
from mino.models.layers import (
    CanonicalPropagationLayer,
    ComplexPairSymbolAction,
    IdentityPseudodifferentialBranch,
    SymbolModulationLayer,
    _helmholtz_shell_features,
)
from mino.training.train import _complex_pair_phase_loss, _complex_pair_relative_loss, _helmholtz_residual_proxy


def test_all_models_preserve_shape() -> None:
    x = torch.randn(2, 1, 32, 32)
    for name in ["MiNO", "MiNO-Core", "MiNO-Plus", "FNOStyle", "WNOStyle", "PDNOStyle", "LocalKernel", "UNetStyle", "HybridSpectralUNet"]:
        model = build_model(name)
        y = model(x)
        assert y.shape == x.shape


def test_all_models_support_channel_change() -> None:
    x = torch.randn(2, 3, 32, 32)
    for name in ["MiNO", "MiNO-Core", "MiNO-Plus", "FNOStyle", "WNOStyle", "PDNOStyle", "LocalKernel", "UNetStyle", "HybridSpectralUNet"]:
        model = build_model(name, in_channels=3, out_channels=1)
        y = model(x)
        assert y.shape == (2, 1, 32, 32)


def test_mino_builder_uses_carrier_bound_defaults() -> None:
    core = build_model("MiNO", model_kwargs={"width": 12, "depth": 1, "max_modes": 6})
    plus = build_model("MiNO-Plus", model_kwargs={"width": 12, "depth": 1, "max_modes": 6})
    assert core.transported_input_scale > 0.0
    assert core.transported_synthesis_scale > 0.0
    assert plus.core.transported_input_scale > 0.0
    assert plus.core.transported_synthesis_scale > 0.0


def test_mino_plus_diagnostics_and_proxy_losses_are_finite() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "window_type": "gaussian",
            "mode_strategy": "shell_balanced",
            "transport_stencil": 4,
        },
    )
    x = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    losses = model.proxy_losses_from_diagnostics(diagnostics, target, proxy_temperature=0.1)
    assert diagnostics["prediction"].shape == x.shape
    assert torch.isfinite(losses["transport_proxy"])
    assert torch.isfinite(losses["symbol_proxy"])


def test_mino_plus_calculus_branches_are_finite_and_dissipative() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "window_type": "gaussian",
            "mode_strategy": "shell_balanced",
            "transport_stencil": 4,
            "pdo_symbol_scale": 0.1,
            "dissipative_symbol_scale": 0.1,
        },
    )
    x = torch.randn(2, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    block_diagnostics = diagnostics["block_diagnostics"]
    assert diagnostics["prediction"].shape == x.shape
    assert block_diagnostics
    for block in block_diagnostics:
        assert torch.isfinite(block["pdo_identity_norm"])
        assert torch.isfinite(block["dissipative_symbol_norm"])
        assert torch.isfinite(block["dissipative_multiplier_mean"])
        assert 0.0 < float(block["dissipative_multiplier_mean"].detach()) <= 1.0


def test_hamiltonian_transport_and_gabor_frame_are_finite() -> None:
    for parameterization in ("hamiltonian_euler", "hamiltonian_verlet"):
        model = build_model(
            "MiNO-Plus",
            model_kwargs={
                "frame_type": "gabor_gaussian",
            "transport_parameterization": parameterization,
            "sparse_topk": True,
            "transport_stencil": 4,
            "transport_highpass_cutoff": 0.25,
            "width": 16,
            "depth": 1,
            "max_modes": 8,
            },
        )
        x = torch.randn(1, 1, 32, 32)
        diagnostics = model.forward_with_diagnostics(x)
        assert diagnostics["prediction"].shape == x.shape
        assert torch.isfinite(diagnostics["tokenizer_reconstruction_error"])
        assert torch.isfinite(diagnostics["tokenizer_covering_radius"])
        block = diagnostics["block_diagnostics"][0]
        assert torch.isfinite(block["canonical_defect_proxy"])
        assert torch.isfinite(block["symbol_order_proxy"])
        assert torch.isfinite(diagnostics["core_high_frequency_norm"])
        assert torch.isfinite(diagnostics["refine_high_frequency_norm"])


def test_transported_input_carrier_diagnostics_are_finite() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "frame_type": "gabor_gaussian",
            "transport_parameterization": "hamiltonian_verlet",
            "transport_stencil": 4,
            "skip_lowpass_cutoff": 0.2,
            "transported_synthesis_scale": 1.0,
            "transported_input_scale": 1.0,
            "width": 16,
            "depth": 1,
            "max_modes": 8,
        },
    )
    x = torch.randn(1, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    assert diagnostics["prediction"].shape == x.shape
    assert torch.isfinite(diagnostics["transported_input_norm"])
    assert torch.isfinite(diagnostics["transported_input_shift_norm"])


def test_atom_splat_transported_synthesis_is_finite() -> None:
    for synthesis_mode in ("atom_splat", "patch_fold"):
        model = build_model(
            "MiNO-Plus",
            model_kwargs={
                "frame_type": "gabor_gaussian",
                "transport_parameterization": "hamiltonian_verlet",
                "transport_stencil": 4,
                "skip_lowpass_cutoff": 0.2,
                "transported_synthesis_scale": 1.0,
                "transported_input_scale": 1.0,
                "transported_synthesis_mode": synthesis_mode,
                "width": 12,
                "depth": 1,
                "max_modes": 6,
            },
        )
        x = torch.randn(1, 1, 32, 32)
        diagnostics = model.forward_with_diagnostics(x)
        assert diagnostics["prediction"].shape == x.shape
        assert torch.isfinite(diagnostics["transported_synthesis_shift_norm"])
        assert torch.isfinite(diagnostics["transported_input_norm"])


def test_learned_transported_landing_decoder_is_finite() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "frame_type": "gabor_gaussian",
            "transport_parameterization": "hamiltonian_verlet",
            "transport_stencil": 4,
            "skip_lowpass_cutoff": 0.2,
            "transported_synthesis_scale": 1.0,
            "transported_input_scale": 1.0,
            "transported_decoder_channels": 8,
            "transported_decoder_scale": 1.0,
            "width": 12,
            "depth": 1,
            "max_modes": 6,
        },
    )
    x = torch.randn(1, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    assert diagnostics["prediction"].shape == x.shape
    assert torch.isfinite(diagnostics["transported_landing_norm"])
    assert torch.isfinite(diagnostics["transported_landing_gate"])


def test_mino_plus_field_corrector_is_finite() -> None:
    x = torch.randn(1, 3, 32, 32)
    for input_mode in ("input_only", "input_core", "input_core_carrier"):
        model = build_model(
            "MiNO-Plus",
            in_channels=3,
            out_channels=1,
            model_kwargs={
                "field_corrector": "hybrid",
                "field_corrector_scale": 1.0,
                "field_corrector_width": 8,
                "field_corrector_input_mode": input_mode,
                "width": 12,
                "depth": 1,
                "max_modes": 6,
            },
        )
        diagnostics = model.forward_with_diagnostics(x)
        assert diagnostics["prediction"].shape == (1, 1, 32, 32)
        assert torch.isfinite(diagnostics["field_correction_norm"])


def test_learned_transported_landing_decoder_is_transport_gated() -> None:
    model = build_model(
        "MiNO-Core",
        model_kwargs={
            "transported_decoder_channels": 4,
            "transported_decoder_scale": 1.0,
            "transported_decoder_transport_gate": True,
            "width": 12,
            "depth": 1,
            "max_modes": 6,
        },
    )
    assert model.transported_decoder is not None
    for parameter in model.transported_decoder.parameters():
        torch.nn.init.constant_(parameter, 0.1)
    zero = torch.zeros(2, 1, 16, 16)
    lifted = torch.randn(2, 1, 16, 16)
    gated = model.transported_landing_correction(zero, zero, lifted)
    assert torch.allclose(gated, torch.zeros_like(gated), atol=1e-6)
    transported = torch.randn(2, 1, 16, 16)
    active = model.transported_landing_correction(transported, zero, lifted)
    assert torch.isfinite(active).all()


def test_spectral_symbol_and_anisotropic_gabor_frame_are_finite() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "frame_type": "anisotropic_gabor",
            "symbol_parameterization": "spectral_order",
            "transport_stencil": 3,
            "width": 12,
            "depth": 1,
            "max_modes": 6,
        },
    )
    x = torch.randn(1, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    assert diagnostics["prediction"].shape == x.shape
    assert torch.isfinite(diagnostics["tokenizer_reconstruction_error"])
    block = diagnostics["block_diagnostics"][0]
    assert torch.isfinite(block["symbol_norm"])
    assert torch.isfinite(block["symbol_order_proxy"])
    assert torch.isfinite(block["symbol_seminorm_proxy"])


def test_directional_gabor_frame_is_finite() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "frame_type": "directional_gabor",
            "symbol_parameterization": "spectral_order",
            "transport_stencil": 3,
            "width": 12,
            "depth": 1,
            "max_modes": 6,
        },
    )
    x = torch.randn(1, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    assert diagnostics["prediction"].shape == x.shape
    assert torch.isfinite(diagnostics["tokenizer_reconstruction_error"])
    assert torch.isfinite(diagnostics["tokenizer_covering_radius"])


def test_branched_mino_preserves_shape_and_reports_gates() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "frame_type": "anisotropic_gabor",
            "transport_parameterization": "hamiltonian_verlet",
            "symbol_parameterization": "spectral_order",
            "num_canonical_branches": 3,
            "branch_routing": "metadata_softmax",
            "transported_synthesis_scale": 1.0,
            "transported_input_scale": 1.0,
            "transported_decoder_channels": 4,
            "transported_decoder_scale": 1.0,
            "edge_symbol_parameterization": "local_packet_kernel",
            "width": 12,
            "depth": 1,
            "max_modes": 6,
            "transport_stencil": 3,
        },
    )
    x = torch.randn(1, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    assert diagnostics["prediction"].shape == x.shape
    weights = diagnostics["branch_weights"]
    assert weights is not None
    assert weights.shape[-1] == 3
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-6)
    branch_metadata = diagnostics["branch_final_metadata"]
    assert branch_metadata is not None
    assert branch_metadata.shape[2] == 3
    block = diagnostics["block_diagnostics"][0]
    for key in (
        "branch_entropy",
        "branch_diversity",
        "branch_usage_max",
        "branch_spread",
        "local_tube_coordinate_norm",
        "edge_symbol_deviation_proxy",
    ):
        assert torch.isfinite(block[key])


def test_frequency_prior_branch_routing_preserves_gate_simplex() -> None:
    for routing in ("frequency_softmax", "metadata_frequency_softmax"):
        model = build_model(
            "MiNO-Plus",
            model_kwargs={
                "frame_type": "directional_gabor",
                "num_canonical_branches": 4,
                "branch_routing": routing,
                "branch_prior_strength": 2.0,
                "width": 12,
                "depth": 1,
                "max_modes": 6,
                "transport_stencil": 3,
            },
        )
        diagnostics = model.forward_with_diagnostics(torch.randn(1, 1, 32, 32))
        weights = diagnostics["branch_weights"]
        assert weights is not None
        assert weights.shape[-1] == 4
        assert torch.isfinite(weights).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-6)


def test_single_branch_mode_on_branched_mino_uses_first_gate() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "num_canonical_branches": 3,
            "branch_routing": "single",
            "width": 12,
            "depth": 1,
            "max_modes": 6,
            "transport_stencil": 3,
        },
    )
    x = torch.randn(1, 1, 32, 32)
    diagnostics = model.forward_with_diagnostics(x)
    weights = diagnostics["branch_weights"]
    assert weights is not None
    assert torch.allclose(weights[..., 0], torch.ones_like(weights[..., 0]), atol=1e-6)
    assert torch.allclose(weights[..., 1:], torch.zeros_like(weights[..., 1:]), atol=1e-6)


def test_symbol_parameterizations_are_multiplier_only_when_theorem_facing() -> None:
    metadata = torch.randn(2, 5, 5)
    zero_features = torch.zeros(2, 5, 8)
    for parameterization in ("spectral_order", "helmholtz_resolvent"):
        for branch in (
            SymbolModulationLayer(width=8, symbol_parameterization=parameterization),
            IdentityPseudodifferentialBranch(width=8, symbol_parameterization=parameterization),
        ):
            output = branch(zero_features, metadata)
            assert torch.allclose(output, torch.zeros_like(output), atol=1e-7)


def test_helmholtz_resolvent_symbol_is_finite() -> None:
    metadata = torch.randn(2, 7, 5)
    metadata[..., 2:4] = torch.randn(2, 7, 2) + torch.linspace(0.2, 2.0, 7).view(1, 7, 1)
    features = torch.randn(2, 7, 8)
    for branch in (
        SymbolModulationLayer(width=8, symbol_parameterization="helmholtz_resolvent"),
        IdentityPseudodifferentialBranch(width=8, symbol_parameterization="helmholtz_resolvent"),
    ):
        output = branch(features, metadata)
        assert output.shape == features.shape
        assert torch.isfinite(output).all()


def test_symbol_branch_uses_local_tube_coordinate() -> None:
    layer = SymbolModulationLayer(width=8, symbol_parameterization="spectral_order")
    metadata = torch.randn(2, 7, 5)
    metadata[..., 2:4] = metadata[..., 2:4] + torch.linspace(0.2, 1.4, 7).view(1, 7, 1)
    features = torch.randn(2, 7, 8)
    local_tube = torch.randn(2, 7, 4)
    output = layer(features, metadata, local_tube)
    loss = output.square().mean()
    loss.backward()
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert layer.local_kernel_proj.weight.grad is not None
    assert torch.isfinite(layer.local_kernel_proj.weight.grad).all()
    assert float(layer.local_kernel_proj.weight.grad.abs().sum()) > 0.0


def test_canonical_propagation_reports_local_tube_coordinate() -> None:
    layer = CanonicalPropagationLayer(
        width=8,
        stencil_size=3,
        sparse_topk=True,
        edge_symbol_parameterization="local_packet_kernel",
    )
    features = torch.randn(2, 9, 8)
    metadata = torch.randn(9, 5)
    output, updated_metadata = layer(features, metadata)
    output.square().mean().backward()
    assert output.shape == features.shape
    assert updated_metadata.shape == (2, 9, 5)
    assert layer.last_local_tube_coordinate is not None
    assert layer.last_local_tube_coordinate.shape == (2, 9, 4)
    assert torch.isfinite(layer.last_local_tube_coordinate).all()
    assert torch.isfinite(layer.last_edge_symbol_deviation_proxy)
    assert layer.edge_symbol is not None
    assert layer.edge_symbol.net[-1].weight.grad is not None
    assert torch.isfinite(layer.edge_symbol.net[-1].weight.grad).all()


def test_helmholtz_resolvent_symbol_sanitizes_nonfinite_inputs() -> None:
    metadata = torch.randn(2, 7, 5)
    features = torch.randn(2, 7, 8)
    metadata[0, 0, 2] = float("nan")
    metadata[0, 1, 3] = float("inf")
    features[0, 2, 4] = float("nan")
    branch = SymbolModulationLayer(width=8, symbol_parameterization="helmholtz_resolvent")
    output = branch(features, metadata)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()


def test_helmholtz_training_losses_sanitize_nonfinite_fields() -> None:
    prediction = torch.randn(2, 2, 8, 8)
    target = torch.randn(2, 2, 8, 8)
    inputs = torch.randn(2, 2, 8, 8)
    prediction[0, 0, 0, 0] = float("nan")
    target[0, 1, 0, 1] = float("inf")
    assert torch.isfinite(_complex_pair_relative_loss(prediction, target))
    assert torch.isfinite(_complex_pair_phase_loss(prediction, target))
    assert torch.isfinite(
        _helmholtz_residual_proxy(prediction, inputs, wavenumber=24.0, refractive_index=1.0)
    )


def test_complex_pair_symbol_action_is_zero_preserving_and_exact() -> None:
    action = ComplexPairSymbolAction(width=4)
    features = torch.randn(2, 3, 4)
    real_multiplier = torch.full((2, 3, 1), 2.0)
    imag_multiplier = torch.full((2, 3, 1), 3.0)
    zero_output = action(torch.zeros_like(features), real_multiplier, imag_multiplier)
    assert torch.allclose(zero_output, torch.zeros_like(zero_output))
    output = action(features, real_multiplier, imag_multiplier)
    assert torch.allclose(output, 2.0 * features, atol=1e-6)
    with torch.no_grad():
        action.imag_lift.weight.copy_(torch.eye(4))
        action.imag_project.weight.copy_(torch.eye(4))
    exact_output = action(features, real_multiplier, imag_multiplier)
    expected = (2.0 * features - 3.0 * features) + (2.0 * features + 3.0 * features)
    assert torch.allclose(exact_output, expected, atol=1e-6)


def test_complex_pair_losses_are_inert_for_scalar_and_exact_for_pairs() -> None:
    scalar_prediction = torch.randn(2, 1, 8, 8)
    scalar_target = torch.randn(2, 1, 8, 8)
    assert _complex_pair_relative_loss(scalar_prediction, scalar_target).item() == 0.0
    assert _complex_pair_phase_loss(scalar_prediction, scalar_target).item() == 0.0

    target = torch.randn(2, 2, 8, 8)
    prediction = target.clone()
    assert torch.allclose(_complex_pair_relative_loss(prediction, target), prediction.new_tensor(0.0))
    assert torch.allclose(_complex_pair_phase_loss(prediction, target), prediction.new_tensor(0.0), atol=1e-6)

    shifted = torch.stack((-target[:, 1], target[:, 0]), dim=1)
    assert float(_complex_pair_phase_loss(shifted, target)) < 1e-5


def test_calibrated_helmholtz_shell_features_peak_on_characteristic_radius() -> None:
    metadata = torch.zeros(1, 3, 5)
    metadata[..., 0:2] = torch.tensor([[[0.75, 0.5], [0.75, 0.5], [0.75, 0.5]]])
    metadata[..., 2:4] = torch.tensor([[[0.4, 0.0], [1.0, 0.0], [1.8, 0.0]]])
    _, _, shell_distance, envelope, outgoing_flux, real, imag, outgoing_gate, shell_center = _helmholtz_shell_features(
        metadata,
        shell_radius=1.0,
        refractive_index=1.0,
        absorption=0.1,
        cap=20.0,
    )
    assert torch.argmin(shell_distance.abs()).item() == 1
    assert torch.argmax(envelope).item() == 1
    assert torch.isfinite(outgoing_flux).all()
    assert torch.isfinite(real).all()
    assert torch.isfinite(imag).all()
    assert torch.isfinite(outgoing_gate).all()
    assert torch.isfinite(shell_center).all()


def test_helmholtz_resolvent_diagnostics_are_reported_by_mino_plus() -> None:
    model = build_model(
        "MiNO-Plus",
        model_kwargs={
            "width": 12,
            "depth": 1,
            "max_modes": 6,
            "symbol_parameterization": "helmholtz_resolvent",
            "helmholtz_shell_radius": 0.7,
            "helmholtz_absorption": 0.08,
            "helmholtz_resolvent_cap": 8.0,
        },
    )
    diagnostics = model.forward_with_diagnostics(torch.randn(1, 1, 32, 32))
    block = diagnostics["block_diagnostics"][0]
    for key in (
        "helmholtz_shell_distance_proxy",
        "helmholtz_resolvent_envelope_proxy",
        "helmholtz_outgoing_flux_proxy",
        "helmholtz_resolvent_real_proxy",
        "helmholtz_resolvent_imag_proxy",
        "helmholtz_outgoing_gate_proxy",
        "helmholtz_shell_center_proxy",
        "helmholtz_complex_latent_imag_energy_proxy",
    ):
        assert key in block
        assert torch.isfinite(block[key])


def test_frequency_order_proxy_backpropagates_to_token_values() -> None:
    metadata = torch.randn(2, 7, 5)
    metadata[..., 2:4] = torch.randn(2, 7, 2) + torch.linspace(0.1, 1.0, 7).view(1, 7, 1)
    token_values = torch.randn(2, 7, 4, requires_grad=True)
    proxy = _frequency_order_proxy(metadata, token_values)
    proxy.backward()
    assert token_values.grad is not None
    assert torch.isfinite(token_values.grad).all()
    assert float(token_values.grad.abs().sum()) > 0.0


def test_finite_symbol_seminorm_proxy_backpropagates_to_token_values() -> None:
    metadata = torch.randn(2, 9, 5)
    metadata[..., :4] = torch.randn(2, 9, 4)
    token_values = torch.randn(2, 9, 4, requires_grad=True)
    proxy = _finite_symbol_seminorm_proxy(metadata, token_values)
    proxy.backward()
    assert token_values.grad is not None
    assert torch.isfinite(token_values.grad).all()
    assert float(token_values.grad.abs().sum()) > 0.0


def test_sparse_topk_matches_dense_masked_topk_shape_and_value() -> None:
    torch.manual_seed(5)
    dense = CanonicalPropagationLayer(width=8, stencil_size=3, sparse_topk=False)
    sparse = CanonicalPropagationLayer(width=8, stencil_size=3, sparse_topk=True)
    sparse.load_state_dict(dense.state_dict())
    features = torch.randn(2, 6, 8)
    metadata = torch.randn(6, 5)
    dense_out, dense_meta = dense(features, metadata)
    sparse_out, sparse_meta = sparse(features, metadata)
    assert dense_out.shape == sparse_out.shape
    assert dense_meta.shape == sparse_meta.shape
    assert torch.allclose(dense_out, sparse_out, atol=1e-5, rtol=1e-5)
