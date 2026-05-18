from __future__ import annotations

import torch

from mino.metrics.wavefront import (
    amplitude_relative_error,
    boundary_trace_relative_error,
    complex_relative_l2,
    high_frequency_relative_error,
    packet_consistency,
    packet_threshold_wavefront_localization_error,
    phase_error,
    relative_l2,
    sobolev_relative_error,
    symbol_order_scaling_error,
)


def test_metrics_are_zero_on_identity() -> None:
    x = torch.randn(3, 1, 32, 32)
    assert torch.allclose(relative_l2(x, x), torch.zeros(3), atol=1e-6)
    assert torch.allclose(complex_relative_l2(x, x), torch.zeros(3), atol=1e-6)
    assert torch.allclose(amplitude_relative_error(x, x), torch.zeros(3), atol=1e-6)
    assert torch.allclose(boundary_trace_relative_error(x, x), torch.zeros(3), atol=1e-6)
    assert torch.allclose(phase_error(x, x), torch.zeros(3), atol=1e-6)
    assert torch.allclose(packet_consistency(x, x), torch.zeros(3), atol=1e-6)


def test_microlocal_metric_proxies_are_finite() -> None:
    x = torch.randn(2, 1, 32, 32)
    y = torch.roll(x, shifts=2, dims=-1)
    z = torch.randn(2, 2, 32, 32)
    z_shifted = torch.roll(z, shifts=2, dims=-1)
    assert torch.isfinite(packet_threshold_wavefront_localization_error(y, x)).all()
    assert torch.isfinite(sobolev_relative_error(y, x, order=2.0)).all()
    assert torch.isfinite(high_frequency_relative_error(y, x)).all()
    assert torch.isfinite(symbol_order_scaling_error(y, x)).all()
    assert torch.isfinite(complex_relative_l2(z_shifted, z)).all()
    assert torch.isfinite(amplitude_relative_error(z_shifted, z)).all()
    assert torch.isfinite(boundary_trace_relative_error(z_shifted, z)).all()
