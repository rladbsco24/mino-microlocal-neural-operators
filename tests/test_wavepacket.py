from __future__ import annotations

import torch

from mino.models.wavepacket import (
    AnisotropicGaborWavepacketTokenizer,
    LocalFourierWavepacketTokenizer,
    RectangularFourierWavepacketTokenizer,
    packet_transport_symbol_proxy,
)


def test_wavepacket_roundtrip_with_full_modes() -> None:
    x = torch.randn(2, 1, 16, 16)
    tokenizer = LocalFourierWavepacketTokenizer(in_channels=1, patch_size=16, stride=16, max_modes=None, window_type="boxcar")
    encoding = tokenizer(x)
    reconstructed = tokenizer.synthesize(encoding.features, encoding)
    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, atol=5e-4, rtol=5e-4)


def test_hann_synthesis_stays_finite_for_small_coefficients() -> None:
    x = torch.randn(2, 1, 32, 32)
    tokenizer = LocalFourierWavepacketTokenizer(in_channels=1, patch_size=16, stride=8, max_modes=12, window_type="hann")
    encoding = tokenizer(x)
    small_features = torch.full_like(encoding.features, 1e-3)
    reconstructed = tokenizer.synthesize(small_features, encoding)
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()
    assert float(reconstructed.abs().max()) < 1.0


def test_gaussian_shell_balanced_proxy_terms_are_finite() -> None:
    x = torch.randn(2, 1, 32, 32)
    tokenizer = LocalFourierWavepacketTokenizer(
        in_channels=1,
        patch_size=16,
        stride=8,
        max_modes=12,
        window_type="gaussian",
        mode_strategy="shell_balanced",
    )
    prediction_encoding = tokenizer(x)
    target_encoding = tokenizer(0.5 * x)
    proxy = packet_transport_symbol_proxy(
        prediction_encoding.features,
        prediction_encoding.metadata,
        prediction_encoding,
        target_encoding.features,
        target_encoding,
        temperature=0.1,
    )
    assert torch.isfinite(proxy["transport_proxy"])
    assert torch.isfinite(proxy["symbol_proxy"])
    assert float(proxy["transport_proxy"].item()) >= 0.0
    assert float(proxy["symbol_proxy"].item()) >= 0.0


def test_rectangular_and_anisotropic_gabor_tokenizers_are_finite() -> None:
    x = torch.randn(1, 1, 32, 32)
    rectangular = RectangularFourierWavepacketTokenizer(
        in_channels=1,
        patch_shape=(8, 16),
        stride_shape=(4, 8),
        max_modes=6,
    )
    encoding = rectangular(x)
    reconstructed = rectangular.synthesize(encoding.features, encoding)
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()

    anisotropic = AnisotropicGaborWavepacketTokenizer(
        in_channels=1,
        patch_shapes=((8, 16), (16, 8)),
        stride_shapes=((4, 8), (8, 4)),
        max_modes=(4, 4),
    )
    multi_encoding = anisotropic(x)
    multi_reconstructed = anisotropic.synthesize(multi_encoding.features, multi_encoding)
    assert multi_reconstructed.shape == x.shape
    assert torch.isfinite(multi_reconstructed).all()
