from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

_TCNO_MODULE = None


def _default_tcno_root() -> Path:
    return Path(__file__).resolve().parents[3] / "tcno_icml2026_draft"


def _load_tcno_module(tcno_root: Path | None = None):
    global _TCNO_MODULE
    if _TCNO_MODULE is not None:
        return _TCNO_MODULE
    root = tcno_root or _default_tcno_root()
    script_path = root / "scripts" / "run_trainable_tcno_benchmark.py"
    if not script_path.exists():
        raise FileNotFoundError(f"TCNO benchmark script not found: {script_path}")
    script_dir = script_path.parent
    for candidate in (script_dir, root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    spec = importlib.util.spec_from_file_location("mino_tcno_benchmark", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load TCNO module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _TCNO_MODULE = module
    return module


def _make_tcno_config(
    module,
    *,
    hidden_channels: int = 32,
    n_layers: int = 3,
    n_modes: int = 12,
    alpha: float = 1.0,
    shrink_lambda: float = 0.015,
    patch_size: int = 32,
    patch_stride: int = 16,
    top_k_bins: int = 4,
    transport_eps: float = 1e-3,
    window_type: str = "hann",
    transport_mode: str = "learned_local",
) -> object:
    return module.BenchmarkConfig(
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_modes=n_modes,
        alpha=alpha,
        shrink_lambda=shrink_lambda,
        patch_size=patch_size,
        patch_stride=patch_stride,
        top_k_bins=top_k_bins,
        transport_eps=transport_eps,
        window_type=window_type,
        transport_mode=transport_mode,
        device="cpu",
    )


class _InputAdapter(nn.Module):
    def __init__(self, in_channels: int, expected_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.expected_channels = expected_channels
        if in_channels == expected_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Conv2d(in_channels, expected_channels, kernel_size=1)
            nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
            for out_idx in range(expected_channels):
                self.proj.weight.data[out_idx, out_idx % in_channels, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        if self.in_channels == 1 and self.expected_channels == 3:
            return x.repeat(1, 3, 1, 1)
        return self.proj(x)


class TCNOOfficialWrapper(nn.Module):
    def __init__(
        self,
        tcno_model_name: str,
        *,
        in_channels: int = 3,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 3,
        n_modes: int = 12,
        alpha: float = 1.0,
        shrink_lambda: float = 0.015,
        patch_size: int = 32,
        patch_stride: int = 16,
        top_k_bins: int = 4,
        transport_eps: float = 1e-3,
        window_type: str = "hann",
        transport_mode: str = "learned_local",
        tcno_root: Path | None = None,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("The TCNO official wrappers currently support out_channels=1 only.")
        module = _load_tcno_module(tcno_root)
        config = _make_tcno_config(
            module,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            n_modes=n_modes,
            alpha=alpha,
            shrink_lambda=shrink_lambda,
            patch_size=patch_size,
            patch_stride=patch_stride,
            top_k_bins=top_k_bins,
            transport_eps=transport_eps,
            window_type=window_type,
            transport_mode=transport_mode,
        )
        self.adapter = _InputAdapter(in_channels=in_channels, expected_channels=3)
        self.model = module.build_model(tcno_model_name, config)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(self.adapter(x))


OFFICIAL_MODEL_ALIASES = {
    "fno": "FNO",
    "convfno": "Conv-FNO",
    "conv-fno": "Conv-FNO",
    "wnoofficial": "WNO-style",
    "wno-official": "WNO-style",
    "wno": "WNO-style",
    "uno": "UNO",
    "fno+learnedlocaltc": "FNO+LearnedLocalTC",
    "fnolearnedlocaltc": "FNO+LearnedLocalTC",
}


def build_official_model(
    name: str,
    *,
    in_channels: int = 3,
    out_channels: int = 1,
    model_kwargs: dict[str, object] | None = None,
) -> nn.Module:
    key = name.lower().replace("_", "").replace(" ", "").replace("-", "")
    if key not in OFFICIAL_MODEL_ALIASES:
        raise ValueError(f"Unknown official baseline: {name}")
    kwargs = dict(model_kwargs or {})
    return TCNOOfficialWrapper(
        OFFICIAL_MODEL_ALIASES[key],
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs,
    )
