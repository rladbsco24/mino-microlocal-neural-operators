from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data import (  # noqa: E402
    build_benchmark_loaders,
    get_scenario_spec,
    list_profile_scenarios,
    load_dmno_reference_rows,
    load_tcno_reference_rows,
)
from mino.metrics.wavefront import count_parameters  # noqa: E402
from mino.models.mino import build_model  # noqa: E402
from mino.training.train import evaluate_model, fit_model  # noqa: E402


STAGE_DEFAULTS: dict[str, dict[str, object]] = {
    "stage0_smoke": {
        "profile": "stage0_smoke",
        "models": "MiNO-Core,MiNO-Plus,FNO,UNO",
        "seeds": "7",
        "epochs": 2,
        "batch_size": 2,
    },
    "stage1_breadth": {
        "profile": "stage1_breadth",
        "models": "MiNO-Core,MiNO-Plus",
        "seeds": "7",
        "epochs": 8,
        "batch_size": 2,
    },
    "stage2_shortlist": {
        "profile": "stage2_shortlist",
        "models": "MiNO-Core,MiNO-Plus",
        "seeds": "7,11,19",
        "epochs": 16,
        "batch_size": 2,
    },
    "stage3_flagship": {
        "profile": "stage3_flagship",
        "models": "MiNO-Core,MiNO-Plus",
        "seeds": "7,11,19",
        "epochs": 24,
        "batch_size": 2,
    },
}


def _parse_int_tuple(raw: str) -> tuple[int, ...] | None:
    if not raw.strip():
        return None
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def make_model_variant(model_name: str, model_kwargs: dict[str, object] | None) -> str:
    if not model_kwargs:
        return model_name
    normalized = model_name.lower().replace("_", "").replace(" ", "")
    if normalized not in {"mino", "minocore", "minoplus"}:
        return model_name
    parts = [model_name]
    for key in (
        "width",
        "depth",
        "patch_size",
        "stride",
        "max_modes",
        "window_type",
        "low_frequency_scale",
        "transport_scale",
        "transport_stencil",
        "local_refine_channels",
        "mode_strategy",
        "frame_patch_sizes",
        "frame_strides",
        "frame_max_modes",
    ):
        if key not in model_kwargs:
            continue
        value = str(model_kwargs[key]).replace(".", "p").replace(" ", "")
        parts.append(f"{key[:3]}{value}")
    return "_".join(parts)


def _summary_stat(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.stdev(values))


def build_model_kwargs(model_name: str, args: argparse.Namespace) -> dict[str, object] | None:
    normalized = model_name.lower().replace("_", "").replace(" ", "")
    if normalized in {"mino", "minocore"}:
        frame_patch_sizes = _parse_int_tuple(args.core_frame_patch_sizes)
        frame_strides = _parse_int_tuple(args.core_frame_strides)
        frame_max_modes = _parse_int_tuple(args.core_frame_max_modes)
        return {
            "width": args.core_width,
            "depth": args.core_depth,
            "patch_size": args.core_patch_size,
            "stride": args.core_stride,
            "max_modes": args.core_max_modes,
            "window_type": args.core_window_type,
            "mode_strategy": args.core_mode_strategy,
            "low_frequency_scale": args.core_low_frequency_scale,
            "transport_scale": args.core_transport_scale,
            "transport_stencil": args.core_transport_stencil,
            "frame_patch_sizes": frame_patch_sizes,
            "frame_strides": frame_strides,
            "frame_max_modes": frame_max_modes,
        }
    if normalized == "minoplus":
        frame_patch_sizes = _parse_int_tuple(args.plus_frame_patch_sizes)
        frame_strides = _parse_int_tuple(args.plus_frame_strides)
        frame_max_modes = _parse_int_tuple(args.plus_frame_max_modes)
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
            "frame_patch_sizes": frame_patch_sizes,
            "frame_strides": frame_strides,
            "frame_max_modes": frame_max_modes,
        }
    if normalized in {"fno", "conv-fno", "convfno", "wno", "wnoofficial", "wno-official", "uno", "fno+learnedlocaltc", "fnolearnedlocaltc"}:
        return {
            "hidden_channels": args.official_hidden_channels,
            "n_layers": args.official_n_layers,
            "n_modes": args.official_n_modes,
            "patch_size": args.official_patch_size,
            "patch_stride": args.official_patch_stride,
            "top_k_bins": args.official_top_k_bins,
            "transport_eps": args.official_transport_eps,
            "window_type": args.official_window_type,
        }
    return None


def run_one(
    scenario: str,
    model_name: str,
    model_variant: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    tcno_cache_root: Path | None,
    model_kwargs: dict[str, object] | None,
    transport_proxy_weight: float,
    symbol_proxy_weight: float,
    proxy_temperature: float,
) -> dict[str, object]:
    loaders = build_benchmark_loaders(
        scenario_name=scenario,
        batch_size=batch_size,
        seed=seed,
        tcno_cache_root=tcno_cache_root,
    )
    spec = loaders.spec
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(
        model_name,
        in_channels=loaders.in_channels,
        out_channels=loaders.out_channels,
        model_kwargs=model_kwargs,
    )
    model = model.to(device)
    history = fit_model(
        model,
        loaders.train_loader,
        loaders.val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        restore_best=True,
        transport_proxy_weight=transport_proxy_weight,
        symbol_proxy_weight=symbol_proxy_weight,
        proxy_temperature=proxy_temperature,
    )
    test_metrics = evaluate_model(model, loaders.test_loader, device=device, criterion=torch.nn.MSELoss())
    return {
        "scenario": scenario,
        "family": spec.family,
        "regime": spec.regime,
        "source_kind": spec.source_kind,
        "model": model_name,
        "model_variant": model_variant,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip_norm": -1.0 if grad_clip_norm is None else grad_clip_norm,
        "transport_proxy_weight": transport_proxy_weight,
        "symbol_proxy_weight": symbol_proxy_weight,
        "proxy_temperature": proxy_temperature,
        "in_channels": loaders.in_channels,
        "out_channels": loaders.out_channels,
        "height": loaders.spatial_shape[0],
        "width": loaders.spatial_shape[1],
        "parameters": count_parameters(model),
        "runtime_seconds": history["runtime_seconds"],
        "best_val_loss": min(row["val_loss"] for row in history["history"]),
        "test_loss": test_metrics.loss,
        "test_relative_l2": test_metrics.relative_l2,
        "test_phase_error": test_metrics.phase_error,
        "test_packet_consistency": test_metrics.packet_consistency,
        "test_topk_capture": None,
        "reference_source": None,
    }


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["model_variant"])), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (scenario, model_variant), group in sorted(grouped.items()):
        relative = [float(row["test_relative_l2"]) for row in group if row["test_relative_l2"] is not None]
        phase = [float(row["test_phase_error"]) for row in group if row["test_phase_error"] is not None]
        packet = [float(row["test_packet_consistency"]) for row in group if row["test_packet_consistency"] is not None]
        topk = [float(row["test_topk_capture"]) for row in group if row["test_topk_capture"] is not None]
        runtime = [float(row["runtime_seconds"]) for row in group if row["runtime_seconds"] is not None]
        rel_mean, rel_std = _summary_stat(relative)
        phase_mean, phase_std = _summary_stat(phase)
        packet_mean, packet_std = _summary_stat(packet)
        topk_mean, topk_std = _summary_stat(topk)
        runtime_mean, runtime_std = _summary_stat(runtime)
        summary_rows.append(
            {
                "scenario": scenario,
                "family": group[0]["family"],
                "regime": group[0]["regime"],
                "source_kind": group[0]["source_kind"],
                "model_variant": model_variant,
                "mean_test_relative_l2": rel_mean,
                "std_test_relative_l2": rel_std,
                "mean_test_phase_error": phase_mean,
                "std_test_phase_error": phase_std,
                "mean_test_packet_consistency": packet_mean,
                "std_test_packet_consistency": packet_std,
                "mean_test_topk_capture": topk_mean,
                "std_test_topk_capture": topk_std,
                "mean_runtime_seconds": runtime_mean,
                "std_runtime_seconds": runtime_std,
                "runs": len(group),
            }
        )
    return summary_rows


def aggregate_pair_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        family = str(row["family"])
        if row["regime"] not in {"control", "positive"}:
            continue
        key = (family, str(row["model_variant"]))
        grouped.setdefault(key, {}).setdefault(str(row["regime"]), []).append(float(row["test_relative_l2"]))
    pair_rows: list[dict[str, object]] = []
    for (family, model_variant), regime_map in sorted(grouped.items()):
        control = regime_map.get("control", [])
        positive = regime_map.get("positive", [])
        if not control or not positive:
            continue
        control_mean, control_std = _summary_stat(control)
        positive_mean, positive_std = _summary_stat(positive)
        pair_mean, pair_std = _summary_stat(control + positive)
        pair_rows.append(
            {
                "family": family,
                "model_variant": model_variant,
                "control_test_relative_l2_mean": control_mean,
                "control_test_relative_l2_std": control_std,
                "positive_test_relative_l2_mean": positive_mean,
                "positive_test_relative_l2_std": positive_std,
                "pair_mean_relative_l2_mean": pair_mean,
                "pair_mean_relative_l2_std": pair_std,
            }
        )
    return pair_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiNO flagship benchmark campaign.")
    parser.add_argument("--campaign-stage", default="", choices=["", *STAGE_DEFAULTS.keys()])
    parser.add_argument("--profile", default="")
    parser.add_argument("--scenarios", default="", help="Optional comma-separated override for scenario names.")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--transport-proxy-weight", type=float, default=0.0)
    parser.add_argument("--symbol-proxy-weight", type=float, default=0.0)
    parser.add_argument("--proxy-temperature", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="generated/benchmark_flagship")
    parser.add_argument("--tcno-cache-root", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-tcno-reference", action="store_true")
    parser.add_argument("--include-dmno-reference", action="store_true")
    parser.add_argument("--reference-models", default="FNO,Conv-FNO,WNO-style,UNO,FNO+LearnedLocalTC")
    parser.add_argument("--dmno-reference-models", default="DCR-NO")
    parser.add_argument("--core-width", type=int, default=64)
    parser.add_argument("--core-depth", type=int, default=4)
    parser.add_argument("--core-patch-size", type=int, default=16)
    parser.add_argument("--core-stride", type=int, default=8)
    parser.add_argument("--core-max-modes", type=int, default=12)
    parser.add_argument("--core-window-type", default="hann", choices=["hann", "boxcar", "gaussian"])
    parser.add_argument("--core-mode-strategy", default="radial", choices=["radial", "shell_balanced"])
    parser.add_argument("--core-low-frequency-scale", type=float, default=0.1)
    parser.add_argument("--core-transport-scale", type=float, default=0.03)
    parser.add_argument("--core-transport-stencil", type=int, default=4)
    parser.add_argument("--core-frame-patch-sizes", default="")
    parser.add_argument("--core-frame-strides", default="")
    parser.add_argument("--core-frame-max-modes", default="")
    parser.add_argument("--plus-width", type=int, default=64)
    parser.add_argument("--plus-depth", type=int, default=6)
    parser.add_argument("--plus-patch-size", type=int, default=16)
    parser.add_argument("--plus-stride", type=int, default=8)
    parser.add_argument("--plus-max-modes", type=int, default=12)
    parser.add_argument("--plus-window-type", default="hann", choices=["hann", "boxcar", "gaussian"])
    parser.add_argument("--plus-mode-strategy", default="radial", choices=["radial", "shell_balanced"])
    parser.add_argument("--plus-low-frequency-scale", type=float, default=0.05)
    parser.add_argument("--plus-transport-scale", type=float, default=0.03)
    parser.add_argument("--plus-transport-stencil", type=int, default=8)
    parser.add_argument("--plus-local-refine-channels", type=int, default=32)
    parser.add_argument("--plus-frame-patch-sizes", default="")
    parser.add_argument("--plus-frame-strides", default="")
    parser.add_argument("--plus-frame-max-modes", default="")
    parser.add_argument("--official-hidden-channels", type=int, default=32)
    parser.add_argument("--official-n-layers", type=int, default=3)
    parser.add_argument("--official-n-modes", type=int, default=12)
    parser.add_argument("--official-patch-size", type=int, default=32)
    parser.add_argument("--official-patch-stride", type=int, default=16)
    parser.add_argument("--official-top-k-bins", type=int, default=4)
    parser.add_argument("--official-transport-eps", type=float, default=1e-3)
    parser.add_argument("--official-window-type", default="hann", choices=["hann", "rect", "gaussian"])
    return parser.parse_args()


def resolve_stage_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.campaign_stage:
        if not args.profile:
            args.profile = "mixed_smoke"
        if not args.models:
            args.models = "MiNO-Core,MiNO-Plus"
        if not args.seeds:
            args.seeds = "7"
        if args.epochs <= 0:
            args.epochs = 4
        if args.batch_size <= 0:
            args.batch_size = 4
        return args
    defaults = STAGE_DEFAULTS[args.campaign_stage]
    if not args.profile:
        args.profile = str(defaults["profile"])
    if not args.models:
        args.models = str(defaults["models"])
    if not args.seeds:
        args.seeds = str(defaults["seeds"])
    if args.epochs <= 0:
        args.epochs = int(defaults["epochs"])
    if args.batch_size <= 0:
        args.batch_size = int(defaults["batch_size"])
    return args


def main() -> None:
    args = resolve_stage_defaults(parse_args())
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    tcno_cache_root = Path(args.tcno_cache_root) if args.tcno_cache_root else None

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    scenarios = (
        [item.strip() for item in args.scenarios.split(",") if item.strip()]
        if args.scenarios.strip()
        else list(list_profile_scenarios(args.profile))
    )
    grad_clip_norm = None if args.grad_clip_norm <= 0 else args.grad_clip_norm

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        spec = get_scenario_spec(scenario, tcno_cache_root=tcno_cache_root)
        if spec.evaluation_only:
            continue
        for model_name in models:
            for seed in seeds:
                model_kwargs = build_model_kwargs(model_name, args)
                model_variant = make_model_variant(model_name, model_kwargs)
                result_path = output_dir / f"{scenario}_{model_variant.lower()}_seed{seed}.json"
                if args.skip_existing and result_path.exists():
                    rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                    continue
                row = run_one(
                    scenario=scenario,
                    model_name=model_name,
                    model_variant=model_variant,
                    seed=seed,
                    device=device,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    grad_clip_norm=grad_clip_norm,
                    tcno_cache_root=tcno_cache_root,
                    model_kwargs=model_kwargs,
                    transport_proxy_weight=args.transport_proxy_weight,
                    symbol_proxy_weight=args.symbol_proxy_weight,
                    proxy_temperature=args.proxy_temperature,
                )
                result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
                rows.append(row)

    scenario_set = set(scenarios)
    if args.include_tcno_reference:
        reference_models = {item.strip() for item in args.reference_models.split(",") if item.strip()}
        rows.extend(load_tcno_reference_rows(scenarios=scenario_set, models=reference_models))
    if args.include_dmno_reference:
        dmno_models = {item.strip() for item in args.dmno_reference_models.split(",") if item.strip()}
        rows.extend(load_dmno_reference_rows(scenarios=scenario_set, models=dmno_models))

    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    csv_path = output_dir / "benchmark_results.csv"
    write_csv(csv_path, rows)

    summary_rows = aggregate_rows(rows)
    summary_path = output_dir / "benchmark_summary.csv"
    write_csv(summary_path, summary_rows)

    pair_rows = aggregate_pair_rows(rows)
    pair_path = output_dir / "benchmark_pair_summary.csv"
    write_csv(pair_path, pair_rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {pair_path}")


if __name__ == "__main__":
    main()
