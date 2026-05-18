from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data.rollout import build_sequence_loaders, get_sequence_scenario_spec, list_rollout_profile_scenarios  # noqa: E402
from mino.metrics.wavefront import count_parameters  # noqa: E402
from mino.models.mino import build_model  # noqa: E402
from mino.training.rollout import evaluate_rollout_model, fit_one_step_rollout_model  # noqa: E402


HARDWARE_PROFILES: dict[str, dict[str, object]] = {
    "local_smoke": {
        "epochs": 2,
        "batch_size": 1,
        "rollout_steps": "4",
        "max_train_samples": 4,
        "max_val_samples": 2,
        "max_test_samples": 2,
        "profile": "rollout_smoke",
    },
    "paper_v100": {
        "epochs": 100,
        "batch_size": 4,
        "rollout_steps": "10,20,50",
        "max_train_samples": 0,
        "max_val_samples": 0,
        "max_test_samples": 0,
        "profile": "stage3_rollout_diagnostic",
    },
    "paper_4090": {
        "epochs": 500,
        "batch_size": 8,
        "rollout_steps": "10,20,50",
        "max_train_samples": 0,
        "max_val_samples": 0,
        "max_test_samples": 0,
        "profile": "stage3_rollout_diagnostic",
    },
    "paper_a100": {
        "epochs": 500,
        "batch_size": 8,
        "rollout_steps": "10,20,50",
        "max_train_samples": 0,
        "max_val_samples": 0,
        "max_test_samples": 0,
        "profile": "stage3_rollout_diagnostic",
    },
}

METRIC_FIELDS = (
    "mean_relative_l2",
    "final_relative_l2",
    "rollout_stability",
    "spectral_energy_error",
    "enstrophy_error",
    "high_frequency_energy_drift",
    "transport_budget",
    "symbol_budget",
    "residual_budget",
    "packet_trajectory_consistency",
    "packet_amplitude_error",
    "packet_residual_energy",
    "packet_energy_cascade_error",
    "runtime_seconds",
)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_csv(raw: str) -> list[int]:
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


def make_run_id(row: dict[str, object]) -> str:
    return "_".join(
        safe_slug(str(row[key]))
        for key in ("scenario", "model", "seed", "rollout_steps", "hardware_profile")
        if key in row
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def mean_std(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("row_type") != "run":
            continue
        key = (str(row["scenario"]), str(row["model_variant"]), int(row["rollout_steps"]))
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for (scenario, model_variant, rollout_steps), group in sorted(grouped.items()):
        out: dict[str, object] = {
            "scenario": scenario,
            "model_variant": model_variant,
            "rollout_steps": rollout_steps,
            "family": group[0]["family"],
            "regime": group[0]["regime"],
            "source_kind": group[0]["source_kind"],
            "runs": len(group),
        }
        for field in METRIC_FIELDS:
            mean, std = mean_std([float(row[field]) for row in group if row.get(field) is not None])
            out[f"mean_{field}"] = mean
            out[f"std_{field}"] = std
        summary.append(out)
    return summary


def build_model_kwargs(model_name: str, args: argparse.Namespace) -> dict[str, object] | None:
    normalized = model_name.lower().replace("_", "").replace(" ", "").replace("-", "")
    if normalized in {"mino", "minocore"}:
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
            "frame_patch_sizes": parse_int_tuple(args.core_frame_patch_sizes),
            "frame_strides": parse_int_tuple(args.core_frame_strides),
            "frame_max_modes": parse_int_tuple(args.core_frame_max_modes),
        }
    if normalized == "minoplus":
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
            "frame_patch_sizes": parse_int_tuple(args.plus_frame_patch_sizes),
            "frame_strides": parse_int_tuple(args.plus_frame_strides),
            "frame_max_modes": parse_int_tuple(args.plus_frame_max_modes),
        }
    if normalized in {"fno", "convfno", "wno", "wnoofficial", "uno", "fnolearnedlocaltc"}:
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


def model_variant(model_name: str, kwargs: dict[str, object] | None) -> str:
    if not kwargs:
        return model_name
    normalized = model_name.lower().replace("_", "").replace(" ", "").replace("-", "")
    if normalized not in {"mino", "minocore", "minoplus"}:
        return model_name
    parts = [model_name]
    for key in ("width", "depth", "patch_size", "stride", "max_modes", "window_type", "mode_strategy", "transport_stencil"):
        value = kwargs.get(key)
        if value is not None:
            parts.append(f"{key[:3]}{str(value).replace('.', 'p')}")
    return "_".join(parts)


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def run_with_oom_fallback(run_fn: Callable[[int, bool], dict[str, object]], batch_size: int) -> dict[str, object]:
    try:
        return run_fn(batch_size, False)
    except RuntimeError as error:
        if batch_size <= 1 or not is_cuda_oom(error):
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return run_fn(1, True)


def make_plan_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sequence_cache_root = Path(args.sequence_cache_root) if args.sequence_cache_root else None
    for scenario in args.scenario_list:
        spec = get_sequence_scenario_spec(scenario, sequence_cache_root=sequence_cache_root)
        for model_name in args.model_list:
            kwargs = build_model_kwargs(model_name, args)
            variant = model_variant(model_name, kwargs)
            for seed in args.seed_list:
                for horizon in args.rollout_step_list:
                    rows.append(
                        {
                            "row_type": "plan",
                            "scenario": scenario,
                            "family": spec.family,
                            "regime": spec.regime,
                            "source_kind": spec.source_kind,
                            "model": model_name,
                            "model_variant": variant,
                            "seed": seed,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "rollout_steps": horizon,
                            "hardware_profile": args.hardware_profile,
                        }
                    )
    if args.resume_from > 0:
        rows = rows[args.resume_from :]
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    return rows


def run_one(row: dict[str, object], args: argparse.Namespace, device: torch.device, batch_size: int, oom_fallback: bool) -> dict[str, object]:
    scenario = str(row["scenario"])
    model_name = str(row["model"])
    seed = int(row["seed"])
    horizon = int(row["rollout_steps"])
    sequence_cache_root = Path(args.sequence_cache_root) if args.sequence_cache_root else None
    loaders = build_sequence_loaders(
        scenario,
        batch_size=batch_size,
        seed=seed,
        sequence_cache_root=sequence_cache_root,
        rollout_steps=horizon,
        synthetic_size=args.synthetic_size if args.synthetic_size > 0 else None,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    kwargs = build_model_kwargs(model_name, args)
    model = build_model(model_name, in_channels=loaders.in_channels, out_channels=loaders.out_channels, model_kwargs=kwargs)
    model = model.to(device)
    history = fit_one_step_rollout_model(
        model,
        loaders.train_loader,
        loaders.val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=None if args.grad_clip_norm <= 0 else args.grad_clip_norm,
        restore_best=True,
        transport_proxy_weight=args.transport_proxy_weight,
        symbol_proxy_weight=args.symbol_proxy_weight,
        proxy_temperature=args.proxy_temperature,
    )
    metrics = evaluate_rollout_model(
        model,
        loaders.test_loader,
        device,
        proxy_temperature=args.proxy_temperature,
        max_batches=args.max_eval_batches,
    )
    out = {
        **row,
        "row_type": "run",
        "requested_batch_size": args.batch_size,
        "batch_size": batch_size,
        "oom_fallback": oom_fallback,
        "device": str(device),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "transport_proxy_weight": args.transport_proxy_weight,
        "symbol_proxy_weight": args.symbol_proxy_weight,
        "proxy_temperature": args.proxy_temperature,
        "parameters": count_parameters(model),
        "runtime_seconds": history["runtime_seconds"],
        "best_val_loss": min((item["val_loss"] for item in history["history"]), default=math.nan),
        "loss": metrics.loss,
        "mean_relative_l2": metrics.mean_relative_l2,
        "final_relative_l2": metrics.final_relative_l2,
        "rollout_stability": metrics.rollout_stability,
        "spectral_energy_error": metrics.spectral_energy_error,
        "enstrophy_error": metrics.enstrophy_error,
        "high_frequency_energy_drift": metrics.high_frequency_energy_drift,
        "transport_budget": metrics.transport_budget,
        "symbol_budget": metrics.symbol_budget,
        "residual_budget": metrics.residual_budget,
        "packet_trajectory_consistency": metrics.packet_trajectory_consistency,
        "packet_amplitude_error": metrics.packet_amplitude_error,
        "packet_residual_energy": metrics.packet_residual_energy,
        "packet_energy_cascade_error": metrics.packet_energy_cascade_error,
    }
    out["run_id"] = make_run_id(out)
    return out


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    profile = HARDWARE_PROFILES[args.hardware_profile]
    if args.profile == "":
        args.profile = str(profile["profile"])
    if args.scenarios == "":
        args.scenarios = ",".join(list_rollout_profile_scenarios(args.profile))
    if args.models == "":
        args.models = "MiNO-Plus"
    if args.seeds == "":
        args.seeds = "7,11,19"
    if args.rollout_steps == "":
        args.rollout_steps = str(profile["rollout_steps"])
    if args.epochs < 0:
        args.epochs = int(profile["epochs"])
    if args.batch_size < 0:
        args.batch_size = int(profile["batch_size"])
    for field in ("max_train_samples", "max_val_samples", "max_test_samples"):
        if getattr(args, field) < 0:
            setattr(args, field, int(profile[field]))
    if args.output == "":
        args.output = str(Path("generated") / "long_rollout" / args.hardware_profile)
    args.scenario_list = parse_csv(args.scenarios)
    args.model_list = parse_csv(args.models)
    args.seed_list = parse_int_csv(args.seeds)
    args.rollout_step_list = parse_int_csv(args.rollout_steps)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or plan MiNO Navier-Stokes long-rollout campaigns.")
    parser.add_argument("--hardware-profile", choices=sorted(HARDWARE_PROFILES), default="local_smoke")
    parser.add_argument("--profile", default="")
    parser.add_argument("--scenarios", "--scenario", dest="scenarios", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--rollout-steps", default="")
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=-1)
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--resume-from", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence-cache-root", default="")
    parser.add_argument("--synthetic-size", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=-1)
    parser.add_argument("--max-val-samples", type=int, default=-1)
    parser.add_argument("--max-test-samples", type=int, default=-1)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--transport-proxy-weight", type=float, default=0.02)
    parser.add_argument("--symbol-proxy-weight", type=float, default=0.02)
    parser.add_argument("--proxy-temperature", type=float, default=0.05)
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
    parser.add_argument("--plus-max-modes", type=int, default=16)
    parser.add_argument("--plus-window-type", default="gaussian", choices=["hann", "boxcar", "gaussian"])
    parser.add_argument("--plus-mode-strategy", default="shell_balanced", choices=["radial", "shell_balanced"])
    parser.add_argument("--plus-low-frequency-scale", type=float, default=0.05)
    parser.add_argument("--plus-transport-scale", type=float, default=0.03)
    parser.add_argument("--plus-transport-stencil", type=int, default=12)
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
    return resolve_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = make_plan_rows(args)
    write_csv(output_dir / "long_rollout_plan.csv", plan_rows)
    manifest: dict[str, Any] = {
        "hardware_profile": args.hardware_profile,
        "profile": args.profile,
        "scenarios": args.scenario_list,
        "models": args.model_list,
        "seeds": args.seed_list,
        "rollout_steps": args.rollout_step_list,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
        "planned_rows": len(plan_rows),
        "output_dir": str(output_dir),
    }
    if args.dry_run:
        write_json(output_dir / "manifest.json", manifest)
        print(f"[dry-run] wrote {output_dir / 'long_rollout_plan.csv'}")
        print(f"[dry-run] planned_rows={len(plan_rows)}")
        return

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results: list[dict[str, object]] = []
    for row in plan_rows:
        run_id = make_run_id(row)
        result_path = output_dir / f"{run_id}.json"
        if args.skip_existing and result_path.exists():
            print(f"[skip] {run_id}", flush=True)
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        def _run(batch_size: int, oom_fallback: bool) -> dict[str, object]:
            print(
                f"[run] scenario={row['scenario']} model={row['model']} "
                f"seed={row['seed']} horizon={row['rollout_steps']} "
                f"epochs={args.epochs} batch={batch_size}",
                flush=True,
            )
            return run_one(row, args, device, batch_size, oom_fallback)

        result = run_with_oom_fallback(_run, args.batch_size)
        write_json(result_path, result)
        results.append(result)
        write_csv(output_dir / "long_rollout_results.csv", results)
        write_csv(output_dir / "long_rollout_summary.csv", aggregate_rows(results))
        print(
            f"[done] {run_id} mean_rel={result['mean_relative_l2']:.6g} "
            f"final_rel={result['final_relative_l2']:.6g}",
            flush=True,
        )
    manifest["device"] = str(device)
    manifest["completed_rows"] = len(results)
    write_json(output_dir / "manifest.json", manifest)
    write_csv(output_dir / "long_rollout_results.csv", results)
    write_csv(output_dir / "long_rollout_summary.csv", aggregate_rows(results))
    print(f"Wrote {output_dir / 'long_rollout_results.csv'}")
    print(f"Wrote {output_dir / 'long_rollout_summary.csv'}")


if __name__ == "__main__":
    main()
