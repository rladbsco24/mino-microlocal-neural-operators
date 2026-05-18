"""Collect all MiNO experiment outputs into a single index.

The collector is intentionally read-only with respect to existing experiment
directories. It scans generated outputs, copies no checkpoints, and writes a
separate index directory that can be regenerated at any time while long runs are
continuing in the background.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

RUN_CSV_NAMES = {
    "benchmark_results.csv": "one_step_benchmark",
    "empirical_closure_results.csv": "empirical_closure",
    "long_rollout_results.csv": "long_rollout",
}

SUMMARY_CSV_NAMES = {
    "benchmark_summary.csv": "one_step_benchmark",
    "benchmark_pair_summary.csv": "one_step_pair_summary",
    "empirical_closure_summary.csv": "empirical_closure",
    "empirical_closure_proxy_deltas.csv": "proxy_delta",
    "empirical_closure_proxy_delta.csv": "proxy_delta",
    "empirical_closure_ablation_deltas.csv": "ablation_delta",
    "long_rollout_summary.csv": "long_rollout",
}

PLAN_CSV_NAMES = {
    "empirical_closure_plan.csv": "empirical_closure_plan",
    "long_rollout_plan.csv": "long_rollout_plan",
}

CANONICAL_RUN_FIELDS = [
    "collection_timestamp",
    "track",
    "source_dir",
    "source_file",
    "run_id",
    "scenario",
    "family",
    "regime",
    "source_kind",
    "model",
    "model_variant",
    "seed",
    "epochs",
    "batch_size",
    "hardware_profile",
    "loss_config",
    "ablation",
    "rollout_steps",
    "test_relative_l2",
    "mean_relative_l2",
    "final_relative_l2",
    "test_loss",
    "best_val_loss",
    "test_phase_error",
    "test_packet_consistency",
    "transport_budget",
    "symbol_budget",
    "residual_budget",
    "final_train_transport_proxy",
    "final_train_symbol_proxy",
    "spectral_energy_error",
    "enstrophy_error",
    "high_frequency_energy_drift",
    "runtime_seconds",
    "parameters",
    "height",
    "width",
    "reference_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", default="generated", help="Generated-output root to scan.")
    parser.add_argument("--output", default="generated/experiment_index", help="Directory for index CSV/JSON files.")
    parser.add_argument("--include-dry-runs", action="store_true", help="Include dry-run plan directories in inventory.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    if preferred_fields:
        fields.extend(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def detect_track(path: Path) -> str:
    name = path.name
    if name in RUN_CSV_NAMES:
        return RUN_CSV_NAMES[name]
    if name in SUMMARY_CSV_NAMES:
        return SUMMARY_CSV_NAMES[name]
    if name in PLAN_CSV_NAMES:
        return PLAN_CSV_NAMES[name]
    return "unknown"


def normalize_run(row: dict[str, str], path: Path, timestamp: str) -> dict[str, Any]:
    out = {field: "" for field in CANONICAL_RUN_FIELDS}
    out.update(row)
    out["collection_timestamp"] = timestamp
    out["track"] = detect_track(path)
    out["source_dir"] = rel_path(path.parent)
    out["source_file"] = rel_path(path)
    if not out.get("run_id"):
        parts = [
            str(out.get("scenario", "")),
            str(out.get("model", "")),
            str(out.get("seed", "")),
            str(out.get("loss_config", "")),
            str(out.get("ablation", "")),
            str(out.get("rollout_steps", "")),
        ]
        out["run_id"] = "_".join(part for part in parts if part)
    return out


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def primary_error(row: dict[str, Any]) -> float | None:
    for key in ("test_relative_l2", "mean_relative_l2", "final_relative_l2"):
        value = to_float(row.get(key))
        if value is not None:
            return value
    return None


def aggregate_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        error = primary_error(row)
        if error is None:
            continue
        key = (
            str(row.get("track", "")),
            str(row.get("scenario", "")),
            str(row.get("model", "")),
            str(row.get("loss_config", "")),
            str(row.get("ablation", "")),
        )
        groups[key].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        errors = [primary_error(row) for row in group]
        clean_errors = [value for value in errors if value is not None]
        if not clean_errors:
            continue
        best = min(group, key=lambda row: primary_error(row) or float("inf"))
        aggregate_rows.append(
            {
                "track": key[0],
                "scenario": key[1],
                "model": key[2],
                "loss_config": key[3],
                "ablation": key[4],
                "runs": len(clean_errors),
                "mean_primary_error": mean(clean_errors),
                "std_primary_error": pstdev(clean_errors) if len(clean_errors) > 1 else 0.0,
                "best_primary_error": min(clean_errors),
                "best_seed": best.get("seed", ""),
                "best_run_id": best.get("run_id", ""),
                "source_dirs": ";".join(sorted({str(row.get("source_dir", "")) for row in group})),
            }
        )
    return aggregate_rows


def inventory_rows(generated_root: Path, include_dry_runs: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(generated_root.rglob("*.csv")):
        if not include_dry_runs and "dryrun" in rel_path(path).lower():
            continue
        kind = "other"
        if path.name in RUN_CSV_NAMES:
            kind = "run"
        elif path.name in SUMMARY_CSV_NAMES:
            kind = "summary"
        elif path.name in PLAN_CSV_NAMES:
            kind = "plan"
        csv_rows = read_csv(path)
        rows.append(
            {
                "source_file": rel_path(path),
                "source_dir": rel_path(path.parent),
                "kind": kind,
                "track": detect_track(path),
                "rows": len(csv_rows),
                "bytes": path.stat().st_size,
                "last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return rows


def load_run_rows(generated_root: Path) -> list[dict[str, Any]]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for path in sorted(generated_root.rglob("*.csv")):
        if path.name not in RUN_CSV_NAMES:
            continue
        if "dryrun" in rel_path(path).lower():
            continue
        for row in read_csv(path):
            rows.append(normalize_run(row, path, timestamp))
    return rows


def write_snapshot(output_dir: Path, run_rows: list[dict[str, Any]], best_rows: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> None:
    tracks: dict[str, int] = defaultdict(int)
    for row in run_rows:
        tracks[str(row.get("track", ""))] += 1
    lines = [
        "# MiNO Experiment Index",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Completed Run Rows",
    ]
    for track, count in sorted(tracks.items()):
        lines.append(f"- `{track}`: `{count}` rows")
    lines.extend(
        [
            f"- Total run rows: `{len(run_rows)}`",
            f"- Indexed CSV files: `{len(inventory)}`",
            "",
            "## Current Best Rows By Scenario/Model",
        ]
    )
    for row in sorted(best_rows, key=lambda item: (str(item["track"]), str(item["scenario"]), float(item["best_primary_error"]))):
        lines.append(
            "- "
            f"`{row['track']}` / `{row['scenario']}` / `{row['model']}`"
            f" / `{row['loss_config'] or '-'} / {row['ablation'] or '-'}`:"
            f" best `{float(row['best_primary_error']):.6g}`,"
            f" mean `{float(row['mean_primary_error']):.6g}` over `{row['runs']}` run(s)"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    generated_root = (ROOT / args.generated).resolve()
    output_dir = (ROOT / args.output).resolve()
    if not generated_root.exists():
        raise FileNotFoundError(f"Missing generated root: {generated_root}")

    run_rows = load_run_rows(generated_root)
    inventory = inventory_rows(generated_root, include_dry_runs=args.include_dry_runs)
    best_rows = aggregate_best(run_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "unified_runs.csv", run_rows, preferred_fields=CANONICAL_RUN_FIELDS)
    write_csv(output_dir / "best_by_scenario_model.csv", best_rows)
    write_csv(output_dir / "experiment_inventory.csv", inventory)
    write_snapshot(output_dir, run_rows, best_rows, inventory)

    manifest = {
        "generated_root": rel_path(generated_root),
        "output_dir": rel_path(output_dir),
        "run_rows": len(run_rows),
        "best_rows": len(best_rows),
        "inventory_rows": len(inventory),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
