from __future__ import annotations

import csv
import json
from pathlib import Path


def _default_tcno_root() -> Path:
    return Path(__file__).resolve().parents[3] / "tcno_icml2026_draft"


def _default_dmno_root() -> Path:
    return Path(__file__).resolve().parents[3] / "DMNO_EXP"


def load_tcno_reference_rows(
    scenarios: set[str] | None = None,
    models: set[str] | None = None,
    tcno_root: Path | None = None,
) -> list[dict[str, object]]:
    root = tcno_root or _default_tcno_root()
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    runs_root = root / "generated" / "trainable_runs"
    if runs_root.exists():
        best_payloads: dict[tuple[str, str, int], tuple[tuple[float, float], dict[str, object], Path]] = {}
        for result_path in runs_root.glob("**/result.json"):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            scenario = payload.get("scenario_name")
            model_name = payload.get("model_name")
            seed = int(payload.get("seed", -1))
            if scenario is None or model_name is None:
                continue
            if scenarios is not None and scenario not in scenarios:
                continue
            if models is not None and model_name not in models:
                continue
            key = (scenario, model_name, seed)
            train = payload.get("train", {})
            test = payload.get("test", {})
            epochs = float(train.get("epochs") or 0.0)
            relative_l2 = float(test.get("relative_l2_mean") or float("inf"))
            score = (epochs, -relative_l2)
            current = best_payloads.get(key)
            if current is None or score > current[0]:
                best_payloads[key] = (score, payload, result_path)
        for key, (_, payload, result_path) in best_payloads.items():
            scenario, model_name, seed = key
            seen.add(key)
            train = payload.get("train", {})
            test = payload.get("test", {})
            rows.append(
                {
                    "scenario": scenario,
                    "family": payload.get("family"),
                    "regime": payload.get("setting"),
                    "source_kind": "tcno_reference",
                    "model": model_name,
                    "model_variant": f"{model_name} [tcno-ref]",
                    "seed": seed,
                    "epochs": train.get("epochs"),
                    "batch_size": train.get("batch_size"),
                    "learning_rate": None,
                    "weight_decay": None,
                    "grad_clip_norm": None,
                    "in_channels": 3,
                    "out_channels": 1,
                    "height": 192,
                    "width": 192,
                    "parameters": train.get("param_count"),
                    "runtime_seconds": train.get("runtime_seconds"),
                    "best_val_loss": train.get("best_val_loss"),
                    "test_loss": None,
                    "test_relative_l2": test.get("relative_l2_mean"),
                    "test_phase_error": test.get("phase_error_mean"),
                    "test_packet_consistency": None,
                    "test_topk_capture": test.get("topk_capture_mean"),
                    "reference_source": str(result_path),
                }
            )
    csv_path = root / "generated" / "trainable_tcno_seeded_results.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scenario = row["scenario_name"]
                model_name = row["model_name"]
                seed = int(row["seed"])
                if scenarios is not None and scenario not in scenarios:
                    continue
                if models is not None and model_name not in models:
                    continue
                key = (scenario, model_name, seed)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "scenario": scenario,
                        "family": row["family"],
                        "regime": row["setting"],
                        "source_kind": "tcno_reference",
                        "model": model_name,
                        "model_variant": f"{model_name} [tcno-ref]",
                        "seed": seed,
                        "epochs": None,
                        "batch_size": None,
                        "learning_rate": None,
                        "weight_decay": None,
                        "grad_clip_norm": None,
                        "in_channels": 3,
                        "out_channels": 1,
                        "height": 192,
                        "width": 192,
                        "parameters": float(row["param_count"]),
                        "runtime_seconds": float(row["runtime_seconds"]),
                        "best_val_loss": None,
                        "test_loss": None,
                        "test_relative_l2": float(row["relative_l2_mean"]),
                        "test_phase_error": float(row["phase_error_mean"]),
                        "test_packet_consistency": None,
                        "test_topk_capture": float(row["topk_capture_mean"]),
                        "reference_source": str(csv_path),
                    }
                )
    return rows


def load_dmno_reference_rows(
    dmno_root: Path | None = None,
    scenarios: set[str] | None = None,
    scenario_prefixes: tuple[str, ...] = (),
    models: set[str] | None = None,
) -> list[dict[str, object]]:
    root = dmno_root or _default_dmno_root()
    artifacts_root = root / "artifacts"
    if not artifacts_root.exists():
        return []
    rows: list[dict[str, object]] = []
    for result_path in artifacts_root.glob("**/results/**/result.json"):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        scenario = payload.get("scenario_name")
        if scenario is None:
            continue
        if scenarios is not None and scenario not in scenarios:
            continue
        if scenario_prefixes and not any(scenario.startswith(prefix) for prefix in scenario_prefixes):
            continue
        model_name = payload.get("model_name", "DCR-NO")
        if models is not None and model_name not in models:
            continue
        split_metrics = payload.get("split_metrics", {})
        test_metrics = split_metrics.get("test", {})
        extras = payload.get("extras", {})
        rows.append(
            {
                "scenario": scenario,
                "family": extras.get("pair_name", scenario),
                "regime": extras.get("mode", "dmno_reference"),
                "source_kind": "dmno_reference",
                "model": model_name,
                "model_variant": f"{model_name} [dmno-ref]",
                "seed": int(extras.get("run_seed", -1)),
                "epochs": extras.get("history_length"),
                "batch_size": None,
                "learning_rate": extras.get("lr"),
                "weight_decay": None,
                "grad_clip_norm": None,
                "in_channels": None,
                "out_channels": 1,
                "height": None,
                "width": None,
                "parameters": None,
                "runtime_seconds": None,
                "best_val_loss": extras.get("final_val_loss"),
                "test_loss": None,
                "test_relative_l2": test_metrics.get("relative_l2"),
                "test_phase_error": None,
                "test_packet_consistency": None,
                "test_topk_capture": None,
                "reference_source": str(result_path),
            }
        )
    return rows
