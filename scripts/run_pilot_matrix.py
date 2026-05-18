from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data.synthetic import build_dataloaders
from mino.metrics.wavefront import count_parameters
from mino.models.mino import build_model
from mino.training.train import evaluate_model, fit_model


def run_one(
    scenario: str,
    model_name: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    size: int,
    train_count: int,
    val_count: int,
    test_count: int,
    seed: int,
) -> dict[str, object]:
    train_loader, val_loader, test_loader = build_dataloaders(
        scenario=scenario,
        batch_size=batch_size,
        size=size,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        seed=seed,
    )
    model = build_model(model_name)
    history = fit_model(model, train_loader, val_loader, device=device, epochs=epochs)
    test_metrics = evaluate_model(model, test_loader, device=device, criterion=torch.nn.MSELoss())
    return {
        "scenario": scenario,
        "model": model_name,
        "epochs": epochs,
        "seed": seed,
        "parameters": count_parameters(model),
        "runtime_seconds": history["runtime_seconds"],
        "test_loss": test_metrics.loss,
        "test_relative_l2": test_metrics.relative_l2,
        "test_phase_error": test_metrics.phase_error,
        "test_packet_consistency": test_metrics.packet_consistency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny pilot benchmark across scenarios and baselines.")
    parser.add_argument("--models", default="MiNO,FNOStyle,WNOStyle,PDNOStyle,LocalKernel,UNetStyle")
    parser.add_argument("--scenarios", default="darcy,navier_stokes,helmholtz,wave")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="generated/pilot")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for model_name in models:
            rows.append(
                run_one(
                    scenario=scenario,
                    model_name=model_name,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    size=args.size,
                    train_count=args.train_count,
                    val_count=args.val_count,
                    test_count=args.test_count,
                    seed=args.seed,
                )
            )

    json_path = output_dir / "pilot_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = output_dir / "pilot_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
