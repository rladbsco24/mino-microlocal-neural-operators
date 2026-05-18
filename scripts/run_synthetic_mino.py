from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic MiNO benchmark.")
    parser.add_argument("--scenario", default="helmholtz", choices=["darcy", "navier_stokes", "helmholtz", "wave"])
    parser.add_argument("--model", default="MiNO", choices=["MiNO", "FNOStyle", "WNOStyle", "PDNOStyle", "LocalKernel", "UNetStyle"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="generated")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        scenario=args.scenario,
        batch_size=args.batch_size,
        size=args.size,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    model = build_model(args.model)
    history = fit_model(model, train_loader, val_loader, device=device, epochs=args.epochs)
    test_metrics = evaluate_model(model, test_loader, device=device, criterion=torch.nn.MSELoss())

    results = {
        "scenario": args.scenario,
        "model": args.model,
        "seed": args.seed,
        "parameters": count_parameters(model),
        "runtime_seconds": history["runtime_seconds"],
        "history": history["history"],
        "test": {
            "loss": test_metrics.loss,
            "relative_l2": test_metrics.relative_l2,
            "phase_error": test_metrics.phase_error,
            "packet_consistency": test_metrics.packet_consistency,
        },
    }
    output_path = output_dir / f"{args.scenario}_{args.model.lower()}_seed{args.seed}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["test"], indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
