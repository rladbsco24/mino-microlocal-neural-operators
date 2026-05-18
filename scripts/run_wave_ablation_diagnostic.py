from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data import build_benchmark_loaders  # noqa: E402
from mino.models.mino import build_model  # noqa: E402
from mino.training.train import evaluate_model, fit_model  # noqa: E402


class ZeroPropagation(nn.Module):
    def forward(self, features: Tensor, metadata: Tensor) -> tuple[Tensor, Tensor]:
        return torch.zeros_like(features), metadata


class ZeroSymbol(nn.Module):
    def forward(self, features: Tensor, metadata: Tensor) -> Tensor:
        return torch.zeros_like(features)


class ZeroField(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]))


def apply_ablation(model: nn.Module, ablation: str, out_channels: int) -> None:
    if ablation == "full":
        return
    if ablation == "no_transport":
        for block in model.core.blocks:
            block.propagation = ZeroPropagation()
        return
    if ablation == "freeze_transport":
        for block in model.core.blocks:
            block.propagation.transport_scale = 0.0
        return
    if ablation == "no_symbol":
        for block in model.core.blocks:
            block.symbol = ZeroSymbol()
        return
    if ablation == "no_lowfreq":
        model.low_frequency_scale = 0.0
        return
    if ablation == "no_local_refine":
        model.local_refine = ZeroField(out_channels)
        return
    raise ValueError(f"Unknown ablation: {ablation}")


def first_batch_prediction(model: nn.Module, loader, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = model(inputs)
    return inputs.cpu(), targets.cpu(), preds.cpu()


def save_wave_figure(path: Path, input_field: Tensor, target: Tensor, predictions: dict[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panels: list[tuple[str, Tensor]] = [
        ("input", input_field[0, 0]),
        ("target", target[0, 0]),
    ]
    for name, pred in predictions.items():
        panels.append((name, pred[0, 0]))
        panels.append((f"|{name}-target|", (pred[0, 0] - target[0, 0]).abs()))

    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    vmax = max(float(t.abs().max().item()) for _, t in panels[:2] + [(k, v[0, 0]) for k, v in predictions.items()])
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, tensor) in zip(axes.ravel(), panels):
        if title.startswith("|"):
            image = ax.imshow(tensor.numpy(), cmap="magma")
        else:
            image = ax.imshow(tensor.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(image, ax=ax, shrink=0.72, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small wave_synth MiNO ablation diagnostic.")
    parser.add_argument("--output", default="generated/wave_ablation_diagnostic")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ablations",
        default="full,no_transport,freeze_transport,no_symbol,no_lowfreq,no_local_refine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loaders = build_benchmark_loaders("wave_synth", batch_size=args.batch_size, seed=args.seed)
    ablations = [item.strip() for item in args.ablations.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    prediction_panels: dict[str, Tensor] = {}
    input_panel: Tensor | None = None
    target_panel: Tensor | None = None
    for ablation in ablations:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = build_model("MiNO-Plus", in_channels=loaders.in_channels, out_channels=loaders.out_channels)
        apply_ablation(model, ablation, loaders.out_channels)
        model = model.to(device)
        history = fit_model(
            model,
            loaders.train_loader,
            loaders.val_loader,
            device=device,
            epochs=args.epochs,
            learning_rate=1e-4,
            weight_decay=1e-4,
            grad_clip_norm=1.0,
            restore_best=True,
        )
        metrics = evaluate_model(model, loaders.test_loader, device=device, criterion=nn.MSELoss())
        row = {
            "scenario": "wave_synth",
            "model": "MiNO-Plus",
            "ablation": ablation,
            "seed": args.seed,
            "epochs": args.epochs,
            "test_relative_l2": metrics.relative_l2,
            "test_phase_error": metrics.phase_error,
            "test_packet_consistency": metrics.packet_consistency,
            "runtime_seconds": history["runtime_seconds"],
        }
        rows.append(row)
        (output_dir / f"{ablation}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        if ablation in {"full", "no_transport", "no_symbol", "no_local_refine"}:
            inputs, targets, preds = first_batch_prediction(model, loaders.test_loader, device)
            if input_panel is None:
                input_panel = inputs
                target_panel = targets
            prediction_panels[ablation] = preds

    csv_path = output_dir / "wave_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if input_panel is not None and target_panel is not None:
        save_wave_figure(output_dir / "wave_ablation_prediction.png", input_panel, target_panel, prediction_panels)
        manuscript_figure = ROOT / "manuscript" / "jmlr" / "figures" / "wave_ablation_prediction.png"
        save_wave_figure(manuscript_figure, input_panel, target_panel, prediction_panels)

    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
