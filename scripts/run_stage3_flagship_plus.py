from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = ROOT / "scripts" / "run_mino_benchmark.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the MiNO stage-3 flagship benchmark with richer packet settings.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--models", default="MiNO-Plus,UNO,Conv-FNO,FNO+LearnedLocalTC")
    parser.add_argument("--output", default="generated/benchmark_stage3_flagship_plus")
    parser.add_argument("--scenarios", default="")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--transport-proxy-weight", type=float, default=0.02)
    parser.add_argument("--symbol-proxy-weight", type=float, default=0.02)
    parser.add_argument("--proxy-temperature", type=float, default=0.05)
    parser.add_argument("--plus-window-type", default="gaussian", choices=["hann", "boxcar", "gaussian"])
    parser.add_argument("--plus-mode-strategy", default="shell_balanced", choices=["radial", "shell_balanced"])
    parser.add_argument("--plus-transport-stencil", type=int, default=12)
    parser.add_argument("--plus-max-modes", type=int, default=16)
    parser.add_argument("--plus-patch-size", type=int, default=16)
    parser.add_argument("--plus-stride", type=int, default=8)
    parser.add_argument("--plus-local-refine-channels", type=int, default=48)
    parser.add_argument("--include-tcno-reference", action="store_true")
    parser.add_argument("--include-dmno-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--campaign-stage",
        "stage3_flagship",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seeds",
        args.seeds,
        "--models",
        args.models,
        "--output",
        args.output,
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--transport-proxy-weight",
        str(args.transport_proxy_weight),
        "--symbol-proxy-weight",
        str(args.symbol_proxy_weight),
        "--proxy-temperature",
        str(args.proxy_temperature),
        "--plus-window-type",
        args.plus_window_type,
        "--plus-mode-strategy",
        args.plus_mode_strategy,
        "--plus-transport-stencil",
        str(args.plus_transport_stencil),
        "--plus-max-modes",
        str(args.plus_max_modes),
        "--plus-patch-size",
        str(args.plus_patch_size),
        "--plus-stride",
        str(args.plus_stride),
        "--plus-local-refine-channels",
        str(args.plus_local_refine_channels),
    ]
    if args.scenarios.strip():
        command.extend(["--scenarios", args.scenarios])
    if args.include_tcno_reference:
        command.append("--include-tcno-reference")
    if args.include_dmno_reference:
        command.append("--include-dmno-reference")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
