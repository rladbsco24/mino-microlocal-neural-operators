from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from mino.data.benchmark import build_benchmark_loaders
from mino.metrics.wavefront import phase_error, relative_l2
from mino.models.mino import build_model
from mino.training.train import fit_model


def _limit_loader(loader: DataLoader, limit: int, *, shuffle: bool) -> DataLoader:
    if limit <= 0 or limit >= len(loader.dataset):
        return loader
    return DataLoader(Subset(loader.dataset, range(limit)), batch_size=loader.batch_size, shuffle=shuffle)


def _field2d(tensor: torch.Tensor, *, channel: int = 0) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor[0, channel].detach().cpu()
    if tensor.ndim == 3:
        return tensor[channel].detach().cpu()
    if tensor.ndim == 2:
        return tensor.detach().cpu()
    raise ValueError(f"Unsupported tensor shape for plotting: {tuple(tensor.shape)}")


def _mino_plus_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "width": args.width,
        "depth": args.depth,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "max_modes": args.max_modes,
        "window_type": "gaussian",
        "mode_strategy": "shell_balanced",
        "transport_scale": 0.03,
        "transport_stencil": args.transport_stencil,
        "local_refine_channels": 32,
        "local_refine_scale": args.local_refine_scale,
        "route_bias_init": -4.0,
        "refine_lowpass_cutoff": 0.25,
        "transport_highpass_cutoff": 0.25,
        "transport_parameterization": "hamiltonian_verlet",
        "sparse_topk": True,
        "frame_type": args.frame_type,
        "symbol_parameterization": args.symbol_parameterization,
        "wavefront_confidence_scale": args.wavefront_confidence_scale,
        "skip_lowpass_cutoff": args.skip_lowpass_cutoff,
        "transported_synthesis_scale": args.transported_synthesis_scale,
        "transported_input_scale": args.transported_input_scale,
        "transported_decoder_channels": args.transported_decoder_channels,
        "transported_decoder_scale": args.transported_decoder_scale,
        "token_refine_scale": args.token_refine_scale,
        "num_canonical_branches": args.num_canonical_branches,
        "branch_routing": args.branch_routing,
        "branch_prior_strength": args.branch_prior_strength,
        "branch_entropy_weight": args.branch_entropy_weight,
        "branch_diversity_weight": args.branch_diversity_weight,
    }


def _plot_panel(ax: plt.Axes, image: torch.Tensor, title: str, cmap: str, *, symmetric: bool = False) -> None:
    array = image.numpy()
    if symmetric:
        limit = float(max(abs(array.min()), abs(array.max()), 1e-8))
        handle = ax.imshow(array, cmap=cmap, vmin=-limit, vmax=limit)
    else:
        handle = ax.imshow(array, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate one Helmholtz prediction figure.")
    parser.add_argument("--scenario", default="helmholtz_local_window_control")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-modes", type=int, default=16)
    parser.add_argument("--transport-stencil", type=int, default=12)
    parser.add_argument("--local-refine-scale", type=float, default=0.15)
    parser.add_argument(
        "--frame-type",
        default="gabor_gaussian",
        choices=["gabor_gaussian", "multiscale_gabor", "anisotropic_gabor", "directional_gabor"],
    )
    parser.add_argument(
        "--symbol-parameterization",
        default="helmholtz_resolvent",
        choices=["spectral_order", "helmholtz_resolvent"],
    )
    parser.add_argument("--wavefront-confidence-scale", type=float, default=2.0)
    parser.add_argument("--skip-lowpass-cutoff", type=float, default=0.16)
    parser.add_argument("--transported-synthesis-scale", type=float, default=1.0)
    parser.add_argument("--transported-input-scale", type=float, default=1.0)
    parser.add_argument("--transported-decoder-channels", type=int, default=16)
    parser.add_argument("--transported-decoder-scale", type=float, default=1.0)
    parser.add_argument("--token-refine-scale", type=float, default=0.0)
    parser.add_argument("--num-canonical-branches", type=int, default=3)
    parser.add_argument(
        "--branch-routing",
        default="metadata_softmax",
        choices=["metadata_softmax", "metadata_frequency_softmax", "frequency_softmax", "uniform", "single"],
    )
    parser.add_argument("--branch-prior-strength", type=float, default=0.0)
    parser.add_argument("--branch-entropy-weight", type=float, default=0.01)
    parser.add_argument("--branch-diversity-weight", type=float, default=0.01)
    parser.add_argument("--ablation", default="full", choices=["full", "single_branch"])
    parser.add_argument("--output-dir", default="generated/figures/helmholtz_single_restore")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    loaders = build_benchmark_loaders(args.scenario, batch_size=args.batch_size, seed=args.seed)
    loaders = type(loaders)(
        train_loader=_limit_loader(loaders.train_loader, args.max_train_samples, shuffle=True),
        val_loader=_limit_loader(loaders.val_loader, args.max_val_samples, shuffle=False),
        test_loader=_limit_loader(loaders.test_loader, args.max_test_samples, shuffle=False),
        in_channels=loaders.in_channels,
        out_channels=loaders.out_channels,
        spatial_shape=loaders.spatial_shape,
        spec=loaders.spec,
    )
    model = build_model(
        "MiNO-Plus",
        in_channels=loaders.in_channels,
        out_channels=loaders.out_channels,
        model_kwargs=_mino_plus_kwargs(args),
    ).to(device)
    if args.ablation == "single_branch":
        model.core.num_canonical_branches = 1
        for block in model.core.blocks:
            if hasattr(block, "branch_routing"):
                block.branch_routing = "single"

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
        transport_proxy_weight=0.10,
        symbol_proxy_weight=0.05,
        proxy_temperature=0.05,
        core_field_weight=1.0,
        residual_energy_weight=0.05,
        route_l1_weight=0.01,
        canonical_loss_weight=0.01,
        symbol_order_loss_weight=0.005,
        symbol_order_target=0.0,
        packet_space_loss_weight=0.01,
        highfreq_core_loss_weight=0.05,
        highfreq_cutoff=args.skip_lowpass_cutoff,
        branch_entropy_weight=args.branch_entropy_weight,
        branch_diversity_weight=args.branch_diversity_weight,
        core_warmup_epochs=max(1, args.epochs // 4),
        freeze_refinement_epochs=max(1, args.epochs // 4),
    )

    test_dataset = loaders.test_loader.dataset
    sample_index = max(0, min(args.sample_index, len(test_dataset) - 1))
    inputs, targets = test_dataset[sample_index]
    inputs = inputs.unsqueeze(0).to(device)
    targets = targets.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        diagnostics = model.forward_with_diagnostics(inputs)
        prediction = diagnostics["prediction"]
        core_prediction = diagnostics.get("core_prediction", prediction)
        refine_correction = diagnostics.get("refine_correction", torch.zeros_like(prediction))

    rel = float(relative_l2(prediction, targets).mean().item())
    phase = float(phase_error(prediction, targets).mean().item())
    error = (prediction - targets).abs()
    core_error = (core_prediction - targets).abs()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_seed{args.seed}_sample{sample_index}_{args.frame_type}_{args.ablation}_L{args.num_canonical_branches}"
    figure_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    _plot_panel(axes[0, 0], _field2d(inputs), "input/source ch0", "viridis")
    _plot_panel(axes[0, 1], _field2d(targets), "target Helmholtz field", "RdBu_r", symmetric=True)
    _plot_panel(axes[0, 2], _field2d(prediction), f"MiNO-Plus prediction\nrel L2={rel:.3f}", "RdBu_r", symmetric=True)
    _plot_panel(axes[1, 0], _field2d(error), "|prediction-target|", "magma")
    _plot_panel(axes[1, 1], _field2d(core_prediction), "core prediction", "RdBu_r", symmetric=True)
    _plot_panel(axes[1, 2], _field2d(refine_correction), "low-pass refinement", "RdBu_r", symmetric=True)
    fig.suptitle(f"{args.scenario}: regenerated {args.ablation} Helmholtz figure", fontsize=11)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    payload = {
        "scenario": args.scenario,
        "seed": args.seed,
        "epochs": args.epochs,
        "sample_index": sample_index,
        "frame_type": args.frame_type,
        "ablation": args.ablation,
        "num_canonical_branches": args.num_canonical_branches,
        "model": "MiNO-Plus",
        "relative_l2": rel,
        "phase_error": phase,
        "runtime_seconds": history["runtime_seconds"],
        "figure": str(figure_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
