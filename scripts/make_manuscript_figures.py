from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "manuscript" / "jmlr" / "figures"
VIS_DIR = ROOT / "generated" / "experiment_visualizations"


SCENARIO_LABELS = {
    "wave_bicharacteristic_control": "bichar.",
    "wave_chirp_propagation": "chirp",
    "wave_two_packet_no_interaction": "two-packet",
    "wave_variable_speed_nocaustic": "var-speed",
    "helmholtz_local_window_control": "Helm. local",
    "helmholtz_highk_control": "Helm. high-k ctl.",
    "wave_synth": "wave synth",
    "diffusion_neumann_control": "diffusion",
    "helmholtz_variable_control": "Helm. var.",
    "helmholtz_variable_positive": "Helm. var.+",
    "helmholtz_highk_positive": "Helm. high-k",
    "helmholtz_highk_ood_control": "Helm. high-k OOD ctl.",
    "helmholtz_highk_ood_positive": "Helm. high-k OOD",
}

ABLATION_LABELS = {
    "full": "full",
    "full_no_flow_supervision": "no flow sup.",
    "core_only": "core only",
    "no_local_refine": "no local",
    "residual_limited_refine": "res. limited",
    "no_transport": "no transp.",
    "identity_transport": "identity",
    "no_transport_no_carrier": "no tr./car.",
    "single_branch": "single br.",
    "no_branch_routing": "uniform route",
    "no_transported_synthesis": "no tr. synth.",
    "no_transported_input_carrier": "no input carrier",
    "no_landing_decoder": "no landing",
    "randomized_metadata": "rand. meta",
    "no_symbol": "no symbol",
    "source_only_symbol": "src-only sym.",
    "no_edge_symbol": "no edge sym.",
    "no_resolvent_phase": "no res. phase",
    "oracle_transport": "oracle tr.",
    "oracle_symbol": "oracle sym.",
    "fno_plus_same_refine": "FNO+ref",
    "uno_plus_same_refine": "UNO+ref",
    "wno_plus_same_refine": "WNO+ref",
}

BRANCH_ID_ABLATIONS = [
    "full",
    "core_only",
    "no_local_refine",
    "residual_limited_refine",
    "no_transport",
    "identity_transport",
    "randomized_metadata",
    "no_symbol",
    "oracle_transport",
    "oracle_symbol",
    "fno_plus_same_refine",
    "uno_plus_same_refine",
    "wno_plus_same_refine",
]

BRANCH_ID_SCENARIOS = [
    "wave_bicharacteristic_control",
    "wave_variable_speed_nocaustic",
    "helmholtz_local_window_control",
    "wave_synth",
]

BRANCH_ID_V3_CONTROL_ABLATIONS = [
    "full",
    "full_no_flow_supervision",
    "no_transport",
    "identity_transport",
    "no_transport_no_carrier",
    "randomized_metadata",
    "no_symbol",
]

BRANCH_ID_V3_CONTROL_SCENARIOS = [
    "wave_bicharacteristic_control",
    "wave_chirp_propagation",
    "wave_two_packet_no_interaction",
    "wave_variable_speed_nocaustic",
    "wave_synth",
]

HELMHOLTZ_BRANCHED_SCENARIOS = [
    "helmholtz_highk_control",
    "helmholtz_highk_positive",
    "helmholtz_variable_positive",
]

HELMHOLTZ_BRANCHED_ABLATIONS = [
    "full",
    "single_branch",
    "no_branch_routing",
    "no_transport",
    "no_transport_no_carrier",
    "no_symbol",
]
HELMHOLTZ_HIGHK_CAREFUL_SCENARIOS = [
    "helmholtz_highk_control",
    "helmholtz_highk_positive",
    "helmholtz_highk_ood_control",
    "helmholtz_highk_ood_positive",
]
HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS = [
    "full",
    "single_branch",
    "no_branch_routing",
    "no_transport",
    "no_transported_synthesis",
    "no_transported_input_carrier",
    "no_landing_decoder",
    "no_transport_no_carrier",
    "no_symbol",
    "source_only_symbol",
    "no_edge_symbol",
    "no_resolvent_phase",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "nan", "NaN", "None"}:
        return float("nan")
    return float(value)


def _read_json_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("row_type", "run") == "run":
            rows.append(payload)
    return rows


def _write_branch_id_bars(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    scenarios = [
        "wave_bicharacteristic_control",
        "wave_variable_speed_nocaustic",
        "helmholtz_local_window_control",
        "wave_synth",
    ]
    ablations = ["full", "no_transport", "randomized_metadata", "no_symbol"]
    labels = ["Full", "No transport", "Random meta", "No symbol", "Best same-refine"]
    colors = ["#183A37", "#D55E00", "#CC79A7", "#0072B2", "#5F6C37"]

    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    values: list[list[float]] = []
    errors: list[list[float]] = []
    best_baselines: list[float] = []
    best_baseline_err: list[float] = []
    for scenario in scenarios:
        scenario_values = []
        scenario_errors = []
        for ablation in ablations:
            row = lookup[(scenario, ablation)]
            scenario_values.append(_as_float(row, "mean_test_relative_l2"))
            scenario_errors.append(_as_float(row, "std_test_relative_l2"))
        same_refine = [
            lookup[(scenario, ablation)]
            for ablation in ("fno_plus_same_refine", "uno_plus_same_refine", "wno_plus_same_refine")
            if (scenario, ablation) in lookup
        ]
        best = min(same_refine, key=lambda row: _as_float(row, "mean_test_relative_l2"))
        best_baselines.append(_as_float(best, "mean_test_relative_l2"))
        best_baseline_err.append(_as_float(best, "std_test_relative_l2"))
        values.append(scenario_values)
        errors.append(scenario_errors)

    matrix = np.array([row + [best_baselines[i]] for i, row in enumerate(values)])
    err_matrix = np.array([row + [best_baseline_err[i]] for i, row in enumerate(errors)])

    x = np.arange(len(scenarios))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    for idx, label in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            matrix[:, idx],
            width,
            yerr=err_matrix[:, idx],
            label=label,
            color=colors[idx],
            alpha=0.92,
            capsize=2,
            linewidth=0,
        )
    ax.set_ylabel("Relative $\\ell^2$ error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("Completed branch-identifiability test")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_branch_id_geometry_gap(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    scenarios = [
        "wave_bicharacteristic_control",
        "wave_variable_speed_nocaustic",
        "helmholtz_local_window_control",
        "wave_synth",
    ]
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    full_l2 = [_as_float(lookup[(scenario, "full")], "mean_test_relative_l2") for scenario in scenarios]
    random_l2 = [_as_float(lookup[(scenario, "randomized_metadata")], "mean_test_relative_l2") for scenario in scenarios]
    full_flow = [_as_float(lookup[(scenario, "full")], "mean_test_canonical_flow_error") for scenario in scenarios]
    random_flow = [
        _as_float(lookup[(scenario, "randomized_metadata")], "mean_test_canonical_flow_error")
        for scenario in scenarios
    ]

    x = np.arange(len(scenarios))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].bar(x - width / 2, full_l2, width, label="Full", color="#183A37")
    axes[0].bar(x + width / 2, random_l2, width, label="Randomized metadata", color="#CC79A7")
    axes[0].set_title("Field error barely changes")
    axes[0].set_ylabel("Relative $\\ell^2$")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, full_flow, width, label="Full", color="#183A37")
    axes[1].bar(x + width / 2, random_flow, width, label="Randomized metadata", color="#CC79A7")
    axes[1].set_title("Canonical-flow proxy is corrupted")
    axes[1].set_ylabel("Canonical-flow proxy error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], rotation=20, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", frameon=False)
    fig.suptitle("Mechanism gap: metadata corruption is real but not field-deciding", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_branch_id_full_heatmap(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    matrix = np.full((len(BRANCH_ID_ABLATIONS), len(BRANCH_ID_SCENARIOS)), np.nan)
    for i, ablation in enumerate(BRANCH_ID_ABLATIONS):
        for j, scenario in enumerate(BRANCH_ID_SCENARIOS):
            row = lookup.get((scenario, ablation))
            if row is not None:
                matrix[i, j] = _as_float(row, "mean_test_relative_l2")

    fig, ax = plt.subplots(figsize=(8.0, 7.4))
    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(np.nanpercentile(finite_values, 92)) if finite_values.size else 1.0
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=max(vmax, 1e-6), aspect="auto")
    ax.set_xticks(np.arange(len(BRANCH_ID_SCENARIOS)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in BRANCH_ID_SCENARIOS], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(BRANCH_ID_ABLATIONS)))
    ax.set_yticklabels([ABLATION_LABELS[a] for a in BRANCH_ID_ABLATIONS])
    ax.set_title("Full branch-identifiability matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value > 0.55 * max(vmax, 1e-6) else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, "--", ha="center", va="center", fontsize=7, color="gray")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean relative $\\ell^2$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_branch_id_v3_controls_relative_l2(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in BRANCH_ID_V3_CONTROL_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    ablations = [ablation for ablation in BRANCH_ID_V3_CONTROL_ABLATIONS if any((s, ablation) in lookup for s in scenarios)]
    if not ablations:
        return

    matrix = np.full((len(scenarios), len(ablations)), np.nan)
    err_matrix = np.full_like(matrix, np.nan)
    for i, scenario in enumerate(scenarios):
        for j, ablation in enumerate(ablations):
            row = lookup.get((scenario, ablation))
            if row is not None:
                matrix[i, j] = _as_float(row, "mean_test_relative_l2")
                err_matrix[i, j] = _as_float(row, "std_test_relative_l2")

    colors = ["#183A37", "#789262", "#D55E00", "#E69F00", "#7A5195", "#CC79A7", "#0072B2"]
    x = np.arange(len(scenarios))
    width = min(0.12, 0.82 / max(len(ablations), 1))
    fig, ax = plt.subplots(figsize=(12.2, 5.1))
    for idx, ablation in enumerate(ablations):
        offset = (idx - (len(ablations) - 1) / 2) * width
        yerr = err_matrix[:, idx]
        yerr = None if np.all(~np.isfinite(yerr)) else yerr
        ax.bar(
            x + offset,
            matrix[:, idx],
            width,
            yerr=yerr,
            label=ABLATION_LABELS.get(ablation, ablation),
            color=colors[idx % len(colors)],
            alpha=0.92,
            capsize=2,
            linewidth=0,
        )
    ax.set_ylabel("Relative $\\ell^2$ error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=15, ha="right")
    ax.set_title("Registered branch-id v3 controls: field error")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_branch_id_v3_controls_wavefront_proxy(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in BRANCH_ID_V3_CONTROL_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    ablations = [
        ablation
        for ablation in ["full", "full_no_flow_supervision", "no_transport", "identity_transport", "randomized_metadata"]
        if any((s, ablation) in lookup for s in scenarios)
    ]
    if not ablations:
        return

    metrics = [
        ("mean_test_packet_wavefront_localization_error", "Packet-WF localization proxy"),
        ("mean_test_wf_transport_error_proxy", "WF transport proxy"),
        ("mean_test_canonical_flow_error", "Canonical-flow proxy"),
    ]
    colors = ["#183A37", "#789262", "#D55E00", "#E69F00", "#CC79A7"]
    x = np.arange(len(scenarios))
    width = min(0.14, 0.82 / max(len(ablations), 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.8), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        values = np.full((len(scenarios), len(ablations)), np.nan)
        for i, scenario in enumerate(scenarios):
            for j, ablation in enumerate(ablations):
                row = lookup.get((scenario, ablation))
                if row is not None:
                    values[i, j] = _as_float(row, metric)
        for idx, ablation in enumerate(ablations):
            offset = (idx - (len(ablations) - 1) / 2) * width
            ax.bar(
                x + offset,
                values[:, idx],
                width,
                label=ABLATION_LABELS.get(ablation, ablation),
                color=colors[idx % len(colors)],
                alpha=0.92,
                linewidth=0,
            )
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmax(finite) > 20 * max(np.nanmin(finite[finite > 0]) if np.any(finite > 0) else 1e-8, 1e-8):
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(5, len(ablations)), loc="upper center", frameon=False)
    fig.suptitle("Registered branch-id v3 controls: microlocal proxy diagnostics", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_branch_id_v3_controls_key_deltas(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    comparisons = [
        ("no_flow_minus_full", "full_no_flow_supervision", "full"),
        ("no_transport_minus_full", "no_transport", "full"),
        ("identity_minus_full", "identity_transport", "full"),
        ("randomized_minus_full", "randomized_metadata", "full"),
        ("no_carrier_minus_no_transport", "no_transport_no_carrier", "no_transport"),
    ]
    metrics = [
        "mean_test_relative_l2",
        "mean_test_canonical_flow_error",
        "mean_test_packet_wavefront_localization_error",
        "mean_test_wf_transport_error_proxy",
    ]
    out: list[dict[str, object]] = []
    for scenario in BRANCH_ID_V3_CONTROL_SCENARIOS:
        for label, candidate, baseline in comparisons:
            candidate_row = lookup.get((scenario, candidate))
            baseline_row = lookup.get((scenario, baseline))
            if candidate_row is None or baseline_row is None:
                continue
            record: dict[str, object] = {
                "scenario": scenario,
                "comparison": label,
                "candidate": candidate,
                "baseline": baseline,
            }
            for metric in metrics:
                record[f"delta_{metric.replace('mean_test_', '')}"] = _as_float(candidate_row, metric) - _as_float(
                    baseline_row,
                    metric,
                )
                record[f"candidate_{metric.replace('mean_test_', '')}"] = _as_float(candidate_row, metric)
                record[f"baseline_{metric.replace('mean_test_', '')}"] = _as_float(baseline_row, metric)
            out.append(record)
    _write_csv(output_path, out)


def _fmt_cell(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def _write_branch_id_v3_controls_latex_table(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in BRANCH_ID_V3_CONTROL_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    ablations = [
        "full",
        "full_no_flow_supervision",
        "no_transport",
        "identity_transport",
        "no_transport_no_carrier",
        "randomized_metadata",
        "no_symbol",
    ]
    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Registered \texttt{branch\_id\_v3\_controls} field-error checks. Entries are mean relative $\ell^2$ over three seeds; lower is better.}",
        r"\label{tab:branchidv3controls}",
        r"\scriptsize",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Scenario & Full & No flow & No tr. & Identity & No tr./car. & Rand. meta & No sym. \\",
        r"\midrule",
    ]
    body: list[str] = []
    for scenario in scenarios:
        cells = []
        for ablation in ablations:
            row = lookup.get((scenario, ablation))
            cells.append(_fmt_cell(_as_float(row, "mean_test_relative_l2")) if row is not None else "--")
        body.append(f"{SCENARIO_LABELS.get(scenario, scenario)} & " + " & ".join(cells) + r" \\")
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def _write_live_progress(campaign_dir: Path, output_path: Path) -> None:
    rows = _read_json_rows(campaign_dir)
    scenarios = [scenario for scenario in BRANCH_ID_SCENARIOS if any(row.get("scenario") == scenario for row in rows)]
    extra = sorted({str(row.get("scenario")) for row in rows} - set(scenarios))
    scenarios.extend(extra)
    ablations = [ablation for ablation in BRANCH_ID_ABLATIONS if any(row.get("ablation") == ablation for row in rows)]
    if not scenarios or not ablations:
        return
    counts = np.zeros((len(ablations), len(scenarios)), dtype=float)
    means = np.full((len(ablations), len(scenarios)), np.nan)
    for i, ablation in enumerate(ablations):
        for j, scenario in enumerate(scenarios):
            subset = [row for row in rows if row.get("scenario") == scenario and row.get("ablation") == ablation]
            counts[i, j] = len(subset)
            values = [float(row["test_relative_l2"]) for row in subset if "test_relative_l2" in row]
            if values:
                means[i, j] = float(np.mean(values))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.0), constrained_layout=True)
    progress = counts / 3.0
    progress_image = axes[0].imshow(progress, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title(f"Live progress: {campaign_dir.name}")
    axes[0].set_xticks(np.arange(len(scenarios)))
    axes[0].set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(ablations)))
    axes[0].set_yticklabels([ABLATION_LABELS.get(a, a) for a in ablations])
    for i in range(progress.shape[0]):
        for j in range(progress.shape[1]):
            axes[0].text(j, i, f"{int(counts[i, j])}/3", ha="center", va="center", fontsize=7)
    fig.colorbar(progress_image, ax=axes[0], fraction=0.035, pad=0.02, label="fraction complete")

    finite_values = means[np.isfinite(means)]
    vmax = float(np.nanpercentile(finite_values, 92)) if finite_values.size else 1.0
    metric_image = axes[1].imshow(means, cmap="YlOrRd", vmin=0.0, vmax=max(vmax, 1e-6), aspect="auto")
    axes[1].set_title("Provisional mean relative $\\ell^2$")
    axes[1].set_xticks(np.arange(len(scenarios)))
    axes[1].set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=25, ha="right")
    axes[1].set_yticks(np.arange(len(ablations)))
    axes[1].set_yticklabels([])
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            value = means[i, j]
            if np.isfinite(value):
                axes[1].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
            else:
                axes[1].text(j, i, "--", ha="center", va="center", fontsize=7, color="gray")
    fig.colorbar(metric_image, ax=axes[1], fraction=0.035, pad=0.02, label="mean relative $\\ell^2$")
    fig.suptitle("Ongoing experiment visualization; provisional, not a manuscript claim", y=1.02)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_stage2_snapshot(summary_path: Path, output_path: Path) -> None:
    rows = _read_csv(summary_path)
    synthetic_wrapper_path = ROOT / "generated" / "benchmark_synth_official_e16" / "benchmark_summary.csv"
    synthetic_wrapper_rows = _read_csv(synthetic_wrapper_path) if synthetic_wrapper_path.exists() else []
    scenarios = [
        "wave_synth",
        "diffusion_neumann_control",
        "helmholtz_variable_control",
        "helmholtz_highk_positive",
    ]
    models = ["MiNO-Core", "MiNO-Plus", "Best non-MiNO"]
    colors = ["#507DBC", "#183A37", "#D55E00"]
    data: list[list[float]] = []
    errs: list[list[float]] = []
    for scenario in scenarios:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        core = next(row for row in scenario_rows if row["model_variant"] == "MiNO-Core")
        plus = next(row for row in scenario_rows if row["model_variant"] == "MiNO-Plus")
        refs = [
            row
            for row in scenario_rows
            if "MiNO" not in row["model_variant"] and row["source_kind"] in {"tcno_reference", "dmno_reference"}
        ]
        if refs:
            best = min(refs, key=lambda row: _as_float(row, "mean_test_relative_l2"))
        else:
            local_refs = [row for row in scenario_rows if "MiNO" not in row["model_variant"]]
            if not local_refs and scenario == "wave_synth":
                local_refs = [
                    row
                    for row in synthetic_wrapper_rows
                    if row["scenario"] == scenario and "MiNO" not in row["model_variant"]
                ]
            if not local_refs:
                raise ValueError(f"No non-MiNO comparison row available for {scenario}")
            best = min(local_refs, key=lambda row: _as_float(row, "mean_test_relative_l2"))
        data.append(
            [
                _as_float(core, "mean_test_relative_l2"),
                _as_float(plus, "mean_test_relative_l2"),
                _as_float(best, "mean_test_relative_l2"),
            ]
        )
        errs.append(
            [
                _as_float(core, "std_test_relative_l2"),
                _as_float(plus, "std_test_relative_l2"),
                _as_float(best, "std_test_relative_l2"),
            ]
        )

    matrix = np.array(data)
    err_matrix = np.array(errs)
    x = np.arange(len(scenarios))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for idx, label in enumerate(models):
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            matrix[:, idx],
            width,
            yerr=err_matrix[:, idx],
            label=label,
            color=colors[idx],
            alpha=0.92,
            capsize=2,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Relative $\\ell^2$ error, log scale")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("Stage-2 shortlist snapshot: positive regimes and hard-regime gap")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_synthesis_mode_probe(summary_path: Path, output_path: Path, table_path: Path) -> None:
    rows = _read_csv(summary_path)
    scenarios = ["wave_chirp_propagation", "wave_two_packet_no_interaction"]
    modes = ["warp_existing", "atom_splat", "patch_fold", "learned_landing"]
    ablations = ["full", "no_transport", "uno_plus_same_refine"]
    mode_labels = {
        "warp_existing": "warp",
        "atom_splat": "atom splat",
        "patch_fold": "patch fold",
        "learned_landing": "learned landing",
    }
    ablation_labels = {
        "full": "full",
        "no_transport": "no transport",
        "uno_plus_same_refine": "UNO+ref",
    }
    colors = {
        "full": "#183A37",
        "no_transport": "#D55E00",
        "uno_plus_same_refine": "#5F6C37",
    }

    lookup = {(row["mode"], row["scenario"], row["ablation"]): row for row in rows}
    fig, axes = plt.subplots(1, len(scenarios), figsize=(11.2, 4.4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    width = 0.23
    x = np.arange(len(modes))
    for ax, scenario in zip(axes, scenarios):
        for idx, ablation in enumerate(ablations):
            values = []
            for mode in modes:
                row = lookup.get((mode, scenario, ablation))
                values.append(_as_float(row, "rel_l2") if row is not None else float("nan"))
            ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                label=ablation_labels[ablation],
                color=colors[ablation],
                alpha=0.92,
            )
        ax.set_title(SCENARIO_LABELS[scenario])
        ax.set_xticks(x)
        ax.set_xticklabels([mode_labels[mode] for mode in modes], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Relative $\\ell^2$ error")
    fig.suptitle("Executable landing probe: analytic landing fails, learned finite landing narrows the gap", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    selected_modes = ["warp_existing", "atom_splat", "patch_fold", "learned_landing"]
    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Single-seed executable landing probe.  This table is diagnostic, not a replacement for the registered three-seed mechanism tables.  It tests whether the remaining UNO gap is partly a finite landing problem rather than a canonical-flow problem.}",
        r"\label{tab:synthesis-mode-probe}",
        r"\small",
        r"\begin{tabular}{p{0.23\textwidth}p{0.18\textwidth}ccc}",
        r"\toprule",
        r"Scenario & Landing map & Full & No transport & UNO+ref \\",
        r"\midrule",
    ]
    body: list[str] = []
    for scenario in scenarios:
        for mode in selected_modes:
            values = []
            for ablation in ablations:
                row = lookup.get((mode, scenario, ablation))
                values.append(_as_float(row, "rel_l2") if row is not None else float("nan"))
            scenario_label = SCENARIO_LABELS[scenario] if mode == selected_modes[0] else ""
            body.append(
                "{} & {} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                    scenario_label,
                    mode_labels[mode],
                    values[0],
                    values[1],
                    values[2],
                )
            )
        if scenario != scenarios[-1]:
            body.append(r"\addlinespace")
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def _write_learned_landing_controls(summary_path: Path, output_path: Path, table_path: Path) -> None:
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in BRANCH_ID_V3_CONTROL_SCENARIOS if (scenario, "full") in lookup]
    ablations = [
        "full",
        "full_no_flow_supervision",
        "no_transport",
        "no_transport_no_carrier",
        "randomized_metadata",
        "no_symbol",
    ]
    labels = ["Full", "No flow sup.", "No transport", "No tr./car.", "Rand. meta", "No symbol"]
    colors = ["#183A37", "#56B4E9", "#D55E00", "#9B2226", "#CC79A7", "#0072B2"]
    if not scenarios:
        return

    matrix: list[list[float]] = []
    err_matrix: list[list[float]] = []
    for scenario in scenarios:
        values = []
        errors = []
        for ablation in ablations:
            row = lookup.get((scenario, ablation))
            values.append(_as_float(row, "mean_test_relative_l2") if row is not None else float("nan"))
            errors.append(_as_float(row, "std_test_relative_l2") if row is not None else float("nan"))
        matrix.append(values)
        err_matrix.append(errors)

    values_np = np.array(matrix, dtype=float)
    errors_np = np.array(err_matrix, dtype=float)
    x = np.arange(len(scenarios))
    width = 0.13
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    for idx, label in enumerate(labels):
        ax.bar(
            x + (idx - (len(labels) - 1) / 2) * width,
            values_np[:, idx],
            width,
            yerr=errors_np[:, idx],
            label=label,
            color=colors[idx],
            alpha=0.92,
            capsize=2,
            linewidth=0,
        )
    ax.set_ylabel("Relative $\\ell^2$ error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_title("Three-seed learned finite-landing controls")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Three-seed learned finite-landing controls.  Entries are mean relative $\ell^2$; lower is better.  The transport-conditioned decoder reduces absolute field error relative to the analytic finite-landing maps, while transport removal and carrier removal remain visible.}",
        r"\label{tab:learned-landing-controls}",
        r"\scriptsize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Scenario & Full & No flow & No tr. & No tr./car. & Rand. meta & No sym. \\",
        r"\midrule",
    ]
    body: list[str] = []
    for scenario, row_values in zip(scenarios, values_np):
        body.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
                SCENARIO_LABELS[scenario],
                *row_values,
            )
        )
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def _best_reference_by_scenario(stage2_summary_path: Path) -> dict[str, tuple[float, str]]:
    if not stage2_summary_path.exists():
        return {}
    rows = _read_csv(stage2_summary_path)
    out: dict[str, tuple[float, str]] = {}
    for scenario in {row.get("scenario", "") for row in rows}:
        candidates = [
            row
            for row in rows
            if row.get("scenario") == scenario
            and "MiNO" not in row.get("model_variant", "")
            and np.isfinite(_as_float(row, "mean_test_relative_l2"))
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: _as_float(row, "mean_test_relative_l2"))
        out[scenario] = (_as_float(best, "mean_test_relative_l2"), best.get("model_variant", "reference"))
    return out


def _write_helmholtz_branched_bars(summary_path: Path, stage2_summary_path: Path, output_path: Path, table_path: Path) -> None:
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in HELMHOLTZ_BRANCHED_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    ablations = [ablation for ablation in HELMHOLTZ_BRANCHED_ABLATIONS if any((s, ablation) in lookup for s in scenarios)]
    refs = _best_reference_by_scenario(stage2_summary_path)
    labels = [ABLATION_LABELS.get(ablation, ablation) for ablation in ablations]
    colors = ["#183A37", "#507DBC", "#E69F00", "#D55E00", "#9B2226", "#0072B2"]
    x = np.arange(len(scenarios))
    width = min(0.12, 0.75 / max(len(ablations), 1))
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    for index, ablation in enumerate(ablations):
        values = []
        errors = []
        for scenario in scenarios:
            row = lookup.get((scenario, ablation))
            values.append(_as_float(row, "mean_test_relative_l2") if row is not None else float("nan"))
            errors.append(_as_float(row, "std_test_relative_l2") if row is not None else float("nan"))
        ax.bar(
            x + (index - (len(ablations) - 1) / 2) * width,
            values,
            width,
            yerr=errors,
            label=labels[index],
            color=colors[index % len(colors)],
            alpha=0.92,
            capsize=2,
            linewidth=0,
        )
    for idx, scenario in enumerate(scenarios):
        if scenario in refs:
            value, label = refs[scenario]
            ax.hlines(value, idx - 0.46, idx + 0.46, color="#111111", linestyles="dashed", linewidth=1.2)
            ax.text(idx, value, f"best ref. {value:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Relative $\\ell^2$ error (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=12, ha="right")
    ax.set_title("Branched MiNO high-$k$ Helmholtz diagnostic")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    header = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Branched MiNO high-$k$ Helmholtz diagnostic. Entries are mean relative $\ell^2$ over available seeds under the local sample-capped protocol; lower is better. The table tests whether finite canonical-branch structure reduces the high-$k$ gap at this budget, not a global Helmholtz resolvent theorem or full-budget benchmark.}",
        r"\label{tab:helmholtz-branched-highk}",
        r"\scriptsize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Scenario & Full & Single br. & Uniform route & No tr. & No tr./car. & No sym. \\",
        r"\midrule",
    ]
    body: list[str] = []
    for scenario in scenarios:
        cells = []
        for ablation in HELMHOLTZ_BRANCHED_ABLATIONS:
            row = lookup.get((scenario, ablation))
            cells.append(_fmt_cell(_as_float(row, "mean_test_relative_l2")) if row is not None else "--")
        body.append(f"{SCENARIO_LABELS.get(scenario, scenario)} & " + " & ".join(cells) + r" \\")
    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(header + body + footer) + "\n", encoding="utf-8")


def _write_helmholtz_branch_diagnostics(summary_path: Path, output_path: Path) -> None:
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in HELMHOLTZ_BRANCHED_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    metrics = [
        ("mean_test_branch_entropy", "Branch entropy"),
        ("mean_test_branch_diversity", "Branch diversity"),
        ("mean_test_high_frequency_relative_error_proxy", "High-frequency error"),
    ]
    ablations = [ablation for ablation in ("full", "single_branch", "no_branch_routing") if any((s, ablation) in lookup for s in scenarios)]
    colors = ["#183A37", "#507DBC", "#E69F00"]
    x = np.arange(len(scenarios))
    width = min(0.20, 0.75 / max(len(ablations), 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(14.5, 4.5), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for index, ablation in enumerate(ablations):
            values = []
            for scenario in scenarios:
                row = lookup.get((scenario, ablation))
                values.append(_as_float(row, metric) if row is not None else float("nan"))
            ax.bar(
                x + (index - (len(ablations) - 1) / 2) * width,
                values,
                width,
                label=ABLATION_LABELS.get(ablation, ablation),
                color=colors[index % len(colors)],
                alpha=0.92,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(ablations), loc="upper center", frameon=False)
    fig.suptitle("Branched Helmholtz routing diagnostics", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_helmholtz_highk_careful_bars(
    summary_path: Path,
    stage2_summary_path: Path,
    output_path: Path,
    table_path: Path,
) -> None:
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in HELMHOLTZ_HIGHK_CAREFUL_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    ablations = [
        ablation
        for ablation in HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS
        if any((scenario, ablation) in lookup for scenario in scenarios)
    ]
    refs = _best_reference_by_scenario(stage2_summary_path)
    labels = [ABLATION_LABELS.get(ablation, ablation) for ablation in ablations]
    colors = ["#183A37", "#507DBC", "#E69F00", "#D55E00", "#6A4C93", "#008080", "#7F5539", "#9B2226", "#0072B2"]
    x = np.arange(len(scenarios))
    width = min(0.09, 0.72 / max(len(ablations) + 1, 1))

    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(scenarios)), 4.5))
    for idx, ablation in enumerate(ablations):
        values = [_as_float(lookup.get((scenario, ablation)), "mean_test_relative_l2") for scenario in scenarios]
        ax.bar(
            x + (idx - len(ablations) / 2) * width,
            values,
            width=width,
            color=colors[idx % len(colors)],
            label=labels[idx],
        )
    ref_values = [refs.get(scenario, math.nan) for scenario in scenarios]
    ax.scatter(
        x + (len(ablations) / 2) * width,
        ref_values,
        marker="D",
        color="#111111",
        label="best imported ref.",
        zorder=4,
    )
    ax.set_ylabel("relative $\\ell^2$")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=12, ha="right")
    ax.set_title("Careful high-$k$ Helmholtz attribution")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    table_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Scenario", *[ABLATION_LABELS.get(ablation, ablation) for ablation in HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS]]
    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Careful high-$k$ Helmholtz attribution campaign. Entries are mean relative $\ell^2$ over available seeds; lower is better. This table separates branch routing, transported synthesis, input carrier, learned landing, transport, and symbol paths.}",
        r"\label{tab:helmholtz-highk-careful}",
        r"\scriptsize",
        r"\begin{tabular}{l" + "c" * len(HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS) + r"}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        cells = []
        for ablation in HELMHOLTZ_HIGHK_CAREFUL_ABLATIONS:
            row = lookup.get((scenario, ablation))
            cells.append(_fmt_cell(_as_float(row, "mean_test_relative_l2")) if row is not None else "--")
        body.append(f"{SCENARIO_LABELS.get(scenario, scenario)} & " + " & ".join(cells) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    table_path.write_text("\n".join(body), encoding="utf-8")


def _write_helmholtz_highk_careful_diagnostics(summary_path: Path, output_path: Path) -> None:
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    lookup = {(row["scenario"], row["ablation"]): row for row in rows}
    scenarios = [scenario for scenario in HELMHOLTZ_HIGHK_CAREFUL_SCENARIOS if (scenario, "full") in lookup]
    if not scenarios:
        return
    metrics = [
        ("mean_test_branch_entropy", "Branch entropy"),
        ("mean_test_branch_diversity", "Branch diversity"),
        ("mean_test_branch_spread", "Branch spread"),
        ("mean_test_high_frequency_relative_error_proxy", "High-frequency error"),
    ]
    ablations = [
        ablation
        for ablation in ("full", "single_branch", "no_branch_routing", "no_transport")
        if any((scenario, ablation) in lookup for scenario in scenarios)
    ]
    colors = ["#183A37", "#507DBC", "#E69F00", "#D55E00"]
    x = np.arange(len(scenarios))
    width = min(0.18, 0.75 / max(len(ablations), 1))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.5), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for idx, ablation in enumerate(ablations):
            values = [_as_float(lookup.get((scenario, ablation)), metric) for scenario in scenarios]
            ax.bar(
                x + (idx - (len(ablations) - 1) / 2) * width,
                values,
                width=width,
                color=colors[idx % len(colors)],
                label=ABLATION_LABELS.get(ablation, ablation),
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(s, s.replace("_", " ")) for s in scenarios], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(ablations), loc="upper center", frameon=False)
    fig.suptitle("Careful high-$k$ Helmholtz routing/carrier diagnostics", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from completed MiNO result artifacts.")
    parser.add_argument("--output-dir", default=str(FIGURE_DIR))
    parser.add_argument(
        "--branch-id-summary",
        default=str(
            ROOT
            / "generated"
            / "empirical_closure"
            / "branch_id_v2_core_first"
            / "empirical_closure_summary.csv"
        ),
    )
    parser.add_argument(
        "--stage2-summary",
        default=str(ROOT / "generated" / "benchmark_stage2_shortlist_v1" / "benchmark_summary.csv"),
    )
    parser.add_argument(
        "--branch-id-v3-controls-summary",
        default=str(
            ROOT
            / "generated"
            / "empirical_closure"
            / "branch_id_v3_controls"
            / "empirical_closure_summary.csv"
        ),
    )
    parser.add_argument(
        "--synthesis-mode-summary",
        default=str(ROOT / "generated" / "empirical_closure" / "synthesis_mode_probe_summary.csv"),
    )
    parser.add_argument(
        "--learned-landing-summary",
        default=str(
            ROOT
            / "generated"
            / "empirical_closure"
            / "learned_landing_controls"
            / "empirical_closure_summary.csv"
        ),
    )
    parser.add_argument(
        "--helmholtz-branched-summary",
        default=str(
            ROOT
            / "generated"
            / "empirical_closure"
            / "helmholtz_branched_highk"
            / "empirical_closure_summary.csv"
        ),
    )
    parser.add_argument(
        "--helmholtz-careful-summary",
        default=str(
            ROOT
            / "generated"
            / "empirical_closure"
            / "helmholtz_highk_careful"
            / "empirical_closure_summary.csv"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_summary = Path(args.branch_id_summary)
    stage2_summary = Path(args.stage2_summary)
    branch_v3_controls_summary = Path(args.branch_id_v3_controls_summary)
    synthesis_mode_summary = Path(args.synthesis_mode_summary)
    learned_landing_summary = Path(args.learned_landing_summary)
    helmholtz_branched_summary = Path(args.helmholtz_branched_summary)
    helmholtz_careful_summary = Path(args.helmholtz_careful_summary)

    _write_branch_id_bars(branch_summary, output_dir / "branch_id_v2_mechanism_bars.png")
    _write_branch_id_geometry_gap(branch_summary, output_dir / "branch_id_v2_geometry_gap.png")
    _write_branch_id_full_heatmap(branch_summary, output_dir / "branch_id_v2_full_heatmap.png")
    if branch_v3_controls_summary.exists():
        _write_branch_id_v3_controls_relative_l2(
            branch_v3_controls_summary,
            output_dir / "branch_id_v3_controls_relative_l2.png",
        )
        _write_branch_id_v3_controls_wavefront_proxy(
            branch_v3_controls_summary,
            output_dir / "branch_id_v3_controls_wavefront_proxy.png",
        )
        _write_branch_id_v3_controls_key_deltas(
            branch_v3_controls_summary,
            VIS_DIR / "branch_id_v3_controls_key_deltas.csv",
        )
        _write_branch_id_v3_controls_latex_table(
            branch_v3_controls_summary,
            ROOT / "manuscript" / "jmlr" / "sections" / "branch_id_v3_controls_table.tex",
        )
    if synthesis_mode_summary.exists():
        _write_synthesis_mode_probe(
            synthesis_mode_summary,
            output_dir / "synthesis_mode_probe_relative_l2.png",
            ROOT / "manuscript" / "jmlr" / "sections" / "synthesis_mode_probe_table.tex",
        )
    if learned_landing_summary.exists():
        _write_learned_landing_controls(
            learned_landing_summary,
            output_dir / "learned_landing_controls_relative_l2.png",
            ROOT / "manuscript" / "jmlr" / "sections" / "learned_landing_controls_table.tex",
        )
    if helmholtz_branched_summary.exists():
        _write_helmholtz_branched_bars(
            helmholtz_branched_summary,
            stage2_summary,
            output_dir / "helmholtz_branched_highk_relative_l2.png",
            ROOT / "manuscript" / "jmlr" / "sections" / "helmholtz_branched_highk_table.tex",
        )
        _write_helmholtz_branch_diagnostics(
            helmholtz_branched_summary,
            output_dir / "helmholtz_branched_highk_routing_diagnostics.png",
        )
    if helmholtz_careful_summary.exists():
        _write_helmholtz_highk_careful_bars(
            helmholtz_careful_summary,
            stage2_summary,
            output_dir / "helmholtz_highk_careful_relative_l2.png",
            ROOT / "manuscript" / "jmlr" / "sections" / "helmholtz_highk_careful_table.tex",
        )
        _write_helmholtz_highk_careful_diagnostics(
            helmholtz_careful_summary,
            output_dir / "helmholtz_highk_careful_diagnostics.png",
        )
    _write_stage2_snapshot(stage2_summary, output_dir / "stage2_regime_snapshot.png")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "branch_id_v2_spectral_binding",
        VIS_DIR / "branch_id_v2_spectral_binding_live.png",
    )
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "branch_id_v3_binding",
        VIS_DIR / "branch_id_v3_binding_live.png",
    )
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "branch_id_v3_carrier",
        VIS_DIR / "branch_id_v3_carrier_live.png",
    )
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "branch_id_v3_carrier_final",
        VIS_DIR / "branch_id_v3_carrier_final_live.png",
    )
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "branch_id_v3_controls",
        VIS_DIR / "branch_id_v3_controls_live.png",
    )
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "helmholtz_branched_highk",
        VIS_DIR / "helmholtz_branched_highk_live.png",
    )
    _write_live_progress(
        ROOT / "generated" / "empirical_closure" / "helmholtz_highk_careful",
        VIS_DIR / "helmholtz_highk_careful_live.png",
    )
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
