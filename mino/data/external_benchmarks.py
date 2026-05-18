from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalBenchmarkSpec:
    """Curated external PDE benchmark target.

    This registry is deliberately metadata-only: it does not download data or
    change the executed MiNO experiments.  Its role is to make the paper's next
    benchmark targets explicit and reproducible.
    """

    pde_family: str
    benchmark_id: str
    suite: str
    dataset_or_task: str
    source_url: str
    paper_or_release: str
    primary_regime: str
    difficulty_tags: tuple[str, ...]
    canonical_mino_relevance: str
    recommended_role: str
    loader_status: str
    local_scenario_hint: str
    baseline_targets: tuple[str, ...]
    priority: str
    notes: str = ""


PDEBENCH_URL = "https://github.com/pdebench/PDEBench"
PDEARENA_URL = "https://github.com/pdearena/pdearena"
THE_WELL_URL = "https://github.com/PolymathicAI/the_well"
POSEIDON_URL = "https://github.com/camlab-ethz/poseidon"
RPB_URL = "https://zenodo.org/records/10406879"
MGCFNN_URL = "https://openreview.net/forum?id=ThhQyIruEs"
HELMHOLTZ_DIFFUSION_URL = "https://arxiv.org/abs/2602.04082"
HP_FEM_POLLUTION_URL = "https://arxiv.org/abs/2202.06939"


_BENCHMARKS: tuple[ExternalBenchmarkSpec, ...] = (
    ExternalBenchmarkSpec(
        pde_family="diffusion_reaction",
        benchmark_id="pdebench_2d_diffusion_reaction",
        suite="PDEBench",
        dataset_or_task="2D diffusion-reaction",
        source_url=PDEBENCH_URL,
        paper_or_release="PDEBench, NeurIPS 2022",
        primary_regime="parabolic / dissipative symbol",
        difficulty_tags=("stiff reaction", "time-dependent", "parameter variation"),
        canonical_mino_relevance="Tests identity-canonical dissipative-symbol branch rather than canonical transport.",
        recommended_role="main external diffusion benchmark",
        loader_status="adapter_required",
        local_scenario_hint="diffusion_neumann_control,diffusion_neumann_positive",
        baseline_targets=("FNO", "U-Net", "PINN", "WNO/UNO if locally implemented"),
        priority="P0",
        notes="Use to validate the diffusion success outside the local TCNO cache.",
    ),
    ExternalBenchmarkSpec(
        pde_family="poisson_elliptic",
        benchmark_id="poseidon_poisson_gauss",
        suite="Poseidon downstream tasks",
        dataset_or_task="Poisson-Gauss",
        source_url=POSEIDON_URL,
        paper_or_release="Poseidon, NeurIPS 2024",
        primary_regime="elliptic / identity-canonical pseudodifferential",
        difficulty_tags=("elliptic smoothing", "foundation-model downstream", "operator norm"),
        canonical_mino_relevance="Tests whether finite S^m/PDO identity branch beats spectral/local baselines.",
        recommended_role="main Poisson benchmark",
        loader_status="adapter_required",
        local_scenario_hint="poisson_robin_control,poisson_robin_positive",
        baseline_targets=("Poseidon", "scOT", "CNO", "FNO", "UNO", "WNO"),
        priority="P1",
        notes="Current local Poisson results trail WNO; this is a calibration target, not a claim target yet.",
    ),
    ExternalBenchmarkSpec(
        pde_family="wave",
        benchmark_id="poseidon_wave_layer_gauss",
        suite="Poseidon downstream tasks",
        dataset_or_task="Wave-Layer / Wave-Gauss",
        source_url=POSEIDON_URL,
        paper_or_release="Poseidon, NeurIPS 2024",
        primary_regime="hyperbolic / finite-speed canonical transport",
        difficulty_tags=("propagating wavefronts", "layered medium", "time-dependent"),
        canonical_mino_relevance="Closest external match to carrier-bound transport and packet-wavefront diagnostics.",
        recommended_role="main external wave benchmark",
        loader_status="adapter_required",
        local_scenario_hint="wave_bicharacteristic_control,wave_chirp_propagation,wave_two_packet_no_interaction",
        baseline_targets=("Poseidon", "scOT", "CNO", "FNO", "UNO"),
        priority="P0",
        notes="Use after local branch_id_v3/learned-landing controls are stable.",
    ),
    ExternalBenchmarkSpec(
        pde_family="helmholtz",
        benchmark_id="poseidon_helmholtz",
        suite="Poseidon downstream tasks",
        dataset_or_task="Helmholtz",
        source_url=POSEIDON_URL,
        paper_or_release="Poseidon, NeurIPS 2024",
        primary_regime="frequency-domain wave / outgoing FIO branches",
        difficulty_tags=("high-frequency", "standing/outgoing waves", "branching risk"),
        canonical_mino_relevance="Direct stress test for finite-branch canonical relation and anisotropic packets.",
        recommended_role="generic Helmholtz benchmark before specialized high-k solver comparison",
        loader_status="adapter_required",
        local_scenario_hint="helmholtz_highk_control,helmholtz_highk_positive,helmholtz_variable_positive",
        baseline_targets=("Poseidon", "scOT", "CNO", "UNO", "FNO"),
        priority="P0",
        notes="Use for public comparability; do not promote a flagship high-k claim from this row alone.",
    ),
    ExternalBenchmarkSpec(
        pde_family="helmholtz",
        benchmark_id="mgcfnn_high_wavenumber_helmholtz",
        suite="ICLR 2025 specialized Helmholtz solver",
        dataset_or_task="High-wavenumber heterogeneous Helmholtz equations",
        source_url=MGCFNN_URL,
        paper_or_release="MGCFNN, ICLR 2025",
        primary_regime="high-k Helmholtz / neural multigrid-Fourier solver",
        difficulty_tags=("high-wavenumber", "heterogeneous medium", "specialized solver"),
        canonical_mino_relevance="Specialized baseline required before claiming high-k Helmholtz competitiveness.",
        recommended_role="flagship high-k specialized baseline",
        loader_status="adapter_required",
        local_scenario_hint="helmholtz_highk_flagship",
        baseline_targets=("MGCFNN", "FNO", "multigrid/Fourier neural solver references"),
        priority="P0",
        notes="A MiNO high-k claim should report whether anisotropic carrier-bound resolvent reconstruction closes this specialized-solver gap.",
    ),
    ExternalBenchmarkSpec(
        pde_family="helmholtz",
        benchmark_id="probabilistic_high_frequency_helmholtz",
        suite="High-frequency Helmholtz diffusion benchmark",
        dataset_or_task="Conditional diffusion operator for high-frequency Helmholtz",
        source_url=HELMHOLTZ_DIFFUSION_URL,
        paper_or_release="Zou, Lanthaler, Salahshoor 2026",
        primary_regime="high-frequency Helmholtz / probabilistic operator learning",
        difficulty_tags=("phase sensitivity", "uncertainty propagation", "spectral bias"),
        canonical_mino_relevance="Tests whether deterministic packet transport preserves phase coherence against probabilistic high-k solvers.",
        recommended_role="flagship high-k probabilistic baseline",
        loader_status="adapter_required",
        local_scenario_hint="helmholtz_highk_flagship",
        baseline_targets=("conditional diffusion operator", "FNO", "UNO", "WNO"),
        priority="P0",
        notes="Use phase error, energy/H1-style error, and uncertainty-aware metrics separately; do not compare only relative L2.",
    ),
    ExternalBenchmarkSpec(
        pde_family="helmholtz",
        benchmark_id="hp_fem_pollution_reference",
        suite="Classical high-frequency Helmholtz reference",
        dataset_or_task="hp-FEM pollution-effect scaling reference",
        source_url=HP_FEM_POLLUTION_URL,
        paper_or_release="Spence 2023 / hp-FEM pollution-effect reference",
        primary_regime="high-k Helmholtz / DOF scaling",
        difficulty_tags=("pollution effect", "wavenumber scaling", "classical numerical reference"),
        canonical_mino_relevance="Defines the scaling language for any pollution-resistant learned representation claim.",
        recommended_role="scaling yardstick, not a neural baseline",
        loader_status="metadata_only",
        local_scenario_hint="helmholtz_highk_flagship",
        baseline_targets=("hp-FEM scaling law", "reference solver residual"),
        priority="P1",
        notes="A learned method should report kL, points per wavelength, residual accuracy, and error-vs-k degradation before using pollution-resistant language.",
    ),
    ExternalBenchmarkSpec(
        pde_family="acoustic_scattering",
        benchmark_id="the_well_acoustic_scattering",
        suite="The Well",
        dataset_or_task="acoustic scattering dataset",
        source_url=THE_WELL_URL,
        paper_or_release="The Well, NeurIPS 2024 Datasets and Benchmarks",
        primary_regime="wave scattering / oscillatory field prediction",
        difficulty_tags=("large-scale", "scattering", "realistic simulator", "dataset scale"),
        canonical_mino_relevance="Natural external benchmark for microlocal scattering beyond local Helmholtz caches.",
        recommended_role="large-scale follow-up benchmark",
        loader_status="adapter_required",
        local_scenario_hint="helmholtz_local_window_control,helmholtz_highk_positive",
        baseline_targets=("The-Well FNO configs", "The-Well benchmark models", "Poseidon if compatible"),
        priority="P1",
        notes="The full collection is large; prefer streaming/small split first.",
    ),
    ExternalBenchmarkSpec(
        pde_family="navier_stokes_incompressible",
        benchmark_id="poseidon_ns_sines_gauss_sl",
        suite="Poseidon downstream tasks",
        dataset_or_task="NS-Sines / NS-Gauss / NS-SL / FNS-KF",
        source_url=POSEIDON_URL,
        paper_or_release="Poseidon, NeurIPS 2024",
        primary_regime="incompressible fluid / nonlinear advection-diffusion",
        difficulty_tags=("nonlinear transport", "vorticity", "rollout", "forcing"),
        canonical_mino_relevance="Tests nonlinear transport-diffusion coupling; current MiNO should treat this as future nonlinear MiNO.",
        recommended_role="negative-to-extension benchmark",
        loader_status="adapter_required",
        local_scenario_hint="navier_stokes_synth",
        baseline_targets=("Poseidon", "scOT", "CNO", "FNO", "UNO"),
        priority="P1",
        notes="Do not use as main claim before pressure/projection and nonlinear coupling are implemented.",
    ),
    ExternalBenchmarkSpec(
        pde_family="navier_stokes_incompressible",
        benchmark_id="pdearena_navier_stokes_2d",
        suite="PDEArena",
        dataset_or_task="NavierStokes-2D",
        source_url=PDEARENA_URL,
        paper_or_release="PDEArena framework",
        primary_regime="time-dependent incompressible flow",
        difficulty_tags=("autoregressive rollout", "standard neural PDE surrogate", "HF dataset"),
        canonical_mino_relevance="Useful reproducible NS comparison with PDEArena training/eval protocol.",
        recommended_role="secondary NS benchmark",
        loader_status="adapter_required",
        local_scenario_hint="navier_stokes_synth",
        baseline_targets=("PDEArena FNO", "PDEArena U-Net/ResNet", "UNO if added"),
        priority="P2",
        notes="Use after Poseidon NS targets because Poseidon has stronger foundation-model baselines.",
    ),
    ExternalBenchmarkSpec(
        pde_family="compressible_euler",
        benchmark_id="poseidon_compressible_euler",
        suite="Poseidon downstream tasks",
        dataset_or_task="CE-RP / CE-CRP / CE-KH / CE-Gauss / CE-RM / GCE-RT",
        source_url=POSEIDON_URL,
        paper_or_release="Poseidon, NeurIPS 2024",
        primary_regime="compressible hyperbolic conservation law",
        difficulty_tags=("shocks", "interfaces", "Kelvin-Helmholtz", "Rayleigh-Taylor"),
        canonical_mino_relevance="Tests microlocal limitations in nonsmooth wavefront/shock regimes.",
        recommended_role="future stress benchmark",
        loader_status="adapter_required",
        local_scenario_hint="none",
        baseline_targets=("Poseidon", "scOT", "CNO"),
        priority="P2",
        notes="Current smooth-packet theory does not cover shocks; keep as limitation/future scope.",
    ),
    ExternalBenchmarkSpec(
        pde_family="darcy",
        benchmark_id="pdebench_darcy_flow_2d",
        suite="PDEBench",
        dataset_or_task="2D Darcy flow",
        source_url=PDEBENCH_URL,
        paper_or_release="PDEBench, NeurIPS 2022",
        primary_regime="elliptic coefficient-to-solution map",
        difficulty_tags=("rough coefficient", "elliptic smoothing", "operator learning classic"),
        canonical_mino_relevance="Baseline elliptic operator-learning sanity check; not a transport claim.",
        recommended_role="breadth benchmark",
        loader_status="adapter_required",
        local_scenario_hint="darcy_synth",
        baseline_targets=("FNO", "U-Net", "PINN", "UNO", "WNO"),
        priority="P2",
        notes="Current synthetic Darcy is not reliable enough for a main claim.",
    ),
    ExternalBenchmarkSpec(
        pde_family="allen_cahn_reaction_diffusion",
        benchmark_id="rpb_allen_cahn",
        suite="Representative PDE Benchmarks / CNO",
        dataset_or_task="Allen-Cahn equation",
        source_url=RPB_URL,
        paper_or_release="Representative PDE Benchmarks, NeurIPS 2023 dataset release",
        primary_regime="reaction-diffusion / interface dynamics",
        difficulty_tags=("interfaces", "metastability", "OOD split"),
        canonical_mino_relevance="Tests dissipative symbol plus interface-local packet defects.",
        recommended_role="secondary parabolic benchmark",
        loader_status="adapter_required",
        local_scenario_hint="none",
        baseline_targets=("CNO", "FNO", "U-Net"),
        priority="P2",
        notes="Useful if diffusion_neumann success needs a harder parabolic benchmark.",
    ),
    ExternalBenchmarkSpec(
        pde_family="transport",
        benchmark_id="rpb_smooth_discontinuous_transport",
        suite="Representative PDE Benchmarks / CNO",
        dataset_or_task="Smooth Transport / Discontinuous Transport",
        source_url=RPB_URL,
        paper_or_release="Representative PDE Benchmarks, NeurIPS 2023 dataset release",
        primary_regime="pure transport",
        difficulty_tags=("advection", "discontinuity", "OOD split"),
        canonical_mino_relevance="Clean external test for canonical transport without symbol complications.",
        recommended_role="mechanism benchmark",
        loader_status="adapter_required",
        local_scenario_hint="wave_chirp_propagation,wave_two_packet_no_interaction",
        baseline_targets=("CNO", "FNO", "U-Net"),
        priority="P1",
        notes="Good bridge between local branch-id probes and public transport benchmarks.",
    ),
)


def list_external_benchmarks(
    *,
    pde_family: str | None = None,
    priority: str | None = None,
) -> list[ExternalBenchmarkSpec]:
    specs = list(_BENCHMARKS)
    if pde_family is not None:
        specs = [spec for spec in specs if spec.pde_family == pde_family]
    if priority is not None:
        specs = [spec for spec in specs if spec.priority == priority]
    return specs


def external_benchmarks_by_family() -> dict[str, list[ExternalBenchmarkSpec]]:
    grouped: dict[str, list[ExternalBenchmarkSpec]] = {}
    for spec in _BENCHMARKS:
        grouped.setdefault(spec.pde_family, []).append(spec)
    return grouped


def benchmark_plan_rows(
    *,
    priorities: set[str] | None = None,
    families: set[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in _BENCHMARKS:
        if priorities is not None and spec.priority not in priorities:
            continue
        if families is not None and spec.pde_family not in families:
            continue
        row = asdict(spec)
        row["difficulty_tags"] = ";".join(spec.difficulty_tags)
        row["baseline_targets"] = ";".join(spec.baseline_targets)
        rows.append(row)
    return rows


def write_external_benchmark_csv(path: Path, rows: list[dict[str, object]] | None = None) -> None:
    output_rows = rows if rows is not None else benchmark_plan_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not output_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)


def write_external_benchmark_markdown(path: Path, rows: list[dict[str, object]] | None = None) -> None:
    output_rows = rows if rows is not None else benchmark_plan_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Curated External PDE Benchmark Plan",
        "",
        "| PDE family | Priority | Suite | Task | Role | MiNO relevance | Loader |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in output_rows:
        lines.append(
            "| {pde_family} | {priority} | {suite} | {dataset_or_task} | {recommended_role} | "
            "{canonical_mino_relevance} | {loader_status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
