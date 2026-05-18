from __future__ import annotations

import csv

from mino.data import (
    ExternalBenchmarkSpec,
    benchmark_plan_rows,
    external_benchmarks_by_family,
    list_external_benchmarks,
    write_external_benchmark_csv,
    write_external_benchmark_markdown,
)


def test_external_benchmark_registry_covers_core_pde_families() -> None:
    grouped = external_benchmarks_by_family()

    assert "wave" in grouped
    assert "helmholtz" in grouped
    assert "navier_stokes_incompressible" in grouped
    assert "diffusion_reaction" in grouped
    assert "poisson_elliptic" in grouped
    assert all(isinstance(spec, ExternalBenchmarkSpec) for specs in grouped.values() for spec in specs)


def test_external_benchmark_priority_filter_selects_claim_targets() -> None:
    p0_specs = list_external_benchmarks(priority="P0")

    assert {spec.pde_family for spec in p0_specs} >= {"wave", "helmholtz", "diffusion_reaction"}
    assert all(spec.priority == "P0" for spec in p0_specs)


def test_benchmark_plan_rows_are_flat_export_rows() -> None:
    rows = benchmark_plan_rows(priorities={"P0", "P1"})

    assert rows
    assert all(isinstance(row["difficulty_tags"], str) for row in rows)
    assert all(isinstance(row["baseline_targets"], str) for row in rows)
    assert any(row["benchmark_id"] == "poseidon_helmholtz" for row in rows)
    assert any(row["benchmark_id"] == "mgcfnn_high_wavenumber_helmholtz" for row in rows)
    assert any(row["benchmark_id"] == "probabilistic_high_frequency_helmholtz" for row in rows)


def test_external_benchmark_exports(tmp_path) -> None:
    rows = benchmark_plan_rows(priorities={"P0"})
    csv_path = tmp_path / "benchmarks.csv"
    md_path = tmp_path / "benchmarks.md"

    write_external_benchmark_csv(csv_path, rows)
    write_external_benchmark_markdown(md_path, rows)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        exported_rows = list(csv.DictReader(handle))
    assert len(exported_rows) == len(rows)
    assert "poseidon_helmholtz" in {row["benchmark_id"] for row in exported_rows}

    markdown = md_path.read_text(encoding="utf-8")
    assert "Curated External PDE Benchmark Plan" in markdown
    assert "Poseidon downstream tasks" in markdown
