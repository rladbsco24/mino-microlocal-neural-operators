# Microlocal Neural Operators (MiNO)

MiNO is a research codebase for the manuscript:

> **Microlocal Neural Operators: Sparse Canonical Packet Matrices for Oscillatory Operator Learning**

The project studies neural operators whose internal state is a sparse packet matrix indexed by phase-space wave packets.  The codebase contains the PyTorch implementation, theorem-facing diagnostics, benchmark runners, manuscript source, and a Lean 4 finite-packet companion.

## Repository Layout

- `mino/` - PyTorch models, layers, metrics, data loaders, and training utilities.
- `scripts/` - experiment runners, figure generation, certificate export, and artifact collection.
- `tests/` - unit tests for models, metrics, and benchmark plumbing.
- `lean/` - Lean 4 formalization of the finite packet/landing algebra.
- `manuscript/` - JMLR manuscript source, tables, and figures.
- `configs/` - experiment and run configuration files.
- `certificates/` - finite symbolic certificate outputs used by the manuscript.
- `.github/workflows/` - lightweight CI checks.

Generated experiments, logs, checkpoints, temporary outputs, and Lean build artifacts are intentionally excluded from Git.  They can be regenerated from the scripts below.

## Installation

```powershell
python -m pip install -e .
```

For development, install the package in editable mode from the repository root.

## Quick Checks

Run Python tests:

```powershell
python -m pytest -q
```

Run a targeted metric test:

```powershell
python -m pytest -q tests/test_metrics.py
```

Compile key Python files:

```powershell
python -m py_compile mino/models/layers.py mino/models/mino.py scripts/run_mino_empirical_closure.py
```

Build the Lean companion:

```powershell
cd lean
lake build
```

Build the JMLR manuscript:

```powershell
cd manuscript/jmlr
bibtex mino_jmlr
pdflatex -interaction=nonstopmode mino_jmlr.tex
pdflatex -interaction=nonstopmode mino_jmlr.tex
```

## Core Experiment Runners

Plan the empirical closure campaign without running it:

```powershell
python scripts/run_mino_empirical_closure.py --campaign smoke --dry-run
```

Run the carrier-bound branch-identifiability campaign:

```powershell
python scripts/run_mino_empirical_closure.py --campaign branch_id_v3_controls --skip-existing
```

Run the high-frequency Helmholtz diagnostic profile:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_8gb --skip-existing --device cuda
```

Run the stricter high-k flagship protocol, if hardware allows:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_flagship --skip-existing --device cuda
```

Collect completed experiment metadata:

```powershell
python scripts/collect_mino_experiments.py --output generated/experiment_index
```

## Manuscript Scope

The current paper is a MiNO-Direct mechanism/theory paper.  Its central claim is not universal benchmark dominance.  The claim is that sparse canonical packet matrices are an executable and falsifiable neural-operator primitive for oscillatory PDEs.

The main theory is an operator-level packet-matrix theorem in the `ell^2/L^2` energy norm.  The Lean companion checks the finite packet landing algebra.  High-frequency Helmholtz is treated as a stress diagnostic unless a full frequency-scaling campaign with phase, radiation, complex-field, and specialized-baseline comparisons is completed.

## Reproducibility Notes

- Raw generated outputs are excluded from Git by default because they can be large and may be modified by long-running experiments.
- Use `--skip-existing` to resume interrupted campaigns safely.
- The runner records scenario, ablation, seed, model hyperparameters, runtime, relative error, phase error, wavefront proxies, Egorov proxies, and high-k Helmholtz diagnostics in CSV form.
- If submitting an artifact package, include selected `generated/` CSV summaries and a manifest separately from the source repository.

## Citation

This repository is under active manuscript development.  A formal citation entry should be added once the public preprint or proceedings version is fixed.
