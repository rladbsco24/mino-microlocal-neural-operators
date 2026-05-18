# MiNO: Microlocal Neural Operator Code and Artifacts

This repository contains the executable code, tests, Lean finite-packet companion, configuration files, and curated execution results for MiNO.

MiNO is a research codebase for sparse canonical packet-matrix neural operators.  The repository is intentionally code-first: writing sources, PDFs, and private notes are not included on the current branch.

## Repository Layout

- `mino/` - PyTorch models, layers, metrics, data loaders, and training utilities.
- `scripts/` - experiment runners, certificate export, benchmark indexing, and artifact utilities.
- `tests/` - unit tests for models, metrics, and benchmark plumbing.
- `lean/` - Lean 4 companion for finite packet/landing algebra.
- `configs/` - experiment and run configuration files.
- `certificates/` - finite symbolic certificate outputs.
- `results/` - curated CSV summaries, manifests, and selected run indexes from local experiments.
- `.github/workflows/` - lightweight CI checks.

Raw generated outputs, checkpoints, caches, and long-running logs remain excluded from Git.  Use `generated/` for local runs and promote only selected summaries to `results/`.

## Installation

```powershell
python -m pip install -e .
```

Run from the repository root.

## Quick Checks

Run Python tests:

```powershell
python -m pytest -q
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

## Core Experiment Runners

Plan a campaign without running it:

```powershell
python scripts/run_mino_empirical_closure.py --campaign smoke --dry-run
```

Run the carrier-bound branch-identifiability controls:

```powershell
python scripts/run_mino_empirical_closure.py --campaign branch_id_v3_controls --skip-existing --device cuda
```

Run the learned-landing controls:

```powershell
python scripts/run_mino_empirical_closure.py --campaign learned_landing_controls --skip-existing --device cuda
```

Run the high-frequency Helmholtz diagnostic profile:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_8gb --skip-existing --device cuda
```

Collect completed experiment metadata locally:

```powershell
python scripts/collect_mino_experiments.py --output generated/experiment_index
```

## Results

Curated outputs are stored in `results/`.  They are small, reviewable CSV/manifest snapshots copied from local `generated/` runs.  The full raw run directories are intentionally not tracked.

The result bundle includes:

- controlled wave branch-identifiability runs;
- learned-landing controls;
- Egorov/Jacobian diagnostic rows;
- calculus-identity and symbol-ablation probes;
- transport parameterization probes;
- high-k Helmholtz diagnostics and competitive-profile smoke/midpack runs;
- stage-2 benchmark shortlist summaries.

See `results/README.md` for the exact contents.

## Reproducibility Boundary

This branch is a source-and-results repository, not a writing-source package.  A formal artifact release should additionally record:

- repository commit hash;
- exact campaign commands;
- hardware and driver details;
- raw generated outputs or a separate archival bundle;
- Lean build log;
- Python package versions.

Do not commit transient checkpoints, raw caches, or long-running log directories to this repository.
