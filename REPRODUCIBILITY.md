# Reproducibility Checklist

This repository tracks source code, tests, finite certificates, Lean files, configuration files, and curated result summaries.  Large raw experiment outputs remain local under `generated/` unless they are promoted into `results/`.

## Environment

```powershell
python -m pip install -e .
python -m pytest -q
```

Lean:

```powershell
cd lean
lake build
```

## Main Campaigns

Mechanism-identifiability controls:

```powershell
python scripts/run_mino_empirical_closure.py --campaign branch_id_v3_controls --skip-existing --device cuda
```

Learned-landing controls:

```powershell
python scripts/run_mino_empirical_closure.py --campaign learned_landing_controls --skip-existing --device cuda
```

High-k Helmholtz diagnostic:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_8gb --skip-existing --device cuda
```

High-k flagship protocol, if hardware allows:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_flagship --skip-existing --device cuda
```

Experiment index:

```powershell
python scripts/collect_mino_experiments.py --output generated/experiment_index
```

## Curated Results

The tracked `results/` tree contains CSV summaries, manifests, and selected run indexes.  It is not a complete raw-run archive.  Re-run campaigns into `generated/`, inspect the outputs, then copy only stable CSV/manifest artifacts into `results/`.

## Artifact Release Checklist

For a formal external artifact release, include:

- source repository commit hash;
- exact command log for each campaign;
- hardware, CUDA, and driver details;
- selected `generated/**/empirical_closure_results.csv` files;
- selected summary CSV files;
- raw per-seed JSON outputs when required to audit aggregation;
- Lean build log;
- Python environment lock file or package-version dump.

Do not commit transient checkpoints, raw caches, or long-running log directories to the source repository.
