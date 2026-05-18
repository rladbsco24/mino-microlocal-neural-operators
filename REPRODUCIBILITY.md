# Reproducibility Checklist

This file records the intended source-only repository workflow.  Large generated artifacts are excluded from Git and should be attached separately when preparing a formal artifact package.

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

Manuscript:

```powershell
cd manuscript/jmlr
bibtex mino_jmlr
pdflatex -interaction=nonstopmode mino_jmlr.tex
pdflatex -interaction=nonstopmode mino_jmlr.tex
```

## Main Campaigns

Mechanism-identifiability:

```powershell
python scripts/run_mino_empirical_closure.py --campaign branch_id_v3_controls --skip-existing --device cuda
```

Learned landing controls:

```powershell
python scripts/run_mino_empirical_closure.py --campaign branch_id_v3 --skip-existing --device cuda
```

High-k Helmholtz diagnostic:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_8gb --skip-existing --device cuda
```

High-k flagship protocol:

```powershell
python scripts/run_mino_empirical_closure.py --campaign helmholtz_highk_flagship --skip-existing --device cuda
```

## Artifact Packaging

For a formal artifact release, include:

- source repository commit hash;
- selected `generated/**/empirical_closure_results.csv` files;
- selected summary CSV files;
- exact command log for each campaign;
- hardware description;
- Lean build log;
- manuscript PDF compiled from the same commit.

Do not commit transient checkpoints, raw caches, or long-running log directories to the source repository.
