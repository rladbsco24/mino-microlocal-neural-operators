# Curated MiNO Execution Results

This directory contains selected execution summaries copied from local `generated/` runs.  It is intended to keep the GitHub repository useful without committing large raw caches, checkpoints, logs, or writing assets.

Included categories:

- `experiment_index/` - aggregate run index and best-by-scenario summaries.
- `empirical_closure/branch_id_v3_carrier_final/` - carrier-bound branch-identifiability summaries.
- `empirical_closure/branch_id_v3_controls/` - transport, carrier, symbol, and randomized-metadata controls.
- `empirical_closure/learned_landing_controls/` - learned finite-landing controls.
- `empirical_closure/egorov_jacobian_bichar_3seed/` - Egorov/Jacobian diagnostic summaries.
- `empirical_closure/calculus_id_carrier_theory_core/` - calculus-identity carrier/theory-core probes.
- `empirical_closure/symbol_ablation_diffusion_lowfan_3seed/` - diffusion symbol-ablation probe.
- `empirical_closure/transport_param_*_lowfan_3seed/` - transport parameterization probes.
- `empirical_closure/helmholtz_branched_highk/` - multibranch high-k Helmholtz diagnostic summaries.
- `empirical_closure/helmholtz_highk_8gb_corrected_full/` - corrected high-k Helmholtz profile summaries.
- `empirical_closure/helmholtz_edge_kernel_minipack/` - edge-local kernel diagnostic summaries.
- `empirical_closure/highk_competitive_*` - high-k competitive-profile smoke and midpack summaries.
- `benchmark_stage2_shortlist_v1/` - stage-2 benchmark shortlist summaries.

Only CSV summaries, `manifest.json` files, and local README files are tracked here.  Full per-seed raw JSON files remain in `generated/` unless a specific audit requires promoting them.

To regenerate an index locally:

```powershell
python scripts/collect_mino_experiments.py --output generated/experiment_index
```
