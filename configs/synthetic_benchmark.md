# Synthetic Benchmark Contract

This file fixes the initial internal benchmark contract for the MiNO scaffold.

## Scenarios

- `darcy`: smooth elliptic surrogate with spectral attenuation
- `navier_stokes`: advect-diffuse surrogate with resolution transfer stress
- `helmholtz`: oscillatory local phase-drift surrogate
- `wave`: high-frequency packet transport surrogate

## Metrics

- relative `L2`
- phase error
- packet consistency
- runtime in seconds
- parameter count

## Baselines

- `MiNO`
- `FNOStyle`
- `WNOStyle`
- `PDNOStyle`
- `LocalKernel`
- `UNetStyle`

## Immediate Goal

The first executable goal is not leaderboard publication. It is to validate the core design claim:

> once local oscillation and packet transport dominate, canonical propagation should matter more than point-only or frequency-only updates.

## High-k Helmholtz Flagship Contract

The high-frequency Helmholtz track is the natural flagship stress regime for
the microlocal packet-matrix primitive, but it requires a stricter contract
than the local synthetic smoke rows.

### Problem specification

- 2D variable-coefficient Helmholtz with heterogeneous medium.
- Multiple source positions and outgoing/PML or equivalent absorbing boundary
  treatment.
- Report nondimensional frequency `kL`, wavelengths across the domain, points
  per wavelength, grid size, and reference solver residual.
- Include in-distribution high-k rows and frequency extrapolation rows
  (`train k <= k_train`, `test k > k_train`).

### Required metrics

- complex relative `L2` when real/imaginary channels are available
- amplitude error
- wrapped or unwrapped phase error
- PDE residual and boundary/radiation residual
- outgoing flux or receiver/far-field trace error when available
- runtime, memory, and error degradation as `k` increases

### Baseline tiers

- General neural operators: FNO, UNO/U-FNO, WNO, CNO/CANO-style attention if
  available.
- Helmholtz-specialized neural solvers: MGCFNN-style neural multigrid/Fourier
  solver and probabilistic/diffusion Helmholtz operator references.
- Classical references: reference solver residual and hp-FEM/multigrid scaling
  language for pollution-effect claims.

### MiNO protocol

The registered future command path is:

```powershell
python scripts\run_mino_empirical_closure.py --campaign helmholtz_highk_flagship --helmholtz-profile flagship --seeds 7,11,19 --skip-existing --device cuda --dry-run
```

Remove `--dry-run` only when the external baseline/adaptor contract is fixed.
The campaign uses anisotropic/directional packets, carrier-bound finite
landing, the `helmholtz_resolvent` symbol hook, explicit real/imaginary losses
when the dataset supplies complex channels, and OOD high-k rows.  A strong
claim requires slower high-k degradation than the general neural operator
baselines and a credible comparison against at least one specialized
Helmholtz solver tier.
