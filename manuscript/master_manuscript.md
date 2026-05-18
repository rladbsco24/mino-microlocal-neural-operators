# Microlocal Neural Operators: Canonical Propagation as a Design Principle for Operator Learning

## Core Claim

Neural operators should preserve canonical propagation. This is the correct primitive for oscillatory operator learning, and it should be built into the backbone rather than recovered as an emergent side effect of point-space or frequency-space correlation.

## Abstract Draft

We introduce the Microlocal Neural Operator (MiNO), a neural operator backbone built around analytic wavepacket tokenization, canonical phase-space transport, and local symbol modulation. The design is motivated by a discrete microlocal propagation theorem: for a finite packet system, the coefficient error of an approximate propagation layer is controlled by the sum of transport, symbol, and truncation errors. This theorem gives a direct architectural justification for separating transport from symbol correction. MiNO reduces to lightweight spectral behavior in smooth low-frequency regimes, while remaining expressive enough to track local phase drift and dominant wavefront geometry in oscillatory settings. We implement MiNO as a transport-aware wavepacket backbone and evaluate it on smooth and oscillatory synthetic operator surrogates spanning Darcy-type, Navier-Stokes-type, Helmholtz-type, and wave-propagation-type families. The repository also includes a Lean 4 companion that machine-checks the finite theorem spine used by the manuscript. The resulting program positions microlocal propagation as a design principle for the next generation of neural operators.

## Main Sections

1. Introduction
2. Why canonical propagation is the right primitive
3. Analytic wavepacket tokenization
4. Microlocal Neural Operator architecture
5. Microlocal Propagation Theorem
6. Approximation Corollary
7. Synthetic and trainable benchmarks
8. Ablations: removing transport, removing symbol modulation, removing low-frequency residual
9. Machine-proof companion
10. Limitations and FoCM path

## Main Theorem Statement

For a finite packet system, let the reference operator transport packet coefficients through a reference transport map and reference symbol. Let the approximate MiNO layer use an approximate transport, an approximate symbol, and a residual truncation term. If the transport mismatch, symbol mismatch, and residual truncation are uniformly bounded, then the induced coefficient error is bounded by the corresponding transport, symbol, and truncation budgets. Consequently, the induced finite `l1` error is bounded by the same budget times the packet count.

## Experimental Spine

- Smooth surrogate: Darcy
- Dynamics surrogate: Navier-Stokes
- Oscillatory surrogate: variable-coefficient Helmholtz
- High-frequency transport surrogate: scalar wave propagation

## Ablation Spine

- remove canonical transport
- freeze metadata transport
- remove symbol modulation
- remove low-frequency residual branch
- replace local packet frame by global FFT
