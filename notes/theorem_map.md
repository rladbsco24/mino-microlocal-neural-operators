# Theorem-to-Code Map

## Main Theorem

**Microlocal Propagation Theorem**

Finite statement implemented in Lean:

- exact reference layer: packet coefficients are moved by a reference transport and scaled by a reference symbol
- approximate MiNO layer: packet coefficients are moved by an approximate transport, scaled by an approximate symbol, and corrected by a residual truncation term
- theorem: coefficient error is controlled by transport error, symbol error, and truncation error

Lean files:

- `lean/MiNO/Microlocal/FinitePacket.lean`
- `lean/MiNO/Microlocal/Propagation.lean`
- `lean/MiNO/Microlocal/Approximation.lean`

## Supporting Corollary

**Approximation Corollary**

If the approximate transport, symbol, and residual components fit the reference finite operator within an admissible budget, then the induced `l1` operator error falls below the target tolerance.

## Code Alignment

- `mino/models/wavepacket.py`: analytic packet state construction
- `mino/models/layers.py`: approximate canonical propagation and symbol modulation
- `mino/models/mino.py`: full MiNO backbone
- `scripts/export_transport_certificate.py`: symbolic decomposition of the theorem identity
- `scripts/check_interval_certificate.py`: interval-style check for a concrete finite budget
