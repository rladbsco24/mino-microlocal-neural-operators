from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


def main() -> None:
    a_hat, a, x_hat, x, r = sp.symbols("a_hat a x_hat x r", real=True)
    decomposition = sp.expand(a_hat * x_hat + r - a * x)
    split = sp.expand(a * (x_hat - x) + (a_hat - a) * x_hat + r)
    certificate = {
        "identity_lhs": str(decomposition),
        "identity_rhs": str(split),
        "difference_simplifies_to": str(sp.simplify(decomposition - split)),
        "bound_template": "|a|*|x_hat - x| + |a_hat - a|*|x_hat| + |r|",
    }
    root = Path(__file__).resolve().parents[1]
    output = root / "certificates" / "transport_decomposition.json"
    output.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
