from __future__ import annotations

import json
from pathlib import Path

from mpmath import iv


def main() -> None:
    exact_symbol = iv.mpf([0.0, 1.2])
    transport_error = iv.mpf([0.0, 0.03])
    symbol_error = iv.mpf([0.0, 0.02])
    coeff_bound = iv.mpf([0.0, 1.5])
    truncation_error = iv.mpf([0.0, 0.01])
    packet_count = 64

    pointwise_budget = exact_symbol * transport_error + symbol_error * coeff_bound + truncation_error
    l1_budget = packet_count * pointwise_budget
    payload = {
        "exact_symbol_bound": [float(exact_symbol.a), float(exact_symbol.b)],
        "transport_error": [float(transport_error.a), float(transport_error.b)],
        "symbol_error": [float(symbol_error.a), float(symbol_error.b)],
        "coefficient_bound": [float(coeff_bound.a), float(coeff_bound.b)],
        "truncation_error": [float(truncation_error.a), float(truncation_error.b)],
        "pointwise_budget_upper": float(pointwise_budget.b),
        "l1_budget_upper": float(l1_budget.b),
    }
    root = Path(__file__).resolve().parents[1]
    output = root / "certificates" / "interval_check.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
