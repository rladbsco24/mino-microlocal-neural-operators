from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data import (
    benchmark_plan_rows,
    write_external_benchmark_csv,
    write_external_benchmark_markdown,
)


def _parse_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the curated external PDE benchmark plan without running experiments."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("generated/external_benchmark_plan/pde_benchmarks.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("generated/external_benchmark_plan/pde_benchmarks.md"),
        help="Markdown output path.",
    )
    parser.add_argument(
        "--priorities",
        default=None,
        help="Comma-separated priority filter, e.g. P0,P1.",
    )
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated PDE-family filter, e.g. helmholtz,wave.",
    )
    args = parser.parse_args()

    rows = benchmark_plan_rows(
        priorities=_parse_filter(args.priorities),
        families=_parse_filter(args.families),
    )
    write_external_benchmark_csv(args.output_csv, rows)
    write_external_benchmark_markdown(args.output_md, rows)
    print(f"Exported {len(rows)} benchmark rows")
    print(f"CSV: {args.output_csv}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
