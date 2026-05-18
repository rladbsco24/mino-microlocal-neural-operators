import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mino.data import load_dmno_reference_rows, load_tcno_reference_rows  # noqa: E402
from scripts.run_mino_benchmark import aggregate_pair_rows, aggregate_rows, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate existing benchmark JSON files into CSV summaries.")
    parser.add_argument("--input", required=True, help="Directory containing per-run JSON files.")
    parser.add_argument("--include-tcno-reference", action="store_true")
    parser.add_argument("--include-dmno-reference", action="store_true")
    parser.add_argument("--reference-models", default="FNO,Conv-FNO,WNO-style,UNO,FNO+LearnedLocalTC")
    parser.add_argument("--dmno-reference-models", default="DCR-NO")
    return parser.parse_args()


def load_json_rows(input_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def main() -> None:
    args = parse_args()
    input_dir = ROOT / args.input
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing benchmark directory: {input_dir}")

    rows = load_json_rows(input_dir)
    if not rows:
        raise RuntimeError(f"No JSON result files found in {input_dir}")

    scenario_set = {str(row["scenario"]) for row in rows}
    if args.include_tcno_reference:
        reference_models = {item.strip() for item in args.reference_models.split(",") if item.strip()}
        rows.extend(load_tcno_reference_rows(scenarios=scenario_set, models=reference_models))
    if args.include_dmno_reference:
        dmno_models = {item.strip() for item in args.dmno_reference_models.split(",") if item.strip()}
        rows.extend(load_dmno_reference_rows(scenarios=scenario_set, models=dmno_models))

    write_csv(input_dir / "benchmark_results.csv", rows)
    write_csv(input_dir / "benchmark_summary.csv", aggregate_rows(rows))
    write_csv(input_dir / "benchmark_pair_summary.csv", aggregate_pair_rows(rows))
    print(f"Wrote aggregate CSV files under {input_dir}")


if __name__ == "__main__":
    main()
