"""Validate VN2 per-round submission CSV files.

Checks the competition-facing constraints:
- six round files exist by default
- each file matches the submission template row order
- columns are Store, Product, 0
- exactly 599 rows
- no missing values
- integer, non-negative order quantities
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_ROWS = 599
REQUIRED_COLUMNS = ["Store", "Product", "0"]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_one(path: Path, template: pd.DataFrame) -> dict[str, int]:
    _assert(path.exists(), f"Missing submission file: {path}")
    df = pd.read_csv(path)

    _assert(list(df.columns) == REQUIRED_COLUMNS, f"{path}: columns must be {REQUIRED_COLUMNS}, got {list(df.columns)}")
    _assert(len(df) == EXPECTED_ROWS, f"{path}: expected {EXPECTED_ROWS} rows, got {len(df)}")
    _assert(df[["Store", "Product", "0"]].notna().all().all(), f"{path}: contains missing values")

    expected_pairs = template[["Store", "Product"]].reset_index(drop=True)
    actual_pairs = df[["Store", "Product"]].reset_index(drop=True)
    _assert(actual_pairs.equals(expected_pairs), f"{path}: Store/Product row order does not match template")

    qty = df["0"]
    numeric = pd.to_numeric(qty, errors="raise")
    _assert((numeric >= 0).all(), f"{path}: contains negative order quantities")
    _assert((numeric.round(0) == numeric).all(), f"{path}: order quantities must be integers")

    return {
        "rows": int(len(df)),
        "nonzero": int((numeric > 0).sum()),
        "total_units": int(numeric.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-dir", default="submissions")
    ap.add_argument("--template", default="Data/Week 0 - Submission Template.csv")
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    template_path = Path(args.template)
    _assert(template_path.exists(), f"Missing template: {template_path}")
    template = pd.read_csv(template_path)
    _assert(len(template) == EXPECTED_ROWS, f"Template expected {EXPECTED_ROWS} rows, got {len(template)}")

    sub_dir = Path(args.submission_dir)
    summaries = []
    for r in range(1, args.rounds + 1):
        path = sub_dir / f"round_{r}.csv"
        summary = validate_one(path, template)
        summary["round"] = r
        summaries.append(summary)

    print("OK: submission validation passed")
    for s in summaries:
        print(f"round_{s['round']}.csv: rows={s['rows']} nonzero={s['nonzero']} total_units={s['total_units']}")


if __name__ == "__main__":
    main()
