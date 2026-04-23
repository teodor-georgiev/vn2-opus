"""Parse experiments.csv robustly and show CV std per experiment.

The CSV has schema drift over time. We extract `fold_costs` (semicolon-separated)
from each row and compute mean/std/range.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CSV = ROOT / "benchmark" / "experiments.csv"

rows = []
with open(CSV, encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for fields in reader:
        if len(fields) < 3:
            continue
        ts = fields[0]
        policy = fields[1]
        # Find a field that contains ";" (fold_costs) — at varying positions.
        fold_costs_str = next((v for v in fields if ";" in v and not v.startswith("Ensemble")), None)
        if fold_costs_str is None:
            # Older rows have no per-fold breakdown.
            continue
        try:
            fold_costs = [float(x) for x in fold_costs_str.split(";") if x]
        except ValueError:
            continue
        if not fold_costs:
            continue
        # cv_mean_cost should be column 2 by convention.
        try:
            cv_mean = float(fields[2])
        except Exception:
            cv_mean = float(np.mean(fold_costs))
        # Try to grab val_cost — it varies in position.
        # Heuristic: look for a value ~half cv_mean (since val is ~8wk vs cv is ~6wk×folds)
        # Skip val for now, just compute fold stats.
        std = float(np.std(fold_costs, ddof=1)) if len(fold_costs) > 1 else 0.0
        cmin = float(np.min(fold_costs))
        cmax = float(np.max(fold_costs))
        rows.append({
            "timestamp": ts,
            "policy": policy,
            "n_folds": len(fold_costs),
            "cv_mean": cv_mean,
            "cv_std": std,
            "cv_min": cmin,
            "cv_max": cmax,
            "cv_range": cmax - cmin,
            "cv_cv_pct": std / cv_mean * 100 if cv_mean else 0,
        })

if not rows:
    print("No fold-level data found in experiments.csv.")
    sys.exit(0)

df = pd.DataFrame(rows)
df = df.drop_duplicates("policy", keep="last")
df = df.sort_values("cv_mean")

print(f"=== CV per-fold std audit ({len(df)} unique policies with fold-level data) ===\n")

# Top by CV mean (best policies)
top = df.head(20)
print("TOP 20 by CV mean (lowest cost):")
print(top[["policy", "n_folds", "cv_mean", "cv_std", "cv_min", "cv_max", "cv_cv_pct"]].to_string(
    index=False,
    formatters={
        "cv_mean": "{:,.1f}".format,
        "cv_std": "{:,.1f}".format,
        "cv_min": "{:,.1f}".format,
        "cv_max": "{:,.1f}".format,
        "cv_cv_pct": "{:.1f}%".format,
    },
))

print("\n=== Distribution stats (across all policies) ===")
print(f"  CV std: median={df['cv_std'].median():,.1f}  mean={df['cv_std'].mean():,.1f}  "
      f"min={df['cv_std'].min():,.1f}  max={df['cv_std'].max():,.1f}")
print(f"  CV CV%: median={df['cv_cv_pct'].median():.1f}%  mean={df['cv_cv_pct'].mean():.1f}%")

# Save full table for inspection
df.to_csv("cv_std_audit.csv", index=False)
print(f"\nFull table: cv_std_audit.csv ({len(df)} policies)")
