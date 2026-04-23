"""Pretty-print the experiments.csv: top results by CV and by sacred."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
RESULTS_CSV = ROOT / "benchmark" / "experiments.csv"


def run(top_n: int = 15):
    if not RESULTS_CSV.exists():
        print(f"No results file at {RESULTS_CSV}")
        return
    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("Empty results file.")
        return

    df = df.copy()
    df["sacred"] = df["policy"].str.contains(r"\[SACRED", regex=True, na=False)

    sacred = df[df["sacred"]].copy()
    sacred = sacred.sort_values("cv_mean_cost", ascending=True)

    cv = df[~df["sacred"]].copy().sort_values("cv_mean_cost", ascending=True)

    def _show(d, key, title, n):
        print(f"\n{title}")
        print("-" * 100)
        cols = ["policy", "cv_mean_cost", "val_cost", "fc_mae", "fc_bias"]
        cols = [c for c in cols if c in d.columns]
        d = d[cols].head(n)
        d = d.assign(**{k: d[k].round(2) for k in cols if k != "policy" and d[k].dtype != object})
        print(d.to_string(index=False))

    _show(cv, "cv_mean_cost", f"TOP {top_n} CV (lower = better)", top_n)
    _show(sacred, "cv_mean_cost", f"SACRED EVALS (cv_mean_cost column shows sacred here)", top_n)

    print(f"\nTotal rows: {len(df)}  (CV entries: {len(cv)}, sacred entries: {len(sacred)})")
    print(f"File: {RESULTS_CSV}")
