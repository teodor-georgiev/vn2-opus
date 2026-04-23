"""Compare prediction quality and cost on VAL vs SACRED windows."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.cv_harness import VAL_START, build_extended_sales  # noqa: E402
from policies import EnsemblePolicy, _seasonal_ma_forecast  # noqa: E402

SACRED_START = 157
HORIZON = 8

full_sales, full_in_stock, master = build_extended_sales()


def fit_predict_window(ws: int) -> pd.DataFrame:
    """Fit ensemble forecaster up to week `ws`, return prediction matrix [n_sku x H]."""
    sales_hist = full_sales.iloc[:, :ws]
    in_stock_hist = full_in_stock.iloc[:, :ws]
    ens = EnsemblePolicy(coverage_weeks=3, w_ma=0.25,
                         censoring_strategy="mean_impute", random_state=42)
    ens._fit(sales_hist, in_stock_hist)
    ml = ens._forecaster.predict(horizon=HORIZON)
    ma = _seasonal_ma_forecast(sales_hist, in_stock_hist, horizon=HORIZON)
    common = ml.columns[: min(len(ml.columns), len(ma.columns))]
    blend = 0.25 * ma[common].reindex(ml.index).fillna(0) + 0.75 * ml[common]
    return blend


def window_audit(name: str, ws: int):
    print(f"\n=== {name} (window_start={ws}) ===")
    pred = fit_predict_window(ws)  # [n_sku, 8]
    cols = sorted(full_sales.columns)
    actual = full_sales[cols[ws: ws + HORIZON]]  # [n_sku, 8]
    actual.columns = pred.columns

    # Per-week totals (sum across SKUs)
    total_pred_w = pred.sum(axis=0).values
    total_actual_w = actual.sum(axis=0).values

    # Per-week absolute error and pct error
    abs_err_w = np.abs(total_actual_w - total_pred_w)
    rel_err_w = abs_err_w / np.abs(total_actual_w) * 100

    # Per-week table
    df = pd.DataFrame({
        "wk": [f"wk{i+1}" for i in range(HORIZON)],
        "actual_total": total_actual_w.astype(int),
        "predicted_total": total_pred_w.round(0).astype(int),
        "diff": (total_pred_w - total_actual_w).round(0).astype(int),
        "abs_err": abs_err_w.round(0).astype(int),
        "pct_err": rel_err_w.round(1),
    })
    print(df.to_string(index=False))

    # Aggregated metrics across all SKUs and weeks
    # Per-SKU per-week errors (all 599 x 8 = 4792 observations)
    per_obs_err = (pred.values - actual.values).flatten()
    per_obs_actual = actual.values.flatten()
    mae = np.mean(np.abs(per_obs_err))
    wape = np.sum(np.abs(per_obs_err)) / np.sum(np.abs(per_obs_actual)) * 100
    bias = np.mean(per_obs_err)

    print(f"\n  Aggregate (per SKU per week, n={len(per_obs_err)}):")
    print(f"    MAE  = {mae:.2f}")
    print(f"    WAPE = {wape:.2f}%")
    print(f"    bias = {bias:+.2f}  (negative => model under-forecasts)")

    # Total-totals metrics (sum across SKUs)
    print(f"\n  Per-week-total accuracy:")
    total_mae = abs_err_w.mean()
    total_wape = abs_err_w.sum() / total_actual_w.sum() * 100
    print(f"    MAE  = {total_mae:.0f} units")
    print(f"    WAPE = {total_wape:.2f}%")
    print(f"    Total actual demand across 8 weeks = {total_actual_w.sum():,.0f}")
    print(f"    Total predicted demand 8 weeks    = {total_pred_w.sum():,.0f}")
    print(f"    Aggregate forecast bias           = {total_pred_w.sum() - total_actual_w.sum():+,.0f} units")


print(f"VAL_START = {VAL_START}, SACRED_START = {SACRED_START}")
window_audit("VAL", VAL_START)
window_audit("SACRED", SACRED_START)

# Also report cost numbers from logged experiments for our best policy.
print("\n\n=== COST comparison (from logged experiments) ===")
print("Policy: cap2_a0.65_m1.0 (our champion)")
print(f"  VAL competition_cost  = 2,593 EUR")
print(f"  SACRED competition_cost = 3,349.60 EUR")
print(f"  Ratio sacred/VAL = {3349.6/2593:.2f}x")
