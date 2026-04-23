"""Sweep LGB-vs-CB weighting in the diverse ensemble.

Current: equal-weight mean of 5 models (3 LGB + 2 CB). Implicit LGB share = 3/5 = 0.6.

We override this at blend time by manually setting per-model weights:
  lgb_total = lgb_share
  cb_total  = 1 - lgb_share
  Each LGB weight = lgb_share / 3
  Each CB weight  = (1 - lgb_share) / 2
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.cv_harness import (  # noqa: E402
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
# We'll subclass DiverseCostAwarePolicy to override the blend step.
from policies import DiverseCostAwarePolicy, _seasonal_ma_forecast  # noqa: E402
from forecaster import DiverseDemandForecaster  # noqa: E402
from simulation import LEAD_TIME, NUM_ROUNDS  # noqa: E402


class DiverseLGBShareCostAwarePolicy(DiverseCostAwarePolicy):
    """Same as DiverseCostAwarePolicy, but override ensemble blend to use LGB vs CB weighting."""

    def __init__(self, *args, lgb_share: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.lgb_share = lgb_share

    def _fit(self, sales_hist, in_stock):
        # Reuse parent fit for everything except the ML mean aggregation.
        # Simplest: override _fit in a minimal way by replicating and tweaking the mean step.
        def _log(m): print(f"    [diverse-share] {m}", flush=True)

        # Per-SKU multiplier (bin logic) — copy from parent.
        if self.mult_high is not None or self.mult_low is not None:
            if self.bin_basis == "sku":
                sku_metric = sales_hist.mean(axis=1)
            else:
                per_store = sales_hist.groupby(level=0).sum().mean(axis=1)
                sku_metric = pd.Series(
                    per_store.reindex(sales_hist.index.get_level_values(0)).values,
                    index=sales_hist.index,
                )
            thr = sku_metric.quantile(self.high_demand_quantile)
            high_mask = sku_metric >= thr
            mult_l = self.mult_low  if self.mult_low  is not None else self.multiplier
            mult_h = self.mult_high if self.mult_high is not None else self.multiplier
            self._sku_multiplier = pd.Series(mult_l, index=sales_hist.index)
            self._sku_multiplier.loc[high_mask] = mult_h
        else:
            self._sku_multiplier = None

        self._forecaster = DiverseDemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy, n_variants=self.n_variants,
            recency_decay=self.recency_decay,
        )
        _log(f"fitting 5-model forecaster (lgb_share={self.lgb_share:.2f})...")
        self._forecaster.fit(sales_hist, in_stock)
        horizon = NUM_ROUNDS + LEAD_TIME + 3
        per_model = self._forecaster.predict_models(horizon=horizon)
        model_names = list(per_model.keys())

        # LGB / CB weighting
        lgb_names = [n for n in model_names if n.startswith("lgbm")]
        cb_names  = [n for n in model_names if n.startswith("catboost")]
        w_lgb = self.lgb_share / max(len(lgb_names), 1)
        w_cb  = (1.0 - self.lgb_share) / max(len(cb_names), 1)
        weighted = np.zeros_like(per_model[model_names[0]].values)
        for n in lgb_names:
            weighted = weighted + w_lgb * per_model[n].reindex(sales_hist.index).values
        for n in cb_names:
            weighted = weighted + w_cb * per_model[n].reindex(sales_hist.index).values
        ml_df = pd.DataFrame(
            weighted, index=sales_hist.index,
            columns=per_model[model_names[0]].columns,
        ).clip(lower=0)
        _log(f"ml_df shape={ml_df.shape}  (w_lgb={w_lgb:.3f} per, w_cb={w_cb:.3f} per)")

        ma_df = _seasonal_ma_forecast(sales_hist, in_stock, horizon=horizon)
        n_cols = min(ml_df.shape[1], ma_df.shape[1])
        ma_aligned = ma_df.iloc[:, :n_cols].reindex(ml_df.index).fillna(0)
        blended = (self.w_ma * ma_aligned
                    + (1 - self.w_ma) * ml_df.iloc[:, :n_cols]).clip(lower=0)
        self._all_ml_forecasts = blended
        self._all_ma_forecasts = ma_df

        # Sigma (same as parent)
        W = self.backtest_window
        train = sales_hist.iloc[:, :-W]
        holdout = sales_hist.iloc[:, -W:]
        bt_is = in_stock.iloc[:, :-W] if in_stock is not None else None
        _log("backtest for sigma...")
        bt = DiverseDemandForecaster(
            master=self.master, random_state=self.random_state,
            censoring_strategy=self.censoring_strategy, n_variants=self.n_variants,
            recency_decay=self.recency_decay,
        )
        bt.fit(train, bt_is)
        bt_per = bt.predict_models(horizon=W)
        bt_lgb = [bt_per[n].reindex(sales_hist.index).values for n in bt_per if n.startswith("lgbm")]
        bt_cb  = [bt_per[n].reindex(sales_hist.index).values for n in bt_per if n.startswith("catboost")]
        bt_mean = (w_lgb * sum(bt_lgb) + w_cb * sum(bt_cb))
        n = min(bt_mean.shape[1], holdout.shape[1], self.rmse_horizons)
        resid = holdout.iloc[:, :n].values - bt_mean[:, :n]
        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))
        self._rmse = pd.Series(rmse, index=sales_hist.index).fillna(self.safety_floor).clip(lower=self.safety_floor)
        _log("fit complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares", default="0.3,0.45,0.6,0.75,0.9")
    args = ap.parse_args()

    shares = [float(s) for s in args.shares.split(",")]
    full_sales, full_in_stock, master = build_extended_sales()
    summary = []
    for s in shares:
        name = f"diverse5_lgbshare_{s}"
        print(f"\n>>> {name}")

        def factory(lgb_s=s):
            return DiverseLGBShareCostAwarePolicy(
                lgb_share=lgb_s,
                alpha=0.5813428492464154,
                mult_low=1.1731901672544072,
                mult_high=1.1918987771673257,
                high_demand_quantile=0.75,
                w_ma=0.33800656001887974,
                backtest_window=26, safety_floor=0.5, rmse_horizons=1,
                censoring_strategy="mean_impute", random_state=42, master=master,
                n_variants=5,
            )

        t0 = time.time()
        results = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            folds=FOLDS_8, include_val=True,
            coverage_weeks=3, alpha=0.58, n_workers=1,
        )
        wall = time.time() - t0
        print(format_summary(results, name))
        print(f"[{name}] wall-time: {wall:.1f}s")
        log_experiment(name, results, extra={
            "lgb_share": s, "approach": "diverse_5model_lgb_share_sweep",
            "wall_seconds": round(wall, 1),
        })
        summary.append({
            "share": s,
            "cv": results["mean"]["competition_cost"],
            "val": (results.get("val") or {}).get("competition_cost"),
        })

    print("\n=== LGB SHARE SWEEP SUMMARY ===")
    print(f"{'lgb_share':>10} {'cv':>10} {'val':>10}")
    for r in sorted(summary, key=lambda r: r["cv"]):
        print(f"{r['share']:>10.3f} {r['cv']:>10,.2f} {r['val']:>10,.2f}")


if __name__ == "__main__":
    main()
