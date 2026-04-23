"""Honest CV harness for VN2 policies.

Protocol
--------
History: weeks 0..156 (pre-competition, 2021-04-12 .. 2024-04-08)
Sacred : weeks 157..164 (hidden competition, touch ONCE at the end)

CV folds    : pseudo-competitions on history. 8-week rollout windows, first 2
              weeks ignored (setup), next 6 weeks scored as competition_cost.
              Fold window-start indices: 93, 109, 125, 141 (so scored weeks are
              95..100, 111..116, 127..132, 143..148 — disjoint, later-the-year
              seasonal coverage).
VAL         : window-start 149 (scored weeks 151..156). Final pre-competition
              check.
SACRED      : run_simulation() via simulation.InventorySimulator (weeks 1..8).

For each (fold, policy), we record:
  - cost metrics: total_cost, setup_cost, competition_cost (+ holding/shortage)
  - forecast accuracy at each round vs next-3-week actual demand:
      MAE, WAPE, bias, and (if the policy exposes quantile forecasts)
      pinball loss at alpha=0.833 (our critical ratio).

Use `score_policies(policy_factories, folds=..., include_val=True)` to compare.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Ensure project root on sys.path.
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime  # noqa: E402

from simulation import (  # noqa: E402
    HOLDING_COST,
    LEAD_TIME,
    SHORTAGE_COST,
    TOTAL_WEEKS,
    load_initial_data,
    run_window,
)

ALPHA = 0.833  # critical ratio for shortage=1, holding=0.2
FOLDS = {
    "cv0": 93,
    "cv1": 109,
    "cv2": 125,
    "cv3": 141,
}
# Denser 8-fold variant: all scored periods disjoint; same VAL window.
FOLDS_8 = {
    "cv0": 85,
    "cv1": 93,
    "cv2": 101,
    "cv3": 109,
    "cv4": 117,
    "cv5": 125,
    "cv6": 133,
    "cv7": 141,
}
VAL_START = 149


@dataclass
class ForecastLog:
    """Per-round forecast record for accuracy evaluation."""
    round_idx: int
    window_start: int
    forecast: pd.Series  # length = n_future periods (typically 3)
    # forecast is a Series indexed by (Store, Product), values = sum over next N weeks.
    actual: pd.Series | None = None  # filled in post-sim
    # Optional: quantile-specific forecast for pinball loss.
    quantile_forecast: pd.Series | None = None
    alpha: float = 0.5


class ForecastTrackingPolicy:
    """Wraps a policy factory so we can extract forecasts alongside orders.

    The wrapped policy must expose either:
      - `get_last_forecast() -> pd.Series`: sum-of-next-`coverage` weeks demand
      - an attribute `last_forecast` (same shape)
    If neither is present, forecast accuracy is skipped for that policy.
    """

    def __init__(self, inner, coverage_weeks: int = 3, alpha: float | None = None):
        self.inner = inner
        self.coverage_weeks = coverage_weeks
        self.alpha = alpha
        self.logs: list[ForecastLog] = []
        self._window_start: int | None = None

    def set_window(self, window_start: int):
        self._window_start = window_start
        self.logs = []

    def __call__(self, sim, round_idx, sales_hist):
        order = self.inner(sim, round_idx, sales_hist)
        fc = None
        if hasattr(self.inner, "last_forecast"):
            fc = self.inner.last_forecast
        elif hasattr(self.inner, "get_last_forecast"):
            fc = self.inner.get_last_forecast()
        if fc is not None:
            self.logs.append(
                ForecastLog(
                    round_idx=round_idx,
                    window_start=self._window_start or -1,
                    forecast=pd.Series(fc).astype(float),
                    quantile_forecast=(
                        pd.Series(self.inner.last_quantile_forecast).astype(float)
                        if hasattr(self.inner, "last_quantile_forecast")
                        and self.inner.last_quantile_forecast is not None
                        else None
                    ),
                    alpha=self.alpha if self.alpha is not None else 0.5,
                )
            )
        return order


def score_forecast_logs(
    logs: list[ForecastLog],
    full_sales: pd.DataFrame,
    coverage_weeks: int = 3,
) -> dict:
    """Compute MAE/WAPE/bias (+pinball if alpha-quantile provided) over logs.

    For each log, the "actual" is the SUM of sales for the next `coverage_weeks`
    starting at the week the order will arrive (round_idx + LEAD_TIME + 1
    relative to window_start; that's when the order arrives).
    """
    if not logs:
        return {"n": 0}

    sales_cols = sorted(full_sales.columns)
    mae, wape_num, wape_den, bias = 0.0, 0.0, 0.0, 0.0
    pinball, pinball_n = 0.0, 0
    n = 0
    for lg in logs:
        # Order placed at round_idx r arrives at relative week r + LEAD_TIME + 1
        # (within the window). We forecast the sum over the next coverage_weeks
        # starting from that arrival week.
        arrival_rel = lg.round_idx + LEAD_TIME + 1
        start = lg.window_start + arrival_rel
        end = start + coverage_weeks
        if end > len(sales_cols):
            continue
        actual_slice = full_sales[sales_cols[start:end]].sum(axis=1)
        actual_slice = actual_slice.reindex(lg.forecast.index)
        diff = (lg.forecast - actual_slice)
        mae += diff.abs().sum()
        wape_num += diff.abs().sum()
        wape_den += actual_slice.abs().sum()
        bias += diff.sum()
        n += len(diff)
        if lg.quantile_forecast is not None:
            q = lg.quantile_forecast.reindex(lg.forecast.index)
            err = (actual_slice - q)
            pinball += (np.maximum(lg.alpha * err, (lg.alpha - 1) * err)).sum()
            pinball_n += len(err)

    out = {
        "n_obs": n,
        "mae": float(mae / max(n, 1)),
        "wape": float(wape_num / wape_den) if wape_den > 0 else float("nan"),
        "bias": float(bias / max(n, 1)),
    }
    if pinball_n > 0:
        out["pinball_0.833"] = float(pinball / pinball_n)
    return out


def _run_one_window(args):
    """Top-level helper for parallel execution (must be picklable)."""
    policy_factory, ws, full_sales, full_in_stock, master, coverage_weeks, alpha = args
    inner = policy_factory()
    wrapper = ForecastTrackingPolicy(inner, coverage_weeks=coverage_weeks, alpha=alpha)
    wrapper.set_window(ws)
    res = run_window(
        full_sales=full_sales,
        window_start=ws,
        full_in_stock=full_in_stock,
        master=master,
        n_weeks=TOTAL_WEEKS,
        policy_fn=wrapper,
    )
    acc = score_forecast_logs(wrapper.logs, full_sales, coverage_weeks=coverage_weeks)
    return {
        "window_start": ws,
        "total_cost": res["total_cost"],
        "setup_cost": res["setup_cost"],
        "competition_cost": res["competition_cost"],
        "comp_holding": res["competition_holding"],
        "comp_shortage": res["competition_shortage"],
        **{f"fc_{k}": v for k, v in acc.items()},
    }


def score_policy_on_folds(
    policy_factory: Callable[[], object],
    full_sales: pd.DataFrame,
    full_in_stock: pd.DataFrame | None,
    master: pd.DataFrame | None,
    folds: dict[str, int] = FOLDS,
    include_val: bool = True,
    coverage_weeks: int = 3,
    alpha: float | None = None,
    n_workers: int = 1,
) -> dict:
    """Run a policy across CV folds (and optionally VAL).

    When n_workers > 1, folds run in parallel via joblib loky backend.
    Inner LightGBM/CatBoost auto-parallelism is CPU-bound so we do not
    multiply it too aggressively; leave default n_jobs.
    """
    tasks = [(name, (policy_factory, ws, full_sales, full_in_stock, master, coverage_weeks, alpha))
             for name, ws in folds.items()]
    if include_val:
        tasks.append(("val", (policy_factory, VAL_START, full_sales, full_in_stock, master, coverage_weeks, alpha)))

    if n_workers and n_workers > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_run_one_window)(args) for _, args in tasks
        )
    else:
        outs = [_run_one_window(args) for _, args in tasks]

    rows = []
    val = None
    for (name, _), out in zip(tasks, outs):
        if name == "val":
            val = out
        else:
            rows.append({"fold": name, **out})
    per_fold = pd.DataFrame(rows).set_index("fold")
    mean = per_fold.select_dtypes(include=[np.number]).mean().to_dict()
    return {"per_fold": per_fold, "mean": mean, "val": val}


def build_extended_sales() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return full sales (weeks 0..156 history + 157..164 sacred) and in_stock.

    History comes from Week 0 - Sales.csv / In Stock.csv. Sacred comes from the
    per-week CSVs in Data/.
    """
    sales_hist, in_stock_hist, _initial, master = load_initial_data()

    # Load hidden weeks 1..8 and append.
    from simulation import DATA_DIR  # local import to avoid top-level cycle
    hidden = {}
    for wk in range(1, 9):
        f = list(DATA_DIR.glob(f"Week {wk} - *Sales.csv"))
        if f:
            df = pd.read_csv(f[0]).set_index(["Store", "Product"])
            df.columns = pd.to_datetime(df.columns)
            hidden[df.columns[-1]] = df[df.columns[-1]]
    hidden_df = pd.DataFrame(hidden)
    full_sales = pd.concat([sales_hist, hidden_df], axis=1)
    # In-stock during hidden weeks is unknown; assume True (1). This only affects
    # censoring logic in the forecaster when it's fit with data >= week 157,
    # which for CV (window_start <= 149) never happens.
    idx = in_stock_hist.index
    hidden_is = pd.DataFrame(True, index=idx, columns=hidden_df.columns)
    full_in_stock = pd.concat([in_stock_hist, hidden_is], axis=1)
    return full_sales, full_in_stock, master


RESULTS_CSV = ROOT / "benchmark" / "experiments.csv"


def log_experiment(name: str, results: dict, extra: dict | None = None,
                   csv_path: Path | None = None) -> None:
    """Append one row per experiment to benchmark/experiments.csv.

    Columns: timestamp, policy, cv_mean_cost, cv_mean_holding, cv_mean_shortage,
             val_cost, val_holding, val_shortage, fc_mae, fc_wape, fc_bias,
             fc_pinball_0.833, fold_costs (semicolon-separated), +extra keys.
    """
    csv_path = Path(csv_path) if csv_path is not None else RESULTS_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    mean = results.get("mean", {})
    val = results.get("val") or {}
    per_fold = results.get("per_fold")
    fold_costs = ""
    if per_fold is not None and "competition_cost" in per_fold.columns:
        fold_costs = ";".join(f"{c:.2f}" for c in per_fold["competition_cost"].tolist())

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "policy": name,
        "cv_mean_cost": mean.get("competition_cost", float("nan")),
        "cv_mean_holding": mean.get("comp_holding", float("nan")),
        "cv_mean_shortage": mean.get("comp_shortage", float("nan")),
        "val_cost": val.get("competition_cost", float("nan")),
        "val_holding": val.get("comp_holding", float("nan")),
        "val_shortage": val.get("comp_shortage", float("nan")),
        "fc_mae": mean.get("fc_mae", float("nan")),
        "fc_wape": mean.get("fc_wape", float("nan")),
        "fc_bias": mean.get("fc_bias", float("nan")),
        "fc_pinball_0.833": mean.get("fc_pinball_0.833", float("nan")),
        "fold_costs": fold_costs,
    }
    if extra:
        row.update(extra)

    df = pd.DataFrame([row])
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


# Published baselines from prior runs; lets the gate warn when VAL drifts off.
BASELINE_VAL_COST = 2855.0  # Ensemble w_ma=0.25 x1.05 VAL (April 2026)
BASELINE_CV_COST = 3654.0


def gated_sacred_eval(
    name: str,
    policy_factory,
    full_sales: pd.DataFrame,
    full_in_stock: pd.DataFrame | None,
    master: pd.DataFrame | None,
    baseline_val_cost: float = BASELINE_VAL_COST,
    val_tolerance: float = 0.10,
    coverage_weeks: int = 2,
    alpha: float | None = None,
    n_workers: int = 1,
    force_sacred: bool = False,
    folds: dict | None = None,
) -> dict:
    """Model-selection gate: run CV+VAL, check VAL coherence, only then touch sacred.

    Decision rule:
      - Run CV (4 folds) + VAL (1 window).
      - If |VAL_cost - baseline_val_cost| / baseline_val_cost > val_tolerance:
          flag; do NOT run sacred unless force_sacred=True.
      - Otherwise run sacred ONCE, log, return all three numbers.

    Returns dict with cv_mean, val, sacred (or None), flagged, reason.
    """
    from simulation import InventorySimulator  # avoid cycle at import time

    # 1) Full CV + VAL.
    res = score_policy_on_folds(
        policy_factory=policy_factory,
        full_sales=full_sales, full_in_stock=full_in_stock, master=master,
        coverage_weeks=coverage_weeks, alpha=alpha,
        folds=folds if folds is not None else FOLDS,
        n_workers=n_workers, include_val=True,
    )
    cv_cost = res["mean"]["competition_cost"]
    val_cost = res["val"]["competition_cost"]

    # 2) Coherence check.
    rel = (val_cost - baseline_val_cost) / baseline_val_cost
    flagged = (rel > val_tolerance)
    reason = ""
    if flagged:
        reason = (
            f"VAL {val_cost:,.0f} is {rel:+.1%} vs baseline {baseline_val_cost:,.0f} "
            f"(tolerance {val_tolerance:.0%})"
        )

    log_experiment(name, res, extra={
        "cv_cost": cv_cost, "val_cost": val_cost,
        "val_rel_vs_baseline": rel,
        "flagged": flagged, "reason": reason,
    })

    out = {
        "name": name,
        "cv_mean": cv_cost,
        "val": val_cost,
        "val_rel_vs_baseline": rel,
        "flagged": flagged,
        "reason": reason,
        "sacred": None,
    }

    print(f"\n[GATE] {name}")
    print(f"  CV mean: {cv_cost:,.2f}  (baseline {BASELINE_CV_COST:,.0f}, "
          f"delta {cv_cost - BASELINE_CV_COST:+.0f})")
    print(f"  VAL:     {val_cost:,.2f}  (baseline {baseline_val_cost:,.0f}, "
          f"{rel:+.1%})")

    # 3) Sacred (gated).
    if flagged and not force_sacred:
        print(f"  --> BLOCKED: {reason}")
        print(f"  --> Re-inspect CV trials, check for overfit, OR pass force_sacred=True.")
        return out

    sim = InventorySimulator()
    sres = sim.run_simulation(policy_factory())
    out["sacred"] = sres["competition_cost"]
    log_experiment(
        f"{name} [SACRED]",
        {"mean": {"competition_cost": sres["competition_cost"],
                  "comp_holding": sres["competition_holding"],
                  "comp_shortage": sres["competition_shortage"]},
         "val": None, "per_fold": None},
        extra={"sacred": True, "gate_passed": True,
               "val_rel_vs_baseline": rel},
    )
    print(f"  SACRED:  {sres['competition_cost']:,.2f}  "
          f"(h={sres['competition_holding']:,.0f}, s={sres['competition_shortage']:,.0f})")
    return out


def format_summary(results: dict, name: str) -> str:
    per_fold = results["per_fold"]
    mean = results["mean"]
    val = results.get("val")
    lines = [
        f"\n=== {name} ===",
        per_fold[["competition_cost", "comp_holding", "comp_shortage"] + [c for c in per_fold.columns if c.startswith("fc_") and c != "fc_n_obs"]].to_string(float_format=lambda x: f"{x:,.2f}"),
        f"  MEAN  competition_cost = {mean['competition_cost']:,.2f}  "
        f"(holding={mean['comp_holding']:,.2f}, shortage={mean['comp_shortage']:,.2f})",
    ]
    if "fc_mae" in mean:
        lines.append(
            f"  MEAN  forecast MAE={mean.get('fc_mae', float('nan')):.2f}  "
            f"WAPE={mean.get('fc_wape', float('nan')):.3f}  "
            f"bias={mean.get('fc_bias', float('nan')):+.2f}"
            + (f"  pinball@0.833={mean['fc_pinball_0.833']:.3f}" if "fc_pinball_0.833" in mean else "")
        )
    if val is not None:
        lines.append(
            f"  VAL   competition_cost = {val['competition_cost']:,.2f}  "
            f"(holding={val['comp_holding']:,.2f}, shortage={val['comp_shortage']:,.2f})"
        )
        if "fc_mae" in val:
            lines.append(
                f"  VAL   forecast MAE={val['fc_mae']:.2f}  WAPE={val['fc_wape']:.3f}  bias={val['fc_bias']:+.2f}"
                + (f"  pinball@0.833={val['fc_pinball_0.833']:.3f}" if "fc_pinball_0.833" in val else "")
            )
    return "\n".join(lines)
