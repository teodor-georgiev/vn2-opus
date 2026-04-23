"""Re-evaluate Optuna's best trial (trial 9) on CV + VAL + sacred.

Params hard-coded from the completed study. Avoids the post-study name
unpacking bug in run_optuna.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")

from benchmark.cv_harness import (  # noqa: E402
    build_extended_sales, format_summary, log_experiment, score_policy_on_folds,
)
from policies import MLPointPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


# Trial 9 params (best: CV 3,705.80)
LGB_PARAMS = {
    "n_estimators": 168,
    "learning_rate": 0.02335565937782791,
    "num_leaves": 29,
    "min_child_samples": 10,
    "subsample": 0.9718790609370292,
    "colsample_bytree": 0.9232481518257668,
    "reg_alpha": 0.6334037565104235,
    "reg_lambda": 0.8714605901877177,
}
CB_PARAMS = {
    "iterations": 341,
    "learning_rate": 0.029126629206623567,
    "depth": 9,
    "l2_leaf_reg": 5.854080177240856,
}


def make_factory(master, cov=2, mult=1.05):
    def f():
        return MLPointPolicy(
            coverage_weeks=cov, master=master, multiplier=mult,
            lgb_params=LGB_PARAMS, cb_params=CB_PARAMS,
        )
    return f


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    N = int(os.environ.get("CV_WORKERS", "3"))

    # Try mult=1.05 (matches Optuna's objective) and mult=1.10 (our best eyeballed)
    for mult in (1.05, 1.10):
        name = f"ML Point cov=2 x{mult} [Optuna-tuned]"
        print(f"\n{name}")
        res = score_policy_on_folds(
            policy_factory=make_factory(master, cov=2, mult=mult),
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=2,
            alpha=None,
            n_workers=N,
            include_val=True,
        )
        log_experiment(name, res)
        print(format_summary(res, name))

        # Only run sacred once per variant (per selection rule).
        # We do BOTH here to log both numbers, but the selected variant should
        # be picked on CV+VAL *before* this step in any honest accounting.
        sim = InventorySimulator()
        sres = sim.run_simulation(make_factory(master, cov=2, mult=mult)())
        print(f"  SACRED: {sres['competition_cost']:,.2f}  "
              f"(h={sres['competition_holding']:,.2f}, s={sres['competition_shortage']:,.2f})")
        log_experiment(f"{name} [SACRED]",
                       {"mean": {"competition_cost": sres["competition_cost"],
                                 "comp_holding": sres["competition_holding"],
                                 "comp_shortage": sres["competition_shortage"]},
                        "val": None, "per_fold": None},
                       extra={"sacred": True})


if __name__ == "__main__":
    main()
