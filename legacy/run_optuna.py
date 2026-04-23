"""Optuna tuning of LGBM + CatBoost hyperparams for MLPointPolicy cov=2 x1.05.

Objective: minimize CV mean competition_cost over 4 folds (no VAL to save time).
~20 trials, ~30-45 min on this box.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import optuna  # noqa: E402

from benchmark.cv_harness import (  # noqa: E402
    build_extended_sales,
    log_experiment,
    score_policy_on_folds,
)
from policies import MLPointPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def make_factory(master, lgb_params, cb_params, cov=2, mult=1.05):
    def f():
        return MLPointPolicy(
            coverage_weeks=cov, master=master, multiplier=mult,
            lgb_params=lgb_params, cb_params=cb_params,
        )
    return f


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    N = int(os.environ.get("CV_WORKERS", "3"))

    def objective(trial: optuna.Trial) -> float:
        lgb_params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("lgb_lr", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 40),
            "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("lgb_reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 0.0, 1.0),
        }
        cb_params = {
            "iterations": trial.suggest_int("cb_iterations", 100, 400),
            "learning_rate": trial.suggest_float("cb_lr", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("cb_depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("cb_l2", 1.0, 10.0),
        }
        res = score_policy_on_folds(
            policy_factory=make_factory(master, lgb_params, cb_params),
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=2,
            alpha=None,
            n_workers=N,
            include_val=False,  # VAL added after optimization
        )
        cost = res["mean"]["competition_cost"]
        trial.set_user_attr("holding", res["mean"]["comp_holding"])
        trial.set_user_attr("shortage", res["mean"]["comp_shortage"])
        return cost

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    n_trials = int(os.environ.get("OPTUNA_TRIALS", "25"))
    print(f"Starting Optuna study with {n_trials} trials ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\nBest trial:")
    print(f"  value (CV mean comp_cost): {study.best_value:,.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Re-run best params with VAL + sacred.
    best = study.best_params
    lgb_params = {k.replace("lgb_", ""): v for k, v in best.items() if k.startswith("lgb_")}
    cb_params = {k.replace("cb_", ""): v for k, v in best.items() if k.startswith("cb_")}

    print("\nRe-evaluating best params on CV + VAL + sacred ...")
    res = score_policy_on_folds(
        policy_factory=make_factory(master, lgb_params, cb_params),
        full_sales=full_sales, full_in_stock=full_in_stock, master=master,
        coverage_weeks=2, alpha=None, n_workers=N, include_val=True,
    )
    print(f"  CV mean   : {res['mean']['competition_cost']:,.2f}")
    print(f"  VAL       : {res['val']['competition_cost']:,.2f}")
    log_experiment("ML Point cov=2 x1.05 [Optuna-tuned]", res,
                   extra={"n_trials": n_trials, "best_params": str(best)})

    sim = InventorySimulator()
    policy = make_factory(master, lgb_params, cb_params)()
    sres = sim.run_simulation(policy)
    print(f"  SACRED    : {sres['competition_cost']:,.2f}  "
          f"(h={sres['competition_holding']:,.2f}, s={sres['competition_shortage']:,.2f})")
    log_experiment(
        "ML Point cov=2 x1.05 [Optuna-tuned, SACRED]",
        {"mean": {"competition_cost": sres["competition_cost"],
                  "comp_holding": sres["competition_holding"],
                  "comp_shortage": sres["competition_shortage"]},
         "val": None, "per_fold": None},
        extra={"sacred": True, "best_params": str(best)},
    )


if __name__ == "__main__":
    main()
