"""Optuna hyperparameter search for MLPointPolicy (cov=2 x1.05)."""
from __future__ import annotations

import os

import optuna

from benchmark.cv_harness import build_extended_sales, log_experiment, score_policy_on_folds
from policies import MLPointPolicy
from experiments._shared import get_n_workers


def _mk(master, lgb_params, cb_params, cov=2, mult=1.05, censoring="mean_impute"):
    def f():
        return MLPointPolicy(
            coverage_weeks=cov, master=master, multiplier=mult,
            lgb_params=lgb_params, cb_params=cb_params,
            censoring_strategy=censoring,
        )
    return f


def run(n_trials: int | None = None):
    fs, fis, master = build_extended_sales()
    N = get_n_workers()
    if n_trials is None:
        n_trials = int(os.environ.get("OPTUNA_TRIALS", "20"))

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
            policy_factory=_mk(master, lgb_params, cb_params),
            full_sales=fs, full_in_stock=fis, master=master,
            coverage_weeks=2, alpha=None, n_workers=N, include_val=False,
        )
        return res["mean"]["competition_cost"]

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    print(f"Starting Optuna study with {n_trials} trials ...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest trial: value={study.best_value:,.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Re-eval best on VAL + sacred.
    best = study.best_params
    name_map_lgb = {
        "lgb_n_estimators": "n_estimators",
        "lgb_lr": "learning_rate",
        "lgb_num_leaves": "num_leaves",
        "lgb_min_child_samples": "min_child_samples",
        "lgb_subsample": "subsample",
        "lgb_colsample": "colsample_bytree",
        "lgb_reg_alpha": "reg_alpha",
        "lgb_reg_lambda": "reg_lambda",
    }
    name_map_cb = {
        "cb_iterations": "iterations",
        "cb_lr": "learning_rate",
        "cb_depth": "depth",
        "cb_l2": "l2_leaf_reg",
    }
    lgb_params = {name_map_lgb[k]: v for k, v in best.items() if k in name_map_lgb}
    cb_params = {name_map_cb[k]: v for k, v in best.items() if k in name_map_cb}

    print("\nRe-evaluating best on CV + VAL + SACRED ...")
    res = score_policy_on_folds(
        policy_factory=_mk(master, lgb_params, cb_params),
        full_sales=fs, full_in_stock=fis, master=master,
        coverage_weeks=2, alpha=None, n_workers=N, include_val=True,
    )
    print(f"  CV mean: {res['mean']['competition_cost']:,.2f}")
    print(f"  VAL    : {res['val']['competition_cost']:,.2f}")
    log_experiment("ML Point cov=2 x1.05 [Optuna-tuned]", res,
                   extra={"n_trials": n_trials, "best_params": str(best)})

    # Sacred (single-shot).
    from simulation import InventorySimulator
    sim = InventorySimulator()
    sres = sim.run_simulation(_mk(master, lgb_params, cb_params)())
    print(f"  SACRED : {sres['competition_cost']:,.2f}")
    log_experiment(
        "ML Point cov=2 x1.05 [Optuna-tuned, SACRED]",
        {"mean": {"competition_cost": sres["competition_cost"],
                  "comp_holding": sres["competition_holding"],
                  "comp_shortage": sres["competition_shortage"]},
         "val": None, "per_fold": None},
        extra={"sacred": True, "best_params": str(best)},
    )
