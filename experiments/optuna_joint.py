"""Joint Optuna over forecaster hyperparams AND ensemble weights.

16-dim search space but with NARROWED ranges centered on known-good values
to curb CV overfitting. First trial seeded with the optuna_weights winner
so we can't regress below it.
"""
from __future__ import annotations

import os

import optuna

from benchmark.cv_harness import build_extended_sales, log_experiment, score_policy_on_folds
from policies import EnsemblePolicy
from experiments._shared import get_n_workers
from simulation import InventorySimulator


# Prior optuna_weights winner (CV 3,625.55, sacred 3,590).
PRIOR_WEIGHTS = dict(
    w_ma=0.2875,
    w_lgb_share=0.6971,
    multiplier=1.0731,
    safety_units=0.5962,
)
# LGB / CB defaults (our fixed baseline in forecaster.py).
PRIOR_LGB = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
)
PRIOR_CB = dict(iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=3.0)


def _mk(master, weights, lgb_params, cb_params, censoring="mean_impute"):
    def f():
        return EnsemblePolicy(
            coverage_weeks=2,
            w_ma=weights["w_ma"],
            multiplier=weights["multiplier"],
            safety_units=weights["safety_units"],
            w_lgb_share=weights["w_lgb_share"],
            master=master, censoring_strategy=censoring,
        )
    return f


def run(n_trials: int | None = None):
    fs, fis, master = build_extended_sales()
    N = get_n_workers()
    if n_trials is None:
        n_trials = int(os.environ.get("OPTUNA_TRIALS", "40"))

    def objective(trial: optuna.Trial) -> float:
        # Ensemble weights — narrow ranges around prior best.
        w_ma = trial.suggest_float("w_ma", 0.15, 0.45)
        w_lgb_share = trial.suggest_float("w_lgb_share", 0.40, 0.95)
        multiplier = trial.suggest_float("multiplier", 1.00, 1.15)
        safety_units = trial.suggest_float("safety_units", 0.0, 1.5)

        # LGBM hyperparams — narrow ranges around defaults.
        lgb_params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 150, 350),
            "learning_rate": trial.suggest_float("lgb_lr", 0.03, 0.10, log=True),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 45),
            "min_child_samples": trial.suggest_int("lgb_min_child_samples", 8, 25),
            "subsample": trial.suggest_float("lgb_subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.7, 0.95),
            "reg_alpha": trial.suggest_float("lgb_reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 0.0, 0.5),
        }
        cb_params = {
            "iterations": trial.suggest_int("cb_iterations", 150, 350),
            "learning_rate": trial.suggest_float("cb_lr", 0.03, 0.10, log=True),
            "depth": trial.suggest_int("cb_depth", 5, 8),
            "l2_leaf_reg": trial.suggest_float("cb_l2", 2.0, 8.0),
        }

        # Attach hyperparams via MLPointPolicy-style injection.
        def factory():
            p = EnsemblePolicy(
                coverage_weeks=2,
                w_ma=w_ma, multiplier=multiplier,
                safety_units=safety_units,
                w_lgb_share=w_lgb_share,
                master=master, censoring_strategy="mean_impute",
            )
            p.lgb_params = lgb_params
            p.cb_params = cb_params
            return p

        res = score_policy_on_folds(
            policy_factory=factory,
            full_sales=fs, full_in_stock=fis, master=master,
            coverage_weeks=2, alpha=None, n_workers=N, include_val=False,
        )
        return res["mean"]["competition_cost"]

    sampler = optuna.samplers.TPESampler(seed=42)
    # Persist the study to sqlite so we can resume / analyze later.
    from pathlib import Path as _P
    study_db = _P("benchmark") / "optuna_joint.db"
    study_db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", sampler=sampler,
        study_name="vn2_optuna_joint",
        storage=f"sqlite:///{study_db.as_posix()}",
        load_if_exists=True,
    )

    # Seed first trial with our prior-best config so Optuna never does worse.
    seed_params = dict(
        **PRIOR_WEIGHTS,
        lgb_n_estimators=PRIOR_LGB["n_estimators"],
        lgb_lr=PRIOR_LGB["learning_rate"],
        lgb_num_leaves=PRIOR_LGB["num_leaves"],
        lgb_min_child_samples=PRIOR_LGB["min_child_samples"],
        lgb_subsample=PRIOR_LGB["subsample"],
        lgb_colsample=PRIOR_LGB["colsample_bytree"],
        lgb_reg_alpha=PRIOR_LGB["reg_alpha"],
        lgb_reg_lambda=PRIOR_LGB["reg_lambda"],
        cb_iterations=PRIOR_CB["iterations"],
        cb_lr=PRIOR_CB["learning_rate"],
        cb_depth=PRIOR_CB["depth"],
        cb_l2=PRIOR_CB["l2_leaf_reg"],
    )
    study.enqueue_trial(seed_params)

    print(f"Starting joint Optuna with {n_trials} trials (seeded with prior winner)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest trial: CV mean = {study.best_value:,.2f}")
    for k, v in study.best_params.items():
        vv = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k}: {vv}")

    # Re-eval best on CV + VAL + SACRED via gated flow.
    bp = study.best_params
    lgb_params = {
        "n_estimators": bp["lgb_n_estimators"],
        "learning_rate": bp["lgb_lr"],
        "num_leaves": bp["lgb_num_leaves"],
        "min_child_samples": bp["lgb_min_child_samples"],
        "subsample": bp["lgb_subsample"],
        "colsample_bytree": bp["lgb_colsample"],
        "reg_alpha": bp["lgb_reg_alpha"],
        "reg_lambda": bp["lgb_reg_lambda"],
    }
    cb_params = {
        "iterations": bp["cb_iterations"],
        "learning_rate": bp["cb_lr"],
        "depth": bp["cb_depth"],
        "l2_leaf_reg": bp["cb_l2"],
    }

    def final_factory():
        p = EnsemblePolicy(
            coverage_weeks=2,
            w_ma=bp["w_ma"], multiplier=bp["multiplier"],
            safety_units=bp["safety_units"],
            w_lgb_share=bp["w_lgb_share"],
            master=master, censoring_strategy="mean_impute",
        )
        p.lgb_params = lgb_params
        p.cb_params = cb_params
        return p

    # Use the gated evaluator to enforce CV+VAL coherence check.
    from benchmark.cv_harness import gated_sacred_eval
    out = gated_sacred_eval(
        name="Ensemble [Optuna-joint]",
        policy_factory=final_factory,
        full_sales=fs, full_in_stock=fis, master=master,
        val_tolerance=0.10, force_sacred=False,
        n_workers=N, coverage_weeks=2,
    )
    if out["flagged"]:
        print("\nJoint tuning blocked by VAL gate. Sticking with prior optuna_weights winner.")

    # Save winner config as JSON alongside the CSV entry.
    import json
    from datetime import datetime
    from pathlib import Path as _P
    winner = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "study_name": "vn2_optuna_joint",
        "n_trials": n_trials,
        "cv_value": float(study.best_value),
        "val": out.get("val"),
        "sacred": out.get("sacred"),
        "flagged": out["flagged"],
        "reason": out.get("reason", ""),
        "weights": {k: bp[k] for k in ("w_ma", "w_lgb_share", "multiplier", "safety_units")},
        "lgb_params": lgb_params,
        "cb_params": cb_params,
    }
    out_path = _P("best_winner_joint.json")
    out_path.write_text(json.dumps(winner, indent=2))
    print(f"\nSaved winner config to {out_path}")
    return out
