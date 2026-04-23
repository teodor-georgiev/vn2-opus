"""Optuna over ENSEMBLE WEIGHTS, with sqlite persistence + consensus-of-top-K.

Search space:
  w_ma          in [0.00, 0.70]
  w_lgb_share   in [0.00, 1.00]
  multiplier    in [0.95, 1.20]
  safety_units  in [-1.0, 2.0]

After optimization:
  1. Save study to sqlite (benchmark/optuna_weights.db).
  2. Save winner config as best_winner_weights.json.
  3. Build CONSENSUS of top-K trials (Bayesian model averaging).
  4. Run gated_sacred_eval on both single-best AND consensus.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import optuna

from benchmark.cv_harness import build_extended_sales, gated_sacred_eval, log_experiment, score_policy_on_folds
from policies import ConsensusPolicy, EnsemblePolicy
from experiments._shared import get_n_workers


def _mk(master, w_ma, w_lgb_share, multiplier, safety_units,
        censoring="mean_impute"):
    def f():
        return EnsemblePolicy(
            coverage_weeks=2, w_ma=w_ma, multiplier=multiplier,
            safety_units=safety_units, w_lgb_share=w_lgb_share,
            master=master, censoring_strategy=censoring,
        )
    return f


def run(n_trials: int | None = None, top_k: int = 5):
    fs, fis, master = build_extended_sales()
    N = get_n_workers()
    if n_trials is None:
        n_trials = int(os.environ.get("OPTUNA_TRIALS", "30"))

    def objective(trial: optuna.Trial) -> float:
        w_ma = trial.suggest_float("w_ma", 0.0, 0.70)
        w_lgb_share = trial.suggest_float("w_lgb_share", 0.0, 1.0)
        multiplier = trial.suggest_float("multiplier", 0.95, 1.20)
        safety_units = trial.suggest_float("safety_units", -1.0, 2.0)
        res = score_policy_on_folds(
            policy_factory=_mk(master, w_ma, w_lgb_share, multiplier, safety_units),
            full_sales=fs, full_in_stock=fis, master=master,
            coverage_weeks=2, alpha=None, n_workers=N, include_val=False,
        )
        trial.set_user_attr("holding", res["mean"]["comp_holding"])
        trial.set_user_attr("shortage", res["mean"]["comp_shortage"])
        return res["mean"]["competition_cost"]

    # Persistent sqlite study.
    sampler = optuna.samplers.TPESampler(seed=42)
    study_db = Path("benchmark") / "optuna_weights.db"
    study_db.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize", sampler=sampler,
        study_name="vn2_optuna_weights",
        storage=f"sqlite:///{study_db.as_posix()}",
        load_if_exists=True,
    )
    print(f"Starting Optuna weight-search with {n_trials} new trials "
          f"(existing trials in study: {len(study.trials)})")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nBest trial: CV mean = {study.best_value:,.2f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.4f}")

    bp = study.best_params
    single_factory = _mk(master, bp["w_ma"], bp["w_lgb_share"], bp["multiplier"], bp["safety_units"])

    # Gated eval of single-best.
    print("\n--- Gated eval: SINGLE-BEST ---")
    single_out = gated_sacred_eval(
        name="Ensemble [Optuna-weights, single-best]",
        policy_factory=single_factory,
        full_sales=fs, full_in_stock=fis, master=master,
        val_tolerance=0.10, force_sacred=False,
        n_workers=N, coverage_weeks=2,
    )

    # Consensus of top-K.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top = sorted(completed, key=lambda t: t.value)[:top_k]
    print(f"\n--- Top-{top_k} by CV for consensus ---")
    for t in top:
        p = t.params
        print(f"  Trial {t.number}: CV={t.value:.2f}  "
              f"w_ma={p['w_ma']:.3f} w_lgb={p['w_lgb_share']:.3f} "
              f"mult={p['multiplier']:.3f} safety={p['safety_units']:.3f}")

    factories = [
        _mk(master, t.params["w_ma"], t.params["w_lgb_share"],
            t.params["multiplier"], t.params["safety_units"])
        for t in top
    ]

    def consensus_factory():
        return ConsensusPolicy([f() for f in factories])

    print(f"\n--- Gated eval: CONSENSUS of top-{top_k} ---")
    consensus_out = gated_sacred_eval(
        name=f"Ensemble [Optuna-weights, consensus-top{top_k}]",
        policy_factory=consensus_factory,
        full_sales=fs, full_in_stock=fis, master=master,
        val_tolerance=0.10, force_sacred=False,
        n_workers=N, coverage_weeks=2,
    )

    # Save both.
    out_path = Path("best_winner_weights.json")
    out_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_trials_total": len(study.trials),
        "study_name": study.study_name,
        "single_best": {
            "cv_value": float(study.best_value),
            "params": bp,
            "gate": {k: v for k, v in single_out.items() if k != "name"},
        },
        f"consensus_top{top_k}": {
            "trial_ids": [t.number for t in top],
            "trial_cvs": [float(t.value) for t in top],
            "gate": {k: v for k, v in consensus_out.items() if k != "name"},
        },
    }, indent=2, default=str))
    print(f"\nSaved: {out_path}")

    # Summary.
    print("\n" + "=" * 70)
    print(f"  {'Selection':<40s} {'CV':>8s} {'VAL':>8s} {'SACRED':>8s}")
    print("  " + "-" * 66)
    for name, out in [("Single-best", single_out), (f"Consensus top-{top_k}", consensus_out)]:
        sac = f"{out['sacred']:,.0f}" if out["sacred"] is not None else "GATED"
        print(f"  {name:<40s} {out['cv_mean']:>8,.0f} {out['val']:>8,.0f} {sac:>8s}")
