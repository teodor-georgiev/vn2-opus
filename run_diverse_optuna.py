"""Optuna tuner for 5-model Diverse policy.

Searches (alpha, multiplier, w_ma) on 8-fold CV+VAL.

Objective: CV mean competition_cost (primary) + gating check on VAL.
To avoid CV overfitting: report top-5 trials with both CV and VAL, only shortlist
those where VAL is also within 50 EUR of baseline champion VAL (2,570).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import optuna  # noqa: E402

from benchmark.cv_harness import (  # noqa: E402
    FOLDS_8, build_extended_sales, log_experiment, score_policy_on_folds,
)
from policies import DiverseCostAwarePolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="optuna_diverse5.json")
    args = ap.parse_args()

    full_sales, full_in_stock, master = build_extended_sales()
    baseline_val = 2570.4  # 5-model eq-weight champion

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.45, 0.85)
        mult  = trial.suggest_float("mult",  0.90, 1.15)
        w_ma  = trial.suggest_float("w_ma",  0.00, 0.45)

        def factory(a=alpha, m=mult, w=w_ma):
            return DiverseCostAwarePolicy(
                alpha=a, multiplier=m, w_ma=w,
                backtest_window=26, safety_floor=0.5, rmse_horizons=1,
                censoring_strategy="mean_impute", random_state=42, master=master,
                n_variants=5,
            )

        t0 = time.time()
        results = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales, full_in_stock=full_in_stock, master=master,
            folds=FOLDS_8, include_val=True,
            coverage_weeks=3, alpha=alpha, n_workers=1,
        )
        wall = time.time() - t0
        cv  = results["mean"]["competition_cost"]
        val = (results.get("val") or {}).get("competition_cost", float("nan"))
        trial_records.append({"alpha": alpha, "mult": mult, "w_ma": w_ma,
                               "cv": cv, "val": val, "wall": round(wall, 1)})
        print(f"  trial {trial.number}: alpha={alpha:.3f} mult={mult:.3f} w_ma={w_ma:.3f} "
              f"=> CV={cv:.1f}  VAL={val:.1f}  ({wall:.0f}s)", flush=True)
        return cv

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\n=== OPTUNA DONE ({args.trials} trials) ===")
    print(f"best CV: {study.best_value:,.2f}")
    print(f"best params: {study.best_params}")

    # Rank by CV and by VAL separately
    print("\nTop 5 by CV:")
    for r in sorted(trial_records, key=lambda x: x["cv"])[:5]:
        print(f"  alpha={r['alpha']:.3f} mult={r['mult']:.3f} w_ma={r['w_ma']:.3f} "
              f"=> CV={r['cv']:.1f}  VAL={r['val']:.1f}")
    print("\nTop 5 by VAL:")
    for r in sorted(trial_records, key=lambda x: x["val"])[:5]:
        print(f"  alpha={r['alpha']:.3f} mult={r['mult']:.3f} w_ma={r['w_ma']:.3f} "
              f"=> CV={r['cv']:.1f}  VAL={r['val']:.1f}")

    # Pareto-ish best: lowest CV among those where VAL is not regressing.
    shortlist = [r for r in trial_records if r["val"] <= baseline_val + 30]
    if shortlist:
        best = min(shortlist, key=lambda x: x["cv"])
        print(f"\nCV+VAL-gated winner (VAL within 30 EUR of baseline {baseline_val}):")
        print(f"  alpha={best['alpha']:.3f} mult={best['mult']:.3f} w_ma={best['w_ma']:.3f} "
              f"=> CV={best['cv']:.1f}  VAL={best['val']:.1f}")
        with open(args.out, "w") as f:
            json.dump({
                "chosen": best,
                "all_trials": trial_records,
                "best_cv_params": study.best_params,
                "best_cv_value": study.best_value,
            }, f, indent=2)
        print(f"Saved {args.out}")
    else:
        print("\nNo trials with VAL within 30 EUR of baseline. Shortlist empty.")


if __name__ == "__main__":
    main()
