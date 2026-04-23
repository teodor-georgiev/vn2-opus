"""Optuna tuner for 5-model Diverse with per-demand-bin multipliers.

Joint search: (alpha, mult_low, mult_high, w_ma). 15 trials.
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
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--high-q", type=float, default=0.75)
    ap.add_argument("--out", default="optuna_diverse5_bin.json")
    args = ap.parse_args()

    full_sales, full_in_stock, master = build_extended_sales()
    baseline_val = 2595.60  # Optuna strict-CV winner VAL (current champion's VAL)
    champion_cv = 4249.05

    records = []

    def objective(trial: optuna.Trial) -> float:
        alpha      = trial.suggest_float("alpha",      0.50, 0.75)
        mult_low   = trial.suggest_float("mult_low",   1.00, 1.25)
        mult_high  = trial.suggest_float("mult_high",  0.90, 1.30)
        w_ma       = trial.suggest_float("w_ma",       0.20, 0.45)

        def factory(a=alpha, ml=mult_low, mh=mult_high, w=w_ma):
            return DiverseCostAwarePolicy(
                alpha=a, multiplier=1.0,  # unused when mult_high/mult_low set
                mult_high=mh, mult_low=ml, high_demand_quantile=args.high_q,
                w_ma=w,
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
        records.append({"alpha": alpha, "mult_low": mult_low, "mult_high": mult_high,
                        "w_ma": w_ma, "cv": cv, "val": val, "wall": round(wall, 1)})
        print(f"  trial {trial.number}: a={alpha:.3f} ml={mult_low:.3f} mh={mult_high:.3f} w={w_ma:.3f} "
              f"=> CV={cv:.1f}  VAL={val:.1f}  ({wall:.0f}s)", flush=True)
        return cv

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    print(f"\n=== OPTUNA-BIN DONE ({args.trials} trials) ===")
    print(f"best CV: {study.best_value:,.2f}  (champion CV: {champion_cv:,.2f})")
    print(f"best params: {study.best_params}")

    print("\nTop 5 by CV:")
    for r in sorted(records, key=lambda x: x["cv"])[:5]:
        print(f"  a={r['alpha']:.3f} ml={r['mult_low']:.3f} mh={r['mult_high']:.3f} "
              f"w={r['w_ma']:.3f} => CV={r['cv']:.1f}  VAL={r['val']:.1f}")

    print("\nTop 5 by VAL:")
    for r in sorted(records, key=lambda x: x["val"])[:5]:
        print(f"  a={r['alpha']:.3f} ml={r['mult_low']:.3f} mh={r['mult_high']:.3f} "
              f"w={r['w_ma']:.3f} => CV={r['cv']:.1f}  VAL={r['val']:.1f}")

    # Gate: best CV within trials where VAL is within 30 EUR of champion VAL
    gated = [r for r in records if r["val"] <= baseline_val + 30]
    if gated:
        best = min(gated, key=lambda x: x["cv"])
        print(f"\nCV+VAL-gated winner (VAL within 30 EUR of {baseline_val}):")
        print(f"  a={best['alpha']:.3f} ml={best['mult_low']:.3f} mh={best['mult_high']:.3f} "
              f"w={best['w_ma']:.3f} => CV={best['cv']:.1f}  VAL={best['val']:.1f}")
        with open(args.out, "w") as f:
            json.dump({"chosen": best, "all_trials": records,
                       "best_cv_params": study.best_params,
                       "best_cv_value": study.best_value}, f, indent=2)
        print(f"Saved {args.out}")
    else:
        print(f"\nNo trials with VAL within 30 EUR of baseline {baseline_val}.")


if __name__ == "__main__":
    main()
