"""Run the three low-risk improvements end-to-end:
  (1) Seed-averaged eval of best policy on sacred.
  (3) Per-SKU coverage routing based on ADI / cv^2 segmentation.
  (Also logs every run to experiments.csv.)

Optuna (2) and NegBinom (4) are in separate scripts for isolation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from benchmark.cv_harness import (  # noqa: E402
    build_extended_sales,
    format_summary,
    log_experiment,
    score_policy_on_folds,
)
from policies import MLPointPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def classify_skus(full_sales: pd.DataFrame, history_end: int = 156) -> pd.DataFrame:
    """Compute per-SKU demand statistics used for coverage routing.

    Returns DataFrame with columns: mean, std, cv2, nonzero_frac, adi.
    """
    hist = full_sales.iloc[:, :history_end]
    hist = hist.fillna(0.0)
    stats = pd.DataFrame(index=full_sales.index)
    stats["mean"] = hist.mean(axis=1)
    stats["std"] = hist.std(axis=1)
    stats["cv2"] = (stats["std"] / stats["mean"].clip(lower=0.01)) ** 2
    # nonzero fraction and ADI (average demand interval).
    nz = (hist > 0).astype(int)
    stats["nonzero_frac"] = nz.mean(axis=1)
    # ADI = number of periods / number of nonzero periods (Syntetos-Boylan).
    nz_counts = nz.sum(axis=1).clip(lower=1)
    stats["adi"] = hist.shape[1] / nz_counts
    return stats


def coverage_rule_volume(stats: pd.DataFrame) -> pd.Series:
    """Simple volume-based rule:
    mean >= 3   -> cov=3
    1 <= mean <3 -> cov=2
    else (low)   -> cov=1
    """
    cov = pd.Series(2, index=stats.index, dtype=int)
    cov[stats["mean"] >= 3] = 3
    cov[stats["mean"] < 1] = 1
    return cov


def coverage_rule_syntetos(stats: pd.DataFrame) -> pd.Series:
    """Syntetos-Boylan demand classification:
    smooth (ADI<1.32, cv2<0.49)      -> cov=3
    intermittent (ADI>=1.32, cv2<0.49)-> cov=2
    erratic (ADI<1.32, cv2>=0.49)     -> cov=2
    lumpy (ADI>=1.32, cv2>=0.49)      -> cov=1
    """
    cov = pd.Series(2, index=stats.index, dtype=int)
    smooth = (stats["adi"] < 1.32) & (stats["cv2"] < 0.49)
    lumpy = (stats["adi"] >= 1.32) & (stats["cv2"] >= 0.49)
    cov[smooth] = 3
    cov[lumpy] = 1
    return cov


# -----------------  (1) Seed averaging  -----------------

def seed_average_sacred(master, cov=2, multiplier=1.10, seeds=(42, 123, 456, 789, 1234)):
    """Run best policy 5x with different seeds on sacred. Report mean ± std."""
    costs = []
    for seed in seeds:
        sim = InventorySimulator()
        policy = MLPointPolicy(
            coverage_weeks=cov, master=sim.master, multiplier=multiplier,
            random_state=seed,
        )
        res = sim.run_simulation(policy)
        costs.append(res["competition_cost"])
        print(f"  seed={seed:<5d} comp_cost={res['competition_cost']:,.2f}  "
              f"(h={res['competition_holding']:,.2f}, s={res['competition_shortage']:,.2f})")
    mean = float(np.mean(costs))
    std = float(np.std(costs, ddof=1))
    print(f"\n  MEAN = {mean:,.2f}  STD = {std:,.2f}  "
          f"[range {min(costs):,.2f} .. {max(costs):,.2f}]")
    log_experiment(
        f"ML Point cov={cov} x{multiplier} [SACRED, 5-seed mean]",
        {"mean": {"competition_cost": mean}, "val": None, "per_fold": None},
        extra={"sacred": True, "n_seeds": len(seeds), "seeds_std": std,
               "seed_range": f"{min(costs):.2f}..{max(costs):.2f}"},
    )
    return costs, mean, std


# -----------------  (3) Per-SKU coverage  -----------------

def mk_per_sku(master, coverage_per_sku, multiplier=1.05, cov_default=2):
    def f():
        return MLPointPolicy(
            coverage_weeks=cov_default, master=master,
            multiplier=multiplier, coverage_per_sku=coverage_per_sku,
        )
    return f


def main():
    full_sales, full_in_stock, master = build_extended_sales()
    N = int(os.environ.get("CV_WORKERS", "3"))

    # ----- (1) Seed averaging -----
    print("\n" + "=" * 70)
    print("  (1) SEED-AVERAGED EVAL ON SACRED")
    print("=" * 70)
    print("\n  cov=2 x1.10 [5 seeds]")
    seed_average_sacred(master, cov=2, multiplier=1.10)
    print("\n  cov=2 x1.05 [5 seeds]")
    seed_average_sacred(master, cov=2, multiplier=1.05)

    # ----- (3) Per-SKU coverage -----
    print("\n\n" + "=" * 70)
    print("  (3) PER-SKU COVERAGE ROUTING")
    print("=" * 70)
    stats = classify_skus(full_sales, history_end=149)  # classify using pre-VAL data only
    print(f"\n  SKU stats: n={len(stats)}  mean avg demand={stats['mean'].mean():.2f}  "
          f"median={stats['mean'].median():.2f}")

    cov_vol = coverage_rule_volume(stats)
    cov_syn = coverage_rule_syntetos(stats)
    print(f"  Volume rule   coverage distribution: {cov_vol.value_counts().sort_index().to_dict()}")
    print(f"  Syntetos rule coverage distribution: {cov_syn.value_counts().sort_index().to_dict()}")

    experiments = [
        ("Per-SKU [volume] x1.00",   mk_per_sku(master, cov_vol, multiplier=1.00), cov_vol),
        ("Per-SKU [volume] x1.10",   mk_per_sku(master, cov_vol, multiplier=1.10), cov_vol),
        ("Per-SKU [syntetos] x1.00", mk_per_sku(master, cov_syn, multiplier=1.00), cov_syn),
        ("Per-SKU [syntetos] x1.10", mk_per_sku(master, cov_syn, multiplier=1.10), cov_syn),
    ]
    for name, factory, cov_series in experiments:
        print(f"\nRunning: {name}")
        res = score_policy_on_folds(
            policy_factory=factory,
            full_sales=full_sales,
            full_in_stock=full_in_stock,
            master=master,
            coverage_weeks=2,  # default/fallback (per-SKU overrides)
            alpha=None,
            n_workers=N,
        )
        log_experiment(name, res)
        print(format_summary(res, name))

        # Also evaluate on sacred.
        sim = InventorySimulator()
        policy = factory()
        sres = sim.run_simulation(policy)
        print(f"  SACRED competition_cost = {sres['competition_cost']:,.2f} "
              f"(h={sres['competition_holding']:,.2f}, s={sres['competition_shortage']:,.2f})")
        log_experiment(
            f"{name} [SACRED]",
            {"mean": {"competition_cost": sres["competition_cost"],
                      "comp_holding": sres["competition_holding"],
                      "comp_shortage": sres["competition_shortage"]},
             "val": None, "per_fold": None},
            extra={"sacred": True},
        )

    print("\n\nDone. Full history in benchmark/experiments.csv")


if __name__ == "__main__":
    main()
