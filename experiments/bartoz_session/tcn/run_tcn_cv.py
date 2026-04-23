"""Evaluate TCNPolicy on CV + VAL folds.

Usage:
    python run_tcn_cv.py --folds cv0          # one fold (smoke test)
    python run_tcn_cv.py --folds all          # 4 CV folds
    python run_tcn_cv.py --folds all --val    # 4 CV + VAL
    python run_tcn_cv.py --folds all --val --mult 1.0 --cov 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import (  # noqa: E402
    FOLDS,
    VAL_START,
    build_extended_sales,
    format_summary,
    log_experiment,
    score_policy_on_folds,
)
from policies import TCNPolicy  # noqa: E402
from tcn_forecaster import TCNConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="cv0", help="'cv0'..'cv3' | 'all'")
    ap.add_argument("--val", action="store_true", help="also run VAL fold (ws=149)")
    ap.add_argument("--cov", type=int, default=2, help="coverage weeks")
    ap.add_argument("--mult", type=float, default=1.0, help="multiplier on mu_sum")
    ap.add_argument("--safety", type=float, default=0.0, help="safety units added")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--tcn-layers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--fine-tune-epochs", type=int, default=None)
    ap.add_argument("--p-time-aug", type=float, default=None)
    ap.add_argument("--p-week-aug", type=float, default=None)
    ap.add_argument("--p-static-aug", type=float, default=None)
    ap.add_argument("--no-softplus", action="store_true", help="disable softplus on decoder output")
    ap.add_argument("--name", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.folds == "all":
        folds = FOLDS
    else:
        if args.folds not in FOLDS:
            raise SystemExit(f"Unknown fold: {args.folds}. Options: {list(FOLDS)} or 'all'")
        folds = {args.folds: FOLDS[args.folds]}

    # Build TCN config from overrides.
    cfg = TCNConfig()
    for k in ("horizon", "hidden", "tcn_layers", "batch_size", "max_epochs",
              "patience", "fine_tune_epochs"):
        v = getattr(args, k.replace("_", "-"), None) if False else getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    # argparse stores hyphens as underscores in namespace
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.patience is not None:
        cfg.patience = args.patience
    if args.fine_tune_epochs is not None:
        cfg.fine_tune_epochs = args.fine_tune_epochs
    if args.horizon is not None:
        cfg.horizon = args.horizon
    if args.hidden is not None:
        cfg.hidden = args.hidden
    if args.tcn_layers is not None:
        cfg.tcn_layers = args.tcn_layers
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.p_time_aug is not None:
        cfg.p_time_aug = args.p_time_aug
    if args.p_week_aug is not None:
        cfg.p_week_aug = args.p_week_aug
    if args.p_static_aug is not None:
        cfg.p_static_aug = args.p_static_aug
    if args.no_softplus:
        cfg.use_softplus_decoder = False

    full_sales, full_in_stock, master = build_extended_sales()

    name = args.name or f"tcn_cov{args.cov}_m{args.mult}_h{cfg.horizon}_hd{cfg.hidden}"
    print(f"[{name}] folds={list(folds)} val={args.val} "
          f"cov={args.cov} mult={args.mult} cfg_horizon={cfg.horizon} hidden={cfg.hidden}")

    t0 = time.time()

    def factory():
        return TCNPolicy(
            coverage_weeks=args.cov,
            multiplier=args.mult,
            safety_units=args.safety,
            cfg=cfg,
            verbose=args.verbose,
        )

    results = score_policy_on_folds(
        factory,
        full_sales=full_sales,
        full_in_stock=full_in_stock,
        master=master,
        folds=folds,
        include_val=args.val,
        coverage_weeks=args.cov,
        alpha=0.833,
        n_workers=1,   # single-GPU; keep serial
    )

    print(format_summary(results, name))
    print(f"\n[{name}] total wall-time: {time.time() - t0:.1f}s")

    extras = {
        "cov": args.cov, "mult": args.mult, "safety": args.safety,
        "horizon": cfg.horizon, "hidden": cfg.hidden, "tcn_layers": cfg.tcn_layers,
        "max_epochs": cfg.max_epochs, "patience": cfg.patience,
        "fine_tune_epochs": cfg.fine_tune_epochs,
    }
    log_experiment(name, results, extra=extras)


if __name__ == "__main__":
    main()
