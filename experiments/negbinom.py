"""NegBinom newsvendor grid over (coverage, alpha)."""
from __future__ import annotations

from benchmark.cv_harness import build_extended_sales
from negbinom_policy import NegBinomPolicy
from experiments._shared import print_summary_table, run_and_log, sacred_eval_and_log


def run():
    fs, fis, master = build_extended_sales()
    grid = [
        (2, 0.50), (2, 0.60), (2, 0.70), (2, 0.833),
        (3, 0.50), (3, 0.60), (3, 0.70),
    ]
    rows = []
    for cov, alpha in grid:
        name = f"NegBinom cov={cov} alpha={alpha}"

        def factory(c=cov, a=alpha):
            return NegBinomPolicy(coverage_weeks=c, alpha=a, multiplier=1.0)
        res = run_and_log(name, factory, fs, fis, master, coverage_weeks=cov, alpha=alpha)
        sres = sacred_eval_and_log(name, factory)
        rows.append((name, res, sres))
    print_summary_table(rows)
