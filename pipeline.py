"""VN2 experiment orchestrator.

Usage:
    python pipeline.py list                      # list all experiments
    python pipeline.py run <name>                # run one experiment by name
    python pipeline.py run <name> --kwarg val    # pass kwargs to experiment
    python pipeline.py report [--top-n 20]       # print experiments.csv summary
    python pipeline.py sacred --cov 2 --multiplier 1.05   # one-shot sacred

Examples:
    python pipeline.py run ensemble
    python pipeline.py run seed_avg --cov 2 --multiplier 1.10
    python pipeline.py run optuna --n_trials 25
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_kv(items: list[str]) -> dict:
    """Turn ['--cov', '2', '--multiplier', '1.05'] into {cov: 2, multiplier: 1.05}.

    Values are parsed via literal_eval when possible (so '1.05' -> float, '5' -> int,
    'True' -> bool). Otherwise left as str.
    """
    out: dict = {}
    i = 0
    while i < len(items):
        k = items[i]
        if not k.startswith("--"):
            raise SystemExit(f"Unexpected arg: {k!r}")
        key = k[2:].replace("-", "_")
        if i + 1 >= len(items) or items[i + 1].startswith("--"):
            # boolean flag
            out[key] = True
            i += 1
        else:
            v = items[i + 1]
            try:
                v = ast.literal_eval(v)
            except Exception:
                pass
            out[key] = v
            i += 2
    return out


def main():
    from experiments import REGISTRY  # lazy

    p = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list registered experiments")

    rp = sub.add_parser("run", help="run an experiment by name")
    rp.add_argument("name", help="experiment name (use `list` to see options)")
    rp.add_argument("kwargs", nargs=argparse.REMAINDER,
                    help="optional --key value pairs passed to the experiment")

    rep = sub.add_parser("report", help="show experiments.csv summary")
    rep.add_argument("--top-n", type=int, default=15)

    sp = sub.add_parser("sacred", help="one-shot sacred evaluation of a policy")
    sp.add_argument("--cov", type=int, default=2)
    sp.add_argument("--multiplier", type=float, default=1.05)
    sp.add_argument("--censoring", default="interpolate")
    sp.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.cmd == "list" or args.cmd is None:
        print("Registered experiments:")
        for name, meta in sorted(REGISTRY.items()):
            print(f"  {name:<20s} {meta['description']}")
        return

    if args.cmd == "run":
        if args.name not in REGISTRY:
            print(f"Unknown experiment: {args.name!r}. Try `list`.", file=sys.stderr)
            sys.exit(2)
        kwargs = _parse_kv(args.kwargs or [])
        print(f"Running experiment: {args.name}  kwargs={kwargs}")
        REGISTRY[args.name]["fn"](**kwargs)
        return

    if args.cmd == "report":
        REGISTRY["report"]["fn"](top_n=args.top_n)
        return

    if args.cmd == "sacred":
        REGISTRY["sacred"]["fn"](
            cov=args.cov, multiplier=args.multiplier,
            censoring=args.censoring, random_state=args.seed,
        )
        return


if __name__ == "__main__":
    main()
