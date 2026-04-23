"""Train PPO+GAE on VN2 episodes; eval each iteration on VAL episode.

Episodes are pre-built once (cached forecaster fits) and reused across PPO iters.
Training episodes = the 8-fold CV windows (FOLDS_8 starts).
Val episode = window_start = VAL_START (149).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark.cv_harness import FOLDS_8, VAL_START, build_extended_sales  # noqa: E402
from rl_env import EnvConfig, make_episodes  # noqa: E402
from rl_policy import PPOConfig, PPOTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--rollouts-per-iter", type=int, default=4,
                    help="number of (random) train episodes rolled per PPO update")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="rl_best.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[rl] device={args.device}")
    print("[rl] loading data...")
    full_sales, full_in_stock, master = build_extended_sales()

    print(f"[rl] building train episodes (8-fold CV starts)...")
    train_starts = list(FOLDS_8.values())
    train_eps = make_episodes(full_sales, full_in_stock, master, train_starts, EnvConfig())

    print(f"[rl] building val episode (window_start={VAL_START})...")
    val_eps = make_episodes(full_sales, full_in_stock, master, [VAL_START], EnvConfig())

    cfg = PPOConfig(lr=args.lr)
    trainer = PPOTrainer(cfg, device=args.device)

    # Initial val eval
    val0 = trainer.evaluate(val_eps, deterministic=True)
    print(f"[rl] iter 0 (init policy): VAL={val0:,.2f}")

    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    rng = np.random.default_rng(args.seed)
    train_costs_recent = []

    for it in range(1, args.iters + 1):
        # Sample episodes for this iteration.
        ep_idx = rng.choice(len(train_eps), size=args.rollouts_per_iter, replace=True)
        all_obs, all_act, all_logp, all_adv, all_ret = [], [], [], [], []
        ep_costs = []
        for i in ep_idx:
            traj = trainer.collect_trajectory(train_eps[i], deterministic=False)
            all_obs.append(traj["obs"])
            all_act.append(traj["act"])
            all_logp.append(traj["logp_old"])
            all_adv.append(traj["adv"])
            all_ret.append(traj["ret"])
            ep_costs.append(traj["episode_total_cost"])
        batch = {
            "obs": np.concatenate(all_obs, axis=0),
            "act": np.concatenate(all_act, axis=0),
            "logp_old": np.concatenate(all_logp, axis=0),
            "adv": np.concatenate(all_adv, axis=0),
            "ret": np.concatenate(all_ret, axis=0),
        }
        # Re-normalize advantages across the merged batch.
        adv = batch["adv"]
        batch["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        info = trainer.update(batch)
        train_costs_recent.append(float(np.mean(ep_costs)))
        if len(train_costs_recent) > 5:
            train_costs_recent.pop(0)

        # Periodic VAL eval
        if it % 2 == 0 or it == 1 or it == args.iters:
            val_cost = trainer.evaluate(val_eps, deterministic=True)
            train_avg = float(np.mean(train_costs_recent))
            print(f"[rl] iter {it:3d}  train_cost~{train_avg:.0f}  VAL={val_cost:,.2f}  "
                  f"loss_pi={info['loss_pi']:+.3f} loss_v={info['loss_v']:.3f} entropy={info['entropy']:+.3f}",
                  flush=True)
            if val_cost < best_val:
                best_val = val_cost
                best_state = {k: v.cpu().clone() for k, v in trainer.net.state_dict().items()}
                torch.save(best_state, args.save)
                patience_left = args.patience
                print(f"[rl] -> new best VAL={best_val:.2f}, saved -> {args.save}", flush=True)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[rl] early stop at iter {it} (no improvement in {args.patience} evals)")
                    break

    print(f"\n[rl] DONE. best VAL = {best_val:,.2f}  (vs CostAware-baseline VAL=2,593)")


if __name__ == "__main__":
    main()
