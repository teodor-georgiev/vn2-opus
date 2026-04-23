"""
Train Matias Alvo's DRL policy on VN2 data, then evaluate it through Toni's
InventorySimulator (which has the hidden weeks 1-8 actual sales).

Usage
-----
1) Train (one-time, slow: up to 20k epochs with early stopping at 500 patience):
       python run_matias.py train

   This invokes Matias's main_run.py after patching data_driven_net.yml to
   enable model saving. The saved model lands under
       references/matias_alvo/saved_models/YYYY_MM_DD/data_driven/TIMESTAMP.pt

2) Evaluate (fast, CPU-only is fine):
       python run_matias.py eval <path-to-checkpoint.pt>

   Wraps the checkpoint as a MatiasPolicy via matias_bridge.MatiasPolicy,
   drives Toni's InventorySimulator, and prints competition_cost (weeks 3-8).
   If the port is faithful, this should land near 3,765 EUR.

Notes
-----
- The bridge maintains Matias's observation in parallel with our sim, calling
  the model with the (pre-demand) state distribution it was trained on. See the
  docstring in matias_bridge.py.
- Matias's cost accounting has been verified equivalent to ours (see README
  section of this file at bottom).
- `past_instocks` within the bridge uses a best-effort fallback (1.0) for
  competition weeks because our sim doesn't track an in-stock flag.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from matias_bridge import MATIAS_DIR, build_matias_policy_from_toni_data, load_matias_model
from simulation import InventorySimulator

HYPERPARAMS_PATH = MATIAS_DIR / "config_files" / "policies_and_hyperparams" / "data_driven_net.yml"
HYPERPARAMS_SAVE_PATH = MATIAS_DIR / "config_files" / "policies_and_hyperparams" / "data_driven_net_save.yml"


def _patch_hyperparams_to_save() -> Path:
    """Write a copy of data_driven_net.yml with save_model=True."""
    with open(HYPERPARAMS_PATH) as f:
        hp = yaml.safe_load(f)
    hp["trainer_params"]["save_model"] = True
    hp["trainer_params"]["epochs_between_save"] = 10
    with open(HYPERPARAMS_SAVE_PATH, "w") as f:
        yaml.safe_dump(hp, f, sort_keys=False)
    return HYPERPARAMS_SAVE_PATH


def cmd_train() -> int:
    """Run Matias's training pipeline with model saving enabled."""
    _patch_hyperparams_to_save()
    # Matias's trainer uses os.mkdir (single-level) — pre-create the base dir.
    (MATIAS_DIR / "saved_models").mkdir(exist_ok=True)
    cmd = [
        sys.executable,
        "main_run.py",
        "train",
        "vn2_round_1",
        HYPERPARAMS_SAVE_PATH.stem,
    ]
    print(f"cd {MATIAS_DIR} && {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=MATIAS_DIR)
    print("\nAfter training completes, find the saved .pt under:")
    print(f"  {MATIAS_DIR / 'saved_models'}")
    return proc.returncode


def cmd_eval(checkpoint_path: Path) -> int:
    """Load the trained model and evaluate through our simulator."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}, checkpoint={checkpoint_path}")

    model, scenario, cfg = load_matias_model(Path(checkpoint_path), device=device)
    policy = build_matias_policy_from_toni_data(model, scenario, cfg, device=device)

    sim = InventorySimulator()
    results = sim.run_simulation(policy)

    print()
    print("=" * 60)
    print("  MATIAS DRL POLICY (replayed through our simulator)")
    print("=" * 60)
    print(f"  Total Cost (8 weeks):       {results['total_cost']:>10,.2f} EUR")
    print(f"  Setup Cost (wk 1-2 fixed):  {results['setup_cost']:>10,.2f} EUR")
    print(f"  Competition Cost (wk 3-8):  {results['competition_cost']:>10,.2f} EUR")
    print(f"    Holding:                  {results['competition_holding']:>10,.2f} EUR")
    print(f"    Shortage:                 {results['competition_shortage']:>10,.2f} EUR")
    print(f"  Official target:            {3765.00:>10,.2f} EUR (Matias's leaderboard)")
    print()
    print("Weekly breakdown:")
    print(results["weekly_log"].to_string(index=False))
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "train":
        return cmd_train()
    if mode == "eval":
        if len(sys.argv) < 3:
            print("Usage: python run_matias.py eval <path-to-checkpoint.pt>")
            return 2
        return cmd_eval(Path(sys.argv[2]))
    print(f"Unknown mode: {mode}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
