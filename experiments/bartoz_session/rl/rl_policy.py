"""Actor-Critic + PPO+GAE for VN2 Stage 2b.

Shared MLP across SKUs. Per-SKU obs (6-dim) → policy outputs (mean, log_std)
for two actions (mult_raw, buffer_raw); critic outputs scalar value.

PPO with:
 - clipped surrogate objective (epsilon = 0.2)
 - value loss (MSE on returns)
 - entropy bonus (encourages exploration)
 - GAE-Lambda advantages (gamma=0.99, lam=0.95)

We treat each SKU's 6-step-trajectory as an independent rollout for the same
shared network — i.e. N_SKU * 6 transitions per episode.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    obs_dim: int = 6
    action_dim: int = 2
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    batch_size: int = 4096
    log_std_init: float = -0.5
    log_std_min: float = -2.5
    log_std_max: float = 1.0
    grad_clip: float = 0.5


class ActorCritic(nn.Module):
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        h = cfg.hidden
        self.shared = nn.Sequential(
            nn.Linear(cfg.obs_dim, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
        )
        self.mu_head = nn.Linear(h, cfg.action_dim)
        self.value_head = nn.Linear(h, 1)
        # Learnable log_std shared across the batch (per action dim).
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.log_std_init))
        self.cfg = cfg

    def get_dist(self, obs: torch.Tensor):
        h = self.shared(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.clamp(self.cfg.log_std_min, self.cfg.log_std_max)
        std = log_std.exp()
        dist = torch.distributions.Independent(
            torch.distributions.Normal(mu, std), 1
        )
        return dist, h

    def forward(self, obs: torch.Tensor):
        dist, h = self.get_dist(obs)
        v = self.value_head(h).squeeze(-1)
        return dist, v

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist, h = self.get_dist(obs)
        if deterministic:
            a = dist.mean
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        v = self.value_head(h).squeeze(-1)
        return a, logp, v


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """rewards/dones: [T], values: [T+1] (last is bootstrap). Returns (adv, ret)."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterm - values[t]
        last_gae = delta + gamma * lam * nonterm * last_gae
        adv[t] = last_gae
    ret = adv + values[:T]
    return adv, ret


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(cfg).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def collect_trajectory(self, env, deterministic: bool = False) -> dict:
        """Run one episode. Returns per-step (obs, action, logp, value, reward, done).

        Each "step" emits N_SKU transitions (for the shared policy). We stack them.
        """
        cfg = self.cfg
        obs_list, act_list, logp_list, val_list, rew_list, done_list = [], [], [], [], [], []
        info_last = {}

        obs_dict = env.reset()
        n_sku = obs_dict["obs"].shape[0]
        for r in range(6):  # NUM_ROUNDS
            obs_t = torch.as_tensor(obs_dict["obs"], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                a, logp, v = self.net.act(obs_t, deterministic=deterministic)
            a_np = a.cpu().numpy().astype(np.float32)
            obs_dict_next, reward, done, info = env.step(a_np)
            # Per-step quantities are scalar (reward applies to the whole batch); we
            # broadcast to per-SKU so each SKU "owns" the same reward for its action.
            obs_list.append(obs_dict["obs"])  # [N, 6]
            act_list.append(a_np)             # [N, 2]
            logp_list.append(logp.cpu().numpy())  # [N]
            val_list.append(v.cpu().numpy())      # [N]
            # Distribute the WEEKLY reward proportional to per-SKU cost contribution
            # would be ideal, but that requires more bookkeeping. For an MVP, give
            # every SKU the average per-SKU reward (reward / N).
            rew_list.append(np.full(n_sku, reward / max(n_sku, 1), dtype=np.float32))
            done_list.append(np.full(n_sku, float(done), dtype=np.float32))
            if not done:
                obs_dict = obs_dict_next
            else:
                info_last = info
                break

        # Bootstrap value for the last state.
        if done:
            last_v = np.zeros(n_sku, dtype=np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_dict["obs"], dtype=torch.float32, device=self.device)
                _, last_v_t = self.net.forward(obs_t)
                last_v = last_v_t.cpu().numpy()

        # Stack to per-SKU per-step arrays: time x sku flattened.
        obs = np.concatenate(obs_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        logp = np.concatenate(logp_list, axis=0)
        val = np.concatenate(val_list, axis=0)
        rew = np.concatenate(rew_list, axis=0)
        dones = np.concatenate(done_list, axis=0)

        # GAE per SKU separately. We stored time-major [T*N], reorder per SKU.
        T = len(obs_list)
        N = n_sku
        rew_tn = np.array(rew_list)         # [T, N]
        val_tn = np.array(val_list)          # [T, N]
        done_tn = np.array(done_list)        # [T, N]
        # Append bootstrap value:
        val_tn_full = np.concatenate([val_tn, last_v[None, :]], axis=0)  # [T+1, N]

        adv = np.zeros((T, N), dtype=np.float32)
        ret = np.zeros((T, N), dtype=np.float32)
        for n in range(N):
            a_n, r_n = compute_gae(rew_tn[:, n], val_tn_full[:, n], done_tn[:, n],
                                   self.cfg.gamma, self.cfg.lam)
            adv[:, n] = a_n
            ret[:, n] = r_n

        # Normalize advantages.
        adv_flat = adv.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        ret_flat = ret.reshape(-1)

        return {
            "obs": obs, "act": act, "logp_old": logp,
            "adv": adv_flat, "ret": ret_flat,
            "episode_total_cost": info_last.get("episode_total_cost", float("nan")),
        }

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(batch["logp_old"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)

        N = obs.shape[0]
        idx = np.arange(N)
        loss_pi_acc = loss_v_acc = ent_acc = 0.0
        n_batches = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, cfg.batch_size):
                mb = idx[start: start + cfg.batch_size]
                dist, v = self.net.forward(obs[mb])
                logp = dist.log_prob(act[mb])
                entropy = dist.entropy().mean()

                ratio = (logp - logp_old[mb]).exp()
                surr1 = ratio * adv[mb]
                surr2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = F.mse_loss(v, ret[mb])
                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                self.opt.step()

                loss_pi_acc += float(loss_pi.item())
                loss_v_acc += float(loss_v.item())
                ent_acc += float(entropy.item())
                n_batches += 1

        return {
            "loss_pi": loss_pi_acc / max(n_batches, 1),
            "loss_v": loss_v_acc / max(n_batches, 1),
            "entropy": ent_acc / max(n_batches, 1),
        }

    def evaluate(self, episodes, deterministic: bool = True) -> float:
        """Mean episode_total_cost across given eval episodes."""
        costs = []
        for ep in episodes:
            traj = self.collect_trajectory(ep, deterministic=deterministic)
            costs.append(traj["episode_total_cost"])
        return float(np.mean(costs)) if costs else float("nan")
