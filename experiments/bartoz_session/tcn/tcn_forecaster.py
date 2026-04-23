"""TCN + FiLM + Fourier-seasonal-head forecaster.

Adapted from Bartosz Szabłowski's winning VN2 architecture (see pdf/Bartoz.pdf).

Pipeline:
    Past window X = sales[t-L:t], stock = in_stock[t-L:t]        (2 channels, L=20)
    Scale factor S = max(mean(sales[t-53:t] | in_stock) * 53, 1)
    X_scaled = X / S;  X_norm = (X_scaled - mu) / sigma            (fit on train)
    TCN( X_norm ) -> h_last [hidden]
    FiLM( h_last, emb(store, product, week) ) -> h'
    Decoder( h' ) -> base [H]                                      (H=5)
    SeasonalHead(base, store_product_id, future_week) -> Y_scaled
    Y_hat = Y_scaled * S

Loss (train): Masked Huber on scaled Y (mask = future in_stock)
Fine-tune:    Same loss but on unscaled Y (magnitude-aware calibration)

We train with H=5 instead of the deck's H=3 so a cov-based order-up-to policy
can use arrival weeks r+3..r+5 (horizons 3,4,5). Bartosz's H=3 is paired with
the RL ordering policy that uses only d3; we're skipping the RL stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class TCNConfig:
    context_len: int = 20
    horizon: int = 5          # deck says 3; we extend to 5 for cov-based orders
    scale_len: int = 53
    hidden: int = 64
    tcn_kernel: int = 3
    tcn_layers: int = 4       # dilations 1,2,4,8 -> receptive field > L
    store_embed: int = 6
    product_embed: int = 12
    series_embed: int = 16    # legacy, unused after seasonal-head rewrite
    fourier_k: int = 4
    dropout: float = 0.1
    input_dropout: float = 0.1
    # Architectural toggles (so we can ablate). All True = full Bartosz arch.
    use_softplus_decoder: bool = True
    # Augmentations (train-mode only)
    p_time_aug: float = 0.5
    time_aug_mul_std: float = 0.10
    time_aug_add_std: float = 0.10
    p_week_aug: float = 0.30
    week_aug_max_shift: int = 2
    p_static_aug: float = 0.30
    # Training
    batch_size: int = 512
    lr: float = 1e-4
    fine_tune_lr: float = 3e-5
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    max_epochs: int = 60
    patience: int = 15
    fine_tune_epochs: int = 20
    # Misc
    huber_delta: float = 1.0
    val_frac: float = 0.10    # last fraction of windows (chronological) used as validation


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
def _week_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """ISO week number (1..53), zero-indexed for embedding."""
    return dates.isocalendar().week.values.astype(np.int64) - 1  # 0..52


def _build_id_maps(index: pd.MultiIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map (Store, Product) MultiIndex to contiguous store_id, product_id, series_id.

    Returns three int arrays of length = len(index).
    """
    stores = index.get_level_values(0).values
    products = index.get_level_values(1).values
    unique_stores = np.unique(stores)
    unique_products = np.unique(products)
    store_map = {s: i for i, s in enumerate(unique_stores)}
    product_map = {p: i for i, p in enumerate(unique_products)}
    store_ids = np.array([store_map[s] for s in stores], dtype=np.int64)
    product_ids = np.array([product_map[p] for p in products], dtype=np.int64)
    series_ids = np.arange(len(index), dtype=np.int64)
    return store_ids, product_ids, series_ids


class WindowDataset(Dataset):
    """Sliding windows over a (n_series, T) sales/stock matrix.

    Each sample:
      x_sales [L], x_stock [L]           past window (scaled then standardized)
      scale [scalar]                     S (for rescaling predictions)
      store_id, product_id, series_id    ints
      week_ctx [L], week_fut [H]         week-of-year indices
      y_scaled [H]                       target / S
      y_raw [H]                          target, unscaled (for fine-tune loss)
      mask [H]                           1.0 where in_stock[t+1:t+H+1] is True
    """

    def __init__(
        self,
        sales: np.ndarray,         # [N, T] float
        in_stock: np.ndarray,      # [N, T] bool
        dates: pd.DatetimeIndex,
        store_ids: np.ndarray,
        product_ids: np.ndarray,
        series_ids: np.ndarray,
        cfg: TCNConfig,
        t_starts: np.ndarray,      # [S] valid start indices (t such that t-L >= 0 and t+H <= T)
        norm_mu: float | None = None,
        norm_sigma: float | None = None,
    ):
        assert sales.shape == in_stock.shape
        sales_f = sales.astype(np.float32)
        stock_f = in_stock.astype(np.float32)
        self.cfg = cfg
        self.N = sales_f.shape[0]
        self.ts = t_starts.astype(np.int64)
        self.n_t = len(self.ts)
        self.n_windows = self.N * self.n_t
        L, H, SL = cfg.context_len, cfg.horizon, cfg.scale_len

        # Per-window scale, vectorized via cumulative sums across time.
        # cs_in_sales[:, t] = sum_{s<t} sales[:, s] * stock[:, s]
        # cs_stock[:, t]    = sum_{s<t} stock[:, s]
        T = sales_f.shape[1]
        cs_in_sales = np.zeros((self.N, T + 1), dtype=np.float64)
        cs_stock = np.zeros((self.N, T + 1), dtype=np.float64)
        cs_in_sales[:, 1:] = np.cumsum(sales_f * stock_f, axis=1)
        cs_stock[:, 1:] = np.cumsum(stock_f, axis=1)
        lo = np.maximum(0, self.ts - SL)
        in_sum = cs_in_sales[:, self.ts] - cs_in_sales[:, lo]   # [N, n_t]
        in_cnt = cs_stock[:, self.ts] - cs_stock[:, lo]         # [N, n_t]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(in_cnt > 0, in_sum / np.maximum(in_cnt, 1) * SL, 1.0)
        self.scales = np.maximum(scale, 1.0).astype(np.float32)  # [N, n_t]

        # Materialize per-window slices: x_sales (raw), x_stock, y_raw, mask.
        # Indices for L-step past window: ts[:, None] - L + arange(L) -> [n_t, L]
        ctx_idx = self.ts[:, None] + np.arange(-L, 0)[None, :]   # [n_t, L]
        fut_idx = self.ts[:, None] + np.arange(0, H)[None, :]    # [n_t, H]
        # Materialize as [N, n_t, L/H] using fancy indexing on the time axis.
        self.x_sales_raw = sales_f[:, ctx_idx]      # [N, n_t, L]
        self.x_stock = stock_f[:, ctx_idx]          # [N, n_t, L]
        self.y_raw = sales_f[:, fut_idx]            # [N, n_t, H]
        self.mask = stock_f[:, fut_idx]             # [N, n_t, H]
        # Scaled targets.
        self.y_scaled = (self.y_raw / self.scales[:, :, None]).astype(np.float32)

        # Standardization parameters and final x_norm.
        if norm_mu is not None and norm_sigma is not None:
            self.norm_mu = float(norm_mu)
            self.norm_sigma = float(norm_sigma)
            x_scaled = self.x_sales_raw / self.scales[:, :, None]
            self.x_norm = ((x_scaled - self.norm_mu) / self.norm_sigma).astype(np.float32)
        else:
            self.norm_mu = None
            self.norm_sigma = None
            self.x_norm = None  # filled in via .set_norm()

        # Static covariates broadcast.
        self.store_ids = store_ids.astype(np.int64)
        self.product_ids = product_ids.astype(np.int64)
        self.series_ids = series_ids.astype(np.int64)
        self.n_stores = int(self.store_ids.max()) + 1
        self.n_products = int(self.product_ids.max()) + 1
        # Week-of-year per ctx/fut window (same for every series).
        week = _week_of_year(dates).astype(np.int64)
        self.week_ctx = week[ctx_idx]                # [n_t, L]
        self.week_fut = week[fut_idx]                # [n_t, H]

        # Augmentation flag (set via .set_training()).
        self.training = False

    def set_training(self, flag: bool):
        self.training = bool(flag)

    def __len__(self):
        return self.n_windows

    def compute_scaled_x_stats(self) -> tuple[float, float]:
        """Mean and std of X_scaled = x / scale across all (N, n_t, L) elements."""
        x_scaled = self.x_sales_raw / self.scales[:, :, None]
        mu = float(x_scaled.mean())
        var = float((x_scaled * x_scaled).mean()) - mu * mu
        return mu, math.sqrt(max(var, 1e-8))

    def set_norm(self, mu: float, sigma: float):
        self.norm_mu = float(mu)
        self.norm_sigma = float(sigma)
        x_scaled = self.x_sales_raw / self.scales[:, :, None]
        self.x_norm = ((x_scaled - self.norm_mu) / self.norm_sigma).astype(np.float32)

    def __getitem__(self, idx):
        i = idx // self.n_t
        t_pos = idx % self.n_t
        cfg = self.cfg

        x_sales = self.x_norm[i, t_pos]               # [L] float32
        x_stock = self.x_stock[i, t_pos]              # [L] float32
        store_id = int(self.store_ids[i])
        product_id = int(self.product_ids[i])
        week_ctx = self.week_ctx[t_pos]               # [L] int64
        week_fut = self.week_fut[t_pos]               # [H] int64

        if self.training:
            # TimeAugmenter: noise on (already-standardized) sales channel.
            if np.random.rand() < cfg.p_time_aug:
                eps_m = np.random.normal(0.0, cfg.time_aug_mul_std)
                eps_a = np.random.normal(0.0, cfg.time_aug_add_std, size=x_sales.shape).astype(np.float32)
                x_sales = (x_sales * (1.0 + eps_m) + eps_a).astype(np.float32)
            # WeekAugmenter: shift calendar feature by small int.
            if np.random.rand() < cfg.p_week_aug and cfg.week_aug_max_shift > 0:
                shift = np.random.randint(-cfg.week_aug_max_shift, cfg.week_aug_max_shift + 1)
                week_ctx = (week_ctx + shift) % 53
                week_fut = (week_fut + shift) % 53
            # StaticCovAugmenter: with prob, swap one of (store_id, product_id) for a random other.
            if np.random.rand() < cfg.p_static_aug:
                if np.random.rand() < 0.5 and self.n_stores > 1:
                    store_id = int(np.random.randint(0, self.n_stores))
                elif self.n_products > 1:
                    product_id = int(np.random.randint(0, self.n_products))

        return {
            "x_sales": torch.from_numpy(np.ascontiguousarray(x_sales)),
            "x_stock": torch.from_numpy(np.ascontiguousarray(x_stock)),
            "scale": torch.tensor(self.scales[i, t_pos], dtype=torch.float32),
            "store_id": torch.tensor(store_id, dtype=torch.long),
            "product_id": torch.tensor(product_id, dtype=torch.long),
            "series_id": torch.tensor(self.series_ids[i], dtype=torch.long),
            "week_ctx": torch.from_numpy(np.ascontiguousarray(week_ctx)),
            "week_fut": torch.from_numpy(np.ascontiguousarray(week_fut)),
            "y_scaled": torch.from_numpy(self.y_scaled[i, t_pos]),
            "y_raw": torch.from_numpy(self.y_raw[i, t_pos].astype(np.float32)),
            "mask": torch.from_numpy(self.mask[i, t_pos]),
        }


# --------------------------------------------------------------------------- #
# Model components                                                            #
# --------------------------------------------------------------------------- #
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=self.pad, dilation=dilation)

    def forward(self, x):
        o = self.conv(x)
        return o[..., :-self.pad] if self.pad else o


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel, dilation)
        self.norm1 = nn.GroupNorm(8, channels) if channels >= 8 else nn.Identity()
        self.norm2 = nn.GroupNorm(8, channels) if channels >= 8 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = F.gelu(self.norm2(self.conv2(h)))
        h = self.drop(h)
        return x + h


class TCN(nn.Module):
    def __init__(self, in_ch, hidden, kernel, n_layers, dropout):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock(hidden, kernel, 2 ** i, dropout) for i in range(n_layers)]
        )

    def forward(self, x_btc: torch.Tensor) -> torch.Tensor:
        # x_btc: [B, L, C] -> we want [B, C, L]
        h = x_btc.transpose(1, 2)
        h = self.proj(h)
        for b in self.blocks:
            h = b(h)
        return h[:, :, -1]  # [B, hidden]


class FiLM(nn.Module):
    """Bartosz-style FiLM: gamma = (1 + tanh(MLP)), beta = MLP.

    cond -> Linear -> SiLU -> Dropout -> Linear -> [gamma_raw, beta]
    out  = (1 + tanh(gamma_raw)) * h + beta
    """

    def __init__(self, cond_dim, hidden, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * hidden),
        )
        # Zero-init final layer => gamma=beta=0 at start => identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.hidden = hidden

    def forward(self, h, cond):
        gb = self.net(cond)
        gamma_raw, beta = gb[:, :self.hidden], gb[:, self.hidden:]
        gamma = torch.tanh(gamma_raw)         # bounded [-1, 1]
        return (1.0 + gamma) * h + beta


class SeasonalHead(nn.Module):
    """Bartosz-style multiplicative Fourier seasonal head.

    cond -> MLP (Linear, SiLU, Dropout, Linear) -> A[K], B[K], gate_logit[1]
    A_h = tanh(A);  B_h = tanh(B);  gate = sigmoid(gate_logit)
    sum_k = sum_k(A_h * sin(2 pi k w / 52) + B_h * cos(...))    [B, H]
    factor = exp(gate * tanh(sum_k))                              [B, H]
    out = base * factor

    Output is bounded to roughly [base * exp(-1), base * exp(+1)] = [0.37 base, 2.72 base].
    """

    def __init__(self, cond_dim, K, dropout: float = 0.1):
        super().__init__()
        self.K = K
        hidden = max(cond_dim, 32)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * K + 1),       # A | B | gate_logit
        )
        # Zero-init => seasonal factor = exp(0) = 1 (identity) at training start.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, base, cond, week_fut):
        # base: [B, H]  cond: [B, cond_dim]  week_fut: [B, H] in 0..52
        out = self.mlp(cond)                                        # [B, 2K+1]
        A = torch.tanh(out[:, :self.K])                             # [B, K]
        B = torch.tanh(out[:, self.K:2 * self.K])                   # [B, K]
        gate = torch.sigmoid(out[:, -1:])                           # [B, 1]

        w = (week_fut.float() + 1.0).unsqueeze(-1)                  # [B, H, 1] (1..53)
        k = torch.arange(1, self.K + 1, device=w.device, dtype=w.dtype)
        theta = 2 * math.pi * w * k / 52.0                          # [B, H, K]
        sum_k = (
            A.unsqueeze(1) * torch.sin(theta) + B.unsqueeze(1) * torch.cos(theta)
        ).sum(-1)                                                   # [B, H]
        factor = torch.exp(gate * torch.tanh(sum_k))                # [B, H]
        return base * factor


class TCNForecaster(nn.Module):
    """TCN -> FiLM -> MLP+Softplus -> * SeasonalHead.

    n_series kept for API back-compat but unused (seasonal head uses cond MLP).
    """

    def __init__(self, cfg: TCNConfig, n_stores: int, n_products: int,
                 n_series: int | None = None, n_weeks: int = 53):
        super().__init__()
        self.cfg = cfg
        self.tcn = TCN(in_ch=2, hidden=cfg.hidden, kernel=cfg.tcn_kernel,
                       n_layers=cfg.tcn_layers, dropout=cfg.dropout)
        self.store_emb = nn.Embedding(n_stores, cfg.store_embed)
        self.product_emb = nn.Embedding(n_products, cfg.product_embed)
        cond_dim = cfg.store_embed + cfg.product_embed
        self.film = FiLM(cond_dim, cfg.hidden, dropout=cfg.dropout)
        decoder_layers: list[nn.Module] = [
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.horizon),
        ]
        if cfg.use_softplus_decoder:
            decoder_layers.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder_layers)
        self.seasonal = SeasonalHead(cond_dim, cfg.fourier_k, dropout=cfg.dropout)
        self.input_dropout = nn.Dropout(cfg.input_dropout)

    def forward(self, batch):
        x_s = batch["x_sales"].unsqueeze(-1)    # [B, L, 1]
        x_k = batch["x_stock"].unsqueeze(-1)    # [B, L, 1]
        x = torch.cat([x_s, x_k], dim=-1)       # [B, L, 2]
        x = self.input_dropout(x)
        h = self.tcn(x)                         # [B, hidden]

        # Conditioning = concatenated [e_store, e_prod]; week goes only to the seasonal head.
        cond = torch.cat(
            [
                self.store_emb(batch["store_id"]),
                self.product_emb(batch["product_id"]),
            ],
            dim=-1,
        )
        h = self.film(h, cond)                  # [B, hidden]
        base = self.decoder(h)                  # [B, H], non-negative via softplus
        out = self.seasonal(base, cond, batch["week_fut"])
        return out                              # scaled space


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #
def masked_huber(pred, target, mask, delta=1.0):
    """Mean over valid (mask>0) elements."""
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    loss = 0.5 * quad * quad + delta * lin
    w = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / w


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def _prepare_arrays(sales_hist: pd.DataFrame, in_stock: pd.DataFrame | None,
                    cfg: TCNConfig):
    """Align in_stock to sales_hist columns; missing columns -> True."""
    sales_hist = sales_hist.sort_index(axis=1)
    if in_stock is None:
        in_stock_a = pd.DataFrame(True, index=sales_hist.index, columns=sales_hist.columns)
    else:
        in_stock_a = in_stock.reindex(columns=sales_hist.columns).reindex(sales_hist.index)
        in_stock_a = in_stock_a.fillna(True).astype(bool)
    sales_arr = sales_hist.fillna(0.0).values.astype(np.float32)
    stock_arr = in_stock_a.values.astype(bool)
    dates = pd.DatetimeIndex(sales_hist.columns)
    return sales_arr, stock_arr, dates


def train_tcn(
    sales_hist: pd.DataFrame,
    in_stock: pd.DataFrame | None,
    cfg: TCNConfig | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train the TCN forecaster on sales_hist. Returns dict with model + id maps."""
    cfg = cfg or TCNConfig()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sales_arr, stock_arr, dates = _prepare_arrays(sales_hist, in_stock, cfg)
    N, T = sales_arr.shape
    L, H, SL = cfg.context_len, cfg.horizon, cfg.scale_len
    min_t = max(L, 1)  # need L past, also reasonable SL coverage
    max_t = T - H
    if max_t <= min_t:
        raise ValueError(f"Not enough history: T={T} requires T >= L+H = {L+H}")

    all_ts = np.arange(min_t, max_t + 1)
    # Chronological train/val split of windows.
    n_val = max(1, int(len(all_ts) * cfg.val_frac))
    train_ts = all_ts[:-n_val]
    val_ts = all_ts[-n_val:]

    store_ids, product_ids, series_ids = _build_id_maps(sales_hist.index)
    n_stores = int(store_ids.max()) + 1
    n_products = int(product_ids.max()) + 1
    n_series = int(series_ids.max()) + 1

    # Build train/val datasets (without norm), compute stats on train, apply to both.
    train_ds = WindowDataset(sales_arr, stock_arr, dates, store_ids, product_ids, series_ids,
                             cfg, train_ts)
    val_ds = WindowDataset(sales_arr, stock_arr, dates, store_ids, product_ids, series_ids,
                           cfg, val_ts)
    mu, sigma = train_ds.compute_scaled_x_stats()
    train_ds.set_norm(mu, sigma)
    val_ds.set_norm(mu, sigma)
    train_ds.set_training(True)    # enable augmentations on train only
    val_ds.set_training(False)
    if verbose:
        print(f"[TCN] N={N} T={T}  train_windows={N*len(train_ts):,}  val_windows={N*len(val_ts):,}")
        print(f"[TCN] x_scaled stats: mu={mu:.4f} sigma={sigma:.4f}")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device == "cuda"))

    model = TCNForecaster(cfg, n_stores, n_products, n_series).to(device)

    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "Norm" in n:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=cfg.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, cfg.max_epochs - cfg.warmup_epochs), eta_min=1e-6
            ),
        ],
        milestones=[cfg.warmup_epochs],
    )
    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))

    def _epoch(loader, training: bool, unscaled: bool = False):
        model.train(training)
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda"), dtype=torch.float16):
                pred = model(batch)  # scaled space
                if unscaled:
                    pred_u = pred * batch["scale"].unsqueeze(-1)
                    loss = masked_huber(pred_u, batch["y_raw"], batch["mask"], cfg.huber_delta)
                else:
                    loss = masked_huber(pred, batch["y_scaled"], batch["mask"], cfg.huber_delta)
            if training:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            total_loss += float(loss.detach()) * batch["y_scaled"].shape[0]
            n += batch["y_scaled"].shape[0]
        return total_loss / max(n, 1)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(cfg.max_epochs):
        tr = _epoch(train_loader, training=True)
        scheduler.step()
        with torch.no_grad():
            va = _epoch(val_loader, training=False)
        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if verbose and (epoch < 3 or epoch % 5 == 0 or bad_epochs >= cfg.patience):
            print(f"[TCN] ep {epoch:3d}  lr={opt.param_groups[0]['lr']:.1e}  "
                  f"train={tr:.4f}  val={va:.4f}  best={best_val:.4f}  bad={bad_epochs}")
        if bad_epochs >= cfg.patience:
            if verbose:
                print(f"[TCN] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------- Fine-tune: scale-aware calibration on unscaled values. -------- #
    if cfg.fine_tune_epochs > 0:
        for g in opt.param_groups:
            g["lr"] = cfg.fine_tune_lr
        best_ft_val = float("inf")
        best_ft_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
        ft_patience = max(3, cfg.patience // 2)
        for epoch in range(cfg.fine_tune_epochs):
            tr = _epoch(train_loader, training=True, unscaled=True)
            with torch.no_grad():
                va = _epoch(val_loader, training=False, unscaled=True)
            if va < best_ft_val - 1e-4:
                best_ft_val = va
                best_ft_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if verbose and (epoch < 2 or epoch % 5 == 0 or bad_epochs >= ft_patience):
                print(f"[TCN-ft] ep {epoch:3d}  train={tr:.4f}  val={va:.4f}  best={best_ft_val:.4f}  bad={bad_epochs}")
            if bad_epochs >= ft_patience:
                if verbose:
                    print(f"[TCN-ft] early stop at epoch {epoch}")
                break
        model.load_state_dict(best_ft_state)

    return {
        "model": model,
        "cfg": cfg,
        "mu": mu,
        "sigma": sigma,
        "store_ids": store_ids,
        "product_ids": product_ids,
        "series_ids": series_ids,
        "index": sales_hist.index,
        "device": device,
        "best_val_scaled": best_val,
    }


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_tcn(trained: dict, sales_hist: pd.DataFrame,
                in_stock: pd.DataFrame | None) -> pd.DataFrame:
    """Predict the next H weeks for every series, using the last context window
    of sales_hist. Returns DataFrame [n_series, H] with future-date columns.

    The input sales_hist/in_stock may be longer than the training data (as the
    CV simulation appends actual weeks); we use the tail.
    """
    model: TCNForecaster = trained["model"]
    cfg: TCNConfig = trained["cfg"]
    mu, sigma = trained["mu"], trained["sigma"]
    device = trained["device"]
    store_ids = trained["store_ids"]
    product_ids = trained["product_ids"]
    series_ids = trained["series_ids"]
    L, H, SL = cfg.context_len, cfg.horizon, cfg.scale_len

    sales_hist = sales_hist.sort_index(axis=1).reindex(trained["index"])
    sales_arr, stock_arr, dates = _prepare_arrays(sales_hist, in_stock, cfg)

    N, T = sales_arr.shape
    t = T  # predict from end of history
    if t < L:
        raise ValueError(f"Need at least L={L} history; got T={T}")

    # Build per-series tensors (vectorized).
    ctx = sales_arr[:, t - L:t]                       # [N, L]
    stk = stock_arr[:, t - L:t].astype(np.float32)    # [N, L]
    lo = max(0, t - SL)
    s_ctx = sales_arr[:, lo:t]                        # [N, <=SL]
    is_ctx = stock_arr[:, lo:t].astype(np.float32)    # [N, <=SL]
    in_sum = (s_ctx * is_ctx).sum(axis=1)
    in_cnt = is_ctx.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(in_cnt > 0, in_sum / np.maximum(in_cnt, 1) * SL, 1.0)
    scales = np.maximum(scale, 1.0).astype(np.float32)              # [N]
    x_sales = ((ctx / scales[:, None] - mu) / sigma).astype(np.float32)
    x_stock = stk

    # Week-of-year for context and future.
    week_ctx = _week_of_year(dates[t - L:t])
    last_date = dates[t - 1]
    fut_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=H, freq="W-MON")
    week_fut = _week_of_year(pd.DatetimeIndex(fut_dates))

    # Batched inference.
    model.eval()
    preds = np.zeros((N, H), dtype=np.float32)
    B = 1024
    for i0 in range(0, N, B):
        i1 = min(N, i0 + B)
        batch = {
            "x_sales": torch.from_numpy(x_sales[i0:i1]).to(device),
            "x_stock": torch.from_numpy(x_stock[i0:i1]).to(device),
            "scale": torch.from_numpy(scales[i0:i1]).to(device),
            "store_id": torch.from_numpy(store_ids[i0:i1]).to(device),
            "product_id": torch.from_numpy(product_ids[i0:i1]).to(device),
            "series_id": torch.from_numpy(series_ids[i0:i1]).to(device),
            "week_ctx": torch.from_numpy(np.broadcast_to(week_ctx, (i1 - i0, L)).copy()).to(device),
            "week_fut": torch.from_numpy(np.broadcast_to(week_fut, (i1 - i0, H)).copy()).to(device),
        }
        out = model(batch)  # scaled
        out = out * batch["scale"].unsqueeze(-1)
        preds[i0:i1] = out.detach().cpu().numpy()

    preds = np.clip(preds, 0.0, None)
    return pd.DataFrame(preds, index=sales_hist.index, columns=fut_dates)
