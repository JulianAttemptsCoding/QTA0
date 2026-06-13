"""Walk-forward training driver (sec10.3).

Splits the panel by date into train / val / test(OOS) with purge+embargo, standardizes the 18
channels on TRAIN stats only (fit-scope, sec0.3.2), trains a quantile model (f3 or f4) with
AdamW + cosine schedule + early stopping on val pinball, records per-epoch train/val loss, and
writes OOS predictions. Multiple seeds are quantile-averaged (sec10.4).

Artifacts -> reports/runs/<run_id>/: loss_history.csv, oos_predictions.parquet, config.json.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from ..common.config import REPORTS_DIR, load_config
from ..common.registry import log_trial
from .panel_dataset import build_panel
from .experts.tcn_ci import F3Model
from .experts.hybrid_xs import F4Model

TAUS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
MEDIAN_IDX = 3


def pinball_torch(preds, y, taus):
    # preds (N,Q), y (N,), taus (Q,)
    e = y.unsqueeze(1) - preds
    return torch.maximum(taus * e, (taus - 1) * e).sum(dim=1).mean()


def pairwise_rank_loss(q50, y):
    """Per-date pairwise logistic ranking on the median head (revision_plan §4.2 Option A).

    For every ordered pair (i, j) with y_i > y_j, penalize softplus(-(q50_i - q50_j)) so the model
    is rewarded for ORDERING names, not just matching the marginal. One date = one batch, so this
    is purely cross-sectional. Returns 0 if no orderable pairs in the batch.
    """
    if q50.shape[0] < 2:
        return q50.new_zeros(())
    dq = q50.unsqueeze(1) - q50.unsqueeze(0)          # (N,N) q50_i - q50_j
    dy = y.unsqueeze(1) - y.unsqueeze(0)              # (N,N) y_i - y_j
    mask = dy > 0
    if not bool(mask.any()):
        return q50.new_zeros(())
    return torch.nn.functional.softplus(-dq[mask]).mean()


def _split(batches, train_end, val_end, purge=2, embargo=5):
    """Date split with a purge+embargo gap between regimes (sec10.3)."""
    tr = [b for b in batches if b.date <= train_end]
    va = [b for b in batches if train_end < b.date <= val_end]
    te = [b for b in batches if b.date > val_end]
    # purge/embargo: drop the first `purge+embargo` val/test batches adjacent to the boundary
    gap = purge + embargo
    return tr, va[gap:], te[gap:]


def _channel_stats(train_batches):
    xs = np.concatenate([b.x_seq.reshape(-1, b.x_seq.shape[1]) for b in train_batches], axis=0)
    # x_seq is (N,18,L) -> reshape to (N*L?,18)? actually channels are axis 1
    allx = np.concatenate([b.x_seq.transpose(0, 2, 1).reshape(-1, 18) for b in train_batches], axis=0)
    mu = np.nanmean(allx, axis=0)
    sd = np.nanstd(allx, axis=0) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def _apply_stats(b, mu, sd, device, demean=False):
    x = (b.x_seq.transpose(0, 2, 1) - mu) / sd       # standardize per channel
    x = x.transpose(0, 2, 1)                          # back to (N,18,L)
    y = b.y.astype(np.float32)
    if demean:                                        # §4.3 cross-sectional demean (relative target)
        y = y - np.nanmean(y)
    return (torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(b.u, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def _eval_loss(model, batches, mu, sd, taus_t, device, demean=False):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in batches:
            x, u, y = _apply_stats(b, mu, sd, device, demean)
            loss = pinball_torch(model(x, u), y, taus_t)
            tot += float(loss) * len(b.y); n += len(b.y)
    return tot / max(n, 1)


def train_one_seed(model_kind, batches_tr, batches_va, batches_te, mu, sd,
                   epochs=25, lr=1e-3, seed=0, device="cpu", patience=5,
                   lambda_rank=0.0, demean=False, weight_decay=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    model = (F4Model() if model_kind == "f4" else F3Model()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    taus_t = torch.tensor(TAUS, dtype=torch.float32, device=device)

    # `patience`: epochs of no val improvement tolerated before stopping. Set high to let the
    # train/val curves run PAST the divergence point (train pinball ↓ while val pinball ↑) so the
    # overfit onset is visible in loss_history; predictions still use the best-val checkpoint.
    history, best_val, best_state, bad = [], float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(len(batches_tr))
        tot, n = 0.0, 0
        for i in order:
            b = batches_tr[i]
            x, u, y = _apply_stats(b, mu, sd, device, demean)
            opt.zero_grad()
            q = model(x, u)
            loss = pinball_torch(q, y, taus_t)
            if lambda_rank > 0.0:
                loss = loss + lambda_rank * pairwise_rank_loss(q[:, MEDIAN_IDX], y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss) * len(b.y); n += len(b.y)
        sched.step()
        tr_loss = tot / max(n, 1)
        # val pinball is the EARLY-STOP signal and the S1 metric (pure pinball, no rank term)
        va_loss = _eval_loss(model, batches_va, mu, sd, taus_t, device, demean)
        history.append({"epoch": ep, "seed": seed, "train_pinball": tr_loss, "val_pinball": va_loss})
        if va_loss < best_val - 1e-6:
            best_val, best_state, bad = va_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # OOS predictions
    preds = []
    model.eval()
    with torch.no_grad():
        for b in batches_te:
            x, u, _ = _apply_stats(b, mu, sd, device)
            q = model(x, u).cpu().numpy()
            for j, ent in enumerate(b.entities):
                preds.append({"date": b.date, "entity": ent, "y_true": b.y[j],
                              **{f"q{int(TAUS[k]*100):02d}": q[j, k] for k in range(len(TAUS))}})
    return pd.DataFrame(history), pd.DataFrame(preds), best_val


def train(cfg: dict | None = None, model_kind: str = "f4", epochs: int = 25,
          seeds: int = 1, train_end="2020-12-31", val_end="2021-12-31",
          run_id: str | None = None, patience: int = 5,
          lambda_rank: float = 0.0, demean: bool = False,
          lr: float = 1e-3, weight_decay: float = 1e-4) -> dict:
    cfg = cfg or load_config()
    from ..common.config import CURATED
    feats = pd.read_parquet(CURATED / "features.parquet")
    prices = pd.read_parquet(CURATED / "prices_daily.parquet")
    labels = pd.read_parquet(CURATED / "labels.parquet")

    batches = build_panel(feats, prices, labels)
    train_end = dt.date.fromisoformat(train_end); val_end = dt.date.fromisoformat(val_end)
    tr, va, te = _split(batches, train_end, val_end)
    if not tr or not va or not te:
        raise RuntimeError(f"empty split: train={len(tr)} val={len(va)} test={len(te)}")
    mu, sd = _channel_stats(tr)

    run_id = run_id or f"{model_kind}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = REPORTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_hist, pred_frames, val_losses = [], [], []
    for s in range(seeds):
        h, p, bv = train_one_seed(model_kind, tr, va, te, mu, sd, epochs=epochs,
                                  seed=1338 + s, patience=patience,
                                  lambda_rank=lambda_rank, demean=demean,
                                  lr=lr, weight_decay=weight_decay)
        all_hist.append(h); pred_frames.append(p.assign(seed=s)); val_losses.append(bv)
        log_trial("hpo", f"{model_kind} seed {s} train", cfg_hash(cfg), study_id=run_id)

    hist = pd.concat(all_hist, ignore_index=True)
    # quantile-average predictions across seeds
    qcols = [f"q{int(t*100):02d}" for t in TAUS]
    preds = (pd.concat(pred_frames, ignore_index=True)
             .groupby(["date", "entity"], as_index=False)
             .agg({**{c: "mean" for c in qcols}, "y_true": "first"}))

    # --- S1 sanity gate (§4.1): model val pinball vs unconditional train-quantile baseline ---
    from .baselines import baseline_pinball, s1_gate
    y_tr = np.concatenate([(b.y - np.nanmean(b.y)) if demean else b.y for b in tr])
    y_va = np.concatenate([(b.y - np.nanmean(b.y)) if demean else b.y for b in va])
    base_pin = baseline_pinball(y_tr, y_va)
    s1 = s1_gate(float(np.mean(val_losses)), base_pin)
    # q50 cross-sectional dispersion diagnostic (§4.2): median over OOS dates of per-date std(q50)
    q50_disp = float(preds.groupby("date")["q50"].std().median())

    hist.to_csv(run_dir / "loss_history.csv", index=False)
    preds.to_parquet(run_dir / "oos_predictions.parquet", index=False)
    (run_dir / "config.json").write_text(json.dumps(
        {"model": model_kind, "epochs": epochs, "seeds": seeds,
         "train_end": str(train_end), "val_end": str(val_end),
         "lambda_rank": lambda_rank, "demean": demean,
         "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "best_val_pinball": float(np.mean(val_losses)),
         "baseline_val_pinball": base_pin, "s1": s1, "q50_dispersion": q50_disp},
        indent=2), encoding="utf-8")
    print(f"[train] {model_kind} run={run_id} val_pinball={np.mean(val_losses):.5f} "
          f"baseline={base_pin:.5f} S1_margin={s1['margin']:.4f} pass={s1['pass']} "
          f"q50_disp={q50_disp:.4f} batches={len(tr)}/{len(va)}/{len(te)}")
    return {"run_id": run_id, "run_dir": str(run_dir), "history": hist,
            "predictions": preds, "val_pinball": float(np.mean(val_losses)),
            "s1": s1, "baseline_val_pinball": base_pin, "q50_dispersion": q50_disp}


def cfg_hash(cfg):
    from ..common.hashing import config_hash
    return config_hash(cfg)
