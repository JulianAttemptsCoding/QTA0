"""CPCV training driver + combined-OOS aggregation (revision_plan §3A / A3).

For each CPCV fold: select train/test batches by the date-axis position masks (purge+embargo already
applied by cpcv_paths), carve an embargoed chronological val slice from the fold's train for early
stopping, fit channel-stats on the kept-train only, train `seeds` ranking-augmented models, quantile-
average their test predictions, and persist per-fold artifacts. A full run aggregates every fold's
OOS predictions into one combined out-of-sample series spanning the whole window (each session OOS
in C(n_groups-1, k_test-1) folds) and writes the path-Sharpe distribution.

Sharding: pass fold_start/fold_end to compute a subset of folds (one Vertex job per shard); run the
aggregation step locally once all shards' fold_*_oos.parquet are present.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import CURATED, REPORTS_DIR, load_config
from ..common.registry import log_trial
from ..validation.cpcv import cpcv_paths, combined_oos_coverage
from .baselines import baseline_pinball, s1_gate
from .panel_dataset import build_panel
from .train import _channel_stats, train_one_seed, TAUS, cfg_hash

QCOLS = [f"q{int(t * 100):02d}" for t in TAUS]


def _load_batches():
    feats = pd.read_parquet(CURATED / "features.parquet")
    prices = pd.read_parquet(CURATED / "prices_daily.parquet")
    labels = pd.read_parquet(CURATED / "labels.parquet")
    batches = build_panel(feats, prices, labels)
    batches.sort(key=lambda b: b.date)
    return batches, prices


def _fold_path_sharpe(preds_fold: pd.DataFrame, roo, sprd, k: int = 35) -> dict:
    """Quick dollar-neutral L/S net Sharpe on a fold's OOS predictions (one CPCV path)."""
    from ..backtest.run import backtest_long_short
    bt = backtest_long_short(preds_fold, prices=None, k=k, roo=roo, sprd=sprd)
    if bt.empty:
        return {"sharpe": float("nan"), "n_days": 0, "mean_ret": float("nan")}
    r = bt["ret_net"].to_numpy()
    sd = r.std(ddof=1)
    return {"sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
            "n_days": int(len(r)), "mean_ret": float(r.mean())}


def run_cpcv(model_kind: str = "f4", n_groups: int = 8, k_test: int = 2,
             purge: int = 2, embargo: int = 5, epochs: int = 20, seeds: int = 2,
             patience: int = 6, lambda_rank: float = 0.3, demean: bool = True,
             val_frac: float = 0.15, fold_start: int = 0, fold_end: int | None = None,
             run_id: str | None = None, out_dir: str | Path | None = None,
             k_book: int = 35) -> dict:
    cfg = load_config()
    run_id = run_id or f"{model_kind}_cpcv_{dt.datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(out_dir) if out_dir else (REPORTS_DIR / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    batches, prices = _load_batches()
    dates = sorted({b.date for b in batches})
    pos = {d: i for i, d in enumerate(dates)}
    by_pos: dict[int, list] = {}
    for b in batches:
        by_pos.setdefault(pos[b.date], []).append(b)

    plan = cpcv_paths(len(dates), n_groups=n_groups, k_test=k_test, purge=purge, embargo=embargo)
    cov = combined_oos_coverage(plan)
    print(f"[cpcv] dates={len(dates)} groups={n_groups} k_test={k_test} folds={plan.n_folds} "
          f"oos/group={plan.oos_count_per_group} coverage[min,max]=[{cov.min()},{cov.max()}]")

    # precompute realized returns + spreads once (for per-fold path Sharpe)
    from ..backtest.run import _open_to_open, _spread
    roo = _open_to_open(prices)
    sprd = _spread(None)

    fold_end = plan.n_folds if fold_end is None else min(fold_end, plan.n_folds)
    paths = []
    for f in plan.folds[fold_start:fold_end]:
        train_batches = [bb for p in f.train_pos for bb in by_pos.get(int(p), [])]
        test_batches = [bb for p in f.test_pos for bb in by_pos.get(int(p), [])]
        train_batches.sort(key=lambda b: b.date)
        if not train_batches or not test_batches:
            print(f"[cpcv] fold {f.fold_id} empty; skip"); continue

        n_val = max(int(len(train_batches) * val_frac), 1)
        val_batches = train_batches[-n_val:]
        kept_train = train_batches[:max(len(train_batches) - n_val - embargo, 1)]
        mu, sd = _channel_stats(kept_train)

        pred_frames, val_losses = [], []
        for s in range(seeds):
            _, p, bv = train_one_seed(model_kind, kept_train, val_batches, test_batches, mu, sd,
                                      epochs=epochs, seed=1338 + s, patience=patience,
                                      lambda_rank=lambda_rank, demean=demean)
            pred_frames.append(p); val_losses.append(bv)
            log_trial("hpo", f"{model_kind} cpcv fold{f.fold_id} seed{s}",
                      cfg_hash(cfg), study_id=run_id)

        preds = (pd.concat(pred_frames, ignore_index=True)
                 .groupby(["date", "entity"], as_index=False)
                 .agg({**{c: "mean" for c in QCOLS}, "y_true": "first"}))
        preds["fold"] = f.fold_id

        # fold S1 (val) + q50 dispersion
        y_tr = np.concatenate([(b.y - np.nanmean(b.y)) if demean else b.y for b in kept_train])
        y_va = np.concatenate([(b.y - np.nanmean(b.y)) if demean else b.y for b in val_batches])
        s1 = s1_gate(float(np.mean(val_losses)), baseline_pinball(y_tr, y_va))
        q50_disp = float(preds.groupby("date")["q50"].std().median())
        path = _fold_path_sharpe(preds, roo, sprd, k=k_book)

        preds.to_parquet(run_dir / f"fold_{f.fold_id:02d}_oos.parquet", index=False)
        meta = {"fold_id": f.fold_id, "test_groups": list(f.test_groups),
                "n_train": len(kept_train), "n_val": len(val_batches), "n_test": len(test_batches),
                "val_pinball": float(np.mean(val_losses)), "s1": s1, "q50_dispersion": q50_disp,
                "path_sharpe": path["sharpe"], "path_n_days": path["n_days"],
                "path_mean_ret": path["mean_ret"]}
        (run_dir / f"fold_{f.fold_id:02d}_meta.json").write_text(json.dumps(meta, indent=2),
                                                                 encoding="utf-8")
        paths.append(meta)
        print(f"[cpcv] fold {f.fold_id:2d} groups={f.test_groups} "
              f"val_pin={meta['val_pinball']:.5f} S1={s1['margin']:+.4f}({s1['pass']}) "
              f"q50disp={q50_disp:.4f} pathSharpe={path['sharpe']:.3f} ndays={path['n_days']}")

    (run_dir / "config.json").write_text(json.dumps(
        {"model": model_kind, "n_groups": n_groups, "k_test": k_test, "purge": purge,
         "embargo": embargo, "epochs": epochs, "seeds": seeds, "patience": patience,
         "lambda_rank": lambda_rank, "demean": demean, "val_frac": val_frac,
         "n_folds": plan.n_folds, "fold_start": fold_start, "fold_end": fold_end,
         "n_dates": len(dates), "coverage_min": int(cov.min()), "coverage_max": int(cov.max())},
        indent=2), encoding="utf-8")
    return {"run_id": run_id, "run_dir": str(run_dir), "paths": paths, "plan": plan}


def aggregate_combined(run_dir: str | Path) -> dict:
    """Combine all fold_*_oos.parquet into one OOS series (mean q across folds per date,entity)
    and gather the per-fold path Sharpes into paths.csv."""
    run_dir = Path(run_dir)
    fold_files = sorted(run_dir.glob("fold_*_oos.parquet"))
    if not fold_files:
        raise FileNotFoundError(f"no fold_*_oos.parquet in {run_dir}")
    allp = pd.concat([pd.read_parquet(f) for f in fold_files], ignore_index=True)
    agg = {**{c: "mean" for c in QCOLS}, "y_true": "first", "fold": "count"}
    combined = allp.groupby(["date", "entity"], as_index=False).agg(agg)
    combined = combined.rename(columns={"fold": "n_folds"})
    combined.to_parquet(run_dir / "oos_predictions.parquet", index=False)

    metas = [json.loads(p.read_text()) for p in sorted(run_dir.glob("fold_*_meta.json"))]
    paths_df = pd.DataFrame(metas)
    paths_df.to_csv(run_dir / "paths.csv", index=False)
    return {"n_obs": len(combined), "n_folds_present": len(fold_files),
            "date_min": str(combined["date"].min()), "date_max": str(combined["date"].max()),
            "path_sharpe_mean": float(paths_df["path_sharpe"].mean()),
            "path_sharpe_std": float(paths_df["path_sharpe"].std())}
