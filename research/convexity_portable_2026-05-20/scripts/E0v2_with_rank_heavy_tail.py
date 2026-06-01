"""E0 v2 — re-test E0 with red-team's fixes.

Changes from E0 v1:
  1. Heavy-tail features (funding_rate, funding_rate_z_7d, funding_rate_1d_change,
     idio_vol_to_btc_1h, idio_vol_to_btc_1d) → POOLED RANK TRANSFORM on fold-0
     train + z-score. Per linear_model arc convention. (Was: winsor p1/p99 + z.)
  2. Add `bars_since_high` and `autocorr_pctile_7d` to the feature set (univariate IC
     +0.046 and -0.028 respectively per red-team's diagnostic).
  3. Add LOFO aggregate diagnostic: AUC_full and delta after dropping each fold
     individually and after dropping the best-2 folds.

Everything else identical: broad 110-panel, 4h cadence, per-sym trailing 30d p95
event detector, time-OOS only, vol baseline = idio_vol_to_btc_1d (pre-committed).

Gate (unchanged): AUC_full ≥ 0.53 AND (AUC_full − AUC_vol_only) ≥ 0.015 → PASS.
Plus LOFO sensitivity: if drop-best-fold delta < 0.015 → fold-concentrated → CLOSE.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"; OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"

N_FOLDS = 9
WIN_30D = 180
MIN_PRIOR = 30
EMBARGO_DAYS = 1

STANDARD = [
    "return_1d", "atr_pct", "obv_z_1d", "vwap_slope_96",
    "bars_since_high_xs_rank",
    "corr_to_btc_1d",
    "beta_to_btc_change_5d",
    "dom_btc_z_1d", "dom_btc_change_288b",
    "corr_to_btc_change_3d",
    "return_8h", "vol_zscore_4h_over_7d",
    # red-team additions
    "bars_since_high", "autocorr_pctile_7d",
]
HEAVY = [
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
]
FEATS = STANDARD + HEAVY            # 14 + 5 = 19 features
VOL_BASELINE = "idio_vol_to_btc_1d"


def build_events(panel):
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    panel["prior_abs_alpha"] = panel.groupby("symbol")["alpha_beta"].shift(1).abs()
    panel["trail_p95"] = (panel.groupby("symbol")["prior_abs_alpha"]
                              .transform(lambda s: s.rolling(WIN_30D, min_periods=MIN_PRIOR)
                                                    .quantile(0.95).shift(1)))
    panel["event"] = (panel["prior_abs_alpha"] >= panel["trail_p95"]).fillna(False)
    return panel


def fit_preproc(train_df, standard_cols, heavy_cols):
    """Fold-0 train rank-transform for HEAVY + winsor-z for STANDARD."""
    standard_stats = {}
    for c in standard_cols:
        v = train_df[c].dropna()
        if len(v) < 50:
            standard_stats[c] = {"lo": 0.0, "hi": 1.0, "mu": 0.0, "sd": 1.0}; continue
        lo = float(v.quantile(0.01)); hi = float(v.quantile(0.99))
        vc = v.clip(lo, hi)
        standard_stats[c] = {"lo": lo, "hi": hi,
                             "mu": float(vc.mean()),
                             "sd": float(vc.std()) or 1.0}
    heavy_stats = {}
    for c in heavy_cols:
        v = train_df[c].dropna().to_numpy()
        if len(v) < 50:
            heavy_stats[c] = {"vals": np.array([0.0, 1.0]), "mu": 0.0, "sd": 1.0}; continue
        sv = np.sort(v)
        # rank-on-train then z-score the ranks using train rank mean+std
        ranks = np.searchsorted(sv, v, side="left") / max(1, len(sv) - 1)
        heavy_stats[c] = {"vals": sv, "mu": float(ranks.mean()),
                          "sd": float(ranks.std()) or 1.0}
    return standard_stats, heavy_stats


def apply_preproc(df, standard_cols, heavy_cols, sstats, hstats):
    rows = len(df)
    out = np.zeros((rows, len(standard_cols) + len(heavy_cols)), dtype=float)
    for i, c in enumerate(standard_cols):
        s = sstats[c]; v = df[c].to_numpy()
        v = np.clip(v, s["lo"], s["hi"])
        out[:, i] = (v - s["mu"]) / s["sd"]
    for j, c in enumerate(heavy_cols):
        h = hstats[c]; v = df[c].to_numpy()
        # rank against fold-0 train values, then z-score using fold-0 train rank stats
        with np.errstate(invalid="ignore"):
            r = np.searchsorted(h["vals"], v, side="left") / max(1, len(h["vals"]) - 1)
        out[:, len(standard_cols) + j] = (r - h["mu"]) / h["sd"]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def main():
    t0 = time.time()
    print("=== E0 v2 (rank-transform heavy-tail + added features) ===\n", flush=True)
    cols = ["symbol", "open_time", "alpha_beta"] + STANDARD + HEAVY
    cols = list(dict.fromkeys(cols))
    p = pd.read_parquet(PANEL, columns=cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p["open_time"].dt.minute == 0) & (p["open_time"].dt.hour % 4 == 0)]
    p = p.dropna(subset=["alpha_beta"]).reset_index(drop=True)
    p = build_events(p)
    evt = p[p["event"]].copy()
    evt["target_sign"] = np.sign(evt["alpha_beta"])
    evt = evt[evt["target_sign"] != 0].reset_index(drop=True)
    print(f"  event bars: {len(evt):,} | long share {(evt['target_sign']>0).mean():.3f}",
          flush=True)

    times = sorted(evt["open_time"].unique())
    n_times = len(times)
    fold_size = n_times // N_FOLDS

    per_fold = []
    for f in range(N_FOLDS):
        i0 = f * fold_size
        i1 = min((f + 1) * fold_size, n_times - 1) if f < N_FOLDS - 1 else n_times
        oos_start = pd.Timestamp(times[i0])
        oos_end = pd.Timestamp(times[i1 - 1])
        embargo_cut = oos_start - pd.Timedelta(days=EMBARGO_DAYS)
        train = evt[evt["open_time"] < embargo_cut]
        test = evt[(evt["open_time"] >= oos_start) & (evt["open_time"] <= oos_end)]
        if len(train) < 200 or len(test) < 30:
            print(f"  fold {f}: skipped", flush=True); continue

        sstats, hstats = fit_preproc(train, STANDARD, HEAVY)
        Xtr = apply_preproc(train, STANDARD, HEAVY, sstats, hstats)
        Xte = apply_preproc(test, STANDARD, HEAVY, sstats, hstats)
        ytr = train["target_sign"].to_numpy()
        yte = test["target_sign"].to_numpy()
        yte_b = (yte > 0).astype(int)

        m_full = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
        pred_full = m_full.predict(Xte)
        try: auc_full = roc_auc_score(yte_b, pred_full)
        except ValueError: auc_full = np.nan

        vol_idx = STANDARD.index(VOL_BASELINE) if VOL_BASELINE in STANDARD else \
                  (len(STANDARD) + HEAVY.index(VOL_BASELINE))
        Xtr_v = Xtr[:, [vol_idx]]
        Xte_v = Xte[:, [vol_idx]]
        m_vol = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr_v, ytr)
        pred_vol = m_vol.predict(Xte_v)
        try: auc_vol = roc_auc_score(yte_b, pred_vol)
        except ValueError: auc_vol = np.nan

        per_fold.append({
            "fold": f,
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "alpha": float(m_full.alpha_),
            "auc_full": float(auc_full) if auc_full == auc_full else None,
            "auc_vol": float(auc_vol) if auc_vol == auc_vol else None,
            "delta": float(auc_full - auc_vol) if (auc_full == auc_full and auc_vol == auc_vol) else None,
            "frac_long_oos": float((yte > 0).mean()),
        })
        print(f"  fold {f}: n_tr={len(train):>5,} n_te={len(test):>5,} α={m_full.alpha_:>6.2f} "
              f"AUC_full={auc_full:.4f} AUC_vol={auc_vol:.4f} Δ={auc_full-auc_vol:+.4f}",
              flush=True)

    # aggregates
    aucs_full = [r["auc_full"] for r in per_fold if r["auc_full"] is not None]
    aucs_vol = [r["auc_vol"] for r in per_fold if r["auc_vol"] is not None]
    deltas = [r["delta"] for r in per_fold if r["delta"] is not None]
    auc_full_mean = float(np.mean(aucs_full))
    auc_vol_mean = float(np.mean(aucs_vol))
    delta_mean = float(np.mean(deltas))

    # LOFO sensitivity
    lofo = []
    for drop in range(len(per_fold)):
        keep = [r for i, r in enumerate(per_fold) if i != drop]
        ka = [r["auc_full"] for r in keep if r["auc_full"] is not None]
        kv = [r["auc_vol"] for r in keep if r["auc_vol"] is not None]
        kd = [r["delta"] for r in keep if r["delta"] is not None]
        lofo.append({
            "drop_fold": per_fold[drop]["fold"],
            "drop_fold_delta": round(per_fold[drop]["delta"], 4) if per_fold[drop]["delta"] is not None else None,
            "remaining_auc_full_mean": round(float(np.mean(ka)), 4) if ka else None,
            "remaining_delta_mean": round(float(np.mean(kd)), 4) if kd else None,
        })
    # drop best-2 (by delta)
    sorted_by_d = sorted(per_fold, key=lambda r: -(r["delta"] or -np.inf))
    drop_best2 = sorted_by_d[2:]
    db2_full = [r["auc_full"] for r in drop_best2 if r["auc_full"] is not None]
    db2_d = [r["delta"] for r in drop_best2 if r["delta"] is not None]
    drop_best2_summary = {
        "dropped_fold_ids": [sorted_by_d[0]["fold"], sorted_by_d[1]["fold"]],
        "remaining_auc_full_mean": round(float(np.mean(db2_full)), 4) if db2_full else None,
        "remaining_delta_mean": round(float(np.mean(db2_d)), 4) if db2_d else None,
    }

    # gate
    gate_auc = auc_full_mean >= 0.53
    gate_delta = delta_mean >= 0.015
    # robustness: drop-best-fold delta must still be ≥ 0.015
    worst_lofo_delta = min(l["remaining_delta_mean"] for l in lofo if l["remaining_delta_mean"] is not None)
    gate_lofo = worst_lofo_delta >= 0.015

    if not gate_auc:
        verdict = "CLOSE — AUC_full < 0.53"
    elif not gate_delta:
        verdict = "CLOSE — delta < 0.015"
    elif not gate_lofo:
        verdict = "CLOSE — drop-best-fold delta < 0.015 (fold-concentrated)"
    else:
        verdict = "PASS — proceed to designing universe-portability test + strategy plan"

    out = {
        "version": "E0v2",
        "n_features": len(FEATS),
        "feature_set": FEATS,
        "heavy_tail_rank_transformed": HEAVY,
        "n_event_bars": int(len(evt)),
        "aggregate": {
            "auc_full_mean": round(auc_full_mean, 4),
            "auc_vol_mean": round(auc_vol_mean, 4),
            "delta_mean": round(delta_mean, 4),
            "n_folds": len(per_fold),
            "folds_auc>=0.53": sum(1 for a in aucs_full if a >= 0.53),
            "folds_delta>=0.015": sum(1 for d in deltas if d >= 0.015),
        },
        "lofo_drop_one_fold": lofo,
        "drop_best_2_folds": drop_best2_summary,
        "gate_auc": bool(gate_auc),
        "gate_delta": bool(gate_delta),
        "gate_lofo_delta": bool(gate_lofo),
        "verdict": verdict,
        "per_fold": per_fold,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "E0v2_results.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"\n=== AGGREGATE ===", flush=True)
    print(f"  AUC_full mean:        {auc_full_mean:.4f}  (gate ≥0.53: {'PASS' if gate_auc else 'FAIL'})", flush=True)
    print(f"  AUC_vol mean:         {auc_vol_mean:.4f}", flush=True)
    print(f"  delta mean:           {delta_mean:+.4f}  (gate ≥0.015: {'PASS' if gate_delta else 'FAIL'})", flush=True)
    print(f"  worst-LOFO delta:     {worst_lofo_delta:+.4f}  (gate ≥0.015: {'PASS' if gate_lofo else 'FAIL'})", flush=True)
    print(f"  drop-best-2 AUC_full: {drop_best2_summary['remaining_auc_full_mean']:.4f}", flush=True)
    print(f"  drop-best-2 delta:    {drop_best2_summary['remaining_delta_mean']:+.4f}", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"\n[elapsed {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
