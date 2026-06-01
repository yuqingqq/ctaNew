"""E0 — HL-agnostic event-conditional sign-predictability test.

Single falsification: does sign of next 4h alpha_vs_btc_realized
have any portable predictability conditional on a magnitude trigger,
on the broad 110-panel? Time-OOS only, BTC-frame features only, no sym_id.

Method (locked):
  - Panel: outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet
    (110 syms, alpha_beta = BTC-residual 4h forward target, full-PIT features).
  - Subsample to 4h cadence: keep only bars at open_time {00,04,08,12,16,20}:00.
    Eliminates 48× overlap in 5m bars; gives clean non-overlapping 4h cycles.
  - Per-symbol event: |alpha_beta[t-1]| ≥ rolling-30d (=180 4h-bars) p95 of
    |alpha_beta|, all values strictly through t-2 (PIT, no overlap with target).
    Event indicates the PRIOR 4h block was a high-magnitude residual.
  - Target: sign(alpha_beta[t]) — the NEXT 4h block's residual sign.
  - Features (V2-set, BTC-frame, no sym_id): 17 features pre-committed below.
  - Vol baseline: idio_vol_to_btc_1d only (pre-committed, no post-hoc swap).
  - Model: RidgeCV with α ∈ {0.01, 0.1, 1, 10, 100}.
  - Folds: 9 time-OOS, expanding-window train, 1-day embargo.

Gates (binary):
  PASS  iff AUC_full ≥ 0.53 AND (AUC_full − AUC_vol_only) ≥ 0.015
  CLOSE otherwise

No strategy. No cost. No placebos. Just signal existence.
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
WIN_30D = 180          # 30 days × 6 4h-bars/day
MIN_PRIOR = 30         # min prior 4h-bars to compute p95
EMBARGO_DAYS = 1

FEATS = [
    # 11 frame-neutral (W17 subset present)
    "return_1d", "atr_pct", "obv_z_1d", "vwap_slope_96",
    "bars_since_high_xs_rank",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d",
    # 4 BTC-frame replacements
    "dom_btc_z_1d", "dom_btc_change_288b",
    "corr_to_btc_change_3d", "idio_vol_to_btc_1d",
    # V2 short-horizon adds
    "return_8h", "vol_zscore_4h_over_7d",
]
VOL_BASELINE = "idio_vol_to_btc_1d"     # PRE-COMMITTED, no swap


def build_events(panel, win=WIN_30D, min_prior=MIN_PRIOR):
    """Per-symbol PIT event detection on the 4h-aligned series."""
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    panel["prior_abs_alpha"] = panel.groupby("symbol")["alpha_beta"].shift(1).abs()
    panel["trail_p95"] = (panel.groupby("symbol")["prior_abs_alpha"]
                              .transform(lambda s: s.rolling(win, min_periods=min_prior).quantile(0.95).shift(1)))
    panel["event"] = (panel["prior_abs_alpha"] >= panel["trail_p95"]).fillna(False)
    return panel


def winsor_z_fit(train_df, cols):
    stats = {}
    for c in cols:
        v = train_df[c].dropna()
        if len(v) < 50:
            stats[c] = {"lo": 0.0, "hi": 1.0, "mu": 0.0, "sd": 1.0}
            continue
        lo = float(v.quantile(0.01)); hi = float(v.quantile(0.99))
        vc = v.clip(lo, hi)
        mu = float(vc.mean()); sd = float(vc.std()) or 1.0
        stats[c] = {"lo": lo, "hi": hi, "mu": mu, "sd": sd}
    return stats


def winsor_z_apply(df, cols, stats):
    X = df[cols].copy()
    for c in cols:
        s = stats[c]
        X[c] = X[c].clip(s["lo"], s["hi"])
        X[c] = (X[c] - s["mu"]) / s["sd"]
    X = X.fillna(0.0)
    return X.values


def main():
    t0 = time.time()
    print("=== E0 broad-universe convexity sign-predictability ===\n", flush=True)
    p = pd.read_parquet(PANEL,
                        columns=["symbol", "open_time", "alpha_beta"] + FEATS)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    print(f"  raw panel: {len(p):,} rows × {p['symbol'].nunique()} syms", flush=True)

    # 4h-aligned subsample (open_time hour in {0,4,8,12,16,20}, minute 0)
    p = p[(p["open_time"].dt.minute == 0) & (p["open_time"].dt.hour % 4 == 0)]
    p = p.dropna(subset=["alpha_beta"]).reset_index(drop=True)
    print(f"  4h-aligned + alpha_beta non-null: {len(p):,} rows", flush=True)

    # event detection
    p = build_events(p)
    evt = p[p["event"]].copy()
    print(f"  event bars (PIT): {len(evt):,} ({len(evt)/len(p)*100:.2f}% of 4h-bars)",
          flush=True)
    evt["target_sign"] = np.sign(evt["alpha_beta"])
    evt = evt[evt["target_sign"] != 0].reset_index(drop=True)
    print(f"  event bars w/ non-zero target: {len(evt):,}", flush=True)
    print(f"  target balance: P(long)={(evt['target_sign']>0).mean():.3f}  "
          f"P(short)={(evt['target_sign']<0).mean():.3f}", flush=True)

    # 9 time-OOS folds
    times = sorted(evt["open_time"].unique())
    n_times = len(times)
    fold_size = n_times // N_FOLDS
    print(f"  9 folds, ~{fold_size} unique times per fold "
          f"(span {pd.Timestamp(times[0]).date()} → {pd.Timestamp(times[-1]).date()})\n",
          flush=True)

    per_fold = []
    aucs_full, aucs_vol = [], []
    for f in range(N_FOLDS):
        i0 = f * fold_size
        i1 = min((f + 1) * fold_size, n_times - 1) if f < N_FOLDS - 1 else n_times
        oos_start = pd.Timestamp(times[i0])
        oos_end = pd.Timestamp(times[i1 - 1])
        embargo_cut = oos_start - pd.Timedelta(days=EMBARGO_DAYS)

        train = evt[evt["open_time"] < embargo_cut]
        test = evt[(evt["open_time"] >= oos_start) & (evt["open_time"] <= oos_end)]
        if len(train) < 200 or len(test) < 30:
            print(f"  fold {f}: skipped (n_train={len(train)}, n_test={len(test)})",
                  flush=True)
            continue

        stats = winsor_z_fit(train, FEATS)
        Xtr = winsor_z_apply(train, FEATS, stats)
        Xte = winsor_z_apply(test, FEATS, stats)
        ytr = train["target_sign"].to_numpy()
        yte = test["target_sign"].to_numpy()
        yte_b = (yte > 0).astype(int)

        # full model
        m_full = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
        pred_full = m_full.predict(Xte)
        try:
            auc_full = roc_auc_score(yte_b, pred_full)
        except ValueError:
            auc_full = np.nan

        # vol-only baseline
        Xtr_v = Xtr[:, [FEATS.index(VOL_BASELINE)]]
        Xte_v = Xte[:, [FEATS.index(VOL_BASELINE)]]
        m_vol = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr_v, ytr)
        pred_vol = m_vol.predict(Xte_v)
        try:
            auc_vol = roc_auc_score(yte_b, pred_vol)
        except ValueError:
            auc_vol = np.nan

        per_fold.append({
            "fold": f,
            "oos_start": str(oos_start.date()),
            "oos_end": str(oos_end.date()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "alpha": float(m_full.alpha_),
            "auc_full": round(float(auc_full), 4) if auc_full == auc_full else None,
            "auc_vol_only": round(float(auc_vol), 4) if auc_vol == auc_vol else None,
            "delta": round(float(auc_full - auc_vol), 4) if (auc_full == auc_full and auc_vol == auc_vol) else None,
            "frac_long_oos": round(float((yte > 0).mean()), 3),
        })
        if auc_full == auc_full: aucs_full.append(auc_full)
        if auc_vol == auc_vol:   aucs_vol.append(auc_vol)

        print(f"  fold {f}: n_tr={len(train):>5,} n_te={len(test):>5,}  "
              f"α={m_full.alpha_:>6.2f}  AUC_full={auc_full:.4f}  AUC_vol={auc_vol:.4f}  "
              f"Δ={auc_full-auc_vol:+.4f}  long%={(yte>0).mean()*100:.1f}",
              flush=True)

    # aggregate
    auc_full_mean = float(np.mean(aucs_full)) if aucs_full else np.nan
    auc_vol_mean = float(np.mean(aucs_vol)) if aucs_vol else np.nan
    delta_mean = auc_full_mean - auc_vol_mean

    # gate
    gate_auc_full = auc_full_mean >= 0.53
    gate_delta = delta_mean >= 0.015
    if not gate_auc_full:
        verdict = "CLOSE — AUC_full < 0.53 (no portable sign-predictability on broad universe)"
    elif not gate_delta:
        verdict = "CLOSE — AUC_full − AUC_vol < 0.015 (classifier is vol detector)"
    else:
        verdict = "PASS — proceed to designing universe-portability tests + strategy plan"

    # also check fold consistency
    n_folds_above_53 = sum(1 for a in aucs_full if a >= 0.53)
    n_folds_delta_above_015 = sum(1 for r in per_fold if r["delta"] is not None and r["delta"] >= 0.015)

    out = {
        "scope": "Broad 110-panel, event-conditional sign-predictability, time-OOS",
        "n_event_bars": int(len(evt)),
        "long_share": round(float((evt["target_sign"] > 0).mean()), 4),
        "auc_full_mean": round(auc_full_mean, 4),
        "auc_vol_only_mean": round(auc_vol_mean, 4),
        "delta_mean": round(delta_mean, 4),
        "n_folds_auc_full>=0.53": n_folds_above_53,
        "n_folds_delta>=0.015": n_folds_delta_above_015,
        "gate_auc_full>=0.53": bool(gate_auc_full),
        "gate_delta>=0.015": bool(gate_delta),
        "verdict": verdict,
        "per_fold": per_fold,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "E0_results.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"\n=== AGGREGATE ===", flush=True)
    print(f"  AUC_full mean: {auc_full_mean:.4f}", flush=True)
    print(f"  AUC_vol mean:  {auc_vol_mean:.4f}", flush=True)
    print(f"  delta mean:    {delta_mean:+.4f}", flush=True)
    print(f"  folds AUC ≥ 0.53:  {n_folds_above_53}/{len(aucs_full)}", flush=True)
    print(f"  folds Δ ≥ 0.015:   {n_folds_delta_above_015}/{len(per_fold)}", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"\n[elapsed {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
