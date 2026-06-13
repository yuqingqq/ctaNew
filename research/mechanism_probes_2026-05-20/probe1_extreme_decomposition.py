"""Probe #1 — Extreme-winners vs extreme-losers feature decomposition (descriptive).

PURE DATA ARCHAEOLOGY (no strategy claim, no PIT issues — features at entry
were PIT per R0; the "label" is realized outcome used purely for grouping).

Take entered LONG legs from the production sim. Top decile by realized
contribution = "extreme winners"; bottom decile = "extreme losers".
Decompose ~40 PIT features at entry: which features systematically differ
between the two groups? Cohen's d + KS, ranked by effect size. Calibrate
against a label-shuffled NULL baseline (100 shuffles) so multi-testing
noise is bounded.

Output: ranked features that empirically distinguish the kinds of trades
that pay vs hurt. These become mechanism hypotheses for the NEXT probe
(does any of them, used predictively + OOS-symbol, identify the trade kind?).
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/mechanism_probes_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CW = REPO / "research/portable_alpha_2026-05-19/results/_cache/close_wide.parquet"

# Panel PIT features (R0-clean); exclude any target/forward column
PANEL_FEATS = [
    # WINNER_21 + funding extras
    "atr_pct", "return_1d", "ema_slope_20_1h",
    "dom_level_vs_bk", "dom_z_1d_vs_bk", "dom_change_288b_vs_bk",
    "bk_ema_slope_4h", "idio_vol_1d_vs_bk", "idio_ret_48b_vs_bk",
    "corr_change_3d_vs_bk", "beta_short_vs_bk",
    "obv_z_1d", "vwap_slope_96", "vwap_zscore",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "funding_streak_pos", "funding_streak_neg",
    "corr_to_btc_1d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
    "beta_to_btc_change_5d", "idio_ret_to_btc_12b", "idio_ret_to_btc_48b",
    # Higher-moment / lifecycle-style features in panel (PIT)
    "idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b",
    "name_idio_share_1d", "name_factor_loading_1d",
    "btc_ret_48b", "btc_realized_vol_1d", "btc_realized_vol_30d",
    "btc_ema_slope_4h",
    # Microstructure (4h aggregates) — PIT
    "tfi_4h", "signed_volume_4h", "aggr_ratio_4h", "avg_trade_size_4h",
    "buy_count_4h",
]


def cohens_d(a, b):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10: return np.nan
    s = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def main():
    t0 = time.time()
    legs = pd.read_parquet(LEGS)
    legs["time"] = pd.to_datetime(legs["time"], utc=True)
    L = legs[legs["side"] == "long"].copy()
    p10, p90 = L["contrib_bps"].quantile([0.10, 0.90])
    L["grp"] = np.where(L["contrib_bps"] >= p90, "WIN",
                np.where(L["contrib_bps"] <= p10, "LOSE", "MID"))
    nW = int((L["grp"] == "WIN").sum()); nL = int((L["grp"] == "LOSE").sum())
    print(f"longs={len(L)} extreme-winners (top-10%)={nW} extreme-losers (bot-10%)={nL}")
    print(f"  win contrib mean={L[L.grp=='WIN'].contrib_bps.mean():+.1f} bps,"
          f"  lose contrib mean={L[L.grp=='LOSE'].contrib_bps.mean():+.1f} bps")

    # Merge panel features at entry — keep only cols actually present
    panel_cols = set(pd.read_parquet(PANEL, columns=None).head(1).columns)
    feats_present = [c for c in PANEL_FEATS if c in panel_cols]
    missing = [c for c in PANEL_FEATS if c not in panel_cols]
    if missing: print(f"  panel missing (dropped): {missing}")
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time"] + feats_present)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    D = L.rename(columns={"time": "open_time"}).merge(
        pan, on=["symbol", "open_time"], how="left")

    # Lifecycle features from close_wide (PIT .shift(1))
    cw = pd.read_parquet(CW)
    px = cw[[c for c in cw.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:]).sort_index()
    life_df = {
        "r24":  px.pct_change(288).shift(1),
        "r72":  px.pct_change(864).shift(1),
        "r7d":  px.pct_change(2016).shift(1),
        "dist_20d_hi": (px / px.rolling(5760, min_periods=288).max() - 1.0).shift(1),
        "runup_z": ((px.pct_change(288)
                     - px.pct_change(288).rolling(2016, min_periods=288).mean())
                    / px.pct_change(288).rolling(2016, min_periods=288).std()).shift(1),
    }
    for name, df in life_df.items():
        m = df.reset_index().melt("open_time", var_name="symbol", value_name=name)
        m["open_time"] = pd.to_datetime(m["open_time"], utc=True)
        D = D.merge(m, on=["symbol", "open_time"], how="left")

    ALL_FEATS = feats_present + list(life_df.keys())
    W = D[D.grp == "WIN"]; Lo = D[D.grp == "LOSE"]
    results = []
    for f in ALL_FEATS:
        a, b = W[f].dropna(), Lo[f].dropna()
        if len(a) < 30 or len(b) < 30: continue
        d = cohens_d(a, b)
        try: ks = float(stats.ks_2samp(a, b, alternative="two-sided").statistic)
        except Exception: ks = np.nan
        med_w, med_l = float(a.median()), float(b.median())
        results.append({"feature": f, "cohen_d": round(d, 3) if d == d else None,
                        "ks": round(ks, 3) if ks == ks else None,
                        "median_W": round(med_w, 5), "median_L": round(med_l, 5),
                        "n_W": int(len(a)), "n_L": int(len(b))})
    results.sort(key=lambda r: abs(r["cohen_d"] or 0), reverse=True)

    # Null baseline: shuffle grp 100 times, max |d| under null
    rng = np.random.RandomState(20260520)
    null_max = []
    for _ in range(100):
        grp = D["grp"].to_numpy().copy(); rng.shuffle(grp)
        Wn = D[(grp == "WIN")]; Ln = D[(grp == "LOSE")]
        mx = 0.0
        for f in ALL_FEATS:
            a, b = Wn[f].dropna(), Ln[f].dropna()
            if len(a) < 30 or len(b) < 30: continue
            d = cohens_d(a, b)
            if d == d: mx = max(mx, abs(d))
        null_max.append(mx)
    null_p95 = float(np.percentile(null_max, 95))

    out = {
        "n_long": len(L), "n_extreme_winners": nW, "n_extreme_losers": nL,
        "null_max_d_p95_multitest_bar": round(null_p95, 3),
        "top_15_features_by_|cohen_d|": results[:15],
        "interpretation_keys": {
            "cohen_d_+": "winners have HIGHER value than losers at entry",
            "cohen_d_-": "winners have LOWER value than losers at entry",
            "vs_null": "|cohen_d| > null_p95 = the gap exceeds what label-shuffling produces ~95% of the time across the same multi-feature scan",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe1_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nnull-baseline max|d| p95 over 100 shuffles = {null_p95:.3f}")
    print(f"\ntop 15 features by |Cohen's d| (winners − losers at entry):\n")
    print(f"  {'feature':<28} {'d':>8} {'KS':>6} {'med_W':>10} {'med_L':>10} "
          f"{'vs_null':>8}")
    for r in results[:15]:
        flag = "**" if abs(r["cohen_d"] or 0) > null_p95 else "  "
        print(f"  {r['feature']:<28} {r['cohen_d']:>+8.3f} {r['ks']:>6.3f} "
              f"{r['median_W']:>+10.4g} {r['median_L']:>+10.4g} {flag}")
    print(f"\n(** = exceeds the null-baseline multi-test p95 bar)")
    print("PROBE1_DONE", flush=True)


if __name__ == "__main__":
    main()
