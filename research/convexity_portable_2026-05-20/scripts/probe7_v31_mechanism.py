"""Probe #7 — what does V3.1 actually capture?

Not asking "is convexity predictable" — answered NO by E0. Asking the
SEPARATE question: at the time V3.1 picks a symbol (long or short), what
feature signature does that symbol show RELATIVE TO THE CROSS-SECTION at
that moment? This is interpretability, not prediction.

Method (PIT-clean, no model retraining):
  1. Load entered_legs.parquet (3448 picks from V3.1 51-panel run).
  2. Join each pick to the panel at entry time → get feature values.
  3. Per cycle (open_time), compute cross-sectional z-score of each feature
     across all eligible symbols at that time.
  4. For each pick, the symbol's xs_z gives a vector "how outlier was the
     pick on each feature when it was selected."
  5. Compute the mean xs_z signature for:
        - long top-quintile by realized contrib (the big winners)
        - long bottom-quintile (longs that bombed)
        - short top-quintile by realized contrib (the big short winners)
        - short bottom-quintile (shorts that bombed)
  6. The signature reveals what feature pattern V3.1 implicitly chose.

Specifically these characterize the strategy type:
  - high return_1d at long-entry → momentum-following
  - low return_1d at long-entry → contrarian / mean-reversion
  - high atr_pct, low corr_to_btc_1d → idiosyncratic volatility hunter
  - high funding_rate at long-entry → squeezing shorts (long against negative carry)
  - low funding_rate at long-entry → trend-following with carry
  - low dom_change → name diverging from basket
  - high vol_zscore_4h_over_7d → recent volatility expansion

Reading the signature tells us what regime / feature class drives V3.1 picks.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"; OUT.mkdir(parents=True, exist_ok=True)
LEGS = REPO / "research/convexity_forensic_2026-05-19/entered_legs.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"

# Comprehensive feature set covering momentum / vol / corr / funding / dom
SIG_FEATS = [
    # momentum / return
    "return_1d", "return_8h",
    # volatility
    "atr_pct", "idio_vol_to_btc_1d", "idio_vol_to_btc_1h", "vol_zscore_4h_over_7d",
    # correlation / beta
    "corr_to_btc_1d", "corr_to_btc_change_3d", "beta_to_btc_change_5d",
    # idio / dominance
    "idio_vol_1d_vs_bk", "name_idio_share_1d",
    "dom_level_vs_bk", "dom_change_288b_vs_bk",
    # funding (heavy-tail)
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change", "funding_streak_pos",
    # microstructure
    "obv_z_1d", "vwap_slope_96", "mfi", "aggr_ratio_4h",
    # ranks / state
    "bars_since_high_xs_rank",
]


def main():
    t0 = time.time()
    L = pd.read_parquet(LEGS)
    L["time"] = pd.to_datetime(L["time"], utc=True)

    # load panel (only needed columns)
    cols_avail = pd.read_parquet(PANEL, columns=None).head(1).columns
    feats_present = [c for c in SIG_FEATS if c in cols_avail]
    print(f"  feats present in panel: {len(feats_present)}/{len(SIG_FEATS)}", flush=True)
    pan_cols = ["symbol", "open_time"] + feats_present
    pan = pd.read_parquet(PANEL, columns=pan_cols)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)

    # cross-sectional z-score per cycle (open_time)
    print("  computing cross-sectional z per cycle...", flush=True)
    pan_z = pan.copy()
    for c in feats_present:
        pan_z[c] = pan_z.groupby("open_time")[c].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))
    # rename to mark as xs-z
    z_cols = {c: f"{c}_xsz" for c in feats_present}
    pan_z = pan_z.rename(columns=z_cols)[["symbol", "open_time"] + list(z_cols.values())]

    # join legs to xs-z
    M = L.rename(columns={"time": "open_time"}).merge(
        pan_z, on=["symbol", "open_time"], how="left")
    print(f"  merged: {len(M):,} legs, "
          f"join missing rate {M[list(z_cols.values())[0]].isna().mean()*100:.1f}%",
          flush=True)
    M = M.dropna(subset=list(z_cols.values()))

    # split long/short, top/bottom quintile by realized contrib
    def quintile_signature(df):
        if len(df) < 50: return None
        q_lo, q_hi = df["contrib_bps"].quantile([0.20, 0.80])
        top = df[df["contrib_bps"] >= q_hi]
        bot = df[df["contrib_bps"] <= q_lo]
        mid = df[(df["contrib_bps"] > q_lo) & (df["contrib_bps"] < q_hi)]
        out = {}
        for c in z_cols.values():
            t = top[c].dropna(); b = bot[c].dropna(); m = mid[c].dropna()
            if len(t) < 10 or len(b) < 10: continue
            out[c] = {
                "top_mean": round(float(t.mean()), 3),
                "bot_mean": round(float(b.mean()), 3),
                "mid_mean": round(float(m.mean()), 3),
                "spread_top_minus_bot": round(float(t.mean() - b.mean()), 3),
                "cohen_d_top_vs_bot": round(float(
                    (t.mean() - b.mean()) / np.sqrt((t.var() + b.var()) / 2)
                    if (t.var() + b.var()) > 0 else 0), 3),
                "t_stat_top_vs_bot": round(float(
                    stats.ttest_ind(t, b, equal_var=False, nan_policy="omit").statistic), 2),
            }
        # rank features by |spread|
        sorted_feats = sorted(out.keys(),
                              key=lambda k: -abs(out[k]["spread_top_minus_bot"]))
        return {"by_feature": out, "top_features_by_spread": sorted_feats}

    longs = M[M["side"] == "long"].copy()
    shorts = M[M["side"] == "short"].copy()
    long_sig = quintile_signature(longs)
    short_sig = quintile_signature(shorts)

    # per-symbol-class characterization: top 5 long winners + their feature signatures
    big_long_syms = (longs.groupby("symbol")["contrib_bps"].sum()
                          .sort_values(ascending=False).head(5).index.tolist())
    big_short_syms = (shorts.groupby("symbol")["contrib_bps"].sum()
                            .sort_values(ascending=False).head(5).index.tolist())
    per_sym_signature = {}
    for kind, sym_list, df_side in (("long", big_long_syms, longs),
                                       ("short", big_short_syms, shorts)):
        for sym in sym_list:
            d = df_side[df_side["symbol"] == sym]
            sig = {}
            for c in z_cols.values():
                v = d[c].dropna()
                if len(v) < 5: continue
                sig[c] = round(float(v.mean()), 3)
            per_sym_signature[f"{kind}_{sym}"] = {
                "n_picks": int(len(d)),
                "total_contrib_bps": int(d["contrib_bps"].sum()),
                "mean_xsz": sig,
            }

    out = {
        "n_legs_total": int(len(L)),
        "n_legs_joined": int(len(M)),
        "long_signature": long_sig,
        "short_signature": short_sig,
        "big_winner_per_sym_signatures": per_sym_signature,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe7_v31_mechanism.json").write_text(
        json.dumps(out, indent=2, default=str))

    # printout
    print(f"\n=== LONG-SIDE SIGNATURE (n={len(longs):,}) ===")
    print(f"  Top quintile (big winners) vs bottom quintile (losers), xs-z mean spread:")
    print(f"  {'feature':<32} {'top':>8} {'mid':>8} {'bot':>8} {'spread':>8} {'cohen_d':>8} {'t-stat':>8}")
    if long_sig:
        for fname in long_sig["top_features_by_spread"][:15]:
            d = long_sig["by_feature"][fname]
            print(f"  {fname:<32} {d['top_mean']:>+8.2f} {d['mid_mean']:>+8.2f} "
                  f"{d['bot_mean']:>+8.2f} {d['spread_top_minus_bot']:>+8.2f} "
                  f"{d['cohen_d_top_vs_bot']:>+8.2f} {d['t_stat_top_vs_bot']:>+8.2f}")
    print(f"\n=== SHORT-SIDE SIGNATURE (n={len(shorts):,}) ===")
    if short_sig:
        for fname in short_sig["top_features_by_spread"][:15]:
            d = short_sig["by_feature"][fname]
            print(f"  {fname:<32} {d['top_mean']:>+8.2f} {d['mid_mean']:>+8.2f} "
                  f"{d['bot_mean']:>+8.2f} {d['spread_top_minus_bot']:>+8.2f} "
                  f"{d['cohen_d_top_vs_bot']:>+8.2f} {d['t_stat_top_vs_bot']:>+8.2f}")

    print(f"\n=== BIG WINNER PER-SYMBOL FEATURE SIGNATURES ===")
    for k, v in per_sym_signature.items():
        print(f"  {k}: n={v['n_picks']}, total_contrib={v['total_contrib_bps']:+d} bps")
        # show top 5 most-extreme features
        top_feats = sorted(v["mean_xsz"].items(), key=lambda x: -abs(x[1]))[:5]
        for fname, z in top_feats:
            print(f"    {fname:<32} {z:>+6.2f}")
    print("\nPROBE7_DONE")


if __name__ == "__main__":
    main()
