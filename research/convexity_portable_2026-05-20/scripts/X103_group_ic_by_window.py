"""X103 — Per-feature-group IC by time window: did ANY group survive the mid-2025 decay?

Docs (X30, favorable window) reported per-sym |IC| by group: v3 0.0338 (highest),
BASE 0.0286, crossX 0.0250, cohort 0.0175, aggT 0.0171. X101/X102 showed the MODEL's
cross-sec IC decayed to ~0 since mid-2025. Question: is the decay uniform across feature
groups, or does some group (esp. v3 idio-vol) still carry signal recently?

Metric: per-symbol Spearman(feature, alpha_vs_btc_realized) averaged across symbols
(matches the docs' "per-sym |IC|"). Report mean |IC| AND signed mean IC per group, per
window: 2024 / 2025H1 / 2025H2 / 2026, and recent = 2025H2+2026. 4h-aligned rows only.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_3yr_v5.parquet"
TGT = "alpha_vs_btc_realized"

GROUPS = {
    "BASE": ["return_1d","atr_pct","obv_z_1d","vwap_slope_96","bars_since_high",
             "bars_since_high_xs_rank","autocorr_pctile_7d","corr_to_btc_1d",
             "beta_to_btc_change_5d","idio_vol_to_btc_1h","idio_vol_to_btc_1d",
             "funding_rate","funding_rate_z_7d","funding_rate_1d_change"],
    "cohort": ["rvol_7d","ret_3d","btc_rvol_7d"],
    "aggT": ["aggr_ratio_4h","tfi_4h","buy_count_4h","signed_volume_4h","avg_trade_size_4h"],
    "crossX": ["bn_perp_okx_perp_z","bn_perp_okx_spot_z","okx_perp_spot_z","bn_perp_cb_spot_z",
               "okx_cb_spot_z","bn_spot_okx_spot_z","bn_spot_cb_spot_z"],
    "v3": ["idio_max_abs_12b","idio_skew_1d","idio_kurt_1d","name_idio_share_1d"],
}
ALL_FEATS = [f for g in GROUPS.values() for f in g]


def persym_ic(df, feat):
    """mean signed per-symbol Spearman(feat, target); also returns mean |IC|."""
    ics=[]
    for sym,g in df.groupby("symbol", sort=False):
        s=g[[feat,TGT]].dropna()
        if len(s)<50: continue
        c=s[feat].corr(s[TGT], method="spearman")
        if np.isfinite(c): ics.append(c)
    if not ics: return np.nan, np.nan, 0
    a=np.array(ics)
    return a.mean(), np.abs(a).mean(), len(a)


def main():
    t0=time.time()
    print("=== X103 per-group IC by window ===\n", flush=True)
    cols=["symbol","open_time",TGT]+ALL_FEATS
    df=pd.read_parquet(PANEL, columns=cols)
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    df=df[(df["open_time"].dt.hour%4==0)&(df["open_time"].dt.minute==0)]
    print(f"rows(4h): {len(df):,}  syms: {df['symbol'].nunique()}  "
          f"range {df['open_time'].min().date()}->{df['open_time'].max().date()}\n", flush=True)

    def win(name, lo, hi):
        m=(df["open_time"]>=pd.Timestamp(lo,tz="UTC"))&(df["open_time"]<pd.Timestamp(hi,tz="UTC"))
        return name, df[m]
    windows=[win("2024",     "2024-01-01","2025-01-01"),
             win("2025H1",   "2025-01-01","2025-07-01"),
             win("2025H2",   "2025-07-01","2026-01-01"),
             win("2026",     "2026-01-01","2026-07-01"),
             win("recent_H2+26","2025-07-01","2026-07-01")]

    # per-group mean |IC| (signed mean IC in parens)
    print(f"  {'group':<8}" + "".join(f"{w:>16}" for w,_ in windows))
    grp_abs={g:{} for g in GROUPS}; grp_sgn={g:{} for g in GROUPS}
    feat_detail={}
    for g,feats in GROUPS.items():
        for w,sub in windows:
            sgns=[]; abss=[]
            for f in feats:
                s,a,n=persym_ic(sub,f)
                if np.isfinite(a): sgns.append(s); abss.append(a)
                if w=="recent_H2+26": feat_detail.setdefault(g,[]).append((f,s,a))
            grp_abs[g][w]=np.mean(abss) if abss else np.nan
            grp_sgn[g][w]=np.mean(sgns) if sgns else np.nan
        cells="".join(f"{grp_abs[g][w]:>+8.4f}({grp_sgn[g][w]:>+.3f})"[:16].rjust(16) for w,_ in windows)
        print(f"  {g:<8}{cells}", flush=True)

    print("\n  (each cell = mean|IC| (signed meanIC). Compare to favorable-window: v3 0.0338, BASE 0.0286, crossX 0.0250)\n")

    print("=== recent (2025H2+2026) per-FEATURE |IC|, top signal-carriers ===")
    flat=[(g,f,s,a) for g,lst in feat_detail.items() for (f,s,a) in lst]
    flat.sort(key=lambda x:-x[3])
    print(f"  {'group':<8}{'feature':<24}{'signedIC':>10}{'|IC|':>8}")
    for g,f,s,a in flat[:12]:
        print(f"  {g:<8}{f:<24}{s:>+10.4f}{a:>8.4f}", flush=True)

    print(f"\nVERDICT: if all groups' recent |IC| collapsed toward favorable-window noise floor (~0.01-0.015),")
    print(f"decay is uniform/market-wide. If v3 stays elevated, idio-vol still carries signal. Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
