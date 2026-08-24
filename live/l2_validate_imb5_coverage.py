"""Validate imb5 (WIDE-book ±5% imbalance) coverage specifically — the feature under incremental test. A null result
is only trustworthy if imb5 is DENSELY covered in BOTH eras (the original fetch had OOS ~28% covered = underpowered).
Per symbol, per era: imb5 non-null bar count, imb5 NaN% (should be ~0 — ±5% levels exist since 2023-01, unlike the
recent-only ±0.2% near-touch), day-density of imb5, largest imb5 gap, and imb5-vs-imb1 parity (imb5 should track imb1
since both come from the same 30s snapshots). Flags any symbol with thin/sparse/gappy imb5 in either era. Read-only,
safe during the backfill (skips mid-write files).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
CACHE = "/home/yuqing/ctaNew/data/ml/cache"
REC0 = pd.Timestamp("2025-10-01", tz="UTC")
# expected trading days per era (for density denominator): OOS 2023-01-01..2025-09-30, RECENT 2025-10-01..2026-07-14
OOS_DAYS = (pd.Timestamp("2025-09-30", tz="UTC") - pd.Timestamp("2023-01-01", tz="UTC")).days + 1

def era_density(dates, lo, hi):
    """% of days in [lo,hi] (clipped to the symbol's own listing span) that have an imb5 obs."""
    dd = dates[(dates >= lo) & (dates < hi)]
    if not len(dd): return 0, 0, 0
    span = (dd.max() - dd.min()).days + 1
    ncov = pd.Series(dd.date).nunique()
    du = pd.to_datetime(pd.Series(sorted(set(dd.date))))
    maxgap = int((du.diff().dt.days.fillna(1) - 1).max()) if len(du) > 1 else 0
    return ncov, round(ncov / span * 100) if span else 0, maxgap

def validate(f):
    sym = Path(f).stem[3:]
    try:
        d = pd.read_parquet(f)
    except Exception as e:
        return {"sym": sym, "err": type(e).__name__}
    if "l2_imb5" not in d.columns:
        return {"sym": sym, "err": "no_imb5"}
    ix = pd.to_datetime(pd.Index(d.index), utc=True)
    d = d.set_index(ix).sort_index()
    m5 = d["l2_imb5"].notna(); m1 = d["l2_imb1"].notna()
    i5 = d.index[m5]                                   # timestamps where imb5 present
    oos_ix, rec_ix = i5[i5 < REC0], i5[i5 >= REC0]
    _, oos_dens, oos_gap = era_density(oos_ix, pd.Timestamp("2023-01-01", tz="UTC"), REC0)
    _, rec_dens, rec_gap = era_density(rec_ix, REC0, pd.Timestamp("2026-07-15", tz="UTC"))
    return {"sym": sym, "err": np.nan, "bars": len(d),
            "imb5_oos": int(len(oos_ix)), "imb5_rec": int(len(rec_ix)),
            "imb5_nan%": round(d["l2_imb5"].isna().mean() * 100, 1),
            "imb5_vs_imb1_gap": int((m1 & ~m5).sum()),   # bars with imb1 but NOT imb5 (parity: should be ~0)
            "oos_dens%": oos_dens, "oos_maxgap_d": oos_gap, "rec_dens%": rec_dens, "rec_maxgap_d": rec_gap}

def main():
    V = pd.DataFrame([validate(f) for f in glob.glob(CACHE + "/l2_*.parquet")])
    errs = V[V["err"].notna()]; ok = V[V["err"].isna()].copy()
    both = (ok.imb5_oos >= 200) & (ok.imb5_rec >= 200)
    rec_ok = ok.imb5_rec >= 200
    print(f"=== imb5 (WIDE-BOOK) COVERAGE  ({len(V)} cache files) ===")
    if len(errs):
        noimb5 = errs[errs.err == "no_imb5"]; unread = errs[errs.err != "no_imb5"]
        print(f"  NOT yet re-fetched (no imb5 col): {len(noimb5)}  {list(noimb5.sym)[:15]}")
        if len(unread): print(f"  unreadable (mid-write?): {list(unread.sym)}")
    print(f"  with imb5: {len(ok)}/{len(V)}   | both-era (>=200 imb5 bars each): {int(both.sum())}   | recent-only: {int((rec_ok & ~both).sum())}")
    print(f"  imb5 NaN% within rows:  median {ok['imb5_nan%'].median():.1f}%  (p90 {ok['imb5_nan%'].quantile(.9):.1f}%)   [expect ~0; ±5% levels exist since 2023-01]")
    print(f"  imb5-vs-imb1 parity (bars w/ imb1 but no imb5): median {int(ok.imb5_vs_imb1_gap.median())}  (p90 {int(ok.imb5_vs_imb1_gap.quantile(.9))})   [expect ~0]")
    print(f"  OOS day-density: median {ok['oos_dens%'].median():.0f}%  (p10 {ok['oos_dens%'].quantile(.1):.0f}%)  | OOS max-gap median {ok.oos_maxgap_d.median():.0f}d (p90 {ok.oos_maxgap_d.quantile(.9):.0f}d)")
    print(f"  REC day-density: median {ok['rec_dens%'].median():.0f}%  (p10 {ok['rec_dens%'].quantile(.1):.0f}%)  | REC max-gap median {ok.rec_maxgap_d.median():.0f}d (p90 {ok.rec_maxgap_d.quantile(.9):.0f}d)")
    print(f"  both-era imb5 bars: OOS median {int(ok[both].imb5_oos.median())}  RECENT median {int(ok[both].imb5_rec.median())}")
    # separate "newer listing, legit no OOS" (recent-only, expected) from genuine coverage defects
    recent_only = ok[(ok.imb5_oos < 200) & (ok.imb5_rec >= 200)]
    defect = ok[((ok['oos_dens%'] < 70) | (ok.oos_maxgap_d > 30) | (ok['imb5_nan%'] > 5) | (ok.imb5_vs_imb1_gap > 50)) & (ok.imb5_oos >= 200)]
    print(f"\n  recent-only (newer listing, legit no OOS): {len(recent_only)}  {list(recent_only.sym)[:20]}")
    print(f"\n=== GENUINE both-era imb5 coverage DEFECTS ({len(defect)}: OOS-dens<70% OR OOS-gap>30d OR NaN>5% OR imb1-no-imb5>50, among both-era syms) ===")
    for _, r in defect.sort_values("oos_dens%").head(30).iterrows():
        print(f"  {r['sym']:13s} imb5 oos {int(r['imb5_oos']):4d} rec {int(r['imb5_rec']):4d} | oosDens {int(r['oos_dens%']):3d}% oosGap {int(r['oos_maxgap_d']):3d}d | NaN {r['imb5_nan%']}% | imb1noImb5 {int(r['imb5_vs_imb1_gap'])}")
    if not len(defect): print("  (none — every both-era symbol has dense, parity-clean imb5)")
    print(f"\n(symbols still showing no_imb5 = backfill not yet reached them; re-run at REFETCHDONE)")
    print("IMB5COVDONE")

if __name__ == "__main__":
    main()
