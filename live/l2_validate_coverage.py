"""Validate the COLLECTED L2 bookDepth cache coverage/quality (not just file existence). Per symbol: date span,
day-level density (missing bookDepth days within span), largest internal gap, both-era presence (OOS 2023→2025-09 +
recent 2025-10+), the previously-missing middle (2024-08→2025-09) fill, and feature NaN rates. Aggregates + flags any
problem symbol (thin span, big gap, low density, high NaN, single-era). Read-only; safe to run while the backfill
writes (skips any file mid-write). Run now (partial) + again at completion.
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
CACHE = "/home/yuqing/ctaNew/data/ml/cache"
REC0 = pd.Timestamp("2025-10-01", tz="UTC"); OOS0 = pd.Timestamp("2023-01-01", tz="UTC")
MID0, MID1 = pd.Timestamp("2024-08-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC")

def validate(f):
    sym = Path(f).stem[3:]
    try:
        d = pd.read_parquet(f)
    except Exception as e:
        return {"sym": sym, "err": type(e).__name__}
    ix = pd.to_datetime(pd.Index(d.index), utc=True).sort_values()
    if not len(ix): return {"sym": sym, "err": "empty"}
    span = (ix.max() - ix.min()).days + 1
    dcov = pd.Series(ix.date).nunique(); dens = round(dcov / span * 100) if span else 0
    du = pd.to_datetime(pd.Series(sorted(set(ix.date))))
    gaps = du.diff().dt.days.fillna(1) - 1
    max_gap = int(gaps.max()) if len(gaps) else 0
    nan = lambda c: round(d[c].isna().mean() * 100) if c in d.columns else 100
    return {"sym": sym, "bars": len(d), "start": str(ix.min())[:10], "end": str(ix.max())[:10], "span_d": span,
            "dens%": dens, "maxgap_d": max_gap, "oos": int((ix < REC0).sum()), "rec": int((ix >= REC0).sum()),
            "mid": int(((ix >= MID0) & (ix < MID1)).sum()), "imb1_nan%": nan("l2_imb1"), "imb02_nan%": nan("l2_imb02")}

def main():
    V = pd.DataFrame([validate(f) for f in glob.glob(CACHE + "/l2_*.parquet")])
    errs = V[V.get("err").notna()] if "err" in V else V.iloc[:0]
    ok = V[V["err"].isna()] if "err" in V else V
    ok = ok.copy()
    both = (ok.oos >= 200) & (ok.rec >= 200)
    fullrange = ok.start <= "2023-06-30"          # reached back into early OOS
    midfilled = ok.mid > 100
    print(f"=== L2 CACHE COVERAGE ({len(V)} symbols; {len(errs)} unreadable) ===")
    print(f"  both-era (>=200 bars each): {int(both.sum())}/{len(ok)}   | recent-only: {int((~both & (ok.rec>=200)).sum())}   | OOS-only: {int((~both & (ok.oos>=200)).sum())}")
    print(f"  reach back to <=2023-06 (deep OOS): {int(fullrange.sum())}/{len(ok)}   | middle-gap (2024-08..2025-09) filled: {int(midfilled.sum())}/{len(ok)}")
    print(f"  day-density: median {ok['dens%'].median():.0f}%  (p10 {ok['dens%'].quantile(.1):.0f}%)   | max internal gap median {ok.maxgap_d.median():.0f}d (p90 {ok.maxgap_d.quantile(.9):.0f}d)")
    print(f"  span days: median {ok.span_d.median():.0f}   | imb1 NaN median {ok['imb1_nan%'].median():.0f}%  | imb02(near-touch) NaN median {ok['imb02_nan%'].median():.0f}% (recent-only feature)")
    # flags
    flag = ok[(ok['dens%'] < 85) | (ok.maxgap_d > 20) | (ok['imb1_nan%'] > 5) | (ok.bars < 500)]
    print(f"\n=== FLAGGED ({len(flag)} symbols: density<85% OR maxgap>20d OR imb1-NaN>5% OR <500 bars) ===")
    for _, r in flag.sort_values("dens%").head(25).iterrows():
        print(f"  {r['sym']:14s} bars {r['bars']:5d} {r['start']}..{r['end']} dens {r['dens%']:3d}% maxgap {r['maxgap_d']:3d}d oos {r['oos']:4d} rec {r['rec']:4d} imb1NaN {r['imb1_nan%']}%")
    if len(errs): print("  UNREADABLE:", list(errs.sym))
    print(f"\n(note: symbols the resume backfill has not yet re-fetched still show the OLD partial coverage — re-run at completion)")
    print("VALDONE")

if __name__ == "__main__":
    main()
