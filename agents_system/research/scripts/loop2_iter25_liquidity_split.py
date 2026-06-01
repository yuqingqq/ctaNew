"""LOOP2 iter-25 — PRINCIPLED two-book split rule by PIT liquidity.

The original flow/price split (69/87) was a data-availability artifact (we fetched flow for the
liquid names first). Now all 175 syms have flow, so we must DESIGN the split. Hypothesis (from
Phase VII): flow helps on liquid/homogeneous names, hurts on thin ones → put the most-liquid N in
the flow book (BookA, V0+flow preds), the rest in the price book (BookB, V0 preds).

Liquidity proxy = trailing-30d dollar volume (flow total_volume × last_price), measured AS OF
OOS start (2025-10-04) — ex-ante / PIT (uses only data ≤ OOS start; liquidity ranks are stable
over the 8-mo window). Writes per-N BookA/BookB pred files (filtered unified_fullflow /
unified_v0full) + random-split placebos of the SAME sizes, for the bot to replay + combine.
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
FULLFLOW = REPO/"live/state/convexity/unified_fullflow_preds.parquet"   # V0+flow, all syms
V0FULL   = REPO/"live/state/convexity/unified_v0full_preds.parquet"     # V0-only, all syms
OUT = REPO/"live/state/convexity/split"; OUT.mkdir(parents=True, exist_ok=True)
OOS0 = pd.Timestamp("2025-10-04", tz="UTC")
N_GRID = [40, 55, 70, 85]
PLACEBO_N = 70          # placebo random splits sized at this N
PLACEBO_SEEDS = 8

def trailing_dvol_at(asof: pd.Timestamp, lookback_days=30) -> dict:
    """Per-sym trailing dollar volume (total_volume*last_price) over [asof-lookback, asof)."""
    lo = asof - pd.Timedelta(days=lookback_days)
    dvol = {}
    for fp in sorted(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet"))):
        sym = Path(fp).stem.replace("flow_", "")
        try:
            f = pd.read_parquet(fp, columns=["total_volume", "last_price"])
        except Exception:
            continue
        idx = pd.to_datetime(f.index, utc=True)
        m = (idx >= lo) & (idx < asof)
        if m.sum() < 50:   # need enough bars to be a real liquidity estimate
            dvol[sym] = np.nan; continue
        dvol[sym] = float((f["total_volume"].to_numpy()[m] * f["last_price"].to_numpy()[m]).sum())
    return dvol

def main():
    full = pd.read_parquet(FULLFLOW); v0 = pd.read_parquet(V0FULL)
    for d in (full, v0):
        d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    oos_syms = sorted(set(full[full.open_time >= OOS0]["symbol"].unique()))
    print(f"OOS syms with preds: {len(oos_syms)}")

    dvol = trailing_dvol_at(OOS0)
    ranked = sorted([s for s in oos_syms if np.isfinite(dvol.get(s, np.nan))],
                    key=lambda s: -dvol[s])
    print(f"ranked-by-liquidity syms: {len(ranked)} (top 8: {ranked[:8]})")
    print(f"bottom 8: {ranked[-8:]}")
    json.dump({s: dvol[s] for s in ranked}, open(OUT/"dvol_rank.json", "w"))

    def write_split(bookA_syms, tag):
        bookA_syms = set(bookA_syms)
        bookB_syms = [s for s in ranked if s not in bookA_syms]
        a = full[full["symbol"].isin(bookA_syms)].copy()
        b = v0[v0["symbol"].isin(bookB_syms)].copy()
        a.to_parquet(OUT/f"bookA_{tag}.parquet"); b.to_parquet(OUT/f"bookB_{tag}.parquet")
        return len(bookA_syms), len(bookB_syms)

    manifest = {}
    for N in N_GRID:
        na, nb = write_split(ranked[:N], f"liq{N}")
        manifest[f"liq{N}"] = dict(bookA=na, bookB=nb, kind="liquidity")
        print(f"  liq{N}: BookA(flow)={na} BookB(price)={nb}")
    # placebo: random splits of size PLACEBO_N (use index-based RNG, no Math.random pathology)
    rng = np.random.default_rng(12345)
    for seed in range(PLACEBO_SEEDS):
        perm = list(rng.permutation(ranked))
        na, nb = write_split(perm[:PLACEBO_N], f"rand{seed}")
        manifest[f"rand{seed}"] = dict(bookA=na, bookB=nb, kind="placebo")
    json.dump(manifest, open(OUT/"manifest.json", "w"))
    print(f"\nwrote {len(manifest)} split pred-file pairs to {OUT}")
    print("DONE")

if __name__ == "__main__":
    main()
