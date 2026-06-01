"""LOOP2 iter-26 — THOROUGH two-book split study.

Generates split pred-file pairs (BookA=flow/unified_fullflow, BookB=price/unified_v0full) for:
  (a) fine liquidity-N curve: N in {20..140}, ranked by trailing-30d $vol as-of OOS start (PIT/ex-ante)
  (b) placebo distributions: 20 random splits each at N in {50,70,90}  -> size-matched null
  (c) per-fold PIT liquidity re-ranking at N in {70,85} (rank each fold, not static)
  (d) alternative criterion: flow data-quality (trailing non-zero-flow-bar fraction) top-N at {70,85}
A parallel runner replays each pair (K=3) + combines; an analyzer ranks liq vs placebo.
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
FULLFLOW = REPO/"live/state/convexity/unified_fullflow_preds.parquet"
V0FULL   = REPO/"live/state/convexity/unified_v0full_preds.parquet"
OUT = REPO/"live/state/convexity/split2"; OUT.mkdir(parents=True, exist_ok=True)
OOS0 = pd.Timestamp("2025-10-04", tz="UTC")
CUTS = [pd.Timestamp(t, tz="UTC") for t in
        ["2025-10-04","2025-11-01","2025-12-01","2026-01-01","2026-02-01","2026-03-01","2026-04-01","2026-05-01","2026-05-27"]]
N_CURVE = [20,30,40,50,60,70,80,90,100,110,120,140]
PLACEBO_NS = [50,70,90]; PLACEBO_SEEDS = 20
PITLIQ_NS = [70,85]; FQ_NS = [70,85]

def flow_metrics_multi(asofs, lookback_days=30):
    """ONE pass over flow files: returns {asof: (dvol_dict, fq_dict)} for all as-of dates."""
    res = {a: ({}, {}) for a in asofs}
    files = sorted(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet")))
    print(f"  flow_metrics_multi: {len(files)} files × {len(asofs)} as-of dates (single pass)", flush=True)
    for j, fp in enumerate(files):
        sym = Path(fp).stem.replace("flow_", "")
        try:
            f = pd.read_parquet(fp, columns=["total_volume", "last_price"])
        except Exception:
            continue
        idx = pd.to_datetime(f.index, utc=True)
        tv = f["total_volume"].to_numpy(); dv = tv * f["last_price"].to_numpy()
        for a in asofs:
            lo = a - pd.Timedelta(days=lookback_days)
            m = (idx >= lo) & (idx < a)
            dd, ff = res[a]
            if m.sum() < 50: dd[sym] = np.nan; ff[sym] = np.nan
            else: dd[sym] = float(dv[m].sum()); ff[sym] = float((tv[m] > 0).mean())
        if (j+1) % 50 == 0: print(f"    {j+1}/{len(files)}", flush=True)
    return res

def main():
    print("START", flush=True)
    full = pd.read_parquet(FULLFLOW); v0 = pd.read_parquet(V0FULL)
    for d in (full, v0): d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    oos_syms = sorted(set(full[full.open_time >= OOS0]["symbol"].unique()))
    print(f"loaded preds; OOS syms {len(oos_syms)}", flush=True)

    asofs = [OOS0] + [CUTS[i] for i in range(len(CUTS)-1)]
    metr = flow_metrics_multi(asofs)
    dvol0, fq0 = metr[OOS0]
    ranked = sorted([s for s in oos_syms if np.isfinite(dvol0.get(s, np.nan))], key=lambda s: -dvol0[s])
    fq_ranked = sorted([s for s in oos_syms if np.isfinite(fq0.get(s, np.nan))], key=lambda s: -fq0[s])
    print(f"ranked {len(ranked)}; top8 {ranked[:8]}", flush=True)

    fold_rank = {}
    for i in range(len(CUTS)-1):
        dv, _ = metr[CUTS[i]]
        fold_rank[i] = sorted([s for s in oos_syms if np.isfinite(dv.get(s, np.nan))], key=lambda s: -dv[s])
    def fold_of(ts):
        for i in range(len(CUTS)-1):
            if CUTS[i] <= ts < CUTS[i+1]: return i
        return len(CUTS)-2

    manifest = {}
    def write_static(bookA_syms, tag):
        A = set(bookA_syms); B = [s for s in ranked if s not in A]
        full[full["symbol"].isin(A)].to_parquet(OUT/f"bookA_{tag}.parquet")
        v0[v0["symbol"].isin(set(B))].to_parquet(OUT/f"bookB_{tag}.parquet")
        manifest[tag] = dict(bookA=len(A), bookB=len(B))

    # (a) liquidity curve
    for N in N_CURVE: write_static(ranked[:N], f"liq{N}")
    # (b) placebo
    rng = np.random.default_rng(20260531)
    for N in PLACEBO_NS:
        for s in range(PLACEBO_SEEDS):
            write_static(list(rng.permutation(ranked))[:N], f"rand{N}_{s}")
    # (d) flow-quality criterion
    for N in FQ_NS: write_static(fq_ranked[:N], f"fq{N}")
    # (c) per-fold PIT liquidity: BookA = rows where sym is top-N IN ITS OWN FOLD (vectorized)
    cut_edges = pd.DatetimeIndex(CUTS)
    full_fold = pd.cut(full["open_time"], bins=cut_edges, labels=False, right=False)
    v0_fold = pd.cut(v0["open_time"], bins=cut_edges, labels=False, right=False)
    for N in PITLIQ_NS:
        # membership key = (fold, symbol)
        memb = {(i, s) for i in fold_rank for s in fold_rank[i][:N]}
        keyA = list(zip(full_fold.fillna(-1).astype(int), full["symbol"]))
        ina = pd.Series([k in memb for k in keyA], index=full.index)
        full[ina].to_parquet(OUT/f"bookA_pitliq{N}.parquet")
        keyB = list(zip(v0_fold.fillna(-1).astype(int), v0["symbol"]))
        inb = pd.Series([k not in memb for k in keyB], index=v0.index)
        v0[inb].to_parquet(OUT/f"bookB_pitliq{N}.parquet")
        manifest[f"pitliq{N}"] = dict(bookA=int(ina.sum()), bookB=int(inb.sum()), perfold=True)

    json.dump(manifest, open(OUT/"manifest.json", "w"))
    json.dump({"liq_rank": ranked, "fq_rank": fq_ranked}, open(OUT/"ranks.json", "w"))
    print(f"wrote {len(manifest)} split pairs -> {OUT}")
    print("GEN DONE")

if __name__ == "__main__": main()
