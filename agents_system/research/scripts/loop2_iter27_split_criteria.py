"""LOOP2 iter-27 — additional split criteria (pre-registered) vs the Phase-VIII placebo.
Criteria (rank eligible syms, top-N -> flow BookA, rest -> price BookB), PIT as-of OOS start:
  lts  : large-trade activity (trailing mean large_trade_volume/total_volume)   [high->flow]
  lowbeta/highbeta : corr_to_btc_1d                                              [both dirs]
  lowvol/highvol   : rvol_7d                                                     [both dirs]
  comp : composite = mean(liq-rank, flowquality-rank)                            [best-of-both]
Generated at N in {70,90} (where placebo distributions exist in split2/). Compared to rand{70,90}_*.
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path("/home/yuqing/ctaNew")
FULLFLOW = REPO/"live/state/convexity/unified_fullflow_preds.parquet"
V0FULL   = REPO/"live/state/convexity/unified_v0full_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT = REPO/"live/state/convexity/split2"   # shared dir (reuse placebo)
OOS0 = pd.Timestamp("2025-10-04", tz="UTC")
NS = [70, 90]

def trailing_flow(asof, lookback_days=30):
    lo = asof - pd.Timedelta(days=lookback_days); lts = {}
    for fp in sorted(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet"))):
        sym = Path(fp).stem.replace("flow_", "")
        try: f = pd.read_parquet(fp, columns=["total_volume","large_trade_volume"])
        except: continue
        idx = pd.to_datetime(f.index, utc=True); m = (idx >= lo) & (idx < asof)
        if m.sum() < 50: lts[sym] = np.nan; continue
        tv = f["total_volume"].to_numpy()[m]; lv = f["large_trade_volume"].to_numpy()[m]
        denom = tv.sum()
        lts[sym] = float(lv.sum()/denom) if denom > 0 else np.nan
    return lts

def trailing_panel_feat(col, asof, lookback_days=30):
    lo = asof - pd.Timedelta(days=lookback_days)
    p = pd.read_parquet(PANEL, columns=["symbol","open_time",col])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p.open_time >= lo) & (p.open_time < asof)]
    return p.groupby("symbol")[col].mean().to_dict()

def main():
    print("START iter27", flush=True)
    full = pd.read_parquet(FULLFLOW); v0 = pd.read_parquet(V0FULL)
    for d in (full, v0): d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    oos = sorted(set(full[full.open_time >= OOS0]["symbol"].unique()))
    # liquidity + flow-quality ranks (for composite) — reuse ranks.json from iter26
    ranks = json.load(open(OUT/"ranks.json"))
    liq_rank = ranks["liq_rank"]; fq_rank = ranks["fq_rank"]
    liq_pos = {s: i for i, s in enumerate(liq_rank)}; fq_pos = {s: i for i, s in enumerate(fq_rank)}
    print("computing criteria...", flush=True)
    lts = trailing_flow(OOS0)
    corr = trailing_panel_feat("corr_to_btc_1d", OOS0)
    rvol = trailing_panel_feat("rvol_7d", OOS0)
    def rank_by(d, reverse=True):
        return sorted([s for s in oos if np.isfinite(d.get(s, np.nan))], key=lambda s: -d[s] if reverse else d[s])
    crits = {
        "lts":     rank_by(lts, True),                                  # high large-trade -> flow
        "highbeta":rank_by(corr, True),                                 # high corr_to_btc -> flow
        "lowbeta": rank_by(corr, False),                                # low corr (idiosyncratic) -> flow
        "highvol": rank_by(rvol, True),
        "lowvol":  rank_by(rvol, False),
        "comp":    sorted([s for s in oos if s in liq_pos and s in fq_pos],
                          key=lambda s: liq_pos[s] + fq_pos[s]),         # best avg of liq+flowquality
    }
    manifest = json.load(open(OUT/"manifest.json"))
    for cname, ranked in crits.items():
        for N in NS:
            A = set(ranked[:N]); B = [s for s in liq_rank if s not in A]
            tag = f"{cname}{N}"
            full[full["symbol"].isin(A)].to_parquet(OUT/f"bookA_{tag}.parquet")
            v0[v0["symbol"].isin(set(B))].to_parquet(OUT/f"bookB_{tag}.parquet")
            manifest[tag] = dict(bookA=len(A), bookB=len(B), crit=cname)
    json.dump(manifest, open(OUT/"manifest.json", "w"))
    new = [f"{c}{N}" for c in crits for N in NS]
    json.dump(new, open(OUT/"iter27_tags.json", "w"))
    print(f"wrote {len(new)} new criterion splits: {new}", flush=True)
    print("GEN DONE")

if __name__ == "__main__": main()
