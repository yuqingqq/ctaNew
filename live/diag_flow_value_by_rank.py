"""ROOT-CAUSE of the N=65 split symptom: WHY do names beyond ~rank-65-by-rvol hurt the flow book?
Hypothesis: flow features (VPIN/TFI) only carry signal where trading activity is high enough to measure
them; their per-symbol IC + coverage decay with rank, so applying them to marginal names injects noise.
If true, the real criterion is per-name FLOW RELIABILITY, not a hard rvol cutoff. We also test which
continuous variable (rvol / atr / idio_vol / flow-coverage) best predicts flow-feature value -> the
principled split axis.  Read-only.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.train_twobook_models as tt
V0 = tt.V0
FLOW_KEY = ["fl_vpin","fl_vpin_1d","fl_tfi","fl_tfi_1d","fl_bs_imb","fl_kyle"]  # core flow signals
OOS = pd.Timestamp("2025-10-04", tz="UTC")

F, flowcols = tt.build_flow()
flowk = [c for c in FLOW_KEY if c in flowcols]
PAN = pd.read_parquet(tt.PANEL, columns=["symbol","open_time","return_pct","rvol_7d","atr_pct","idio_vol_to_btc_1h"])
PAN["open_time"] = pd.to_datetime(PAN["open_time"], utc=True)
PAN = PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].merge(F, on=["symbol","open_time"], how="left")
PAN = PAN[PAN.open_time >= OOS]
g = PAN.groupby("open_time"); PAN["fwd"] = PAN["return_pct"] - g["return_pct"].transform("mean")
# cross-sectional rvol rank per cycle -> per-symbol mean rank (the split axis)
PAN["rvol_rank"] = g["rvol_7d"].rank(ascending=False)
nsym = g["symbol"].transform("size")

rows = []
for sym, gg in PAN.groupby("symbol"):
    if len(gg) < 100: continue
    rec = {"symbol": sym, "n": len(gg), "mean_rvol_rank": gg["rvol_rank"].mean(),
           "rvol": gg["rvol_7d"].mean(), "atr": gg["atr_pct"].mean(), "idio": gg["idio_vol_to_btc_1h"].mean()}
    # flow-feature coverage + per-symbol IC vs forward demeaned return
    covs, ics = [], []
    for c in flowk:
        cov = gg[c].notna().mean(); covs.append(cov)
        sub = gg[[c,"fwd"]].dropna()
        if len(sub) > 50 and sub[c].std() > 0:
            ics.append(abs(sub[c].corr(sub["fwd"], method="spearman")))
    rec["flow_cov"] = float(np.mean(covs)) if covs else np.nan
    rec["flow_absIC"] = float(np.mean(ics)) if ics else np.nan
    rows.append(rec)
R = pd.DataFrame(rows).dropna(subset=["flow_absIC"])
R["rank_bucket"] = pd.cut(R["mean_rvol_rank"], [0,65,80,110,999], labels=["top-65","65-80","80-110","110+"])

print(f"flow features analyzed: {flowk}\n")
print("=== flow-feature value + coverage by rvol-rank bucket (the split axis) ===")
agg = R.groupby("rank_bucket").agg(n_syms=("symbol","size"), mean_flow_absIC=("flow_absIC","mean"),
        mean_flow_cov=("flow_cov","mean"), mean_atr=("atr","mean"), mean_idio=("idio","mean")).round(4)
print(agg.to_string())
print("\nINTERP: if mean_flow_absIC and/or mean_flow_cov DROP sharply past top-65 -> flow signal decays with rank")
print("        -> N=65 is a PROXY for flow-feature reliability, not a magic number.\n")

print("=== which continuous var best predicts per-symbol flow value (|corr| with flow_absIC) ===")
for v in ["mean_rvol_rank","rvol","atr","idio","flow_cov"]:
    c = R[v].corr(R["flow_absIC"], method="spearman")
    print(f"  {v:16s}: spearman vs flow_absIC = {c:+.3f}")
print("  (the strongest |corr| is the PRINCIPLED split axis; if flow_cov wins, route by coverage not rvol)")

print("\n=== the marginal band (rvol-rank 65-110): what's different about these names? ===")
marg = R[(R.mean_rvol_rank>65)&(R.mean_rvol_rank<=110)].sort_values("mean_rvol_rank")
print(marg[["symbol","mean_rvol_rank","flow_cov","flow_absIC","rvol","atr"]].round(3).head(20).to_string(index=False))
print(f"\n  marginal-band flow_cov mean={marg.flow_cov.mean():.3f} vs top-65 {R[R.mean_rvol_rank<=65].flow_cov.mean():.3f} "
      f"| flow_absIC {marg.flow_absIC.mean():.4f} vs {R[R.mean_rvol_rank<=65].flow_absIC.mean():.4f}")
R.to_csv(REPO/"live/state/opt_loop/flow_value_by_rank.csv", index=False)
