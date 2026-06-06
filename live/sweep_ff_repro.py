"""Full-history reproduction sweep: confirm the live bot reproduces the gen records when fed the same
forward-filled funding (+ the healed 174-cohort xs_rank).

For each of the 41 record cycles: build the bot's bar (predict_at_close), substitute the gen's ff funding
(MY_funding_panel_94syms), score both books with the frozen 5.29 models, and compare pred_long/pred_short
to the gen's (MY_PANEL_features_ALL94) for all 94 low-vol symbols. The records are selection(gen preds),
so bot preds == gen preds  <=>  bot picks == records picks. Read-only; touches no live state.
"""
from __future__ import annotations
import sys, json, pickle, warnings
from pathlib import Path
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
import live.predict_at_close as pac, live.decide_v1 as d

FCOLS = ["funding_rate", "funding_rate_z_7d", "funding_rate_1d_change"]
recs = pd.read_csv(REPO / "docs/recent_cycles_v2.csv"); recs["open_time"] = pd.to_datetime(recs["open_time"], utc=True)
boundaries = sorted(recs["open_time"].unique())
gen = pd.read_parquet(REPO / "docs/xref/MY_PANEL_features_ALL94_0529_0604.parquet"); gen["open_time"] = pd.to_datetime(gen["open_time"], utc=True)
ff = pd.read_parquet(REPO / "docs/xref/MY_funding_panel_94syms.parquet"); ff["open_time"] = pd.to_datetime(ff["open_time"], utc=True)
longm = pickle.load(open(REPO / "live/models/convexity_v1_long_model.pkl", "rb"))["models"]
shortm = pickle.load(open(REPO / "live/models/convexity_v1_short_model.pkl", "rb"))["models"]
excl = set(json.load(open(d.UNIV))["exclude_high_vol"])


def topk(df, col, k):
    return set(df.sort_values(col, ascending=False).head(k)["symbol"])


rows = []
for B in boundaries:
    try:
        bar = pac.build_bar(B, drop_unlabeled=False)
    except Exception as e:
        print(f"  {B} build FAIL {str(e)[:50]}", flush=True); continue
    if bar is None or not len(bar):
        print(f"  {B} no bar", flush=True); continue
    cur = d._with_residrev(bar, B)
    gff = ff[ff["open_time"] == B].set_index("symbol")
    for c in FCOLS:
        cur[c] = cur["symbol"].map(gff[c]).fillna(cur[c])
    lp = d._score(cur, longm); sp = d._score(cur, shortm)
    pcl = [c for c in lp.columns if "pred" in c.lower()][0]
    pcs = [c for c in sp.columns if "pred" in c.lower()][0]
    lp = lp[~lp["symbol"].isin(excl)][["symbol", pcl]].rename(columns={pcl: "bot_long"})
    sp = sp[~sp["symbol"].isin(excl)][["symbol", pcs]].rename(columns={pcs: "bot_short"})
    gB = gen[gen["open_time"] == B][["symbol", "pred_long", "pred_short"]]
    m = gB.merge(lp, on="symbol").merge(sp, on="symbol")
    if not len(m):
        print(f"  {B} no overlap", flush=True); continue
    dl = (m["bot_long"] - m["pred_long"]).abs(); ds = (m["bot_short"] - m["pred_short"]).abs()
    # pick reproduction: do the bot's top-3 / bot-3 match the gen's? (selection-agnostic upper bound)
    long_match = all(topk(m.rename(columns={"bot_long": "x"}), "x", 3) == topk(m.rename(columns={"pred_long": "x"}), "x", 3)
                     for _ in [0])
    short_match = topk(m.assign(x=-m.bot_short), "x", 3) == topk(m.assign(x=-m.pred_short), "x", 3)
    rows.append({"B": str(B)[:16], "n": len(m), "max_dL": dl.max(), "max_dS": ds.max(),
                 "top3L_match": long_match, "bot3S_match": short_match})
    print(f"  {str(B)[:16]}  n={len(m):2d}  max|dLong|={dl.max():.2e}  max|dShort|={ds.max():.2e}  "
          f"top3L={'OK' if long_match else 'DIFF'}  bot3S={'OK' if short_match else 'DIFF'}", flush=True)

res = pd.DataFrame(rows)
print("\n===== SWEEP SUMMARY =====")
print(f"cycles checked: {len(res)} / {len(boundaries)}")
print(f"overall max |pred_long diff|:  {res['max_dL'].max():.3e}")
print(f"overall max |pred_short diff|: {res['max_dS'].max():.3e}")
print(f"cycles with top-3 long ranking == gen:  {res['top3L_match'].sum()}/{len(res)}")
print(f"cycles with bot-3 short ranking == gen: {res['bot3S_match'].sum()}/{len(res)}")
ok = res["max_dL"].max() < 1e-2 and res["max_dS"].max() < 1e-2
print(f"\nREPRODUCED (preds match gen within 1e-2 on all {len(res)} cycles): {ok}")
