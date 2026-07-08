"""Analyze the opt2025 sweep: per-fold robustness + nested-OOS param selection + honest verdict.

Anti-overfit gates a lever must pass to be ACCEPTED:
  (1) 2025 Sharpe lift > 0 AND all-period dense not hurt (generalizes, not 2025-only);
  (2) folds_beat_baseline majority (>= half of folds improve) — not 1-2 lucky folds;
  (3) NESTED-OOS: picking the param from past folds and applying to the next fold must STILL beat baseline
      (tuned continuous params usually fail this — the K3/decay lesson).
Reads live/state/v3loop/opt2025/<tag>/cyc_fold.csv (pnl_bps + fold per cycle).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
REPO=Path("/home/yuqing/ctaNew"); OUT=REPO/"live/state/v3loop/opt2025"
ANN=np.sqrt(365)

def load(tag):
    f=OUT/tag/"cyc_fold.csv"
    if not f.exists(): return None
    c=pd.read_csv(f); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
    return c.sort_values("open_time").set_index("open_time")
def dsh(s):
    d=(s.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
def fold_pnl(c):  # total bps per fold
    return c.groupby("fold")["pnl_bps"].sum()

def compare(base_tag, cand_tags):
    base=load(base_tag);
    if base is None: print("no baseline yet"); return
    bf=fold_pnl(base); b25=dsh(base.loc['2025-01-01':'2025-12-31','pnl_bps']); bdense=dsh(base.loc['2025-01-01':'2026-06-04','pnl_bps'])
    print(f"BASELINE {base_tag}: 2025 {b25:+.3f}  dense {bdense:+.3f}")
    rows=[]
    for t in cand_tags:
        c=load(t)
        if c is None: continue
        cf=fold_pnl(c)
        common=bf.index.intersection(cf.index)
        beat=int((cf[common]>bf[common]).sum()); nf=len(common)
        rows.append(dict(tag=t,s2025=round(dsh(c.loc['2025-01-01':'2025-12-31','pnl_bps']),3),
                         dense=round(dsh(c.loc['2025-01-01':'2026-06-04','pnl_bps']),3),
                         folds_beat=f"{beat}/{nf}",d_pnl25=round(c.loc['2025-01-01':'2025-12-31','pnl_bps'].sum()-base.loc['2025-01-01':'2025-12-31','pnl_bps'].sum(),0)))
    df=pd.DataFrame(rows).sort_values("s2025",ascending=False)
    print(df.to_string(index=False))
    return df

def nested_oos(base_tag, cand_tags):
    """Per fold t: pick the cand (incl baseline) with best cumulative pnl over folds<t, take its fold-t pnl.
    Reports nested daily Sharpe vs always-baseline."""
    series={t:load(t) for t in [base_tag]+cand_tags}; series={k:v for k,v in series.items() if v is not None}
    if base_tag not in series: print("no baseline"); return
    folds=sorted(series[base_tag]["fold"].dropna().unique())
    fp={t:fold_pnl(c) for t,c in series.items()}
    nested_pick=[]
    for i,f in enumerate(folds):
        if i==0: pick=base_tag
        else:
            past=folds[:i]; cum={t:fp[t].reindex(past).fillna(0).sum() for t in series}
            pick=max(cum,key=cum.get)
        nested_pick.append((f,pick))
    # build nested daily pnl by concatenating each fold's chosen-config cycles
    parts=[]
    for f,pick in nested_pick:
        c=series[pick]; parts.append(c[c["fold"]==f]["pnl_bps"])
    nested=pd.concat(parts).sort_index()
    n25=nested.loc['2025-01-01':'2025-12-31']; ndense=nested.loc['2025-01-01':'2026-06-04']
    base=series[base_tag]
    print(f"\nNESTED-OOS over {len(cand_tags)} cands: 2025 {dsh(n25):+.3f} (base {dsh(base.loc['2025-01-01':'2025-12-31','pnl_bps']):+.3f})  "
          f"dense {dsh(ndense):+.3f} (base {dsh(base.loc['2025-01-01':'2026-06-04','pnl_bps']):+.3f})")
    from collections import Counter
    print("  picks:",dict(Counter(p for _,p in nested_pick)))

if __name__=="__main__":
    led=pd.read_csv(OUT/"ledger.csv") if (OUT/"ledger.csv").exists() else pd.DataFrame()
    if len(led):
        print("=== LEDGER ==="); print(led[[c for c in ["phase","tag","dense","s2025","maxDD25","y2025","y2026","y2024","y2023","y2022"] if c in led.columns]].to_string(index=False))
    base="baseline"
    for ph in ["P1","P3","P4"]:
        tags=[t for t in led[led.phase==ph]["tag"]] if len(led) else []
        if tags:
            print(f"\n========== {ph} ==========")
            compare(base,tags); nested_oos(base,tags)
