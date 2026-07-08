"""8h optimization loop — goal: a better, ROBUST 2025 Sharpe on the convexity strategy.

Anti-overfit guardrail: 2025 is one thin-edge window; optimizing directly for it = overfit. So a lever is
ACCEPTED only if it improves via a generalizable mechanism — validated across ALL folds + matched placebo +
nested-OOS — and we report its 2025 effect as a consequence. A lever that helps only 2025 is REJECTED.

Fast path: every candidate is an env-override REPLAY against the canonical full-history preds (175-univ
fullhist_mpit, dense +1.33 / 2025 +0.38). No pred regen. Each replay ~2-4 min. Records per-year + per-fold
Sharpe (fold = the monthly walk-forward fold of the preds) so a post-step can do nested-OOS + placebo.

Phases (env-only, no bot code change): P1 DISP_GATE sweep, P3 L/S K-asymmetry, P4 SHORT_FUND_FLOOR.
(P2 pred-floor needs a gated bot edit — added in a later wave.)  Writes ledger incrementally.
"""
import sys, os, time, subprocess, json
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
PY=sys.executable
OUT=REPO/"live/state/v3loop/opt2025"; OUT.mkdir(parents=True,exist_ok=True)
LEDGER=OUT/"ledger.csv"
PREDS=REPO/"live/state/v3loop/fullhist_mpit/base.parquet"
PREDS_LONG=REPO/"live/state/v3loop/fullhist_mpit/long.parquet"
META=REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
ANN=np.sqrt(365)
t0=time.time()

PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")

# open_time -> fold map (from the preds folds = monthly walk-forward windows)
_f=pd.read_parquet(PREDS,columns=["open_time","fold"]); _f["open_time"]=pd.to_datetime(_f["open_time"],utc=True)
OT2FOLD=_f.groupby("open_time")["fold"].first()

def dsh(s):
    d=(s.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
def maxdd(s):
    eq=s.fillna(0).cumsum(); return float((eq-eq.cummax()).min())

def run_cfg(tag, overrides):
    sd=OUT/tag; sd.mkdir(exist_ok=True)
    env=dict(os.environ); env.update(PROD); env.update(overrides)
    env.update(PYTHONPATH=str(REPO),CONVEXITY_STATE=str(sd),CONVEXITY_PREDS_PATH=str(PREDS),
               CONVEXITY_PREDS_LONG=str(PREDS_LONG),CONVEXITY_UNIVERSE_META=str(META))
    r=subprocess.run([PY,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=str(REPO),
                     stdout=open(sd/"run.log","w"),stderr=subprocess.STDOUT)
    c=pd.read_csv(sd/"cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
    c=c.sort_values("open_time").set_index("open_time")
    dense=c.loc['2025-01-01':'2026-06-04','pnl_bps']; y25=c.loc['2025-01-01':'2025-12-31','pnl_bps']
    rec=dict(tag=tag,dense=round(dsh(dense),3),s2025=round(dsh(y25),3),
             maxDD25=round(maxdd(dense),0),pnl25=round(y25.sum(),0),rc=r.returncode)
    for yr,g in c.groupby(c.index.year): rec[f"y{yr}"]=round(dsh(g["pnl_bps"]),3)
    # per-fold daily-pnl saved for nested-OOS/placebo post-step
    c["fold"]=c.index.map(OT2FOLD)
    c[["pnl_bps","fold"]].to_csv(sd/"cyc_fold.csv")
    return rec

def log_rec(phase,rec):
    rec=dict(phase=phase,**rec,elapsed=round(time.time()-t0))
    df=pd.DataFrame([rec])
    hdr=not LEDGER.exists()
    df.to_csv(LEDGER,mode="a",header=hdr,index=False)
    yrs=" ".join(f"{k}={rec[k]:+.2f}" for k in rec if k.startswith("y2"))
    print(f"[{phase}] {rec['tag']:28s} dense {rec['dense']:+.3f}  2025 {rec['s2025']:+.3f}  maxDD25 {rec['maxDD25']:+.0f}  | {yrs}  [{rec['elapsed']}s]",flush=True)
    return rec

# ============================== PLAN ==============================
def _main():
  PLAN=[]
  # baseline (reuse production default)
  PLAN.append(("P0","baseline",{}))
  # P1: DISP_GATE dispersion de-gross — pctile x lookback
  for pct in ["0.20","0.30","0.40","0.50"]:
    for lb in ["252","504"]:
        PLAN.append(("P1",f"disp_p{pct}_lb{lb}",{"DISP_GATE":"1","DISP_GATE_PCTILE":pct,"DISP_GATE_LOOKBACK":lb}))
  # P3: L/S asymmetry — short-heavier (short=alpha) and long down-weight via K
  for ks,kl in [("3","3"),("4","2"),("5","2"),("4","3"),("3","2"),("5","3")]:
    PLAN.append(("P3",f"ks{ks}_kl{kl}",{"STRAT_K_SHORT":ks,"STRAT_K_LONG":kl}))
  # P4: SHORT_FUND_FLOOR — drop shorts with PIT funding below floor (collect carry / avoid pay)
  for fl in ["-5","-2","0","2","5"]:
    PLAN.append(("P4",f"fundfloor_{fl}",{"SHORT_FUND_FLOOR":fl}))

  print(f"PLAN: {len(PLAN)} configs",flush=True)
  done=set()
  if LEDGER.exists(): done=set(pd.read_csv(LEDGER)["tag"])
  for phase,tag,ov in PLAN:
    if tag in done: print(f"skip {tag} (done)",flush=True); continue
    try:
        log_rec(phase,run_cfg(tag,ov))
    except Exception as e:
        print(f"ERR {tag}: {e}",flush=True)
  print("DONE phase_2025_opt wave-1",flush=True)

if __name__=="__main__":
    _main()
