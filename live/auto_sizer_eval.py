"""Re-evaluate the AUTO_SIZER (btc_ret_30d-bucket PIT report-card) on the HONEST daily metric, with its matched
placebo (AUTO_RAND_SEED>0 = throttle RANDOM buckets, same count). Runs real + N placebo replays in parallel,
post-processes honest daily-Sharpe + tail, ranks real vs placebo. Apples-to-apples vs blunt bear bg=0.5.
"""
import os, sys, subprocess, json
import concurrent.futures as cf
import numpy as np, pandas as pd
ROOT="/home/yuqing/ctaNew"; BASE=f"{ROOT}/live/state/exp_xs94/baseline"; PY=sys.executable
PROD=dict(COST_BPS_LEG="4.5",STRAT_K="3",SIDE_MODE="default",XS_LEAN="1",CONVEXITY_PIT_DVOL="1",BEAR_MODE="equal",
          STOP_SKIP_REGIMES="bear",SIDE_BETA_NEUT="0",BEAR_K="2",SIZING_MODE="inv_vol",LONG_MAX_RET3D="0.20")
ANN=np.sqrt(365)
def run(label, ov):
    out=f"{ROOT}/live/state/v3loop/as_{label}"; os.makedirs(out,exist_ok=True)
    env=dict(os.environ); env.update(PROD); env.update(ov)
    env.update(PYTHONPATH=ROOT,CONVEXITY_STATE=out,CONVEXITY_PREDS_PATH=f"{BASE}/base_mpit.parquet",
               CONVEXITY_PREDS_LONG=f"{BASE}/long_mpit.parquet"); env.pop("CONVEXITY_UNIVERSE_META",None)
    r=subprocess.run([PY,"-m","live.convexity_paper_bot","--replay-all"],env=env,cwd=ROOT,
                     stdout=open(f"{out}/run.log","w"),stderr=subprocess.STDOUT)
    cc=f"{out}/cycles.csv"
    return label, (cc if r.returncode==0 and os.path.exists(cc) else None)
def honest(cc):
    c=pd.read_csv(cc); c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c=c.sort_values("open_time")
    s=c.set_index("open_time")["pnl_bps"].fillna(0)/1e4; d=s.resample("1D").sum()
    eq=s.cumsum(); x=c["pnl_bps"].dropna()
    return dict(dsharpe=float(d.mean()/d.std()*ANN), maxdd=float((eq-eq.cummax()).min()),
                worst1=float(x.nsmallest(max(1,len(x)//100)).sum()), totpnl=float(x.sum()))
variants=[("real",{"AUTO_SIZER":"1"})]+[(f"plac{k}",{"AUTO_SIZER":"1","AUTO_RAND_SEED":str(k)}) for k in range(1,9)]
res={}
with cf.ThreadPoolExecutor(max_workers=9) as ex:
    for label,cc in ex.map(lambda a: run(*a), variants):
        res[label]=honest(cc) if cc else None
        print(f"done {label}", flush=True)
real=res["real"]; plac=[res[f"plac{k}"] for k in range(1,9) if res.get(f"plac{k}")]
ds=np.array([p["dsharpe"] for p in plac]); w1=np.array([p["worst1"] for p in plac])
print(json.dumps(dict(
  auto_real=dict(dsharpe=round(real["dsharpe"],3),maxdd=round(real["maxdd"]),worst1=round(real["worst1"]),totpnl=round(real["totpnl"])),
  placebo_dsharpe=dict(mean=round(ds.mean(),3),min=round(ds.min(),3),max=round(ds.max(),3),
                       rank_pctile=round(100*(ds<real["dsharpe"]).mean())),
  placebo_worst1=dict(mean=round(w1.mean()),rank_pctile=round(100*(w1<real["worst1"]).mean())),
  n_placebo=len(plac)), indent=2))
