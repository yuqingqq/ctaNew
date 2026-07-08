"""Fixed harness for the vanilla-v4 tuning-validation loop. Runs VANILLA v4 (raw two-book, gates OFF) + a set of
env OVERRIDES through the real bot (fee+funding+slippage) on BOTH windows (OOS 2023-25 + in-sample 2025-10+),
returns net metrics. Agents propose {name, env:{overrides}}; the protocol is locked here.
Usage: python3 live/tuning_harness.py <config.json> <out.json>
config.json = {"name": str, "env": {"REGIME_GATE":"1", ...}}   (env values are strings)
VANILLA = gates all OFF, K_long=1/K_short=2, inv-vol sizing, residual target.
"""
import sys, json, subprocess, os
from pathlib import Path
import pandas as pd, numpy as np
R="/home/yuqing/ctaNew"
cfg=json.load(open(sys.argv[1])); OUT=sys.argv[2]
name=cfg.get("name","cfg"); ov=cfg.get("env",{})
EXEC={"COST_BPS_LEG":"9","FEE_BPS_FILL":"4.5","CHARGE_FUNDING":"1","CONVEXITY_PIT_DVOL":"1","XS_LEAN":"1",
      "CONVEXITY_UNIVERSE_META":"outputs/vBTC_features/panel_expanded_v0.parquet",
      "DEPTH_COST_CSV":"live/state/v3loop/persym_cost_cal.csv","DEPTH_COST_TIER":"cost_10k",
      "CONVEXITY_DVOL_CACHE_PKL":"live/state/v3loop/ddloop/_dvol_cache.pkl"}
VANILLA={"STRAT_K":"2","STRAT_K_LONG":"1","BEAR_K":"0","SIDE_MODE":"default","SIZING_MODE":"inv_sqrt_vol",
         "REGIME_GATE":"0","BEAR_DEPTH_RAMP":"0","BULL_MODE":"default","CONC_CAP":"0.99","CONC_CAP_SINGLE_EXEMPT":"1",
         "STOP_SKIP_REGIMES":"side,bear,bull","SHORT_MIN_RET3D":"-999","LONG_MAX_RET3D":"999"}
WIN={"oos":("hl_v4base_oos","hl_v4long_oos"),"ins":("hl_tgt_res_base","hl_tgt_res_long")}
def run(win):
    pb,pl=WIN[win]; sd=f"{R}/live/state/longtail/tune/{name}/{win}"; Path(sd).mkdir(parents=True,exist_ok=True)
    env=dict(os.environ); env.update(EXEC); env.update(VANILLA); env.update({k:str(v) for k,v in ov.items()})
    env["CONVEXITY_PREDS_PATH"]=f"{R}/live/state/convexity/{pb}/v0full_hl60.parquet"
    env["CONVEXITY_PREDS_LONG"]=f"{R}/live/state/convexity/{pl}/v0full_hl60.parquet"
    env["CONVEXITY_STATE"]=sd; env["PYTHONPATH"]=R
    with open(f"{sd}/run.log","w") as lg:
        subprocess.run([sys.executable,"-m","live.convexity_paper_bot","--replay-all"],cwd=R,env=env,stdout=lg,stderr=subprocess.STDOUT,timeout=400)
    s=json.load(open(f"{sd}/replay_summary.json"))
    c=pd.read_csv(f"{sd}/cycles.csv")
    pc="pnl_bps" if "pnl_bps" in c.columns else [x for x in c.columns if x.lower()=="pnl"][0]
    reg={r:round(float(v)) for r,v in c.groupby("regime")[pc].sum().items()}
    c["open_time"]=pd.to_datetime(c["open_time"],utc=True); c["yr"]=c["open_time"].dt.year
    yr={int(k):round(float(v)) for k,v in c.groupby("yr")[pc].sum().items()}
    return {"sharpe":round(float(s["Sharpe_ann"]),3),"pnl":round(float(s["totPnL_bps"])),
            "maxdd":round(float(s["maxDD_bps"])),"stop_pct":round(float(s.get("stop_engaged_pct",0)),0),
            "per_regime":reg,"per_year":yr}
res={"name":name,"env":ov}
for w in ["oos","ins"]:
    try: res[w]=run(w)
    except Exception as e: res[w]={"error":str(e)[:200]}
json.dump(res,open(OUT,"w"),indent=1)
o=res.get("oos",{}); i=res.get("ins",{})
print(f"[{name}] OOS Sh {o.get('sharpe','ERR')} pnl {o.get('pnl','?')} | INS Sh {i.get('sharpe','ERR')} pnl {i.get('pnl','?')}")
