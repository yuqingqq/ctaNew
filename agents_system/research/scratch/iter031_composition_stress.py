"""iter-031 Step 3 standalone — composition stress (random subset draws), with klines CACHED
across draws to avoid re-reading per draw. Builds the champion base book + stop on random subsets.
"""
from __future__ import annotations
import time, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import importlib.util as _ilu

REPO=Path("/home/yuqing/ctaNew")
SCRIPTS=REPO/"research/convexity_portable_2026-05-20/scripts"
OUT=REPO/"outputs/iter031"; OUT.mkdir(parents=True, exist_ok=True)
_s123=_ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py"); x123=_ilu.module_from_spec(_s123); _s123.loader.exec_module(x123)
_s125=_ilu.spec_from_file_location("x125", SCRIPTS/"X125_volnorm_stop.py"); x125=_ilu.module_from_spec(_s125); _s125.loader.exec_module(x125)

# monkeypatch load_close with an in-memory cache so repeated builds are fast
_CLOSE_CACHE={}
_orig_load=x123.load_close
def cached_load(sym):
    if sym not in _CLOSE_CACHE:
        _CLOSE_CACHE[sym]=_orig_load(sym)
    return _CLOSE_CACHE[sym]
x123.load_close=cached_load

build_universe=x123.build_universe; gross_unit=x125.gross_unit; metrics=x125.metrics
run_volnorm=x125.run_volnorm_heldbook; PRIMARY_COST=x125.PRIMARY_COST; REC_K=x125.REC_K
HL70_PREDS=x123.HL70_PREDS; EXT_PREDS=x123.EXT_PREDS
SEED=12345; N_DRAWS=25

_tmp=[]
def build_subset(preds_path, label, syms):
    cols=["symbol","open_time","pred","return_pct","fold"]
    d=pd.read_parquet(preds_path, columns=cols); d=d[d["symbol"].isin(syms)].copy()
    tf=tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/tmp"); d.to_parquet(tf.name, index=False); _tmp.append(tf.name)
    return build_universe(Path(tf.name), label)

def champ(U):
    base=gross_unit(U["cyc"]["base"], U["rs"], PRIMARY_COST)*1e4; bm=metrics(base)
    pnl,g,st,rt,_=run_volnorm(U["cyc"]["base"], U["rs"], PRIMARY_COST, REC_K); sm=metrics(pnl*1e4)
    return bm, sm

def main():
    t0=time.time(); rng=np.random.default_rng(SEED)
    hl70=sorted(pd.read_parquet(HL70_PREDS, columns=["symbol"])["symbol"].unique())
    ext=sorted(pd.read_parquet(EXT_PREDS, columns=["symbol"])["symbol"].unique())
    rows=[]; summ=[]
    for univ, preds, full, Ns in [("HL70", HL70_PREDS, hl70, [50, 40, 60]),
                                   ("EXT",  EXT_PREDS, ext, [15, 18])]:
        for N in Ns:
            print(f"\n[{univ}] random {N}-of-{len(full)} ({N_DRAWS} draws, base+stop @4.5bps)", flush=True)
            sh=[]; dd=[]; cal=[]; ssh=[]; sdd=[]; scal=[]
            for i in range(N_DRAWS):
                pick=sorted(rng.choice(full, size=N, replace=False).tolist())
                U=build_subset(preds, f"{univ}_r{N}_{i}", pick)
                bm,sm=champ(U)
                sh.append(bm["Sharpe"]); dd.append(bm["maxDD"]); cal.append(bm["Calmar"])
                ssh.append(sm["Sharpe"]); sdd.append(sm["maxDD"]); scal.append(sm["Calmar"])
                rows.append(dict(univ=univ,N=N,draw=i,base_Sharpe=bm["Sharpe"],base_maxDD=bm["maxDD"],
                                 base_Calmar=bm["Calmar"],stop_Sharpe=sm["Sharpe"],stop_maxDD=sm["maxDD"],
                                 stop_Calmar=sm["Calmar"],syms=";".join(pick)))
                for f in _tmp:
                    try: Path(f).unlink()
                    except Exception: pass
                _tmp.clear()
            sh=np.array(sh); dd=np.array(dd); cal=np.array(cal); ssh=np.array(ssh); sdd=np.array(sdd)
            print(f"  BASE Sharpe mean {sh.mean():+.2f} std {sh.std():.2f} min {sh.min():+.2f} max {sh.max():+.2f}"
                  f" | maxDD mean {dd.mean():+.0f} worst {dd.min():+.0f} | Calmar mean {cal.mean():+.2f}", flush=True)
            print(f"  STOP Sharpe mean {ssh.mean():+.2f} std {ssh.std():.2f} min {ssh.min():+.2f} max {ssh.max():+.2f}"
                  f" | maxDD mean {sdd.mean():+.0f} worst {sdd.min():+.0f}", flush=True)
            summ.append(dict(univ=univ,N=N,n_draws=N_DRAWS,base_Sharpe_mean=float(sh.mean()),
                base_Sharpe_std=float(sh.std()),base_Sharpe_min=float(sh.min()),base_Sharpe_max=float(sh.max()),
                base_maxDD_mean=float(dd.mean()),base_maxDD_worst=float(dd.min()),base_Calmar_mean=float(cal.mean()),
                stop_Sharpe_mean=float(ssh.mean()),stop_Sharpe_std=float(ssh.std()),stop_Sharpe_min=float(ssh.min()),
                stop_maxDD_worst=float(sdd.min())))
    pd.DataFrame(rows).to_csv(OUT/"composition_stress_draws.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT/"composition_stress_summary.csv", index=False)
    print(f"\nSaved -> composition_stress_*.csv  [{time.time()-t0:.0f}s]", flush=True)

if __name__=="__main__":
    main()
