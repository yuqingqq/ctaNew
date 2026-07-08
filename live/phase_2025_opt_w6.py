"""WAVE-6: address K=2's known weakness (underperforms alt-bull 2023-H1/2024-H1) via REGIME-CONDITIONAL breadth —
wider K in bull (BULL_K), keep K=2 in side/bear. Mechanistically motivated; validated per-fold + nested-OOS (regime
timing => overfit risk, like the disp gate). Base = K=2 (STRAT_K_SHORT=2 STRAT_K_LONG=2)."""
import sys; from pathlib import Path; import pandas as pd
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.phase_2025_opt as w1
K2={"STRAT_K_SHORT":"2","STRAT_K_LONG":"2"}
PLAN=[]
# regime-conditional K: keep side/bear at 2, widen bull (where K=2 underperforms)
for bk in ["3","4","5"]:
    PLAN.append(("W6rk",f"k2_bullk{bk}",{**K2,"BULL_K":bk}))
# inverse check: widen BEAR instead (is it bull-specific?)
for bek in ["3","4"]:
    PLAN.append(("W6rk",f"k2_beark{bek}",{**K2,"BEAR_K":bek}))
done=set(pd.read_csv(w1.LEDGER)["tag"]) if w1.LEDGER.exists() else set()
print(f"WAVE-6 PLAN: {len(PLAN)}",flush=True)
for phase,tag,ov in PLAN:
    if tag in done: print(f"skip {tag}",flush=True); continue
    try: w1.log_rec(phase,w1.run_cfg(tag,ov))
    except Exception as e: print(f"ERR {tag}: {e}",flush=True)
print("DONE phase_2025_opt wave-6",flush=True)
