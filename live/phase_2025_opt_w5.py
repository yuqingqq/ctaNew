"""WAVE-5: levers that may STACK with the K=2 win (base = STRAT_K_SHORT=2 STRAT_K_LONG=2). Each an env-sweep replay
vs canonical preds; per-fold pnl logged for nested-OOS. Includes a CONVICTION PLACEBO (random-2 short via randshort)
to confirm K=2's edge is the top-conviction SELECTION, not just holding fewer positions."""
import sys; from pathlib import Path; import pandas as pd
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.phase_2025_opt as w1
K2={"STRAT_K_SHORT":"2","STRAT_K_LONG":"2"}
PLAN=[]
# conviction placebo: random-2 short + top-2 long (randshort mode) vs K2 top-both
PLAN.append(("W5cp","k2_randshort",{**K2,"SIDE_MODE":"randshort","RANDSHORT_SEED":"1"}))
PLAN.append(("W5cp","k2_randshort2",{**K2,"SIDE_MODE":"randshort","RANDSHORT_SEED":"7"}))
# sizing within the 2-name basket
for sm in ["inv_vol","inv_sqrt_vol","volcap"]:
    PLAN.append(("W5sz",f"k2_{sm}",{**K2,"SIZING_MODE":sm}))
# sleeve count (cost amortization vs freshness) on K2
for h in ["4","8","9"]:
    PLAN.append(("W5hold",f"k2_hold{h}",{**K2,"STRAT_HOLD":h}))
done=set(pd.read_csv(w1.LEDGER)["tag"]) if w1.LEDGER.exists() else set()
print(f"WAVE-5 PLAN: {len(PLAN)}",flush=True)
for phase,tag,ov in PLAN:
    if tag in done: print(f"skip {tag}",flush=True); continue
    try: w1.log_rec(phase,w1.run_cfg(tag,ov))
    except Exception as e: print(f"ERR {tag}: {e}",flush=True)
print("DONE phase_2025_opt wave-5",flush=True)
