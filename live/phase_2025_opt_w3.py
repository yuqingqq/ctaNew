"""WAVE-3: clean SYMMETRIC K-ladder {1,4,5} (have 2,3) to confirm breadth-reduction is a monotone discrete
choice, not a grid-search peak. Plus ks2 with sleeve-decay (the other broad lever) as a stack check."""
import sys; from pathlib import Path; import pandas as pd
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.phase_2025_opt as w1
PLAN=[("P7","ks1_kl1",{"STRAT_K_SHORT":"1","STRAT_K_LONG":"1"}),
      ("P7","ks4_kl4",{"STRAT_K_SHORT":"4","STRAT_K_LONG":"4"}),
      ("P7","ks5_kl5",{"STRAT_K_SHORT":"5","STRAT_K_LONG":"5"}),
      ("P7b","ks2_kl2_decay1.5",{"STRAT_K_SHORT":"2","STRAT_K_LONG":"2","SLEEVE_DECAY_TAU":"1.5"})]
done=set(pd.read_csv(w1.LEDGER)["tag"]) if w1.LEDGER.exists() else set()
print(f"WAVE-3 PLAN: {len(PLAN)}",flush=True)
for phase,tag,ov in PLAN:
    if tag in done: print(f"skip {tag}",flush=True); continue
    try: w1.log_rec(phase,w1.run_cfg(tag,ov))
    except Exception as e: print(f"ERR {tag}: {e}",flush=True)
print("DONE phase_2025_opt wave-3",flush=True)
