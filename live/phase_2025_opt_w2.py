"""8h loop WAVE-2 — pred-magnitude/EV floor + sleeve-decay + bear-gross, layered on the wave-1 survivor.

Requires the gated PRED_FLOOR edit in convexity_paper_bot.py (default 0.0 = production no-op). Each candidate
is an env-override replay vs canonical preds. Reuses the wave-1 driver's run_cfg/log_rec machinery.
The wave-1 winning DISP_GATE config is read from ledger (best nested-OOS) and layered under P2/P5/P6 where noted.
"""
import sys
from pathlib import Path
import pandas as pd
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.phase_2025_opt as w1   # reuse run_cfg/log_rec/PROD/OUT/LEDGER

# NOTE: DISP_GATE was REJECTED in wave-1 (fails nested-OOS: honest 2025 +0.27 < baseline +0.38; folds_beat ~50%).
# So wave-2 tests levers STANDALONE — no layering on the overfit disp gate.

PLAN=[]
# P3b: LONG-BOOK REDUCTION depth — wave-1 WIN was ks3_kl2 (cut longs 3->2), nested-OOS 2025 +0.68/dense +1.40.
# Probe how far to cut longs (long=beta-hedge drag, DDI-2). kl0 = short-only diagnostic.
for ks,kl in [("3","1"),("4","1"),("2","1"),("3","0"),("4","0"),("2","2")]:
    PLAN.append(("P3b",f"ks{ks}_kl{kl}",{"STRAT_K_SHORT":ks,"STRAT_K_LONG":kl}))
# P2: pred-magnitude / EV floor in default mode (skip low-conviction legs). Thin-regime lever.
for fl in ["0.10","0.20","0.30","0.40","0.50","0.70"]:
    PLAN.append(("P2",f"predfloor_{fl}",{"PRED_FLOOR":fl}))
# P2c: EV floor COMBINED with the wave-1 winner (cut longs to 2)
for fl in ["0.20","0.30","0.40"]:
    PLAN.append(("P2c",f"predfloor_{fl}_kl2",{"PRED_FLOOR":fl,"STRAT_K_SHORT":"3","STRAT_K_LONG":"2"}))
# P5: sleeve age-decay (cost amortization vs freshness) — vBTC found equal optimal; verify for convexity
for tau in ["1.5","3","6"]:
    PLAN.append(("P5",f"decay_tau{tau}",{"SLEEVE_DECAY_TAU":tau}))
# P6: bear gross mult (de-risk bear) standalone
for bg in ["0.5","0.75"]:
    PLAN.append(("P6",f"beargross_{bg}",{"BEAR_GROSS_MULT":bg}))

done=set()
if w1.LEDGER.exists(): done=set(pd.read_csv(w1.LEDGER)["tag"])
print(f"WAVE-2 PLAN: {len(PLAN)} configs",flush=True)
for phase,tag,ov in PLAN:
    if tag in done: print(f"skip {tag} (done)",flush=True); continue
    try: w1.log_rec(phase,w1.run_cfg(tag,ov))
    except Exception as e: print(f"ERR {tag}: {e}",flush=True)
print("DONE phase_2025_opt wave-2",flush=True)
