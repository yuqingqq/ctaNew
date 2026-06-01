"""iter-024 DECISIVE R4 for Mechanism B (drawdown-DURATION trigger).

Mechanism B looked great on incremental maxDD/Calmar, BUT it de-grosses on 774-3673 cycles (vs the
depth-stop's 15-92) -> it is removing a LOT of average exposure. The reactive-track R4 question: does
firing ON the underwater-duration cut the LEFT TAIL better than (i) a CONSTANT de-gross of equal average
gross, and (ii) a matched-%-time RANDOM de-gross? If constant/random match it, B is "run much smaller
(longer)", i.e. ~proportional — the SAME honest verdict as iter-012, NOT a genuinely-different mechanism
that adds incremental skill. Decisive at construction layer (AGENT.md / R4-placebo).

For each universe @4.5bps: take the iter-012 depth-stop book as the layer-on baseline, add the duration
overlay (best D from the pre-check), then compare its maxDD vs:
  (a) constant gross == avg_gross of the duration book (R4 constant-de-gross of equal exposure)
  (b) 200 random de-gross masks of the SAME #cycles-at-floor, SAME g_floor (R4 matched-%-time placebo).
Reuses iter024 engine. Modifies nothing prior.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import importlib.util as _ilu

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
_s = _ilu.spec_from_file_location("it24", SCRIPTS/"iter024_reactive_risk_precheck.py")
it24 = _ilu.module_from_spec(_s); _s.loader.exec_module(it24)
build_universe = it24.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = it24.HL70_PREDS, it24.EXT_PREDS, it24.S44_PREDS
heldbook_engine = it24.heldbook_engine
metrics = it24.metrics
x124 = it24.x124
PRIMARY_COST = it24.PRIMARY_COST; GFLOOR = it24.GFLOOR

N = 200
BEST_D = {"HL70": 90, "EXT": 90, "S44": 90}   # best Calmar per universe from the pre-check


def main():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    print("="*110)
    print("iter-024 R4-DECISIVE for Mechanism B (duration trigger): does it beat CONSTANT/ RANDOM de-gross")
    print("  of EQUAL average exposure? (if not -> ~proportional 'run smaller longer', NOT a new mechanism)")
    print("="*110)
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        U = build_universe(pp, name); cyc = U["cyc"]["base"]; rs = U["rs"]
        base = x124.gross_unit(cyc, rs, PRIMARY_COST)*1e4
        # iter-012 depth-stop book (the layer we add to) and the duration book
        pnl012 = heldbook_engine(cyc, rs, PRIMARY_COST)*1e4
        D = BEST_D[name]
        pnlB, dfire, durf, gB = heldbook_engine(cyc, rs, PRIMARY_COST, dur_D=D, return_diag=True)
        pnlB = pnlB*1e4
        mB = metrics(pnlB); m012 = metrics(pnl012)
        avgG = gB.mean(); n_floor = int((gB < 1.0).sum())
        # (a) constant de-gross of equal AVERAGE gross on the depth-stop book pnl
        const = pnl012*avgG
        mc = metrics(const)
        # (b) matched-%-time random de-gross: from the depth-stop book, push n_floor random cycles to the
        #     SAME relative floor (gB on those cycles), rest unchanged. Approx via scaling pnl012.
        floor_ratio = avgG  # average gross of the duration book; random matches #cycles at GFLOOR
        n_at_floor = int((np.abs(gB - GFLOOR) < 1e-6).sum())
        mdds = np.empty(N)
        for i in range(N):
            pick = rng.choice(len(pnl012), size=n_at_floor, replace=False)
            gg = np.ones(len(pnl012)); gg[pick] = GFLOOR
            mdds[i] = metrics(pnl012*gg)["maxDD"]
        rank = float((mB["maxDD"] > mdds).mean()*100)
        print(f"\n--- {name} (depth-stop maxDD {m012['maxDD']:+.0f} Calmar {m012['Calmar']:+.2f}) "
              f"+ duration D={D} ---")
        print(f"  duration book: maxDD {mB['maxDD']:+.0f} Calmar {mB['Calmar']:+.2f} Sharpe {mB['Sharpe']:+.2f} "
              f"avgGross {avgG:.2f}  (#cyc at floor {n_at_floor})")
        print(f"  (a) CONSTANT de-gross x{avgG:.2f} of depth-stop book: maxDD {mc['maxDD']:+.0f} "
              f"Calmar {mc['Calmar']:+.2f} Sharpe {mc['Sharpe']:+.2f}  -> "
              f"{'duration BEATS const tail' if mB['maxDD']>mc['maxDD'] else 'CONST matches/better = ~proportional'}")
        print(f"  (b) RANDOM matched-%-time de-gross ({N} seeds): p50 maxDD {np.percentile(mdds,50):+.0f} "
              f"p95(best) {np.percentile(mdds,95):+.0f}  -> duration ranks p{rank:.0f} "
              f"{'PASS(>=p95 skill)' if rank>=95 else '(~proportional, NOT skill)'}")
    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
