"""iter-024 PRE-CHECK — genuinely-DIFFERENT reactive risk mechanisms layered ON TOP of the adopted
iter-012 vol-norm equity-DD stop. Decisive question (reactive track, R2/R4/R5/R7): does a NEW mechanism
cut maxDD / raise Calmar INCREMENTALLY beyond the iter-012 stop, ACROSS HL70+EXT+S44, and NOT just match
random / proportional?

Two candidate mechanisms (each layered on the iter-012 stop, measure the INCREMENTAL delta):

  A. POSITION-LEVEL worst-leg stop (vs book-level de-gross). At each cycle drop the held-book legs whose
     OWN trailing realized contribution (PIT through t-1) is in the worst tail. Mechanism prior: iter-006
     root-cause says the DD is the LONG leg of high-beta fallen alts in a correlated alt-bear -> if the
     bleeding legs are persistent + identifiable, cutting THEM specifically caps the same tail with less
     total exposure removed (better Calmar) than the book-level de-gross. DECISIVE construction-layer
     placebo (AGENT.md): worst-leg-cut vs RANDOM-leg-cut of equal count. If random does as well, the
     "which leg is worst" carries no forward info (correlated selloff => near-random) -> NO-CANDIDATE.

  B. DRAWDOWN-DURATION trigger (depth-orthogonal axis). The iter-012 stop fires on DD DEPTH (>= k*sigma).
     A duration trigger fires when the book has been continuously underwater >= D bars REGARDLESS of depth
     (a slow grind the depth-stop misses). Mechanism prior (lit: duration vs depth are distinct risk axes,
     prolonged shallow grind != sharp deep dip). To ADD anything it must (i) fire UNCORRELATED with the
     depth-stop (catch a DIFFERENT tail) and (ii) incrementally cut maxDD/Calmar across universes.

All PIT: every trigger uses realized equity / leg-contribution through t-1, lagged. Reuses X123 universe
+ X125 iter-012 stop machinery. Modifies nothing prior.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import importlib.util as _ilu

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"

_s123 = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_s123); _s123.loader.exec_module(x123)
build_universe = x123.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
HOLD = x123.HOLD

_s124 = _ilu.spec_from_file_location("x124", SCRIPTS/"X124_reactive_dd_stop.py")
x124 = _ilu.module_from_spec(_s124); _s124.loader.exec_module(x124)
metrics = x124.metrics

SEED = 12345
ANN = 6*365
PRIMARY_COST = 4.5e-4
COSTS_BPS = [1.0, 3.0, 4.5]

# iter-012 fixed reactive policy
GFLOOR = 0.40; HEAL = 0.50; RDAYS = 90; VOL_WIN = 180; WARMUP = 60; SQRT_WIN = np.sqrt(VOL_WIN); REC_K = 2.0
N_PLACEBO = 200

EPISODES = [
    ("2022_luna", "2022-05-01", "2022-07-31"),
    ("2022_ftx", "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4", "2025-09-01", "2025-12-31"),
]


# ---- the iter-012 stop, returning per-cycle gross AND the depth-trigger firing flag (for orthogonality)
def _depth_stop_state(incr_so_far, eq, peak, stopped, stop_peak, trough, stop_t, t, k):
    """Resolve the iter-012 depth-stop decision at cycle t given realized state through t-1."""
    dd = eq - peak
    if t >= 2:
        lo = max(0, t - VOL_WIN); seg = incr_so_far[lo:t]; seg = seg[np.isfinite(seg)]
        sigma = float(seg.std()) if len(seg) >= 2 else 0.0
    else:
        sigma = 0.0
    trig = k * sigma * SQRT_WIN
    can_fire = (t >= WARMUP) and (sigma > 0)
    fired = False
    if not stopped:
        if can_fire and (-dd >= trig):
            stopped = True; fired = True; stop_peak = peak; trough = eq; stop_t = t
    else:
        trough = min(trough, eq)
        gap = stop_peak - trough
        healed = (gap > 0) and ((eq - trough) >= HEAL*gap)
        timed = (t - stop_t) >= RDAYS
        if (healed and eq > trough) or timed:
            stopped = False
    return stopped, stop_peak, trough, stop_t, fired


def heldbook_engine(cyc_w, rs, cost, leg_keep_fn=None, gross_fn=None,
                    k=REC_K, dur_D=None, return_diag=False):
    """One causal pass. Optional layers on top of the iter-012 depth-stop:
      leg_keep_fn(t, net_dict, leg_contrib_trailing) -> set of symbols to KEEP (position-level stop, A)
      dur_D : if set, ALSO de-gross to GFLOOR when underwater >= dur_D consecutive bars (duration, B)
    gross from the depth-stop is always active (the iter-012 baseline we layer on).
    Returns pnl (fraction). If return_diag, also (depth_fire, dur_fire, gross) arrays.
    """
    n = len(cyc_w)
    pnl = np.empty(n); incr = np.empty(n)
    gross_arr = np.ones(n); depth_fire = np.zeros(n, bool); dur_fire = np.zeros(n, bool)
    # trailing per-leg contribution buffer: list of dicts {sym: realized contrib bps} per past cycle
    leg_hist = []
    prev = {}
    eq = 0.0; peak = 0.0
    stopped = False; stop_peak = 0.0; trough = 0.0; stop_t = 0
    underwater_run = 0
    for t in range(n):
        # ---- depth-stop (iter-012) decision from state through t-1 ----
        stopped, stop_peak, trough, stop_t, fired = _depth_stop_state(
            incr, eq, peak, stopped, stop_peak, trough, stop_t, t, k)
        depth_fire[t] = fired
        g = GFLOOR if stopped else 1.0
        # ---- duration overlay (B): underwater run length through t-1 ----
        if dur_D is not None:
            if (t >= WARMUP) and (underwater_run >= dur_D):
                if g == 1.0:
                    dur_fire[t] = True
                g = min(g, GFLOOR)
        gross_arr[t] = g
        # ---- build net book ----
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items():
                net[s] = net.get(s, 0.0) + wt/HOLD
        # ---- position-level leg stop (A): keep only allowed legs ----
        if leg_keep_fn is not None and len(net) > 0:
            # trailing contribution per leg: sum over last VOL_WIN cycles of realized contrib
            contrib = {}
            for d in leg_hist[-VOL_WIN:]:
                for s, c in d.items():
                    contrib[s] = contrib.get(s, 0.0) + c
            keep = leg_keep_fn(net, contrib)
            net = {s: v for s, v in net.items() if s in keep}
        scaled = {s: g*v for s, v in net.items()}
        alls = set(scaled) | set(prev)
        turn = sum(abs(scaled.get(s, 0.0) - prev.get(s, 0.0)) for s in alls)
        rl = rs[t]
        # realized PnL + per-leg contribution
        leg_c = {}
        c = 0.0
        for s, w in scaled.items():
            r = rl.get(s, np.nan)
            if np.isfinite(r):
                lc = w*r; c += lc; leg_c[s] = lc*1e4   # bps contribution this cycle (post-gross)
        if not np.isfinite(c):
            c = 0.0
        pnl[t] = c - turn*0.5*cost
        prev = scaled
        leg_hist.append(leg_c)
        step = pnl[t]*1e4
        if not np.isfinite(step):
            step = 0.0
        incr[t] = step
        eq += step
        if eq > peak:
            peak = eq; underwater_run = 0
        else:
            underwater_run += 1
    if return_diag:
        return pnl, depth_fire, dur_fire, gross_arr
    return pnl


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*120)
    print("iter-024 PRE-CHECK — reactive risk mechanisms layered ON TOP of iter-012 vol-norm depth-stop")
    print("="*120)
    panels = {}
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        print(f"[build] {name}", flush=True)
        U = build_universe(pp, name)
        cyc = U["cyc"]["base"]; rs = U["rs"]
        base = x124.gross_unit(cyc, rs, PRIMARY_COST)*1e4
        # iter-012 stop (depth only) = the layer we improve on
        pnl_012, dfire, _, g012 = heldbook_engine(cyc, rs, PRIMARY_COST, return_diag=True)
        panels[name] = dict(U=U, cyc=cyc, rs=rs, base=base,
                            pnl012=pnl_012*1e4, dfire=dfire, g012=g012)

    print("\n" + "="*120)
    print("REFERENCE: base (no stop) vs iter-012 depth-stop @4.5bps")
    print("="*120)
    for name in panels:
        P = panels[name]; bm = metrics(P["base"]); sm = metrics(P["pnl012"])
        print(f"  {name:>5}: base maxDD {bm['maxDD']:+.0f} Calmar {bm['Calmar']:+.2f} | "
              f"iter012 maxDD {sm['maxDD']:+.0f} ({(1-sm['maxDD']/bm['maxDD'])*100:+.1f}%) "
              f"Calmar {sm['Calmar']:+.2f} Sharpe {sm['Sharpe']:+.2f} "
              f"(%degross {(P['g012']<1).mean()*100:.0f})")

    # ============================================================ MECHANISM A — position-level worst-leg stop
    print("\n" + "="*120)
    print("MECHANISM A — POSITION-LEVEL worst-leg stop ON TOP of iter-012. Drop legs whose trailing")
    print("  realized contribution (PIT t-1) is in worst Q-tile. INCREMENTAL vs iter-012 + RANDOM-leg placebo.")
    print("="*120)
    for name in panels:
        P = panels[name]; cyc = P["cyc"]; rs = P["rs"]
        sm012 = metrics(P["pnl012"])
        print(f"\n--- {name} (iter-012 maxDD {sm012['maxDD']:+.0f} Calmar {sm012['Calmar']:+.2f}) ---")
        for q in (0.20, 0.33):
            def keep_worst_cut(net, contrib, q=q):
                # rank held legs by trailing contribution; drop the worst q-fraction (those bleeding most)
                if len(net) <= 2:
                    return set(net)
                items = sorted(net.keys(), key=lambda s: contrib.get(s, 0.0))  # ascending = worst first
                ncut = max(1, int(round(len(items)*q)))
                return set(items[ncut:])
            pnl_A = heldbook_engine(cyc, rs, PRIMARY_COST, leg_keep_fn=keep_worst_cut)*1e4
            mA = metrics(pnl_A)
            ddRed_inc = (1 - mA["maxDD"]/sm012["maxDD"])*100 if sm012["maxDD"] < 0 else np.nan
            # construction-layer placebo: drop the SAME count of RANDOM legs each cycle
            mdds = []; cals = []
            for i in range(40):   # cheaper: 40 seeds for the pre-check
                rg = np.random.default_rng(1000+i)
                def keep_rand(net, contrib, q=q, rg=rg):
                    if len(net) <= 2:
                        return set(net)
                    items = list(net.keys()); ncut = max(1, int(round(len(items)*q)))
                    drop = set(rg.choice(items, size=ncut, replace=False))
                    return set(items) - drop
                pnl_R = heldbook_engine(cyc, rs, PRIMARY_COST, leg_keep_fn=keep_rand)*1e4
                mR = metrics(pnl_R); mdds.append(mR["maxDD"]); cals.append(mR["Calmar"])
            mdds = np.array(mdds); cals = np.array(cals)
            rank_dd = float((mA["maxDD"] > mdds).mean()*100)     # higher maxDD (less negative) = better
            rank_cal = float((mA["Calmar"] > cals).mean()*100)
            verdict = "PASS" if (rank_dd >= 95 and ddRed_inc > 0) else "FAIL(~random/no-inc)"
            print(f"  cut worst {q:.0%}: maxDD {mA['maxDD']:+.0f} (incDD {ddRed_inc:+.1f}% vs i012) "
                  f"Calmar {mA['Calmar']:+.2f} Sharpe {mA['Sharpe']:+.2f} | "
                  f"random-leg p50 maxDD {np.percentile(mdds,50):+.0f} Calmar {np.percentile(cals,50):+.2f} "
                  f"-> real rank p{rank_dd:.0f}(DD)/p{rank_cal:.0f}(Cal)  {verdict}", flush=True)

    # ============================================================ MECHANISM B — drawdown-DURATION trigger
    print("\n" + "="*120)
    print("MECHANISM B — DRAWDOWN-DURATION trigger ON TOP of iter-012. De-gross when underwater >= D bars")
    print("  (depth-orthogonal). (1) orthogonality of duration-fire vs depth-fire; (2) INCREMENTAL maxDD/Calmar.")
    print("="*120)
    for name in panels:
        P = panels[name]; cyc = P["cyc"]; rs = P["rs"]; dfire = P["dfire"]
        sm012 = metrics(P["pnl012"])
        print(f"\n--- {name} (iter-012 maxDD {sm012['maxDD']:+.0f} Calmar {sm012['Calmar']:+.2f}, "
              f"depth fires {int(dfire.sum())}) ---")
        for D in (30, 60, 90):
            pnl_B, df2, durf, gB = heldbook_engine(cyc, rs, PRIMARY_COST, dur_D=D, return_diag=True)
            pnl_B = pnl_B*1e4; mB = metrics(pnl_B)
            ddRed_inc = (1 - mB["maxDD"]/sm012["maxDD"])*100 if sm012["maxDD"] < 0 else np.nan
            # orthogonality: of the cycles where duration fires NEW de-gross, how many already depth-degrossed?
            dur_only = durf & (gB > GFLOOR + 1e-9) if False else durf  # durf marks cycles dur pushed to floor while depth==1
            n_dur = int(durf.sum())
            # overlap: duration-de-grossed cycles that the depth-stop ALSO had de-grossed
            depth_degrossed = P["g012"] < 1.0
            overlap = int((durf & depth_degrossed).sum())
            frac_new = (1 - overlap/n_dur)*100 if n_dur > 0 else 0.0
            print(f"  D={D:>3}: maxDD {mB['maxDD']:+.0f} (incDD {ddRed_inc:+.1f}% vs i012) "
                  f"Calmar {mB['Calmar']:+.2f} Sharpe {mB['Sharpe']:+.2f} | dur-fires {n_dur} "
                  f"({frac_new:.0f}% NOT already depth-degrossed = orthogonal tail)", flush=True)

    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
