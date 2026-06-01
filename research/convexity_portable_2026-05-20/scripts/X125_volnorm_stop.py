"""X125 — iter-012 PORTABLE: VOL-NORMALIZED reactive equity-drawdown stop (mechanical, PIT, NOT alpha).

REACTIVE risk-control track. iter-011 (X124) built a working equity-DD stop but its trigger was an
ABSOLUTE bps threshold (X=1600 off peak). Different universes have different natural equity scales, so
one absolute X cannot be right everywhere -> X124 PASSED R6 nested-OOS on HL70 but FAILED on EXT (forward
ddRed -7.3% at +44.8% cost: X drifts deep on the longer panels and barely fires) and S44. This iteration
makes the trigger SELF-NORMALIZING so a single UNITLESS parameter `k` generalizes across HL70+EXT+S44
under nested-OOS WITHOUT per-universe tuning.

THE PORTABLE TRIGGER (research/handoff.md spec):
  De-gross the whole held book to g_floor=0.40 when the strategy's OWN drawdown-from-peak
      (peak - eq) >= k * sigma(trailing-180-bar equity increments) * sqrt(win)
  where `k` is a UNITLESS "sigmas of equity" multiple (recommended k=2.0). sigma / peak / DD all computed
  through t-1 (PIT). Re-enter (gross->1) when equity heals 50% of the DD back toward the peak (and
  eq > trough) OR after 90 bars (~15d) timeout. WARMUP 60 bars: no trigger can fire before bar 60.

WHY VOL-NORMALIZED IS PORTABLE: the trigger threshold k*sigma*sqrt(win) auto-scales to each universe's
  own equity volatility, so the same unitless k="sigmas of equity" has the same MEANING on every universe
  -> the nested-OOS selector lands on the same k on every universe/fold and the choice transports.

Reuses X124's gross-aware held-book engine VERBATIM (gross applied to positions BEFORE turnover/cost) and
X123's build_universe (preds + klines pipeline, X117 production base book). The ONLY change vs X124 is the
TRIGGER FORM (absolute-X -> vol-normalized-k). Same fixed re-entry policy (g_floor / heal / timeout) and
the same canonical accounting at cost {1,3,4.5}bps. Modifies NOTHING prior (no baseline scripts/preds).

PIT (R1): equity, running peak, and the trailing-180 sigma of equity INCREMENTS are ALL computed from
  realized PnL through t-1 (the increment at tau is gross[tau]*pnl[tau]*1e4, already realized at decision
  time t for tau<=t-1). The gross applied to cycle t is fixed from that t-1 state; pnl[t] is realized and
  equity advanced only AFTER. The held book overlaps HOLD=6 sleeves; the equity we gate on is realized
  through t-1. No forward peek. k is unitless and self-normalizing -> same meaning per universe.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

import importlib.util as _ilu

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
OUT = REPO/"research/convexity_portable_2026-05-20/results"

# reuse X123 universe pipeline + X124 held-book engine verbatim ---------------------------------------
_s123 = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_s123); _s123.loader.exec_module(x123)
build_universe = x123.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
HOLD = x123.HOLD

_s124 = _ilu.spec_from_file_location("x124", SCRIPTS/"X124_reactive_dd_stop.py")
x124 = _ilu.module_from_spec(_s124); _s124.loader.exec_module(x124)
metrics = x124.metrics          # n / tot / maxDD / Sharpe / Calmar / cvar1 (NaN-guarded)
gross_unit = x124.gross_unit    # gross==1.0 reproduces the X117/X123 base book
const_degross = x124.const_degross

SEED = 12345
N_PLACEBO = 200
ANN = 6*365                # cycles/yr (4h horizon, held-book per-cycle)
COSTS_BPS = [1.0, 3.0, 4.5]
PRIMARY_COST = 4.5e-4      # production calibration (X117 = +1.93 / -5674 at 4.5bps)

# fixed reactive policy (NOT tuned; identical to iter-011 X124) ---------------------------------------
GFLOOR = 0.40              # de-gross to 40% gross (book keeps participating so equity can heal)
HEAL = 0.50                # re-enter once equity heals half the drawdown back toward the peak
RDAYS = 90                 # OR after 90 bars (~15d) time fail-safe
VOL_WIN = 180              # trailing window for sigma of equity increments (~30d) — fixed policy
WARMUP = 60                # no trigger can fire before this many bars — fixed policy
SQRT_WIN = np.sqrt(VOL_WIN)

# UNITLESS k sweep (the ONLY knob; flagged for R6 nested-OOS) -----------------------------------------
K_GRID = [1.5, 2.0, 2.5, 3.0]
REC_K = 2.0                # the recommended unitless multiple

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


# --------------------------------------------------------------------------- vol-normalized DD stop
def run_volnorm_heldbook(cyc_w, rs, cost, k, g_floor=GFLOOR, heal=HEAL, timeout=RDAYS,
                         vol_win=VOL_WIN, warmup=WARMUP):
    """CANONICAL run: one causal forward pass that (1) builds the de-grossed gross[t] from the
    strategy's OWN realized equity through t-1 with a VOL-NORMALIZED trigger, and (2) computes per-cycle
    realized PnL by scaling the book BEFORE turnover/cost (X124 heldbook mechanics inline).

    Trigger: fire (de-gross to g_floor) when  (peak - eq) >= k * sigma * sqrt(vol_win)  where sigma is
    the std of the strategy's OWN realized equity INCREMENTS over the trailing `vol_win` bars through
    t-1. Because the equity / sigma depend on the gross and the gross depends on them, we resolve all in
    one pass: gross[t] is fixed from realized state through t-1, then pnl[t] realized, then equity (and
    the increment buffer) advanced. No look-ahead. WARMUP: no trigger before bar `warmup`.

    Returns (pnl_bps, gross, in_stop, n_roundtrips, thresh) where thresh[t] is the bps trigger level
    in effect at t (for diagnostics)."""
    n = len(cyc_w)
    pnl = np.empty(n, dtype=np.float64)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool); thresh = np.full(n, np.nan)
    incr = np.empty(n, dtype=np.float64)          # realized equity increment per cycle (bps), filled as we go
    prev = {}                                     # previous GROSS-SCALED net positions
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    for t in range(n):
        # ---- vol-normalized stop decision from realized state THROUGH t-1 (PIT) ----
        dd = eq - peak                            # <= 0
        # trailing sigma of equity increments over the last vol_win realized cycles (through t-1)
        if t >= 2:
            lo = max(0, t - vol_win)
            seg = incr[lo:t]                      # realized increments for cycles lo..t-1
            seg = seg[np.isfinite(seg)]
            sigma = float(seg.std()) if len(seg) >= 2 else 0.0
        else:
            sigma = 0.0
        trig = k * sigma * SQRT_WIN               # bps drawdown that fires the stop
        thresh[t] = trig
        can_fire = (t >= warmup) and np.isfinite(sigma) and (sigma > 0)
        if not stopped:
            if can_fire and (-dd >= trig):
                stopped = True; rt += 1; stop_peak = peak; trough = eq; stop_t = t
        else:
            trough = min(trough, eq)
            gap = stop_peak - trough
            healed = (gap > 0) and ((eq - trough) >= heal*gap)
            timed = (t - stop_t) >= timeout
            if (healed and eq > trough) or timed:
                stopped = False
        g = g_floor if stopped else 1.0
        gross[t] = g; in_stop[t] = stopped
        # ---- realize cycle t PnL: scale positions BEFORE turnover/cost (X124 mechanics) ----
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items():
                net[s] = net.get(s, 0.0) + wt/HOLD
        scaled = {s: g*v for s, v in net.items()}
        alls = set(scaled) | set(prev)
        turn = sum(abs(scaled.get(s, 0.0) - prev.get(s, 0.0)) for s in alls)
        rl = rs[t]
        c = sum(scaled.get(s, 0.0)*rl.get(s, 0.0) for s in scaled if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c):
            c = 0.0
        pnl[t] = c - turn*0.5*cost
        prev = scaled
        # ---- advance equity (realized) and record the increment ----
        step = pnl[t]*1e4                          # bps equity increment for cycle t
        if not np.isfinite(step):
            step = 0.0
        incr[t] = step
        eq += step
        if eq > peak:
            peak = eq
    return pnl, gross, in_stop, rt, thresh


def run_volnorm_scalar(pnl_base_bps, k, g_floor=GFLOOR, heal=HEAL, timeout=RDAYS,
                       vol_win=VOL_WIN, warmup=WARMUP):
    """Scalar-approx vol-normalized stop on a precomputed cost-netted base-PnL series (scales the
    already-netted pnl by gross). Used in the R6 nested-OOS / R4-placebo loops where rebuilding the full
    held book per fold/seed is prohibitive (same family as the X124 scalar approx). The trailing-sigma is
    of the REALIZED (scaled) equity increments through t-1. Returns (gross, in_stop, n_roundtrips)."""
    pb = np.asarray(pnl_base_bps, dtype=np.float64)
    pb = np.where(np.isfinite(pb), pb, 0.0)
    n = len(pb)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool); incr = np.empty(n, dtype=np.float64)
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    for t in range(n):
        dd = eq - peak
        if t >= 2:
            lo = max(0, t - vol_win); seg = incr[lo:t]; seg = seg[np.isfinite(seg)]
            sigma = float(seg.std()) if len(seg) >= 2 else 0.0
        else:
            sigma = 0.0
        trig = k * sigma * SQRT_WIN
        can_fire = (t >= warmup) and (sigma > 0)
        if not stopped:
            if can_fire and (-dd >= trig):
                stopped = True; rt += 1; stop_peak = peak; trough = eq; stop_t = t
        else:
            trough = min(trough, eq)
            gap = stop_peak - trough
            healed = (gap > 0) and ((eq - trough) >= heal*gap)
            timed = (t - stop_t) >= timeout
            if (healed and eq > trough) or timed:
                stopped = False
        g = g_floor if stopped else 1.0
        gross[t] = g; in_stop[t] = stopped
        step = g*pb[t]
        incr[t] = step if np.isfinite(step) else 0.0
        eq += incr[t]
        if eq > peak:
            peak = eq
    return gross, in_stop, rt


# --------------------------------------------------------------------------- builds
def build_panel(name, preds_path):
    print(f"\n[build] {name}", flush=True)
    U = build_universe(preds_path, name)
    base_by_cost = {cb: gross_unit(U["cyc"]["base"], U["rs"], cb*1e-4)*1e4 for cb in COSTS_BPS}
    return U, base_by_cost


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*120, flush=True)
    print("X125 — VOL-NORMALIZED (PORTABLE) REACTIVE EQUITY-DRAWDOWN STOP", flush=True)
    print(f"  de-gross to g_floor={GFLOOR} when own DD-from-peak >= k * sigma(trailing-{VOL_WIN} equity "
          f"increments) * sqrt({VOL_WIN}); k UNITLESS (rec k={REC_K})", flush=True)
    print(f"  re-enter on {HEAL:.0%} heal OR {RDAYS}-bar timeout; warmup {WARMUP}; PIT (state through "
          f"t-1); costs {COSTS_BPS} bps. seed {SEED}", flush=True)
    print("="*120, flush=True)

    panels = {}
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        U, base_by_cost = build_panel(name, pp)
        panels[name] = dict(U=U, base=base_by_cost)

    # ---- base reproduction (must match X117 on HL70) ----
    print("\n" + "="*120, flush=True)
    print("BASE REPRODUCTION (gross=1.0 everywhere == X117/X123 base book) @ each cost", flush=True)
    print("="*120, flush=True)
    for name in ("HL70", "EXT", "S44"):
        for cb in COSTS_BPS:
            m = metrics(panels[name]["base"][cb])
            tag = "  <- X117 target +1.93/-5674" if (name == "HL70" and cb == 4.5) else ""
            print(f"  {name:>5} @{cb:>4.1f}bps  Sharpe {m['Sharpe']:+.2f}  maxDD {m['maxDD']:+.0f}  "
                  f"Calmar {m['Calmar']:+.2f}  totPnL {m['tot']:+.0f}{tag}", flush=True)

    # ============================================================ R2/R3 DD-vs-COST + k-SWEEP
    print("\n" + "="*120, flush=True)
    print("R2/R3 — DD-vs-COST TRADE-OFF + k-SWEEP {1.5,2.0,2.5,3.0} per universe per cost", flush=True)
    print("  (gross applied to positions BEFORE turnover/cost; canonical held-book accounting)", flush=True)
    print("="*120, flush=True)
    curve_rows = []
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]
        for cb in COSTS_BPS:
            base = panels[name]["base"][cb]; bm = metrics(base)
            print(f"\n--- {name} @ {cb:.1f}bps : base Sharpe {bm['Sharpe']:+.2f} maxDD {bm['maxDD']:+.0f} "
                  f"Calmar {bm['Calmar']:+.2f} totPnL {bm['tot']:+.0f} ---", flush=True)
            print(f"{'k':>6}{'maxDD':>9}{'ddRed%':>8}{'totPnL':>9}{'totCost%':>9}{'Sharpe':>8}"
                  f"{'Calmar':>8}{'%stop':>7}{'RT':>5}{'avgG':>6}", flush=True)
            for k in K_GRID:
                pnl, gross, stop, rt, _ = run_volnorm_heldbook(U["cyc"]["base"], U["rs"], cb*1e-4, k)
                pnl = pnl*1e4; m = metrics(pnl)
                ddRed = (1-m["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else np.nan
                totCost = (1-m["tot"]/bm["tot"])*100 if bm["tot"] != 0 else np.nan
                rec = (k == REC_K)
                print(f"{k:>6.1f}{m['maxDD']:>9.0f}{ddRed:>8.1f}{m['tot']:>9.0f}{totCost:>9.1f}"
                      f"{m['Sharpe']:>8.2f}{m['Calmar']:>8.2f}{stop.mean()*100:>7.1f}{rt:>5}"
                      f"{gross.mean():>6.2f}{'  <-- RECOMMENDED' if rec else ''}", flush=True)
                curve_rows.append(dict(universe=name, cost_bps=cb, k=k, maxDD=m["maxDD"], ddRed=ddRed,
                                       totPnL=m["tot"], totCost=totCost, Sharpe=m["Sharpe"],
                                       Calmar=m["Calmar"], cvar1=m["cvar1"], pct_stop=stop.mean()*100,
                                       roundtrips=rt, avg_gross=gross.mean(), base_maxDD=bm["maxDD"],
                                       base_totPnL=bm["tot"], base_Sharpe=bm["Sharpe"],
                                       base_Calmar=bm["Calmar"]))
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_parquet(OUT/"X125_tradeoff_curve.parquet", index=False)

    # ============================================================ R4 vs CONSTANT de-gross + placebo
    print("\n" + "="*120, flush=True)
    print("R4 — STOP (triggered) vs CONSTANT de-gross of EQUAL AVERAGE EXPOSURE: does triggering ON the", flush=True)
    print("  drawdown cut the LEFT TAIL better than always running smaller? (g_floor=0.40, sweep k) @4.5bps", flush=True)
    print("="*120, flush=True)
    r4_rows = []
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]; base = panels[name]["base"][PRIMARY_COST*1e4]; bm = metrics(base)
        print(f"\n--- {name} @4.5bps (base maxDD {bm['maxDD']:+.0f}, CVaR1%/cyc {bm['cvar1']:+.2f}) ---",
              flush=True)
        print(f"{'k':>6}{'avgG':>6}{'STOP_maxDD':>11}{'CONST_maxDD':>12}{'STOP-CONST':>11}"
              f"{'STOP_tot':>9}{'CONST_tot':>10}  read", flush=True)
        for k in K_GRID:
            pnl, gross, stop, rt, _ = run_volnorm_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, k)
            pnl = pnl*1e4; ag = gross.mean()
            sm = metrics(pnl); cm = metrics(const_degross(base, ag))
            better = sm["maxDD"] > cm["maxDD"]
            print(f"{k:>6.1f}{ag:>6.2f}{sm['maxDD']:>11.0f}{cm['maxDD']:>12.0f}"
                  f"{sm['maxDD']-cm['maxDD']:>+11.0f}{sm['tot']:>9.0f}{cm['tot']:>10.0f}"
                  f"  {'STOP better tail' if better else 'const matches/better'}", flush=True)
            r4_rows.append(dict(universe=name, k=k, avg_gross=ag, stop_maxDD=sm["maxDD"],
                                const_maxDD=cm["maxDD"], stop_minus_const=sm["maxDD"]-cm["maxDD"],
                                stop_tot=sm["tot"], const_tot=cm["tot"]))
    pd.DataFrame(r4_rows).to_parquet(OUT/"X125_r4_const_degross.parquet", index=False)

    # ---- R4-PLACEBO: stop vs RANDOM de-gross of matched %-time + matched floor (200 seeds) ----
    print("\n" + "-"*120, flush=True)
    print(f"R4-PLACEBO — STOP vs RANDOM de-gross of MATCHED %-time (same #stopped cycles, same g_floor,", flush=True)
    print(f"  {N_PLACEBO} seeds): does triggering ON the realized DD cap maxDD better than random? @4.5bps",
          flush=True)
    print("-"*120, flush=True)
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]; base = panels[name]["base"][PRIMARY_COST*1e4]
        pnl, gross, stop, rt, _ = run_volnorm_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, REC_K)
        real_m = metrics(pnl*1e4); n_stop = int(stop.sum())
        mdds = np.empty(N_PLACEBO)
        for i in range(N_PLACEBO):
            pick = rng.choice(len(base), size=n_stop, replace=False)
            gg = np.ones(len(base)); gg[pick] = GFLOOR
            mdds[i] = metrics(base*gg)["maxDD"]
        rank = float((real_m["maxDD"] > mdds).mean()*100)
        print(f"  {name} k={REC_K}: real maxDD {real_m['maxDD']:+.0f} ({n_stop} stopped, "
              f"{stop.mean()*100:.0f}% time); random matched p50 {np.percentile(mdds,50):+.0f} "
              f"p95(best) {np.percentile(mdds,95):+.0f} -> real ranks p{rank:.0f} "
              f"{'PASS' if rank>=95 else '(proportional, not skill — expected)'}", flush=True)

    # ============================================================ R5 per-episode (EXT) + episode-LOFO
    print("\n" + "="*120, flush=True)
    print("R5 (DECISIVE) — cross-episode tail-capping (EXT) + episode-LOFO. one rule on running EXT equity;", flush=True)
    print("  maxDD WITHIN each episode window. @4.5bps", flush=True)
    print("="*120, flush=True)
    U = panels["EXT"]["U"]
    ot = pd.to_datetime(pd.Series(U["times"]), utc=True).values
    base = panels["EXT"]["base"][PRIMARY_COST*1e4]
    r5_rows = []
    for k in (REC_K, 2.5, 3.0):
        pnl, gross, stop, rt, _ = run_volnorm_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, k)
        gpnl = pnl*1e4
        print(f"\n  k={k} (avg gross {gross.mean():.2f}, {stop.mean()*100:.1f}% time stopped, {rt} round-trips):",
              flush=True)
        print(f"    {'episode':<13}{'base_maxDD':>11}{'stop_maxDD':>11}{'ddRed%':>8}{'base_tot':>10}{'stop_tot':>10}",
              flush=True)
        capped = 0; n_ep = 0
        for nm, s, e in EPISODES:
            m = (ot >= np.datetime64(s)) & (ot <= np.datetime64(e))
            if m.sum() < 40:
                continue
            n_ep += 1; bm = metrics(base[m]); sm = metrics(gpnl[m])
            red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
            if red >= 10:
                capped += 1
            print(f"    {nm:<13}{bm['maxDD']:>11.0f}{sm['maxDD']:>11.0f}{red:>8.1f}"
                  f"{bm['tot']:>10.0f}{sm['tot']:>10.0f}", flush=True)
            r5_rows.append(dict(k=k, episode=nm, base_maxDD=bm["maxDD"], stop_maxDD=sm["maxDD"], ddRed=red))
        print(f"    => episodes with >=10% maxDD reduction: {capped}/{n_ep}", flush=True)
    pd.DataFrame(r5_rows).to_parquet(OUT/"X125_r5_episodes.parquet", index=False)

    print(f"\n  EXT episode-LOFO (drop each episode, recompute the stop on the remaining series, k={REC_K}):",
          flush=True)
    pnl_all, _, _, _, _ = run_volnorm_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, REC_K)
    bA = metrics(base); sA = metrics(pnl_all*1e4)
    print(f"    ALL: base maxDD {bA['maxDD']:+.0f} -> stop {sA['maxDD']:+.0f} "
          f"(ddRed {(1-sA['maxDD']/bA['maxDD'])*100:+.1f}%)", flush=True)
    cyc_base = U["cyc"]["base"]; rs = U["rs"]
    for nm, s, e in EPISODES:
        keep = ~((ot >= np.datetime64(s)) & (ot <= np.datetime64(e)))
        idx = np.where(keep)[0]
        if len(idx) < 100:
            continue
        cyc_k = [cyc_base[i] for i in idx]; rs_k = [rs[i] for i in idx]
        base_k = gross_unit(cyc_k, rs_k, PRIMARY_COST)*1e4
        pnl_k, _, _, _, _ = run_volnorm_heldbook(cyc_k, rs_k, PRIMARY_COST, REC_K)
        bm = metrics(base_k); sm = metrics(pnl_k*1e4)
        red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
        cost = (1-sm["tot"]/bm["tot"])*100 if bm["tot"] != 0 else np.nan
        print(f"    drop {nm:<12}: base maxDD {bm['maxDD']:+.0f} -> stop {sm['maxDD']:+.0f} "
              f"ddRed {red:+.1f}%  totCost {cost:+.1f}%", flush=True)

    # ============================================================ R6 NESTED-OOS of k (ALL THREE univ)
    print("\n" + "="*120, flush=True)
    print("R6 (THE HEADLINE) — NESTED-OOS of unitless k: pick k on PAST folds (max ddRed under <=25% cost", flush=True)
    print("  budget), apply to the NEXT fold; measure realized FORWARD maxDD-capping. ALL THREE universes.", flush=True)
    print("  PASS = forward ddRed > +5% AND forward cost < 40% on EVERY universe. @4.5bps (scalar-approx).", flush=True)
    print("="*120, flush=True)
    r6_pass = {}
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]; base = panels[name]["base"][PRIMARY_COST*1e4]
        fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        oos_b, oos_s, chosen = [], [], []
        for i in range(1, len(folds)):
            past = np.isin(fold_arr, folds[:i]); fut = fold_arr == folds[i]
            pp = base[past]
            best_k, best_score = None, -1e18
            for k in K_GRID:
                gp, _, _ = run_volnorm_scalar(pp, k)
                bm = metrics(pp); sm = metrics(pp*gp)
                if bm["maxDD"] >= 0:
                    continue
                ddred = 1-sm["maxDD"]/bm["maxDD"]
                cost = (1-sm["tot"]/bm["tot"]) if bm["tot"] != 0 else 1.0
                if cost <= 0.25 and ddred > best_score:
                    best_score, best_k = ddred, k
            if best_k is None:
                best_k = max(K_GRID)               # nothing met budget -> deepest k (least intrusive)
            pf = base[fut]
            gf, _, _ = run_volnorm_scalar(pf, best_k)
            oos_b.append(pf); oos_s.append(pf*gf); chosen.append((int(folds[i]), best_k))
        ob = metrics(np.concatenate(oos_b)); os_ = metrics(np.concatenate(oos_s))
        red = (1-os_["maxDD"]/ob["maxDD"])*100 if ob["maxDD"] < 0 else 0.0
        cost = (1-os_["tot"]/ob["tot"])*100 if ob["tot"] != 0 else np.nan
        ok = (red > 5.0) and np.isfinite(cost) and (cost < 40.0)
        r6_pass[name] = ok
        print(f"\n  {name}: nested-OOS chosen k per fold: {chosen}", flush=True)
        print(f"    OOS base maxDD {ob['maxDD']:+.0f}  Sharpe {ob['Sharpe']:+.2f}  totPnL {ob['tot']:+.0f}",
              flush=True)
        print(f"    OOS stop maxDD {os_['maxDD']:+.0f}  Sharpe {os_['Sharpe']:+.2f}  totPnL {os_['tot']:+.0f}",
              flush=True)
        print(f"    => forward ddRed {red:+.1f}% at forward totCost {cost:+.1f}%  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    n_pass = sum(r6_pass.values())
    print(f"\n  R6 SUMMARY: k-family PASSES nested-OOS on {n_pass}/3 universes "
          f"({', '.join(n for n,v in r6_pass.items() if v) or 'none'})  "
          f"{'<-- PORTABLE (target 3/3)' if n_pass==3 else ''}", flush=True)

    # ============================================================ R7 re-entry sanity (recommended k)
    print("\n" + "="*120, flush=True)
    print(f"R7 — re-entry sanity at the RECOMMENDED config (k={REC_K}, g_floor={GFLOOR}, heal={HEAL}, "
          f"timeout={RDAYS}, warmup={WARMUP}). HL70 @4.5bps", flush=True)
    print("="*120, flush=True)
    U = panels["HL70"]["U"]; base = panels["HL70"]["base"][PRIMARY_COST*1e4]
    pnl, gross, stop, rt, _ = run_volnorm_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, REC_K)
    m = metrics(pnl*1e4); bm = metrics(base)
    reentries = int(((gross[1:] == 1.0) & (gross[:-1] == GFLOOR)).sum())
    print(f"  round-trips (stop firings): {rt}; re-entries (gross floor->1): {reentries}; "
          f"%time stopped: {stop.mean()*100:.1f}%", flush=True)
    print(f"  g_floor={GFLOOR}>0 so equity keeps healing while stopped -> no frozen-equity permanent kill;",
          flush=True)
    print(f"  eq>trough guard -> never buys back AT the trough.", flush=True)
    print(f"  RECOMMENDED HL70: base maxDD {bm['maxDD']:+.0f} -> {m['maxDD']:+.0f} "
          f"(ddRed {(1-m['maxDD']/bm['maxDD'])*100:+.1f}%), Sharpe {bm['Sharpe']:+.2f}->{m['Sharpe']:+.2f}, "
          f"Calmar {bm['Calmar']:+.2f}->{m['Calmar']:+.2f}, totPnL {bm['tot']:+.0f}->{m['tot']:+.0f} "
          f"(cost {(1-m['tot']/bm['tot'])*100:+.1f}%)", flush=True)

    print(f"\nartifacts: X125_tradeoff_curve.parquet, X125_r4_const_degross.parquet, X125_r5_episodes.parquet",
          flush=True)
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
