"""X124 — iter-011 CANONICAL: REACTIVE equity-drawdown stop (mechanical, PIT, NOT prediction).

Reactive risk-control track (NOT alpha). iters 5-10 proved no free signal PREDICTS the alt-bear; this
iteration builds a MECHANICAL, REACTIVE damage-control layer that de-grosses the live book when a
drawdown is ALREADY underway and caps the tail for capital preservation — accepting it cannot forecast
and will whipsaw. Deliverable = a characterized DD-vs-cost trade-off curve + the recommended config,
graded against the reactive-track gates R1-R7 (shared/evaluation_contract.md).

RECOMMENDED CONFIG (research handoff):
  Equity-drawdown stop. De-gross the whole held book to gross = g_floor=0.40 when the strategy's OWN
  cumulative equity is >= X=1600 bps below its running peak (equity/peak/DD computed through t-1, PIT).
  Re-enter (gross -> 1.0) when equity heals 50% of the drawdown back toward the peak (and is above the
  trough — never buy back AT the trough) OR after 90 bars (~15d) as a time fail-safe.

WHY THIS SCRIPT (vs the research probe iter011_reactive_dd_stop.py):
  The research probe scaled the already-cost-netted per-cycle `pnl_base` scalar by gross, which scales
  cost linearly (an approximation). This CANONICAL version rebuilds the held book from per-symbol
  weights and applies the gross scale to the POSITIONS BEFORE turnover/cost, so turnover (hence cost)
  is recomputed exactly under the time-varying gross — and it runs at cost {1,3,4.5}bps. This is the
  correct accounting the evaluation contract needs (R3 cost realism / G8).

PIT (R1):  the gross applied to cycle t is decided from equity THROUGH t-1 (running peak & DD on
  realized pnl to t-1). The held book overlaps HOLD=6 sleeves, so the per-cycle realized PnL at t is
  itself the return of a book of sleeves entered t-HOLD+1..t; the equity we gate on is realized through
  t-1 (already known), so the trigger is strictly point-in-time. The ONLY parameter is the threshold X
  (flagged for R6 nested-OOS); g_floor / heal / timeout are fixed policy choices, not tuned.

Reuses X123's build_universe verbatim (preds + klines pipeline, X117 base book) and a gross-aware
held-book engine. Modifies NOTHING prior (no baseline scripts, no cached preds).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

# reuse the X123 universe/engine pipeline verbatim (X117 production base book) ----------------------
import importlib.util as _ilu

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
OUT = REPO/"research/convexity_portable_2026-05-20/results"

_spec = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(x123)
build_universe = x123.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
HOLD = x123.HOLD

SEED = 12345
N_PLACEBO = 200
ANN = 6*365                # cycles/yr (4h horizon, held-book per-cycle)
COSTS_BPS = [1.0, 3.0, 4.5]
PRIMARY_COST = 4.5e-4      # production calibration (X117 = +1.93 / -5674 at 4.5bps)

# fixed reactive re-entry policy (NOT tuned; avoids the frozen-equity permanent-kill pathology) ------
GFLOOR = 0.40              # de-gross to 40% gross (book keeps participating so equity can heal)
HEAL = 0.50                # re-enter once equity heals half the drawdown back toward the peak
RDAYS = 90                 # OR after 90 bars (~15d) time fail-safe

# DD-vs-cost trade-off sweep (deep thresholds: fire ONLY in catastrophic tails) ----------------------
X_GRID = [800, 1200, 1600, 2000, 2500, 3000]
REC_X = 1600               # the recommended threshold

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


# --------------------------------------------------------------------------- metrics
def metrics(pnl_bps):
    pb = np.asarray(pnl_bps, dtype=np.float64)
    pb = pb[np.isfinite(pb)]
    if len(pb) < 3:
        return dict(n=len(pb), tot=np.nan, maxDD=np.nan, Sharpe=np.nan, Calmar=np.nan, cvar1=np.nan)
    eq = np.cumsum(pb)
    dd = eq - np.maximum.accumulate(eq)
    mdd = float(dd.min())
    sd = pb.std()
    sh = float(pb.mean()/sd*np.sqrt(ANN)) if sd > 0 else np.nan
    cal = float(pb.mean()*ANN/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    k = max(1, int(len(pb)*0.01))
    cvar1 = float(np.sort(pb)[:k].mean())
    return dict(n=len(pb), tot=float(eq[-1]), maxDD=mdd, Sharpe=sh, Calmar=cal, cvar1=cvar1)


# --------------------------------------------------------------------------- gross-aware held book
def heldbook_gross(cyc_w, rs, cost, gross):
    """X123 held-book engine with a PER-CYCLE GROSS SCALE applied to POSITIONS BEFORE turnover/cost.

    At cycle t the net book is the HOLD-sleeve average of cyc_w[t-HOLD+1..t], then scaled by gross[t].
    Turnover = sum_s |g_t*net_t(s) - g_{t-1}*net_{t-1}(s)| (recomputed exactly under time-varying gross,
    so when gross drops the de-grossing trade itself pays cost; cost is NOT just linearly scaled).
    pnl[t] = g_t * (book gross return) - turnover * 0.5 * cost.  Returns per-cycle net PnL (fraction)."""
    n = len(cyc_w)
    prev = {}                 # previous cycle's GROSS-SCALED net positions
    pnl = np.empty(n, dtype=np.float64)
    for t in range(n):
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items():
                net[s] = net.get(s, 0.0) + wt/HOLD
        g = gross[t]
        scaled = {s: g*v for s, v in net.items()}
        alls = set(scaled) | set(prev)
        turn = sum(abs(scaled.get(s, 0.0) - prev.get(s, 0.0)) for s in alls)
        rl = rs[t]
        c = sum(scaled.get(s, 0.0)*rl.get(s, 0.0) for s in scaled if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c):
            c = 0.0
        pnl[t] = c - turn*0.5*cost
        prev = scaled
    return pnl


def gross_unit(cyc_w, rs, cost):
    """Convenience: gross == 1.0 everywhere reproduces the X117/X123 base book."""
    return heldbook_gross(cyc_w, rs, cost, np.ones(len(cyc_w)))


# --------------------------------------------------------------------------- reactive equity-DD stop
def run_stop_on_scalar(pnl_base_bps, X_bps, g_floor=GFLOOR, heal=HEAL, timeout=RDAYS):
    """Forward state-machine on a SCALAR per-cycle base-PnL series (approx accounting: scales the
    already-cost-netted pnl). Used for R5 episode-LOFO / R6 nested-OOS where rebuilding the full held
    book per held-out slice is prohibitive. Equity is the strategy's OWN realized (scaled) equity."""
    pb = np.asarray(pnl_base_bps, dtype=np.float64)
    pb = np.where(np.isfinite(pb), pb, 0.0)
    n = len(pb)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool)
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    for t in range(n):
        dd = eq - peak                              # <= 0, equity through t-1 (PIT)
        if not stopped:
            if -dd >= X_bps:
                stopped = True; rt += 1; stop_peak = peak; trough = eq; stop_t = t
        else:
            trough = min(trough, eq)
            gap = stop_peak - trough
            healed = (gap > 0) and ((eq - trough) >= heal*gap)
            timed = (t - stop_t) >= timeout
            if (healed and eq > trough) or timed:
                stopped = False
        if stopped:
            gross[t] = g_floor; in_stop[t] = True
        eq += gross[t]*pb[t]
        if eq > peak:
            peak = eq
    return gross, in_stop, rt


def run_stop_heldbook(cyc_w, rs, cost, X_bps, g_floor=GFLOOR, heal=HEAL, timeout=RDAYS):
    """CANONICAL run: a single forward pass that (1) builds the de-grossed gross[t] from the strategy's
    OWN realized equity through t-1, and (2) computes the per-cycle realized PnL by scaling the book
    BEFORE turnover/cost (heldbook_gross mechanics inline). Because the equity depends on the gross and
    the gross depends on the equity, we resolve both in one causal pass: gross[t] is fixed from equity
    through t-1, then pnl[t] is realized, then equity advances. No look-ahead.
    Returns (pnl_stopped_bps, gross, in_stop, n_roundtrips)."""
    n = len(cyc_w)
    pnl = np.empty(n, dtype=np.float64)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool)
    prev = {}                                       # previous GROSS-SCALED net positions
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    for t in range(n):
        # ---- stop decision from equity THROUGH t-1 (PIT) ----
        dd = eq - peak
        if not stopped:
            if -dd >= X_bps:
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
        # ---- realize cycle t PnL: scale positions BEFORE turnover/cost ----
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
        # ---- advance equity (realized) ----
        eq += pnl[t]*1e4                            # bps equity
        if eq > peak:
            peak = eq
    return pnl, gross, in_stop, rt


def const_degross(pnl_base_bps, avg_gross):
    """R4 control: a CONSTANT flat gross applied to every cycle (scalar approx, equal AVERAGE exposure
    to the stop). For the held-book this is exact since constant gross scales positions AND turnover
    uniformly -> pnl_const = g * pnl_base."""
    return np.asarray(pnl_base_bps, dtype=np.float64)*avg_gross


# --------------------------------------------------------------------------- builds (cache the held books)
def build_panel(name, preds_path):
    print(f"\n[build] {name}", flush=True)
    U = build_universe(preds_path, name)
    base_by_cost = {cb: gross_unit(U["cyc"]["base"], U["rs"], cb*1e-4)*1e4 for cb in COSTS_BPS}
    return U, base_by_cost


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*120, flush=True)
    print("X124 — REACTIVE EQUITY-DRAWDOWN STOP (canonical: gross applied BEFORE turnover/cost)", flush=True)
    print(f"  de-gross to g_floor={GFLOOR} when own equity >= X bps below running peak (PIT, through t-1);", flush=True)
    print(f"  re-enter when equity heals {HEAL:.0%} of the DD OR after {RDAYS} bars (~{RDAYS//6}d). "
          f"costs {COSTS_BPS} bps.", flush=True)
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

    # ============================================================ R2/R3 DD-vs-COST TRADE-OFF CURVE
    print("\n" + "="*120, flush=True)
    print("R2/R3 — DD-vs-COST TRADE-OFF CURVE (sweep X x g_floor) per universe per cost", flush=True)
    print("  (gross applied to positions BEFORE turnover/cost; canonical held-book accounting)", flush=True)
    print("="*120, flush=True)
    curve_rows = []
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]
        for cb in COSTS_BPS:
            base = panels[name]["base"][cb]
            bm = metrics(base)
            print(f"\n--- {name} @ {cb:.1f}bps : base Sharpe {bm['Sharpe']:+.2f} maxDD {bm['maxDD']:+.0f} "
                  f"Calmar {bm['Calmar']:+.2f} totPnL {bm['tot']:+.0f} ---", flush=True)
            print(f"{'gfloor':>7}{'X_bps':>7}{'maxDD':>9}{'ddRed%':>8}{'totPnL':>9}{'totCost%':>9}"
                  f"{'Sharpe':>8}{'Calmar':>8}{'%stop':>7}{'RT':>5}{'avgG':>6}", flush=True)
            for gf in (0.0, 0.40):
                for X in X_GRID:
                    pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], cb*1e-4, X,
                                                             g_floor=gf)
                    pnl = pnl*1e4
                    m = metrics(pnl)
                    ddRed = (1-m["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else np.nan
                    totCost = (1-m["tot"]/bm["tot"])*100 if bm["tot"] != 0 else np.nan
                    rec = (gf == 0.40 and X == REC_X)
                    print(f"{gf:>7.2f}{X:>7.0f}{m['maxDD']:>9.0f}{ddRed:>8.1f}{m['tot']:>9.0f}"
                          f"{totCost:>9.1f}{m['Sharpe']:>8.2f}{m['Calmar']:>8.2f}"
                          f"{stop.mean()*100:>7.1f}{rt:>5}{gross.mean():>6.2f}"
                          f"{'  <-- RECOMMENDED' if rec else ''}", flush=True)
                    curve_rows.append(dict(universe=name, cost_bps=cb, g_floor=gf, X=X,
                                           maxDD=m["maxDD"], ddRed=ddRed, totPnL=m["tot"], totCost=totCost,
                                           Sharpe=m["Sharpe"], Calmar=m["Calmar"], cvar1=m["cvar1"],
                                           pct_stop=stop.mean()*100, roundtrips=rt, avg_gross=gross.mean(),
                                           base_maxDD=bm["maxDD"], base_totPnL=bm["tot"],
                                           base_Sharpe=bm["Sharpe"], base_Calmar=bm["Calmar"]))
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_parquet(OUT/"X124_tradeoff_curve.parquet", index=False)

    # ============================================================ R4 vs CONSTANT de-gross + placebo
    print("\n" + "="*120, flush=True)
    print("R4 — STOP (triggered) vs CONSTANT de-gross of EQUAL AVERAGE EXPOSURE: does triggering ON the", flush=True)
    print("  drawdown cut the LEFT TAIL better than just always running smaller? (recommended g_floor=0.40)", flush=True)
    print("="*120, flush=True)
    r4_rows = []
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]; base = panels[name]["base"][PRIMARY_COST*1e4]
        bm = metrics(base)
        print(f"\n--- {name} @4.5bps (base maxDD {bm['maxDD']:+.0f}, base CVaR1%/cyc {bm['cvar1']:+.2f}) ---",
              flush=True)
        print(f"{'X_bps':>7}{'avgG':>6}{'STOP_maxDD':>11}{'CONST_maxDD':>12}{'STOP-CONST':>11}"
              f"{'STOP_tot':>9}{'CONST_tot':>10}  read", flush=True)
        for X in X_GRID:
            pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, X)
            pnl = pnl*1e4
            ag = gross.mean()
            sm = metrics(pnl); cm = metrics(const_degross(base, ag))
            better = sm["maxDD"] > cm["maxDD"]      # less-negative = caps tail better
            print(f"{X:>7.0f}{ag:>6.2f}{sm['maxDD']:>11.0f}{cm['maxDD']:>12.0f}"
                  f"{sm['maxDD']-cm['maxDD']:>+11.0f}{sm['tot']:>9.0f}{cm['tot']:>10.0f}"
                  f"  {'STOP better tail' if better else 'const matches/better'}", flush=True)
            r4_rows.append(dict(universe=name, X=X, avg_gross=ag, stop_maxDD=sm["maxDD"],
                                const_maxDD=cm["maxDD"], stop_minus_const=sm["maxDD"]-cm["maxDD"],
                                stop_tot=sm["tot"], const_tot=cm["tot"]))
    pd.DataFrame(r4_rows).to_parquet(OUT/"X124_r4_const_degross.parquet", index=False)

    # ---- R4-PLACEBO: stop vs RANDOM de-gross of matched %-time + matched floor (200 seeds) ----
    print("\n" + "-"*120, flush=True)
    print("R4-PLACEBO — STOP vs RANDOM de-gross of MATCHED %-time (same #stopped cycles, same g_floor,", flush=True)
    print(f"  {N_PLACEBO} seeds): does triggering ON the realized DD cap maxDD better than random? HL70+EXT @4.5bps",
          flush=True)
    print("-"*120, flush=True)
    for name in ("HL70", "EXT"):
        U = panels[name]["U"]; base = panels[name]["base"][PRIMARY_COST*1e4]
        for X in (REC_X, 2000):
            pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, X)
            real_m = metrics(pnl*1e4)
            n_stop = int(stop.sum())
            mdds = np.empty(N_PLACEBO)
            for i in range(N_PLACEBO):
                pick = rng.choice(len(base), size=n_stop, replace=False)
                gg = np.ones(len(base)); gg[pick] = GFLOOR
                # scalar approx for placebo (random mask -> scale base pnl): matches research probe family
                mdds[i] = metrics(base*gg)["maxDD"]
            rank = float((real_m["maxDD"] > mdds).mean()*100)
            print(f"  {name} X={X}: real maxDD {real_m['maxDD']:+.0f} ({n_stop} stopped, {stop.mean()*100:.0f}% time); "
                  f"random matched p50 {np.percentile(mdds,50):+.0f} p95(best) {np.percentile(mdds,95):+.0f} "
                  f"-> real ranks p{rank:.0f} {'PASS' if rank>=95 else 'FAIL (proportional, not skill)'}",
                  flush=True)

    # ============================================================ R5 per-episode (EXT) + episode-LOFO
    print("\n" + "="*120, flush=True)
    print("R5 (DECISIVE) — cross-episode tail-capping (EXT multi-episode) + episode-LOFO of the DD-reduction", flush=True)
    print("  one mechanical rule on the running EXT equity; maxDD WITHIN each episode window. @4.5bps", flush=True)
    print("="*120, flush=True)
    U = panels["EXT"]["U"]
    ot = pd.to_datetime(pd.Series(U["times"]), utc=True).values
    base = panels["EXT"]["base"][PRIMARY_COST*1e4]
    r5_rows = []
    for X in (REC_X, 2000, 2500):
        pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, X)
        gpnl = pnl*1e4
        print(f"\n  X={X} bps (avg gross {gross.mean():.2f}, {stop.mean()*100:.1f}% time stopped, {rt} round-trips):",
              flush=True)
        print(f"    {'episode':<13}{'base_maxDD':>11}{'stop_maxDD':>11}{'ddRed%':>8}{'base_tot':>10}{'stop_tot':>10}",
              flush=True)
        capped = 0; n_ep = 0
        for nm, s, e in EPISODES:
            m = (ot >= np.datetime64(s)) & (ot <= np.datetime64(e))
            if m.sum() < 40:
                continue
            n_ep += 1
            bm = metrics(base[m]); sm = metrics(gpnl[m])
            red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
            if red >= 10:
                capped += 1
            print(f"    {nm:<13}{bm['maxDD']:>11.0f}{sm['maxDD']:>11.0f}{red:>8.1f}"
                  f"{bm['tot']:>10.0f}{sm['tot']:>10.0f}", flush=True)
            r5_rows.append(dict(X=X, episode=nm, base_maxDD=bm["maxDD"], stop_maxDD=sm["maxDD"], ddRed=red))
        print(f"    => episodes with >=10% maxDD reduction: {capped}/{n_ep}", flush=True)

    print("\n  EXT episode-LOFO (drop each episode, recompute the stop on the remaining series, X=2000):",
          flush=True)
    Xl = 2000
    pnl_all, _, _, _ = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, Xl)
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
        pnl_k, _, _, _ = run_stop_heldbook(cyc_k, rs_k, PRIMARY_COST, Xl)
        bm = metrics(base_k); sm = metrics(pnl_k*1e4)
        red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
        cost = (1-sm["tot"]/bm["tot"])*100 if bm["tot"] != 0 else np.nan
        print(f"    drop {nm:<12}: base maxDD {bm['maxDD']:+.0f} -> stop {sm['maxDD']:+.0f} "
              f"ddRed {red:+.1f}%  totCost {cost:+.1f}%", flush=True)

    # ============================================================ R6 nested-OOS of the threshold
    print("\n" + "="*120, flush=True)
    print("R6 — NESTED-OOS of the threshold X: pick X on PAST folds (max ddRed under <=25% cost budget),", flush=True)
    print("  apply to the NEXT fold; measure realized FORWARD maxDD-capping. HL70 (production) + EXT. @4.5bps",
          flush=True)
    print("  (uses scalar-approx accounting on the precomputed base pnl so each fold's stop is cheap)", flush=True)
    print("="*120, flush=True)
    for name in ("HL70", "EXT"):
        U = panels[name]["U"]; base = panels[name]["base"][PRIMARY_COST*1e4]
        fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        oos_b, oos_s, chosen = [], [], []
        for i in range(1, len(folds)):
            past = np.isin(fold_arr, folds[:i]); fut = fold_arr == folds[i]
            pp = base[past]
            best_X, best_score = None, -1e18
            for X in X_GRID:
                gp, _, _ = run_stop_on_scalar(pp, X)
                bm = metrics(pp); sm = metrics(pp*gp)
                if bm["maxDD"] >= 0:
                    continue
                ddred = 1-sm["maxDD"]/bm["maxDD"]
                cost = (1-sm["tot"]/bm["tot"]) if bm["tot"] != 0 else 1.0
                if cost <= 0.25 and ddred > best_score:
                    best_score, best_X = ddred, X
            if best_X is None:
                best_X = max(X_GRID)                # nothing met budget -> deepest (least intrusive)
            pf = base[fut]
            gf, _, _ = run_stop_on_scalar(pf, best_X)
            oos_b.append(pf); oos_s.append(pf*gf); chosen.append((int(folds[i]), best_X))
        ob = metrics(np.concatenate(oos_b)); os_ = metrics(np.concatenate(oos_s))
        red = (1-os_["maxDD"]/ob["maxDD"])*100 if ob["maxDD"] < 0 else 0.0
        cost = (1-os_["tot"]/ob["tot"])*100 if ob["tot"] != 0 else np.nan
        print(f"\n  {name}: nested-OOS chosen X per fold: {chosen}", flush=True)
        print(f"    OOS base maxDD {ob['maxDD']:+.0f}  Sharpe {ob['Sharpe']:+.2f}  totPnL {ob['tot']:+.0f}",
              flush=True)
        print(f"    OOS stop maxDD {os_['maxDD']:+.0f}  Sharpe {os_['Sharpe']:+.2f}  totPnL {os_['tot']:+.0f}",
              flush=True)
        print(f"    => forward ddRed {red:+.1f}% at forward totCost {cost:+.1f}%  "
              f"{'PASS' if red > 0 else 'FAIL (no forward cut)'}", flush=True)

    # ============================================================ R7 re-entry sanity (recommended config)
    print("\n" + "="*120, flush=True)
    print("R7 — re-entry sanity at the RECOMMENDED config (X=1600, g_floor=0.40, heal=0.5, timeout=90). HL70 @4.5bps",
          flush=True)
    print("="*120, flush=True)
    U = panels["HL70"]["U"]; base = panels["HL70"]["base"][PRIMARY_COST*1e4]
    pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, REC_X)
    m = metrics(pnl*1e4); bm = metrics(base)
    # count timeout vs heal re-entries by replaying the gross transitions
    reentries = int(((gross[1:] == 1.0) & (gross[:-1] == GFLOOR)).sum())
    print(f"  round-trips (stop firings): {rt}; re-entries (gross floor->1): {reentries}; "
          f"%time stopped: {stop.mean()*100:.1f}%", flush=True)
    print(f"  g_floor={GFLOOR}>0 so equity keeps healing while stopped -> no frozen-equity permanent kill.",
          flush=True)
    print(f"  RECOMMENDED HL70: base maxDD {bm['maxDD']:+.0f} -> {m['maxDD']:+.0f} "
          f"(ddRed {(1-m['maxDD']/bm['maxDD'])*100:+.1f}%), Sharpe {bm['Sharpe']:+.2f}->{m['Sharpe']:+.2f}, "
          f"Calmar {bm['Calmar']:+.2f}->{m['Calmar']:+.2f}, totPnL {bm['tot']:+.0f}->{m['tot']:+.0f} "
          f"(cost {(1-m['tot']/bm['tot'])*100:+.1f}%)", flush=True)

    print(f"\nartifacts: X124_tradeoff_curve.parquet, X124_r4_const_degross.parquet", flush=True)
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
