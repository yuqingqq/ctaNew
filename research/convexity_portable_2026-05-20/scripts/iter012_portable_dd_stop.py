"""iter-012 — PORTABLE / parameter-free REACTIVE equity-drawdown stop.

REACTIVE risk-control track (NOT alpha). iter-011 built a working equity-DD stop but its trigger is an
ABSOLUTE bps threshold (X=1600 bps off peak). Different universes have different natural equity
scales/vol, so one absolute X cannot be right everywhere -> iter-011 PASSED nested-OOS on HL70 but
FAILED on EXT (-7.3% ddRed / +44.8% cost: X drifts deep, barely fires). This iteration fixes that one
weakness by making the trigger SELF-NORMALIZING so a single UNITLESS parameter generalizes across
HL70 + EXT + S44 under nested-OOS, WITHOUT per-universe tuning.

Same fixed re-entry policy as iter-011 (g_floor=0.40, heal=0.50, timeout=90 bars) -> R7 carries over.
All triggers PIT: equity/peak/DD/vol computed through t-1, expanding/trailing only (R1).

PORTABLE TRIGGER FORMS (each has ONE unitless knob):
  (a) VOLNORM   — fire when (peak - eq) >= k * trailing_vol_of_equity_increments.  k in "sigmas of
                  equity" (unitless). Self-scales to each universe's own equity volatility.
  (b) PCTILE    — fire when current DD is in the worst q-quantile of the strategy's OWN expanding DD
                  distribution (PIT, expanding). q unitless (worst fraction).
  (c) MAXDDFRAC — fire when current DD > f * trailing-N-bar max DD (the strategy's own recent worst).
                  f unitless fraction.

Compared head-to-head against iter-011's ABSOLUTE-X trigger (the thing we're trying to beat on EXT).

Reuses X123 build_universe + X124 held-book mechanics verbatim (gross applied BEFORE turnover/cost).
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

_spec = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(x123)
build_universe = x123.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
HOLD = x123.HOLD

SEED = 12345
N_PLACEBO = 200
ANN = 6*365
COSTS_BPS = [1.0, 3.0, 4.5]
PRIMARY_COST = 4.5e-4

# fixed reactive re-entry policy (carried from iter-011, NOT tuned)
GFLOOR = 0.40
HEAL = 0.50
RDAYS = 90

# trailing windows for the self-normalizing stats (in bars; ~30d=180 bars, ~60d=360). Fixed policy,
# NOT a per-universe tuned param. The UNITLESS knob is k/q/f, not the window.
VOL_WIN = 180        # trailing window for equity-increment vol (~30d)
DD_WIN = 360         # trailing window for max-DD-frac reference (~60d)
MIN_OBS = 60         # warm-up before any self-normalizing trigger can fire (~10d)

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]

# UNITLESS parameter grids (these are the knobs that must generalize across universes)
K_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]            # volnorm: sigmas of equity
Q_GRID = [0.80, 0.85, 0.90, 0.95]             # pctile: worst-q of own expanding DD distribution
F_GRID = [0.60, 0.75, 0.90, 1.10]             # maxddfrac: fraction of trailing max DD
X_GRID_ABS = [1200, 1600, 2000, 2500, 3000]   # iter-011 absolute reference (the thing to beat)


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


# --------------------------------------------------------------------------- portable trigger predicate
def trigger_fires(form, param, eq, peak, eq_incr_hist, dd_hist):
    """PIT decision: should the stop ARM at this cycle given equity state THROUGH t-1?
      eq          : current equity (through t-1)
      peak        : running peak (through t-1)
      eq_incr_hist: list of realized equity increments (gross-scaled pnl bps) through t-1
      dd_hist     : list of realized DD-from-peak values (<=0) through t-1
    Returns True to ARM the stop. All inputs are realized-to-t-1 -> R1 PIT-clean."""
    dd = peak - eq                          # >= 0, current drawdown magnitude
    if dd <= 0:
        return False
    n = len(eq_incr_hist)
    if n < MIN_OBS:
        return False
    if form == "absolute":                  # iter-011 reference
        return dd >= param
    if form == "volnorm":                   # (a) k sigmas of trailing equity-increment vol
        hist = eq_incr_hist[-VOL_WIN:]
        sig = float(np.std(hist))
        if sig <= 0:
            return False
        # cumulative-DD scale ~ sigma * sqrt(window-ish); compare DD to k*sigma*sqrt(VOL_WIN)
        return dd >= param * sig * np.sqrt(min(n, VOL_WIN))
    if form == "pctile":                    # (b) worst-q of OWN expanding DD distribution
        # dd_hist holds <=0 DD values; current DD magnitude vs the q-quantile of past DD magnitudes
        mags = -np.asarray(dd_hist)         # >=0
        thr = float(np.quantile(mags, param))
        return dd >= thr and thr > 0
    if form == "maxddfrac":                 # (c) fraction of trailing-N-bar max DD
        recent = dd_hist[-DD_WIN:]
        mdd_recent = -min(recent) if recent else 0.0  # >=0
        if mdd_recent <= 0:
            return False
        return dd >= param * mdd_recent
    raise ValueError(form)


# --------------------------------------------------------------------------- canonical held-book stop
def run_stop_heldbook(cyc_w, rs, cost, form, param, g_floor=GFLOOR, heal=HEAL, timeout=RDAYS):
    """Single causal forward pass. gross[t] fixed from equity THROUGH t-1 (portable trigger), then
    pnl[t] realized by scaling positions BEFORE turnover/cost. Returns (pnl_frac, gross, in_stop, rt)."""
    n = len(cyc_w)
    pnl = np.empty(n, dtype=np.float64)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool)
    prev = {}
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    eq_incr_hist = []                       # realized equity increments (bps) through t-1
    dd_hist = []                            # realized DD-from-peak (<=0) through t-1
    for t in range(n):
        dd = eq - peak                      # <= 0
        if not stopped:
            if trigger_fires(form, param, eq, peak, eq_incr_hist, dd_hist):
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
        # realize cycle t
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
        # advance equity
        incr = pnl[t]*1e4
        eq += incr
        eq_incr_hist.append(incr)
        dd_hist.append(eq - peak)
        if eq > peak:
            peak = eq
    return pnl, gross, in_stop, rt


def run_stop_scalar(pnl_base_bps, form, param, g_floor=GFLOOR, heal=HEAL, timeout=RDAYS):
    """Scalar-approx accounting (scale already-cost-netted pnl) for cheap LOFO / nested-OOS slices.
    Returns (gross, in_stop, rt)."""
    pb = np.asarray(pnl_base_bps, dtype=np.float64)
    pb = np.where(np.isfinite(pb), pb, 0.0)
    n = len(pb)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool)
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    eq_incr_hist = []; dd_hist = []
    for t in range(n):
        if not stopped:
            if trigger_fires(form, param, eq, peak, eq_incr_hist, dd_hist):
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
        incr = g*pb[t]
        eq += incr
        eq_incr_hist.append(incr)
        dd_hist.append(eq - peak)
        if eq > peak:
            peak = eq
    return gross, in_stop, rt


def gross_unit(cyc_w, rs, cost):
    return run_stop_heldbook(cyc_w, rs, cost, "absolute", 1e18)[0]  # threshold never fires -> base


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*120, flush=True)
    print("iter-012 — PORTABLE / parameter-free REACTIVE equity-DD stop (self-normalizing triggers)", flush=True)
    print(f"  fixed policy g_floor={GFLOOR} heal={HEAL} timeout={RDAYS}; vol_win={VOL_WIN} dd_win={DD_WIN} warmup={MIN_OBS}",
          flush=True)
    print("="*120, flush=True)

    panels = {}
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        print(f"\n[build] {name}", flush=True)
        U = build_universe(pp, name)
        base_by_cost = {cb: gross_unit(U["cyc"]["base"], U["rs"], cb*1e-4)*1e4 for cb in COSTS_BPS}
        panels[name] = dict(U=U, base=base_by_cost)

    print("\n" + "="*120, flush=True)
    print("BASE REPRODUCTION (no stop) @4.5bps", flush=True)
    for name in ("HL70", "EXT", "S44"):
        m = metrics(panels[name]["base"][4.5])
        print(f"  {name:>5}  Sharpe {m['Sharpe']:+.2f}  maxDD {m['maxDD']:+.0f}  Calmar {m['Calmar']:+.2f}  "
              f"totPnL {m['tot']:+.0f}", flush=True)

    FORMS = [("absolute", X_GRID_ABS), ("volnorm", K_GRID), ("pctile", Q_GRID), ("maxddfrac", F_GRID)]

    # ============================================================ R2/R3 trade-off curve per form/universe
    print("\n" + "="*120, flush=True)
    print("R2/R3 — DD-vs-COST per trigger FORM x param x universe @4.5bps (held-book canonical)", flush=True)
    print("="*120, flush=True)
    curve_rows = []
    for form, grid in FORMS:
        print(f"\n##### FORM = {form} #####", flush=True)
        for name in ("HL70", "EXT", "S44"):
            U = panels[name]["U"]; base = panels[name]["base"][4.5]; bm = metrics(base)
            print(f"\n--- {name} (base maxDD {bm['maxDD']:+.0f} Calmar {bm['Calmar']:+.2f} "
                  f"tot {bm['tot']:+.0f}) ---", flush=True)
            print(f"{'param':>7}{'maxDD':>9}{'ddRed%':>8}{'totPnL':>9}{'cost%':>7}{'Sharpe':>8}"
                  f"{'Calmar':>8}{'%stop':>7}{'RT':>5}{'avgG':>6}", flush=True)
            for p in grid:
                pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, form, p)
                m = metrics(pnl*1e4)
                ddRed = (1-m["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else np.nan
                cost = (1-m["tot"]/bm["tot"])*100 if bm["tot"] != 0 else np.nan
                print(f"{p:>7.2f}{m['maxDD']:>9.0f}{ddRed:>8.1f}{m['tot']:>9.0f}{cost:>7.1f}"
                      f"{m['Sharpe']:>8.2f}{m['Calmar']:>8.2f}{stop.mean()*100:>7.1f}{rt:>5}{gross.mean():>6.2f}",
                      flush=True)
                curve_rows.append(dict(form=form, universe=name, param=p, maxDD=m["maxDD"], ddRed=ddRed,
                                       totPnL=m["tot"], cost=cost, Sharpe=m["Sharpe"], Calmar=m["Calmar"],
                                       pct_stop=stop.mean()*100, rt=rt, avg_gross=gross.mean(),
                                       base_maxDD=bm["maxDD"], base_tot=bm["tot"]))
    pd.DataFrame(curve_rows).to_parquet(OUT/"iter012_tradeoff.parquet", index=False)

    # ============================================================ R6 nested-OOS per form (THE TARGET)
    print("\n" + "="*120, flush=True)
    print("R6 — NESTED-OOS of the UNITLESS param per FORM: pick param on PAST folds (max ddRed under", flush=True)
    print("  <=25% cost budget), apply to NEXT fold. PASS = forward ddRed>0 AND cost bounded on BOTH HL70+EXT.",
          flush=True)
    print("  Self-normalizing form WINS if one form's param generalizes forward on every universe.", flush=True)
    print("="*120, flush=True)
    r6_rows = []
    for form, grid in FORMS:
        print(f"\n##### FORM = {form} #####", flush=True)
        for name in ("HL70", "EXT", "S44"):
            U = panels[name]["U"]; base = panels[name]["base"][4.5]
            fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
            folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
            oos_b, oos_s, chosen = [], [], []
            for i in range(1, len(folds)):
                past = np.isin(fold_arr, folds[:i]); fut = fold_arr == folds[i]
                pp = base[past]; bm = metrics(pp)
                best_p, best_score = None, -1e18
                for p in grid:
                    gp, _, _ = run_stop_scalar(pp, form, p)
                    sm = metrics(pp*gp)
                    if bm["maxDD"] >= 0:
                        continue
                    ddred = 1-sm["maxDD"]/bm["maxDD"]
                    cost = (1-sm["tot"]/bm["tot"]) if bm["tot"] != 0 else 1.0
                    if cost <= 0.25 and ddred > best_score:
                        best_score, best_p = ddred, p
                if best_p is None:
                    best_p = max(grid) if form != "pctile" else max(grid)  # least-intrusive end
                pf = base[fut]
                gf, _, _ = run_stop_scalar(pf, form, best_p)
                oos_b.append(pf); oos_s.append(pf*gf); chosen.append((int(folds[i]), best_p))
            ob = metrics(np.concatenate(oos_b)); os_ = metrics(np.concatenate(oos_s))
            red = (1-os_["maxDD"]/ob["maxDD"])*100 if ob["maxDD"] < 0 else 0.0
            cost = (1-os_["tot"]/ob["tot"])*100 if ob["tot"] != 0 else np.nan
            verdict = "PASS" if (red > 5 and cost < 40) else "FAIL"
            print(f"  {name}: chosen {chosen}", flush=True)
            print(f"    OOS base maxDD {ob['maxDD']:+.0f} -> stop {os_['maxDD']:+.0f}  "
                  f"forward ddRed {red:+.1f}% at cost {cost:+.1f}%  [{verdict}]", flush=True)
            r6_rows.append(dict(form=form, universe=name, ddRed=red, cost=cost, verdict=verdict))
    pd.DataFrame(r6_rows).to_parquet(OUT/"iter012_nested_oos.parquet", index=False)

    # R6 summary: which form passes on ALL THREE?
    r6 = pd.DataFrame(r6_rows)
    print("\n  --- R6 PORTABILITY SUMMARY (form passes if PASS on HL70 AND EXT AND S44) ---", flush=True)
    for form, _ in FORMS:
        sub = r6[r6.form == form]
        passes = (sub.verdict == "PASS").sum()
        allpass = passes == 3
        print(f"    {form:>10}: {passes}/3 universes PASS  {'<== PORTABLE' if allpass else ''}", flush=True)

    # ============================================================ R5 cross-episode + LOFO (best portable form)
    print("\n" + "="*120, flush=True)
    print("R5 — cross-episode tail-capping (EXT) + episode-LOFO, for EACH form at a MID param. @4.5bps", flush=True)
    print("="*120, flush=True)
    U = panels["EXT"]["U"]
    ot = pd.to_datetime(pd.Series(U["times"]), utc=True).values
    base = panels["EXT"]["base"][4.5]
    mid = {"absolute": 2000, "volnorm": 3.0, "pctile": 0.90, "maxddfrac": 0.90}
    r5_rows = []
    for form, _ in FORMS:
        p = mid[form]
        pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, form, p)
        gpnl = pnl*1e4
        print(f"\n  FORM={form} param={p} (avg gross {gross.mean():.2f}, {stop.mean()*100:.0f}% stopped, {rt} RT):",
              flush=True)
        capped = 0; nep = 0
        for nm, s, e in EPISODES:
            m = (ot >= np.datetime64(s)) & (ot <= np.datetime64(e))
            if m.sum() < 40:
                continue
            nep += 1
            bm = metrics(base[m]); sm = metrics(gpnl[m])
            red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
            if red >= 10:
                capped += 1
            print(f"    {nm:<13} base {bm['maxDD']:>+8.0f} -> stop {sm['maxDD']:>+8.0f}  ddRed {red:>+6.1f}%",
                  flush=True)
            r5_rows.append(dict(form=form, episode=nm, ddRed=red))
        print(f"    => {capped}/{nep} episodes >=10% capped", flush=True)
        # LOFO on EXT
        cyc_base = U["cyc"]["base"]; rs = U["rs"]
        lofo = []
        for nm, s, e in EPISODES:
            keep = ~((ot >= np.datetime64(s)) & (ot <= np.datetime64(e)))
            idx = np.where(keep)[0]
            if len(idx) < 100:
                continue
            cyc_k = [cyc_base[i] for i in idx]; rs_k = [rs[i] for i in idx]
            base_k = gross_unit(cyc_k, rs_k, PRIMARY_COST)*1e4
            pnl_k, _, _, _ = run_stop_heldbook(cyc_k, rs_k, PRIMARY_COST, form, p)
            bk = metrics(base_k); sk = metrics(pnl_k*1e4)
            red = (1-sk["maxDD"]/bk["maxDD"])*100 if bk["maxDD"] < 0 else 0.0
            lofo.append(f"drop {nm}:{red:+.0f}%")
        print(f"    LOFO: {'  '.join(lofo)}", flush=True)
    pd.DataFrame(r5_rows).to_parquet(OUT/"iter012_r5_episodes.parquet", index=False)

    # ============================================================ R4 vs constant de-gross (best portable form)
    print("\n" + "="*120, flush=True)
    print("R4 — STOP vs CONSTANT de-gross of EQUAL avg exposure (honesty gate; expect ~proportional).", flush=True)
    print("  STOP_maxDD - CONST_maxDD (positive = stop caps tail better). per FORM at mid param. @4.5bps", flush=True)
    print("="*120, flush=True)
    for form, _ in FORMS:
        p = mid[form]
        print(f"\n  FORM={form} param={p}:", flush=True)
        for name in ("HL70", "EXT", "S44"):
            U = panels[name]["U"]; base = panels[name]["base"][4.5]
            pnl, gross, stop, rt = run_stop_heldbook(U["cyc"]["base"], U["rs"], PRIMARY_COST, form, p)
            ag = gross.mean()
            sm = metrics(pnl*1e4); cm = metrics(base*ag)
            print(f"    {name:>5} avgG {ag:.2f}  STOP_maxDD {sm['maxDD']:>+8.0f}  CONST_maxDD {cm['maxDD']:>+8.0f}"
                  f"  STOP-CONST {sm['maxDD']-cm['maxDD']:>+7.0f}", flush=True)

    print(f"\nartifacts: iter012_tradeoff.parquet, iter012_nested_oos.parquet, iter012_r5_episodes.parquet",
          flush=True)
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
