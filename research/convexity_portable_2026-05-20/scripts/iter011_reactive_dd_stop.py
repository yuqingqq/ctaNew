"""iter-011 — REACTIVE damage-control layer (NOT prediction): cap the drawdown TAIL.

Human directive: stop trying to PREDICT the alt-bear (iters 5-10 proved no free signal leads).
Instead build a MECHANICAL, REACTIVE stop that de-grosses when a drawdown is ALREADY HAPPENING
and caps the tail for live capital preservation — accepting it cannot forecast and will whipsaw.

Reactive-track gates (evaluation_contract.md R1-R7):
  R1 look-ahead  — equity/flag uses only realized-to-(t-1) info, lagged. PIT.
  R2 tail reduce — meaningfully cut maxDD vs baseline on HL70.
  R3 bounded cost— state the return/Sharpe give-up; report the full DD-vs-cost trade-off curve.
  R4 vs const-de-gross — does triggering ON the DD cut the LEFT TAIL better than a CONSTANT flat
                         gross reduction of EQUAL AVERAGE EXPOSURE? (the reframed G4)
  R5 cross-episode (DECISIVE) — same mechanical rule caps DD across EXT episodes + S44; episode-LOFO.
  R6 nested-OOS  — choose threshold on past folds, apply forward; forward DD-capping must hold.
  R7 re-entry sanity — defined re-entry, no buy-back-at-top pathology.

Mechanisms (all PIT, mechanical, with a defined re-entry rule):
  (a) EQUITY-DRAWDOWN STOP: track running cum-equity on realized pnl_base; when current equity is in a
      drawdown >= X bps below its running peak (computed from data through t-1), de-gross the book for
      cycle t to a floor g_floor (0 = full stop). Re-enter (gross->1) when equity recovers to within
      (1-recover_frac)*peak-gap of the peak. Sweep deep thresholds X (fires ONLY in catastrophic tails).
  (b) FAST-FLAG de-risk: use iter-010 fast crash-onset metrics (flag ~21d earlier) to de-gross during a
      crash; re-enter when the fast metric normalizes. Accept whipsaw.
  (c) COMBO: fast-flag ARMS the stop; equity-DD CONFIRMS (de-gross only when both fire).

For each, on HL70 + EXT(multi-episode) + S44: measure the DD-vs-COST trade-off (maxDD reduction,
totPnL/Sharpe/Calmar cost, %time-in-stop, # whipsaw round-trips). Plus R4 constant-de-gross control,
R5 episode-LOFO, R6 nested-OOS of the threshold.

Reuses the X123 held-book per-cycle panels verbatim (pnl_base = production held-book per-cycle return on
constant gross notional; ×1e4 = bps; equity additive). Modifies nothing prior.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
RES = REPO/"research/convexity_portable_2026-05-20/results"
PANEL = {nm: RES/f"X123_altbear_short_{nm}.parquet" for nm in ("HL70", "S44", "EXT")}
FASTP = RES/"iter010_fast_metrics_EXT.parquet"
SEED = 12345
N_PLACEBO = 200
ANN = 6*365  # cycles/yr (4h horizon, held-book per-cycle)
# default reactive re-entry policy (sane, avoids frozen-equity permanent kill):
GFLOOR = 0.4     # de-gross to 40% gross (keep participating so equity can heal -> re-entry possible)
HEAL = 0.5       # re-enter once equity heals half the drawdown back toward the peak
RDAYS = 90       # OR after 90 bars (~15d) time fail-safe

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


def metrics(pnl_bps):
    pb = pd.Series(pnl_bps).dropna().values
    if len(pb) < 3:
        return dict(n=len(pb), tot=np.nan, maxDD=np.nan, Sharpe=np.nan, Calmar=np.nan)
    eq = np.cumsum(pb)
    dd = eq - np.maximum.accumulate(eq)
    mdd = dd.min()
    sh = pb.mean()/pb.std()*np.sqrt(ANN) if pb.std() > 0 else np.nan
    cal = pb.mean()*ANN/abs(mdd) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return dict(n=len(pb), tot=eq[-1], maxDD=mdd, Sharpe=sh, Calmar=cal)


def equity_dd_gross(pnl_bps, X_bps, g_floor=0.4, reenter_heal=0.5, reenter_days=None):
    """REACTIVE equity-drawdown stop with a SANE re-entry rule. PIT: the gross applied to cycle t is
    decided from the equity curve THROUGH t-1 (running peak & DD computed on realized pnl to t-1).

    Mechanics: de-gross to g_floor while in-stop. Trigger when DD-from-peak (through t-1) >= X_bps.
    The book KEEPS participating at g_floor>0 so its equity can heal (avoids the frozen-equity
    permanent-kill pathology of g_floor=0).
    Re-entry (gross->1) when EITHER:
      - equity has healed `reenter_heal` of the gap from the stop-time trough back toward the peak
        (heal=0.5 => recovered half the drawdown), OR
      - `reenter_days` (in 4h bars) have elapsed since the stop fired (time-based fail-safe).
    Returns (gross_series, in_stop_bool, n_roundtrips)."""
    n = len(pnl_bps)
    gross = np.ones(n)
    in_stop = np.zeros(n, dtype=bool)
    eq = 0.0
    peak = 0.0
    stopped = False
    roundtrips = 0
    stop_peak = 0.0
    stop_t = 0
    trough = 0.0
    for t in range(n):
        dd = eq - peak  # <= 0, computed on equity through t-1
        if not stopped:
            if -dd >= X_bps:
                stopped = True
                roundtrips += 1
                stop_peak = peak
                stop_t = t
                trough = eq
        else:
            trough = min(trough, eq)
            healed = (eq - trough) >= reenter_heal * (stop_peak - trough) and (stop_peak - trough) > 0
            timed = (reenter_days is not None) and ((t - stop_t) >= reenter_days)
            # only re-enter once the worst is past (eq above trough) — never buy back AT the trough
            if (healed and eq > trough) or timed:
                stopped = False
        if stopped:
            gross[t] = g_floor
            in_stop[t] = True
        eq += gross[t] * pnl_bps[t]
        if eq > peak:
            peak = eq
    return gross, in_stop, roundtrips


def flag_gross(flag_bool, g_floor=0.0):
    """Fast-flag de-risk: gross = g_floor while flag (already PIT .shift(1)-lagged in iter010) is True."""
    g = np.ones(len(flag_bool))
    g[flag_bool] = g_floor
    rt = int(((~np.r_[[False], flag_bool[:-1]]) & flag_bool).sum())  # entries into flag
    return g, flag_bool.copy(), rt


def apply_gross(pnl_bps, gross):
    return pnl_bps * gross


def left_tail(pnl_bps, q=0.01):
    """sum of the worst-q cumulative drawdown info: report maxDD and CVaR (mean of worst q% cycles)."""
    pb = np.asarray(pnl_bps)
    k = max(1, int(len(pb)*q))
    worst = np.sort(pb)[:k]
    return worst.mean()


def run_panel(name, X_grid, g_floor=GFLOOR):
    p = pd.read_parquet(PANEL[name]).sort_values("open_time").reset_index(drop=True)
    pnl = (p["pnl_base"].fillna(0.0).values)*1e4  # bps
    base = metrics(pnl)
    rows = []
    for X in X_grid:
        g, stop, rt = equity_dd_gross(pnl, X, g_floor=g_floor, reenter_heal=HEAL, reenter_days=RDAYS)
        m = metrics(apply_gross(pnl, g))
        rows.append(dict(X=X, **{f"m_{k}": v for k, v in m.items()},
                         pct_stop=stop.mean()*100, roundtrips=rt,
                         avg_gross=g.mean(),
                         ddRed=(1-m["maxDD"]/base["maxDD"])*100,
                         totCost=(1-m["tot"]/base["tot"])*100 if base["tot"] != 0 else np.nan))
    return base, pd.DataFrame(rows), pnl, p


def const_degross_matched(pnl_bps, avg_gross):
    """R4 control: a CONSTANT flat gross = avg_gross applied to every cycle (equal average exposure)."""
    return metrics(pnl_bps*avg_gross)


def main():
    rng = np.random.default_rng(SEED)
    # deep-tail thresholds (bps off running peak). HL70 baseline maxDD ~5674 bps.
    X_GRID = [800, 1200, 1600, 2000, 2500, 3000, 4000]

    print("="*120)
    print(f"iter-011 REACTIVE EQUITY-DRAWDOWN STOP — DD-vs-COST trade-off")
    print(f"  X = de-gross to g_floor={GFLOOR} when equity is >= X bps below its running peak (PIT, through t-1);")
    print(f"  re-enter (gross=1) when equity heals {HEAL:.0%} of the DD OR after {RDAYS} bars (~{RDAYS//6}d).")
    print("="*120)

    panel_base = {}
    panel_curve = {}
    panel_pnl = {}
    panel_df = {}
    for name in ("HL70", "EXT", "S44"):
        base, curve, pnl, pdf = run_panel(name, X_GRID, g_floor=GFLOOR)
        panel_base[name] = base; panel_curve[name] = curve; panel_pnl[name] = pnl; panel_df[name] = pdf
        print(f"\n--- {name}: baseline Sharpe {base['Sharpe']:+.2f} maxDD {base['maxDD']:+.0f} "
              f"Calmar {base['Calmar']:+.2f} totPnL {base['tot']:+.0f} ---")
        print(f"{'X_bps':>7}{'maxDD':>9}{'ddRed%':>8}{'totPnL':>9}{'totCost%':>9}"
              f"{'Sharpe':>8}{'Calmar':>8}{'%stop':>7}{'RT':>5}{'avgG':>6}")
        for _, r in curve.iterrows():
            print(f"{r['X']:>7.0f}{r['m_maxDD']:>9.0f}{r['ddRed']:>8.1f}{r['m_tot']:>9.0f}"
                  f"{r['totCost']:>9.1f}{r['m_Sharpe']:>8.2f}{r['m_Calmar']:>8.2f}"
                  f"{r['pct_stop']:>7.1f}{int(r['roundtrips']):>5}{r['avg_gross']:>6.2f}")

    # ---------------- R4: trigger-ON-DD vs CONSTANT de-gross of equal average exposure ----------------
    print("\n" + "="*120)
    print("R4 — STOP (triggered) vs CONSTANT de-gross of EQUAL AVERAGE EXPOSURE (does the stop cut the")
    print("  LEFT TAIL better than just always running smaller at the same avg gross?)")
    print("="*120)
    for name in ("HL70", "EXT", "S44"):
        base = panel_base[name]; pnl = panel_pnl[name]; curve = panel_curve[name]
        print(f"\n--- {name} (base maxDD {base['maxDD']:+.0f}, base CVaR1%/cyc {left_tail(pnl):+.1f}) ---")
        print(f"{'X_bps':>7}{'avgG':>6}{'STOP_maxDD':>11}{'CONST_maxDD':>12}{'STOP-CONST':>11}"
              f"{'STOP_tot':>9}{'CONST_tot':>10}  read")
        for _, r in curve.iterrows():
            g, stop, rt = equity_dd_gross(pnl, r["X"], g_floor=GFLOOR, reenter_heal=HEAL, reenter_days=RDAYS)
            ag = g.mean()
            stop_m = metrics(apply_gross(pnl, g))
            const_m = const_degross_matched(pnl, ag)
            better = stop_m["maxDD"] > const_m["maxDD"]  # less-negative = better tail cap
            print(f"{r['X']:>7.0f}{ag:>6.2f}{stop_m['maxDD']:>11.0f}{const_m['maxDD']:>12.0f}"
                  f"{stop_m['maxDD']-const_m['maxDD']:>+11.0f}{stop_m['tot']:>9.0f}{const_m['tot']:>10.0f}"
                  f"  {'STOP better tail' if better else 'const matches/better'}")

    # ---------------- R5: cross-episode tail-capping on the EXT multi-episode panel + episode-LOFO ----
    print("\n" + "="*120)
    print("R5 (DECISIVE) — does ONE mechanical rule cap DD across MULTIPLE EXT episodes? per-episode maxDD")
    print("  (stop fires globally on the running EXT equity; we report the maxDD WITHIN each episode window)")
    print("="*120)
    pdf = panel_df["EXT"]; pnl = panel_pnl["EXT"]
    ot = pd.to_datetime(pdf["open_time"], utc=True).values
    # choose a representative deep X to characterize per-episode, plus show a couple
    for X in [1600, 2000, 2500]:
        g, stop, rt = equity_dd_gross(pnl, X, g_floor=GFLOOR, reenter_heal=HEAL, reenter_days=RDAYS)
        gpnl = apply_gross(pnl, g)
        print(f"\n  X={X} bps (avg gross {g.mean():.2f}, {stop.mean()*100:.1f}% time stopped, {rt} round-trips):")
        print(f"    {'episode':<13}{'base_maxDD':>11}{'stop_maxDD':>11}{'ddRed%':>8}{'base_tot':>10}{'stop_tot':>10}")
        capped = 0
        for nm, s, e in EPISODES:
            m = (ot >= np.datetime64(s)) & (ot <= np.datetime64(e))
            if m.sum() < 40:
                continue
            bm = metrics(pnl[m]); sm = metrics(gpnl[m])
            red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
            if red >= 10:
                capped += 1
            print(f"    {nm:<13}{bm['maxDD']:>11.0f}{sm['maxDD']:>11.0f}{red:>8.1f}"
                  f"{bm['tot']:>10.0f}{sm['tot']:>10.0f}")
        print(f"    => episodes with >=10% maxDD reduction: {capped}/{sum(1 for nm,s,e in EPISODES if ((ot>=np.datetime64(s))&(ot<=np.datetime64(e))).sum()>=40)}")

    # episode-LOFO: drop each episode, does the GLOBAL ddRed survive? (recompute stop on the held-out series)
    print("\n  EXT episode-LOFO (drop each episode, recompute stop on remaining series, X=2000):")
    Xl = 2000
    g_all, st_all, _ = equity_dd_gross(pnl, Xl, GFLOOR, HEAL, RDAYS)
    base_all = metrics(pnl); stop_all = metrics(apply_gross(pnl, g_all))
    print(f"    ALL: base maxDD {base_all['maxDD']:+.0f} -> stop {stop_all['maxDD']:+.0f} "
          f"(ddRed {(1-stop_all['maxDD']/base_all['maxDD'])*100:+.1f}%)")
    for nm, s, e in EPISODES:
        keep = ~((ot >= np.datetime64(s)) & (ot <= np.datetime64(e)))
        pk = pnl[keep]
        if len(pk) < 100:
            continue
        gk, _, _ = equity_dd_gross(pk, Xl, GFLOOR, HEAL, RDAYS)
        bm = metrics(pk); sm = metrics(apply_gross(pk, gk))
        red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
        print(f"    drop {nm:<12}: base maxDD {bm['maxDD']:+.0f} -> stop {sm['maxDD']:+.0f} ddRed {red:+.1f}%  "
              f"totCost {(1-sm['tot']/bm['tot'])*100:+.1f}%")

    # ---------------- R6: nested-OOS of the threshold (choose X on PAST folds, apply forward) ----------
    print("\n" + "="*120)
    print("R6 — NESTED-OOS of the threshold: pick X on PAST folds (best in-sample ddRed-per-cost), apply")
    print("  to the NEXT fold; measure the realized FORWARD maxDD-capping on HL70 (the production universe)")
    print("="*120)
    for name in ("HL70", "EXT"):
        pdf = panel_df[name]; pnl = panel_pnl[name]
        folds = sorted(pd.unique(pdf["fold"].dropna()))
        fold_arr = pdf["fold"].values
        # objective for picking X on past data: maximize ddRed while totCost <= 25% (a risk-pref budget)
        oos_pnl_base = []
        oos_pnl_stop = []
        chosen = []
        for i in range(1, len(folds)):
            past = np.isin(fold_arr, folds[:i])
            fut = fold_arr == folds[i]
            pp = pnl[past]
            best_X, best_score = None, -1e18
            for X in X_GRID:
                gp, _, _ = equity_dd_gross(pp, X, GFLOOR, HEAL, RDAYS)
                bm = metrics(pp); sm = metrics(apply_gross(pp, gp))
                if bm["maxDD"] >= 0:
                    continue
                ddred = (1-sm["maxDD"]/bm["maxDD"])
                cost = (1-sm["tot"]/bm["tot"]) if bm["tot"] != 0 else 1.0
                if cost <= 0.25:
                    score = ddred  # maximize tail cap under a 25% cost budget
                    if score > best_score:
                        best_score, best_X = score, X
            if best_X is None:
                best_X = max(X_GRID)  # nothing met budget -> the deepest (least intrusive) threshold
            # apply chosen X to the FUTURE fold (stop state computed on the future fold's own equity)
            pf = pnl[fut]
            gf, _, _ = equity_dd_gross(pf, best_X, GFLOOR, HEAL, RDAYS)
            oos_pnl_base.append(pf)
            oos_pnl_stop.append(apply_gross(pf, gf))
            chosen.append((folds[i], best_X))
        ob = metrics(np.concatenate(oos_pnl_base)); os_ = metrics(np.concatenate(oos_pnl_stop))
        red = (1-os_["maxDD"]/ob["maxDD"])*100 if ob["maxDD"] < 0 else 0.0
        print(f"\n  {name}: nested-OOS chosen X per fold: {chosen}")
        print(f"    OOS base   maxDD {ob['maxDD']:+.0f}  Sharpe {ob['Sharpe']:+.2f}  totPnL {ob['tot']:+.0f}")
        print(f"    OOS stop   maxDD {os_['maxDD']:+.0f}  Sharpe {os_['Sharpe']:+.2f}  totPnL {os_['tot']:+.0f}")
        print(f"    => forward ddRed {red:+.1f}%, forward totCost "
              f"{(1-os_['tot']/ob['tot'])*100:+.1f}%")

    # ---------------- ARM (b): FAST-FLAG de-risk (iter-010 metrics) on EXT ----------------
    print("\n" + "="*120)
    print("ARM (b) — FAST-FLAG reactive de-risk (iter-010 onset metrics flag ~21d earlier). EXT panel.")
    print("  de-gross to 0 while flag fires; re-enter when it normalizes. (PIT: metrics already .shift(1))")
    print("="*120)
    fp = pd.read_parquet(FASTP).sort_values("open_time").reset_index(drop=True)
    fpnl = fp["pnl_base"].fillna(0.0).values*1e4
    fbase = metrics(fpnl)
    print(f"  EXT base: maxDD {fbase['maxDD']:+.0f} Sharpe {fbase['Sharpe']:+.2f} totPnL {fbase['tot']:+.0f}")
    print(f"  {'flag':<20}{'maxDD':>9}{'ddRed%':>8}{'totPnL':>9}{'totCost%':>9}{'Sharpe':>8}"
          f"{'%stop':>7}{'RT':>5}  vsCONST(maxDD)")
    FAST_DEFS = {
        "alt_1d": ("lt", 0.0), "alt_3d": ("lt", 0.0), "alt_7d": ("lt", 0.0),
        "alt_dd20": ("lt", -0.05), "alt_rvol_spike": ("gt", 1.25),
        "breadth_below_ma7": ("gt", 0.6), "alt_accel_3d": ("lt", 0.0),
    }
    for k, (mode, thr) in FAST_DEFS.items():
        s = fp[k]
        flag = (s < thr).values if mode == "lt" else (s > thr).values
        flag = np.where(np.isnan(s.values), False, flag)
        g, stop, rt = flag_gross(flag, 0.0)
        m = metrics(apply_gross(fpnl, g))
        cm = const_degross_matched(fpnl, g.mean())
        red = (1-m["maxDD"]/fbase["maxDD"])*100
        print(f"  {k:<20}{m['maxDD']:>9.0f}{red:>8.1f}{m['tot']:>9.0f}"
              f"{(1-m['tot']/fbase['tot'])*100:>9.1f}{m['Sharpe']:>8.2f}{stop.mean()*100:>7.1f}{rt:>5}"
              f"  const {cm['maxDD']:+.0f} ({'flag better' if m['maxDD']>cm['maxDD'] else 'const>=flag'})")

    # ---------------- ARM (c): COMBO — fast-flag ARMS, equity-DD CONFIRMS ----------------
    print("\n" + "="*120)
    print("ARM (c) — COMBO: de-gross only when BOTH a fast-flag is on AND equity is in a DD>=X. (EXT)")
    print("="*120)
    # use the best-leading fast flag (breadth_below_ma7) as the arm; X confirms.
    arm = fp["breadth_below_ma7"]
    armflag = np.where(np.isnan(arm.values), False, (arm > 0.6).values)
    for X in [800, 1200, 1600]:
        # equity-DD stop but only allowed to ARM when armflag is true at decision time; SANE re-entry
        # (heal/timed) and g_floor=GFLOOR — same policy as arm (a) for a fair characterization.
        n = len(fpnl); gross = np.ones(n); eq = 0.0; peak = 0.0; stopped = False; rt = 0; instop = np.zeros(n, bool)
        stop_peak = 0.0; trough = 0.0; stop_t = 0
        for t in range(n):
            dd = eq - peak
            if not stopped:
                if (-dd >= X) and armflag[t]:
                    stopped = True; rt += 1; stop_peak = peak; trough = eq; stop_t = t
            else:
                trough = min(trough, eq)
                healed = (eq - trough) >= HEAL*(stop_peak - trough) and (stop_peak - trough) > 0
                timed = (t - stop_t) >= RDAYS
                if (healed and eq > trough) or timed:
                    stopped = False
            if stopped:
                gross[t] = GFLOOR; instop[t] = True
            eq += gross[t]*fpnl[t]
            if eq > peak:
                peak = eq
        m = metrics(apply_gross(fpnl, gross))
        cm = const_degross_matched(fpnl, gross.mean())
        red = (1-m["maxDD"]/fbase["maxDD"])*100
        print(f"  X={X:>5} armed by breadth: maxDD {m['maxDD']:+.0f} (ddRed {red:+.1f}%) totPnL {m['tot']:+.0f} "
              f"(cost {(1-m['tot']/fbase['tot'])*100:+.1f}%) Sharpe {m['Sharpe']:+.2f} %stop {instop.mean()*100:.1f} RT {rt} "
              f"| const {cm['maxDD']:+.0f}")

    # ---------------- R4-strict placebo: stop vs RANDOM de-gross masks of matched %-time + matched avg gross
    print("\n" + "="*120)
    print("R4-PLACEBO — equity-DD STOP vs RANDOM de-gross of MATCHED %-time (200 seeds): does triggering ON")
    print("  the realized DD cut the LEFT TAIL (maxDD) better than random de-gross of equal exposure? HL70+EXT")
    print("="*120)
    for name in ("HL70", "EXT"):
        pnl = panel_pnl[name]; base = panel_base[name]
        for X in [1600, 2000]:
            g, stop, rt = equity_dd_gross(pnl, X, GFLOOR, HEAL, RDAYS)
            real_m = metrics(apply_gross(pnl, g))
            n_stop = int(stop.sum())
            mdds = []
            for _ in range(N_PLACEBO):
                pick = rng.choice(len(pnl), size=n_stop, replace=False)
                gg = np.ones(len(pnl)); gg[pick] = 0.0
                mdds.append(metrics(apply_gross(pnl, gg))["maxDD"])
            mdds = np.array(mdds)
            # rank: how often does the real stop cap the tail BETTER (less negative maxDD) than random?
            rank = (real_m["maxDD"] > mdds).mean()*100
            print(f"  {name} X={X}: real maxDD {real_m['maxDD']:+.0f} ({n_stop} stopped); "
                  f"random matched-%time maxDD p50 {np.percentile(mdds,50):+.0f} p95(best) "
                  f"{np.percentile(mdds,95):+.0f} -> real ranks p{rank:.0f} "
                  f"{'PASS (DD-triggered beats random)' if rank>=95 else 'mixed/FAIL'}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
