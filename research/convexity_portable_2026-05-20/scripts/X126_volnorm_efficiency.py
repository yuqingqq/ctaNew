"""X126 — iter-013 EFFICIENCY variants of the portable vol-normalized reactive DD stop.

REACTIVE risk-control track. iter-012 (X125) shipped the PORTABLE vol-normalized stop:
  de-gross book to g_floor=0.40 when (peak-eq) >= k*sigma(trailing-180 equity incr)*sqrt(180),
  re-enter on 50%-heal OR 90-bar timeout. k=2.0 unitless. PASS R6 3/3, R5 LOFO, R4 ~proportional.
It WORKS but it CHURNS: ~15 de-gross/re-gross round-trips/402d, each pays turnover cost, 51% time
at reduced gross.

R4 already proved the tail-cap is ~PROPORTIONAL (you cannot SELECT the tail with skill). So we do NOT
try to beat proportional. The ONLY honest optimization is EFFICIENCY: cut the whipsaw / transaction
waste so you get the SAME DD-cap at LOWER return cost (Pareto-improve the trade-off curve).

EFFICIENCY VARIANTS (all PIT, unitless/parameter-free forms preferred to stay portable + not overfit):
  - V0  binary           : iter-012 baseline (de-gross to g_floor binary; heal=0.50).            [REFERENCE]
  - GRAD smooth de-gross : gross = continuous f(DD/trigger) ramping 1.0 -> g_floor between the
                           de-gross trigger and 2x trigger. Fewer hard 1.0<->0.40 round-trips,
                           turnover spread out / smaller. PARAMETER-FREE shape (linear ramp, ratio 2x).
  - HYST re-entry band   : de-gross at k*sigma, re-gross only after a FULLER heal (heal=0.90 instead
                           of 0.50) -> wider dead-band, fewer flip-flops. heal is the only knob (nested-OOS).
  - COOL min-hold        : after re-entry, cannot re-fire for `cool` bars (cooldown) -> cuts churn.
  - CONF confirmation lag: require the DD>=trigger condition to persist M bars before de-grossing
                           -> fewer false single-bar triggers.

Pareto metric: a variant Pareto-improves iff at the SAME (or more) maxDD-reduction it pays STRICTLY
LESS cost (less turnover give-up), robustly on HL70+EXT+S44, while keeping R5 (episode-LOFO) and R6
(nested-OOS). Because each variant has its own avg-gross, we compare on the (ddRed, cost) plane and on
the cost-at-matched-avg-gross (turnover efficiency holding average exposure fixed).

Reuses X123 build_universe + X124 held-book engine (gross BEFORE turnover/cost) verbatim. Same fixed
policy (g_floor=0.40, vol_win=180, warmup=60, timeout=90) unless a variant explicitly changes ONE knob.
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

_s123 = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_s123); _s123.loader.exec_module(x123)
build_universe = x123.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
HOLD = x123.HOLD

_s124 = _ilu.spec_from_file_location("x124", SCRIPTS/"X124_reactive_dd_stop.py")
x124 = _ilu.module_from_spec(_s124); _s124.loader.exec_module(x124)
metrics = x124.metrics
gross_unit = x124.gross_unit
const_degross = x124.const_degross

SEED = 12345
ANN = 6*365
COSTS_BPS = [1.0, 3.0, 4.5]
PRIMARY_COST = 4.5e-4

GFLOOR = 0.40
HEAL = 0.50
RDAYS = 90
VOL_WIN = 180
WARMUP = 60
SQRT_WIN = np.sqrt(VOL_WIN)
REC_K = 2.0
K_GRID = [1.5, 2.0, 2.5, 3.0]

EPISODES = [
    ("2022_luna",   "2022-05-01", "2022-07-31"),
    ("2022_ftx",    "2022-11-01", "2023-01-31"),
    ("2024_summer", "2024-06-01", "2024-09-30"),
    ("2025_q4",     "2025-09-01", "2025-12-31"),
]


# ============================================================ generalized vol-norm stop (held book)
def run_eff_heldbook(cyc_w, rs, cost, k, variant="binary",
                     g_floor=GFLOOR, heal=HEAL, timeout=RDAYS, vol_win=VOL_WIN, warmup=WARMUP,
                     grad_ratio=2.0, cool=0, confirm=1):
    """One causal forward pass. The gross[t] is built from realized equity through t-1 with the
    vol-normalized trigger trig = k*sigma*sqrt(vol_win); the variant changes HOW gross moves between
    1.0 and g_floor and WHEN it can change. pnl[t] computed by scaling positions BEFORE turnover/cost
    (X124 held-book mechanics inline). PIT throughout. Returns (pnl, gross, in_stop, rt, turnover_sum).

    variants:
      binary : g = g_floor if stopped else 1.0  (iter-012)
      grad   : SMOOTH ramp. Independent of the stop state-machine. Compute the "stress" ratio
               r = (-dd)/trig (>=1 means at the binary fire point). gross = clamp linearly:
                  r <= 1            -> 1.0
                  1 < r < grad_ratio-> linear 1.0 .. g_floor
                  r >= grad_ratio   -> g_floor
               No round-trips state needed (continuous), but we still gate by warmup. This spreads
               de-grossing across small steps -> turnover is incremental, not a 0.60-jump.
      hyst   : binary state-machine but re-enter only after a FULLER heal (heal arg, e.g. 0.90).
      cool   : binary state-machine; after a re-entry, cannot re-fire for `cool` bars.
      conf   : binary state-machine; require -dd>=trig for `confirm` consecutive bars before firing.
    """
    n = len(cyc_w)
    pnl = np.empty(n, dtype=np.float64)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool)
    incr = np.empty(n, dtype=np.float64)
    prev = {}
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    last_reentry_t = -10**9
    confirm_count = 0
    turn_sum = 0.0
    for t in range(n):
        dd = eq - peak
        if t >= 2:
            lo = max(0, t - vol_win); seg = incr[lo:t]; seg = seg[np.isfinite(seg)]
            sigma = float(seg.std()) if len(seg) >= 2 else 0.0
        else:
            sigma = 0.0
        trig = k * sigma * SQRT_WIN
        can_fire = (t >= warmup) and (sigma > 0)

        if variant == "grad":
            # continuous, stateless ramp on stress ratio (still PIT: dd,sigma through t-1)
            if (not can_fire) or trig <= 0:
                g = 1.0
            else:
                r = (-dd) / trig
                if r <= 1.0:
                    g = 1.0
                elif r >= grad_ratio:
                    g = g_floor
                else:
                    frac = (r - 1.0) / (grad_ratio - 1.0)        # 0..1
                    g = 1.0 + frac * (g_floor - 1.0)
            in_stop[t] = (g < 1.0)
            if g < 1.0:
                # count a "round-trip" each time we ENTER the de-grossed band from full
                if t == 0 or gross[t-1] >= 1.0:
                    rt += 1
        else:
            # state-machine variants
            fire_cond = can_fire and (-dd >= trig)
            if variant == "conf":
                confirm_count = confirm_count + 1 if fire_cond else 0
                fire_now = confirm_count >= confirm
            else:
                fire_now = fire_cond
            if not stopped:
                cooled = (t - last_reentry_t) >= cool
                if fire_now and cooled:
                    stopped = True; rt += 1; stop_peak = peak; trough = eq; stop_t = t
            else:
                trough = min(trough, eq)
                gap = stop_peak - trough
                healed = (gap > 0) and ((eq - trough) >= heal*gap)
                timed = (t - stop_t) >= timeout
                if (healed and eq > trough) or timed:
                    stopped = False; last_reentry_t = t
            g = g_floor if stopped else 1.0
            in_stop[t] = stopped

        gross[t] = g
        # ---- realize cycle t PnL ----
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items():
                net[s] = net.get(s, 0.0) + wt/HOLD
        scaled = {s: g*v for s, v in net.items()}
        alls = set(scaled) | set(prev)
        turn = sum(abs(scaled.get(s, 0.0) - prev.get(s, 0.0)) for s in alls)
        turn_sum += turn
        rl = rs[t]
        c = sum(scaled.get(s, 0.0)*rl.get(s, 0.0) for s in scaled if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(c):
            c = 0.0
        pnl[t] = c - turn*0.5*cost
        prev = scaled
        step = pnl[t]*1e4
        incr[t] = step if np.isfinite(step) else 0.0
        eq += incr[t]
        if eq > peak:
            peak = eq
    return pnl, gross, in_stop, rt, turn_sum


def run_eff_scalar(pnl_base_bps, k, variant="binary",
                   g_floor=GFLOOR, heal=HEAL, timeout=RDAYS, vol_win=VOL_WIN, warmup=WARMUP,
                   grad_ratio=2.0, cool=0, confirm=1):
    """Scalar approx (scales already-netted base pnl by gross) for R6 nested-OOS / placebo loops."""
    pb = np.asarray(pnl_base_bps, dtype=np.float64); pb = np.where(np.isfinite(pb), pb, 0.0)
    n = len(pb)
    gross = np.ones(n); in_stop = np.zeros(n, dtype=bool); incr = np.empty(n, dtype=np.float64)
    eq = 0.0; peak = 0.0; stopped = False; rt = 0
    stop_peak = 0.0; trough = 0.0; stop_t = 0
    last_reentry_t = -10**9; confirm_count = 0
    for t in range(n):
        dd = eq - peak
        if t >= 2:
            lo = max(0, t - vol_win); seg = incr[lo:t]; seg = seg[np.isfinite(seg)]
            sigma = float(seg.std()) if len(seg) >= 2 else 0.0
        else:
            sigma = 0.0
        trig = k * sigma * SQRT_WIN
        can_fire = (t >= warmup) and (sigma > 0)
        if variant == "grad":
            if (not can_fire) or trig <= 0:
                g = 1.0
            else:
                r = (-dd)/trig
                if r <= 1.0: g = 1.0
                elif r >= grad_ratio: g = g_floor
                else: g = 1.0 + (r-1.0)/(grad_ratio-1.0)*(g_floor-1.0)
            in_stop[t] = (g < 1.0)
            if g < 1.0 and (t == 0 or gross[t-1] >= 1.0): rt += 1
        else:
            fire_cond = can_fire and (-dd >= trig)
            if variant == "conf":
                confirm_count = confirm_count + 1 if fire_cond else 0
                fire_now = confirm_count >= confirm
            else:
                fire_now = fire_cond
            if not stopped:
                cooled = (t - last_reentry_t) >= cool
                if fire_now and cooled:
                    stopped = True; rt += 1; stop_peak = peak; trough = eq; stop_t = t
            else:
                trough = min(trough, eq); gap = stop_peak - trough
                healed = (gap > 0) and ((eq - trough) >= heal*gap)
                timed = (t - stop_t) >= timeout
                if (healed and eq > trough) or timed:
                    stopped = False; last_reentry_t = t
            g = g_floor if stopped else 1.0
            in_stop[t] = stopped
        gross[t] = g
        step = g*pb[t]; incr[t] = step if np.isfinite(step) else 0.0
        eq += incr[t]
        if eq > peak: peak = eq
    return gross, in_stop, rt


# variant configs (all at k=REC_K; the ONLY ones with a knob beyond k: hyst.heal, cool.cool, conf.confirm)
VARIANTS = [
    ("binary",  dict(variant="binary")),                                 # iter-012 reference
    ("grad2x",  dict(variant="grad", grad_ratio=2.0)),                   # parameter-free ramp shape
    ("grad3x",  dict(variant="grad", grad_ratio=3.0)),
    ("hyst90",  dict(variant="hyst", heal=0.90)),                        # fuller-heal dead-band
    ("hyst75",  dict(variant="hyst", heal=0.75)),
    ("cool30",  dict(variant="cool", cool=30)),                          # ~5d cooldown
    ("cool60",  dict(variant="cool", cool=60)),                          # ~10d
    ("conf3",   dict(variant="conf", confirm=3)),                        # persist 3 bars (~12h)
    ("conf6",   dict(variant="conf", confirm=6)),                        # persist 6 bars (~1d)
]


def build_panel(name, preds_path):
    print(f"\n[build] {name}", flush=True)
    U = build_universe(preds_path, name)
    base_by_cost = {cb: gross_unit(U["cyc"]["base"], U["rs"], cb*1e-4)*1e4 for cb in COSTS_BPS}
    return U, base_by_cost


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*120, flush=True)
    print("X126 — EFFICIENCY VARIANTS of the portable vol-norm DD stop (Pareto: same DD-cap @ lower cost)", flush=True)
    print("="*120, flush=True)

    panels = {}
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        U, base_by_cost = build_panel(name, pp)
        panels[name] = dict(U=U, base=base_by_cost)

    print("\nBASE (gross=1) reproduction @4.5bps:", flush=True)
    for name in ("HL70", "EXT", "S44"):
        m = metrics(panels[name]["base"][4.5])
        print(f"  {name:>5} Sharpe {m['Sharpe']:+.2f} maxDD {m['maxDD']:+.0f} Calmar {m['Calmar']:+.2f} "
              f"tot {m['tot']:+.0f}", flush=True)

    # ============================================================ MAIN: variant trade-off @ k=2.0
    print("\n" + "="*120, flush=True)
    print("VARIANT TRADE-OFF @ k=2.0, 4.5bps (canonical held book). Pareto = same/more ddRed @ less cost.", flush=True)
    print("  Also report turnover (sum |dpos|) and cost-at-matched-avg-gross.", flush=True)
    print("="*120, flush=True)
    rows = []
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]; base = panels[name]["base"][4.5]; bm = metrics(base)
        print(f"\n--- {name} @4.5bps  base maxDD {bm['maxDD']:+.0f} Sharpe {bm['Sharpe']:+.2f} "
              f"Calmar {bm['Calmar']:+.2f} tot {bm['tot']:+.0f} ---", flush=True)
        print(f"{'variant':>9}{'maxDD':>8}{'ddRed%':>8}{'tot':>8}{'cost%':>7}{'Sharpe':>7}{'Calmar':>7}"
              f"{'%stop':>7}{'RT':>4}{'avgG':>6}{'turn':>8}{'cost/ddRed':>11}", flush=True)
        for vname, vcfg in VARIANTS:
            pnl, gross, stop, rt, turn = run_eff_heldbook(U["cyc"]["base"], U["rs"], 4.5e-4, REC_K, **vcfg)
            pnl = pnl*1e4; m = metrics(pnl)
            ddRed = (1-m["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else np.nan
            cost = (1-m["tot"]/bm["tot"])*100 if bm["tot"] != 0 else np.nan
            cpr = cost/ddRed if (ddRed and ddRed > 0) else np.nan   # cost per unit ddRed (lower=more efficient)
            print(f"{vname:>9}{m['maxDD']:>8.0f}{ddRed:>8.1f}{m['tot']:>8.0f}{cost:>7.1f}{m['Sharpe']:>7.2f}"
                  f"{m['Calmar']:>7.2f}{stop.mean()*100:>7.1f}{rt:>4}{gross.mean():>6.2f}{turn:>8.0f}"
                  f"{cpr:>11.2f}", flush=True)
            rows.append(dict(universe=name, variant=vname, maxDD=m["maxDD"], ddRed=ddRed, tot=m["tot"],
                             cost=cost, Sharpe=m["Sharpe"], Calmar=m["Calmar"], pct_stop=stop.mean()*100,
                             roundtrips=rt, avg_gross=gross.mean(), turnover=turn, cost_per_ddRed=cpr,
                             base_maxDD=bm["maxDD"], base_tot=bm["tot"]))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT/"X126_variant_tradeoff.parquet", index=False)

    # ---- PARETO CHECK: for each variant, cost AT MATCHED avg-gross (const-degross of equal exposure)
    print("\n" + "="*120, flush=True)
    print("PARETO @ matched avg-gross: variant ddRed & cost vs the CONST-degross of the SAME avg-gross.", flush=True)
    print("  A variant is turnover-efficient if its cost <= const-degross cost at the same avg exposure", flush=True)
    print("  AND its ddRed is comparable/better. (binary is the iter-012 reference row.)", flush=True)
    print("="*120, flush=True)
    for name in ("HL70", "EXT", "S44"):
        U = panels[name]["U"]; base = panels[name]["base"][4.5]; bm = metrics(base)
        print(f"\n--- {name} @4.5bps ---", flush=True)
        print(f"{'variant':>9}{'avgG':>6}{'var_ddRed%':>11}{'var_cost%':>10}{'const_ddRed%':>13}"
              f"{'const_cost%':>12}  verdict", flush=True)
        for vname, vcfg in VARIANTS:
            pnl, gross, stop, rt, turn = run_eff_heldbook(U["cyc"]["base"], U["rs"], 4.5e-4, REC_K, **vcfg)
            pnl = pnl*1e4; m = metrics(pnl); ag = gross.mean()
            cm = metrics(const_degross(base, ag))
            v_dd = (1-m["maxDD"]/bm["maxDD"])*100; v_c = (1-m["tot"]/bm["tot"])*100
            c_dd = (1-cm["maxDD"]/bm["maxDD"])*100; c_c = (1-cm["tot"]/bm["tot"])*100
            verdict = "tail>const" if m["maxDD"] > cm["maxDD"] else "~const"
            print(f"{vname:>9}{ag:>6.2f}{v_dd:>11.1f}{v_c:>10.1f}{c_dd:>13.1f}{c_c:>12.1f}  {verdict}",
                  flush=True)

    # ============================================================ R5 episode-LOFO for top variants
    print("\n" + "="*120, flush=True)
    print("R5 — EXT cross-episode + LOFO for binary vs the efficiency variants @ k=2.0, 4.5bps", flush=True)
    print("="*120, flush=True)
    U = panels["EXT"]["U"]; base = panels["EXT"]["base"][4.5]
    ot = pd.to_datetime(pd.Series(U["times"]), utc=True).values
    for vname, vcfg in VARIANTS:
        pnl, gross, stop, rt, turn = run_eff_heldbook(U["cyc"]["base"], U["rs"], 4.5e-4, REC_K, **vcfg)
        gpnl = pnl*1e4
        caps = []
        for nm, s, e in EPISODES:
            m = (ot >= np.datetime64(s)) & (ot <= np.datetime64(e))
            if m.sum() < 40: continue
            bm = metrics(base[m]); sm = metrics(gpnl[m])
            red = (1-sm["maxDD"]/bm["maxDD"])*100 if bm["maxDD"] < 0 else 0.0
            caps.append((nm, red))
        capped = sum(1 for _, r in caps if r >= 10)
        print(f"  {vname:>9}: episodes>=10% {capped}/{len(caps)}  " +
              "  ".join(f"{nm}:{r:+.0f}%" for nm, r in caps), flush=True)

    # ============================================================ R6 nested-OOS (k AND knob if any)
    print("\n" + "="*120, flush=True)
    print("R6 — NESTED-OOS: pick k (and the variant's own knob via grid) on PAST folds (max ddRed under", flush=True)
    print("  <=25% cost), apply FORWARD. PASS = fwd ddRed>+5% AND cost<40% on EVERY universe. @4.5bps.", flush=True)
    print("="*120, flush=True)
    # for parameter-free variants (binary, grad), only k is nested. for knobbed variants, nest (k, knob).
    KNOB_GRID = {
        "binary": [{}],
        "grad":   [{"grad_ratio": gr} for gr in (2.0, 3.0)],
        "hyst":   [{"heal": h} for h in (0.50, 0.75, 0.90)],
        "cool":   [{"cool": c} for c in (0, 30, 60)],
        "conf":   [{"confirm": c} for c in (1, 3, 6)],
    }
    r6_summary = {}
    for vfamily in ("binary", "grad", "hyst", "cool", "conf"):
        npass = 0; detail = {}
        for name in ("HL70", "EXT", "S44"):
            U = panels[name]["U"]; base = panels[name]["base"][4.5]
            fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
            folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
            oos_b, oos_s = [], []
            for i in range(1, len(folds)):
                past = np.isin(fold_arr, folds[:i]); fut = fold_arr == folds[i]
                pp = base[past]; bm = metrics(pp)
                best, best_score = None, -1e18
                for k in K_GRID:
                    for knob in KNOB_GRID[vfamily]:
                        cfg = dict(variant=vfamily, **knob) if vfamily != "binary" else dict(variant="binary")
                        gp, _, _ = run_eff_scalar(pp, k, **cfg)
                        sm = metrics(pp*gp)
                        if bm["maxDD"] >= 0: continue
                        ddred = 1-sm["maxDD"]/bm["maxDD"]
                        cost = (1-sm["tot"]/bm["tot"]) if bm["tot"] != 0 else 1.0
                        if cost <= 0.25 and ddred > best_score:
                            best_score, best = ddred, (k, knob)
                if best is None:
                    best = (max(K_GRID), KNOB_GRID[vfamily][-1])
                k_sel, knob_sel = best
                cfg = dict(variant=vfamily, **knob_sel) if vfamily != "binary" else dict(variant="binary")
                pf = base[fut]; gf, _, _ = run_eff_scalar(pf, k_sel, **cfg)
                oos_b.append(pf); oos_s.append(pf*gf)
            ob = metrics(np.concatenate(oos_b)); os_ = metrics(np.concatenate(oos_s))
            red = (1-os_["maxDD"]/ob["maxDD"])*100 if ob["maxDD"] < 0 else 0.0
            cost = (1-os_["tot"]/ob["tot"])*100 if ob["tot"] != 0 else np.nan
            ok = (red > 5.0) and np.isfinite(cost) and (cost < 40.0)
            detail[name] = (red, cost, ok)
            if ok: npass += 1
        r6_summary[vfamily] = (npass, detail)
        ds = "  ".join(f"{n}:{d[0]:+.0f}%/{d[1]:+.0f}%{'P' if d[2] else 'F'}" for n, d in detail.items())
        print(f"  {vfamily:>7}: nested-OOS {npass}/3  {ds}", flush=True)

    print(f"\nartifacts: X126_variant_tradeoff.parquet", flush=True)
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
