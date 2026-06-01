"""X130 — iter-019 CANONICAL: transaction-cost-aware NO-TRADE BAND on the held-book net weights.

ALPHA track (it changes the realized net weights). The alpha champion (HL70 regime-hybrid
held-book, K=5/side, 6 overlapping sleeves) is UNCHANGED — the band is a pure cost-efficiency
layer that suppresses sub-threshold rebalances of names that churn at the rank cutoff
(iter-016: sleeves 2-6 hold stale, mildly-anti-signal positions; iter-019 pre-check:
56% of trades are small churn = 45% of turnover, cost paid for ~no signal).

MECHANISM (exactly as the research handoff specifies):
  Each cycle t, compute the target net weights `target_t` (the 6-sleeve average of cyc_w,
  i.e. the X117 production book — unchanged). Then EXECUTE a banded book:
      exe[s] = target[s]   if |target[s] - held[s]| >= BAND   (trade)
      exe[s] = held[s]      otherwise                          (HOLD; no trade)
  Turnover/cost is computed on the ACTUAL executed changes only. BAND=0 reproduces X117.

PIT (G1): the band uses ONLY the current target (built from preds/mom/beta that are all
  trailing/.shift(1) inside build_universe) and the PREVIOUSLY-EXECUTED weights `held`
  (realized through t-1). No future information enters the execution decision. The only
  parameter is BAND (flagged for G3 nested-OOS). Full leg = 1/K = 0.200.

GROSS-PnL DIAGNOSTIC (the decisive cost-only test): we emit pre-cost (gross) PnL alongside
  net PnL per band. A band that is cost-ONLY leaves gross PnL ~flat (it just trades less of
  the same bet); a band whose gross PnL DRIFTS is changing the bet (the δ=0.05 trap — gross
  jumps because skipping large rank-boundary moves holds a stale book that happens to win
  in-sample). nested-OOS of δ is the decisive gate: it must reject the δ=0.05 region.

Reuses X123.build_universe verbatim (preds + klines pipeline -> cyc["base"] net-target
builder, rs return maps, fold_by_time). Modifies NOTHING prior (no baseline scripts, no
cached preds).
"""
from __future__ import annotations
import time
from pathlib import Path
import importlib.util as _ilu
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCRIPTS = REPO/"research/convexity_portable_2026-05-20/scripts"
OUT = REPO/"research/convexity_portable_2026-05-20/results"

_spec = _ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(x123)
build_universe = x123.build_universe
HL70_PREDS, EXT_PREDS, S44_PREDS = x123.HL70_PREDS, x123.EXT_PREDS, x123.S44_PREDS
HOLD = x123.HOLD
K = x123.K

SEED = 12345
N_PLACEBO = 200
ANN = 6*365                       # cycles/yr (4h horizon, held-book per-cycle)
COSTS_BPS = [1.0, 3.0, 4.5]
PRIMARY_COST = 4.5e-4             # production calibration (X117 = +1.93 / -5674 at 4.5bps)
FULL_LEG = 1.0/K                  # 0.200

# pre-registered band sweep (handoff: {0,0.005,0.01,0.02,0.03} center 0.02; +0.05/0.08 trap probes)
BAND_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
# nested-OOS candidate set (forward-chosen): keep the trap (0.05) IN the menu so the test must reject it
BAND_OOS_GRID = [0.0, 0.01, 0.02, 0.03, 0.05]


# --------------------------------------------------------------------------- metrics
def metrics(pnl_bps):
    pb = np.asarray(pnl_bps, dtype=np.float64)
    pb = pb[np.isfinite(pb)]
    if len(pb) < 3:
        return dict(n=len(pb), tot=np.nan, maxDD=np.nan, Sharpe=np.nan, Calmar=np.nan)
    eq = np.cumsum(pb)
    dd = eq - np.maximum.accumulate(eq)
    mdd = float(dd.min())
    sd = pb.std()
    sh = float(pb.mean()/sd*np.sqrt(ANN)) if sd > 0 else np.nan
    cal = float(pb.mean()*ANN/abs(mdd)) if (mdd < 0 and np.isfinite(mdd)) else np.nan
    return dict(n=len(pb), tot=float(eq[-1]), maxDD=mdd, Sharpe=sh, Calmar=cal)


# --------------------------------------------------------------------------- net-target builder
def net_targets(cyc_w):
    """Per-cycle TARGET net weights = the 6-sleeve average of cyc_w[t-HOLD+1..t] (the X117 book).
    This is the production target the band gates against. PIT: cyc_w entries are built from
    trailing/.shift(1) features inside build_universe; the sleeve average uses only sleeves
    entered at or before t."""
    n = len(cyc_w)
    nets = []
    for t in range(n):
        active = cyc_w[max(0, t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s, wt in w.items():
                net[s] = net.get(s, 0.0) + wt/HOLD
        nets.append(net)
    return nets


# --------------------------------------------------------------------------- BANDED held-book engine
def heldbook_band(nets, rs, cost, band, return_gross=False):
    """Execute the net targets with a per-symbol no-trade band.

    held = last EXECUTED weights (realized through t-1, PIT). At cycle t:
        for each symbol in target|held:
            move to target  iff |target - held| >= band  (count its turnover)
            else hold        (carry the held weight, no trade, no turnover)
    pnl[t] = sum_s exe[s]*ret_t[s] - turnover*0.5*cost   (cost on EXECUTED changes only)
    Returns per-cycle net PnL (fraction). If return_gross, also returns per-cycle GROSS
    (pre-cost) PnL and total executed turnover — to verify the band is cost-only."""
    n = len(nets)
    held = {}
    pnl = np.empty(n, dtype=np.float64)
    gross = np.empty(n, dtype=np.float64)
    tot_turn = 0.0
    for t in range(n):
        target = nets[t]
        exe = dict(held)
        turn = 0.0
        for s in set(target) | set(held):
            tg = target.get(s, 0.0)
            pv = held.get(s, 0.0)
            if abs(tg - pv) >= band:
                exe[s] = tg
                turn += abs(tg - pv)
            else:
                exe[s] = pv                                   # HOLD — carry, no trade
        exe = {s: w for s, w in exe.items() if abs(w) > 1e-9}
        rl = rs[t]
        g = sum(exe.get(s, 0.0)*rl.get(s, 0.0) for s in exe if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(g):
            g = 0.0
        gross[t] = g
        pnl[t] = g - turn*0.5*cost
        tot_turn += turn
        held = exe
    if return_gross:
        return pnl, gross, tot_turn
    return pnl


# --------------------------------------------------------------------------- placebo (G4 matched random skip)
def heldbook_random_skip(nets, rs, cost, n_skip_target, rng):
    """Matched random-turnover-skip placebo. The real band SKIPS the small (sub-δ) trades.
    The control instead skips n_skip_target trades chosen AT RANDOM from the same per-cycle
    candidate trade set, of MATCHED count — so it cuts the same NUMBER of small trades but at
    random positions (not specifically the rank-boundary churn). If the band only matches this,
    the honest equivalent is 'trade a bit less at random'. PIT-irrelevant (placebo only)."""
    n = len(nets)
    # build the full per-cycle trade list (symbol, |target-held|) under BAND=0 execution
    held = {}
    cand = []                                                  # list of (t, sym, target_val, held_val, size)
    base_held_seq = []                                         # snapshot held BEFORE each cycle (for replay)
    for t in range(n):
        base_held_seq.append(dict(held))
        target = nets[t]
        for s in set(target) | set(held):
            tg = target.get(s, 0.0); pv = held.get(s, 0.0)
            ch = abs(tg - pv)
            if ch > 1e-12:
                cand.append((t, s, tg, pv, ch))
        # advance held to full target (BAND=0 baseline execution)
        exe = {s: w for s, w in target.items() if abs(w) > 1e-9}
        held = exe
    cand = np.array([(t, c[4]) for t, c in zip([x[0] for x in cand], cand)], dtype=float) if False else cand
    # choose n_skip_target trades at random to SKIP (carry held); rank-agnostic
    m = len(cand)
    if n_skip_target <= 0 or n_skip_target >= m:
        return heldbook_band(nets, rs, cost, 0.0)
    skip_set = set(rng.choice(m, size=n_skip_target, replace=False).tolist())
    skip_by_cycle = {}
    for i, (t, s, tg, pv, ch) in enumerate(cand):
        if i in skip_set:
            skip_by_cycle.setdefault(t, set()).add(s)
    # replay execution: at cycle t, move every candidate symbol EXCEPT those in skip set
    held = {}
    pnl = np.empty(n, dtype=np.float64)
    for t in range(n):
        target = nets[t]
        skips = skip_by_cycle.get(t, set())
        exe = dict(held)
        turn = 0.0
        for s in set(target) | set(held):
            tg = target.get(s, 0.0); pv = held.get(s, 0.0)
            if s in skips:
                exe[s] = pv                                    # randomly skipped -> hold
            else:
                if abs(tg - pv) > 1e-12:
                    turn += abs(tg - pv)
                exe[s] = tg
        exe = {s: w for s, w in exe.items() if abs(w) > 1e-9}
        rl = rs[t]
        g = sum(exe.get(s, 0.0)*rl.get(s, 0.0) for s in exe if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(g):
            g = 0.0
        pnl[t] = g - turn*0.5*cost
        held = exe
    return pnl


def count_band_skips(nets, band):
    """Number of trades the band SKIPS (sub-δ moves), for matching the placebo skip count."""
    held = {}
    n_skip = 0
    for t in range(len(nets)):
        target = nets[t]
        exe = dict(held)
        for s in set(target) | set(held):
            tg = target.get(s, 0.0); pv = held.get(s, 0.0)
            ch = abs(tg - pv)
            if ch < 1e-12:
                continue
            if ch >= band:
                exe[s] = tg
            else:
                exe[s] = pv
                n_skip += 1
        held = {s: w for s, w in exe.items() if abs(w) > 1e-9}
    return n_skip


# --------------------------------------------------------------------------- block-bootstrap paired CI
def paired_ci_by_fold(diff_pnl_bps, fold_arr, rng, n_boot=2000):
    """Block-bootstrap (block = fold) the paired per-cycle PnL diff (banded - base). CI on the
    MEAN per-cycle diff. CI clearing zero => G6 pass."""
    folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
    blocks = [np.where(fold_arr == f)[0] for f in folds]
    blocks = [b for b in blocks if len(b) > 0]
    if not blocks:
        return np.nan, np.nan, np.nan
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[j] for j in pick])
        means[i] = np.nanmean(diff_pnl_bps[idx])
    return float(np.nanmean(diff_pnl_bps)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- builds
def build_panel(name, preds_path):
    print(f"\n[build] {name}", flush=True)
    U = build_universe(preds_path, name)
    nets = net_targets(U["cyc"]["base"])
    return U, nets


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*120, flush=True)
    print("X130 — TRANSACTION-COST-AWARE NO-TRADE BAND on held-book net weights (cost-only layer)", flush=True)
    print(f"  execute |target - held| >= BAND else HOLD; cost on executed changes only. "
          f"full leg = {FULL_LEG:.3f}. costs {COSTS_BPS} bps.", flush=True)
    print(f"  PIT: band uses current target + previously-EXECUTED weights only (no future). "
          f"BAND is the only knob (G3 nested-OOS).", flush=True)
    print("="*120, flush=True)

    panels = {}
    for name, pp in (("HL70", HL70_PREDS), ("EXT", EXT_PREDS), ("S44", S44_PREDS)):
        U, nets = build_panel(name, pp)
        panels[name] = dict(U=U, nets=nets)

    # ---- BASE reproduction (BAND=0 must reproduce X117 on HL70: +1.93 / -5674 @4.5bps) ----
    print("\n" + "="*120, flush=True)
    print("BASE REPRODUCTION (BAND=0 == X117/X123 base book) @ each cost", flush=True)
    print("="*120, flush=True)
    for name in ("HL70", "EXT", "S44"):
        nets = panels[name]["nets"]; U = panels[name]["U"]
        for cb in COSTS_BPS:
            m = metrics(heldbook_band(nets, U["rs"], cb*1e-4, 0.0)*1e4)
            tag = "  <- X117 target +1.93/-5674" if (name == "HL70" and cb == 4.5) else ""
            print(f"  {name:>5} @{cb:>4.1f}bps  Sharpe {m['Sharpe']:+.2f}  maxDD {m['maxDD']:+.0f}  "
                  f"Calmar {m['Calmar']:+.2f}  totPnL {m['tot']:+.0f}{tag}", flush=True)

    # ============================================================ δ-SWEEP per universe per cost
    print("\n" + "="*120, flush=True)
    print("G2/G8 — δ-SWEEP: Sharpe/maxDD/Calmar/totPnL/turnover + GROSS PnL (cost-only check) per universe", flush=True)
    print("  GROSS flat across δ = cost-only (good); GROSS drifting = band changes the bet (the δ=0.05 trap).", flush=True)
    print("="*120, flush=True)
    sweep_rows = []
    for name in ("HL70", "EXT", "S44"):
        nets = panels[name]["nets"]; U = panels[name]["U"]
        for cb in COSTS_BPS:
            base_pnl = heldbook_band(nets, U["rs"], cb*1e-4, 0.0)
            bm = metrics(base_pnl*1e4)
            print(f"\n--- {name} @ {cb:.1f}bps  (base Calmar {bm['Calmar']:+.2f}) ---", flush=True)
            print(f"  {'band':>7}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}"
                  f"{'turnover':>10}{'grossPnL':>10}{'foldsPos':>9}", flush=True)
            fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
            folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
            for band in BAND_GRID:
                pnl, gross, turn = heldbook_band(nets, U["rs"], cb*1e-4, band, return_gross=True)
                m = metrics(pnl*1e4)
                gtot = gross.sum()*1e4
                # folds where banded Calmar >= base Calmar
                fp = 0; nf = 0
                base_b = base_pnl
                for f in folds:
                    fm = fold_arr == f
                    if fm.sum() < 3:
                        continue
                    nf += 1
                    cb_ = metrics(base_b[fm]*1e4)["Calmar"]; ca_ = metrics(pnl[fm]*1e4)["Calmar"]
                    if np.isfinite(ca_) and np.isfinite(cb_) and ca_ >= cb_:
                        fp += 1
                print(f"  {band:>7.3f}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}"
                      f"{m['tot']:>+9.0f}{turn:>10.1f}{gtot:>+10.0f}{fp:>6}/{nf}", flush=True)
                sweep_rows.append(dict(universe=name, cost_bps=cb, band=band, Sharpe=m["Sharpe"],
                                       maxDD=m["maxDD"], Calmar=m["Calmar"], totPnL=m["tot"],
                                       turnover=turn, grossPnL=gtot, folds_pos=fp, n_folds=nf))
    pd.DataFrame(sweep_rows).to_parquet(OUT/"X130_band_sweep.parquet", index=False)

    # ============================================================ G4 matched random-skip placebo
    print("\n" + "="*120, flush=True)
    print("G4 — MATCHED RANDOM-TURNOVER-SKIP placebo: skip the SAME NUMBER of trades at RANDOM (not the", flush=True)
    print(f"  rank-boundary churn). Does targeting small sub-δ trades beat random skip of equal count? "
          f"({N_PLACEBO} seeds) @4.5bps", flush=True)
    print("="*120, flush=True)
    g4_rows = []
    for name in ("HL70", "EXT"):
        nets = panels[name]["nets"]; U = panels[name]["U"]
        for band in (0.02, 0.03):
            real = heldbook_band(nets, U["rs"], PRIMARY_COST, band)
            rm = metrics(real*1e4)
            n_skip = count_band_skips(nets, band)
            cals = np.empty(N_PLACEBO); tots = np.empty(N_PLACEBO)
            for i in range(N_PLACEBO):
                pp = heldbook_random_skip(nets, U["rs"], PRIMARY_COST, n_skip, rng)
                mm = metrics(pp*1e4); cals[i] = mm["Calmar"]; tots[i] = mm["tot"]
            crank = float(np.nanmean(cals < rm["Calmar"])*100)
            trank = float(np.nanmean(tots < rm["tot"])*100)
            print(f"  {name} band={band:.3f} (skips {n_skip} trades): real Calmar {rm['Calmar']:+.2f} "
                  f"totPnL {rm['tot']:+.0f}", flush=True)
            print(f"      placebo Calmar p50 {np.nanpercentile(cals,50):+.2f} p95 {np.nanpercentile(cals,95):+.2f} "
                  f"-> rank p{crank:.0f} {'PASS(>=p95)' if crank>=95 else 'FAIL'}", flush=True)
            print(f"      placebo totPnL p50 {np.nanpercentile(tots,50):+.0f} p95 {np.nanpercentile(tots,95):+.0f} "
                  f"-> rank p{trank:.0f} {'PASS(>=p95)' if trank>=95 else 'FAIL'}", flush=True)
            g4_rows.append(dict(universe=name, band=band, n_skip=n_skip, real_Calmar=rm["Calmar"],
                                real_totPnL=rm["tot"], plac_Cal_p50=np.nanpercentile(cals,50),
                                plac_Cal_p95=np.nanpercentile(cals,95), Cal_rank=crank,
                                plac_tot_p50=np.nanpercentile(tots,50), plac_tot_p95=np.nanpercentile(tots,95),
                                tot_rank=trank))
    pd.DataFrame(g4_rows).to_parquet(OUT/"X130_g4_placebo.parquet", index=False)

    # ============================================================ G6 paired CI (per-cycle, vs base)
    print("\n" + "="*120, flush=True)
    print("G6 — PAIRED block-bootstrap CI of per-cycle PnL diff (banded - base) vs baseline. "
          f"CI must clear 0. @4.5bps", flush=True)
    print("="*120, flush=True)
    g6_rows = []
    for name in ("HL70", "EXT", "S44"):
        nets = panels[name]["nets"]; U = panels[name]["U"]
        fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
        base = heldbook_band(nets, U["rs"], PRIMARY_COST, 0.0)*1e4
        for band in (0.02, 0.03):
            banded = heldbook_band(nets, U["rs"], PRIMARY_COST, band)*1e4
            diff = banded - base
            mean_d, lo, hi = paired_ci_by_fold(diff, fold_arr, rng)
            clears = (lo > 0) or (hi < 0)
            print(f"  {name} band={band:.3f}: mean per-cycle diff {mean_d:+.3f}bps  "
                  f"95%CI [{lo:+.3f}, {hi:+.3f}]  {'CLEARS 0' if clears else 'CROSSES 0 (G6 FAIL)'}", flush=True)
            g6_rows.append(dict(universe=name, band=band, mean_diff=mean_d, ci_lo=lo, ci_hi=hi, clears=clears))
    pd.DataFrame(g6_rows).to_parquet(OUT/"X130_g6_paired_ci.parquet", index=False)

    # also emit per-cycle banded vs base (HL70 + EXT) for downstream placebo/bootstrap by Eval
    for name in ("HL70", "EXT"):
        nets = panels[name]["nets"]; U = panels[name]["U"]
        out = {"open_time": pd.to_datetime(U["times"]),
               "fold": [U["fold_by_time"].get(t, -1) for t in U["times"]],
               "regime": U["regimes"]}
        out["pnl_base"] = heldbook_band(nets, U["rs"], PRIMARY_COST, 0.0)
        for band in (0.01, 0.02, 0.03, 0.05):
            out[f"pnl_band_{int(band*1000):03d}"] = heldbook_band(nets, U["rs"], PRIMARY_COST, band)
        pd.DataFrame(out).to_parquet(OUT/f"X130_percycle_{name}.parquet", index=False)
        print(f"  per-cycle -> X130_percycle_{name}.parquet", flush=True)

    # ============================================================ G3 nested-OOS of δ (DECISIVE)
    print("\n" + "="*120, flush=True)
    print("G3 (DECISIVE) — NESTED-OOS of δ: choose δ on PAST folds (max Calmar), apply to NEXT fold;", flush=True)
    print("  measure realized FORWARD Calmar/Sharpe. MUST reject the δ=0.05 trap. HL70 (production) + EXT. @4.5bps",
          flush=True)
    print(f"  candidate δ menu (trap 0.05 kept IN): {BAND_OOS_GRID}", flush=True)
    print("="*120, flush=True)
    for name in ("HL70", "EXT"):
        nets = panels[name]["nets"]; U = panels[name]["U"]
        fold_arr = np.array([U["fold_by_time"].get(t, -1) for t in U["times"]])
        folds = sorted(f for f in pd.unique(fold_arr) if f >= 0)
        # precompute per-band per-cycle PnL once (PnL within a fold-mask = that fold's contribution)
        pnl_by_band = {b: heldbook_band(nets, U["rs"], PRIMARY_COST, b)*1e4 for b in BAND_OOS_GRID}
        base_pnl = pnl_by_band[0.0]
        oos_base, oos_band, chosen = [], [], []
        for i in range(1, len(folds)):
            past = np.isin(fold_arr, folds[:i])
            fut = fold_arr == folds[i]
            # choose δ maximizing PAST Calmar (in-sample on past folds)
            best_b, best_cal = 0.0, -1e18
            for b in BAND_OOS_GRID:
                cal = metrics(pnl_by_band[b][past])["Calmar"]
                if np.isfinite(cal) and cal > best_cal:
                    best_cal, best_b = cal, b
            oos_base.append(base_pnl[fut])
            oos_band.append(pnl_by_band[best_b][fut])
            chosen.append((int(folds[i]), best_b))
        ob = metrics(np.concatenate(oos_base)); osd = metrics(np.concatenate(oos_band))
        # forward folds_positive (banded fold Calmar >= base fold Calmar over the OOS folds)
        fp = 0; nf = 0
        for fi, b in chosen:
            fm = fold_arr == fi
            if fm.sum() < 3:
                continue
            nf += 1
            cb_ = metrics(base_pnl[fm])["Calmar"]; ca_ = metrics(pnl_by_band[b][fm])["Calmar"]
            if np.isfinite(ca_) and np.isfinite(cb_) and ca_ >= cb_:
                fp += 1
        n_trap = sum(1 for _, b in chosen if b >= 0.05)
        print(f"\n  {name}: nested-OOS chosen δ per fold: {chosen}", flush=True)
        print(f"    OOS base   Sharpe {ob['Sharpe']:+.2f}  maxDD {ob['maxDD']:+.0f}  "
              f"Calmar {ob['Calmar']:+.2f}  totPnL {ob['tot']:+.0f}", flush=True)
        print(f"    OOS banded Sharpe {osd['Sharpe']:+.2f}  maxDD {osd['maxDD']:+.0f}  "
              f"Calmar {osd['Calmar']:+.2f}  totPnL {osd['tot']:+.0f}", flush=True)
        lift = osd["Calmar"] - ob["Calmar"]
        print(f"    => forward Calmar lift {lift:+.2f}  forward folds_pos {fp}/{nf}  "
              f"(δ=0.05 chosen in {n_trap}/{len(chosen)} folds)  "
              f"{'PASS (forward Calmar >= base, trap not dominating)' if lift >= 0 else 'FAIL (no forward lift)'}",
              flush=True)

    print(f"\nartifacts: X130_band_sweep.parquet, X130_g4_placebo.parquet, X130_g6_paired_ci.parquet, "
          f"X130_percycle_{{HL70,EXT}}.parquet", flush=True)
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
