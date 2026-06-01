"""X120_eval_gates — Evaluation harness for the iter-002 correlation-aware side gate.

Runs the full evaluation_contract.md gate suite on X120 (corr-aware sideways regime
gate; THR tuned in {0.60,0.70,0.80}).

Per the Review's CRITICAL note: the G4 matched side-pool placebo MUST re-derive the
held-book PnL under each randomized FLAT mask (flatting a side sleeve changes
turnover/overlap for HOLD subsequent cycles). We therefore import X120's
build_universe and replicate its heldbook with an EXPLICIT mask argument
(heldbook_mask) — for the real gate the mask is the gate's own decision; for placebos
the mask is a random count-matched subset of the is_side pool. Zeroing pnl_base rows
is NOT used.

Gates: G1 (confirm Review PASS), G2 (in-sample), G3 (nested-OOS THR), G4 (matched
side-pool placebo, >=200 seeds, re-derived held book), G5 (per-fold + LOFO),
G6 (block-bootstrap paired CI), G7 (HL70 vs S44), G8 (cost {1,3,4.5} bps).
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO/"research/convexity_portable_2026-05-20/scripts"))
import X120_corr_regime_gate as X  # noqa

OUT = REPO/"research/convexity_portable_2026-05-20/results"
HOLD = X.HOLD
THRS = X.THRS
COSTS = X.COSTS_BPS
N_PLACEBO = 500
N_BOOT = 2000
SEED = 20260525
BASE_CALMAR = 1.68
BASE_SHARPE = 1.93
BASE_MAXDD = -5674


def heldbook_mask(times, cyc_w, rs, regimes, flat_mask, cost):
    """Identical to X120.heldbook but takes an EXPLICIT per-cycle flat mask (bool array
    over all cycles; True => that cycle's NEW sleeve is FLAT/empty). Re-derives the held
    book so turnover/overlap decay under FLATs is faithfully accounted (NOT row-zeroing)."""
    prev = {}; pnl = []
    n = len(cyc_w)
    for t in range(n):
        active = [({} if flat_mask[k] else cyc_w[k]) for k in range(max(0, t-HOLD+1), t+1)]
        net = {}
        for w in active:
            for s, wt in w.items():
                net[s] = net.get(s, 0) + wt/HOLD
        alls = set(net) | set(prev)
        turn = sum(abs(net.get(s, 0)-prev.get(s, 0)) for s in alls)
        rl = rs[t]
        cyc = sum(net.get(s, 0)*rl.get(s, 0.0) for s in net if np.isfinite(rl.get(s, 0.0)))
        if not np.isfinite(cyc):
            cyc = 0.0
        pnl.append(cyc - turn*0.5*cost)
        prev = net
    return np.asarray(pnl, dtype=np.float64)


def metrics(pnl):
    return X.stats(pnl)


def fold_dd(pnl):
    eq = np.cumsum(pnl*1e4)
    return (eq - np.maximum.accumulate(eq)).min()


# ---------------------------------------------------------------------------
def load_universe(label, preds_path):
    times, cyc_w, rs, fold_by_time, regimes, pr_lag = X.build_universe(preds_path, label)
    regimes = np.array(regimes)
    is_side = regimes == "side"
    fold_arr = np.array([fold_by_time.get(t, -1) for t in times])
    # reconstruct each THR's gate mask exactly as X120 does
    gate_masks = {}
    for thr in THRS:
        m = np.zeros(len(times), dtype=bool)
        for t in range(len(times)):
            if regimes[t] == "side" and np.isfinite(pr_lag[t]) and pr_lag[t] >= thr:
                m[t] = True
        gate_masks[thr] = m
    return dict(label=label, times=times, cyc_w=cyc_w, rs=rs, regimes=regimes,
                is_side=is_side, fold_arr=fold_arr, pr_lag=pr_lag, gate_masks=gate_masks)


# ---- G2 in-sample: base + each THR at each cost ----
def g2_insample(U):
    print(f"\n=== G2 in-sample [{U['label']}] base vs THR x cost ===", flush=True)
    rows = []
    for cost_bps in COSTS:
        cost = cost_bps*1e-4
        base = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'],
                             np.zeros(len(U['times']), bool), cost)
        sb = metrics(base)
        rows.append((cost_bps, 'base', sb))
        for thr in THRS:
            g = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'],
                              U['gate_masks'][thr], cost)
            sg = metrics(g)
            rows.append((cost_bps, f'gate@{thr:.2f}', sg))
    print(f"  {'cost':>5}{'arm':>12}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    for cb, arm, s in rows:
        print(f"  {cb:>5.1f}{arm:>12}{s['sharpe']:>+8.2f}{s['maxDD']:>+9.0f}{s['calmar']:>+8.2f}{s['totPnL']:>+9.0f}{s['pct_pos']:>7.1f}", flush=True)
    return rows


# ---- G3 nested-OOS THR selection ----
def g3_nested(U, cost_bps=4.5):
    """Walk-forward: for each fold f (in sorted order, skipping first), select THR by
    argmax in-sample Calmar over ALL strictly-earlier folds, apply to fold f. Concatenate
    per-cycle gated PnL (gate decision = chosen THR's mask restricted to fold f cycles;
    base elsewhere within the concatenation is irrelevant since we measure the nested
    series). We must re-derive the held book ONCE with a composite mask = union over
    folds of (chosen THR mask within that fold)."""
    cost = cost_bps*1e-4
    folds = sorted(set(U['fold_arr'].tolist()) - {-1})
    base = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'],
                         np.zeros(len(U['times']), bool), cost)
    # precompute per-THR gated PnL & per-fold Calmar for selection
    thr_pnl = {thr: heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'],
                                  U['gate_masks'][thr], cost) for thr in THRS}
    composite_mask = np.zeros(len(U['times']), bool)
    chosen = {}
    print(f"\n=== G3 nested-OOS THR [{U['label']}] @ {cost_bps}bps ===", flush=True)
    for f in folds:
        past = U['fold_arr'] < f
        past = past & (U['fold_arr'] != -1)
        if past.sum() < 10:
            # no history -> structural fallback
            sel = X.THR_FALLBACK
            chosen[f] = (sel, 'fallback')
        else:
            best_thr, best_cal = None, -1e9
            for thr in THRS:
                cal = metrics(thr_pnl[thr][past])['calmar']
                cal = cal if np.isfinite(cal) else -1e9
                if cal > best_cal:
                    best_cal, best_thr = cal, thr
            sel = best_thr
            chosen[f] = (sel, f'cal={best_cal:+.2f}')
        cur = U['fold_arr'] == f
        composite_mask |= (U['gate_masks'][sel] & cur)
        print(f"  fold {f}: chosen THR={sel:.2f} ({chosen[f][1]})", flush=True)
    nested = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'], composite_mask, cost)
    # evaluate only over folds that had a selection (all are graded forward; warmup fold 1
    # for HL70 starts at fold 2). We grade the full concatenated series across graded folds.
    graded = U['fold_arr'] != -1
    sn = metrics(nested[graded])
    sb = metrics(base[graded])
    print(f"  nested-OOS: Sharpe {sn['sharpe']:+.2f} maxDD {sn['maxDD']:+.0f} Calmar {sn['calmar']:+.2f} totPnL {sn['totPnL']:+.0f}", flush=True)
    print(f"  base(graded): Sharpe {sb['sharpe']:+.2f} maxDD {sb['maxDD']:+.0f} Calmar {sb['calmar']:+.2f} totPnL {sb['totPnL']:+.0f}", flush=True)
    return dict(nested=sn, base_graded=sb, chosen=chosen)


# ---- G4 matched side-pool placebo (re-derived held book) ----
def g4_placebo(U, thr, n_seeds=N_PLACEBO, cost_bps=4.5):
    cost = cost_bps*1e-4
    real_mask = U['gate_masks'][thr]
    n_flat = int(real_mask.sum())
    side_idx = np.where(U['is_side'])[0]
    real = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'], real_mask, cost)
    sr = metrics(real)
    rng = np.random.default_rng(SEED)
    cal_dist = np.empty(n_seeds); dd_dist = np.empty(n_seeds); sh_dist = np.empty(n_seeds)
    for i in range(n_seeds):
        pick = rng.choice(side_idx, size=n_flat, replace=False)
        m = np.zeros(len(U['times']), bool); m[pick] = True
        p = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'], m, cost)
        s = metrics(p)
        cal_dist[i] = s['calmar'] if np.isfinite(s['calmar']) else -1e9
        dd_dist[i] = s['maxDD']
        sh_dist[i] = s['sharpe']
    pct_cal = float((cal_dist < sr['calmar']).mean()*100)
    # maxDD: less negative (closer to 0) is better -> percentile of being >= placebo
    pct_dd = float((dd_dist < sr['maxDD']).mean()*100)   # frac of placebos with WORSE (more neg) DD
    pct_sh = float((sh_dist < sr['sharpe']).mean()*100)
    print(f"\n=== G4 matched side-pool placebo [{U['label']}] THR={thr:.2f}, {n_flat} FLATs, {n_seeds} seeds @ {cost_bps}bps ===", flush=True)
    print(f"  real:  Calmar {sr['calmar']:+.2f} maxDD {sr['maxDD']:+.0f} Sharpe {sr['sharpe']:+.2f}", flush=True)
    print(f"  placebo Calmar: mean {cal_dist[cal_dist>-1e8].mean():+.2f} p50 {np.percentile(cal_dist,50):+.2f} p95 {np.percentile(cal_dist,95):+.2f} max {cal_dist.max():+.2f}", flush=True)
    print(f"  placebo maxDD:  mean {dd_dist.mean():+.0f} p50 {np.percentile(dd_dist,50):+.0f} p95(best) {np.percentile(dd_dist,95):+.0f} best {dd_dist.max():+.0f}", flush=True)
    print(f"  placebo Sharpe: mean {sh_dist.mean():+.2f} p95 {np.percentile(sh_dist,95):+.2f}", flush=True)
    print(f"  => real percentile: Calmar p{pct_cal:.0f}  maxDD p{pct_dd:.0f}  Sharpe p{pct_sh:.0f}", flush=True)
    return dict(real=sr, n_flat=n_flat, pct_calmar=pct_cal, pct_maxdd=pct_dd, pct_sharpe=pct_sh,
                cal_p95=float(np.percentile(cal_dist, 95)), cal_mean=float(cal_dist[cal_dist>-1e8].mean()),
                dd_p95=float(np.percentile(dd_dist, 95)))


# ---- G5 per-fold + LOFO ----
def g5_perfold(U, thr, cost_bps=4.5):
    cost = cost_bps*1e-4
    base = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'],
                         np.zeros(len(U['times']), bool), cost)
    gate = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'], U['gate_masks'][thr], cost)
    folds = sorted(set(U['fold_arr'].tolist()) - {-1})
    print(f"\n=== G5 per-fold [{U['label']}] THR={thr:.2f} @ {cost_bps}bps ===", flush=True)
    print(f"  {'fold':>5}{'baseDD':>9}{'gateDD':>9}{'DDimpr%':>9}{'baseSh':>8}{'gateSh':>8}{'dPnL':>9}", flush=True)
    n_dd_better = 0; n_sh_better = 0; n_eval = 0; fold_dpnl = {}
    for f in folds:
        m = U['fold_arr'] == f
        if m.sum() < 3: continue
        n_eval += 1
        bdd = fold_dd(base[m]); gdd = fold_dd(gate[m])
        impr = (1-abs(gdd)/abs(bdd))*100 if bdd < 0 else np.nan
        bsh = X.ann(base[m]); gsh = X.ann(gate[m])
        dpnl = (gate[m].sum()-base[m].sum())*1e4
        fold_dpnl[f] = dpnl
        if np.isfinite(impr) and impr > 0: n_dd_better += 1
        if gsh > bsh: n_sh_better += 1
        print(f"  {f:>5}{bdd:>+9.0f}{gdd:>+9.0f}{impr:>+9.1f}{bsh:>+8.2f}{gsh:>+8.2f}{dpnl:>+9.0f}", flush=True)
    print(f"  DD improved {n_dd_better}/{n_eval} folds; Sharpe improved {n_sh_better}/{n_eval} folds", flush=True)
    # LOFO on Calmar
    print(f"  --- LOFO (drop one fold, recompute aggregate Calmar lift) ---", flush=True)
    full_lift = metrics(gate)['calmar'] - metrics(base)['calmar']
    print(f"  full-sample Calmar lift: {full_lift:+.2f}", flush=True)
    for f in folds:
        keep = (U['fold_arr'] != f) & (U['fold_arr'] != -1)
        lift = metrics(gate[keep])['calmar'] - metrics(base[keep])['calmar']
        print(f"  drop fold {f}: Calmar lift {lift:+.2f}  (dPnL of fold {f}: {fold_dpnl.get(f, 0):+.0f})", flush=True)
    return dict(n_dd_better=n_dd_better, n_sh_better=n_sh_better, n_eval=n_eval, full_lift=full_lift)


# ---- G6 block-bootstrap paired CI ----
def g6_bootstrap(U, thr, cost_bps=4.5, n_boot=N_BOOT):
    cost = cost_bps*1e-4
    base = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'],
                         np.zeros(len(U['times']), bool), cost)
    gate = heldbook_mask(U['times'], U['cyc_w'], U['rs'], U['regimes'], U['gate_masks'][thr], cost)
    diff = (gate - base)*1e4   # per-cycle bps diff
    graded = U['fold_arr'] != -1
    folds = sorted(set(U['fold_arr'][graded].tolist()))
    fold_blocks = {f: np.where(U['fold_arr'] == f)[0] for f in folds}
    rng = np.random.default_rng(SEED+1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.choice(folds, size=len(folds), replace=True)
        idx = np.concatenate([fold_blocks[f] for f in picks])
        means[b] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    obs = diff[graded].mean()
    print(f"\n=== G6 paired CI [{U['label']}] THR={thr:.2f} @ {cost_bps}bps (block-boot by fold, {n_boot}) ===", flush=True)
    print(f"  mean per-cycle (gate-base) diff: {obs:+.3f} bps; 95% CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    print(f"  CI {'CLEARS' if (lo>0 or hi<0) else 'CROSSES'} zero", flush=True)
    return dict(mean=obs, ci_lo=lo, ci_hi=hi, clears=bool(lo > 0 or hi < 0))


def main():
    t0 = time.time()
    print("=== X120 EVALUATION GATES (iter-002 corr-aware side gate) ===", flush=True)
    HL = load_universe("HL70", X.HL70_PREDS)
    S44 = load_universe("S44", X.S44_PREDS)

    res = {}
    # G2
    res['g2_HL70'] = g2_insample(HL)
    res['g2_S44'] = g2_insample(S44)

    # G3 nested-OOS (PRIMARY: HL70)
    res['g3_HL70'] = g3_nested(HL, 4.5)
    res['g3_S44'] = g3_nested(S44, 4.5)

    # Determine the THR to gate-test for G4/G5/G6: in-sample best Calmar on HL70 @4.5
    cost = 4.5e-4
    best_thr, best_cal = None, -1e9
    for thr in THRS:
        g = heldbook_mask(HL['times'], HL['cyc_w'], HL['rs'], HL['regimes'], HL['gate_masks'][thr], cost)
        c = metrics(g)['calmar']
        if c > best_cal: best_cal, best_thr = c, thr
    print(f"\n[in-sample best HL70 THR by Calmar @4.5bps = {best_thr:.2f} (Calmar {best_cal:+.2f})]", flush=True)

    # G4 placebo on the best in-sample THR AND the fallback THR
    res['g4_HL70_best'] = g4_placebo(HL, best_thr)
    if best_thr != X.THR_FALLBACK:
        res['g4_HL70_fallback'] = g4_placebo(HL, X.THR_FALLBACK)

    # G5 / G6 on the same THR
    res['g5_HL70'] = g5_perfold(HL, best_thr)
    res['g6_HL70'] = g6_bootstrap(HL, best_thr)
    # also report S44 per-fold/CI for G7 context
    res['g5_S44'] = g5_perfold(S44, best_thr)
    res['g6_S44'] = g6_bootstrap(S44, best_thr)

    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
