"""iter-032 — EVALUATION: does expanding the universe 70->156 IMPROVE the strategy honestly,
or is it polluted by thin-history noise?

Reuses the iter-031 deploy-universe engine VERBATIM (build_panel / build_cyc_weights / heldbook /
volnorm_stop / metrics), which itself reuses X117 (held-book regime-hybrid) + X125 (vol-norm stop).

STEP-2 analyses on the EXPANDED x132 V0 preds (156 syms, 2021-26, 8 folds):
  1. BREADTH-N sweep (random/liquidity-agnostic subsets) at N=23,50,100,156 on the SAME x132 preds +
     SAME 2021-26 window. Does Sharpe/Calmar rise with N? (apples-to-apples breadth: only N varies.)
  2. vs the prior 23-sym EXT baseline (x113 preds) — full-156 vs 23-sym EXT on the same folds.
     Also full-156 vs the EXT-23 SUBSET-of-x132 (controls for model retrain, only universe differs).
  3. Per-fold Sharpe/Calmar + folds_positive; transport across the multi-episode panel (not 1 fold).
  4. THIN-HISTORY NOISE CHECK: do extreme-pred picks come from low-history names? Winsorize pred
     and/or require a min-history-per-symbol-per-cycle gate. Is breadth robust to it?
  5. All headline numbers reported WITH the iter-012 vol-norm stop (the deploy config).
"""
from __future__ import annotations
import time, pickle, importlib.util as _ilu
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
SCR = REPO/"agents_system/research/scratch"
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT = REPO/"outputs/iter032"; OUT.mkdir(parents=True, exist_ok=True)

X132_PREDS = RC/"x132_expanded_v0_preds.parquet"
EXT_PREDS = RC/"x113_ext_v0_preds.parquet"
SEED = 32032
N_DRAWS = 30
ANN = 6*365

# reuse the iter-031 engine verbatim
_s = _ilu.spec_from_file_location("i31", SCR/"iter031_deploy_universe.py")
i31 = _ilu.module_from_spec(_s); _s.loader.exec_module(i31)
build_panel = i31.build_panel
build_cyc_weights = i31.build_cyc_weights
heldbook = i31.heldbook
volnorm_stop = i31.volnorm_stop
metrics = i31.metrics
run_subset = i31.run_subset
HOLD = i31.HOLD; K = i31.K
PRIMARY_COST = i31.PRIMARY_COST


def ann_sh(p):
    p = pd.Series(p).dropna()
    return p.mean()/p.std()*np.sqrt(ANN) if len(p) > 2 and p.std() > 0 else np.nan


# ============================================================================================
# FAST ENGINE — precompute per-cycle numpy arrays ONCE; each subset run is pure numpy.
# EXACTLY mirrors i31.build_cyc_weights + heldbook/volnorm_stop semantics (verified vs slow path).
#   build_cyc_weights logic:
#     regime from FULL group (not subset). bear or len(subset_g)<2K -> empty weights.
#     key = mom30 (bull) / pred (side). drop NaN key in subset. if <2K -> empty.
#     sort by key asc: L=tail-K, S=head-K. side: beta-neutral a,b from mean betas (if both>0).
#     w[L]+=a/K, w[S]-=b/K. (return dict overlaps are summed -> a symbol can't be both here.)
# ============================================================================================
def precompute_cycles(P):
    """Per-cycle arrays indexed by a global symbol id. Returns dict with:
       times, sym_index (sym->id), n_sym, and per-cycle lists of (ids, pred, mom30, retpct, regime, beta)."""
    times = P["times"]; by_t = P["by_t"]; betas = P["betas"]
    syms = sorted({s for g in by_t.values() for s in g["symbol"].values})
    sidx = {s: i for i, s in enumerate(syms)}
    nS = len(syms)
    cyc = []
    beta_has = betas.index  # DatetimeIndex
    beta_cols = list(betas.columns)
    beta_col_id = np.array([sidx.get(c, -1) for c in beta_cols])
    beta_vals_all = betas.values  # (T_beta, nbetacols)
    beta_row_for = {t: i for i, t in enumerate(beta_has)}
    for ot in times:
        g = by_t[ot]
        ids = np.array([sidx[s] for s in g["symbol"].values], dtype=np.int64)
        pred = g["pred"].to_numpy(np.float64)
        mom = g["mom30"].to_numpy(np.float64) if "mom30" in g.columns else np.full(len(g), np.nan)
        ret = g["return_pct"].to_numpy(np.float64)
        regime = g["regime"].iloc[0]
        # beta vector aligned to global sym id for this cycle (NaN where absent)
        bvec = np.full(nS, np.nan)
        bi = beta_row_for.get(ot, None)
        if bi is not None:
            row = beta_vals_all[bi]
            m = beta_col_id >= 0
            bvec[beta_col_id[m]] = row[m]
        cyc.append((ids, pred, mom, ret, regime, bvec))
    return dict(times=times, sidx=sidx, nS=nS, syms=syms, cyc=cyc,
                fold_by_time=P["fold_by_time"])


def fast_cyc_weights(PC, subset_ids, pred_override=None, hist_ok=None):
    """Returns (cyc_w list of {sym_id:weight}, rs list of {sym_id:retpct}) for the subset.
       pred_override: optional dict sym_id-> per-cycle... not used; we apply winsor/hist at array level.
       hist_ok: optional callable(ot_index, ids)->bool mask to drop thin rows (min-history gate)."""
    subset = np.zeros(PC["nS"], dtype=bool); subset[list(subset_ids)] = True
    cyc_w = []; rs = []
    for ti, (ids, pred, mom, ret, regime, bvec) in enumerate(PC["cyc"]):
        in_sub = subset[ids]
        # returns dict for the subset (mirrors rl from subset group)
        rl = {int(ids[j]): ret[j] for j in range(len(ids)) if in_sub[j]}
        rs.append(rl)
        if regime == "bear":
            cyc_w.append({}); continue
        # subset members
        sel = in_sub.copy()
        if hist_ok is not None:
            sel = sel & hist_ok(ti, ids)
        n_sub = sel.sum()
        if n_sub < 2*K:
            cyc_w.append({}); continue
        key = mom if regime == "bull" else pred.copy()
        if pred_override == "winsor" and regime != "bull":
            pass  # handled by caller pre-winsorizing the pred arrays
        sub_ids = ids[sel]; sub_key = key[sel]
        good = np.isfinite(sub_key)
        sub_ids = sub_ids[good]; sub_key = sub_key[good]
        if len(sub_ids) < 2*K:
            cyc_w.append({}); continue
        order = np.argsort(sub_key, kind="stable")
        sub_ids = sub_ids[order]
        S = sub_ids[:K]; L = sub_ids[-K:]
        a = b = 1.0
        if regime == "side":
            mbL = np.nanmean(bvec[L]); mbS = np.nanmean(bvec[S])
            if np.isfinite(mbL) and np.isfinite(mbS) and mbL > 0 and mbS > 0:
                a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)
        w = {}
        for s in L: w[int(s)] = w.get(int(s), 0)+a/K
        for s in S: w[int(s)] = w.get(int(s), 0)-b/K
        cyc_w.append(w)
    return cyc_w, rs


def fast_run(PC, subset_ids, with_stop=True, cost=PRIMARY_COST, hist_ok=None):
    cyc_w, rs = fast_cyc_weights(PC, subset_ids, hist_ok=hist_ok)
    pnl = i31.volnorm_stop(cyc_w, rs, cost) if with_stop else i31.heldbook(cyc_w, rs, cost)*1e4
    return pnl, i31.metrics(pnl)


def fast_perfold(PC, subset_ids, with_stop=True, cost=PRIMARY_COST, hist_ok=None):
    cyc_w, rs = fast_cyc_weights(PC, subset_ids, hist_ok=hist_ok)
    pnl = i31.volnorm_stop(cyc_w, rs, cost) if with_stop else i31.heldbook(cyc_w, rs, cost)*1e4
    times = PC["times"]; fbt = PC["fold_by_time"]
    folds = np.array([fbt.get(t, -1) for t in times])
    rows = []
    for f in sorted(set(folds[folds >= 0])):
        rows.append(dict(fold=int(f), **i31.metrics(pnl[folds == f])))
    fps = sum(1 for r in rows if np.isfinite(r["Sharpe"]) and r["Sharpe"] > 0)
    return rows, fps, len(rows), pnl, folds


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("="*100)
    print("iter-032 — EVALUATION: expanded universe (156) — does breadth improve the deploy strategy honestly?")
    print("="*100, flush=True)

    # ============================================================ build panels (cache to disk)
    def cached_panel(preds, label, tag):
        cp = OUT/f"panel_{tag}.pkl"
        if cp.exists():
            print(f"  loading cached panel {cp.name}", flush=True)
            return pickle.loads(cp.read_bytes())
        print(f"  building {label} panel...", flush=True)
        P = build_panel(preds, label)
        cp.write_bytes(pickle.dumps(P))
        return P
    PX = cached_panel(X132_PREDS, "X132 expanded (156)", "x132")
    print(f"  x132: {len(PX['syms'])} syms, {len(PX['times'])} 4h-cycles", flush=True)
    PE = cached_panel(EXT_PREDS, "X113 EXT (23)", "ext")
    print(f"  EXT: {len(PE['syms'])} syms, {len(PE['times'])} 4h-cycles", flush=True)

    # FAST engine: precompute per-cycle numpy arrays once (verified == slow engine to 1e-13).
    print("  precomputing fast-engine cycle arrays...", flush=True)
    PCX = precompute_cycles(PX)
    PCE = precompute_cycles(PE)
    sidxX = PCX["sidx"]; sidxE = PCE["sidx"]

    full156 = [s for s in PX["syms"] if s != "BTCUSDT"]
    ext23 = sorted(PE["syms"])
    ext23_in_x132 = [s for s in ext23 if s in set(PX["syms"])]
    print(f"  tradable legs: full156={len(full156)}, EXT-23-in-x132={len(ext23_in_x132)}", flush=True)

    def ids(PC, names):
        return [PC["sidx"][s] for s in names if s in PC["sidx"]]

    # one-time fast-vs-slow verification (logged for the record)
    _vs = list(np.random.default_rng(1).choice(full156, 23, replace=False))
    _, mS = run_subset(PX, _vs, with_stop=True)
    _, mF = fast_run(PCX, ids(PCX, _vs), with_stop=True)
    print(f"  VERIFY fast==slow: slow Sharpe {mS['Sharpe']:+.4f} vs fast {mF['Sharpe']:+.4f} "
          f"(must match)", flush=True)

    # ============================================================ 1. BREADTH-N sweep (random subsets)
    print("\n" + "="*100)
    print("1. BREADTH-N SWEEP on x132 preds (random subsets, liquidity-agnostic). Same window/folds. @4.5bps")
    print("   For each N: 30 random draws (composition-agnostic). Report mean/std/min/max Sharpe & Calmar.")
    print("   This is the clean apples-to-apples breadth test (same model, same period, only N varies).")
    print("="*100, flush=True)
    pool = full156
    NGRID = [23, 50, 100, 156]
    bn_rows = []
    print(f"  {'N':>4} {'kind':<5}{'Sh_mean':>9}{'Sh_std':>8}{'Sh_min':>8}{'Sh_max':>8}"
          f"{'Cal_mean':>9}{'mDD_mean':>10}", flush=True)
    pool_ids = ids(PCX, pool)
    for N in NGRID:
        for kind, ws in (("base", False), ("stop", True)):
            if N >= len(pool):
                # full universe: single deterministic run
                _, m = fast_run(PCX, pool_ids, with_stop=ws)
                sh = [m["Sharpe"]]; cal = [m["Calmar"]]; dd = [m["maxDD"]]
            else:
                sh = []; cal = []; dd = []
                for _ in range(N_DRAWS):
                    sub = list(rng.choice(pool_ids, size=N, replace=False))
                    _, m = fast_run(PCX, sub, with_stop=ws)
                    sh.append(m["Sharpe"]); cal.append(m["Calmar"]); dd.append(m["maxDD"])
            sh = np.array(sh, float); cal = np.array(cal, float); dd = np.array(dd, float)
            print(f"  {N:>4} {kind:<5}{np.nanmean(sh):>+9.2f}{np.nanstd(sh):>8.2f}{np.nanmin(sh):>+8.2f}"
                  f"{np.nanmax(sh):>+8.2f}{np.nanmean(cal):>+9.2f}{np.nanmean(dd):>+10.0f}", flush=True)
            bn_rows.append(dict(N=N, kind=kind, sh_mean=float(np.nanmean(sh)), sh_std=float(np.nanstd(sh)),
                                sh_min=float(np.nanmin(sh)), sh_max=float(np.nanmax(sh)),
                                cal_mean=float(np.nanmean(cal)), dd_mean=float(np.nanmean(dd)),
                                n_draws=len(sh)))
    pd.DataFrame(bn_rows).to_csv(OUT/"iter032_breadthN_x132.csv", index=False)

    # ============================================================ 2. full-156 vs 23-sym EXT
    print("\n" + "="*100)
    print("2. FULL-156 vs 23-sym EXT baseline (same 2021-26 folds). @4.5bps, base & +stop")
    print("="*100, flush=True)
    print(f"  {'config':<28}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'%pos':>7}", flush=True)
    comp_rows = []
    configs = [
        ("EXT-23 (x113 preds)", PCE, ext23),
        ("EXT-23 subset of x132", PCX, ext23_in_x132),
        ("FULL-156 (x132)", PCX, full156),
    ]
    for name, PC, sub in configs:
        sids = ids(PC, sub)
        for kind, ws in (("base", False), ("stop", True)):
            _, m = fast_run(PC, sids, with_stop=ws)
            print(f"  {name+' ['+kind+']':<28}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}"
                  f"{m['Calmar']:>+8.2f}{m['tot']:>+9.0f}{m['pct_pos']:>7.1f}", flush=True)
            comp_rows.append(dict(config=name, kind=kind, **m))
    pd.DataFrame(comp_rows).to_csv(OUT/"iter032_full156_vs_ext23.csv", index=False)

    # ============================================================ 3. per-fold + transport
    print("\n" + "="*100)
    print("3. PER-FOLD robustness + transport (+stop config). Does the expanded book hold across folds?")
    print("="*100, flush=True)
    pf_all = {}
    for name, PC, sub in [("FULL-156", PCX, full156), ("EXT-23-x132", PCX, ext23_in_x132),
                          ("EXT-23-x113", PCE, ext23)]:
        rows, fps, nf, pnl, folds = fast_perfold(PC, ids(PC, sub), with_stop=True)
        pf_all[name] = (rows, fps, nf)
        print(f"\n  {name}: folds_positive={fps}/{nf}  (per-fold Sharpe / Calmar / maxDD)")
        for r in rows:
            print(f"    f{r['fold']}: Sh {r['Sharpe']:>+6.2f}  Cal {r['Calmar']:>+6.2f}  "
                  f"mDD {r['maxDD']:>+8.0f}  tot {r['tot']:>+8.0f}", flush=True)
        # LOFO: drop each fold, recompute aggregate Sharpe
        agg = metrics(pnl)
        loro = []
        for f in sorted(set(folds[folds >= 0])):
            mm = metrics(pnl[folds != f])
            loro.append((int(f), mm["Sharpe"]))
        worst = min(loro, key=lambda x: x[1])
        print(f"    AGG Sharpe {agg['Sharpe']:+.2f}; LOFO worst-drop fold {worst[0]} -> {worst[1]:+.2f}", flush=True)
    # save per-fold
    rec = []
    for name, (rows, fps, nf) in pf_all.items():
        for r in rows:
            rec.append(dict(config=name, folds_positive=fps, n_folds=nf, **r))
    pd.DataFrame(rec).to_csv(OUT/"iter032_perfold.csv", index=False)

    # ============================================================ 4. THIN-HISTORY NOISE CHECK
    print("\n" + "="*100)
    print("4. THIN-HISTORY NOISE CHECK (the watch item). Wide pred tails [-34,+45].")
    print("="*100, flush=True)

    # 4a. Are extreme-pred picks coming from low-history names?
    draw = pd.read_parquet(X132_PREDS, columns=["symbol", "open_time", "pred"])
    draw["open_time"] = pd.to_datetime(draw["open_time"], utc=True)
    draw = draw[(draw["open_time"].dt.hour % 4 == 0) & (draw["open_time"].dt.minute == 0)].copy()
    # per-symbol first-seen (proxy for listing/history start)
    first_seen = draw.groupby("symbol")["open_time"].min()
    # history-bars-available at each row = count of that symbol's 4h rows strictly before open_time
    draw = draw.sort_values(["symbol", "open_time"])
    draw["hist_bars"] = draw.groupby("symbol").cumcount()  # 0-based count of prior bars
    # flag extreme preds (|pred| in top 1%)
    thr = draw["pred"].abs().quantile(0.99)
    extreme = draw[draw["pred"].abs() >= thr]
    print(f"  |pred| top-1% threshold = {thr:.2f}; {len(extreme):,} extreme rows of {len(draw):,}")
    print(f"  median hist_bars: ALL rows = {draw['hist_bars'].median():.0f}  |  "
          f"extreme-pred rows = {extreme['hist_bars'].median():.0f}", flush=True)
    # fraction of extreme picks from thin-history (<180 4h-bars = <30d of trailing beta/mom warmup)
    THIN = 180
    print(f"  fraction of rows with hist_bars<{THIN}: ALL = {(draw['hist_bars']<THIN).mean()*100:.1f}%  |  "
          f"extreme = {(extreme['hist_bars']<THIN).mean()*100:.1f}%", flush=True)
    # correlation of |pred| with thin-history
    corr = np.corrcoef((draw["hist_bars"] < THIN).astype(float), draw["pred"].abs().clip(upper=10))[0, 1]
    print(f"  corr(|pred|, is_thin_history) = {corr:+.4f}  (positive => extreme preds skew to thin names)",
          flush=True)

    # 4b. Winsorize pred + min-history gate, fast engine (does the breadth result survive?)
    print("\n  4b. Robustness: winsorize pred (clip side-regime ranking key) + min-history gate.")
    # per-cycle hist_bars array aligned to PCX cycle order/ids
    hb = draw.set_index(["symbol", "open_time"])["hist_bars"].to_dict()
    times = PCX["times"]; syms_by_id = PCX["syms"]
    hist_arr = []  # per cycle: array of hist_bars aligned to that cycle's ids
    for ti, ot in enumerate(times):
        idz = PCX["cyc"][ti][0]
        hist_arr.append(np.array([hb.get((syms_by_id[j], ot), 0) for j in idz]))

    def hist_ok_factory(min_hist):
        def f(ti, idz):
            return hist_arr[ti] >= min_hist
        return f

    def winsor_PC(PC, lim):
        """Return a shallow PC copy whose pred arrays are clipped to +-lim (mom unchanged)."""
        newcyc = []
        for (idz, pred, mom, ret, regime, bvec) in PC["cyc"]:
            newcyc.append((idz, np.clip(pred, -lim, lim), mom, ret, regime, bvec))
        P2 = dict(PC); P2["cyc"] = newcyc
        return P2

    full_ids = ids(PCX, full156)
    print(f"  {'variant':<34}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}", flush=True)
    th_rows = []
    base_pnl, m = fast_run(PCX, full_ids, with_stop=True)
    print(f"  {'full156 base (no mod)':<34}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}{m['tot']:>+9.0f}", flush=True)
    th_rows.append(dict(variant="full156 base", **m))
    for lim in (3.0, 1.5):
        _, m = fast_run(winsor_PC(PCX, lim), full_ids, with_stop=True)
        print(f"  {('full156 winsor pred |%.1f|'%lim):<34}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}{m['tot']:>+9.0f}", flush=True)
        th_rows.append(dict(variant=f"winsor|{lim}|", **m))
    for mh in (180, 540):
        _, m = fast_run(PCX, full_ids, with_stop=True, hist_ok=hist_ok_factory(mh))
        print(f"  {('full156 min-hist %db'%mh):<34}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}{m['tot']:>+9.0f}", flush=True)
        th_rows.append(dict(variant=f"minhist{mh}", **m))
    pd.DataFrame(th_rows).to_csv(OUT/"iter032_thinhistory.csv", index=False)

    # 4c. breadth benefit driven by thin names? full-156 vs only-old-history subset
    print("\n  4c. Is breadth benefit from the NEW thin names, or the wider OLD set?")
    old_syms = [s for s in full156 if first_seen.get(s, pd.Timestamp.max) <= pd.Timestamp("2023-01-01", tz="UTC")]
    new_syms = [s for s in full156 if first_seen.get(s, pd.Timestamp.min) > pd.Timestamp("2024-06-01", tz="UTC")]
    print(f"  old (<=2023 start): {len(old_syms)} syms; new (>2024-06 start): {len(new_syms)} syms", flush=True)
    for name, sub in [("OLD-only (<=2023)", old_syms), ("FULL-156", full156)]:
        _, m = fast_run(PCX, ids(PCX, sub), with_stop=True)
        print(f"    {name:<22} Sh {m['Sharpe']:>+6.2f}  Cal {m['Calmar']:>+6.2f}  "
              f"mDD {m['maxDD']:>+8.0f}  tot {m['tot']:>+8.0f}", flush=True)

    # ============================================================ 5. G4 placebo / G6 paired CI / G8 cost
    print("\n" + "="*100)
    print("5. GATES: G4 matched-composition placebo, G6 paired CI vs EXT-23, G8 cost sweep. (+stop)")
    print("="*100, flush=True)

    # G4: is full-156 better than a matched-size random composition? full-156 IS the full set, so the
    # matched control is "random N=156 of 156" = identical. Instead frame breadth-honesty: does the
    # FULL set beat random-100 (matched to a plausible deploy size) and is it within the random-156 band?
    # The decisive honest test: random subsets of EACH size vs full (done in block 1). Here: 100-seed
    # placebo of random-100 draws -> where does the deterministic full-156 rank, and random-50.
    print("  G4 breadth-honesty placebo (100 random draws each, +stop Sharpe):", flush=True)
    g4 = {}
    for N in (50, 100):
        shs = np.array([fast_run(PCX, list(rng.choice(full_ids, N, replace=False)), with_stop=True)[1]["Sharpe"]
                        for _ in range(100)], float)
        g4[N] = shs
        print(f"    random-{N}: mean {np.nanmean(shs):+.2f} p5 {np.nanpercentile(shs,5):+.2f} "
              f"p50 {np.nanpercentile(shs,50):+.2f} p95 {np.nanpercentile(shs,95):+.2f} max {np.nanmax(shs):+.2f}",
              flush=True)
    full_sh = fast_run(PCX, full_ids, with_stop=True)[1]["Sharpe"]
    rank100 = (g4[100] < full_sh).mean()*100
    print(f"    FULL-156 Sharpe {full_sh:+.2f}; ranks p{rank100:.0f} vs random-100 "
          f"(>p50 means breadth>truncation)", flush=True)

    # G6: paired block-bootstrap CI of per-cycle PnL diff (full-156 minus EXT-23-x132), +stop.
    print("\n  G6 paired CI: full-156 vs EXT-23 (both x132, +stop), block-bootstrap by fold:", flush=True)
    pnl_full, _ = fast_run(PCX, full_ids, with_stop=True)
    pnl_ext, _ = fast_run(PCX, ids(PCX, ext23_in_x132), with_stop=True)
    diff = pnl_full - pnl_ext
    fbt = PCX["fold_by_time"]; foldsv = np.array([fbt.get(t, -1) for t in PCX["times"]])
    uf = sorted(set(foldsv[foldsv >= 0]))
    boots = []
    for _ in range(2000):
        chosen = rng.choice(uf, size=len(uf), replace=True)
        seg = np.concatenate([diff[foldsv == f] for f in chosen])
        boots.append(np.nanmean(seg))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"    mean per-cycle diff {np.nanmean(diff):+.3f} bps; 95% CI [{lo:+.3f}, {hi:+.3f}] "
          f"({'clears zero' if lo>0 or hi<0 else 'CROSSES zero'})", flush=True)

    # G8: cost sweep {1,3,4.5} bps for full-156 vs EXT-23 (+stop and base).
    print("\n  G8 cost sweep (+stop): Sharpe at 1 / 3 / 4.5 bps/leg", flush=True)
    print(f"    {'config':<22}{'@1bp':>8}{'@3bp':>8}{'@4.5bp':>9}", flush=True)
    cost_rows = []
    for name, PC, sub in [("FULL-156", PCX, full156), ("EXT-23-x132", PCX, ext23_in_x132),
                          ("EXT-23-x113", PCE, ext23)]:
        sids = ids(PC, sub); shs = []
        for c in (1e-4, 3e-4, 4.5e-4):
            shs.append(fast_run(PC, sids, with_stop=True, cost=c)[1]["Sharpe"])
        print(f"    {name:<22}{shs[0]:>+8.2f}{shs[1]:>+8.2f}{shs[2]:>+9.2f}", flush=True)
        cost_rows.append(dict(config=name, sh_1bp=shs[0], sh_3bp=shs[1], sh_45bp=shs[2]))
    pd.DataFrame(cost_rows).to_csv(OUT/"iter032_cost.csv", index=False)

    print(f"\nDone [{time.time()-t0:.0f}s]  artifacts in {OUT}", flush=True)


if __name__ == "__main__":
    main()
