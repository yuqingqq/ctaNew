"""iter-033 — TRAINING-CONFIG study. iter-032 found the real lever was RETRAINING on
more data (+0.86->+1.06 Sharpe just from a larger training set), NOT more symbols.

So: study the TRAINING WINDOW (expanding vs rolling-N) and the RETRAIN CADENCE (fold
count) on the EXPANDED 156-sym panel. Does old data help (expanding) or hurt (regime
drift -> rolling)? Is fresher (more-frequent) retrain better?

Method:
  - Panel is already built (outputs/vBTC_features/panel_expanded_v0.parquet, 175 syms,
    2021-2026, 4h-sampled, PIT target_z). We ONLY vary the fold/window construction in
    a per-symbol RidgeCV walk-forward (x6.train_per_sym_ridge machinery, reimplemented
    here to add a rolling-window lower bound + variable fold count).
  - WINDOW: expanding (current) vs rolling-{1yr,2yr,3yr}.
  - CADENCE: N_FOLDS in {9 (~7mo, current), 18 (~3.5mo), 27 (~2.3mo)}.
  - For each (window, cadence): regenerate walk-forward V0 preds, attach a COMMON
    9-fold evaluation-fold id (fixed time partition, apples-to-apples per-fold), then
    run the iter-031/iter-032 deploy engine (full-156, regime-hybrid held-book + iter-012
    vol-norm stop) -> Sharpe / maxDD / Calmar / per-fold / folds_positive / IC.
  - Honest tests: per-fold robustness + LOFO; NESTED-OOS the config choice (pick config
    on past eval-folds, apply to next -> does the chosen config beat current expanding-9?).

Eval is on a FIXED 9-bin common time grid for ALL configs so per-fold numbers are
comparable regardless of training cadence. We OVERRIDE the preds' own `fold` column with
this common grid before feeding the engine.
"""
from __future__ import annotations
import time, json, importlib.util as ilu
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV

REPO = Path("/home/yuqing/ctaNew")
SCR = REPO/"agents_system/research/scratch"
OUT = REPO/"outputs/iter033"; OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDDIR = OUT/"preds"; PREDDIR.mkdir(exist_ok=True)

# x6 (training machinery + constants/feature lists/preproc)
_s = ilu.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = ilu.module_from_spec(_s); _s.loader.exec_module(x6)
# iter031 deploy engine (build_panel/heldbook/volnorm_stop/metrics) + iter032 fast engine
_s2 = ilu.spec_from_file_location("i31", SCR/"iter031_deploy_universe.py")
i31 = ilu.module_from_spec(_s2); _s2.loader.exec_module(i31)
_s3 = ilu.spec_from_file_location("i32", SCR/"iter032_expanded_universe.py")
i32 = ilu.module_from_spec(_s3); _s3.loader.exec_module(i32)

ANN = 6*365
N_EVAL_FOLDS = 9          # fixed common evaluation grid
YEAR_BARS = {"1yr": int(365*24/4), "2yr": int(2*365*24/4), "3yr": int(3*365*24/4), "exp": None}


def get_folds_n(panel, n_folds):
    """Walk-forward folds with arbitrary n_folds (mirrors x6.get_folds with N_FOLDS=n_folds)."""
    times = sorted(panel["open_time"].unique())
    n = len(times); fs = n // n_folds
    folds = []
    for f in range(n_folds):
        i0 = f*fs
        i1 = min((f+1)*fs, n-1) if f < n_folds-1 else n
        oos_start = pd.Timestamp(times[i0]); oos_end = pd.Timestamp(times[i1-1])
        embargo_cut = oos_start - pd.Timedelta(days=x6.EMBARGO_DAYS)
        folds.append((f, oos_start, oos_end, embargo_cut))
    return folds


def train_per_sym_ridge_window(panel, folds, feat_cols, window_bars=None):
    """x6.train_per_sym_ridge, but the per-fold TRAIN set is restricted to a trailing
    `window_bars` 4h-bars (rolling) instead of all history (expanding) when window_bars
    is not None. The trailing window is measured per the fold's embargo_cut on open_time."""
    all_preds = []
    for f, ts, te, ec in folds:
        if window_bars is None:
            lo_cut = None
        else:
            # trailing window: keep train rows whose open_time is within window_bars*4h before ec
            lo_cut = ec - pd.Timedelta(hours=4*window_bars)
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        if lo_cut is not None:
            train_all = train_all[train_all["open_time"] >= lo_cut]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 300:
                continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30:
                continue
            sstats, hstats = x6.fit_preproc(gtr, feat_cols)
            Xtr = x6.apply_preproc(gtr, feat_cols, sstats, hstats)
            Xte = x6.apply_preproc(gte, feat_cols, sstats, hstats)
            try:
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(Xtr, gtr["target_z"].to_numpy())
                pred = m.predict(Xte)
            except Exception:
                continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized", "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
        if out_frames:
            all_preds.append(pd.concat(out_frames, ignore_index=True))
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def assign_common_eval_fold(apd):
    """Override the preds' `fold` with a FIXED 9-bin time partition so per-fold metrics
    are comparable across configs of any training cadence."""
    times = np.array(sorted(apd["open_time"].unique()))
    n = len(times); fs = n // N_EVAL_FOLDS
    fold_of = {}
    for f in range(N_EVAL_FOLDS):
        i0 = f*fs; i1 = min((f+1)*fs, n) if f < N_EVAL_FOLDS-1 else n
        for t in times[i0:i1]:
            fold_of[t] = f
    apd = apd.copy()
    apd["fold"] = apd["open_time"].map(fold_of).astype(int)
    return apd


def build_and_eval(panel, feats, window_key, n_folds, full156_names, tag):
    """Regenerate preds for (window, cadence), eval full-156 +stop. Returns metrics dict
    + per-fold rows + the pnl array (common 9-fold grid)."""
    pred_path = PREDDIR/f"preds_{tag}.parquet"
    if pred_path.exists():
        apd = pd.read_parquet(pred_path)
        apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    else:
        folds = get_folds_n(panel, n_folds)
        apd = train_per_sym_ridge_window(panel, folds, feats, window_bars=YEAR_BARS[window_key])
        apd = assign_common_eval_fold(apd)
        apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    # build deploy panel + fast engine on these preds
    P = i31.build_panel(pred_path, tag)
    PC = i32.precompute_cycles(P)
    full_ids = [PC["sidx"][s] for s in full156_names if s in PC["sidx"]]
    rows, fps, nf, pnl, foldsv = i32.fast_perfold(PC, full_ids, with_stop=True)
    _, m = i32.fast_run(PC, full_ids, with_stop=True)
    # LOFO worst-drop
    uf = sorted(set(foldsv[foldsv >= 0]))
    loro = [(int(f), i31.metrics(pnl[foldsv != f])["Sharpe"]) for f in uf]
    worst = min(loro, key=lambda x: x[1])
    return dict(tag=tag, window=window_key, n_folds=n_folds, ic=ic,
                Sharpe=m["Sharpe"], maxDD=m["maxDD"], Calmar=m["Calmar"], tot=m["tot"],
                pct_pos=m["pct_pos"], folds_positive=fps, n_eval_folds=nf,
                lofo_worst_fold=worst[0], lofo_worst_sh=worst[1]), rows, pnl, foldsv


def main():
    t0 = time.time()
    print("="*100)
    print("iter-033 — TRAINING-CONFIG (window x cadence) on the 156-sym expanded panel")
    print("="*100, flush=True)
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    print(f"  panel {len(panel):,} rows x {panel['symbol'].nunique()} syms; feats={len(feats)}")
    full156 = [s for s in sorted(panel["symbol"].unique()) if s != "BTCUSDT"]
    print(f"  full universe legs (ex-BTC) = {len(full156)}", flush=True)

    # GRID: (window, cadence). expanding-9 = the incumbent (matches x132 build).
    GRID = [
        ("exp", 9), ("exp", 18), ("exp", 27),
        ("3yr", 9), ("3yr", 18),
        ("2yr", 9), ("2yr", 18),
        ("1yr", 9), ("1yr", 18),
    ]
    results = {}; perfold = {}; pnls = {}; foldsvs = {}
    print("\n" + "="*100)
    print("GRID: regenerate preds + eval full-156 +stop  (Sharpe / maxDD / Calmar / IC / folds+)")
    print("="*100, flush=True)
    print(f"  {'tag':<14}{'Sharpe':>8}{'maxDD':>9}{'Calmar':>8}{'totPnL':>9}{'IC':>9}{'fpos':>7}{'LOFO':>9}", flush=True)
    for wk, nf in GRID:
        tag = f"{wk}_nf{nf}"
        tt = time.time()
        m, rows, pnl, foldsv = build_and_eval(panel, feats, wk, nf, full156, tag)
        results[tag] = m; perfold[tag] = rows; pnls[tag] = pnl; foldsvs[tag] = foldsv
        print(f"  {tag:<14}{m['Sharpe']:>+8.2f}{m['maxDD']:>+9.0f}{m['Calmar']:>+8.2f}"
              f"{m['tot']:>+9.0f}{m['ic']:>+9.4f}{m['folds_positive']:>4}/{m['n_eval_folds']:<2}"
              f"{m['lofo_worst_sh']:>+9.2f}  [{time.time()-tt:.0f}s]", flush=True)
    pd.DataFrame(results.values()).to_csv(OUT/"iter033_grid.csv", index=False)

    # per-fold table (common 9-fold grid)
    print("\n" + "="*100)
    print("PER-FOLD Sharpe (common 9-fold eval grid)")
    print("="*100, flush=True)
    folds_present = sorted({r["fold"] for rows in perfold.values() for r in rows})
    hdr = "  " + f"{'config':<14}" + "".join([f"   f{f}" for f in folds_present])
    print(hdr, flush=True)
    pf_rec = []
    for tag, rows in perfold.items():
        shmap = {r["fold"]: r["Sharpe"] for r in rows}
        line = "  " + f"{tag:<14}" + "".join([f"{shmap.get(f, np.nan):>+6.2f}" for f in folds_present])
        print(line, flush=True)
        for r in rows:
            pf_rec.append(dict(config=tag, **r))
    pd.DataFrame(pf_rec).to_csv(OUT/"iter033_perfold.csv", index=False)

    # ============================================================ NESTED-OOS the config choice
    # For each eval-fold f (f>=2), choose the config with best CUMULATIVE Sharpe over folds [0..f-1],
    # then record that chosen config's fold-f PnL. Compare the assembled nested-OOS PnL/Sharpe to the
    # incumbent (exp_nf9) applied to ALL folds. A config family that only wins with hindsight fails here.
    print("\n" + "="*100)
    print("NESTED-OOS config selection: pick best config on PAST eval-folds, apply forward.")
    print("   Decisive: does honest forward selection beat the incumbent exp_nf9?")
    print("="*100, flush=True)
    tags = list(results.keys())
    # per-config per-fold pnl array on common grid
    fold_pnl = {}  # tag -> {fold -> pnl array}
    for tag in tags:
        pnl = pnls[tag]; fv = foldsvs[tag]
        fold_pnl[tag] = {int(f): pnl[fv == f] for f in sorted(set(fv[fv >= 0]))}
    all_folds = sorted({f for d in fold_pnl.values() for f in d})

    def sharpe(arr):
        arr = np.asarray(arr, float); arr = arr[np.isfinite(arr)]
        return arr.mean()/arr.std()*np.sqrt(ANN) if len(arr) > 2 and arr.std() > 0 else np.nan

    def cum_sharpe(tag, upto):  # cumulative Sharpe over folds [0..upto-1]
        seg = np.concatenate([fold_pnl[tag][f] for f in all_folds if f < upto and f in fold_pnl[tag]])
        return sharpe(seg)

    nested_pnl = []; chosen = []
    incumbent = "exp_nf9"
    for f in all_folds:
        if f < 2:  # need >=2 past folds to choose; use incumbent for the warmup folds
            pick = incumbent
        else:
            scored = [(t, cum_sharpe(t, f)) for t in tags]
            scored = [(t, s) for t, s in scored if np.isfinite(s)]
            pick = max(scored, key=lambda x: x[1])[0] if scored else incumbent
        chosen.append((f, pick))
        nested_pnl.append(fold_pnl[pick].get(f, np.array([])))
    nested = np.concatenate([p for p in nested_pnl if len(p)])
    inc_full = pnls[incumbent]
    print("  chosen config per eval-fold (nested-OOS): " +
          ", ".join([f"f{f}:{p}" for f, p in chosen]), flush=True)
    print(f"  NESTED-OOS assembled  Sharpe {sharpe(nested):+.2f}  maxDD {i31.metrics(nested)['maxDD']:+.0f}"
          f"  Calmar {i31.metrics(nested)['Calmar']:+.2f}", flush=True)
    mi = i31.metrics(inc_full)
    print(f"  INCUMBENT exp_nf9     Sharpe {sharpe(inc_full):+.2f}  maxDD {mi['maxDD']:+.0f}"
          f"  Calmar {mi['Calmar']:+.2f}", flush=True)
    # also: best static config in-sample (the hindsight pick) for context
    best_static = max(results.values(), key=lambda r: r["Sharpe"])
    print(f"  BEST-STATIC (hindsight) {best_static['tag']}: Sharpe {best_static['Sharpe']:+.2f}"
          f"  Calmar {best_static['Calmar']:+.2f}", flush=True)

    # paired CI nested vs incumbent (align on common folds)
    # build per-fold mean-diff bootstrap
    diff_by_fold = []
    for f in all_folds:
        pick = dict(chosen)[f]
        a = fold_pnl[pick].get(f, np.array([])); b = fold_pnl[incumbent].get(f, np.array([]))
        nlen = min(len(a), len(b))
        if nlen > 0:
            diff_by_fold.append(a[:nlen] - b[:nlen])
    rng = np.random.default_rng(33)
    boots = []
    for _ in range(2000):
        ch = rng.choice(len(diff_by_fold), size=len(diff_by_fold), replace=True)
        seg = np.concatenate([diff_by_fold[i] for i in ch])
        boots.append(np.nanmean(seg))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    md = np.nanmean(np.concatenate(diff_by_fold))
    print(f"  paired (nested - incumbent) per-cycle diff {md:+.3f} bps; 95% CI [{lo:+.3f},{hi:+.3f}] "
          f"({'clears 0' if lo > 0 or hi < 0 else 'CROSSES 0'})", flush=True)

    json.dump({"nested_sharpe": float(sharpe(nested)), "incumbent_sharpe": float(sharpe(inc_full)),
               "best_static": best_static["tag"], "chosen": [[int(f), p] for f, p in chosen],
               "paired_ci": [float(lo), float(hi)], "paired_mean": float(md)},
              open(OUT/"iter033_nested.json", "w"), indent=2)
    print(f"\nDONE [{time.time()-t0:.0f}s]  artifacts in {OUT}", flush=True)


if __name__ == "__main__":
    main()
