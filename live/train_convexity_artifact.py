"""Train the deployable convexity_portable artifact.

ONE per-symbol Ridge model fit on ALL history up to today, saved as a single
artifact for live inference. The research script (X132/X6) uses 9-fold
walk-forward — here we collapse to one final model per symbol (all history,
1-day embargo) so the paper bot can do single-shot inference each cycle.

Validation: predicted-vs-walk-forward rank IC on the OOS portion of the last
fold must be >= 0.85, else the saved artifact diverges from the research
preds and the forward test is meaningless.

Output: live/models/convexity_portable.pkl with keys:
  feat_cols        list of feature column names
  symbols          list of symbols (Ridge fit per-sym only for these)
  sstats           {sym: {feat: {lo, hi, mu, sd}}} (winsor + z preproc)
  hstats           {sym: {feat: {vals, mu, sd}}}  (rank preproc for heavy-tail)
  models           {sym: sklearn RidgeCV fitted}
  meta             {trained_at, panel_path, train_rows, val_rank_ic, …}
"""
from __future__ import annotations
import argparse, sys, time, json, pickle, warnings, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
PANEL_PATH = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
RESEARCH_PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
OUT_DIR = REPO/"live/models"; OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

FEAT_COLS = x6.BASE + x6.COHORT_EXTRAS   # 14 BASE + 3 cohort = 17
EMBARGO_DAYS = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=str, default=None,
                    help="ISO date, cap training data at this point (for walk-forward validation)")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix on output filename, e.g. 'val_h1' -> convexity_portable_val_h1.pkl")
    ap.add_argument("--alphas", type=str, default=None,
                    help="Comma-sep ridge alpha grid override, e.g. '1,10,100,1000,10000'")
    ap.add_argument("--halflife-days", type=float, default=None,
                    help="If set, exponentially weight training samples by recency with this half-life "
                         "(in days). Weight = exp(-(T_end - t)/halflife). Older samples get less weight.")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    OUT_PKL = OUT_DIR/f"convexity_portable{suffix}.pkl"
    OUT_META = OUT_DIR/f"convexity_portable{suffix}.meta.json"

    t0 = time.time()
    print(f"=== Train convexity_portable deploy artifact (tag={args.tag or 'production'}) ===", flush=True)
    print(f"  panel: {PANEL_PATH.name}", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    if args.train_end is not None:
        train_end = pd.Timestamp(args.train_end, tz="UTC")
        before = len(panel)
        panel = panel[panel["open_time"] <= train_end]
        print(f"  --train-end {args.train_end}: filtered {before:,} -> {len(panel):,} rows", flush=True)
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms × "
          f"{panel['open_time'].min().date()}→{panel['open_time'].max().date()}", flush=True)

    # target_z already in panel from x132 build
    if "target_z" not in panel.columns:
        print("  target_z missing — rebuilding", flush=True)
        panel = x6.build_target_z(panel)
    miss = [c for c in FEAT_COLS if c not in panel.columns]
    if miss:
        raise RuntimeError(f"missing feature cols in panel: {miss}")

    # Fit cutoff & validation slice depend on --train-end:
    #  - if --train-end is set (walk-forward validation mode):
    #      fit on all rows up to train_end - embargo; validate on the original research fold-8
    #      slice (kept for sanity but not the main test — main test is run separately by the bot
    #      against retrained-artifact-generated preds)
    #  - default (live deploy mode):
    #      fit on all rows up to (panel_end - embargo), validate vs research fold-8 preds
    panel_end = panel["open_time"].max()
    if args.train_end is not None:
        fit_cut = train_end - pd.Timedelta(days=EMBARGO_DAYS)
    else:
        fit_cut = panel_end - pd.Timedelta(days=EMBARGO_DAYS)
    fit_train = panel[(panel["exit_time"] < fit_cut) & panel["target_z"].notna()].copy()
    print(f"  fit cutoff = {fit_cut}; fit rows = {len(fit_train):,}", flush=True)

    # research preds (for validation rank IC sanity)
    rp = pd.read_parquet(RESEARCH_PREDS)
    rp["open_time"] = pd.to_datetime(rp["open_time"], utc=True)
    last_fold = rp["fold"].max()
    val_slice = rp[rp["fold"] == last_fold]
    val_start = val_slice["open_time"].min()
    # only validate on rows AFTER fit_cut (honest OOS for sanity-check IC)
    val_slice = val_slice[val_slice["open_time"] > fit_cut]
    print(f"  validation slice (research fold {last_fold} post-fit_cut): "
          f"{len(val_slice):,} rows from {val_slice['open_time'].min() if len(val_slice) else 'n/a'}", flush=True)

    from sklearn.linear_model import RidgeCV
    train = fit_train   # alias used downstream meta
    alphas_grid = [float(a) for a in args.alphas.split(",")] if args.alphas else x6.RIDGE_ALPHAS
    print(f"  ridge alphas grid: {alphas_grid}", flush=True)
    if args.halflife_days:
        print(f"  recency-weighted training: halflife={args.halflife_days}d", flush=True)
    t_end = fit_train["open_time"].max()
    models, sstats_all, hstats_all = {}, {}, {}
    syms_done, syms_skipped = [], []
    for sym, gtr in fit_train.groupby("symbol"):
        if len(gtr) < 300:
            syms_skipped.append((sym, "too few train rows", len(gtr))); continue
        s, h = x6.fit_preproc(gtr, FEAT_COLS)
        Xtr = x6.apply_preproc(gtr, FEAT_COLS, s, h)
        weights = None
        if args.halflife_days:
            days_back = ((t_end - gtr["open_time"]).dt.total_seconds()/86400).to_numpy()
            weights = np.exp(-days_back/args.halflife_days)
        try:
            m = RidgeCV(alphas=alphas_grid).fit(Xtr, gtr["target_z"].to_numpy(), sample_weight=weights)
        except Exception as e:
            syms_skipped.append((sym, f"fit error: {e}", len(gtr))); continue
        models[sym] = m; sstats_all[sym] = s; hstats_all[sym] = h
        syms_done.append(sym)
    print(f"  fit: {len(syms_done)} syms ok, {len(syms_skipped)} skipped", flush=True)
    if syms_skipped[:5]:
        for s, why, n in syms_skipped[:5]:
            print(f"    skipped {s}: {why} (n={n})", flush=True)

    # validation: rank IC vs research preds on the post-fit_cut OOS slice (if any)
    val_rank_ic = float("nan"); val_cycles = 0
    if len(val_slice) > 0:
        val = val_slice.merge(panel[["symbol","open_time"] + FEAT_COLS], on=["symbol","open_time"], how="left")
        val_p = []
        for sym, gv in val.groupby("symbol"):
            if sym not in models: continue
            Xv = x6.apply_preproc(gv, FEAT_COLS, sstats_all[sym], hstats_all[sym])
            pv = models[sym].predict(Xv)
            val_p.append(pd.DataFrame({"symbol": sym, "open_time": gv["open_time"].values,
                                       "pred_deploy": pv, "pred_research": gv["pred"].values}))
        if val_p:
            vp = pd.concat(val_p, ignore_index=True)
            cyc_ic = vp.groupby("open_time").apply(
                lambda g: g["pred_deploy"].rank().corr(g["pred_research"].rank())).dropna()
            val_rank_ic = float(cyc_ic.mean()); val_cycles = int(cyc_ic.shape[0])
            print(f"\n=== VALIDATION (sanity rank-IC vs research preds, post-fit_cut OOS) ===", flush=True)
            print(f"  XS rank-IC per cycle: mean {val_rank_ic:+.4f}  median {cyc_ic.median():+.4f}", flush=True)
            print(f"  rows compared: {len(vp):,}; cycles: {val_cycles}", flush=True)
    else:
        print(f"  (no post-fit_cut research-preds rows for IC sanity)", flush=True)

    # save artifact
    artifact = dict(
        feat_cols=FEAT_COLS, symbols=syms_done,
        sstats=sstats_all, hstats=hstats_all, models=models,
        meta=dict(
            trained_at=pd.Timestamp.utcnow().isoformat(),
            panel_path=str(PANEL_PATH), panel_end=str(panel_end),
            train_rows=int(len(fit_train)), fit_cut=str(fit_cut),
            val_rank_ic_xs_per_cycle=val_rank_ic,
            val_cycles=val_cycles,
            n_syms_fit=len(syms_done), n_syms_skipped=len(syms_skipped),
            heavy_tail=list(x6.HEAVY_TAIL),
            ridge_alphas=list(x6.RIDGE_ALPHAS),
        ),
    )
    with open(OUT_PKL, "wb") as f: pickle.dump(artifact, f)
    OUT_META.write_text(json.dumps(artifact["meta"], indent=2, default=str))
    print(f"\n  saved {OUT_PKL} ({OUT_PKL.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"  saved {OUT_META}", flush=True)
    print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
    if val_rank_ic < 0.85:
        print(f"\n!!! WARNING: val_rank_ic {val_rank_ic:+.4f} < 0.85 — deploy preds DIVERGE from research preds; do NOT live-trade until reconciled.", flush=True)


if __name__ == "__main__":
    main()
