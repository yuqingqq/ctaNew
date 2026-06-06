"""Train + cache the per-symbol two-book models ONCE (monthly retrain), so the live cycle only has to
PREDICT new bars (predict_twobook_incremental.py) instead of re-fitting all 8 walk-forward folds × 159
syms × 2 books (~2500 fits) every cycle.

Fits, per symbol: a FLOW model (per-sym RidgeCV on V0+flow, recency-60) and a PRICE model (V0 only),
on all labelled data with exit_time < fit_cut. Exact same machinery as loop2_iter28 gen(). Saves the
fitted model + preprocessing to live/models/twobook_{flow,price}_models.pkl.

Usage: python3 live/train_twobook_models.py [--fit-cut 2026-05-26]  (default: latest panel date - 1d embargo)
"""
import sys, glob, pickle, json, argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
import warnings; warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
MODELS = REPO/"live/models"
V0 = x6.BASE + x6.COHORT_EXTRAS; EMB = pd.Timedelta(days=1); HL = 60.0
# LEAN feature set (2026-06-02 funding-fix + prune study): drop funding (redundant with xs-z target +
# 2/3 noise), keep only VPIN/TFI flow (the other 10 flow feats are noise by IC+coef+LOO). rvol_7d KEPT
# (its LOO "improvement" was 78%-one-fold snooping). Price book = V0_LEAN(14); flow book = +VPIN/TFI(18).
V0_LEAN = [f for f in V0 if not f.startswith("funding")]
FLOW_KEEP = ["fl_vpin", "fl_vpin_1d", "fl_tfi", "fl_tfi_1d"]
RR = ["resid_rev_2", "resid_rev_3"]   # v1 resid_rev long-ranker features (book B); fit on V0_LEAN+RR, frozen single-fit


def build_flow():
    rows = []
    for fp in sorted(glob.glob(str(REPO/"data/ml/cache/flow_*.parquet"))):
        sym = Path(fp).stem.replace("flow_", "")
        try: f = pd.read_parquet(fp)
        except Exception: continue
        R = lambda c, how: getattr(f[c].resample("4h", label="right", closed="right"), how)()
        agg = pd.DataFrame({"tfi": R("tfi", "mean"), "sv_z": R("signed_volume_z", "mean"), "vpin": R("vpin", "mean"),
            "kyle": R("kyle_lambda", "mean"), "aggr": R("aggressor_count_ratio", "mean"),
            "lgvol": R("large_trade_volume", "sum"), "totvol": R("total_volume", "sum"),
            "bv": R("buy_volume", "sum"), "sv": R("sell_volume", "sum")})
        agg["lg_share"] = agg["lgvol"]/agg["totvol"].replace(0, np.nan)
        agg["bs_imb"] = (agg["bv"]-agg["sv"])/(agg["bv"]+agg["sv"]).replace(0, np.nan)
        agg = agg[agg.index.hour % 4 == 0]; feats = {}
        for c in ["tfi", "sv_z", "vpin", "kyle", "aggr", "lg_share", "bs_imb"]:
            feats["fl_"+c] = agg[c].shift(1); feats["fl_"+c+"_1d"] = agg[c].rolling(6, min_periods=3).mean().shift(1)
        ff = pd.DataFrame(feats); ff["symbol"] = sym; ff["open_time"] = ff.index; rows.append(ff.reset_index(drop=True))
    F = pd.concat(rows, ignore_index=True); F["open_time"] = pd.to_datetime(F["open_time"], utc=True)
    return F, [c for c in F.columns if c.startswith("fl_")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fit-cut", default=None); a = ap.parse_args()
    F, flowcols = build_flow(); flowsyms = set(F.symbol.unique())
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "exit_time", "return_pct", "alpha_vs_btc_realized"]+V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].merge(F, on=["symbol", "open_time"], how="left")
    gc = pan.groupby("open_time"); mu = gc["return_pct"].transform("mean"); sd = gc["return_pct"].transform("std").replace(0, np.nan)
    pan["xs_z"] = ((pan["return_pct"]-mu)/sd).clip(-10, 10)
    # resid_rev (v1 long-ranker feature): -Σ trailing PAST per-bar BTC-residual alpha (PIT; 4h label => no overlap).
    pan = pan.sort_values(["symbol", "open_time"])
    _a = pan.groupby("symbol")["alpha_vs_btc_realized"]
    pan["resid_rev_2"] = -_a.transform(lambda s: s.shift(1).rolling(2).sum())   # 8h
    pan["resid_rev_3"] = -_a.transform(lambda s: s.shift(1).rolling(3).sum())   # 12h
    for c in RR: pan[c] = pan[c].fillna(0.0)
    fit_cut = pd.Timestamp(a.fit_cut, tz="UTC") if a.fit_cut else (pan["open_time"].max().normalize() - EMB)
    tr = pan[(pan.exit_time < fit_cut) & pan["xs_z"].notna()]
    t_end = tr["open_time"].max()
    flow_models, price_models, price_residrev_models = {}, {}, {}
    nf = npr = nrr = 0
    for sym, g in tr.groupby("symbol"):
        if len(g) < 300: continue
        w = np.exp(-((t_end - g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        y = g["xs_z"].to_numpy()
        # v1 base PRICE model — FULL V0 (funding RESTORED 2026-06-04: helps book-B-only +0.68 vs V0_LEAN; the
        # 2026-06-02 funding-drop was a two-book result, does NOT hold for book-B-only). Ranks the SHORT leg.
        try:
            s, h = x6.fit_preproc(g, V0); X = x6.apply_preproc(g, V0, s, h)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, y, sample_weight=w)
            price_models[sym] = (m, s, h, V0); npr += 1
        except Exception: pass
        # v1 resid_rev model (FULL V0 + resid_rev) — LONG-ranker for book B (dual-pred). Base price model ranks shorts.
        try:
            feats = V0 + RR
            s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, y, sample_weight=w)
            price_residrev_models[sym] = (m, s, h, feats); nrr += 1
        except Exception: pass
        # v1 is BOOK-B-ONLY — flow model (book A) DROPPED. (block kept disabled for reference; not trained)
        if False:
            fk = [c for c in FLOW_KEEP if c in flowcols]
            try:
                feats = V0_LEAN + fk
                s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, y, sample_weight=w)
                flow_models[sym] = (m, s, h, feats); nf += 1
            except Exception: pass
    MODELS.mkdir(parents=True, exist_ok=True)
    meta = {"fit_cut": str(fit_cut), "t_end": str(t_end), "HL": HL,
            "features": "full_V0", "funding": "restored_2026-06-04", "book": "low-vol single book", "v0": list(V0)}
    # convexity v1 = ONE book (low-vol). SHORT model = base V0; LONG model = V0+resid_rev (dual-pred). No flow/book-A.
    pickle.dump({"models": price_models, "meta": meta}, open(MODELS/"convexity_v1_short_model.pkl", "wb"))
    rr_meta = dict(meta); rr_meta["resid_rev_feats"] = RR
    pickle.dump({"models": price_residrev_models, "meta": rr_meta}, open(MODELS/"convexity_v1_long_model.pkl", "wb"))

    # FROZEN volatility split (champion = STATIC ranking at retrain; rolling re-rank hurts). The top-N by
    # trailing-30d rvol_7d as-of fit_cut go to the flow book; computed ONCE here on the training box (full
    # history) and shipped via git so the live server uses the IDENTICAL 80-set rather than recomputing
    # off its own warmup data (which would drift the split day-to-day). Daily script LOADS this, no recompute.
    SPLIT_N = 80; OOS_START = pd.Timestamp("2025-10-04", tz="UTC")
    lo = fit_cut - pd.Timedelta(days=30)
    rv = (pan[(pan.open_time >= lo) & (pan.open_time < fit_cut)]
          .groupby("symbol")["rvol_7d"].mean())
    # universe = symbols present in the book-B preds file since OOS start (excludes brand-new thin listings).
    # Fall back to price-model keys if absent (fresh box). Book A is dropped; this split only marks the top-N
    # high-vol names to EXCLUDE (book B = the rest).
    fp = REPO/"live/state/convexity/hl/v0full_hl60.parquet"
    if fp.exists():
        op = pd.read_parquet(fp, columns=["symbol", "open_time"]); op["open_time"] = pd.to_datetime(op["open_time"], utc=True)
        oos_univ = set(op[op.open_time >= OOS_START].symbol.unique())
    else:
        oos_univ = set(price_models)
    cand = [s for s in oos_univ if np.isfinite(rv.get(s, np.nan))]
    ranked = sorted(cand, key=lambda s: -rv[s])
    flow_book = ranked[:SPLIT_N]   # the EXCLUDED high-vol set (book B = universe minus this)
    # universe.json: the top-N high-vol names to EXCLUDE (book = the rest). 'exclude_high_vol' is the key (kept
    # 'flow_book' alias for back-compat with any old loader).
    split = {"asof": str(fit_cut), "n": len(flow_book), "rvol_window_days": 30,
             "exclude_high_vol": flow_book, "flow_book": flow_book}
    json.dump(split, open(MODELS/"convexity_v1_universe.json", "w"), indent=0)
    print(f"convexity v1 (single low-vol book): {npr} short(base) models, {nrr} long(resid_rev) models; "
          f"fit_cut={fit_cut.date()} t_end={t_end} → live/models/convexity_v1_{{short,long}}_model.pkl")
    print(f"universe: exclude top-{len(flow_book)} high-vol (rvol_7d as-of {fit_cut.date()}) "
          f"→ live/models/convexity_v1_universe.json")

    # regenerate the exec-server WS collector feed list (full universe minus dead/halted) so it stays in sync
    # with this retrain's universe — the 175-XS cross-section needs every live symbol fed.
    try:
        import live.gen_collector_universe as gcu; gcu.main()
    except Exception as e:
        print(f"WARN collector_universe regen failed: {e}")


if __name__ == "__main__":
    main()
