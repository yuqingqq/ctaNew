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
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "exit_time", "return_pct"]+V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour % 4 == 0) & (pan.open_time.dt.minute == 0)].merge(F, on=["symbol", "open_time"], how="left")
    gc = pan.groupby("open_time"); mu = gc["return_pct"].transform("mean"); sd = gc["return_pct"].transform("std").replace(0, np.nan)
    pan["xs_z"] = ((pan["return_pct"]-mu)/sd).clip(-10, 10)
    fit_cut = pd.Timestamp(a.fit_cut, tz="UTC") if a.fit_cut else (pan["open_time"].max().normalize() - EMB)
    tr = pan[(pan.exit_time < fit_cut) & pan["xs_z"].notna()]
    t_end = tr["open_time"].max()
    flow_models, price_models = {}, {}
    nf = npr = 0
    for sym, g in tr.groupby("symbol"):
        if len(g) < 300: continue
        w = np.exp(-((t_end - g["open_time"]).dt.total_seconds().to_numpy()/86400.0)/HL)
        y = g["xs_z"].to_numpy()
        # price model (V0)
        try:
            s, h = x6.fit_preproc(g, V0); X = x6.apply_preproc(g, V0, s, h)
            m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, y, sample_weight=w)
            price_models[sym] = (m, s, h, V0); npr += 1
        except Exception: pass
        # flow model (V0+flow) if flow populated
        if (sym in flowsyms) and g[flowcols].notna().any().all():
            try:
                feats = V0 + flowcols
                s, h = x6.fit_preproc(g, feats); X = x6.apply_preproc(g, feats, s, h)
                m = RidgeCV(alphas=x6.RIDGE_ALPHAS).fit(X, y, sample_weight=w)
                flow_models[sym] = (m, s, h, feats); nf += 1
            except Exception: pass
    MODELS.mkdir(parents=True, exist_ok=True)
    meta = {"fit_cut": str(fit_cut), "t_end": str(t_end), "HL": HL, "flowcols": flowcols}
    pickle.dump({"models": flow_models, "meta": meta}, open(MODELS/"twobook_flow_models.pkl", "wb"))
    pickle.dump({"models": price_models, "meta": meta}, open(MODELS/"twobook_price_models.pkl", "wb"))

    # FROZEN volatility split (champion = STATIC ranking at retrain; rolling re-rank hurts). The top-N by
    # trailing-30d rvol_7d as-of fit_cut go to the flow book; computed ONCE here on the training box (full
    # history) and shipped via git so the live server uses the IDENTICAL 80-set rather than recomputing
    # off its own warmup data (which would drift the split day-to-day). Daily script LOADS this, no recompute.
    SPLIT_N = 80; OOS_START = pd.Timestamp("2025-10-04", tz="UTC")
    lo = fit_cut - pd.Timedelta(days=30)
    rv = (pan[(pan.open_time >= lo) & (pan.open_time < fit_cut)]
          .groupby("symbol")["rvol_7d"].mean())
    # universe = the SAME set the validated daily logic ranks: symbols present in the flow-preds file since
    # OOS start (excludes brand-new thin listings that just crossed the 300-bar model threshold). Fall back
    # to flow-model keys only if the preds file is absent (fresh box before first predict run).
    fp = REPO/"live/state/convexity/hl/fullflow_hl60.parquet"
    if fp.exists():
        op = pd.read_parquet(fp, columns=["symbol", "open_time"]); op["open_time"] = pd.to_datetime(op["open_time"], utc=True)
        oos_univ = set(op[op.open_time >= OOS_START].symbol.unique())
    else:
        oos_univ = set(flow_models)
    cand = [s for s in oos_univ if np.isfinite(rv.get(s, np.nan))]
    ranked = sorted(cand, key=lambda s: -rv[s])
    flow_book = ranked[:SPLIT_N]
    split = {"asof": str(fit_cut), "n": len(flow_book), "rvol_window_days": 30, "flow_book": flow_book}
    json.dump(split, open(MODELS/"twobook_split.json", "w"), indent=0)
    print(f"trained: {nf} flow models, {npr} price models; fit_cut={fit_cut.date()} t_end={t_end} "
          f"→ live/models/twobook_*_models.pkl")
    print(f"frozen split: flow_book N={len(flow_book)} (top rvol_7d as-of {fit_cut.date()}) "
          f"→ live/models/twobook_split.json")


if __name__ == "__main__":
    main()
