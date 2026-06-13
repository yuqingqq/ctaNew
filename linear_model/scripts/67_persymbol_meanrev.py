"""Step 67: per-SYMBOL model on the mean-rev-v2 architecture (the genuinely
untested path — SEG/CAL were on the LGBM sleeve, not this engine), + an honest
quantification of the real diversified blue-chip alpha size.

Part A: train an INDEPENDENT Ridge per symbol (V2 features, walk-forward 9-fold,
        each symbol uses only its own rows), feed pred_z into the SAME mean-rev-v2
        cross-sectional-z engine. Measure vs pooled:
          - overall per-cycle IC          (pooled = -0.013)
          - nested-OOS Sharpe + CI
          - per-TRADE mean net bps + bootstrap CI  (no annualization inflation)
          - survives drop-BIO+VVV-retrain?  (the de-concentration test)
          - random-pool placebo
Part B: honest blue-chip alpha SIZE for pooled vs per-symbol — per-trade and
        per-cycle bps with CI, and the post-de-concentration "robust core".

Honest prior: per-symbol full-sample IC mean -0.0099 (mostly negative, only
large-sample-stable); per-symbol models have ~1/44 the data → expected ≈0/neg.
But it has NOT been tried on THIS architecture; this is the clean decisive test.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    sp = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

psl = _imp("psl", "scripts/phase_ah_sleeve.py")
s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s65 = _imp("s65", "linear_model/scripts/65_tail_attrib_deconc.py")
s59 = s64.s59
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL_111 = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
OUT = REPO / "linear_model/results/step67_persymbol"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s64.OOS, s64.BLOCK
ALPHAS = s58.ALPHAS
AUTO = s58.AUTO_THRESH


def build_panel(drop):
    listings = s58.get_listings()
    hl = pd.read_csv(HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= 2e6)]["symbol"])
    panel = pd.read_parquet(PANEL_111)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[panel["symbol"].isin(keep)
                  & ~panel["symbol"].isin(set(drop) | {"BTCUSDT"})].copy()
    folds = _multi_oos_splits(panel)
    f0 = _slice(panel, folds[0])[0].index
    sg = panel.loc[f0].groupby("symbol")["alpha_beta"].std()
    med = float(sg.dropna().median())
    panel["sigma_idio"] = panel["symbol"].map(sg).fillna(med).clip(lower=1e-6)
    panel = s58.build_target_z(panel, f0)
    tm = panel["open_time"].between(_slice(panel, folds[0])[0].open_time.min(),
                                    _slice(panel, folds[0])[0].open_time.max())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            listings[s] = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    X, fc = s58.build_v2_features(panel, tm)
    px = panel[["symbol", "open_time", "alpha_beta", "target_z",
                "autocorr_pctile_7d"]].merge(
        X.drop(columns=["alpha_beta", "target_z", "autocorr_pctile_7d"]),
        on=["symbol", "open_time"], how="left")
    return panel, px, fc, folds


def train_persymbol(px, folds, fc):
    """Independent RidgeCV per symbol, walk-forward; OOS pred_z."""
    out = []
    for fid in range(10):
        if fid >= len(folds):
            continue
        tr, _, te = _slice(px, folds[fid])
        tr = tr[tr["autocorr_pctile_7d"] >= AUTO].dropna(subset=["target_z"])
        te = te.dropna(subset=["target_z"]).copy()
        if len(te) < 50:
            continue
        preds = np.full(len(te), np.nan)
        te_sym = te["symbol"].to_numpy()
        for sym, g in tr.groupby("symbol"):
            if len(g) < 300:
                continue
            m = RidgeCV(alphas=ALPHAS, scoring="r2", fit_intercept=True)
            m.fit(g[fc].to_numpy(np.float32), g["target_z"].to_numpy(np.float32))
            sel = te_sym == sym
            if sel.any():
                preds[sel] = m.predict(
                    te.loc[sel, fc].to_numpy(np.float32)).astype(np.float32)
        d = te[["symbol", "open_time", "alpha_beta"]].copy()
        d["pred_z"] = preds
        d["fold"] = fid
        out.append(d)
    return pd.concat(out, ignore_index=True).dropna(subset=["pred_z"])


def finalize(apd, panel):
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    ex = panel[["symbol", "open_time", "return_pct", "exit_time"]].copy()
    ex["exit_time"] = pd.to_datetime(ex["exit_time"], utc=True)
    apd = apd.merge(ex, on=["symbol", "open_time"], how="left")
    tt = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    dic = s58.compute_trailing_ic(apd, tt, 90)
    apd = apd.merge(dic, on=["symbol", "open_time"], how="left")
    apd["trail_ic"] = apd["trail_ic"].fillna(0)
    apd["pred_B"] = apd["pred_z"] * apd["trail_ic"]
    apd["pred"] = apd["pred_B"]
    return apd


def cyc_ic(apd):
    o = apd.dropna(subset=["pred_z", "alpha_beta"])
    ic = o.groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    return float(ic.mean())


def evaluate(apd, label):
    aw, pzw, tw, fw, bw, sig, nsy = _piv(apd)
    s65.COST = s64.COST
    nd, ntr, npo = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    n = nd["net"].to_numpy()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=1000)[1:]
    fp = sum(1 for _, g in nd.groupby("fold") if s59._sharpe(g["net"]) > 0)
    # per-trade bps + bootstrap CI (NO annualization)
    tr_pf = tr_mean = tr_lo = tr_hi = np.nan
    if len(ntr):
        c = ntr["cum_bps"].to_numpy()
        tr_pf = (c[c > 0].sum() / -c[c < 0].sum()) if (c < 0).any() else np.inf
        tr_mean = float(c.mean())
        bs = [np.random.default_rng(k).choice(c, len(c)).mean()
              for k in range(1000)]
        tr_lo, tr_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    # per-cycle bps + CI
    cb = [np.random.default_rng(k).choice(n, len(n)).mean() for k in range(1000)]
    pc_lo, pc_hi = float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))
    ic = cyc_ic(apd)
    print(f"  [{label}] N={nsy} IC={ic:+.4f} | nested Sh {sh:+.2f} "
          f"[{lo:+.2f},{hi:+.2f}] fp={fp}/9 | per-CYCLE net {n.mean():+.2f} bps "
          f"CI[{pc_lo:+.2f},{pc_hi:+.2f}] | per-TRADE {tr_mean:+.1f} bps "
          f"CI[{tr_lo:+.1f},{tr_hi:+.1f}] PF {tr_pf:.2f} | tot {n.sum():,.0f}",
          flush=True)
    return dict(label=label, N=nsy, ic=ic, sharpe=sh, sh_lo=lo, sh_hi=hi, fp=fp,
                pc_mean=float(n.mean()), pc_lo=pc_lo, pc_hi=pc_hi,
                tr_mean=tr_mean, tr_lo=tr_lo, tr_hi=tr_hi, tr_pf=tr_pf,
                total=float(n.sum()))


def _piv(apd):
    syms = sorted(apd["symbol"].unique())
    f0 = apd[apd["fold"] == 0]
    sg = f0.groupby("symbol")["alpha_beta"].std()
    s64.sig_med = float(sg.dropna().median())
    sig = sg.fillna(s64.sig_med).to_dict()
    samp = sorted(apd[apd["fold"].isin(OOS)]["open_time"].unique())[::BLOCK]
    aw = apd.pivot_table(index="open_time", columns="symbol",
                         values="alpha_beta", aggfunc="first").sort_index()
    pzw = apd.pivot_table(index="open_time", columns="symbol",
                          values="pred_z", aggfunc="first").sort_index()
    tw = apd.pivot_table(index="open_time", columns="symbol",
                         values="trail_ic", aggfunc="first").sort_index()
    fw, _ = s59.infer_funding(syms, samp)
    bw = s64.recover_beta(apd)
    return aw, pzw, tw, fw, bw, sig, len(syms)


def main():
    print("=" * 100, flush=True)
    print("  STEP 67: per-symbol model on mean-rev-v2 + honest blue-chip alpha",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    res = []

    print("\n--- POOLED baseline (reuse Step 62 predictions) ---", flush=True)
    pooled = pd.read_parquet(REPO / "linear_model/results/step62_bluechip44/predictions.parquet")
    pooled["open_time"] = pd.to_datetime(pooled["open_time"], utc=True)
    res.append(evaluate(pooled, "pooled-44"))

    print("\n--- PART A: PER-SYMBOL model on mean-rev-v2 (44-sym) ---", flush=True)
    panel, px, fc, folds = build_panel([])
    aps = train_persymbol(px, folds, fc)
    aps = finalize(aps, panel)
    aps.to_parquet(OUT / "persym_predictions.parquet", index=False)
    res.append(evaluate(aps, "persym-44"))

    print("\n--- de-concentration: drop BIO+VVV, RETRAIN per-symbol ---", flush=True)
    panel2, px2, fc2, folds2 = build_panel(["BIOUSDT", "VVVUSDT"])
    aps2 = finalize(train_persymbol(px2, folds2, fc2), panel2)
    res.append(evaluate(aps2, "persym-drop2-42"))

    pd.DataFrame(res).to_csv(OUT / "summary.csv", index=False)
    print(f"\n{'='*100}\n  ANSWER: how large is the real blue-chip alpha?",
          flush=True)
    print(f"{'='*100}", flush=True)
    for r in res:
        verdict = ("REAL+robust" if (r["tr_lo"] > 0 and r["sh_lo"] > 0)
                   else "not-significant (CI crosses 0)")
        print(f"  {r['label']:16s}: per-trade {r['tr_mean']:+.1f}bps "
              f"CI[{r['tr_lo']:+.1f},{r['tr_hi']:+.1f}] | per-cycle "
              f"{r['pc_mean']:+.2f}bps CI[{r['pc_lo']:+.2f},{r['pc_hi']:+.2f}] | "
              f"IC {r['ic']:+.4f} | {verdict}", flush=True)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
