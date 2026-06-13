"""Step 66: settle the two real concerns about mean-rev-v2 on its ACTUAL design
(single-name + BTC-hedge allowed; bothsides was invalid).

  Part A — COST-STRESS: rerun the nested-OOS design at effective round-trip
            cost 9 / 18 / 27 bps (cost-mult 1x/2x/3x of the 2.25 bps/|Δw| rate).
            Tests the gross-PF≈1.06 fragility. If NET dies at 2x → too thin.

  Part B — ITERATIVE DROP-&-RETRAIN (Step 61 methodology): drop the top
            idiosyncratic tail driver, RETRAIN V2 on the reduced universe,
            rerun nested-OOS, find the new driver, repeat. Distinguishes:
            (1) BIO-specific fluke (dies on drop-1),
            (2) relocates to next volatile name (same structural pattern),
            (3) survives → genuinely diversified convex edge.

Reuses validated machinery: s58 (V2 retrain), s64 (engine consts/β/GRID),
s65.nested+runL (logged nested engine — smoke-validated).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

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
STEP62 = REPO / "linear_model/results/step62_bluechip44/predictions.parquet"
OUT = REPO / "linear_model/results/step66_coststress_iterdrop"
OUT.mkdir(parents=True, exist_ok=True)
OOS, BLOCK = s64.OOS, s64.BLOCK
BASE_COST = s64.COST                       # 2.25 bps per unit |Δw|
VOL_THRESH = 2e6
MAXIT = 8


def retrain(dropped):
    """V2 Ridge on (HL & vol>=$2M 44) − dropped. Mirrors Step 62."""
    listings = s58.get_listings()
    hl = pd.read_csv(HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= VOL_THRESH)]["symbol"])
    panel = pd.read_parquet(PANEL_111)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[panel["symbol"].isin(keep)
                  & ~panel["symbol"].isin(set(dropped) | {"BTCUSDT"})].copy()
    folds_all = _multi_oos_splits(panel)
    f0 = _slice(panel, folds_all[0])[0].index
    sg = panel.loc[f0].groupby("symbol")["alpha_beta"].std()
    med = float(sg.dropna().median())
    panel["sigma_idio"] = panel["symbol"].map(sg).fillna(med).clip(lower=1e-6)
    panel = s58.build_target_z(panel, f0)
    tm = panel["open_time"].between(_slice(panel, folds_all[0])[0].open_time.min(),
                                    _slice(panel, folds_all[0])[0].open_time.max())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            listings[s] = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    X, fc = s58.build_v2_features(panel, tm)
    px = panel[["symbol", "open_time", "alpha_beta", "target_z",
                "autocorr_pctile_7d"]].merge(
        X.drop(columns=["alpha_beta", "target_z", "autocorr_pctile_7d"]),
        on=["symbol", "open_time"], how="left")
    apd = s58.train_ridge(px, folds_all, fc)
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


def pivots(apd):
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


def metrics(nd, ntr, npo):
    n = nd["net"].to_numpy(); tot = n.sum()
    sh = s59._sharpe(n)
    lo, hi = block_bootstrap_ci(n, statistic=s59._sharpe, block_size=7,
                                n_boot=1000)[1:]
    fp = sum(1 for _, g in nd.groupby("fold") if s59._sharpe(g["net"]) > 0)
    a = np.abs(n); tm = a >= np.quantile(a, 0.95)
    pf = drv = sms = np.nan
    if len(ntr):
        c = ntr["cum_bps"]
        pf = (c[c > 0].sum() / -c[c < 0].sum()) if (c < 0).any() else np.inf
    if len(npo):
        pt = npo[npo["time"].isin(set(nd[tm]["time"]))]
        ag = pt.groupby("symbol")["contrib_bps"].sum().sort_values(ascending=False)
        if len(ag):
            drv = ag.index[0]
            sm = sum(v for k, v in ag.items() if k in s65.SEMI_MEME)
            sms = sm / ag.sum() * 100 if ag.sum() else np.nan
    return dict(sharpe=sh, lo=lo, hi=hi, fp=fp, total=tot, pf_gross=pf,
                top_driver=drv, semi_meme_pct=sms,
                body95=s59._sharpe(n[~tm]))


def nested_at(apd, cost_mult):
    aw, pzw, tw, fw, bw, sig, nsy = pivots(apd)
    s65.COST = BASE_COST * cost_mult
    try:
        nd, ntr, npo = s65.nested(apd, aw, fw, pzw, tw, sig, bw, "design")
    finally:
        s65.COST = BASE_COST
    return nd, ntr, npo, nsy


def main():
    print("=" * 100, flush=True)
    print("  STEP 66: cost-stress + iterative drop-&-retrain (mean-rev-v2 design)",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    apd0 = pd.read_parquet(STEP62)
    apd0["open_time"] = pd.to_datetime(apd0["open_time"], utc=True)

    print("\n========== PART A: COST-STRESS (44-sym, design) ==========", flush=True)
    rowsA = []
    for mult in (1.0, 2.0, 3.0):
        nd, ntr, npo, nsy = nested_at(apd0, mult)
        m = metrics(nd, ntr, npo)
        rt = 9 * mult
        print(f"  cost {mult:.0f}x (~{rt:.0f}bps RT): nested Sharpe "
              f"{m['sharpe']:+.2f} [{m['lo']:+.2f},{m['hi']:+.2f}] fp={m['fp']}/9 "
              f"total {m['total']:,.0f}bps body95 {m['body95']:+.2f} "
              f"PFgross {m['pf_gross']:.2f}", flush=True)
        rowsA.append(dict(cost_mult=mult, rt_bps=rt, **{k: m[k] for k in
                     ("sharpe", "lo", "hi", "fp", "total", "body95", "pf_gross")}))
    pd.DataFrame(rowsA).to_csv(OUT / "partA_coststress.csv", index=False)
    a2 = rowsA[1]
    print(f"  => VERDICT A: at 2x cost net total {a2['total']:,.0f}bps, "
          f"Sharpe {a2['sharpe']:+.2f}, CI_lo {a2['lo']:+.2f} "
          f"=> {'SURVIVES' if a2['lo'] > 0 and a2['total'] > 0 else 'DIES (too thin)'}",
          flush=True)

    print("\n========== PART B: ITERATIVE DROP-&-RETRAIN (1x cost) ==========",
          flush=True)
    dropped = []
    rowsB = []
    for it in range(MAXIT):
        ts = time.time()
        apd = apd0 if not dropped else retrain(dropped)
        nd, ntr, npo, nsy = nested_at(apd, 1.0)
        m = metrics(nd, ntr, npo)
        print(f"  [it{it}] dropped={dropped} N={nsy} -> nested "
              f"{m['sharpe']:+.2f} [{m['lo']:+.2f},{m['hi']:+.2f}] fp={m['fp']}/9 "
              f"total {m['total']:,.0f} PFg {m['pf_gross']:.2f} "
              f"body95 {m['body95']:+.2f} top_drv={m['top_driver']} "
              f"semimeme {m['semi_meme_pct']:.0f}% ({time.time()-ts:.0f}s)",
              flush=True)
        rowsB.append(dict(it=it, n_dropped=len(dropped),
                          dropped=";".join(dropped) or "-", n_syms=nsy,
                          **{k: m[k] for k in ("sharpe", "lo", "hi", "fp",
                             "total", "pf_gross", "body95", "top_driver",
                             "semi_meme_pct")}))
        pd.DataFrame(rowsB).to_csv(OUT / "partB_iterdrop.csv", index=False)
        if m["total"] <= 0 or m["lo"] < 0 or (m["pf_gross"] < 1.0) or nsy < 30:
            why = ("total<=0" if m["total"] <= 0 else "CI_lo<0" if m["lo"] < 0
                   else "PFgross<1.0" if m["pf_gross"] < 1.0 else "N<30")
            print(f"  STOP ({why}): edge gone after dropping {len(dropped)} "
                  f"({dropped}). Pattern => "
                  f"{'BIO-fluke' if len(dropped)<=1 else 'relocates/structural'}",
                  flush=True)
            break
        if m["top_driver"] is None or (isinstance(m["top_driver"], float)):
            print("  STOP: no identifiable tail driver.", flush=True)
            break
        dropped = dropped + [m["top_driver"]]
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
