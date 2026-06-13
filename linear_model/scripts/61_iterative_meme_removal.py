"""Step 61: iteratively remove long-tail meme drivers + RETRAIN until the tail
edge is exhausted. Answers: what % of the universe are the 'alpha tokens'?

Each iteration:
  1. retrain V2 Ridge on (110 − cumulative_dropped)   [Step 58 machinery]
  2. causal aggregator (+funding, proven immaterial) → Sharpe, CI, folds+
  3. naive per-symbol gross attribution → rank the dominant contributors
     (naive sum is invalid for PnL magnitude but a fine SELECTION ranking —
      it correctly flagged SIREN/JELLY then PIPPIN/BROCCOLI; measurement is
      the retrain+placebo, selection is the attribution)
  4. drop the symbols making up the top ~60% of positive gross (2..6/iter)
  5. checkpoint a row to summary.csv

Stop when causal Sharpe ≤ 0.3, or CI crosses 0 with no dominant tail
(top-1 contributor < 20% of positive gross), or n<70, or 12 iters.
Final placebo (P1/P2) on the last model for the verdict.

The cumulative dropped count / 110 at the stop = the 'alpha-token %'.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name, rel):
    sp = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

psl = _imp("psl", "scripts/phase_ah_sleeve.py")
s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s59 = _imp("s59", "linear_model/scripts/59_clean108_funding.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v4_xs import block_bootstrap_ci

PANEL_111 = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
OUT = REPO / "linear_model/results/step61_iterative_removal"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
N_PLACEBO = 100
MAX_ITERS = 12
STOP_SHARPE = 0.30


def attribution(records, apd_full):
    al = apd_full.set_index(["open_time", "symbol"])["alpha_beta"].to_dict()
    contrib = defaultdict(float)
    for _, r in records.iterrows():
        if not r["traded"]:
            continue
        tt = r["time"]; L = list(r["long_basket"]); S = list(r["short_basket"])
        nL, nS = max(len(L), 1), max(len(S), 1)
        for x in L:
            a = al.get((tt, x))
            if a is not None and not pd.isna(a):
                contrib[x] += a * 1e4 / nL
        for x in S:
            a = al.get((tt, x))
            if a is not None and not pd.isna(a):
                contrib[x] += -a * 1e4 / nS
    ad = pd.DataFrame({"symbol": list(contrib),
                        "contrib_bps": [contrib[k] for k in contrib]})
    return ad.sort_values("contrib_bps", ascending=False).reset_index(drop=True)


def run_once(dropped):
    """Full retrain + causal evaluation on (110 − dropped). Returns metrics."""
    listings = s58.get_listings()
    panel = pd.read_parquet(PANEL_111)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[~panel["symbol"].isin(set(dropped) | {"BTCUSDT"})].copy()
    syms = sorted(panel["symbol"].unique())

    folds_all = _multi_oos_splits(panel)
    fold0_train_idx = _slice(panel, folds_all[0])[0].index
    tr0 = panel.loc[fold0_train_idx]
    sig = tr0.groupby("symbol")["alpha_beta"].std()
    med = float(sig.dropna().median())
    panel["sigma_idio"] = panel["symbol"].map(sig).fillna(med).clip(lower=1e-6)
    panel = s58.build_target_z(panel, fold0_train_idx)

    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    X, feat_cols = s58.build_v2_features(panel, train_mask)
    panel_x = panel[["symbol", "open_time", "alpha_beta", "target_z",
                      "autocorr_pctile_7d"]].merge(
        X.drop(columns=["alpha_beta", "target_z", "autocorr_pctile_7d"]),
        on=["symbol", "open_time"], how="left")
    apd = s58.train_ridge(panel_x, folds_all, feat_cols)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]
    extra = panel[["symbol", "open_time", "return_pct", "exit_time"]].copy()
    extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
    apd = apd.merge(extra, on=["symbol", "open_time"], how="left")

    syms_set = set(syms)
    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cut = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in syms_set if listings.get(s) and listings[s] <= cut}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    df_ic = s58.compute_trailing_ic(apd, sampled_t, 90)
    apd = apd.merge(df_ic, on=["symbol", "open_time"], how="left")
    apd["trail_ic"] = apd["trail_ic"].fillna(0)
    apd["pred_B"] = apd["pred_z"] * apd["trail_ic"]
    # CRITICAL (matches validated Step 58b): rolling-IC universe is selected on
    # pred_z; the protocol then trades on pred_B. Building the universe on pred_B
    # picks the wrong symbols and gave the bogus -0.66 in the first 61 run.
    apd["pred"] = apd["pred_z"]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)
    apd["pred"] = apd["pred_B"]
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                  values="alpha_A", aggfunc="first").sort_index()
    fund_block, _ = s59.infer_funding(syms, sampled_t)

    records = psl.run_production_protocol_save_sleeves(apd, universe)
    df = s59.aggregate_causal_funding(records, alpha_wide, fund_block)
    net = df["net_pnl_bps"].to_numpy()
    sh = s59._sharpe(net)
    lo, hi = block_bootstrap_ci(net, statistic=s59._sharpe, block_size=7,
                                 n_boot=1000)[1:]
    fp = s59.folds_positive(df)
    ad = attribution(records, apd)
    pos = ad[ad.contrib_bps > 0]["contrib_bps"]
    pos_tot = pos.sum() if len(pos) else 1.0
    a = np.abs(net); order = np.argsort(-a)
    top5pct = net[order[:max(1, int(len(net) * 0.05))]].sum() / net.sum() * 100 \
        if net.sum() else 0.0
    return dict(n=len(syms), sharpe=sh, ci_lo=lo, ci_hi=hi, folds_pos=fp,
                top1_share=float(ad.contrib_bps.iloc[0] / pos_tot * 100),
                top5_share=float(ad.head(5).contrib_bps.sum() / pos_tot * 100),
                top5pct_cycles=float(top5pct), ad=ad, apd=apd,
                universe=universe, alpha_wide=alpha_wide, fund_block=fund_block,
                sampled_t=sampled_t)


def main():
    print("=" * 100, flush=True)
    print("  STEP 61: iterative meme removal — how many 'alpha tokens'?", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    dropped = ["SIRENUSDT", "JELLYJELLYUSDT"]   # start at the known clean-108
    rows = []
    last = None
    for it in range(MAX_ITERS):
        ts = time.time()
        try:
            r = run_once(dropped)
        except Exception as e:
            print(f"  iter {it}: FAILED {type(e).__name__}: {e}", flush=True)
            break
        pct = len(dropped) / 110 * 100
        top = r["ad"].head(4)["symbol"].tolist()
        print(f"\n[iter {it}] dropped={len(dropped)} ({pct:.0f}% of 110)  "
              f"n={r['n']}  Sharpe={r['sharpe']:+.2f} [{r['ci_lo']:+.2f},"
              f"{r['ci_hi']:+.2f}]  folds+={r['folds_pos']}/9  "
              f"top1={r['top1_share']:.0f}%  top5={r['top5_share']:.0f}%  "
              f"top5%cyc={r['top5pct_cycles']:.0f}%  ({time.time()-ts:.0f}s)",
              flush=True)
        print(f"           next drivers: " + ", ".join(
            f"{s}={r['ad'].set_index('symbol').contrib_bps[s]:+.0f}" for s in top),
            flush=True)
        rows.append({k: r[k] for k in ("n", "sharpe", "ci_lo", "ci_hi",
                     "folds_pos", "top1_share", "top5_share", "top5pct_cycles")}
                    | {"n_dropped": len(dropped), "pct_dropped": pct,
                       "dropped": ";".join(dropped)})
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
        last = (dropped.copy(), r)

        # stop conditions
        if r["sharpe"] <= STOP_SHARPE:
            print(f"\n  STOP: Sharpe {r['sharpe']:+.2f} ≤ {STOP_SHARPE} — "
                  f"tail edge exhausted at {len(dropped)} dropped "
                  f"({pct:.0f}% of universe).", flush=True)
            break
        if r["ci_lo"] < 0 and r["top1_share"] < 20:
            print(f"\n  STOP: CI crosses 0 and no dominant tail "
                  f"(top1 {r['top1_share']:.0f}%) — exhausted at {pct:.0f}%.",
                  flush=True)
            break
        if r["n"] < 70:
            print(f"\n  STOP: universe shrunk to {r['n']} (<70).", flush=True)
            break

        # select next batch: symbols making up top ~60% of positive gross
        pos = r["ad"][r["ad"].contrib_bps > 0].reset_index(drop=True)
        csum = pos.contrib_bps.cumsum() / pos.contrib_bps.sum()
        kbatch = int((csum < 0.60).sum()) + 1
        kbatch = max(2, min(6, kbatch))
        dropped = dropped + pos.head(kbatch)["symbol"].tolist()

    # final placebo on the last surviving model
    if last is not None:
        dl, r = last
        print(f"\n  Final P1/P2 placebo on model with {len(dl)} dropped "
              f"({len(dl)/110*100:.0f}% of universe), Sharpe={r['sharpe']:+.2f}:",
              flush=True)
        apd = r["apd"].copy()
        ul = s59.build_liquidity_universe(r["sampled_t"],
                                           sorted(apd["symbol"].unique()), 30)
        for nm, uv in [("P1", ul), ("P2", r["universe"])]:
            ps = []
            for sd in range(N_PLACEBO):
                rp = psl.run_production_protocol_save_sleeves(apd, uv,
                                                               placebo_seed=sd)
                dp = s59.aggregate_causal_funding(rp, r["alpha_wide"],
                                                   r["fund_block"])
                ps.append(s59._sharpe(dp["net_pnl_bps"].to_numpy()))
            ps = np.array(ps); p95 = float(np.percentile(ps, 95))
            print(f"    {nm}: p95={p95:+.2f}  real rank "
                  f"p{(ps < r['sharpe']).mean()*100:.0f}  "
                  f"{'PASS' if r['sharpe'] > p95 else 'FAIL'}", flush=True)

    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
