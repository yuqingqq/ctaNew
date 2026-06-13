"""R2 — profit-lever stack vs the R1 deployable baseline.

R2a: rvol_7d/ret_3d/btc_rvol_7d as MODEL features (predictions from
     R2a_retrain.py), run through the V3.1 sleeve + cap frontier.
R2b: longer effective hold via equal-weight overlapping sleeves at
     {24h ref, 48h, 72h} on the R1 baseline predictions; + a tail-stressed
     cost (3x realized-unit cost on top-vol-decile legs).
R2c: R2a predictions at the best R2b hold.

Gates (PLAN.md v3): deployable criteria 1-6 + LOFO single-fold sign-flip
kill on the lift vs R1. Pre-registered: >=1 lever clears criteria 1-6 with
lift >= +0.3 over R1 (paired-block-bootstrap CI excluding 0) AND no LOFO
sign-flip; else the levers are refuted (recorded honestly).
"""
from __future__ import annotations
import json, sys, time, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research/portable_alpha_2026-05-19/scripts"))
import phase_ah_sleeve as PA
import R1_baseline_frontier as R1
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "research/portable_alpha_2026-05-19/results"
APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
R2A = REPO / "research/portable_alpha_2026-05-19/results/_cache/all_predictions_R2a.parquet"
HE = R1.HE
CPY = R1.CPY


def _sh(x):
    return R1._sharpe(np.asarray(x, float))


def aggregate_hold(records, fwd, sigma, advc, hold_bars, n_sleeves,
                   cap_frac=np.inf, cost_mode="flat45", tail_q=None):
    """Equal-weight overlapping sleeves at arbitrary hold; optional per-name
    cap and a tail-stressed realized cost (tail_q = sigma cross-sec quantile
    above which the realized unit cost is tripled)."""
    bar = pd.Timedelta(minutes=5)
    queue = deque(maxlen=n_sleeves)
    prev = {}
    rows = []
    flat9 = cost_mode == "flat9"
    realized = cost_mode in ("realized", "tail")
    for _, rec in records.iterrows():
        t, fold = rec["time"], rec["fold"]
        lb, sb = rec["long_basket"], rec["short_basket"]
        if lb and sb:
            queue.append({"entry_time": t, "longs": lb, "shorts": sb})
        queue = deque([s for s in queue if (t - s["entry_time"]) < hold_bars * bar],
                      maxlen=n_sleeves)
        tw = defaultdict(float)
        sw = 1.0 / n_sleeves
        for sl in queue:
            L, S = sl["longs"], sl["shorts"]
            if not L or not S:
                continue
            for s in L:
                tw[s] += sw / len(L)
            for s in S:
                tw[s] -= sw / len(S)
        tw = R1._apply_cap(dict(tw), cap_frac)
        g = 0.0
        if t in fwd.index:
            r = fwd.loc[t]
            for s, w in prev.items():
                rv = r.get(s, np.nan)
                if rv == rv:
                    g += w * rv * 1e4
        sig_t = sigma.loc[t] if (tail_q is not None and t in sigma.index) else None
        thr = np.nanquantile(sig_t.to_numpy(), tail_q) if sig_t is not None else None
        cost = 0.0
        for s in set(tw) | set(prev):
            d = abs(tw.get(s, 0.0) - prev.get(s, 0.0))
            if d == 0:
                continue
            if realized:
                u = advc.get(s, R1.COST_UNIT)
                if thr is not None and sig_t is not None:
                    sv = sig_t.get(s, np.nan)
                    if sv == sv and sv >= thr:
                        u *= 3.0
            elif flat9:
                u = R1.COST_UNIT * 2.0
            else:
                u = R1.COST_UNIT
            cost += d * u
        rows.append({"time": t, "fold": fold, "net_pnl_bps": g - cost,
                     "gross_pnl_bps": g, "cost_bps": cost})
        prev = dict(tw)
    return pd.DataFrame(rows)


def lofo(lever_df, base_df):
    """Lift = Sh(lever)-Sh(base) on matched cycles; sign-flip if removing any
    single OOS fold flips the lift sign."""
    m = lever_df.merge(base_df, on="time", suffixes=("_L", "_B"))
    folds = sorted(m["fold_L"].dropna().unique())
    lift_all = _sh(m["net_pnl_bps_L"]) - _sh(m["net_pnl_bps_B"])
    flips = []
    for f in folds:
        sub = m[m["fold_L"] != f]
        lf = _sh(sub["net_pnl_bps_L"]) - _sh(sub["net_pnl_bps_B"])
        if np.sign(lf) != np.sign(lift_all) and abs(lift_all) > 1e-9:
            flips.append((int(f), round(float(lf), 3)))
    # paired block bootstrap of per-cycle (lever-base) net diff
    diff = (m["net_pnl_bps_L"] - m["net_pnl_bps_B"]).to_numpy()
    try:
        _, lo, hi = block_bootstrap_ci(diff, statistic=lambda x: float(np.mean(x)),
                                       block_size=R1.BLOCK, n_boot=2000)
    except Exception:
        lo = hi = np.nan
    return {"lift_sharpe": round(float(lift_all), 3),
            "lofo_signflips": flips,
            "paired_diff_ci": [round(float(lo), 3), round(float(hi), 3)],
            "paired_diff_excludes_0": bool(lo > 0 or hi < 0)}


def run_preds(path, panel_syms, listings, fwd, sigma, advc, holds, caps, costs):
    apd = pd.read_parquet(path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= ts}

    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    u = PA.build_rolling_ic_universe(apd, tt[::HE], PA.TOP_N, elig)
    rec = PA.run_production_protocol_save_sleeves(apd, u)
    res = {}
    for (hb, nsl, hname) in holds:
        for c in caps:
            for cost in costs:
                df = aggregate_hold(rec, fwd, sigma, advc, hb, nsl,
                                    cap_frac=c, cost_mode=cost,
                                    tail_q=(0.90 if cost == "tail" else None))
                net = df["net_pnl_bps"].to_numpy()
                fs = df.groupby("fold")["net_pnl_bps"].sum()
                res[f"{hname}|cap{c}|{cost}"] = {
                    "sharpe": round(_sh(net), 3),
                    "folds_pos": int((fs > 0).sum()),
                    "maxDD": round(R1._max_dd(net), 0),
                    "totPnL": round(float(net.sum()), 0), "df": df}
    return res, rec


def main():
    t0 = time.time()
    apd0 = pd.read_parquet(APD)
    apd0["open_time"] = pd.to_datetime(apd0["open_time"], utc=True)
    panel_syms = sorted(apd0["symbol"].unique())
    listings = PA.get_listings()
    fwd, sigma, advc = R1.build_caches(apd0, panel_syms)

    # R1 baseline reference (24h, uncapped, flat45)
    base_all, _ = run_preds(APD, panel_syms, listings, fwd, sigma, advc,
                            [(288, 6, "24h")], [np.inf], ["flat45"])
    base_df = base_all["24h|capinf|flat45"]["df"]
    base_sh = base_all["24h|capinf|flat45"]["sharpe"]
    print(f"R1 baseline (24h uncapped flat45) Sharpe {base_sh:+.3f}", flush=True)

    summary = {"base_sharpe": base_sh, "levers": {}}

    # ---- R2a: +3 model features (24h, caps inf & 1/3, costs) -----------
    if R2A.exists():
        r2a, _ = run_preds(R2A, panel_syms, listings, fwd, sigma, advc,
                           [(288, 6, "24h")], [np.inf, 1/3],
                           ["flat45", "flat9", "realized"])
        for k, v in r2a.items():
            lf = lofo(v["df"], base_df)
            summary["levers"][f"R2a|{k}"] = {**{x: v[x] for x in
                ("sharpe", "folds_pos", "maxDD", "totPnL")}, **lf}
            print(f"  R2a {k}: Sh {v['sharpe']:+.3f} f+{v['folds_pos']}/9 "
                  f"lift {lf['lift_sharpe']:+.3f} flips={lf['lofo_signflips']} "
                  f"pairCI{lf['paired_diff_ci']}", flush=True)
    else:
        print("  R2A predictions not found — run R2a_retrain.py first", flush=True)

    # ---- R2b: longer hold on R1 baseline preds -------------------------
    r2b, _ = run_preds(APD, panel_syms, listings, fwd, sigma, advc,
                       [(288, 6, "24h"), (576, 12, "48h"), (864, 18, "72h")],
                       [np.inf, 1/3], ["flat45", "realized", "tail"])
    for k, v in r2b.items():
        if k.startswith("24h|capinf|flat45"):
            continue
        lf = lofo(v["df"], base_df)
        summary["levers"][f"R2b|{k}"] = {**{x: v[x] for x in
            ("sharpe", "folds_pos", "maxDD", "totPnL")}, **lf}
        print(f"  R2b {k}: Sh {v['sharpe']:+.3f} f+{v['folds_pos']}/9 "
              f"lift {lf['lift_sharpe']:+.3f} flips={lf['lofo_signflips']} "
              f"pairCI{lf['paired_diff_ci']}", flush=True)

    # ---- R2c: R2a preds at best R2b hold -------------------------------
    if R2A.exists():
        b48 = summary["levers"].get("R2b|48h|capinf|flat45", {})
        b72 = summary["levers"].get("R2b|72h|capinf|flat45", {})
        best_hold = (576, 12, "48h") if b48.get("sharpe", -9) >= b72.get("sharpe", -9) \
            else (864, 18, "72h")
        r2c, _ = run_preds(R2A, panel_syms, listings, fwd, sigma, advc,
                           [best_hold], [np.inf, 1/3], ["flat45", "realized"])
        for k, v in r2c.items():
            lf = lofo(v["df"], base_df)
            summary["levers"][f"R2c|{k}"] = {**{x: v[x] for x in
                ("sharpe", "folds_pos", "maxDD", "totPnL")}, **lf}
            print(f"  R2c {k}: Sh {v['sharpe']:+.3f} f+{v['folds_pos']}/9 "
                  f"lift {lf['lift_sharpe']:+.3f} flips={lf['lofo_signflips']}",
                  flush=True)

    summary["elapsed_s"] = round(time.time() - t0, 1)
    (OUT / "R2_results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nR2 done [{summary['elapsed_s']}s] -> {OUT}/R2_results.json", flush=True)


if __name__ == "__main__":
    main()
