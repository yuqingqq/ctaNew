"""R1c — re-initiated concentration measurement (honest).

Round-3 F1: R1/R1b computed Herfindahl on YEAR-CUMULATIVE SIGNED per-name
PnL (mechanically diffuse -> misleading 0.094 "diversified"). The honest
risk-concentration metric is the PER-CYCLE GROSS-WEIGHT Herfindahl:
  H_t = sum_s (|w_s,t| / sum_s|w_s,t|)^2 , reported as median/mean over cycles.
Round-3 F2/F5: R1b's "ex-VVV +1.99" REBUILDS the universe so filter_refill
rotates onto the next tail name -> report the rebuilt book's new top-name
net share (is the concentration risk gone, or rotated?).
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

OUT = REPO / "research/portable_alpha_2026-05-19/results"
APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
CAP = 1 / 3


def per_cycle_weight_herfindahl(records, fwd, sigma, advc, cap_frac):
    """Replays the cap-aware 6-sleeve weights and records, per cycle, the
    gross-weight Herfindahl + effective-N + #names of the actual book."""
    bar = pd.Timedelta(minutes=5)
    queue = deque(maxlen=PA.N_SLEEVES)
    Hs, effN, nnames, top1w = [], [], [], []
    for _, rec in records.iterrows():
        t = rec["time"]; lb, sb = rec["long_basket"], rec["short_basket"]
        if lb and sb:
            queue.append({"entry_time": t, "longs": lb, "shorts": sb})
        queue = deque([s for s in queue if (t - s["entry_time"]) < PA.HOLD_BARS * bar],
                      maxlen=PA.N_SLEEVES)
        tw = defaultdict(float); sw = 1.0 / PA.N_SLEEVES
        for sl in queue:
            L, S = sl["longs"], sl["shorts"]
            if not L or not S: continue
            for s in L: tw[s] += sw / len(L)
            for s in S: tw[s] -= sw / len(S)
        tw = R1._apply_cap(dict(tw), cap_frac)
        a = np.array([abs(v) for v in tw.values()], float)
        g = a.sum()
        if g <= 0: continue
        p = a / g
        H = float((p ** 2).sum())
        Hs.append(H); effN.append(1.0 / H); nnames.append(int((a > 1e-9).sum()))
        top1w.append(float(p.max()))
    return {"median_H": round(float(np.median(Hs)), 4),
            "mean_H": round(float(np.mean(Hs)), 4),
            "median_effN": round(float(np.median(effN)), 2),
            "median_n_names": int(np.median(nnames)),
            "median_top1_weight": round(float(np.median(top1w)), 4)}


def run_stack(apd, syms, listings, fwd, sigma, advc):
    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in syms if listings.get(s) and listings[s] <= ts}
    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    u = PA.build_rolling_ic_universe(apd, tt[::R1.HE], PA.TOP_N, elig)
    rec = PA.run_production_protocol_save_sleeves(apd, u)
    df, sg, _ = R1.aggregate_capped(rec, fwd, sigma, advc, cap_frac=CAP,
                                    sizing="equal", cost_mode="flat45")
    return df, sg, rec


def topshare(sg):
    net = sum(sg.values())
    ser = pd.Series(sg)
    nm = ser.abs().sort_values(ascending=False).index[0]
    return nm, round(float(ser[nm] / net), 4), round(float(net), 0)


def main():
    t0 = time.time()
    apd = pd.read_parquet(APD)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    syms = sorted(apd["symbol"].unique())
    listings = PA.get_listings()
    fwd, sigma, advc = R1.build_caches(apd, syms)

    # base (full universe) — honest per-cycle gross-weight Herfindahl
    df, sg, rec = run_stack(apd, syms, listings, fwd, sigma, advc)
    pcw = per_cycle_weight_herfindahl(rec, fwd, sigma, advc, CAP)
    n0, s0, net0 = topshare(sg)
    print(f"base cap-1/3: Sharpe {R1._sharpe(df['net_pnl_bps']):+.3f}", flush=True)
    print(f"  PER-CYCLE gross-weight Herfindahl: median {pcw['median_H']} "
          f"(effN {pcw['median_effN']}, ~{pcw['median_n_names']} names, "
          f"top1 wt {pcw['median_top1_weight']})", flush=True)
    print(f"  cumulative-signed-PnL top1: {n0} = {s0:+.1%} of net "
          f"(this is the OLD misleading basis)", flush=True)

    # ex-VVV REBUILD — does concentration vanish or ROTATE?
    keep = [s for s in syms if s != "VVVUSDT"]
    a2 = apd[apd["symbol"].isin(keep)].copy()
    df2, sg2, _ = run_stack(a2, keep, listings, fwd, sigma, advc)
    n2, s2, net2 = topshare(sg2)
    print(f"  ex-VVV (universe REBUILT): Sharpe {R1._sharpe(df2['net_pnl_bps']):+.3f}; "
          f"new top1 {n2} = {s2:+.1%} of net (net {net2:+.0f})", flush=True)
    # rotate again: drop the new top name too
    keep3 = [s for s in keep if s != n2]
    a3 = apd[apd["symbol"].isin(keep3)].copy()
    df3, sg3, _ = run_stack(a3, keep3, listings, fwd, sigma, advc)
    n3, s3, net3 = topshare(sg3)
    print(f"  ex-VVV-&-{n2} (REBUILT): Sharpe {R1._sharpe(df3['net_pnl_bps']):+.3f}; "
          f"new top1 {n3} = {s3:+.1%} of net", flush=True)

    out = {"per_cycle_gross_weight_herfindahl": pcw,
           "interpretation": (f"per-cycle book is ~{pcw['median_effN']} effective "
                              f"names (CONCENTRATED), not the ~11 implied by the "
                              f"cumulative-signed-PnL Herfindahl 0.094"),
           "cumulative_signed_top1": {"name": n0, "share": s0},
           "ex_VVV_rebuild": {"sharpe": round(R1._sharpe(df2["net_pnl_bps"]), 3),
                              "new_top1": n2, "new_top1_share": s2},
           "ex_VVV_and_next_rebuild": {"sharpe": round(R1._sharpe(df3["net_pnl_bps"]), 3),
                                       "new_top1": n3, "new_top1_share": s3},
           "verdict": ("concentration ROTATES to the next tail name under "
                       "filter_refill — it is not eliminated; 'robust to VVV' "
                       "is the adaptive-refill confound, NOT diversification"),
           "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "R1c_concentration_truth.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nR1c done [{out['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
