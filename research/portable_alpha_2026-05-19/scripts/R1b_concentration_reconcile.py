"""R1b — reconcile the R1 prediction-miss honestly.

R1 pre-registered "Herfindahl >= 0.40, concentrated"; observed gross-attribution
Herfindahl ~= 0.094, top-1 ~= 25%. PROGRESS.md says "~62% of net PnL from VVV".
These are DIFFERENT metrics. This script computes BOTH on the SAME V3.1 base
records to explain the discrepancy and test the real deployability question:

  (1) gross-risk concentration : Herfindahl of |per-cycle-attributed PnL|
      (already in R1; ~0.09 -> per-cycle exposure is broadly diversified)
  (2) net-cumulative concentration : signed cumulative per-name PnL as a share
      of TOTAL net PnL (the PROGRESS.md "62% from VVV" metric)
  (3) ex-VVV reconstruction : drop VVVUSDT from the panel, rebuild universe +
      records + 6-sleeve aggregate, measure Sharpe vs +2.23
  (4) drop-only-top1 : same, dropping whichever name has the largest net share

Per PLAN.md: a prediction miss rewrites the Diagnosis, not the gate.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research/portable_alpha_2026-05-19/scripts"))
import phase_ah_sleeve as PA
import R1_baseline_frontier as R1

OUT = REPO / "research/portable_alpha_2026-05-19/results"


def run_stack(apd, panel_syms, listings, fwd, sigma, advc):
    def elig(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC") - pd.Timedelta(days=PA.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= ts}
    tt = sorted(apd[apd["fold"].isin(PA.OOS_FOLDS)]["open_time"].unique())
    sampled = tt[::R1.HE]
    u = PA.build_rolling_ic_universe(apd, sampled, PA.TOP_N, elig)
    rec = PA.run_production_protocol_save_sleeves(apd, u)
    df, sg, sc = R1.aggregate_capped(rec, fwd, sigma, advc, cap_frac=np.inf,
                                     sizing="equal", cost_mode="flat45")
    return df, sg


def main():
    t0 = time.time()
    apd = pd.read_parquet(R1.APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = PA.get_listings()
    fwd, sigma, advc = R1.build_caches(apd, panel_syms)

    df, sg = run_stack(apd, panel_syms, listings, fwd, sigma, advc)
    net_total = float(df["net_pnl_bps"].sum())
    sh_base = R1._sharpe(df["net_pnl_bps"].to_numpy())

    # net-cumulative per-name share of TOTAL net PnL (PROGRESS.md metric)
    ser = pd.Series(sg).sort_values(ascending=False)
    share = (ser / net_total).round(4)
    top = ser.head(8)
    top1_name = ser.abs().sort_values(ascending=False).index[0]
    top1_share_of_net = float(ser[top1_name] / net_total)
    top3_share_of_net = float(ser.reindex(ser.abs().sort_values(ascending=False)
                                          .index[:3]).sum() / net_total)
    gross_H = R1._herfindahl(sg)

    print(f"  base V3.1 uncapped: Sharpe {sh_base:+.3f}, total_net {net_total:+.0f} bps",
          flush=True)
    print(f"  gross-risk Herfindahl (|per-cycle pnl|) = {gross_H:.4f} "
          f"(eff ~{1/gross_H:.0f} names)  <- R1 metric", flush=True)
    print(f"  net-cumulative: top1 {top1_name} = {top1_share_of_net:+.1%} of net PnL; "
          f"top3 = {top3_share_of_net:+.1%}  <- PROGRESS.md metric", flush=True)
    print("  top names by net cumulative PnL (bps / share-of-total-net):", flush=True)
    for s, v in top.items():
        print(f"    {s:<14} {v:+9.0f}  {v/net_total:+.1%}", flush=True)

    # ex-VVV and drop-only-top1 reconstructions
    recon = {}
    for label, drop in [("ex_VVVUSDT", "VVVUSDT"), (f"ex_top1_{top1_name}", top1_name)]:
        keep = [s for s in panel_syms if s != drop]
        a2 = apd[apd["symbol"].isin(keep)].copy()
        d2, _ = run_stack(a2, keep, listings, fwd, sigma, advc)
        s2 = R1._sharpe(d2["net_pnl_bps"].to_numpy())
        recon[label] = {"dropped": drop, "sharpe": round(s2, 3),
                        "delta_vs_base": round(s2 - sh_base, 3),
                        "total_net": round(float(d2["net_pnl_bps"].sum()), 0)}
        print(f"  {label}: Sharpe {s2:+.3f} (Δ {s2-sh_base:+.3f} vs base "
              f"{sh_base:+.3f})", flush=True)

    out = {"base_sharpe": round(sh_base, 3), "base_total_net": round(net_total, 0),
           "gross_herfindahl": round(gross_H, 4),
           "net_top1_name": top1_name,
           "net_top1_share_of_total": round(top1_share_of_net, 4),
           "net_top3_share_of_total": round(top3_share_of_net, 4),
           "top8_net": {s: round(float(v), 0) for s, v in top.items()},
           "reconstructions": recon, "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "R1b_concentration.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nR1b done [{out['elapsed_s']}s]", flush=True)


if __name__ == "__main__":
    main()
