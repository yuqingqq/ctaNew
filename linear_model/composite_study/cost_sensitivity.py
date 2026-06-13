"""Composite Study — cost sensitivity of THE linear alpha model
(market-neutral leak-free F_core portfolio, the +0.62-at-VIP-0 book).
Net Sharpe / cumPnL / maxDD + block-bootstrap CI at several cost-per-unit
levels, with the 3.5 bps interpretations labelled explicitly.

Cost convention: c = bps per unit |Δw|. A position FLIP (−1→+1) is |Δ|=2.
  VIP-0 baseline           c = 2.25  (flip = 4.5 bps)   [arc default]
  "3.5 bps round-trip"     c = 1.75  (flip = 3.5 bps)
  "3.5 bps per side/unit"  c = 3.50  (flip = 7.0 bps)
Honest: this is the leak-free STATIONARY ceiling number; CI shown; the
established forward expectation is below it (non-stationarity haircut,
CI-crosses-0, fold-fragility — independent of cost). Production LGBM
unaffected.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


s94 = _imp("s94", "linear_model/scripts/94_info_ceiling_d1.py")
s94b = _imp("s94b", "linear_model/scripts/94b_info_ceiling_d1_grouped.py")
from ml.research.alpha_v4_xs import block_bootstrap_ci
ANN = np.sqrt(365.0 * 6.0)
OUTD = REPO / "linear_model/composite_study/results"


def main():
    print("=" * 100, flush=True)
    print("  COMPOSITE STUDY — cost sensitivity of the linear alpha model "
          "(market-neutral leak-free book)", flush=True)
    print("=" * 100, flush=True)
    dec, syms, btc, pan = s94.build(universe_oi=False)
    LEAK = s94.LEAK
    FEATS = [c for c in dec.columns if c not in LEAK and
             pd.api.types.is_numeric_dtype(dec[c])]
    if "s_t" not in FEATS:
        FEATS.append("s_t")
    d = dec.dropna(subset=FEATS + ["tz", "alpha_beta"]).reset_index(drop=True)
    rid, _ = s94b.grouped_oof(d, FEATS)
    d["pred"] = rid
    d = d[~d["pred"].isna()].sort_values(["symbol", "open_time"]).reset_index(drop=True)
    d["pos"] = np.sign(d["pred"])
    d["dp"] = d.groupby("symbol")["pos"].diff().abs().fillna(d["pos"].abs())
    pt = d.groupby("open_time").agg(
        gross=("alpha_beta", lambda s: (d.loc[s.index, "pos"] *
               d.loc[s.index, "alpha_beta"]).mean()),
        turn=("dp", "mean")).reset_index()
    g = pt["gross"].to_numpy() * 1e4                    # bps/cyc, cost-free
    tn = pt["turn"].to_numpy()                          # |Δw|/cyc
    gsh = g.mean()/g.std(ddof=1)*ANN
    print(f"\n  cycles={len(pt)}  GROSS Sharpe={gsh:+.2f}  "
          f"gross={g.mean():+.2f} bps/cyc  avg turnover |Δ|/cyc={tn.mean():.3f}",
          flush=True)
    print(f"  (+1.5 = the pre-registered economic bar; CI from block "
          f"bootstrap; this is the leak-free STATIONARY ceiling)\n", flush=True)
    grid = [(0.0, "zero cost (upper bound)"),
            (1.00, "≈HL-maker ~2bps RT"),
            (1.75, "**3.5 bps ROUND-TRIP**"),
            (2.25, "VIP-0 baseline (4.5 RT) [arc default]"),
            (3.50, "3.5 bps per side/unit (7.0 RT)")]
    print(f"  {'c/unit':>7s} {'meaning':38s} {'netSh':>6s} "
          f"{'CI(Sharpe)':>16s} {'net_bps':>8s} {'cumPnL':>8s} {'maxDD':>7s} "
          f"{'>+1.5?':>6s}", flush=True)
    rows = []
    for c, lab in grid:
        net = g - tn * c
        sh = net.mean()/net.std(ddof=1)*ANN
        lo, hi = block_bootstrap_ci(
            net, statistic=lambda z: z.mean()/z.std(ddof=1)*ANN
            if z.std(ddof=1) > 1e-12 else 0.0, block_size=7, n_boot=1000)[1:]
        eq = np.cumsum(net)
        dd = (eq - np.maximum.accumulate(eq)).min()
        pas = "PASS" if lo > 1.5 else ("~" if hi > 1.5 else "no")
        print(f"  {c:7.2f} {lab:38s} {sh:+6.2f} [{lo:+6.2f},{hi:+6.2f}] "
              f"{net.mean():+8.2f} {eq[-1]:+8.0f} {dd:+7.0f} {pas:>6s}",
              flush=True)
        rows.append(dict(c_per_unit=c, meaning=lab, net_sh=round(sh, 2),
                         ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         net_bps=round(net.mean(), 2), cumpnl=round(eq[-1]),
                         maxdd=round(dd), gate_pass=pas))
    pd.DataFrame(rows).to_csv(OUTD/"cost_sensitivity.csv", index=False)
    print("""
  HONEST READING:
   • Cost sensitivity is REAL and monotone — lower execution cost raises
     the number (gross is fixed; only the turnover·c drag changes).
   • At **3.5 bps round-trip (c=1.75)** read the netSh row + its CI.
   • BUT this is the leak-free STATIONARY-ceiling Sharpe. The
     pre-registered +1.5 bar is on THIS ceiling precisely because the real
     walk-forward / non-stationarity haircut runs ~2–3× and the CI here
     crosses 0. A lower cost lifts the cost term; it does NOT fix the
     non-stationarity, CI-crosses-0, or fold-fragility (independent
     findings: 92b/93/99). So even a favorable cost only matters if the
     row CLEARS +1.5 with CI excluding 0 — otherwise the constraint is
     unchanged. Production LGBM unaffected.
""", flush=True)
    print(f"Saved {OUTD}/cost_sensitivity.csv", flush=True)


if __name__ == "__main__":
    main()
