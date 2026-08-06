"""Clean, independent verification of the crypto alpha-residual edge — reproduce the numbers I'd been
quoting from docs, using the EXACT deployed pipeline (v0_feature_ablation.build_panel / gen).

Checks:
  (1) BASELINE rank-IC both eras (should reproduce ~+0.030 recent / ~+0.021 OOS if the doc is right).
  (2) LONG vs SHORT leg split (verify the "short-concentrated" claim at book level): per bar, quintile
      by prediction; long-leg P&L = mean fwd-alpha of top quintile; short-leg P&L = −mean fwd-alpha of
      bottom quintile. If short >> long, short-concentrated.
Run:  python3 -u -m live.verify_edge
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, gen, perbar_ic, V0, RECENT_CUTS, OOS_CUTS


def day_ci(ic: pd.Series, n_boot=2000, seed=1):
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True))
    s["d"] = s.index.floor("1D")
    grp = [x["v"].values for _, x in s.groupby("d")]
    rng = np.random.default_rng(seed); k = len(grp)
    boot = [np.concatenate([grp[i] for i in rng.integers(0, k, k)]).mean() for _ in range(n_boot)]
    return float(ic.mean()), *np.percentile(boot, [2.5, 97.5])


def leg_split(P, k=0.2):
    rows = []
    for t, g in P.groupby("open_time"):
        if len(g) < 5:
            continue
        g = g.sort_values("pred")
        nk = max(1, int(len(g) * k))
        rows.append((t, g["alpha_A"].iloc[-nk:].mean(), g["alpha_A"].iloc[:nk].mean()))
    d = pd.DataFrame(rows, columns=["t", "long_a", "short_a"])
    long_pnl = d["long_a"].mean() * 1e4          # long top-quintile: P&L = +fwd alpha
    short_pnl = (-d["short_a"]).mean() * 1e4     # short bottom-quintile: P&L = −fwd alpha
    return long_pnl, short_pnl, d


def main():
    PAN = build_panel()
    print(f"panel {len(PAN):,} rows | {PAN.symbol.nunique()} syms | V0={len(V0)} feats\n", flush=True)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        P = gen(PAN, V0, cuts)
        ic = perbar_ic(P)
        m, lo, hi = day_ci(ic)
        lp, sp, _ = leg_split(P)
        print(f"===== {era} =====", flush=True)
        print(f"  BASELINE rank-IC {m:+.4f} [{lo:+.4f},{hi:+.4f}]  (doc claims "
              f"{'+0.030' if era=='RECENT' else '+0.021'})", flush=True)
        print(f"  LEG SPLIT (quintile L/S, per-bar fwd-alpha, bps): "
              f"long {lp:+.2f} | short {sp:+.2f} | total {lp+sp:+.2f} | "
              f"short share {sp/(abs(lp)+abs(sp)+1e-9)*100:.0f}%", flush=True)
        print("", flush=True)
    print("VERIFYDONE", flush=True)


if __name__ == "__main__":
    main()
