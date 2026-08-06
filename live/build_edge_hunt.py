"""Extend the mechanism-driven hunt: (1) momentum COMPOSITE (multi-horizon skip-recent) — does combining 14d+30d
strengthen / stabilize the thin single-14d edge? (2) slow CARRY (funding_rate_z_7d) — a structurally different
(non-behavioral) root, only ever tested as a fast feature; screen it as a slow cross-sectional sleeve.
For each: orthogonalized IC vs the edge {return_1d,ret_3d,rvol,atr,idio_vol}, both eras, + QUARTERLY stability
(persistent same-sign = real; flips = artifact). Fast (no pipeline). Run: python3 -u -m live.build_edge_hunt
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel
from live.build_alpha_beta_decomp import FULL
from live.orthogonal_harness import screen, _fmt

CTRL = ["return_1d", "ret_3d", "rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]
TGT = "alpha_vs_btc_realized"
CANDS = ["mom_14_3", "mom_comp", "carry"]


def quarterly(d, fcol):
    dd = d.dropna(subset=[fcol] + CTRL + [TGT])
    rows = []
    for t, g in dd.groupby("open_time"):
        if len(g) < 15:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in CTRL])
        b, *_ = np.linalg.lstsq(X, g[fcol].to_numpy(), rcond=None)
        rows.append((t, spearmanr(g[fcol].to_numpy() - X @ b, g[TGT].to_numpy()).correlation))
    P = pd.DataFrame(rows, columns=["t", "ic"]).dropna().set_index("t")
    q = P.groupby(P.index.to_period("Q"))["ic"].mean()
    return q


def main():
    PAN = build_panel()
    miss = [c for c in set(["return_pct"] + CTRL) if c not in PAN.columns]
    ex = pd.read_parquet(FULL, columns=["symbol", "open_time"] + miss)
    ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
    PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left").sort_values(["symbol", "open_time"])
    r = PAN.groupby("symbol")["return_pct"]
    PAN["mom_14_3"] = r.transform(lambda s: s.shift(18).rolling(66).sum())
    PAN["mom_30_7"] = r.transform(lambda s: s.shift(42).rolling(138).sum())
    for c in ["mom_14_3", "mom_30_7"]:
        PAN[c + "_z"] = PAN.groupby("open_time")[c].transform(lambda s: (s - s.mean()) / s.std())
    PAN["mom_comp"] = PAN[["mom_14_3_z", "mom_30_7_z"]].mean(axis=1)
    PAN["carry"] = PAN["funding_rate_z_7d"]

    res = screen(PAN, CANDS, controls=CTRL)
    print("orthogonalized IC vs edge (both eras); + = momentum/long-high, sign shows direction:", flush=True)
    print(f"  {'cand':<12}{'RAW OOS':<26}{'ORTH OOS':<26}{'ORTH RECENT':<26}", flush=True)
    for c in CANDS:
        print(f"  {c:<12}{_fmt(res[c]['raw']['OOS']):<26}{_fmt(res[c]['orth']['OOS']):<26}"
              f"{_fmt(res[c]['orth']['RECENT']):<26}", flush=True)
    print("\nQUARTERLY orthogonal IC (persistence acid test):", flush=True)
    for c in ["mom_comp", "carry"]:
        q = quarterly(PAN, c)
        pos = int((q > 0).sum())
        print(f"  {c}: {pos}/{len(q)} quarters positive | mean {q.mean():+.4f} | worst {q.min():+.3f} "
              f"best {q.max():+.3f}", flush=True)
        print("    " + "  ".join(f"{str(k)[2:]}:{v:+.3f}" for k, v in q.items()), flush=True)
    print("\nEDGEHUNTDONE", flush=True)


if __name__ == "__main__":
    main()
