"""Apply the cost-aware 'no-trade band' idea (implementable-efficient-frontier / band-turnover-regularization
literature) to our strategy — the lever that targets our binding constraint (cost).

Baseline: full-rebalance quintile L/S every 4h (turnover ~0.40, net-negative at retail).
No-trade band (Schmitt trigger): ENTER long at rank>=1-e / short at rank<=e; HOLD until the name exits the
wider band (rank<1-x long / rank>x short). Holds persistent names -> cuts turnover -> raises break-even.

Sweep (e, x); measure turnover, gross, NET at a cost grid, both eras. Question: does the band flip the net
edge positive at achievable (fee-tier/maker) cost, and by how much does it raise break-even?
Run: python3 -u -m live.build_notrade_band
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

COST_GRID = [24.0, 12.0, 6.0, 3.0]
PYR = 6 * 365.0
BANDS = [(0.2, 0.2), (0.2, 0.4), (0.2, 0.5), (0.1, 0.4)]   # (enter, exit); e==x = full rebalance baseline


def band_pos(rank):
    def f(r, e, x):
        n = len(r); pos = np.empty(n, np.int8); p = 0
        for i in range(n):
            ri = r[i]
            if ri >= 1 - e: p = 1
            elif ri <= e: p = -1
            elif p == 1 and ri >= 1 - x: p = 1
            elif p == -1 and ri <= x: p = -1
            else: p = 0
            pos[i] = p
        return pos
    return f


def turnover(bt, sym, pos):
    order = np.lexsort((bt, sym)); so, po = sym[order], pos[order]
    same = so[1:] == so[:-1]
    out = []
    for side in (1, -1):
        cur = (po == side); prev = cur[:-1] & same
        out.append(1.0 - (prev & cur[1:]).sum() / max(prev.sum(), 1))
    return float(np.mean(out))


def portfolio(d, pos):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    ok = (nl >= 3) & (ns >= 3)
    return sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    f = band_pos(None)
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        d["rank"] = d.groupby("open_time")["pred"].rank(pct=True)
        print(f"===== {era} =====", flush=True)
        for e, x in BANDS:
            pos = np.concatenate([f(g["rank"].to_numpy(), e, x) for _, g in d.groupby("symbol", sort=False)])
            gross = portfolio(d, pos); turn = turnover(d["open_time"].to_numpy("datetime64[ns]"),
                                                       d["symbol"].to_numpy(), pos)
            gm, gsd = gross.mean(), gross.std()
            tag = "(baseline full-rebal)" if e == x else f"(band {e}->{x})"
            be = gm * 1e4 / max(turn, 1e-9)
            nets = "  ".join(f"{c:g}:{(gm - turn*c/1e4)/gsd*np.sqrt(PYR):+.2f}" for c in COST_GRID)
            print(f"  enter {e} exit {x} {tag:<22} gross {gm*1e4:+.2f}bps | turn {turn:.2f} | "
                  f"break-even {be:.1f}bps | netSharpe@cost {nets}", flush=True)
        print("", flush=True)
    print("BANDDONE", flush=True)


if __name__ == "__main__":
    main()
