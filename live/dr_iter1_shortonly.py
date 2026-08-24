"""Detail-review loop — iteration 1 (D1): is this actually a SHORT book?

Measured in dp_phase3_diligence.py on the held-out window: A long -0.71 / A short +1.30, B long -0.39 /
B short +0.95. BOTH long legs lose money. I flagged this, offered it as an option, and never tested it.

The honest counter-hypothesis, stated first: the long leg may be earning its keep as a HEDGE rather than as
alpha. Leg Sharpes computed on the BTC-residual return already net out beta, so they cannot settle that —
removing the longs could raise beta exposure and volatility more than it raises return.

Constructions, all on the same preds / universe / band, held-out and select windows:
  ls          symmetric long/short quintiles (incumbent)
  short_only  bottom quintile only, market exposure neutralised with a long BASKET position sized to the
              short book's realised beta (estimated on the OTHER window — never in-sample)
  short_tilt  asymmetric 70/30 short/long
  long_only   top quintile only, basket-hedged — negative control

Gates: G1 short_only beats ls on held-out net, paired 7d-block CI on the DELTA excluding 0; G2 also holds in
SELECT; G3 drawdown/skew do not materially worsen. Falsifier: G1 fails -> the long leg is doing hedging work
the leg decomposition hides, and the symmetric book stands.
Run: python3 -u -m live.dr_iter1_shortonly
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (ERAS, block_ci, build_panel, get_preds, paired_block_ci,
                                    pit_adv, sharpe, tag_ci)
from live.build_alpha_beta_decomp import FULL
from live.cl_iter4_capacity import build, cost_tiers
from live.mc_oi_universe import topn, N as NTOP

SEL = (pd.Timestamp("2023-06-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
HO = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
COST_BPS = 2.0                      # measured small-clip cost, per side
PYR = 6 * 365.0


def load():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    lab = PAN.rename(columns={"alpha_vs_btc_realized": "alpha_A"})[["symbol", "open_time", "alpha_A"]]
    P = pd.concat([get_preds(e) for e in ERAS], ignore_index=True).drop_duplicates(
        ["symbol", "open_time"]).sort_values(["symbol", "open_time"])
    P = P.drop(columns=[c for c in ("alpha_A", "return_pct") if c in P.columns]).merge(
        lab, on=["symbol", "open_time"], how="left").merge(RP, on=["symbol", "open_time"], how="left")
    A = pit_adv(); P["date"] = P["open_time"].dt.floor("1D")
    return P.merge(A, on=["symbol", "date"], how="left").dropna(subset=["tadv", "alpha_A", "return_pct"])


def legs(d):
    """Weights from the band construction, split into long and short books, plus the equal-weight basket."""
    W, Aa = build(d, "band")
    R = d.pivot_table(index="symbol", columns="open_time", values="return_pct").reindex_like(Aa).fillna(0.0)
    mask = d.pivot_table(index="symbol", columns="open_time", values="return_pct").notna().astype(float)
    basket = (R * mask).sum(axis=0) / mask.sum(axis=0).replace(0, np.nan)      # equal-weight market proxy
    return W, R, Aa, basket


def series(W, R, Aa, basket, mode, beta):
    """Return (raw book return incl. hedge, turnover) for a construction."""
    pos = W.clip(lower=0); neg = W.clip(upper=0)
    if mode == "ls":
        Wu = W
    elif mode == "short_only":
        Wu = neg
    elif mode == "short_tilt":
        Wu = 0.7 * neg / max(abs(neg.sum().mean()), 1e-9) + 0.3 * pos / max(pos.sum().mean(), 1e-9)
        Wu = neg * 0.7 + pos * 0.3
    elif mode == "long_only":
        Wu = pos
    g = (Wu * R).sum(axis=0)
    net_exposure = Wu.sum(axis=0)                       # dollar imbalance to hedge with the basket
    hedged = g - net_exposure * basket * beta
    dW = Wu.diff(axis=1).abs()
    turn = 0.25 * dW.sum(axis=0) + 0.25 * net_exposure.diff().abs()   # hedge leg trades too
    return hedged.iloc[1:], turn.iloc[1:]


def run(d, beta_by_mode):
    W, R, Aa, basket = legs(d)
    out = {}
    for mode in ("ls", "short_only", "short_tilt", "long_only"):
        g, turn = series(W, R, Aa, basket, mode, beta_by_mode.get(mode, 1.0))
        net = g - turn * COST_BPS * 2 / 1e4
        j = pd.concat([net.rename("net"), g.rename("gross"), turn.rename("t")], axis=1).dropna()
        j.index = pd.to_datetime(j.index, utc=True)
        out[mode] = j.groupby(j.index.floor("1D")).sum()
    return out


def est_beta(d):
    """Hedge ratio per construction, estimated on THIS window (applied to the OTHER one)."""
    W, R, Aa, basket = legs(d)
    b = {}
    for mode in ("ls", "short_only", "short_tilt", "long_only"):
        pos = W.clip(lower=0); neg = W.clip(upper=0)
        Wu = {"ls": W, "short_only": neg, "short_tilt": neg * 0.7 + pos * 0.3, "long_only": pos}[mode]
        g = (Wu * R).sum(axis=0)
        ex = Wu.sum(axis=0)
        j = pd.concat([g.rename("g"), (ex * basket).rename("x")], axis=1).dropna()
        j = j[j["x"].abs() > 1e-12]
        b[mode] = float(np.polyfit(j["x"], j["g"], 1)[0]) if len(j) > 50 else 1.0
    return b


def main():
    P = load()
    win = {}
    for nm, (t0, t1) in (("SELECT", SEL), ("HOLDOUT", HO)):
        win[nm] = topn(P[(P.open_time >= t0) & (P.open_time < t1)], "tadv", NTOP)
    # hedge ratios estimated OUT of the evaluation window
    b_sel = est_beta(win["HOLDOUT"])
    b_ho = est_beta(win["SELECT"])
    print("hedge ratios (estimated on the OTHER window):", flush=True)
    print("  " + ", ".join(f"{k} sel={b_sel[k]:+.2f}/ho={b_ho[k]:+.2f}" for k in b_sel), flush=True)

    res = {"SELECT": run(win["SELECT"], b_sel), "HOLDOUT": run(win["HOLDOUT"], b_ho)}
    for nm in ("SELECT", "HOLDOUT"):
        print(f"\n================ {nm} ================", flush=True)
        print(f"  {'construction':<14}{'turn':>7}{'gross':>8}{'net':>8}{'net CI':>22}"
              f"{'maxDD':>9}{'skew':>7}", flush=True)
        for mode in ("ls", "short_only", "short_tilt", "long_only"):
            s = res[nm][mode]["net"]
            if len(s) < 50:
                continue
            lo, hi = block_ci(s.to_numpy(), block=7)
            eq = np.cumsum(s.to_numpy())
            dd = float((eq - np.maximum.accumulate(eq)).min())
            print(f"  {mode:<14}{res[nm][mode]['t'].mean():>7.3f}"
                  f"{s.mean()/s.std()*np.sqrt(365) + (res[nm][mode]['t'].mean()*COST_BPS*2/1e4)*0:>8.2f}"
                  f"{s.mean()/s.std()*np.sqrt(365):>8.2f}"
                  f"{f'[{lo:+.2f},{hi:+.2f}] {tag_ci(lo,hi)}':>22}{dd*100:>9.1f}%{float(pd.Series(s).skew()):>7.2f}",
                  flush=True)

    print("\n=== GATES — paired Δ vs the symmetric L/S book ===", flush=True)
    for nm in ("SELECT", "HOLDOUT"):
        base = res[nm]["ls"]["net"]
        cells = []
        for mode in ("short_only", "short_tilt", "long_only"):
            v = res[nm][mode]["net"]
            idx = base.index.intersection(v.index)
            dd, lo, hi = paired_block_ci(base.loc[idx].to_numpy(), v.loc[idx].to_numpy(), block=7)
            cells.append(f"{mode} {dd:+.2f}[{lo:+.2f},{hi:+.2f}]{tag_ci(lo,hi)}")
        print(f"  {nm:<9}" + "   ".join(cells), flush=True)
    hb = res["HOLDOUT"]
    idx = hb["ls"]["net"].index.intersection(hb["short_only"]["net"].index)
    d1, l1, _ = paired_block_ci(hb["ls"]["net"].loc[idx].to_numpy(),
                                hb["short_only"]["net"].loc[idx].to_numpy(), block=7)
    print(f"\n  G1 (short_only beats ls held-out, CI>0): {'PASS' if l1 > 0 else 'FAIL'}", flush=True)
    print("\nDRITER1DONE", flush=True)


if __name__ == "__main__":
    main()
