"""Cost/turnover loop — iteration 3.

H3: the low-vol factor's ~0 book Sharpe (iter2) is a leg-VARIANCE artifact of equal-weighting a book that is
sorted ON volatility — the short leg holds the raciest names, so book std ~= short-leg std and Sharpe is
deflated by roughly the leg-vol ratio. Risk-weighting should recover it.

Constructions (all on per-name BTC-residual returns, dollar-neutral, PIT weights):
  eq        equal-weight quintile                                  (iter2 baseline)
  ivol      inverse-vol weights within each leg (w ∝ 1/rvol_7d)
  cont      continuous rank weights over the FULL cross-section, inverse-vol scaled (harvests the middle)
  volscale  per-name equal RISK contribution (w ∝ 1/rvol, applied to the quintile book, then risk-normalized)

Reports the realized short-leg/long-leg return-vol ratio (G1: the mechanism, measured not assumed) and carries
the INCUMBENT through every construction as the control (A2).

Gates: G1 leg-vol ratio >= 1.5 both eras; G2 paired Δ(reweighted − eq) gross CI>0 both eras for rvol_slow;
G3 recovered gross CI>0 both eras; G4 net@ps CI>0 both eras. See live/COST_TURNOVER_LOOP.md.
Run: python3 -u -m live.cl_iter3_weighting
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from live.cost_loop_harness import (
    ERAS, CACHE, add_slow_signals, block_ci, build_panel, cost_map, get_preds,
    maxdd, paired_block_ci, restrict_topn, sharpe, tag_ci,
)

SIGS = ["fast", "rvol_slow", "stat"]
CONS = ["eq", "ivol", "cont", "volscale"]
TIERS = [(40, "top40"), (999, "all")]


def prep(era, rvol):
    p = add_slow_signals(get_preds(era)).merge(rvol, on=["symbol", "open_time"], how="left")
    p["vrank"] = p.groupby("open_time")["rvol_7d"].rank(pct=True)
    p["rvol_slow"] = -p.sort_values(["symbol", "open_time"]).groupby("symbol")["vrank"].transform(
        lambda s: s.shift(1).expanding(min_periods=30).mean())
    p["fast"] = p["pred"]
    # PIT inverse-vol scaler, winsorized so a near-zero rvol cannot take the whole book
    v = p["rvol_7d"].replace(0, np.nan)
    lo = p.groupby("open_time")["rvol_7d"].transform(lambda s: s.quantile(0.05))
    p["iv"] = 1.0 / v.clip(lower=lo).fillna(v.median())
    return p


def _norm_sides(P):
    pos = P.clip(lower=0); neg = P.clip(upper=0)
    return pos.div(pos.sum().replace(0, np.nan), axis=1).fillna(0.0) \
        + neg.div(neg.sum().abs().replace(0, np.nan), axis=1).fillna(0.0)


def weights(d: pd.DataFrame, sig: str, con: str, q: float = 0.2):
    """Returns (W, A, legmask) as names x bars. A = per-name residual return."""
    x = d.dropna(subset=[sig, "alpha_A", "iv"]).copy()
    x["rk"] = x.groupby("open_time")[sig].rank(pct=True)
    if con == "cont":
        # continuous, demeaned rank across the whole cross-section, scaled by inverse vol
        x["raw"] = (x["rk"] - x.groupby("open_time")["rk"].transform("mean")) * x["iv"]
    else:
        side = np.where(x["rk"] >= 1 - q, 1.0, np.where(x["rk"] <= q, -1.0, 0.0))
        if con == "eq":
            x["raw"] = side
        elif con in ("ivol", "volscale"):
            x["raw"] = side * x["iv"]
    A = x.pivot_table(index="symbol", columns="open_time", values="alpha_A").fillna(0.0)
    P = x.pivot_table(index="symbol", columns="open_time", values="raw", fill_value=0.0).reindex_like(A)
    W = _norm_sides(P)
    if con == "volscale":                      # equalize the two legs' EX-ANTE risk, not their dollars
        IV = x.pivot_table(index="symbol", columns="open_time", values="rvol_7d").reindex_like(A)
        rl = (W.clip(lower=0) * IV).sum(axis=0)
        rs = (W.clip(upper=0).abs() * IV).sum(axis=0)
        scale = (rl / rs.replace(0, np.nan)).clip(0.2, 5.0).fillna(1.0)
        W = W.clip(lower=0) + W.clip(upper=0).mul(scale, axis=1)
    return W, A, x


def book_stats(W, A, cost):
    gross = (W * A).sum(axis=0)
    dW = W.diff(axis=1).abs()
    c, med = cost
    cvec = pd.Series([c.get(s, med) for s in W.index], index=W.index)
    j = pd.DataFrame({"g": gross, "t": 0.25 * dW.sum(axis=0),
                      "c_ps": 0.25 * dW.mul(cvec, axis=0).sum(axis=0) / 1e4,
                      "gl": (W.clip(lower=0) * A).sum(axis=0),
                      "gs": (W.clip(upper=0) * A).sum(axis=0)})
    return j.iloc[1:].dropna(subset=["g", "t"])


def main():
    pc = cost_map()
    PAN = build_panel()
    rvol = PAN[["symbol", "open_time", "rvol_7d"]]
    D = {era: prep(era, rvol) for era in ERAS}
    books = {}
    for era in ERAS:
        for n, lab in TIERS:
            d = restrict_topn(D[era], n)
            for sig in SIGS:
                for con in CONS:
                    W, A, _ = weights(d, sig, con)
                    books[(era, lab, sig, con)] = book_stats(W, A, pc)
        print(f"[{era}] books built", flush=True)

    rows = []
    print("\n============ LEVELS (per-name residual books, 7d-block CI) ============", flush=True)
    for era in ERAS:
        for _, lab in TIERS:
            print(f"\n----- {era} / {lab} -----", flush=True)
            print(f"  {'signal':<11}{'con':<10}{'turn':<7}{'legvol':<8}{'grossSh [CI]':<27}"
                  f"{'net@ps [CI]':<27}{'maxDD':<8}", flush=True)
            for sig in SIGS:
                for con in CONS:
                    j = books[(era, lab, sig, con)]
                    lv = float(j["gs"].std() / j["gl"].std()) if j["gl"].std() > 0 else np.nan
                    glo, ghi = block_ci(j["g"].to_numpy())
                    nps = (j["g"] - j["c_ps"]).to_numpy()
                    nlo, nhi = block_ci(nps)
                    r = dict(era=era, tier=lab, sig=sig, con=con, turn=float(j["t"].mean()), legvol=lv,
                             gross=sharpe(j["g"]), g_lo=glo, g_hi=ghi, net_ps=sharpe(nps),
                             n_lo=nlo, n_hi=nhi, maxdd=maxdd(j["g"]) * 1e4)
                    rows.append(r)
                    gc = f"{r['gross']:+.2f} [{glo:+.2f},{ghi:+.2f}] {tag_ci(glo, ghi)}"
                    nc = f"{r['net_ps']:+.2f} [{nlo:+.2f},{nhi:+.2f}] {tag_ci(nlo, nhi)}"
                    print(f"  {sig:<11}{con:<10}{r['turn']:<7.3f}{lv:<8.2f}{gc:<27}{nc:<27}"
                          f"{r['maxdd']:<8.0f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(CACHE / "iter3_levels.csv", index=False)

    print("\n============ PAIRED Δ(construction − eq), same signal ============", flush=True)
    prows = []
    for era in ERAS:
        for _, lab in TIERS:
            print(f"\n----- {era} / {lab} -----", flush=True)
            for sig in SIGS:
                je = books[(era, lab, sig, "eq")]
                cells = []
                for con in CONS[1:]:
                    jv = books[(era, lab, sig, con)]
                    idx = je.index.intersection(jv.index)
                    dd, lo, hi = paired_block_ci(je.loc[idx, "g"].to_numpy(), jv.loc[idx, "g"].to_numpy())
                    cells.append(f"{con} {dd:+.2f}[{lo:+.2f},{hi:+.2f}]{tag_ci(lo, hi)}")
                    prows.append(dict(era=era, tier=lab, sig=sig, con=con, d=dd, lo=lo, hi=hi))
                print(f"  {sig:<11}" + "   ".join(f"{c:<30}" for c in cells), flush=True)
    P = pd.DataFrame(prows); P.to_csv(CACHE / "iter3_paired.csv", index=False)

    print("\n============ GATE READ ============", flush=True)
    lv = {e: T[(T.era == e) & (T.tier == "top40") & (T.sig == "rvol_slow") & (T.con == "eq")].iloc[0].legvol
          for e in ERAS}
    g1 = all(v >= 1.5 for v in lv.values())
    print(f"  G1 leg-vol ratio >=1.5 both eras (rvol eq, top40): "
          f"{ {k: round(v, 2) for k, v in lv.items()} } -> {'PASS' if g1 else 'FAIL'}", flush=True)
    best = None
    for con in CONS[1:]:
        ok = all(P[(P.era == e) & (P.tier == "top40") & (P.sig == "rvol_slow") & (P.con == con)].iloc[0].lo > 0
                 for e in ERAS)
        print(f"  G2 Δ({con} − eq) gross CI>0 both eras (rvol, top40): {'PASS' if ok else 'FAIL'}", flush=True)
        if ok and best is None:
            best = con
    if best:
        g3 = all(T[(T.era == e) & (T.tier == "top40") & (T.sig == "rvol_slow") & (T.con == best)].iloc[0].g_lo > 0
                 for e in ERAS)
        g4 = all(T[(T.era == e) & (T.tier == "top40") & (T.sig == "rvol_slow") & (T.con == best)].iloc[0].n_lo > 0
                 for e in ERAS)
        print(f"  G3 recovered gross CI>0 both eras ({best}): {'PASS' if g3 else 'FAIL'}", flush=True)
        print(f"  G4 recovered net@ps CI>0 both eras ({best}): {'PASS' if g4 else 'FAIL'}", flush=True)
    else:
        print("  G3/G4 not evaluated — no construction passed G2", flush=True)
    print("\n  CONTROL (A2) — same Δ for the INCUMBENT (a generic lift would contradict the closed", flush=True)
    print("  vol-weighting/convex-sizing nulls and should be trusted less than those results):", flush=True)
    for era in ERAS:
        sub = P[(P.era == era) & (P.tier == "top40") & (P.sig == "fast")]
        print(f"    {era:<8}" + "  ".join(f"{r.con} {r.d:+.2f}[{r.lo:+.2f},{r.hi:+.2f}]{tag_ci(r.lo, r.hi)}"
                                          for r in sub.itertuples()), flush=True)
    print("\nITER3DONE", flush=True)


if __name__ == "__main__":
    main()
