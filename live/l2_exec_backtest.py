"""Depth-aware EXECUTION backtest for the convexity-v4 book (WS1 / E2-E4).

Question: how much of the strategy's paper selection-spread survives realistic
market impact, and how much does depth-aware sizing recover?

Book (per 4h bar `open_time`): rank the cross-section by `pred`; LONG top-1,
SHORT bottom-2. Paper per-bar selection-spread (return units, dollar-neutral,
gross-normalised) = 0.5*long_alpha_A - 0.5*mean(bottom2 alpha_A).

Impact model: reuse `bookdepth_impact.impact_pct` (piecewise-linear one-side
cumulative-depth curve through 0.2%/1%/5% points). One-side depth reconstructed
from bookDepth: d1 = exp(l2_liq1)/2, d02 = l2_touch*exp(l2_liq1)/2,
d5 = l2_slope*exp(l2_liq1)/2. Round-trip (entry+exit) cost per leg = 2*impact_pct.

Sizing / normalisation (dollar-neutral, matches the paper-spread weights 0.5/0.25/0.25):
  per-side notional S (x-axis "per-leg" size = the top-1 LONG leg = total short side).
  EQUAL-$   : long = S, each short = S/2 (gross 2S).
  DEPTH-AWARE: each leg capped at min(depth_cap=S, its one-side 1% depth); the
    dollar-neutral per-side deployment P = min(cap_long, cap_s2+cap_s3) <= S,
    shorts split in proportion to their caps. Both books' per-bar return is
    normalised by the SAME target gross 2S, so de-levering thin bars is scored
    honestly (gives up alpha on thin bars but saves convex impact).

Metric: per-bar net spread -> daily-summed -> Sharpe*sqrt(365) (path-independent).

STRICTLY PIT: a position decided at open_time=T uses only depth from the bar
ending at T (l2 index == T-4h).

CAVEAT (prominent): this is BINANCE bookDepth, but the book EXECUTES on
Hyperliquid. The naive prior is HL is thinner -> real impact higher -> these
numbers optimistic. The one available real-fill cross-check (slippage.csv,
2026-05-30) actually shows the model is the SAME order of magnitude and if
anything HARSHER than real HL on thin alts at ~$12k, softer on tiny liquid
trades. So treat these as a fair proxy at small size, NOT a guaranteed
best-case; HL depth for larger trades / stressed tape is unmeasured here.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

from live.bookdepth_impact import impact_pct  # reuse the vetted impact function

ROOT = Path("/home/yuqing/ctaNew")
L2_DIR = ROOT / "data/ml/cache"
PRED = {
    "RECENT": ROOT / "live/state/convexity/hl_tgt_res_base_honest/v0full_hl60.parquet",
    "OOS":    ROOT / "live/state/convexity/hl_v4base_oos_honest/v0full_hl60.parquet",
}
BAR = pd.Timedelta("4h")
TOUCH_FALLBACK = 0.14   # structural 0.2%/1% depth ratio; used only where l2_touch
                        # never observed (added to the feed ~2026-01). Pooled
                        # observed median ~0.137, so this is a minor factor and
                        # matters only for the smallest sizes.
# span sub-$50k (where the raw book actually lives) up to the task's headline grid
SIZES = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 500_000,
         1_000_000, 5_000_000, 20_000_000]
HEADLINE = {50_000, 200_000, 1_000_000, 5_000_000, 20_000_000}
vimpact = np.vectorize(impact_pct, otypes=[float])


# ----------------------------------------------------------------------------- depth
def _ns(idx_or_series):
    """UTC epoch-ns int64 from a tz-aware DatetimeIndex/Series (resolution-agnostic)."""
    return idx_or_series.values.astype("datetime64[ns]").astype("int64")


def load_depth(symbols):
    """Return dict sym -> DataFrame(index=int64 UTC-ns, cols d02,d1,d5). PIT touch fill."""
    out = {}
    for sym in symbols:
        f = L2_DIR / f"l2_{sym}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["l2_liq1", "l2_touch", "l2_slope"]).sort_index()
        d = d[~d.index.duplicated(keep="last")]
        d = d[np.isfinite(d["l2_liq1"])]
        if not len(d):
            continue
        # clips keep cumulative depth monotonic (d02<d1<d5) so impact_pct is well-formed
        touch = d["l2_touch"].ffill().fillna(TOUCH_FALLBACK).clip(0.01, 0.999)  # PIT ffill
        slope = d["l2_slope"].ffill().bfill().clip(lower=1.001)                 # ~never NaN
        L = np.exp(d["l2_liq1"].astype(float))                 # total notional within 1%
        out[sym] = pd.DataFrame({
            "d1":  (L / 2.0).values,
            "d02": (touch * L / 2.0).clip(lower=1.0).values,
            "d5":  (slope * L / 2.0).values,
        }, index=pd.Index(_ns(d.index), name="ts"))
    return out


# ------------------------------------------------------------------------- selection
def build_selection(pred_path):
    """Per open_time: long sym + 2 short syms, their alpha_A, and paper spread."""
    df = pd.read_parquet(pred_path)[["symbol", "open_time", "alpha_A", "pred"]].dropna()
    recs = []
    for T, g in df.groupby("open_time", sort=True):
        if g["symbol"].nunique() < 3:
            continue
        g = g.sort_values("pred")
        s2, s3 = g.iloc[0], g.iloc[1]          # bottom-2 by pred -> SHORT
        lg = g.iloc[-1]                         # top-1 by pred    -> LONG
        recs.append((T, lg["symbol"], s2["symbol"], s3["symbol"],
                     float(lg["alpha_A"]), float(s2["alpha_A"]), float(s3["alpha_A"])))
    sel = pd.DataFrame(recs, columns=["open_time", "long", "s2", "s3",
                                      "a_long", "a_s2", "a_s3"])
    sel["paper"] = 0.5 * sel["a_long"] - 0.25 * sel["a_s2"] - 0.25 * sel["a_s3"]
    # PIT decision-bar L2 index = open_time - 4h, as UTC epoch-ns int64
    sel["dec_ns"] = _ns(sel["open_time"]) - int(BAR.value)
    return sel


def attach_depth(sel, depth):
    """Look up each leg's (d02,d1,d5) at the decision bar. Drop bars w/ any leg missing."""
    # per-symbol O(1) lookup {ns -> (d02,d1,d5)}
    lut = {sym: dict(zip(df.index.values, df[["d02", "d1", "d5"]].values))
           for sym, df in depth.items()}
    dec = sel["dec_ns"].values
    for col, tag in [("long", "L"), ("s2", "A"), ("s3", "B")]:
        vals = np.full((len(sel), 3), np.nan)
        for i, sym in enumerate(sel[col].values):
            r = lut.get(sym, {}).get(dec[i])
            if r is not None:
                vals[i] = r
        sel[[f"d02_{tag}", f"d1_{tag}", f"d5_{tag}"]] = vals
    n0 = len(sel)
    ok = sel[[f"d1_{t}" for t in ("L", "A", "B")]].notna().all(axis=1)
    kept = sel[ok].reset_index(drop=True)
    return kept, n0, len(kept)


# --------------------------------------------------------------------------- backtest
def sharpe(net_spread, index):
    s = pd.Series(net_spread, index=index)
    daily = s.groupby(s.index.floor("D")).sum()
    if daily.std() == 0 or len(daily) < 5:
        return np.nan
    return daily.mean() / daily.std() * np.sqrt(365)


def rt(N, tag, sel):
    """Round-trip impact FRACTION for notional N (scalar or array) on leg `tag`."""
    return 2.0 * vimpact(N, sel[f"d02_{tag}"].values, sel[f"d1_{tag}"].values,
                         sel[f"d5_{tag}"].values)


def equal_dollar(sel, S):
    """Per-bar net spread, equal-$ dollar-neutral: long S, each short S/2, gross 2S."""
    rL = rt(S, "L", sel); rA = rt(S / 2, "A", sel); rB = rt(S / 2, "B", sel)
    impact = 0.5 * rL + 0.25 * rA + 0.25 * rB          # $impact / 2S
    return sel["paper"].values - impact


def depth_aware(sel, S):
    """Per-bar net spread, depth-aware dollar-neutral, normalised by target gross 2S.
    cap_i = min(S, d1_i); P = min(cap_L, cap_A+cap_B); shorts split ∝ cap."""
    cL = np.minimum(S, sel["d1_L"].values)
    cA = np.minimum(S, sel["d1_A"].values)
    cB = np.minimum(S, sel["d1_B"].values)
    cS = cA + cB
    P = np.minimum(cL, cS)                              # per-side deployed <= S
    nA = P * cA / np.maximum(cS, 1.0)                   # short notionals (sum = P)
    nB = P * cB / np.maximum(cS, 1.0)
    alpha_dollar = P * sel["a_long"].values - nA * sel["a_s2"].values - nB * sel["a_s3"].values
    imp_dollar = (P * vimpact(P, sel.d02_L, sel.d1_L, sel.d5_L) * 2.0
                  + nA * vimpact(nA, sel.d02_A, sel.d1_A, sel.d5_A) * 2.0
                  + nB * vimpact(nB, sel.d02_B, sel.d1_B, sel.d5_B) * 2.0)
    net = (alpha_dollar - imp_dollar) / (2.0 * S)      # normalise by target gross
    util = (2.0 * P) / (2.0 * S)                        # gross utilisation
    return net, util


def gated(sel, S, X):
    """Equal-$ at S but DROP any leg whose one-way impact_pct > X; reweight dollar-neutral.
    long dropped -> skip bar; one short dropped -> 1L/1S (0.5/0.5); both -> skip."""
    iL = vimpact(S, sel.d02_L, sel.d1_L, sel.d5_L)
    iA = vimpact(S / 2, sel.d02_A, sel.d1_A, sel.d5_A)
    iB = vimpact(S / 2, sel.d02_B, sel.d1_B, sel.d5_B)
    net = np.full(len(sel), np.nan)
    for i in range(len(sel)):
        if iL[i] > X:
            continue                                    # no long -> skip bar
        keepA, keepB = iA[i] <= X, iB[i] <= X
        if not keepA and not keepB:
            continue                                    # no short -> skip bar
        aL = sel["a_long"].values[i]
        if keepA and keepB:                             # 1L / 2S  (0.5/0.25/0.25)
            paper = 0.5 * aL - 0.25 * sel["a_s2"].values[i] - 0.25 * sel["a_s3"].values[i]
            imp = 0.5 * 2 * iL[i] + 0.25 * 2 * iA[i] + 0.25 * 2 * iB[i]
        else:                                           # 1L / 1S  (0.5/0.5)
            aS = sel["a_s2"].values[i] if keepA else sel["a_s3"].values[i]
            iS = iA[i] if keepA else iB[i]
            paper = 0.5 * aL - 0.5 * aS
            imp = 0.5 * 2 * iL[i] + 0.5 * 2 * iS
        net[i] = paper - imp
    return net


# ------------------------------------------------------------------------------- run
def run_era(era, pred_path):
    print(f"\n{'='*88}\n{era} era  ({pred_path.name})\n{'='*88}")
    sel = build_selection(pred_path)
    syms = pd.unique(sel[["long", "s2", "s3"]].values.ravel())
    depth = load_depth(syms)
    sel, n0, nk = attach_depth(sel, depth)
    cov = sel["open_time"].agg(["min", "max"])
    print(f"bars total={n0}  with-PIT-depth={nk} ({nk/n0*100:.0f}%)  "
          f"L2-covered window {cov['min'].date()} .. {cov['max'].date()}")
    paper_sh = sharpe(sel["paper"].values, sel["open_time"])
    print(f"PAPER (cost-free) Sharpe on the L2-covered bars = {paper_sh:+.3f}\n")

    print(f"{'per-side S':>11} {'gross 2S':>10} | {'EQUAL-$ Sh':>10} {'%paper':>7} | "
          f"{'DEPTH-AW Sh':>11} {'%paper':>7} {'util':>5} | {'Δ vs eq':>7} {'%lost rec':>9}")
    rows = {}
    for S in SIZES:
        eq = equal_dollar(sel, S)
        da, util = depth_aware(sel, S)
        she = sharpe(eq, sel["open_time"]); shd = sharpe(da, sel["open_time"])
        rec = (shd - she) / (paper_sh - she) * 100 if (paper_sh - she) > 1e-6 else np.nan
        rows[S] = (she, shd)
        mark = "*" if S in HEADLINE else " "
        print(f"{mark}${S/1e6:8.3f}M {'$'+format(2*S/1e6,'.2f')+'M':>10} | "
              f"{she:+10.3f} {she/paper_sh*100:6.0f}% | {shd:+11.3f} {shd/paper_sh*100:6.0f}% "
              f"{util.mean()*100:4.0f}% | {shd-she:+7.3f} {rec:8.0f}%")

    # AUM ceiling: first S where Sharpe crosses 0 (linear interp in log-size)
    def crossing(col):
        xs = SIZES; ys = [rows[S][col] for S in xs]
        for i in range(len(xs) - 1):
            if ys[i] > 0 >= ys[i + 1]:                 # interp 0-crossing in log-size
                frac = ys[i] / (ys[i] - ys[i + 1])
                lx = np.log10(xs[i]) + frac * (np.log10(xs[i + 1]) - np.log10(xs[i]))
                return 10 ** lx
        return None
    def fmt(c):
        return f"per-side ~${c/1e3:.1f}k (gross ~${2*c/1e3:.1f}k)" if c else "n/a (never >0 on grid)"
    print(f"\nAUM CEILING (realizable Sharpe -> 0):  equal-$ {fmt(crossing(0))}"
          f"  |  depth-aware {fmt(crossing(1))}")

    # Liquidity gating near/above the ceiling (drop legs too thin at this size)
    print("\nLIQUIDITY GATING (equal-$, drop legs whose one-way impact > X; renormalise):")
    for S in (10_000, 25_000, 100_000):
        base = sharpe(equal_dollar(sel, S), sel["open_time"])
        line = [f"  S=${S/1e3:>4.0f}k no-gate {base:+6.2f}"]
        for X in (0.02, 0.01, 0.005, 0.002, 0.001):
            g = gated(sel, S, X)
            m = ~np.isnan(g)
            sh = sharpe(g[m], sel["open_time"][m]) if m.sum() > 5 else np.nan
            line.append(f"| {X*100:.1f}%:{sh:+5.2f}({m.mean()*100:2.0f}%)")
        print(" ".join(line))
    return sel, depth


def sanity_vs_hl():
    """Order-of-magnitude check: modeled Binance impact vs real Hyperliquid fills."""
    print(f"\n{'='*88}\nSANITY: modeled BINANCE impact vs REAL HYPERLIQUID fills "
          f"(slippage.csv, 2026-05-30)\n{'='*88}")
    s = pd.read_csv(ROOT / "live/state/convexity_twobook/slippage.csv")
    depth = load_depth(s["symbol"].unique())   # self-contained: all fill names
    dec = int(pd.Timestamp("2026-05-30 12:00", tz="UTC").value)   # T-4h for the 16:00 bar
    print(f"{'symbol':11} {'book':>4} {'$notional':>10} {'model_1way':>10} {'real_slip':>9} {'real_tot':>8}")
    mods, reals = [], []
    for _, r in s.iterrows():
        dd = depth.get(r["symbol"])
        if dd is None or dec not in dd.index:
            continue
        row = dd.loc[dec]
        m = impact_pct(r["leg_notional_usd"], row["d02"], row["d1"], row["d5"]) * 1e4
        mods.append(m); reals.append(r["slippage_bps"])
        print(f"{r['symbol']:11} {r['book']:>4} {r['leg_notional_usd']:10.0f} "
              f"{m:9.1f}b {r['slippage_bps']:8.1f}b {r['total_cost_bps']:7.1f}b")
    mods, reals = np.array(mods), np.array(reals)
    print(f"\n  mean modeled(Binance) {mods.mean():.1f}bps vs real(HL slippage) {reals.mean():.1f}bps"
          f"  ->  real / model = {reals.mean()/max(mods.mean(),1e-9):.2f}x")
    print("  Same order of magnitude. Model is if anything HARSHER than real HL on the thin\n"
          "  book-A alts (~2x) and softer on the tiny liquid book-B trades. So the capacity\n"
          "  numbers are NOT wildly optimistic at these sizes; treat them as a fair proxy,\n"
          "  with the caveat that HL depth for LARGER trades / stressed tape is unmeasured here.")


def main():
    run_era("RECENT", PRED["RECENT"])
    run_era("OOS", PRED["OOS"])
    sanity_vs_hl()
    print("\nDONE")


if __name__ == "__main__":
    main()
