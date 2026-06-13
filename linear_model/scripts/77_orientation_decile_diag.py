"""Step 77: A_4h orientation diagnostic — deciles / bands / K-sweep.

NOT a backtest. NOT optimization. NOT 24h (Step-76 Part B invalid). Answers the
one user question on the Step-76 Part-A composite (the +0.0517 / t +8.46 / 9-9
4h `signed_all_shrunk_ic_weighted`):

  > Is the positive rank-IC monetizable in some band, or is it rank signal
  > that disappears / inverts at the tradable extremes?

The composite SCORE is reconstructed by importing Step-76's own helpers
(`fit_weights`, `assign_folds`, `build_panel`, cross-sectional-z) verbatim, so
the diagnosed score is byte-identical to the one that produced A_4h — no
divergence risk. For every OOS decision cycle we emit (score, fwd alpha_beta)
per symbol, then report:

  1. decile returns (+ overall & fold-level monotonicity)
  2. quintile / band returns
  3. middle-vs-extreme behaviour
  4. long/short spread by K = 1, 2, 3, 5, 10 (mean, t, %+, per-leg)
  5. interior-band L/S combos (does moving off the tail turn spread positive?)

Decision logic (user-stated, applied in the printed verdict):
  - deciles monotonic but extremes bad -> entry rule needs redesign
  - deciles NOT monotonic            -> IC not economically useful
"""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step77_orientation_decile_diag"
OUT.mkdir(parents=True, exist_ok=True)
OOS = s76.OOS
BLOCK = s76.BLOCK
DROP = s76.DROP
Y = "alpha_beta"


def build_scores() -> tuple[pd.DataFrame, list]:
    """Reconstruct the exact Step-76 Part-A composite score per (cycle, sym)."""
    panel, px, fc, folds = s76.s67.build_panel(DROP)
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    grid = sorted(px["open_time"].unique())[::BLOCK]
    dec = px[px["open_time"].isin(set(grid))].copy()
    dec = s76.assign_folds(dec, folds)
    rows = []
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=[Y])
        if len(tr) < 500:
            continue
        w = s76.fit_weights(tr, fc, Y)                       # identical to s76
        wv = np.array([w[f] for f in fc], float)
        te = dec[dec["fold"] == k].dropna(subset=[Y]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            X = g[fc].to_numpy(float)
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            sd[sd <= 1e-12] = 1.0
            Z = np.nan_to_num((X - mu) / sd)                 # cross-sectional z
            yv = g[Y].to_numpy(float)
            if np.std(yv) <= 1e-12:
                continue
            sc = Z @ wv
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": yv}))
    return pd.concat(rows, ignore_index=True), fc


def _binned(df: pd.DataFrame, nbin: int) -> pd.DataFrame:
    """Per-cycle equal-count bins by score; mean fwd y (bps) per bin, then
    average across cycles (cycle-equal-weighted)."""
    per = []
    for t, g in df.groupby("open_time", sort=True):
        if len(g) < nbin * 2:
            continue
        r = g["score"].rank(method="first")
        try:
            b = pd.qcut(r, nbin, labels=False, duplicates="drop")
        except ValueError:
            continue
        gg = g.assign(b=b)
        per.append(gg.groupby("b")["y"].mean() * 1e4)
    if not per:
        return pd.DataFrame()
    M = pd.concat(per, axis=1).T                              # cycles x bins
    out = pd.DataFrame({"bin": M.columns,
                        "mean_bps": M.mean().values,
                        "t": M.mean().values /
                             (M.std(ddof=1).values / np.sqrt(M.count().values)),
                        "pos_pct": (M > 0).mean().values,
                        "n_cyc": M.count().values})
    return out


def _mono(tbl: pd.DataFrame) -> float:
    if len(tbl) < 3:
        return np.nan
    return float(pd.Series(tbl["mean_bps"].values).corr(
        pd.Series(tbl["bin"].values), method="spearman"))


def _ksweep(df: pd.DataFrame, ks=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    res = []
    grp = list(df.groupby("open_time", sort=True))
    for K in ks:
        L, S, SP = [], [], []
        for _, g in grp:
            if len(g) < 2 * K:
                continue
            gs = g.sort_values("score", ascending=False)
            lo = gs.head(K)["y"].mean() * 1e4
            sh = gs.tail(K)["y"].mean() * 1e4
            L.append(lo)
            S.append(sh)
            SP.append(lo - sh)
        SP = np.array(SP, float)
        L = np.array(L, float)
        S = np.array(S, float)
        res.append({
            "K": K, "n_cyc": len(SP),
            "long_bps": float(L.mean()), "short_bps": float(S.mean()),
            "spread_bps": float(SP.mean()),
            "spread_t": float(SP.mean() / (SP.std(ddof=1) / np.sqrt(len(SP))))
            if len(SP) > 2 else np.nan,
            "spread_pos_pct": float((SP > 0).mean()),
        })
    return pd.DataFrame(res)


def _band_ls(df: pd.DataFrame, combos) -> pd.DataFrame:
    """L/S using decile bands (long set vs short set of deciles)."""
    res = []
    for name, longs, shorts in combos:
        SP = []
        for t, g in df.groupby("open_time", sort=True):
            if len(g) < 20:
                continue
            r = g["score"].rank(method="first")
            try:
                b = pd.qcut(r, 10, labels=False, duplicates="drop")
            except ValueError:
                continue
            gg = g.assign(b=b)
            lo = gg[gg["b"].isin(longs)]["y"]
            sh = gg[gg["b"].isin(shorts)]["y"]
            if len(lo) and len(sh):
                SP.append((lo.mean() - sh.mean()) * 1e4)
        SP = np.array(SP, float)
        res.append({"band": name, "n_cyc": len(SP),
                    "spread_bps": float(SP.mean()) if len(SP) else np.nan,
                    "spread_t": float(SP.mean() / (SP.std(ddof=1) /
                                np.sqrt(len(SP)))) if len(SP) > 2 else np.nan,
                    "pos_pct": float((SP > 0).mean()) if len(SP) else np.nan})
    return pd.DataFrame(res)


def main():
    print("=" * 92, flush=True)
    print("  STEP 77: A_4h orientation diagnostic (deciles / bands / K-sweep)",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()

    print("\nReconstructing Step-76 Part-A composite score...", flush=True)
    df, fc = build_scores()
    print(f"  {len(df):,} (cycle,symbol) rows, {df['open_time'].nunique()} "
          f"OOS cycles, {df['symbol'].nunique()} syms, {len(fc)} feats",
          flush=True)
    # sanity: reproduce the headline cycle-IC
    ic = df.groupby("open_time").apply(
        lambda g: g["score"].corr(g["y"], method="spearman")).dropna()
    print(f"  sanity cycle-IC mean={ic.mean():+.4f} (Step-76 A_4h was +0.0517)",
          flush=True)

    dec = _binned(df, 10)
    qui = _binned(df, 5)
    dec.to_csv(OUT / "decile_returns.csv", index=False)
    qui.to_csv(OUT / "quintile_returns.csv", index=False)
    dmono = _mono(dec)
    qmono = _mono(qui)
    print(f"\n--- DECILES (0=low score, 9=high; IC>0 ⇒ expect rising) "
          f"monotonicity ρ={dmono:+.3f} ---", flush=True)
    for _, r in dec.iterrows():
        print(f"  D{int(r['bin'])}: {r['mean_bps']:+7.2f} bps  t={r['t']:+5.2f} "
              f"pos={r['pos_pct']*100:4.0f}%", flush=True)
    print(f"\n--- QUINTILES  monotonicity ρ={qmono:+.3f} ---", flush=True)
    for _, r in qui.iterrows():
        print(f"  Q{int(r['bin'])}: {r['mean_bps']:+7.2f} bps  t={r['t']:+5.2f}",
              flush=True)

    # middle vs extreme
    if len(dec) == 10:
        ext = dec[dec["bin"].isin([0, 9])]["mean_bps"].mean()
        mid = dec[dec["bin"].isin([3, 4, 5, 6])]["mean_bps"].mean()
        inner = dec[dec["bin"].isin([1, 2, 7, 8])]["mean_bps"].mean()
        print(f"\n--- MIDDLE vs EXTREME ---", flush=True)
        print(f"  extreme deciles (D0,D9) mean: {ext:+.2f} bps", flush=True)
        print(f"  inner    deciles (D1-2,7-8): {inner:+.2f} bps", flush=True)
        print(f"  middle   deciles (D3-6)   : {mid:+.2f} bps", flush=True)

    ks = _ksweep(df)
    ks.to_csv(OUT / "topk_spread.csv", index=False)
    print(f"\n--- LONG/SHORT SPREAD by K (the tradability question) ---",
          flush=True)
    for _, r in ks.iterrows():
        print(f"  K={int(r['K']):2d}: long={r['long_bps']:+7.2f} "
              f"short={r['short_bps']:+7.2f} spread={r['spread_bps']:+7.2f} "
              f"t={r['spread_t']:+5.2f} pos={r['spread_pos_pct']*100:4.0f}%",
              flush=True)

    combos = [
        ("D9 / D0", [9], [0]),
        ("D8-9 / D0-1", [8, 9], [0, 1]),
        ("D7-9 / D0-2", [7, 8, 9], [0, 1, 2]),
        ("D6-8 / D1-3 (interior)", [6, 7, 8], [1, 2, 3]),
        ("D5-9 / D0-4 (half-split)", [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]),
    ]
    bl = _band_ls(df, combos)
    bl.to_csv(OUT / "band_ls.csv", index=False)
    print(f"\n--- INTERIOR-BAND L/S (does moving off the tail help?) ---",
          flush=True)
    for _, r in bl.iterrows():
        print(f"  {r['band']:26s} spread={r['spread_bps']:+7.2f} bps "
              f"t={r['spread_t']:+5.2f} pos={r['pos_pct']*100:4.0f}%",
              flush=True)

    # fold-level decile monotonicity
    frows = []
    for k, g in df.groupby("fold", sort=True):
        fb = _binned(g, 10)
        frows.append({"fold": int(k), "mono_rho": _mono(fb),
                      "D0_bps": float(fb.loc[fb["bin"] == 0, "mean_bps"].iloc[0])
                      if (fb["bin"] == 0).any() else np.nan,
                      "D9_bps": float(fb.loc[fb["bin"] == 9, "mean_bps"].iloc[0])
                      if (fb["bin"] == 9).any() else np.nan})
    fdf = pd.DataFrame(frows)
    fdf.to_csv(OUT / "fold_decile_mono.csv", index=False)
    print(f"\n--- FOLD-LEVEL decile monotonicity ---", flush=True)
    for _, r in fdf.iterrows():
        print(f"  fold {int(r['fold'])}: ρ={r['mono_rho']:+.3f}  "
              f"D0={r['D0_bps']:+6.1f}  D9={r['D9_bps']:+6.1f}", flush=True)
    fmono_pos = int((fdf["mono_rho"] > 0).sum())

    # data-driven verdict
    print("\n" + "=" * 92, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 92, flush=True)
    best_k = ks.loc[ks["spread_bps"].idxmax()]
    best_band = bl.loc[bl["spread_bps"].idxmax()]
    any_tradable = (best_k["spread_bps"] > 0 and best_k["spread_t"] > 2.0) or \
                   (best_band["spread_bps"] > 0 and best_band["spread_t"] > 2.0)
    monotonic = (abs(dmono) >= 0.6 and fmono_pos >= 6) if dmono == dmono else False
    if not monotonic:
        v = ("DECILES NOT MONOTONIC (ρ={:+.2f}, fold+ {}/9) -> the +0.052 IC is "
             "rank noise without a monotone payoff; NOT economically useful. "
             "Orientation lever is real statistically but dead economically."
             ).format(dmono, fmono_pos)
    elif any_tradable:
        v = ("MONOTONIC and a tradable band exists (best: K/band spread "
             "{:+.2f} bps t {:+.2f}) -> entry rule must move OFF the raw "
             "top-3/bottom-3; redesign entry to the profitable band, then "
             "(and only then) backtest.").format(
            max(best_k["spread_bps"], best_band["spread_bps"]),
            max(best_k["spread_t"], best_band["spread_t"]))
    else:
        v = ("MONOTONIC mid-rank but NO band clears a tradable L/S (all K and "
             "interior bands ≤0 or insignificant) -> broad ranking info that "
             "does not monetize anywhere; entry-rule redesign unlikely to "
             "rescue it. Treat as not economically useful pending the "
             "leakage audit.")
    print(f"  decile ρ={dmono:+.3f}  quintile ρ={qmono:+.3f}  "
          f"fold-mono+ {fmono_pos}/9", flush=True)
    print(f"  best K spread {best_k['spread_bps']:+.2f} bps (K={int(best_k['K'])}, "
          f"t {best_k['spread_t']:+.2f}); best band {best_band['spread_bps']:+.2f} "
          f"bps ({best_band['band']}, t {best_band['spread_t']:+.2f})", flush=True)
    print(f"\n  {v}", flush=True)
    pd.DataFrame([{"decile_rho": dmono, "quintile_rho": qmono,
                   "fold_mono_pos": fmono_pos,
                   "best_k_spread": float(best_k["spread_bps"]),
                   "best_band_spread": float(best_band["spread_bps"]),
                   "verdict": v}]).to_csv(OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
