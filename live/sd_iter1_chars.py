"""Signal-diversity loop — iteration 1 (S1): do the liquidity characteristics the crypto literature ranks
above price/vol actually add to our book?

Runs the REAL pipeline (x6 preproc + per-symbol RidgeCV + HL=60 + exit_time purge + 1d embargo), same harness
that reproduces the documented +0.030/+0.021 baseline. For each candidate: add it to V0_LEAN, measure the
PAIRED delta in per-bar rank-IC with a day-clustered bootstrap CI, both eras.

Also reported for every candidate, because the OI/ADV post-mortem showed a "new" signal can be the known
low-vol factor in a new hat (A2):
  - the feature's own standalone rank-IC
  - its rank-IC after cross-sectional residualization on the vol rank
  - its cross-sectional correlation with the vol rank

New features get the rank (empirical-CDF) preprocessing path rather than winsor+z — they are heavy-tailed
liquidity measures and rank is the distribution-free choice, matching how funding/idio_vol are handled.

Gates (live/SIGNAL_DIVERSITY_LOOP.md): G1 paired Δ rank-IC CI>0 in BOTH eras; G2 survivor still adds after
vol-residualization; G3 converts at book level held-out. Falsifier: nothing clears G1 -> S1 null.
Run: python3 -u -m live.sd_iter1_chars
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import CACHE, ERAS, CUTS, build_panel
from live.v0_feature_ablation import V0, gen
from live.build_alpha_beta_decomp import x6
from live.sd_features import FEATS as NEWF

RNG = np.random.default_rng(17)
CHARS = CACHE / "sd_chars.parquet"
ALL = NEWF + ["past_alpha_7d"]


def build():
    PAN = build_panel()
    C = pd.read_parquet(CHARS)
    C["open_time"] = pd.to_datetime(C["open_time"], utc=True)
    PAN = PAN.merge(C, on=["symbol", "open_time"], how="left")
    PAN = PAN.sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    # trailing 7d sum of the 4h forward-alpha label, shifted a FULL bar (the label looks forward)
    PAN["past_alpha_7d"] = a.transform(lambda s: s.shift(1).rolling(42, min_periods=20).sum())
    x6.HEAVY_TAIL = set(x6.HEAVY_TAIL) | set(ALL)      # rank-transform the new heavy-tailed characteristics
    return PAN.reset_index(drop=True)


def perbar_ic(P, col="pred", tgt="alpha_A"):
    return P.groupby("open_time").apply(
        lambda g: spearmanr(g[col], g[tgt]).correlation if len(g) >= 10 else np.nan).dropna()


def day_ci_diff(a, b, nb=3000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]
    gg = [x["d"].to_numpy() for _, x in j.groupby(pd.to_datetime(j.index, utc=True).floor("1D"))]
    boot = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5])


def day_ci(s, nb=3000):
    gg = [x.to_numpy() for _, x in s.groupby(pd.to_datetime(s.index, utc=True).floor("1D"))]
    b = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cached_gen(PAN, feats, cuts, tag, era):
    fp = CACHE / f"sd1_{tag}_{era}.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["open_time"] = pd.to_datetime(d["open_time"], utc=True); return d
    P = gen(PAN, feats, cuts)
    if P.empty:
        return P
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P.to_parquet(fp, index=False)
    return P


def xs_resid(d, y, x):
    def f(g):
        xv = g[x].to_numpy(); yv = g[y].to_numpy()
        if len(g) < 10 or np.std(xv) == 0:
            return pd.Series(np.nan, index=g.index)
        b = np.polyfit(xv, yv, 1)[0]
        r = yv - b * xv
        return pd.Series(r - r.mean(), index=g.index)
    return d.groupby("open_time", group_keys=False).apply(f)


def main():
    PAN = build()
    cov = {c: PAN[c].notna().mean() for c in ALL}
    print("panel merged. new-feature coverage on the 4h panel:", flush=True)
    for c, v in cov.items():
        print(f"  {c:<18}{v*100:.1f}%", flush=True)

    # ---------- standalone character diagnostics (no model): A2 ----------
    print("\n============ A2 — is each characteristic just the vol factor again? ============", flush=True)
    for era in ERAS:
        c0, c1 = CUTS[era][0], CUTS[era][-1]
        d = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)].copy()
        d = d.rename(columns={"alpha_vs_btc_realized": "alpha_A"}).dropna(subset=["alpha_A", "rvol_7d"])
        d["vol_rank"] = d.groupby("open_time")["rvol_7d"].rank(pct=True)
        print(f"\n----- {era} -----", flush=True)
        print(f"  {'characteristic':<20}{'own IC':<20}{'IC | vol-resid':<22}{'xs corr w/ vol':<15}", flush=True)
        for c in ALL:
            s = d.dropna(subset=[c]).copy()
            if len(s) < 5000:
                print(f"  {c:<20}(insufficient)", flush=True); continue
            s["r"] = s.groupby("open_time")[c].rank(pct=True)
            ic = perbar_ic(s, "r"); lo, hi = day_ci(ic)
            s["res"] = xs_resid(s, "r", "vol_rank")
            ic2 = perbar_ic(s.dropna(subset=["res"]), "res"); lo2, hi2 = day_ci(ic2)
            rho = s.groupby("open_time").apply(
                lambda g: spearmanr(g["r"], g["vol_rank"]).correlation).dropna().mean()
            t2 = "SIG" if (lo2 > 0 or hi2 < 0) else "spans0"
            print(f"  {c:<20}{f'{ic.mean():+.4f}[{lo:+.4f},{hi:+.4f}]':<20}"
                  f"{f'{ic2.mean():+.4f}[{lo2:+.4f},{hi2:+.4f}] {t2}':<22}{rho:<+15.3f}", flush=True)

    # ---------- G1: incremental value through the real pipeline ----------
    print("\n============ G1 — paired Δ rank-IC vs V0_LEAN (real per-symbol Ridge pipeline) ============",
          flush=True)
    res = {}
    for era in ERAS:
        cuts = CUTS[era]
        base = perbar_ic(cached_gen(PAN, list(V0), cuts, "base", era))
        res[("base", era)] = base
        print(f"\n----- {era} -----   baseline V0_LEAN rank-IC {base.mean():+.4f}   "
              f"[gate ~+0.030 RECENT / +0.021 OOS]", flush=True)
        for c in ALL:
            P = cached_gen(PAN, list(V0) + [c], cuts, c, era)
            if P.empty:
                print(f"  +{c:<20} (failed)", flush=True); continue
            ic = perbar_ic(P); res[(c, era)] = ic
            d, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{c:<20} IC {ic.mean():+.4f}   Δ {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}", flush=True)
        P = cached_gen(PAN, list(V0) + ALL, cuts, "allnew", era)
        if not P.empty:
            ic = perbar_ic(P); res[("allnew", era)] = ic
            d, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{'ALL 6 together':<20} IC {ic.mean():+.4f}   Δ {d:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}",
                  flush=True)

    print("\n============ G1 GATE READ ============", flush=True)
    winners = []
    for c in ALL + ["allnew"]:
        ok = True
        for era in ERAS:
            if (c, era) not in res:
                ok = False; break
            _, lo, _ = day_ci_diff(res[("base", era)], res[(c, era)])
            ok = ok and lo > 0
        print(f"  {c:<22}{'PASS both eras' if ok else 'fail'}", flush=True)
        if ok and c != "allnew":
            winners.append(c)
    print(f"\n  G1 survivors: {winners if winners else 'NONE -> S1 is a null, move to S2'}", flush=True)
    print("\nSDITER1DONE", flush=True)


if __name__ == "__main__":
    main()
