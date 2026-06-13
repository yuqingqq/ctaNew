"""Same-universe re-baselining (owner-requested 2026-05-19).

Honest cleanup, NOT a §4 re-litigation: the §4 verdict (linear line fails
on its own hl42 universe) STANDS. This answers the distinct question the
owner raised — §4 compared a VVV-excluded linear book (hl42, 42 syms)
against a V3.1 baseline whose +2.23/+2.747 is ~62% VVVUSDT. Re-run the
linear book on the EXACT V3.1 candidate universe (50 syms incl. VVV/BIO,
the symbols hl42 deliberately drops as degenerate) and set the correct
apples-to-apples baseline. Key diagnostic: does the linear (β-residual)
model ALSO concentrate on VVV — i.e. is production's edge a universe /
symbol-selection artifact any model captures, or is V3.1's machinery
doing real work the linear book can't replicate?

Reuses harness_v3's 3-agent-cleared pieces UNCHANGED (preprocess,
strict_sigma_idio, model_envelope, walk_forward masks, the 3 BLOCKING
self-checks must re-pass on this universe). Production LGBM untouched.
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


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = _imp("harness_v3", "linear_model/composite_study/harness_v3.py")
S4 = _imp("ensemble_s4", "linear_model/composite_study/ensemble_s4.py")
s92, load_close, trail = H.s92, H.load_close, H.trail
L, BLOCK, OOS = H.L, H.BLOCK, H.OOS
sh = S4.sh
APD = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PS = REPO / "outputs/vBTC_sleeve_horizon/production_sleeves.parquet"
V31_CSV = REPO / "outputs/vBTC_sleeve_horizon/per_cycle_V3.1_equal6_baseline.csv"
OUTD = REPO / "linear_model/composite_study/results"


def load_panel_v31_universe():
    """harness_v3.load_panel logic, but universe = the V3.1 candidate
    pool (all_predictions.parquet symbols, ex-BTC) — NOT hl42, NOT
    excluding BIO/VVV. Everything else (s_t, folds, 4h grid) identical."""
    pan = pd.read_parquet(s92.PANEL)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["exit_time"] = pd.to_datetime(pan["exit_time"], utc=True)
    v31 = set(pd.read_parquet(APD, columns=["symbol"])["symbol"].unique())
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in v31 and s != "BTCUSDT")
    btc = load_close("BTCUSDT").set_index("open_time")["close"]
    btc_rL = trail(btc).rename("ret_btc_L")
    parts = []
    for s in syms:
        c = load_close(s)
        if c is None or len(c) < L + 1000:
            continue
        c = c.set_index("open_time")
        df = pd.concat([trail(c["close"]).rename("ret_asset_L"), btc_rL],
                       axis=1).reset_index()
        df["symbol"] = s
        parts.append(df)
    sig = pd.concat(parts, ignore_index=True)
    sig["open_time"] = pd.to_datetime(sig["open_time"], utc=True)
    d = pan.merge(sig, on=["symbol", "open_time"], how="inner")
    d["s_t"] = (d["ret_asset_L"]
                - d["beta_btc_pit"] * d["ret_btc_L"]).astype("float64")
    d = d.dropna(subset=["s_t", "alpha_beta"]).sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    folds = H._multi_oos_splits(d)
    d["fold"] = -1
    for fid in range(len(folds)):
        d.loc[H._slice(d, folds[fid])[2].index, "fold"] = fid
    oos = d[d["fold"].isin(OOS)].copy()
    grid = sorted(oos["open_time"].unique())[::BLOCK]
    dec = oos[oos["open_time"].isin(set(grid))].sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    return dec, folds


def v31_matched(cov_times, drop_vvv=False):
    """Correct baseline = production V3.1 per-cycle net on the cycle grid
    the linear book actually covers. Full = the +2.23 artifact series;
    ex-VVV = equal-wt pre-sleeve/pre-cost reconstruction from the actual
    traded baskets (directional; the concentration structure is
    weighting-invariant — labelled approximate)."""
    if not drop_vvv:
        v = pd.read_csv(V31_CSV)
        v["open_time"] = pd.to_datetime(v["time"], utc=True)
        v = v[v["open_time"].isin(cov_times)]
        return v["net_pnl_bps"].to_numpy(), v["open_time"].to_numpy()
    ap = pd.read_parquet(APD)[["symbol", "open_time", "return_pct"]]
    ap["open_time"] = pd.to_datetime(ap["open_time"], utc=True)
    ret = ap.set_index(["symbol", "open_time"])["return_pct"].to_dict()
    ps = pd.read_parquet(PS)
    ps["time"] = pd.to_datetime(ps["time"], utc=True)
    ps = ps[ps["time"].isin(cov_times)]
    pc = []
    for _, r in ps.iterrows():
        if not r["traded"]:
            pc.append(0.0)
            continue
        Lg = [s for s in r["long_basket"] if s != "VVVUSDT"]
        Sh = [s for s in r["short_basket"] if s != "VVVUSDT"]
        pl = np.mean([ret.get((s, r["time"]), 0.0) for s in Lg]) if Lg else 0.0
        psh = np.mean([ret.get((s, r["time"]), 0.0) for s in Sh]) if Sh else 0.0
        pc.append(pl - psh)
    return np.asarray(pc), ps["time"].to_numpy()


def attribute(pf, predcol):
    """Per-symbol signed PnL of the naive sign(pred) equal-weight linear
    β-residual book (does it concentrate like V3.1?)."""
    f = pf.copy()
    f["pos"] = np.sign(f[predcol])
    f.loc[f["pos"] == 0, "pos"] = 1.0
    n = f.groupby("open_time")["symbol"].transform("count")
    f["c"] = f["pos"] / n * f["alpha_beta"] * 1e4
    g = f.groupby("symbol")["c"].sum().sort_values()
    ab = g.abs().sort_values(ascending=False)
    return g, ab


def main():
    print("=" * 92, flush=True)
    print("  SAME-UNIVERSE RE-BASELINING (owner-requested; diagnostic, "
          "NOT a §4 re-litigation)", flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()
    dec, folds = load_panel_v31_universe()
    print(f"  universe = V3.1 candidate pool: {dec.symbol.nunique()} syms "
          f"(incl. VVV/BIO that hl42 drops) | rows={len(dec)} "
          f"cycles={dec.open_time.nunique()} "
          f"{dec.open_time.min().date()}→{dec.open_time.max().date()}",
          flush=True)
    if not H.run_selfchecks(dec):
        print("\n  ABORT — self-checks failed on this universe.", flush=True)
        return
    pf = H.walk_forward(dec, folds, members=("ridge_best", "lgbm_es"),
                        verbose=False)
    cov = sorted(pd.unique(pf["open_time"]))
    print(f"  linear book covers {len(cov)} cycles, "
          f"{pf.symbol.nunique()} syms, folds {sorted(pf.fold.unique())}",
          flush=True)
    covset = set(pd.to_datetime(cov))
    vfull, vt = v31_matched(covset, drop_vvv=False)
    vexv, _ = v31_matched(covset, drop_vvv=True)
    sh_v31 = sh(vfull)
    sh_v31_exv = sh(vexv)
    print(f"\n  CORRECT BASELINE (V3.1 on the matched {len(vfull)}-cycle "
          f"grid): full Sharpe={sh_v31:+.3f} | ex-VVV (approx recon)="
          f"{sh_v31_exv:+.3f}", flush=True)

    rows = []
    for member in ("ridge_best", "lgbm_es"):
        g, ab = attribute(pf, member)
        tot = g.sum()
        topname = ab.index[0]
        vvv = float(g.get("VVVUSDT", 0.0))
        top3 = 100 * ab.head(3).sum() / ab.sum() if ab.sum() > 1e-9 else 0
        print(f"\n  ── linear {member} on V3.1 universe ──", flush=True)
        print(f"    per-symbol concentration: top-3={top3:.0f}% of |PnL| "
              f"| biggest={topname} {g[topname]:+.1f}bps | "
              f"VVVUSDT={vvv:+.1f}bps ({100*vvv/tot if abs(tot)>1e-9 else 0:+.0f}% of net)",
              flush=True)
        print(f"    worst3 {list(g.head(3).index)} {g.head(3).round(0).tolist()}"
              f" | best3 {list(g.tail(3).index[::-1])} "
              f"{g.tail(3).iloc[::-1].round(0).tolist()}", flush=True)
        for cost, cl in ((H.MAKER, "MAKER"), (H.COST, "COST2.25")):
            lb = H.linear_book(pf, member, cost=cost).rename(
                columns={"net": "lin"})
            m = lb[lb["open_time"].isin(covset)].sort_values("open_time")
            lin_sh = sh(m["lin"].to_numpy())
            mm = m.merge(pd.DataFrame({"open_time": pd.to_datetime(vt),
                                       "v31": vfull}), on="open_time",
                         how="inner").sort_values(
                "open_time").reset_index(drop=True)
            corr = float(np.corrcoef(mm["lin"], mm["v31"])[0, 1]) \
                if len(mm) > 3 else np.nan
            # nested var-min blend vs the CORRECT (full) V3.1 baseline
            # (mm already carries 'fold' from linear_book — no re-merge)
            ef = sorted(mm["fold"].dropna().unique())
            w = np.zeros(len(mm))
            for i, fdi in enumerate(ef):
                msk = (mm["fold"] == fdi).to_numpy()
                if i == 0:
                    continue
                pr = mm[mm["fold"].isin(ef[:i])]
                L0, V0 = pr["lin"].to_numpy(), pr["v31"].to_numpy()
                den = L0.var() + V0.var() - 2 * np.cov(L0, V0)[0, 1]
                w[msk] = float(np.clip(
                    (V0.var() - np.cov(L0, V0)[0, 1]) / den
                    if abs(den) > 1e-12 else 0.0, 0, 1))
            bl = w * mm["lin"].to_numpy() + (1 - w) * mm["v31"].to_numpy()
            lift = sh(bl) - sh(mm["v31"].to_numpy())
            print(f"    [{cl:8}] linear standalone Sharpe={lin_sh:+.3f} | "
                  f"corr(lin,V31)={corr:+.3f} | nested blend lift vs "
                  f"V3.1-full={lift:+.3f}", flush=True)
            rows.append(dict(member=member, cost=cl, lin_sh=lin_sh,
                             corr=corr, lift_vs_v31full=lift,
                             v31_full=sh_v31, v31_exVVV=sh_v31_exv,
                             vvv_bps=vvv, top3_pct=top3))
    R = pd.DataFrame(rows)
    R.to_csv(OUTD / "same_universe_baseline.csv", index=False)
    print("\n" + "=" * 92, flush=True)
    print("  READING (diagnostic): if linear is still weak AND does NOT "
          "concentrate on VVV → V3.1's +2.23 is a VVV/selection artifact "
          "the β-residual book structurally cannot (and should not) "
          "replicate; the §4 negative stands and the correct same-universe "
          "baseline is V3.1-full (VVV-inflated) vs the documented "
          "ex-VVV ~broad number. Production LGBM unaffected.", flush=True)
    print(f"\nSaved {OUTD}/same_universe_baseline.csv  "
          f"Total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
