"""Step 76: minimal orientation falsifier + proper 24h target rebuild.

User-approved bounded sequence (NOT a rabbit hole):

  Part A  — minimal orientation on the current 4h target.
            ONE composite only: signed_all_shrunk_ic_weighted (no top-k suite,
            no backtest). Per-fold pre-fold train IC -> orient by sign,
            weight by |IC| shrunk by its own signal/noise (parameter-free:
            shrink = |mean_IC| / (|mean_IC| + SE_IC)). Evaluate OOS cycle-IC /
            spread / fold stability only. HARD FAIL if IC < +0.02 or t < 3.0.

  Part B  — proper 24h residual target (Step-75's 24h was a cheap additive
            alpha_beta-sum proxy; it showed +16 bps half-weight spread so it
            earns one clean rebuild). True 24h forward beta-residual:
            compounded 6x non-overlapping 4h forward returns, PIT beta at
            entry, fold-0-frozen per-symbol sigma. Same composite, same gate,
            non-overlapping (stride-6) cadence so the t-stat is not
            autocorrelation-inflated.

Pre-registered gate (fixed before run, same as NEXT_PLAN.md Phase 1.5/2):
  PASS iff OOS cycle-IC mean >= +0.02 AND t >= +3.0.

If BOTH parts FAIL -> orientation is confirmed dead by direct test; the only
remaining plan branch is Step-3 new interaction features (price x volume x
volatility), to be decided separately. No trading engine is invoked here.
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


s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s67 = _imp("s67", "linear_model/scripts/67_persymbol_meanrev.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step76_minimal_orientation"
OUT.mkdir(parents=True, exist_ok=True)
OOS = sorted(s64.OOS)               # [1..9]
BLOCK = s64.BLOCK                   # 48 bars = 4h
DROP = ["BIOUSDT", "VVVUSDT"]
GATE_IC, GATE_T = 0.02, 3.0


def assign_folds(df: pd.DataFrame, folds: list) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(df, folds[fid])[2]
        df.loc[te.index, "fold"] = fid
    return df


def per_cycle_ic(df: pd.DataFrame, col: str, y: str) -> np.ndarray:
    """Spearman IC of `col` vs `y` per decision cycle (one value per cycle)."""
    out = []
    for _, g in df.groupby("open_time", sort=True):
        v = g[[col, y]].dropna()
        if len(v) < 5 or v[col].std() <= 1e-12 or v[y].std() <= 1e-12:
            continue
        out.append(v[col].corr(v[y], method="spearman"))
    return np.array(out, float)


def fit_weights(train: pd.DataFrame, fc: list, y: str) -> dict:
    """signed_all_shrunk_ic_weighted: per-feature pre-fold train IC -> sign x
    |IC| x (|IC|/(|IC|+SE)). No hard selection, no tunable knob."""
    w = {}
    for f in fc:
        ic = per_cycle_ic(train, f, y)
        if len(ic) < 5:
            w[f] = 0.0
            continue
        m = float(ic.mean())
        se = float(ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 2 else np.inf
        shrink = abs(m) / (abs(m) + se) if (abs(m) + se) > 0 else 0.0
        w[f] = np.sign(m) * abs(m) * shrink
    return w


def composite_eval(dec: pd.DataFrame, fc: list, folds: list, y: str,
                    stride: int, label: str) -> dict:
    """Build the signed shrunk-IC composite per fold, evaluate OOS only."""
    ic_rows, fold_ic = [], {}
    eq_ic_rows = []                 # informational: pure signed-equal sibling
    tb = []
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0]
        tr = tr.dropna(subset=[y])
        if len(tr) < 500:
            continue
        w = fit_weights(tr, fc, y)
        wv = np.array([w[f] for f in fc], float)
        we = np.array([np.sign(w[f]) for f in fc], float)   # signed-equal
        te = dec[(dec["fold"] == k)].dropna(subset=[y]).copy()
        te_t = sorted(te["open_time"].unique())[::stride]
        te = te[te["open_time"].isin(set(te_t))]
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            X = g[fc].to_numpy(float)
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            sd[sd <= 1e-12] = 1.0
            Z = (X - mu) / sd                       # cross-sectional z per cycle
            Z = np.nan_to_num(Z)
            yv = g[y].to_numpy(float)
            if np.std(yv) <= 1e-12:
                continue
            for wts, sink in ((wv, ic_rows), (we, eq_ic_rows)):
                sc = Z @ wts
                if np.std(sc) <= 1e-12:
                    continue
                ic = pd.Series(sc).corr(pd.Series(yv), method="spearman")
                sink.append({"open_time": t, "fold": k, "ic": float(ic)})
            # top/bottom 3 spread on the gated composite
            sc = Z @ wv
            order = np.argsort(-sc)
            if len(order) >= 6:
                top = yv[order[:3]].mean() * 1e4
                bot = yv[order[-3:]].mean() * 1e4
                tb.append(top - bot)
    ic_df = pd.DataFrame(ic_rows)
    eq_df = pd.DataFrame(eq_ic_rows)

    def _stats(d):
        if d.empty:
            return dict(n=0, mean=np.nan, median=np.nan, t=np.nan, fp=0)
        n = len(d)
        sd = d["ic"].std(ddof=1) if n > 2 else np.nan
        t = float(d["ic"].mean() / (sd / np.sqrt(n))) if (sd and sd > 0) else np.nan
        fp = int((d.groupby("fold")["ic"].mean() > 0).sum())
        return dict(n=n, mean=float(d["ic"].mean()),
                    median=float(d["ic"].median()), t=t, fp=fp)

    g = _stats(ic_df)
    e = _stats(eq_df)
    tb = np.array(tb, float)
    spread = float(tb.mean()) if len(tb) else np.nan
    passed = (not np.isnan(g["mean"]) and g["mean"] >= GATE_IC
              and not np.isnan(g["t"]) and g["t"] >= GATE_T)
    if not ic_df.empty:
        ic_df.to_csv(OUT / f"{label}_cycle_ic.csv", index=False)
    print(f"\n  [{label}] signed_all_shrunk_ic_weighted: "
          f"OOS cycle-IC mean={g['mean']:+.4f} median={g['median']:+.4f} "
          f"t={g['t']:+.2f} n={g['n']} folds+={g['fp']}/9 | "
          f"top-bottom={spread:+.2f} bps | "
          f"{'PASS' if passed else 'FAIL'} (need IC>=+{GATE_IC} & t>=+{GATE_T})",
          flush=True)
    print(f"  [{label}] (info) signed-equal sibling: IC mean={e['mean']:+.4f} "
          f"t={e['t']:+.2f} folds+={e['fp']}/9", flush=True)
    return dict(label=label, ic_mean=g["mean"], ic_median=g["median"],
                ic_t=g["t"], n_cycles=g["n"], folds_pos=g["fp"],
                top_bottom_bps=spread, gate_pass=bool(passed),
                eq_ic_mean=e["mean"], eq_ic_t=e["t"], eq_folds_pos=e["fp"])


def build_24h_target(dec: pd.DataFrame, panel: pd.DataFrame,
                     folds: list) -> pd.DataFrame:
    """Proper 24h forward beta-residual: compounded 6 non-overlapping 4h fwd
    returns, PIT beta at entry, fold-0-frozen per-symbol sigma (clip 1e-6).
    Replaces Step-75's additive alpha_beta-sum proxy."""
    cols = ["symbol", "open_time", "return_pct", "btc_ret_fwd", "beta_btc_pit"]
    p = panel[cols].copy()
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    d = dec.merge(p, on=["symbol", "open_time"], how="left")
    d = d.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    n = 24 // 4                                     # 6 non-overlapping 4h blocks
    gr = d.groupby("symbol", sort=False)
    comp_r = np.ones(len(d))
    comp_b = np.ones(len(d))
    for j in range(n):
        comp_r *= (1.0 + gr["return_pct"].shift(-j * BLOCK).to_numpy())
        comp_b *= (1.0 + gr["btc_ret_fwd"].shift(-j * BLOCK).to_numpy())
    ret24 = comp_r - 1.0
    btc24 = comp_b - 1.0
    d["resid24"] = ret24 - d["beta_btc_pit"].to_numpy() * btc24
    f0 = _slice(d, folds[0])[0]
    sg = f0.groupby("symbol")["resid24"].std()
    med = float(sg.dropna().median())
    d["_sig24"] = d["symbol"].map(sg).fillna(med).clip(lower=1e-6)
    return d


def main():
    print("=" * 92, flush=True)
    print("  STEP 76: minimal orientation (4h) + proper 24h rebuild — IC gate only",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()

    print("\nBuilding drop-BIO+VVV HL>=2M testbed (V2 22 feats)...", flush=True)
    panel, px, fc, folds = s67.build_panel(DROP)
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    grid = sorted(px["open_time"].unique())[::BLOCK]      # global 4h decision grid
    dec = px[px["open_time"].isin(set(grid))].copy()
    dec = assign_folds(dec, folds)
    print(f"  decision frame: {len(dec):,} rows, "
          f"{dec['open_time'].nunique()} cycles, {dec['symbol'].nunique()} syms, "
          f"{len(fc)} features", flush=True)

    res = []
    print("\n--- PART A: minimal orientation, current 4h alpha_beta target ---",
          flush=True)
    res.append(composite_eval(dec, fc, folds, "alpha_beta", 1, "A_4h"))

    print("\n--- PART B: proper 24h forward beta-residual rebuild ---", flush=True)
    dec24 = build_24h_target(dec, panel, folds)
    res.append(composite_eval(dec24, fc, folds, "resid24", 6, "B_24h_proper"))

    out = pd.DataFrame(res)
    out.to_csv(OUT / "summary.csv", index=False)

    print("\n" + "=" * 92, flush=True)
    print("  PRE-REGISTERED VERDICT (gate fixed before run: IC>=+0.02 & t>=+3.0)",
          flush=True)
    print("=" * 92, flush=True)
    for r in res:
        print(f"  {r['label']:14s} -> {'PASS' if r['gate_pass'] else 'FAIL'} "
              f"(IC {r['ic_mean']:+.4f}, t {r['ic_t']:+.2f}, "
              f"folds+ {r['folds_pos']}/9)", flush=True)
    any_pass = any(r["gate_pass"] for r in res)
    if any_pass:
        v = ("AT LEAST ONE PASS -> orientation/target is a real lever; proceed "
             "to the next plan step for the passing config only.")
    else:
        v = ("BOTH FAIL -> orientation directly refuted on 4h AND proper 24h. "
             "Step-75 kill stands. Only remaining branch = Step-3 new "
             "interaction features (price x volume x volatility), decided "
             "separately. No more plain price/vol/funding variants.")
    print(f"\n  {v}", flush=True)
    pd.DataFrame([{"any_pass": any_pass, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
