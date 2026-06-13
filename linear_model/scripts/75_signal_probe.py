"""Step 75 (Phase 1.5): cheap falsifiers run BEFORE the orientation composites.

Decides whether orientation can possibly be the fix, before building the eight
Phase-2 composites. Implements the NEXT_PLAN.md Phase 1.5 spec exactly:

  1. Feature-IC sign persistence: per V2 feature, per-fold OOS cycle-IC at the
     4h decision cadence; sign-agreement across the 9 OOS folds and
     rho(fold_k IC -> fold_{k+1} IC).
  2. Multivariate OOS reference: pooled RidgeCV on all 22 features, walk-forward
     (the as-built train_ridge), OOS cycle-IC mean/median/t + top-bottom spread.
     A cheap MSE-linear reference, NOT a mathematical ceiling.
  3. Horizon pre-check: repeat (2) for 4h / 8h / 12h / 24h non-overlapping
     residual targets (cheap IC only, no trading). Front-loads Phase 5.

Pre-registered gates (from NEXT_PLAN.md, fixed before this script ran):
  - Sign persistence PASS: >= 8 of the top-10 |IC| features hold the same sign
    in >= 7/9 folds AND mean rho(fold_k -> fold_{k+1} IC) >= +0.20.
  - Multivariate reference PASS: OOS cycle-IC mean >= +0.02 with t >= 3.0 at
    ANY tested horizon.

Testbed (Phase 1 deliverable): current drop-BIO+VVV HL>=2M universe, V2 22
features, causal-engine fold protocol. No trading engine is invoked here.
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


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s64 = _imp("s64", "linear_model/scripts/64_meanrev_v2_backtest.py")
s67 = _imp("s67", "linear_model/scripts/67_persymbol_meanrev.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step75_signal_probe"
OUT.mkdir(parents=True, exist_ok=True)
OOS = set(s64.OOS)            # {1..9}
BLOCK = s64.BLOCK             # 48 bars = 4h decision cadence
DROP = ["BIOUSDT", "VVVUSDT"]
HORIZONS_H = [4, 8, 12, 24]   # hours
WINSOR = 5.0                  # target_z clip, matches project convention


def assign_folds(df: pd.DataFrame, folds: list) -> pd.DataFrame:
    """Label each row with its OOS test fold (fid), -1 if train/embargo only."""
    df = df.copy()
    df["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(df, folds[fid])[2]
        df.loc[te.index, "fold"] = fid
    return df


def decision_times(df: pd.DataFrame) -> set:
    """OOS decision-cadence open_times: every BLOCK-th unique OOS timestamp."""
    d = df[df["fold"].isin(OOS)]
    return set(sorted(d["open_time"].unique())[::BLOCK])


def cycle_ic_frame(df: pd.DataFrame, score: str, y: str = "alpha_beta") -> pd.DataFrame:
    """Per-open_time Spearman IC of `score` vs `y`, with fold attached."""
    rows = []
    for t, g in df.dropna(subset=[score, y]).groupby("open_time", sort=True):
        if len(g) < 5 or g[score].std() <= 1e-12 or g[y].std() <= 1e-12:
            continue
        rows.append({"open_time": t, "fold": int(g["fold"].iloc[0]),
                     "ic": float(g[score].corr(g[y], method="spearman"))})
    return pd.DataFrame(rows)


def ic_stats(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = len(ic)
    sd = ic.std(ddof=1) if n > 2 else np.nan
    t = float(ic.mean() / (sd / np.sqrt(n))) if (n > 2 and sd and sd > 0) else np.nan
    return {"n_cycles": n, "ic_mean": float(ic.mean()) if n else np.nan,
            "ic_median": float(ic.median()) if n else np.nan, "ic_t": t}


def top_bottom(df: pd.DataFrame, score: str, k: int = 3) -> dict:
    sp = []
    for _, g in df.dropna(subset=[score, "alpha_beta"]).groupby("open_time"):
        if len(g) < 2 * k:
            continue
        gs = g.sort_values(score, ascending=False)
        sp.append(gs.head(k)["alpha_beta"].mean() * 1e4
                  - gs.tail(k)["alpha_beta"].mean() * 1e4)
    sp = np.array(sp, float)
    return {"top_minus_bottom_bps": float(sp.mean()) if len(sp) else np.nan,
            "ls_half_weight_bps": float(0.5 * sp.mean()) if len(sp) else np.nan,
            "spread_pos_pct": float((sp > 0).mean()) if len(sp) else np.nan}


# ---------------------------------------------------------------------------
# PART 1 — feature-IC sign persistence
# ---------------------------------------------------------------------------
def part1_sign_persistence(px: pd.DataFrame, fc: list) -> tuple[pd.DataFrame, dict]:
    st = decision_times(px)
    dec = px[px["fold"].isin(OOS) & px["open_time"].isin(st)].copy()
    print(f"  decision frame: {len(dec):,} rows, "
          f"{dec['open_time'].nunique()} cycles, {dec['symbol'].nunique()} syms",
          flush=True)
    rows = []
    for f in fc:
        cif = cycle_ic_frame(dec, f)
        if cif.empty:
            continue
        full = float(cif["ic"].mean())
        fold_ic = (cif.groupby("fold")["ic"].mean()
                   .reindex(sorted(OOS)))                     # f1..f9
        s_full = np.sign(full) if full != 0 else 0.0
        agree = int(((np.sign(fold_ic) == s_full) & (s_full != 0)).sum())
        v = fold_ic.to_numpy(float)
        ok = ~np.isnan(v)
        if ok.sum() >= 4 and np.nanstd(v[:-1]) > 0 and np.nanstd(v[1:]) > 0:
            a, b = v[:-1], v[1:]
            m = ~np.isnan(a) & ~np.isnan(b)
            rho = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 3 else np.nan
        else:
            rho = np.nan
        row = {"feature": f, "full_ic": full, "abs_ic": abs(full),
               "n_folds_sign_agree": agree, "rho_lag1": rho}
        for fid in sorted(OOS):
            row[f"ic_f{fid}"] = float(fold_ic.get(fid, np.nan))
        rows.append(row)
    fdf = pd.DataFrame(rows).sort_values("abs_ic", ascending=False).reset_index(drop=True)
    fdf["in_top10"] = False
    fdf.loc[:9, "in_top10"] = True
    top = fdf[fdf["in_top10"]]
    n_persist = int((top["n_folds_sign_agree"] >= 7).sum())
    mean_rho = float(top["rho_lag1"].mean())
    gate = (n_persist >= 8) and (mean_rho >= 0.20)
    fdf.to_csv(OUT / "feature_ic_persistence.csv", index=False)
    print(f"\n  top-10 |IC| features (sign agreement / 9 folds, rho_lag1):",
          flush=True)
    for _, r in top.iterrows():
        print(f"    {r['feature']:26s} IC={r['full_ic']:+.4f} "
              f"sign_agree={int(r['n_folds_sign_agree'])}/9 "
              f"rho={r['rho_lag1']:+.3f}", flush=True)
    print(f"\n  SIGN-PERSISTENCE GATE: {n_persist}/10 features persist >=7/9 "
          f"(need >=8); mean rho_lag1={mean_rho:+.3f} (need >=+0.20) "
          f"-> {'PASS' if gate else 'FAIL'}", flush=True)
    return fdf, {"n_persist_top10": n_persist, "mean_rho_top10": mean_rho,
                 "pass": bool(gate)}


# ---------------------------------------------------------------------------
# PARTS 2 & 3 — multivariate reference at 4h, repeated per horizon
# ---------------------------------------------------------------------------
def build_horizon_px(px: pd.DataFrame, panel: pd.DataFrame, folds: list,
                      n_cyc: int, hours: int) -> pd.DataFrame:
    """Per-bar forward non-overlapping sum of alpha_beta over n_cyc 4h blocks.

    alpha_beta[t] is the [t, t+4h] beta-residual; the H-horizon residual is the
    sum of the next n_cyc non-overlapping 4h blocks. Cheap additive proxy for
    an IC-only pre-check (no trading). target_z rebuilt with fold-0-frozen
    per-symbol sigma (cross-sym median fallback), matching project convention.
    exit_time extended to t+H for honest label purging.
    """
    d = px.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    if n_cyc == 1:
        d["alpha_H"] = d["alpha_beta"]
    else:
        parts = [d.groupby("symbol")["alpha_beta"].shift(-j * BLOCK)
                 for j in range(n_cyc)]
        d["alpha_H"] = sum(parts)            # NaN if window incomplete -> dropped
    # fold-0 train sigma per symbol, frozen
    f0 = _slice(d, folds[0])[0]
    sg = f0.groupby("symbol")["alpha_H"].std()
    med = float(sg.dropna().median())
    sig = d["symbol"].map(sg).fillna(med).clip(lower=1e-6)
    d["target_z"] = (d["alpha_H"] / sig).clip(-WINSOR, WINSOR).astype("float32")
    d["alpha_beta"] = d["alpha_H"]           # train_ridge reports cyc_ic vs this
    ex = panel[["symbol", "open_time", "exit_time"]].copy()
    ex["exit_time"] = pd.to_datetime(ex["exit_time"], utc=True)
    d = d.merge(ex, on=["symbol", "open_time"], how="left")
    d["exit_time"] = d["open_time"] + pd.Timedelta(hours=hours)
    return d


def eval_pooled(preds: pd.DataFrame, folds: list, stride_cyc: int) -> dict:
    """Cycle-IC of pooled pred_z vs alpha_beta at non-overlapping cadence."""
    preds = assign_folds(preds, folds)
    d = preds[preds["fold"].isin(OOS)].copy()
    tt = sorted(d["open_time"].unique())[::(BLOCK * stride_cyc)]
    d = d[d["open_time"].isin(set(tt))]
    cif = cycle_ic_frame(d, "pred_z")
    st = ic_stats(cif["ic"]) if not cif.empty else ic_stats(pd.Series(dtype=float))
    st.update(top_bottom(d, "pred_z"))
    fp = 0
    if not cif.empty:
        fp = int((cif.groupby("fold")["ic"].mean() > 0).sum())
    st["folds_pos"] = fp
    return st


def main():
    print("=" * 92, flush=True)
    print("  STEP 75 (Phase 1.5): cheap falsifiers before orientation composites",
          flush=True)
    print("=" * 92, flush=True)
    t0 = time.time()

    print("\nBuilding current drop-BIO+VVV HL>=2M testbed (V2, 22 feats)...",
          flush=True)
    panel, px, fc, folds = s67.build_panel(DROP)
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    print(f"  panel rows={len(px):,}, symbols={px['symbol'].nunique()}, "
          f"features={len(fc)}", flush=True)
    pxf = assign_folds(px, folds)

    print("\n--- PART 1: feature-IC sign persistence (4h cadence) ---", flush=True)
    _, g1 = part1_sign_persistence(pxf, fc)

    print("\n--- PARTS 2&3: pooled-Ridge multivariate reference per horizon ---",
          flush=True)
    hz_rows = []
    g2_pass = False
    for h in HORIZONS_H:
        n_cyc = h // 4
        print(f"\n  [{h}h] retraining pooled RidgeCV (n_cyc={n_cyc})...",
              flush=True)
        if h == 4:
            preds = s58.train_ridge(px, folds, fc)
        else:
            px_h = build_horizon_px(px, panel, folds, n_cyc, h)
            preds = s58.train_ridge(px_h, folds, fc)
        st = eval_pooled(preds, folds, stride_cyc=n_cyc)
        passed = (not np.isnan(st["ic_mean"]) and st["ic_mean"] >= 0.02
                  and not np.isnan(st["ic_t"]) and st["ic_t"] >= 3.0)
        g2_pass = g2_pass or passed
        hz_rows.append({"horizon_h": h, "n_cyc": n_cyc, **st,
                        "gate_pass": bool(passed)})
        print(f"  [{h}h] OOS cycle-IC mean={st['ic_mean']:+.4f} "
              f"median={st['ic_median']:+.4f} t={st['ic_t']:+.2f} "
              f"n={st['n_cycles']} | top-bottom={st['top_minus_bottom_bps']:+.2f} "
              f"bps | folds+={st['folds_pos']}/9 -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)

    hz = pd.DataFrame(hz_rows)
    hz.to_csv(OUT / "horizon_reference.csv", index=False)
    hz[hz["horizon_h"] == 4].to_csv(OUT / "multivariate_reference.csv", index=False)

    # ---- pre-registered verdict ----
    print("\n" + "=" * 92, flush=True)
    print("  PRE-REGISTERED VERDICT (gates fixed in NEXT_PLAN.md before run)",
          flush=True)
    print("=" * 92, flush=True)
    sp = g1["pass"]
    print(f"  Sign-persistence gate ........ {'PASS' if sp else 'FAIL'} "
          f"({g1['n_persist_top10']}/10 persist, mean rho "
          f"{g1['mean_rho_top10']:+.3f})", flush=True)
    print(f"  Multivariate-reference gate .. {'PASS' if g2_pass else 'FAIL'} "
          f"(IC>=+0.02 & t>=3.0 at any horizon)", flush=True)
    if not sp and not g2_pass:
        verdict = ("BOTH FAIL -> orientation/model layer unlikely the fix; no "
                   "persistent cross-sectional signal in V2 at 4-24h. Skip "
                   "Phase 2/3; Phase 4 only under the kill clock, or stop.")
    elif sp and not g2_pass:
        verdict = ("SIGN-PERSIST ONLY -> features carry stable but tiny signal; "
                   "Phase 2 may help at the margin, absolute level is the wall.")
    else:
        verdict = ("MULTIVARIATE REFERENCE PASSES -> extractable linear signal "
                   "the current model wastes; Phase 2/3 is the right lever.")
    print(f"\n  {verdict}", flush=True)
    pd.DataFrame([{
        "sign_persistence_pass": sp,
        "n_persist_top10": g1["n_persist_top10"],
        "mean_rho_top10": g1["mean_rho_top10"],
        "multivariate_reference_pass": g2_pass,
        "verdict": verdict,
    }]).to_csv(OUT / "verdict.csv", index=False)
    print(f"\nSaved outputs under {OUT}", flush=True)
    print(f"Total: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
