"""Step 89 (Stage 3): per-symbol OI model + per-symbol PIT normalization.

Builds the per-symbol model the way the user argued it should be tested
(Step-67 was on the leaky pre-Step-72 engine + wrong normalization; never
re-done rigorously). PRODUCES + VALIDATES the intermediate artifact; renders
NO verdict (that is Step 90, with placebo + pre-registered gate).

Design:
  - 23 symbols with full clean PIT OI history (build_btc_oi_features, audited).
  - Per-symbol PIT trailing-z normalization of every OI feature: at row t,
    z = (x_t - mean(x_{t-W..t-1})) / std(x_{t-W..t-1}), W=90d, strictly past
    (rolling .shift(1)). x_t is itself PIT (shift(1) in the OI builder).
    This is the Step-68 self-std recipe applied correctly per symbol.
  - Per-symbol INDEPENDENT model (one model per symbol on its own series):
    pred_ridge = RidgeCV on that symbol's walk-forward train (autocorr filter,
    target_z); pred_signed = per-symbol train-IC-sign equal-weight (the
    estimator-robustness reference, Step-84 G4 lesson).
  - Walk-forward folds = global date-based _multi_oos_splits; per symbol
    via _slice on that symbol's subframe.

INTERMEDIATE VALIDATION (printed + saved; gates the next stage):
  V1 normalization PIT: independent strictly-past recompute exact-match on a
     sample (symbol,feature); corr with a LEAKY (non-shifted-stats) version
     must be < 1 (proves it is genuinely past-only, not look-ahead).
  V2 per-symbol OOS IC sane: corr(pred, alpha_beta) within each symbol over
     its OOS rows — distribution finite, not all ~0, not exploding.
  V3 per-symbol feature IC-sign stability across that symbol's folds
     (Insight-1 applied PER SYMBOL — directly tests the user's hypothesis
     that per-symbol OI relations are more stationary than pooled).
Saves OOS preds -> outputs for Step 90. No backtest / no verdict here.
"""
from __future__ import annotations
import importlib.util, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n, r):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
OI = REPO / "outputs/vBTC_features_oi/oi_panel.parquet"
OUTD = REPO / "linear_model/results/step89_per_symbol_oi"
OUTD.mkdir(parents=True, exist_ok=True)
ALPHAS = s58.ALPHAS
AUTO = s58.AUTO_THRESH
DAY = 288
W = 90 * DAY                      # per-symbol normalization window (90d)
MINP = 7 * DAY                    # need >=7d to z-score
OOS = list(range(1, 10))
FEATS = ["oi_chg_1h", "oi_chg_4h", "oi_chg_1d", "oi_z_1d", "oi_z_7d",
         "oiv_z_1d", "ls_count_z_1d", "ls_count_chg_4h", "ls_top_z_1d",
         "ls_taker_z_1d", "ls_taker_chg_4h"]


def per_symbol_pit_z(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """z = (x - past-mean)/past-std within each symbol, strictly past
    (rolling .shift(1)). df sorted by [symbol, open_time]."""
    out = df.copy()
    g = out.groupby("symbol", sort=False)
    for c in cols:
        mu = g[c].transform(lambda s: s.rolling(W, min_periods=MINP).mean().shift(1))
        sd = g[c].transform(lambda s: s.rolling(W, min_periods=MINP).std().shift(1))
        out[c + "_n"] = ((out[c] - mu) / sd.replace(0, np.nan)).astype("float32")
    return out


def main():
    print("=" * 96, flush=True)
    print("  STEP 89 (Stage 3): per-symbol OI model + per-symbol PIT norm",
          flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "alpha_beta",
                                          "sigma_idio", "autocorr_pctile_7d",
                                          "exit_time"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    # target_z is derived (not stored): clip(alpha_beta/sigma_idio, ±WINSOR),
    # exactly as s58.build_target_z.
    WS = float(getattr(s58, "WINSORIZE_SIGMA", 5.0))
    pan["target_z"] = (pan["alpha_beta"] / pan["sigma_idio"]).clip(
        -WS, WS).astype("float32")
    oi = pd.read_parquet(OI)
    oi["open_time"] = pd.to_datetime(oi["open_time"], utc=True)
    df = pan.merge(oi, on=["symbol", "open_time"], how="inner").sort_values(
        ["symbol", "open_time"]).reset_index(drop=True)
    syms = sorted(df["symbol"].unique())
    print(f"  joined: {len(df):,} rows, {len(syms)} syms "
          f"({df.open_time.min().date()}→{df.open_time.max().date()})",
          flush=True)

    df = per_symbol_pit_z(df, FEATS)
    NF = [c + "_n" for c in FEATS]
    folds = _multi_oos_splits(df)
    df["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(df, folds[fid])[2]
        df.loc[te.index, "fold"] = fid

    # ---------- V1: per-symbol PIT normalization audit ----------
    print("\n--- V1 per-symbol PIT-normalization audit ---", flush=True)
    okV1 = True
    for s, c in [("SOLUSDT", "oi_z_1d"), ("ADAUSDT", "oi_chg_1d")]:
        d = df[df.symbol == s].sort_values("open_time").reset_index(drop=True)
        x = d[c].astype(float)
        # independent strictly-past recompute
        mp = x.rolling(W, min_periods=MINP).mean().shift(1)
        sp = x.rolling(W, min_periods=MINP).std().shift(1)
        ind = (x - mp) / sp.replace(0, np.nan)
        # LEAKY version (stats include current row -> uses t)
        leak = (x - x.rolling(W, min_periods=MINP).mean()) / \
               x.rolling(W, min_periods=MINP).std().replace(0, np.nan)
        st = d[c + "_n"].astype(float)
        v = pd.DataFrame({"st": st, "ind": ind, "lk": leak}).dropna(
            subset=["st", "ind"])
        c_ind = float(v["st"].corr(v["ind"]))
        md = float((v["st"] - v["ind"]).abs().max())
        c_lk = float(v["st"].corr(v["lk"]))
        good = c_ind > 0.9999 and md < 1e-6 and c_lk < 0.9999
        okV1 &= good
        print(f"  {s}/{c}: corr(stored,indep_PAST)={c_ind:.6f} maxdiff={md:.1e}"
              f"  corr(stored,LEAKY)={c_lk:.4f} (<1 ⇒ genuinely past-only) "
              f"-> {'OK' if good else 'CHECK'}", flush=True)

    # ---------- per-symbol independent model, walk-forward ----------
    print("\n--- per-symbol model (RidgeCV + signed-equal), walk-forward ---",
          flush=True)
    preds = []
    sign_rows = []                                  # for V3
    for s in syms:
        d = df[df.symbol == s]
        for k in OOS:
            if k >= len(folds):
                continue
            tr, _, te = _slice(d, folds[k])
            tr = tr[(tr["autocorr_pctile_7d"] >= AUTO)].dropna(
                subset=["target_z"] + NF)
            te = te.dropna(subset=NF).copy()
            if len(tr) < 800 or len(te) < 50:
                continue
            Xtr = tr[NF].to_numpy(np.float32)
            ytr = tr["target_z"].to_numpy(np.float32)
            Xte = te[NF].to_numpy(np.float32)
            # per-symbol feature IC sign on train (time-series corr)
            ics = {c: float(np.corrcoef(tr[c], tr["alpha_beta"])[0, 1])
                   if tr[c].std() > 1e-12 and tr["alpha_beta"].std() > 1e-12
                   else 0.0 for c in NF}
            for c in NF:
                sign_rows.append({"symbol": s, "fold": k, "feature": c,
                                  "ic": ics[c]})
            sgn = np.array([np.sign(ics[c]) or 1.0 for c in NF], float)
            m = RidgeCV(alphas=ALPHAS, scoring="r2",
                        fit_intercept=False).fit(Xtr, ytr)
            te = te.assign(
                pred_ridge=m.predict(Xte).astype(np.float32),
                pred_signed=(Xte @ sgn).astype(np.float32))
            preds.append(te[["symbol", "open_time", "fold", "alpha_beta",
                             "target_z", "pred_ridge", "pred_signed"]])
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(OUTD / "per_symbol_oi_preds.parquet", index=False)
    SR = pd.DataFrame(sign_rows)
    SR.to_csv(OUTD / "per_symbol_feature_ic.csv", index=False)

    # ---------- V2: per-symbol OOS IC sanity ----------
    print("\n--- V2 per-symbol OOS IC (corr(pred, alpha_beta) within symbol) ---",
          flush=True)
    rows = []
    for s, g in P.groupby("symbol"):
        gg = g.dropna(subset=["pred_ridge", "alpha_beta"])
        if len(gg) < 200 or gg["pred_ridge"].std() < 1e-12:
            continue
        rows.append({"symbol": s, "n": len(gg),
                     "ic_ridge": float(np.corrcoef(gg["pred_ridge"],
                                                   gg["alpha_beta"])[0, 1]),
                     "ic_signed": float(np.corrcoef(gg["pred_signed"],
                                                    gg["alpha_beta"])[0, 1])
                     if gg["pred_signed"].std() > 1e-12 else np.nan})
    ICs = pd.DataFrame(rows)
    ICs.to_csv(OUTD / "per_symbol_oos_ic.csv", index=False)
    finite = np.isfinite(ICs["ic_ridge"]).all()
    print(f"  n_syms={len(ICs)}  ic_ridge mean={ICs.ic_ridge.mean():+.4f} "
          f"median={ICs.ic_ridge.median():+.4f} "
          f"%pos={100*(ICs.ic_ridge>0).mean():.0f}  "
          f"[min {ICs.ic_ridge.min():+.3f}, max {ICs.ic_ridge.max():+.3f}]",
          flush=True)
    print(f"  ic_signed mean={ICs.ic_signed.mean():+.4f} "
          f"%pos={100*(ICs.ic_signed>0).mean():.0f} | all finite={finite}",
          flush=True)

    # ---------- V3: per-symbol feature IC-sign stability across folds ----------
    print("\n--- V3 per-symbol feature IC-sign stability (Insight-1 per sym) ---",
          flush=True)
    st_rows = []
    for (s, f), g in SR.groupby(["symbol", "feature"]):
        v = g["ic"].dropna()
        if len(v) < 5:
            continue
        pos = int((v > 0).sum())
        neg = int((v < 0).sum())
        st_rows.append({"symbol": s, "feature": f,
                        "same_sign_frac": max(pos, neg) / len(v),
                        "mean_ic": float(v.mean())})
    STB = pd.DataFrame(st_rows)
    STB.to_csv(OUTD / "per_symbol_sign_stability.csv", index=False)
    consistent = float((STB["same_sign_frac"] >= 7/9).mean())
    print(f"  (symbol,feature) pairs sign-consistent ≥7/9 folds: "
          f"{100*consistent:.0f}%  (mean same-sign frac "
          f"{STB.same_sign_frac.mean():.2f})", flush=True)
    print(f"  → for comparison, pooled cross-sectional sign-persistence was "
          f"≈noise (Step-75 rho +0.05); this measures the per-symbol analog",
          flush=True)

    v1 = "PASS" if okV1 else "FAIL"
    print(f"\n  INTERMEDIATE VALIDATION: V1(norm-PIT)={v1}  V2(IC finite/sane)="
          f"{'PASS' if finite and len(ICs)>=15 else 'CHECK'}  V3 recorded",
          flush=True)
    print(f"  preds saved → {OUTD/'per_symbol_oi_preds.parquet'} "
          f"({len(P):,} rows). NO verdict here — Step 90 does rigorous eval.",
          flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
