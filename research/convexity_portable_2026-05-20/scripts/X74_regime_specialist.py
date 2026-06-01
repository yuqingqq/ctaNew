"""X74 — PIT-routed regime-specialist models on 3-year panel.

Idea (user): train different models per regime, route by PIT regime at inference.
Now feasible with 6 bull folds in the 3-year data.

Design (all PIT — routing uses trailing BTC 30d return, no hindsight):
  - Regime label per row: bull if BTC trailing-30d-ret > +10%, else non_bull
  - Specialist A: Ridge Per-sym trained ONLY on bull rows (within each fold's train set)
  - Specialist B: Ridge Per-sym trained ONLY on non-bull rows
  - At inference for an OOS row: use its PIT regime to pick which specialist predicts
  - Walk-forward, build sleeve, compare to:
      * single V0 (all-regime) = +0.12
      * bull-filter @0.10 = +1.13

Also a simpler baseline: regime-routed = use single-V0 pred in non-bull, ZERO in bull
(= the bull-gate, for reference).
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
OUT = REPO/"research/convexity_portable_2026-05-20/results"; CACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

ALPHAS = [0.01, 0.1, 1, 10, 100]


def btc_regime_pit():
    files = sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float64)
    ret30 = btc/btc.shift(8640) - 1
    out = ret30.to_frame("btc_ret_30d").reset_index()
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    return out


def train_specialist_persym(panel, folds, feats, regime_mask_col, train_regime, label):
    """Per-sym Ridge, but training rows restricted to those where regime_mask_col==train_regime.
    Predictions generated for ALL OOS rows (routing decided later)."""
    preds = []
    for f, ts, te, ec in folds:
        tr = panel[(panel["open_time"] < ec) & (panel[regime_mask_col] == train_regime)]
        oos = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(tr) < 500 or len(oos) == 0: continue
        for sym, g_oos in oos.groupby("symbol"):
            g_tr = tr[tr["symbol"] == sym]
            if len(g_tr) < 100:
                # fallback: pooled within regime
                g_tr = tr
            Xtr = g_tr[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
            ytr = g_tr["target_z"].values if "target_z" in g_tr else g_tr["alpha_A"].values
            mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
            Xtr_s = (Xtr-mu)/sd
            try:
                model = RidgeCV(alphas=ALPHAS).fit(Xtr_s, ytr)
            except Exception:
                continue
            Xoos = g_oos[feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
            Xoos_s = (Xoos-mu)/sd
            p = model.predict(Xoos_s)
            d = g_oos[["symbol","open_time","alpha_A","return_pct","exit_time"]].copy()
            d["pred"] = p; d["fold"] = f
            preds.append(d)
    return pd.concat(preds, ignore_index=True) if preds else None


def main():
    t0 = time.time()
    print("=== X74 PIT-routed regime-specialist models (3-year) ===\n", flush=True)
    panel = pd.read_parquet(REPO/"outputs/vBTC_features/panel_3yr_v0.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)

    reg = btc_regime_pit()
    panel = panel.merge(reg, on="open_time", how="left")
    panel["regime_pit"] = np.where(panel["btc_ret_30d"] > 0.10, "bull", "nonbull")
    if "target_z" not in panel.columns:
        panel = x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32"))

    feats = [f for f in x6.BASE + x6.COHORT_EXTRAS if f in panel.columns]
    folds = x6.get_folds(panel)
    print(f"Panel {len(panel):,} rows; regime split: {panel['regime_pit'].value_counts().to_dict()}")
    print(f"Features: {len(feats)}\n")

    # Specialist B: trained on non-bull rows only
    print("Training non-bull specialist...", flush=True)
    predB = train_specialist_persym(panel, folds, feats, "regime_pit", "nonbull", "specB")
    # Specialist A: trained on bull rows only
    print("Training bull specialist...", flush=True)
    predA = train_specialist_persym(panel, folds, feats, "regime_pit", "bull", "specA")

    if predB is None:
        print("non-bull specialist failed"); return

    # Route: for each OOS row use its PIT regime → bull rows get specA pred, nonbull get specB pred
    base = panel[["symbol","open_time","regime_pit"]].drop_duplicates(["symbol","open_time"])
    routed = predB.rename(columns={"pred":"pred_B"}).merge(
        (predA[["symbol","open_time","pred"]].rename(columns={"pred":"pred_A"}) if predA is not None else
         predB[["symbol","open_time"]].assign(pred_A=0.0)),
        on=["symbol","open_time"], how="left")
    routed = routed.merge(base, on=["symbol","open_time"], how="left")
    routed["pred"] = np.where(routed["regime_pit"]=="bull",
                               routed["pred_A"].fillna(0.0), routed["pred_B"])
    apd_routed = routed[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]]
    p_routed = CACHE/"x74_routed_preds.parquet"; apd_routed.to_parquet(p_routed, index=False)
    m_r = x6.run_sleeve_on_preds(p_routed, "x74_routed")
    print(f"\n[Routed specialists] Sharpe={m_r.get('sharpe',0):+.2f} folds={m_r.get('folds_pos','?')} conc={m_r.get('concentration','?')}")

    # Variant: non-bull specialist only, ZERO in bull (specialist + gate)
    apd_bgate = routed.copy()
    apd_bgate["pred"] = np.where(apd_bgate["regime_pit"]=="bull", 0.0, apd_bgate["pred_B"])
    p_bg = CACHE/"x74_specB_bullzero_preds.parquet"
    apd_bgate[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]].to_parquet(p_bg, index=False)
    m_bg = x6.run_sleeve_on_preds(p_bg, "x74_specB_bullzero")
    print(f"[non-bull specialist + zero-in-bull] Sharpe={m_bg.get('sharpe',0):+.2f} folds={m_bg.get('folds_pos','?')}")

    # Variant: non-bull specialist used everywhere (no routing) — does regime-specialized training alone help?
    apd_Bonly = predB.copy()
    p_Bo = CACHE/"x74_specB_only_preds.parquet"
    apd_Bonly[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]].to_parquet(p_Bo, index=False)
    m_Bo = x6.run_sleeve_on_preds(p_Bo, "x74_specB_only")
    print(f"[non-bull specialist everywhere] Sharpe={m_Bo.get('sharpe',0):+.2f} folds={m_Bo.get('folds_pos','?')}")

    print(f"\nReference: single V0 all-regime +0.12; bull-filter@0.10 +1.13")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
