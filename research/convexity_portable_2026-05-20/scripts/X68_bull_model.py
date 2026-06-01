"""X68 — Build a dedicated BULL-regime model (momentum-oriented).

Current V5_mv3 is a sideways/mean-reversion play that FAILS in bull (fold 8 hurts).
Hypothesis: a momentum-feature model captures bull-regime alpha better.

Build momentum features:
  mom_7d, mom_14d, mom_30d (PIT returns)
  rel_strength_7d, rel_strength_30d (sym ret - btc ret)
  mom_30d_vol_adj (mom_30d / rvol_7d)
  bars_since_high, bars_since_high_xs_rank (breakout — already in panel)

Train Ridge Per-sym on momentum features, eval:
  1. Overall Sharpe (all regimes)
  2. Bull-fold (0, 8) contribution
  3. Regime ensemble: bull-model in bull, V5_mv3 elsewhere
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
KLINES = REPO / "data/ml/test/parquet/klines"
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def load_close(sym):
    sd = KLINES / sym / "5m"
    if not sd.exists(): return None
    dfs = [pd.read_parquet(f, columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time")["close"].astype(np.float32)


def build_momentum_features(syms):
    btc = load_close("BTCUSDT")
    btc_7d = (btc / btc.shift(2016) - 1)
    btc_30d = (btc / btc.shift(8640) - 1)
    out = []
    for sym in syms:
        c = load_close(sym)
        if c is None: continue
        btc7 = btc_7d.reindex(c.index).ffill()
        btc30 = btc_30d.reindex(c.index).ffill()
        mom_7d = (c / c.shift(2016) - 1).shift(1)
        mom_14d = (c / c.shift(4032) - 1).shift(1)
        mom_30d = (c / c.shift(8640) - 1).shift(1)
        rvol_7d = np.log(c / c.shift(1)).rolling(2016, min_periods=288).std().shift(1)
        df = pd.DataFrame({
            "symbol": sym,
            "open_time": c.index,
            "mom_7d": mom_7d.astype(np.float32).values,
            "mom_14d": mom_14d.astype(np.float32).values,
            "mom_30d": mom_30d.astype(np.float32).values,
            "rel_strength_7d": (mom_7d - btc7.shift(1)).astype(np.float32).values,
            "rel_strength_30d": (mom_30d - btc30.shift(1)).astype(np.float32).values,
            "mom_30d_vol_adj": (mom_30d / (rvol_7d * np.sqrt(8640) + 1e-6)).astype(np.float32).values,
        })
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main():
    print("=== X68 bull-regime momentum model ===\n")
    # Load HL-50 panel
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_hl70_v5_full.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    canonical = sorted(set(pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
        columns=["symbol"])["symbol"].unique()) - {"BTCUSDT"})
    panel = panel[panel["symbol"].isin(canonical)]
    print(f"HL-50 panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms")

    # Build momentum features
    print("Building momentum features...")
    mom = build_momentum_features(canonical)
    mom["open_time"] = pd.to_datetime(mom["open_time"], utc=True)
    panel = panel.merge(mom, on=["symbol","open_time"], how="left")
    print(f"  merged momentum features")

    panel = x6.build_target_z(panel)
    if "bars_since_high_xs_rank" not in panel.columns:
        panel["bars_since_high_xs_rank"] = (panel.groupby("open_time")["bars_since_high"]
                                            .rank(pct=True).astype("float32"))
    folds = x6.get_folds(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")

    # Bull-model features (momentum-oriented)
    mom_feats = ["mom_7d","mom_14d","mom_30d","rel_strength_7d","rel_strength_30d",
                 "mom_30d_vol_adj","bars_since_high","bars_since_high_xs_rank",
                 "return_1d","ret_3d","corr_to_btc_1d","rvol_7d"]
    mom_feats = [f for f in mom_feats if f in panel.columns]
    print(f"\nBull-model features ({len(mom_feats)}): {mom_feats}")

    apd = x6.train_per_sym_ridge(panel, folds, mom_feats, label="x68_bull")
    pred_path = CACHE / "x68_bull_model_preds.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    print(f"\nTrained: {len(apd):,} rows, IC={ic:+.4f}")

    m = x6.run_sleeve_on_preds(pred_path, "x68_bull")
    print(f"Bull-model overall: Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')}")

    # LOFO to find bull-fold contributions (folds 0, 8 are BULL)
    print(f"\nBull-model LOFO (folds 0 and 8 are BULL regime):")
    base_sh = m.get("sharpe", 0) or 0
    for f in [0, 8]:
        apd_d = apd[apd["fold"] != f]
        tmp = CACHE / f"x68_drop{f}_preds.parquet"
        apd_d.to_parquet(tmp, index=False)
        md = x6.run_sleeve_on_preds(tmp, f"x68_drop{f}")
        sh = md.get("sharpe", 0) or 0
        print(f"  drop fold {f}: {sh:+.2f} (Δ {sh-base_sh:+.2f})")

    # Regime ensemble: bull-model in bull, V5_mv3 elsewhere
    print(f"\n=== Regime ensemble: bull-model in BULL, V5_mv3 in sideways/bear ===")
    btc = load_close("BTCUSDT")
    btc_30d_ret = (btc / btc.shift(8640) - 1).astype(np.float32)
    btc_df = btc_30d_ret.to_frame("btc_30d_ret").reset_index()
    btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], utc=True)

    v5mv3 = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    v5mv3["open_time"] = pd.to_datetime(v5mv3["open_time"], utc=True)
    # Normalize both per-fold
    apd["pred_n"] = apd.groupby("fold")["pred"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))
    v5mv3["pred_n"] = v5mv3.groupby("fold")["pred"].transform(lambda x: (x-x.mean())/(x.std()+1e-8))

    merged = v5mv3[["symbol","open_time","alpha_A","return_pct","exit_time","fold","pred_n"]].rename(
        columns={"pred_n":"pred_v5mv3"})
    merged = merged.merge(apd[["symbol","open_time","pred_n"]].rename(columns={"pred_n":"pred_bull"}),
                           on=["symbol","open_time"], how="left")
    merged = merged.merge(btc_df, on="open_time", how="left")

    for thr in [0.10, 0.15, 0.20]:
        m2 = merged.copy()
        in_bull = m2["btc_30d_ret"] > thr
        m2["pred"] = np.where(in_bull, m2["pred_bull"], m2["pred_v5mv3"]).astype(np.float32)
        apd2 = m2[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]].copy()
        tmp = CACHE / f"x68_ensemble_t{thr}_preds.parquet"
        apd2.to_parquet(tmp, index=False)
        mm = x6.run_sleeve_on_preds(tmp, f"x68_ens_t{thr}")
        sh = mm.get("sharpe", 0) or 0
        fp = mm.get("folds_pos", "?")
        conc = mm.get("concentration", "?")
        print(f"  bull-model+V5mv3 (thr={thr}): Sharpe={sh:+.2f} folds={fp} conc={conc}")

    print(f"\nReference: V5_mv3 alone +1.74, X66 V5_mv3+V0 ensemble +2.08")


if __name__ == "__main__":
    main()
