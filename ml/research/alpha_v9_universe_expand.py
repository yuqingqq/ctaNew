"""Universe expansion test: ORIG25 vs FULL39 under conv+PM gate.

ORIG25 was the canonical universe pre-PM-gate (25 perps that existed at
v6_clean baseline, "fixed don't-redo" per memory). With PM gate active,
larger universe might:
  + offer more selective top-K (rank pool larger)
  + give more "high conviction" persistent names per cycle
  − include newer, lower-quality perps that dilute alpha
  − increase cross-asset correlation noise

Test: same conv+PM stack on:
  A. ORIG25 (current production), K=7  →  validated Sharpe +2.75
  B. FULL39 (ORIG25 + 14 newer perps), K=7
  C. FULL39 with K=11 (preserves K/N selection fraction)

Quick paired multi-OOS (10 folds) on each.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, list_universe,
    make_xs_alpha_labels, build_basket, build_kline_features,
    add_basket_features, add_engineered_flow_features,
    add_xs_rank_features, XS_RANK_SOURCES,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v8_h48_audit import (
    aggregate_4h_flow, AGGTRADE_4H_TO_ADD, CACHE_DIR,
)
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

HORIZON = 48
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/universe_expand"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}


def _sharpe(x: np.ndarray) -> float:
    if len(x) == 0 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def build_wide_panel_for(universe: list[str], horizon: int = HORIZON) -> pd.DataFrame:
    """Build wide panel for arbitrary universe (mirrors alpha_v8_h48_audit.build_wide_panel
    but parameterized on the universe rather than hard-coded ORIG25)."""
    print(f"Building wide panel for {len(universe)} syms...")
    feats = {s: build_kline_features(s) for s in universe}
    feats = {s: f for s, f in feats.items() if not f.empty}
    if len(feats) < 5:
        raise RuntimeError(f"too few symbols with features: {len(feats)}")
    closes = pd.DataFrame({s: feats[s]["close"] for s in feats}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats.keys()))}

    enriched = {}
    for s in feats:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        cache = CACHE_DIR / f"flow_{s}.parquet"
        if cache.exists():
            flow = pd.read_parquet(cache)
            if flow.index.tz is None:
                flow.index = flow.index.tz_localize("UTC")
            f = f.join(aggregate_4h_flow(flow), how="left")
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, horizon)

    rank_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols = list({s for s, d in XS_RANK_SOURCES.items() if d in rank_cols})
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                       + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                       + src_cols + AGGTRADE_4H_TO_ADD) - set(rank_cols))

    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                          + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows ({panel['symbol'].nunique()} symbols)")
    return panel


def run_config(panel: pd.DataFrame, top_k: int, label: str) -> dict:
    """Run multi-OOS conv+PM on the given panel + K. Returns summary."""
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    avail_feats = [c for c in v6_clean if c in panel.columns]
    print(f"\n  {label}: {len(folds)} folds, K={top_k}, n_features={len(avail_feats)}")

    cycles_baseline = []
    cycles_stacked = []
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            print(f"    fold {fold['fid']:>2}: skipped"); continue
        Xt = tr[avail_feats].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_feats].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        models = [_train(Xt, yt_, Xc, yc_, seed=s) for s in ENSEMBLE_SEEDS]
        Xtest = test[avail_feats].to_numpy(dtype=np.float32)
        pred_test = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                              for m in models], axis=0)

        df_b = evaluate_stacked(test, pred_test, use_conv_gate=False, use_pm_gate=False, top_k=top_k)
        df_s = evaluate_stacked(test, pred_test, use_conv_gate=True, use_pm_gate=True, top_k=top_k)
        for _, row in df_b.iterrows():
            cycles_baseline.append({"fold": fold["fid"], "time": row["time"], "net": row["net_bps"]})
        for _, row in df_s.iterrows():
            cycles_stacked.append({
                "fold": fold["fid"], "time": row["time"], "net": row["net_bps"],
                "skipped": row["skipped"], "n_long": row["n_long"], "n_short": row["n_short"],
                "cost": row["cost_bps"], "long_to": row["long_turnover"],
            })
        print(f"    fold {fold['fid']:>2}: base={df_b['net_bps'].mean():+.2f}  "
              f"stk={df_s['net_bps'].mean():+.2f}  ({time.time()-t0:.0f}s)")

    df_b = pd.DataFrame(cycles_baseline)
    df_s = pd.DataFrame(cycles_stacked)
    base_net = df_b["net"].to_numpy()
    stk_net = df_s["net"].to_numpy()
    sh_b, lo_b, hi_b = block_bootstrap_ci(base_net, statistic=_sharpe, block_size=7, n_boot=2000)
    sh_s, lo_s, hi_s = block_bootstrap_ci(stk_net, statistic=_sharpe, block_size=7, n_boot=2000)
    n_min = min(len(base_net), len(stk_net))
    delta = stk_net[:n_min] - base_net[:n_min]
    rng = np.random.default_rng(42)
    block = 7; n_blocks = int(np.ceil(len(delta) / block)); n_boot = 2000
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, len(delta) - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:len(delta)]
        boot_means[i] = delta[idx].mean()
    d_lo, d_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "label": label, "top_k": top_k, "n_universe": panel["symbol"].nunique(),
        "n_cycles": len(stk_net),
        "baseline_net": base_net.mean(), "baseline_sharpe": sh_b, "baseline_ci": [lo_b, hi_b],
        "stacked_net": stk_net.mean(), "stacked_sharpe": sh_s, "stacked_ci": [lo_s, hi_s],
        "delta_net": delta.mean(), "delta_ci": [d_lo, d_hi],
        "delta_sh": sh_s - sh_b,
        "K_avg": (df_s["n_long"].mean() + df_s["n_short"].mean()) / 2,
        "skip_pct": df_s["skipped"].mean() * 100,
        "stacked_cost": df_s["cost"].mean(),
        "stacked_turnover": df_s["long_to"].mean(),
    }


def main():
    universe_full = sorted([s for s in list_universe(min_days=200)])
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"ORIG25 ({len(orig25)} syms)")
    print(f"FULL ({len(universe_full)} syms): adds {sorted(NEW_SYMBOLS & set(universe_full))}")

    # Build panels
    panel_orig25 = build_wide_panel_for(orig25)
    panel_full = build_wide_panel_for(universe_full)

    # Run configs
    configs = [
        (panel_orig25, 7, "ORIG25 K=7"),
        (panel_full, 7, "FULL K=7 (more selective)"),
        (panel_full, 11, "FULL K=11 (preserves K/N=0.28)"),
    ]
    results = []
    for panel, k, label in configs:
        r = run_config(panel, top_k=k, label=label)
        results.append(r)

    print("\n" + "=" * 110)
    print("UNIVERSE EXPANSION COMPARISON  (multi-OOS, h=48, 4.5 bps/leg, conv+PM gates)")
    print("=" * 110)
    print(f"{'config':<35}  {'N':>3}  {'K':>2}  "
          f"{'baseline_Sh':>11}  {'stacked_Sh':>10}  {'Δsh':>6}  "
          f"{'stacked_net':>11}  {'CI':>16}  {'K_avg':>6}  {'cost':>5}  {'skip%':>5}")
    for r in results:
        print(f"{r['label']:<35}  {r['n_universe']:>3}  {r['top_k']:>2}  "
              f"{r['baseline_sharpe']:>+11.2f}  {r['stacked_sharpe']:>+10.2f}  "
              f"{r['delta_sh']:>+6.2f}  {r['stacked_net']:>+11.2f}  "
              f"[{r['stacked_ci'][0]:+.2f},{r['stacked_ci'][1]:+.2f}]  "
              f"{r['K_avg']:>6.2f}  {r['stacked_cost']:>5.2f}  {r['skip_pct']:>5.1f}")

    pd.DataFrame(results).to_csv(OUT_DIR / "expansion_summary.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
