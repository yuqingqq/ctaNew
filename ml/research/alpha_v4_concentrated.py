"""Cross-sectional v4 — concentrated selection + high-conviction filter.

Two questions:
  1. Does concentrating to top-1 / bottom-1 (instead of top-5 / bottom-5) increase
     spread_alpha enough to overcome cost? Hypothesis: yes — the alpha is
     concentrated in a few extreme picks per bar; averaging across 5 dilutes it.
  2. Does filtering bars by prediction-spread magnitude (only trade when
     (top_pred - bottom_pred) is in the top 20% of cal magnitudes) help?
     Hypothesis: only a fraction of bars have strong cross-sectional dispersion;
     those should have larger realized spread.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from features_ml.cross_sectional import (
    XS_FEATURE_COLS, assemble_universe, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs import (
    _train, _stack_xs_panel, _walk_forward_splits, _holdout_split, _slice,
    HORIZON, REGIME_CUTOFF, ENSEMBLE_SEEDS, NAKED_COST_BPS_PER_LEG,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _eval_portfolio(test, yt, *, n_per_side: int, conviction_quantile: float | None,
                     cal_pred_spread: float | None) -> dict:
    """Compute portfolio P&L with optional concentrated picks and conviction filter."""
    df = test[["open_time", "symbol", "return_pct", "alpha_realized"]].copy()
    df["pred"] = yt
    bars = []
    for t, g in df.groupby("open_time"):
        if len(g) < max(2 * n_per_side, 5): continue
        sorted_g = g.sort_values("pred")
        bot = sorted_g.head(n_per_side)
        top = sorted_g.tail(n_per_side)
        pred_spread = top["pred"].mean() - bot["pred"].mean()
        if conviction_quantile is not None and cal_pred_spread is not None:
            if pred_spread < cal_pred_spread:
                continue
        bars.append({
            "time": t, "n": len(g), "pred_spread": pred_spread,
            "spread_ret_bps": (top["return_pct"].mean() - bot["return_pct"].mean()) * 1e4,
            "spread_alpha_bps": (top["alpha_realized"].mean() - bot["alpha_realized"].mean()) * 1e4,
        })
    bdf = pd.DataFrame(bars)
    if bdf.empty:
        return {"n_bars": 0}
    cost_total = 2 * NAKED_COST_BPS_PER_LEG
    bdf["net_bps"] = bdf["spread_ret_bps"] - cost_total
    return {
        "n_bars": len(bdf),
        "spread_ret": bdf["spread_ret_bps"].mean(),
        "spread_alpha": bdf["spread_alpha_bps"].mean(),
        "net": bdf["net_bps"].mean(),
        "win": (bdf["net_bps"] > 0).mean(),
        "std": bdf["spread_ret_bps"].std(),
        "sharpe": bdf["spread_ret_bps"].mean() / bdf["spread_ret_bps"].std() if bdf["spread_ret_bps"].std() > 0 else np.nan,
    }


def _cal_pred_spread_quantile(cal, yc, *, n_per_side: int, q: float) -> float:
    """Return the qth-quantile of (top-n - bot-n) prediction spreads on cal."""
    df = cal[["open_time", "symbol"]].copy(); df["pred"] = yc
    spreads = []
    for t, g in df.groupby("open_time"):
        if len(g) < max(2 * n_per_side, 5): continue
        sorted_g = g.sort_values("pred")
        spreads.append(sorted_g.tail(n_per_side)["pred"].mean() -
                        sorted_g.head(n_per_side)["pred"].mean())
    if not spreads: return float("nan")
    return float(np.quantile(spreads, q))


def main():
    universe = list_universe(min_days=200)
    log.info("universe: %d symbols", len(universe))

    pkg = assemble_universe(universe, horizon=HORIZON)
    feats_by_sym = pkg["feats_by_sym"]
    basket_close = pkg["basket_close"]
    labels_by_sym = make_xs_alpha_labels(feats_by_sym, basket_close, HORIZON)
    panel = _stack_xs_panel(feats_by_sym, labels_by_sym, cols=XS_FEATURE_COLS)
    panel = panel.dropna(subset=["autocorr_pctile_7d"])

    print("=" * 90)
    print("v4 CONCENTRATED + CONVICTION-FILTERED — top-N / bottom-N variations")
    print("=" * 90)

    configs = []
    for n in (1, 2, 5):
        configs.append({"n_per_side": n, "conviction_q": None})
        configs.append({"n_per_side": n, "conviction_q": 0.80})  # only top 20% conviction bars

    for mode, fold_fn in [("walk-forward", lambda: _walk_forward_splits(panel)),
                           ("OOS holdout",   lambda: _holdout_split(panel))]:
        print(f"\n--- {mode} ---")
        folds = fold_fn()
        agg = {tuple(c.items()): [] for c in configs}

        for fold in folds:
            train, cal, test = _slice(panel, fold)
            train_f = train[train["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            cal_f = cal[cal["autocorr_pctile_7d"] >= 1 - REGIME_CUTOFF]
            if len(train_f) < 1000 or len(cal_f) < 200: continue

            X_train = train_f[XS_FEATURE_COLS].to_numpy()
            y_train = train_f["demeaned_target"].to_numpy()
            X_cal = cal_f[XS_FEATURE_COLS].to_numpy()
            y_cal = cal_f["demeaned_target"].to_numpy()
            models = [_train(X_train, y_train, X_cal, y_cal, seed=s) for s in ENSEMBLE_SEEDS]

            yc = np.mean([m.predict(X_cal, num_iteration=m.best_iteration) for m in models], axis=0)
            yt = np.mean([m.predict(test[XS_FEATURE_COLS].to_numpy(),
                                       num_iteration=m.best_iteration) for m in models], axis=0)

            for c in configs:
                cps = None
                if c["conviction_q"] is not None:
                    cps = _cal_pred_spread_quantile(cal_f, yc, n_per_side=c["n_per_side"],
                                                       q=c["conviction_q"])
                r = _eval_portfolio(test, yt, n_per_side=c["n_per_side"],
                                     conviction_quantile=c["conviction_q"], cal_pred_spread=cps)
                if r.get("n_bars", 0) > 0:
                    agg[tuple(c.items())].append(r)

        print(f"\n  config                              n_bars  spread_ret  spread_alpha     net    sharpe   win")
        for c in configs:
            rs = agg[tuple(c.items())]
            if not rs: continue
            label = f"top-{c['n_per_side']}/bot-{c['n_per_side']}"
            if c["conviction_q"] is not None:
                label += f", convict>q{int(c['conviction_q']*100)}"
            n_bars = sum(r["n_bars"] for r in rs)
            sr = np.mean([r["spread_ret"] for r in rs])
            sa = np.mean([r["spread_alpha"] for r in rs])
            net = np.mean([r["net"] for r in rs])
            sh = np.mean([r["sharpe"] for r in rs])
            wn = np.mean([r["win"] for r in rs])
            print(f"  {label:<35}  {n_bars:>6}     {sr:+6.2f}        {sa:+6.2f}    {net:+6.2f}    {sh:+.3f}   {wn*100:.1f}%")


if __name__ == "__main__":
    main()
