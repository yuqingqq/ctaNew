"""Detail-review loop — iteration 3 (D3): the equity kill used a 15-name universe; the DEPLOYED config was 11.

I reported the equity strategy dead on `alpha_v7_honest.py`'s hard split (-0.20) and flagged a caveat that it
tested "the full S&P 100" while the deployed config was 11 hand-picked names. Reading the code, my caveat was
HALF right:
  - the honest script does NOT trade the S&P 100; it passes allowed=set(XYZ_IN_SP100), a 15-name set
  - but XYZ_IN_SP100 = the 11 deployed Tier A+B names PLUS AMD, COST, INTC, LLY — exactly the four Tier-C
    names `docs/STATUS.md` says were dropped because "they hurt backtest at realistic cost"

So the kill was measured on a universe the operators had already rejected. That is a real, testable
difference and it is the kind of detail this loop exists to catch.

Identical hard split (train <=2019 frozen, test 2020-2026, PIT rolling gate, same features, same TOP_K,
same cost) with only the EXECUTION universe varied:
    full15   XYZ_IN_SP100 (what produced -0.20)
    tier_ab  the 11 deployed names
    tierC    the 4 dropped names alone — diagnostic: were they really the problem?

Honest prior: the per-year decay in the 15-name test (2020 +1.07, 2021 +0.95, 2022 -0.16, 2023 -1.93,
2024 -0.90, 2025 -0.35, 2026 -1.79) looks like signal decay, not universe contamination, so tier_ab is
unlikely to rescue it. But it has never been measured and I should not assert it.

Gate: tier_ab hard-split active Sharpe > 0 with CI excluding 0 AND not negative in the last 3 years.
Falsifier: fails -> the equity programme is dead on the DEPLOYED universe, measured rather than assumed.
Run: python3 -u -m live.dr_iter3_eq11
"""
from __future__ import annotations

import importlib.util as iu
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
TIER_AB = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "MU", "NFLX", "NVDA", "ORCL", "PLTR", "TSLA"]
TIER_C = ["AMD", "COST", "INTC", "LLY"]


def main():
    spec = iu.spec_from_file_location("h", REPO / "ml/research/alpha_v7_honest.py")
    m = iu.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    logging.getLogger().setLevel(logging.WARNING)

    panel, earnings, _ = m.load_universe()
    anchors = m.load_anchors()
    panel = m.add_returns_and_basket(panel)
    panel = m.add_residual_and_label(panel, m.HOLD_DAYS)
    panel, fA = m.add_features_A(panel)
    panel, fB = m.add_features_B_fixed(panel, earnings)
    panel["sym_id"] = panel["symbol"].astype("category").cat.codes
    regime = m.compute_regime_indicators(panel, anchors)
    label = f"fwd_resid_{m.HOLD_DAYS}d"
    feats = fA + fB + ["sym_id"]
    print(f"panel {len(panel):,} rows | {len(feats)} features | label {label}", flush=True)

    train_end = pd.Timestamp("2019-12-31", tz="UTC")
    test_start = pd.Timestamp("2020-01-01", tz="UTC")
    test_end = panel["ts"].max()

    universes = [("full15  (what produced -0.20)", list(m.XYZ_IN_SP100)),
                 ("tier_ab (the DEPLOYED 11)", TIER_AB),
                 ("tierC   (the 4 dropped)", TIER_C)]
    print(f"\n{'universe':<30}{'n_reb':>7}{'net/d bps':>11}{'active Sh':>11}{'95% CI':>20}", flush=True)
    out = {}
    for name, uni in universes:
        try:
            raw = m.hard_split_test(panel, feats, label, train_end, test_start, test_end,
                                    top_k=m.TOP_K, cost_bps_side=m.COST_BPS_SIDE,
                                    hold_days=m.HOLD_DAYS, allowed=set(uni))
            if raw.empty:
                print(f"{name:<30} empty", flush=True); continue
            gated = m.gate_rolling(raw, regime, m.GATE_PCTILE, m.GATE_WINDOW_DAYS)
            r = m.report(name, gated, m.HOLD_DAYS)
            out[name] = (gated, r)
            lo, hi = m.bootstrap_active_sharpe_ci(gated, m.HOLD_DAYS)
            mm = r.get("m", {})
            print(f"{name:<30}{mm.get('n_rebal', len(gated)):>7}"
                  f"{mm.get('net_bps_per_day', np.nan):>11.2f}"
                  f"{mm.get('active_sharpe_annu', np.nan):>11.2f}"
                  f"{f'[{lo:+.2f},{hi:+.2f}]':>20}", flush=True)
        except Exception as e:
            print(f"{name:<30} ERR {type(e).__name__}: {str(e)[:90]}", flush=True)

    print("\n=== per-year, deployed 11-name universe ===", flush=True)
    key = "tier_ab (the DEPLOYED 11)"
    if key in out:
        g = out[key][0].copy()
        tcol = "ts" if "ts" in g.columns else g.columns[0]
        g[tcol] = pd.to_datetime(g[tcol], utc=True)
        g["yr"] = g[tcol].dt.year
        ncol = [c for c in g.columns if "net" in c.lower()]
        ncol = ncol[0] if ncol else None
        if ncol:
            for y, s in g.groupby("yr")[ncol]:
                sh = s.mean() / s.std() * np.sqrt(252 / m.HOLD_DAYS) if s.std() > 0 else np.nan
                print(f"  {y}  n={len(s):>3}  net/d {s.mean():>8.2f} bps  Sh {sh:>6.2f}", flush=True)
    print("\nDRITER3DONE", flush=True)


if __name__ == "__main__":
    main()
