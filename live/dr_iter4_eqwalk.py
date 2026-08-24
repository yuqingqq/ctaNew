"""Detail-review loop — iteration 4 (D3b): does the equity +1.82 survive more observations, and is it
uncorrelated with crypto?

D3 found the equity kill was measured on the wrong universe: the deployed 11 names give a hard-split Sharpe
of +1.82, not the -0.20 I reported. But that rests on 40 rebalances with a CI of [-1.00,+5.14]. Two things
must hold before it counts as a sleeve:
  G1  it survives WALK-FORWARD (far more rebalances than one hard split), CI excluding 0
  G2  its return series is near-uncorrelated with the crypto sleeves — the only reason to want it
  G3  it is not carried by a single year

Walk-forward is a weaker test than a frozen hard split (the training set grows), so a pass here is necessary
but not sufficient; it is being used to buy sample size, and that trade-off is stated rather than hidden.

Run: python3 -u -m live.dr_iter4_eqwalk
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


def crypto_sleeves():
    """Daily net series for the two crypto sleeves on the held-out window (from the deployment work)."""
    try:
        from live.dp_phase1_consolidate import sleeve_momentum
        from live.dp_phase1_review import sleeve_reversal_basis
        a = sleeve_reversal_basis(4.2, "raw")
        b = sleeve_momentum(4.2)
        return {"crypto_reversal": a, "crypto_momentum": b}
    except Exception as e:
        print(f"  (crypto sleeves unavailable: {type(e).__name__} {str(e)[:70]})", flush=True)
        return {}


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
    folds = m.make_folds(panel, train_min_days=365 * 3, test_days=365)
    print(f"panel {len(panel):,} rows | {len(feats)} feats | {len(folds)} walk-forward folds", flush=True)

    print("\n=== G1 — walk-forward, deployed 11 names vs the 15-name superset ===", flush=True)
    series = {}
    for nm, uni in (("tier_ab (11)", TIER_AB), ("full15", list(m.XYZ_IN_SP100))):
        raw = m.walk_forward(panel, feats, label, folds, top_k=m.TOP_K,
                             cost_bps_side=m.COST_BPS_SIDE, hold_days=m.HOLD_DAYS, allowed=set(uni))
        if raw.empty:
            print(f"  {nm:<16} empty", flush=True); continue
        g = m.gate_rolling(raw, regime, m.GATE_PCTILE, m.GATE_WINDOW_DAYS)
        r = m.report(nm, g, m.HOLD_DAYS)
        mm = r.get("m", {})
        lo, hi = m.bootstrap_active_sharpe_ci(g, m.HOLD_DAYS)
        print(f"  {nm:<16}n={mm.get('n_rebal', len(g)):>4}  net/d {mm.get('net_bps_per_day', np.nan):>7.2f} bps"
              f"  Sharpe {mm.get('active_sharpe_annu', np.nan):>6.2f}  CI [{lo:+.2f},{hi:+.2f}]"
              f"  {'SIG' if lo > 0 else 'spans0'}", flush=True)
        series[nm] = g

    key = "tier_ab (11)"
    if key not in series:
        print("\nno tier_ab series"); return
    g = series[key].copy()
    tcol = "ts" if "ts" in g.columns else g.columns[0]
    g[tcol] = pd.to_datetime(g[tcol], utc=True)
    ncol = [c for c in g.columns if "net" in c.lower()][0]

    print("\n=== G3 — by year (deployed 11) ===", flush=True)
    g["yr"] = g[tcol].dt.year
    pos = tot = 0
    for y, s in g.groupby("yr")[ncol]:
        if len(s) < 5:
            print(f"  {y}  n={len(s):>3}  (too few)", flush=True); continue
        sh = s.mean() / s.std() * np.sqrt(252 / m.HOLD_DAYS) if s.std() > 0 else np.nan
        tot += 1; pos += 1 if sh > 0 else 0
        print(f"  {y}  n={len(s):>3}  net/d {s.mean():>8.2f} bps  Sharpe {sh:>6.2f}", flush=True)
    print(f"  positive years (n>=5 rebalances): {pos}/{tot}", flush=True)

    print("\n=== G2 — correlation with the crypto sleeves ===", flush=True)
    eq = g.set_index(tcol)[ncol].rename("equity")
    eqd = eq.groupby(eq.index.floor("1D")).sum()
    cs = crypto_sleeves()
    if not cs:
        print("  skipped", flush=True)
    else:
        for nm, s in cs.items():
            s = s.copy(); s.index = pd.to_datetime(s.index, utc=True)
            j = pd.concat([eqd.rename("eq"), s.rename(nm)], axis=1).dropna()
            if len(j) < 30:
                print(f"  {nm:<20} only {len(j)} overlapping days", flush=True); continue
            print(f"  {nm:<20} overlap {len(j):>4} days   corr {j['eq'].corr(j[nm]):+.3f}", flush=True)
    print("\nDRITER4DONE", flush=True)


if __name__ == "__main__":
    main()
