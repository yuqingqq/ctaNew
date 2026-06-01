"""Probe #9 — random-rule placebo for Probe #8's Mode B rule.

The red-team review identified that Probe #8's rule (idio_vol_to_btc_1d > +2σ
AND corr_to_btc_1d < -1.5σ AND funding_rate < -1.5σ AND return_1d > 0) was
derived from VVV's individual signature (the single biggest contributor),
NOT from the aggregate Probe #7 signature — and 3 of 4 conditions are the
OPPOSITE direction of the aggregate signature.

This raises the multiple-testing concern: with 22 features in Probe #7's
signature panel and ~5 reasonable threshold values × 2 directions, the
implicit search space is huge. A +4.16 Sharpe might be the top of that
grid, not a real mechanism.

This probe runs the random-rule placebo:
  1. Use the 22 SIG_FEATS from Probe #7.
  2. Sample 500 random 4-feature combinations.
  3. For each: sample random thresholds (independent Normal(0, 1.5)) and
     directional inequalities (uniform ±).
  4. Apply the random rule on the 51-panel (same as Probe #8).
  5. Compute net Sharpe for each random rule. Filter to rules with at least
     200 fires (same scale as the real rule's 893) to be apples-to-apples
     on sample size.
  6. Report the distribution: where does the REAL rule's +4.16 rank?

Pre-committed pass criterion:
  Real Sharpe ≥ p99 of random-rule distribution → mechanism has real signal
  Real Sharpe < p95 → result is grid-search noise; REJECT
  Real Sharpe in [p95, p99) → marginal; needs further investigation
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"; OUT.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"

SIG_FEATS = [
    "return_1d", "atr_pct", "idio_vol_to_btc_1d", "idio_vol_to_btc_1h",
    "corr_to_btc_1d", "beta_to_btc_change_5d",
    "idio_vol_1d_vs_bk", "name_idio_share_1d",
    "dom_level_vs_bk", "dom_change_288b_vs_bk",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
    "funding_streak_pos", "obv_z_1d", "vwap_slope_96",
    "mfi", "aggr_ratio_4h", "bars_since_high_xs_rank",
    "idio_skew_1d", "idio_kurt_1d", "idio_max_abs_12b",
]

COST_RT_BPS = 9.0
N_RANDOM_RULES = 500
MIN_FIRES = 200
SEED = 20260520


def sharpe(r):
    r = np.asarray(r, dtype=float); r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(288 * 365 / 48))


def main():
    t0 = time.time()
    print("=== Probe #9 random-rule placebo ===", flush=True)
    cols_avail = pd.read_parquet(PANEL, columns=None).head(1).columns
    feats_present = [c for c in SIG_FEATS if c in cols_avail]
    print(f"  features present: {len(feats_present)}/{len(SIG_FEATS)}", flush=True)

    p = pd.read_parquet(PANEL,
        columns=["symbol", "open_time", "alpha_vs_btc_realized"] + feats_present)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[(p["open_time"].dt.minute == 0) & (p["open_time"].dt.hour % 4 == 0)]
    p = p.dropna(subset=["alpha_vs_btc_realized"]).reset_index(drop=True)
    print(f"  panel: {len(p):,} rows", flush=True)

    # xs-z per cycle for all features
    print("  computing xs-z per cycle...", flush=True)
    for c in feats_present:
        p[c + "_xsz"] = p.groupby("open_time")[c].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))
    p["target_bps"] = p["alpha_vs_btc_realized"] * 10000.0
    print(f"  xs-z done [{time.time()-t0:.0f}s]", flush=True)

    # the REAL rule
    real_fires = ((p["idio_vol_to_btc_1d_xsz"] > 2.0) &
                  (p["corr_to_btc_1d_xsz"] < -1.5) &
                  (p["funding_rate_xsz"] < -1.5) &
                  (p["return_1d_xsz"] > 0)).fillna(False)
    real_returns = (p["target_bps"] - COST_RT_BPS)[real_fires].dropna()
    real_sharpe = sharpe(real_returns)
    real_n = int(real_fires.sum())
    print(f"  REAL rule: n_fires={real_n}, net_mean={real_returns.mean():+.1f} bps, "
          f"Sharpe={real_sharpe:.3f}", flush=True)

    # random rules
    rng = np.random.RandomState(SEED)
    placebo_results = []
    print(f"  running {N_RANDOM_RULES} random rules...", flush=True)
    for r in range(N_RANDOM_RULES):
        # sample 4 features without replacement
        feat_idx = rng.choice(len(feats_present), size=4, replace=False)
        feats = [feats_present[i] for i in feat_idx]
        # sample thresholds (normal, scaled to match Probe #8's thresholds in magnitude)
        thresholds = rng.normal(loc=0, scale=1.5, size=4)
        # sample directions
        directions = rng.choice(['gt', 'lt'], size=4)
        # build mask
        mask = pd.Series(True, index=p.index)
        for f, t, d in zip(feats, thresholds, directions):
            col = p[f + "_xsz"]
            if d == 'gt':
                mask = mask & (col > t)
            else:
                mask = mask & (col < t)
        mask = mask.fillna(False)
        n_fires = int(mask.sum())
        if n_fires < MIN_FIRES:
            continue
        # also random direction for the trade itself (long or short)
        trade_sign = rng.choice([+1, -1])
        rets = (trade_sign * p["target_bps"] - COST_RT_BPS)[mask].dropna()
        s = sharpe(rets)
        placebo_results.append({
            "rule_id": r,
            "feats": feats,
            "thresholds": [round(float(t), 2) for t in thresholds],
            "directions": directions.tolist(),
            "trade_sign": int(trade_sign),
            "n_fires": n_fires,
            "net_mean_bps": round(float(rets.mean()), 1),
            "sharpe": round(s, 3),
        })
        if (r + 1) % 50 == 0:
            sharpes = [x["sharpe"] for x in placebo_results]
            print(f"  ... {r+1}/{N_RANDOM_RULES} ({len(placebo_results)} kept), "
                  f"sharpe p50={np.percentile(sharpes, 50):.2f} "
                  f"p95={np.percentile(sharpes, 95):.2f} "
                  f"max={max(sharpes):.2f}", flush=True)

    sharpes = np.array([x["sharpe"] for x in placebo_results])
    if len(sharpes) == 0:
        print("  no random rules met MIN_FIRES floor!", flush=True)
        return

    # where does real rank
    n_geq_real = int((sharpes >= real_sharpe).sum())
    n_geq_real_abs = int((np.abs(sharpes) >= abs(real_sharpe)).sum())
    pct_below = float((sharpes < real_sharpe).mean())

    distribution = {
        "n_random_rules_kept": int(len(sharpes)),
        "real_sharpe": round(real_sharpe, 3),
        "real_n_fires": real_n,
        "placebo_p05": round(float(np.percentile(sharpes, 5)), 3),
        "placebo_p25": round(float(np.percentile(sharpes, 25)), 3),
        "placebo_p50": round(float(np.percentile(sharpes, 50)), 3),
        "placebo_p75": round(float(np.percentile(sharpes, 75)), 3),
        "placebo_p90": round(float(np.percentile(sharpes, 90)), 3),
        "placebo_p95": round(float(np.percentile(sharpes, 95)), 3),
        "placebo_p99": round(float(np.percentile(sharpes, 99)), 3),
        "placebo_max": round(float(sharpes.max()), 3),
        "placebo_min": round(float(sharpes.min()), 3),
        "placebo_mean": round(float(sharpes.mean()), 3),
        "real_rank_pct": round(pct_below * 100, 2),
        "n_random_geq_real": n_geq_real,
        "n_random_geq_real_abs_value": n_geq_real_abs,
    }

    if real_sharpe >= distribution["placebo_p99"]:
        verdict = f"PASS — real Sharpe at p{distribution['real_rank_pct']:.1f} ≥ p99 of random rules ({distribution['placebo_p99']})"
    elif real_sharpe >= distribution["placebo_p95"]:
        verdict = f"MARGINAL — real Sharpe at p{distribution['real_rank_pct']:.1f}, between p95 and p99"
    else:
        verdict = f"REJECT — real Sharpe at p{distribution['real_rank_pct']:.1f} below p95; grid-search noise"

    # top 5 random rules (most positive)
    top_rules = sorted(placebo_results, key=lambda x: -x["sharpe"])[:5]
    bottom_rules = sorted(placebo_results, key=lambda x: x["sharpe"])[:5]

    out = {
        "real_rule": {
            "feats": ["idio_vol_to_btc_1d", "corr_to_btc_1d", "funding_rate", "return_1d"],
            "thresholds": [+2.0, -1.5, -1.5, 0.0],
            "directions": ["gt", "lt", "lt", "gt"],
            "trade_sign": +1,
            "sharpe": round(real_sharpe, 3),
            "n_fires": real_n,
        },
        "placebo_distribution": distribution,
        "verdict": verdict,
        "top_5_random_rules_by_sharpe": top_rules,
        "bottom_5_random_rules_by_sharpe": bottom_rules,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "probe9_random_rule_placebo.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"\n=== PLACEBO DISTRIBUTION ({len(sharpes)} valid random rules) ===", flush=True)
    for k in ["placebo_p05", "placebo_p25", "placebo_p50", "placebo_p75",
              "placebo_p90", "placebo_p95", "placebo_p99", "placebo_max", "placebo_mean"]:
        print(f"  {k:<20} {distribution[k]}", flush=True)
    print(f"\n  REAL Sharpe: {real_sharpe:.3f}", flush=True)
    print(f"  REAL rank: p{distribution['real_rank_pct']:.1f}", flush=True)
    print(f"  n_random ≥ real: {n_geq_real}/{len(sharpes)}", flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"\nTop 3 random rules:", flush=True)
    for r in top_rules[:3]:
        print(f"  Sharpe={r['sharpe']:+.3f}, n={r['n_fires']}, feats={r['feats']}, "
              f"thr={r['thresholds']}, dir={r['directions']}, trade_sign={r['trade_sign']}", flush=True)
    print(f"\n[elapsed {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
