"""Phase I: regime-conditional residual IC audit on WINNER_21 redundancy pairs.

For each (dropped, kept) feature pair from H1 redundancy clusters:
  1. Residualize the dropped feature against the kept feature (linear regression
     of dropped ~ kept, take residuals).
  2. Compute Spearman IC of residual vs alpha_A in 4 pre-registered regimes.
  3. Flag cells with |residual IC| ≥ 0.04 as candidates for regime-conditional use.

Pairs tested:
  atr_pct                  vs idio_vol_to_btc_1h          (volatility cluster)
  dom_change_288b_vs_bk    vs return_1d                   (return-magnitude cluster)
  corr_to_btc_1d           vs idio_vol_1d_vs_bk_xs_rank   (cross-asset corr cluster)
  mfi                      vs obv_z_1d                    (volume cluster)
  price_volume_corr_20     vs obv_z_1d                    (volume cluster)

Regimes (each defined by per-bar quantile of a panel feature):
  all_obs       — full sample (baseline)
  high_disp     — xs_alpha_dispersion_48b ≥ 70th percentile (conv_gate active)
  btc_down      — btc_ret_288b ≤ 25th percentile
  funding_stress — |funding_rate| ≥ 75th percentile
  high_vol      — idio_vol_1d_vs_bk_xs_rank ≥ 75th percentile

Multi-test discipline: 5 pairs × 5 regimes = 25 cells. Pre-register threshold
|IC| ≥ 0.04 as practically meaningful (well above noise floor at n>100K).

Output:
  outputs/vBTC_regime_residual/regime_residual_ic.csv
  outputs/vBTC_regime_residual/regime_obs_counts.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
OUT = REPO / "outputs/vBTC_regime_residual"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("atr_pct", "idio_vol_to_btc_1h", "volatility"),
    ("dom_change_288b_vs_bk", "return_1d", "return-magnitude"),
    ("corr_to_btc_1d", "idio_vol_1d_vs_bk_xs_rank", "cross-asset-corr"),
    ("mfi", "obv_z_1d", "volume_a"),
    ("price_volume_corr_20", "obv_z_1d", "volume_b"),
]

# threshold for "practical" residual IC; per CLAUDE.md, +0.10 is suspicious-of-leakage
# so we use 0.04 — clearly above noise but plausibly informative
IC_THRESH = 0.04


def regime_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Build boolean per-bar regime masks."""
    masks = {}
    # all observations
    masks["all_obs"] = pd.Series(True, index=panel.index)

    if "xs_alpha_dispersion_48b" in panel.columns:
        q = panel["xs_alpha_dispersion_48b"].quantile(0.70)
        masks["high_disp"] = panel["xs_alpha_dispersion_48b"] >= q
    if "btc_ret_288b" in panel.columns:
        q = panel["btc_ret_288b"].quantile(0.25)
        masks["btc_down"] = panel["btc_ret_288b"] <= q
    if "funding_rate" in panel.columns:
        q = panel["funding_rate"].abs().quantile(0.75)
        masks["funding_stress"] = panel["funding_rate"].abs() >= q
    if "idio_vol_1d_vs_bk_xs_rank" in panel.columns:
        q = panel["idio_vol_1d_vs_bk_xs_rank"].quantile(0.75)
        masks["high_vol"] = panel["idio_vol_1d_vs_bk_xs_rank"] >= q
    return masks


def main():
    print("=== Phase I: regime-conditional residual IC audit ===\n", flush=True)

    panel = pd.read_parquet(PANEL)
    print(f"  panel: {len(panel):,} rows, {panel.symbol.nunique()} syms", flush=True)

    masks = regime_masks(panel)
    print(f"\n  Regime counts (% of total rows):")
    obs_counts = []
    for r, m in masks.items():
        pct = m.sum() / len(panel) * 100
        print(f"    {r:<18}  {m.sum():>10,}  ({pct:>5.1f}%)", flush=True)
        obs_counts.append({"regime": r, "n_obs": int(m.sum()), "pct": pct})
    pd.DataFrame(obs_counts).to_csv(OUT / "regime_obs_counts.csv", index=False)

    print(f"\n--- Residual IC by regime ---", flush=True)
    print(f"  pair (dropped | kept)               | regime              "
          f"| n_obs    |   raw_IC | resid_IC | r_dropped→kept | flag", flush=True)

    rows = []
    for dropped, kept, cluster in PAIRS:
        for r_name, mask in masks.items():
            sub = panel.loc[mask, [dropped, kept, "alpha_A"]].dropna()
            n = len(sub)
            if n < 10_000:
                continue
            y = sub[dropped].values.astype(float)
            X = sub[[kept]].values.astype(float)
            reg = LinearRegression().fit(X, y)
            resid = y - reg.predict(X)
            r2_pair = sub[dropped].corr(sub[kept])  # pairwise Pearson r
            raw_ic = sub[dropped].rank().corr(sub["alpha_A"].rank())
            resid_ic = pd.Series(resid).rank().corr(sub["alpha_A"].rank())
            flag = ""
            if abs(resid_ic) >= IC_THRESH:
                flag = "  CANDIDATE"
            print(f"  {dropped:<24} | {kept:<22} | {r_name:<18} "
                  f"| {n:>8,} | {raw_ic:>+8.4f} | {resid_ic:>+8.4f} | "
                  f"{r2_pair:>+12.3f}    | {flag}", flush=True)
            rows.append({
                "dropped": dropped, "kept": kept, "cluster": cluster,
                "regime": r_name, "n_obs": n,
                "raw_IC": raw_ic, "residual_IC": resid_ic,
                "pearson_r_dropped_kept": r2_pair, "is_candidate": abs(resid_ic) >= IC_THRESH,
            })
        print()  # blank line between pairs

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "regime_residual_ic.csv", index=False)

    print(f"\n--- Candidates (|residual IC| ≥ {IC_THRESH}) ---", flush=True)
    cands = df[df["is_candidate"]]
    if len(cands) == 0:
        print(f"  NONE — no feature×regime cell exceeds threshold.", flush=True)
        print(f"  Verdict: redundant features remain non-informative at regime level.", flush=True)
    else:
        # Multi-test correction: Bonferroni at α=0.05, n_tests = 5×5 = 25 → α_corrected = 0.002
        # For Spearman IC at n=N, |IC| > 1.96/sqrt(N) is significant uncorrected.
        # Bonferroni: |IC| > 3.06/sqrt(N) (≈ z=3.06 for two-sided p=0.002).
        for _, row in cands.iterrows():
            n = row["n_obs"]
            bonf_threshold = 3.06 / np.sqrt(n)
            sig = "SIG_BONF" if abs(row["residual_IC"]) > bonf_threshold else "weak"
            print(f"  {row['dropped']:<24} in {row['regime']:<18} "
                  f"residual_IC={row['residual_IC']:+.4f}  "
                  f"(Bonferroni threshold={bonf_threshold:.4f})  → {sig}",
                  flush=True)

    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
