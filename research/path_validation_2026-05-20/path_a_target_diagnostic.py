"""Path (a) target_A clip diagnostic.

The 111-panel target_A is clipped to exactly ±5 (100% within); the 51-panel
target_A is unclipped (min −127, max +164; 0.4% of rows would exceed ±5).
Before committing to a 2–10h retrain of 111-panel without the clip, measure:

  1. Per-symbol: what % of target_A rows would clip at ±5?
  2. What percentile is ±5 per symbol (e.g., 99th vs 95th)?
  3. Per-symbol: skew & kurtosis of target_A (heavy tails?)
  4. Which symbols are the clip-heavy ones? Are they:
     - large-cap stable names (clip would be wrong) or
     - meme/new-listing rotation names (clip might be defensible) or
     - small-cap thin-liquidity names (clip masks real signal)

If clip-heavy symbols are large-cap stable: clip is a hack that breaks ranking.
If clip-heavy symbols are meme/new: clip is regularization that may or may
not help; either way the FIX should be per-symbol winsorization at a fixed
percentile (e.g., 99.5%), not a fixed value (±5) that doesn't respect each
symbol's natural target_A scale.

Output: per-symbol clip rate, p99/p99.5/p995/p999 percentiles, skew/kurt,
and a recommendation for which preprocessing to use in (a)'s retrain.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/path_validation_2026-05-20"; OUT.mkdir(parents=True, exist_ok=True)
PANEL_51 = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"


def main():
    t0 = time.time()
    cols = ["symbol", "open_time", "target_A"]
    p = pd.read_parquet(PANEL_51, columns=cols)
    p = p.dropna(subset=["target_A"])
    print(f"panel rows {len(p):,}, symbols {p['symbol'].nunique()}", flush=True)

    overall = {
        "n_rows": int(len(p)),
        "min": round(float(p["target_A"].min()), 2),
        "max": round(float(p["target_A"].max()), 2),
        "p001":  round(float(p["target_A"].quantile(0.001)), 3),
        "p01":   round(float(p["target_A"].quantile(0.01)), 3),
        "p05":   round(float(p["target_A"].quantile(0.05)), 3),
        "p50":   round(float(p["target_A"].quantile(0.50)), 3),
        "p95":   round(float(p["target_A"].quantile(0.95)), 3),
        "p99":   round(float(p["target_A"].quantile(0.99)), 3),
        "p995":  round(float(p["target_A"].quantile(0.995)), 3),
        "p999":  round(float(p["target_A"].quantile(0.999)), 3),
        "frac_lt_-5": round(float((p["target_A"] < -5).mean()), 5),
        "frac_gt_+5": round(float((p["target_A"] >  5).mean()), 5),
        "frac_abs_gt_5": round(float((p["target_A"].abs() > 5).mean()), 5),
        "skew": round(float(stats.skew(p["target_A"], nan_policy="omit")), 3),
        "kurt": round(float(stats.kurtosis(p["target_A"], nan_policy="omit")), 3),
    }

    # per-symbol
    per_sym = []
    for sym, g in p.groupby("symbol"):
        n = len(g); t = g["target_A"]
        if n < 100: continue
        clip_lo = float((t < -5).mean())
        clip_hi = float((t > 5).mean())
        per_sym.append({
            "symbol": sym, "n": n,
            "clip_lo_frac": round(clip_lo, 5),
            "clip_hi_frac": round(clip_hi, 5),
            "clip_total_frac": round(clip_lo + clip_hi, 5),
            "p99":   round(float(t.quantile(0.99)), 3),
            "p995":  round(float(t.quantile(0.995)), 3),
            "p999":  round(float(t.quantile(0.999)), 3),
            "p01":   round(float(t.quantile(0.01)), 3),
            "p005":  round(float(t.quantile(0.005)), 3),
            "p001":  round(float(t.quantile(0.001)), 3),
            "min": round(float(t.min()), 2),
            "max": round(float(t.max()), 2),
            "skew": round(float(stats.skew(t, nan_policy="omit")), 3),
            "kurt": round(float(stats.kurtosis(t, nan_policy="omit")), 3),
        })
    per_sym = pd.DataFrame(per_sym).sort_values("clip_total_frac", ascending=False)

    top_clip = per_sym.head(15).to_dict("records")
    no_clip = per_sym[per_sym["clip_total_frac"] == 0]
    n_no_clip = len(no_clip)
    n_clip_any = (per_sym["clip_total_frac"] > 0).sum()
    n_clip_heavy = (per_sym["clip_total_frac"] > 0.005).sum()  # >0.5% rows

    # Symbols whose ±5 is below the 99% mark (suggesting clip would mask
    # meaningful tail observations, not just extreme outliers).
    inside99 = per_sym[(per_sym["p99"] > 5) | (per_sym["p01"] < -5)]

    out = {
        "scope": "51-panel target_A clip diagnostic",
        "overall": overall,
        "n_symbols_total": len(per_sym),
        "n_symbols_no_clip_at_pm5": int(n_no_clip),
        "n_symbols_any_clip_at_pm5": int(n_clip_any),
        "n_symbols_heavy_clip_>0.5pct": int(n_clip_heavy),
        "n_symbols_with_p99_or_p01_beyond_5": int(len(inside99)),
        "top_15_clip_heavy_symbols": top_clip,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "path_a_target_diagnostic.json").write_text(json.dumps(out, indent=2, default=str))

    # printout
    print(f"\n=== overall target_A distribution ===")
    for k, v in overall.items(): print(f"  {k:<22} {v}")
    print(f"\n=== per-symbol summary ===")
    print(f"  n symbols: {len(per_sym)}")
    print(f"  no clip at ±5:           {n_no_clip}")
    print(f"  any clip at ±5:          {n_clip_any}")
    print(f"  >0.5% clip (heavy):      {n_clip_heavy}")
    print(f"  p99 or p01 beyond ±5:    {len(inside99)} (clip masks real tail)")
    print(f"\n=== top 15 clip-heavy symbols ===")
    print(f"  {'sym':<14} {'n':>7} {'clip%':>7} {'p99':>7} {'p001':>7} {'p999':>7} {'min':>9} {'max':>9} {'skew':>6} {'kurt':>7}")
    for r in top_clip:
        print(f"  {r['symbol']:<14} {r['n']:>7} {r['clip_total_frac']*100:>6.2f}  "
              f"{r['p99']:>+7.2f} {r['p001']:>+7.2f} {r['p999']:>+7.2f} "
              f"{r['min']:>+9.1f} {r['max']:>+9.1f} "
              f"{r['skew']:>+6.2f} {r['kurt']:>+7.1f}")
    print("\nDIAG_DONE")


if __name__ == "__main__":
    main()
