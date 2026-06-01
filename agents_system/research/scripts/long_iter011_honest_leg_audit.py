"""LONG-PRED iter-011 — Honest leg-by-leg audit

User question: "It still does not seem like v6 is a good solution, can we check
whether long fails AND short is also not good enough?"

This script measures per-leg edge magnitudes WITHOUT spinning the result. The
question to answer:
  (a) Is the long leg's edge negligible/negative in H2 (confirming "long broken")?
  (b) Is the short leg's edge actually large enough to deploy after cost?
  (c) What's the net per-cycle expected PnL from V6's construction, with
      statistical significance?

If short edge is +X bps gross/cycle but cost+aggregation eat X-1 bps, V6 is
making +1 bps/cycle. That's near-zero alpha.

We measure with BOTH static and hl=14d preds, on the BIG OOS window, with
proper statistical bootstrap CI.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
PREDS_STATIC = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PREDS_HL14 = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
COST_PER_LEG = 4.5  # bps
N_BOOT = 2000

def bootstrap_ci(arr, fn, n=N_BOOT, ci=95):
    arr = np.asarray(arr)
    rng = np.random.default_rng(42)
    boots = np.array([fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n)])
    lo, hi = np.percentile(boots, [(100-ci)/2, 100-(100-ci)/2])
    return float(lo), float(hi)

def per_cycle_edges(preds: pd.DataFrame, allowlist=None, K=5):
    """For each cycle: compute top-K edge, bot-K edge (vs median and vs mean)."""
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[(preds["open_time"].dt.hour%4==0) & (preds["open_time"].dt.minute==0)]
    if allowlist is not None:
        allowed_set = set((r.open_time, r.symbol) for _, r in allowlist.iterrows())
        preds = preds[preds.apply(lambda r: (r["open_time"], r["symbol"]) in allowed_set, axis=1)]
    rows = []
    for ot, g in preds.groupby("open_time"):
        if len(g) < 2*K: continue
        g = g.sort_values("pred")
        u_mean = g["return_pct"].mean(); u_median = g["return_pct"].median()
        top = g.tail(K)["return_pct"].mean()
        bot = g.head(K)["return_pct"].mean()
        # Per-leg PnL contribution for a 5L/5S basket:
        #   long leg: +1.0 weight equally in K names -> realized = top
        #   short leg: -1.0 weight equally in K names -> realized = -bot
        #   spread per cycle = top - bot
        rows.append(dict(open_time=ot, n=len(g),
                         top_vs_mean=top - u_mean,
                         top_vs_median=top - u_median,
                         bot_vs_mean=u_mean - bot,
                         bot_vs_median=u_median - bot,
                         top_abs=top, bot_abs=bot,
                         spread=top - bot,
                         u_mean=u_mean, u_median=u_median))
    return pd.DataFrame(rows)

def report(df, label, K=5):
    if len(df)==0:
        print(f"  {label}: (no data)"); return
    print(f"\n  {label} (n_cycles={len(df)}):")
    for metric in ["top_vs_mean","top_vs_median","bot_vs_mean","bot_vs_median","spread"]:
        v = df[metric].values * 1e4
        mean_bps = v.mean()
        lo, hi = bootstrap_ci(v, np.mean)
        # one-side t-test: is mean > 0?
        t = mean_bps / (v.std()/np.sqrt(len(v))) if v.std()>0 else float("nan")
        sig = "★" if abs(t) > 1.96 else " "
        print(f"    {metric:<18} {mean_bps:+7.2f} bps  CI95 [{lo:+6.2f}, {hi:+6.2f}]  t={t:+5.2f} {sig}")
    print(f"\n    REAL alpha cost-comparison (NET = gross_edge − round-trip cost):")
    cost_RT = 2 * COST_PER_LEG  # round-trip per position (open+close)
    long_net = df["top_vs_median"].mean()*1e4 - cost_RT/K  # per-cycle cost spread over K positions
    short_net = df["bot_vs_median"].mean()*1e4 - cost_RT/K
    print(f"    Long leg net (top_vs_median − cost/K): {long_net:+.2f} bps/cycle ({'POSITIVE' if long_net>0 else 'NEGATIVE'})")
    print(f"    Short leg net (bot_vs_median − cost/K): {short_net:+.2f} bps/cycle ({'POSITIVE' if short_net>0 else 'NEGATIVE'})")

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-011: Honest leg-by-leg audit ===\n", flush=True)
    print("Question: is V6's short leg actually deploying meaningful alpha, or marginal?\n")

    # Load preds
    static = pd.read_parquet(PREDS_STATIC, columns=["symbol","open_time","pred","return_pct"])
    hl14 = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred","return_pct"])

    allowlist = pd.read_parquet(ALLOWLIST) if ALLOWLIST.exists() else None
    if allowlist is not None:
        allowlist["open_time"] = pd.to_datetime(allowlist["open_time"], utc=True)

    for preds_label, preds_df in [("STATIC preds (original)", static), ("hl=14d preds (V6 model)", hl14)]:
        print(f"\n{'='*70}")
        print(f"=== {preds_label} ===")
        print(f"{'='*70}")

        edges_all = per_cycle_edges(preds_df, allowlist=None)
        edges_all["open_time"] = pd.to_datetime(edges_all["open_time"], utc=True)
        h1 = edges_all[edges_all["open_time"] < H2_START]
        h2 = edges_all[edges_all["open_time"] >= H2_START]

        print(f"\n[FULL UNIVERSE, K=5]")
        report(h1, "H1 (Oct→Jan22)")
        report(h2, "H2 (Jan22→May26)")

        # K=3 (V6's actual K for shorts) on full universe
        edges_k3 = per_cycle_edges(preds_df, allowlist=None, K=3)
        h2_k3 = edges_k3[edges_k3["open_time"] >= H2_START]
        print(f"\n[FULL UNIVERSE, K=3 (V6's short basket size)]")
        report(h2_k3, "H2 K=3")

        # filtered universe (V6's actual universe)
        if allowlist is not None:
            edges_filt = per_cycle_edges(preds_df, allowlist=allowlist, K=3)
            edges_filt["open_time"] = pd.to_datetime(edges_filt["open_time"], utc=True)
            h2_filt = edges_filt[edges_filt["open_time"] >= H2_START]
            h1_filt = edges_filt[edges_filt["open_time"] < H2_START]
            print(f"\n[FILTERED UNIVERSE (W180 τ0.02), K=3 — V6's actual selection space]")
            report(h1_filt, "H1 filtered")
            report(h2_filt, "H2 filtered")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"=== FINAL HONEST VERDICT ===")
    print(f"{'='*70}")
    edges_v6_h2 = per_cycle_edges(hl14, allowlist=allowlist, K=3)
    edges_v6_h2["open_time"] = pd.to_datetime(edges_v6_h2["open_time"], utc=True)
    edges_v6_h2 = edges_v6_h2[edges_v6_h2["open_time"] >= H2_START]
    bot_edge = edges_v6_h2["bot_vs_median"].mean() * 1e4
    top_edge = edges_v6_h2["top_vs_median"].mean() * 1e4
    print(f"\nV6's H2 reality check (hl=14d preds, filtered universe, K=3):")
    print(f"  Long  top-K=3 vs median: {top_edge:+.2f} bps/cycle")
    print(f"  Short bot-K=3 vs median: {bot_edge:+.2f} bps/cycle  ← V6 captures only this")
    print(f"  After 6-sleeve aggregation loss (~7 bps from iter-003) + cost: real per-cycle ~{bot_edge - 7 - 3:+.2f} bps")
    if bot_edge > 12:
        print(f"  ✓ Short edge MATERIALLY positive (>{12} bps) — V6 has real alpha")
    elif bot_edge > 5:
        print(f"  ≈ Short edge MARGINAL ({bot_edge:.1f} bps) — alpha barely covers construction losses")
    else:
        print(f"  ✗ Short edge TOO SMALL — V6 is mostly 'trade less' not 'real alpha'")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
