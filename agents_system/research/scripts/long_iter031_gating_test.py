"""LONG-PRED iter-031 — Gating mechanism test.

Replace fixed K=5 with conviction-based gating: |pred| > T_threshold.

Test variants:
  V0: baseline K=5/K=5 (fixed)
  V1: Gate T=0.5, variable basket sizes
  V2: Gate T=1.0
  V3: Gate T=1.5
  V4: Gate T=2.0 (strict)
  V5: Asymmetric — strict gate on shorts, loose on longs (T_short=1.5, T_long=0.5)

For each variant:
  - Track basket sizes per cycle (avg longs, avg shorts, % cycles with no shorts, etc.)
  - Compute total PnL per cycle (long_picks_ret - short_picks_ret) / |basket|
  - Compute selection alpha (vs market median)
  - Add BTC hedge variant for net-long cycles

Output: which gate produces the highest selection alpha at acceptable trade frequency.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K_BASELINE = 5

def evaluate(df_period, label, T_long=None, T_short=None, K=None, hedge=None):
    """Compute per-cycle returns under given selection rule.

    T_long/T_short = z-score thresholds on pred (if None, use K)
    hedge = None | 'market_basket' (offset net long with basket short)
    """
    rows = []
    for ot, gc in df_period.groupby("open_time"):
        if len(gc) < 10: continue
        market_med = gc["return_pct"].median()
        # Select longs and shorts
        if T_long is not None:
            longs = gc[gc["pred"] > T_long]
            shorts = gc[gc["pred"] < -T_short]
        else:
            longs = gc.nlargest(K, "pred")
            shorts = gc.nsmallest(K, "pred")
        n_long, n_short = len(longs), len(shorts)
        # PnL calculation
        long_ret = longs["return_pct"].mean() if n_long else 0
        short_ret = shorts["return_pct"].mean() if n_short else 0
        # Equal $-weight: each position is 1/max(n,1) of side allocation
        # For comparison purposes, just track average return per side
        # Dollar-neutral basket: long $1, short $1 → PnL = long_ret - short_ret
        if n_long == 0 and n_short == 0:
            pnl = 0
        elif n_long > 0 and n_short > 0:
            pnl = long_ret - short_ret  # equal $ both sides
        elif n_long > 0 and n_short == 0:
            if hedge == "market_basket":
                pnl = long_ret - market_med  # offset with basket short
            else:
                pnl = long_ret  # net long, no hedge
        elif n_short > 0 and n_long == 0:
            pnl = -short_ret
        # Selection alphas (vs market median basket)
        long_alpha = (long_ret - market_med) if n_long else None
        short_alpha = (market_med - short_ret) if n_short else None
        rows.append({
            "open_time": ot, "n_long": n_long, "n_short": n_short,
            "pnl": pnl, "long_alpha": long_alpha, "short_alpha": short_alpha,
            "market_med": market_med,
        })
    return pd.DataFrame(rows)

def summarize(res, label):
    """Print summary stats."""
    pnl_bps = res["pnl"].values * 1e4
    n = len(res)
    long_alphas = res["long_alpha"].dropna().values * 1e4
    short_alphas = res["short_alpha"].dropna().values * 1e4
    pct_no_long = (res["n_long"]==0).mean()*100
    pct_no_short = (res["n_short"]==0).mean()*100
    avg_long_size = res[res["n_long"]>0]["n_long"].mean()
    avg_short_size = res[res["n_short"]>0]["n_short"].mean()
    return {
        "label": label,
        "n_cycles": n,
        "pnl_mean": pnl_bps.mean(),
        "pnl_t": pnl_bps.mean() / (pnl_bps.std()/np.sqrt(n)) if pnl_bps.std() > 0 else 0,
        "long_alpha_mean": long_alphas.mean() if len(long_alphas) else np.nan,
        "long_alpha_t": long_alphas.mean()/(long_alphas.std()/np.sqrt(len(long_alphas))) if len(long_alphas) and long_alphas.std()>0 else np.nan,
        "short_alpha_mean": short_alphas.mean() if len(short_alphas) else np.nan,
        "short_alpha_t": short_alphas.mean()/(short_alphas.std()/np.sqrt(len(short_alphas))) if len(short_alphas) and short_alphas.std()>0 else np.nan,
        "pct_no_long": pct_no_long, "pct_no_short": pct_no_short,
        "avg_long_size": avg_long_size, "avg_short_size": avg_short_size,
    }

def main():
    t0 = time.time()
    print("=== iter-031: Gating mechanism test ===\n", flush=True)

    print("loading data...", flush=True)
    preds = pd.read_parquet(PREDS)
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    print(f"  preds: {len(preds):,} rows", flush=True)
    print(f"  pred dist: mean={preds['pred'].mean():.4f} std={preds['pred'].std():.4f}", flush=True)
    print(f"  pred p99={preds['pred'].quantile(0.99):.4f} p01={preds['pred'].quantile(0.01):.4f}", flush=True)
    print(f"  pred p95={preds['pred'].quantile(0.95):.4f} p05={preds['pred'].quantile(0.05):.4f}", flush=True)

    # Z-score predictions per cycle (cross-sectionally) for gating
    # Since preds are not on uniform scale, use rank-normalized predictions
    print("\nbuilding cycle-z-scored predictions for gating...", flush=True)
    preds["pred_z"] = preds.groupby("open_time")["pred"].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)

    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        print(f"\n\n{'='*70}\n=== {period_label}: {s.date()} → {e.date()} ===\n{'='*70}", flush=True)
        sub = preds[(preds["open_time"]>=s) & (preds["open_time"]<e)].copy()
        # Replace pred with pred_z for gating
        sub["pred"] = sub["pred_z"]

        all_r = []
        # Baseline K=5
        res = evaluate(sub, "V0_K5_K5", K=K_BASELINE)
        all_r.append(summarize(res, "V0 K=5/K=5 (baseline)"))
        # Gates
        for T in [0.5, 1.0, 1.5, 2.0]:
            res = evaluate(sub, f"V_T{T}", T_long=T, T_short=T)
            all_r.append(summarize(res, f"V T={T} symmetric"))
            # Plus market-basket hedge for net-long cycles
            res = evaluate(sub, f"V_T{T}_hedge", T_long=T, T_short=T, hedge="market_basket")
            all_r.append(summarize(res, f"V T={T} + basket hedge"))
        # Asymmetric: strict short, loose long
        res = evaluate(sub, "V_asym", T_long=0.5, T_short=1.5)
        all_r.append(summarize(res, "V asymmetric (T_l=0.5, T_s=1.5)"))
        res = evaluate(sub, "V_asym2", T_long=0.5, T_short=2.0)
        all_r.append(summarize(res, "V asymmetric (T_l=0.5, T_s=2.0)"))

        rdf = pd.DataFrame(all_r)
        print(f"\n{'variant':<34} {'avg #L':>7} {'avg #S':>7} {'%noS':>5} {'PnL bps':>9} {'t':>5} {'L_alpha':>8} {'S_alpha':>8}")
        print("-"*120)
        for _, r in rdf.iterrows():
            sig = "★" if abs(r["pnl_t"])>1.96 else " "
            l_sig = "★" if not np.isnan(r["long_alpha_t"]) and abs(r["long_alpha_t"])>1.96 else " "
            s_sig = "★" if not np.isnan(r["short_alpha_t"]) and abs(r["short_alpha_t"])>1.96 else " "
            print(f"  {r['label']:<32} {r['avg_long_size']:>6.1f}  {r['avg_short_size']:>6.1f}  {r['pct_no_short']:>4.0f}%  "
                  f"{r['pnl_mean']:>+7.2f}{sig} {r['pnl_t']:>+4.2f} "
                  f"{r['long_alpha_mean']:>+6.2f}{l_sig} {r['short_alpha_mean']:>+6.2f}{s_sig}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
