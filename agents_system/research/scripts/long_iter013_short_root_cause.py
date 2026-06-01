"""LONG-PRED iter-013 — Why doesn't short work consistently?

HYPOTHESIS: V0 model is a mean-reversion learner. It picks:
  - Top-K: recently-weak names → expects bounce up (works in crypto: buy-the-dip)
  - Bot-K: recently-strong names → expects revert down (FAILS: crypto momentum persists)

Diagnostic:
  (a) For each cycle, measure TRAILING characteristics of top-K and bot-K
      (e.g., trailing 1d/3d/7d return, trailing vol, recent momentum)
  (b) Show the model's selection pattern: does it pick top-K = recent losers?
                                          does it pick bot-K = recent winners?
  (c) Conditional analysis: short bot-K performance conditional on its trailing momentum:
      - "Genuine weak" (negative trailing momentum) — does shorting work?
      - "Just had a rally" (positive trailing momentum) — does shorting fail more?

If hypothesis confirmed: propose anti-momentum filter for shorts.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS_HL14 = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-013: Short signal root cause ===\n", flush=True)

    # Load preds + key trailing features
    preds = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred","return_pct"])
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[(preds["open_time"].dt.hour%4==0) & (preds["open_time"].dt.minute==0)]
    panel_feats = pd.read_parquet(PANEL, columns=["symbol","open_time","return_1d","ret_3d","rvol_7d","atr_pct"])
    panel_feats["open_time"] = pd.to_datetime(panel_feats["open_time"], utc=True)
    panel_feats = panel_feats[(panel_feats["open_time"].dt.hour%4==0) & (panel_feats["open_time"].dt.minute==0)]
    df = preds.merge(panel_feats, on=["symbol","open_time"], how="left")
    print(f"loaded {len(df):,} rows × {df['symbol'].nunique()} syms", flush=True)

    # (a) per-cycle: what TRAILING features do top-K and bot-K have?
    print("\n=== (a) Trailing characteristics of top-K vs bot-K by pred ===")
    print("(higher trailing return = recently strong; lower = recently weak)\n")

    for label,(s,e) in [("H1",(H1_START,H2_START)),("H2",(H2_START,H2_END))]:
        sub = df[(df["open_time"]>=s)&(df["open_time"]<e)].dropna(subset=["pred","return_1d","ret_3d","rvol_7d"])
        rows = []
        for ot, g in sub.groupby("open_time"):
            if len(g)<10: continue
            g = g.sort_values("pred")
            for K in [5]:
                top = g.tail(K); bot = g.head(K)
                rows.append(dict(open_time=ot,
                    top_ret1d=top["return_1d"].mean(), bot_ret1d=bot["return_1d"].mean(),
                    top_ret3d=top["ret_3d"].mean(), bot_ret3d=bot["ret_3d"].mean(),
                    top_rvol=top["rvol_7d"].mean(), bot_rvol=bot["rvol_7d"].mean(),
                    top_atr=top["atr_pct"].mean(), bot_atr=bot["atr_pct"].mean()))
        rdf = pd.DataFrame(rows)
        print(f"  {label} (n_cycles={len(rdf)}):")
        print(f"    Top-K trailing 1d return:  {rdf['top_ret1d'].mean()*100:+.3f}%  (recent: {'losers' if rdf['top_ret1d'].mean()<0 else 'winners'})")
        print(f"    Bot-K trailing 1d return:  {rdf['bot_ret1d'].mean()*100:+.3f}%  (recent: {'losers' if rdf['bot_ret1d'].mean()<0 else 'winners'})")
        print(f"    Top-K trailing 3d return:  {rdf['top_ret3d'].mean()*100:+.3f}%")
        print(f"    Bot-K trailing 3d return:  {rdf['bot_ret3d'].mean()*100:+.3f}%")
        print(f"    Top-K trailing rvol_7d:    {rdf['top_rvol'].mean()*100:.3f}")
        print(f"    Bot-K trailing rvol_7d:    {rdf['bot_rvol'].mean()*100:.3f}")
        print(f"    PATTERN: top-K = {'recent losers (mean-rev: bounce expected)' if rdf['top_ret1d'].mean()<rdf['bot_ret1d'].mean() else 'recent winners'}")
        print(f"             bot-K = {'recent winners (mean-rev: revert down expected)' if rdf['bot_ret1d'].mean()>rdf['top_ret1d'].mean() else 'recent losers'}")
        print()

    # (b) CRITICAL TEST — short performance CONDITIONAL on trailing momentum
    print("\n=== (b) Short performance by trailing momentum bucket — does short work for 'genuine weak' but fail for 'recent winners'? ===\n")
    for label,(s,e) in [("H1",(H1_START,H2_START)),("H2",(H2_START,H2_END))]:
        sub = df[(df["open_time"]>=s)&(df["open_time"]<e)].dropna(subset=["pred","return_1d","return_pct"])
        # For each cycle, take bot-K=5 by pred, split into "genuine weak" (trailing ret < 0) vs "just rallied" (>0)
        weak_rets = []; strong_rets = []; baseline_rets = []
        for ot, g in sub.groupby("open_time"):
            if len(g)<10: continue
            g = g.sort_values("pred")
            bot = g.head(5)
            baseline_rets.extend(bot["return_pct"].values)
            weak = bot[bot["return_1d"] < 0]
            strong = bot[bot["return_1d"] > 0]
            weak_rets.extend(weak["return_pct"].values)
            strong_rets.extend(strong["return_pct"].values)
        # SHORT P&L: shorting → +PnL when realized < 0; -PnL when > 0
        # Edge for short = -mean(realized): positive = profitable
        wr = np.array(weak_rets); sr = np.array(strong_rets); br = np.array(baseline_rets)
        print(f"  {label}:")
        print(f"    Baseline bot-K=5 short edge (-mean(realized)):           {-br.mean()*1e4:+.2f} bps/cycle  n={len(br):,}")
        print(f"    Subset 'genuine weak' (trailing 1d<0) short edge:        {-wr.mean()*1e4:+.2f} bps/cycle  n={len(wr):,}")
        print(f"    Subset 'just rallied' (trailing 1d>0) short edge:        {-sr.mean()*1e4:+.2f} bps/cycle  n={len(sr):,}")
        # statistical significance
        if len(wr)>10 and len(sr)>10:
            from scipy.stats import ttest_ind
            t, p = ttest_ind(wr, sr)
            print(f"    Δ weak−strong (t-test): t={t:+.2f}  p={p:.3f}")
        print()

    # (c) Proposed fix preview: anti-momentum filter for shorts
    print("\n=== (c) Proposed fix: anti-momentum filter on shorts ===\n")
    print("If 'genuine weak' shorts have edge > +5 bps AND 'just rallied' shorts have edge < 0,")
    print("then filtering bot-K to only short names with trailing 1d<0 would deploy a better short signal.")
    print("Test variant: bot-K filtered to NAMES WITH NEGATIVE TRAILING 1d RETURN ONLY.\n")
    for label,(s,e) in [("H2",(H2_START,H2_END))]:
        sub = df[(df["open_time"]>=s)&(df["open_time"]<e)].dropna(subset=["pred","return_1d","return_pct"])
        edges_baseline = []; edges_filtered = []
        for ot, g in sub.groupby("open_time"):
            if len(g)<10: continue
            g = g.sort_values("pred")
            # baseline: bot-K=5 unconditional
            bot_base = g.head(5)
            # filtered: among bot-K=10, take 5 with most-negative trailing 1d
            bot_cand = g.head(10).sort_values("return_1d").head(5)
            edges_baseline.append(-bot_base["return_pct"].mean())
            if len(bot_cand)>=3:
                edges_filtered.append(-bot_cand["return_pct"].mean())
        eb = np.array(edges_baseline)*1e4; ef = np.array(edges_filtered)*1e4
        print(f"  {label}:")
        print(f"    Baseline short edge:         {eb.mean():+.2f} bps")
        print(f"    Anti-momentum filter edge:   {ef.mean():+.2f} bps")
        print(f"    Improvement:                 {ef.mean()-eb.mean():+.2f} bps")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
