"""Distinguish (A) market-uniformity-killing-tail-edge vs (B) model-misranking
as the cause of H2 underperformance.

Test 1: tail-edge per cycle = top-K mean realized return - all-syms mean. Same for bot-K
        (with sign flip). If H2 tail-edge collapses, mechanism (A) — alts are coupled,
        tails carry no info. If H2 tail-edge holds positive, mechanism (B) — rankings
        wrong, even though edge would exist if predictions were better.

Test 2: rank stability = Jaccard overlap of top-K (and bot-K) between consecutive cycles.
        High stability + bad PnL = model picking same wrong names. Low stability + bad PnL
        = model randomizing. (Note: bull regime uses mom30 instead; we look at side only.)

Both diagnostics on hysteresis-N=3 side regime only, full 2025-10 → 2026-05 OOS.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
K = 5

print("loading preds...")
d = pd.read_parquet(PREDS)
d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
print(f"  {len(d):,} rows × {d['symbol'].nunique()} syms × {d['open_time'].min().date()}→{d['open_time'].max().date()}")

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

# Per cycle: sort by pred, get top-K and bot-K, look up their return_pct
def per_cycle_tail(g):
    if len(g)<2*K: return None
    g = g.sort_values("pred")
    L = g.tail(K); S = g.head(K)
    return pd.Series(dict(
        all_mean=g["return_pct"].mean(),
        top_mean=L["return_pct"].mean(),
        bot_mean=S["return_pct"].mean(),
        top_edge=L["return_pct"].mean() - g["return_pct"].mean(),  # top-K relative to all
        bot_edge=g["return_pct"].mean() - S["return_pct"].mean(),  # all relative to bot-K
        spread_K=L["return_pct"].mean() - S["return_pct"].mean(),  # long-short pre-cost gross
        pred_disp=g["pred"].std(),
        n_syms=len(g),
        top_syms=tuple(sorted(L["symbol"].tolist())),
        bot_syms=tuple(sorted(S["symbol"].tolist())),
    ))

cyc = d.groupby("open_time").apply(per_cycle_tail).dropna()
cyc.index = cyc.index.get_level_values(0)
cyc["open_time"] = cyc.index
cyc = cyc.reset_index(drop=True)
cyc["open_time"] = pd.to_datetime(cyc["open_time"], utc=True)
print(f"\n=== TEST 1: TAIL-EDGE (top-K & bot-K realized return spread vs all-syms mean) ===")
for label,(s,e) in [("FULL OOS",(H1[0],H2[1])), ("H1",H1), ("H2",H2)]:
    sub = cyc[(cyc["open_time"]>=s)&(cyc["open_time"]<e)]
    print(f"\n{label} (n_cycles={len(sub)}):")
    print(f"  top-K edge (long basket - all): mean {sub['top_edge'].mean()*1e4:+.1f} bps  std {sub['top_edge'].std()*1e4:.1f}  %positive {100*(sub['top_edge']>0).mean():.1f}%")
    print(f"  bot-K edge (all - short basket): mean {sub['bot_edge'].mean()*1e4:+.1f} bps  std {sub['bot_edge'].std()*1e4:.1f}  %positive {100*(sub['bot_edge']>0).mean():.1f}%")
    print(f"  long-short spread (gross alpha): mean {sub['spread_K'].mean()*1e4:+.1f} bps  std {sub['spread_K'].std()*1e4:.1f}  %positive {100*(sub['spread_K']>0).mean():.1f}%")
    print(f"  pred_disp:                       mean {sub['pred_disp'].mean():.3f}")
    print(f"  all-syms mean return:            mean {sub['all_mean'].mean()*1e4:+.1f} bps  (market drift)")

# === TEST 2: rank stability ===
print(f"\n\n=== TEST 2: RANK STABILITY (Jaccard overlap of top-K / bot-K vs previous cycle) ===")
cyc = cyc.sort_values("open_time").reset_index(drop=True)
top_overlaps, bot_overlaps = [], []
for i in range(1, len(cyc)):
    prev_top = set(cyc.iloc[i-1]["top_syms"]); cur_top = set(cyc.iloc[i]["top_syms"])
    prev_bot = set(cyc.iloc[i-1]["bot_syms"]); cur_bot = set(cyc.iloc[i]["bot_syms"])
    top_overlaps.append((cyc.iloc[i]["open_time"], len(prev_top & cur_top)/K))
    bot_overlaps.append((cyc.iloc[i]["open_time"], len(prev_bot & cur_bot)/K))
top_df = pd.DataFrame(top_overlaps, columns=["open_time","top_jacc"])
bot_df = pd.DataFrame(bot_overlaps, columns=["open_time","bot_jacc"])
stab = top_df.merge(bot_df, on="open_time")

for label,(s,e) in [("FULL OOS",(H1[0],H2[1])), ("H1",H1), ("H2",H2)]:
    sub = stab[(stab["open_time"]>=s)&(stab["open_time"]<e)]
    print(f"\n{label} (n={len(sub)}):")
    print(f"  top-K overlap with previous cycle (Jaccard, mean): {sub['top_jacc'].mean():.3f}  (1.0=identical, 0=disjoint)")
    print(f"  bot-K overlap with previous cycle: {sub['bot_jacc'].mean():.3f}")
    print(f"  % cycles where top-K is fully replaced (overlap=0): {100*(sub['top_jacc']==0).mean():.1f}%")
    print(f"  % cycles where top-K is unchanged (overlap=1.0):   {100*(sub['top_jacc']==1.0).mean():.1f}%")

# === TEST 3: monthly tail-edge trajectory ===
print(f"\n\n=== TEST 3: monthly tail-edge trajectory ===")
cyc["month"] = cyc["open_time"].dt.to_period("M").astype(str)
mm = cyc.groupby("month").agg(n=("spread_K","count"),
    top_edge_bps=("top_edge",lambda x:x.mean()*1e4),
    bot_edge_bps=("bot_edge",lambda x:x.mean()*1e4),
    spread_bps=("spread_K",lambda x:x.mean()*1e4),
    pred_disp=("pred_disp","mean"),
    all_mean_bps=("all_mean",lambda x:x.mean()*1e4))
print(mm.round(2).tail(20).to_string())
