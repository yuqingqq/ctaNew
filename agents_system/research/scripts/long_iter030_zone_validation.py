"""LONG-PRED iter-030 — Zone validation test.

Hypothesis (to validate):
  Zone 1 (ret_3d ∈ [-3%, +3%]): forward 4h has FEATURE-DRIVEN variance — longs work here
  Zone 2 (ret_3d ∈ [+3%, +10%]): forward 4h dominated by POSITIVE momentum continuation — shorts fail here
  Zone 3 (ret_3d > +10%): true mean reversion zone — but rarely picked

Tests:
  T1: Unconditional mean forward 4h return per ret_3d bucket
      → does forward return monotonically rise with ret_3d (momentum) or have a non-monotonic peak?
  T2: Per-zone model selection alpha
      → model TOP-K vs RANDOM K vs BOT-K mean forward return per zone
      → Confirms: in long-zone model picks winners; in short-zone model can't distinguish
  T3: Per-zone forward return variance (or std)
      → high variance = feature-discriminable zone
      → low variance = uniform behavior, features irrelevant
  T4: Where do model's longs and shorts actually land?
      → fraction of model's TOP-K and BOT-K picks in each zone
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5

# Bucket boundaries for ret_3d (3-day return)
ZONE_EDGES = [-1.0, -0.10, -0.05, -0.03, 0.0, 0.03, 0.05, 0.10, 0.20, 1.0]
ZONE_LABELS = ["<-10%", "-10/-5%", "-5/-3%", "-3/0%", "0/+3%", "+3/+5%", "+5/+10%", "+10/+20%", ">+20%"]

def main():
    t0 = time.time()
    print("=== iter-030: Zone hypothesis validation ===\n", flush=True)

    print("loading data...", flush=True)
    preds = pd.read_parquet(PREDS)
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    cols = ["symbol","open_time","return_pct","ret_3d"]
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    df = preds.merge(panel[["symbol","open_time","ret_3d"]], on=["symbol","open_time"], how="left")
    df = df[(df["open_time"] >= H1_START) & (df["open_time"] < H2_END)].copy()
    df = df.dropna(subset=["ret_3d","return_pct"])
    df["zone"] = pd.cut(df["ret_3d"], bins=ZONE_EDGES, labels=ZONE_LABELS, include_lowest=True)
    print(f"  {len(df):,} obs total", flush=True)

    # =========================================
    # T1: UNCONDITIONAL mean forward return per ret_3d zone
    # =========================================
    print(f"\n=== T1: Mean forward 4h return per ret_3d zone (validates momentum vs reversion) ===\n", flush=True)
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        print(f"\n  {period_label}:")
        print(f"    {'zone':<12} {'n':>9} {'mean fwd bps':>14} {'t-stat':>9} {'std bps':>10} signif")
        print(f"    {'-'*70}")
        for zone in ZONE_LABELS:
            z = sub[sub["zone"]==zone]
            if len(z) < 100: continue
            a = z["return_pct"].values * 1e4
            mean = a.mean(); std = a.std(); se = std/np.sqrt(len(a))
            t = mean/se if se>0 else float("nan")
            sig = "★" if abs(t)>1.96 else " "
            direction = "↑MOMENTUM CONTINUES" if mean>5 else ("↓MEAN REVERTS" if mean<-5 else "≈NEUTRAL")
            print(f"    {zone:<12} {len(a):>7,}  {mean:>+10.2f}    {t:>+6.2f}    {std:>8.2f}  {sig} {direction if abs(t)>1.5 else ''}")

    # =========================================
    # T2: Per-zone model TOP-K vs RANDOM vs BOT-K mean forward return
    # =========================================
    print(f"\n=== T2: Per-zone model selection alpha (within-zone) ===\n", flush=True)
    print("  For each zone+cycle, compute model top-K, bot-K, random-K mean forward return")
    print("  If model adds value in a zone: top-K > random > bot-K within that zone\n")
    rng = np.random.default_rng(42)
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        rows = []
        for ot, gc in sub.groupby("open_time"):
            for zone in ZONE_LABELS:
                gz = gc[gc["zone"]==zone]
                if len(gz) < 5: continue  # need at least 5 names in zone for top-K
                top_k = min(3, len(gz))
                top = gz.nlargest(top_k, "pred")["return_pct"].mean()
                bot = gz.nsmallest(top_k, "pred")["return_pct"].mean()
                # Random K from same zone
                rand_idx = rng.choice(len(gz), size=top_k, replace=False)
                rnd = gz.iloc[rand_idx]["return_pct"].mean()
                rows.append({"zone": zone, "top": top, "bot": bot, "rnd": rnd, "n_zone": len(gz)})
        rdf = pd.DataFrame(rows)
        print(f"\n  {period_label}:")
        print(f"    {'zone':<12} {'n_obs':>7} {'top-K bps':>11} {'random bps':>12} {'bot-K bps':>11} {'top-rnd':>9} {'rnd-bot':>9}")
        for zone in ZONE_LABELS:
            z = rdf[rdf["zone"]==zone]
            if len(z) < 30: continue
            top_m = z["top"].mean()*1e4
            bot_m = z["bot"].mean()*1e4
            rnd_m = z["rnd"].mean()*1e4
            top_minus_rnd = top_m - rnd_m
            rnd_minus_bot = rnd_m - bot_m
            value_long = "✓" if top_minus_rnd > 5 else ("?" if top_minus_rnd > 0 else "✗")
            value_short = "✓" if rnd_minus_bot > 5 else ("?" if rnd_minus_bot > 0 else "✗")
            print(f"    {zone:<12} {len(z):>5,}  {top_m:>+8.2f}    {rnd_m:>+8.2f}    {bot_m:>+8.2f}   {top_minus_rnd:>+6.2f}{value_long}  {rnd_minus_bot:>+6.2f}{value_short}")
        print(f"    (top-rnd > 0 = model adds long alpha in zone; rnd-bot > 0 = model adds short alpha in zone)")

    # =========================================
    # T3: Within-zone forward return variance (feature-discriminability)
    # =========================================
    print(f"\n=== T3: Within-zone forward return std (high = features can discriminate) ===\n", flush=True)
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        print(f"\n  {period_label}:")
        print(f"    {'zone':<12} {'std bps':>10} {'IQR bps':>10} {'discriminable':>14}")
        for zone in ZONE_LABELS:
            z = sub[sub["zone"]==zone]
            if len(z) < 100: continue
            a = z["return_pct"].values * 1e4
            std = a.std()
            iqr = np.percentile(a, 75) - np.percentile(a, 25)
            # High std/IQR = high cross-sectional dispersion → features could discriminate
            verdict = "HIGH dispersion" if std > 200 else ("MED" if std > 150 else "LOW dispersion")
            print(f"    {zone:<12} {std:>8.2f}    {iqr:>8.2f}    {verdict}")

    # =========================================
    # T4: Where do model picks actually land? (zone distribution)
    # =========================================
    print(f"\n=== T4: Distribution of model's TOP-K and BOT-K picks across zones ===\n", flush=True)
    for period_label, s, e in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
        top_picks, bot_picks = [], []
        for ot, gc in sub.groupby("open_time"):
            if len(gc) < 2*K: continue
            top_picks.extend(gc.nlargest(K, "pred")["zone"].tolist())
            bot_picks.extend(gc.nsmallest(K, "pred")["zone"].tolist())
        top_counts = pd.Series(top_picks).value_counts(normalize=True).reindex(ZONE_LABELS).fillna(0)
        bot_counts = pd.Series(bot_picks).value_counts(normalize=True).reindex(ZONE_LABELS).fillna(0)
        print(f"\n  {period_label}: where model's picks land (% of K picks in each zone)")
        print(f"    {'zone':<12} {'long picks %':>13} {'short picks %':>14}")
        for zone in ZONE_LABELS:
            print(f"    {zone:<12} {top_counts[zone]*100:>10.1f}%      {bot_counts[zone]*100:>10.1f}%")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
