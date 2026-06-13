"""Step 26: Audit volume-surge features (recent / average ratios).

Build candidates from raw kline data:
  vol_surge_4h_over_7d   = sum(quote_volume, 4h) / mean(daily quote_volume, 7d)
  vol_surge_1d_over_30d  = sum(quote_volume, 1d) / mean(daily quote_volume, 30d)
  vol_surge_4h_over_30d  = sum(quote_volume, 4h) / mean(4h quote_volume over 30d)

For each: audit shape, IC vs target_bps, NaN rate, redundancy with existing
features (correlation with log_dollar_volume_7d, volume_stability_30d).

If any have monotonic shape + |spearman| > 0.01 + low redundancy, propose
adding to next R3 variant.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

KLINES_DIR = REPO / "data/ml/test/parquet/klines"
TARGETS = REPO / "linear_model/data/targets.parquet"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
V3_5M = REPO / "outputs/vBTC_features_btc_v3/panel_v3_5m.parquet"
OUT = REPO / "linear_model/results"

BAR_PER_DAY = 288
BAR_PER_HOUR = 12
BAR_4H = 48


def shape_diag(decile_targets):
    d = decile_targets.values
    if len(d) < 10: return "insufficient"
    rho = stats.spearmanr(range(10), d).statistic
    mid = np.mean(d[3:7])
    tails = np.mean(d[[0,1,2,7,8,9]])
    if rho > 0.7:   return "monotonic_up"
    elif rho < -0.7: return "monotonic_down"
    elif mid > tails + abs(np.std(d)) * 0.5: return "inverted_u (BAD)"
    elif mid < tails - abs(np.std(d)) * 0.5: return "u_shape (sq fix)"
    else: return "noisy"


def main():
    print("=== Step 26: Volume-surge ratio audit ===\n", flush=True)
    t0 = time.time()

    # Build vol surge features from raw klines
    print("Loading kline quote_volume per symbol + computing surge ratios...",
          flush=True)
    all_surge = []
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        sym = sym_dir.name
        if sym == "BTCUSDT": continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        df = pd.concat([pd.read_parquet(f, columns=["open_time","quote_volume"])
                          for f in files], ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.sort_values("open_time").reset_index(drop=True)
        df["symbol"] = sym

        qv = df["quote_volume"].astype(float)

        # 4h volume = sum of 48 5m bars
        vol_4h = qv.rolling(BAR_4H, min_periods=BAR_4H).sum()
        # 1d volume = sum of 288 5m bars
        vol_1d = qv.rolling(BAR_PER_DAY, min_periods=BAR_PER_DAY).sum()
        # 1h volume = sum of 12 bars
        vol_1h = qv.rolling(BAR_PER_HOUR, min_periods=BAR_PER_HOUR).sum()

        # Trailing means (NOT including current period for PIT cleanliness)
        # Mean 4h vol over trailing 7d (excluding most recent 4h to avoid overlap)
        # 7d × 6 4h-buckets/day = 42 4h-windows
        # Use shifted rolling mean: at time t, use 4h windows ending at t-48 backwards
        # Easier: use 7d backward mean of 5m volume, multiply by 48 to get 4h-equivalent average
        mean_5m_7d = qv.rolling(7*BAR_PER_DAY, min_periods=4*BAR_PER_DAY).mean()
        avg_4h_over_7d = mean_5m_7d * BAR_4H  # equivalent to mean 4h volume

        mean_5m_30d = qv.rolling(30*BAR_PER_DAY, min_periods=15*BAR_PER_DAY).mean()
        avg_1d_over_30d = mean_5m_30d * BAR_PER_DAY
        avg_4h_over_30d = mean_5m_30d * BAR_4H

        # Surge ratios (PIT: use last completed window for "recent" + trailing mean
        # shifted so it doesn't include the recent window)
        # vol_4h_over_7d_avg = vol_4h / avg_4h_over_7d (shifted by 4h+1 to be strict PIT)
        df["vol_surge_4h_over_7d"] = (vol_4h.shift(BAR_4H+1) /
                                       avg_4h_over_7d.shift(BAR_4H+1)
                                       .replace(0, np.nan))
        df["vol_surge_1d_over_30d"] = (vol_1d.shift(BAR_PER_DAY+1) /
                                        avg_1d_over_30d.shift(BAR_PER_DAY+1)
                                        .replace(0, np.nan))
        df["vol_surge_4h_over_30d"] = (vol_4h.shift(BAR_4H+1) /
                                        avg_4h_over_30d.shift(BAR_4H+1)
                                        .replace(0, np.nan))
        df["vol_surge_1h_over_24h"] = (vol_1h.shift(BAR_PER_HOUR+1) /
                                        (mean_5m_7d.shift(BAR_PER_HOUR+1)
                                          * BAR_PER_HOUR).replace(0, np.nan))

        all_surge.append(df[["symbol","open_time","vol_surge_4h_over_7d",
                              "vol_surge_1d_over_30d", "vol_surge_4h_over_30d",
                              "vol_surge_1h_over_24h"]])

    surge = pd.concat(all_surge, ignore_index=True)
    surge.to_parquet(OUT / "vol_surge_features.parquet", index=False)
    print(f"  Built {len(surge):,} rows × 4 surge features ({time.time()-t0:.0f}s)",
          flush=True)

    # Load targets
    tgt = pd.read_parquet(TARGETS, columns=["symbol","open_time","alpha_beta",
                                              "autocorr_pctile_7d"])
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    df = tgt.merge(surge, on=["symbol","open_time"], how="left")

    # Fold-0 training slice
    panel = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    folds = _multi_oos_splits(panel)
    # Get train period from _slice on the actual panel
    panel_for_slice = panel.merge(tgt[["symbol","open_time"]],
                                    on=["symbol","open_time"], how="inner")
    train_slice, _, _ = _slice(panel_for_slice, folds[0])
    train_lo = train_slice["open_time"].min()
    train_hi = train_slice["open_time"].max()
    print(f"  train slice: {train_lo} → {train_hi}", flush=True)
    train0 = df[(df.open_time >= train_lo) & (df.open_time <= train_hi)
                 & (df.autocorr_pctile_7d >= 0.5) & df.alpha_beta.notna()]
    target_bps = train0["alpha_beta"].clip(-0.1, 0.1) * 1e4
    print(f"\n  Fold-0 train slice: {len(train0):,} rows", flush=True)

    # ===== Audit each surge feature =====
    print(f"\n{'='*120}", flush=True)
    print(f"  SURGE FEATURE AUDIT", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"  {'feature':<28} {'nan%':>6} {'skew':>7} {'kurt':>8} "
          f"{'p1':>7} {'p99':>7} {'pearson':>9} {'spearman':>10} {'shape':<24}",
          flush=True)
    surge_feats = ["vol_surge_4h_over_7d", "vol_surge_1d_over_30d",
                   "vol_surge_4h_over_30d", "vol_surge_1h_over_24h"]
    audit = []
    for f in surge_feats:
        s = train0[f]
        valid = ~s.isna() & ~target_bps.isna()
        if valid.sum() < 1000: continue
        ss = s[valid].clip(s.quantile(0.005), s.quantile(0.995))
        yy = target_bps[valid]
        nan_pct = (1 - valid.mean()) * 100
        skew = stats.skew(ss); kurt = stats.kurtosis(ss)
        p1, p50, p99 = ss.quantile([0.01, 0.5, 0.99]).values
        pr = stats.pearsonr(ss, yy).statistic
        sr = stats.spearmanr(ss, yy).statistic
        try:
            q = pd.qcut(ss, 10, labels=False, duplicates="drop")
            dec = yy.groupby(q).mean()
            shape = shape_diag(dec)
        except Exception:
            shape = "binning_fail"
        audit.append({"feature":f, "nan_pct":nan_pct, "skew":skew, "kurt":kurt,
                      "p1":p1, "p99":p99, "pearson":pr, "spearman":sr, "shape":shape})
        print(f"  {f:<28} {nan_pct:>5.1f}% {skew:>+7.2f} {kurt:>+8.1f} "
              f"{p1:>7.2f} {p99:>7.2f} {pr:>+9.4f} {sr:>+10.4f} {shape:<24}",
              flush=True)

    # ===== Decile means (more detail for top candidates) =====
    print(f"\n{'='*120}", flush=True)
    print(f"  DECILE MEANS for each surge feature (target_bps)", flush=True)
    print(f"{'='*120}", flush=True)
    for f in surge_feats:
        s = train0[f]
        valid = ~s.isna() & ~target_bps.isna()
        if valid.sum() < 1000: continue
        ss = s[valid].clip(s.quantile(0.005), s.quantile(0.995))
        yy = target_bps[valid]
        try:
            q = pd.qcut(ss, 10, labels=False, duplicates="drop")
            dec = yy.groupby(q).agg(["mean","count"])
            print(f"\n  {f}:", flush=True)
            for i, row in dec.iterrows():
                bar = "█" * int(abs(row["mean"]/2))
                sign = "+" if row["mean"] >= 0 else "-"
                print(f"    decile {int(i)}: mean = {row['mean']:>+8.2f} bps "
                      f"(n={int(row['count']):>6,})  {sign}{bar}", flush=True)
        except Exception:
            print(f"  {f}: decile binning failed", flush=True)

    # ===== Correlation with existing volume features =====
    print(f"\n{'='*120}", flush=True)
    print(f"  CORRELATION with existing features (redundancy check)", flush=True)
    print(f"{'='*120}", flush=True)
    # Load existing features from various sources
    v3 = pd.read_parquet(V3_5M, columns=["symbol","open_time",
                                          "log_dollar_volume_7d",
                                          "volume_stability_30d",
                                          "amihud_illiq_30d"])
    v3["open_time"] = pd.to_datetime(v3["open_time"], utc=True)
    base = pd.read_parquet(PANEL, columns=["symbol","open_time","volume_ma_50",
                                            "idio_vol_to_btc_1h"])
    base["open_time"] = pd.to_datetime(base["open_time"], utc=True)
    df_corr = df.merge(v3, on=["symbol","open_time"], how="left")
    df_corr = df_corr.merge(base, on=["symbol","open_time"], how="left")
    train_corr = df_corr[(df_corr.open_time < train_lo)
                          & (df_corr.autocorr_pctile_7d >= 0.5)]
    existing = ["log_dollar_volume_7d", "volume_stability_30d",
                "amihud_illiq_30d", "volume_ma_50", "idio_vol_to_btc_1h"]
    print(f"  {'surge':<28} | " + " ".join(f"{e[:15]:<15}" for e in existing), flush=True)
    for sf in surge_feats:
        row = [f"  {sf:<28} |"]
        for e in existing:
            try:
                c = train_corr[[sf, e]].dropna().corr(method="spearman").iloc[0,1]
                row.append(f"{c:+.3f}".ljust(15))
            except Exception:
                row.append("n/a".ljust(15))
        print(" ".join(row), flush=True)

    pd.DataFrame(audit).to_csv(OUT / "vol_surge_audit.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
