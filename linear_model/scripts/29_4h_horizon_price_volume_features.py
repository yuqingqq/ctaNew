"""Step 29: Build 4h-horizon features + price-volume joint features.

The trading horizon is 4h (48 5m bars). R3 features are mostly 1d-7d windows
— horizon mismatch. Plus R3 has no PRICE×VOLUME joint features.

Build the following from raw kline data:

A. SHORT-HORIZON MOMENTUM (matching 4h target):
   1. return_4h     = close[t]/close[t-48] - 1
   2. return_8h     = close[t]/close[t-96] - 1
   3. return_24h    = close[t]/close[t-288] - 1
   (already have return_1d = 1d which is too long for direct prediction)

B. SHORT-HORIZON VOLATILITY:
   4. realized_vol_4h  = std of 5m returns over 48 bars
   5. realized_vol_24h = std of 5m returns over 288 bars

C. PRICE-VOLUME JOINT (the key new family):
   6. volprice_momentum_4h = return_4h × vol_4h_surge
      (positive return + surge = confirmed up; negative + surge = confirmed down)
   7. volume_weighted_return_4h = sum(ret_5m × vol_5m) / sum(vol_5m) over 4h
      (where money actually moved, not just where close moved)
   8. vwap_dev_4h = (close - vwap_4h) / atr_pct
      (normalized distance from where money was traded)
   9. price_volume_corr_4h = Spearman(price_5m, vol_5m) over 48 bars
      (do prices and volumes move together within the 4h window?)

D. ABNORMAL VOLUME:
   10. vol_zscore_4h_over_7d = (vol_4h - mean(vol_4h, 7d)) / std(vol_4h, 7d)
       (z-score of 4h volume vs its own 7d distribution)

For each: audit shape, IC, redundancy with R3.
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
OUT = REPO / "linear_model/results"

BAR_PER_DAY = 288
BAR_PER_HOUR = 12
BAR_4H = 48
BAR_8H = 96
BAR_24H = 288


def shape_diag(decile_targets):
    d = decile_targets.values
    if len(d) < 10: return "insufficient"
    rho = stats.spearmanr(range(10), d).statistic
    mid = np.mean(d[3:7])
    tails = np.mean(d[[0,1,2,7,8,9]])
    if rho > 0.7:   return "monotonic_up"
    elif rho < -0.7: return "monotonic_down"
    elif mid > tails + abs(np.std(d)) * 0.5: return "inverted_u"
    elif mid < tails - abs(np.std(d)) * 0.5: return "u_shape"
    else: return "noisy"


def build_features_for_symbol(sym_dir):
    """Build all candidate features for one symbol from raw 5m klines."""
    m5 = sym_dir / "5m"
    files = sorted(m5.glob("*.parquet"))
    if not files: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time","close","high","low","quote_volume"])
                      for f in files], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)

    close = df["close"].astype(float)
    qv = df["quote_volume"].astype(float)

    # 5m bar-to-bar return for some calcs
    ret_5m = close.pct_change()

    # ===== A. Short-horizon momentum =====
    df["return_4h"]  = close.pct_change(BAR_4H).shift(1)
    df["return_8h"]  = close.pct_change(BAR_8H).shift(1)
    df["return_24h"] = close.pct_change(BAR_24H).shift(1)

    # ===== B. Short-horizon volatility =====
    df["realized_vol_4h"]  = ret_5m.rolling(BAR_4H, min_periods=BAR_4H).std().shift(1)
    df["realized_vol_24h"] = ret_5m.rolling(BAR_24H, min_periods=BAR_24H).std().shift(1)

    # ===== C. Price-volume joint =====
    # vol_4h_surge: 4h volume / 7d avg 4h volume (similar to vol_surge but better-named)
    vol_4h = qv.rolling(BAR_4H, min_periods=BAR_4H).sum()
    mean_5m_7d = qv.rolling(7*BAR_PER_DAY, min_periods=4*BAR_PER_DAY).mean()
    avg_4h_over_7d = mean_5m_7d * BAR_4H
    vol_4h_surge = (vol_4h / avg_4h_over_7d.replace(0, np.nan)).shift(BAR_4H+1)

    # 1. volprice_momentum_4h = return_4h × log(vol_4h_surge)
    #    using log to avoid extreme values from vol_surge tail
    df["volprice_momentum_4h"] = df["return_4h"] * np.log1p(vol_4h_surge.fillna(1))

    # 2. volume_weighted_return_4h: weighted avg of 5m returns by 5m volume
    # = sum(ret_5m * vol_5m, window) / sum(vol_5m, window)
    weighted_ret = (ret_5m * qv).rolling(BAR_4H, min_periods=BAR_4H).sum()
    total_vol = qv.rolling(BAR_4H, min_periods=BAR_4H).sum()
    df["volume_weighted_return_4h"] = (weighted_ret / total_vol.replace(0, np.nan)).shift(1)

    # 3. vwap_dev_4h: (close - vwap_4h) / atr_pct_4h
    # vwap_4h = sum(close × volume) / sum(volume) over last 4h
    cvw = (close * qv).rolling(BAR_4H, min_periods=BAR_4H).sum()
    vw = qv.rolling(BAR_4H, min_periods=BAR_4H).sum()
    vwap_4h = (cvw / vw.replace(0, np.nan))
    # atr_4h: high-low range avg
    tr = (df["high"] - df["low"]).astype(float)
    atr_4h = tr.rolling(BAR_4H, min_periods=BAR_4H).mean()
    df["vwap_dev_4h"] = ((close - vwap_4h) / (atr_4h.replace(0, np.nan))).shift(1)

    # 4. price_volume_corr_4h: rolling Spearman between price changes and volume changes
    # Spearman is expensive; use Pearson on log_changes as proxy
    log_ret = np.log(close).diff()
    log_vol = np.log(qv.replace(0, np.nan)).diff()
    cov = log_ret.rolling(BAR_4H, min_periods=BAR_4H).cov(log_vol)
    sd_r = log_ret.rolling(BAR_4H, min_periods=BAR_4H).std()
    sd_v = log_vol.rolling(BAR_4H, min_periods=BAR_4H).std()
    df["price_volume_corr_4h"] = (cov / (sd_r * sd_v).replace(0, np.nan)).shift(1)

    # ===== D. Abnormal volume z-score =====
    # vol_4h_z = (vol_4h - mean_4h_7d) / std_4h_7d
    # Use 5m volume aggregated: rolling 7d std of 5m volume × √48
    vol_4h_history = qv.rolling(BAR_4H).sum()  # 4h vol at every 5m bar (rolling)
    # Standardize: use the rolling distribution of vol_4h over past 7d (504 4h-buckets)
    # Simpler: use 5m vol mean+std × √48 as approx
    mean_5m_7d_v = qv.rolling(7*BAR_PER_DAY, min_periods=4*BAR_PER_DAY).mean()
    std_5m_7d_v = qv.rolling(7*BAR_PER_DAY, min_periods=4*BAR_PER_DAY).std()
    df["vol_zscore_4h_over_7d"] = ((vol_4h_history - BAR_4H * mean_5m_7d_v)
                                     / (np.sqrt(BAR_4H) * std_5m_7d_v.replace(0, np.nan))).shift(BAR_4H+1)

    return df[["open_time"] + [
        "return_4h", "return_8h", "return_24h",
        "realized_vol_4h", "realized_vol_24h",
        "volprice_momentum_4h", "volume_weighted_return_4h",
        "vwap_dev_4h", "price_volume_corr_4h",
        "vol_zscore_4h_over_7d",
    ]]


def main():
    print("=== Step 29: 4h-horizon + price-volume joint features ===\n",
          flush=True)
    t0 = time.time()

    # Build features per symbol
    print("Building features from raw klines...", flush=True)
    all_feats = []
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        sym = sym_dir.name
        if sym == "BTCUSDT": continue
        feats = build_features_for_symbol(sym_dir)
        if feats is None: continue
        feats["symbol"] = sym
        all_feats.append(feats)
    feat_panel = pd.concat(all_feats, ignore_index=True)
    print(f"  built {len(feat_panel):,} rows × {feat_panel.shape[1]} cols "
          f"({time.time()-t0:.0f}s)", flush=True)
    feat_panel.to_parquet(OUT / "step29_features.parquet", index=False)

    # Audit
    tgt = pd.read_parquet(TARGETS, columns=["symbol","open_time","alpha_beta",
                                              "autocorr_pctile_7d"])
    tgt["open_time"] = pd.to_datetime(tgt["open_time"], utc=True)
    df = tgt.merge(feat_panel, on=["symbol","open_time"], how="left")

    # Fold-0 training rows for audit
    folds = _multi_oos_splits(tgt)
    train0_t = _slice(tgt, folds[0])[0]["open_time"]
    train_lo, train_hi = train0_t.min(), train0_t.max()
    train0 = df[(df.open_time >= train_lo) & (df.open_time <= train_hi)
                 & (df.autocorr_pctile_7d >= 0.5)
                 & df.alpha_beta.notna()]
    target_bps = (train0["alpha_beta"] * 1e4).clip(-1000, 1000)
    print(f"  fold-0 train: {len(train0):,} rows\n", flush=True)

    feats_to_audit = ["return_4h", "return_8h", "return_24h",
                       "realized_vol_4h", "realized_vol_24h",
                       "volprice_momentum_4h", "volume_weighted_return_4h",
                       "vwap_dev_4h", "price_volume_corr_4h",
                       "vol_zscore_4h_over_7d"]

    print(f"{'='*120}", flush=True)
    print(f"  AUDIT (4h-target prediction)", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"  {'feature':<32} {'nan%':>6} {'skew':>7} {'kurt':>8} "
          f"{'pearson':>9} {'spearman':>10} {'shape':<14}", flush=True)
    audit_results = []
    for f in feats_to_audit:
        s = train0[f]
        valid = s.notna() & target_bps.notna()
        if valid.sum() < 1000: continue
        ss = s[valid].clip(s.quantile(0.005), s.quantile(0.995))
        yy = target_bps[valid]
        nan_pct = (1 - valid.mean()) * 100
        skew = stats.skew(ss); kurt = stats.kurtosis(ss)
        pr = stats.pearsonr(ss, yy).statistic
        sr = stats.spearmanr(ss, yy).statistic
        try:
            q = pd.qcut(ss, 10, labels=False, duplicates="drop")
            dec = yy.groupby(q).mean()
            shape = shape_diag(dec)
        except Exception:
            shape = "binning_fail"
        audit_results.append({"feature":f, "nan_pct":nan_pct,
                              "skew":skew, "kurt":kurt,
                              "pearson":pr, "spearman":sr, "shape":shape})
        print(f"  {f:<32} {nan_pct:>5.1f}% {skew:>+7.2f} {kurt:>+8.1f} "
              f"{pr:>+9.4f} {sr:>+10.4f} {shape:<14}", flush=True)

    # Decile breakdowns for the most promising features
    print(f"\n{'='*120}", flush=True)
    print(f"  DECILE BREAKDOWN for high-|spearman| features", flush=True)
    print(f"{'='*120}", flush=True)
    df_audit = pd.DataFrame(audit_results).sort_values("spearman", key=abs,
                                                        ascending=False)
    for _, r in df_audit.head(6).iterrows():
        f = r["feature"]
        s = train0[f]
        valid = s.notna() & target_bps.notna()
        ss = s[valid].clip(s.quantile(0.005), s.quantile(0.995))
        yy = target_bps[valid]
        q = pd.qcut(ss, 10, labels=False, duplicates="drop")
        dec = yy.groupby(q).agg(["mean","count"])
        print(f"\n  {f} (spearman {r['spearman']:+.4f}, shape {r['shape']}):",
              flush=True)
        for i, row in dec.iterrows():
            bar = "█" * int(abs(row["mean"]/2))
            sign = "+" if row["mean"] >= 0 else "-"
            print(f"    decile {int(i)}: mean = {row['mean']:>+8.2f} bps  "
                  f"(n={int(row['count']):>6,})  {sign}{bar}", flush=True)

    df_audit.to_csv(OUT / "step29_audit.csv", index=False)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
