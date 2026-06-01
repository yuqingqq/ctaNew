"""X4 — Cross-exchange basis features from OKX spot + perp data.

Stages:
  1. Load OKX SPOT + OKX SWAP 1h klines (from data/ml/cache/okx_*_<SYM>_1h.parquet)
  2. Load Binance Perp 5m closes (from data/ml/test/parquet/klines/<SYM>/5m/)
  3. Align all three to a common 4h cycle grid (open_time hour % 4 == 0, minute 0)
  4. Compute four candidate basis features:
       - bn_okx_perp_basis_bps   = (binance_perp − okx_perp) / mid * 10000
       - bn_okx_spot_basis_bps   = (binance_perp − okx_spot) / mid * 10000
       - okx_perp_spot_basis_bps = (okx_perp − okx_spot) / mid * 10000  (OKX's funding-equiv)
       - bn_perp_okx_perp_minus_spot = bn_okx_perp_basis - bn_okx_spot_basis (decomposition)
  5. PIT preprocessing:
       - Per-symbol trailing-30d z (PIT)
       - Cross-sectional z per cycle (across all available syms at that t)
  6. IC test vs next-4h alpha_vs_btc_realized:
       - Pooled IC
       - Per-symbol IC
       - Per-fold IC (9 walk-forward folds matched to V3.1)
       - Multiple-testing null baseline (100 label shuffles)

Pass criterion: aggregate IC > 0.02 with t-stat > 3 AND consistent across folds.
If pass, the feature is a candidate for V3.1 retrain.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = REPO / "data/ml/cache"
KLINES = REPO / "data/ml/test/parquet/klines"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
N_FOLDS = 9


def load_okx_set(kind: str):
    """kind in {'spot','swap'}. Returns long-format df indexed by (symbol, open_time)."""
    files = list(CACHE.glob(f"okx_{kind}_*_1h.parquet"))
    frames = []
    for f in files:
        sym = f.stem.replace(f"okx_{kind}_", "").replace("_1h", "")
        try:
            df = pd.read_parquet(f, columns=["open_time", "close"])
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df["symbol"] = sym
            df = df.rename(columns={"close": f"okx_{kind}_close"})
            frames.append(df)
        except Exception as e:
            print(f"  err loading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    print(f"  OKX {kind}: {len(out):,} rows × {out['symbol'].nunique()} syms",
          flush=True)
    return out


def load_binance_perp_closes(syms, start, end):
    """Load Binance perp 5m closes for given syms over time window."""
    frames = []
    for sym in syms:
        sd = KLINES / sym / "5m"
        if not sd.exists(): continue
        files = sorted(sd.glob("*.parquet"))
        # Filter by date range (file names are dates)
        dfs = []
        for f in files:
            try:
                date_str = f.stem
                # quick date filter
                if "2025-04" in date_str or "2025-05" in date_str or "2025-06" in date_str \
                   or "2025-07" in date_str or "2025-08" in date_str or "2025-09" in date_str \
                   or "2025-10" in date_str or "2025-11" in date_str or "2025-12" in date_str \
                   or "2026-01" in date_str or "2026-02" in date_str or "2026-03" in date_str \
                   or "2026-04" in date_str or "2026-05" in date_str:
                    df = pd.read_parquet(f, columns=["open_time", "close"])
                    dfs.append(df)
            except Exception: pass
        if not dfs: continue
        df = pd.concat(dfs, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["symbol"] = sym
        df = df.rename(columns={"close": "bn_perp_close"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_features(df):
    """df has bn_perp_close + okx_swap_close + okx_spot_close + symbol + open_time.
    Compute basis features + per-symbol trailing-z + cross-sectional z per cycle."""
    df = df.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    # basis features (in bps)
    # mid = average of the two prices being compared
    mid_perp = (df["bn_perp_close"] + df["okx_swap_close"]) / 2
    df["bn_okx_perp_basis_bps"] = (df["bn_perp_close"] - df["okx_swap_close"]) / mid_perp * 10000.0

    mid_spot = (df["bn_perp_close"] + df["okx_spot_close"]) / 2
    df["bn_okx_spot_basis_bps"] = (df["bn_perp_close"] - df["okx_spot_close"]) / mid_spot * 10000.0

    mid_okx = (df["okx_swap_close"] + df["okx_spot_close"]) / 2
    df["okx_perp_spot_basis_bps"] = (df["okx_swap_close"] - df["okx_spot_close"]) / mid_okx * 10000.0

    df["bn_perp_minus_okx_minus_spot"] = (df["bn_okx_perp_basis_bps"] -
                                          df["bn_okx_spot_basis_bps"])

    feat_cols = ["bn_okx_perp_basis_bps", "bn_okx_spot_basis_bps",
                 "okx_perp_spot_basis_bps", "bn_perp_minus_okx_minus_spot"]

    # Per-symbol trailing-30d (180 4h-bars) z; PIT (shift 1 after rolling)
    for c in feat_cols:
        df[c + "_z"] = df.groupby("symbol")[c].transform(
            lambda s: ((s - s.rolling(180, min_periods=30).mean()) /
                       s.rolling(180, min_periods=30).std().replace(0, np.nan)).shift(1))

    # Cross-sectional z per cycle
    for c in feat_cols:
        df[c + "_xsz"] = df.groupby("open_time")[c].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))

    return df, feat_cols


def ic_test(df, feat, target_col, label):
    """Pooled IC + per-fold + per-symbol breakdown."""
    v = df[feat]; t = df[target_col]
    valid = v.notna() & t.notna()
    if valid.sum() < 100:
        return {"feat": feat, "label": label, "n": int(valid.sum()), "status": "insufficient"}
    ic_pooled = float(v[valid].corr(t[valid]))
    n = int(valid.sum())
    se = 1.0 / np.sqrt(n)
    tstat = ic_pooled / se
    # null: shuffle target labels within symbol
    nulls = []
    rng = np.random.RandomState(42)
    for s in range(100):
        v_sh = v[valid].sample(frac=1, random_state=s).to_numpy()
        nulls.append(float(np.corrcoef(v_sh, t[valid])[0, 1]))
    null_p95 = float(np.percentile(np.abs(nulls), 95))

    # per-fold
    times = sorted(df["open_time"].unique())
    fs = len(times) // N_FOLDS
    fold_ics = []
    for f in range(N_FOLDS):
        i0 = f * fs
        i1 = min((f+1)*fs, len(times)-1) if f < N_FOLDS-1 else len(times)
        ts = pd.Timestamp(times[i0]); te = pd.Timestamp(times[i1-1])
        mask = (df["open_time"] >= ts) & (df["open_time"] <= te) & valid
        if mask.sum() < 30: continue
        fic = float(v[mask].corr(t[mask]))
        fold_ics.append(fic)

    # per-symbol (top contributors)
    per_sym = {}
    for sym, g in df[valid].groupby("symbol"):
        if len(g) < 30: continue
        per_sym[sym] = round(float(g[feat].corr(g[target_col])), 4)

    return {
        "feat": feat, "label": label,
        "n_valid": n,
        "ic_pooled": round(ic_pooled, 5),
        "se": round(se, 5),
        "t_stat": round(tstat, 2),
        "null_p95_abs": round(null_p95, 5),
        "beats_null_p95": bool(abs(ic_pooled) > null_p95),
        "per_fold_ic": [round(x, 4) for x in fold_ics],
        "n_folds": len(fold_ics),
        "folds_positive": sum(1 for x in fold_ics if x > 0),
        "per_symbol_ic_min": min(per_sym.values()) if per_sym else None,
        "per_symbol_ic_max": max(per_sym.values()) if per_sym else None,
        "per_symbol_ic_median": float(np.median(list(per_sym.values()))) if per_sym else None,
        "n_syms_evaluable": len(per_sym),
    }


def main():
    t0 = time.time()
    print("=== X4 cross-exchange basis feature test ===\n", flush=True)

    # Load OKX
    okx_swap = load_okx_set("swap")
    okx_spot = load_okx_set("spot")
    if okx_swap.empty or okx_spot.empty:
        print("  OKX data not ready — collection still running. Wait.")
        return

    # Common syms
    syms_swap = set(okx_swap["symbol"].unique())
    syms_spot = set(okx_spot["symbol"].unique())
    syms_both = sorted(syms_swap & syms_spot)
    print(f"  OKX both spot+swap: {len(syms_both)} syms", flush=True)

    # Time range
    okx_swap = okx_swap[okx_swap["symbol"].isin(syms_both)]
    okx_spot = okx_spot[okx_spot["symbol"].isin(syms_both)]
    print(f"  OKX time: {okx_swap['open_time'].min()} → {okx_swap['open_time'].max()}",
          flush=True)

    # Load Binance perp closes
    print("  loading Binance perp 5m closes...", flush=True)
    bn = load_binance_perp_closes(syms_both,
                                   okx_swap["open_time"].min(),
                                   okx_swap["open_time"].max())
    print(f"  Binance perp: {len(bn):,} rows × {bn['symbol'].nunique()} syms", flush=True)

    # Merge: on (symbol, open_time)
    # OKX is 1h bars. Binance is 5m. Align to OKX hourly grid by filtering Binance to top-of-hour bars.
    bn["open_time"] = pd.to_datetime(bn["open_time"], utc=True)
    bn_h = bn[(bn["open_time"].dt.minute == 0)].copy()
    print(f"  Binance perp at top-of-hour: {len(bn_h):,} rows", flush=True)

    m = bn_h.merge(okx_swap, on=["symbol", "open_time"], how="inner")
    m = m.merge(okx_spot, on=["symbol", "open_time"], how="inner")
    print(f"  merged 1h: {len(m):,} rows × {m['symbol'].nunique()} syms", flush=True)

    # 4h-align (cycle grid)
    m4 = m[m["open_time"].dt.hour % 4 == 0].copy()
    print(f"  4h-aligned: {len(m4):,} rows", flush=True)

    # Compute features
    m4, feat_cols = compute_features(m4)
    print(f"  feature columns: {feat_cols}", flush=True)

    # Join target: alpha_vs_btc_realized from 51-panel
    panel = pd.read_parquet(PANEL,
        columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    m4 = m4.merge(panel, on=["symbol", "open_time"], how="left")
    n_with_target = m4["alpha_vs_btc_realized"].notna().sum()
    print(f"  with target: {n_with_target:,} of {len(m4):,} rows", flush=True)

    # IC tests
    target_col = "alpha_vs_btc_realized"
    results = []
    for c in feat_cols:
        for variant in [c, c + "_z", c + "_xsz"]:
            r = ic_test(m4, variant, target_col,
                        f"{c[:20]}{'-trail-z' if variant.endswith('_z') else '-xs-z' if variant.endswith('_xsz') else '-raw'}")
            results.append(r)

    # Output
    print(f"\n=== IC RESULTS ===", flush=True)
    print(f"  {'feature':<45} {'n':>8} {'IC':>9} {'t':>6} {'p95-null':>9} "
          f"{'beats?':>8} {'folds+':>7}", flush=True)
    for r in results:
        if "ic_pooled" not in r: continue
        print(f"  {r['feat']:<45} {r['n_valid']:>8,} "
              f"{r['ic_pooled']:>+9.5f} {r['t_stat']:>+6.2f} "
              f"{r['null_p95_abs']:>9.5f} {'YES' if r['beats_null_p95'] else 'no':>8} "
              f"{r['folds_positive']}/{r['n_folds']:>3}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "X4_cross_exchange_features.json").write_text(
        json.dumps({"results": results, "elapsed_s": round(time.time()-t0, 1)},
                   indent=2, default=str))
    print(f"\n[elapsed {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
