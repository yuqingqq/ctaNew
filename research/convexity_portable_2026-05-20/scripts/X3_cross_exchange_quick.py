"""X3 — quick cross-exchange feature test on the EXISTING HL klines cache.

Loads HL 5m klines that already exist in data/ml/cache/, aligns with Binance
panel, builds Binance-HL basis as a candidate feature, and checks IC vs
next-4h alpha_vs_btc_realized.

This is the pre-test BEFORE deciding to fetch a full 6-month HL backfill.
If even with short window we see IC > 0.02, signal worth pursuing.
If IC < 0.01, cross-exchange basis on majors carries no portable signal.

Feature candidates:
  1. basis_bps = (price_binance - price_hl) / price_binance * 10000
  2. basis_z = per-symbol trailing-24h z-score of basis_bps (PIT)
  3. basis_change_4h = basis_bps(t) - basis_bps(t-4h)
  4. basis_xs_rank per cycle

Target: next-4h alpha_vs_btc_realized (from 51-panel)
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = REPO / "data/ml/cache"
PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"


def load_hl_klines_all_syms():
    """Load all available HL 5m klines."""
    files = list(CACHE.glob("hl_klines_*_5m.parquet"))
    print(f"  HL klines files (5m): {len(files)}", flush=True)
    frames = []
    for f in files:
        sym = f.stem.replace("hl_klines_", "").replace("_5m", "")
        try:
            df = pd.read_parquet(f)
            # need close prices + open_time
            if "open_time" not in df.columns:
                # might use 'timestamp' or other
                cols = df.columns.tolist()
                if "timestamp" in cols:
                    df = df.rename(columns={"timestamp": "open_time"})
                elif "t" in cols:
                    df = df.rename(columns={"t": "open_time"})
            if "close" not in df.columns and "c" in df.columns:
                df = df.rename(columns={"c": "close"})
            df = df[["open_time", "close"]].copy()
            df["symbol"] = sym
            df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
            df = df.rename(columns={"close": "hl_close"})
            frames.append(df)
        except Exception as e:
            print(f"  err {sym}: {e}", flush=True)
    if not frames:
        return pd.DataFrame()
    hl = pd.concat(frames, ignore_index=True)
    print(f"  HL data: {len(hl):,} rows × {hl['symbol'].nunique()} syms", flush=True)
    print(f"  HL time range: {hl['open_time'].min()} → {hl['open_time'].max()}", flush=True)
    return hl


def main():
    t0 = time.time()
    print("=== X3 cross-exchange basis quick test ===\n", flush=True)

    hl = load_hl_klines_all_syms()
    if len(hl) == 0:
        print("NO HL DATA")
        return
    hl_syms = sorted(hl["symbol"].unique())
    print(f"  HL syms: {hl_syms[:5]} ... ({len(hl_syms)} total)", flush=True)

    # load Binance panel: need close + target
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[p["symbol"].isin(hl_syms)].copy()
    print(f"  Binance panel rows (overlap): {len(p):,}", flush=True)

    # Need Binance close from klines (panel doesn't have it directly)
    # Load from data/ml/test/parquet/klines/<sym>/5m/*.parquet
    KLINES = REPO / "data/ml/test/parquet/klines"
    bn_closes = []
    for sym in hl_syms:
        sd = KLINES / sym / "5m"
        if not sd.exists(): continue
        # only load recent files (those overlapping HL window: 2026-04 forward)
        files = sorted(sd.glob("*.parquet"))
        # filter to files dated 2026-04 or later
        recent_files = [f for f in files if "2026-04" in f.stem or "2026-05" in f.stem or "2026-03" in f.stem]
        if not recent_files:
            recent_files = files[-3:]  # last 3 days
        dfs = []
        for f in recent_files:
            try:
                df = pd.read_parquet(f, columns=["open_time", "close"])
                df["symbol"] = sym
                dfs.append(df)
            except Exception: pass
        if dfs:
            bn_closes.append(pd.concat(dfs, ignore_index=True))
    if not bn_closes:
        print("no Binance closes loadable for HL syms")
        return
    bn = pd.concat(bn_closes, ignore_index=True)
    bn["open_time"] = pd.to_datetime(bn["open_time"], utc=True)
    bn = bn.rename(columns={"close": "bn_close"})
    print(f"  Binance closes (5m, recent): {len(bn):,} rows", flush=True)

    # join HL + Binance closes
    m = bn.merge(hl, on=["symbol", "open_time"], how="inner")
    print(f"  merged 5m bars: {len(m):,} rows", flush=True)
    if len(m) == 0:
        print("no overlap")
        return

    # 4h-align (open_time hour % 4 == 0, minute == 0)
    m = m[(m["open_time"].dt.minute == 0) & (m["open_time"].dt.hour % 4 == 0)].copy()
    print(f"  4h-aligned bars: {len(m):,}", flush=True)
    print(f"  overlap time: {m['open_time'].min()} → {m['open_time'].max()}", flush=True)

    # compute basis
    m["basis_bps"] = (m["bn_close"] - m["hl_close"]) / m["bn_close"] * 10000.0
    m["basis_abs"] = m["basis_bps"].abs()

    # per-symbol trailing z (use available window per-sym; min 12 prior 4h bars)
    m = m.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    m["basis_mean"] = m.groupby("symbol")["basis_bps"].transform(
        lambda s: s.rolling(36, min_periods=12).mean().shift(1))
    m["basis_std"] = m.groupby("symbol")["basis_bps"].transform(
        lambda s: s.rolling(36, min_periods=12).std().shift(1))
    m["basis_z"] = (m["basis_bps"] - m["basis_mean"]) / m["basis_std"].replace(0, np.nan)

    # basis change 4h (1-bar lag of 4h basis)
    m["basis_change_4h"] = m["basis_bps"] - m.groupby("symbol")["basis_bps"].shift(1)

    # cross-sectional rank of basis_bps per cycle
    m["basis_xs_rank"] = m.groupby("open_time")["basis_bps"].rank(pct=True)

    # join target
    m = m.merge(p, on=["symbol", "open_time"], how="left")
    m = m.dropna(subset=["alpha_vs_btc_realized"])
    print(f"  with target: {len(m):,} rows × {m['symbol'].nunique()} syms\n", flush=True)

    # IC table
    target = m["alpha_vs_btc_realized"]
    feats_to_test = ["basis_bps", "basis_abs", "basis_z", "basis_change_4h",
                      "basis_xs_rank"]
    print(f"  {'feature':<22} {'IC':>10} {'n_valid':>10} {'SE':>8} {'t-stat':>8}")
    results = {}
    for f in feats_to_test:
        v = m[f]
        valid = v.notna() & target.notna()
        n = valid.sum()
        if n < 30: continue
        ic = float(v[valid].corr(target[valid]))
        se = 1.0 / np.sqrt(n)
        t = ic / se
        print(f"  {f:<22} {ic:>+10.5f} {n:>10,} {se:>8.5f} {t:>+8.2f}")
        results[f] = {"ic": round(ic, 5), "n": int(n), "se": round(se, 5), "t": round(t, 2)}

    # Per-symbol IC for the strongest aggregate feature
    best_f = max(results.keys(), key=lambda k: abs(results[k]["ic"]))
    print(f"\n  Per-symbol IC for '{best_f}':")
    per_sym_ic = {}
    for sym, g in m.groupby("symbol"):
        v = g[best_f]; t = g["alpha_vs_btc_realized"]
        valid = v.notna() & t.notna()
        if valid.sum() < 20: continue
        ic = float(v[valid].corr(t[valid]))
        per_sym_ic[sym] = {"ic": round(ic, 4), "n": int(valid.sum())}
    sorted_ic = sorted(per_sym_ic.items(), key=lambda x: x[1]["ic"])
    print(f"    {'sym':<14} {'IC':>8} {'n':>6}")
    for sym, d in sorted_ic[:5]:
        print(f"    {sym:<14} {d['ic']:>+8.3f} {d['n']:>6}")
    print(f"    ...")
    for sym, d in sorted_ic[-5:]:
        print(f"    {sym:<14} {d['ic']:>+8.3f} {d['n']:>6}")
    ics = [d["ic"] for d in per_sym_ic.values()]
    print(f"    mean IC: {np.mean(ics):+.4f}, median: {np.median(ics):+.4f}, "
          f"std: {np.std(ics):.4f}", flush=True)

    out = {
        "n_4h_bars_with_basis_and_target": int(len(m)),
        "n_syms": int(m["symbol"].nunique()),
        "time_range": [str(m["open_time"].min()), str(m["open_time"].max())],
        "feature_ICs": results,
        "best_feature": best_f,
        "per_symbol_IC_for_best": per_sym_ic,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "X3_cross_exchange_quick.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[elapsed {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
