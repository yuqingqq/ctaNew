"""Add bars_since_low_xs_rank to the volaug panel -> volaug2.

Symmetric partner of bars_since_high (features_ml/regime_features.py:34):
  high_1d = close.rolling(288).max(); is_new_high = close==high_1d
  bars_since_high = (1-is_new_high).groupby(is_new_high.cumsum()).cumcount()
=> bars_since_low uses the trailing-1d *min*; same 288 window; same .shift(1)
+ cross-sectional pct-rank treatment the full-PIT builder applies to
bars_since_high (build_btc_features_111_full_pit Step 3 shift + Step 4 xs-rank).

A genuinely-new non-monotone "distance-from-recent-extreme" feature (fits the
Step-80a squared/U-shape finding). Output = volaug + bars_since_low_xs_rank,
volaug spine (incl 9 vol cols) byte-identical (LEFT-join, row-count asserted)
so Steps 76-84 stay directly comparable.

PIT: rolling-288 min includes bar t (window ends at t), then .shift(1) per
symbol so the value at row t reflects state at t-1 — exactly the same PIT
level as bars_since_high (the established convention). Audit runs on the raw
shifted bsl BEFORE the xs-rank: independent recompute exact-match + shift
proof + look-ahead IC<0.10.
"""
from __future__ import annotations
import sys, time, warnings, glob, gc
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

VOLAUG = REPO / "outputs/vBTC_features_btc_only_111_volaug/panel_btc_only_111_volaug.parquet"
KL = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_features_btc_only_111_volaug2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DAY = 288


def load_close(sym):
    fs = sorted(glob.glob(str(KL / sym / "5m" / "*.parquet")))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["open_time", "close"])
                   for f in fs], ignore_index=True)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True, errors="coerce")
    return (d.dropna(subset=["open_time"]).drop_duplicates("open_time")
              .sort_values("open_time").reset_index(drop=True))


def bsl_raw(close: pd.Series) -> pd.Series:
    """bars since last trailing-1d (288-bar) low of close — pre-shift."""
    low_1d = close.rolling(DAY).min()
    is_new_low = (close == low_1d).astype(int)
    return (1 - is_new_low).groupby(is_new_low.cumsum()).cumcount()


def main():
    print("=== Build bars_since_low_xs_rank -> volaug2 ===\n", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(VOLAUG)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    syms = sorted(panel["symbol"].unique())
    n0, c0 = len(panel), len(panel.columns)
    print(f"volaug spine: {n0:,} rows x {c0} cols, {len(syms)} syms", flush=True)

    parts, skip = [], []
    for i, s in enumerate(syms):
        d = load_close(s)
        if d is None or len(d) < 1000:
            skip.append(s)
            continue
        bsl = bsl_raw(d["close"].astype(float)).shift(1)        # PIT shift(1)
        parts.append(pd.DataFrame({"symbol": s, "open_time": d["open_time"],
                                   "bsl": bsl.values}))
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(syms)} ({time.time()-t0:.0f}s)", flush=True)
    allb = pd.concat(parts, ignore_index=True)
    allb["open_time"] = pd.to_datetime(allb["open_time"], utc=True)
    allb["symbol"] = allb["symbol"].astype("category")
    panel["symbol"] = panel["symbol"].astype("category")
    m = panel.merge(allb, on=["symbol", "open_time"], how="left")
    del allb, parts
    gc.collect()
    assert len(m) == n0, f"ROW COUNT CHANGED {len(m)} != {n0}"
    assert all(c in m.columns for c in panel.columns), "lost a spine col"
    print(f"\nmerged {len(m):,} rows (spine preserved); raw bsl non-NaN "
          f"{m['bsl'].notna().mean()*100:.1f}%, skipped {len(skip)}", flush=True)

    # ---- PIT audit on the raw shifted bsl (before xs-rank) ----
    print("\n--- PIT audit (raw shifted bsl) ---", flush=True)
    okA = True
    for sym in ["ETHUSDT", "SOLUSDT"]:
        d = load_close(sym)
        raw = bsl_raw(d["close"].astype(float))
        ref = pd.DataFrame({"open_time": d["open_time"],
                            "indep_shift1": raw.shift(1).values,
                            "unshift": raw.values})
        mm = m[m.symbol == sym][["open_time", "bsl"]].merge(
            ref, on="open_time").dropna(subset=["bsl", "indep_shift1"])
        corr1 = float(mm["bsl"].corr(mm["indep_shift1"]))
        maxd = float((mm["bsl"] - mm["indep_shift1"]).abs().max())
        cunsh = float(mm["bsl"].corr(mm["unshift"]))
        good = corr1 > 0.999 and maxd < 1e-6
        okA &= good
        print(f"  {sym}: n={len(mm):,} corr(stored,indep_SHIFT1)={corr1:.5f} "
              f"maxdiff={maxd:.1e}  corr(stored,UNSHIFTED)={cunsh:.3f}  "
              f"-> {'OK (= t-1 state, not bar t)' if good else 'MISMATCH'}",
              flush=True)

    print("\n--- look-ahead IC vs fwd alpha_beta (OOS decision cycles) ---",
          flush=True)
    folds = _multi_oos_splits(m)
    m["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(m, folds[fid])[2]
        m.loc[te.index, "fold"] = fid
    oos = m[m["fold"].between(1, 9)]
    grid = sorted(oos["open_time"].unique())[::48]
    dec = oos[oos["open_time"].isin(set(grid))]
    ics = []
    for _, g in dec.groupby("open_time"):
        v = g[["bsl", "alpha_beta"]].dropna()
        if len(v) >= 5 and v["bsl"].std() > 1e-12 and v["alpha_beta"].std() > 1e-12:
            ics.append(v["bsl"].corr(v["alpha_beta"], method="spearman"))
    mu = float(np.mean(ics)) if ics else np.nan
    okB = abs(mu) < 0.10
    print(f"  bars_since_low mean cycle-IC = {mu:+.4f}  n={len(ics)}  "
          f"{'ok (<0.10)' if okB else '<<< SUSPECT'}", flush=True)

    # ---- xs-rank per cycle (after shift; matches full-PIT Step 4) ----
    m["bars_since_low_xs_rank"] = m.groupby("open_time")["bsl"].rank(pct=True)
    m = m.drop(columns=["bsl", "fold"])
    cov = m["bars_since_low_xs_rank"].notna().mean()
    out = OUT_DIR / "panel_btc_only_111_volaug2.parquet"
    m.to_parquet(out, index=False)
    print(f"\nPIT audit: {'PASS' if (okA and okB) else 'FAIL'}", flush=True)
    print(f"bars_since_low_xs_rank non-NaN {cov*100:.1f}%  "
          f"({len(m):,} rows x {len(m.columns)} cols, +1 vs volaug)", flush=True)
    print(f"Saved: {out}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
