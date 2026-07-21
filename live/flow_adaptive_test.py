"""Necessary test — does the ADAPTIVE rule beat zero under HONEST (regime-aware) inference?

The naive ρ=0.8 (SE~0.16) assumes windows are independent; they aren't (only ~2-3 regimes in 3.5y),
so effective-N ≈ #regimes. Two honest tests via MOVING-BLOCK bootstrap (block length L ≈ 1 regime,
so the resample preserves the regime structure and the CI reflects the true effective-N):
  (A) block-bootstrap CI on ρ(IC_t, IC_{t+1})              -> is the persistence even precisely estimated?
  (B) block-bootstrap CI on the WALK-FORWARD adaptive edge -> the tradeable quantity:
        realized_i = sign(IC_{i-1}) * IC_i   (trade the trailing-window sign, PIT, no look-ahead)
        report overall + BOTH-ERA + vs static (fixed-sign) baseline.
Adaptive is real only if (B) is CI-off-zero BOTH eras AND above the sub-cost bar. Prior: fails effective-N.
"""
import os, glob
import numpy as np, pandas as pd
os.environ.setdefault("MALLOC_ARENA_MAX", "2"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys; sys.path.insert(0, "/home/yuqing/ctaNew")
from live.flow_harness import partial_xsic
from live.flow_iter2_vol import _load_sym, RET, VOL

FEAT = "signed_pressure_5min"
CUT = pd.Timestamp("2025-10-01", tz="UTC")


def window_series(D, tgt, W, min_ts=20):
    bt = D["bar_time"]; t0 = bt.min()
    wid = ((bt - t0).dt.days // W).to_numpy()
    ics, starts = {}, {}
    for w in np.unique(wid):
        ic = partial_xsic(D, FEAT, RET + VOL, tgt, row_mask=(wid == w))
        if len(ic) >= min_ts:
            ics[int(w)] = float(ic.mean())
            starts[int(w)] = t0 + pd.Timedelta(days=int(w) * W)
    idx = sorted(ics)
    return np.array([ics[i] for i in idx]), pd.DatetimeIndex([starts[i] for i in idx])


def _blocks(N, L, rng):
    nb = int(np.ceil(N / L)); smax = max(N - L, 0)
    idx = np.concatenate([np.arange(s, s + L) for s in rng.integers(0, smax + 1, nb)])
    return idx[idx < N][:N]


def mbb(x, L, stat, n=3000, seed=1):
    x = np.asarray(x, float); N = len(x)
    if N < 4 or L >= N:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = [stat(x[_blocks(N, L, rng)]) for _ in range(n)]
    vals = [v for v in vals if np.isfinite(v)]
    return tuple(np.nanpercentile(vals, [2.5, 50, 97.5])) if vals else (np.nan,) * 3


def rho_stat(x):
    return np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 3 and x[:-1].std() > 0 else np.nan


def main():
    syms = sorted(p.split("/")[-1][:-8] for p in glob.glob(
        "/home/yuqing/ctaNew/data/ml/cache/research/flow_slim_v3/*.parquet"))
    D = pd.concat([x for x in (_load_sym(s) for s in syms) if x is not None], ignore_index=True)
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    print(f"panel {len(D):,} rows | {D.bar_time.min().date()}..{D.bar_time.max().date()}")
    print("block length L ~ 1 regime (12x30d / 6x60d); CI = moving-block bootstrap (regime-aware)\n")

    for tgt in ["fwd_5m", "fwd_1h"]:
        for W, L in [(30, 12), (60, 6)]:
            x, dt = window_series(D, tgt, W)
            if len(x) < 6:
                print(f"### {tgt} W={W}d: too few windows ({len(x)})"); continue
            rho = rho_stat(x); rlo, rmed, rhi = mbb(x, L, rho_stat)
            # walk-forward adaptive realized edge (trade trailing sign)
            adr = np.sign(x[:-1]) * x[1:]
            addt = dt[1:]
            a_lo, a_med, a_hi = mbb(adr, L, np.nanmean)
            s_lo, s_med, s_hi = mbb(x[1:], L, np.nanmean)    # static fixed-sign baseline = mean IC
            oos = adr[addt < CUT]; rec = adr[addt >= CUT]
            o_lo, _, o_hi = mbb(oos, min(L, max(len(oos) // 2, 2)), np.nanmean)
            r_lo, _, r_hi = mbb(rec, min(L, max(len(rec) // 2, 2)), np.nanmean)
            print(f"### {tgt}  W={W}d  N={len(x)}  (OOS {len(oos)} / REC {len(rec)} windows)")
            print(f"  rho={rho:+.2f}  block-CI [{rlo:+.2f},{rhi:+.2f}]   (naive SE claimed ~{1/np.sqrt(len(x)-1):.2f})")
            print(f"  ADAPTIVE realized IC = {np.nanmean(adr):+.4f}  block-CI [{a_lo:+.4f},{a_hi:+.4f}]"
                  f"  vs STATIC {np.nanmean(x[1:]):+.4f} [{s_lo:+.4f},{s_hi:+.4f}]")
            print(f"  both-era adaptive: OOS {np.nanmean(oos):+.4f} [{o_lo:+.4f},{o_hi:+.4f}] | "
                  f"REC {np.nanmean(rec):+.4f} [{r_lo:+.4f},{r_hi:+.4f}]", flush=True)
        print()
    print("read: adaptive is REAL only if its realized-edge block-CI is off-zero BOTH eras and above cost.")
    print("ADAPTIVETESTDONE", flush=True)


if __name__ == "__main__":
    main()
