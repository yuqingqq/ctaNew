"""Extreme-pump mirror of the extreme-crash strategy (docs/EXTREME_PUMP.md).

Question: after >= half of the liquid Binance USDT perps RISE >= X within 15 minutes, does shorting the
hardest-pumped fifth for 4 h pay? Panels, universe and LIQ are research/flush_signal.py's (unchanged).

  python3 -m research.extreme_pump calibrate                  # outcome-blind: fire counts per threshold
"""
from __future__ import annotations

import argparse
import json
import math
import warnings

import numpy as np
import pandas as pd

import research.flush_signal as fs
import research.replay as rp

UPS = (0.03, 0.05, 0.07, 0.10, 0.15)


def ucol(x: float) -> str:
    return f"u{int(round(x * 100))}"


USER = ["n_u", "m15", *[ucol(x) for x in UPS]]


def up_series(C, V, LIQ, keep: np.ndarray, j0: int, j1: int) -> dict[str, np.ndarray]:
    """Per minute t in [j0, j1): universe size (the flush detector's predicate), median 15-min log return,
    and the share of U(t) with 15-minute return >= +x."""
    out = {c: np.full(j1 - j0, np.nan) for c in USER}
    lg = {x: math.log(1 + x) for x in UPS}
    for s in range(j0, j1, fs.CHUNK):
        e = min(s + fs.CHUNK, j1)
        cur = np.asarray(C[:, s:e], np.float64)
        vv = np.asarray(V[:, s - fs.W15 + 1:e], np.float64)
        cv = np.zeros((vv.shape[0], vv.shape[1] + 1))
        np.cumsum(vv, axis=1, out=cv[:, 1:])
        vol15 = cv[:, fs.W15:] - cv[:, :-fs.W15]
        del vv, cv
        with np.errstate(all="ignore"):
            r15 = np.log(cur / np.asarray(C[:, s - fs.W15:e - fs.W15], np.float64))
        u = keep[:, None] & (LIQ[:, np.arange(s, e) // 15] >= fs.LIQ_MIN) & np.isfinite(r15) & (vol15 > 0)
        n = u.sum(axis=0)
        ok = n >= fs.N_MIN
        sl = slice(s - j0, e - j0)
        out["n_u"][sl] = n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["m15"][sl] = np.where(ok, np.nanmedian(np.where(u, r15, np.nan), axis=0), np.nan)
        for x in UPS:
            out[ucol(x)][sl] = np.where(ok, (u & (r15 >= lg[x])).sum(axis=0) / np.maximum(n, 1), np.nan)
        del cur, r15, u, vol15
    return out


def calibrate() -> None:
    """Outcome-blind: on the discovery window (2022-01..2025-08), count 24 h-refractory fires per threshold,
    and how many fall within 24 h after a crash-side WATCH minute (median coin down >= 3% in 15 min)."""
    segs = fs._hist_segments()
    parts = []
    for k, sg in enumerate(segs):
        syms, idx, C, V = fs._load(sg)
        LIQ = np.load(fs.OUT / f"liq_hist_{k}.npy")
        keep = fs.static_keep(syms)
        j0, j1 = idx.get_loc(sg["own"][0]), idx.get_loc(sg["own"][1])
        d = pd.DataFrame(up_series(C, V, LIQ, keep, j0, j1))
        d.insert(0, "time", idx[j0:j1])
        parts.append(d)
        rp.mem(f"calibrate {k + 1}/{len(segs)}")
        del C, V, LIQ
    ser = pd.concat(parts, ignore_index=True)
    ser.to_parquet(fs.OUT / "pump_series_disc.parquet")
    down = pd.concat([pd.read_parquet(fs.OUT / f"series_hist_{k}.parquet") for k in range(len(segs))])
    crash = np.sort(down.loc[down["owned"] & (down["s3"] >= 0.5), "time"].dt.tz_convert(None).to_numpy())
    tn = ser["time"].dt.tz_convert(None).to_numpy()
    days = ser.groupby(ser["time"].dt.date)
    print(f"minutes {len(ser)}, days {days.ngroups}")
    for share in (0.25, 0.5):
        for x in UPS:
            f = fs.fire_times(tn, (ser[ucol(x)] >= share).to_numpy())
            after = sum(bool(len(crash)) and np.any((crash <= t) & (crash > t - np.timedelta64(1440, "m"))) for t in f)
            print(f"share >= {share:.0%}, up >= {x:.0%}: fires {len(f):4d} | within 24 h after a crash-side WATCH {after:3d} "
                  f"| independent {len(f) - after:3d}")
    peak = days["u5"].max()
    bins, lab = [0, .01, .05, .10, .25, .50, 1.01], ["<1%", "1-5%", "5-10%", "10-25%", "25-50%", ">=50%"]
    print("daily peak share up >= 5% in 15 min:", pd.cut(peak, bins, right=False, labels=lab).value_counts().reindex(lab).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["calibrate"])
    a = ap.parse_args()
    {"calibrate": calibrate}[a.phase]()


if __name__ == "__main__":
    main()
