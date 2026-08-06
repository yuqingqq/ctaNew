"""iter11: does the "illiquidity premium" (illiquid drifts up) actually HOLD, or is it size/vol/reversal
in disguise + survivorship?

Daily cross-sectional, from the aggTrade flow files (they carry dollar volume for the SIZE control).
Tests:
  (1) RAW IC of kyle_lambda / vpin vs fwd 1d/3d/5d (reconfirm iter9).
  (2) PARTIAL IC controlling [trailing-5d return (reversal/momentum), daily realized vol, log dollar volume
      (size)] — is there an illiquidity premium BEYOND the known factors?  <-- the decisive test.
  (3) MATURITY split: does the partial premium hold in long-listed names (survivorship-robust) or only in
      the newest/thinnest survivors?
Both eras, block-bootstrap CI. A real premium must clear (2) both-era AND (3) in mature names.
Run:  python3 -m live.emergent_iter11_validate
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from live.flow_harness import CUT, partial_xsic, xsic
from live.emergent_harness import AGG, block_ci

MIN_BARS = 100
BLK = {"f1": 3, "f3": 7, "f5": 10}


def daily_from_agg(path: str) -> pd.DataFrame | None:
    sym = path.split("/")[-1].replace("flow_", "").replace(".parquet", "")
    try:
        d = pd.read_parquet(path, columns=["kyle_lambda", "vpin", "total_volume", "last_price", "vwap"])
    except Exception:
        return None
    d.index = pd.to_datetime(d.index, utc=True)
    d["dv"] = d["total_volume"] * d["vwap"]
    d["lr"] = np.log(d["last_price"]).diff()
    day = d.index.floor("1D")
    g = d.groupby(day)
    daily = pd.DataFrame({
        "kyle": g["kyle_lambda"].mean(), "vpin": g["vpin"].mean(),
        "dvol": g["dv"].sum(), "rv": g["lr"].std(),
        "close": g["last_price"].last(), "n": g["lr"].size()})
    daily = daily[daily["n"] >= MIN_BARS]
    if len(daily) < 30:
        return None
    daily["symbol"] = sym
    return daily.reset_index(names="day")


def build() -> pd.DataFrame:
    files = sorted(glob.glob(f"{AGG}/flow_*.parquet"))
    frames = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for f in as_completed({ex.submit(daily_from_agg, p): p for p in files}):
            r = f.result()
            if r is not None:
                frames.append(r)
    D = pd.concat(frames, ignore_index=True).sort_values(["symbol", "day"])
    g = D.groupby("symbol")
    D["lr"] = g["close"].transform(lambda s: np.log(s).diff())
    D["tr5"] = g["lr"].transform(lambda s: s.rolling(5).sum())
    gl = D.groupby("symbol")["lr"]
    D["f1"] = gl.shift(-1)
    D["f3"] = sum(gl.shift(-i) for i in (1, 2, 3))
    D["f5"] = sum(gl.shift(-i) for i in (1, 2, 3, 4, 5))
    D["ldvol"] = np.log(D["dvol"].clip(lower=1))
    D["hist"] = g["day"].transform("size")   # #daily obs = maturity proxy
    D["bar_time"] = pd.to_datetime(D["day"], utc=True)
    return D.dropna(subset=["kyle", "tr5", "rv", "ldvol"])


def line(D, feat, tgt, ctrls, mask, label):
    raw = xsic(D, feat, tgt, row_mask=mask)
    par = partial_xsic(D, feat, ctrls, tgt, row_mask=mask)
    ra, rl, ru = block_ci(raw, block_days=BLK[tgt])
    pa, pl, pu = block_ci(par, block_days=BLK[tgt])
    ps = "*" if (pl > 0 or pu < 0) else " "
    print(f"    {label:<22}{tgt}: raw {ra:+.4f} | partial(rev,vol,size) {pa:+.4f}[{pl:+.4f},{pu:+.4f}]{ps}",
          flush=True)


def main():
    D = build()
    ctrls = ["tr5", "rv", "ldvol"]
    print(f"daily rows {len(D):,} | {D['symbol'].nunique()} syms | controls = {ctrls}\n", flush=True)
    masks = {"OOS": (D["bar_time"] < CUT).to_numpy(), "REC": (D["bar_time"] >= CUT).to_numpy()}
    for feat in ("kyle", "vpin"):
        print(f"=== {feat} ===", flush=True)
        for era in ("OOS", "REC"):
            for tgt in ("f1", "f3", "f5"):
                line(D, feat, tgt, ctrls, masks[era], f"{era} all")
        # maturity split (survivorship proxy): mature = top-half by history, on the 3d horizon
        med = D["hist"].median()
        mat = (D["hist"] >= med).to_numpy()
        for era in ("OOS", "REC"):
            line(D, feat, "f3", ctrls, masks[era] & mat, f"{era} MATURE")
        print("", flush=True)
    print("VERDICT logic: a real illiquidity premium must keep partial(rev,vol,size) CI off-0, both eras, "
          "AND survive in MATURE names. If partial collapses vs raw -> it was size/vol/reversal, not a premium.",
          flush=True)


if __name__ == "__main__":
    main()
