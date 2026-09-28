"""What happens after a signal? Forward price paths after every NEW / REVERSAL alert of
the live (strict, logic v4) setting on the Binance-USDM replay panel.

Descriptive, not a validated strategy:
  - entry at the alert's minute close (the alert price), and 1 / 5 minutes later
    (human reaction time); returns are DIRECTION-ADJUSTED (+ = the move continued)
  - raw and market-adjusted (minus the median liquid coin over the same span)
  - means with UTC-day-clustered bootstrap 95% CIs (1,000 draws); a window with < 5
    complete days gets a point estimate only (CLAUDE.md rule 8)
  - gross of costs: a round trip on these coins costs roughly 0.1-0.3% (taker fees
    + slippage in fast markets), more on thin books
Windows: calibration 08-02..08-31, 09-01..09-19 (read during development), and
09-20..09-27 (consumed: QNT) reported separately.

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.post_signal
"""
from __future__ import annotations

import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "ARROW_DEFAULT_MEMORY_POOL": "system"}.items():
    os.environ.setdefault(_k, _v)

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from research.common import Stale, dir_simple, is_tradfi, mkt_median, panel_meta  # noqa: E402
from research.replay import OUT_DIR, PANEL_DIR, load_panel, mem  # noqa: E402

FWD = (5, 15, 60, 240, 1440)
WINDOWS = {"aug": ("2026-08-02", "2026-09-01"), "sep01_19": ("2026-09-01", "2026-09-20"),
           "sep20_27_consumed": ("2026-09-20", "2026-09-28")}
RNG = np.random.default_rng(20260928)


def forward(al: pd.DataFrame, C, LIQ, syms: list[str], delay: int, stale: Stale) -> pd.DataFrame:
    """v2 (review 2026-09-28): market median excludes the coin and frozen coins; windows with
    frozen (post-delisting) bars -> NaN."""
    T = C.shape[1]
    rows = []
    for a in al.itertuples():
        i = syms.index(a.asset)
        t0 = int(a.t) + delay
        if t0 >= T:
            continue
        c = np.asarray(C[i, t0: min(T, t0 + 1441)], np.float64)
        p0 = c[0]
        if not np.isfinite(p0):
            continue
        d = int(a.direction)
        rec = {"time": a.time, "day": a.time.floor("D"), "asset": a.asset, "kind": a.kind, "dir": d,
               "horizon": a.horizon if isinstance(a.horizon, str) else "", "move_pct": a.move_pct,
               "alert_px": a.price, "entry_px": p0}
        for k in FWD:
            if k < len(c) and np.isfinite(c[k]) and not stale.coin(i, t0, t0 + k):
                r = d * np.log(c[k] / p0)
                mk = mkt_median(C, LIQ, t0, t0 + k, stale, exclude=i)
                rec[f"r{k}"] = r
                rec[f"x{k}"] = r - d * (mk if np.isfinite(mk) else 0.0)
            else:
                rec[f"r{k}"] = rec[f"x{k}"] = np.nan
        if len(c) > 240 and not stale.coin(i, t0, t0 + 240):
            path = d * np.log(c[:241] / p0)
            rec["mfe240"], rec["mae240"] = np.nanmax(path), np.nanmin(path)
        else:
            rec["mfe240"] = rec["mae240"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def boot_mean(df: pd.DataFrame, col: str, draws: int = 1000):
    """Arithmetic mean of SIMPLE returns (exp(log) - 1), day-clustered bootstrap CI."""
    x = df[["day", col, "dir"]].dropna().copy()
    if x.empty:
        return None
    x[col] = dir_simple(x[col].to_numpy(), x["dir"].to_numpy())
    days = x["day"].unique()
    point = float(x[col].mean())
    if len(days) < 5:
        return {"mean": point, "ci": None, "n": int(len(x)), "days": int(len(days))}
    g = {d: v[col].to_numpy() for d, v in x.groupby("day")}
    ms = []
    for _ in range(draws):
        pick = RNG.choice(days, size=len(days), replace=True)
        v = np.concatenate([g[d] for d in pick])
        ms.append(v.mean())
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return {"mean": point, "ci": [float(lo), float(hi)], "n": int(len(x)), "days": int(len(days))}


def table(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df)), "hit_rate_1h": round(float((df["r60"] > 0).mean()), 3) if len(df) else None}
    for k in FWD:
        out[f"r{k}"] = boot_mean(df, f"r{k}")
        out[f"x{k}"] = boot_mean(df, f"x{k}")
    med = lambda c: float(np.nanmedian(dir_simple(df[c].to_numpy(), df["dir"].to_numpy()))) if df[c].notna().any() else None  # noqa: E731
    out["mfe240_median"], out["mae240_median"] = med("mfe240"), med("mae240")
    out["r60_median"], out["r1440_median"] = med("r60"), med("r1440")
    return out


def main() -> None:
    syms, idx, C, V = load_panel("2026-08-01", "2026-09-28")
    LIQ = np.load(OUT_DIR / "blocks_2026-08-01_2026-09-28_r15.npz")["LIQ"]
    al = pd.read_parquet(OUT_DIR / "cal_v4" / "alerts.parquet")
    al = al[al["kind"].isin(["NEW", "REVERSAL"]) & ~al["asset"].map(is_tradfi)].copy()
    stale = Stale(panel_meta(PANEL_DIR, OUT_DIR / "panel_2026-08-01_2026-09-28", syms, idx), len(syms))
    mem("loaded")
    res, frames = {}, {}
    for delay in (0, 1, 5):
        f = forward(al, C, LIQ, syms, delay, stale)
        frames[delay] = f
        for wname, (a, b) in WINDOWS.items():
            w = f[(f["time"] >= pd.Timestamp(a, tz="UTC")) & (f["time"] < pd.Timestamp(b, tz="UTC"))]
            for kind in ("NEW", "REVERSAL"):
                for dname, dsel in (("pump", 1), ("dump", -1)):
                    x = w[(w["kind"] == kind) & (w["dir"] == dsel)]
                    res[f"delay{delay}|{wname}|{kind}|{dname}"] = table(x)
                x = w[(w["kind"] == "NEW")]
                for grp, hs in (("short(1m-15m)", ("1m", "5m", "15m")), ("long(1h-24h)", ("1h", "4h", "24h"))):
                    res[f"delay{delay}|{wname}|NEW|{grp}"] = table(x[x["horizon"].isin(hs)])
        mem(f"delay {delay}")
    out = OUT_DIR / "post_signal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    frames[0].to_parquet(out / "signals_delay0.parquet")
    frames[1].to_parquet(out / "signals_delay1.parquet")
    print("rows", {d: len(f) for d, f in frames.items()})


if __name__ == "__main__":
    main()
