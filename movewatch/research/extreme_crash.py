"""Extreme-crash strategy (docs/EXTREME_CRASH.md): after the median liquid coin falls >= 10% in 15 minutes,
buy the hardest-hit fifth and sell 4 h later.

The machinery is research/flush_signal.py (frozen at cb690cd), unchanged; this module only configures it
(windows, tiers, minimum, k) inside build/validate, never at import.

  python3 -m research.extreme_crash build    --window {disc,early}
  python3 -m research.extreme_crash validate --window {disc,early}

disc  = 2022-01..2025-08, the flush study's window (consumed; discovery record, descriptive)
early = 2020-01..2021-12, untouched for this question (the confirmation test, read once)
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research.flush_signal as fs
import research.replay as rp

ROOT = Path(__file__).resolve().parents[1]
DECL = ROOT / "docs" / "EXTREME_CRASH.md"
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests" / "test_extreme_crash.py"
PRIMARY = ("E1", "H")
MIN_EVENTS = 10
PLATEAU = {"EXTREME_D8": (0.08, 0.50), "EXTREME_D12": (0.12, 0.50),
           "EXTREME_S25": (0.10, 0.25), "EXTREME_S75": (0.10, 0.75)}
PLATEAU_MIN_FIRES = 3
FLAGS = ["early (2020-2021) is untouched for this question but not perfectly blind: the large crashes are public "
         "knowledge and other repo programs read 2020-2021 klines for cross-sectional alpha",
         "funding excluded; costs are a calm-market slippage model x3; real crash fills are untested",
         "disc (2022-01..2025-08) is consumed: its numbers are the discovery record, not evidence"]


def _early_segments() -> list[dict]:
    d0, d1 = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")
    segs = []
    for a in pd.date_range(d0, d1, freq="QS", inclusive="left"):
        b = a + pd.DateOffset(months=3)
        segs.append(dict(panel=(str(max(a - pd.Timedelta(days=2), d0).date()), str(min(b + pd.Timedelta(days=2), d1).date())),
                         panel_dir=fs.HIST / "usdm_1m_early",
                         own=(max(a, pd.Timestamp("2020-01-02", tz="UTC")), min(b, pd.Timestamp("2021-12-30", tz="UTC")))))
    return segs


def configure() -> None:
    """Point the frozen flush machinery at this declaration (module globals, read at call time)."""
    fs.DEPTHS = tuple(sorted(set(fs.DEPTHS) | {0.08, 0.12}))
    fs.SER = ["n_u", "m15", "m5", *[fs.scol(d) for d in fs.DEPTHS]]
    fs.TIERS = {**fs.TIERS, **PLATEAU}
    fs.MIN_EVENTS = MIN_EVENTS
    fs.K = 1
    fs.CELLS = [PRIMARY]
    fs.WINDOWS["disc"] = fs._hist_segments()
    fs.WINDOWS["early"] = _early_segments()


def require_frozen() -> None:
    bad = [p.name for p in (DECL, SCRIPT, TESTS, fs.SCRIPT) if not fs.committed_clean(p)]
    if bad:
        raise SystemExit(f"REFUSED: {', '.join(bad)} not committed clean; window early is read only after the freeze")


def receipt_path(window: str) -> Path:
    return fs.OUT / f"receipt_extreme_{window}.json"


def basket_path(C, V, LIQ, keep: np.ndarray, t: int, hold: int = fs.P_HOLD) -> np.ndarray:
    """Equal-weight E1-H basket value relative to entry, minute by minute over the hold."""
    u, r15 = fs.universe_at(C, V, LIQ, keep, t)
    h = u[np.argsort(r15, kind="stable")[:math.ceil(len(u) / 5)]]
    e = t + 1
    P = pd.DataFrame(np.asarray(C[h, e:e + hold + 1], np.float64).T).ffill().to_numpy()
    ok = np.isfinite(P[0]) & (P[0] > 0)
    return (P[:, ok] / P[0, ok] - 1.0).mean(axis=1)


def build(window: str) -> None:
    configure()
    if window == "early":
        require_frozen()
        if receipt_path(window).exists():
            raise SystemExit("REFUSED: early was already validated (read once); not rebuilding")
        man = fs.HIST / "usdm_1m_early" / "_manifest.json"
        if not man.exists():
            raise SystemExit("REFUSED: early download not finished (_manifest.json missing)")
        if json.loads(man.read_text())["counts"].get("error", 0):
            raise SystemExit("REFUSED: early download has symbols with errors; re-run the download first")
    fs.build(window)
    # ---- intra-hold path of every EXTREME basket (descriptive), reconciled to the recorded 4 h gross
    ev = pd.read_parquet(fs.OUT / f"events_{window}.parquet")
    ex = ev[ev["tier"] == "EXTREME"]
    rows = []
    for k, g in ex.groupby("seg"):
        sg = fs.WINDOWS[window][k]
        syms, idx, C, V = fs._load(sg)
        LIQ = np.load(fs.OUT / f"liq_{window}_{k}.npy")
        keep = fs.static_keep(syms)
        for r in g.itertuples():
            t = int((r.time - idx[0]) / fs.MIN)
            path = basket_path(C, V, LIQ, keep, t)
            if not abs(path[-1] - getattr(r, fs.col("E1", "H", fs.P_HOLD, "gross"))) <= 1e-6:
                raise SystemExit(f"SELFTEST basket path does not reconcile at {r.time}: {path[-1]}")
            rows.append({"time": r.time, "worst": float(path.min()), "worst_at_min": int(path.argmin())})
        rp.mem(f"{window} paths {k}")
        del C, V, LIQ
    pd.DataFrame(rows, columns=["time", "worst", "worst_at_min"]).to_parquet(fs.OUT / f"paths_{window}.parquet")
    print(json.dumps({"window": window, "EXTREME_paths": len(rows), "path_selftest": "OK"}))


def plateau_split(ev: pd.DataFrame, cc: str) -> tuple[dict, dict]:
    """All plateau variants (fires, mean) and the ones the verdict requires (>= PLATEAU_MIN_FIRES fires)."""
    allv, req = {}, {}
    for v in PLATEAU:
        x = ev.loc[ev["tier"] == v, cc]
        allv[v] = {"fires": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan")}
        if len(x) >= PLATEAU_MIN_FIRES:
            req[v] = float(x.mean())
    return allv, req


def validate(window: str) -> dict:
    configure()
    receipt = receipt_path(window)
    if window == "early":
        require_frozen()
        if receipt.exists():
            raise SystemExit("REFUSED: early receipt exists; the window is read once")
    bst = json.loads((fs.OUT / f"build_{window}.json").read_text())
    if bst["script_blob"] != fs.blob(fs.SCRIPT):
        raise SystemExit("REFUSED: research/flush_signal.py changed since this window was built")
    ev = pd.read_parquet(fs.OUT / f"events_{window}.parquet")
    pool = pd.read_parquet(fs.OUT / f"pool_{window}.parquet")
    paths = pd.read_parquet(fs.OUT / f"paths_{window}.parquet")
    fal = fs.falsifiers(pool)
    ex = ev[ev["tier"] == "EXTREME"].sort_values("time")
    cc = fs.col(*PRIMARY)
    allv, req = plateau_split(ev, cc)
    prim = fs._brief(fs.evaluate(ex[cc].to_numpy(), ex["time"], pool[cc].to_numpy(), pool["time"], req))
    wk = fs._weeks(ex["time"])
    d = {"plateau_all": allv,
         "other_cells": {f"{e}_{b}": fs._brief(fs.cell_stats(ex[fs.col(e, b)].to_numpy(), wk))
                         for e, b in (("E1", "M"), ("E2", "H"), ("E2", "M"))},
         "holds_stress": {f"E1_H_{h}_{s}": fs._brief(fs.cell_stats(ex[fs.col("E1", "H", h, s)].to_numpy(), wk))
                          for h in fs.HOLDS for s in (*fs.STRESS, "gross")},
         "per_year": ex.groupby(ex["time"].dt.year)[cc].agg(["size", "mean"]).round(5).to_dict("index"),
         "pool_mean": float(pool[cc].mean()),
         "flush_tier_E1_H": fs._brief(fs.cell_stats(ev.loc[ev["tier"] == "FLUSH", cc].to_numpy(),
                                                    fs._weeks(ev.loc[ev["tier"] == "FLUSH", "time"])))}
    pm = paths.set_index("time")
    d["events"] = [{"time": f"{r.time:%Y-%m-%d %H:%M}", "coins": int(r.n_u), "basket": int(getattr(r, "E1_H_n")),
                    "m15_pct": round(100 * (math.exp(r.m15) - 1), 2),
                    "E1_H_pct": round(100 * getattr(r, cc), 2), "E1_M_pct": round(100 * getattr(r, fs.col("E1", "M")), 2),
                    "worst_during_hold_pct": round(100 * float(pm.loc[r.time, "worst"]), 2)}
                   for r in ex.itertuples()]
    res = {"window": window,
           "label": "TEST (untouched window, read once)" if window == "early" else "DISCOVERY (consumed window; descriptive only)",
           "declaration_commit": fs.last_commit(DECL), "script_commit": fs.last_commit(SCRIPT),
           "flush_signal_commit": fs.last_commit(fs.SCRIPT), "head": fs._git("rev-parse", "HEAD").stdout.strip(),
           "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "flags": FLAGS,
           "build": {k: v for k, v in bst.items() if k != "segments"},
           "thin_universe_minutes": sum(s["THIN_UNIVERSE_minutes"] for s in bst["segments"]),
           "owned_minutes": sum(s["owned_minutes"] for s in bst["segments"]),
           "falsifiers": fal, "k": 1, "primary": {"cell": "EXTREME_E1_H_240_s3", **prim}, "descriptive": d}
    receipt.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"window": window, "fires": bst["fires"], "primary": res["primary"], "falsifiers": fal},
                     indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "validate"])
    ap.add_argument("--window", choices=["disc", "early"], required=True)
    a = ap.parse_args()
    (build if a.phase == "build" else validate)(a.window)


if __name__ == "__main__":
    main()
