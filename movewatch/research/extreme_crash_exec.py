"""Execution stress for the extreme-crash strategy (docs/EXTREME_CRASH.md, "Execution stress test").

Worst fill (buy the entry minute's high, sell the exit minute's low) and entry latency (t+2..t+10),
over all 31 EXTREME fires. Consumed windows: a sensitivity measurement, not a selection.

  python3 -m research.extreme_crash_exec
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import research.btc_hedge as bh
import research.extreme_crash as ec
import research.flush_signal as fs
import research.replay as rp

LAGS = (1, 2, 3, 5, 10)
REQUIRED = ("worst_fill", "lag2", "lag3", "lag5")
GROUPS = {"test_2020_21": ("early",), "all_31": ("early", "disc", "rehearsal")}


def worst_fill(sg: dict, idx: pd.DatetimeIndex, syms: list[str], members: np.ndarray, e: int, x: int) -> np.ndarray:
    """Per member: low(exit minute) / high(entry minute) - 1, from the per-symbol 1m parquet files."""
    t0, t1 = idx[e], idx[x]
    out = np.full(len(members), np.nan)
    for j, i in enumerate(members):
        d = pd.read_parquet(sg["panel_dir"] / f"{syms[i]}.parquet", columns=["high", "low"])
        d = d[~d.index.duplicated()].loc[t0:t1].astype(np.float64)
        if t0 not in d.index or not np.isfinite(d.loc[t0, "high"]) or d.loc[t0, "high"] <= 0:
            continue
        lo = d.loc[t1, "low"] if t1 in d.index and np.isfinite(d.loc[t1, "low"]) else d["low"].dropna().iloc[-1]
        out[j] = lo / d.loc[t0, "high"] - 1.0
    return out


def run() -> dict:
    ec.configure()
    rows = []
    for w in ("early", "disc", "rehearsal"):
        ev = pd.read_parquet(fs.OUT / f"events_{w}.parquet")
        ex = ev[ev["tier"] == "EXTREME"]
        for k, g in ex.groupby("seg"):
            sg = fs.WINDOWS[w][k]
            syms, idx, C, V = fs._load(sg)
            LIQ = np.load(fs.OUT / f"liq_{w}_{k}.npy")
            keep = fs.static_keep(syms)
            for r in g.itertuples():
                t = int((r.time - idx[0]) / fs.MIN)
                u, r15 = fs.universe_at(C, V, LIQ, keep, t)
                h = u[np.argsort(r15, kind="stable")[:math.ceil(len(u) / 5)]]
                cost = np.array([2 * (bh.FEE + fs.P_STRESS * bh.slip(float(v))) for v in LIQ[h, t // 15]])
                rec = {"window": w, "time": r.time}
                for lag in LAGS:
                    e, x = t + lag, t + lag + fs.P_HOLD
                    c0 = np.asarray(C[h, e], np.float64)
                    c1 = np.asarray(C[h, x], np.float64)
                    for j in np.flatnonzero(~np.isfinite(c1)):          # last finite close in [entry, exit]
                        row = np.asarray(C[h[j], e:x + 1], np.float64)
                        f = np.flatnonzero(np.isfinite(row))
                        c1[j] = row[f[-1]] if len(f) else np.nan
                    ok = np.isfinite(c0) & (c0 > 0) & np.isfinite(c1)
                    rec[f"lag{lag}"] = float((c1[ok] / c0[ok] - 1.0 - cost[ok]).mean())
                if not abs(rec["lag1"] - getattr(r, fs.col("E1", "H"))) <= 1e-6:
                    raise SystemExit(f"SELFTEST lag1 does not reproduce E1-H at {r.time}: {rec['lag1']}")
                wf = worst_fill(sg, idx, syms, h, t + 1, t + 1 + fs.P_HOLD)
                ok = np.isfinite(wf)
                rec["worst_fill"] = float((wf[ok] - cost[ok]).mean())
                rec["worst_fill_coins"] = int(ok.sum())
                rows.append(rec)
            rp.mem(f"exec {w} seg {k}")
            del C, V, LIQ
    df = pd.DataFrame(rows).sort_values("time")
    res: dict = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "script_commit": fs.last_commit(ec.ROOT / "research" / "extreme_crash_exec.py"),
                 "n_events": int(len(df)), "selftest_lag1": "OK", "groups": {}}
    failing = []
    for gname, wins in GROUPS.items():
        s = df[df["window"].isin(wins)]
        wk = fs._weeks(s["time"])
        cells = {}
        for c in ("lag1", *[f"lag{x}" for x in LAGS[1:]], "worst_fill"):
            st = fs.cell_stats(s[c].to_numpy(), wk)
            st["wins"] = int((s[c] > 0).sum())
            cells[c] = fs._brief(st) | {"wins": st["wins"]}
            if c in REQUIRED and not (st["status"] == "OK" and st["mean"] > 0 and st["ci"][0] > 0):
                failing.append(f"{gname}:{c}")
        res["groups"][gname] = cells
    res["failing"] = failing
    res["verdict"] = "ROBUST" if not failing else "FRAGILE"
    res["events"] = [{"time": f"{r.time:%Y-%m-%d %H:%M}", "window": r.window,
                      **{c: round(100 * getattr(r, c), 2) for c in ("lag1", "lag2", "lag3", "lag5", "lag10", "worst_fill")}}
                     for r in df.itertuples()]
    (fs.OUT / "exec_stress_extreme.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1, default=str))
    return res


if __name__ == "__main__":
    run()
