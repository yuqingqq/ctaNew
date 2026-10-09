"""Short the heavy-volume pumps, with the tail cut (docs/HEAVY_FADE.md).

  python3 -m research.heavy_fade rows --window {screen,conf}   # event rows + trade cells (conf: after the freeze)
  python3 -m research.heavy_fade screen                        # consumed windows -> docs/HEAVY_FADE_FREEZE.json
  python3 -m research.heavy_fade validate                      # untouched replays 2020-01..2025-08, read once
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research.btc_hedge as bh
import research.extreme_crash as ec
import research.flush_signal as fs
import research.persistence_study as ps
import research.replay as rp
import research.signal_search as ss
from research.common import Stale, is_tradfi, panel_meta

ROOT = Path(__file__).resolve().parents[1]
OUT = rp.OUT_DIR / "heavy_fade"
DECL = ROOT / "docs" / "HEAVY_FADE.md"
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests" / "test_heavy_fade.py"
FREEZE = ROOT / "docs" / "HEAVY_FADE_FREEZE.json"
RECEIPT = OUT / "receipt_conf.json"
MIN = pd.Timedelta("1min")
_A2 = next(c for c in json.loads((ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json").read_text())["cells"] if c["id"] == "A2")
QUIET, HEAVY = float(_A2["cuts"][0]), float(_A2["cuts"][1])
ENTRY = 241
CELLS = {"S24": (1441, None), "S72": (4321, None), "S24_stop25": (1441, 0.25), "S72_stop25": (4321, 0.25)}
DESC = {f"S{h}_stop{int(s * 100)}": ((1441 if h == 24 else 4321), s) for h in (24, 72) for s in (0.15, 0.40)}
MIN_N, MIN_WEEKS, DRAWS, SEED = 100, 5, 4000, 20261009
SCREEN = ("discovery", "holdout", "val_a", "val_b")


def conf_segments() -> list[dict]:
    segs = []
    for sg in ec._early_segments() + fs._hist_segments():
        p0, p1 = sg["panel"]
        a, b = pd.Timestamp(p0, tz="UTC"), pd.Timestamp(p1, tz="UTC")
        segs.append(dict(panel=(p0, p1), panel_dir=sg["panel_dir"], alerts=f"hfv_{p0}",
                         sig=(a + pd.Timedelta(hours=28), b - pd.Timedelta(hours=73))))
    return segs


def committed_clean(p: Path) -> bool:
    return fs.committed_clean(p)


# ----------------------------------------------------------------------------- rows
def short_trade(row: np.ndarray, entry_px: float, stop: float | None) -> tuple[int, float]:
    """Exit offset into `row` (the closes from entry+1 .. exit) and the exit price."""
    if stop is not None:
        hit = np.flatnonzero(row >= entry_px * (1 + stop))          # NaN compares False
        if len(hit):
            return int(hit[0]), float(row[hit[0]])
    return len(row) - 1, float(row[-1])


def segment_rows(name: str, w: dict, sig: pd.DataFrame, st: dict) -> list[dict]:
    """Rows for one panel from its (already filtered and de-duplicated) NEW alerts."""
    rp.PANEL_DIR = w["panel_dir"]
    syms, idx, C, V = rp.load_panel(*w["panel"])
    pdir = rp.OUT_DIR / f"panel_{w['panel'][0]}_{w['panel'][1]}"
    stale = Stale(panel_meta(w["panel_dir"], pdir, syms, idx), len(syms))
    LIQ = np.load(rp.OUT_DIR / f"blocks_{w['panel'][0]}_{w['panel'][1]}_r15.npz")["LIQ"]
    T = C.shape[1]
    t_ms0 = int(idx[0].value // 1_000_000)
    df = sig.copy()
    df["t"] = ((df["time"] - idx[0]) // MIN).astype(int) - 1
    df["i"] = df["asset"].map(syms.index)
    px = np.array([float(C[i, t]) for i, t in zip(df["i"], df["t"])])
    bad = ~np.isclose(px, df["price"].to_numpy(float), rtol=1e-5, atol=0)
    st["T_RECONCILE_MISMATCH"] = st.get("T_RECONCILE_MISMATCH", 0) + int(bad.sum())
    if bad.sum():
        raise SystemExit(f"[{name}] trigger-minute reconciliation failed: {int(bad.sum())} rows")
    st["SELF_BTC"] = st.get("SELF_BTC", 0) + int((df["asset"] == "BTCUSDT").sum())
    df = df[df["asset"] != "BTCUSDT"]
    stock = np.array([bh.stock_like(a, float(C[i, t])) for a, i, t in zip(df["asset"], df["i"], df["t"])], dtype=bool)
    st["TRADFI_PRICE"] = st.get("TRADFI_PRICE", 0) + int(stock.sum())
    df = df[~stock]

    def ok_win(i, j0, j1):
        return 0 <= j0 and j1 < T and np.isfinite(C[i, j0]) and np.isfinite(C[i, j1]) and not stale.coin(i, j0, j1)

    rows, pend = [], []
    for r in df.itertuples():
        i, t, d = int(r.i), int(r.t), int(r.direction)
        rec = {"time": r.time, "asset": r.asset, "dir": d, "window": name, "t": t, "i": i, "liq": float(LIQ[i, t // 15])}
        if ok_win(i, t - 179, t + 240):
            after = float(np.asarray(V[i, t + 61:t + 241], np.float64).sum())
            before = float(np.asarray(V[i, t - 179:t + 1], np.float64).sum())
            rec["vol_persist"] = math.log10(after / before) if after > 0 and before > 0 else np.nan
        else:
            rec["vol_persist"] = np.nan
        if d == 1:
            e = t + ENTRY
            for cell, (x, stop) in {**CELLS, **DESC}.items():
                if not ok_win(i, e, t + x):
                    rec[f"r_{cell}"] = np.nan
                    continue
                p0 = float(C[i, e])
                k, p1 = short_trade(np.asarray(C[i, e + 1:t + x + 1], np.float64), p0, stop)
                rec[f"r_{cell}"] = p1 / p0 - 1.0
                rec[f"exit_{cell}"] = e + 1 + k
                rec[f"stopped_{cell}"] = bool(stop is not None and e + 1 + k < t + x)
                pend.append((len(rows), cell, r.asset, t_ms0 + e * 60_000 + 59_999, t_ms0 + (e + 1 + k) * 60_000 + 59_999))
        rows.append(rec)
    fund = bh.Funding()
    pairs = {(a, m) for _, _, a, f0, f1 in pend for m in fund.months(f0, f1)}
    st.setdefault("funding", []).append(bh.fetch_funding(pairs))
    for n, cell, a, f0, f1 in pend:
        rows[n][f"fund_{cell}"] = fund.total(a, f0, f1)
    rp.mem(f"rows {name}")
    del C, V, LIQ
    return rows


def _alerts(w: dict) -> pd.DataFrame:
    al = pd.read_parquet(rp.OUT_DIR / w["alerts"] / "alerts.parquet")
    lo, hi = (pd.Timestamp(x, tz="UTC") if isinstance(x, str) else x for x in w["sig"])
    return al[(al["kind"] == "NEW") & (al["time"] >= lo) & (al["time"] < hi)].drop_duplicates(["asset", "time"])


def build_rows(window: str) -> pd.DataFrame:
    bh.REF.update(bh.load_stock_ref())
    st: dict = {}
    if window == "screen":                              # as the signal search: de-dup per window, V seam after
        out = []
        for name in SCREEN:
            w = ss.WINDOWS[name]
            sig = _alerts(w)
            st[f"{name}_new"] = int(len(sig))
            sig = ps.dedupe_72h(sig[~sig["asset"].map(is_tradfi)]).reset_index(drop=True)
            out += segment_rows(name, w, sig, st)
        df = pd.DataFrame(out)
        v = df[df["window"].isin(["val_a", "val_b"])].sort_values("time")
        keep, last = [], {}
        for r in v.itertuples():
            lt = last.get(r.asset)
            if lt is None or r.time - lt >= pd.Timedelta(hours=72):
                keep.append(r.Index)
                last[r.asset] = r.time
        st["SEAM_DUPLICATE"] = int(len(v) - len(keep))
        df = pd.concat([df[~df["window"].isin(["val_a", "val_b"])], df.loc[keep]], ignore_index=True)
    else:
        if not (FREEZE.exists() and all(committed_clean(p) for p in (FREEZE, DECL, SCRIPT, TESTS))):
            raise SystemExit("REFUSED: freeze, declaration, builder and tests must be committed before the conf rows")
        if RECEIPT.exists():
            raise SystemExit("REFUSED: conf already validated (read once)")
        segs = conf_segments()
        missing = [s["alerts"] for s in segs if not (rp.OUT_DIR / s["alerts"] / "alerts.parquet").exists()]
        if missing:
            raise SystemExit(f"REFUSED: replays missing: {missing}")
        al = pd.concat([_alerts(s).assign(seg=k) for k, s in enumerate(segs)], ignore_index=True)
        st["new"] = int(len(al))
        al = ps.dedupe_72h(al[~al["asset"].map(is_tradfi)]).reset_index(drop=True)
        st["after_72h_dedup"] = int(len(al))
        out = []
        for k, s in enumerate(segs):
            out += segment_rows(s["alerts"], s, al[al["seg"] == k].drop(columns="seg"), st)
        df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"rows_{window}.parquet")
    st["rows"] = int(len(df))
    st["pumps"] = int((df["dir"] == 1).sum())
    st["heavy_pumps"] = int(((df["dir"] == 1) & (df["vol_persist"] > HEAVY)).sum())
    (OUT / f"rows_{window}.json").write_text(json.dumps(st, indent=1, default=str))
    print(json.dumps({k: v for k, v in st.items() if k != "funding"}, default=str))
    return df


def net(df: pd.DataFrame, cell: str, stress: int = 1) -> np.ndarray:
    """Short P&L net of costs, plus funding received by the short."""
    cost = np.array([2 * (bh.FEE + stress * bh.slip(v)) for v in df["liq"].to_numpy(float)])
    return -df[f"r_{cell}"].to_numpy(float) - cost + df[f"fund_{cell}"].to_numpy(float)


def reconcile(df: pd.DataFrame) -> dict:
    """The rebuilt screen rows must equal the signal-search rows (event set, vol_persist, S72 net)."""
    ref = pd.concat([ss.load_rows(("discovery", "holdout")), ss.load_rows(ss.VAL)], ignore_index=True)
    v = ref[ref["window"].isin(["val_a", "val_b"])].sort_values("time")
    keep, last = [], {}
    for r in v.itertuples():
        lt = last.get(r.asset)
        if lt is None or r.time - lt >= pd.Timedelta(hours=72):
            keep.append(r.Index)
            last[r.asset] = r.time
    ref = pd.concat([ref[~ref["window"].isin(["val_a", "val_b"])], ref.loc[keep]], ignore_index=True)
    a = df[df["dir"] == 1].set_index(["asset", "time"]).sort_index()
    b = ref[ref["dir"] == 1].set_index(["asset", "time"]).sort_index()
    res = {"rows": int(len(a)), "ref_rows": int(len(b)), "set_equal": bool(a.index.equals(b.index))}
    if res["set_equal"]:
        vp = a["vol_persist"].to_numpy(float), b["vol_persist"].to_numpy(float)
        res["vol_persist_mismatch"] = int((~((np.isnan(vp[0]) & np.isnan(vp[1])) | (np.abs(vp[0] - vp[1]) <= 1e-12))).sum())
        n1 = net(a.reset_index(), "S72")
        n2 = ss.net(b.reset_index(), "x4_72", False)
        res["S72_net_mismatch"] = int((~((np.isnan(n1) & np.isnan(n2)) | (np.abs(n1 - n2) <= 1e-9))).sum())
    return res


# ----------------------------------------------------------------------------- statistics
def stats(v: np.ndarray, times: pd.Series, seed: int = SEED) -> dict:
    ok = np.isfinite(v)
    v = v[ok]
    times = pd.Series(pd.DatetimeIndex(times)[ok])
    n = len(v)
    if n == 0:
        return {"status": "INSUFFICIENT", "n": 0}
    wk = ps.cluster_weeks(times).to_numpy()
    nw = len(np.unique(wk))
    base = {"n": int(n), "weeks": int(nw), "mean": float(v.mean()), "median": float(np.median(v)),
            "win": float((v > 0).mean()), "p5": float(np.percentile(v, 5)), "worst": float(v.min())}
    if n < MIN_N or nw < MIN_WEEKS:
        return {"status": "INSUFFICIENT", **base}
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    rng = np.random.default_rng(seed)
    b = np.array([v[np.concatenate([pos[j] for j in rng.integers(0, len(uw), len(uw))])].mean() for _ in range(DRAWS)])
    return {"status": "OK", **base, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "p": float((1 + (b <= 0).sum()) / (DRAWS + 1)), "drop_best": float(np.sort(v)[:-1].mean())}


def verdict(c: dict, k: int) -> str:
    if c["status"] != "OK":
        return "INSUFFICIENT"
    core = c["mean"] > 0 and c["ci"][0] > 0
    return "VALIDATED" if core and c["p"] <= 0.05 / k and c["drop_best"] > 0 else "SUGGESTIVE" if core else "FAILED"


def heavy(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["dir"] == 1) & (df["vol_persist"] > HEAVY)]


def screen() -> None:
    df = pd.read_parquet(OUT / "rows_screen.parquet")
    rec = reconcile(df)
    print("reconciliation", rec, flush=True)
    if not (rec["set_equal"] and rec["vol_persist_mismatch"] == 0 and rec["S72_net_mismatch"] == 0):
        raise SystemExit("reconciliation failed")
    h = heavy(df)
    s_win, v_win = h[h["window"].isin(["discovery", "holdout"])], h[h["window"].isin(["val_a", "val_b"])]
    res, carry = {}, []
    for cell in CELLS:
        pooled = stats(net(h, cell), h["time"])
        ms, mv = float(np.nanmean(net(s_win, cell))), float(np.nanmean(net(v_win, cell)))
        res[cell] = {"pooled": pooled, "mean_S": ms, "mean_V": mv}
        if pooled["status"] == "OK" and pooled["ci"][0] > 0 and ms > 0 and mv > 0:
            carry.append(cell)
    desc = {c: stats(net(h, c), h["time"]) for c in DESC}
    fz = {"declaration": "docs/HEAVY_FADE.md", "heavy_cut": HEAVY, "reconciliation": rec, "carried": carry,
          "k": len(carry), "screen": res, "descriptive": desc,
          "stopped_share": {c: float(h[f"stopped_{c}"].mean()) for c in (*CELLS, *DESC) if "stop" in c}}
    FREEZE.write_text(json.dumps(fz, indent=1, default=str))
    print(json.dumps({"carried": carry, "k": len(carry), "screen": res}, indent=1, default=str))


def falsifiers(times: pd.Series, k: int) -> dict:
    rng = np.random.default_rng(SEED + 7)
    n = len(times)
    planted = rng.normal(0.05, 0.10, n)
    noise = rng.normal(0.0, 0.10, n)
    noise -= noise.mean()
    out = {"planted": verdict(stats(planted, times), k), "noise": verdict(stats(noise, times), k),
           "empty": verdict(stats(np.array([]), times.iloc[:0]), k)}
    if out != {"planted": "VALIDATED", "noise": "FAILED", "empty": "INSUFFICIENT"}:
        raise SystemExit(f"FALSIFIER failed: {out}")
    return out


def validate() -> None:
    if RECEIPT.exists():
        raise SystemExit("REFUSED: conf already validated (read once)")
    if not all(committed_clean(p) for p in (FREEZE, DECL, SCRIPT, TESTS)):
        raise SystemExit("REFUSED: freeze, declaration, builder and tests must be committed")
    fz = json.loads(FREEZE.read_text())
    k = max(fz["k"], 1)
    df = pd.read_parquet(OUT / "rows_conf.parquet")
    h = heavy(df)
    fal = falsifiers(h["time"].reset_index(drop=True), k)
    cells = {}
    for cell in CELLS:
        c = stats(net(h, cell), h["time"])
        c["carried"] = cell in fz["carried"]
        if c["carried"]:
            c["verdict"] = verdict(c, k)
        c["stopped_share"] = float(h[f"stopped_{cell}"].mean()) if "stop" in cell else None
        cells[cell] = c
    pumps = df[df["dir"] == 1]
    quiet = pumps[pumps["vol_persist"] <= QUIET]
    desc = {"stops": {c: stats(net(h, c), h["time"]) for c in DESC},
            "stress3": {c: stats(net(h, c, 3), h["time"]) for c in CELLS},
            "all_pumps": {c: stats(net(pumps, c), pumps["time"]) for c in CELLS},
            "quiet_pumps_short": {c: stats(net(quiet, c), quiet["time"]) for c in CELLS},
            "per_year": {c: pd.Series(net(h, c), index=h.index).groupby(h["time"].dt.year).agg(["size", "mean", "median"])
                         .round(5).to_dict("index") for c in CELLS}}
    st = json.loads((OUT / "rows_conf.json").read_text())
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "freeze_commit": fs.last_commit(FREEZE), "script_commit": fs.last_commit(SCRIPT), "k": k,
           "rows": {kk: v for kk, v in st.items() if kk != "funding"}, "falsifiers": fal, "cells": cells, "descriptive": desc}
    RECEIPT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({kk: v for kk, v in out.items() if kk != "descriptive"}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["rows", "screen", "validate"])
    ap.add_argument("--window", choices=["screen", "conf"])
    a = ap.parse_args()
    if a.phase == "rows":
        build_rows(a.window)
    else:
        {"screen": screen, "validate": validate}[a.phase]()


if __name__ == "__main__":
    main()
