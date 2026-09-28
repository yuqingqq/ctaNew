"""Replay the live detector (movewatch.signals + movewatch.alerts) over a 1m panel.

Panel: Binance USDM 1m klines, one symbol = one asset (single venue). Replay
evaluates at minute closes, so replay detection times are an UPPER bound on live
detection time (live evaluates rolling windows every second; up to 59 s earlier).

Parity with live (movewatch/engine.py):
  - baselines refreshed every `refresh_min` minutes from minutes
    [t0 - 1440, t0 - 60) via the SAME signals.baselines_from_minutes()
  - liquidity = 24h USD volume at the last refresh (live: vol24h at refresh)
  - market median = median r_h over instruments with liq >= market_liq_usd
  - alert episodes via the SAME alerts.AlertEngine

MEMORY (hard requirement: run inside the default research_slot cap of 4 GiB
address space with a buffer):
  - the (n x T) close/volume panel lives in disk-backed .npy memmaps
    (data/replay/panel_<start>_<end>/), never as in-RAM copies;
  - returns / market medians / candidates are computed in time chunks;
  - candidates are pre-filtered at the grid's MINIMUM multipliers (floor, z, v)
    and the fixed liquidity floor -- lossless for every grid point;
  - mem() logs RSS / peak / vsize after every stage and aborts cleanly when RSS
    exceeds --mem-budget-gib (default 2.0 = half the slot cap).
  Launch:  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh <core> -- python3 -m research.replay ...

Big-move events (ex-post, for scoring only -- never used by the detector):
  pump: C_t / min(C over trailing 24h) >= 1 + pump_pct   (default 30%)
  dump: C_t / max(C over trailing 24h) <= 1 - dump_pct   (default 25%)
  contiguous qualifying minutes (merged across gaps < 6h) form one event;
  start = the trailing-24h extreme before the event's peak/trough.
  captured = log(peak / P_first_alert) / log(peak / P_start): the share of the
  move still AHEAD when the first same-direction alert fired (1 = at the start,
  0 = at the peak; undetected -> NaN).
"""
from __future__ import annotations

import os

# Before numpy / pyarrow load: single-threaded math, system allocator for Arrow
# (jemalloc/mimalloc reserve large virtual ranges that count against prlimit --as).
for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "ARROW_DEFAULT_MEMORY_POOL": "system"}.items():
    os.environ.setdefault(_k, _v)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from types import SimpleNamespace  # noqa: E402

from movewatch.alerts import AlertCfg, AlertEngine, govern  # noqa: E402
from movewatch.signals import DEFAULT_RULES, H_NAME, HORIZONS, Rules, baselines_from_minutes, fire  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL_DIR = ROOT / "data" / "hist" / "usdm_1m"
OUT_DIR = ROOT / "data" / "replay"
STABLES = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "BUSDUSDT", "DAIUSDT", "USDEUSDT",
           "EURUSDT", "USD1USDT", "XUSDUSDT", "BFUSDUSDT", "PAXGUSDT", "XAUTUSDT", "RLUSDUSDT"}

GRID = {"floor_mult": [0.8, 1.0, 1.25, 1.5], "z_mult": [0.75, 1.0, 1.25], "v_mult": [0.67, 1.0, 1.5],
        "liq_min": [2e6]}
BUDGET = {"new_per_day": 12.0, "messages_per_day": 40.0}   # declared before any result was seen
MEM_BUDGET_GIB = 2.0


# ----------------------------------------------------------------------------- memory guard
def mem(stage: str) -> dict:
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmRSS", "VmHWM", "VmSize")):
                d[line.split(":")[0]] = int(line.split()[1]) / 1048576.0   # kB -> GiB
    print(f"[mem] {stage}: rss {d['VmRSS']:.2f} GiB · peak {d['VmHWM']:.2f} GiB · vsize {d['VmSize']:.2f} GiB",
          flush=True)
    if d["VmRSS"] > MEM_BUDGET_GIB:
        raise SystemExit(f"memory guard: rss {d['VmRSS']:.2f} GiB > budget {MEM_BUDGET_GIB} GiB at '{stage}'")
    return d


# ----------------------------------------------------------------------------- panel (memmap)
def load_panel(start: str, end: str):
    """(syms, idx, C, V): C/V are read-only float32 memmaps of shape (n, T).
    Built once, streaming one symbol at a time into preallocated files."""
    pdir = OUT_DIR / f"panel_{start}_{end}"
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    if not (pdir / "done").exists():
        pdir.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in PANEL_DIR.glob("*.parquet") if p.stem not in STABLES)
        n, T = len(files), len(idx)
        C = np.lib.format.open_memmap(pdir / "C.npy", mode="w+", dtype=np.float32, shape=(n, T))
        V = np.lib.format.open_memmap(pdir / "V.npy", mode="w+", dtype=np.float32, shape=(n, T))
        syms, status = [], {}
        for k, p in enumerate(files):
            df = pd.read_parquet(p, columns=["close", "quote_volume"])
            pos = ((df.index - idx[0]) // pd.Timedelta("1min")).to_numpy()
            ok = (pos >= 0) & (pos < T)
            row_c = np.full(T, np.nan, np.float32)
            row_v = np.zeros(T, np.float32)
            row_c[pos[ok]] = df["close"].to_numpy(np.float32)[ok]
            row_v[pos[ok]] = np.nan_to_num(df["quote_volume"].to_numpy(np.float32)[ok])
            if ok.sum() < 1440:                 # < 1 day in range: excluded, recorded as a status
                row_c[:] = np.nan
                status[p.stem] = "excluded_lt_1day"
            else:
                status[p.stem] = "ok"
            C[k], V[k] = row_c, row_v
            syms.append(p.stem)
            del df
        C.flush(); V.flush()
        del C, V
        (pdir / "syms.json").write_text(json.dumps(syms))
        (pdir / "status.json").write_text(json.dumps(status))
        (pdir / "done").write_text("ok")
    syms = json.loads((pdir / "syms.json").read_text())
    C = np.load(pdir / "C.npy", mmap_mode="r")
    V = np.load(pdir / "V.npy", mmap_mode="r")
    return syms, idx, C, V


# ----------------------------------------------------------------------------- features
def compute_blocks(C, V, refresh_min: int = 15):
    """Baselines per refresh block b (minutes [b*R, (b+1)*R)), from [b*R-1440, b*R-60).
    LIQ[:, b] = USD volume over [b*R-1440, b*R) via per-block sums (no n x T cumsum)."""
    n, T = C.shape
    R = refresh_min
    nb = (T + R - 1) // R
    SIG = np.full((n, nb), np.nan, np.float32)
    V1M = np.full((n, nb), np.nan, np.float32)
    S = np.zeros((n, nb), np.float64)
    for b in range(nb):
        S[:, b] = V[:, b * R:(b + 1) * R].sum(axis=1, dtype=np.float64)
    cs = np.zeros((n, nb + 1), np.float64)
    np.cumsum(S, axis=1, out=cs[:, 1:])
    w = 1440 // R
    bb = np.arange(nb)
    LIQ = (cs[:, bb] - cs[:, np.maximum(bb - w, 0)]).astype(np.float32)
    del S, cs
    for b in range(nb):
        t0 = b * R
        a, z = max(0, t0 - 1440), t0 - 60
        if z - a >= 121:
            s, v, _ = baselines_from_minutes(np.asarray(C[:, a:z], np.float64), np.asarray(V[:, a:z], np.float64))
            SIG[:, b], V1M[:, b] = s, v
    return SIG, V1M, LIQ


def candidate_table(C, V, SIG, V1M, LIQ, rules: Rules, refresh_min: int, pre_floor: float = 1.0,
                    pre_z: float = 1.0, pre_v: float = 1.0, chunk: int = 4096) -> pd.DataFrame:
    """Rows (i, t, h, e, r, z, vm, liq, mkt) that pass the trigger predicate at multipliers
    (pre_floor, pre_z, pre_v) and the liquidity floor. With the grid's MINIMUM multipliers
    this is a lossless superset for every grid point. Computed in time chunks."""
    n, T = C.shape
    R = refresh_min
    parts = []
    for h in HORIZONS:
        hm = h // 60
        rule = rules.horizons[h]
        for a in range(hm, T, chunk):
            b = min(T, a + chunk)
            lo = a - hm
            with np.errstate(divide="ignore", invalid="ignore"):
                L = np.log(np.asarray(C[:, lo:b], np.float64))
            r = (L[:, hm:] - L[:, :-hm]).astype(np.float32)            # times a..b-1
            del L
            blk = np.arange(a, b) // R
            liq = LIQ[:, blk]
            mk = np.where((liq >= rules.market_liq_usd) & np.isfinite(r), r, np.nan)
            cnt = np.isfinite(mk).sum(axis=0)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m = np.nanmedian(mk, axis=0)
            m = np.where(cnt >= 10, m, 0.0).astype(np.float32)
            del mk
            e = r - m[None, :]
            with np.errstate(invalid="ignore"):
                sel = (np.abs(e) >= pre_floor * rule.floor) & (liq >= rules.liq_min_usd)
            ii, jj = np.nonzero(sel)
            del sel
            if ii.size:
                tt = a + jj
                bk = tt // R
                cv = np.cumsum(np.asarray(V[:, lo:b], np.float64), axis=1)   # chunk-relative
                trel = tt - lo                                               # >= hm
                volh = cv[ii, trel] - np.where(trel - hm >= 0, cv[ii, np.maximum(trel - hm, 0)], 0.0)
                # cv[t] includes V[lo..t]; subtracting cv[t-hm] leaves V[t-hm+1..t]
                del cv
                sig = SIG[ii, bk]
                v1m = V1M[ii, bk]
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = e[ii, jj] / (sig * math.sqrt(hm))
                    vm = volh / (v1m * hm)
                    keep = (np.abs(z) >= pre_z * rule.zmin) & (vm >= pre_v * rule.vmin)
                if keep.any():
                    parts.append(pd.DataFrame({
                        "i": ii[keep].astype(np.int32), "t": tt[keep].astype(np.int32), "h": np.int32(h),
                        "e": e[ii[keep], jj[keep]], "r": r[ii[keep], jj[keep]],
                        "z": z[keep].astype(np.float32), "vm": vm[keep].astype(np.float32),
                        "liq": liq[ii[keep], jj[keep]], "mkt": m[jj[keep]],
                    }))
            del r, e, liq
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["i", "t", "h", "e", "r", "z", "vm", "liq", "mkt"])


def triggers(cand: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    """Apply the exact live predicate (signals.fire) to the candidate table."""
    keep = np.zeros(len(cand), bool)
    for h, rule in rules.horizons.items():
        sel = (cand["h"] == h).to_numpy()
        if not sel.any():
            continue
        c = cand[sel]
        keep[sel] = fire(c["e"].to_numpy(), c["z"].to_numpy(), c["vm"].to_numpy(),
                         c["liq"].to_numpy(), rule, rules.liq_min_usd)
    tr = cand[keep].copy()
    tr["dir"] = np.sign(tr["e"]).astype(np.int8)
    tr["absr"] = tr["r"].abs()
    # per (asset, minute): the fired horizon with the largest raw |r| sets the anchor
    tr = tr.sort_values(["i", "t", "absr"], ascending=[True, True, False]).drop_duplicates(["i", "t"])
    return tr


# ----------------------------------------------------------------------------- alert run
def run_alerts(tr: pd.DataFrame, C, syms: list[str], idx: pd.DatetimeIndex, acfg: AlertCfg) -> pd.DataFrame:
    eng = AlertEngine(acfg)
    ts0 = idx[0].timestamp()
    out = []
    T = C.shape[1]
    for i, g in tr.groupby("i", sort=True):
        sym = syms[i]
        row_c = np.asarray(C[i], np.float64)
        trig = {int(t): row for t, row in zip(g["t"].to_numpy(), g.itertuples(index=False))}
        tlist = sorted(trig)
        k = 0
        t = tlist[0]
        while t < T:
            px = row_c[t]
            ts = ts0 + 60.0 * (t + 1)          # evaluated at the minute close
            if t in trig and np.isfinite(px):
                row = trig[t]
                al = eng.on_trigger(sym, ts, px, int(row.dir), float(row.r), H_NAME[int(row.h)],
                                    metrics={"z": float(row.z), "vm": float(row.vm), "e": float(row.e),
                                             "mkt": float(row.mkt), "liq": float(row.liq)})
            elif np.isfinite(px):
                al = eng.on_price(sym, ts, px)
            else:
                al = []
            for a in al:
                d = asdict(a)
                d.update(d.pop("metrics"))
                d["t"] = t
                out.append(d)
            if (sym, 1) in eng.active or (sym, -1) in eng.active:
                t += 1
            else:
                while k < len(tlist) and tlist[k] <= t:
                    k += 1
                if k >= len(tlist):
                    break
                t = tlist[k]
    al = pd.DataFrame(out)
    if len(al):
        al["time"] = pd.to_datetime(al["ts"], unit="s", utc=True)
    return al


# ----------------------------------------------------------------------------- scoring
def big_move_events(C, syms: list[str], pump_pct=30.0, dump_pct=25.0, merge_gap=360, win=1440) -> pd.DataFrame:
    ev = []
    for i in range(C.shape[0]):
        c = pd.Series(np.asarray(C[i], np.float64))
        if c.notna().sum() < 1440:
            continue
        lo = c.rolling(win, min_periods=60).min()
        hi = c.rolling(win, min_periods=60).max()
        for direction, cond in ((+1, c / lo >= 1 + pump_pct / 100), (-1, c / hi <= 1 - dump_pct / 100)):
            q = np.flatnonzero(cond.to_numpy())
            if q.size == 0:
                continue
            for grp in np.split(q, np.flatnonzero(np.diff(q) > merge_gap) + 1):
                a, b = grp[0], grp[-1]
                seg = c.iloc[a: b + 1]
                pk = int(seg.idxmax() if direction > 0 else seg.idxmin())
                pre = c.iloc[max(0, pk - win): pk + 1]
                st = int(pre.idxmin() if direction > 0 else pre.idxmax())
                ev.append(dict(i=i, sym=syms[i], dir=direction, t_start=st, t_peak=pk,
                               p_start=float(c.iloc[st]), p_peak=float(c.iloc[pk]),
                               move_pct=(c.iloc[pk] / c.iloc[st] - 1) * 100))
    return pd.DataFrame(ev)


def score_events(ev: pd.DataFrame, al: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return ev
    ev = ev.copy()
    first_t, first_px, first_kind = [], [], []
    by = {k: g for k, g in al.groupby(["asset", "direction"])} if len(al) else {}
    for r in ev.itertuples(index=False):
        g = by.get((r.sym, r.dir))
        a = g[(g["t"] >= r.t_start) & (g["t"] <= r.t_peak)] if g is not None else None
        if a is not None and len(a):
            a0 = a.sort_values("t").iloc[0]
            first_t.append(int(a0["t"])); first_px.append(float(a0["price"])); first_kind.append(a0["kind"])
        else:
            first_t.append(-1); first_px.append(np.nan); first_kind.append(None)
    ev["t_first"] = first_t
    ev["p_first"] = first_px
    ev["first_kind"] = first_kind
    det = ev["t_first"] >= 0
    ev["detected"] = det
    with np.errstate(divide="ignore", invalid="ignore"):
        ev["captured"] = np.where(det, np.log(ev["p_peak"] / ev["p_first"]) / np.log(ev["p_peak"] / ev["p_start"]),
                                  np.nan)
    ev["lead_min"] = np.where(det, ev["t_peak"] - ev["t_first"], np.nan)
    ev["delay_min"] = np.where(det, ev["t_first"] - ev["t_start"], np.nan)
    return ev


def summarize(al: pd.DataFrame, ev: pd.DataFrame, idx: pd.DatetimeIndex, t_lo: int, t_hi: int) -> dict:
    days = (t_hi - t_lo) / 1440.0
    a = al[(al["t"] >= t_lo) & (al["t"] < t_hi)] if len(al) else al
    e = ev[(ev["t_peak"] >= t_lo) & (ev["t_peak"] < t_hi)] if len(ev) else ev

    def evs(d):
        x = e[e["dir"] == d] if len(e) else e
        if not len(x):
            return {"n": 0, "captured_ge_half": 0}
        return {
            "n": int(len(x)),
            "detected_before_peak": int(x["detected"].sum()),
            "detect_rate": round(float(x["detected"].mean()), 3),
            "captured_median": round(float(np.nanmedian(x["captured"])), 3) if x["detected"].any() else None,
            "captured_ge_half": int((x["captured"] >= 0.5).sum()),
            "lead_min_median": float(np.nanmedian(x["lead_min"])) if x["detected"].any() else None,
        }
    kinds = a["kind"].value_counts().to_dict() if len(a) else {}
    per_day = (a.groupby(a["t"] // 1440).size() if len(a) else pd.Series(dtype=int))
    # delivery under the live BurstGovernor (the budget itself is on RAW counts, the stricter)
    gov_total, gov_max = 0, 0
    if len(a):
        srt = a.sort_values("ts")
        for _, dd in srt.groupby(srt["t"] // 1440):
            nm, ndg = govern([SimpleNamespace(kind=k, ts=float(t)) for k, t in zip(dd["kind"], dd["ts"])])
            gov_total += nm + ndg
            gov_max = max(gov_max, nm + ndg)
    return {
        "window": [str(idx[t_lo]), str(idx[min(t_hi, len(idx)) - 1])], "days": round(days, 2),
        "messages_per_day": round(len(a) / days, 2) if days else None,
        "new_per_day": round(kinds.get("NEW", 0) / days, 2) if days else None,
        "max_messages_one_day": int(per_day.max()) if len(per_day) else 0,
        "delivered_per_day_with_burst_governor": round(gov_total / days, 2) if days else None,
        "max_delivered_one_day_with_burst_governor": int(gov_max),
        "kinds": {k: int(v) for k, v in kinds.items()},
        "pump_events": evs(+1), "dump_events": evs(-1),
    }


def good(sm: dict) -> int:
    return sum(sm[k].get("captured_ge_half", 0) for k in ("pump_events", "dump_events"))


def grid(C, syms, idx, cand, ev0, split_t, out: Path) -> pd.DataFrame:
    """Calibration grid on [day 2, split) ONLY. Objective (declared up front): maximize
    calibration events detected with captured >= 0.5, subject to BUDGET; tie-break on
    fewer messages/day. Validation numbers are computed for the record but never used
    to choose."""
    rows = []
    for fm, zm, vmm, lq in ((a, b, c, d) for a in GRID["floor_mult"] for b in GRID["z_mult"]
                            for c in GRID["v_mult"] for d in GRID["liq_min"]):
                t0 = time.time()
                rules = DEFAULT_RULES.scaled(fm, zm, vmm)
                rules.liq_min_usd = lq
                al = run_alerts(triggers(cand, rules), C, syms, idx, AlertCfg())
                ev = score_events(ev0, al)
                cal = summarize(al, ev, idx, 1440, split_t)
                val = summarize(al, ev, idx, split_t, C.shape[1])
                rows.append({"floor_mult": fm, "z_mult": zm, "v_mult": vmm, "liq_min": lq,
                             "cal_new_per_day": cal["new_per_day"], "cal_msgs_per_day": cal["messages_per_day"],
                             "cal_good": good(cal),
                             "cal_events": cal["pump_events"]["n"] + cal["dump_events"]["n"],
                             "val_new_per_day": val["new_per_day"], "val_msgs_per_day": val["messages_per_day"],
                             "val_good": good(val),
                             "val_events": val["pump_events"]["n"] + val["dump_events"]["n"],
                             "secs": round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
                del al, ev
                mem(f"grid {fm}/{zm}/{vmm}/{lq:.0e}")
    g = pd.DataFrame(rows)
    g.to_csv(out / "grid.csv", index=False)
    return g


def pick(g: pd.DataFrame) -> pd.Series:
    ok = g[(g["cal_new_per_day"] <= BUDGET["new_per_day"]) & (g["cal_msgs_per_day"] <= BUDGET["messages_per_day"])]
    if ok.empty:
        raise SystemExit("no grid point meets the declared budget -- widen the grid, do not move the budget")
    return ok.sort_values(["cal_good", "cal_msgs_per_day"], ascending=[False, True]).iloc[0]


def asset_report(al: pd.DataFrame, ev: pd.DataFrame, sym: str) -> dict:
    a = al[al["asset"] == sym] if len(al) else al
    e = ev[ev["sym"] == sym] if len(ev) else ev
    return {
        "alerts": [{"time": str(r.time), "kind": r.kind, "dir": int(r.direction), "price": round(float(r.price), 6),
                    "move_pct": round(float(r.move_pct), 1), "horizon": r.horizon,
                    "anchor_px": round(float(r.anchor_px), 6)} for r in a.itertuples()] if len(a) else [],
        "events": [{"dir": int(r.dir), "start": int(r.t_start), "peak": int(r.t_peak),
                    "move_pct": round(float(r.move_pct), 1), "detected": bool(r.detected),
                    "captured": None if not r.detected else round(float(r.captured), 3),
                    "lead_min": None if not r.detected else float(r.lead_min)} for r in e.itertuples()]
        if len(e) else [],
    }


def main() -> None:
    global MEM_BUDGET_GIB, PANEL_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-09-28")
    ap.add_argument("--split", default="2026-09-01", help="calibration < split <= validation")
    ap.add_argument("--consumed-from", default="2026-09-20",
                    help="validation minutes from here on were SEEN during development (CALIBRATION.md 1a)")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--floor-mult", type=float, default=1.0)
    ap.add_argument("--z-mult", type=float, default=1.0)
    ap.add_argument("--v-mult", type=float, default=1.0)
    ap.add_argument("--refresh-min", type=int, default=15)
    ap.add_argument("--liq-min", type=float, default=None)
    ap.add_argument("--panel-dir", default=None, help="per-symbol 1m parquets (default data/hist/usdm_1m)")
    ap.add_argument("--grid", action="store_true", help="calibration grid, then the picked point")
    ap.add_argument("--mem-budget-gib", type=float, default=MEM_BUDGET_GIB)
    ap.add_argument("--grid-floor", default=None, help="override GRID floor multipliers, e.g. 1.5,2,2.5,3")
    ap.add_argument("--grid-z", default=None)
    ap.add_argument("--grid-v", default=None)
    ap.add_argument("--grid-liq", default=None, help="liquidity floors in USD, e.g. 2e6,5e6,1e7")
    args = ap.parse_args()
    MEM_BUDGET_GIB = args.mem_budget_gib
    if args.panel_dir:
        PANEL_DIR = Path(args.panel_dir)
    for key, val in (("floor_mult", args.grid_floor), ("z_mult", args.grid_z), ("v_mult", args.grid_v),
                     ("liq_min", args.grid_liq)):
        if val:
            GRID[key] = [float(x) for x in val.split(",")]

    t0 = time.time()
    out = OUT_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    mem("start")
    syms, idx, C, V = load_panel(args.start, args.end)
    print(f"panel {C.shape} {time.time()-t0:.0f}s", flush=True)
    mem("panel")
    bpath = OUT_DIR / f"blocks_{args.start}_{args.end}_r{args.refresh_min}.npz"
    if bpath.exists():
        z = np.load(bpath)
        SIG, V1M, LIQ = z["SIG"], z["V1M"], z["LIQ"]
    else:
        SIG, V1M, LIQ = compute_blocks(C, V, args.refresh_min)
        np.savez(bpath, SIG=SIG, V1M=V1M, LIQ=LIQ)
    print(f"baselines {time.time()-t0:.0f}s", flush=True)
    mem("baselines")
    split_t = int(idx.get_indexer([pd.Timestamp(args.split, tz="UTC")])[0])
    cons_t = int(idx.get_indexer([pd.Timestamp(args.consumed_from, tz="UTC")])[0])
    if args.grid:
        cand = candidate_table(C, V, SIG, V1M, LIQ, DEFAULT_RULES, args.refresh_min,
                               pre_floor=min(GRID["floor_mult"]), pre_z=min(GRID["z_mult"]),
                               pre_v=min(GRID["v_mult"]))
    else:
        cand = candidate_table(C, V, SIG, V1M, LIQ, DEFAULT_RULES, args.refresh_min,
                               pre_floor=args.floor_mult, pre_z=args.z_mult, pre_v=args.v_mult)
    print(f"candidates {len(cand)} {time.time()-t0:.0f}s", flush=True)
    mem("candidates")
    ev0 = big_move_events(C, syms)
    print(f"events {len(ev0)} {time.time()-t0:.0f}s", flush=True)
    mem("events")
    if args.grid:
        g = grid(C, syms, idx, cand, ev0, split_t, out)
        best = pick(g)
        print("PICKED (calibration only):", best.to_dict(), flush=True)
        args.floor_mult, args.z_mult, args.v_mult = float(best["floor_mult"]), float(best["z_mult"]), float(best["v_mult"])
        args.liq_min = float(best["liq_min"])
    rules = DEFAULT_RULES.scaled(args.floor_mult, args.z_mult, args.v_mult)
    if args.liq_min is not None:
        rules.liq_min_usd = args.liq_min
    al = run_alerts(triggers(cand, rules), C, syms, idx, AlertCfg())
    ev = score_events(ev0, al)
    T = C.shape[1]
    summary = {
        "multipliers": {"floor": args.floor_mult, "z": args.z_mult, "v": args.v_mult},
        "grid": GRID if args.grid else None,
        "budget": BUDGET,
        "rules": {H_NAME[h]: asdict(r) for h, r in rules.horizons.items()},
        "liq_min_usd": rules.liq_min_usd, "n_symbols": len(syms),
        "calibration": summarize(al, ev, idx, 1440, split_t),
        "validation_untouched": summarize(al, ev, idx, split_t, cons_t),
        "consumed_seen_in_development": summarize(al, ev, idx, cons_t, T),
        "positive_control_QNTUSDT": asset_report(al, ev, "QNTUSDT"),
        "negative_control_new_alerts": {s: int(((al["asset"] == s) & (al["kind"] == "NEW")).sum()) if len(al) else 0
                                        for s in ("BTCUSDT", "ETHUSDT")},
    }
    al.to_parquet(out / "alerts.parquet")
    ev.to_parquet(out / "events.parquet")
    summary["memory_peak_gib"] = round(mem("final")["VmHWM"], 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
