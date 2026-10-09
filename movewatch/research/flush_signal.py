"""Flush signal (docs/FLUSH_SIGNAL.md): after half the liquid market drops >= 5% in 15 minutes, buy
the market (M) or the hardest-hit fifth (H). Declared before results; the hist window is read once.

  python3 -m research.flush_signal build    --window {rehearsal,hist}
  python3 -m research.flush_signal validate --window {rehearsal,hist}

build prints counts only. validate computes every verdict from the declared predicates and writes
data/replay/flush_signal/receipt_<window>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import research.btc_hedge as bh
import research.replay as rp
from research.common import is_tradfi

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
OUT = rp.OUT_DIR / "flush_signal"
DECL = ROOT / "docs" / "FLUSH_SIGNAL.md"
SCRIPT = Path(__file__).resolve()
TESTS = ROOT / "tests" / "test_flush_signal.py"
MIN = pd.Timedelta("1min")

INDEX = {"BTCDOMUSDT", "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT"}
LIQ_MIN, N_MIN, W15, W5 = 5e6, 50, 15, 5
REFRACT, TURN_MAX = 1440, 60
HOLDS, STRESS = (60, 240, 1440), (1, 3, 5)
P_HOLD, P_STRESS = 240, 3
CHUNK = 5_000
DRAWS, NULL_DRAWS, POOL_PER_HOUR = 4000, 2000, 300
MIN_EVENTS, MIN_WEEKS = 15, 5
SEED = 20261009
PLANT = 0.05
DEPTHS = (0.03, 0.04, 0.05, 0.06, 0.10)
TIERS = {"WATCH": (0.03, 0.50), "FLUSH": (0.05, 0.50), "EXTREME": (0.10, 0.50),
         "FLUSH_D4": (0.04, 0.50), "FLUSH_D6": (0.06, 0.50), "FLUSH_S25": (0.05, 0.25), "FLUSH_S75": (0.05, 0.75)}
PLATEAU = ("FLUSH_D4", "FLUSH_D6", "FLUSH_S25", "FLUSH_S75")
CELLS = [("E1", "M"), ("E1", "H"), ("E2", "M"), ("E2", "H")]
K = len(CELLS)
FLAGS = ["hist is untouched for this question but not perfectly blind: large crashes in 2022-2025 are public "
         "knowledge, and other repo programs read 2022-2025 klines for cross-sectional alpha",
         "funding excluded (a 4 h hold spans at most one settlement; in flushes funding usually pays longs)",
         "costs are a calm-market slippage model x stress; real crash fills are untested"]


def scol(d: float) -> str:
    return f"s{int(round(d * 100))}"


def col(entry: str, basket: str, hold: int = P_HOLD, stress: int | str = P_STRESS) -> str:
    return f"{entry}_{basket}_{hold}_" + (stress if isinstance(stress, str) else f"s{stress}")


# ----------------------------------------------------------------------------- windows
def _hist_segments() -> list[dict]:
    d0, d1 = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-09-01", tz="UTC")
    segs = []
    for a in pd.date_range(d0, d1, freq="QS", inclusive="left"):
        b = a + pd.DateOffset(months=3)
        segs.append(dict(panel=(str(max(a - pd.Timedelta(days=2), d0).date()), str(min(b + pd.Timedelta(days=2), d1).date())),
                         panel_dir=HIST / "usdm_1m_hist",
                         own=(max(a, pd.Timestamp("2022-01-02", tz="UTC")), min(b, pd.Timestamp("2025-08-30", tz="UTC")))))
    return segs


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


WINDOWS = {
    "rehearsal": [
        dict(panel=("2025-09-01", "2026-01-01"), panel_dir=HIST / "usdm_1m_val", own=(_ts("2025-10-01"), _ts("2025-12-30"))),
        dict(panel=("2025-12-01", "2026-04-01"), panel_dir=HIST / "usdm_1m_val", own=(_ts("2025-12-30"), _ts("2026-03-30"))),
        dict(panel=("2026-04-01", "2026-08-01"), panel_dir=HIST / "usdm_1m_holdout", own=(_ts("2026-04-02"), _ts("2026-07-30"))),
        dict(panel=("2026-08-01", "2026-09-28"), panel_dir=HIST / "usdm_1m", own=(_ts("2026-08-02"), _ts("2026-09-26"))),
    ],
    "hist": _hist_segments(),
}


# ----------------------------------------------------------------------------- provenance / guards
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def committed_clean(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    return (_git("ls-files", "--error-unmatch", rel).returncode == 0
            and _git("diff", "--quiet", "HEAD", "--", rel).returncode == 0)


def require_frozen() -> None:
    bad = [p.name for p in (DECL, SCRIPT, TESTS) if not committed_clean(p)]
    if bad:
        raise SystemExit(f"REFUSED: {', '.join(bad)} not committed clean; the hist window is read only after the freeze")


def blob(path: Path) -> str:
    return _git("hash-object", str(path)).stdout.strip()


def last_commit(path: Path) -> str:
    return _git("log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))).stdout.strip()


# ----------------------------------------------------------------------------- detector
def static_keep(syms: list[str]) -> np.ndarray:
    return np.array([s not in INDEX and s not in rp.STABLES and not is_tradfi(s) for s in syms])


def liq_blocks(V) -> np.ndarray:
    """LIQ[:, b] = USD volume over [b*15 - 1440, b*15), the same computation as replay.compute_blocks."""
    n, T = V.shape
    R = 15
    nb = (T + R - 1) // R
    S = np.zeros((n, nb), np.float64)
    for b in range(nb):
        S[:, b] = V[:, b * R:(b + 1) * R].sum(axis=1, dtype=np.float64)
    cs = np.zeros((n, nb + 1), np.float64)
    np.cumsum(S, axis=1, out=cs[:, 1:])
    bb = np.arange(nb)
    return (cs[:, bb] - cs[:, np.maximum(bb - 1440 // R, 0)]).astype(np.float32)


SER = ["n_u", "m15", "m5", *[scol(d) for d in DEPTHS]]


def minute_series(C, V, LIQ, keep: np.ndarray, j0: int, j1: int) -> dict[str, np.ndarray]:
    """Per minute t in [j0, j1): universe size, median 15-min and 5-min log return, share_X."""
    out = {c: np.full(j1 - j0, np.nan) for c in SER}
    lg = {d: math.log(1 - d) for d in DEPTHS}
    for s in range(j0, j1, CHUNK):
        e = min(s + CHUNK, j1)
        cur = np.asarray(C[:, s:e], np.float64)
        vv = np.asarray(V[:, s - W15 + 1:e], np.float64)
        cv = np.zeros((vv.shape[0], vv.shape[1] + 1))
        np.cumsum(vv, axis=1, out=cv[:, 1:])
        vol15 = cv[:, W15:] - cv[:, :-W15]
        del vv, cv
        with np.errstate(all="ignore"):
            r15 = np.log(cur / np.asarray(C[:, s - W15:e - W15], np.float64))
            r5 = np.log(cur / np.asarray(C[:, s - W5:e - W5], np.float64))
        u = keep[:, None] & (LIQ[:, np.arange(s, e) // 15] >= LIQ_MIN) & np.isfinite(r15) & (vol15 > 0)
        n = u.sum(axis=0)
        ok = n >= N_MIN
        sl = slice(s - j0, e - j0)
        out["n_u"][sl] = n
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["m15"][sl] = np.where(ok, np.nanmedian(np.where(u, r15, np.nan), axis=0), np.nan)
            out["m5"][sl] = np.where(ok, np.nanmedian(np.where(u & np.isfinite(r5), r5, np.nan), axis=0), np.nan)
        for d in DEPTHS:
            out[scol(d)][sl] = np.where(ok, (u & (r15 <= lg[d])).sum(axis=0) / np.maximum(n, 1), np.nan)
        del cur, r15, r5, u, vol15
    return out


def universe_at(C, V, LIQ, keep: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of U(t) and their 15-minute log returns (the detector's predicate, one minute)."""
    with np.errstate(all="ignore"):
        r15 = np.log(np.asarray(C[:, t], np.float64) / np.asarray(C[:, t - W15], np.float64))
    vol15 = np.asarray(V[:, t - W15 + 1:t + 1], np.float64).sum(axis=1)
    u = np.flatnonzero(keep & (LIQ[:, t // 15] >= LIQ_MIN) & np.isfinite(r15) & (vol15 > 0))
    return u, r15[u]


def naive_minute(C, V, LIQ, keep: np.ndarray, t: int) -> dict[str, float]:
    """Self-test reference: the same minute, one coin at a time."""
    r15s, r5s = [], []
    for i in np.flatnonzero(keep):
        if not LIQ[i, t // 15] >= LIQ_MIN:
            continue
        c, c15, c5 = float(C[i, t]), float(C[i, t - W15]), float(C[i, t - W5])
        if not (math.isfinite(c) and math.isfinite(c15) and c > 0 and c15 > 0):
            continue
        if not float(np.asarray(V[i, t - W15 + 1:t + 1], np.float64).sum()) > 0:
            continue
        r15s.append(math.log(c / c15))
        if math.isfinite(c5) and c5 > 0:
            r5s.append(math.log(c / c5))
    n = len(r15s)
    res = {"n_u": float(n)}
    if n < N_MIN:
        return res | {c: float("nan") for c in SER if c != "n_u"}
    res["m15"], res["m5"] = float(np.median(r15s)), float(np.median(r5s)) if r5s else float("nan")
    for d in DEPTHS:
        res[scol(d)] = sum(r <= math.log(1 - d) for r in r15s) / n
    return res


def fire_times(times: np.ndarray, cond: np.ndarray) -> list:
    """First minute of each run of `cond`, unless the same tier fired in the previous 24 h."""
    out, last = [], None
    gap = np.timedelta64(REFRACT, "m")
    for t in times[cond]:
        if last is None or t - last >= gap:
            out.append(t)
            last = t
    return out


def e2_offset(m5: np.ndarray, p: int) -> tuple[int, bool]:
    """E2 entry offset from the trigger: first u in [t+1, t+60] with median 5-min return > 0 -> u+1,
    else t+61 (timeout). `p` is the trigger's position in `m5`."""
    hit = np.flatnonzero(m5[p + 1:p + TURN_MAX + 1] > 0)
    return (int(hit[0]) + 2, False) if len(hit) else (TURN_MAX + 1, True)


# ----------------------------------------------------------------------------- trades
def basket_trades(C, V, LIQ, keep: np.ndarray, t: int, entries: dict[str, int]) -> tuple[dict, dict]:
    u, r15 = universe_at(C, V, LIQ, keep, t)
    members = {"M": u, "H": u[np.argsort(r15, kind="stable")[:math.ceil(len(u) / 5)]]}
    slip = {k: np.array([bh.slip(float(v)) for v in LIQ[m, t // 15]]) for k, m in members.items()}
    rec = {"n_u": len(u)}
    for en, e in entries.items():
        c0 = np.asarray(C[:, e], np.float64)
        for hold in HOLDS:
            x = e + hold
            c1 = np.asarray(C[:, x], np.float64)
            for bn, m in members.items():
                ok = np.isfinite(c0[m]) & (c0[m] > 0)
                mm, sp = m[ok], slip[bn][ok]
                p1 = c1[mm].copy()
                bad = np.flatnonzero(~np.isfinite(p1))
                for k in bad:                                   # last finite close in [entry, exit]
                    row = np.asarray(C[mm[k], e:x + 1], np.float64)
                    p1[k] = row[np.flatnonzero(np.isfinite(row))[-1]]
                r = p1 / c0[mm] - 1.0
                rec[col(en, bn, hold, "gross")] = float(r.mean()) if len(r) else float("nan")
                for st in STRESS:
                    rec[col(en, bn, hold, st)] = float((r - 2 * (bh.FEE + st * sp)).mean()) if len(r) else float("nan")
                if hold == P_HOLD:
                    rec[f"{en}_{bn}_n"], rec[f"{en}_{bn}_no_entry"], rec[f"{en}_{bn}_last_price"] = len(mm), int((~ok).sum()), len(bad)
    return rec, members


def price_reconcile(sg: dict, idx: pd.DatetimeIndex, syms: list[str], t: int, e: int, members: np.ndarray, got: float) -> float:
    """Self-test: gross E1-M 4 h return from the per-symbol 1m parquet files, not the panel."""
    t0, t1 = idx[e], idx[e + P_HOLD]
    rs = []
    for i in members:
        s = pd.read_parquet(sg["panel_dir"] / f"{syms[i]}.parquet", columns=["close"])["close"]
        s = s[~s.index.duplicated()].loc[t0:t1].astype(np.float64)
        if t0 not in s.index or not np.isfinite(s.loc[t0]) or s.loc[t0] <= 0:
            continue
        c1 = s.loc[t1] if t1 in s.index and np.isfinite(s.loc[t1]) else s.dropna().iloc[-1]
        rs.append(c1 / s.loc[t0] - 1.0)
    want = float(np.mean(rs))
    if not abs(want - got) <= 1e-6:
        raise SystemExit(f"SELFTEST price reconciliation failed at {idx[t]}: panel {got:.8f} vs parquet {want:.8f}")
    return want


# ----------------------------------------------------------------------------- build
def _load(sg: dict):
    rp.PANEL_DIR = sg["panel_dir"]
    syms, idx, C, V = rp.load_panel(*sg["panel"])
    own0, own1 = sg["own"]
    if own0 - idx[0] < pd.Timedelta(minutes=1440):          # LIQ needs the full 24 h before a block
        raise SystemExit(f"panel {sg['panel']}: lookback margin too short for {own0}")
    if idx[-1] - own1 < pd.Timedelta(minutes=TURN_MAX + 2 + max(HOLDS)):
        raise SystemExit(f"panel {sg['panel']}: forward margin too short for {own1}")
    return syms, idx, C, V


def build(window: str) -> None:
    segs = WINDOWS[window]
    receipt = OUT / f"receipt_{window}.json"
    if window == "hist":
        require_frozen()
        if receipt.exists():
            raise SystemExit("REFUSED: hist was already validated (read once); not rebuilding")
        man = HIST / "usdm_1m_hist" / "_manifest.json"
        if not man.exists():
            raise SystemExit("REFUSED: hist download not finished (_manifest.json missing)")
        if json.loads(man.read_text())["counts"].get("error", 0):
            raise SystemExit("REFUSED: hist download has symbols with errors; re-run the download first")
    OUT.mkdir(parents=True, exist_ok=True)
    st: dict = {"window": window, "script_blob": blob(SCRIPT), "head": _git("rev-parse", "HEAD").stdout.strip(),
                "segments": []}
    # ---- pass 1: minute series per panel
    for k, sg in enumerate(segs):
        syms, idx, C, V = _load(sg)
        LIQ = liq_blocks(V)
        if window == "rehearsal":                       # self-test 3: LIQ vs the replay engine's blocks
            ref = np.load(rp.OUT_DIR / f"blocks_{sg['panel'][0]}_{sg['panel'][1]}_r15.npz")["LIQ"]
            if ref.shape != LIQ.shape or not np.allclose(ref, LIQ, rtol=1e-6, atol=1.0, equal_nan=True):
                raise SystemExit(f"SELFTEST LIQ reconciliation failed for {sg['panel']}")
        np.save(OUT / f"liq_{window}_{k}.npy", LIQ)
        keep = static_keep(syms)
        j0, j1own = idx.get_loc(sg["own"][0]), idx.get_loc(sg["own"][1])
        j1 = min(j1own + 120, len(idx))
        ser = pd.DataFrame(minute_series(C, V, LIQ, keep, j0, j1))
        ser.insert(0, "time", idx[j0:j1])
        ser["owned"] = np.arange(j1 - j0) < (j1own - j0)
        ser.to_parquet(OUT / f"series_{window}_{k}.parquet")
        st["segments"].append({"panel": list(sg["panel"]), "own": [str(x) for x in sg["own"]], "n_syms": len(syms),
                               "kept": int(keep.sum()), "owned_minutes": int(j1own - j0),
                               "THIN_UNIVERSE_minutes": int((ser["owned"] & (ser["n_u"] < N_MIN)).sum())})
        rp.mem(f"{window} series {k + 1}/{len(segs)} {sg['panel'][0]}")
        del C, V, LIQ, ser
    allser = pd.concat([pd.read_parquet(OUT / f"series_{window}_{k}.parquet").assign(seg=k) for k in range(len(segs))])
    allser["tn"] = allser["time"].dt.tz_convert(None)          # naive UTC datetime64 for numpy work
    own = allser[allser["owned"]].reset_index(drop=True)
    # ---- fires per tier / variant (refractory applied across panels)
    ev = []
    times = own["tn"].to_numpy()
    for tier, (d, sh) in TIERS.items():
        ev += [{"tier": tier, "tn": t} for t in fire_times(times, (own[scol(d)] >= sh).to_numpy())]
    ev = pd.DataFrame(ev, columns=["tier", "tn"])
    # ---- random-time pool: >= 24 h from any WATCH minute, 300 per UTC hour
    watch = np.sort(allser.loc[allser["s3"] >= TIERS["WATCH"][1], "tn"].to_numpy())
    cand = own.loc[own["n_u"] >= N_MIN, "tn"].to_numpy()
    if len(watch):
        pos = np.searchsorted(watch, cand)
        dist = np.minimum(np.abs(watch[np.minimum(pos, len(watch) - 1)] - cand), np.abs(cand - watch[np.maximum(pos - 1, 0)]))
        cand = cand[dist >= np.timedelta64(REFRACT, "m")]
    rng = np.random.default_rng(SEED)
    hours = pd.DatetimeIndex(cand).hour
    pool = []
    for h in range(24):
        c = cand[hours == h]
        pool += list(c[rng.choice(len(c), size=min(POOL_PER_HOUR, len(c)), replace=False)])
    todo = pd.concat([ev, pd.DataFrame({"tier": "POOL", "tn": pool})], ignore_index=True)
    todo["time"] = pd.DatetimeIndex(todo.pop("tn")).tz_localize("UTC")
    # ---- pass 2: trades in the panel that owns each trigger
    rows, recon = [], []
    rs = np.random.default_rng(SEED + 1)
    for k, sg in enumerate(segs):
        sel = todo[(todo["time"] >= sg["own"][0]) & (todo["time"] < sg["own"][1])]
        syms, idx, C, V = _load(sg)
        LIQ = np.load(OUT / f"liq_{window}_{k}.npy")
        keep = static_keep(syms)
        ser = pd.read_parquet(OUT / f"series_{window}_{k}.parquet")
        j0 = idx.get_loc(ser["time"].iloc[0])
        m5 = ser["m5"].to_numpy()
        # self-test 1: vectorised series == naive recompute (every trigger minute + 200 random owned minutes)
        trig = [int((t - idx[0]) / MIN) for t in sel.loc[sel["tier"] != "POOL", "time"]]
        rand = list(rs.integers(j0, j0 + int(ser["owned"].sum()), 200))
        for t in trig + rand:
            got, ref = ser.iloc[t - j0], naive_minute(C, V, LIQ, keep, t)
            for c in SER:
                a, b = float(got[c]), ref[c]
                if not ((math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12):
                    raise SystemExit(f"SELFTEST detector reconciliation failed at {idx[t]} {c}: {a} vs {b}")
        for r in sel.itertuples():
            t = int((r.time - idx[0]) / MIN)
            off, timeout = e2_offset(m5, t - j0)
            rec, members = basket_trades(C, V, LIQ, keep, t, {"E1": t + 1, "E2": t + off})
            row = {"tier": r.tier, "time": r.time, "seg": k, "e2_offset": off, "E2_TIMEOUT": timeout,
                   "m15": float(ser["m15"].iloc[t - j0]), "s5": float(ser["s5"].iloc[t - j0]), **rec}
            rows.append(row)
            if len(recon) < 3:                                  # self-test 2 (events come first in `sel`)
                recon.append(price_reconcile(sg, idx, syms, t, t + 1, members["M"], rec[col("E1", "M", P_HOLD, "gross")]))
        rp.mem(f"{window} trades {k + 1}/{len(segs)}")
        del C, V, LIQ
    if len(recon) < 3:
        raise SystemExit(f"SELFTEST price reconciliation ran on only {len(recon)} triggers")
    df = pd.DataFrame(rows)
    df[df["tier"] != "POOL"].to_parquet(OUT / f"events_{window}.parquet")
    df[df["tier"] == "POOL"].to_parquet(OUT / f"pool_{window}.parquet")
    st |= {"fires": ev["tier"].value_counts().reindex(list(TIERS), fill_value=0).astype(int).to_dict(),
           "pool": int((df["tier"] == "POOL").sum()),
           "pool_per_hour_min": int(pd.DatetimeIndex(df.loc[df["tier"] == "POOL", "time"]).hour.value_counts().min()),
           "E2_TIMEOUT_flush": int(df.loc[df["tier"] == "FLUSH", "E2_TIMEOUT"].sum()),
           "NO_ENTRY_PRICE_coins_flush": int(df.loc[df["tier"] == "FLUSH", [f"{e}_{b}_no_entry" for e, b in CELLS]].to_numpy().sum()),
           "LAST_PRICE_coins_flush": int(df.loc[df["tier"] == "FLUSH", [f"{e}_{b}_last_price" for e, b in CELLS]].to_numpy().sum()),
           "selftests": {"detector": "OK", "price_reconciliation_n": len(recon), "liq": "OK" if window == "rehearsal" else "n/a"}}
    (OUT / f"build_{window}.json").write_text(json.dumps(st, indent=1, default=str))
    print(json.dumps({k: v for k, v in st.items() if k != "segments"}, default=str))


# ----------------------------------------------------------------------------- statistics
def _weeks(times: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(times).tz_convert(None).to_period("W").start_time.to_numpy()


def _blocks(wk: np.ndarray, rng, draws: int):
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == w) for w in uw]
    for _ in range(draws):
        yield np.concatenate([pos[j] for j in rng.integers(0, len(uw), len(uw))])


def cell_stats(vals: np.ndarray, wk: np.ndarray, seed: int = SEED, draws: int = DRAWS) -> dict:
    ok = np.isfinite(vals)
    v, w = vals[ok], wk[ok]
    n, nw = int(len(v)), int(len(np.unique(w)))
    if n < MIN_EVENTS or nw < MIN_WEEKS:
        return {"status": "INSUFFICIENT", "n": n, "weeks": nw, "mean": float(v.mean()) if n else float("nan")}
    b = np.array([v[s].mean() for s in _blocks(w, np.random.default_rng(seed), draws)])
    return {"status": "OK", "n": n, "weeks": nw, "mean": float(v.mean()), "median": float(np.median(v)),
            "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "p": float((1 + (b <= 0).sum()) / (draws + 1)), "win": float((v > 0).mean())}


def null_test(obs: float, ev_hours: np.ndarray, pool_vals: np.ndarray, pool_hours: np.ndarray,
              seed: int = SEED, draws: int = NULL_DRAWS) -> dict:
    rng = np.random.default_rng(seed)
    by = {h: pool_vals[(pool_hours == h) & np.isfinite(pool_vals)] for h in np.unique(ev_hours)}
    if any(len(x) == 0 for x in by.values()):
        return {"status": "NO_POOL_FOR_HOUR", "p_null": float("nan")}
    mat = np.column_stack([rng.choice(by[h], size=draws) for h in ev_hours])
    nm = mat.mean(axis=1)
    return {"status": "OK", "p_null": float((1 + (nm >= obs).sum()) / (draws + 1)), "null_mean": float(nm.mean()),
            "null_ci": [float(np.percentile(nm, 2.5)), float(np.percentile(nm, 97.5))]}


def verdict(c: dict) -> str:
    if c["status"] != "OK":
        return "INSUFFICIENT"
    core = c["mean"] > 0 and c["ci"][0] > 0
    full = (core and c["p"] <= 0.05 / K and c.get("p_null", float("nan")) <= 0.05
            and all(np.isfinite(x) and x > 0 for x in c["plateau"].values()) and c["drop_best"] > 0)
    return "VALIDATED" if full else "SUGGESTIVE" if core else "FAILED"


def evaluate(vals: np.ndarray, times: pd.Series, pool_vals: np.ndarray, pool_times: pd.Series,
             plateau: dict[str, float], seed: int = SEED) -> dict:
    c = cell_stats(vals, _weeks(times), seed)
    v = vals[np.isfinite(vals)]
    c["drop_best"] = float(np.sort(v)[:-1].mean()) if len(v) > 1 else float("nan")
    c["top_event_share"] = float(v.max() / v.sum()) if len(v) and v.sum() > 0 else float("nan")
    c["plateau"] = plateau
    if c["status"] == "OK":
        c |= null_test(c["mean"], pd.DatetimeIndex(times).hour.to_numpy()[np.isfinite(vals)], pool_vals,
                       pd.DatetimeIndex(pool_times).hour.to_numpy(), seed)
    c["verdict"] = verdict(c)
    return c


def falsifiers(pool: pd.DataFrame) -> dict:
    """Instrument checks: planted must validate, null-as-events must not, empty is INSUFFICIENT."""
    rng = np.random.default_rng(SEED + 7)
    out = {}
    for en, bn in CELLS:
        cc = col(en, bn)
        p = pool[np.isfinite(pool[cc])].reset_index(drop=True)
        a = p.iloc[rng.choice(len(p), 30, replace=False)]
        b = p.iloc[rng.choice(len(p), 30, replace=False)]
        planted = a[cc].to_numpy() + PLANT
        pl = evaluate(planted, a["time"], p[cc].to_numpy(), p["time"], {v: float(planted.mean()) for v in PLATEAU})
        nul = evaluate(b[cc].to_numpy(), b["time"], p[cc].to_numpy(), p["time"], {v: float(b[cc].mean()) for v in PLATEAU})
        emp = evaluate(np.array([]), pd.Series([], dtype="datetime64[ns, UTC]"), p[cc].to_numpy(), p["time"], {})
        out[f"{en}_{bn}"] = {"planted": pl["verdict"], "null_as_events": nul["verdict"], "empty": emp["verdict"]}
        if pl["verdict"] != "VALIDATED" or nul["verdict"] == "VALIDATED" or emp["verdict"] != "INSUFFICIENT":
            raise SystemExit(f"FALSIFIER failed for {en}_{bn}: {out[f'{en}_{bn}']}")
    return out


def _brief(c: dict) -> dict:
    keys = ("status", "n", "weeks", "mean", "median", "ci", "p", "win", "p_null", "null_mean", "drop_best",
            "top_event_share", "plateau", "verdict")
    return {k: c[k] for k in keys if k in c}


def validate(window: str) -> dict:
    receipt = OUT / f"receipt_{window}.json"
    if window == "hist":
        require_frozen()
        if receipt.exists():
            raise SystemExit("REFUSED: hist receipt exists; the window is read once")
    bst = json.loads((OUT / f"build_{window}.json").read_text())
    if bst["script_blob"] != blob(SCRIPT):
        raise SystemExit("REFUSED: the builder changed since this window was built")
    ev = pd.read_parquet(OUT / f"events_{window}.parquet")
    pool = pd.read_parquet(OUT / f"pool_{window}.parquet")
    fal = falsifiers(pool)
    fl = ev[ev["tier"] == "FLUSH"].sort_values("time")
    res: dict = {"window": window, "label": "REHEARSAL (consumed year; descriptive only, not a test)"
                 if window == "rehearsal" else "TEST (untouched window, read once)",
                 "declaration_commit": last_commit(DECL), "script_commit": last_commit(SCRIPT),
                 "head": _git("rev-parse", "HEAD").stdout.strip(),
                 "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "flags": FLAGS, "build": {k: v for k, v in bst.items() if k != "segments"}, "falsifiers": fal, "k": K}
    cells = {}
    for en, bn in CELLS:
        cc = col(en, bn)
        plateau = {v: float(ev.loc[ev["tier"] == v, cc].mean()) if (ev["tier"] == v).any() else float("nan")
                   for v in PLATEAU}
        cells[f"{en}_{bn}"] = _brief(evaluate(fl[cc].to_numpy(), fl["time"], pool[cc].to_numpy(), pool["time"], plateau))
    res["cells"] = cells
    res["n_validated"] = sum(c["verdict"] == "VALIDATED" for c in cells.values())
    # ---- descriptive (decides nothing)
    wk = _weeks(fl["time"])
    d: dict = {"H_minus_M": {en: _brief(cell_stats((fl[col(en, "H")] - fl[col(en, "M")]).to_numpy(), wk))
                             for en in ("E1", "E2")}}
    for tier in ("WATCH", "EXTREME"):
        t_ = ev[ev["tier"] == tier]
        d[tier] = {f"{en}_{bn}": _brief(cell_stats(t_[col(en, bn)].to_numpy(), _weeks(t_["time"])))
                   for en, bn in CELLS}
    d["holds_stress"] = {f"{en}_{bn}_{h}_{s}": _brief(cell_stats(fl[col(en, bn, h, s)].to_numpy(), wk))
                         for en, bn in CELLS for h in HOLDS for s in (*STRESS, "gross")}
    d["pool_mean"] = {f"{en}_{bn}": float(pool[col(en, bn)].mean()) for en, bn in CELLS}
    d["per_year"] = {f"{en}_{bn}": fl.groupby(fl["time"].dt.year)[col(en, bn)].agg(["size", "mean"]).round(5).to_dict("index")
                     for en, bn in CELLS}
    d["events"] = [{"time": f"{r.time:%Y-%m-%d %H:%M}", "n_u": int(r.n_u), "s5": round(r.s5, 3),
                    "m15_pct": round(100 * (math.exp(r.m15) - 1), 2), "e2_offset_min": int(r.e2_offset),
                    "E2_TIMEOUT": bool(r.E2_TIMEOUT),
                    **{f"{en}_{bn}_pct": round(100 * getattr(r, col(en, bn)), 2) for en, bn in CELLS},
                    "E1_M_gross_pct": round(100 * getattr(r, col("E1", "M", P_HOLD, "gross")), 2),
                    "E1_H_gross_pct": round(100 * getattr(r, col("E1", "H", P_HOLD, "gross")), 2)}
                   for r in fl.itertuples()]
    res["descriptive"] = d
    receipt.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"window": window, "fires": bst["fires"], "cells": cells, "falsifiers": fal}, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "validate"])
    ap.add_argument("--window", choices=list(WINDOWS), required=True)
    a = ap.parse_args()
    (build if a.phase == "build" else validate)(a.window)


if __name__ == "__main__":
    main()
