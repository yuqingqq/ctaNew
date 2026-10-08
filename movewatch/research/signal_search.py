"""Signal search. Protocol: docs/SIGNAL_SEARCH.md (declared in commit 2e49375, before any data).

Phases:
  build --window W  rows for one window: the persistence features (same code), the +4 h positioning
                    (runners.first4h), the ten new alert-time features, x4h, and the trade legs
                    (raw return + funding per span). V windows refuse unless the freeze is committed.
  screen            S1/S2 only: self-tests (reconciliation, falsifiers), IC per cell, the pass rule;
                    writes docs/SIGNAL_SEARCH_FREEZE.json (computed). Commit it before building V.
  validate          V only, ONCE: refuses without a committed freeze or with an existing receipt.

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.signal_search build --window holdout
"""
from __future__ import annotations

import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "ARROW_DEFAULT_MEMORY_POOL": "system"}.items():
    os.environ.setdefault(_k, _v)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

import research.btc_hedge as bh  # noqa: E402
import research.persistence_study as ps  # noqa: E402
import research.replay as rp  # noqa: E402
from movewatch.instruments import split_multiplier  # noqa: E402
from research.common import Stale, metrics_row, mkt_median, panel_meta  # noqa: E402
from research.runners import base_of, first4h  # noqa: E402
from research.signal_features import fetch_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
OUT = rp.OUT_DIR / "signal_search"
FREEZE = ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json"
RECEIPT = OUT / "validation_receipt.json"
DECLARATION_COMMIT = "2e49375"
UPBIT = HIST / "upbit_1m"
WINDOWS = {
    "discovery": dict(ps.WINDOWS["discovery"]),
    "holdout": dict(ps.WINDOWS["holdout"]),
    "val_a": dict(panel=("2025-09-01", "2026-01-01"), panel_dir=HIST / "usdm_1m_val", alerts="val120a",
                  sig=("2025-10-01", "2025-12-29")),
    "val_b": dict(panel=("2025-12-01", "2026-04-01"), panel_dir=HIST / "usdm_1m_val", alerts="val120b",
                  sig=("2025-12-29", "2026-03-29")),
}
SCREEN, VAL = ("discovery", "holdout"), ("val_a", "val_b")
SPANS = {"x4h": (1, 241), "x72h": (1, 4321), "x4_72": (241, 4321)}       # minutes after trigger minute t
DRAWS = 4000
MIN_N, MIN_WEEKS = 100, 5
SEED = 20261008
MIN = pd.Timedelta(minutes=1)

NEW = ["trade_size", "wick15", "hi7", "btc_24h", "asia", "weekend", "upbit_listed", "kimchi", "upbit_share",
       "oi_chg_1h"]
BINARY = {"asia", "weekend", "upbit_listed", "has_spot"}
CARRIED = [   # Group A: (id, feature, pop, outcome, declared sign on continuation)
    ("A1", "mkt_breadth", "all", "x4h", -1),
    ("A2", "vol_persist", "pumps", "x4_72", -1),
    ("A3", "premium_h4", "pumps", "x4_72", +1),
    ("A4", "taker_ls_4h", "pumps", "x4_72", +1),
    ("A5", "has_spot", "pumps", "x72h", -1),
    ("A6", "age_days", "pumps", "x72h", -1),
    ("A7", "calm_7d", "pumps", "x72h", +1),
]
BASE_RATE = ("A8", "pumps", "x72h")         # median < 0


# ----------------------------------------------------------------------------- git guards
def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def committed_clean(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    return (_git("ls-files", "--error-unmatch", rel).returncode == 0
            and _git("diff", "--quiet", "HEAD", "--", rel).returncode == 0)


def head() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def require_freeze() -> dict:
    if not FREEZE.exists() or not committed_clean(FREEZE):
        raise SystemExit(f"REFUSED: {FREEZE.name} missing or not committed; the V window is read only after the freeze")
    return json.loads(FREEZE.read_text())


# ----------------------------------------------------------------------------- Upbit (rate-limited, cached)
_LOCK = threading.Lock()
_LAST: list[float] = []
_S = requests.Session()
_S.headers["User-Agent"] = "movewatch-research/0.1"


def _throttle(per_sec: float = 7.0) -> None:
    with _LOCK:
        now = time.time()
        while _LAST and now - _LAST[0] > 1.0:
            _LAST.pop(0)
        if len(_LAST) >= per_sec:
            time.sleep(1.0 - (now - _LAST[0]) + 0.01)
        _LAST.append(time.time())


def upbit_candles(market: str, to: pd.Timestamp, count: int):
    """Upbit KRW 1m candles that START before `to` (Upbit `to` is exclusive). Returns a list,
    "no_market" (404: never listed OR since delisted, indistinguishable), or None (error, uncached)."""
    p = UPBIT / market / f"{to:%Y%m%d%H%M}_{count}.json"
    if p.exists():
        return json.loads(p.read_text())
    for k in range(5):
        _throttle()
        try:
            r = _S.get("https://api.upbit.com/v1/candles/minutes/1",
                       params={"market": market, "to": to.strftime("%Y-%m-%dT%H:%M:%SZ"), "count": count}, timeout=15)
        except requests.RequestException:
            time.sleep(1 + k)
            continue
        if r.status_code == 429:
            time.sleep(1.5 * (k + 1))
            continue
        if r.status_code == 404:
            res = "no_market"
        elif r.ok:
            res = [{"t": c["candle_date_time_utc"], "px": c["trade_price"], "krw": c["candle_acc_trade_price"]}
                   for c in r.json()]
        else:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res))
        return res
    return None


def upbit_feats(base: str, mult: float, d: int, time_close: pd.Timestamp, perp_px: float, perp_qv60: float) -> dict:
    tm = time_close - MIN                                   # open time of trigger minute t
    out = {"upbit_listed": np.nan, "kimchi": np.nan, "upbit_share": np.nan, "upbit_status": "OK"}
    c = upbit_candles(f"KRW-{base}", time_close, 200)
    if c is None:
        out["upbit_status"] = "UPBIT_ERROR"
        return out
    if c == "no_market":
        out["upbit_listed"], out["upbit_status"] = 0.0, "UPBIT_NO_DATA"
        return out
    starts = pd.to_datetime([x["t"] for x in c], utc=True) if c else pd.DatetimeIndex([], tz="UTC")
    listed = bool(len(starts) and starts.max() >= tm - pd.Timedelta(hours=24))
    out["upbit_listed"] = float(listed)
    if not listed:
        out["upbit_status"] = "UPBIT_NOT_TRADING"
        return out
    u = upbit_candles("KRW-USDT", time_close, 30)
    if not isinstance(u, list) or not u:
        out["upbit_status"] = "NO_USDT_RATE"
        return out
    ut = pd.to_datetime([x["t"] for x in u], utc=True)
    if ut.max() < tm - pd.Timedelta(minutes=30):
        out["upbit_status"] = "NO_USDT_RATE"
        return out
    usdt = float(u[int(np.argmax(ut))]["px"])
    in_hr = [x for x, s in zip(c, starts) if s >= tm - pd.Timedelta(minutes=59)]
    up_usd = sum(x["krw"] for x in in_hr) / usdt
    if np.isfinite(perp_qv60) and up_usd + perp_qv60 > 0:
        out["upbit_share"] = up_usd / (up_usd + perp_qv60)
    k = int(np.argmax(starts))
    if starts[k] >= tm - pd.Timedelta(minutes=10) and perp_px > 0:
        kim = math.log(c[k]["px"] / usdt) - math.log(perp_px / mult)
        if abs(kim) > 0.5:
            out["upbit_status"] = "PRICE_MISMATCH"
        else:
            out["kimchi"] = d * kim
    else:
        out["upbit_status"] = "UPBIT_STALE_PRICE"
    return out


# ----------------------------------------------------------------------------- build
def minute_feats(rows: pd.DataFrame, panel_dir: Path) -> pd.DataFrame:
    """B1 trade_size, B2 wick15, B3 hi7, and the perp close/volume for B8/B9, from the per-symbol
    1m parquets (data through the trigger minute t only)."""
    recs = {}
    for asset, g in rows.groupby("asset"):
        df = pd.read_parquet(panel_dir / f"{asset}.parquet", columns=["high", "low", "close", "quote_volume", "trades"])
        df = df[~df.index.duplicated()].sort_index()
        first = df.index[0]
        for r in g.itertuples():
            tm = r.time - MIN
            d = int(r.dir)
            rec = {"trade_size": np.nan, "wick15": np.nan, "hi7": np.nan, "perp_px": np.nan, "perp_qv60": np.nan,
                   "minute_status": "OK"}
            if tm not in df.index:
                rec["minute_status"] = "NO_BAR_T"
                recs[r.Index] = rec
                continue
            ct = float(df.at[tm, "close"])
            w = df.loc[tm - pd.Timedelta(minutes=59): tm]
            rec["perp_px"], rec["perp_qv60"] = ct, float(w["quote_volume"].sum())
            b = df.loc[tm - pd.Timedelta(days=7, minutes=60): tm - pd.Timedelta(minutes=60)]
            if len(w) >= 30 and len(b) >= 1440 and w["trades"].sum() > 0 and b["trades"].sum() > 0:
                upt_w = w["quote_volume"].sum() / w["trades"].sum()
                upt_b = b["quote_volume"].sum() / b["trades"].sum()
                rec["trade_size"] = math.log10(upt_w / upt_b) if upt_w > 0 and upt_b > 0 else np.nan
            w15 = df.loc[tm - pd.Timedelta(minutes=14): tm]
            rng = float(w["high"].max() - w["low"].min())
            if rng > 0:
                rec["wick15"] = ((float(w15["high"].max()) - ct) / rng if d > 0 else (ct - float(w15["low"].min())) / rng)
            h = df.loc[tm - pd.Timedelta(days=7): tm - pd.Timedelta(minutes=60), "close"]
            if first <= tm - pd.Timedelta(days=7) + pd.Timedelta(minutes=60) and len(h) >= 1440:
                rec["hi7"] = math.log(ct / float(h.max())) if d > 0 else -math.log(ct / float(h.min()))
            else:
                rec["minute_status"] = "SHORT_HISTORY"
            recs[r.Index] = rec
        del df
    return pd.DataFrame.from_dict(recs, orient="index")


def build_window(window: str) -> None:
    if window in VAL:
        require_freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    w = WINDOWS[window]
    ps.WINDOWS[window] = w
    ps.OUT = OUT
    status: dict = {}
    df = ps.build(window, status)
    st = status[window]
    st.pop("random_baseline", None)          # an outcome statistic of the window: not read here
    rp.PANEL_DIR = w["panel_dir"]
    syms, idx, C, V = rp.load_panel(*w["panel"])
    pdir = rp.OUT_DIR / f"panel_{w['panel'][0]}_{w['panel'][1]}"
    stale = Stale(panel_meta(w["panel_dir"], pdir, syms, idx), len(syms))
    LIQ = np.load(rp.OUT_DIR / f"blocks_{w['panel'][0]}_{w['panel'][1]}_r15.npz")["LIQ"]
    T, ib = C.shape[1], syms.index("BTCUSDT")
    bh.REF.update(bh.load_stock_ref())
    # ---- trigger minute t (alert `time` stamps the close of minute t) + reconciliation vs the alert price
    al = pd.read_parquet(rp.OUT_DIR / w["alerts"] / "alerts.parquet")
    al = al[al.kind == "NEW"][["asset", "time", "price"]].drop_duplicates(["asset", "time"])
    df = df.merge(al, on=["asset", "time"], how="left")
    df["t"] = ((df["time"] - idx[0]) // MIN).astype(int) - 1
    df["i"] = df["asset"].map(syms.index)
    px_t = np.array([float(C[i, t]) for i, t in zip(df["i"], df["t"])])
    bad = ~np.isclose(px_t, df["price"].to_numpy(float), rtol=1e-5, atol=0)
    st["T_RECONCILE_MISMATCH"] = int(bad.sum())
    if bad.sum():
        raise SystemExit(f"[{window}] trigger-minute reconciliation failed: {int(bad.sum())} rows")
    # ---- exclusions (counted statuses)
    st["SELF_BTC"] = int((df["asset"] == "BTCUSDT").sum())
    df = df[df["asset"] != "BTCUSDT"]
    stock = np.array([bh.stock_like(a, float(C[i, t])) for a, i, t in zip(df["asset"], df["i"], df["t"])], dtype=bool)
    st["TRADFI_PRICE"] = int(stock.sum())
    df = df[~stock].reset_index(drop=True)
    st["rows"] = int(len(df))
    print(f"[{window}] rows {len(df)} (BTC {st['SELF_BTC']}, stock-price {st['TRADFI_PRICE']})", flush=True)

    def ok_win(i, j0, j1):
        return 0 <= j0 and j1 < T and np.isfinite(C[i, j0]) and np.isfinite(C[i, j1]) and not stale.coin(i, j0, j1)

    # ---- x4h + trade legs per span (raw simple return of the coin, funding settled in (entry, exit])
    fund = bh.Funding()
    t_ms0 = idx[0].value // 10 ** 6
    pairs = set()
    for r in df.itertuples():
        for a, b in SPANS.values():
            pairs |= {(r.asset, m) for m in fund.months(t_ms0 + (r.t + a) * 60_000, t_ms0 + (r.t + b) * 60_000 + 59_999)}
    st["funding"] = bh.fetch_funding(pairs)
    legs = []
    for r in df.itertuples():
        i, t, d = int(r.i), int(r.t), int(r.dir)
        rec = {"liq": float(LIQ[i, t // 15])}
        e, x = t + 1, t + 241
        if ok_win(i, e, x):
            mk = mkt_median(C, LIQ, e, x, stale, exclude=i)
            rec["x4h"] = d * (math.log(C[i, x] / C[i, e]) - (mk if np.isfinite(mk) else 0.0))
        else:
            rec["x4h"] = np.nan
        for name, (a, b) in SPANS.items():
            j0, j1 = t + a, t + b
            if ok_win(i, j0, j1):
                rec[f"rc_{name}"] = float(C[i, j1]) / float(C[i, j0]) - 1.0
                rec[f"fund_{name}"] = fund.total(r.asset, t_ms0 + j0 * 60_000 + 59_999, t_ms0 + j1 * 60_000 + 59_999)
            else:
                rec[f"rc_{name}"] = rec[f"fund_{name}"] = np.nan
        c0, c1 = float(C[ib, t - 1440]) if t >= 1440 else np.nan, float(C[ib, t])
        rec["btc_24h"] = d * math.log(c1 / c0) if np.isfinite(c0) and np.isfinite(c1) and c0 > 0 else np.nan
        legs.append(rec)
    df = pd.concat([df, pd.DataFrame(legs, index=df.index)], axis=1)
    rp.mem(f"{window} legs")
    # ---- calendar (B5, B6)
    df["asia"] = (df["time"].dt.hour < 8).astype(float)
    df["weekend"] = (df["time"].dt.dayofweek >= 5).astype(float)
    # ---- +4 h positioning (runners.first4h: oi_chg_4h, top_ls_chg_4h, taker_ls_4h, premium_h4)
    df = first4h(df)
    # ---- B10 oi_chg_1h (Vision metrics, a row is usable once its 5-min span has ended)
    days = {(r.asset, d) for r in df.itertuples()
            for d in {(r.time - pd.Timedelta(minutes=65)).strftime("%Y-%m-%d"), r.time.strftime("%Y-%m-%d")}}
    met, _ = fetch_all(days)
    oi = []
    for r in df.itertuples():
        ds = sorted({(r.time - pd.Timedelta(minutes=65)).strftime("%Y-%m-%d"), r.time.strftime("%Y-%m-%d")})
        parts = [met.get((r.asset, d)) for d in ds]
        parts = [p for p in parts if p is not None and len(p)]
        m = pd.concat(parts).sort_index() if parts else None
        if m is not None:
            m = m[~m.index.duplicated()]
        o1 = metrics_row(m, r.time, "sum_open_interest")
        o0 = metrics_row(m, r.time - pd.Timedelta(minutes=60), "sum_open_interest")
        oi.append(math.log(o1 / o0) if o0 > 0 and o1 > 0 else np.nan)
    df["oi_chg_1h"] = oi
    rp.mem(f"{window} metrics")
    # ---- B1-B3 + perp close/volume (1m parquets)
    df = pd.concat([df, minute_feats(df, w["panel_dir"])], axis=1)
    rp.mem(f"{window} minute features")
    # ---- B7-B9 Upbit
    t0 = time.time()

    def one(r):
        raw = r.asset[:-4] if r.asset.endswith(("USDT", "USDC")) else r.asset
        return upbit_feats(base_of(r.asset), split_multiplier(raw)[1], int(r.dir), r.time, r.perp_px, r.perp_qv60)
    with ThreadPoolExecutor(4) as ex:
        up = list(ex.map(one, list(df.itertuples())))
    df = pd.concat([df, pd.DataFrame(up, index=df.index)], axis=1)
    st["upbit_status"] = df["upbit_status"].value_counts().to_dict()
    st["minute_status"] = df["minute_status"].value_counts().to_dict()
    st["upbit_seconds"] = round(time.time() - t0)
    st["missing"] = {f: int(df[f].isna().sum()) for f in NEW + [c[1] for c in CARRIED] + list(SPANS)}
    df.to_parquet(OUT / f"rows_{window}.parquet")
    (OUT / f"status_{window}.json").write_text(json.dumps(status, indent=1, default=str))
    print(json.dumps(status, default=str)[:3000], flush=True)
    rp.mem(f"{window} done")


# ----------------------------------------------------------------------------- statistics
def _rank(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank().to_numpy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _blocks(wk: np.ndarray, rng, draws: int):
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    for _ in range(draws):
        yield np.concatenate([pos[k] for k in rng.integers(0, len(uw), len(uw))])


def cell(df: pd.DataFrame, feat: str, out: str, seed: int = SEED, draws: int = DRAWS) -> dict:
    x, y = df[feat].to_numpy(float), df[out].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    wk = df["week"].to_numpy()[ok]
    x, y = x[ok], y[ok]
    n, nw = int(ok.sum()), int(len(np.unique(wk)))
    if n < MIN_N or nw < MIN_WEEKS or len(np.unique(x)) < 2:
        return {"status": "INSUFFICIENT", "n": n, "weeks": nw}
    ic = spearman(x, y)
    rng = np.random.default_rng(seed)
    b = np.array([spearman(x[s], y[s]) for s in _blocks(wk, rng, draws)])
    b = b[np.isfinite(b)]
    return {"status": "OK", "n": n, "weeks": nw, "ic": round(ic, 4),
            "ci": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "p_pos": (1 + int((b <= 0).sum())) / (len(b) + 1), "p_neg": (1 + int((b >= 0).sum())) / (len(b) + 1)}


def median_cell(df: pd.DataFrame, out: str, seed: int = SEED, draws: int = DRAWS) -> dict:
    y = df[out].to_numpy(float)
    ok = np.isfinite(y)
    wk, y = df["week"].to_numpy()[ok], y[ok]
    if len(y) < MIN_N or len(np.unique(wk)) < MIN_WEEKS:
        return {"status": "INSUFFICIENT", "n": int(len(y))}
    rng = np.random.default_rng(seed)
    b = np.array([np.median(y[s]) for s in _blocks(wk, rng, draws)])
    return {"status": "OK", "n": int(len(y)), "median_pct": round((math.exp(float(np.median(y))) - 1) * 100, 2),
            "ci_pct": [round((math.exp(float(np.percentile(b, q))) - 1) * 100, 2) for q in (2.5, 97.5)],
            "p_neg": (1 + int((b >= 0).sum())) / (len(b) + 1)}


def excludes0(c: dict) -> bool:
    return c.get("status") == "OK" and (c["ci"][0] > 0 or c["ci"][1] < 0)


def screen_pass(s1: dict, s2: dict, pooled: dict) -> bool:
    if any(c.get("status") != "OK" for c in (s1, s2, pooled)):
        return False
    sg = np.sign(pooled["ic"])
    return bool(sg != 0 and np.sign(s1["ic"]) == sg and np.sign(s2["ic"]) == sg and excludes0(pooled))


def verdict(c: dict, sign: int, k: int) -> str:
    if c.get("status") != "OK":
        return "INSUFFICIENT"
    if np.sign(c["ic"]) != sign or not excludes0(c):
        return "FAILED"
    p = c["p_pos"] if sign > 0 else c["p_neg"]
    return "VALIDATED" if p <= 0.05 / k else "SUGGESTIVE"


def pop_of(df: pd.DataFrame, pop: str) -> pd.DataFrame:
    return df if pop == "all" else df[df["dir"] == (1 if pop == "pumps" else -1)]


# ----------------------------------------------------------------------------- economics
def net(df: pd.DataFrame, span: str, follow: bool) -> np.ndarray:
    side = df["dir"].to_numpy(float) * (1 if follow else -1)
    rc, f = df[f"rc_{span}"].to_numpy(float), df[f"fund_{span}"].to_numpy(float)
    cost = np.array([2 * (bh.FEE + bh.slip(v)) for v in df["liq"].to_numpy(float)])
    return side * rc - cost - side * f


def mean_ci(vals: np.ndarray, wk: np.ndarray, seed: int, base: tuple | None = None) -> dict:
    ok = np.isfinite(vals)
    v, w = vals[ok], wk[ok]
    if len(v) < 20 or len(np.unique(w)) < MIN_WEEKS:
        return {"n": int(len(v)), "status": "INSUFFICIENT"}
    rng = np.random.default_rng(seed)
    b = np.array([v[s].mean() for s in _blocks(w, rng, 2000)])
    res = {"n": int(len(v)), "mean_pct": round(100 * v.mean(), 2),
           "ci_pct": [round(100 * float(np.percentile(b, q)), 2) for q in (2.5, 97.5)],
           "median_pct": round(100 * float(np.median(v)), 2), "win": round(float((v > 0).mean()), 3),
           "p5_pct": round(100 * float(np.percentile(v, 5)), 2), "status": "OK"}
    return res


def economics(df: pd.DataFrame, c: dict, seed: int) -> dict:
    sub = pop_of(df, c["pop"])
    span, x = c["outcome"], sub[c["feature"]].to_numpy(float)
    wk = sub["week"].to_numpy()
    if c["cuts"] is None:
        lo, hi = x == 0, x == 1
    else:
        lo, hi = x <= c["cuts"][0], x > c["cuts"][1]
    fade_grp, follow_grp = (lo, hi) if c["sign"] > 0 else (hi, lo)
    out = {}
    for name, grp, follow in (("fade", fade_grp, False), ("follow", follow_grp, True)):
        v_all = net(sub, span, follow)
        v_grp = np.where(grp, v_all, np.nan)
        out[name] = {"group": mean_ci(v_grp, wk, seed), "all_events": mean_ci(v_all, wk, seed),
                     "group_minus_all_pct": diff_ci(v_grp, v_all, wk, seed)}
    return out


def diff_ci(v_grp: np.ndarray, v_all: np.ndarray, wk: np.ndarray, seed: int) -> dict:
    """mean(group) - mean(all events), the same week draws for both (the group is a subset)."""
    if np.isfinite(v_grp).sum() < 20:
        return {"status": "INSUFFICIENT"}
    rng = np.random.default_rng(seed)
    b = []
    for s in _blocks(wk, rng, 2000):
        g, a = v_grp[s], v_all[s]
        if np.isfinite(g).any() and np.isfinite(a).any():
            b.append(np.nanmean(g) - np.nanmean(a))
    return {"diff": round(100 * (np.nanmean(v_grp) - np.nanmean(v_all)), 2),
            "ci": [round(100 * float(np.percentile(b, q)), 2) for q in (2.5, 97.5)]}


def partial_ic(df: pd.DataFrame, feat: str, out: str, controls=("liq_pre", "mkt_breadth", "calm_7d")) -> float:
    cols = [feat, out, *[c for c in controls if c != feat]]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < MIN_N:
        return float("nan")
    R = d.rank().to_numpy()
    X = np.column_stack([np.ones(len(d)), R[:, 2:]])
    res = [R[:, k] - X @ np.linalg.lstsq(X, R[:, k], rcond=None)[0] for k in (0, 1)]
    return round(float(np.corrcoef(*res)[0, 1]), 4)


# ----------------------------------------------------------------------------- self-tests
def falsifiers(df: pd.DataFrame) -> dict:
    """The instrument must flag a planted feature, refuse a shuffled one, and call all-missing INSUFFICIENT."""
    rng = np.random.default_rng(7)
    d = pop_of(df, "pumps").copy()
    y = d["x72h"].to_numpy(float)
    z = (_rank(np.nan_to_num(y)) - len(y) / 2) / (len(y) / math.sqrt(12))
    d["planted"] = np.where(np.isfinite(y), 0.15 * z + math.sqrt(1 - 0.0225) * rng.standard_normal(len(y)), np.nan)
    d["noise"] = rng.standard_normal(len(y))
    d["empty"] = np.nan
    res = {}
    for f in ("planted", "noise", "empty"):
        s1 = cell(d[d.window == "discovery"], f, "x72h", draws=1000)
        s2 = cell(d[d.window == "holdout"], f, "x72h", draws=1000)
        pl = cell(d, f, "x72h", draws=1000)
        res[f] = {"pass": screen_pass(s1, s2, pl), "pooled": pl,
                  "verdict_k10": verdict(pl, int(np.sign(pl.get("ic", 0)) or 1), 10)}
    ok = (res["planted"]["pass"] and res["planted"]["verdict_k10"] == "VALIDATED" and not res["noise"]["pass"]
          and res["noise"]["verdict_k10"] != "VALIDATED" and res["empty"]["pooled"]["status"] == "INSUFFICIENT")
    res["ok"] = bool(ok)
    return res


def reconcile(df: pd.DataFrame) -> dict:
    out = {}
    for wname in SCREEN:
        ref = pd.read_parquet(rp.OUT_DIR / "persistence" / f"features_{wname}.parquet")[["asset", "time", "x72h"]]
        m = df[df.window == wname][["asset", "time", "x72h"]].merge(ref, on=["asset", "time"], suffixes=("", "_ref"))
        a, b = m["x72h"].to_numpy(float), m["x72h_ref"].to_numpy(float)
        bad = ~((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=0, atol=1e-9))
        out[f"x72h_{wname}"] = {"matched": int(len(m)), "rows": int((df.window == wname).sum()), "mismatch": int(bad.sum())}
    return out


def reconcile_premium(df: pd.DataFrame) -> dict:
    """premium_h4 here vs the runners study's value (same function, same Vision files)."""
    pop = pd.read_parquet(rp.OUT_DIR / "runners" / "pumps.parquet")
    m = df[["window", "asset", "time", "premium_h4"]].merge(pop[["window", "asset", "time", "premium_h4"]],
                                                           on=["window", "asset", "time"], suffixes=("", "_ref"))
    both = m["premium_h4"].notna() & m["premium_h4_ref"].notna()
    bad = both & ~np.isclose(m["premium_h4"], m["premium_h4_ref"], rtol=0, atol=1e-12)
    return {"matched": int(len(m)), "both_finite": int(both.sum()), "mismatch": int(bad.sum())}


# ----------------------------------------------------------------------------- phases
def load_rows(windows) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(OUT / f"rows_{w}.parquet") for w in windows], ignore_index=True)


def screen() -> None:
    df = load_rows(SCREEN)
    res: dict = {"declaration": "docs/SIGNAL_SEARCH.md", "declaration_commit": DECLARATION_COMMIT,
                 "rows": df.groupby(["window", "dir"]).size().rename(lambda x: str(x)).to_dict()}
    rec = reconcile(df)
    rec["premium_h4"] = reconcile_premium(df)
    res["reconciliation"] = rec
    print("reconciliation", json.dumps(rec), flush=True)
    if any(v.get("mismatch", 0) for v in rec.values()) or any(v["matched"] != v["rows"] for k, v in rec.items()
                                                              if k.startswith("x72h")):
        raise SystemExit("reconciliation failed")
    fz = falsifiers(df)
    res["falsifiers"] = fz
    print("falsifiers ok:", fz["ok"], {k: (v["pass"], v["verdict_k10"], v["pooled"].get("ic")) for k, v in fz.items()
                                      if k != "ok"}, flush=True)
    if not fz["ok"]:
        raise SystemExit("falsifier failed: the instrument cannot be trusted")
    cells, carried = [], []
    for f in NEW:
        for pop in ("pumps", "dumps"):
            for out in ("x4h", "x72h"):
                sub = pop_of(df, pop)
                s1 = cell(sub[sub.window == "discovery"], f, out)
                s2 = cell(sub[sub.window == "holdout"], f, out)
                pl = cell(sub, f, out)
                ok = screen_pass(s1, s2, pl)
                both = ok and excludes0(s1) and excludes0(s2)
                sus = pl.get("status") == "OK" and abs(pl["ic"]) > 0.10
                cells.append({"feature": f, "pop": pop, "outcome": out, "S1": s1, "S2": s2, "pooled": pl,
                              "screen_pass": ok, "both_windows": both, "suspicious": sus})
                print(f"{f:13s} {pop:5s} {out:5s} S1 {s1.get('ic', s1['status'])!s:>8} S2 {s2.get('ic', s2['status'])!s:>8} "
                      f"pooled {pl.get('ic', pl['status'])!s:>8} {pl.get('ci', '')} n={pl['n']}"
                      f"{'  PASS' if ok else ''}{' both' if both else ''}{' SUSPICIOUS' if sus else ''}", flush=True)
    res["cells"] = cells
    # Group A on S, for context only (they were chosen on these windows)
    res["carried_on_S"] = {cid: cell(pop_of(df, pop), f, out) for cid, f, pop, out, _ in CARRIED}
    res["carried_on_S"]["A8"] = median_cell(pop_of(df, "pumps"), "x72h")
    # ---- freeze (computed): A1-A8 + every screen-passing B cell
    for cid, f, pop, out, sg in CARRIED:
        carried.append({"id": cid, "feature": f, "pop": pop, "outcome": out, "sign": sg})
    b = [c for c in cells if c["screen_pass"]]
    for k, c in enumerate(b, 1):
        carried.append({"id": f"B{NEW.index(c['feature']) + 1}.{c['pop']}.{c['outcome']}", "feature": c["feature"],
                        "pop": c["pop"], "outcome": c["outcome"], "sign": int(np.sign(c["pooled"]["ic"]))})
    for c in carried:
        x = pop_of(df, c["pop"])[c["feature"]].to_numpy(float)
        x = x[np.isfinite(x)]
        c["cuts"] = None if c["feature"] in BINARY else [float(np.percentile(x, 100 / 3)), float(np.percentile(x, 200 / 3))]
    k = len(carried) + 1                                   # + A8, the base-rate test
    res["freeze_preview"] = {"k": k, "cells": carried}
    (OUT / "screen_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\ncarried to V: {len(carried)} IC cells + A8 base rate = k {k}", flush=True)
    for c in carried:
        print("  ", c["id"], c["feature"], c["pop"], c["outcome"], c["sign"], c["cuts"], flush=True)
    if not committed_clean(Path(__file__)):
        print("NOT writing the freeze: research/signal_search.py has uncommitted changes (rule 12)", flush=True)
        return
    FREEZE.write_text(json.dumps({"declaration": "docs/SIGNAL_SEARCH.md", "declaration_commit": DECLARATION_COMMIT,
                                  "script_commit": head(), "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                                  "k": k, "base_rate": {"id": "A8", "pop": "pumps", "outcome": "x72h", "sign": -1},
                                  "cells": carried}, indent=1))
    print(f"freeze written: {FREEZE} (commit it before building V)", flush=True)


def validate() -> None:
    fz = require_freeze()
    if RECEIPT.exists():
        raise SystemExit(f"REFUSED: {RECEIPT} exists; V is read once")
    for w in VAL:
        if not (OUT / f"rows_{w}.parquet").exists():
            raise SystemExit(f"REFUSED: rows_{w}.parquet missing (build it first)")
    v = load_rows(VAL).sort_values("time")
    # 72 h per-coin de-dup across the val_a / val_b seam (rows are actions)
    keep, last, seam = [], {}, 0
    for r in v.itertuples():
        lt = last.get(r.asset)
        if lt is None or r.time - lt >= pd.Timedelta(hours=72):
            keep.append(r.Index)
            last[r.asset] = r.time
        else:
            seam += 1
    v = v.loc[keep].copy()
    v["week"] = ps.cluster_weeks(v["time"])
    k = fz["k"]
    rec = {"freeze": fz, "freeze_file_commit": _git("log", "-1", "--format=%H", "--", str(FREEZE.relative_to(ROOT))).stdout.strip(),
           "script_commit_at_validation": head(), "run_utc": pd.Timestamp.now(tz="UTC").isoformat(),
           "rows": v.groupby("dir").size().rename(lambda x: str(x)).to_dict(), "SEAM_DUPLICATE": seam,
           "weeks": int(v["week"].nunique()), "flag": "A6/A7 overlap thematically with the pump->dump thread's era reads"}
    S = load_rows(SCREEN)
    results = []
    for c in fz["cells"]:
        cc = cell(pop_of(v, c["pop"]), c["feature"], c["outcome"])
        vd = verdict(cc, c["sign"], k)
        r = {**c, "V": cc, "verdict": vd}
        if vd == "VALIDATED":
            r["economics_V"] = economics(v, c, SEED)
            r["economics_S"] = economics(S, c, SEED)
            # TRADABLE (declared): a side whose V mean-net CI is above 0 and whose S1+S2 mean is above 0
            r["tradable"] = bool(any(
                r["economics_V"][kk]["group"].get("status") == "OK" and r["economics_V"][kk]["group"]["ci_pct"][0] > 0
                and r["economics_S"][kk]["group"].get("mean_pct", -1) > 0 for kk in ("fade", "follow")))
            r["partial_ic_V"] = partial_ic(pop_of(v, c["pop"]), c["feature"], c["outcome"])
        results.append(r)
        print(f"{c['id']:22s} {c['feature']:13s} {c['pop']:5s} {c['outcome']:5s} sign {c['sign']:+d}  "
              f"V ic {cc.get('ic', cc['status'])} {cc.get('ci', '')} n={cc['n']}  -> {vd}", flush=True)
    br = median_cell(pop_of(v, "pumps"), "x72h")
    br["verdict"] = ("VALIDATED" if br.get("status") == "OK" and br["ci_pct"][1] < 0 and br["p_neg"] <= 0.05 / k
                     else "SUGGESTIVE" if br.get("status") == "OK" and br["ci_pct"][1] < 0 else "FAILED")
    rec["A8_base_rate"] = br
    print("A8 pumps median x72h", br, flush=True)
    rec["cells"] = results
    rec["descriptive"] = {
        "dumps_median_x72h": median_cell(pop_of(v, "dumps"), "x72h"),
        "pump_fade_all": {s: mean_ci(net(pop_of(v, "pumps"), s, False), pop_of(v, "pumps")["week"].to_numpy(), SEED)
                          for s in ("x4h", "x72h")},
    }
    rec["n_validated"] = sum(r["verdict"] == "VALIDATED" for r in results) + (br["verdict"] == "VALIDATED")
    RECEIPT.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec["descriptive"], default=str), flush=True)
    print(f"receipt: {RECEIPT}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "screen", "validate"])
    ap.add_argument("--window", choices=list(WINDOWS))
    a = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    if a.phase == "build":
        build_window(a.window)
    elif a.phase == "screen":
        screen()
    else:
        validate()


if __name__ == "__main__":
    main()
