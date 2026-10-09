"""Upbit listing announcements -> Binance USDM price reaction (docs/UPBIT_LISTINGS.md).

  python3 -m research.upbit_listings fetch     # all 'trade' notices from Upbit's announcement API (no prices)
  python3 -m research.upbit_listings census    # outcome-blind: new-KRW-listing events per year with a Binance perp
  python3 -m research.upbit_listings paths --window {disc,conf}   # per-event paths, trades, same-coin null
  python3 -m research.upbit_listings screen    # discovery (2025-09..2026-09, partly consumed): carry passing cells
  python3 -m research.upbit_listings validate  # confirmation (2020..2025-08, untouched): read once after the freeze
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import research.btc_hedge as bh

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "hist" / "upbit_announcements.json"
API = "https://api-manager.upbit.com/api/v1/announcements"
HIST_DIRS = ["usdm_1m_early", "usdm_1m_hist", "usdm_1m_val", "usdm_1m_holdout", "usdm_1m"]
NEW_LISTING = "신규 거래지원 안내"                       # "new trading support notice"


def fetch() -> None:
    out, page = [], 1
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    while True:
        for k in range(8):                                   # the API rate-limits (429): back off and retry
            r = s.get(API, params={"os": "web", "page": page, "per_page": 20, "category": "trade"}, timeout=20)
            if r.status_code != 429:
                break
            time.sleep(10 * (k + 1))
        r.raise_for_status()
        d = r.json()["data"]
        out += d["notices"]
        if page >= d["total_pages"]:
            break
        page += 1
        time.sleep(2.5)
    ANN.write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print(f"notices {len(out)}, pages {page}, first {out[-1]['listed_at']}, last {out[0]['listed_at']}")


LISTING_WORDS = ("신규 거래지원", "디지털 자산 추가", "신규 상장", "코인 추가")   # 2024+, 2020-23, 2019, 2018
NOT_TICKERS = {"KRW", "BTC", "USDT", "ETH"}


def parse(notices: list[dict]) -> pd.DataFrame:
    """One row per ticker: its FIRST notice that opens a KRW market (later notices on the same listing,
    e.g. delays or name changes, are dropped)."""
    rows = []
    for n in notices:
        t = n["title"]
        if not any(w in t for w in LISTING_WORDS) or not ("KRW" in t or "원화" in t):
            continue
        for tk in set(re.findall(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9]{1,14})(?![A-Za-z0-9])", t)) - NOT_TICKERS:
            rows.append({"id": n["id"], "time": pd.Timestamp(n["first_listed_at"]).tz_convert("UTC"),
                         "ticker": tk, "title": t})
    df = pd.DataFrame(rows).sort_values("time")
    return df.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def perp_coverage(ticker: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(dir, first, last) of Binance USDM 1m data for <ticker>USDT (or 1000<ticker>USDT) across the corpora."""
    cov = []
    for d in HIST_DIRS:
        for sym in (f"{ticker}USDT", f"1000{ticker}USDT"):
            p = ROOT / "data" / "hist" / d / f"{sym}.parquet"
            if p.exists():
                idx = pd.read_parquet(p, columns=["close"]).index
                if len(idx):
                    cov.append((d, sym, idx.min(), idx.max()))
    return cov


def census() -> None:
    ev = parse(json.loads(ANN.read_text()))
    ok = []
    for r in ev.itertuples():
        hit = [(d, s) for d, s, a, b in perp_coverage(r.ticker) if a <= r.time - pd.Timedelta(hours=1) and b >= r.time + pd.Timedelta(hours=5)]
        ok.append(hit[0] if hit else (None, None))
    ev["dir"], ev["symbol"] = [h[0] for h in ok], [h[1] for h in ok]
    ev.to_parquet(ROOT / "data" / "replay" / "upbit_listing_events.parquet")
    by = ev.groupby(ev["time"].dt.year).agg(krw_listings=("ticker", "size"), with_binance_perp=("symbol", lambda x: x.notna().sum()))
    print(by.to_string())
    print("total", len(ev), "with perp", int(ev["symbol"].notna().sum()))


# ---------------------------------------------------------------------- event study
OUT = ROOT / "data" / "replay" / "upbit_listings"
FREEZE = ROOT / "docs" / "UPBIT_LISTINGS_FREEZE.json"
RECEIPT = OUT / "receipt_conf.json"
SPLIT = pd.Timestamp("2025-09-01", tz="UTC")              # disc >= SPLIT (partly consumed); conf < SPLIT (untouched)
STRESS, SEED, DRAWS, N_NULL, MIN_EVENTS, MIN_WEEKS = 3, 20261009, 4000, 20, 20, 5
PATH = (0, 1, 2, 5, 15, 60, 240, 1440)
CELLS = {**{f"long_{h}": ("long", 1, 1 + h) for h in (5, 15, 60, 240)},            # buy close a+1, sell close a+1+h
         **{f"short_{o}_1440": ("short", o, 1440) for o in (15, 60, 240)}}        # short close a+o, cover close a+1440


def _git(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)


def committed_clean(p: Path) -> bool:
    rel = str(p.relative_to(ROOT))
    return _git("ls-files", "--error-unmatch", rel).returncode == 0 and _git("diff", "--quiet", "HEAD", "--", rel).returncode == 0


def _px(s: pd.Series, t: pd.Timestamp) -> float:
    """Close of minute t, else the last close before it (status counted by the caller)."""
    v = s.get(t)
    if v is not None and np.isfinite(v):
        return float(v)
    x = s.loc[:t].dropna()
    return float(x.iloc[-1]) if len(x) else float("nan")


def trades_at(close: pd.Series, a: pd.Timestamp, cost: float) -> dict:
    m = pd.Timedelta(minutes=1)
    out = {}
    for name, (side, o, x) in CELLS.items():
        p0, p1 = _px(close, a + o * m), _px(close, a + x * m)
        r = p1 / p0 - 1.0
        out[name] = (r if side == "long" else -r) - cost
    return out


def event_rows(ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, nulls = [], []
    rng = np.random.default_rng(SEED)
    m = pd.Timedelta(minutes=1)
    for r in ev.itertuples():
        d = pd.read_parquet(ROOT / "data" / "hist" / r.dir / f"{r.symbol}.parquet", columns=["close", "quote_volume"])
        d = d[~d.index.duplicated()]
        a = r.time.floor("min")
        d = d.loc[a - pd.Timedelta(days=31):a + pd.Timedelta(minutes=1500)]
        if d.index.max() < a + 1442 * m:
            rows.append({"ticker": r.ticker, "time": r.time, "status": "NO_24H_DATA"})
            continue
        close = d["close"].astype(np.float64)
        liq = float(d["quote_volume"].loc[a - 1440 * m:a - m].sum())
        cost = 2 * (bh.FEE + STRESS * bh.slip(liq))
        pre = _px(close, a - m)
        rec = {"ticker": r.ticker, "symbol": r.symbol, "time": r.time, "status": "OK", "liq": liq, "cost": cost,
               "pre_drift_60": pre / _px(close, a - 61 * m) - 1.0,
               **{f"path_{k}": _px(close, a + k * m) / pre - 1.0 for k in PATH}, **trades_at(close, a, cost)}
        rows.append(rec)
        cand = close.loc[a - pd.Timedelta(days=30):a - pd.Timedelta(days=2)].index
        cand = cand[cand.hour == a.hour] if (cand.hour == a.hour).sum() >= N_NULL else cand
        for t in cand[rng.choice(len(cand), size=min(N_NULL, len(cand)), replace=False)] if len(cand) else []:
            nulls.append({"time": r.time, **trades_at(close, t, cost)})
    return pd.DataFrame(rows), pd.DataFrame(nulls)


def paths(window: str) -> None:
    ev = pd.read_parquet(ROOT / "data" / "replay" / "upbit_listing_events.parquet")
    ev = ev[ev["symbol"].notna()]
    ev = ev[ev["time"] >= SPLIT] if window == "disc" else ev[ev["time"] < SPLIT]
    if window == "conf":
        if not (FREEZE.exists() and committed_clean(FREEZE) and committed_clean(Path(__file__).resolve())):
            raise SystemExit("REFUSED: the freeze file and this script must be committed before the confirmation window is read")
        if RECEIPT.exists():
            raise SystemExit("REFUSED: confirmation already read (read once)")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nulls = event_rows(ev)
    rows.to_parquet(OUT / f"events_{window}.parquet")
    nulls.to_parquet(OUT / f"null_{window}.parquet")
    print(json.dumps({"window": window, "events": int(len(rows)), "status": rows["status"].value_counts().to_dict(),
                      "null_rows": int(len(nulls))}))


def _weeks(t: pd.Series) -> np.ndarray:
    return pd.DatetimeIndex(t).tz_convert(None).to_period("W").start_time.to_numpy()


def stats(v: np.ndarray, wk: np.ndarray, seed: int = SEED) -> dict:
    ok = np.isfinite(v)
    v, wk = v[ok], wk[ok]
    n, nw = len(v), len(np.unique(wk))
    if n < MIN_EVENTS or nw < MIN_WEEKS:
        return {"status": "INSUFFICIENT", "n": int(n), "mean": float(v.mean()) if n else float("nan")}
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == w) for w in uw]
    rng = np.random.default_rng(seed)
    b = np.array([v[np.concatenate([pos[j] for j in rng.integers(0, len(uw), len(uw))])].mean() for _ in range(DRAWS)])
    return {"status": "OK", "n": int(n), "weeks": int(nw), "mean": float(v.mean()), "median": float(np.median(v)),
            "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "p": float((1 + (b <= 0).sum()) / (DRAWS + 1)),
            "win": float((v > 0).mean()), "drop_best": float(np.sort(v)[:-1].mean())}


def null_p(obs: float, ev_times: pd.Series, null: pd.DataFrame, c: str, seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    groups = [null.loc[null["time"] == t, c].dropna().to_numpy() for t in ev_times]
    groups = [g for g in groups if len(g)]
    nm = np.array([np.mean([g[rng.integers(len(g))] for g in groups]) for _ in range(2000)])
    return float((1 + (nm >= obs).sum()) / 2001)


def screen() -> None:
    e = pd.read_parquet(OUT / "events_disc.parquet")
    e = e[e["status"] == "OK"]
    nl = pd.read_parquet(OUT / "null_disc.parquet")
    wk = _weeks(e["time"])
    res, carry = {}, []
    for c in CELLS:
        st = stats(e[c].to_numpy(), wk)
        st["null_mean"] = float(nl[c].mean())
        res[c] = st
        if st["status"] == "OK" and st["mean"] > 0 and st["ci"][0] > 0:
            carry.append(c)
    path = {f"path_{k}": {"median": float(e[f"path_{k}"].median()), "mean": float(e[f"path_{k}"].mean())} for k in PATH}
    path["pre_drift_60"] = {"median": float(e["pre_drift_60"].median()), "mean": float(e["pre_drift_60"].mean())}
    freeze = {"declaration": "docs/UPBIT_LISTINGS.md", "script_commit": _git("rev-parse", "HEAD").stdout.strip(),
              "screen_window": "2025-09-01..2026-09-26 (partly consumed)", "carried": carry, "k": len(carry),
              "screen": res, "path": path, "n_disc": int(len(e))}
    FREEZE.write_text(json.dumps(freeze, indent=1))
    print(json.dumps({"carried": carry, "k": len(carry), "n": len(e),
                      "cells": {c: {k: v for k, v in r.items() if k in ("status", "n", "mean", "ci", "win", "null_mean")}
                                for c, r in res.items()}, "path": path}, indent=1))


def validate() -> None:
    if RECEIPT.exists():
        raise SystemExit("REFUSED: confirmation already read (read once)")
    fz = json.loads(FREEZE.read_text())
    e = pd.read_parquet(OUT / "events_conf.parquet")
    e = e[e["status"] == "OK"]
    nl = pd.read_parquet(OUT / "null_conf.parquet")
    wk = _weeks(e["time"])
    k = max(fz["k"], 1)
    # falsifiers: planted (null draws + 10%) must validate; null-as-events must not
    rng = np.random.default_rng(SEED + 7)
    fal = {}
    for c in fz["carried"]:
        draw = np.array([rng.choice(nl.loc[nl["time"] == t, c].to_numpy()) for t in e["time"]])
        pl, nu = stats(draw + 0.10, wk), stats(draw, wk)
        fal[c] = {"planted_ok": bool(pl["status"] == "OK" and pl["ci"][0] > 0 and pl["p"] <= 0.05 / k),
                  "null_ok": bool(not (nu["status"] == "OK" and nu["ci"][0] > 0 and nu["p"] <= 0.05 / k))}
        if not (fal[c]["planted_ok"] and fal[c]["null_ok"]):
            raise SystemExit(f"FALSIFIER failed for {c}: {fal[c]}")
    res = {}
    for c in CELLS:
        st = stats(e[c].to_numpy(), wk)
        if st["status"] == "OK":
            st["p_null"] = null_p(st["mean"], e["time"], nl, c)
        st["carried"] = c in fz["carried"]
        if st["carried"]:
            ok = st["status"] == "OK" and st["mean"] > 0 and st["ci"][0] > 0
            st["verdict"] = ("VALIDATED" if ok and st["p"] <= 0.05 / k and st["p_null"] <= 0.05 and st["drop_best"] > 0
                             else "SUGGESTIVE" if ok else "INSUFFICIENT" if st["status"] != "OK" else "FAILED")
        res[c] = st
    path = {f"path_{x}": {"median": float(e[f"path_{x}"].median()), "mean": float(e[f"path_{x}"].mean())} for x in PATH}
    out = {"run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "freeze_commit":
           _git("log", "-1", "--format=%H", "--", str(FREEZE.relative_to(ROOT))).stdout.strip(),
           "n": int(len(e)), "k": k, "falsifiers": fal, "cells": res, "path": path,
           "per_year": {c: e.groupby(e["time"].dt.year)[c].agg(["size", "mean"]).round(5).to_dict("index") for c in fz["carried"]}}
    RECEIPT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "per_year"}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["fetch", "census", "paths", "screen", "validate"])
    ap.add_argument("--window", choices=["disc", "conf"])
    a = ap.parse_args()
    if a.phase == "paths":
        return paths(a.window)
    {"fetch": fetch, "census": census, "screen": screen, "validate": validate}[a.phase]()


if __name__ == "__main__":
    main()
