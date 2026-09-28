"""Do extra metrics (OI, positioning, premium, breadth, detection latency, ...) tell which
signals continue? Protocol declared in docs/SIGNAL_FEATURES.md BEFORE this ran.

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.signal_features
Outputs: data/replay/signal_features/{features.parquet, results.json}
"""
from __future__ import annotations

import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "ARROW_DEFAULT_MEMORY_POOL": "system"}.items():
    os.environ.setdefault(_k, _v)

import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from movewatch.instruments import split_concat_symbol, split_multiplier  # noqa: E402
from research.common import METRICS_SPAN, Stale, is_tradfi, mkt_median, panel_meta  # noqa: E402
from research.replay import OUT_DIR, PANEL_DIR, load_panel, mem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
VISION = "https://data.binance.vision/data/futures/um/daily"
H_ORD = {"1m": 1, "5m": 2, "15m": 3, "1h": 4, "4h": 5, "24h": 6}
H_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "24h": 1440}
FEATURES = ["oi_chg_1h", "oi_chg_4h", "top_ls", "top_ls_chg_4h", "acct_ls", "taker_ls_1h", "premium",
            "premium_chg_1h", "taker_buy_share_15m", "extension", "minutes_running", "trigger_h", "z", "vm",
            "log_liq", "mkt_breadth", "venue_count", "prior_signals_7d"]
WINDOWS = {"aug": ("2026-08-02", "2026-09-01"), "sep01_19": ("2026-09-01", "2026-09-20"),
           "sep20_27_consumed": ("2026-09-20", "2026-09-28")}
RNG = np.random.default_rng(20260928)
S = requests.Session()


# ----------------------------------------------------------------------------- Vision downloads
def _vision(kind: str, sym: str, day: str) -> pd.DataFrame | None:
    path = HIST / f"vision_{kind}" / sym / f"{day}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = (f"{VISION}/metrics/{sym}/{sym}-metrics-{day}.zip" if kind == "metrics"
           else f"{VISION}/premiumIndexKlines/{sym}/1m/{sym}-1m-{day}.zip")
    for k in range(4):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0]).decode()
            if kind == "metrics":
                df = pd.read_csv(io.StringIO(raw))
                df["ts"] = pd.to_datetime(df["create_time"], utc=True)
                df = df.drop(columns=["create_time", "symbol"]).sort_values("ts").set_index("ts").astype(float)
            else:
                lines = [ln.split(",") for ln in raw.splitlines() if ln and ln[0].isdigit()]
                df = pd.DataFrame({"ts": pd.to_datetime([int(x[0]) for x in lines], unit="ms", utc=True),
                                   "premium": [float(x[4]) for x in lines]}).set_index("ts").sort_index()
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            return df
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(1.5 * (k + 1))
    return None


def fetch_all(pairs: set[tuple[str, str]]) -> tuple[dict, dict]:
    jobs = [(k, s, d) for (s, d) in sorted(pairs) for k in ("metrics", "premium")]
    out = {"metrics": {}, "premium": {}}
    with ThreadPoolExecutor(8) as ex:
        for (k, s, d), df in zip(jobs, ex.map(lambda j: _vision(*j), jobs)):
            out[k][(s, d)] = df
    return out["metrics"], out["premium"]


def series(store: dict, sym: str, t_end: pd.Timestamp, lookback_min: int) -> pd.DataFrame | None:
    days = sorted({(t_end - pd.Timedelta(minutes=lookback_min + 5)).strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d")})
    parts = [store.get((sym, d)) for d in days]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


# ----------------------------------------------------------------------------- venue breadth
def venue_bases() -> dict[str, set]:
    """Bases listed on each of the 12 feeds TODAY (static; mild look-ahead, stated)."""
    def base_of(raw):
        return split_multiplier(raw)[0].upper()
    g = lambda u, **kw: S.get(u, timeout=30, **kw).json()  # noqa: E731
    v: dict[str, set] = {}
    px = g("https://data-api.binance.vision/api/v3/ticker/price")
    v["binance_spot"] = {base_of(p["symbol"][:-4]) for p in px if p["symbol"].endswith("USDT")}
    v["binance_usdm"] = {base_of(s[:-4]) for s in
                         (ROOT / "data" / "usdm_symbol_dirs.txt").read_text().split() if s.endswith("USDT")}
    txt = S.get("https://public.bybit.com/trading/", timeout=30).text
    v["bybit_linear"] = {base_of(s[:-4]) for s in re.findall(r'href="([A-Z0-9]+USDT)/"', txt)}
    for t, name in (("SPOT", "okx_spot"), ("SWAP", "okx_swap")):
        d = g("https://www.okx.com/api/v5/public/instruments", params={"instType": t})["data"]
        v[name] = {base_of(x["instId"].split("-")[0]) for x in d if x.get("instCategory") == "1"}
    v["coinbase"] = {base_of(p["base_currency"]) for p in g("https://api.exchange.coinbase.com/products")
                     if p.get("status") == "online"}
    v["upbit"] = {m["market"].split("-", 1)[1] for m in g("https://api.upbit.com/v1/market/all") if m["market"].startswith("KRW-")}
    v["bithumb"] = {m["market"].split("-", 1)[1] for m in g("https://api.bithumb.com/v1/market/all") if m["market"].startswith("KRW-")}
    v["bitget_spot"] = {base_of(x["baseCoin"]) for x in g("https://api.bitget.com/api/v2/spot/public/symbols")["data"]}
    v["bitget_fut"] = {base_of(x["baseCoin"]) for x in
                       g("https://api.bitget.com/api/v2/mix/market/contracts", params={"productType": "USDT-FUTURES"})["data"]}
    meta = S.post("https://api.hyperliquid.xyz/info", json={"type": "meta"}, timeout=30).json()
    v["hyperliquid"] = {base_of(u["name"]) for u in meta["universe"] if not u.get("isDelisted")}
    v["bybit_spot"] = v["binance_spot"] & v["bybit_linear"]   # proxy: Bybit spot REST is geo-blocked here
    return v


# ----------------------------------------------------------------------------- features
def build(sig: pd.DataFrame, syms, idx, C, LIQ, metrics, premium, venues, stale: Stale) -> pd.DataFrame:
    """v2 (review 2026-09-28): metrics point-in-time (a row stamped T spans [T, T+5 min)),
    market median excludes the coin and frozen coins, outcomes on frozen bars -> NaN."""
    T = C.shape[1]
    ts0 = idx[0]
    taker = {}
    rows = []
    for a in sig.itertuples():
        i = syms.index(a.asset)
        t = int(a.t)
        d = int(a.direction)
        ta = a.time                                   # alert time = close of minute t
        rec = {"time": ta, "day": ta.floor("D"), "asset": a.asset, "dir": d, "horizon": a.horizon}
        # ---- outcomes (entry one minute after the alert)
        e = t + 1
        blk = e // 15
        liq_mask = LIQ[:, blk] >= 5e6
        for k in (60, 240, 1440):
            if e + k < T and np.isfinite(C[i, e]) and np.isfinite(C[i, e + k]) and not stale.coin(i, e, e + k):
                mk = mkt_median(C, LIQ, e, e + k, stale, exclude=i)
                rec[f"x{k}"] = d * (np.log(C[i, e + k] / C[i, e]) - (mk if np.isfinite(mk) else 0.0))
            else:
                rec[f"x{k}"] = np.nan
        # ---- metrics (OI, positioning)
        m = series(metrics, a.asset, ta, 250)
        if m is not None:
            m = m[m.index + METRICS_SPAN <= ta]           # rows whose 5-minute span ended by the alert
        def at(df, col, minutes_ago):
            if df is None or not len(df):
                return np.nan
            x = df[df.index + METRICS_SPAN <= ta - pd.Timedelta(minutes=minutes_ago)]
            return float(x[col].iloc[-1]) if len(x) else np.nan
        oi0, oi1, oi4 = at(m, "sum_open_interest", 0), at(m, "sum_open_interest", 60), at(m, "sum_open_interest", 240)
        rec["oi_chg_1h"] = oi0 / oi1 - 1 if oi1 and oi1 > 0 else np.nan
        rec["oi_chg_4h"] = oi0 / oi4 - 1 if oi4 and oi4 > 0 else np.nan
        tl0, tl4 = at(m, "sum_toptrader_long_short_ratio", 0), at(m, "sum_toptrader_long_short_ratio", 240)
        rec["top_ls"] = d * np.log(tl0) if tl0 and tl0 > 0 else np.nan
        rec["top_ls_chg_4h"] = d * np.log(tl0 / tl4) if tl0 and tl4 and tl0 > 0 and tl4 > 0 else np.nan
        al0 = at(m, "count_long_short_ratio", 0)
        rec["acct_ls"] = d * np.log(al0) if al0 and al0 > 0 else np.nan
        if m is not None and len(m):
            tk = m[m.index + METRICS_SPAN > ta - pd.Timedelta(minutes=60)]["sum_taker_long_short_vol_ratio"]
            tk = tk[tk > 0]
            rec["taker_ls_1h"] = d * float(np.log(tk).mean()) if len(tk) else np.nan
        else:
            rec["taker_ls_1h"] = np.nan
        # ---- premium index (perp vs spot index)
        p = series(premium, a.asset, ta, 60)
        def prem(minutes_ago):
            if p is None:
                return np.nan
            x = p[p.index <= ta - pd.Timedelta(minutes=minutes_ago + 1)]   # kline open <= t-1m -> closed by t
            return float(x["premium"].iloc[-1]) if len(x) else np.nan
        p0, p60 = prem(0), prem(60)
        rec["premium"] = d * p0
        rec["premium_chg_1h"] = d * (p0 - p60) if np.isfinite(p0) and np.isfinite(p60) else np.nan
        # ---- taker buy share (USDM klines)
        if a.asset not in taker:
            df = pd.read_parquet(HIST / "usdm_1m" / f"{a.asset}.parquet", columns=["quote_volume", "taker_buy_quote"])
            taker[a.asset] = df[~df.index.duplicated()]
        tb = taker[a.asset]
        w = tb[(tb.index > idx[t] - pd.Timedelta(minutes=15)) & (tb.index <= idx[t])]
        qv = float(w["quote_volume"].sum())
        rec["taker_buy_share_15m"] = d * (float(w["taker_buy_quote"].sum()) / qv - 0.5) if qv > 0 else np.nan
        # ---- extension / detection latency
        c = np.asarray(C[i, max(0, t - 1439): t + 1], np.float64)
        if np.isfinite(c).sum() > 60:
            j = int(np.nanargmin(c) if d > 0 else np.nanargmax(c))
            rec["extension"] = d * np.log(c[-1] / c[j])
            rec["minutes_running"] = len(c) - 1 - j
        else:
            rec["extension"] = rec["minutes_running"] = np.nan
        # ---- signal strength, liquidity
        rec["trigger_h"] = H_ORD.get(a.horizon, np.nan)
        rec["z"], rec["vm"], rec["log_liq"] = abs(float(a.z)), float(a.vm), float(np.log10(max(a.liq, 1.0)))
        # ---- market breadth over the trigger horizon
        hm = H_MIN.get(a.horizon, 60)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_all = np.log(np.asarray(C[:, t], np.float64) / np.asarray(C[:, t - hm], np.float64))
        lm = (LIQ[:, t // 15] >= 5e6) & np.isfinite(r_all)
        lm[i] = False
        rec["mkt_breadth"] = float((d * r_all[lm] >= 0.02).mean()) if lm.sum() >= 10 else np.nan
        # ---- venue breadth (static) + repeat movers
        sp = split_concat_symbol(a.asset)
        base = split_multiplier(sp[0])[0].upper() if sp else a.asset
        rec["venue_count"] = sum(1 for v in venues.values() if base in v)
        rec["prior_signals_7d"] = int(((sig["asset"] == a.asset) & (sig["time"] < ta)
                                       & (sig["time"] >= ta - pd.Timedelta(days=7))).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- statistics
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 10:
        return np.nan
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def ic_with_ci(df: pd.DataFrame, feat: str, out: str = "x240", draws: int = 1000) -> dict:
    x = df[["day", feat, out]].dropna()
    ic = spearman(x[feat].to_numpy(), x[out].to_numpy())
    days = x["day"].unique()
    res = {"ic": ic, "n": int(len(x)), "days": int(len(days)), "ci": None}
    if len(days) < 5 or not np.isfinite(ic):
        return res
    groups = [g.index.to_numpy() for _, g in x.groupby("day")]
    fx, oy = x[feat].to_numpy(), x[out].to_numpy()
    pos = {k: np.flatnonzero(x.index.isin(g)) for k, g in enumerate(groups)}
    bs = []
    for _ in range(draws):
        pick = RNG.integers(0, len(groups), len(groups))
        ii = np.concatenate([pos[k] for k in pick])
        bs.append(spearman(fx[ii], oy[ii]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    res["ci"] = [float(lo), float(hi)]
    return res


def max_null(df: pd.DataFrame, out: str = "x240", perms: int = 500) -> float:
    """95th percentile of max |IC| over all features when the outcome is shuffled within day."""
    base = df[["day", out] + FEATURES].copy()
    mx = []
    for _ in range(perms):
        sh = base.copy()
        sh[out] = sh.groupby("day")[out].transform(lambda s: s.sample(frac=1.0, random_state=int(RNG.integers(1e9))).to_numpy())
        ics = []
        for f in FEATURES:
            x = sh[[f, out]].dropna()
            ics.append(abs(spearman(x[f].to_numpy(), x[out].to_numpy())))
        mx.append(np.nanmax(ics))
    return float(np.percentile(mx, 95))


def terciles(df: pd.DataFrame, feat: str, out: str = "x240") -> list | None:
    from research.common import dir_simple
    x = df[[feat, out, "dir"]].dropna()
    if len(x) < 30 or x[feat].nunique() < 3:
        return None
    q = pd.qcut(x[feat], 3, labels=False, duplicates="drop")       # value-based: ties stay together
    x = x.assign(simple=dir_simple(x[out].to_numpy(), x["dir"].to_numpy()))
    return [round(float(v) * 100, 2) for v in x.groupby(q, observed=True)["simple"].mean().to_numpy()]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts-tag", default="study120", help="replay tag whose NEW alerts form the population")
    ap.add_argument("--out-tag", default="signal_features")
    ap.add_argument("--null", action="store_true", default=None, help="compute the permutation null (default: study120 only)")
    args = ap.parse_args()
    out_dir = OUT_DIR / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    syms, idx, C, V = load_panel("2026-08-01", "2026-09-28")
    LIQ = np.load(OUT_DIR / "blocks_2026-08-01_2026-09-28_r15.npz")["LIQ"]
    al = pd.read_parquet(OUT_DIR / args.alerts_tag / "alerts.parquet")
    sig = al[al["kind"] == "NEW"].copy()
    n_tradfi = int(sig["asset"].map(is_tradfi).sum())
    sig = sig[~sig["asset"].map(is_tradfi)].reset_index(drop=True)
    stale = Stale(panel_meta(PANEL_DIR, OUT_DIR / "panel_2026-08-01_2026-09-28", syms, idx), len(syms))
    print(f"tradfi excluded {n_tradfi}; stale runs {len(stale.i)}", flush=True)
    strict = pd.read_parquet(OUT_DIR / "cal_v4" / "alerts.parquet")
    strict_keys = set(zip(strict.loc[strict["kind"] == "NEW", "asset"], strict.loc[strict["kind"] == "NEW", "t"]))
    mem("loaded")
    pairs = set()
    for a in sig.itertuples():
        for dd in (a.time, a.time - pd.Timedelta(hours=5)):
            pairs.add((a.asset, dd.strftime("%Y-%m-%d")))
    t0 = time.time()
    metrics, premium = fetch_all(pairs)
    print(f"vision: {len(pairs)} symbol-days, metrics missing {sum(v is None for v in metrics.values())}, "
          f"premium missing {sum(v is None for v in premium.values())}, {time.time()-t0:.0f}s", flush=True)
    mem("vision")
    venues = venue_bases()
    print("venue list sizes:", {k: len(v) for k, v in venues.items()}, flush=True)
    feats = build(sig, syms, idx, C, LIQ, metrics, premium, venues, stale)
    feats["strict"] = [(a, int(t)) in strict_keys for a, t in zip(sig["asset"], sig["t"])]
    feats.to_parquet(out_dir / "features.parquet")
    mem("features")
    res = {"exclusions": {f: int(feats[f].isna().sum()) for f in FEATURES + ["x60", "x240", "x1440"]},
           "n": int(len(feats)), "tradfi_excluded": n_tradfi, "version": "v2-post-review"}
    for wname, (a, b) in WINDOWS.items():
        w = feats[(feats["time"] >= pd.Timestamp(a, tz="UTC")) & (feats["time"] < pd.Timestamp(b, tz="UTC"))]
        for pop, sel in (("all", w), ("pumps", w[w["dir"] > 0]), ("strict", w[w["strict"]])):
            key = f"{wname}|{pop}"
            res[key] = {"n": int(len(sel)), "features": {}}
            for f in FEATURES:
                res[key]["features"][f] = {"x240": ic_with_ci(sel, f), "x60": ic_with_ci(sel, f, "x60"),
                                           "x1440": ic_with_ci(sel, f, "x1440"), "terciles_x240": terciles(sel, f)}
            do_null = args.null if args.null is not None else args.alerts_tag == "study120"
            if wname == "aug" and pop in ("all", "pumps") and do_null:
                res[key]["null_max_abs_ic_95"] = max_null(sel)
            print(f"{key}: n={len(sel)}", flush=True)
    mem("stats")
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
