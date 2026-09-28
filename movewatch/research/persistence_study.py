"""What separates moves that LAST from moves that FADE? Protocol: docs/PERSISTENCE_STUDY.md
(declared before the holdout was downloaded). Discovery = Aug 2 - Sep 24 2026 (seen);
holdout = Apr 2 - Jul 28 2026 (consumed by the first run on 2026-09-28).

v2 (post-review 2026-09-28): fixes applied — Vision metrics point-in-time (+5 min span),
frozen post-delisting bars excluded with a counted status (outcomes, market median, random
baseline), every lookback day loaded, windows before listing / panel start -> NaN with status,
spot existence point-in-time + ticker aliases, stock/commodity perps excluded, value-based
tercile groups, 2,000-draw bootstrap / 2,000-permutation nulls, arithmetic means and medians
of SIMPLE returns (never exp(mean log) labelled as a mean), x24h analysed, declared combined
score with its top-minus-bottom CI, every robustness number computed here.
Multiple-testing bars: static features -> the declared 16-feature null; the 2 confirmation
features, which the protocol left unassigned, -> the stricter joint 18-feature null.

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.persistence_study
"""
from __future__ import annotations

import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "ARROW_DEFAULT_MEMORY_POOL": "system"}.items():
    os.environ.setdefault(_k, _v)

import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

import research.replay as rp  # noqa: E402
from movewatch.instruments import split_concat_symbol, split_multiplier  # noqa: E402
from movewatch.signals import baselines_from_minutes  # noqa: E402
from research.common import (Stale, arith_pct, arith_pct_dir, days_between, is_tradfi, median_pct,  # noqa: E402
                             median_pct_dir, metrics_row, mkt_median, panel_meta)
from research.signal_features import _vision, spearman  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
OUT = rp.OUT_DIR / "persistence"
S = requests.Session()
RNG = np.random.default_rng(20260929)
DRAWS, PERMS = 2000, 2000
H_ORD = {"1m": 1, "5m": 2, "15m": 3, "1h": 4, "4h": 5, "24h": 6}
H_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "24h": 1440}
STATIC = ["age_days", "calm_7d", "runup_72h", "speed", "liq_pre", "vol_surge", "mkt_breadth", "oi_chg_24h",
          "acct_ls", "top_ls_chg_24h", "premium", "premium_24h", "has_spot", "spot_share_4h",
          "prior_signals_30d", "trigger_h"]
CONFIRM = ["hold_4h", "vol_persist"]
SPOT_ALIAS = {"LUNA2USDT": "LUNAUSDT"}
WINDOWS = {
    "discovery": dict(panel=("2026-08-01", "2026-09-28"), panel_dir=HIST / "usdm_1m", alerts="study120",
                      sig=("2026-08-02", "2026-09-25")),
    "holdout": dict(panel=("2026-04-01", "2026-08-01"), panel_dir=HIST / "usdm_1m_holdout", alerts="holdout120",
                    sig=("2026-04-02", "2026-07-29")),
}


# ----------------------------------------------------------------------------- external data
def listing_dates(syms: list[str]) -> dict[str, str]:
    path = HIST / "listing_dates.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    todo = [s for s in syms if s not in have]

    def one(sym):
        for k in range(3):
            try:
                r = S.get("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
                          params={"prefix": f"data/futures/um/daily/klines/{sym}/1m/", "max-keys": 1}, timeout=20)
                m = re.findall(r"-1m-(\d{4}-\d{2}-\d{2})\.zip<", r.text)
                return sym, (m[0] if m else None)
            except requests.RequestException:
                time.sleep(1 + k)
        return sym, None
    with ThreadPoolExecutor(8) as ex:
        for sym, d in ex.map(one, todo):
            have[sym] = d
    path.write_text(json.dumps(have))
    return have


def spot_candidates(perp: str) -> list[str]:
    """Spot tickers to try, in order: the perp ticker itself (Binance spot keeps e.g.
    1000SATS / 1000CAT / 1MBABYDOGE), the multiplier-stripped ticker, known aliases."""
    out = [perp]
    sp = split_concat_symbol(perp)
    if sp:
        base, mult = split_multiplier(sp[0])
        if mult != 1.0:
            out.append(base + sp[1])
    if perp in SPOT_ALIAS:
        out.append(SPOT_ALIAS[perp])
    return out


def _spot(sym: str, day: str) -> pd.DataFrame | None:
    path = HIST / "vision_spot" / sym / f"{day}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    url = f"https://data.binance.vision/data/spot/daily/klines/{sym}/1m/{sym}-1m-{day}.zip"
    for k in range(4):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404:
                df = pd.DataFrame({"qv": []}, index=pd.DatetimeIndex([], tz="UTC", name="ts"))
            else:
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    raw = z.read(z.namelist()[0]).decode()
                rows = [ln.split(",") for ln in raw.splitlines() if ln and ln[0].isdigit()]
                ot = np.array([int(x[0]) for x in rows], dtype=np.int64)
                if len(ot) and ot[0] > 10 ** 14:          # Vision spot uses microseconds since 2025
                    ot = ot // 1000
                df = pd.DataFrame({"qv": [float(x[7]) for x in rows]},
                                  index=pd.DatetimeIndex(pd.to_datetime(ot, unit="ms", utc=True), name="ts"))
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            return df
        except (requests.RequestException, zipfile.BadZipFile):
            time.sleep(1.5 * (k + 1))
    return None


def fetch(pairs: set, fn) -> dict:
    keys = sorted(pairs)
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(keys, ex.map(lambda k: fn(*k), keys)))


def _cat(store: dict, sym: str, days: list[str]):
    parts = [store.get((sym, d)) for d in days]
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return None
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated()]


# ----------------------------------------------------------------------------- population
def dedupe_72h(sig: pd.DataFrame) -> pd.DataFrame:
    keep, last = [], {}
    for r in sig.sort_values("time").itertuples():
        lt = last.get(r.asset)
        if lt is None or r.time - lt >= pd.Timedelta(hours=72):
            keep.append(r.Index)
            last[r.asset] = r.time
    return sig.loc[keep]


def cluster_weeks(times: pd.Series) -> pd.Series:
    """ISO weeks; a week with < 3 distinct signal days is merged into its neighbour."""
    wk = times.dt.tz_convert(None).dt.to_period("W").dt.start_time
    days = times.dt.tz_convert(None).dt.normalize()
    ndays = days.groupby(wk).nunique()
    order = sorted(ndays.index)
    remap = {}
    for k, w in enumerate(order):
        if ndays[w] < 3:
            remap[w] = order[k + 1] if k + 1 < len(order) else order[k - 1]
    return wk.map(lambda w: remap.get(w, w))


# ----------------------------------------------------------------------------- features
def build(window: str, status: dict) -> pd.DataFrame:
    w = WINDOWS[window]
    rp.PANEL_DIR = w["panel_dir"]
    syms, idx, C, V = rp.load_panel(*w["panel"])
    pdir = rp.OUT_DIR / f"panel_{w['panel'][0]}_{w['panel'][1]}"
    stale = Stale(panel_meta(w["panel_dir"], pdir, syms, idx), len(syms))
    LIQ = np.load(rp.OUT_DIR / f"blocks_{w['panel'][0]}_{w['panel'][1]}_r15.npz")["LIQ"]
    T = C.shape[1]
    t_panel0 = idx[0]
    al = pd.read_parquet(rp.OUT_DIR / w["alerts"] / "alerts.parquet")
    sig = al[(al["kind"] == "NEW") & (al["time"] >= pd.Timestamp(w["sig"][0], tz="UTC"))
             & (al["time"] < pd.Timestamp(w["sig"][1], tz="UTC"))].copy()
    st = status.setdefault(window, {})
    st["new_signals"] = int(len(sig))
    tradfi = sig["asset"].map(is_tradfi)
    st["TRADFI_EXCLUDED"] = int(tradfi.sum())
    sig = sig[~tradfi]
    sig = dedupe_72h(sig).reset_index(drop=True)
    st["after_72h_dedup"] = int(len(sig))
    print(f"[{window}] NEW {st['new_signals']} -> tradfi excluded {st['TRADFI_EXCLUDED']} -> "
          f"{len(sig)} after 72h per-coin de-dup; stale runs in panel: {len(stale.i)}", flush=True)
    rp.mem(f"{window} loaded")
    lst = listing_dates(sorted(set(sig["asset"])))
    mp = {(r.asset, d) for r in sig.itertuples() for d in days_between(r.time, 25)}
    spp = {(c, d) for r in sig.itertuples() for c in spot_candidates(r.asset) for d in days_between(r.time, 5)}
    t0 = time.time()
    metrics = {k[1:]: v for k, v in fetch({("metrics",) + k for k in mp}, _vision).items()}
    premium = {k[1:]: v for k, v in fetch({("premium",) + k for k in mp}, _vision).items()}
    spot = fetch(spp, _spot)
    print(f"[{window}] downloads metrics {len(metrics)} (missing {sum(v is None for v in metrics.values())}), "
          f"premium {len(premium)} (missing {sum(v is None for v in premium.values())}), spot pairs {len(spot)} "
          f"(failed {sum(v is None for v in spot.values())}) {time.time()-t0:.0f}s", flush=True)
    rp.mem(f"{window} downloads")
    kept_times: dict[str, list] = {}
    rows = []
    for a in sig.itertuples():
        i = syms.index(a.asset)
        t = int(a.t)
        d = int(a.direction)
        ta = a.time
        c = lambda j: float(C[i, j]) if 0 <= j < T else np.nan  # noqa: E731
        rec = {"time": ta, "asset": a.asset, "dir": d, "horizon": a.horizon, "window": window}

        def ok_win(j0, j1):
            return (0 <= j0 and j1 < T and np.isfinite(c(j0)) and np.isfinite(c(j1)) and not stale.coin(i, j0, j1))

        def mkt_adj(j0, j1):
            if not ok_win(j0, j1):
                return np.nan
            mk = mkt_median(C, LIQ, j0, j1, stale, exclude=i)
            return d * (math.log(c(j1) / c(j0)) - (mk if np.isfinite(mk) else 0.0))
        # ---- outcomes (status STALE_OR_UNAVAILABLE when the coin's window has frozen bars)
        e, x_end, e2 = t + 1, t + 1 + 4320, t + 241
        rec["x72h"] = mkt_adj(e, x_end)
        rec["x24h"] = mkt_adj(e, e + 1440)
        rec["x4_72"] = mkt_adj(e2, x_end)
        rec["raw72h"] = d * math.log(c(x_end) / c(e)) if ok_win(e, x_end) else np.nan
        rec["raw4_72"] = d * math.log(c(x_end) / c(e2)) if ok_win(e2, x_end) else np.nan
        rec["held72"] = float(rec["raw72h"] >= 0) if np.isfinite(rec["raw72h"]) else np.nan
        # ---- static features (NaN when the window starts before listing / panel start)
        L = lst.get(a.asset)
        rec["age_days"] = math.log10(max((ta.tz_convert(None).normalize() - pd.Timestamp(L)).days, 1)) if L else np.nan
        a0, a1 = t - 8 * 1440, t - 1440
        if ok_win(a0, a1):
            s7, _, _ = baselines_from_minutes(np.asarray(C[i:i + 1, a0:a1], np.float64),
                                              np.asarray(V[i:i + 1, a0:a1], np.float64), min_hist=1440)
            rec["calm_7d"] = math.log10(s7[0]) if np.isfinite(s7[0]) else np.nan
        else:
            rec["calm_7d"] = np.nan
        lg = lambda j0, j1: d * math.log(c(j1) / c(j0)) if ok_win(j0, j1) else np.nan  # noqa: E731
        rec["runup_72h"] = lg(t - 4320, t)
        m1, m24 = lg(t - 60, t), lg(t - 1440, t)
        rec["speed"] = float(np.clip(m1 / max(m24, 0.01), -1, 2)) if np.isfinite(m1) and np.isfinite(m24) else np.nan
        vsum = lambda j0, j1: float(np.asarray(V[i, j0:j1], np.float64).sum())  # noqa: E731
        if ok_win(t - 2880, t - 1441):
            pre = vsum(t - 2880, t - 1440)
            rec["liq_pre"] = math.log10(pre) if pre > 0 else np.nan
        else:
            rec["liq_pre"] = np.nan
        if ok_win(t - 8 * 1440, t):
            wk = vsum(t - 8 * 1440, t - 1440) / 7.0
            now = vsum(t - 1439, t + 1)
            rec["vol_surge"] = math.log10(now / wk) if wk > 0 and now > 0 else np.nan
        else:
            rec["vol_surge"] = np.nan
        hm = H_MIN.get(a.horizon, 60)
        if t - hm >= 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r_all = np.log(np.asarray(C[:, t], np.float64) / np.asarray(C[:, t - hm], np.float64))
            lm = (LIQ[:, t // 15] >= 5e6) & np.isfinite(r_all) & ~stale.mask(t - hm, t)
            lm[i] = False
            rec["mkt_breadth"] = float((d * r_all[lm] >= 0.02).mean()) if lm.sum() >= 10 else np.nan
        else:
            rec["mkt_breadth"] = np.nan
        m = _cat(metrics, a.asset, days_between(ta, 25))
        oi0 = metrics_row(m, ta, "sum_open_interest")
        oi24 = metrics_row(m, ta - pd.Timedelta(hours=24), "sum_open_interest")
        rec["oi_chg_24h"] = oi0 / oi24 - 1 if oi24 > 0 and np.isfinite(oi0) else np.nan
        al0 = metrics_row(m, ta, "count_long_short_ratio")
        rec["acct_ls"] = d * math.log(al0) if al0 > 0 else np.nan
        tl0 = metrics_row(m, ta, "sum_toptrader_long_short_ratio")
        tl24 = metrics_row(m, ta - pd.Timedelta(hours=24), "sum_toptrader_long_short_ratio")
        rec["top_ls_chg_24h"] = d * math.log(tl0 / tl24) if tl0 > 0 and tl24 > 0 else np.nan
        p = _cat(premium, a.asset, days_between(ta, 25))
        if p is not None:
            p = p[p.index <= ta - pd.Timedelta(minutes=1)]                 # klines closed by the alert
            rec["premium"] = d * float(p["premium"].iloc[-1]) if len(p) else np.nan
            p24 = p[p.index > ta - pd.Timedelta(minutes=1441)]
            rec["premium_24h"] = d * float(p24["premium"].mean()) if len(p24) >= 60 else np.nan
        else:
            rec["premium"] = rec["premium_24h"] = np.nan
        # spot existence point-in-time: spot bars that closed before the alert
        rec["has_spot"], rec["spot_share_4h"], rec["spot_symbol"] = np.nan, np.nan, None
        sdays = days_between(ta, 5)
        fetched_all = True
        for cand in spot_candidates(a.asset):
            parts = [spot.get((cand, x)) for x in sdays]
            if any(v is None for v in parts):
                fetched_all = False
                continue
            sp = pd.concat([v for v in parts if len(v)]) if any(len(v) for v in parts) else None
            if sp is not None:
                sp = sp[sp.index <= ta - pd.Timedelta(minutes=1)]
            if sp is not None and len(sp):
                rec["has_spot"], rec["spot_symbol"] = 1.0, cand
                sv = float(sp[sp.index > ta - pd.Timedelta(hours=4, minutes=1)]["qv"].sum())
                pv = vsum(t - 239, t + 1)
                rec["spot_share_4h"] = sv / (sv + pv) if sv + pv > 0 else np.nan
                break
        if rec["has_spot"] != rec["has_spot"] and fetched_all:        # no candidate had pre-alert spot bars
            rec["has_spot"], rec["spot_share_4h"] = 0.0, 0.0
        prev = kept_times.setdefault(a.asset, [])
        rec["prior_signals_30d"] = (sum(1 for x in prev if ta - x <= pd.Timedelta(days=30))
                                    if ta - t_panel0 >= pd.Timedelta(days=30) else np.nan)
        prev.append(ta)
        rec["trigger_h"] = H_ORD.get(a.horizon, np.nan)
        # ---- confirmation features (4 h after the alert)
        rec["hold_4h"] = lg(t, t + 240)
        if ok_win(t - 179, t + 240):
            after, before = vsum(t + 61, t + 241), vsum(t - 179, t + 1)
            rec["vol_persist"] = math.log10(after / before) if after > 0 and before > 0 else np.nan
        else:
            rec["vol_persist"] = np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["week"] = cluster_weeks(df["time"])
    st["STALE_OR_UNAVAILABLE_x72h"] = int(df["x72h"].isna().sum())
    df.to_parquet(OUT / f"features_{window}.parquet")
    # random baseline (liquid coin-minutes in the signal window, frozen windows excluded)
    s0 = idx.get_indexer([pd.Timestamp(w["sig"][0], tz="UTC")])[0]
    s1 = idx.get_indexer([pd.Timestamp(w["sig"][1], tz="UTC")])[0]
    held, adj, tries = [], [], 0
    while len(held) < 5000 and tries < 300000:
        tries += 1
        i = int(RNG.integers(0, C.shape[0]))
        if is_tradfi(syms[i]):
            continue
        t = int(RNG.integers(s0, min(s1, T - 4322)))
        e, x_end = t + 1, t + 1 + 4320
        if LIQ[i, t // 15] < 5e6 or not (np.isfinite(C[i, e]) and np.isfinite(C[i, x_end])) or stale.coin(i, e, x_end):
            continue
        r = math.log(C[i, x_end] / C[i, e])
        mk = mkt_median(C, LIQ, e, x_end, stale, exclude=i)
        held.append(r >= 0)
        adj.append(r - (mk if np.isfinite(mk) else 0.0))
    st["random_baseline"] = {"n": len(held), "raw_held72_share": round(float(np.mean(held)), 3),
                             "mktadj72_arith_mean_pct": round(arith_pct(adj), 2),
                             "mktadj72_median_pct": round(median_pct(adj), 2)}
    rp.mem(f"{window} features")
    return df


# ----------------------------------------------------------------------------- statistics
def ic_ci(df: pd.DataFrame, f: str, out: str) -> dict:
    x = df[["week", f, out]].dropna()
    ic = spearman(x[f].to_numpy(), x[out].to_numpy())
    res = {"ic": ic, "n": int(len(x)), "clusters": int(x["week"].nunique()), "ci": None}
    if res["clusters"] < 5 or not np.isfinite(ic):
        return res
    wk = x["week"].to_numpy()
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    fx, oy = x[f].to_numpy(), x[out].to_numpy()
    bs = []
    for _ in range(DRAWS):
        ii = np.concatenate([pos[k] for k in RNG.integers(0, len(uw), len(uw))])
        bs.append(spearman(fx[ii], oy[ii]))
    res["ci"] = [float(v) for v in np.nanpercentile(bs, [2.5, 97.5])]
    return res


def fw_null(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> float:
    """95th pct of max |IC| over (feature, outcome) pairs when outcomes are shuffled within cluster."""
    outs = sorted({o for _, o in pairs})
    base = df[["week"] + outs + sorted({f for f, _ in pairs})].copy()
    grp = base.groupby("week").indices
    mx = []
    for _ in range(PERMS):
        sh = base.copy()
        for o in outs:
            vals = sh[o].to_numpy().copy()
            for ix in grp.values():
                vals[ix] = vals[ix][RNG.permutation(len(ix))]
            sh[o] = vals
        v = [abs(spearman(*sh[[f, o]].dropna().to_numpy().T)) for f, o in pairs]
        mx.append(np.nanmax(v))
    return float(np.percentile(mx, 95))


def groups(df: pd.DataFrame, f: str, out: str) -> dict | None:
    """Value-based terciles (ties stay together; fewer bins if the feature is lumpy):
    n, arithmetic mean and median of SIMPLE returns per bin, and top-minus-bottom arithmetic
    difference with a cluster-bootstrap CI."""
    x = df[["week", "dir", f, out]].dropna()
    if len(x) < 30 or x[f].nunique() < 2:
        return None
    b = pd.qcut(x[f], 3, labels=False, duplicates="drop") if x[f].nunique() > 2 else (x[f] > x[f].min()).astype(int)
    x = x.assign(bin=b)
    bins = sorted(x["bin"].unique())
    ap = lambda g: arith_pct_dir(g[out], g["dir"])  # noqa: E731
    res = {"bins": [{"n": int((x.bin == k).sum()), "arith_pct": round(ap(x[x.bin == k]), 2),
                     "median_pct": round(median_pct_dir(x[x.bin == k][out], x[x.bin == k]["dir"]), 2)} for k in bins]}
    lo, hi = bins[0], bins[-1]
    diff = lambda d: ap(d[d.bin == hi]) - ap(d[d.bin == lo])  # noqa: E731
    wk = x["week"].to_numpy()
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    bs = [diff(x.iloc[np.concatenate([pos[k] for k in RNG.integers(0, len(uw), len(uw))])]) for _ in range(DRAWS)]
    res["top_minus_bottom_arith_pct"] = round(diff(x), 2)
    res["ci"] = [round(float(v), 2) for v in np.nanpercentile(bs, [2.5, 97.5])] if len(uw) >= 5 else None
    return res


def partial_ic(df: pd.DataFrame, f: str, out: str, controls: list[str]) -> float:
    x = df[[f, out] + controls].dropna()
    if len(x) < 30:
        return float("nan")
    R = x.rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].to_numpy() for c in controls])
    res = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(res(R[f].to_numpy()), res(R[out].to_numpy()))[0, 1])


def robustness(df: pd.DataFrame, f: str, out: str) -> dict:
    x = df.dropna(subset=[f, out])
    wk = [spearman(g[f].to_numpy(), g[out].to_numpy()) for _, g in x.groupby("week") if len(g) >= 15]
    wk = [v for v in wk if v == v]
    within = {}
    for ctrl in ("hold_4h", "liq_pre", "trigger_h"):
        y = x.dropna(subset=[ctrl])
        if y[ctrl].nunique() < 3:
            continue
        b = pd.qcut(y[ctrl], 3, labels=False, duplicates="drop")
        within[ctrl] = [round(spearman(g[f].to_numpy(), g[out].to_numpy()), 3) for _, g in y.groupby(b)]
    raw = "raw4_72" if out == "x4_72" else "raw72h"
    return {"weeks_ge15_signals": len(wk), "weeks_same_sign_as_pooled": int(sum(np.sign(v) == np.sign(spearman(x[f].to_numpy(), x[out].to_numpy())) for v in wk)),
            "within_control_terciles": within,
            "partial_ic_given_hold4h_trigger": round(partial_ic(x, f, out, ["hold_4h", "trigger_h"]), 3) if f != "hold_4h" else None,
            "ic_on_raw_returns": round(spearman(*x[[f, raw]].dropna().to_numpy().T), 3)}


def describe(df: pd.DataFrame) -> dict:
    return {"n": int(df["x72h"].notna().sum()), "x72h_arith_mean_pct": round(arith_pct_dir(df["x72h"], df["dir"]), 2),
            "x72h_median_pct": round(median_pct_dir(df["x72h"], df["dir"]), 2),
            "raw_held72_share": round(float(df["held72"].mean()), 3) if df["held72"].notna().any() else None}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status: dict = {}
    frames = {w: build(w, status) for w in WINDOWS}
    res = {"protocol": "docs/PERSISTENCE_STUDY.md", "version": "v2-post-review", "status": status,
           "features": STATIC, "confirm": CONFIRM}
    for w, df in frames.items():
        res[f"exclusions_{w}"] = {f: int(df[f].isna().sum()) for f in STATIC + CONFIRM + ["x72h", "x24h", "x4_72"]}
        for pop, sel in (("pumps", df[df["dir"] > 0]), ("dumps", df[df["dir"] < 0])):
            res[f"describe_{w}_{pop}"] = describe(sel)
    pairs_static = [(f, "x72h") for f in STATIC]
    pairs_joint = pairs_static + [(f, "x4_72") for f in CONFIRM]
    for pop in ("pumps", "dumps"):
        dsc = frames["discovery"][frames["discovery"]["dir"] == (1 if pop == "pumps" else -1)]
        hld = frames["holdout"][frames["holdout"]["dir"] == (1 if pop == "pumps" else -1)]
        nulls = {}
        if pop == "pumps":
            nulls = {"discovery_static16": fw_null(dsc, pairs_static), "discovery_joint18": fw_null(dsc, pairs_joint),
                     "holdout_joint18": fw_null(hld, pairs_joint)}
            print("nulls", nulls, flush=True)
        table = {}
        for f in STATIC + CONFIRM:
            out = "x4_72" if f in CONFIRM else "x72h"
            dres, hres = ic_ci(dsc, f, out), ic_ci(hld, f, out)
            row = {"outcome": out, "discovery": dres, "holdout": hres,
                   "discovery_x24h": ic_ci(dsc, f, "x24h") if f in STATIC else None,
                   "holdout_x24h": ic_ci(hld, f, "x24h") if f in STATIC else None,
                   "groups_discovery": groups(dsc, f, out), "groups_holdout": groups(hld, f, out)}
            if pop == "pumps":
                bar = nulls["discovery_static16"] if f in STATIC else nulls["discovery_joint18"]
                d_ok = dres["ci"] is not None and (dres["ci"][0] > 0 or dres["ci"][1] < 0) and abs(dres["ic"]) >= bar
                same = np.isfinite(dres["ic"]) and np.isfinite(hres["ic"]) and dres["ic"] * hres["ic"] > 0
                h_ok = hres["ci"] is not None and (hres["ci"][0] > 0 or hres["ci"][1] < 0)
                h_fw = np.isfinite(hres["ic"]) and abs(hres["ic"]) >= nulls["holdout_joint18"]
                row["discovery_bar"] = bar
                row["verdict"] = ("VALIDATED" if d_ok and same and h_ok else "suggestive" if d_ok and same else
                                  "discovery only" if d_ok else
                                  "holdout-only (passes holdout joint null; discovery below its bar)" if (same and h_ok and h_fw) else "no")
                if h_fw or d_ok:
                    row["robustness_holdout"] = robustness(hld, f, out)
                    row["robustness_discovery"] = robustness(dsc, f, out)
            table[f] = row
        res[pop] = {"nulls": nulls, "table": table}
        if pop == "pumps":
            passing = [f for f in STATIC if table[f]["verdict"] in ("VALIDATED", "suggestive", "discovery only")]
            res["combined_features"] = passing
            if passing:
                def score(df):
                    z = pd.DataFrame({f: np.sign(table[f]["discovery"]["ic"]) * df[f].rank(pct=True) for f in passing})
                    return z.mean(axis=1, skipna=True)
                h = hld.assign(score=score(hld))
                res["combined_holdout"] = {"ic": ic_ci(h, "score", "x72h"), "groups": groups(h, "score", "x72h")}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    rp.mem("done")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
