"""BTC-hedged fade after replay NEW alerts. Protocol: docs/BTC_HEDGE_STUDY.md (declared and
committed in 55ae98d before any result). Exploratory: both windows are consumed.

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.btc_hedge
"""
from __future__ import annotations

import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

import research.replay as rp  # noqa: E402
from movewatch.followup import beta_to  # noqa: E402
from research.common import Stale, is_tradfi, mkt_median, panel_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
FUND = HIST / "funding"
OUT = rp.OUT_DIR / "btc_hedge"
WINDOWS = {
    "holdout": dict(panel=("2026-04-01", "2026-08-01"), panel_dir=HIST / "usdm_1m_holdout",
                    features="features_holdout.parquet"),
    "discovery": dict(panel=("2026-08-01", "2026-09-28"), panel_dir=HIST / "usdm_1m",
                      features="features_discovery.parquet"),
}
HOR = {"0_4": (1, 241), "0_24": (1, 1441), "0_72": (1, 4321), "4_72": (241, 4561)}   # (entry, exit) minutes after t
FEE = 5e-4                       # taker, per side, each leg (Binance USDM VIP-0)
BTC_SLIP = 1e-4
TIERS = [(100e6, 5e-4), (20e6, 10e-4), (5e6, 20e-4), (0.0, 40e-4)]   # coin slippage per side by 24h volume
BETA_CLIP = (0.0, 3.0)
N_RANDOM, DRAWS = 2000, 2000
SEED = 20261002
BREADTH = (0.0051, 0.0468)       # live cutoffs: coin-specific <= 0.0051 < partly <= 0.0468 < market-wide
VOLP = (math.log10(0.94), math.log10(2.26))
S = requests.Session()


# ----------------------------------------------------------------------------- stocks
REF: dict[str, list[float]] = {}


def load_stock_ref() -> dict:
    """Venue-labelled stock/commodity prices (movewatch.tradfi.fetch_reference), snapshotted to
    OUT/tradfi_ref.json so the run is reproducible."""
    import asyncio

    import aiohttp

    from movewatch.tradfi import fetch_reference
    p = OUT / "tradfi_ref.json"
    if not p.exists():
        async def go():
            async with aiohttp.ClientSession() as s:
                return await fetch_reference(s)
        p.write_text(json.dumps(asyncio.run(go())))
    return json.loads(p.read_text())


def stock_like(sym: str, px: float) -> bool:
    """Static list (research.common) or a labelled stock base within 2x of its current price
    (the windows are months old; a same-ticker crypto sits orders of magnitude away)."""
    if is_tradfi(sym):
        return True
    base = sym[:-4] if sym.endswith(("USDT", "USDC")) else sym
    prices = REF.get(base)
    return bool(prices and px > 0 and min(abs(math.log(px / p)) for p in prices) <= math.log(2.0))


# ----------------------------------------------------------------------------- funding
def _fund_path(sym: str, month: str) -> Path:
    return FUND / f"{sym}_{month}.csv"


def _fetch_fund(key: tuple[str, str]) -> None:
    sym, month = key
    p = _fund_path(sym, month)
    if p.exists():
        return
    url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{month}.zip"
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 404:
                p.write_text("missing\n")
                return
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            p.write_bytes(z.open(z.namelist()[0]).read())
            return
        except Exception:  # noqa: BLE001 -- retried, then left absent (status NO_FUNDING)
            time.sleep(1 + attempt)


class Funding:
    """Sum of settled funding rates in (t0, t1] (ms) per symbol, from Vision monthly archives."""
    def __init__(self):
        self.cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    def months(self, t0_ms: float, t1_ms: float) -> list[str]:
        a = pd.Timestamp(t0_ms, unit="ms").to_period("M")          # naive UTC
        b = pd.Timestamp(t1_ms, unit="ms").to_period("M")
        return [str(p) for p in pd.period_range(a, b, freq="M")]

    def series(self, sym: str, months: list[str]):
        key = sym + "|" + ",".join(months)
        if key not in self.cache:
            parts = []
            for m in months:
                p = _fund_path(sym, m)
                if not p.exists() or p.read_text()[:7] == "missing":
                    self.cache[key] = None
                    return None
                parts.append(pd.read_csv(p))
            df = pd.concat(parts)
            self.cache[key] = (df["calc_time"].to_numpy(np.int64), df["last_funding_rate"].to_numpy(np.float64))
        return self.cache[key]

    def total(self, sym: str, t0_ms: float, t1_ms: float) -> float:
        s = self.series(sym, self.months(t0_ms, t1_ms))
        if s is None:
            return float("nan")
        ts, r = s
        sel = (ts > t0_ms) & (ts <= t1_ms)
        return float(r[sel].sum())


def fetch_funding(pairs: set[tuple[str, str]]) -> dict:
    FUND.mkdir(parents=True, exist_ok=True)
    todo = sorted(k for k in pairs if not _fund_path(*k).exists())
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(_fetch_fund, todo))
    missing = sum(1 for k in pairs if not _fund_path(*k).exists() or _fund_path(*k).read_text()[:7] == "missing")
    return {"pairs": len(pairs), "downloaded": len(todo), "missing": missing, "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- trades
def slip(liq: float) -> float:
    return next(s for lim, s in TIERS if liq >= lim)


def trade(c0, c1, b0, b1, side: int, beta: float, liq: float, f_coin: float, f_btc: float, slip_mult: float = 1.0) -> dict:
    """P&L per unit coin notional of the fade: coin leg `side` (+1 long / -1 short), BTC leg -side x beta."""
    rc, rb = c1 / c0 - 1.0, b1 / b0 - 1.0
    coin_cost = 2 * (FEE + slip_mult * slip(liq))
    btc_cost = 2 * (FEE + slip_mult * BTC_SLIP)
    fc = -side * f_coin                    # long pays positive funding; short receives it
    out = {"U_gross": side * rc, "U": side * rc - coin_cost + fc}
    for name, b in (("H1", 1.0), ("Hb", beta)):
        g = side * rc - side * b * rb
        out[f"{name}_gross"] = g
        out[name] = g - coin_cost - b * btc_cost + fc + side * b * f_btc
    return out


def events(window: str, w: dict, syms, idx, C, LIQ, stale, status: dict) -> list[dict]:
    feat = pd.read_parquet(rp.OUT_DIR / "persistence" / w["features"])
    T, ib = C.shape[1], syms.index("BTCUSDT")
    st = status.setdefault(window, {"events": int(len(feat))})
    rows = []
    for r in feat.itertuples():
        if r.asset == "BTCUSDT":
            st["SELF"] = st.get("SELF", 0) + 1
            continue
        i = syms.index(r.asset)
        t = int((r.time - idx[0]) // pd.Timedelta("1min")) - 1      # `time` stamps the close of trigger minute t
        if stock_like(r.asset, float(C[i, t])):
            st["TRADFI_PRICE"] = st.get("TRADFI_PRICE", 0) + 1
            continue
        rows.append({"window": window, "asset": r.asset, "i": i, "t": t, "dir": int(r.dir), "time": r.time,
                     "breadth": r.mkt_breadth, "vol_persist": r.vol_persist, "x72h_ref": r.x72h})
    return rows


def random_draws(window: str, syms, idx, C, LIQ, stale, rng) -> list[dict]:
    """Coin-minutes with >= $5M/day at the minute, stocks and BTC excluded (declared null)."""
    T, ib = C.shape[1], syms.index("BTCUSDT")
    ok_sym = np.array([s != "BTCUSDT" and not is_tradfi(s) for s in syms])
    rows, tries = [], 0
    while len(rows) < N_RANDOM and tries < N_RANDOM * 50:
        tries += 1
        t = int(rng.integers(1441, T - 4562))
        cand = np.flatnonzero(ok_sym & (LIQ[:, t // 15] >= 5e6) & np.isfinite(np.asarray(C[:, t + 1], np.float64)))
        if not cand.size:
            continue
        i = int(rng.choice(cand))
        if stock_like(syms[i], float(C[i, t])):
            continue
        rows.append({"window": window, "asset": syms[i], "i": i, "t": t, "dir": 1, "time": idx[t],
                     "breadth": np.nan, "vol_persist": np.nan, "x72h_ref": np.nan})
    return rows


def evaluate(rows: list[dict], syms, idx, C, LIQ, stale, fund: Funding, status: dict, tag: str) -> pd.DataFrame:
    T, ib = C.shape[1], syms.index("BTCUSDT")
    t_ms0 = idx[0].value // 10 ** 6
    out = []
    st = status.setdefault(tag, {})
    for e in rows:
        i, t, d = e["i"], e["t"], e["dir"]
        side = -d                                          # fade
        pos = np.arange(t - 1440, t - 60)
        if pos[0] < 0:
            beta = float("nan")
        else:
            beta = beta_to(np.asarray(C[i, pos], np.float64), np.asarray(C[ib, pos], np.float64))[0]
        liq = float(LIQ[i, t // 15])
        for h, (a, b) in HOR.items():
            j0, j1 = t + a, t + b
            rec = {**{k: e[k] for k in ("window", "asset", "dir", "time", "breadth", "vol_persist", "x72h_ref")}, "h": h,
                   "beta_raw": beta, "liq": liq}
            if j1 >= T or not (np.isfinite(C[i, j0]) and np.isfinite(C[i, j1])) or stale.coin(i, j0, j1):
                rec["status"] = "NO_PRICE_OR_STALE"
            elif not (np.isfinite(C[ib, j0]) and np.isfinite(C[ib, j1])):
                rec["status"] = "NO_BTC"
            elif not np.isfinite(beta):
                rec["status"] = "NO_BETA"
            else:
                m0, m1 = t_ms0 + j0 * 60_000 + 59_999, t_ms0 + j1 * 60_000 + 59_999     # minute closes
                fc, fb = fund.total(syms[i], m0, m1), fund.total("BTCUSDT", m0, m1)
                if not (np.isfinite(fc) and np.isfinite(fb)):
                    rec["status"] = "NO_FUNDING"
                else:
                    bc = float(np.clip(beta, *BETA_CLIP))
                    c0, c1, b0, b1 = (float(C[i, j0]), float(C[i, j1]), float(C[ib, j0]), float(C[ib, j1]))
                    rec.update(trade(c0, c1, b0, b1, side, bc, liq, fc, fb))
                    rec.update({f"{k}_x2": v for k, v in trade(c0, c1, b0, b1, side, bc, liq, fc, fb, 2.0).items()
                                if not k.endswith("_gross")})
                    mk = mkt_median(C, LIQ, j0, j1, stale, exclude=i)
                    rec["M_gross"] = side * (c1 / c0 - 1.0) - side * (math.exp(mk) - 1.0) if np.isfinite(mk) else np.nan
                    rec["mkt_log"] = d * (math.log(c1 / c0) - mk) if np.isfinite(mk) else np.nan
                    rec["beta"], rec["funding_coin"], rec["funding_btc"] = bc, fc, fb
                    rec["rc"], rec["rb"] = c1 / c0 - 1.0, b1 / b0 - 1.0
                    rec["status"] = "OK"
            st[f"{h}:{rec['status']}"] = st.get(f"{h}:{rec['status']}", 0) + 1
            out.append(rec)
    df = pd.DataFrame(out)
    df["week"] = pd.to_datetime(df["time"]).dt.tz_convert(None).dt.to_period("W").dt.start_time
    return df


# ----------------------------------------------------------------------------- statistics
def _draws(wk: np.ndarray, rng) -> list[np.ndarray]:
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    return [np.concatenate([pos[k] for k in rng.integers(0, len(uw), len(uw))]) for _ in range(DRAWS)]


def boot_means(df: pd.DataFrame, col: str, seed: int) -> tuple[float, np.ndarray, int, int]:
    x = df[["week", col]].dropna()
    v = x[col].to_numpy()
    if len(v) == 0:
        return float("nan"), np.array([]), 0, 0
    bs = np.array([v[ii].mean() for ii in _draws(x["week"].to_numpy(), np.random.default_rng(seed))])
    return float(v.mean()), bs, int(len(v)), int(x["week"].nunique())


def stats(df: pd.DataFrame, col: str, seed: int) -> dict:
    m, bs, n, g = boot_means(df, col, seed)
    if n == 0:
        return {"n": 0}
    v = df[col].dropna().to_numpy()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": n, "weeks": g, "mean_pct": 100 * m, "ci_pct": [100 * lo, 100 * hi], "median_pct": 100 * float(np.median(v)),
            "hit": float((v > 0).mean()), "std_pct": 100 * float(v.std()), "p5_pct": 100 * float(np.percentile(v, 5))}


def diff_ci(a: pd.DataFrame, b: pd.DataFrame, col: str, seed: int) -> dict:
    ma, ba, na, _ = boot_means(a, col, seed)
    mb, bb, nb, _ = boot_means(b, col, seed + 1)
    if not (na and nb):
        return {}
    lo, hi = np.percentile(ba - bb, [2.5, 97.5])
    return {"diff_pct": 100 * (ma - mb), "ci_pct": [100 * lo, 100 * hi]}


def summarize(ev: pd.DataFrame, rnd: pd.DataFrame) -> dict:
    res: dict = {}
    pops = {"pumps": lambda d: d[d.dir > 0], "dumps": lambda d: d[d.dir < 0],
            "pumps_coin_specific": lambda d: d[(d.dir > 0) & (d.breadth <= BREADTH[0])],
            "pumps_partly": lambda d: d[(d.dir > 0) & (d.breadth > BREADTH[0]) & (d.breadth <= BREADTH[1])],
            "pumps_market_wide": lambda d: d[(d.dir > 0) & (d.breadth > BREADTH[1])],
            "pumps_quiet_vol": lambda d: d[(d.dir > 0) & (d.vol_persist < VOLP[0])],
            "pumps_heavy_vol": lambda d: d[(d.dir > 0) & (d.vol_persist > VOLP[1])]}
    seed = SEED
    for w in WINDOWS:
        for h in HOR:
            base = ev[(ev.window == w) & (ev.h == h) & (ev.status == "OK")]
            r = rnd[(rnd.window == w) & (rnd.h == h) & (rnd.status == "OK")]
            for p, f in pops.items():
                if p.endswith("_vol") and h != "4_72":
                    continue
                d = f(base)
                key = f"{w}|{p}|{h}"
                seed += 10
                cell = {v: stats(d, v, seed) for v in ("U", "H1", "Hb", "M_gross", "U_gross", "Hb_gross", "Hb_x2")}
                if len(d) > 2:
                    cell["var_ratio_Hb_U_gross"] = float(d["Hb_gross"].var() / d["U_gross"].var())
                    cell["corr_coin_btc_hold"] = float(np.corrcoef(d["rc"], d["rb"])[0, 1])
                    cell["beta_median"] = float(d["beta"].median())
                    cell["funding_coin_mean_pct"] = 100 * float((-(-d["dir"]) * d["funding_coin"]).mean())
                if p == "pumps":
                    cell["random_Hb"] = stats(r, "Hb", seed + 5)
                    cell["random_U"] = stats(r, "U", seed + 6)
                    cell["Hb_minus_random"] = diff_ci(d, r, "Hb", seed + 7)
                res[key] = cell
    # candidate predicate (declared): Hb pumps, CI > 0 and Hb - random CI > 0, in BOTH windows
    cand = {}
    for h in HOR:
        ok = []
        for w in WINDOWS:
            c = res[f"{w}|pumps|{h}"]
            hb, dr = c["Hb"], c.get("Hb_minus_random", {})
            ok.append(bool(hb.get("n") and hb["ci_pct"][0] > 0 and dr and dr["ci_pct"][0] > 0))
        cand[h] = all(ok)
    res["candidate"] = cand
    return res


def reconcile(ev: pd.DataFrame) -> dict:
    """Selftest vs the generating study (rule 3): our 0_72 market-adjusted log return must equal
    persistence_study's x72h on the rows both computed."""
    x = ev[(ev.h == "0_72") & ev["mkt_log"].notna()][["mkt_log", "x72h_ref"]].dropna()
    bad = int((np.abs(x["mkt_log"] - x["x72h_ref"]) > 1e-6).sum())
    return {"rows": int(len(x)), "mismatch": bad}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REF.update(load_stock_ref())
    status: dict = {}
    rng = np.random.default_rng(SEED)
    evs, rnds = [], []
    fund = Funding()
    for w, cfg in WINDOWS.items():
        rp.PANEL_DIR = cfg["panel_dir"]
        syms, idx, C, V = rp.load_panel(*cfg["panel"])
        pdir = rp.OUT_DIR / f"panel_{cfg['panel'][0]}_{cfg['panel'][1]}"
        stale = Stale(panel_meta(cfg["panel_dir"], pdir, syms, idx), len(syms))
        LIQ = np.load(rp.OUT_DIR / f"blocks_{cfg['panel'][0]}_{cfg['panel'][1]}_r15.npz")["LIQ"]
        rp.mem(f"{w} loaded")
        er = events(w, cfg, syms, idx, C, LIQ, stale, status)
        rr = random_draws(w, syms, idx, C, LIQ, stale, rng)
        pairs = set()
        for e in er + rr:
            t0 = idx[0] + pd.Timedelta(minutes=e["t"])
            for m in fund.months(t0.value // 10 ** 6, (t0 + pd.Timedelta(minutes=4561)).value // 10 ** 6):
                pairs.add((e["asset"], m))
                pairs.add(("BTCUSDT", m))
        status[f"{w}_funding"] = fetch_funding(pairs)
        print(f"[{w}] events {len(er)} random {len(rr)} funding {status[f'{w}_funding']}", flush=True)
        evs.append(evaluate(er, syms, idx, C, LIQ, stale, fund, status, f"{w}_events"))
        rnds.append(evaluate(rr, syms, idx, C, LIQ, stale, fund, status, f"{w}_random"))
        rp.mem(f"{w} evaluated")
        del C, V, LIQ
    ev, rnd = pd.concat(evs, ignore_index=True), pd.concat(rnds, ignore_index=True)
    rec = reconcile(ev)
    print("reconciliation vs persistence x72h:", rec, flush=True)
    if rec["rows"] < 100 or rec["mismatch"] > 0.01 * rec["rows"]:
        raise SystemExit(f"reconciliation failed: {rec}")
    res = summarize(ev, rnd)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res["meta"] = {"commit": commit, "status": status, "reconciliation": rec, "costs": {
        "fee_per_side": FEE, "coin_slip_tiers": TIERS, "btc_slip": BTC_SLIP}, "beta_clip": BETA_CLIP}
    ev.drop(columns=["time"]).to_parquet(OUT / "events.parquet")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    for w in WINDOWS:
        print(f"\n== {w}")
        for h in HOR:
            c = res[f"{w}|pumps|{h}"]
            f = lambda s: f"{s['mean_pct']:+6.2f}% [{s['ci_pct'][0]:+.2f},{s['ci_pct'][1]:+.2f}]" if s.get("n") else "n/a"  # noqa: E731
            print(f"pumps {h:5s} n={c['Hb'].get('n', 0):4d}  U {f(c['U'])}  H1 {f(c['H1'])}  Hb {f(c['Hb'])}  "
                  f"M(gross) {f(c['M_gross'])}  random Hb {f(c['random_Hb'])}  var(Hb/U) {c.get('var_ratio_Hb_U_gross', float('nan')):.2f}"
                  f"  corr {c.get('corr_coin_btc_hold', float('nan')):.2f}")
    print("\ncandidate (declared predicate):", res["candidate"])
    rp.mem("done")


if __name__ == "__main__":
    main()
