"""Why do some pumps keep running? Protocol: docs/RUNNERS_STUDY.md (declared, commit 8faad7a).
Part A (structured listings + first-4h positioning) and the blind sample for Part B (news).

  MALLOC_ARENA_MAX=2 .worktrees/research_slot.sh 3 -- python3 -m research.runners            # Part A + sample
  python3 -m research.runners --join-news data/replay/runners/news_labels.json                 # Part B join
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats as ss

import research.replay as rp
from movewatch.instruments import split_multiplier
from research.common import metrics_row
from research.persistence_study import spot_candidates
from research.signal_features import fetch_all

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "hist"
OUT = rp.OUT_DIR / "runners"
RUN, FADE = 0.20, -0.10
SEED = 20261008
N_NEWS = 30
S = requests.Session()
S.headers["User-Agent"] = "movewatch-research/0.1"


# ----------------------------------------------------------------------------- population
def population() -> pd.DataFrame:
    ev = pd.read_parquet(rp.OUT_DIR / "btc_hedge" / "events.parquet")
    ev = ev[(ev.h == "0_72") & (ev.status == "OK") & (ev.dir > 0)][["window", "asset", "x72h_ref", "liq"]]
    feats = pd.concat([pd.read_parquet(rp.OUT_DIR / "persistence" / f"features_{w}.parquet")
                       for w in ("holdout", "discovery")], ignore_index=True)
    m = ev.merge(feats, on=["window", "asset"], how="left")
    m = m[np.isclose(m["x72h"], m["x72h_ref"], rtol=0, atol=1e-9)].drop_duplicates(["window", "asset", "time"])
    assert len(m) == len(ev), (len(m), len(ev))          # every hedge-study row found its feature row once
    m["y"] = np.exp(m["x72h"]) - 1.0
    m["group"] = np.where(m["y"] >= RUN, "RUNNER", np.where(m["y"] <= FADE, "FADER", "MIDDLE"))
    # post-alert (first 4 h) features are compared on the 4 h -> 72 h outcome: the 0 -> 72 h outcome
    # contains those 4 h, which makes the comparison partly tautological (CLAUDE.md rule 9)
    m["y4"] = np.exp(m["x4_72"]) - 1.0
    m["group4"] = np.where(m["y4"] >= RUN, "RUNNER", np.where(m["y4"] <= FADE, "FADER", "MIDDLE"))
    m.loc[m["x4_72"].isna(), "group4"] = "NO_OUTCOME"
    al = pd.concat([pd.read_parquet(rp.OUT_DIR / d / "alerts.parquet") for d in ("holdout120", "study120")])
    al = al[al.kind == "NEW"][["asset", "time", "price"]].drop_duplicates(["asset", "time"])
    m = m.merge(al, on=["asset", "time"], how="left")
    return m.reset_index(drop=True)


def base_of(sym: str) -> str:
    raw = sym[:-4] if sym.endswith(("USDT", "USDC")) else sym
    return split_multiplier(raw)[0].upper()


# ----------------------------------------------------------------------------- listings
def _get(url, params=None):
    for k in range(4):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (k + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(1 + k)
    return None


def binance_spot_first(perp: str) -> str | None:
    for sym in spot_candidates(perp):
        r = _get("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
                 {"prefix": f"data/spot/daily/klines/{sym}/1m/", "max-keys": 1})
        if r is not None and r.ok:
            m = re.findall(r"-1m-(\d{4}-\d{2}-\d{2})\.zip<", r.text)
            if m:
                return m[0]
    return None


def upbit_first(base: str) -> str | None:
    r = _get("https://api.upbit.com/v1/candles/days", {"market": f"KRW-{base}", "count": 200})
    time.sleep(0.12)
    if r is None or not r.ok:
        return None
    rows = r.json()
    if not isinstance(rows, list) or not rows:
        return None
    oldest = min(x["candle_date_time_utc"][:10] for x in rows)
    return "before-window" if len(rows) >= 200 else oldest


def bithumb_first(base: str) -> str | None:
    r = _get(f"https://api.bithumb.com/public/candlestick/{base}_KRW/24h")
    time.sleep(0.1)
    if r is None or not r.ok or r.json().get("status") != "0000":
        return None
    data = r.json().get("data") or []
    return pd.Timestamp(int(data[0][0]), unit="ms").strftime("%Y-%m-%d") if data else None


def coinbase_first(base: str) -> str | None:
    r = _get(f"https://api.exchange.coinbase.com/products/{base}-USD/candles",
             {"granularity": 86400, "start": "2026-01-05T00:00:00Z", "end": "2026-10-01T00:00:00Z"})
    time.sleep(0.15)
    if r is None or not r.ok:
        return None
    rows = r.json()
    if not isinstance(rows, list) or not rows:
        return None
    oldest = pd.Timestamp(min(x[0] for x in rows), unit="s")
    return "before-window" if oldest <= pd.Timestamp("2026-01-06") else oldest.strftime("%Y-%m-%d")


def listings(pop: pd.DataFrame) -> dict:
    path = HIST / "listing_venues.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    perps = sorted(set(pop["asset"]))
    bases = sorted({base_of(s) for s in perps})
    jobs = [("binance_spot", p, binance_spot_first) for p in perps] + \
           [(v, b, f) for b in bases for v, f in (("upbit", upbit_first), ("bithumb", bithumb_first),
                                                  ("coinbase", coinbase_first))]
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in have]
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        for (v, k, _), res in zip(todo, ex.map(lambda j: j[2](j[1]), todo)):
            have[f"{v}|{k}"] = res
    path.write_text(json.dumps(have, indent=0))
    print(f"listing lookups: {len(jobs)} ({len(todo)} new, {time.time() - t0:.0f}s)", flush=True)
    return have


def listing_flags(pop: pd.DataFrame, L: dict) -> pd.DataFrame:
    out = pop.copy()
    d0 = out["time"].dt.tz_convert(None).dt.normalize()
    for v in ("binance_spot", "upbit", "bithumb", "coinbase"):
        key = out["asset"] if v == "binance_spot" else out["asset"].map(base_of)
        raw = key.map(lambda k: L.get(f"{v}|{k}"))
        dt = pd.to_datetime(raw.where(raw.notna() & (raw != "before-window")), errors="coerce")
        out[f"lst_{v}_before"] = ((dt >= d0 - pd.Timedelta(days=7)) & (dt <= d0)).astype(float)
        out[f"lst_{v}_after"] = ((dt > d0) & (dt <= d0 + pd.Timedelta(days=3))).astype(float)
    out["lst_any_before"] = out[[c for c in out if c.startswith("lst_") and c.endswith("_before")]].max(axis=1)
    out["lst_any_after"] = out[[c for c in out if c.startswith("lst_") and c.endswith("_after")]].max(axis=1)
    out["new_perp_14d"] = (10 ** out["age_days"] <= 14).astype(float).where(out["age_days"].notna())   # age_days is log10
    return out


# ----------------------------------------------------------------------------- first 4 h positioning
def first4h(pop: pd.DataFrame) -> pd.DataFrame:
    days = {(r.asset, d.strftime("%Y-%m-%d")) for r in pop.itertuples()
            for d in pd.date_range(r.time.normalize(), (r.time + pd.Timedelta(hours=4)).normalize(), freq="D")}
    t0 = time.time()
    met, prem = fetch_all(days)
    print(f"vision days {len(days)}: metrics missing {sum(v is None for v in met.values())}, "
          f"premium missing {sum(v is None for v in prem.values())} ({time.time() - t0:.0f}s)", flush=True)
    rows = []
    for r in pop.itertuples():
        ds = [d.strftime("%Y-%m-%d") for d in pd.date_range(r.time.normalize(),
                                                           (r.time + pd.Timedelta(hours=4)).normalize(), freq="D")]
        mp = [met.get((r.asset, d)) for d in ds]
        pp = [prem.get((r.asset, d)) for d in ds]
        m = pd.concat([x for x in mp if x is not None and len(x)]) if any(x is not None and len(x) for x in mp) else None
        p = pd.concat([x for x in pp if x is not None and len(x)]) if any(x is not None and len(x) for x in pp) else None
        a, b = r.time, r.time + pd.Timedelta(hours=4)
        rec = {}
        if m is not None:
            m = m[~m.index.duplicated()].sort_index()
            oi0, oi1 = metrics_row(m, a, "sum_open_interest_value"), metrics_row(m, b, "sum_open_interest_value")
            rec["oi_chg_4h"] = math.log(oi1 / oi0) if oi0 > 0 and oi1 > 0 else np.nan
            t0_, t1_ = metrics_row(m, a, "sum_toptrader_long_short_ratio"), metrics_row(m, b, "sum_toptrader_long_short_ratio")
            rec["top_ls_chg_4h"] = math.log(t1_ / t0_) if t0_ > 0 and t1_ > 0 else np.nan
            span = m[(m.index + pd.Timedelta(minutes=5) > a) & (m.index + pd.Timedelta(minutes=5) <= b)]
            rec["taker_ls_4h"] = float(span["sum_taker_long_short_vol_ratio"].mean()) if len(span) else np.nan
        if p is not None:
            p = p[~p.index.duplicated()].sort_index()
            w = p[(p.index >= a + pd.Timedelta(hours=3)) & (p.index + pd.Timedelta(minutes=1) <= b)]
            rec["premium_h4"] = float(w["premium"].mean()) if len(w) else np.nan
        rows.append(rec)
    return pd.concat([pop, pd.DataFrame(rows, index=pop.index)], axis=1)


# ----------------------------------------------------------------------------- statistics
def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (c - h), 100 * (c + h)]


BINARY = ["lst_binance_spot_before", "lst_upbit_before", "lst_bithumb_before", "lst_coinbase_before", "lst_any_before",
          "lst_binance_spot_after", "lst_upbit_after", "lst_bithumb_after", "lst_coinbase_after", "lst_any_after",
          "new_perp_14d", "has_spot"]
POST4 = ["oi_chg_4h", "premium_h4", "taker_ls_4h", "top_ls_chg_4h", "hold_4h", "vol_persist"]
CONT = ["age_days", "runup_72h", "calm_7d", "oi_chg_24h", "acct_ls", "mkt_breadth", "premium"]


def compare(df: pd.DataFrame, feats: list[str], gcol: str = "group") -> dict:
    res = {}
    for f in feats:
        cell = {}
        for w in ("holdout", "discovery", "pooled"):
            d = df if w == "pooled" else df[df.window == w]
            r, fd = d[d[gcol] == "RUNNER"][f].dropna(), d[d[gcol] == "FADER"][f].dropna()
            c = {"n_runner": int(len(r)), "n_fader": int(len(fd)),
                 "excluded": int(d[d[gcol].isin(["RUNNER", "FADER"])][f].isna().sum())}
            if f in BINARY:
                kr, kf = int(r.sum()), int(fd.sum())
                c.update(runner_pct=100 * kr / max(len(r), 1), fader_pct=100 * kf / max(len(fd), 1),
                         runner_ci=wilson(kr, len(r)), fader_ci=wilson(kf, len(fd)),
                         diff_pp=100 * (kr / max(len(r), 1) - kf / max(len(fd), 1)),
                         p=float(ss.fisher_exact([[kr, len(r) - kr], [kf, len(fd) - kf]])[1]) if len(r) and len(fd) else float("nan"))
            else:
                c.update(runner_median=float(r.median()) if len(r) else float("nan"),
                         fader_median=float(fd.median()) if len(fd) else float("nan"),
                         p=float(ss.mannwhitneyu(r, fd).pvalue) if len(r) > 2 and len(fd) > 2 else float("nan"))
                c["diff"] = c["runner_median"] - c["fader_median"]
            cell[w] = c
        key = "diff_pp" if f in BINARY else "diff"
        signs = [np.sign(cell[w][key]) for w in ("holdout", "discovery")]
        cell["consistent"] = bool(signs[0] == signs[1] != 0 and cell["pooled"]["p"] < 0.01)
        res[f] = cell
    return res


def news_sample(pop: pd.DataFrame) -> None:
    rng = random.Random(SEED)
    run = pop[pop.group == "RUNNER"]
    n_h = round(N_NEWS * (run.window == "holdout").mean())
    pick = (run[run.window == "holdout"].sample(n_h, random_state=SEED).index.tolist()
            + run[run.window == "discovery"].sample(N_NEWS - n_h, random_state=SEED).index.tolist())
    used, pairs = set(), []
    for i in pick:
        r = pop.loc[i]
        f = pop[(pop.group == "FADER") & (pop.window == r.window) & (pop.asset != r.asset) & ~pop.index.isin(used)]
        j = (f["time"] - r.time).abs().idxmin()
        used.add(j)
        pairs += [i, j]
    rng.shuffle(pairs)
    blind, key = [], {}
    for k, i in enumerate(pairs):
        r = pop.loc[i]
        eid = f"E{k + 1:02d}"
        blind.append({"id": eid, "token": base_of(r.asset), "binance_perp": r.asset,
                      "alert_time_utc": r.time.strftime("%Y-%m-%d %H:%M"), "alert_price_usd": float(r.price)})
        key[eid] = {"group": r.group, "window": r.window, "asset": r.asset, "y_pct": round(100 * r.y, 1)}
    (OUT / "news_blind.json").write_text(json.dumps(blind, indent=1, ensure_ascii=False))
    (OUT / "news_key.json").write_text(json.dumps(key, indent=1))
    print(f"news sample: {len(blind)} events written blind ({OUT / 'news_blind.json'}); key kept separately", flush=True)


def join_news(path: Path) -> dict:
    labels = {}
    for f in sorted(path.parent.glob(path.name)):          # a glob: news_labels_part*.json
        labels.update(json.loads(f.read_text()))
    key = json.loads((OUT / "news_key.json").read_text())
    rows = []
    for eid, k in key.items():
        lab = labels.get(eid) or {}
        items = lab.get("items", [])
        real = [x for x in items if x.get("category") not in ("PRICE_ONLY", "NONE_FOUND")]
        pre = [x for x in real if x.get("timing") == "before"]
        same = [x for x in real if x.get("timing") == "same_day"]
        post = [x for x in real if x.get("timing") == "after"]
        rows.append({"id": eid, **k, "labelled": bool(lab), "pre_catalyst": bool(pre),
                     "pre_or_same_day": bool(pre or same), "post_catalyst": bool(post),
                     "any_catalyst": bool(real), "price_only_coverage": any(x.get("category") == "PRICE_ONLY" for x in items),
                     "pre_cats": sorted({x["category"] for x in pre + same}), "post_cats": sorted({x["category"] for x in post})})
    d = pd.DataFrame(rows)
    res = {"labelled": int(d.labelled.sum()), "events": int(len(d))}
    for col in ("pre_catalyst", "pre_or_same_day", "post_catalyst", "any_catalyst", "price_only_coverage"):
        r, f = d[(d.group == "RUNNER") & d.labelled][col], d[(d.group == "FADER") & d.labelled][col]
        kr, kf = int(r.sum()), int(f.sum())
        res[col] = {"runner": f"{kr}/{len(r)}", "fader": f"{kf}/{len(f)}", "runner_ci": wilson(kr, len(r)),
                    "fader_ci": wilson(kf, len(f)), "diff_pp": 100 * (kr / max(len(r), 1) - kf / max(len(f), 1)),
                    "p": float(ss.fisher_exact([[kr, len(r) - kr], [kf, len(f) - kf]])[1])}
    cats = {}
    for g in ("RUNNER", "FADER"):
        for col in ("pre_cats", "post_cats"):
            c = pd.Series([x for v in d[(d.group == g) & d.labelled][col] for x in v]).value_counts().to_dict()
            cats[f"{g}_{col}"] = c
    res["categories"] = cats
    res["events_table"] = d.to_dict("records")
    (OUT / "news_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--join-news", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.join_news:
        res = join_news(Path(args.join_news))
        print(json.dumps({k: v for k, v in res.items() if k != "events_table"}, indent=1, ensure_ascii=False))
        return
    pop = population()
    print(f"pumps {len(pop)}: " + ", ".join(f"{w} {dict(pop[pop.window == w].group.value_counts())}"
                                            for w in ("holdout", "discovery")), flush=True)
    L = listings(pop)
    pop = listing_flags(pop, L)
    pop = first4h(pop)
    rp.mem("features built")
    res = {"counts": {w: pop[pop.window == w].group.value_counts().to_dict() for w in ("holdout", "discovery")},
           "counts_4_72": {w: pop[pop.window == w].group4.value_counts().to_dict() for w in ("holdout", "discovery")},
           "compare": compare(pop, BINARY + CONT), "compare_4_72": compare(pop, POST4, "group4")}
    pop.drop(columns=["time"]).assign(time=pop["time"].astype(str)).to_parquet(OUT / "pumps.parquet")
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    news_sample(pop)
    print("counts 4->72:", res["counts_4_72"])
    for title, block in (("0->72 h outcome (pre-alert + listings)", res["compare"]),
                         ("4->72 h outcome (first-4h features)", res["compare_4_72"])):
        print(f"\n{title}\nfeature                     | holdout R vs F            | discovery R vs F          | pooled p   | consistent")
        for f, c in block.items():
            sc = 1e4 if f.startswith("premium") else 1.0              # premium in bp
            if f in BINARY:
                fmt = lambda x: f"{x['runner_pct']:5.1f}% vs {x['fader_pct']:5.1f}%"  # noqa: E731
            else:
                fmt = lambda x, sc=sc: f"{sc * x['runner_median']:+.3f} vs {sc * x['fader_median']:+.3f}"  # noqa: E731
            print(f"{f:27s} | {fmt(c['holdout']):25s} | {fmt(c['discovery']):25s} | {c['pooled']['p']:.2g}"
                  f"{'':6s} | {c['consistent']}")
    rp.mem("done")


if __name__ == "__main__":
    main()
