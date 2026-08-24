"""Detail-review loop — iteration 2 (D2): was the delisting kill over-broad?

The untried-angles sweep killed the "short the delisting announcement" trade on a full-sample Sharpe of
+0.12, with pre-2025-08 at -0.62 and post at +3.38. But its OWN subsample numbers say pre-2025-08
EX-CRISIS is -0.21 — flat, not negative — and the case that did the damage (ANCUSDT, +148.9% in 80 minutes
then halted) had **0 days between announcement and settlement**: trading ceased INSIDE the 4-hour holding
window. That is not a strategy risk, it is an implementation error — you cannot hold a position in a contract
that stops trading.

Settlement date is STATED IN THE ANNOUNCEMENT TITLE, so requiring a minimum announcement-to-settlement lead
is an EX-ANTE, mechanism-based filter, not a fitted one. This rebuilds the study from scratch with it.

Honest prior, stated before running: the sweep's own ex-crisis pre-period is FLAT (-0.21), and 72% of the
post-2025-08 P&L is in Aug-Sep 2025 alone. So the most likely outcome is that the filter removes the
disasters without creating an edge — i.e. the kill stands but for a better-stated reason.

Sources, both free: announcements from the Binance CMS API (articles live under data.catalogs[0].articles —
the sweep's path was wrong); prices from data.binance.vision, which retains delisted symbols.
Run: python3 -u -m live.dr_iter2_delist
"""
from __future__ import annotations

import io
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "live/state/cost_loop"
ANN = CACHE / "delist_announcements.json"
API = ("https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
       "?type=1&catalogId=161&pageNo={p}&pageSize=50")
VISION = "https://data.binance.vision/data/futures/um/daily/klines/{s}/5m/{s}-5m-{d}.zip"
HDR = {"User-Agent": "Mozilla/5.0"}
ENTRY_MIN = 15          # short 15 minutes after the announcement
EXIT_MIN = 240          # cover 4 hours after
RT_BPS = 60.0           # round trip on a delisting-day contract, deliberately conservative
MIN_LEAD_DAYS = [0, 1, 2, 3]


def fetch_announcements() -> list:
    if ANN.exists():
        return json.loads(ANN.read_text())
    arts = []
    for p in range(1, 12):
        try:
            r = urllib.request.Request(API.format(p=p), headers=HDR)
            d = json.load(urllib.request.urlopen(r, timeout=45))
            cats = d.get("data", {}).get("catalogs", [])
            got = []
            for c in cats:
                got += c.get("articles", []) or []
            if not got:
                break
            arts += got
            time.sleep(1.0)
        except Exception as e:
            print(f"  page {p}: {str(e)[:60]}", flush=True)
            break
    seen, out = set(), []
    for a in arts:
        if a.get("id") in seen:
            continue
        seen.add(a["id"]); out.append(a)
    ANN.write_text(json.dumps(out))
    return out


SYMRE = re.compile(r"\b([A-Z0-9]{2,12})USDT\b")
DATERE = re.compile(r"on\s+(\d{4}-\d{2}-\d{2})")


def parse_futures(arts) -> pd.DataFrame:
    rows = []
    for a in arts:
        t = a.get("title", "")
        low = t.lower()
        if "futures" not in low or "delist" not in low and "remov" not in low:
            continue
        if "coin-m" in low or "coin margined" in low:
            continue
        syms = sorted(set(SYMRE.findall(t)))
        if not syms:
            continue
        rel = pd.Timestamp(a["releaseDate"], unit="ms", tz="UTC")
        m = DATERE.search(t)
        settle = pd.Timestamp(m.group(1), tz="UTC") if m else pd.NaT
        rows.append(dict(id=a["id"], title=t, release=rel,
                         settle=m.group(1) if m else None,
                         syms=[s + "USDT" for s in syms]))
    D = pd.DataFrame(rows)
    if D.empty:
        return D
    # build tz-aware in one pass — a NaT from the no-match branch makes the column tz-naive otherwise
    D["settle"] = pd.to_datetime(D["settle"], utc=True, errors="coerce")
    D["release"] = pd.to_datetime(D["release"], utc=True)
    return D


def klines(sym: str, day: str) -> pd.DataFrame | None:
    try:
        r = urllib.request.Request(VISION.format(s=sym, d=day), headers=HDR)
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(r, timeout=60).read()))
        with z.open(z.namelist()[0]) as fh:
            d = pd.read_csv(fh, header=None, usecols=[0, 4], names=["t", "close"])
        d = d[pd.to_numeric(d["t"], errors="coerce").notna()]
        unit = "us" if d["t"].astype("int64").max() > 2e15 else "ms"
        d["t"] = pd.to_datetime(d["t"].astype("int64"), unit=unit, utc=True)
        return d.astype({"close": float}).sort_values("t")
    except Exception:
        return None


def main():
    arts = fetch_announcements()
    F = parse_futures(arts)
    print(f"{len(arts)} delisting articles -> {len(F)} USDT-M futures events", flush=True)
    if F.empty:
        print("no events parsed"); return
    F["lead_days"] = (F["settle"] - F["release"]).dt.total_seconds() / 86400
    print(f"  announcement->settlement lead: median {F['lead_days'].median():.1f}d, "
          f"min {F['lead_days'].min():.1f}d, "
          f"{int((F['lead_days'] < 1).sum())} events settle in <1 day", flush=True)
    print(f"  date range {F['release'].min().date()} -> {F['release'].max().date()}", flush=True)

    rows = []
    for _, e in F.iterrows():
        day = e["release"].strftime("%Y-%m-%d")
        for s in e["syms"]:
            k = klines(s, day)
            if k is None or len(k) < 50:
                continue
            t0 = e["release"]
            pre = k[k["t"] <= t0]
            ent = k[k["t"] >= t0 + pd.Timedelta(minutes=ENTRY_MIN)]
            ex = k[k["t"] >= t0 + pd.Timedelta(minutes=EXIT_MIN)]
            if pre.empty or ent.empty:
                continue
            p_ent = float(ent["close"].iloc[0])
            # if the contract stops trading before the exit, cover at the LAST available print
            p_ex = float(ex["close"].iloc[0]) if not ex.empty else float(k["close"].iloc[-1])
            halted = ex.empty
            rows.append(dict(event=e["id"], symbol=s, release=t0, lead=e["lead_days"],
                             ret_short=-(p_ex / p_ent - 1) * 1e4 - RT_BPS, halted=halted))
        time.sleep(0.15)
    T = pd.DataFrame(rows)
    if T.empty:
        print("no priced events"); return
    T.to_parquet(CACHE / "delist_events.parquet", index=False)
    print(f"\npriced {len(T)} symbol-events across {T.event.nunique()} announcements "
          f"({int(T.halted.sum())} halted before the 4h exit)", flush=True)

    print("\n=== the ex-ante filter: require N days announcement->settlement ===", flush=True)
    print(f"  {'min lead':<10}{'events':>8}{'sym-obs':>9}{'mean bps':>10}{'t (batch-clustered)':>22}"
          f"{'worst batch':>13}", flush=True)
    for L in MIN_LEAD_DAYS:
        s = T[T["lead"].fillna(99) >= L]
        if s.empty:
            continue
        b = s.groupby("event")["ret_short"].mean()
        t = b.mean() / (b.std() / np.sqrt(len(b))) if b.std() > 0 else np.nan
        print(f"  >={L}d{'':<6}{len(b):>8}{len(s):>9}{b.mean():>10.0f}{t:>22.2f}{b.min():>13.0f}",
              flush=True)

    print("\n=== era split at the chosen filter (>=2 days) ===", flush=True)
    s = T[T["lead"].fillna(99) >= 2]
    b = s.groupby(["event"]).agg(r=("ret_short", "mean"), d=("release", "first"))
    for nm, t0, t1 in (("pre-2025-08", "2020-01-01", "2025-08-01"),
                       ("post-2025-08", "2025-08-01", "2027-01-01")):
        e = b[(b.d >= pd.Timestamp(t0, tz="UTC")) & (b.d < pd.Timestamp(t1, tz="UTC"))]
        if len(e) < 3:
            continue
        t = e["r"].mean() / (e["r"].std() / np.sqrt(len(e))) if e["r"].std() > 0 else np.nan
        print(f"  {nm:<14}{len(e):>4} batches  mean {e['r'].mean():>8.0f} bps  t {t:>6.2f}  "
              f"worst {e['r'].min():>8.0f}  best {e['r'].max():>8.0f}", flush=True)
    if len(b) > 5:
        top = b["r"].nlargest(3).sum() / b["r"].sum() if b["r"].sum() != 0 else np.nan
        print(f"\n  concentration: top-3 batches = {100*top:.0f}% of total P&L", flush=True)
    print("\nDRITER2DONE", flush=True)


if __name__ == "__main__":
    main()
