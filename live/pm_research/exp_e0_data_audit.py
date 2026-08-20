"""EXP-E0 — data integrity audit. Runs BEFORE any result is trusted.

This program has a known incident list: a 30 s duplicate-collector overlap, a
~16 min market-side outage, 8 malformed resolution rows written before the
finality bug was fixed, restart shards, and gaps in the TWAP stream. Every one
of those can move a calibration number, so they are quantified here rather than
assumed benign.

Checks
  1 resolution finality + duplicates
  2 TWAP stream: duplicates by (t_event, symbol), gaps, per-coin coverage
  3 knowledge-time lag distribution (recv_ns - payload ts) per coin
  4 per-window TWAP coverage over [t0-5s, T+5s] — the E-M6 admissibility rule
  5 market capture: shard sets, book sanity (crossed/locked/out-of-range)
  6 selection bias: is the resolved+covered subset representative?
  7 up-rate by coin and by day — the drift confound in the calibration result

Run: python3 -u -m live.pm_research.exp_e0_data_audit
"""
from __future__ import annotations

import glob, gzip, json, math, statistics as st
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
COINS = {"btc": "btc/usd", "eth": "eth/usd", "sol": "sol/usd", "xrp": "xrp/usd",
         "doge": "doge/usd", "bnb": "bnb/usd", "hype": "hype/usd"}


def hdr(t):
    print(f"\n{'='*66}\n{t}\n{'='*66}")


def main():
    # ---------- 1 resolutions ----------
    hdr("1. RESOLUTIONS — finality and duplicates")
    rows = [json.loads(l) for l in open(PM / "resolutions.jsonl")]
    final = [r for r in rows if r.get("closed") is True and r.get("winners")]
    nonfinal = [r for r in rows if not (r.get("closed") is True and r.get("winners"))]
    dup = Counter(r["slug"] for r in final)
    print(f"  rows={len(rows)}  final={len(final)}  non-final(garbage)={len(nonfinal)}")
    print(f"  distinct final slugs={len(dup)}  slugs with >1 final row="
          f"{sum(1 for v in dup.values() if v > 1)}")
    if nonfinal[:1]:
        print(f"  example non-final: {json.dumps(nonfinal[0])[:110]}")
    res = {}
    for r in final:
        res[r["slug"]] = bool(r["winners"].get("Up"))   # last write wins

    # ---------- 2 TWAP stream ----------
    hdr("2. TWAP STREAM — duplicates, gaps, coverage")
    ev = defaultdict(list)     # symbol -> (t_event, t_known)
    seen = defaultdict(set)
    dupe = Counter()
    for f in sorted(glob.glob(str(PM / "prices/crypto_prices_twap_sixty/*.csv*"))):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            for ln in fh:
                p = ln.split("\t", 1)
                if len(p) < 2:
                    continue
                try:
                    m = json.loads(p[1]); pl = m.get("payload") or {}
                    s, t = pl.get("symbol"), pl.get("timestamp")
                    if not s or not t:
                        continue
                    t = int(t)
                    if t in seen[s]:
                        dupe[s] += 1
                        continue
                    seen[s].add(t)
                    ev[s].append((t, int(p[0]) // 10**6))
                except Exception:
                    pass
    print(f"  {'symbol':<10} {'ticks':>7} {'dupes':>7} {'span(h)':>8} {'gaps>5s':>8} {'max gap':>8}")
    for s in sorted(ev):
        v = sorted(ev[s]); ts = [a for a, _ in v]
        d = [(b - a) / 1000 for a, b in zip(ts, ts[1:])]
        big = [x for x in d if x > 5]
        print(f"  {s:<10} {len(ts):>7} {dupe[s]:>7} {(ts[-1]-ts[0])/3.6e6:>8.1f} "
              f"{len(big):>8} {max(d) if d else 0:>7.0f}s")

    # ---------- 3 knowledge-time lag ----------
    hdr("3. KNOWLEDGE-TIME LAG  (recv_ns - payload ts), ms")
    print(f"  {'symbol':<10} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>9} {'negative':>9}")
    for s in sorted(ev):
        lag = sorted(k - t for t, k in ev[s])
        neg = sum(1 for x in lag if x < 0)
        q = lambda f: lag[min(int(len(lag) * f), len(lag) - 1)]
        print(f"  {s:<10} {q(.5):>8} {q(.9):>8} {q(.99):>8} {lag[-1]:>9} {neg:>9}")
    print("  negative lag would mean we recorded a tick before it existed (clock fault)")

    # ---------- 4 per-window coverage ----------
    hdr("4. PER-WINDOW TWAP COVERAGE over [t0-5s, T+5s]")
    markets = {}
    for ln in open(PM / "markets.jsonl"):
        try:
            m = json.loads(ln); markets[m["slug"]] = m
        except Exception:
            pass
    idx = {s: sorted(a for a, _ in v) for s, v in ev.items()}
    cov = Counter(); by_coin = defaultdict(lambda: [0, 0])
    for slug, up in res.items():
        m = markets.get(slug)
        if not m:
            cov["no metadata"] += 1; continue
        s = COINS.get(m["coin"]); ts = idx.get(s)
        if not ts:
            cov["no stream"] += 1; continue
        a, b = m["window_start"] * 1000 - 5000, m["window_end"] * 1000 + 5000
        lo, hi = bisect_right(ts, a), bisect_right(ts, b)
        n = hi - lo
        span = (b - a) / 1000
        ok = n >= span * 0.9 and bisect_right(ts, m["window_start"] * 1000) > 0
        cov["admissible" if ok else "sparse/absent"] += 1
        by_coin[m["coin"]][0 if ok else 1] += 1
    for k, v in cov.most_common():
        print(f"  {k:<18} {v}")
    print(f"\n  {'coin':<8} {'admissible':>11} {'excluded':>9}")
    for c in sorted(by_coin):
        print(f"  {c:<8} {by_coin[c][0]:>11} {by_coin[c][1]:>9}")

    # ---------- 5 market capture ----------
    hdr("5. MARKET CAPTURE — shards and book sanity")
    files = glob.glob(str(PM / "raw/*/*.jsonl*"))
    base = Counter(Path(f).name.split(".jsonl")[0] for f in files)
    multi = {k: v for k, v in base.items() if v > 1}
    print(f"  window files={len(files)}  distinct slugs={len(base)}  multi-shard slugs={len(multi)}")
    bad = Counter(); nbook = 0
    for f in sorted(glob.glob(str(PM / "raw/*/btc-updown-5m-*.jsonl*")))[-25:]:
        op = gzip.open if f.endswith(".gz") else open
        try:
            with op(f, "rt") as fh:
                for ln in fh:
                    p = ln.split("\t", 1)
                    if len(p) < 2:
                        continue
                    try:
                        msgs = json.loads(p[1])
                    except Exception:
                        bad["unparsable line"] += 1; continue
                    for mm in (msgs if isinstance(msgs, list) else [msgs]):
                        if mm.get("event_type") != "book":
                            continue
                        nbook += 1
                        bids, asks = mm.get("bids") or [], mm.get("asks") or []
                        if not bids or not asks:
                            bad["one-sided book"] += 1; continue
                        bb = max(float(x["price"]) for x in bids)
                        ba = min(float(x["price"]) for x in asks)
                        if bb >= ba: bad["crossed/locked"] += 1
                        if not (0 < bb < 1 and 0 < ba < 1): bad["out of (0,1)"] += 1
        except Exception:
            bad["unreadable file"] += 1
    print(f"  BTC book snapshots sampled={nbook}")
    for k, v in bad.most_common():
        print(f"    {k:<20} {v}")
    if not bad:
        print("    no anomalies")

    # ---------- 6 selection bias ----------
    hdr("6. SELECTION — is the analysed subset representative?")
    allw = Counter(m["coin"] for m in markets.values())
    resw = Counter(markets[s]["coin"] for s in res if s in markets)
    print(f"  {'coin':<8} {'discovered':>11} {'resolved':>9} {'ratio':>7}")
    for c in sorted(allw):
        r = resw.get(c, 0)
        print(f"  {c:<8} {allw[c]:>11} {r:>9} {r/allw[c]:>7.2f}")

    # ---------- 7 up-rate ----------
    hdr("7. UP-RATE — the drift confound in the calibration result")
    by = defaultdict(list); byday = defaultdict(list)
    for s, up in res.items():
        m = markets.get(s)
        if not m:
            continue
        by[m["coin"]].append(up)
        byday[m["window_start"] // 86400].append(up)
    print(f"  {'coin':<8} {'n':>6} {'up rate':>9}")
    for c in sorted(by):
        v = by[c]; print(f"  {c:<8} {len(v):>6} {sum(v)/len(v):>9.3f}")
    print(f"\n  {'day':<8} {'n':>6} {'up rate':>9}")
    for d in sorted(byday):
        v = byday[d]; print(f"  {d:<8} {len(v):>6} {sum(v)/len(v):>9.3f}")
    allv = [u for v in by.values() for u in v]
    p = sum(allv) / len(allv)
    se = math.sqrt(p * (1 - p) / len(allv))
    print(f"\n  pooled up-rate {p:.4f}  (naive se {se:.4f}, ignores cross-window dependence)")
    print(f"  distance from 0.5: {(p-0.5)/se:+.2f} naive sigma — treat as an UPPER bound "
          f"on significance, since windows share price paths")


if __name__ == "__main__":
    main()
