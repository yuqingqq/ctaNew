"""EXP-BLEND — build p̂ and score it against the book on resolved outcomes.

Target verified by EXP-M6: Up iff S60(T) >= S60(t0), 99.8% winner reproduction.

Model, read at KNOWLEDGE TIME only (t_known = recv_ns, never payload ts):
    r      = T - t                       time remaining
    X_0    = S60 at window open          the strike
    E[X_T] = S60 as last observed at t   martingale forecast
    Var    = sigma^2 * (r - 2w/3)   for r > w      (pre-averaging)
             sigma^2 * r^3/(3w^2)  for r <= w      (in-window r^3 lock-in)
    p_hat  = Phi( (E[X_T] - X_0) / sigma_eff )

sigma is ONE free parameter, fitted by maximum likelihood on realised winners
(the loss function is probability accuracy, not vol accuracy) and applied
WALK-FORWARD: fit on days strictly before d, score day d.

Scored against the venue's own book mid at the same instant. The comparison is
paired per (window, decision time) and clustered by day, because 288 windows a
day ride one underlying path.

Run: python3 -u -m live.pm_research.exp_blend_model
"""
from __future__ import annotations

import glob, gzip, json, math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
COINS = {"btc": "btc/usd", "eth": "eth/usd", "sol": "sol/usd", "xrp": "xrp/usd",
         "doge": "doge/usd", "bnb": "bnb/usd", "hype": "hype/usd"}
W = 60.0                      # verified averaging window, seconds
GRID = [30, 60, 120, 180, 240, 270]     # seconds into the 300 s window
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load_twap60():
    rows = defaultdict(list)
    for f in sorted(glob.glob(str(PM / "prices/crypto_prices_twap_sixty/*.csv*"))):
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            for ln in fh:
                p = ln.split("\t", 1)
                if len(p) < 2:
                    continue
                try:
                    m = json.loads(p[1]); pl = m.get("payload") or {}
                    if pl.get("symbol") and pl.get("timestamp"):
                        rows[pl["symbol"]].append(
                            (int(p[0]) // 10**6, float(pl.get("full_accuracy_value", 0))))
                except Exception:
                    pass
    out = {}
    for k, v in rows.items():
        v.sort()
        tk = [a for a, _ in v]; val = [b for _, b in v]
        out[k] = (tk, val)
    return out


def at_known(series, ms):
    tk, val = series
    i = bisect_right(tk, ms) - 1
    return val[i] if i >= 0 else None


def book_mid_series(slug, up_asset):
    """(t_known_ms, mid) for the Up token, from recorded book snapshots."""
    pat = str(PM) + "/raw/*/" + slug + ".jsonl*"
    pts = []
    for f in sorted(glob.glob(pat)):
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
                        continue
                    for m in (msgs if isinstance(msgs, list) else [msgs]):
                        if m.get("event_type") != "book" or m.get("asset_id") != up_asset:
                            continue
                        bids, asks = m.get("bids") or m.get("buys") or [], m.get("asks") or m.get("sells") or []
                        if not bids or not asks:
                            continue
                        bb = max(float(x["price"]) for x in bids)
                        ba = min(float(x["price"]) for x in asks)
                        if 0 < bb < ba < 1:
                            pts.append((int(p[0]) // 10**6, (bb + ba) / 2))
        except Exception:
            pass
    if not pts:
        return None
    pts.sort()
    return ([a for a, _ in pts], [b for _, b in pts])


def main():
    tw = load_twap60()
    markets = {}
    for ln in open(PM / "markets.jsonl"):
        try:
            m = json.loads(ln); markets[m["slug"]] = m
        except Exception:
            pass
    res = {}
    for ln in open(PM / "resolutions.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("closed") is True and r.get("winners"):
                res[r["slug"]] = bool(r["winners"].get("Up"))
        except Exception:
            pass

    # ---- assemble samples: (day, coin, r, z, up, book_mid) ; z = (E[X_T]-X_0)/sqrt(varfac)
    samples = []
    nb = 0
    for slug, up_won in sorted(res.items()):
        m = markets.get(slug)
        if not m:
            continue
        sym = COINS.get(m["coin"]); s = tw.get(sym)
        if not s:
            continue
        t0, T = m["window_start"] * 1000, m["window_end"] * 1000
        x0 = at_known(s, t0)
        if not x0:
            continue
        toks = m.get("clobTokenIds") or []
        outs = m.get("outcomes")
        up_asset = None
        if toks and outs:
            o = json.loads(outs) if isinstance(outs, str) else outs
            up_asset = toks[o.index("Up")] if "Up" in o else toks[0]
        bs = book_mid_series(slug, up_asset) if up_asset else None
        if bs:
            nb += 1
        day = m["slug"].split("-")[-1]
        day = str(m["window_start"] // 86400)
        for t in GRID:
            ms = t0 + t * 1000
            ex = at_known(s, ms)
            if ex is None:
                continue
            r = 300.0 - t
            varfac = (r - 2 * W / 3) if r > W else (r ** 3) / (3 * W * W)
            if varfac <= 0:
                continue
            z = (ex - x0) / x0 / math.sqrt(varfac)      # relative, per sqrt-second
            bm = at_known(bs, ms) if bs else None
            samples.append((day, m["coin"], r, z, up_won, bm))
    print(f"[blend] windows with book coverage: {nb} | samples: {len(samples)}")

    days = sorted({s[0] for s in samples})
    print(f"[blend] days: {len(days)}")

    def nll(sig, rows):
        tot = 0.0
        for _, _, _, z, up, _ in rows:
            p = min(max(N(z / sig), 1e-6), 1 - 1e-6)
            tot -= math.log(p if up else 1 - p)
        return tot

    def fit(rows):
        lo, hi = 1e-6, 5e-2
        for _ in range(60):
            a = lo + (hi - lo) / 3; b = hi - (hi - lo) / 3
            if nll(a, rows) < nll(b, rows):
                hi = b
            else:
                lo = a
        return (lo + hi) / 2

    sig_all = fit(samples)
    print(f"[blend] pooled sigma (MLE on winners) = {sig_all*1e4:.3f} bps/sqrt(s)"
          f"  ~ {sig_all*math.sqrt(365*24*3600)*100:.0f}%/yr")

    # ---- walk-forward scoring, paired vs the book
    per_day = defaultdict(lambda: {"n": 0, "bm": 0.0, "bb": 0.0, "lm": 0.0, "lb": 0.0})
    for i, d in enumerate(days):
        if i == 0:
            continue
        train = [s for s in samples if s[0] < d]
        test = [s for s in samples if s[0] == d and s[5] is not None]
        if len(train) < 200 or not test:
            continue
        sig = fit(train)
        for _, _, _, z, up, bm in test:
            pm = min(max(N(z / sig), 1e-6), 1 - 1e-6)
            pb = min(max(bm, 1e-6), 1 - 1e-6)
            y = 1.0 if up else 0.0
            a = per_day[d]
            a["n"] += 1
            a["bm"] += (pm - y) ** 2
            a["bb"] += (pb - y) ** 2
            a["lm"] -= math.log(pm if up else 1 - pm)
            a["lb"] -= math.log(pb if up else 1 - pb)

    if not per_day:
        print("[blend] insufficient walk-forward coverage — need >1 day with book data")
        return
    print(f"\n{'day':<8} {'n':>6} {'Brier model':>12} {'Brier book':>11} {'Δ (m-b)':>9}"
          f" {'LL model':>9} {'LL book':>9}")
    deltas = []
    for d in sorted(per_day):
        a = per_day[d]
        bm, bb = a["bm"] / a["n"], a["bb"] / a["n"]
        deltas.append(bm - bb)
        print(f"{d:<8} {a['n']:>6} {bm:>12.4f} {bb:>11.4f} {bm-bb:>+9.4f}"
              f" {a['lm']/a['n']:>9.4f} {a['lb']/a['n']:>9.4f}")
    md = sum(deltas) / len(deltas)
    sd = (sum((x - md) ** 2 for x in deltas) / max(len(deltas) - 1, 1)) ** 0.5
    se = sd / math.sqrt(len(deltas)) if len(deltas) > 1 else float("nan")
    print(f"\nPAIRED ΔBrier (model − book), day-clustered: {md:+.4f}"
          f"  se {se:.4f}  t {md/se if se else float('nan'):+.2f}  ({len(deltas)} days)")
    print("negative Δ = model beats the book")


if __name__ == "__main__":
    main()
