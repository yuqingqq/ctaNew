"""Forward test F1: short heavy-volume pumps +4 h -> +24 h (docs/FORWARD_FADE_F1_v2.md, superseding v1).

  python3 -m research.forward_fade            # counts + statuses only (allowed any time)
  python3 -m research.forward_fade --final    # the one look; refuses below the minimum or a second time
  python3 -m research.forward_fade --cost     # cost fields of the order-book snapshots, no prices (any time)
  python3 -m research.forward_fade --paper    # the live paper ledger + a cross-check with the evaluator's rows

Reads the records of the live recorder frozen for forward test v2 (data/alerts/followups-*.jsonl)
and admits only runs of that frozen code, exactly as research/forward_eval.py does. Funding over
each hold comes from Binance Vision's monthly fundingRate archives (the research source; Binance's
futures REST API answers HTTP 451 from this box); a row whose hold touches a month not yet
published is FUNDING_PENDING. Every pump that does not count carries its first failing status.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research.forward_eval as fe  # noqa: E402

DECLARATION = "docs/FORWARD_FADE_F1_v2.md"
_A2 = next(c for c in json.loads((ROOT / "docs" / "SIGNAL_SEARCH_FREEZE.json").read_text())["cells"] if c["id"] == "A2")
HEAVY = float(_A2["cuts"][1])                      # log10 volume ratio; 0.35259 -> 2.2525x
MEME = set(json.loads((ROOT / "docs" / "coingecko_meme_rank1500_2026-10-09.json").read_text())["symbols"])
FEE, SLIP = 5e-4, 15e-4                            # per side; research average slippage at stress 1 was 0.145%
COST = 2 * (FEE + SLIP)                            # 0.40% round trip
COST_STRESS = 2 * (FEE + 3 * SLIP)                 # 1.00% round trip, slippage x3
LATE = {"FU_4H": 600, "FU_24H": 3600}
DEDUP_S = 72 * 3600
MIN = {"rows": 400, "weeks": 12}
DRAWS, SEED = 4000, 20261010
VISION = "https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"
_PUBLISHED: dict[str, bool] = {}
NAN = float("nan")
UNBLINDED = "2026-10-10 02:30Z: paper P&L shown live by user decision (docs/FORWARD_UNBLIND_2026-10-10.md); read and pass rule unchanged"
BOOK = "data/alerts/fadebook.jsonl"


def base(asset: str) -> str:
    """Same rule as research/premium_angles.py::base (asset names carry no quote suffix live)."""
    s = asset.removesuffix("USDT")
    for pre in ("1000000", "1000", "1M"):
        if s.startswith(pre) and len(s) > len(pre) + 1:
            return s[len(pre):]
    return s


def check(rec: dict | None, kind: str, ref_key, runs: set, end: float | None) -> str:
    if rec is None:
        return "MISSING"
    if rec.get("run") not in runs or (end is not None and rec["t"] >= end):
        return "OTHER_RUN"
    if rec["late_s"] > LATE[kind]:
        return "LATE"
    if not fe._fin(rec.get("px")) or rec["px"] <= 0:
        return "NO_PRICE"
    if rec.get("px_key") != ref_key:
        return "PX_SOURCE_CHANGED"
    return "OK"


def repeats_72h(opens: dict[str, dict], keep: set[str]) -> set[str]:
    """Per asset, any direction: an alert < 72 h after the last kept alert of that asset is a repeat
    (research: persistence_study.dedupe_72h on all NEW alerts)."""
    out, last = set(), {}
    for o in sorted((opens[i] for i in keep), key=lambda o: (o["t"], o["id"])):
        lt = last.get(o["asset"])
        if lt is not None and o["t"] - lt < DEDUP_S:
            out.add(o["id"])
            continue
        last[o["asset"]] = o["t"]
    return out


def months(t0: float, t1: float) -> list[str]:
    a, b = (pd.Timestamp(t, unit="s").to_period("M") for t in (t0, t1))
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def published_month(month: str, head=requests.head) -> bool:
    """A month's funding archives are out once BTCUSDT's is (Binance publishes them together).
    Checks existence only; reads no rate."""
    if month not in _PUBLISHED:
        try:
            ok = head(VISION.format(s="BTCUSDT", m=month), timeout=20, allow_redirects=True).status_code == 200
        except Exception:  # noqa: BLE001 -- unreachable reads as not yet published
            return False
        if not ok:
            return False                              # re-checked next time
        _PUBLISHED[month] = True
    return _PUBLISHED[month]


def rows(recs: list[dict], runs: set, start: float | None, end: float | None, published=None) -> pd.DataFrame:
    published = published or published_month
    opens, f4, f24 = (fe.first_by_id(recs, k) for k in ("FU_OPEN", "FU_4H", "FU_24H"))
    in_test = {i for i, o in opens.items() if o.get("run") in runs and start is not None and o["t"] >= start
               and (end is None or o["t"] < end)}
    dup = fe.duplicates(opens, runs)
    rep = repeats_72h(opens, in_test - dup)
    out = []
    for i, o in opens.items():
        if o["dir"] <= 0:
            continue
        a4, a24 = f4.get(i) or {}, f24.get(i) or {}
        ref = o.get("ref_key") or ""
        s4 = check(f4.get(i), "FU_4H", ref, runs, end)
        s24 = check(f24.get(i), "FU_24H", ref, runs, end)
        vbn = a4.get("vol_ratio_binance_usdm")
        gaps = max(a4.get("gap_min_before", math.inf), a4.get("gap_min_after", math.inf))
        vol_ok = fe._fin(vbn) and vbn > 0 and gaps <= fe.GAP_TOL_MIN
        pend = s24 == "OK" and s4 == "OK" and not all(published(m) for m in months(a4["t"], a24["t"]))
        st = fe._first_fail([("OTHER_RUN", i not in in_test), ("DUPLICATE", i in dup), ("REPEAT_72H", i in rep),
                             ("NOT_BINANCE_USDM", not ref.startswith("binance_usdm:")), ("MEME", base(o["asset"]) in MEME),
                             (f"4H_{s4}", s4 != "OK"), ("VOL_GAP", not vol_ok),
                             ("NOT_HEAVY", vol_ok and math.log10(vbn) <= HEAVY), (f"24H_{s24}", s24 != "OK"),
                             ("FUNDING_PENDING", pend)])
        out.append({"id": i, "asset": o["asset"], "symbol": ref.split(":", 1)[-1], "alert_ts": o["t"], "status": st,
                    "t4": a4.get("t"), "t24": a24.get("t"), "px4": a4.get("px"), "px24": a24.get("px"),
                    "lvr_bn": math.log10(vbn) if vol_ok else NAN})
    d = pd.DataFrame(out)
    if len(d):
        d["week"] = pd.to_datetime(d["alert_ts"], unit="s").dt.to_period("W").dt.start_time
    return d


def _month_csv(symbol: str, month: str, get=requests.get) -> pd.DataFrame | None:
    p = ROOT / "data" / "forward_fade" / "funding" / f"{symbol}_{month}.csv"
    if not p.exists():
        r = get(VISION.format(s=symbol, m=month), timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        p.write_bytes(z.open(z.namelist()[0]).read())
    return pd.read_csv(p)


def funding_sum(symbol: str, t0: float, t1: float, get=requests.get) -> float:
    """Sum of settled funding rates (Vision `last_funding_rate`) with calc_time in (t0, t1] (seconds),
    as research/btc_hedge.Funding.total; NaN when an archive is missing or unreadable."""
    try:
        parts = [_month_csv(symbol, m, get) for m in months(t0, t1)]
        if any(p is None for p in parts):
            return NAN
        df = pd.concat(parts)
        sel = (df["calc_time"] > t0 * 1000) & (df["calc_time"] <= t1 * 1000)
        return float(df.loc[sel, "last_funding_rate"].sum())
    except Exception:  # noqa: BLE001 -- unavailable funding is a status, never a zero
        return NAN


def trades(d: pd.DataFrame, fund=funding_sum) -> pd.DataFrame:
    x = d[d["status"] == "OK"].copy()
    x["gross"] = -(x["px24"] / x["px4"] - 1.0)
    x["funding"] = [fund(s, a, b) for s, a, b in zip(x["symbol"], x["t4"], x["t24"])]
    x.loc[x["funding"].isna(), "status"] = "NO_FUNDING"
    x["net"] = x["gross"] - COST + x["funding"]
    x["net_stress"] = x["gross"] - COST_STRESS + x["funding"]
    return x


def mean_ci(v: np.ndarray, wk: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    uw = np.unique(wk)
    pos = [np.flatnonzero(wk == u) for u in uw]
    bs = np.array([v[np.concatenate([pos[k] for k in rng.integers(0, len(uw), len(uw))])].mean() for _ in range(DRAWS)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci": [float(lo), float(hi)], "n": int(len(v)), "weeks": int(len(uw))}


def final_stats(x: pd.DataFrame) -> dict:
    ok = x[x["status"] == "OK"]
    wk = ok["week"].to_numpy()
    v = ok["net"].to_numpy()
    w, l = v[v > 0], v[v <= 0]
    h3 = mean_ci(v, wk)
    h3["pass"] = bool(h3["mean"] > 0 and h3["ci"][0] > 0)
    return {"H3_net": h3, "gross": mean_ci(ok["gross"].to_numpy(), wk), "funding_mean": float(ok["funding"].mean()),
            "net_stress": mean_ci(ok["net_stress"].to_numpy(), wk),
            "win_rate": float((v > 0).mean()), "median": float(np.median(v)), "worst": float(v.min()),
            "lose_gt_25pct": float((v < -0.25).mean()), "losses_over_wins": float(-l.sum() / w.sum()) if w.size else NAN,
            "no_funding": int((x["status"] == "NO_FUNDING").sum())}


def counts(d: pd.DataFrame, start: float | None, end: float | None, now: float) -> dict:
    return {"test_start": str(pd.Timestamp(start, unit="s")) if start else None,
            "test_end": str(pd.Timestamp(end, unit="s")) if end else None,
            "complete_weeks": fe.complete_weeks(start, end or now) if start else 0,
            "pumps": int(len(d)), "status": d["status"].value_counts().to_dict() if len(d) else {},
            "complete": int((d["status"] == "OK").sum()) if len(d) else 0}


def ready(c: dict) -> bool:
    return c["complete"] >= MIN["rows"] and c["complete_weeks"] >= MIN["weeks"]


def cost_summary(recs: list[dict]) -> dict:
    """Cost fields only (spread, walk-the-book cost, depth): never a price, so it reveals no trade
    outcome and may be read any time. Exit snapshots take the heavy label of their entry."""
    heavy = {r["id"]: r.get("heavy") for r in recs if r.get("leg") == "entry"}
    out = {"snapshots": len(recs), "errors": sum(1 for r in recs if r.get("error"))}
    for leg, side in (("entry", "sell_bps"), ("exit", "buy_bps")):
        for lab, want in (("heavy", True), ("all", None)):
            g = [r for r in recs if r.get("leg") == leg and not r.get("error")
                 and (want is None or heavy.get(r.get("id")) is want)]
            cell = {"n": len(g)}
            if g:
                cell["spread_bps_median"] = float(np.median([r["spread_bps"] for r in g]))
                for n in sorted({k for r in g for k in r[side]}, key=float):
                    v = [r[side].get(n) for r in g]
                    ok = [x for x in v if x is not None]
                    cell[f"{n}_usd"] = {"median_bps": float(np.median(ok)) if ok else None,
                                        "p90_bps": float(np.percentile(ok, 90)) if ok else None,
                                        "book_too_thin": int(sum(x is None for x in v))}
            out[f"{leg}_{lab}"] = cell
    return out


def cost_counts() -> dict:
    recs = []
    for f in sorted((ROOT / "data" / "alerts").glob("fadecost-*.jsonl")):
        recs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return cost_summary(recs)


def paper_report(d: pd.DataFrame | None = None) -> dict:
    """The live paper ledger (shown by user decision, docs/FORWARD_UNBLIND_2026-10-10.md), checked
    against the evaluator's rows when `d` is given."""
    from movewatch.watchers import PaperBook
    b = PaperBook()
    p = ROOT / BOOK
    for line in (p.read_text().splitlines() if p.exists() else []):
        b.apply(json.loads(line))
    print(f"{'entered (UTC)':17} {'asset':10} {'src':8} {'entry':>11} {'exit':>11} {'gross':>8} {'funding':>16} {'net':>8}")
    for t in sorted(b.tr.values(), key=lambda t: t["t4"]):
        when = pd.Timestamp(t["t4"], unit="s").strftime("%m-%d %H:%M")
        if t.get("t24") is None:
            print(f"{when:17} {t['asset'][:10]:10} {t['src']:8} {t['px4']:11.5g} {'open':>11}")
            continue
        if t.get("void"):
            print(f"{when:17} {t['asset'][:10]:10} {t['src']:8} {t['px4']:11.5g} {'void':>11}  ({t['void']})")
            continue
        f, how = b.funding(t)
        net = b.net(t)
        print(f"{when:17} {t['asset'][:10]:10} {t['src']:8} {t['px4']:11.5g} {t['px24']:11.5g} {100 * b.gross(t):+7.2f}% "
              f"{(f'{100 * f:+.3f}% ' + how) if f is not None else 'pending':>16} {(f'{100 * net:+7.2f}%') if net is not None else '      ?':>8}")
    s = b.summary()
    print(b.line())
    if d is not None and len(d):
        ev = set(d.loc[d["status"].isin(["OK", "FUNDING_PENDING"]), "id"])
        bk = {t["id"] for t in b.closed()}
        s["evaluator_check"] = {"evaluator_complete": len(ev), "book_closed": len(bk),
                                "only_in_book": sorted(bk - ev), "only_in_evaluator": sorted(ev - bk)}
        print("cross-check:", s["evaluator_check"])
    return s


def declaration_commit() -> str:
    """The commit that added the declaration (CLAUDE.md rule 12: a freeze is a commit)."""
    out = fe._git(ROOT, "log", "--diff-filter=A", "--format=%H", "--", DECLARATION).decode().split()
    if not out:
        raise SystemExit(f"refused: {DECLARATION} is not committed, so there is no freeze")
    return out[-1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args(argv)
    if args.cost:
        print(json.dumps(cost_counts(), indent=1))
        return
    decl = declaration_commit()
    commit = fe.freeze_commit()                      # the v2 recorder identity this test reads
    sha = fe.fingerprint_at(commit)
    recs = fe.load(ROOT)
    runs, start, end = fe.admission(recs, sha)
    now = time.time()
    d = rows(recs, runs, start, end)
    c = counts(d, start, end, now)
    c.update(declaration_commit=decl, recorder_freeze_commit=commit, recorder_sha=sha, admitted_runs=len(runs), minimum=MIN)
    print(json.dumps(c, indent=1, default=str))
    if args.paper:
        paper_report(d)
        return
    ok = ready(c)
    if not args.final:
        print("counts only (reading rule). minimum met." if ok else "counts only (reading rule). minimum NOT met yet.")
        return
    if not ok:
        raise SystemExit(f"refused: minimum sample not met ({MIN}); the test is read once, at the minimum")
    out = ROOT / "data" / "forward_fade"
    if list(out.glob("receipt_*.json")):
        raise SystemExit(f"refused: a receipt already exists in {out}; the test is read once")
    x = trades(d)
    res = {"declaration_commit": decl, "recorder_freeze_commit": commit, "recorder_sha": sha, "unblinded": UNBLINDED,
           "evaluator_head": fe._git(ROOT, "rev-parse", "HEAD").decode().strip(), "counts": c, **final_stats(x)}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"receipt_{pd.Timestamp.utcnow():%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    print("receipt:", path)


if __name__ == "__main__":
    main()
