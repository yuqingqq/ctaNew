"""DA's verify-first check for a forward day. Run BEFORE anyone scores it.

SURFACE AUTHORISATION (R-126, in-file): R-141(1) made daily admission a gated
act — "DA verifies the day (complete tape, gap rate under bar, entirely
post-freeze), DE appends it to the split, BE's harness scores it" — and
R-153(2) restored DA-VERIFIES-FIRST as a HARD PRECONDITION after the gate ran
out of order on day one. This module implements that standing duty.

WHY IT IS A MODULE AND NOT A SCRIPT I REWRITE EACH NIGHT. The duty is
recurring, the predicates are fixed, and the one time it ran late the day had
already been scored — which turned an ordinary exclusion decision into a
post-hoc call on a visible result (Q-DA-69). An improvised check is also a
check nobody else can run: this one is committed, carries its own falsifiers,
and needs no setup, so any seat can discharge the duty on any night.

THE THREE PREDICATES ARE R-141(1)'s, UNCHANGED. What this adds is the number
Q-DA-69 showed actually decides it: gaps per hour understates the damage, and
WINDOWS AFFECTED is the quantity that matters — 08-25 ran 28.0 gaps/hr, which
sounds survivable, while 231 of 288 btc windows (80.2%) carried a gap.

A DAY THAT FAILS IS EXCLUDED WITH A STATED REASON, NEVER SILENTLY SKIPPED
(R-141(1)). This module states reasons; it does not exclude. The decision is
the policy layer's (rule 14) and the exclusion is the coordinator's to rule.

    python3 live/pm_research/da_forward_day_verify.py --selftest
    python3 live/pm_research/da_forward_day_verify.py verify --day 20260827
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
PM_GAPS = REPO / "data/pm_5min/collector_gaps.jsonl"

WINDOW_S = 300
WINDOWS_PER_DAY = 288
#: CLASS A, from R-141(3). A reporting threshold; no verdict rests on it alone,
#: and the windows-affected figure is published beside it precisely because
#: this one understated the damage on day one.
GAP_BAR_PER_HR = 15.0


def day_bounds(day_token: str) -> tuple[int, int]:
    d = dt.datetime.strptime(day_token, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    lo = int(d.timestamp())
    return lo, lo + 86400


def gap_series(lo: int, hi: int) -> dict[str, Any]:
    """Per-hour PM gap counts, with unparseable lines COUNTED (rule 4)."""
    per_hr: collections.Counter = collections.Counter()
    lost_ns: collections.Counter = collections.Counter()
    causes: collections.Counter = collections.Counter()
    n_lines = n_bad = 0
    for line in PM_GAPS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        if r.get("event") != "gap_closed":
            continue
        s = r.get("gap_start_ns")
        if not s:
            continue
        t = s / 1e9
        if not (lo <= t < hi):
            continue
        hr = int((t - lo) // 3600)
        per_hr[hr] += 1
        lost_ns[hr] += max(0, (r.get("gap_end_ns") or s) - s)
        causes[r.get("cause", "?")] += 1
    hours = sorted(per_hr)
    over = [h for h in hours if per_hr[h] > GAP_BAR_PER_HR]
    span = max(hours) + 1 if hours else 0
    return {
        "ledger_lines": n_lines, "ledger_unparseable": n_bad,
        "n_gaps": sum(per_hr.values()),
        "lost_s": round(sum(lost_ns.values()) / 1e9, 1),
        "hours_observed": span,
        "gaps_per_hour": round(sum(per_hr.values()) / span, 2) if span else None,
        "hours_over_bar": over, "n_hours_over_bar": len(over),
        "worst_hour": max(per_hr.values()) if per_hr else 0,
        "per_hour": {str(h): per_hr[h] for h in hours},
        "causes": dict(causes),
    }


def verify_day(day_token: str, freeze_epoch: float,
               coins: Sequence[str] = ("btc", "eth")) -> dict[str, Any]:
    import warning_window as WW
    import da_hf_pm_alignment as A
    import policy_optimizer_queue_realistic as qr

    lo, hi = day_bounds(day_token)
    iso = dt.datetime.fromtimestamp(lo, dt.timezone.utc).strftime("%Y-%m-%d")
    now = dt.datetime.now(dt.timezone.utc)
    preds: list[dict[str, Any]] = []

    def p(name, ok, detail):
        preds.append({"predicate": name, "pass": bool(ok), "detail": detail})

    # --- 1. entirely post-freeze ------------------------------------------
    sel = WW.select_holdout(freeze_epoch)
    day = sel["days"].get(iso)
    if day is None:
        p("entirely_post_freeze", False,
          f"{iso} absent from the selector -- cannot be verified, not passed")
        adm = tot = {}
        day_closed = None
    else:
        adm, tot = day["n_admissible_by_coin"], day["n_total_by_coin"]
        day_closed = day["day_closed"]
        allpost = bool(tot) and all(adm[c] == tot[c] for c in tot)
        p("entirely_post_freeze", allpost,
          f"day_closed={day_closed}; " + ", ".join(
              f"{c} {adm.get(c)}/{tot.get(c)}" for c in sorted(tot)[:4])
          + (f" (+{len(tot)-4} more)" if len(tot) > 4 else ""))

    # --- 2. complete tape --------------------------------------------------
    w = A.pm_windows([day_token])
    counts = {c: len([x for x in w.get(c, []) if lo <= x < hi]) for c in coins}
    elapsed = min(WINDOWS_PER_DAY, max(0, int((now.timestamp() - lo) // WINDOW_S)))
    expect = WINDOWS_PER_DAY if day_closed else elapsed
    short = {c: expect - n for c, n in counts.items()}
    p("complete_tape", all(v <= 0 for v in short.values()),
      f"expect {expect} ({'closed day' if day_closed else 'elapsed so far'}); "
      + ", ".join(f"{c} {counts[c]} (short {short[c]})" for c in coins))

    # --- 3. gap rate under bar ---------------------------------------------
    gs = gap_series(lo, hi)
    p("gap_rate_under_bar", gs["n_hours_over_bar"] == 0,
      f"{gs['n_gaps']} gaps over {gs['hours_observed']}h = "
      f"{gs['gaps_per_hour']}/hr vs bar {GAP_BAR_PER_HR:.0f}; "
      f"{gs['n_hours_over_bar']} hours over; worst {gs['worst_hour']}; "
      f"{gs['lost_s']}s lost")

    # --- the number Q-DA-69 showed actually decides it ---------------------
    fi = qr.base.fi
    gaps = fi.gaps_by_slug(fi.ERA)
    cov = fi.covered_slugs(fi.ERA)
    affected: dict[str, dict[str, Any]] = {}
    for coin in coins:
        tot_w = aff = 0
        for slug in cov:
            if not slug.startswith(coin + "-"):
                continue
            try:
                ws = int(slug.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if not (lo <= ws < hi):
                continue
            tot_w += 1
            if gaps.get(slug):
                aff += 1
        affected[coin] = {
            "era_covered_windows": tot_w, "gap_affected": aff,
            "gap_affected_pct": round(100.0 * aff / tot_w, 1) if tot_w else None}

    return {
        "instrument": "da_forward_day_verify_v1",
        "authorised_by": "R-141(1); DA-verifies-first restored as a hard "
                         "precondition by R-153(2)",
        "day": iso, "day_token": day_token,
        "as_of_utc": now.isoformat(),
        "freeze_epoch": freeze_epoch,
        "day_closed": day_closed,
        "predicates": preds,
        "all_pass": all(x["pass"] for x in preds),
        "windows_gap_affected": affected,
        "gap_series": gs,
        "decision_note": (
            "This instrument STATES REASONS; it does not exclude. R-141(1): a "
            "day that fails DA's verification is EXCLUDED WITH A STATED "
            "REASON, and that exclusion is the coordinator's to rule, not a "
            "worker's boolean (rule 14). Note also that gaps/hour understates "
            "damage: day one ran 28.0/hr while 80.2% of btc windows carried a "
            "gap -- read windows_gap_affected beside it, never instead of it."),
    }


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    lo, hi = day_bounds("20260827")
    ok(hi - lo == 86400, "a day is 86400s")
    ok(dt.datetime.fromtimestamp(lo, dt.timezone.utc).strftime("%Y-%m-%d")
       == "2026-08-27", "the token maps to the right UTC day")
    ok(day_bounds("20260826")[1] == lo,
       "consecutive days abut exactly -- no gap, no overlap at midnight")
    ok(WINDOWS_PER_DAY * WINDOW_S == 86400, "288 windows tile the day exactly")

    # the gap series must COUNT unparseable lines, never drop them (rule 4)
    import tempfile
    global PM_GAPS
    real = PM_GAPS
    try:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "g.jsonl"
            base = lo + 60
            good = [json.dumps({"event": "gap_closed",
                                "gap_start_ns": int((base + i) * 1e9),
                                "gap_end_ns": int((base + i + 2) * 1e9),
                                "cause": "PING_TIMEOUT"}) for i in range(20)]
            f.write_text("\n".join(good + ["not json at all", ""]),
                         encoding="utf-8")
            PM_GAPS = f
            gs = gap_series(lo, hi)
            ok(gs["ledger_unparseable"] == 1,
               "an unreadable ledger line is COUNTED, not silently dropped")
            ok(gs["n_gaps"] == 20, "gaps inside the day are counted")
            ok(gs["n_hours_over_bar"] == 1 and gs["hours_over_bar"] == [0],
               "20 gaps in hour 0 breaches the bar of 15 and is named")
            ok(gs["worst_hour"] == 20, "the worst hour is reported")
            ok(gs["lost_s"] == 40.0, "lost seconds are summed")
            # FALSIFIER: a quiet day must NOT breach -- else the bar can't fire
            f.write_text("\n".join(good[:5]), encoding="utf-8")
            gq = gap_series(lo, hi)
            ok(gq["n_hours_over_bar"] == 0,
               "5 gaps in an hour does NOT breach -- no false positive")
            ok(gq["n_gaps"] != gs["n_gaps"],
               "the two ledgers get DIFFERENT answers (R-42 mirror)")
            # a gap OUTSIDE the day must not count
            f.write_text(json.dumps({"event": "gap_closed",
                                     "gap_start_ns": int((hi + 10) * 1e9),
                                     "gap_end_ns": int((hi + 12) * 1e9)}),
                         encoding="utf-8")
            ok(gap_series(lo, hi)["n_gaps"] == 0,
               "a gap after midnight belongs to the NEXT day, not this one")
            # non-gap events must be ignored
            f.write_text(json.dumps({"event": "collector_start",
                                     "recv_ns": int((lo + 5) * 1e9)}),
                         encoding="utf-8")
            ok(gap_series(lo, hi)["n_gaps"] == 0,
               "a collector_start is not a gap")
    finally:
        PM_GAPS = real

    print(f"da_forward_day_verify selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["verify"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day", default=None, help="YYYYMMDD, e.g. 20260827")
    ap.add_argument("--freeze-epoch", type=float, default=1787583868.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if not a.day:
        raise SystemExit("--day YYYYMMDD is required; refusing to guess a day")
    rep = verify_day(a.day, a.freeze_epoch)
    text = json.dumps(rep, indent=2, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    print("\nPREDICATES")
    for x in rep["predicates"]:
        print(f"  [{'PASS' if x['pass'] else 'FAIL'}] {x['predicate']}: {x['detail']}")
    print("\nWINDOWS GAP-AFFECTED (the number that decides it, Q-DA-69)")
    for c, v in sorted(rep["windows_gap_affected"].items()):
        print(f"  {c}: {v['gap_affected']}/{v['era_covered_windows']} "
              f"({v['gap_affected_pct']}%)")
    print(f"\nALL PASS: {rep['all_pass']}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
