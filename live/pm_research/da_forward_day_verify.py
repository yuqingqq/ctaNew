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

#: R-211(3). PER-COIN verdicts from this day forward; days BEFORE it are judged
#: by the FROZEN whole-day rule, unchanged. Declared 2026-08-27 while NO 08-28
#: DATA EXISTS -- that is the whole point. Day one (08-27) is failing on btc
#: while eth is clean, and choosing the granularity that rescues eth AFTER
#: seeing that would be selecting on the result (rule 11). So 08-27 is judged
#: whole-day, eth is lost with it, and the better rule starts at a boundary no
#: one has looked past.
#:
#: CLASS A and PROSPECTIVE-ONLY: moving this value later, once a coin-day
#: verdict is visible, would retro-judge a day under a rule chosen to change
#: its answer. If asked to move it after a result is visible, REFUSE and record
#: the refusal (the standing Class-C/D instruction).
PER_COIN_RULE_FROM_DAY = "20260828"


def verdict_granularity(day_token: str) -> str:
    """Which rule judges this day. A DATE predicate, not a caller's flag.

    Deliberately takes no override argument: a granularity that a caller can
    choose is a granularity that gets chosen after seeing the numbers.
    """
    return ("per_coin"
            if dt.datetime.strptime(day_token, "%Y%m%d")
            >= dt.datetime.strptime(PER_COIN_RULE_FROM_DAY, "%Y%m%d")
            else "aggregate_frozen")


def day_bounds(day_token: str) -> tuple[int, int]:
    d = dt.datetime.strptime(day_token, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    lo = int(d.timestamp())
    return lo, lo + 86400


def gap_series(lo: int, hi: int, now: float | None = None,
               coin: str | None = None) -> dict[str, Any]:
    """Per-hour PM gap counts, with unparseable lines COUNTED (rule 4).

    THE DENOMINATOR IS ELAPSED TIME, NOT THE SPAN OF THE GAPS THEMSELVES.
    The first version divided by `max(gap_hour) + 1`, which is fine for a full
    day with gaps in every hour and badly wrong for a partial one: fired 30 s
    after the 08-27 boundary it reported "1 gaps over 1h = 1.0/hr" for a single
    gap in thirty seconds -- understating by ~120x, and in the direction that
    reads as GOOD NEWS. A rate whose denominator comes from the numerator's own
    distribution is not a rate.
    """
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
        if coin is not None and r.get("coin") != coin:
            continue
        hr = int((t - lo) // 3600)
        per_hr[hr] += 1
        lost_ns[hr] += max(0, (r.get("gap_end_ns") or s) - s)
        causes[r.get("cause", "?")] += 1
    hours = sorted(per_hr)
    over = [h for h in hours if per_hr[h] > GAP_BAR_PER_HR]
    now = dt.datetime.now(dt.timezone.utc).timestamp() if now is None else now
    elapsed_h = max(0.0, (min(now, hi) - lo)) / 3600.0
    # Below an hour of tape there is no rate to report. Saying so beats
    # publishing a number that will be read as a trend.
    rate = (round(sum(per_hr.values()) / elapsed_h, 2)
            if elapsed_h >= 1.0 else None)
    return {
        "coin": coin,          # None = every coin, the frozen whole-day basis
        "ledger_lines": n_lines, "ledger_unparseable": n_bad,
        "n_gaps": sum(per_hr.values()),
        "lost_s": round(sum(lost_ns.values()) / 1e9, 1),
        "hours_elapsed": round(elapsed_h, 3),
        "hours_with_a_gap": len(hours),
        "gaps_per_hour": rate,
        "rate_estimable": elapsed_h >= 1.0,
        "rate_note": (None if elapsed_h >= 1.0 else
                      f"only {elapsed_h*3600:.0f}s elapsed -- NO RATE REPORTED; "
                      f"{sum(per_hr.values())} raw gaps so far"),
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
    # CALENDAR closure, not the selector's tape-derived `day_closed`. The
    # selector derives closure from "a strictly later window exists on disk",
    # which at 00:00:30 is not yet true -- so it called 08-26 OPEN thirty
    # seconds after 08-26 ended, and that boolean was feeding this branch. A
    # field that misdescribes closure driving a verdict is the R-139 class in
    # miniature. A day whose end is in the past is closed, and no tape state
    # changes that.
    calendar_closed = now.timestamp() >= hi
    expect = WINDOWS_PER_DAY if calendar_closed else elapsed
    short = {c: expect - n for c, n in counts.items()}
    basis = "closed day (calendar)" if calendar_closed else "elapsed so far"
    disagree = (day_closed is not None and bool(day_closed) != calendar_closed)
    # An empty expectation must NEVER pass: 0 present of 0 expected is the
    # empty-set-passes trap, and it PASSED on the 08-27 arm at 00:00:30.
    p("complete_tape",
      expect > 0 and all(v <= 0 for v in short.values()),
      f"expect {expect} ({basis}); "
      + ", ".join(f"{c} {counts[c]} (short {short[c]})" for c in coins)
      + ("" if expect > 0 else "  <-- NOTHING ELAPSED: cannot pass on an empty "
                              "expectation")
      + (f"  [selector day_closed={day_closed} DISAGREES with the calendar; "
         f"its tape-derived predicate lags the boundary by up to one window]"
         if disagree else ""))

    # --- 3. gap rate under bar ---------------------------------------------
    gs = gap_series(lo, hi, now.timestamp())
    p("gap_rate_under_bar",
      gs["rate_estimable"] and gs["n_hours_over_bar"] == 0,
      (f"{gs['n_gaps']} gaps over {gs['hours_elapsed']}h elapsed = "
       f"{gs['gaps_per_hour']}/hr vs bar {GAP_BAR_PER_HR:.0f}; "
       f"{gs['n_hours_over_bar']} hours over; worst {gs['worst_hour']}; "
       f"{gs['lost_s']}s lost")
      if gs["rate_estimable"] else
      f"NOT ESTIMABLE -- {gs['rate_note']}")

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

    # --- R-211(3): PER-COIN verdicts, days >= PER_COIN_RULE_FROM_DAY only ---
    # Each coin is judged on ITS OWN completeness and ITS OWN hourly gap bars,
    # and coin-days pass or fail independently -- so one coin's degraded feed
    # stops costing every other coin its day. The frozen whole-day predicates
    # above are computed EITHER WAY and stay in the artifact: the day a rule
    # changes is exactly the day both answers should be legible side by side.
    gran = verdict_granularity(day_token)
    per_coin: dict[str, Any] = {}
    if gran == "per_coin":
        for coin in coins:
            cp: list[dict[str, Any]] = []

            def cpp(name, okv, detail):
                cp.append({"predicate": name, "pass": bool(okv),
                           "detail": detail})

            # post-freeze, this coin only
            if day is None:
                cpp("entirely_post_freeze", False,
                    f"{iso} absent from the selector -- not verifiable")
            else:
                _a, _t = adm.get(coin), tot.get(coin)
                cpp("entirely_post_freeze",
                    _t is not None and _t > 0 and _a == _t,
                    f"{coin} {_a}/{_t} admissible/total"
                    + ("" if _t else "  <-- NO WINDOWS: an empty coin-day "
                                     "cannot pass on an empty expectation"))

            # completeness, this coin's own window count
            _n, _short = counts.get(coin, 0), expect - counts.get(coin, 0)
            cpp("complete_tape", expect > 0 and _short <= 0,
                f"expect {expect} ({basis}); {coin} {_n} (short {_short})"
                + ("" if expect > 0 else "  <-- NOTHING ELAPSED"))

            # gap bars computed on THIS COIN's gaps -- the substance of R-211(3)
            _gs = gap_series(lo, hi, now.timestamp(), coin=coin)
            cpp("gap_rate_under_bar",
                _gs["rate_estimable"] and _gs["n_hours_over_bar"] == 0,
                (f"{_gs['n_gaps']} {coin} gaps over {_gs['hours_elapsed']}h = "
                 f"{_gs['gaps_per_hour']}/hr vs bar {GAP_BAR_PER_HR:.0f}; "
                 f"{_gs['n_hours_over_bar']} hours over; worst "
                 f"{_gs['worst_hour']}; {_gs['lost_s']}s lost")
                if _gs["rate_estimable"] else
                f"NOT ESTIMABLE -- {_gs['rate_note']}")

            per_coin[coin] = {
                "predicates": cp,
                "all_pass": all(x["pass"] for x in cp),
                "gap_series": _gs,
                "windows_gap_affected": affected.get(coin),
            }

    # `all_pass` STAYS WHOLE-DAY-STRICT under both rules: a reader that has not
    # been updated for per-coin verdicts must not be handed a day-level True
    # because one coin survived. The independent verdicts live in `per_coin`,
    # so an un-updated reader fails safe and an updated one reads the coin it
    # actually wants (R-211(3): coin-days pass/fail independently, each coin's
    # >=5-day clock on its own passing days).
    _day_all_pass = all(x["pass"] for x in preds)
    if gran == "per_coin" and per_coin:
        _day_all_pass = _day_all_pass and all(v["all_pass"]
                                              for v in per_coin.values())

    return {
        "instrument": "da_forward_day_verify_v1",
        "verdict_granularity": gran,
        "per_coin_rule_from_day": PER_COIN_RULE_FROM_DAY,
        "granularity_note": (
            "R-211(3), declared 2026-08-27 while no 08-28 data existed. Days "
            "before the boundary are judged by the FROZEN whole-day rule; from "
            "the boundary each coin-day passes or fails on its own "
            "completeness and its own hourly gap bars. `all_pass` remains "
            "whole-day-strict under BOTH rules so an un-updated reader fails "
            "safe; per-coin verdicts are in `per_coin`."),
        "per_coin": per_coin,
        "authorised_by": "R-141(1); DA-verifies-first restored as a hard "
                         "precondition by R-153(2)",
        "day": iso, "day_token": day_token,
        "as_of_utc": now.isoformat(),
        "freeze_epoch": freeze_epoch,
        "day_closed_selector": day_closed,
        "day_closed_calendar": now.timestamp() >= hi,
        "predicates": preds,
        "all_pass": _day_all_pass,
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

    # ---- R-211(3): the rule switch is PROSPECTIVE and date-driven ---------
    ok(verdict_granularity("20260827") == "aggregate_frozen",
       "day one (08-27) is judged by the FROZEN whole-day rule -- the new rule "
       "must not reach back and re-judge the day that motivated it")
    ok(verdict_granularity("20260826") == "aggregate_frozen",
       "and neither are earlier days")
    ok(verdict_granularity("20260828") == "per_coin",
       "the per-coin rule starts at the declared boundary")
    ok(verdict_granularity("20260901") == "per_coin", "and holds after it")
    import inspect as _insp
    ok("day_token" in _insp.signature(verdict_granularity).parameters
       and len(_insp.signature(verdict_granularity).parameters) == 1,
       "granularity takes ONLY the date: a rule a caller can override is a "
       "rule that gets overridden after the numbers are visible (rule 11)")

    # ---- R-211(3) BOTH DIRECTIONS on a synthetic btc-degraded/eth-clean day
    import tempfile as _tf2
    global PM_GAPS
    _real2 = PM_GAPS
    try:
        with _tf2.TemporaryDirectory() as td2:
            f2 = Path(td2) / "g.jsonl"
            lo8, hi8 = day_bounds("20260828")
            rowsx = []
            for i in range(20):        # btc: 20 in hour 0 -> OVER the bar of 15
                rowsx.append(json.dumps({
                    "event": "gap_closed", "coin": "btc",
                    "gap_start_ns": int((lo8 + 60 + i) * 1e9),
                    "gap_end_ns": int((lo8 + 62 + i) * 1e9)}))
            for i in range(2):         # eth: 2 in hour 0 -> UNDER the bar
                rowsx.append(json.dumps({
                    "event": "gap_closed", "coin": "eth",
                    "gap_start_ns": int((lo8 + 90 + i) * 1e9),
                    "gap_end_ns": int((lo8 + 91 + i) * 1e9)}))
            f2.write_text("\n".join(rowsx), encoding="utf-8")
            PM_GAPS = f2
            _now8 = lo8 + 7200          # 2h elapsed: rate estimable

            gb = gap_series(lo8, hi8, _now8, coin="btc")
            ge = gap_series(lo8, hi8, _now8, coin="eth")
            ga = gap_series(lo8, hi8, _now8)              # frozen basis

            btc_pass = gb["rate_estimable"] and gb["n_hours_over_bar"] == 0
            eth_pass = ge["rate_estimable"] and ge["n_hours_over_bar"] == 0
            agg_pass = ga["rate_estimable"] and ga["n_hours_over_bar"] == 0

            ok(btc_pass is False,
               "R-211(3) direction 1: the degraded coin FAILS its own bar "
               f"({gb['n_gaps']} btc gaps in hour 0 vs bar 15)")
            ok(eth_pass is True,
               "R-211(3) direction 2: the CLEAN coin PASSES on the same day -- "
               f"({ge['n_gaps']} eth gaps) -- which is the entire point of the "
               "rule: one coin's dead feed stops costing every other coin")
            ok(gb["n_gaps"] == 20 and ge["n_gaps"] == 2,
               "each coin's series counts ONLY its own gaps (R-42 mirror: the "
               "two coins get different answers from one ledger)")
            ok(ga["n_gaps"] == 22 and agg_pass is False,
               "and the FROZEN whole-day basis still pools all 22 and fails -- "
               "so a pre-boundary day is judged exactly as it was before")
            ok(verdict_granularity("20260827") == "aggregate_frozen"
               and agg_pass is False,
               "R-211(3) direction 3: the SAME ledger under a pre-08-28 date "
               "is whole-day FAIL -- eth is not rescued retroactively")
    finally:
        PM_GAPS = _real2

    # the gap series must COUNT unparseable lines, never drop them (rule 4)
    import tempfile
    real = PM_GAPS          # `global PM_GAPS` is declared once, above
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
            # --- the rate denominator is ELAPSED, not gap-derived --------
            f.write_text("\n".join(good[:1]), encoding="utf-8")
            short = gap_series(lo, hi, now=lo + 30)          # 30s elapsed
            ok(short["rate_estimable"] is False
               and short["gaps_per_hour"] is None,
               "under an hour of tape NO RATE is reported -- the 08-27 arm "
               "published '1.0/hr' for one gap in 30s, understating ~120x and "
               "in the direction that reads as abatement")
            ok(short["n_gaps"] == 1 and "30s elapsed" in (short["rate_note"] or ""),
               "the raw count and the reason are reported instead")
            full = gap_series(lo, hi, now=lo + 7200)         # 2h elapsed
            ok(full["rate_estimable"] and full["gaps_per_hour"] == 0.5,
               "with 2h elapsed and 1 gap the rate is 0.5/hr, not 1.0 -- the "
               "denominator is time, not the span of the gaps themselves")
            ok(short["gaps_per_hour"] != full["gaps_per_hour"],
               "identical ledgers, different elapsed -> different answers "
               "(R-42 mirror on the denominator)")
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
    # A DAY THAT FAILS AND AN INSTRUMENT THAT BROKE MUST NOT SHARE AN EXIT
    # CODE. An uncaught exception exits 1, and so does a computed FAIL -- so
    # the nightly log could not distinguish "day one is inadmissible" from
    # "the verifier never ran". R-153(2) makes this a HARD PRECONDITION, and a
    # precondition that can silently no-op is the failure it exists to prevent.
    try:
        rep = verify_day(a.day, a.freeze_epoch)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\nINSTRUMENT FAILURE verifying {a.day}: NOTHING WAS VERIFIED. "
              f"This is exit 4, NOT a failing day -- no verdict was computed.")
        return 4
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
    if rep.get("verdict_granularity") == "per_coin" and rep.get("per_coin"):
        print("\nPER-COIN VERDICTS (R-211(3): coin-days pass/fail independently)")
        for c, v in sorted(rep["per_coin"].items()):
            print(f"  {c}: ALL PASS = {v['all_pass']}")
            for x in v["predicates"]:
                print(f"      [{'PASS' if x['pass'] else 'FAIL'}] "
                      f"{x['predicate']}: {x['detail']}")
    print(f"\nALL PASS (whole-day strict, both rules): {rep['all_pass']}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
