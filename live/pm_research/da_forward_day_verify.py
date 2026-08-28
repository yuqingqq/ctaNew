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
import math
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

#: DAY_BAR_V2 (P1/P2/P3), governing days from this date. Declared in
#: plans/DAY_BAR_V2_PREREGISTRATION.md (dfa0977, amended 368345b) BEFORE any
#: day it judges. Class A and PROSPECTIVE-ONLY, same discipline as the per-coin
#: boundary: moving it after a verdict is visible retro-judges a day under a
#: bar chosen to change its answer.
DAY_BAR_V2_FROM_DAY = "20260829"
DAY_BAR_V2_DOC = "live/pm_research/plans/DAY_BAR_V2_PREREGISTRATION.md"
DAY_BAR_V2_COMMITS = "dfa0977 (declared) + 368345b (amended pre-judgment)"
#: CLASS A thresholds, transcribed from the doc. Not to be tuned here.
P1_LOST_S_PER_HR_MAX = 120.0     # >= 3.33% coverage loss
P2_MATERIAL_SPAN_S = 75.0        # >=25% of a 300s window in gap
P2_MATERIAL_SHARE_MAX = 0.05     # <=14 of 288 windows
P3_ROLLING_60MIN_LOST_S_MAX = 900.0


def bar_regime(day_token: str) -> str:
    """Which bar judges this day. A DATE predicate, no caller override."""
    d = dt.datetime.strptime(day_token, "%Y%m%d")
    if d >= dt.datetime.strptime(DAY_BAR_V2_FROM_DAY, "%Y%m%d"):
        return "day_bar_v2"
    return "count_bar_v1_frozen"


GAP_EVENTS = ("gap_closed", "gap_open_at_exit")
KNOWN_COINS = ("btc", "eth", "sol", "xrp", "doge", "bnb", "hype")


def coin_gap_intervals(lo: int, hi: int, coin: str, path: Path | None = None,
                       diag: dict | None = None
                       ) -> list[tuple[float, float]]:
    """COIN-LEVEL gap intervals overlapping [lo, hi), merged and sorted.

    COIN-LEVEL is mandatory (R-191 scope, restated in the day-bar doc §4.2):
    a gap logged against a NEIGHBOURING window still blinds this one. The
    verifier's older per-slug figure is kept but RENAMED, because the two
    definitions differ by construction on bad days and a bar set on one and
    evaluated on the other is wrong by that difference.
    """
    src = PM_GAPS if path is None else path
    iv = []
    n_bad = 0
    n_structural_bad = 0
    synthesized_ends = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        if not isinstance(r, dict):
            n_structural_bad += 1
            continue
        ev = r.get("event")
        # (4) VALIDATE BEFORE FILTERING BY COIN. Filtering first means a
        # malformed record for another coin -- or one with no coin at all --
        # is never examined, so structural corruption in the ledger is
        # invisible to the coin whose day it silently shrinks.
        if ev in GAP_EVENTS:
            if r.get("coin") not in KNOWN_COINS:
                n_structural_bad += 1
                continue
            _gs, _ge = r.get("gap_start_ns"), r.get("gap_end_ns")
            if not isinstance(_gs, (int, float)) or not math.isfinite(_gs):
                n_structural_bad += 1
                continue
            if ev == "gap_closed":
                if not isinstance(_ge, (int, float)) or not math.isfinite(_ge):
                    n_structural_bad += 1
                    continue
                # strictly end > start: a reversed or zero-length interval is
                # CORRUPTION, and a reversed one previously produced NEGATIVE
                # lost seconds that PASSED every bar.
                if _ge <= _gs:
                    n_structural_bad += 1
                    continue
            elif _ge is not None:
                # an open-at-exit record must NOT carry an end
                n_structural_bad += 1
                continue
        # (c) gap_open_at_exit is the NEVER-RECONNECTED class -- exactly what
        # O1d exists for. Reading only gap_closed silently understates loss the
        # moment such a record appears, and says nothing while doing it.
        if ev not in GAP_EVENTS or r.get("coin") != coin:
            continue
        gs, ge = r.get("gap_start_ns"), r.get("gap_end_ns")
        if ev == "gap_open_at_exit":
            # ONLY this event may synthesise an end, and the scope end it is
            # charged to is recorded explicitly rather than left implicit.
            ge = int(hi * 1e9)
            synthesized_ends += 1
        a, b = gs / 1e9, ge / 1e9
        if b > lo and a < hi:
            iv.append((max(a, lo), min(b, hi)))
    # REFUSE a malformed ledger (doc §4.3). Skipping bad lines and returning
    # the intervals that survived is indistinguishable from a CLEAN DAY -- the
    # exact shape where an unreadable input reads as a pass. An absent file
    # already raises; an unreadable one must too, and say how much it lost.
    if n_structural_bad:
        raise ValueError(
            f"REFUSED: {n_structural_bad} gap record(s) in {src.name} lack a "
            f"usable interval. Dropping them silently shrinks measured loss "
            f"with no trace -- exclusions are statuses, never silent drops.")
    if n_bad:
        raise ValueError(
            f"REFUSED: {n_bad} unparseable line(s) in {src.name}. A bar computed "
            f"over the lines that happened to parse is not a bar over the day -- "
            f"and zero intervals from a broken ledger reads exactly like a day "
            f"with no gaps.")
    if diag is not None:
        diag["synthesized_ends_charged_to_scope_end"] = synthesized_ends
        diag["scope_end_utc"] = hi
    iv.sort()
    out: list[tuple[float, float]] = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


#: R-240: the freeze epoch governs RACE-CLOCK ACCRUAL, not feed health. Those
#: are two different questions and `entirely_post_freeze` answers only the
#: second: a day can be perfectly healthy and still not count, because the
#: candidate's clock had not started. Reporting one number for both makes a
#: good-but-early day read as a bad day.
ACCRUAL_PREDICATE = "entirely_post_freeze"

#: (2) The RAW COUNT bar is SUPERSEDED on day_bar_v2 days. The pre-registration
#: makes raw count a REPORTED DIAGNOSTIC WITH NO BAR -- O1 removes detection lag
#: without reducing disconnects, so a post-fix day keeps failing the count while
#: the actual harm falls ~4x. Leaving it in the composition let the superseded
#: bar VETO a day that passed P1/P2/P3: the fix reads as ineffective when it
#: worked, which is the precise failure the v2 bars were designed to prevent.
#: The FIELDS stay -- only the veto is removed.
SUPERSEDED_ON_V2 = ("gap_rate_under_bar",)


def governing_predicates(preds: list, regime: str) -> list:
    """Predicates that GOVERN the verdict under this regime. Callable."""
    if regime != "day_bar_v2":
        return list(preds)
    return [x for x in preds if x["predicate"] not in SUPERSEDED_ON_V2]


def split_verdict(preds: list, regime: str = "count_bar_v1_frozen") -> dict:
    """Separate DAY QUALITY (feed health) from RACE ACCRUAL (eligibility).

    CALLABLE, so the split can be driven directly rather than inferred from a
    composed boolean -- the lesson from all_pass being computed before the bars
    were appended.
    """
    gov = governing_predicates(preds, regime)
    quality = [x for x in gov if x["predicate"] != ACCRUAL_PREDICATE]
    accrual = [x for x in gov if x["predicate"] == ACCRUAL_PREDICATE]
    q_ok = bool(quality) and all(x["pass"] for x in quality)
    a_ok = bool(accrual) and all(x["pass"] for x in accrual)
    return {
        "day_quality_pass": q_ok,
        "post_freeze_pass": a_ok,
        # a day accrues only if it is BOTH healthy and after the clock started
        "race_accrual_eligible": q_ok and a_ok,
        "why": "feed health and clock eligibility are separate questions; a "
               "healthy day BEFORE the freeze commit is a good day that does "
               "not count, not a bad day",
    }


def compose_all_pass(preds: list, per_coin: dict, bars_v2: dict,
                     regime: str) -> bool:
    """The day verdict, composed from EVERY input that governs it.

    CALLABLE on purpose. The previous version computed all_pass inline BEFORE
    the P1/P2/P3 predicates were appended, so stubbing the bars all-pass or
    all-fail left the verdict IDENTICAL -- recorded, not enforced -- and no
    test could drive the composition to show it. A verdict rule that cannot be
    called cannot be falsified.
    """
    gov = governing_predicates(preds, regime)
    ok = all(x["pass"] for x in gov)
    if per_coin:
        ok = ok and all(v["all_pass"] for v in per_coin.values())
    if regime == "day_bar_v2" and bars_v2:
        ok = ok and all(bool(b.get("P1_pass")) and bool(b.get("P2_pass"))
                        and bool(b.get("P3_pass")) for b in bars_v2.values())
    return ok


def day_bar_v2(lo: int, hi: int, coin: str, elapsed_h: float,
               path: Path | None = None,
               coverage_observed: bool | None = None) -> dict[str, Any]:
    """P1/P2/P3 for one coin-day, from COIN-LEVEL merged gap intervals."""
    iv = coin_gap_intervals(lo, hi, coin, path)
    lost = sum(b - a for a, b in iv)
    # P1 severity: the doc divides by 24, so a PARTIAL day is not comparable.
    p1_rate = lost / 24.0
    # P2 materiality: windows with >=75s of their 300s span intersected
    mat = 0
    for i in range(WINDOWS_PER_DAY):
        w0 = lo + i * WINDOW_S
        w1 = w0 + WINDOW_S
        cov = sum(min(b, w1) - max(a, w0) for a, b in iv if a < w1 and b > w0)
        if cov >= P2_MATERIAL_SPAN_S:
            mat += 1
    # P3 concentration: the EXACT maximum over ALL rolling-hour placements.
    #
    # This previously stepped only 300s-ALIGNED starts, which is not the
    # declared statistic. Codex's executed counterexample: gaps [+100,+600] and
    # [+3200,+3700]; the exact window [+100,+3700] holds 1000s and must FAIL,
    # while aligned stepping reports 900 and PASSES at the <=900 boundary.
    #
    # Coverage of a fixed-width window is piecewise-linear in its start, so the
    # maximum is attained at a breakpoint: a window STARTING at an interval
    # start, or ENDING at an interval end (start = end - 3600), plus the day
    # bounds. Enumerating those candidates is exact, not a finer grid.
    worst = 0.0
    cands = {float(lo)}
    for a, b in iv:
        cands.add(a)
        cands.add(b - 3600.0)
    for h0 in sorted(cands):
        h0 = min(max(h0, float(lo)), float(hi) - 3600.0)
        if h0 < lo or h0 + 3600.0 > hi:
            continue
        h1 = h0 + 3600.0
        worst = max(worst, sum(min(b, h1) - max(a, h0)
                               for a, b in iv if a < h1 and b > h0))
    # NOT-YET-EVALUABLE is not a PASS. A day that has not started has no gaps,
    # so all three bars would read "pass" on zero data -- the empty-set trap
    # this programme has paid for more than once. Elapsed time is P1's
    # denominator and P2/P3's observation window, so below an hour there is
    # nothing to judge and the bars say so instead of passing.
    # (b) AN EMPTY LEDGER IS NOT A FLAWLESS DAY. The not-yet-started guard
    # below covers a day that has not begun; this covers the day that ELAPSED
    # and produced NOTHING -- silence from a dead collector is byte-identical to
    # silence from a perfect one. The bars may only be read where the day is
    # independently known to have been OBSERVED (the tape's own completeness),
    # so "no gaps" means "none happened" rather than "none were recorded".
    # (3) ONLY `is True` EVALUATES. This was `is False`, so the DEFAULT None --
    # i.e. a caller that never supplied evidence -- sailed through as
    # observed-without-evidence. That is the N/A-vacuity class, in the very
    # guard added to close the empty-ledger hole: absence of evidence read as
    # evidence of coverage.
    if coverage_observed is not True:
        return {
            "evaluable": False, "hours_elapsed": round(elapsed_h, 3),
            "coin_level_gap_intervals": len(iv), "lost_seconds": round(lost, 1),
            "P1_pass": False, "P2_pass": False, "P3_pass": False,
            "coverage_observed_arg": repr(coverage_observed),
            "why": "coverage evidence was not AFFIRMATIVELY supplied (only "
                   "coverage_observed is True evaluates; False, None, omitted "
                   "and malformed all refuse). The day is NOT INDEPENDENTLY "
                   "OBSERVED (tape coverage "
                   "absent/short), so an empty gap ledger cannot be read as a "
                   "clean day: silence from a dead collector and silence from a "
                   "perfect one are the same bytes",
            "thresholds": {"P1_max_s_per_hr": P1_LOST_S_PER_HR_MAX,
                           "P2_material_span_s": P2_MATERIAL_SPAN_S,
                           "P2_max_share": P2_MATERIAL_SHARE_MAX,
                           "P3_max_rolling_60min_s": P3_ROLLING_60MIN_LOST_S_MAX}}
    if elapsed_h < 1.0:
        return {
            "evaluable": False, "hours_elapsed": round(elapsed_h, 3),
            "coin_level_gap_intervals": len(iv), "lost_seconds": round(lost, 1),
            "P1_pass": False, "P2_pass": False, "P3_pass": False,
            "why": f"only {elapsed_h:.3f}h elapsed: NOT YET EVALUABLE, and "
                   f"zero gaps on a day that has not happened is not a clean day",
            "thresholds": {"P1_max_s_per_hr": P1_LOST_S_PER_HR_MAX,
                           "P2_material_span_s": P2_MATERIAL_SPAN_S,
                           "P2_max_share": P2_MATERIAL_SHARE_MAX,
                           "P3_max_rolling_60min_s": P3_ROLLING_60MIN_LOST_S_MAX}}
    return {
        "evaluable": True, "hours_elapsed": round(elapsed_h, 3),
        "coin_level_gap_intervals": len(iv),
        "lost_seconds": round(lost, 1),
        "P1_lost_s_per_hr": round(p1_rate, 2),
        "P1_pass": p1_rate <= P1_LOST_S_PER_HR_MAX,
        "P2_material_windows": mat,
        "P2_material_share": round(mat / WINDOWS_PER_DAY, 4),
        "P2_pass": (mat / WINDOWS_PER_DAY) <= P2_MATERIAL_SHARE_MAX,
        "P3_worst_rolling_60min_lost_s": round(worst, 1),
        "P3_pass": worst <= P3_ROLLING_60MIN_LOST_S_MAX,
        "thresholds": {"P1_max_s_per_hr": P1_LOST_S_PER_HR_MAX,
                       "P2_material_span_s": P2_MATERIAL_SPAN_S,
                       "P2_max_share": P2_MATERIAL_SHARE_MAX,
                       "P3_max_rolling_60min_s": P3_ROLLING_60MIN_LOST_S_MAX},
    }


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
        # DUAL-REPORTED (day-bar doc §4.2). The historic field was PER-SLUG;
        # the governing scope is COIN-LEVEL. They differ by construction on bad
        # days -- a gap logged against a neighbouring window still blinds this
        # one -- so a bar set on one and evaluated on the other is wrong by that
        # difference. Both are named for what they are; neither is "the" number.
        _iv = coin_gap_intervals(lo, hi, coin)
        _cl = sum(1 for i in range(WINDOWS_PER_DAY)
                  if any(a < lo + i * WINDOW_S + WINDOW_S and b > lo + i * WINDOW_S
                         for a, b in _iv))
        affected[coin] = {
            "era_covered_windows": tot_w,
            "gap_affected_PER_SLUG": aff,
            "gap_affected_pct_PER_SLUG": round(100.0 * aff / tot_w, 1) if tot_w else None,
            "gap_affected_COIN_LEVEL": _cl,
            "gap_affected_pct_COIN_LEVEL": round(100.0 * _cl / WINDOWS_PER_DAY, 1),
            "scope_note": "COIN_LEVEL is the governing scope (R-191); PER_SLUG "
                          "is retained for continuity and is NOT the bar's "
                          "input. Raw breadth is a REPORTED DIAGNOSTIC with no "
                          "bar -- O1 barely moves it, so a bar here would "
                          "reject good post-fix days forever."}

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

            _gov_cp = governing_predicates(cp, bar_regime(day_token))
            per_coin[coin] = {
                "predicates": cp,
                "governing_predicates": [x["predicate"] for x in _gov_cp],
                "superseded_not_governing": [x["predicate"] for x in cp
                                             if x not in _gov_cp],
                "all_pass": all(x["pass"] for x in _gov_cp),
                "gap_series": _gs,
                "windows_gap_affected": affected.get(coin),
            }

    # `all_pass` STAYS WHOLE-DAY-STRICT under both rules: a reader that has not
    # been updated for per-coin verdicts must not be handed a day-level True
    # because one coin survived. The independent verdicts live in `per_coin`,
    # so an un-updated reader fails safe and an updated one reads the coin it
    # actually wants (R-211(3): coin-days pass/fail independently, each coin's
    # >=5-day clock on its own passing days).

    regime = bar_regime(day_token)
    bars_v2: dict[str, Any] = {}
    if regime == "day_bar_v2":
        for coin in coins:
            _cov = counts.get(coin, 0) > 0 and short.get(coin, 1) <= 0
            b = day_bar_v2(lo, hi, coin, gs["hours_elapsed"],
                           coverage_observed=_cov)
            b["all_pass"] = bool(b["P1_pass"] and b["P2_pass"] and b["P3_pass"])
            b["partial_day"] = not calendar_closed
            if not calendar_closed:
                b["note"] = ("P1 divides lost seconds by 24 as declared, so on "
                             "an OPEN day it is a LOWER BOUND, not the day's "
                             "rate. The closing verdict is the one that judges.")
            bars_v2[coin] = b
            for k in ("P1", "P2", "P3"):
                if not b.get("evaluable"):
                    p(f"{k}_{coin}", False, b.get("why", "not evaluable"))
                    continue
                p(f"{k}_{coin}", b[f"{k}_pass"],
                  {"P1": f"lost {b['lost_seconds']}s = {b['P1_lost_s_per_hr']}/hr "
                         f"vs bar {P1_LOST_S_PER_HR_MAX}",
                   "P2": f"{b['P2_material_windows']} windows >= "
                         f"{P2_MATERIAL_SPAN_S}s in gap = "
                         f"{100*b['P2_material_share']:.2f}% vs bar "
                         f"{100*P2_MATERIAL_SHARE_MAX:.0f}%",
                   "P3": f"worst rolling 60min {b['P3_worst_rolling_60min_lost_s']}s "
                         f"vs bar {P3_ROLLING_60MIN_LOST_S_MAX}"}[k])

    # (a) COMPUTED AFTER every predicate is appended, including P1/P2/P3.
    # It used to be computed BEFORE the bars were added, so stubbing them
    # all-pass or all-fail left all_pass IDENTICAL: the bars were recorded and
    # did not govern. Recomputing here from the final table is the same
    # discipline as the tape verdict's all_pass -- derive it from what is
    # actually in the table, never from a snapshot taken earlier.
    _day_all_pass = compose_all_pass(
        preds, per_coin if gran == "per_coin" else {}, bars_v2, regime)

    _split = split_verdict(preds, regime)

    return {
        "instrument": "da_forward_day_verify_v1",
        "verdict_split": _split,
        "race_accrual_eligible": _split["race_accrual_eligible"],
        "bar_regime": regime,
        "day_bar_v2": bars_v2,
        "day_bar_v2_governing": {
            "from_day": DAY_BAR_V2_FROM_DAY, "doc": DAY_BAR_V2_DOC,
            "commits": DAY_BAR_V2_COMMITS,
            "applies_to_this_day": regime == "day_bar_v2",
            "note": "days before the boundary are judged by the FROZEN count "
                    "bar; the v2 bars are not applied retroactively"},
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

    # ---- (b) an ELAPSED but UNOBSERVED day must not pass -----------------
    import tempfile as _tfb
    _lo, _hi = day_bounds("20260829")
    with _tfb.TemporaryDirectory() as _td:
        _e = Path(_td) / "empty.jsonl"; _e.write_text("", encoding="utf-8")
        _unobs = day_bar_v2(_lo, _hi, "btc", 24.0, _e, coverage_observed=False)
        ok(_unobs["evaluable"] is False and not _unobs["P1_pass"],
           "(b) a FULLY ELAPSED day with an EMPTY ledger and NO observed "
           "coverage does NOT pass -- silence from a dead collector and "
           "silence from a perfect one are the same bytes")
        _obs = day_bar_v2(_lo, _hi, "btc", 24.0, _e, coverage_observed=True)
        ok(_obs["evaluable"] is True and _obs["P1_pass"],
           "and the SAME empty ledger DOES pass when the day is independently "
           "observed -- otherwise the guard would reject every clean day")

        # ---- (c) open gaps counted; structurally bad rows REFUSE -----------
        _open = Path(_td) / "open.jsonl"
        _open.write_text(json.dumps({
            "event": "gap_open_at_exit", "coin": "btc", "slug": "s",
            "gap_start_ns": int((_lo + 3600) * 1e9)}), encoding="utf-8")
        _r = day_bar_v2(_lo, _hi, "btc", 24.0, _open, coverage_observed=True)
        ok(_r["lost_seconds"] > 0 and not _r["P1_pass"],
           "(c) a gap_open_at_exit record (the NEVER-RECONNECTED class O1d "
           "creates) is CHARGED to the scope end, not ignored -- reading only "
           "gap_closed understates loss the moment that record type appears")
        _bad = Path(_td) / "structurally_bad.jsonl"
        _bad.write_text(json.dumps({"event": "gap_closed", "coin": "btc",
                                    "slug": "s"}), encoding="utf-8")
        try:
            day_bar_v2(_lo, _hi, "btc", 24.0, _bad, coverage_observed=True)
            ok(False, "(c) a gap record with no usable interval must REFUSE")
        except ValueError as _ex:
            ok("no trace" in str(_ex) or "usable interval" in str(_ex),
               "(c) a structurally bad gap record REFUSES instead of being "
               "silently dropped -- a silent drop shrinks measured loss with "
               "no trace (rule 4)")

    # ---- (1) SEAM: what the LAUNCHER actually passes, read from ARGV -----
    # The entry-point boundary, mechanized. Every one of the earlier findings
    # was a consumer I never exercised; asserting the launcher's own argv is
    # the check that would have caught the stale epoch without a reviewer.
    _sh = Path(__file__).resolve().parent / "da_midnight_verify.sh"
    if _sh.exists():
        import subprocess as _sp, tempfile as _tfl
        with _tfl.TemporaryDirectory() as _ltd:
            _spy = Path(_ltd) / "spy.py"
            _spy.write_text(
                "import sys, json, pathlib\n"
                "pathlib.Path(%r).write_text(json.dumps(sys.argv))\n"
                % str(Path(_ltd) / "argv.json"), encoding="utf-8")
            _run = Path(_ltd) / "run.sh"
            _run.write_text(
                _sh.read_text(encoding="utf-8")
                   .replace('V=/home/yuqing/ctaNew/live/pm_research/'
                            'da_forward_day_verify.py', f'V={_spy}')
                   .replace('LOG="${DA_MIDNIGHT_LOG:-'
                            '/home/yuqing/ctaNew/data/pm_5min/derived/'
                            '.da_midnight_verify.log}"',
                            f'LOG="{Path(_ltd) / "log"}"'),
                encoding="utf-8")
            _run.chmod(0o755)
            _sp.run(["bash", str(_run)], capture_output=True,
                    env={"PATH": "/usr/bin:/bin",
                         "DA_MIDNIGHT_LOG": str(Path(_ltd) / "log")})
            _argv = json.loads((Path(_ltd) / "argv.json").read_text())
            ok("--freeze-epoch" in _argv,
               "(1) the LAUNCHER passes --freeze-epoch explicitly (no default "
               "exists any more, so an omission would refuse at 00:06Z)")
            _ep = _argv[_argv.index("--freeze-epoch") + 1]
            ok(abs(float(_ep) - 1787897340.0) < 1.0,
               f"(1) and it passes the RULED freeze-commit epoch "
               f"(got {_ep}, want 1787897340 = b3f7f9f) -- read from the "
               f"launcher's own ARGV, not from reading the script")

    # ---- Codex re-review blockers, each RED-FIRST ------------------------
    import tempfile as _tfc
    _lo9, _hi9 = day_bounds("20260829")

    def _wr(td, recs, name="g.jsonl"):
        f = Path(td) / name
        f.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
        return f

    # (2) the SUPERSEDED count bar must not veto a v2 day
    _legacy_fail = [{"predicate": "complete_tape", "pass": True},
                    {"predicate": "gap_rate_under_bar", "pass": False},
                    {"predicate": ACCRUAL_PREDICATE, "pass": True}]
    _bars_ok = {"btc": {"P1_pass": True, "P2_pass": True, "P3_pass": True}}
    ok(compose_all_pass(_legacy_fail, {}, _bars_ok, "day_bar_v2") is True,
       "(2) a v2 day PASSES with P1-P3 green even though the SUPERSEDED raw "
       "count bar fails -- O1 removes detection lag without reducing "
       "disconnects, so a superseded veto makes a working fix read as failed")
    ok(compose_all_pass(_legacy_fail, {}, {}, "count_bar_v1_frozen") is False,
       "(2) and the same table still FAILS under the frozen regime -- "
       "supersession is scoped to the days v2 governs, not retroactive")

    with _tfc.TemporaryDirectory() as _td:
        # (5) Codex's executed counterexample, verbatim
        _ce = _wr(_td, [{"event": "gap_closed", "coin": "btc", "slug": "s",
                         "gap_start_ns": int((_lo9 + 100) * 1e9),
                         "gap_end_ns": int((_lo9 + 600) * 1e9)},
                        {"event": "gap_closed", "coin": "btc", "slug": "s",
                         "gap_start_ns": int((_lo9 + 3200) * 1e9),
                         "gap_end_ns": int((_lo9 + 3700) * 1e9)}], "ce.jsonl")
        _r5 = day_bar_v2(_lo9, _hi9, "btc", 24.0, _ce, coverage_observed=True)
        ok(abs(_r5["P3_worst_rolling_60min_lost_s"] - 1000.0) < 1e-6
           and _r5["P3_pass"] is False,
           "(5) Codex counterexample: the EXACT rolling hour [+100,+3700] holds "
           "1000s and FAILS. 300s-aligned stepping reported 900 and passed at "
           "the boundary -- a grid is not the declared maximum")
        _r5b = day_bar_v2(_lo9, _hi9, "btc", 24.0,
                          _wr(_td, [{"event": "gap_closed", "coin": "btc",
                                     "slug": "s",
                                     "gap_start_ns": int((_lo9 + 100) * 1e9),
                                     "gap_end_ns": int((_lo9 + 600) * 1e9)}],
                              "one.jsonl"), coverage_observed=True)
        ok(_r5b["P3_pass"] is True and _r5b["P3_worst_rolling_60min_lost_s"] == 500.0,
           "(5) positive control: a single 500s gap reports exactly 500s and "
           "passes -- the exact maximum does not inflate a quiet day")

        # (3) only is-True evaluates
        _e = _wr(_td, [], "empty.jsonl")
        for _cov, _lbl in ((None, "OMITTED/None"), (False, "False"),
                           ("yes", "malformed truthy string")):
            _rc = day_bar_v2(_lo9, _hi9, "btc", 24.0, _e, coverage_observed=_cov)
            ok(_rc["evaluable"] is False and not _rc["P1_pass"],
               f"(3) coverage_observed={_lbl} REFUSES -- only an affirmative "
               f"True evaluates; absence of evidence is not evidence")
        ok(day_bar_v2(_lo9, _hi9, "btc", 24.0, _e,
                      coverage_observed=True)["evaluable"] is True,
           "(3) positive control: explicit True still evaluates")

        # (4) structural validation, before coin filtering
        _bad_cases = [
            ([{"event": "gap_closed", "coin": "btc", "slug": "s",
               "gap_start_ns": int((_lo9 + 200) * 1e9),
               "gap_end_ns": int((_lo9 + 150) * 1e9)}],
             "a REVERSED interval (previously lost_seconds=-50 and PASSED)"),
            ([{"event": "gap_closed", "slug": "s",
               "gap_start_ns": int((_lo9 + 10) * 1e9),
               "gap_end_ns": int((_lo9 + 20) * 1e9)}],
             "a record with NO coin (previously ignored silently)"),
            ([{"event": "gap_closed", "coin": "btc", "slug": "s",
               "gap_start_ns": float("inf"),
               "gap_end_ns": int((_lo9 + 20) * 1e9)}],
             "a NON-FINITE stamp"),
            ([{"event": "gap_open_at_exit", "coin": "btc", "slug": "s",
               "gap_start_ns": int((_lo9 + 10) * 1e9),
               "gap_end_ns": int((_lo9 + 20) * 1e9)}],
             "an open-at-exit record that CARRIES an end"),
        ]
        for _i, (_recs, _lbl) in enumerate(_bad_cases):
            try:
                day_bar_v2(_lo9, _hi9, "btc", 24.0,
                           _wr(_td, _recs, f"bad{_i}.jsonl"),
                           coverage_observed=True)
                ok(False, f"(4) {_lbl} must REFUSE")
            except ValueError:
                ok(True, f"(4) {_lbl} REFUSES -- validated BEFORE coin "
                         f"filtering, so corruption cannot hide behind another "
                         f"coin's records")
        # positive control + explicit scope end for the ONLY synthesising event
        _diag: dict = {}
        _iv = coin_gap_intervals(_lo9, _hi9, "btc",
                                 _wr(_td, [{"event": "gap_open_at_exit",
                                            "coin": "btc", "slug": "s",
                                            "gap_start_ns": int((_lo9 + 10) * 1e9)}],
                                     "open.jsonl"), diag=_diag)
        ok(_diag.get("synthesized_ends_charged_to_scope_end") == 1
           and _diag.get("scope_end_utc") == _hi9,
           "(4) gap_open_at_exit is the ONLY event that may synthesise an end, "
           "and the scope end it is charged to is RECORDED, not implicit")

    # ---- R-240: the epoch governs ACCRUAL, not feed health ---------------
    _healthy_early = [{"predicate": "complete_tape", "pass": True},
                      {"predicate": "gap_rate_under_bar", "pass": True},
                      {"predicate": ACCRUAL_PREDICATE, "pass": False}]
    _sv = split_verdict(_healthy_early)
    ok(_sv["day_quality_pass"] is True and _sv["race_accrual_eligible"] is False,
       "a day STRADDLING the freeze is day-quality GOOD but does NOT accrue -- "
       "the epoch's whole job, and reporting one number for both would make a "
       "good-but-early day read as a bad day")
    _healthy_late = [dict(x, **({"pass": True} if x["predicate"] == ACCRUAL_PREDICATE else {}))
                     for x in _healthy_early]
    ok(split_verdict(_healthy_late)["race_accrual_eligible"] is True,
       "a healthy day ENTIRELY AFTER the freeze DOES accrue (positive control)")
    _sick_late = [{"predicate": "gap_rate_under_bar", "pass": False},
                  {"predicate": ACCRUAL_PREDICATE, "pass": True}]
    ok(split_verdict(_sick_late)["race_accrual_eligible"] is False,
       "and an UNHEALTHY day after the freeze does not accrue either -- "
       "accrual needs both halves")

    # ---- (a) the bars must GOVERN the verdict, isolated ------------------
    _clean = [{"predicate": "x", "pass": True}, {"predicate": "y", "pass": True}]
    _pass_bar = {"btc": {"P1_pass": True, "P2_pass": True, "P3_pass": True}}
    _fail_bar = {"btc": {"P1_pass": True, "P2_pass": False, "P3_pass": True}}
    ok(compose_all_pass(_clean, {}, _pass_bar, "day_bar_v2") is True,
       "clean table + passing bars -> verdict PASSES (positive control)")
    ok(compose_all_pass(_clean, {}, _fail_bar, "day_bar_v2") is False,
       "clean table + ONE FAILING BAR -> verdict FAILS. This is the isolated "
       "governance link: before the fix all_pass was computed BEFORE the bars "
       "were appended, so it was identical whether every bar passed or failed")
    ok(compose_all_pass([{"predicate": "x", "pass": False}], {}, _pass_bar,
                        "day_bar_v2") is False,
       "and a failing ordinary predicate still fails the day")
    ok(compose_all_pass(_clean, {}, _fail_bar, "count_bar_v1_frozen") is True,
       "a FAILING bar does NOT fail a day the v2 regime does not govern -- the "
       "bars are not applied retroactively")

    # ---- DAY-BAR V2 falsifiers (doc §4.3): each bar must FIRE ------------
    import tempfile as _tf3
    lo9, hi9 = day_bounds("20260829")
    def _ledger(td, spans):
        f = Path(td) / "g.jsonl"
        f.write_text("\n".join(json.dumps({
            "event": "gap_closed", "coin": "btc", "slug": f"btc-{i}",
            "gap_start_ns": int(a * 1e9), "gap_end_ns": int(b * 1e9)})
            for i, (a, b) in enumerate(spans)), encoding="utf-8")
        return f

    ok(bar_regime("20260828") == "count_bar_v1_frozen"
       and bar_regime("20260829") == "day_bar_v2",
       "day-bar v2 governs from 20260829 and NOT before -- the bar is not "
       "applied to days that closed under the frozen count bar")

    with _tf3.TemporaryDirectory() as td:
        # HIGH-LOSS day: many short gaps, no single window materially hit.
        # 4000s over the day = 166.7 s/hr -> P1 must FAIL, P2 must still pass.
        hi_loss = [(lo9 + 300 * i + 10, lo9 + 300 * i + 10 + 20) for i in range(200)]
        r = day_bar_v2(lo9, hi9, "btc", 24.0, _ledger(td, hi_loss), coverage_observed=True)
        ok(not r["P1_pass"], f"synthetic HIGH-LOSS day FAILS P1 "
                             f"({r['P1_lost_s_per_hr']}/hr vs {P1_LOST_S_PER_HR_MAX})")
        ok(r["P2_pass"], "and still PASSES P2 -- severity and materiality are "
                         "different questions, which is why both exist")

        # LONG-OUTAGE day: one 90-minute outage. P2 and P3 must BOTH fail.
        out = [(lo9 + 3600 * 5, lo9 + 3600 * 5 + 5400)]
        r2 = day_bar_v2(lo9, hi9, "btc", 24.0, _ledger(td, out), coverage_observed=True)
        ok(not r2["P2_pass"], f"synthetic LONG-OUTAGE day FAILS P2 "
                              f"({r2['P2_material_windows']} material windows)")
        ok(not r2["P3_pass"], f"and FAILS P3 "
                              f"({r2['P3_worst_rolling_60min_lost_s']}s in an hour)")
        ok(not r2["P1_pass"] or r2["P1_lost_s_per_hr"] <= P1_LOST_S_PER_HR_MAX,
           "P1 on that day is reported either way -- the bars are independent")

        # POSITIVE CONTROL: a quiet day must pass all three, or the bars are
        # unfalsifiable in the direction that matters for accepting a day.
        quiet = [(lo9 + 3600 * i + 5, lo9 + 3600 * i + 15) for i in range(24)]
        r3 = day_bar_v2(lo9, hi9, "btc", 24.0, _ledger(td, quiet), coverage_observed=True)
        ok(r3["P1_pass"] and r3["P2_pass"] and r3["P3_pass"],
           "positive control: a QUIET day passes all three bars")

        # a day that has NOT HAPPENED must not pass on zero data
        _fut = day_bar_v2(lo9, hi9, "btc", 0.0, _ledger(td, quiet), coverage_observed=True)
        ok(_fut["evaluable"] is False and not _fut["P1_pass"]
           and not _fut["P2_pass"] and not _fut["P3_pass"],
           "a day with nothing elapsed is NOT-YET-EVALUABLE and does NOT pass "
           "-- zero elapsed hours is not a clean day, and the same bars that "
           "pass a quiet day must refuse an empty one")

        # MALFORMED ledger must REFUSE, never silently read as a clean day
        bad = Path(td) / "bad.jsonl"
        bad.write_text('{"event":"gap_closed","coin":"btc"\nnot json\n', encoding="utf-8")
        try:
            day_bar_v2(lo9, hi9, "btc", 24.0, bad, coverage_observed=True)
            ok(False, "a MALFORMED ledger must REFUSE, not read as a clean day")
        except ValueError as e:
            ok("unparseable" in str(e),
               "a MALFORMED ledger REFUSES and names the count -- zero intervals "
               "from a broken ledger would otherwise read exactly like a day "
               "with no gaps, which is a pass obtained by failing to read")
        miss = Path(td) / "absent.jsonl"
        try:
            day_bar_v2(lo9, hi9, "btc", 24.0, miss)
            ok(False, "an ABSENT ledger must raise, not read as zero loss")
        except FileNotFoundError:
            ok(True, "an ABSENT ledger REFUSES (FileNotFoundError), never "
                     "reading as a day with no gaps")

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
    # (e) NO SILENT DEFAULT. The old default (1787583868.0 = 2026-08-24T15:04Z)
    # was 3.63 days stale against the live freeze commit b3f7f9f
    # (2026-08-28T06:09Z), whose own receipt says the clock STARTS AT THE FREEZE
    # COMMIT -- so pre-freeze days passed entirely_post_freeze and could count
    # toward a clock that had not started. Which epoch governs is a RULING, not
    # a default: the launcher must state it, and an unstated one refuses.
    ap.add_argument("--freeze-epoch", type=float, default=None,
                    help="REQUIRED. The governing freeze epoch; there is no "
                         "default, because a stale one judged days silently.")
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
    # (e) checked BEFORE the try: an unstated epoch is a launcher error, not an
    # instrument failure, and must not be reported as one.
    if a.freeze_epoch is None:
        raise SystemExit(
            "REFUSED: --freeze-epoch is required and has no default. It "
            "selects which days count as post-freeze, so a stale value counts "
            "days toward a clock that had not started. State the governing "
            "epoch explicitly.")
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
        # (d) the dual-report rename broke this line and the CLI raised
        # KeyError. The library was validated against the doc's table and the
        # ENTRY POINT was never run -- the consumer half of the same split.
        print(f"  {c}: COIN-LEVEL {v.get('gap_affected_COIN_LEVEL')}/288 "
              f"({v.get('gap_affected_pct_COIN_LEVEL')}%)  |  per-slug "
              f"{v.get('gap_affected_PER_SLUG')}/{v.get('era_covered_windows')} "
              f"({v.get('gap_affected_pct_PER_SLUG')}%)")
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
