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
import re
import statistics
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
CANONICAL_VERDICT_DIR = Path(
    "/home/yuqing/ctaNew/data/pm_5min/derived")  # see --write-reason
MAX_CATCHUP_DAYS = 32          # declared; see days_needing_verdict


def _artifact_closed(path: Path) -> bool | None:
    """True if this artifact was written when its day was already CLOSED.

    None means the file exists but cannot be trusted to answer (unreadable,
    unparseable, wrong day, or the field absent). None is NOT False and NOT
    True: it means RE-VERDICT, because a file that cannot say whether it
    judged a closed day is not evidence that it did.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict) or "day_closed_calendar" not in d:
        return None
    v = d["day_closed_calendar"]
    return True if v is True else (False if v is False else None)


def days_needing_verdict(outdir: Path, closed_token: str,
                         opened_token: str) -> dict[str, Any]:
    """Which days tonight's run must verdict.

    THE LIMITATION THIS CLOSES (R-255(4)). The launcher verified exactly
    `date -u -d yesterday` and `date -u`, RELATIVE TO RUN TIME. A single missed
    night was recovered by Persistent=true; an outage of two or more days
    permanently lost the earlier ones, because the catch-up fires once and its
    "yesterday" is the wrong day. The day list is now DERIVED FROM DISK.

    THE FLOOR IS DERIVED, NOT HARDCODED -- and deriving it wrongly is the
    whole risk here. The obvious rule, "closed days that have tape but no
    verdict", would tonight mint SIX retroactive verdicts (08-20..08-25): the
    gap ledger reaches back to 08-20 while verdicts begin at 08-26, and those
    earlier days are CONSUMED (rule 11). So the floor is the EARLIEST DAY THAT
    ALREADY HAS A VERDICT: catch-up fills holes INSIDE the range we have been
    verdicting and can never invent a backlog behind it. With no verdicts at
    all -- a fresh install -- it falls back to closed+opened rather than
    verdicting all of history.

    A day inside the range needs verdicting when it has no artifact, or when
    its artifact was written while the day was still OPEN. That second case is
    not hypothetical: tonight 08-28's artifact reads
    `day_closed_calendar=False`, because it was last written at 10:48Z with the
    day still running. Without it, a missed night would leave a PARTIAL verdict
    standing as a closed day's final record.

    Tape is deliberately NOT consulted. A day inside the range with no tape
    verdicts as a `complete_tape` FAILURE, which is the correct and informative
    outcome -- skipping it would be a silent drop (rule 4).
    """
    outdir = Path(outdir)
    have: dict[str, bool | None] = {}
    for f in sorted(outdir.glob("da_dayverdict_*.json")):
        tok = f.name[len("da_dayverdict_"):-len(".json")]
        if len(tok) == 8 and tok.isdigit():
            have[tok] = _artifact_closed(f)

    base = [(closed_token, "closed_today"), (opened_token, "open_today")]
    if not have:
        return {"days": base, "floor": None, "truncated": [],
                "why": "no verdict artifacts exist: falling back to "
                       "closed+opened rather than minting a backlog"}

    floor = min(have)
    d0 = dt.datetime.strptime(floor, "%Y%m%d")
    d1 = dt.datetime.strptime(closed_token, "%Y%m%d")
    catchup: list[tuple[str, str]] = []
    d = d0
    while d <= d1:
        tok = d.strftime("%Y%m%d")
        if tok not in (closed_token, opened_token):
            st = have.get(tok, "absent")
            if st is not True:
                catchup.append((tok, "catchup_absent" if st == "absent"
                                else ("catchup_unreadable" if st is None
                                      else "catchup_was_open")))
        d += dt.timedelta(days=1)

    truncated: list[str] = []
    if len(catchup) > MAX_CATCHUP_DAYS:
        # NEVER a silent cap. The dropped days are named in the result and the
        # launcher logs them, because a bounded run that reads as complete is
        # the failure mode a cap introduces.
        truncated = [t for t, _ in catchup[:-MAX_CATCHUP_DAYS]]
        catchup = catchup[-MAX_CATCHUP_DAYS:]

    seen, days = set(), []
    for tok, kind in sorted(catchup + base):
        if tok not in seen:
            seen.add(tok)
            days.append((tok, kind))
    return {"days": days, "floor": floor, "truncated": truncated,
            "n_catchup": len(catchup),
            "why": f"floor {floor} = earliest existing verdict; "
                   f"{len(catchup)} day(s) to catch up"}


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
    producer_ends = 0
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
                # DB2. The committed O1 producer STAMPS a task-exit end on
                # gap_open_at_exit; refusing it made both suites green in
                # isolation while their integration refused the moment O1d
                # fired. A producer-supplied end is USED when it is finite and
                # ordered -- it is better evidence than anything the consumer
                # can synthesize, because the producer knows when the task
                # actually exited.
                if (not isinstance(_ge, (int, float)) or not math.isfinite(_ge)
                        or _ge <= _gs):
                    n_structural_bad += 1
                    continue
        # (c) gap_open_at_exit is the NEVER-RECONNECTED class -- exactly what
        # O1d exists for. Reading only gap_closed silently understates loss the
        # moment such a record appears, and says nothing while doing it.
        if ev not in GAP_EVENTS or r.get("coin") != coin:
            continue
        gs, ge = r.get("gap_start_ns"), r.get("gap_end_ns")
        if ev == "gap_open_at_exit":
            if ge is None:
                # SYNTHESIS IS THE FALLBACK, NOT THE RULE: only when the
                # producer supplied no end at all. The scope end charged to is
                # recorded explicitly rather than left implicit.
                ge = int(hi * 1e9)
                synthesized_ends += 1
            else:
                producer_ends += 1
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
        diag["producer_supplied_ends_used"] = producer_ends
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

#: REPORTED, NEVER GOVERNING -- under every regime.
#:
#: `tape_density` measures a real hole `complete_tape` cannot see: that
#: predicate counts WINDOWS, so a window holding 2 rows beside one holding
#: 17,092 counts as present (measured, eth 08-29 20:10Z). A feed that thins
#: without disconnecting writes no gap row and passes coverage.
#:
#: IT DOES NOT VETO, AND THAT IS DELIBERATE. The 5%-of-median threshold was
#: chosen after seeing which days fail it, and 7 of 13 days fail -- including
#: 08-29, the only day whose verdict reads all_pass. Letting it govern would
#: re-judge already-judged days on a bar picked from their own data, which is
#: rule 11 exactly. The measurement is real; retro-fitting a bar to it is not
#: a measurement. A GOVERNING density bar needs pre-registration against days
#: not yet seen, and that is a coordinator/USER act (rule 14) -- not something
#: this instrument may grant itself by adding a conjunct.
REPORTED_NOT_GOVERNING = ("tape_density",)


def governing_predicates(preds: list, regime: str) -> list:
    """Predicates that GOVERN the verdict under this regime. Callable."""
    out = [x for x in preds if x["predicate"] not in REPORTED_NOT_GOVERNING]
    if regime != "day_bar_v2":
        return out
    return [x for x in out if x["predicate"] not in SUPERSEDED_ON_V2]


#: The density receipt, MEASURED ELSEWHERE and consumed here. BE emits it,
#: this reports it -- the same split as the era ledger and the contamination
#: record: whoever measured emits, whoever judges consumes, neither does both.
PM_TAPE_DENSITY = Path(
    "/home/yuqing/ctaNew/data/pm_5min/derived/tape_density_v1.json")


def tape_density_for(day_token: str, path: Path | None = None) -> dict:
    """Per-coin thin-window counts for one day, or an explicit status.

    THREE OUTCOMES, AND THEY MUST NOT COLLAPSE INTO EACH OTHER:
      MEASURED            -- the receipt covers this day
      UNMEASURED          -- no receipt, or it genuinely does not cover the day
      SCHEMA_UNRECOGNISED -- a receipt IS there and this cannot read it

    The third was missing and the cost was immediate. The receipt changed from
    a list of day rows to {days, note, threshold_sensitivity}; iterating a dict
    yields its KEYS, no row matched, and this reported UNMEASURED -- "no
    measurement for this day" while the measurement sat in the file. A reader
    that turns a schema change into absence is worse than one that crashes,
    because absence is a plausible answer.

    THE WINDOW COUNTS ARE READINGS AT A THRESHOLD, NOT QUANTITIES. BE asserted
    threshold-insensitivity, never computed it, and on measuring found the set
    moves by 114 windows between 0.05 and 0.25. What survives is the DAY count,
    stable at 7 across a tenfold range. So the threshold and the stability
    range travel with every count reported here; a bare count would be the
    conclusion-beside-a-number this whole instrument exists to refuse.
    """
    src = PM_TAPE_DENSITY if path is None else path
    if not src.exists():
        return {"status": "UNMEASURED", "why": f"no density receipt at {src}"}
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"status": "UNMEASURED", "why": f"density receipt unreadable: {e}"}
    if not isinstance(doc, dict) or not isinstance(doc.get("days"), list):
        return {"status": "SCHEMA_UNRECOGNISED",
                "why": (f"density receipt at {src} is not "
                        f"{{days: [...], ...}}: top level is "
                        f"{type(doc).__name__} with "
                        f"{sorted(doc)[:6] if isinstance(doc, dict) else 'n/a'}. "
                        f"A shape this cannot read is NOT the same as no "
                        f"measurement, and must not report as one.")}
    hit = [r for r in doc["days"]
           if isinstance(r, dict) and r.get("day") == day_token]
    if not hit:
        return {"status": "UNMEASURED",
                "why": f"density receipt covers {len(doc['days'])} day(s), "
                       f"not {day_token}"}
    r = hit[-1]
    coins = r.get("coins") or {}
    per_coin = {c: {"n_thin_invisible": v.get("n_thin_invisible"),
                    "n_thin_accounted": v.get("n_thin_accounted"),
                    "n_windows": v.get("n_windows"),
                    "status": v.get("status")}
                for c, v in coins.items() if isinstance(v, dict)}
    curve = [x for x in (doc.get("threshold_sensitivity") or [])
             if isinstance(x, dict)]
    stable = sorted({x["threshold_frac_of_median"] for x in curve
                     if x.get("days_with_invisible_loss")
                     == max((y.get("days_with_invisible_loss") or 0)
                            for y in curve if (y.get("threshold_frac_of_median")
                                               or 1) <= 0.10)}) if curve else []
    return {
        "status": "MEASURED",
        "n_invisible_at_threshold": r.get("total_thin_invisible"),
        "n_accounted_at_threshold": r.get("total_thin_accounted"),
        "threshold_frac_of_median": r.get("threshold_frac_of_median"),
        "threshold_sensitivity": curve,
        "day_count_stable_over": [min(stable), max(stable)] if stable else None,
        "coins_unjudgeable": sorted(c for c, v in per_coin.items()
                                    if v["status"] != "JUDGED"),
        "per_coin": per_coin,
        "source": str(src),
        "governs": False,
        "why_not": ("REPORTED, NOT GOVERNING: the threshold was chosen after "
                    "seeing which days fail it, so letting it veto would "
                    "re-judge judged days on a bar taken from their own data "
                    "(rule 11). A governing bar needs pre-registration "
                    "against unseen days, which is a policy act, not this "
                    "instrument's to grant itself."),
    }


#: CONTENT LIVENESS -- structure frozen 2026-08-31, BEFORE 09-01 exists.
#:
#: complete_tape asks whether windows are present and tape_density asks whether
#: they are full. Neither can say WHOSE fault a thin day is. The discriminator
#: is received-vs-written: the collector's faithfulness (written/received) is
#: independent of the upstream's liveness (received vs its own norm), so the
#: two failures separate cleanly. 08-31 is the motivating case -- 0.51% of
#: normal message rate for 4.1 h while 99.77% of what was seen was written.
#:
#: FOUR STATUSES, and the fourth is not a failure state:
#:   CONTENT_LIVE            received within the day's normal band
#:   CONTENT_THIN_COLLECTOR  thin AND written/received materially below 1
#:   CONTENT_THIN_UPSTREAM   thin AND written/received ~= 1: tape faithful
#:   CONTENT_LIVENESS_UNRESOLVED  received unavailable, or no ratified bar
#:
#: THE THRESHOLDS ARE DELIBERATELY ABSENT AND THIS CODE WILL NOT INVENT THEM.
#: A band and a "materially below 1" are Class-C values; setting them from the
#: days that motivated the rule is rule 11, and setting them after seeing
#: 09-01 consumes 09-01. So the quantities are COMPUTED and REPORTED and the
#: status stays UNRESOLVED until a bar is ratified. That is the honest state:
#: the discriminator is frozen, the number is not mine.
CONTENT_LIVENESS_STATUSES = ("CONTENT_LIVE", "CONTENT_THIN_COLLECTOR",
                             "CONTENT_THIN_UPSTREAM",
                             "CONTENT_LIVENESS_UNRESOLVED")
PM_COLLECTOR_LOG = Path("/home/yuqing/ctaNew/data/pm_5min/collector.log")
#: BOTH stamp forms. From 2026-08-31T22:00Z the status line carries a full
#: ISO date; before it, only HH:MM:SSZ. A DATED line needs no reconstruction
#: and is not subject to the >24h blind spot; a dateless one is walked back
#: from an anchor. Matching only the old form would have matched ZERO lines
#: from tonight and reported "the log does not reach back that far" -- a
#: format change read as ABSENCE, which is the density-receipt defect again,
#: arriving on the night before the day this measure exists to judge.
_HB = re.compile(r"^\[pm\] (?:(\d{4})-(\d{2})-(\d{2})T)?"
                 r"(\d{2}):(\d{2}):(\d{2})Z .*?\bmsgs=(\d+)\b")
_PM_LINE = re.compile(r"^\[pm\] ")


def _heartbeats_dated(text: str, anchor_epoch: float) -> list[tuple[float, int]]:
    """(epoch, cumulative msgs) for heartbeat lines.

    The log stamps HH:MM:SSZ with NO DATE, so dates are reconstructed by
    walking BACKWARD from a known anchor and stepping the date at each
    midnight wrap.

    WHAT THIS CAN AND CANNOT CATCH, stated rather than implied. Reconstructed
    epochs that fail to increase ARE caught and refuse. A silent gap of MORE
    than 24 h is NOT detectable from dateless stamps at all: 12:00 followed by
    12:00 two days later is indistinguishable from one day later, and no check
    on the stamps can tell them apart. That limitation is real and is reported
    rather than guarded, because a guard that cannot fire is not a guard --
    the honest containment is that this measure is used only for days the log
    demonstrably covers with per-minute heartbeats.
    """
    entries, saw_pm = [], False
    for ln in text.splitlines():
        if _PM_LINE.match(ln):
            saw_pm = True
        m = _HB.match(ln)
        if not m:
            continue
        y, mo, d, hh, mm, ss, msgs = m.groups()
        sod = int(hh) * 3600 + int(mm) * 60 + int(ss)
        exact = (dt.datetime(int(y), int(mo), int(d), int(hh), int(mm),
                             int(ss), tzinfo=dt.timezone.utc).timestamp()
                 if y else None)
        entries.append((sod, int(msgs), exact))
    if not entries:
        if saw_pm:
            raise ValueError(
                "REFUSED: the log has [pm] lines but NOT ONE matches the "
                "heartbeat shape. That is a FORMAT CHANGE, not an absence of "
                "history, and reporting it as 'the log does not reach back' "
                "would read a rename as missing data.")
        return []
    # ONE backward walk over the whole file, in file order, because an
    # append-only log's order IS time order. A DATED line fixes the clock
    # exactly; a dateless one is placed on the day of the entry AFTER it,
    # stepping back when that would put it later. Reconstructing the dateless
    # block against its own anchor instead of against the dated block put a
    # pre-22:00Z line on the FOLLOWING day -- caught by the transition-night
    # test, which is the only night the mixed shape exists.
    out, cur = [], None
    for sod, msgs, exact in reversed(entries):
        if exact is not None:
            ts = exact
        else:
            ref = anchor_epoch if cur is None else cur
            ts = int(ref // 86400) * 86400 + sod
            if ts > ref:
                ts -= 86400
        out.append((ts, msgs))
        cur = ts
    out.reverse()
    for i in range(1, len(out)):
        if out[i][0] <= out[i - 1][0]:
            raise ValueError(
                "REFUSED: heartbeat dates do not reconstruct monotonically "
                "-- two stamps resolve to the same instant or step backward, "
                "so their order in time is unknowable and a misdated "
                "heartbeat would attribute one day's traffic to another")
    return out


def content_liveness_for(day_token: str, log_path: Path | None = None,
                         anchor_epoch: float | None = None) -> dict:
    """Received-rate evidence for one day. Computes; never classifies."""
    src = PM_COLLECTOR_LOG if log_path is None else log_path
    lo, hi = day_bounds(day_token)
    if not src.exists():
        return {"status": "CONTENT_LIVENESS_UNRESOLVED",
                "why": f"no collector log at {src}"}
    anchor = (src.stat().st_mtime if anchor_epoch is None else anchor_epoch)
    try:
        hb = _heartbeats_dated(src.read_text(errors="replace"), anchor)
    except ValueError as e:
        return {"status": "CONTENT_LIVENESS_UNRESOLVED", "why": str(e)}
    day = [(t, m) for t, m in hb if lo <= t < hi]
    if len(day) < 2:
        return {"status": "CONTENT_LIVENESS_UNRESOLVED",
                "why": (f"the log carries {len(day)} heartbeat(s) inside "
                        f"{day_token}; received rate needs at least two to "
                        f"difference. The log does not reach back far enough "
                        f"for historic days -- this measure is PROSPECTIVE by "
                        f"construction")}
    rates = []
    for i in range(1, len(day)):
        dt_s = day[i][0] - day[i - 1][0]
        dm = day[i][1] - day[i - 1][1]
        if dt_s > 0 and dm >= 0:
            rates.append(dm / dt_s)
    if not rates:
        return {"status": "CONTENT_LIVENESS_UNRESOLVED",
                "why": "no usable heartbeat interval (counter reset or "
                       "non-monotonic msgs)"}
    med = statistics.median(rates)
    low = [r for r in rates if med > 0 and r / med < 0.10]
    return {
        "status": "CONTENT_LIVENESS_UNRESOLVED",
        "why": ("the discriminator is computed and frozen; NO ratified band "
                "exists, and inventing one from the days that motivated the "
                "rule -- or from 09-01 after seeing it -- is the error this "
                "structure was declared early to avoid"),
        "n_intervals": len(rates),
        "median_msgs_per_s": round(med, 3),
        "min_msgs_per_s": round(min(rates), 3),
        "intervals_below_10pct_of_median": len(low),
        "fraction_of_day_below_10pct": round(len(low) / len(rates), 4),
        "note_10pct": ("10% is a REPORTING cut so the shape is visible, NOT a "
                       "ratified bar and not used to classify anything"),
        "written_received_ratio": None,
        "written_received_note": ("the collector logs RECEIVED; WRITTEN comes "
                                 "from the tape. Pairing them is the second "
                                 "half of the discriminator and needs the "
                                 "per-coin join, not shipped here"),
        "governs": False,
    }


#: THE ERA LEDGER. Admission is keyed on this file, not on quality.
PM_COLLECTOR_RUNS = Path("/home/yuqing/ctaNew/data/pm_5min/collector_runs.jsonl")

#: ERA ADMISSIBILITY IS A RULED INPUT, NEVER DERIVED. A collector era is
#: admissible for race accrual because a ruling says so -- not because its days
#: happen to pass quality. Those are different questions and 08-30 is what
#: happens when only the second one is asked: ETH passed quality on a MIXED-ERA
#: day and was marked `race_accrual_eligible=true`, while the day as a whole
#: read false only because BTC failed QUALITY. The eligibility was wrong for a
#: reason quality can never see.
#:
#: An era absent from this table REFUSES. A new collector version is not
#: admissible by default, and silence is not a ruling.
ERA_ADMISSIBLE = {
    "clob_v3_1": False,   # pre-O1
    "clob_v4": False,     # O1 package; ruled never admissible post-O1 (R-340)
    "clob_v5": True,      # heartbeat repair; admissible once its era starts
    # USER RULING 2026-08-31 ("Yes, admit"), recorded not inferred. DA
    # recommended ADMIT (Q-DA-188) on the ground that admissibility is a
    # question about the DATA: clob_v4_1 changes ONLY the RFC control-ping
    # cadence (3/3 -> 10/10), while row format, timestamps and sub-second
    # validity are identical to clob_v4. Refusing it would exclude admissible
    # data because a keepalive parameter moved.
    #
    # THE CAVEAT THAT TRAVELS WITH THE RULING, in view when it was made:
    # clob_v4_1 gap statistics are NOT comparable to clob_v4 ones, because the
    # CAUSE MIX shifts (~97% PING_TIMEOUT at 3/3 vs ~54% at 10/10, the rest
    # being instantly-detected causes). A bar crossing AT the boundary is a
    # measurement change, not a change in feed health. So a five-day window
    # that SPANS this boundary is heterogeneous in its quality basis: the
    # clock must record the ERA of every accrued day and never compare
    # quality across eras. P1/P2/P3 are NOT adjusted to restore
    # comparability -- recomputing a pre-registered bar voids it.
    "clob_v4_1": True,    # ping 3/3 -> 10/10 rollback; USER 2026-08-31
}


#: ---------------------------------------------------------------------------
#: THE TRANSITION/ROLLBACK RECEIPT CONTRACT (closes Codex V5-0700-R4)
#: ---------------------------------------------------------------------------
#: A ledger row records an ATTEMPT. AN ATTEMPT IS NOT AN ERA. The first guard
#: read every row as an effective transition, so the runbook's own
#: restart-failed row -- a v5 that never started -- minted an admissible v5 era
#: for every later day. Codex executed exactly that.
#:
#: THE FIX IS NOT "ALSO CHECK `aborted`". An absent boolean is indistinguishable
#: from false, so `r.get("aborted")` admits every row that omits the field, and
#: the next failure shape the runbook invents is silently admissible again --
#: the same defect one layer down, which is how this one was born. Every row
#: must ASSERT what it is from a CLOSED vocabulary, and a row that asserts
#: nothing REFUSES.
#:
#: EMITTER AND CONSUMER AGREE ON THIS SHAPE. It is declared here, not inferred.
#: THE STATE MARKERS ARE THE EMITTER'S OWN (bc854d3), WITH ONE ADDITION:
#: exactly ONE of these must be PRESENT AND TRUE on every non-legacy row.
#: `transitioned` is the addition. Without it a plain transition is encoded by
#: the ABSENCE of the other two -- and an absent boolean is indistinguishable
#: from a forgotten one, which is precisely how a restart that never happened
#: became an admissible era. Absence now REFUSES; so does asserting two.
STATE_MARKERS = {
    "transitioned": True,   # the new version RAN; contributes ONE boundary
    "aborted":      False,  # never ran; contributes NO boundary; needs `stage`
    "rollback":     True,   # ran and was REVERTED; contributes its OWN boundary
                            # and needs `stage`, `closes_boundary_utc` and a
                            # verified `collector_start_recv_ns`
}

#: Rows written before the markers existed, pinned BY IDENTITY. Silence is
#: ruled 'transitioned' for exactly these and nothing else -- the same pattern
#: as ERA_ADMISSIBLE, so no new row can inherit the exemption.
LEGACY_ROWS_RULED_TRANSITIONED = {
    ("clob_v4", "clob_v3_1", "2026-08-30T05:30:00Z"),
}


#: ONE spelling, not several. `fromisoformat` accepts a bare date, a space
#: separator, an explicit offset and fractional seconds -- and boundaries are
#: then compared to each other as RAW STRINGS, so two spellings of one instant
#: read as two different eras, while a bare "2026-08-31" silently means
#: midnight and moves a transition by hours. Found by BE's differential fuzz.
CANONICAL_INSTANT = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _instant(value, field: str, lineno: int) -> float:
    """Validate ONE canonical spelling and return its epoch."""
    if not isinstance(value, str) or not CANONICAL_INSTANT.match(value):
        raise ValueError(
            f"REFUSED: era ledger row {lineno} has {field}={value!r}. The one "
            f"accepted spelling is YYYY-MM-DDTHH:MM:SSZ -- boundaries are "
            f"compared to each other as strings, so a second spelling of the "
            f"same instant reads as a different era.")
    try:
        return dt.datetime.fromisoformat(
            value.replace("Z", "+00:00")).timestamp()
    except ValueError as e:
        raise ValueError(
            f"REFUSED: era ledger row {lineno} has {field}={value!r}, which is "
            f"canonically SHAPED but not a real instant ({e})") from None


def _ledger_rows(src: Path) -> list[tuple]:
    """Parse and VALIDATE every ledger row. Refuses rather than guessing."""
    out = []
    for lineno, ln in enumerate(src.read_text().splitlines(), 1):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError as e:
            # The raw error's coordinates are the FRAGMENT's, not the file's:
            # it says "line 1 column 428" for ledger line 7, and an operator
            # reads that as the first row. Name the real line, and refuse by
            # name so both consumers of this ledger agree on malformed.
            raise ValueError(
                f"REFUSED: era ledger row {lineno} is not valid JSON ({e.msg} "
                f"at column {e.colno} OF THAT ROW) -- a torn or concatenated "
                f"line, e.g. an append interrupted mid-write: {ln[:80]!r}"
            ) from None
        b, v = r.get("boundary_utc"), r.get("collector_schema_version")
        if not b or not v:
            raise ValueError(
                f"REFUSED: era ledger row {lineno} lacks boundary_utc or "
                f"collector_schema_version: {ln[:120]}")
        for _f, _val in (("collector_schema_version", v),
                         ("supersedes", r.get("supersedes"))):
            if _val is not None and not isinstance(_val, str):
                raise ValueError(
                    f"REFUSED: era ledger row {lineno} has {_f}={_val!r}, "
                    f"which is {type(_val).__name__}, not a string. Era names "
                    f"are sorted and set-compared together, so a non-string "
                    f"raised a bare TypeError on any day touching two eras -- "
                    f"an UNDECLARED exception where this module promises a "
                    f"declared refusal.")
        _instant(b, "boundary_utc", lineno)
        # THE ERA BEGINS WHEN THE PROCESS BEGINS, not when the ruling says so.
        # This field was validated on `recovered` and `rollback` rows but NOT
        # on plain `transitioned` rows -- the only kind that OPENS an
        # admissible era. A boundary ruled at 00:00:00Z with an ordinary 119 s
        # restart means the OLD version served the day's first 119 s, while the
        # day read era_pure and race-admissible. (BE Q-DA-180 item 1.)
        _cs = r.get("collector_start_recv_ns")
        if _cs is not None:
            if not isinstance(_cs, int) or isinstance(_cs, bool) or _cs <= 0:
                raise ValueError(
                    f"REFUSED: era ledger row {lineno} ({v} @ {b}) has "
                    f"collector_start_recv_ns={_cs!r}, which is not a positive "
                    f"int -- the evidence of when this era actually began")
            if _cs < _instant(b, "boundary_utc", lineno) * 1e9:
                raise ValueError(
                    f"REFUSED: era ledger row {lineno} ({v} @ {b}) declares a "
                    f"collector_start BEFORE its own boundary -- the process "
                    f"it names was already running before the era it opens")
        if "closes_boundary_utc" in r:
            _instant(r["closes_boundary_utc"], "closes_boundary_utc", lineno)
        # EVERY flag is read with `is True`, so a JSON int 1 or the string
        # "yes" reads as FALSE -- which for `recovered` waived the evidence
        # burden while the row still stood as the era of record, and for a
        # second state marker slipped past the asserts-two check. A flag that
        # is PRESENT must be a bool; a truthy non-bool is not a sloppy yes, it
        # is a value this contract cannot read.
        for _flag in (*STATE_MARKERS, "recovered"):
            if _flag in r and not isinstance(r[_flag], bool):
                raise ValueError(
                    f"REFUSED: era ledger row {lineno} ({v} @ {b}) has "
                    f"{_flag}={r[_flag]!r}, which is {type(r[_flag]).__name__}, "
                    f"not a bool. Every flag here is read with `is True`, so a "
                    f"truthy non-bool reads as FALSE and silently waives what "
                    f"it looks like it is asserting.")
        on = [m for m in STATE_MARKERS if r.get(m) is True]
        if len(on) > 1:
            raise ValueError(
                f"REFUSED: era ledger row {lineno} ({v} @ {b}) asserts {on} at "
                f"once -- a row records ONE attempt with ONE outcome")
        if not on:
            if (v, r.get("supersedes"), b) not in LEGACY_ROWS_RULED_TRANSITIONED:
                raise ValueError(
                    f"REFUSED: era ledger row {lineno} ({v} @ {b}) asserts NO "
                    f"state. A row records an ATTEMPT and an attempt is not an "
                    f"era; it must carry exactly one of "
                    f"{sorted(STATE_MARKERS)}=true. Absence is NOT "
                    f"'transitioned' -- reading it that way is how a restart "
                    f"that never happened became an admissible era.")
            on = ["transitioned"]
        st = on[0]
        if st in ("aborted", "rollback") and not str(r.get("stage") or "").strip():
            raise ValueError(
                f"REFUSED: {st} row {lineno} ({v} @ {b}) names no `stage` -- an "
                f"attempt that will not say WHICH path it took is not auditable")
        if r.get("recovered") is True:
            # A RECOVERED row is written after the fact, for a v5 that RAN but
            # whose stamp could never be appended. It is the one row whose
            # boundary is a RECONSTRUCTION rather than a contemporaneous stamp,
            # so it must carry the evidence the stamp would have carried.
            if st != "transitioned":
                raise ValueError(
                    f"REFUSED: row {lineno} ({v} @ {b}) is `recovered` but "
                    f"asserts {st!r}. Recovery records a transition that "
                    f"HAPPENED; an abort or a rollback is not a thing to "
                    f"recover")
            if not str(r.get("stage") or "").strip():
                raise ValueError(
                    f"REFUSED: recovered row {lineno} ({v} @ {b}) names no "
                    f"`stage` -- a retroactive boundary must say WHY it could "
                    f"not be stamped at the time")
            rec = r.get("collector_start_recv_ns")
            if not isinstance(rec, int) or isinstance(rec, bool) or rec <= 0:
                raise ValueError(
                    f"REFUSED: recovered row {lineno} ({v} @ {b}) carries no "
                    f"`collector_start_recv_ns`. Its boundary is a CLAIM about "
                    f"the past, not a stamp; without the {v} process's own "
                    f"start there is nothing to show the span ran at all")
        if r.get("supersedes") == v:
            raise ValueError(
                f"REFUSED: era ledger row {lineno} ({v} @ {b}) supersedes "
                f"ITSELF. The ledger records TRANSITIONS, and a row replacing "
                f"its own version transitions nothing -- but it still mints a "
                f"boundary, which silently makes the day it lands on impure "
                f"and costs a day off the validation clock for no era change. "
                f"A same-version restart is not a transition; if one needs "
                f"recording, it needs its own declared shape.")
        ts = _instant(b, "boundary_utc", lineno)
        out.append((ts, b, v, r.get("supersedes"), st, r))
    return out


def era_timeline(path: Path | None = None) -> dict:
    """(start_epoch, era_name) for EFFECTIVE transitions only, ascending.

    The era in force BEFORE the first recorded boundary is that entry's
    `supersedes` -- the ledger records transitions, so the opening era is named
    only by what it replaced. Attempts that never transitioned contribute
    NOTHING, and an attempt state that cannot be read unambiguously REFUSES.
    """
    src = PM_COLLECTOR_RUNS if path is None else path
    rows = _ledger_rows(src)
    if not rows:
        raise ValueError("REFUSED: era ledger is EMPTY -- with no recorded "
                         "era, no day can be shown to lie inside one")
    for i in range(1, len(rows)):
        if rows[i][0] < rows[i - 1][0]:
            raise ValueError(
                f"REFUSED: era ledger is OUT OF ORDER at {rows[i][1]} (after "
                f"{rows[i-1][1]}). It is append-only and must be chronological; "
                f"sorting it would silently reorder a rollback against the "
                f"transition it closes.")
    spans: list[tuple[float, str]] = []
    recovered: set[float] = set()
    seen: set[str] = set()   # versions that have HELD an effective era
    unevidenced: set[float] = set()
    open_era = open_since = prev_era = None
    open_recovered = False
    open_from_rollback = False
    for ts, b, v, sup, st, r in rows:
        if st == "aborted":
            if open_era is not None and v == open_era:
                raise ValueError(
                    f"REFUSED: AMBIGUOUS attempt state -- an `aborted` row for "
                    f"{v} at {b}, but {v} has been LIVE since {open_since}. A "
                    f"transition that RAN cannot be retracted by an abort row; "
                    f"it must be closed by a `rollback` row carrying a "
                    f"verified restoration. (This is the runbook's stage-4 "
                    f"instruction, and it leaves the live era open forever.)")
            continue
        if st == "rollback":
            if open_era is None:
                raise ValueError(
                    f"REFUSED: rollback row at {b} closes nothing -- no era is "
                    f"open for it to revert")
            if r.get("closes_boundary_utc") != open_since or sup != open_era:
                raise ValueError(
                    f"REFUSED: rollback row at {b} says it reverts "
                    f"{sup!r}@{r.get('closes_boundary_utc')!r}, but the OPEN era is "
                    f"{open_era!r}@{open_since!r} -- a rollback must NAME the "
                    f"transition it reverts")
            if ts <= dt.datetime.fromisoformat(
                    open_since.replace("Z", "+00:00")).timestamp():
                raise ValueError(
                    f"REFUSED: rollback row at {b} is stamped at or BEFORE the "
                    f"era it closes ({open_since}), giving {open_era!r} ZERO "
                    f"WIDTH. The v5 era really ran; its boundary_utc must be "
                    f"the RESTORATION instant, not a copy of the transition's.")
            rec = r.get("collector_start_recv_ns")
            if not isinstance(rec, int) or isinstance(rec, bool) or rec <= 0:
                raise ValueError(
                    f"REFUSED: rollback row at {b} carries no verified "
                    f"restoration receipt (collector_start_recv_ns). Without one, nothing shows the {v} process came "
                    f"back, and DA would name every later day from a version "
                    f"that may not be running.")
        elif st == "transitioned":
            # RECEIPT-CHAIN IDENTITY. `supersedes` must name the era ACTUALLY
            # in force at append time -- the consumer does not trust the
            # emitter to know which era is live. A row that supersedes an era
            # that is not open is malformed chain state, whatever it says.
            if open_era is None:
                if not sup:
                    raise ValueError(
                        f"REFUSED: the first effective row ({v} @ {b}) names no "
                        f"`supersedes` -- the ledger records transitions, so "
                        f"the OPENING era has no other name")
            elif sup != open_era:
                raise ValueError(
                    f"REFUSED: MALFORMED CHAIN -- the row at {b} claims to "
                    f"supersede {sup!r}, but the era in force since "
                    f"{open_since} is {open_era!r}. A transition receipt must "
                    f"name the era it ACTUALLY replaces; superseding an era "
                    f"that is not open means the writer and the ledger "
                    f"disagree about what is running.")
            # A RETURN to ANY version that has already held an era -- not
            # merely the immediately previous one. The harm is not the hop
            # count: a plain `transitioned` row skips the ENTIRE rollback
            # evidence contract, and v4->v5->v6->v4 skips it exactly as
            # completely as v4->v5->v4. Multi-hop is worse in one respect --
            # after two hops nobody remembers which era the missing evidence
            # would have described. (BE V5-P5-1; their rule, adopted.)
            #
            # The one exemption is a RETRY: the era now open was itself
            # restored by a rollback, so the version being returned to is the
            # one that rollback closed, and the evidence already exists.
            if v in seen and not open_from_rollback:
                raise ValueError(
                    f"REFUSED: AMBIGUOUS attempt state -- the row at {b} "
                    f"declares a plain `transitioned` back to {v}, which has "
                    f"ALREADY HELD an era in this chain, and {open_era!r} "
                    f"(open since {open_since}) was never closed. A return to "
                    f"any previously-in-force version is a ROLLBACK: it must "
                    f"declare rollback=true with a stage, a restoration "
                    f"receipt and closes_boundary_utc, or it cannot be told "
                    f"apart from a fresh deploy of {v}.")
        if not spans and sup:
            spans.append((float("-inf"), sup))
        _cs = r.get("collector_start_recv_ns")
        eff = (_cs / 1e9) if _cs is not None else ts
        if spans and spans[-1][0] > eff:
            raise ValueError(
                f"REFUSED: the row at {b} has an effective start "
                f"({dt.datetime.fromtimestamp(eff, dt.timezone.utc)}) BEFORE "
                f"the era before it -- restart delays cannot reorder the chain")
        spans.append((eff, v))
        if _cs is None:
            unevidenced.add(eff)
        if r.get("recovered") is True:
            recovered.add(eff)
        open_recovered = r.get("recovered") is True
        seen.add(v)
        prev_era, open_era, open_since = open_era, v, b
        open_from_rollback = (st == "rollback")
    if not spans:
        raise ValueError(
            "REFUSED: the era ledger records attempts but NOT ONE EFFECTIVE "
            "TRANSITION -- no era can be named, so no day lies inside one")
    if open_recovered:
        raise ValueError(
            f"REFUSED: the recovered era {open_era!r} opened at {open_since} is "
            f"never CLOSED. Recovery exists for a version that ran and was then "
            f"restored; an unclosed recovered row says that version is still "
            f"live yet was never stampable, which is not a state the runbook "
            f"can reach. A half-written recovery bundle refuses rather than "
            f"leaving the era open.")
    return {"spans": spans, "recovered": recovered,
            "unevidenced": unevidenced}


def era_spans(path: Path | None = None) -> list[tuple[float, str]]:
    """(start_epoch, era_name) for EFFECTIVE transitions only, ascending."""
    return era_timeline(path)["spans"]


def day_era_admission(day_token: str, path: Path | None = None,
                      admissible_table: dict | None = None) -> dict:
    """Is this UTC day ENTIRELY inside ONE ADMISSIBLE EFFECTIVE era?

    Two conditions, and BOTH are independent of day quality:
      * PURITY -- no era boundary falls strictly inside the day. A mid-day
        transition means the day's rows come from two collectors, so no coin's
        rows are homogeneous and per-coin quality cannot rescue it.
      * ADMISSIBILITY -- the single era it lies in is ruled admissible.
    """
    tbl = ERA_ADMISSIBLE if admissible_table is None else admissible_table
    lo, hi = day_bounds(day_token)
    _tl = era_timeline(path)
    spans, _recovered = _tl["spans"], _tl["recovered"]
    _unev = _tl["unevidenced"]
    inside = [(t, n) for t, n in spans if lo < t < hi]
    covering = [n for i, (t, n) in enumerate(spans)
                if t <= lo and (i + 1 == len(spans) or spans[i + 1][0] > lo)]
    touched = sorted({n for n in covering} | {n for _, n in inside})
    unknown = [n for n in touched if n not in tbl]
    if unknown:
        raise ValueError(
            f"REFUSED: era(s) {unknown} touch {day_token} but carry NO ruled "
            f"admissibility. A collector version is not admissible by default "
            f"and silence is not a ruling.")
    # Which era-starts does this day actually sit on or cross?
    starts = {t for t, _ in inside} | {t for i, (t, n) in enumerate(spans)
                                       if t <= lo and (i + 1 == len(spans)
                                                       or spans[i + 1][0] > lo)}
    reconstructed = bool(starts & _recovered)
    # An era whose row carries NO process evidence cannot be shown to have
    # begun when it claims, so its purity is unverifiable -- and assuming
    # boundary == start is the assumption that produced the false accept.
    unevidenced = bool(starts & _unev)
    pure = not inside and len(touched) == 1
    # A RECONSTRUCTED boundary is a claim about the past, not a stamp made at
    # the time. Era purity is a contemporaneous predicate -- the whole
    # discipline rests on the boundary being recorded when it happened -- so a
    # day resting on a recovered boundary does not accrue, whatever its
    # quality. PROPOSED CONSERVATIVE DEFAULT, not a ruling: relaxing it is a
    # policy call with its own priced trade-off (rule 14).
    admissible = (pure and not reconstructed and not unevidenced
                  and all(tbl[n] is True for n in touched))
    return {
        "day": day_token, "eras_touched": touched,
        "boundaries_inside_day": [dt.datetime.fromtimestamp(
            t, dt.timezone.utc).isoformat().replace("+00:00", "Z")
            for t, _ in inside],
        "era_pure": pure,
        "era_reconstructed": reconstructed,
        "era_unevidenced_start": unevidenced,
        "era_admissible_ruled": {n: tbl.get(n) for n in touched},
        "race_admissible_by_era": admissible,
        "why": ("a day not lying ENTIRELY inside ONE admissible era cannot "
                "accrue for ANY coin, whatever its quality -- era admission is "
                "a ruled property of the collector, not a measured property of "
                "the feed"),
    }


#: THE ACCRUAL RULE, in one place and in plain words. A day accrues when all
#: FOUR are true, and each is a different question:
#:
#:   1. FINISHED    -- it is a closed UTC day
#:   2. AFTER       -- it lies entirely after the freeze commit
#:   3. ADMISSIBLE  -- it lies entirely inside ONE ruled-admissible era
#:   4. HEALTHY     -- it passes the quality bars
#:
#: Nothing here is redundant and nothing else is required. Any of the four
#: false means the day does not count -- and NOT that the day was bad: (2) and
#: (3) are properties of the clock and the collector, never of the feed.
#:
#: WHAT (3) DOES AND DOES NOT DECIDE -- USER RULING 2026-09-01, recorded here
#: because it was previously ambiguous and read as a quality judgement:
#:
#:   *"i dont care about collector version, as long as the data quality is
#:    good, then we can use to test the model"*
#:
#: ERA IS NOT A QUALITY VERDICT, AND IT NEVER GRADES THE FEED. Across
#: clob_v3_1 -> clob_v4 -> clob_v4_1 the collector_runs ledger states its own
#: semantics as "distributional only; NO row-stamping change": the rows that
#: SURVIVE are recorded identically in every one of those eras. What differs is
#: how much is lost and how the loss is labelled, never the fidelity of what is
#: kept. So among RULED eras, QUALITY ALONE DECIDES -- which is exactly what
#: happens today, because clob_v4_1 is ruled admissible and every forward day
#: is era-pure clob_v4_1, so conjunct (3) is already satisfied and (4) governs.
#:
#: (3) SURVIVES AS AN INTERLOCK, NOT AS A GRADE. Its refusal text is "a
#: collector version is not admissible by default and silence is not a ruling".
#: Its job is the NEXT boundary: if a deploy introduces an era nobody has ruled
#: on, days must not start accruing under an unvetted collector -- the checker
#: refuses and NAMES the version instead. It costs nothing while every live era
#: is ruled, and this programme deployed a collector change on 2026-08-31.
#:
#: CROSS-ERA QUALITY COMPARISON REMAINS INVALID, which is a different claim
#: from the one above and must not be collapsed into it. P1/P2/P3 are
#: era-DEPENDENT IN MAGNITUDE: at ping 3/3 a stall becomes a logged gap in ~3 s,
#: at 10/10 sub-10 s stalls self-heal and are never logged at all (measured:
#: 08-31 1,134 btc gaps / 27.3 s median cumulative; 09-01 84 / 9.7 s, same feed).
#: So a day's bars are comparable to days in ITS OWN era and to its own bar
#: regime, never across a boundary. Forward days are all clob_v4_1, so the
#: forward comparison is internally valid; a historical cross-era table is not.
#:
#: AND THE BAR REGIME IS PART OF "HEALTHY", not a detail: days before
#: 2026-08-29 are governed by `count_bar_v1_frozen` (gap_rate_under_bar), from
#: 08-29 by `day_bar_v2` (P1/P2/P3, gap_rate SUPERSEDED). Reading a v2 bar
#: against a v1-governed day is an anachronism and flips verdicts -- 2026-08-28
#: passes P1 at 114.1 s/hr yet FAILS its actual bar at 20.29 gaps/hr.
ACCRUAL_RULE = ("a day accrues iff FINISHED (closed UTC day) AND AFTER (post "
                "freeze commit) AND ADMISSIBLE (the era is RULED -- an "
                "interlock against an unvetted collector, never a quality "
                "grade) AND HEALTHY (the quality bars of that day's OWN bar "
                "regime). Four conjuncts, four different questions. Among "
                "ruled eras QUALITY ALONE DECIDES (USER 2026-09-01); era "
                "carries no fidelity claim, since clob_v3_1/v4/v4_1 make NO "
                "row-stamping change. Cross-era quality comparison stays "
                "invalid: the bars are era-dependent in magnitude.")


def split_verdict(preds: list, regime: str = "count_bar_v1_frozen",
                  era_admissible: bool | None = None,
                  day_closed: bool | None = None) -> dict:
    """Separate DAY QUALITY (feed health) from RACE ACCRUAL (eligibility).

    CALLABLE, so the split can be driven directly rather than inferred from a
    composed boolean -- the lesson from all_pass being computed before the bars
    were appended. The rule it implements is `ACCRUAL_RULE` above.
    """
    gov = governing_predicates(preds, regime)
    quality = [x for x in gov if x["predicate"] != ACCRUAL_PREDICATE]
    accrual = [x for x in gov if x["predicate"] == ACCRUAL_PREDICATE]
    q_ok = bool(quality) and all(x["pass"] for x in quality)
    a_ok = bool(accrual) and all(x["pass"] for x in accrual)
    # ERA ADMISSION IS REQUIRED, NOT OPTIONAL. Passing None would let a caller
    # obtain eligibility by not asking -- absence of a check is not a passed
    # check, and this is the exact field that was wrong on 08-30.
    if era_admissible is not True and era_admissible is not False:
        raise ValueError(
            "REFUSED: split_verdict needs an explicit era_admissible. "
            "Eligibility must not be obtainable by omitting the era question.")
    # AND SO IS CLOSURE, for the same reason and by the same precedent.
    # `complete_tape` compares against the windows elapsed SO FAR, so it passes
    # mid-day -- which meant a four-hour-old day read eligible, and the nightly
    # (which verdicts the just-OPENED day as well as the closed one) would have
    # written eligible for a six-minute-old day. That was invisible while the
    # stale-era pin was failing those days for an unrelated wrong reason: one
    # wrong answer masking another.
    if day_closed is not True and day_closed is not False:
        raise ValueError(
            "REFUSED: split_verdict needs an explicit day_closed. A day that "
            "has not finished cannot have accrued, and eligibility must not "
            "be obtainable by omitting the question.")
    return {
        "day_quality_pass": q_ok,
        "post_freeze_pass": a_ok,
        "era_admissible": era_admissible,
        "day_closed": day_closed,
        "race_accrual_eligible": q_ok and a_ok and era_admissible
        and day_closed,
        "rule": ACCRUAL_RULE,
        "why": "feed health and clock eligibility are separate questions; a "
               "healthy day BEFORE the freeze commit is a good day that does "
               "not count, not a bad day -- and an UNFINISHED day is not yet "
               "a day at all",
        "era_role": "INTERLOCK, NOT A QUALITY GRADE (USER 2026-09-01). Among "
                    "RULED eras quality alone decides: clob_v3_1/v4/v4_1 make "
                    "NO row-stamping change, so era carries no fidelity claim "
                    "about the rows that survive. This conjunct exists to "
                    "refuse an UNRULED collector at the next boundary, not to "
                    "judge the feed. Cross-era quality COMPARISON stays "
                    "invalid separately: the bars are era-dependent in "
                    "magnitude.",
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


#: Immediate-order 0h. BREADTH IS DISCLOSED, NEVER GATED.
#:
#: `count_bar_v1_frozen`'s gap-RATE predicate was the only bar that ever saw
#: breadth, and `day_bar_v2` SUPERSEDED it (`SUPERSEDED_ON_V2`) in favour of
#: three DURATION bars. 08-28 is the proof that the two come apart: 186/288
#: btc windows carried a gap while P1/P2/P3 all passed. So a forward day can
#: pass its governing bars and still be substantially gap-affected, with
#: nothing in the governing set saying so.
#:
#: THIS DOES NOT CLOSE THAT WITH A NEW BAR. The bars are pre-registered and
#: frozen and 09-01 is the first forward day; a threshold chosen now, knowing
#: which days would pass it, is choosing after seeing (rule 11) and would void
#: the race. A governing breadth predicate needs pre-registration against days
#: not yet seen -- a coordinator/USER act (rule 14) -- and this instrument may
#: not grant it to itself by adding a conjunct. Same disposition as
#: `tape_density` under R-362: REPORTED_NOT_GOVERNING.
DISCLOSURE_ROLE = "REPORTED_NOT_GOVERNING"


def windows_affected_disclosure(lo: int, hi: int, coin: str,
                                elapsed_h: float,
                                path: Path | None = None,
                                intervals: list | None = None,
                                coverage_observed: bool | None = None
                                ) -> dict[str, Any]:
    """Windows touched by a COIN-LEVEL gap, carried beside P1/P2/P3.

    WHAT THE COUNT IS: 5-minute windows whose span intersects at least one
    merged coin-level gap interval. WHAT IT IS NOT: a count of contaminated
    windows. A gap opens a blind interval and forces a modeled queue reset and
    repost; the replay clears state, resynchronizes and re-anchors from the
    next quote, and busy windows carry thousands of `book` snapshots, so an
    overlap does not leave the rest of the window stale (HANDOFF 09:43Z
    correction, which withdrew the opposite claim).

    TWO DENOMINATORS, BOTH CARRIED, because they answer different questions and
    one of them has already been misread: 52/113 (the live rate over windows
    that have actually happened) and 52/288 (progress toward the complete-day
    denominator) are the same numerator and mean different things. Only /288 is
    the CLOSING-day denominator; on a closed day the two coincide.

    ZERO IS NOT A CLEAN CLAIM. An empty gap ledger from a dead collector and
    one from a perfect feed are the same bytes, so the disclosure carries the
    interval count it actually read and flags a zero that arrives without
    affirmative coverage evidence, rather than publishing a reassuring 0%.
    """
    iv = coin_gap_intervals(lo, hi, coin, path) if intervals is None \
        else list(intervals)
    aff = 0
    for i in range(WINDOWS_PER_DAY):
        w0 = lo + i * WINDOW_S
        w1 = w0 + WINDOW_S
        if any(a < w1 and b > w0 for a, b in iv):
            aff += 1
    # COMPLETE windows only. The in-flight window is excluded by construction:
    # a window that has not finished cannot be judged, and counting it would
    # make the live rate flap with where inside the window you look -- the
    # same trap that made a partial day read HEALTHY then UNHEALTHY twenty
    # minutes later.
    elapsed_w = max(0, min(WINDOWS_PER_DAY, int(elapsed_h * 3600.0 // WINDOW_S)))
    # A None, never a 0.0: no elapsed windows means the rate is UNDEFINED, and
    # a 0.0 there would read as "no breadth" -- absence of a measurement
    # wearing the shape of a clean one.
    over_elapsed = (aff / elapsed_w) if elapsed_w else None
    return {
        "role": DISCLOSURE_ROLE,
        "is_a_gate": False,
        "has_threshold": False,
        "governs_all_pass": False,
        "windows_affected_COIN_LEVEL": aff,
        "windows_complete_elapsed": elapsed_w,
        "windows_total": WINDOWS_PER_DAY,
        "affected_over_elapsed": None if over_elapsed is None
        else round(over_elapsed, 4),
        "affected_over_288": round(aff / WINDOWS_PER_DAY, 4),
        "pct_of_elapsed": None if over_elapsed is None
        else round(100.0 * over_elapsed, 1),
        "pct_of_288": round(100.0 * aff / WINDOWS_PER_DAY, 1),
        "coin_level_gap_intervals_read": len(iv),
        "coverage_observed_arg": repr(coverage_observed),
        "zero_affected_is_not_a_clean_claim": (
            aff == 0 and coverage_observed is not True),
        "denominator_note": (
            "affected_over_elapsed is the LIVE rate over COMPLETE elapsed "
            "windows; affected_over_288 is progress toward the complete-day "
            "denominator and is the ONLY one a closing-day receipt quotes. "
            "They coincide on a closed day."),
        "meaning_note": (
            "a DISCLOSURE COUNT of short blind intervals and modeled queue "
            "resets -- NOT a count of contaminated windows, NOT a gate, and "
            "NOT a threshold. The governing bars score DURATION (P1/P2/P3); "
            "breadth is carried BESIDE them, never instead of them, because "
            "v2 retired the only predicate that saw it (08-28: 186/288 "
            "windows touched with all three duration bars passing)."),
    }


def _zero_probe_intervals(path: Path, lo: int, hi: int) -> int:
    """Interval count on an arbitrary ledger -- used only to give the vacuity
    control a THIRD, disagreeing fixture."""
    return len(coin_gap_intervals(lo, hi, "btc", path))


def day_bar_v2(lo: int, hi: int, coin: str, elapsed_h: float,
               path: Path | None = None,
               coverage_observed: bool | None = None) -> dict[str, Any]:
    """P1/P2/P3 for one coin-day, from COIN-LEVEL merged gap intervals."""
    iv = coin_gap_intervals(lo, hi, coin, path)
    lost = sum(b - a for a, b in iv)
    # 0h: carried in EVERY branch, including the refusals. A disclosure that
    # disappears exactly when the bar declines to evaluate is a disclosure the
    # reader does not have on the days it most wants it.
    _disc = windows_affected_disclosure(lo, hi, coin, elapsed_h, path,
                                        intervals=iv,
                                        coverage_observed=coverage_observed)
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
            "windows_affected_disclosure": _disc,
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
            "windows_affected_disclosure": _disc,
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
        "windows_affected_disclosure": _disc,
        "thresholds": {"P1_max_s_per_hr": P1_LOST_S_PER_HR_MAX,
                       "P2_material_span_s": P2_MATERIAL_SPAN_S,
                       "P2_max_share": P2_MATERIAL_SHARE_MAX,
                       "P3_max_rolling_60min_s": P3_ROLLING_60MIN_LOST_S_MAX},
    }


def assert_disclosure_carried(rep: dict) -> None:
    """REFUSE a `day_bar_v2` receipt whose bars carry no 0h breadth disclosure.

    Rule 17, the half a green suite cannot supply. `day_bar_v2` returning the
    field proves the UNIT; it does not prove the RUNNER emits it, and this
    programme has already shipped six falsifier-proven evaluators with zero
    call sites (I11-2) and two green suites over an integration that always
    refused (DB2). So the ARTIFACT refuses to exist without the field, in both
    places a reader looks: the whole-day `day_bar_v2` block and every per-coin
    block. A disclosure the receipt does not carry is not a disclosure.
    """
    if rep.get("bar_regime") != "day_bar_v2":
        return
    need = ("windows_affected_COIN_LEVEL", "windows_complete_elapsed",
            "affected_over_elapsed", "affected_over_288", "role")
    missing: list[str] = []
    scopes = [("day_bar_v2", rep.get("day_bar_v2") or {})]
    scopes.append(("per_coin", {c: (v or {}).get("day_bar_v2")
                                for c, v in (rep.get("per_coin") or {}).items()
                                if isinstance(v, dict)
                                and v.get("day_bar_v2") is not None}))
    n_checked = 0
    for scope, blocks in scopes:
        for coin, b in sorted(blocks.items()):
            if not isinstance(b, dict):
                missing.append(f"{scope}/{coin}: no bar block")
                continue
            n_checked += 1
            d = b.get("windows_affected_disclosure")
            if not isinstance(d, dict):
                missing.append(f"{scope}/{coin}: disclosure absent")
                continue
            gone = [k for k in need if k not in d]
            if gone:
                missing.append(f"{scope}/{coin}: missing {gone}")
    # AN EMPTY SET IS NOT A PASS. A v2-regime report with no bar blocks at all
    # would satisfy a "nothing missing" test while disclosing nothing -- the
    # empty-set trap this programme has paid for repeatedly.
    if n_checked == 0:
        raise SystemExit(
            "REFUSED: a day_bar_v2 receipt with ZERO bar blocks checked. "
            "Nothing missing from nothing is not a disclosure (rule 11/16).")
    if missing:
        raise SystemExit(
            "REFUSED to emit: the 0h windows-affected disclosure is not "
            "carried beside P1/P2/P3 -- " + "; ".join(missing)
            + ". The governing bars score DURATION only; a receipt without "
              "breadth beside them lets a substantially gap-affected day read "
              "as unqualified.")


def closed_label(day_closed, calendar_closed: bool) -> str:
    """Attribute the day-closed flag to WHOSE it is, and name a disagreement.

    Extracted so BOTH branches are testable. The disagreement clause only
    renders while the two differ -- which at 00:06Z they do and an hour later
    they do not -- so an inline version would have shipped a branch no test
    ever entered. That is exactly how `NON_NUMERIC_SIDE` reached the tree
    undefined yesterday with a green suite.
    """
    out = f"selector day_closed={day_closed}"
    if bool(day_closed) != bool(calendar_closed):
        out += (f" (calendar says {calendar_closed}; the tape-derived "
                f"predicate lags the boundary by up to one window)")
    return out


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
    # ERA ADMISSION, COMPUTED ONCE FOR THE DAY. A mixed-era day is mixed for
    # EVERY coin; a coin cannot pass its way out of it, which is exactly what
    # 08-30's ETH did before this guard existed.
    _era = day_era_admission(day_token)
    iso = dt.datetime.fromtimestamp(lo, dt.timezone.utc).strftime("%Y-%m-%d")
    now = dt.datetime.now(dt.timezone.utc)
    preds: list[dict[str, Any]] = []

    def p(name, ok, detail):
        preds.append({"predicate": name, "pass": bool(ok), "detail": detail})

    # --- 1. entirely post-freeze ------------------------------------------
    # THE SELECTOR ERA IS THE DAY'S OWN ERA, derived from the ledger directly
    # above -- never a module constant. `warning_window` used to read
    # `flow_intensity.ERA`, a literal pinned to `clob_v3_1`, an era that closed
    # 2026-08-30T05:30:01Z; every later day was therefore absent from the
    # selector and this predicate failed BY CONSTRUCTION, silently zeroing
    # accrual for any feed quality whatsoever. Deriving it here also makes the
    # verdict SELF-CONSISTENT: `era_admission` said 08-31 was clob_v4/v4_1
    # while this predicate said the day did not exist, in one artifact.
    # A mixed-era day has no single era, so there is nothing to load and
    # nothing to pass -- it is refused BY NAME rather than by absence. Such a
    # day already fails `era_admissible`, so this narrows no day's outcome.
    _touched = _era.get("eras_touched") or []
    _sel_era = _touched[0] if (_era.get("era_pure") and len(_touched) == 1) \
        else None
    if _sel_era is None:
        sel = {"freeze_epoch": freeze_epoch, "era": None, "days": {}}
    else:
        sel = WW.select_holdout(freeze_epoch, era=_sel_era)
    day = sel["days"].get(iso)
    if day is None:
        p("entirely_post_freeze", False,
          (f"{iso} spans eras {_touched} -- a mixed-era day has no single "
           f"era whose windows could be loaded, so this cannot be verified, "
           f"not passed" if _sel_era is None else
           f"{iso} absent from the selector for its own era {_sel_era!r} -- "
           f"cannot be verified, not passed"))
        adm = tot = {}
        day_closed = None
    else:
        adm, tot = day["n_admissible_by_coin"], day["n_total_by_coin"]
        day_closed = day["day_closed"]
        allpost = bool(tot) and all(adm[c] == tot[c] for c in tot)
        # SAY WHOSE FLAG IT IS. This printed a bare `day_closed=False` beside
        # an artifact that separately carries `day_closed_selector=False` AND
        # `day_closed_calendar=True` -- so a human reading the one line saw a
        # closed day called open, while `complete_tape` two predicates down
        # labelled the same disagreement explicitly. A machine reader had both
        # fields; a person had one unattributed word. Observed on the 08-28
        # final verdict (Q-DA-149).
        p("entirely_post_freeze", allpost,
          closed_label(day_closed, now.timestamp() >= hi)
          + "; " + ", ".join(
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

    # --- 2b. tape DENSITY (reported, never governing) -----------------------
    # complete_tape counts WINDOWS. A window holding 2 rows beside one holding
    # 17,092 counts as present, so a feed that thins without disconnecting
    # writes no gap row and passes coverage. Measured elsewhere; reported here
    # with its own n so a reader can see content as well as coverage.
    _td = tape_density_for(day_token)
    p("tape_density",
      _td["status"] == "MEASURED" and _td.get("n_invisible_at_threshold") == 0,
      (f"{_td['n_invisible_at_threshold']} thin window(s) with NO gap row "
       f"covering them AT threshold={_td.get('threshold_frac_of_median')} of "
       f"the per-coin median -- a READING AT A SETTING, not a quantity; the "
       f"day-level verdict is stable over "
       f"{_td.get('day_count_stable_over')}. "
       f"{_td['n_accounted_at_threshold']} accounted for by the gap ledger"
       + (f"; coins not judgeable: {_td['coins_unjudgeable']}"
          if _td.get("coins_unjudgeable") else "")
       + " -- REPORTED ONLY, this predicate does NOT govern all_pass"
       if _td["status"] == "MEASURED" else
       f"{_td['status']} -- {_td['why']}"))

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
    # SAME CORRECTION AS THE SELECTOR, and the reason this one is a UNION: a
    # mixed-era day's windows genuinely come from every era it touches, so
    # taking one would drop the rest. Under the old `fi.ERA` literal both of
    # these read the CLOSED clob_v3_1 era, which is why `era_covered_windows`
    # reported 0 and `gap_affected_PER_SLUG` reported 0/None for 08-31 and
    # 09-01 -- days holding 288 windows each. Neither governs the bar
    # (COIN_LEVEL is the input, R-191, and it was always era-correct), so this
    # fixes two REPORTED numbers that read as clean, not a verdict.
    gaps, cov = {}, set()
    for _e in (_touched or [fi.ERA]):
        gaps.update(fi.gaps_by_slug(_e))
        cov |= fi.covered_slugs(_e)
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

            _reg = bar_regime(day_token)
            # DB1. The per-coin verdict used to be composed BEFORE P1/P2/P3
            # existed, so a coin-day PASSED over a 4,000s outage while its own
            # bar failed -- and PER-COIN VERDICTS FEED PER-COIN CLOCKS, so that
            # day would have accrued. The same ordering defect as the global
            # all_pass: I fixed one consumer and left its twin. The coin's bars
            # are computed HERE and appended to its own table before anything
            # is composed.
            _cov_c = counts.get(coin, 0) > 0 and short.get(coin, 1) <= 0
            _cb = (day_bar_v2(lo, hi, coin, gs["hours_elapsed"],
                              coverage_observed=_cov_c)
                   if _reg == "day_bar_v2" else None)
            if _cb is not None:
                for _k in ("P1", "P2", "P3"):
                    cpp(f"{_k}_bar", _cb.get(f"{_k}_pass", False),
                        _cb.get("why") or f"{_k} on this coin's own gaps")
            _gov_cp = governing_predicates(cp, _reg)
            # SAME era admission for every coin: a mixed-era day is mixed
            # for all of them, and a coin cannot pass its way out of it.
            _csplit = split_verdict(cp, _reg, _era["race_admissible_by_era"],
                                    day_closed=now.timestamp() >= hi)
            per_coin[coin] = {
                "predicates": cp,
                "day_bar_v2": _cb,
                "verdict_split": _csplit,
                "race_accrual_eligible": _csplit["race_accrual_eligible"],
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

    # CALENDAR closure, not the selector's -- the selector's `day_closed`
    # depends on tape arriving, so a stalled collector would make a finished
    # day look unfinished. The question here is only "has the UTC day ended".
    _split = split_verdict(preds, regime, _era["race_admissible_by_era"],
                           day_closed=now.timestamp() >= hi)

    return {
        "instrument": "da_forward_day_verify_v1",
        "verdict_split": _split,
        "era_admission": _era,
        # The era whose windows the accrual predicate actually loaded. Carried
        # because "which population was read" is the question the old literal
        # made unanswerable from the artifact alone.
        "selector_era": _sel_era,
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
        "tape_density": _td,
        # A REPORT FIELD, not a predicate. A predicate carries pass/fail and
        # this has no ratified bar, so a pass/fail here would be invented.
        "content_liveness": content_liveness_for(day_token),
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

        # ---- 0h WINDOWS-AFFECTED DISCLOSURE -------------------------------
        # Both directions, because a control that only ever refuses passes
        # nothing and a control that only ever admits fails nothing (rule 16).
        #
        # POSITIVE CONTROL: it must ADMIT and report the RIGHT count. Three
        # gaps placed inside windows 0, 100 and 287; one of them spans a
        # window boundary and must therefore touch TWO windows, so the
        # expected answer is 4 and a boundary-blind implementation says 3.
        _w = WINDOW_S
        _rows = [
            {"event": "gap_closed", "coin": "btc", "slug": "s0",
             "gap_start_ns": int((_lo + 10) * 1e9),
             "gap_end_ns": int((_lo + 20) * 1e9)},
            {"event": "gap_closed", "coin": "btc", "slug": "s1",
             "gap_start_ns": int((_lo + 100 * _w + _w - 2) * 1e9),
             "gap_end_ns": int((_lo + 101 * _w + 2) * 1e9)},
            {"event": "gap_closed", "coin": "btc", "slug": "s2",
             "gap_start_ns": int((_lo + 287 * _w + 5) * 1e9),
             "gap_end_ns": int((_lo + 287 * _w + 9) * 1e9)},
        ]
        _f = Path(_td) / "disclosure.jsonl"
        _f.write_text("\n".join(json.dumps(r) for r in _rows), encoding="utf-8")
        _d = windows_affected_disclosure(_lo, _hi, "btc", 24.0, _f,
                                         coverage_observed=True)
        ok(_d["windows_affected_COIN_LEVEL"] == 4
           and _d["windows_complete_elapsed"] == 288
           and _d["affected_over_288"] == round(4 / 288, 4)
           and _d["affected_over_elapsed"] == round(4 / 288, 4),
           "0h positive control: the disclosure ADMITS and reports 4 touched "
           "windows (the boundary-spanning gap touches two), with both "
           "denominators agreeing on a CLOSED day")
        ok(_d["coin_level_gap_intervals_read"] == 3,
           "0h vacuity control: the disclosure records how many gap intervals "
           "it actually READ -- a 0-affected report from a reader that read "
           "nothing is the empty-set trap, not a clean day")

        # THE LOAD-BEARING KNOWN-BAD: a disclosure that FAILS spectacularly
        # must leave every governing verdict UNCHANGED. 288 hairline gaps put
        # breadth at 100% while total lost time is 28.8s, so P1/P2/P3 all pass
        # -- and they must, or breadth has quietly become a bar (rule 11).
        _all = [{"event": "gap_closed", "coin": "btc", "slug": f"s{i}",
                 "gap_start_ns": int((_lo + i * _w + 10) * 1e9),
                 "gap_end_ns": int((_lo + i * _w + 10.1) * 1e9)}
                for i in range(WINDOWS_PER_DAY)]
        _fa = Path(_td) / "all_affected.jsonl"
        _fa.write_text("\n".join(json.dumps(r) for r in _all), encoding="utf-8")
        _b_wide = day_bar_v2(_lo, _hi, "btc", 24.0, _fa, coverage_observed=True)
        _b_clean = day_bar_v2(_lo, _hi, "btc", 24.0, _e, coverage_observed=True)
        ok(_b_wide["windows_affected_disclosure"]["pct_of_288"] == 100.0
           and _b_wide["P1_pass"] and _b_wide["P2_pass"] and _b_wide["P3_pass"],
           "0h known-bad: a 100%-BREADTH day still PASSES all three governing "
           "duration bars -- the disclosure REPORTS and does not VETO")
        _gov = ["P1_pass", "P2_pass", "P3_pass", "evaluable"]
        ok([_b_wide[k] for k in _gov] == [_b_clean[k] for k in _gov],
           "0h known-bad: 0% and 100% breadth produce IDENTICAL governing "
           "outcomes, so no verdict depends on the disclosure")
        _pp = [{"predicate": "complete_tape", "pass": True},
               {"predicate": ACCRUAL_PREDICATE, "pass": True}]
        ok(compose_all_pass(_pp, {}, {"btc": _b_wide}, "day_bar_v2") is True
           and compose_all_pass(_pp, {}, {"btc": _b_clean}, "day_bar_v2") is True,
           "0h known-bad: compose_all_pass reads P1/P2/P3 only -- the "
           "100%-breadth day and the clean day both compose to True")
        ok(_b_wide["windows_affected_disclosure"]["coin_level_gap_intervals_read"]
           == WINDOWS_PER_DAY
           and _d["coin_level_gap_intervals_read"] == 3
           and _zero_probe_intervals(_e, _lo, _hi) == 0,
           "0h vacuity control, SECOND fixture: the interval counter reads 3, "
           "288 and 0 on three different ledgers. A single fixture lets a "
           "hardcoded constant satisfy the check -- a mutation that pinned "
           "this to 3 survived until the counts had to disagree")
        ok(_b_wide["windows_affected_disclosure"]["is_a_gate"] is False
           and _b_wide["windows_affected_disclosure"]["governs_all_pass"] is False
           and _b_wide["windows_affected_disclosure"]["role"]
           == "REPORTED_NOT_GOVERNING",
           "0h: the disclosure names its own role, so a downstream reader "
           "cannot mistake it for a predicate (R-362's class)")

        # OPEN-DAY denominators must actually DIFFER, or carrying both is
        # decoration. 9.4939h -> 113 COMPLETE windows (the in-flight one is
        # excluded by construction).
        _open_d = windows_affected_disclosure(_lo, _hi, "btc", 9.4939, _fa,
                                              coverage_observed=True)
        ok(_open_d["windows_complete_elapsed"] == 113
           and _open_d["affected_over_elapsed"] != _open_d["affected_over_288"]
           and _open_d["pct_of_288"] < _open_d["pct_of_elapsed"],
           "0h: on an OPEN day the two denominators differ (affected/elapsed "
           "vs affected/288) -- the pair that stops 52/113 being read as 52/288")

        # A rate with no elapsed windows is UNDEFINED, never 0.0.
        _zero = windows_affected_disclosure(_lo, _hi, "btc", 0.0, _e,
                                            coverage_observed=True)
        ok(_zero["affected_over_elapsed"] is None
           and _zero["pct_of_elapsed"] is None
           and _zero["windows_complete_elapsed"] == 0,
           "0h known-bad: zero elapsed windows yields None, NOT 0.0 -- a 0.0 "
           "there would read as 'no breadth measured clean'")

        # ZERO AFFECTED IS NOT A CLEAN CLAIM without affirmative coverage.
        _z_unobs = windows_affected_disclosure(_lo, _hi, "btc", 24.0, _e,
                                               coverage_observed=None)
        _z_obs = windows_affected_disclosure(_lo, _hi, "btc", 24.0, _e,
                                             coverage_observed=True)
        ok(_z_unobs["zero_affected_is_not_a_clean_claim"] is True
           and _z_obs["zero_affected_is_not_a_clean_claim"] is False,
           "0h: an empty ledger WITHOUT observed coverage is flagged, and the "
           "same emptiness WITH coverage is not -- the flag discriminates "
           "rather than firing on every zero")

        # ---- 0h ARTIFACT-LEVEL GUARD (rule 17's second half) -----------
        _rep_ok = {"bar_regime": "day_bar_v2", "day_bar_v2": {"btc": _b_wide},
                   "per_coin": {"btc": {"day_bar_v2": _b_wide}}}
        assert_disclosure_carried(_rep_ok)          # must NOT raise
        ok(True, "0h guard positive control: a receipt that DOES carry the "
                 "disclosure in both scopes is admitted")
        ok(assert_disclosure_carried({"bar_regime": "count_bar_v1_frozen"})
           is None,
           "0h guard: a v1-regime day is out of scope and is not refused -- "
           "the guard discriminates by regime rather than refusing everything")
        import copy as _cp
        for _scope in ("day_bar_v2", "per_coin"):
            _bad_rep = _cp.deepcopy(_rep_ok)
            _tgt = (_bad_rep["day_bar_v2"]["btc"] if _scope == "day_bar_v2"
                    else _bad_rep["per_coin"]["btc"]["day_bar_v2"])
            _tgt.pop("windows_affected_disclosure")
            try:
                assert_disclosure_carried(_bad_rep)
                ok(False, f"0h guard known-bad ({_scope}) must REFUSE")
            except SystemExit as _e:
                ok("not carried beside P1/P2/P3" in str(_e)
                   and _scope in str(_e),
                   f"0h guard known-bad: a receipt missing the disclosure in "
                   f"{_scope} REFUSES and names the scope")
        _partial = _cp.deepcopy(_rep_ok)
        _partial["day_bar_v2"]["btc"]["windows_affected_disclosure"].pop(
            "affected_over_elapsed")
        try:
            assert_disclosure_carried(_partial)
            ok(False, "0h guard known-bad: a HALF disclosure must REFUSE")
        except SystemExit as _e:
            ok("affected_over_elapsed" in str(_e),
               "0h guard known-bad: carrying only one denominator refuses and "
               "names the missing one -- 0h asks for BOTH")
        try:
            assert_disclosure_carried({"bar_regime": "day_bar_v2",
                                       "day_bar_v2": {}, "per_coin": {}})
            ok(False, "0h guard known-bad: ZERO bar blocks must REFUSE")
        except SystemExit as _e:
            ok("ZERO bar blocks" in str(_e),
               "0h guard known-bad: an EMPTY v2 receipt refuses -- nothing "
               "missing from nothing is not a disclosure (the empty-set trap)")

        # CARRIED IN EVERY BRANCH, including the two refusals.
        _nb1 = day_bar_v2(_lo, _hi, "btc", 24.0, _fa, coverage_observed=False)
        _nb2 = day_bar_v2(_lo, _hi, "btc", 0.5, _fa, coverage_observed=True)
        ok(_nb1["evaluable"] is False and _nb2["evaluable"] is False
           and _nb1["windows_affected_disclosure"]["windows_affected_COIN_LEVEL"] == 288
           and _nb2["windows_affected_disclosure"]["windows_affected_COIN_LEVEL"] == 288,
           "0h: the disclosure is carried in BOTH non-evaluable branches -- a "
           "disclosure that vanishes when the bar refuses is missing on the "
           "days a reader most wants it")

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
                   .replace('LOG="${DA_MIDNIGHT_LOG:-'
                            '/home/yuqing/ctaNew/data/pm_5min/derived/'
                            '.da_midnight_verify.log}"',
                            f'LOG="{Path(_ltd) / "log"}"'),
                encoding="utf-8")
            # The verifier is shimmed through the launcher's OWN documented
            # override, not by string surgery on its source. The previous
            # version replaced the literal `V=/home/...`; when that line
            # gained a default-expansion the replace became a SILENT NO-OP and
            # the seam ran the REAL verifier -- a shim that quietly stops
            # shimming tests nothing and says so in no way at all.
            assert str(Path(_ltd) / "log") in _run.read_text(encoding="utf-8"), (
                "seam shim: the LOG redirect did not match the launcher's "
                "text -- refusing to run a seam test against production paths")
            _run.chmod(0o755)
            # BOTH overrides, or none. This test redirected the LOG and left
            # OUTDIR pointing at production, so every run of the seam test
            # OVERWROTE the real verdict artifacts -- invisibly, because the
            # path is stable and the log was elsewhere. That is how the 00:06Z
            # verdicts for 08-27 and 08-28 were replaced at 09:14Z. An
            # instrument that mutates production state as a side effect of
            # checking it is not a check.
            _out = Path(_ltd) / "out"
            _env = {"PATH": "/usr/bin:/bin",
                    "DA_MIDNIGHT_LOG": str(Path(_ltd) / "log"),
                    "DA_MIDNIGHT_OUTDIR": str(_out),
                    "DA_MIDNIGHT_VERIFY_BIN": str(_spy)}
            _pr = _sp.run(["bash", str(_run)], capture_output=True, env=_env)
            ok(_pr.returncode != 5,
               "(1) the seam run is FULLY isolated: log AND outdir both "
               "redirected, so checking the launcher does not rewrite the "
               "production verdicts it is checking")
            _half = _sp.run(["bash", str(_run)], capture_output=True,
                            env={k: v for k, v in _env.items()
                                 if k != "DA_MIDNIGHT_OUTDIR"})
            ok(_half.returncode == 5 and b"REFUSED" in _half.stderr,
               "(1) and overriding ONLY the log REFUSES (rc 5) -- a half "
               "isolation reads as isolated while writing production "
               "artifacts, which is worse than none")
            # A GUARD THAT WRITES BEFORE IT REFUSES HAS ALREADY DONE THE THING
            # IT REFUSES. Codex batch-2 §7: with only OUTDIR overridden this
            # refused with rc=5 *after* appending its header to the production
            # log (46 bytes, measured). Both refusal directions must now leave
            # the log path untouched, so the check is on the FILE, not the
            # return code -- the return code was already right.
            _dlog = Path(_ltd) / "default.log"
            _run.write_text(_run.read_text(encoding="utf-8").replace(
                str(Path(_ltd) / "log"), str(_dlog)), encoding="utf-8")
            _oonly = _sp.run(["bash", str(_run)], capture_output=True,
                             env={k: v for k, v in _env.items()
                                  if k != "DA_MIDNIGHT_LOG"})
            ok(_oonly.returncode == 5 and not _dlog.exists(),
               "(1) a refusal writes NOTHING: with only OUTDIR overridden the "
               "run refuses AND leaves its log path untouched. The pair guard "
               "was written this morning against 'an isolation that only "
               "covers the visible half' and was ITSELF half-isolated -- it "
               "refused the run and mutated production on the way out")
            _argv = json.loads((Path(_ltd) / "argv.json").read_text())
            ok("--freeze-epoch" in _argv,
               "(1) the LAUNCHER passes --freeze-epoch explicitly (no default "
               "exists any more, so an omission would refuse at 00:06Z)")
            ok("--write-reason" in _argv,
               "(1) the LAUNCHER passes --write-reason explicitly -- a "
               "canonical production write refuses without one, so an "
               "omission here would break the nightly run rather than write "
               "an unattributed artifact")
            _wr = _argv[_argv.index("--write-reason") + 1]
            ok(_wr.startswith("UNATTRIBUTED"),
               f"(1) and a HAND run of the launcher stamps UNATTRIBUTED (got "
               f"{_wr[:40]!r}) -- the unit check reads /proc/self/cgroup, not "
               f"INVOCATION_ID, which is INHERITED by every child of any "
               f"systemd unit and made a hand rehearsal call itself the timer")
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
               "gap_start_ns": int((_lo9 + 20) * 1e9),
               "gap_end_ns": int((_lo9 + 10) * 1e9)}],
             "an open-at-exit record whose producer end PRECEDES its start"),
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
        # ---- DB2: the PRODUCER's shape must be consumed, not refused -------
        _prod = {"event": "gap_open_at_exit", "coin": "btc", "slug": "s",
                 "window_start": _lo9, "cause": "PING_TIMEOUT",
                 "gap_start_ns": int((_lo9 + 3600) * 1e9),
                 "gap_end_ns": int((_lo9 + 3900) * 1e9),
                 "duration_ms": 300000.0,
                 "note": "outage ran to window end; never reconnected"}
        _d2: dict = {}
        _iv2 = coin_gap_intervals(_lo9, _hi9, "btc",
                                  _wr(_td, [_prod], "producer.jsonl"), diag=_d2)
        ok(len(_iv2) == 1 and abs((_iv2[0][1] - _iv2[0][0]) - 300.0) < 1e-6,
           "DB2: a row in the COMMITTED PRODUCER's exact shape (gap_open_at_exit "
           "WITH a finite task-exit end) is CONSUMED and charged its real 300s "
           "-- refusing it made both suites green while the integration always "
           "refused the moment O1d fired")
        ok(_d2.get("producer_supplied_ends_used") == 1
           and _d2.get("synthesized_ends_charged_to_scope_end") == 0,
           "DB2: the PRODUCER's end is USED and counted as such; synthesis did "
           "NOT fire -- the producer knows when its task exited, the consumer "
           "does not")
        _noend = dict(_prod); _noend.pop("gap_end_ns")
        _d3: dict = {}
        _iv3 = coin_gap_intervals(_lo9, _hi9, "btc",
                                  _wr(_td, [_noend], "noend.jsonl"), diag=_d3)
        ok(_d3.get("synthesized_ends_charged_to_scope_end") == 1
           and _iv3 and _iv3[0][1] == float(_hi9),
           "DB2: synthesis is the FALLBACK, firing only when the producer "
           "supplied no end at all, and the scope end is recorded")

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
    _sv = split_verdict(_healthy_early, era_admissible=True, day_closed=True)
    ok(_sv["day_quality_pass"] is True and _sv["race_accrual_eligible"] is False,
       "a day STRADDLING the freeze is day-quality GOOD but does NOT accrue -- "
       "the epoch's whole job, and reporting one number for both would make a "
       "good-but-early day read as a bad day")
    _healthy_late = [dict(x, **({"pass": True} if x["predicate"] == ACCRUAL_PREDICATE else {}))
                     for x in _healthy_early]
    ok(split_verdict(_healthy_late, era_admissible=True,
                     day_closed=True)["race_accrual_eligible"] is True,
       "a healthy day ENTIRELY AFTER the freeze DOES accrue (positive control)")
    _sick_late = [{"predicate": "gap_rate_under_bar", "pass": False},
                  {"predicate": ACCRUAL_PREDICATE, "pass": True}]
    # ---- FINISHED is the fourth conjunct, and it is not implied by the others
    # `complete_tape` measures against the windows elapsed SO FAR, so it PASSES
    # mid-day. Without this, a four-hour-old day read eligible, and the nightly
    # -- which verdicts the just-OPENED day as well as the closed one -- would
    # have written eligible for a SIX-MINUTE-OLD day. KNOWN-BAD: identical
    # inputs, only closure flipped.
    ok(split_verdict(_healthy_late, era_admissible=True,
                     day_closed=False)["race_accrual_eligible"] is False,
       "an UNFINISHED day does NOT accrue, on inputs that accrue when closed "
       "-- every quality bar passes and it is still not a day yet")
    ok(split_verdict(_healthy_late, era_admissible=True,
                     day_closed=False)["day_quality_pass"] is True,
       "and its QUALITY verdict is untouched by that -- an unfinished day is "
       "not a bad day, exactly as a pre-freeze day is not a bad day")
    _nod = ""
    try:
        split_verdict(_healthy_late, era_admissible=True)
    except ValueError as _e:
        _nod = str(_e)
    ok("needs an explicit day_closed" in _nod,
       "and OMITTING closure REFUSES rather than defaulting -- absence of a "
       "check is not a passed check, the same precedent as era_admissible")
    ok(split_verdict(_sick_late, era_admissible=True,
                     day_closed=True)["race_accrual_eligible"] is False,
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

    # ---- R-255(4): CATCH-UP DAY DERIVATION, red-first ------------------
    import tempfile as _tfd

    def _mkdir_verdicts(td, spec):
        """spec: {day_token: True|False|None|"bad"}; True/False = the artifact's
        day_closed_calendar, None = field absent, "bad" = unparseable."""
        for tok, v in spec.items():
            f = Path(td) / f"da_dayverdict_{tok}.json"
            if v == "bad":
                f.write_text("{not json", encoding="utf-8")
            else:
                d = {"day_token": tok}
                if v is not None:
                    d["day_closed_calendar"] = v
                f.write_text(json.dumps(d), encoding="utf-8")

    def _toks(r):
        return [t for t, _ in r["days"]]

    with _tfd.TemporaryDirectory() as td:
        # THE KNOWN-BAD, stated as the thing that must change. Machine down for
        # 08-29 and 08-30, boots 08-31. Verdicts exist through 08-28.
        _mkdir_verdicts(td, {"20260826": True, "20260827": True,
                             "20260828": True})
        _old = ["20260830", "20260831"]          # date -d yesterday, date
        _new = _toks(days_needing_verdict(Path(td), "20260830", "20260831"))
        ok("20260829" not in _old,
           "KNOWN-BAD reproduced: after a TWO-day outage the old rule "
           "(yesterday + today) verdicts 20260830+20260831 and 20260829 is "
           "LOST FOREVER -- the catch-up fires once and its 'yesterday' is the "
           "wrong day")
        ok(_new == ["20260829", "20260830", "20260831"],
           f"and the derived list recovers BOTH missing days: {_new}")

    with _tfd.TemporaryDirectory() as td:
        # POSITIVE CONTROL: tonight's real shape. 08-28's artifact was written
        # while the day was still OPEN, so it is the closed-day run's job.
        _mkdir_verdicts(td, {"20260826": True, "20260827": True,
                             "20260828": False})
        _n = _toks(days_needing_verdict(Path(td), "20260828", "20260829"))
        ok(_n == ["20260828", "20260829"],
           f"NO-GAP POSITIVE CONTROL: a normal morning verdicts exactly "
           f"yesterday+today ({_n}) -- tonight's behaviour is UNCHANGED")

    with _tfd.TemporaryDirectory() as td:
        # THE FLOOR. Verdicts begin 08-26; the tape reaches back to 08-20 and
        # those days are CONSUMED (rule 11). Catch-up must not reach behind
        # the earliest verdict.
        _mkdir_verdicts(td, {"20260826": True, "20260827": True})
        _f = days_needing_verdict(Path(td), "20260828", "20260829")
        ok(_f["floor"] == "20260826"
           and all(t >= "20260826" for t in _toks(_f)),
           f"THE FLOOR IS DERIVED from the earliest existing verdict "
           f"({_f['floor']}), so catch-up fills holes INSIDE the verdicted "
           f"range and can never mint a backlog behind it -- the naive 'tape "
           f"but no verdict' rule would tonight have minted SIX retroactive "
           f"verdicts over CONSUMED days")

    with _tfd.TemporaryDirectory() as td:
        # A PARTIAL artifact for an OLDER day must be re-verdicted once closed.
        _mkdir_verdicts(td, {"20260826": True, "20260827": False,
                             "20260828": True})
        _pt = _toks(days_needing_verdict(Path(td), "20260829", "20260830"))
        ok("20260827" in _pt,
           "a day whose artifact was written while it was still OPEN is "
           "re-verdicted once closed -- otherwise a missed night leaves a "
           "PARTIAL verdict standing as a closed day's final record")
        _mkdir_verdicts(td, {"20260827": True})
        ok("20260827" not in _toks(
               days_needing_verdict(Path(td), "20260829", "20260830")),
           "positive control: once its artifact says the day was closed, it is "
           "NOT re-verdicted (so the rule is not simply 'redo everything')")

    with _tfd.TemporaryDirectory() as td:
        _mkdir_verdicts(td, {"20260826": True, "20260827": "bad",
                             "20260828": None})
        _u = _toks(days_needing_verdict(Path(td), "20260829", "20260830"))
        ok("20260827" in _u and "20260828" in _u,
           "an UNPARSEABLE artifact and one MISSING day_closed_calendar both "
           "need re-verdicting -- a file that cannot say whether it judged a "
           "closed day is not evidence that it did (None is not True)")

    with _tfd.TemporaryDirectory() as td:
        _r0 = days_needing_verdict(Path(td), "20260828", "20260829")
        ok(_toks(_r0) == ["20260828", "20260829"] and _r0["floor"] is None,
           "with NO artifacts at all -- a fresh install -- it falls back to "
           "closed+opened rather than verdicting all of history")

    with _tfd.TemporaryDirectory() as td:
        _spec = {"20260101": True}
        _lots = days_needing_verdict(Path(td.__str__()), "20260301", "20260302")
        _mkdir_verdicts(td, _spec)
        _cap = days_needing_verdict(Path(td), "20260301", "20260302")
        ok(len(_cap["days"]) <= MAX_CATCHUP_DAYS + 2
           and len(_cap["truncated"]) > 0
           and all(t < _cap["days"][0][0] for t in _cap["truncated"]),
           f"a very long outage is CAPPED at {MAX_CATCHUP_DAYS} catch-up days "
           f"and the {len(_cap['truncated'])} dropped days are NAMED in "
           f"`truncated` -- a bounded run that reads as complete is the "
           f"failure a silent cap introduces (rule 4)")

    # ---- the day-closed label, BOTH branches (Q-DA-149 wrinkle) ---------
    ok(closed_label(False, False) == "selector day_closed=False"
       and closed_label(True, True) == "selector day_closed=True",
       "the day-closed flag is ATTRIBUTED to the selector. It printed a bare "
       "`day_closed=False` beside an artifact separately carrying "
       "day_closed_selector AND day_closed_calendar, so a human reading the "
       "one line saw a closed day called open")
    _dis = closed_label(False, True)
    ok("calendar says True" in _dis and "lags the boundary" in _dis,
       f"and a DISAGREEMENT is named in the line itself: {_dis!r}. This branch "
       f"renders only while the two differ -- true at 00:06Z, false an hour "
       f"later -- so testing it inline would have shipped a branch no test "
       f"ever entered, which is how an undefined status reached the tree "
       f"yesterday under a green suite")
    ok("calendar" not in closed_label(True, True),
       "positive control: when they agree the line stays short and says "
       "nothing about the calendar")

    # ---- ERA-ADMISSION GUARD (R-340), red-first ------------------------
    import tempfile as _tfe

    def _ledger(td, rows):
        f = Path(td) / "runs.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return f

    with _tfe.TemporaryDirectory() as _td:
        _mixed = _ledger(_td, [{"collector_schema_version": "clob_v4",
                                "supersedes": "clob_v3_1",
                                "boundary_utc": "2026-08-30T05:30:00Z"}])
        _a30 = day_era_admission("20260830", _mixed)
        ok(_a30["era_pure"] is False
           and _a30["boundaries_inside_day"] == ["2026-08-30T05:30:00Z"]
           and _a30["race_admissible_by_era"] is False,
           "KNOWN-BAD (the 08-30 shape): an era boundary INSIDE the day makes "
           "it era-impure and INADMISSIBLE -- the day's rows come from two "
           "collectors, so no coin's rows are homogeneous")
        # THE EXACT ETH SHAPE: quality-PASSING coin on the mixed-era day
        _pass_preds = [{"predicate": "entirely_post_freeze", "pass": True},
                       {"predicate": "complete_tape", "pass": True},
                       {"predicate": "gap_rate_under_bar", "pass": True}]
        _eth = split_verdict(_pass_preds, "count_bar_v1_frozen",
                             _a30["race_admissible_by_era"],
                             day_closed=True)
        ok(_eth["day_quality_pass"] is True
           and _eth["race_accrual_eligible"] is False,
           "AND THE 08-30 ETH SHAPE ITSELF: a coin that PASSES every quality "
           "predicate on that day is now INELIGIBLE. Before this the same "
           "inputs gave race_accrual_eligible=TRUE, and the whole day read "
           "false only because BTC failed QUALITY -- eligibility wrong for a "
           "reason quality can never see")
        _a29 = day_era_admission("20260829", _mixed)
        ok(_a29["era_pure"] is True
           and _a29["eras_touched"] == ["clob_v3_1"]
           and _a29["race_admissible_by_era"] is False,
           "a day fully inside a NON-admissible era is era-pure but still "
           "ineligible -- purity and admissibility are separate questions")
        _adm = _ledger(_td, [{"collector_schema_version": "clob_v4",
                              "supersedes": "clob_v3_1",
                              "boundary_utc": "2026-08-30T05:30:00Z"},
                             {"collector_schema_version": "clob_v5",
                              "supersedes": "clob_v4",
                              "transitioned": True,
                              "collector_start_recv_ns": 1788159605000000000,
                              "boundary_utc": "2026-08-31T07:00:00Z"}])
        _a901 = day_era_admission("20260901", _adm)
        ok(_a901["era_pure"] and _a901["eras_touched"] == ["clob_v5"]
           and _a901["race_admissible_by_era"] is True,
           "POSITIVE CONTROL: a day fully inside the ADMISSIBLE v5 era is "
           "admissible -- the guard is not simply refusing everything")
        ok(split_verdict(_pass_preds, "count_bar_v1_frozen",
                         _a901["race_admissible_by_era"], day_closed=True
                         )["race_accrual_eligible"] is True,
           "and a quality-passing coin on that day IS eligible, so the guard "
           "gates on era rather than replacing the quality question")
        # ---- the accrual predicate must read the DAY'S era, never a literal
        # `fi.ERA` was `clob_v3_1`, closed 2026-08-30T05:30:01Z, so every later
        # day was absent from the selector and `entirely_post_freeze` failed by
        # construction. Two checks, because either alone is weak: the first
        # proves the era REACHES the loaders, the second proves `verify_day`
        # actually supplies one. A behavioural test alone would pass with
        # `verify_day` still calling the bare constant.
        import warning_window as _WW
        _seen: list[str] = []

        class _FakeFi:
            ERA = "clob_LITERAL_must_not_be_used"

            @staticmethod
            def gaps_by_slug(era):
                _seen.append(("gaps", era))
                return {}

            @staticmethod
            def covered_slugs(era):
                _seen.append(("cov", era))
                return set()

            @staticmethod
            def _archive_paths():
                return {}

            @staticmethod
            def token_map():
                return {}

        _real_fi = _WW.fi
        try:
            _WW.fi = _FakeFi
            _out = _WW.select_holdout(0.0, era="clob_v4_1")
            ok([e for _k, e in _seen] == ["clob_v4_1", "clob_v4_1"]
               and _out["era"] == "clob_v4_1",
               "(B1) an explicit era REACHES both window loaders and is "
               "carried in the result -- the population is named, so a "
               "selector that silently read a dead era is auditable")
            _seen.clear()
            _WW.select_holdout(0.0)
            ok([e for _k, e in _seen] == ["clob_LITERAL_must_not_be_used"] * 2,
               "(B1 CONTROL) omitting it still falls back to the module "
               "constant, so historical consumers are unchanged -- and this "
               "is exactly the path the accrual predicate must NOT take")
        finally:
            _WW.fi = _real_fi
        _src = Path(__file__).read_text(encoding="utf-8")
        _vd = _src[_src.index("def verify_day("):]
        _vd = _vd[:_vd.index("\ndef ")]
        ok("WW.select_holdout(freeze_epoch, era=" in _vd
           and "WW.select_holdout(freeze_epoch)" not in _vd,
           "(B1) verify_day calls the selector WITH an era and never bare -- "
           "a revert to the literal reinstates a predicate that fails for "
           "every day after the era it names, and reads as a data problem")
        ok("fi.gaps_by_slug(fi.ERA)" not in _vd
           and "fi.covered_slugs(fi.ERA)" not in _vd,
           "(B1) and its windows_gap_affected reads the day's OWN eras, not "
           "the literal -- that literal is why era_covered_windows reported 0 "
           "on days holding 288 windows")
        # THE FOURTH CONJUNCT IS ONLY AS GOOD AS WHAT verify_day SUPPLIES.
        # Mutation found this: hardcoding `day_closed=True` at the call site
        # left every split_verdict test green, because they all pass their own
        # value. Enforcing a rule inside the function while the caller feeds it
        # a constant is a FUNCTION fixed and a PATH left open -- the same shape
        # as the era literal two checks above.
        ok(_vd.count("day_closed=now.timestamp() >= hi") == 2
           and "day_closed=True" not in _vd,
           "(RULE) verify_day supplies REAL calendar closure to both the "
           "whole-day and the per-coin split, and never a constant -- "
           "otherwise an unfinished day accrues no matter what the rule says")

        _a831 = day_era_admission("20260831", _adm)
        ok(_a831["era_pure"] is False
           and _a831["race_admissible_by_era"] is False,
           "BOUNDARY-DAY KNOWN-BAD: 08-31 carries the v5 transition at 07:00Z, "
           "so it is mixed v4->v5 and ineligible -- and its quality verdict is "
           "still produced and PRESERVED as honest v4-storm evidence")
        _unk = _ledger(_td, [{"collector_schema_version": "clob_v9",
                              "supersedes": "clob_v4",
                              "transitioned": True,
                              "boundary_utc": "2026-08-30T05:30:00Z"}])
        _refused = ""
        try:
            day_era_admission("20260901", _unk)
        except ValueError as e:
            _refused = str(e)
        ok("NO ruled admissibility" in _refused,
           "an UNRULED era REFUSES: a collector version is not admissible by "
           "default and silence is not a ruling")
        _none = ""
        try:
            split_verdict(_pass_preds, "count_bar_v1_frozen", None,
                          day_closed=True)
        except ValueError as e:
            _none = str(e)
        ok("needs an explicit era_admissible" in _none,
           "and OMITTING the era question REFUSES -- eligibility must not be "
           "obtainable by not asking, which is how the field got its value "
           "before the guard existed")

    # ---- TRANSITION/ROLLBACK RECEIPT STATE MACHINE (Codex V5-0700-R4) ----
    # An ATTEMPT IS NOT AN ERA. Every row below is a stage of the real v5
    # runbook, executed through the real consumer.
    _B = "2026-08-31T07:00:00Z"
    _ns = lambda s: int(dt.datetime.fromisoformat(
        s.replace("Z", "+00:00")).timestamp() * 1e9)
    _v4 = {"collector_schema_version": "clob_v4", "supersedes": "clob_v3_1",
           "boundary_utc": "2026-08-30T05:30:00Z"}          # the LEGACY row
    # Every transitioned fixture now carries the process evidence the contract
    # requires: a start 5 s after its own boundary, an ordinary restart.
    _v5at = lambda b: {"collector_schema_version": "clob_v5",
                       "supersedes": "clob_v4", "boundary_utc": b,
                       "transitioned": True,
                       "collector_start_recv_ns": _ns(b) + 5 * 10**9}
    _v5 = _v5at(_B)
    _ab = lambda st: {"collector_schema_version": "clob_v5", "boundary_utc": _B,
                      "aborted": True, "stage": st}
    _rb = {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
           "boundary_utc": "2026-08-31T07:20:00Z",
           "rollback": True, "stage": "counters_refused", "closes_boundary_utc": _B,
           "collector_start_recv_ns": _ns("2026-08-31T07:20:00Z") + 4 * 10**8}

    def _refusal(rows, day="20260901"):
        with _tfe.TemporaryDirectory() as t:
            try:
                day_era_admission(day, _ledger(t, rows))
                return ""
            except ValueError as e:
                return str(e)

    def _adm_of(rows, day="20260901"):
        with _tfe.TemporaryDirectory() as t:
            return day_era_admission(day, _ledger(t, rows))

    _a = _adm_of([_v4, _ab("restart_failed")])
    ok(_a["eras_touched"] == ["clob_v4"]
       and _a["race_admissible_by_era"] is False,
       "KNOWN-BAD, THE EXECUTED ONE (Codex V5-0700-R4a): the runbook's own "
       "restart-failed row -- a v5 that NEVER STARTED -- must not mint an era. "
       "Before this it returned eras_touched=['clob_v5'], era_pure=True, "
       "race_admissible=TRUE for every later day")
    ok(_adm_of([_v4, _v5])["race_admissible_by_era"] is True,
       "POSITIVE CONTROL on the same ledger: flip that one row to "
       "'transitioned' and 09-01 IS admissible -- the guard reads what the row "
       "ASSERTS, it does not just count rows or refuse everything")
    ok("asserts NO state" in _refusal(
        [_v4, {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
               "boundary_utc": _B}]),
       "and a row that asserts NOTHING refuses: an absent field is not "
       "'transitioned'. Reading `aborted` alone would have admitted this "
       "row -- the same defect one layer down")
    _two = _refusal([_v4, dict(_v5, aborted=True)])
    ok("at once" in _two and "transitioned" in _two and "aborted" in _two,
       "and a row asserting TWO states refuses -- one attempt, one outcome")
    ok("names no `stage`" in _refusal([_v4, {"collector_schema_version":
        "clob_v5", "boundary_utc": _B, "aborted": True}]),
       "an abort that will not say WHERE it stopped is not auditable")
    ok("asserts NO state" in _refusal(
        [_v4, dict(_v4, boundary_utc="2026-09-01T05:30:00Z")], "20260902"),
       "the legacy exemption is pinned BY IDENTITY, not by position: a second "
       "field-identical row at a different boundary still refuses, so no new "
       "row can inherit the exemption")

    ok("AMBIGUOUS attempt state" in _refusal([_v4, _v5, _ab("counters_refused")])
       and "LIVE since" in _refusal([_v4, _v5, _ab("counters_refused")]),
       "KNOWN-BAD (V5-0700-R4b), AND IT IS THE RUNBOOK'S OWN STAGE-4 "
       "INSTRUCTION: once v5 is stamped LIVE, appending an `aborted` row does "
       "not retract it -- the era stays open and DA calls every later day v5 "
       "forever. The abort path cannot describe a transition that RAN")
    _rbk = _adm_of([_v4, _v5, _rb])
    ok(_rbk["eras_touched"] == ["clob_v4"]
       and _rbk["race_admissible_by_era"] is False,
       "VERIFIED ROLLBACK: a rolled_back row carrying a restoration receipt "
       "closes the v5 era, and later days are v4 again -- inadmissible, which "
       "is the whole point of closing it")
    _b31 = _adm_of([_v4, _v5, _rb], "20260831")
    ok(_b31["eras_touched"] == ["clob_v4", "clob_v5"]
       and len(_b31["boundaries_inside_day"]) == 2,
       "and BOTH REAL BOUNDARIES SURVIVE on the day it happened -- the v5 era "
       "existed and is not erased by being reverted; 08-31 shows both")
    ok("ZERO WIDTH" in _refusal([_v4, _v5, dict(_rb, boundary_utc=_B)]),
       "KNOWN-BAD AGAINST THE EMITTER AS COMMITTED (bc854d3): its receipt sets "
       "boundary_utc = BOUNDARY_UTC, the same instant it closes, so the v5 era "
       "opens and shuts at one timestamp and the span that really ran vanishes. "
       "The rollback's boundary is the RESTORATION instant")
    ok("carries no verified restoration receipt" in _refusal(
        [_v4, _v5, {k: v for k, v in _rb.items()
                    if k != "collector_start_recv_ns"}]),
       "a rollback with NO receipt refuses: without one nothing shows v4 came "
       "back, and DA would name later days from a version that may not be "
       "running")
    ok("collector_start BEFORE its own boundary" in _refusal([_v4, _v5, dict(
        _rb, collector_start_recv_ns=_ns("2026-08-31T06:00:00Z"))]),
       "a STALE receipt refuses -- a collector_start from before the restart "
       "proves nothing about it. This now fires one layer EARLIER, at the row "
       "itself, and the old era-level check was SUBSUMED and DELETED rather "
       "than kept as unreachable code: a start >= this row's boundary, which "
       "the zero-width check already forces past the closed era's boundary")
    ok("must NAME the transition it reverts" in _refusal([_v4, _v5, dict(_rb,
        closes_boundary_utc="2026-08-30T05:30:00Z")]),
       "and a rollback pointing at the wrong era refuses")
    ok("AMBIGUOUS attempt state" in _refusal([_v4, _v5,
        {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
         "boundary_utc": "2026-08-31T07:20:00Z",
         "transitioned": True}]),
       "UNDECLARED ROLLBACK refuses: a plain return to the previous version "
       "cannot be told apart from a fresh deploy of it")

    # Coverage found by mutation: each of these refusals could be DELETED with
    # the suite still green, so none of them was a check yet.
    ok("lacks boundary_utc" in _refusal(
        [{"collector_schema_version": "clob_v5", "transitioned": True}]),
       "a row with no boundary_utc refuses -- an era with no start instant "
       "cannot bound any day")
    ok("era ledger is EMPTY" in _refusal([]),
       "an EMPTY ledger refuses rather than reading as 'no boundaries, so the "
       "day is pure' -- absence of a recorded era is not purity")
    ok("NOT ONE EFFECTIVE TRANSITION" in _refusal([_ab("restart_failed")]),
       "a ledger of ONLY attempts refuses: rows exist, none of them ran, so no "
       "era can be named -- the case that looks least like an error and most "
       "like a quiet pass")
    ok("closes nothing" in _refusal([_rb]),
       "a rollback with no open era refuses -- it cannot revert a transition "
       "that is not in the ledger")
    ok("names no `stage`" in _refusal(
        [_v4, _v5, {k: v for k, v in _rb.items() if k != "stage"}]),
       "and a rollback with no `stage` refuses. Mutation found the era_spans "
       "copy of this check to be DEAD -- _ledger_rows already refuses every "
       "stage-less rollback -- so it was deleted; this exercises the live one")
    # The PURITY conjunct is invisible to a refusal-only audit and, with the
    # real ruled table, unfalsifiable: every mixed-era day happens to sort a
    # NON-admissible era first, so dropping `pure` changes no verdict. That is
    # an accident of the table's contents, not a property of the guard -- the
    # moment a later era is ruled admissible, an impure day between two
    # admissible eras would be admitted. Tested against a synthetic table.
    _two_ok = {"clob_v4": True, "clob_v5": True}
    _mix = _adm_of_tbl = None
    with _tfe.TemporaryDirectory() as _t2:
        _mixed_adm = _ledger(_t2, [_v4, _v5])
        _impure = day_era_admission("20260831", _mixed_adm, _two_ok)
        ok(_impure["era_pure"] is False
           and _impure["era_admissible_ruled"] == {"clob_v4": True,
                                                   "clob_v5": True}
           and _impure["race_admissible_by_era"] is False,
           "PURITY IS LOAD-BEARING ON ITS OWN: a day straddling TWO eras that "
           "are BOTH ruled admissible is still inadmissible. The day's rows "
           "still come from two collectors -- admissibility of both endpoints "
           "does not make the day homogeneous")
        ok(day_era_admission("20260901", _mixed_adm, _two_ok
                             )["race_admissible_by_era"] is True,
           "and against the SAME table a PURE day in one of those eras is "
           "admissible, so the check is purity and not blanket refusal")
    # ---- RECOVERY BUNDLE (Codex V5-R3B residual) -----------------------
    # A v5 that RAN but could never be stamped. An `aborted` row cannot encode
    # it -- the span existed -- and it is NOT a rollback of a row that does not
    # exist. TWO rows, in order: the observed transition marked `recovered`,
    # then the standard rollback receipt closing it.
    _recv = dict(_v5, recovered=True, stage="stamp_unwritable_recovery",
                 collector_start_recv_ns=_ns("2026-08-31T07:00:04Z"))
    _bundle = [_v4, _recv, _rb]
    _b = _adm_of(_bundle, "20260901")
    ok(_b["eras_touched"] == ["clob_v4"]
       and _b["race_admissible_by_era"] is False,
       "RECOVERY BUNDLE lands: the v5 span that ran is recorded, closed by its "
       "rollback, and later days are v4 again")
    _bd = _adm_of(_bundle, "20260831")
    ok(_bd["eras_touched"] == ["clob_v4", "clob_v5"]
       and len(_bd["boundaries_inside_day"]) == 2
       and _bd["era_reconstructed"] is True,
       "and BOTH real boundaries survive on the day it happened -- the span is "
       "recorded rather than erased, and flagged RECONSTRUCTED")
    ok(_adm_of([_v4, dict(_recv, boundary_utc="2026-08-30T06:00:00Z",
                          collector_start_recv_ns=_ns("2026-08-30T06:00:04Z")),
                dict(_rb, closes_boundary_utc="2026-08-30T06:00:00Z",
                     boundary_utc="2026-09-03T00:00:00Z",
                     collector_start_recv_ns=_ns("2026-09-03T00:00:04Z"))],
               "20260901")["race_admissible_by_era"] is False,
       "A DAY LYING WHOLLY INSIDE A RECOVERED v5 SPAN DOES NOT ACCRUE. Its "
       "boundary is a CLAIM about the past, not a stamp made at the time, and "
       "era purity is a contemporaneous predicate. Conservative DEFAULT, not a "
       "ruling -- relaxing it is a policy call (rule 14)")
    ok("never CLOSED" in _refusal([_v4, _recv]),
       "KNOWN-BAD (recovered-without-rollback): an unclosed recovered row "
       "refuses. It would say v5 is still live yet was never stampable -- a "
       "state the runbook cannot reach, and the shape a HALF-WRITTEN bundle "
       "leaves behind when the ledger goes unwritable again mid-append")
    ok("must NAME the transition it reverts" in _refusal([_v4, _rb]),
       "KNOWN-BAD (rollback-without-recovered): a rollback naming a v5 "
       "boundary that was NEVER WRITTEN refuses -- Codex's 'do not encode that "
       "case as a rollback of a row that does not exist', enforced. (A "
       "rollback as the very FIRST row hits a different refusal, `closes "
       "nothing`, covered separately -- two distinct shapes, two messages)")
    ok("OUT OF ORDER" in _refusal([_v4, _rb, _recv]),
       "KNOWN-BAD (out-of-order bundle): the bundle's two rows APPENDED in the "
       "wrong order -- rollback at 07:20 written before the 07:00 transition "
       "it closes -- refuses. (My first fixture for this was wrong: I dated "
       "the rollback EARLIER, which is chronologically consistent and trips a "
       "different check. The append order is what the bundle must get right)")
    ok("is `recovered` but asserts" in _refusal(
        [_v4, _v5, {k: v for k, v in dict(_rb, recovered=True).items()}]),
       "`recovered` on a rollback refuses: recovery records a transition that "
       "HAPPENED, and an abort or rollback is not a thing to recover")
    ok("names no `stage`" in _refusal(
        [_v4, {k: v for k, v in _recv.items() if k != "stage"}, _rb]),
       "a recovered row with no `stage` refuses -- a retroactive boundary must "
       "say WHY it could not be stamped at the time. (The generic stage check "
       "covers aborted and rollback rows only; a `transitioned` row needs "
       "none, so recovery carries its own -- and mutation found this one "
       "untested before it shipped)")
    ok("carries no `collector_start_recv_ns`" in _refusal(
        [_v4, {k: v for k, v in _recv.items()
               if k != "collector_start_recv_ns"}, _rb]),
       "a recovered row with no process evidence refuses -- without the v5 "
       "process's own start there is nothing to show the span ran at all")
    ok("collector_start BEFORE its own boundary" in _refusal(
        [_v4, dict(_recv, collector_start_recv_ns=_ns("2026-08-31T06:00:00Z")),
         _rb]),
       "and one whose collector_start predates its own boundary refuses -- "
       "that process was already running before the era it claims to open. "
       "The recovery-specific copy of this check was DELETED as subsumed: "
       "generalising it to every row made two special-case checks unreachable")
    ok(_adm_of([_v4, _v5])["race_admissible_by_era"] is True
       and _adm_of([_v4, _v5])["era_reconstructed"] is False,
       "POSITIVE CONTROL: a NORMALLY stamped v5 era is admissible and not "
       "flagged reconstructed -- recovery is the exception, not the default")

    _self = _refusal([_v4, dict(_v5, supersedes="clob_v5")])
    # ---- TAPE DENSITY: reported, never governing (BE R-362) ------------
    _dense_fail = [{"predicate": "complete_tape", "pass": True},
                   {"predicate": "tape_density", "pass": False}]
    ok(compose_all_pass(_dense_fail, {}, _pass_bar, "day_bar_v2") is True,
       "LOAD-BEARING: a day FAILING tape_density with everything else passing "
       "still has all_pass TRUE. The 5%-of-median threshold was chosen after "
       "seeing which days fail it -- 7 of 13, including 08-29, the only "
       "all_pass day -- so letting it veto would re-judge judged days on a bar "
       "taken from their own data (rule 11). The hole is real; retro-fitting a "
       "bar to it is not a measurement")
    ok(compose_all_pass([{"predicate": "complete_tape", "pass": False},
                         {"predicate": "tape_density", "pass": True}],
                        {}, _pass_bar, "day_bar_v2") is False,
       "POSITIVE CONTROL: a GOVERNING predicate still decides -- "
       "governing_predicates is filtering one name, not draining the set")
    with _tfe.TemporaryDirectory() as _t6:
        _dp = Path(_t6) / "density.json"
        ok(tape_density_for("20260829", _dp)["status"] == "UNMEASURED",
           "an ABSENT receipt is UNMEASURED, never a clean zero -- absence of "
           "measurement is not evidence of density, and the empty-set-passes "
           "trap already fired once on the 08-27 arm")
        _dp.write_text(json.dumps({"days": [{"day": "20260829", "coins": {},
                                    "total_thin_invisible": 0,
                                    "total_thin_accounted": 0,
                                    "threshold_frac_of_median": 0.05}],
                                   "threshold_sensitivity": []}))
        ok(tape_density_for("20260830", _dp)["status"] == "UNMEASURED",
           "and a receipt that covers OTHER days is UNMEASURED for this one -- "
           "a non-empty file is not coverage of the day asked about")
        ok(tape_density_for("20260829", _dp)["n_invisible_at_threshold"] == 0
           and tape_density_for("20260829", _dp)["governs"] is False,
           "POSITIVE CONTROL: a covered day reads MEASURED with its count, and "
           "says in the artifact that it does not govern")
        _dp.write_text("{not json")
        ok(tape_density_for("20260829", _dp)["status"] == "UNMEASURED",
           "an unreadable receipt is UNMEASURED rather than an exception -- a "
           "reported diagnostic must not take the whole verdict down with it")

    with _tfe.TemporaryDirectory() as _t7:
        _dp2 = Path(_t7) / "d.json"
        _dp2.write_text(json.dumps([{"day": "20260829", "coins": {},
                                     "total_thin_invisible": 10}]))
        _sr = tape_density_for("20260829", _dp2)
        ok(_sr["status"] == "SCHEMA_UNRECOGNISED"
           and "NOT the same as no measurement" in _sr["why"],
           "KNOWN-BAD, AND IT HAPPENED: the receipt changed from a LIST of day "
           "rows to {days, note, threshold_sensitivity}. Iterating a dict "
           "yields its KEYS, no row matched, and this reported UNMEASURED -- "
           "'no measurement for this day' while the measurement sat in the "
           "file. A reader that turns a schema change into ABSENCE is worse "
           "than one that crashes, because absence is a plausible answer")
        _dp2.write_text(json.dumps({"days": [], "threshold_sensitivity": []}))
        ok(tape_density_for("20260829", _dp2)["status"] == "UNMEASURED",
           "and a receipt in the RIGHT shape that covers no days is UNMEASURED "
           "-- the two statuses must not collapse into each other")
        _dp2.write_text(json.dumps({"days": [
            {"day": "20260829", "coins": {}, "total_thin_invisible": 10,
             "total_thin_accounted": 0, "threshold_frac_of_median": 0.05}],
            "threshold_sensitivity": [
                {"threshold_frac_of_median": 0.01,
                 "days_with_invisible_loss": 7, "invisible_windows": 249},
                {"threshold_frac_of_median": 0.10,
                 "days_with_invisible_loss": 7, "invisible_windows": 749}]}))
        _ok = tape_density_for("20260829", _dp2)
        ok(_ok["status"] == "MEASURED"
           and _ok["n_invisible_at_threshold"] == 10
           and _ok["day_count_stable_over"] == [0.01, 0.10],
           "POSITIVE CONTROL: a covered day reads MEASURED, and the count is "
           "named AT ITS THRESHOLD with the range over which the DAY verdict "
           "is stable. BE asserted threshold-insensitivity without computing "
           "it and the window set moves by 114 between 0.05 and 0.25 -- a bare "
           "count would be the conclusion-beside-a-number this instrument "
           "exists to refuse")

    # ---- CONTENT LIVENESS: structure frozen, thresholds absent ---------
    with _tfe.TemporaryDirectory() as _t8:
        _lg = Path(_t8) / "c.log"
        _anchor = 1788220800.0            # 2026-09-01T00:00:00Z
        _lg.write_text("".join(
            f"[pm] {(23*3600 + 50*60 + i*60) // 3600 % 24:02d}:"
            f"{((23*3600 + 50*60 + i*60) // 60) % 60:02d}:00Z markets=1 "
            f"msgs={1000 + i*60000}\n" for i in range(20)))
        _cl = content_liveness_for("20260831", _lg, _anchor)
        ok(_cl["status"] == "CONTENT_LIVENESS_UNRESOLVED"
           and _cl.get("n_intervals", 0) >= 1,
           "heartbeats spanning MIDNIGHT are dated by walking backward from an "
           "anchor -- the log stamps HH:MM:SSZ with NO DATE, so a line's day "
           "is reconstructed, never assumed")
        ok(all(x in CONTENT_LIVENESS_STATUSES
               for x in (_cl["status"],)),
           "and the status is drawn from the DECLARED four, not invented")
        _lg.write_text("[pm] 12:00:00Z markets=1 msgs=100\n"
                       "[pm] 12:00:00Z markets=1 msgs=200\n")
        _amb = content_liveness_for("20260831", _lg, _anchor)
        ok(_amb["status"] == "CONTENT_LIVENESS_UNRESOLVED"
           and "unknowable" in _amb["why"],
           "KNOWN-BAD: two stamps resolving to the SAME instant refuse -- "
           "their order in time is unknowable and a misdated heartbeat "
           "attributes one day's traffic to another. (My first fixture for "
           "this stepped the stamps backward, which the backward walk resolves "
           "cleanly: it tested nothing. And a >24h silent gap is NOT "
           "detectable from dateless stamps at all -- that limit is DECLARED "
           "in the docstring rather than guarded, because a guard that cannot "
           "fire is not a guard)")
        _lg.write_text("[pm] 12:00:00Z markets=1 msgs=100\n")
        ok("PROSPECTIVE by construction" in
           content_liveness_for("20260831", _lg, _anchor)["why"],
           "a day with fewer than two heartbeats is UNRESOLVED and SAYS the "
           "measure is prospective -- the log does not reach back, so absence "
           "of evidence is reported as absence, never as liveness")
        ok(content_liveness_for("20260831", Path(_t8) / "nope.log",
                                _anchor)["status"]
           == "CONTENT_LIVENESS_UNRESOLVED",
           "and no log at all is UNRESOLVED rather than an exception")
        _mixed_log = Path(_t8) / "mixed.log"
        _mixed_log.write_text(
            "[pm] 21:00:23Z markets=1 msgs=10\n"
            "[pm] 21:01:23Z markets=1 msgs=20\n"
            "[pm] 2026-08-31T22:00:23Z markets=1 msgs=100\n"
            "[pm] 2026-08-31T22:01:23Z markets=1 msgs=200\n")
        _mx = content_liveness_for("20260831", _mixed_log, _anchor)
        ok(_mx.get("n_intervals") == 3,
           "THE TRANSITION NIGHT ITSELF: a log that is DATELESS before 22:00Z "
           "and DATED after parses BOTH halves. Tonight is the only night this "
           "shape exists, and it is the night before the day this measure "
           "exists to judge")
        _fmt = Path(_t8) / "fmt.log"
        _fmt.write_text("[pm] 22:00:23Z markets=1 nomsgs=1\n"
                        "[pm] 22:01:23Z markets=1 nomsgs=2\n")
        _fr = content_liveness_for("20260831", _fmt, _anchor)
        ok(_fr["status"] == "CONTENT_LIVENESS_UNRESOLVED"
           and "FORMAT CHANGE" in _fr["why"],
           "KNOWN-BAD: a log with [pm] lines where NONE matches the heartbeat "
           "shape is reported as a FORMAT CHANGE, never as 'the log does not "
           "reach back'. My regex required a DATELESS stamp; the status line "
           "gains a full ISO date at 22:00Z tonight, so it would have matched "
           "ZERO lines and reported the day as unmeasurable history -- a "
           "format change read as ABSENCE, which is the density-receipt defect "
           "arriving with four hours' warning instead of after the fact")

    _real = content_liveness_for("20260831")
    ok(_real["status"] == "CONTENT_LIVENESS_UNRESOLVED"
       and _real.get("median_msgs_per_s") is not None,
       "LOAD-BEARING: on the REAL 08-31 log the quantities are COMPUTED and "
       "the status is still UNRESOLVED. The discriminator is frozen and the "
       "band is not mine to set -- a rule that classified here would be "
       "choosing a bar from the day that motivated it")
    ok(_real["fraction_of_day_below_10pct"] > 0.0
       and content_liveness_for("20260829")["fraction_of_day_below_10pct"]
       == 0.0,
       "POSITIVE CONTROL: the measure SEPARATES -- 08-31 (the 4.1 h event) "
       "shows a quarter of the day under a tenth of its own median rate while "
       "08-29 shows none. It discriminates without classifying, which is "
       "exactly the state a pre-registered rule should be in before its bar "
       "is ratified")

    # ---- ERA START = PROCESS START (BE Q-DA-180 item 1) ----------------
    _late = {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
             "transitioned": True, "boundary_utc": "2026-09-01T00:00:00Z",
             "collector_start_recv_ns": _ns("2026-09-01T00:00:00Z")
                                        + 119 * 10**9}
    _tblv5 = {"clob_v3_1": False, "clob_v4": False, "clob_v5": True}
    with _tfe.TemporaryDirectory() as _t5:
        _l = _ledger(_t5, [_v4, _late])
        _a = day_era_admission("20260901", _l, _tblv5)
        ok(_a["era_pure"] is False and _a["race_admissible_by_era"] is False,
           "KNOWN-BAD: a boundary ruled at 00:00:00Z with an ORDINARY 119 s "
           "restart was era_pure and RACE-ADMISSIBLE, while the row's OWN "
           "collector_start says the old version served the day's first 119 s. "
           "collector_start_recv_ns was validated on `recovered` and "
           "`rollback` rows but NOT on plain `transitioned` ones -- the only "
           "kind that OPENS an admissible era")
        _ctl = _ledger(_t5, [_v4, dict(
            _late, boundary_utc="2026-08-31T23:58:00Z",
            collector_start_recv_ns=_ns("2026-08-31T23:58:00Z") + 119 * 10**9)])
        _c = day_era_admission("20260901", _ctl, _tblv5)
        ok(_c["era_pure"] is True and _c["race_admissible_by_era"] is True,
           "POSITIVE CONTROL (BE's, and it must keep passing): 23:58:00Z + "
           "119 s = 23:59:59Z is SAME-DAY, so 09-01 is genuinely pure. The "
           "rule is not 'restarts are suspicious', it is that the era begins "
           "when the PROCESS begins")
        _noev = day_era_admission("20260901", _ledger(_t5, [_v4, {
            "collector_schema_version": "clob_v5", "supersedes": "clob_v4",
            "transitioned": True, "boundary_utc": "2026-08-31T07:00:00Z"}]),
            _tblv5)
        ok(_noev["era_unevidenced_start"] is True
           and _noev["race_admissible_by_era"] is False,
           "and a transitioned row with NO process evidence cannot accrue: "
           "its purity is UNVERIFIABLE, and assuming boundary == start is "
           "exactly the assumption that produced the false accept")
    ok("not a positive int" in _refusal(
        [_v4, dict(_v5, collector_start_recv_ns=1.788e18)]),
       "a non-int collector_start refuses -- float64 cannot hold a nanosecond "
       "epoch exactly, and this is the evidence field the era start is read "
       "from")
    ok("restart delays cannot reorder the chain" in _refusal(
        [_v4, {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
               "transitioned": True, "boundary_utc": "2026-08-31T08:00:00Z",
               "collector_start_recv_ns": _ns("2026-08-31T08:00:00Z")
                                          + 60 * 10**9},
         {"collector_schema_version": "clob_v6", "supersedes": "clob_v5",
          "transitioned": True, "boundary_utc": "2026-08-31T08:00:30Z",
          "collector_start_recv_ns": _ns("2026-08-31T08:00:30Z")}]),
       "and an effective start landing BEFORE the era before it refuses: "
       "boundaries order the chain, but restart delays must not silently "
       "reorder the spans built from them")

    # ---- MULTI-HOP RETURN (BE V5-P5-1) ---------------------------------
    _T = lambda v, sup, b: {"collector_schema_version": v, "supersedes": sup,
                            "transitioned": True, "boundary_utc": b,
                            "collector_start_recv_ns": _ns(b) + 5 * 10**9}
    _hop = [_v4, _T("clob_v5", "clob_v4", "2026-08-31T07:00:00Z"),
            _T("clob_v6", "clob_v5", "2026-08-31T08:00:00Z"),
            _T("clob_v4", "clob_v6", "2026-08-31T09:00:00Z")]
    _hm = _refusal(_hop)
    ok("ALREADY HELD an era" in _hm and "clob_v4" in _hm,
       "KNOWN-BAD (V5-P5-1, BE's rule adopted): v4->v5->v6->v4 with no "
       "rollback evidence was ACCEPTED here -- I only refused a return to the "
       "IMMEDIATELY PREVIOUS era. The harm is not the hop count: a plain "
       "transitioned row skips the ENTIRE rollback contract, and multi-hop is "
       "worse in one way -- after two hops nobody remembers which era the "
       "missing evidence would have described")
    _tbl6 = {"clob_v3_1": False, "clob_v4": False, "clob_v5": True,
             "clob_v6": True}
    with _tfe.TemporaryDirectory() as _t4:
        _ok_hop = _ledger(_t4, [_v4,
                                _T("clob_v5", "clob_v4", "2026-08-31T07:00:00Z"),
                                _T("clob_v6", "clob_v5", "2026-08-31T08:00:00Z"),
                                {"collector_schema_version": "clob_v4",
                                 "supersedes": "clob_v6", "rollback": True,
                                 "stage": "counters_refused",
                                 "closes_boundary_utc": "2026-08-31T08:00:00Z",
                                 "boundary_utc": "2026-08-31T09:00:00Z",
                                 "collector_start_recv_ns":
                                     _ns("2026-08-31T09:00:04Z")}])
        ok(day_era_admission("20260902", _ok_hop, _tbl6
                             )["race_admissible_by_era"] is False,
           "POSITIVE CONTROL: the SAME multi-hop shape WITH a rollback receipt "
           "is accepted -- the rule demands evidence, it does not forbid "
           "returning")
    ok(_adm_of([_v4, _v5, _rb, _v5at("2026-09-01T07:00:00Z")],
               "20260902")["race_admissible_by_era"] is True,
       "AND THE RETRY EXEMPTION SURVIVES: a v5 retry after its own verified "
       "rollback returns to a previously-in-force version and is still "
       "allowed. Read literally, 'any previous version is a return' would "
       "have broken this -- the era now open was RESTORED by a rollback, so "
       "the evidence already exists")

    _torn = _refusal([_v4], "20260901") or ""
    with _tfe.TemporaryDirectory() as _t3:
        _f = Path(_t3) / "runs.jsonl"
        _good = json.dumps(_v5)
        for _label, _text, _row in (
                ("truncated mid-object", json.dumps(_v4) + "\n" + _good[:60], 2),
                ("two objects concatenated",
                 json.dumps(_v4) + "\n" + _good + _good, 2)):
            _f.write_text(_text, encoding="utf-8")
            _msg = ""
            try:
                day_era_admission("20260901", _f)
            except ValueError as e:
                _msg = str(e)
            ok("is not valid JSON" in _msg and f"row {_row}" in _msg
               and "OF THAT ROW" in _msg,
               f"KNOWN-BAD ({_label}): a malformed ledger line refuses BY NAME "
               f"and names the REAL row. It used to leak json's own "
               f"JSONDecodeError, whose 'line 1 column N' is the FRAGMENT's "
               f"coordinate -- an operator reads that as ledger line 1 and "
               f"looks at the wrong row")

    # ---- differential-fuzz findings (BE, 17,729 ledgers) ---------------
    for _spell in ("2026-08-31T07:00:00.500Z", "2026-08-31T07:00:00+00:00",
                   "2026-08-31 07:00:00Z", "2026-08-31"):
        ok("one accepted spelling is YYYY-MM-DDTHH:MM:SSZ" in _refusal(
            [_v4, dict(_v5, boundary_utc=_spell)]),
           f"KNOWN-BAD: boundary spelled {_spell!r} refuses. All four parsed "
           f"before, and boundaries are compared to each other as RAW STRINGS "
           f"-- so a second spelling of one instant read as a different era, "
           f"and a bare date silently meant midnight, moving a transition by "
           f"hours")
    ok(_adm_of([_v4, _v5])["race_admissible_by_era"] is True,
       "POSITIVE CONTROL: the canonical spelling still passes, and it is the "
       "form the real ledger and the emitter already write")
    ok("canonically SHAPED but not a real instant" in _refusal(
        [_v4, {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
         "transitioned": True, "boundary_utc": "2026-02-31T00:00:00Z"}]),
       "and a canonically-shaped IMPOSSIBLE date refuses with a declared "
       "message -- an unparseable boundary used to leak fromisoformat's own "
       "ValueError, which reads as a crash rather than a refusal")
    _mixed_type = _refusal([_v4, {"collector_schema_version": 5,
                                  "supersedes": "clob_v4", "transitioned": True,
                                  "boundary_utc": "2026-08-31T07:00:00Z"}],
                           "20260831")
    ok("not a string" in _mixed_type and "UNDECLARED exception" in _mixed_type,
       "KNOWN-BAD (a CRASH, not a wrong answer): a non-string era name raised "
       "a bare TypeError from sorted() on any day touching TWO eras -- it was "
       "invisible on a single-era day, where one element never gets compared")
    ok("names no `stage`" in _refusal(
        [_v4, _v5, dict(_rb, stage="   ")]),
       "KNOWN-BAD: a whitespace-only `stage` refuses. It was TRUTHINESS-"
       "checked, so '   ' satisfied a field whose entire purpose is to say "
       "WHY -- a blank reason is not a reason")
    ok("names no `stage`" in _refusal(
        [_v4, {k: v for k, v in dict(_recv, stage=" ").items()}, _rb]),
       "and the same on a recovered transition, where the stage explains why "
       "the boundary could not be stamped at the time")

    _int = _refusal([_v4, dict(_v5, recovered=1)])
    ok("not a bool" in _int and "recovered=1" in _int,
       "KNOWN-BAD (cross-consumer divergence, BE's S7): `recovered: 1` was "
       "ACCEPTED here -- read with `is True` it meant NOT recovered, so the "
       "stage and collector_start_recv_ns burden was WAIVED *and* the era "
       "counted for race accrual while its boundary was a reconstruction. "
       "Worse than the emitter-side shape, which only waived the burden")
    _ab1 = _refusal([_v4, dict(_v5, aborted=1)])
    ok("not a bool" in _ab1,
       "and the SAME defect one flag over, which the report did not name: "
       "`aborted: 1` beside `transitioned: true` slipped past the asserts-two "
       "check and stood as a plain transition. Fixed for EVERY flag, not just "
       "the one that was found")
    ok("not a bool" in _refusal([_v4, dict(_v5, transitioned="yes")]),
       "a string flag refuses too -- a truthy non-bool is not a sloppy yes, it "
       "is a value this contract cannot read")
    ok(_adm_of([_v4, _v5])["race_admissible_by_era"] is True,
       "POSITIVE CONTROL: real bools still pass; the check is on TYPE, not on "
       "the flag's presence")
    ok("supersedes ITSELF" in _self,
       "KNOWN-BAD: a row superseding its OWN version refuses. It transitions "
       "nothing, yet it mints a boundary -- and the direction of the harm is "
       "easy to miss: it makes the day it lands on IMPURE, so it fails SAFE on "
       "admissibility while silently costing a day off the five-day clock")
    ok(_adm_of([_v4, _v5])["race_admissible_by_era"] is True,
       "POSITIVE CONTROL: the same row superseding a DIFFERENT version is "
       "still admitted -- the check targets self-reference, not v5 rows")
    _chain = _refusal([_v4, {"collector_schema_version": "clob_v5",
                             "transitioned": True, "supersedes": "clob_v3_1",
                             "boundary_utc": _B}])
    ok("MALFORMED CHAIN" in _chain and "clob_v3_1" in _chain,
       "KNOWN-BAD, THE EXECUTED ONE (round-2): a v5 row claiming it supersedes "
       "clob_v3_1, appended after clob_v4 has been LIVE since 08-30. It was "
       "ADMITTED and made 09-01 admissible -- `supersedes` was only ever "
       "checked on the rollback path. A receipt must name the era it ACTUALLY "
       "replaces; the consumer does not trust the emitter to know what is "
       "running")
    ok(_adm_of([_v4, _v5])["race_admissible_by_era"] is True,
       "POSITIVE CONTROL for the chain: the SAME row with supersedes=clob_v4 "
       "-- the era really in force -- is admitted, so the check reads the "
       "claim rather than rejecting v5 rows")
    ok("must NAME the transition it reverts" in _refusal(
        [_v4, _v5, dict(_rb, supersedes="clob_v3_1")]),
       "and a ROLLBACK receipt superseding the wrong version refuses too, "
       "separately from its closes_boundary_utc -- both conjuncts of the "
       "rollback's chain identity are load-bearing")
    ok("names no `supersedes`" in _refusal(
        [{"collector_schema_version": "clob_v4", "transitioned": True,
          "boundary_utc": "2026-08-30T05:30:00Z"}]),
       "a first effective row with no `supersedes` refuses: the ledger records "
       "transitions, so the OPENING era has no other name")
    ok(_adm_of([_v4, _ab("restart_failed"),
                _v5at("2026-08-31T09:00:00Z")]
               )["race_admissible_by_era"] is True,
       "RETRY AFTER ABORT is permitted and admissible from the retry's own "
       "boundary -- the guard refuses failed attempts, not second attempts")
    ok(_adm_of([_v4, _v5, _rb, _v5at("2026-09-01T07:00:00Z")],
               "20260902")["race_admissible_by_era"] is True,
       "RETRY AFTER A VERIFIED ROLLBACK is admissible too: the earlier v5 era "
       "was explicitly closed, so returning to v5 is unambiguous")
    ok("OUT OF ORDER" in _refusal([_v4, _v5at("2026-09-02T07:00:00Z"),
                                   _v5at("2026-09-01T07:00:00Z")],
                                  "20260903"),
       "a non-chronological append refuses. The old code SORTED, which both "
       "reordered a rollback against the transition it closes and crashed with "
       "a TypeError when two rows shared a boundary")

    print(f"da_forward_day_verify selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["verify", "days"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--closed", default=None)
    ap.add_argument("--opened", default=None)
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
    ap.add_argument("--supersedes", default=None,
                    help="path of the artifact this run REPLACES. The verdict "
                         "path is a CACHE of the current verdict, not a "
                         "receipt -- the nightly mv -f's over it, which is how "
                         "a pre-guard verdict stood at the canonical path with "
                         "nothing naming its correction. Recording the "
                         "replaced artifact's digest and verdict IN A FIELD "
                         "makes the overwrite resolvable by a reader, which "
                         "prose in a log is not.")
    ap.add_argument("--write-reason", default=None,
                    help="WHO is writing this verdict and WHY. Required when "
                         "writing a CANONICAL production verdict, because the "
                         "verdict path is stable: any later write replaces the "
                         "artifact a previous log line still describes, and a "
                         "reader who finds an unexpected as_of has to "
                         "reconstruct the reason from memory. Recorded in the "
                         "artifact as `write_reason`.")
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if a.cmd == "days":
        # Emits `<day>\t<kind>` lines for the launcher, plus `#` diagnostics it
        # copies into the nightly log. Printing the DERIVATION, not just its
        # result, is what lets a reader of the log see WHY a night verdicted
        # the days it did.
        if not (a.outdir and a.closed and a.opened):
            raise SystemExit("REFUSED: days needs --outdir --closed --opened")
        r = days_needing_verdict(Path(a.outdir), a.closed, a.opened)
        print(f"# floor={r['floor']} {r['why']}")
        if r["truncated"]:
            print(f"# TRUNCATED {len(r['truncated'])} day(s) beyond the "
                  f"{MAX_CATCHUP_DAYS}-day cap, NOT verdicted: "
                  f"{','.join(r['truncated'])}")
        for tok, kind in r["days"]:
            print(f"{tok}\t{kind}")
        return 0
    if not a.day:
        raise SystemExit("--day YYYYMMDD is required; refusing to guess a day")
    # A DAY THAT FAILS AND AN INSTRUMENT THAT BROKE MUST NOT SHARE AN EXIT
    # CODE. An uncaught exception exits 1, and so does a computed FAIL -- so
    # the nightly log could not distinguish "day one is inadmissible" from
    # "the verifier never ran". R-153(2) makes this a HARD PRECONDITION, and a
    # precondition that can silently no-op is the failure it exists to prevent.
    # (e) checked BEFORE the try: an unstated epoch is a launcher error, not an
    # instrument failure, and must not be reported as one.
    # (f) A CANONICAL PRODUCTION WRITE MUST SAY WHO AND WHY. Checked here,
    # beside the epoch check and BEFORE any computation, for the same reason:
    # an unattributed write is a caller error, not an instrument failure. The
    # nightly launcher always supplies one, so this bites only hand runs --
    # exactly the population that needs attributing. Scratch and rehearsal
    # paths are unaffected.
    if a.out:
        _op = Path(a.out).resolve()
        if (_op.parent == CANONICAL_VERDICT_DIR
                and _op.name.startswith("da_dayverdict_")
                and _op.name.endswith(".json") and not a.write_reason):
            raise SystemExit(
                f"REFUSED: {_op.name} is a CANONICAL production verdict and "
                f"--write-reason is required. The path is stable, so this "
                f"write REPLACES the artifact the last log line describes; "
                f"without a stated reason a future reader of `as_of_utc` has "
                f"to reconstruct who regenerated it and why.")
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
    # 0h, ENFORCED AT THE ARTIFACT, not only in the unit that computes it.
    assert_disclosure_carried(rep)
    # Provenance, stamped AFTER the verdict: it is not a predicate and must
    # never enter `all_pass`. Always present, so an unattributed write is a
    # visible STATUS rather than a missing key (rule 4).
    rep["write_reason"] = a.write_reason
    # WHAT THIS RUN REPLACES, as a resolvable FIELD. Always present, so "this
    # replaced nothing" is a stated status rather than a missing key.
    rep["supersedes"] = None
    if a.supersedes:
        _prior = Path(a.supersedes)
        if _prior.exists():
            _bytes = _prior.read_bytes()
            try:
                _pj = json.loads(_bytes)
            except json.JSONDecodeError:
                _pj = {}
            rep["supersedes"] = {
                "path": str(_prior),
                "sha256": __import__("hashlib").sha256(_bytes).hexdigest(),
                "as_of_utc": _pj.get("as_of_utc"),
                "race_accrual_eligible": _pj.get("race_accrual_eligible"),
                "all_pass": _pj.get("all_pass"),
                "note": ("the artifact this write replaced; its bytes remain "
                         "in git history, which is the provenance -- this "
                         "path is a CACHE of the current verdict, not a "
                         "receipt"),
            }
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
    if rep.get("bar_regime") == "day_bar_v2" and rep.get("day_bar_v2"):
        print("\nDAY BAR v2 (GOVERNING: duration) + BREADTH (DISCLOSURE ONLY)")
        for c, b in sorted(rep["day_bar_v2"].items()):
            d = b.get("windows_affected_disclosure") or {}
            if not b.get("evaluable"):
                print(f"  {c}: NOT EVALUABLE -- {b.get('why', '')[:70]}")
            else:
                print(f"  {c}: P1 {b['P1_lost_s_per_hr']}/{P1_LOST_S_PER_HR_MAX} "
                      f"s/hr [{'PASS' if b['P1_pass'] else 'FAIL'}]  "
                      f"P2 {b['P2_material_windows']} material "
                      f"[{'PASS' if b['P2_pass'] else 'FAIL'}]  "
                      f"P3 {b['P3_worst_rolling_60min_lost_s']}/"
                      f"{P3_ROLLING_60MIN_LOST_S_MAX}s "
                      f"[{'PASS' if b['P3_pass'] else 'FAIL'}]")
            _oe = d.get("affected_over_elapsed")
            print(f"      breadth (NOT A GATE): "
                  f"{d.get('windows_affected_COIN_LEVEL')}/"
                  f"{d.get('windows_complete_elapsed')} elapsed"
                  f" ({d.get('pct_of_elapsed')}%)"
                  if _oe is not None else
                  f"      breadth (NOT A GATE): "
                  f"{d.get('windows_affected_COIN_LEVEL')}/"
                  f"{d.get('windows_complete_elapsed')} elapsed (rate "
                  f"UNDEFINED -- no complete window yet)")
            print(f"      breadth (NOT A GATE): "
                  f"{d.get('windows_affected_COIN_LEVEL')}/288 closing-day "
                  f"denominator ({d.get('pct_of_288')}%)"
                  + ("  [zero is NOT a clean claim: coverage not observed]"
                     if d.get("zero_affected_is_not_a_clean_claim") else ""))
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
