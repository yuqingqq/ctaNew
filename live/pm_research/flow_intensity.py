"""Flow model: the arrival-intensity clock, non-parametrically.

Executes only the `f_r` and `f_p` layer of BE_FLOWANDFILLS_MODEL_PLAN.md §2.2.
No parametric form, no Hawkes, no self-excitation, no f_book -- deliberately.
A self-exciting term with a constant baseline attributes clock-driven intensity
growth to itself and adopts on in-sample fit, so the clock is measured FIRST and
anything else is tested against it later.

Guards carried from the session, each paid for:
  * R-DUAL      -- arrival counts and descriptive notional throughput are both
                   reported, with distinct units and the 0.02 event type shown.
  * FOLD        -- trades are single-sided; the pair book is one book. Down-token
                   trades enter the unified frame at 1-p with side flipped.
                   Skipping this halves every estimate.
  * ARRIVALS    -- one `last_trade_price` == one taker-order aggregate. Counting
                   OrderFilled legs would inject a state-dependent multiplicity.
  * DENOMINATOR -- numerator and denominator use the SAME window and state
                   intervals. Excluding the micro class changes counts, never
                   exposure.
  * EXPOSURE    -- integrate over observed time using exact gap boundaries. This
                   fixes the DENOMINATOR bias and NOT the selection bias.
  * ERA         -- gap-ledger work is clob_v3_1 only; before 2026-08-20 14:50:21
                   absence of a gap record is NOT evidence of a clean window.

    python3 live/pm_research/flow_intensity.py --selftest
    python3 live/pm_research/flow_intensity.py fr
    python3 live/pm_research/flow_intensity.py fp
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import datetime as dt
import json
import math
import random
import sys
import zlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Iterable

#: RR12-1, AT THE SITE WHERE THE RESOLUTION HAPPENS. CODE ROOT AND DATA ROOT
#: ARE NOT THE SAME TREE, and this module reads DATA. `REPO` was
#: `Path(__file__).resolve().parents[2]`, which is right for CODE and wrong for
#: the tape: `data/` is gitignored and exists ONCE, so from a worktree this
#: resolved to a tree with no `data/pm_5min/raw` and `_archive_paths()`
#: returned an EMPTY MAP.
#:
#: WHAT THAT COST, MEASURED RATHER THAN FEARED. `warning_window.select_holdout`
#: reads its windows through here, and `da_forward_day_verify`'s
#: `entirely_post_freeze` reads the selector. From a bare worktree the selector
#: was empty for EVERY era, so a live `verify_day` returned
#: `post_freeze_pass: FALSE` for EVERY day -- 2026-09-01 and 2026-09-02
#: INCLUDED, the two days that have accrued, whose artifacts read True. The
#: message said "absent from the selector", which reads as a data problem and
#: is a ROOT problem. Measured: 0 archive paths from a worktree against 29,438
#: with the tape mirrored, the SAME code both times.
#:
#: PRODUCTION WAS NEVER WRONG -- the nightly unit runs from the canonical tree,
#: where the old expression and the new resolution return the SAME path. What
#: was wrong is that every VERIFICATION run in a detached scratch worktree read
#: a silently empty tape, so the defect corrupted the instrument used to check
#: the work rather than the work.
#:
#: ONE RULE, IMPORTED, NEVER A SECOND COPY. `pm_tape_density` is the
#: lowest-level reader of this tape and owns the resolution (explicit
#: PM_DATA_ROOT > the code tree IF it carries the tape > canonical), so
#: everything above it agrees by construction instead of by two expressions
#: that can disagree.
CODE_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_tape_density as _TDROOT                              # noqa: E402
DATA_ROOT = _TDROOT.DATA_ROOT
#: WHICH BRANCH ANSWERED, re-exported so a short run is self-explaining rather
#: than merely smaller (DA10-R2's lesson, applied to this module's consumers).
DATA_ROOT_BRANCH = _TDROOT.DATA_ROOT_BRANCH

#: KEPT UNDER ITS OLD NAME ON PURPOSE. Twenty-two modules read `fi.REPO`,
#: `fi.PM`, `fi.RAW`, `fi.GAPS` or `fi.MARKETS`, and every one of them wants
#: the TAPE -- so the name keeps its meaning and only its resolution is
#: repaired. The one consumer that wants the CODE (the register path in
#: `file_vacate_row`) is moved to CODE_ROOT below, which is the whole
#: distinction RR12-1 draws.
REPO = DATA_ROOT
PM = REPO / "data/pm_5min"
RAW = PM / "raw"
GAPS = PM / "collector_gaps.jsonl"
MARKETS = PM / "markets.jsonl"
OUT_MD = Path(__file__).with_name("FLOW_INTENSITY_RESULTS.md")

#: THE REGISTER IS CODE, NOT DATA. It is tracked, it exists in every worktree,
#: and a run from a worktree that filed into the CANONICAL tree's register
#: would be writing to a tree it is not in. Named as a constant so the
#: distinction is checkable rather than buried in a default argument.
REGISTER_DEFAULT = (CODE_ROOT / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min"
                    / "workspace/COORDINATION.md")

WINDOW_S = 300.0
QUOTE_STATE_LAG_S = 0.250  # frozen; shared by numerator and exposure
# Declared in FLOW_MODEL_PROTOCOL_V*.yaml as `minimum_price_bin_dwell_s` and in
# SPEC_REV2 as part of the estimand rather than a post-hoc filter -- but until
# 2026-08-21 it existed only as an editorial fence applied in the write-up, with
# no code path. A bin with 9 s of dwell alone produced a shape_ratio of 295.
MIN_BIN_DWELL_S = 60.0
MICRO_SIZE = 0.02          # labelled single-actor class; prevalence varies by coin
MICRO_TOL = 1e-9
ERA = "clob_v3_1"
def _discover_days() -> tuple[str, ...]:
    """Every collected UTC day on disk, always current.

    DAYS WAS A HARDCODED TUPLE AND WENT STALE FOUR TIMES IN THREE DAYS -- twice
    silently, and the last time within twelve hours of being corrected. A literal
    day list cannot survive a running collector, so it is derived instead.

    THIS GROWS, which is the point, and it is why `provenance(sampled=...)`
    exists: the population change must be VISIBLE in every receipt rather than
    silent. Compare two runs on `days_sampled`, never on this constant.

    A probe pinned to a FROZEN protocol must declare its OWN day tuple -- a
    protocol's design window must not move because a collector kept running.
    `exp_gff1_side` already does this.
    """
    root = PM / "raw"
    if not root.is_dir():
        return ()
    return tuple(sorted(d.name for d in root.iterdir()
                        if d.is_dir() and d.name.isdigit()))


DAYS = _discover_days()


def assert_days_current() -> list[str]:
    """Fail LOUDLY if collected days exist on disk that DAYS does not list.

    This constant went stale silently on 2026-08-22: it omitted that day and its
    1,141 archives, and every probe importing `_archive_paths()` inherited the
    omission without any signal. A window pool that quietly shrinks is the same
    class of defect as a gate that cannot fire -- it reports success while
    measuring less than it claims.

    Probes pinned to a FROZEN protocol should declare their own day tuple rather
    than inherit this one; a protocol's design window must not move because a
    collector kept running.
    """
    on_disk = list(_discover_days())
    missing = [d for d in on_disk if d not in DAYS]
    if missing:
        raise AssertionError(
            f"flow_intensity.DAYS is STALE: {missing} on disk but unlisted. "
            f"DAYS is now DERIVED, so this can only mean it was captured at "
            f"import and the collector has written a new day since.")
    return on_disk

TRADE_MARK = b'"event_type":"last_trade_price"'
QUOTE_MARK = b'"event_type":"price_change"'

# f_r bins: 15 s, 20 per window. BTC runs ~2,424 trades/window and the thinnest
# alts 108-133, so at ~100 windows/coin even the alts hold ~500+ per bin.
FR_BINS = 20
FR_W = WINDOW_S / FR_BINS

FP_EDGES = (0.0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0)


# --------------------------------------------------------------------------
# fold
# --------------------------------------------------------------------------

def fold_price(price: float, is_down: bool) -> float:
    """Express a trade price in the unified (Up-token) frame."""
    return (1.0 - price) if is_down else price


def fold_side(side: str, is_down: bool) -> str:
    """Buying Down is selling Up. Flip the taker side into the unified frame."""
    s = side.upper()
    if not is_down:
        return s
    return "SELL" if s == "BUY" else "BUY"


def r_bin(elapsed_s: float) -> int:
    """Bin index by elapsed time; bin k covers r in (300-(k+1)w, 300-kw]."""
    if not (0.0 <= elapsed_s <= WINDOW_S):
        raise ValueError(f"elapsed {elapsed_s} outside window")
    return min(int(elapsed_s / FR_W), FR_BINS - 1)


def bin_r_mid(k: int) -> float:
    """Seconds remaining at the midpoint of bin k."""
    return WINDOW_S - (k + 0.5) * FR_W


def p_bin(p: float) -> int:
    for i in range(len(FP_EDGES) - 1):
        if FP_EDGES[i] <= p < FP_EDGES[i + 1]:
            return i
    return len(FP_EDGES) - 2


def p_label(i: int) -> str:
    return f"[{FP_EDGES[i]:.2f},{FP_EDGES[i+1]:.2f})"


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------

def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def bin_exposure(gaps: Sequence[tuple[float, float]]) -> list[float]:
    """Observed seconds per f_r bin, given gap intervals relative to window start.

    Gaps are clipped to the window and may straddle bin boundaries.
    """
    out = []
    for k in range(FR_BINS):
        lo, hi = k * FR_W, (k + 1) * FR_W
        lost = sum(overlap(lo, hi, g0, g1) for g0, g1 in gaps)
        out.append(max(0.0, FR_W - lost))
    return out


def _eras() -> list[tuple[int, float, str]]:
    ev = []
    if not GAPS.exists():
        return []
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") in ("collector_start", "collector_stop"):
                ev.append((r["recv_ns"], r["event"], r.get("collector_version")))
    ev.sort()
    out, cur = [], None
    for t, e, v in ev:
        if e == "collector_start":
            if cur:
                out.append((cur[0], t, cur[1]))
            cur = (t, v)
        else:
            if cur:
                out.append((cur[0], t, cur[1]))
                cur = None
    if cur:
        out.append((cur[0], float("inf"), cur[1]))
    return out


def gaps_by_slug(era: str = ERA) -> dict[str, list[tuple[float, float]]]:
    """Gap intervals per slug, relative to window start, for ONE era."""
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    if not GAPS.exists():
        return out
    with GAPS.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("collector_version") != era:
                continue
            s, e, ws = r.get("gap_start_ns"), r.get("gap_end_ns"), r.get("window_start")
            slug = r.get("slug")
            if not (s and e and ws and slug):
                continue
            g0, g1 = s / 1e9 - int(ws), e / 1e9 - int(ws)
            g0, g1 = max(0.0, g0), min(WINDOW_S, g1)
            if g1 > g0:
                out[slug].append((g0, g1))
    return out


def covered_slugs(era: str = ERA) -> set[str]:
    """Slugs whose entire [ws, ws+300] lies inside a single era of `era`."""
    spans = [(a / 1e9, b / 1e9) for a, b, v in _eras() if v == era]
    if not spans:
        return set()
    out = set()
    for day in DAYS:
        d = RAW / day
        if not d.is_dir():
            continue
        for path in d.glob("*.jsonl*.gz"):        # .gz only == immutable
            slug = path.name.split(".jsonl")[0]
            try:
                ws = int(slug.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if any(a <= ws and ws + WINDOW_S <= b for a, b in spans):
                out.add(slug)
    return out


def token_map() -> dict[str, tuple[str, str]]:
    """slug -> (up_token_id, down_token_id). outcomes are ["Up","Down"] in all 3230."""
    out: dict[str, tuple[str, str]] = {}
    if not MARKETS.exists():
        return out
    for line in MARKETS.read_text().splitlines():
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        ids, slug = m.get("clobTokenIds"), m.get("slug")
        if ids and len(ids) == 2 and slug:
            out[slug] = (str(ids[0]), str(ids[1]))
    return out


def _gz_lines(path: Path) -> Iterator[bytes]:
    dec = zlib.decompressobj(zlib.MAX_WBITS | 16)
    tail = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            try:
                out = dec.decompress(chunk)
            except zlib.error:
                return
            if not out:
                continue
            buf = tail + out
            *lines, tail = buf.split(b"\n")
            yield from lines
    if tail:
        yield tail


# Records which days the last _archive_paths() call actually read. Every probe
# consuming that helper stamps this into its receipt via provenance(), so a
# population change is VISIBLE in the output rather than silent. DAYS grew from
# three days to four on 2026-08-22; without this, re-running any probe would
# quietly produce numbers over a different population than the ones published.
_ARCHIVE_DAYS_USED: list[str] = []


def provenance(sampled: "Iterable[str | Path] | None" = None) -> dict[str, object]:
    """Source-day provenance for a receipt. Stamp this in every published result.

    PASS THE SLUGS OR PATHS YOU ACTUALLY SAMPLED. Without them this reports days
    READ, which for any subsampling probe is an UPPER BOUND on day coverage and
    not the population the numbers came from -- `inventory_walk.select` takes the
    first `per_coin` slugs in sorted order, i.e. the EARLIEST, so a newly
    collected day can be globbed and then never displace the sample. That is why
    a run after DAYS grew to four days still reproduced a three-day figure to the
    digit. Comparing two runs on `source_days` alone would have missed it.
    """
    read = list(_ARCHIVE_DAYS_USED)
    out: dict[str, object] = {
        "days_read": read,
        "days_declared": list(DAYS),
        "n_days_read": len(read),
    }
    if sampled is None:
        out["days_sampled"] = None
        out["sampled_is_known"] = False
        out["warning"] = "days_sampled UNKNOWN -- days_read is an UPPER BOUND"
        out["source_days"] = read          # back-compat, over-reports
        return out
    days = set()
    for item in sampled:
        name = Path(item).name if isinstance(item, Path) else str(item)
        m = re.search(r"-(\d{10})(?:\D|$)", name)
        if m:
            days.add(dt.datetime.fromtimestamp(
                int(m.group(1)), dt.timezone.utc).strftime("%Y%m%d"))
    out["days_sampled"] = sorted(days)
    out["sampled_is_known"] = True
    out["n_days_sampled"] = len(days)
    out["source_days"] = sorted(days)
    return out




# Frozen by FLOW_MODEL_PROTOCOL_V5.yaml under ruling R-19 (D-V5-1).
V5_PRIMARY_EVALUATION_START = "2026-08-22"


def promotion_days(primary_start: str = V5_PRIMARY_EVALUATION_START,
                   era: str = ERA) -> dict[str, Any]:
    """The ten-forward-day promotion count, DERIVED. Never write this down.

    D-V5-2 binding condition, in code: the count is computed from V5's own
    `primary_evaluation_start` against the days actually present on disk, at read
    time. A hardcoded day count in a tracking file is the `DAYS`-went-stale
    defect wearing a different hat -- it went stale four times in three days, the
    last within twelve hours of being fixed.

    Counts only COMPLETE days: the newest day on disk is still accruing, so it is
    reported separately rather than silently included.
    """
    days = sorted({slug_day(s) for s in covered_slugs(era)})
    if not days:
        return {"primary_start": primary_start, "n_complete": 0,
                "complete": [], "in_progress": None, "required": 10,
                "remaining": 10, "met": False}
    in_progress = days[-1]
    eligible = [d for d in days if d >= primary_start and d != in_progress]
    return {
        "primary_start": primary_start,
        "era": era,
        "complete": eligible,
        "n_complete": len(eligible),
        "in_progress": in_progress if in_progress >= primary_start else None,
        "required": 10,
        "remaining": max(0, 10 - len(eligible)),
        "met": len(eligible) >= 10,
        "rule": "DERIVED_AT_READ_TIME_NEVER_STORED",
    }



# ---------------------------------------------------------------------------
# R-20: a frozen Class-D bar freezes its TEXT but not its INPUTS.
# ---------------------------------------------------------------------------
# R-1's f* is a function of Class-C MEASURED values -- spread capture and the
# CI endpoint of Layer-1 markout nearest zero -- which Class C obliges the
# coordinator to ADOPT. So a re-publication of Layer-1 silently moves the bar
# unless the bar is anchored BY VALUE. R-20 anchors it.
#
# The values below are the SNAPSHOT AS PUBLISHED AT FREEZE TIME. They are not
# a measurement this module performs and they must never be recomputed here.
# DE's `warning_window_v1_dayseries.json` independently stores the same anchor
# as `frozen_f_low: {btc: 0.309, eth: 0.494}`.
R1_BAR_ANCHOR: dict[str, dict[str, float]] = {
    "btc": {"spread_cents": 0.642, "markout_lo_cents": 0.287, "f_star_low": 0.309},
    "eth": {"spread_cents": 0.778, "markout_lo_cents": 0.759, "f_star_low": 0.494},
}


def f_star(spread_cents: float, markout_lo_cents: float) -> float:
    """R-1's break-even warned share: |markout| / (spread + |markout|)."""
    if spread_cents < 0 or markout_lo_cents < 0:
        raise ValueError("f_star takes magnitudes; pass |markout|, not the signed value")
    denom = spread_cents + markout_lo_cents
    if denom <= 0:
        raise ValueError("spread + |markout| must be positive")
    return markout_lo_cents / denom


def would_move_bar(coin: str, new_spread_cents: float,
                   new_markout_lo_cents: float,
                   measured_r_hi: float | None = None) -> dict[str, Any]:
    """R-20 REPORTING OBLIGATION, made mechanical. This NEVER moves the bar.

    Given a re-published Layer-1 spread and markout, report what f* WOULD have
    been and in which direction the verdict would have moved. The frozen bar is
    unchanged by construction: `f_star_low_operative` is always the anchor.

    R-20 is SYMMETRIC. A re-publication that makes a coin MORE dead is exactly
    as much a finding as one that revives it, and neither moves the bar. The
    point is that verdicts stop drifting with their inputs, not that they drift
    only in the convenient direction.

    `measured_r_hi` is the upper bound of the measured warned share R. Pass it to
    learn whether the counterfactual bar would flip the DEAD verdict; omit it for
    the bar movement alone.
    """
    if coin not in R1_BAR_ANCHOR:
        raise ValueError(f"{coin} is not a verdict coin under R-1; "
                         f"anchored coins are {sorted(R1_BAR_ANCHOR)}")
    anchor = R1_BAR_ANCHOR[coin]
    counterfactual = f_star(new_spread_cents, new_markout_lo_cents)
    delta = counterfactual - anchor["f_star_low"]
    out: dict[str, Any] = {
        "coin": coin,
        "f_star_low_operative": anchor["f_star_low"],   # THE BAR. Never changes.
        "f_star_low_counterfactual": round(counterfactual, 4),
        "delta": round(delta, 4),
        "direction": ("BAR_WOULD_HAVE_RISEN_HARDER_TO_BE_DEAD" if delta > 0
                      else "BAR_WOULD_HAVE_FALLEN_EASIER_TO_BE_DEAD" if delta < 0
                      else "UNCHANGED"),
        "anchor_inputs": dict(anchor),
        "new_inputs": {"spread_cents": new_spread_cents,
                       "markout_lo_cents": new_markout_lo_cents},
        "bar_moved": False,
        "rule": "R-20: publish the new markout as its OWN result; surface the "
                "would-have-moved to the coordinator; do NOT propagate into f*",
    }
    if measured_r_hi is not None:
        dead_now = measured_r_hi < anchor["f_star_low"]
        dead_cf = measured_r_hi < counterfactual
        out["measured_r_hi"] = measured_r_hi
        out["verdict_under_operative_bar"] = "DEAD" if dead_now else "NOT_DEAD"
        out["would_vacate"] = dead_now and not dead_cf

        # R-38 CLAUSE (D): AN AMENDMENT MAY NOT, BY ITSELF, CHANGE A VERDICT.
        # This block previously reported `verdict_under_counterfactual: NOT_DEAD`
        # -- i.e. it amended its way from DEAD to alive, which is exactly what
        # clause (d) forbids. An amendment can only vacate a verdict to
        # UNDETERMINED, and that purchase costs an OBLIGATION, not a result.
        if dead_now and not dead_cf:
            out["verdict_under_counterfactual"] = "UNDETERMINED_PENDING_RERUN"
            out.update(_vacate_provenance(coin, anchor, counterfactual, "DEAD"))
            out["obligation"] = (
                "Re-establishing ANY verdict requires RE-RUNNING the "
                "measurement under the new bar at the original evidentiary "
                "standard. An amendment buys 'not yet determined', never 'alive'.")
            out["escalate"] = (
                "FIRST-CLASS FINDING: a re-publication would VACATE this coin's "
                "R-11 verdict to UNDETERMINED and oblige a re-run. It would not "
                "make the coin not-dead, and the bar does not move.")
        elif (not dead_now) and dead_cf:
            out["verdict_under_counterfactual"] = "UNDETERMINED_PENDING_RERUN"
            out.update(_vacate_provenance(coin, anchor, counterfactual, "NOT_DEAD"))
            out["obligation"] = (
                "Symmetric: an amendment that would make a coin MORE dead is "
                "equally a finding and equally cannot establish the verdict. "
                "It vacates and obliges a re-run.")
            out["escalate"] = (
                "FIRST-CLASS FINDING, adverse direction: a re-publication would "
                "vacate to UNDETERMINED. R-20's symmetry holds under R-38.")
        else:
            out["verdict_under_counterfactual"] = out["verdict_under_operative_bar"]
    return out



# ---------------------------------------------------------------------------
# R-35 (Q-BE-3): day-stratified sampling binds ANY probe whose output is an
# input to a frozen bar -- not only V5 fits.
# ---------------------------------------------------------------------------
# The reason is R-20: a frozen bar is anchored to its inputs BY VALUE, so if
# those inputs came from a sampler that cannot leave one day, the anchor is
# anchored to a BIASED number. `edge_l1_v1`'s Layer-1 markout feeds R-1's f*,
# so Layer-1 is in scope even though it is not a V5 fit.
#
# Two constraints follow, and both are mechanised here rather than remembered.
SAMPLING_RULES = ("EARLIEST_FIRST", "DAY_STRATIFIED", "WHOLE_COVERED_POPULATION")


def assert_poolable(receipts: "Iterable[Mapping[str, Any]]") -> str:
    """R-35 constraint 2: NEVER pool across sampling rules. Returns the rule.

    Different samplers are DIFFERENT POPULATIONS -- the never-pool-across-eras
    rule one level up. A comparison must state which rule each side used, and a
    pooled statistic over mixed rules is not a statistic about anything.

    Refuses loudly rather than warning, because the failure it guards is silent:
    an earliest-first receipt and a day-stratified one have the same shape and
    the same fields, and nothing in the numbers reveals the mixture.
    """
    seen: dict[str, list[str]] = {}
    for r in receipts:
        rule = str(r.get("sampling_rule") or "UNDECLARED")
        seen.setdefault(rule, []).append(str(r.get("protocol") or "<unnamed>"))
    if "UNDECLARED" in seen:
        raise ValueError(
            "R-35: receipt(s) with no declared sampling_rule cannot be pooled: "
            f"{seen['UNDECLARED']}. Declare the rule; do not assume it.")
    if len(seen) > 1:
        raise ValueError(
            "R-35 constraint 2 -- NEVER POOL ACROSS SAMPLING RULES. "
            f"Mixed populations: { {k: v for k, v in seen.items()} }. "
            "Different samplers are different populations; state which rule "
            "each side used and compare, do not pool.")
    return next(iter(seen))


def resampled_markout_is_a_candidate(coin: str, new_spread_cents: float,
                                     new_markout_lo_cents: float,
                                     measured_r_hi: float | None = None
                                     ) -> dict[str, Any]:
    """R-35 constraint 1: a re-sampled Layer-1 markout does NOT move the bar.

    It creates a CANDIDATE requiring a Class-D amendment under R-6 -- which,
    since `ww_v1`'s measurement has already run, must satisfy all three parts of
    the amendment test including (c) invalidating every verdict computed under
    the old bar. That is no longer free.

    This wraps `would_move_bar` so the return value cannot be mistaken for an
    updated bar: the operative value is unchanged and the result is labelled a
    candidate, not a bar.
    """
    out = would_move_bar(coin, new_spread_cents, new_markout_lo_cents,
                         measured_r_hi=measured_r_hi)
    out["status"] = "CANDIDATE_NOT_A_BAR"
    out["requires"] = ("R-6 Class-D amendment: made before the re-run, motivated "
                       "by information that is not the result, and explicitly "
                       "invalidating every verdict computed under the old bar")
    out["amendment_is_free"] = False   # ww_v1's measurement has already run
    return out



def _vacate_provenance(coin: str, anchor: Mapping[str, float],
                       counterfactual: float, was: str) -> dict[str, Any]:
    """R-40 CLAUSE (E): a vacated verdict carries its provenance PERMANENTLY.

    Clause (d) stopped an amendment converting DEAD into alive. It left one
    residual, which the coordinator review found: an amender who vacates and
    never completes the re-run holds the verdict at UNDETERMINED indefinitely --
    four-fifths of the erasure for the price of an unpaid IOU.

    Clause (e) makes limbo LOUD. The vacated verdict keeps its history forever,
    and the vacating amendment owes a register row for the re-run it created.

    `register_row_filed` is False here BY CONSTRUCTION: this function knows the
    obligation exists, not that anyone discharged it. Leaving it False is exactly
    the limbo clause (e) exposes.
    """
    return {
        "vacated_provenance": (
            f"VACATED -- was {was} under bar f*_low={anchor['f_star_low']:.3f} "
            f"(spread {anchor['spread_cents']}, |markout|_lo "
            f"{anchor['markout_lo_cents']}); re-run owed under the amended bar "
            f"f*_low={counterfactual:.4f}"),
        "provenance_is_permanent": True,
        "register_row_required": True,
        "register_row_filed": False,
        "register_row": (
            f"| Q-BE-n | BE | **RE-RUN OWED (R-40 clause e)** -- `ww_v1` {coin} "
            f"verdict VACATED to UNDETERMINED by an amendment to `f*_low` "
            f"({anchor['f_star_low']:.3f} -> {counterfactual:.4f}). Under clause "
            f"(d) the amendment cannot re-establish a verdict; re-running the "
            f"measurement under the new bar at the original evidentiary standard "
            f"is owed. Until it completes, {coin} is UNDETERMINED and was {was}. |"),
    }


def file_vacate_obligation(result: Mapping[str, Any],
                           register: "Path | None" = None,
                           dry_run: bool = True) -> dict[str, Any]:
    """Append the owed-re-run row to the section 0a register. Clause (e).

    `dry_run=True` by default: a function that silently edits a shared,
    coordinator-owned file as a side effect is a worse defect than the one it
    fixes. The caller sees the row and files it deliberately.
    """
    if not result.get("register_row_required"):
        raise ValueError("no vacate obligation on this result; nothing to file")
    row = result["register_row"]
    if dry_run:
        return {"filed": False, "dry_run": True, "row": row,
                "note": "re-call with dry_run=False to append"}
    path = register or REGISTER_DEFAULT
    text = path.read_text()
    marker = "\n## 0. Roles"
    if marker not in text:
        raise ValueError("section 0a register not found; refusing to guess")
    i = text.index(marker)
    path.write_text(text[:i] + "\n" + row + text[i:])
    return {"filed": True, "dry_run": False, "row": row, "register": str(path)}


def slug_day(slug: str) -> str:
    """UTC day of a window slug, from the epoch start it ends in."""
    m = re.search(r"-(\d{10})(?:\D|$)", str(slug))
    if not m:
        raise ValueError(f"no epoch in slug {slug!r}")
    return dt.datetime.fromtimestamp(
        int(m.group(1)), dt.timezone.utc).strftime("%Y-%m-%d")


def sample_days(per_coin: int, era: str = ERA) -> dict[str, Any]:
    """How many UTC DAYS an earliest-N-per-coin sample actually spans.

    THE SECOND SAMPLING BUG, found 2026-08-23. Deriving DAYS from disk fixed the
    population READ. It did not touch the population SAMPLED: every probe selects
    with `sorted(covered_slugs(ERA))` truncated at `per_coin`, and a slug sorts by
    the epoch it ends in, so "the first N" is "the EARLIEST N". The clob_v3_1 era
    opens 2026-08-20 14:50:21, so at per_coin <= 60 the sample never leaves
    2026-08-20 no matter how many days are on disk.

    Every headline replay result in this corpus was computed at per_coin 10-60
    and is therefore ONE UTC DAY -- while two receipts stamped `n_days: 4` on it,
    because they called `provenance()` with no `sampled=`. See FLOW_MODEL_STATE
    section 1f.

    Call this before publishing, and report `n_days_sampled` beside every N.
    """
    paths = _archive_paths()
    tokens = token_map()
    picked: dict[str, list[str]] = {}
    for slug in sorted(covered_slugs(era)):
        coin = slug.split("-")[0]
        if slug not in paths or slug not in tokens:
            continue
        picked.setdefault(coin, [])
        if len(picked[coin]) < per_coin:
            picked[coin].append(slug)
    per: dict[str, Any] = {}
    union: set[str] = set()
    for coin, slugs in sorted(picked.items()):
        days = sorted({slug_day(s) for s in slugs})
        per[coin] = {"n_windows": len(slugs), "days": days, "n_days": len(days)}
        union |= set(days)
    return {
        "per_coin_requested": per_coin,
        "era": era,
        "coins": per,
        "days_sampled": sorted(union),
        "n_days_sampled": len(union),
        "days_declared": list(DAYS),
        "warning": ("EARLIEST-first selection: n_days_sampled is what the numbers "
                    "rest on; days_declared is only what is on disk"),
    }


def report_sample_days(per_coin_values: Sequence[int] = (10, 24, 25, 30, 60, 273, 361),
                       era: str = ERA) -> list[str]:
    """One line per candidate sample size: how many days it would actually span."""
    paths = _archive_paths()
    tokens = token_map()
    avail: dict[str, list[str]] = {}
    for slug in sorted(covered_slugs(era)):
        if slug in paths and slug in tokens:
            avail.setdefault(slug.split("-")[0], []).append(slug)
    L = [f"era={era}  coins={len(avail)}  "
         f"available/coin={min(len(v) for v in avail.values())}"
         f"-{max(len(v) for v in avail.values())}  "
         f"days_declared={','.join(DAYS)}", ""]
    ndays_all = len({slug_day(s) for v in avail.values() for s in v})
    L.append(f"whole era spans {ndays_all} UTC days")
    L.append("")
    L.append(f"{'per_coin':>8} | {'n_days_sampled':>14} | days")
    for n in per_coin_values:
        union = sorted({slug_day(s) for v in avail.values() for s in v[:n]})
        L.append(f"{n:>8} | {len(union):>14} | {','.join(d[5:] for d in union)}")
    return L


def _archive_paths() -> dict[str, Path]:
    out: dict[str, Path] = {}
    used: list[str] = []
    for day in DAYS:
        d = RAW / day
        if not d.is_dir():
            continue
        used.append(day)
        for path in sorted(d.glob("*.jsonl*.gz")):
            out.setdefault(path.name.split(".jsonl")[0], path)
    _ARCHIVE_DAYS_USED[:] = used
    return out


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def window_trades(path: Path, up_id: str, down_id: str) -> list[dict[str, Any]]:
    """Folded taker arrivals for one window. One row per last_trade_price event.

    Execution price is a mark, not the state used to condition ``f_p``.  The
    latter is joined separately from the lagged midpoint-state timeline.
    """
    try:
        ws = int(path.name.split(".jsonl")[0].rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return []
    rows = []
    for line in _gz_lines(path):
        if TRADE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            rn = int(parts[0])
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        el = rn / 1e9 - ws
        if not (0.0 <= el <= WINDOW_S):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "last_trade_price":
                continue
            aid = str(msg.get("asset_id"))
            if aid == up_id:
                is_down = False
            elif aid == down_id:
                is_down = True
            else:
                continue
            try:
                px, sz = float(msg["price"]), float(msg["size"])
                side = str(msg["side"]).upper()
            except (KeyError, ValueError, TypeError):
                continue
            rows.append({
                "elapsed": el,
                "native_price": px,
                "exec_p_up": fold_price(px, is_down),
                "side": fold_side(side, is_down),
                "size": sz,
                "notional": sz * px,        # USDC actually paid; do not use folded p
                "micro": abs(sz - MICRO_SIZE) < MICRO_TOL,
            })
    return rows


def state_segments_from_points(
    points: Sequence[tuple[float, float]],
    gaps: Sequence[tuple[float, float]],
    lag_s: float = QUOTE_STATE_LAG_S,
) -> list[tuple[float, float, float]]:
    """Build half-open, knowledge-admissible ``(start, end, mid)`` segments.

    A quote received at ``t`` becomes usable only at ``t + lag_s``.  A
    collector gap invalidates the current state at its start; that state is
    never carried across the gap.  Only a later quote can establish a new
    segment.  This is intentionally stricter than subtracting gap duration
    from a stale quote interval.
    """
    if lag_s < 0:
        raise ValueError("state lag must be non-negative")

    # Last quote wins when multiple updates share one receive timestamp.
    effective: list[tuple[float, float]] = []
    ordered = sorted(
        ((float(t) + lag_s, float(mid)) for t, mid in points),
        key=lambda point: point[0],
    )
    for t, mid in ordered:
        if not (0.0 <= mid <= 1.0) or t > WINDOW_S:
            continue
        if effective and abs(effective[-1][0] - t) < 1e-12:
            effective[-1] = (t, mid)
        else:
            effective.append((t, mid))

    clipped_gaps = sorted(
        (max(0.0, float(g0)), min(WINDOW_S, float(g1)))
        for g0, g1 in gaps if g1 > g0 and g1 > 0.0 and g0 < WINDOW_S
    )
    out: list[tuple[float, float, float]] = []
    for i, (raw_start, mid) in enumerate(effective):
        start = max(0.0, raw_start)
        end = effective[i + 1][0] if i + 1 < len(effective) else WINDOW_S
        end = min(WINDOW_S, end)
        if end <= start:
            continue

        received = raw_start - lag_s
        if any(g0 < start and g1 > received for g0, g1 in clipped_gaps):
            # A disconnect between receipt and admissibility invalidates the
            # quote even when the 250 ms lag would otherwise mature after the
            # reconnect.
            continue

        # The first gap touching this quote's lifetime kills it.  In
        # particular, a state whose admissibility time lands inside a gap does
        # not reappear at gap_end.
        first_touch = next(
            ((g0, g1) for g0, g1 in clipped_gaps if g1 > start and g0 < end),
            None,
        )
        if first_touch is not None:
            g0, _ = first_touch
            if g0 <= start:
                continue
            end = min(end, g0)
        if end > start:
            out.append((start, end, mid))
    return out


def state_mid_at(segments: Sequence[tuple[float, float, float]],
                 elapsed: float) -> float | None:
    """Return the midpoint state on the half-open segment containing elapsed."""
    starts = [s[0] for s in segments]
    i = bisect.bisect_right(starts, elapsed) - 1
    if i >= 0 and segments[i][0] <= elapsed < segments[i][1]:
        return segments[i][2]
    return None


def state_dwell(segments: Sequence[tuple[float, float, float]]) -> list[float]:
    """Exposure seconds by midpoint-state bin."""
    dwell = [0.0] * (len(FP_EDGES) - 1)
    for start, end, mid in segments:
        dwell[p_bin(mid)] += end - start
    return dwell


def window_mid_segments(path: Path, up_id: str,
                        gaps: Sequence[tuple[float, float]]) -> list[
                            tuple[float, float, float]
                        ]:
    """Lagged Up-midpoint state segments, with stale state killed at gaps."""
    try:
        ws = int(path.name.split(".jsonl")[0].rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return []
    pts: list[tuple[float, float]] = []
    for line in _gz_lines(path):
        if QUOTE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            rn = int(parts[0])
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        el = rn / 1e9 - ws
        if not (0.0 <= el <= WINDOW_S):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "price_change":
                continue
            for pc in msg.get("price_changes", []):
                if str(pc.get("asset_id")) != up_id:
                    continue
                try:
                    b, a = float(pc["best_bid"]), float(pc["best_ask"])
                except (KeyError, TypeError, ValueError):
                    continue
                # boundary quotes are RETAINED -- 0.0 <= b < a <= 1.0. The strict
                # form dropped 5.2% of quotes, all from the tails, in two
                # independent codebases this session.
                if not (0.0 <= b < a <= 1.0):
                    continue
                pts.append((el, (a + b) / 2.0))
    return state_segments_from_points(pts, gaps)


def fp_window_counts(
    rows: Sequence[dict[str, Any]],
    segments: Sequence[tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float], dict[str, int]]:
    """Bin arrivals and exposure on the identical lagged midpoint state.

    ``exec_p_up`` is retained only as an execution mark and a mismatch
    diagnostic.  It never selects the conditioning bin.
    """
    n_bins = len(FP_EDGES) - 1
    cnt = [0.0] * n_bins
    cnt_ex = [0.0] * n_bins
    notl = [0.0] * n_bins
    diag = {"total": len(rows), "admitted": 0, "no_state": 0,
            "exec_state_bin_mismatch": 0, "micro_admitted": 0}
    for row in rows:
        mid = state_mid_at(segments, float(row["elapsed"]))
        if mid is None:
            diag["no_state"] += 1
            continue
        i = p_bin(mid)
        diag["admitted"] += 1
        if p_bin(float(row["exec_p_up"])) != i:
            diag["exec_state_bin_mismatch"] += 1
        cnt[i] += 1.0
        notl[i] += float(row["notional"])
        if row["micro"]:
            diag["micro_admitted"] += 1
        else:
            cnt_ex[i] += 1.0
    return cnt, cnt_ex, notl, diag


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def cluster_bootstrap(per_window: list[tuple[list[float], list[float]]],
                      n_boot: int, seed: int) -> list[tuple[float, float]]:
    """Window-clustered CI for count/exposure ratios, per bin.

    CAVEAT CARRIED EVERYWHERE: window clustering cannot capture day-level common
    factors. With two days these intervals UNDERSTATE true uncertainty.
    """
    if not per_window:
        return []
    n_bins = len(per_window[0][0])
    rng = random.Random(seed)
    draws: list[list[float]] = [[] for _ in range(n_bins)]
    idx = range(len(per_window))
    for _ in range(n_boot):
        pick = [rng.choice(per_window) for _ in idx]
        for k in range(n_bins):
            num = sum(w[0][k] for w in pick)
            den = sum(w[1][k] for w in pick)
            if den > 0:
                draws[k].append(num / den)
    out = []
    for k in range(n_bins):
        d = sorted(draws[k])
        if not d:
            out.append((float("nan"), float("nan")))
            continue
        lo = d[int(0.025 * (len(d) - 1))]
        hi = d[int(0.975 * (len(d) - 1))]
        out.append((lo, hi))
    return out


def profile_ratio(per_window: list[tuple[list[float], list[float]]]) -> list[float]:
    if not per_window:
        return []
    n = len(per_window[0][0])
    out = []
    for k in range(n):
        num = sum(w[0][k] for w in per_window)
        den = sum(w[1][k] for w in per_window)
        out.append(num / den if den > 0 else float("nan"))
    return out


def shape_ratio(prof: Sequence[float]) -> float:
    """max/min over finite positive bins -- the flatness diagnostic."""
    v = [x for x in prof if math.isfinite(x) and x > 0]
    return (max(v) / min(v)) if len(v) >= 2 else float("nan")


# --------------------------------------------------------------------------
# f_r
# --------------------------------------------------------------------------

def fr(n_boot: int = 2000, seed: int = 20260821,
       era_only: bool = True) -> dict[str, Any]:
    """Empirical arrival-count profile per (coin, r-bin), exposure-corrected."""
    paths = _archive_paths()
    toks = token_map()
    gaps = gaps_by_slug(ERA)
    cov = covered_slugs(ERA)
    slugs = sorted(cov if era_only else set(paths))
    slugs = [s for s in slugs if s in paths and s in toks]

    by_coin: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    n_gap_touched = 0
    for slug in slugs:
        up, dn = toks[slug]
        rows = window_trades(paths[slug], up, dn)
        if not rows:
            continue
        g = gaps.get(slug, []) if era_only else []
        if g:
            n_gap_touched += 1
        expo = bin_exposure(g)
        cnt = [0.0] * FR_BINS
        cnt_ex = [0.0] * FR_BINS
        notl = [0.0] * FR_BINS
        notl_ex = [0.0] * FR_BINS
        for r in rows:
            k = r_bin(r["elapsed"])
            cnt[k] += 1.0
            notl[k] += r["notional"]
            if not r["micro"]:
                cnt_ex[k] += 1.0
                notl_ex[k] += r["notional"]
        by_coin[slug.split("-")[0]].append(
            {"slug": slug, "cnt": cnt, "cnt_ex": cnt_ex,
             "notl": notl, "notl_ex": notl_ex, "expo": expo})

    res: dict[str, Any] = {"bins": FR_BINS, "bin_w": FR_W, "era_only": era_only,
                           "n_gap_touched": n_gap_touched, "coins": {}}
    for coin, ws in sorted(by_coin.items()):
        d = {"n_windows": len(ws),
             "n_trades": int(sum(sum(w["cnt"]) for w in ws)),
             "n_micro": int(sum(sum(w["cnt"]) - sum(w["cnt_ex"]) for w in ws)),
             "exposure_s": sum(sum(w["expo"]) for w in ws)}
        for key, num in (("count", "cnt"), ("count_ex_micro", "cnt_ex"),
                         ("notional", "notl"), ("notional_ex_micro", "notl_ex")):
            pw = [(w[num], w["expo"]) for w in ws]
            prof = profile_ratio(pw)
            d[key] = {"profile": prof, "shape_ratio": shape_ratio(prof)}
            if key in ("count", "notional"):
                d[key]["ci"] = cluster_bootstrap(pw, n_boot, seed)
        res["coins"][coin] = d
    return res


# --------------------------------------------------------------------------
# f_p
# --------------------------------------------------------------------------

def fp(per_coin: int = 10, n_boot: int = 2000,
       seed: int = 20260821) -> dict[str, Any]:
    """Arrival intensity per lagged midpoint-state bin.

    Numerator and exposure are assigned by the identical knowledge-admissible
    state timeline.  Execution price is a mark only.  The quote stream is ~97%
    of message volume, so this runs on a deterministic subsample; scope and
    state-join diagnostics are reported.
    """
    paths = _archive_paths()
    toks = token_map()
    gaps = gaps_by_slug(ERA)
    cov = sorted(covered_slugs(ERA))

    picked: dict[str, list[str]] = collections.defaultdict(list)
    for slug in cov:
        coin = slug.split("-")[0]
        if len(picked[coin]) < per_coin and slug in paths and slug in toks:
            picked[coin].append(slug)

    n_bins = len(FP_EDGES) - 1
    res: dict[str, Any] = {"schema_version": 2, "edges": FP_EDGES,
                           "state": "up_midpoint", "state_lag_s": QUOTE_STATE_LAG_S,
                           "per_coin_target": per_coin, "coins": {}}
    for coin, slugs in sorted(picked.items()):
        pw_cnt, pw_ex, pw_notl = [], [], []
        diag_total = collections.Counter()
        for slug in slugs:
            up, dn = toks[slug]
            g = gaps.get(slug, [])
            segments = window_mid_segments(paths[slug], up, g)
            dwell = state_dwell(segments)
            if sum(dwell) <= 0:
                continue
            rows = window_trades(paths[slug], up, dn)
            cnt, cnt_ex, notl, diag = fp_window_counts(rows, segments)
            diag_total.update(diag)
            pw_cnt.append((cnt, dwell))
            pw_ex.append((cnt_ex, dwell))
            pw_notl.append((notl, dwell))
        if not pw_cnt:
            continue
        prof = profile_ratio(pw_cnt)
        res["coins"][coin] = {
            "n_windows": len(pw_cnt),
            "n_trades": diag_total["admitted"],
            "n_trades_total": diag_total["total"],
            "n_trades_state_admitted": diag_total["admitted"],
            "n_trades_no_state": diag_total["no_state"],
            "n_exec_state_bin_mismatch": diag_total["exec_state_bin_mismatch"],
            "n_micro": diag_total["micro_admitted"],
            "dwell_s": (dwell_total := [sum(w[1][i] for w in pw_cnt)
                                       for i in range(n_bins)]),
            # The fence is now CODE, not prose. Bins below the declared minimum
            # are marked non-reportable and published beside the retained set
            # rather than dropped -- the excluded set is part of the result.
            "min_bin_dwell_s": MIN_BIN_DWELL_S,
            "bin_reportable": [d >= MIN_BIN_DWELL_S for d in dwell_total],
            "n_bins_below_min_dwell": sum(d < MIN_BIN_DWELL_S for d in dwell_total),
            "count": {"profile": prof, "shape_ratio": shape_ratio(prof),
                      "ci": cluster_bootstrap(pw_cnt, n_boot, seed)},
            "count_ex_micro": {"profile": profile_ratio(pw_ex)},
            "notional": {"profile": profile_ratio(pw_notl)},
        }
    return res


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

def _expect(label: str, exc: type[BaseException], fn) -> None:
    try:
        fn()
    except exc:
        return
    except BaseException as other:  # noqa: BLE001
        raise AssertionError(f"{label}: expected {exc.__name__}, got {other!r}") from other
    raise AssertionError(f"{label}: expected {exc.__name__}, nothing raised")


def _synth(profile: Sequence[float], n_windows: int = 40
           ) -> list[tuple[list[float], list[float]]]:
    """Windows whose per-bin counts follow `profile` exactly, full exposure."""
    return [([c for c in profile], [FR_W] * len(profile)) for _ in range(n_windows)]


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # 1-4: the fold. Buying Down at 0.3 is selling Up at 0.7.
    ok(abs(fold_price(0.30, True) - 0.70) < 1e-12, "fold price down")
    ok(abs(fold_price(0.30, False) - 0.30) < 1e-12, "fold price up")
    ok(fold_side("BUY", True) == "SELL", "fold side down")
    ok(fold_side("BUY", False) == "BUY", "fold side up")

    # 5-8: r binning and its orientation. Bin 0 is the START of the window,
    # which is the LARGEST r. Getting this backwards would mirror every profile.
    ok(r_bin(0.0) == 0, "r_bin at open")
    ok(r_bin(299.9) == FR_BINS - 1, "r_bin at close")
    ok(abs(bin_r_mid(0) - (WINDOW_S - FR_W / 2)) < 1e-9, "bin 0 has largest r")
    ok(bin_r_mid(0) > bin_r_mid(FR_BINS - 1), "r decreases with bin index")
    _expect("elapsed out of window", ValueError, lambda: r_bin(400.0))

    # 9-12: exposure. A gap covering a whole bin must zero it, not shrink it.
    e = bin_exposure([(0.0, FR_W)])
    ok(abs(e[0]) < 1e-9, "full-bin gap zeroes exposure")
    ok(abs(e[1] - FR_W) < 1e-9, "neighbouring bin untouched")
    e2 = bin_exposure([(0.0, FR_W * 2.5)])
    ok(abs(e2[2] - FR_W * 0.5) < 1e-9, "straddling gap splits correctly")
    ok(abs(sum(bin_exposure([])) - WINDOW_S) < 1e-9, "no gaps == full window")

    # 13-14: CONTROL -- the flatness diagnostic must not fire on flat input and
    # must fire on a ramp. Without this pair, "f_r is non-flat" proves nothing.
    flat = profile_ratio(_synth([10.0] * FR_BINS))
    ok(abs(shape_ratio(flat) - 1.0) < 1e-9, "CONTROL: flat input -> ratio 1")
    ramp = profile_ratio(_synth([1.0 + k for k in range(FR_BINS)]))
    ok(shape_ratio(ramp) > 5.0, "CONTROL: ramped input -> ratio >> 1")

    # 15: CONTROL -- a profile must be invariant to how many identical windows
    # it is built from, or the estimator is really counting windows.
    ok(abs(profile_ratio(_synth([7.0] * FR_BINS, 5))[0]
           - profile_ratio(_synth([7.0] * FR_BINS, 500))[0]) < 1e-12,
       "CONTROL: ratio invariant to window count")

    # 16-17: bootstrap. Identical windows carry no cluster variance.
    ci = cluster_bootstrap(_synth([10.0] * FR_BINS, 30), 200, 1)
    ok(abs(ci[0][1] - ci[0][0]) < 1e-9, "CONTROL: identical windows -> zero CI width")
    mixed = [([1.0] * FR_BINS, [FR_W] * FR_BINS)] * 15 + \
            [([9.0] * FR_BINS, [FR_W] * FR_BINS)] * 15
    ci2 = cluster_bootstrap(mixed, 500, 1)
    ok(ci2[0][1] - ci2[0][0] > 1e-3, "CONTROL: dispersed windows -> positive CI width")

    # 18-19: the micro exclusion changes the NUMERATOR only. If exposure moved
    # with it, the two weightings would not be comparable.
    pw = [([10.0] * FR_BINS, [FR_W] * FR_BINS)]
    pw_ex = [([6.0] * FR_BINS, [FR_W] * FR_BINS)]
    ok(pw[0][1] == pw_ex[0][1], "micro exclusion leaves exposure identical")
    ok(profile_ratio(pw)[0] > profile_ratio(pw_ex)[0], "excluding micro lowers count rate")

    # 20-22: price binning covers [0,1] with no gap and no overlap.
    ok(p_bin(0.0) == 0, "p_bin lower edge")
    ok(p_bin(0.999) == len(FP_EDGES) - 2, "p_bin upper edge")
    ok(p_bin(1.0) == len(FP_EDGES) - 2, "p_bin at 1.0 does not overflow")

    # 23: overlap is symmetric and non-negative on disjoint intervals.
    ok(overlap(0, 1, 2, 3) == 0.0 and overlap(0, 5, 1, 2) == 1.0, "overlap")

    # 24: shape_ratio refuses to invent structure from a single bin.
    ok(math.isnan(shape_ratio([3.0])), "shape_ratio undefined on one bin")

    # 25-29: f_p numerator and denominator share one lagged state.  Execution
    # price is deliberately in the opposite tail and must remain only a mark.
    seg = state_segments_from_points([(0.0, 0.10), (10.0, 0.90)], [], lag_s=0.25)
    ok(seg == [(0.25, 10.25, 0.10), (10.25, WINDOW_S, 0.90)],
       "quote state begins only after frozen lag")
    row = {"elapsed": 1.0, "exec_p_up": 0.90, "notional": 5.0, "micro": False}
    c, cx, nv, dg = fp_window_counts([row], seg)
    ok(c[p_bin(0.10)] == 1.0 and c[p_bin(0.90)] == 0.0,
       "arrival uses midpoint-state bin, not execution-price bin")
    ok(cx[p_bin(0.10)] == 1.0 and nv[p_bin(0.10)] == 5.0
       and dg["exec_state_bin_mismatch"] == 1,
       "count, ex-micro count and notional share one state bin")
    ok(abs(state_dwell(seg)[p_bin(0.10)] - 10.0) < 1e-12,
       "denominator uses the same midpoint-state bin")

    # A gap kills the old state; subtracting the gap and resuming the old quote
    # would create a forbidden stale-state interval from 8.0 to 10.25.
    gseg = state_segments_from_points([(0.0, 0.10), (10.0, 0.90)],
                                      [(5.0, 8.0)], lag_s=0.25)
    ok(gseg == [(0.25, 5.0, 0.10), (10.25, WINDOW_S, 0.90)]
       and state_mid_at(gseg, 8.5) is None,
       "state is not carried across collector gaps")

    # 30: a trade without an admitted state is explicit, never silently binned.
    _, _, _, no_state = fp_window_counts([
        {"elapsed": 8.5, "exec_p_up": 0.10, "notional": 1.0, "micro": True}
    ], gseg)
    ok(no_state["no_state"] == 1 and no_state["admitted"] == 0,
       "missing state is counted and rejected")

    # 31: a quote received before a gap cannot mature after the reconnect.
    interrupted = state_segments_from_points([(4.9, 0.10), (6.0, 0.90)],
                                             [(5.0, 5.1)], lag_s=0.25)
    ok(state_mid_at(interrupted, 5.2) is None
       and state_mid_at(interrupted, 6.25) == 0.90,
       "gap between quote receipt and lag maturity invalidates the quote")

    # 32-36: the EARLIEST-first sampler. These are the checks that would have
    # caught the one-day sample being published as a four-day one.
    ok(slug_day("btc-updown-5m-1787184000") == "2026-08-20", "slug_day decodes")
    ok(slug_day("btc-updown-5m-1787451600") == "2026-08-23", "slug_day rolls over")
    _expect("slug with no epoch", ValueError, lambda: slug_day("btc-updown-5m"))

    # The ordering property the whole defect rests on: sorting slugs sorts TIME,
    # so a truncated sample is the EARLIEST windows, never a spread of them.
    _slugs = ["btc-updown-5m-1787451600", "btc-updown-5m-1787184000",
              "btc-updown-5m-1787270400"]
    ok([slug_day(x) for x in sorted(_slugs)]
       == ["2026-08-20", "2026-08-21", "2026-08-23"],
       "sorting slugs sorts by day -- truncation takes the EARLIEST")

    # provenance() without `sampled=` must SAY it is an upper bound. The two
    # receipts that over-reported n_days: 4 on a one-day sample predate this.
    _pv = provenance()
    ok(_pv["sampled_is_known"] is False and _pv["days_sampled"] is None
       and "UPPER BOUND" in str(_pv["warning"]),
       "provenance() with no sampled= declares itself an upper bound")

    # 37-42: R-20. The bar is anchored BY VALUE and this code cannot move it.
    ok(abs(f_star(0.642, 0.287) - 0.309) < 0.001, "f* reproduces btc's frozen 0.309")
    ok(abs(f_star(0.778, 0.759) - 0.494) < 0.001, "f* reproduces eth's frozen 0.494")
    _expect("signed markout rejected", ValueError, lambda: f_star(0.642, -0.287))

    # R-20's own worked example: btc at |markout|_lo = 0.110 -> 14.6%.
    cf = would_move_bar("btc", 0.642, 0.110, measured_r_hi=0.219)
    ok(abs(cf["f_star_low_counterfactual"] - 0.146) < 0.001,
       "R-20 worked example reproduces 14.6%")
    ok(cf["f_star_low_operative"] == 0.309 and cf["bar_moved"] is False,
       "the operative bar is UNCHANGED by a counterfactual")
    # R-38 clause (d): the counterfactual may VACATE, never revive.
    ok(cf["would_vacate"] is True and cf["verdict_under_operative_bar"] == "DEAD"
       and cf["verdict_under_counterfactual"] == "UNDETERMINED_PENDING_RERUN",
       "R-38(d): an amendment vacates to UNDETERMINED, never to NOT_DEAD")
    ok("RE-RUNNING" in cf["obligation"],
       "R-38(d): vacating buys an OBLIGATION, not a result")

    # R-40 clause (e): limbo is LOUD.
    ok(cf["provenance_is_permanent"] is True
       and cf["vacated_provenance"].startswith("VACATED -- was DEAD under bar"),
       "R-40(e): the vacated verdict carries its history permanently")
    ok(cf["register_row_required"] is True and cf["register_row_filed"] is False,
       "R-40(e): a re-run row is OWED and is never silently marked filed")
    _dry = file_vacate_obligation(cf)
    ok(_dry["filed"] is False and "RE-RUN OWED" in _dry["row"],
       "R-40(e): the owed row is produced; filing is deliberate, not silent")
    _expect("nothing to file when no vacate", ValueError,
            lambda: file_vacate_obligation({"register_row_required": False}))
    ok(all(v != "NOT_DEAD" for k, v in cf.items()
           if k == "verdict_under_counterfactual"),
       "R-38(d): no code path lets an amendment reach a favourable verdict")

    # symmetry: a MORE-dead re-publication is equally a finding and equally inert
    worse = would_move_bar("btc", 0.642, 0.900, measured_r_hi=0.219)
    ok(worse["delta"] > 0 and worse["bar_moved"] is False
       and worse["would_vacate"] is False,
       "a more-adverse re-publication also leaves the bar alone")

    # 43-47: R-35. Pooling across samplers is refused, not warned about.
    _ok_pool = [{"sampling_rule": "DAY_STRATIFIED", "protocol": "a"},
                {"sampling_rule": "DAY_STRATIFIED", "protocol": "b"}]
    ok(assert_poolable(_ok_pool) == "DAY_STRATIFIED", "same rule pools")
    _expect("mixed samplers refused", ValueError, lambda: assert_poolable(
        [{"sampling_rule": "EARLIEST_FIRST"}, {"sampling_rule": "DAY_STRATIFIED"}]))
    _expect("undeclared rule refused", ValueError,
            lambda: assert_poolable([{"protocol": "x"}]))

    _cand = resampled_markout_is_a_candidate("btc", 0.642, 0.110, measured_r_hi=0.219)
    ok(_cand["f_star_low_operative"] == 0.309 and _cand["bar_moved"] is False,
       "a re-sampled markout leaves the operative bar untouched")
    ok(_cand["status"] == "CANDIDATE_NOT_A_BAR" and _cand["amendment_is_free"] is False,
       "and is labelled a candidate whose amendment is no longer free")

    # ----------------------------------------------------------------------
    # RR12-1 AT ITS ORIGIN: the roots, driven in BOTH directions on real trees.
    #
    # The old expression is not argued about here, it is RUN: a scratch code
    # tree with no `data/` is built, this module is copied into it, and a CHILD
    # imports it twice -- once as shipped, once with the pre-fix line restored.
    # One finds the tape, the other finds nothing. That is the defect and its
    # repair in the same check.
    # ----------------------------------------------------------------------
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf

    _PROBE = (
        "import sys, json\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import flow_intensity as fi\n"
        "print(json.dumps({'code_root': str(fi.CODE_ROOT),\n"
        "                  'repo': str(fi.REPO),\n"
        "                  'data_root': str(fi.DATA_ROOT),\n"
        "                  'branch': fi.DATA_ROOT_BRANCH,\n"
        "                  'raw': str(fi.RAW),\n"
        "                  'register_default': str(fi.REGISTER_DEFAULT),\n"
        "                  'n_paths': len(fi._archive_paths())}))\n")

    def _child(tree: Path, prefix_mutant: bool) -> dict:
        """Import this module from `tree` and report what it resolved."""
        src = Path(__file__).resolve().parent
        dst = tree / "live" / "pm_research"
        dst.mkdir(parents=True, exist_ok=True)
        for _n in ("flow_intensity.py", "pm_tape_density.py"):
            _sh.copy2(src / _n, dst / _n)
        if prefix_mutant:
            _t = (dst / "flow_intensity.py").read_text(encoding="utf-8")
            _t = _t.replace("REPO = DATA_ROOT\n",
                            "REPO = Path(__file__).resolve().parents[2]\n", 1)
            (dst / "flow_intensity.py").write_text(_t, encoding="utf-8")
        _pr = tree / "probe.py"
        _pr.write_text(_PROBE, encoding="utf-8")
        _r = _sp.run([sys.executable, str(_pr), str(dst)],
                     capture_output=True, text=True, timeout=600)
        if _r.returncode != 0:                               # pragma: no cover
            raise AssertionError(f"probe failed: {_r.stderr[-400:]}")
        return json.loads(_r.stdout.strip().splitlines()[-1])

    with _tf.TemporaryDirectory() as _t:
        _tree = Path(_t) / "tapeless"
        _fixed = _child(_tree, prefix_mutant=False)
        ok(_fixed["code_root"] == str(_tree)
           and _fixed["repo"] != str(_tree)
           and _fixed["data_root"] != str(_tree)
           and _fixed["branch"] == "3_canonical"
           and _fixed["register_default"].startswith(str(_tree))
           and not _fixed["register_default"].startswith(
               _fixed["data_root"] + "/orch")
           and _fixed["n_paths"] > 0,
           f"RR12-1 REPAIRED, DRIVEN: imported from a code tree with NO "
           f"`data/`, this module resolves CODE_ROOT to that tree and "
           f"DATA_ROOT elsewhere (branch {_fixed['branch']}), and finds "
           f"{_fixed['n_paths']} archive paths")
        _broken = _child(Path(_t) / "tapeless_mutant", prefix_mutant=True)
        ok(_broken["repo"] == _broken["code_root"]
           and _broken["n_paths"] == 0,
           f"RR12-1 KNOWN-BAD, THE ORIGINAL EXPRESSION RESTORED IN A COPY: "
           f"`REPO = Path(__file__).resolve().parents[2]` in the SAME tree "
           f"resolves the tape to the code tree and finds "
           f"{_broken['n_paths']} archive paths. That empty map is what made "
           f"`warning_window.select_holdout` return no days and "
           f"`entirely_post_freeze` read FALSE for every day, 09-01 and "
           f"09-02 included")

        # AND THE CANONICAL BEHAVIOUR IS UNCHANGED -- a tree that CARRIES the
        # tape still resolves to ITSELF, which is what the nightly unit does.
        _own = Path(_t) / "carries"
        (_own / "data" / "pm_5min").mkdir(parents=True)
        (_own / "data" / "pm_5min" / "raw").symlink_to(RAW)
        _r3 = _child(_own, prefix_mutant=False)
        ok(_r3["repo"] == _r3["data_root"] == _r3["code_root"] == str(_own)
           and _r3["branch"] == "2_code_tree_carries_the_tape"
           and _r3["register_default"].startswith(str(_own))
           and _r3["n_paths"] > 0,
           f"RR12-1 REGRESSION CONTROL: a code tree that DOES carry the tape "
           f"still resolves to ITSELF (branch {_r3['branch']}, "
           f"{_r3['n_paths']} paths) -- so in the canonical tree, which is "
           f"where the nightly unit runs, the new resolution returns exactly "
           f"the path the old expression returned. The repair changes what a "
           f"WORKTREE reads and nothing else. **AND THIS IS THE "
           f"CONFIGURATION DA24-R2 FOUND RED**: the roots are EQUAL here, "
           f"which is what broke the register check written as a string "
           f"prefix -- so the case production runs in is now driven "
           f"explicitly rather than being whichever tree the suite happened "
           f"to be launched from")

    # DA24-R2, AND IT IS A REGRESSION I SHIPPED AND REPORTED GREEN. This
    # check first read:
    #     startswith(CODE_ROOT) and not startswith(DATA_ROOT + "/orch")
    # which FAILS whenever CODE_ROOT == DATA_ROOT, because then the two
    # prefixes are the same string -- and the roots are equal in exactly the
    # configuration production runs in (branch 2, a code tree that carries the
    # tape, which is what the canonical tree is). Round 24 reported this module
    # green at 54 from a WORKTREE, where the roots differ; in the canonical
    # tree it was rc 1 on both launchers. A fix that is green where it was
    # tested and red where it runs is worse than no fix, and the two checks
    # here even contradicted each other -- the one below said in as many words
    # that the roots are allowed to be equal.
    #
    # THE PROPERTY, STATED SO IT HOLDS IN BOTH CONFIGURATIONS: the register
    # default is always under CODE_ROOT, and it is under DATA_ROOT only when
    # DATA_ROOT *is* CODE_ROOT. `is_relative_to` compares path components, so
    # it cannot be fooled by a shared string prefix either.
    _reg_under_code = REGISTER_DEFAULT.is_relative_to(CODE_ROOT)
    _reg_under_data = REGISTER_DEFAULT.is_relative_to(DATA_ROOT)
    ok(_reg_under_code
       and (CODE_ROOT == DATA_ROOT or not _reg_under_data),
       f"RR12-1 THE OTHER HALF: the register default is under CODE_ROOT "
       f"({CODE_ROOT}) and under DATA_ROOT only when the two are the same "
       f"tree (equal={CODE_ROOT == DATA_ROOT}, branch "
       f"{DATA_ROOT_BRANCH}). The register is TRACKED and exists in every "
       f"worktree; a run from a worktree that filed into the canonical tree's "
       f"register would be writing to a tree it is not in")
    ok((CODE_ROOT == DATA_ROOT) == (DATA_ROOT_BRANCH
                                    == "2_code_tree_carries_the_tape"),
       f"RR12-1 and the two roots are equal EXACTLY when the resolver took "
       f"branch 2 (equal={CODE_ROOT == DATA_ROOT}, branch "
       f"{DATA_ROOT_BRANCH}) -- so this suite's configuration is a STATED "
       f"fact rather than an accident of where it was launched, and the "
       f"discriminating half of the check above is driven in BOTH "
       f"configurations by the children below")

    print(f"flow_intensity selftest: {checks} checks OK")
    return 0


# --------------------------------------------------------------------------

def _fmt_prof(prof: Sequence[float], every: int = 2) -> str:
    return " ".join(f"{prof[k]:.2f}" for k in range(0, len(prof), every))


def report_fr(res: dict[str, Any]) -> list[str]:
    L = ["## f_r — arrival intensity vs seconds-to-settlement", "",
         f"Bins: {res['bins']} x {res['bin_w']:.0f} s. Exposure-corrected on the "
         f"`{ERA}` covered set ({res['n_gap_touched']} gap-touched windows).", "",
         "`shape_ratio` = max/min across bins. 1.0 is flat.", "",
         "| coin | windows | trades | micro | count/s ratio | notional/s ratio |",
         "|---|---:|---:|---:|---:|---:|"]
    for coin, d in res["coins"].items():
        L.append(f"| {coin} | {d['n_windows']} | {d['n_trades']:,} | {d['n_micro']:,} "
                 f"| {d['count']['shape_ratio']:.2f} | {d['notional']['shape_ratio']:.2f} |")
    L += ["", "### Profiles, arrivals/s by bin (open -> close, every 2nd bin)", "",
          "```"]
    L.append("r_mid  " + " ".join(f"{bin_r_mid(k):5.0f}" for k in range(0, FR_BINS, 2)))
    for coin, d in res["coins"].items():
        L.append(f"{coin:6s} " + " ".join(
            f"{d['count']['profile'][k]:5.2f}" for k in range(0, FR_BINS, 2)))
    L.append("```")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?",
                    choices=["fr", "fp", "sample-days", "promotion-days"],
                    default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--per-coin", type=int, default=10)
    ap.add_argument("--all-windows", action="store_true",
                    help="ignore era/ledger coverage (NO exposure correction)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "promotion-days":
        print(json.dumps(promotion_days(), indent=1))
        return 0
    if a.cmd == "sample-days":
        for line in report_sample_days():
            print(line)
        return 0
    if a.cmd == "fr":
        res = fr(n_boot=a.boot, era_only=not a.all_windows)
        print(json.dumps({c: {k: v for k, v in d.items() if k != "slug"}
                          for c, d in res["coins"].items()}, indent=1)[:200])
        for line in report_fr(res):
            print(line)
        (PM / "derived").mkdir(parents=True, exist_ok=True)
        (PM / "derived/flow_fr.json").write_text(json.dumps(res, indent=1))
        return 0
    if a.cmd == "fp":
        res = fp(per_coin=a.per_coin, n_boot=a.boot)
        (PM / "derived").mkdir(parents=True, exist_ok=True)
        (PM / "derived/flow_fp.json").write_text(json.dumps(res, indent=1))
        for coin, d in res["coins"].items():
            print(f"{coin:6s} n_win={d['n_windows']:3d} "
                  f"admitted={d['n_trades_state_admitted']:6,}/"
                  f"{d['n_trades_total']:6,} no_state={d['n_trades_no_state']:5,} "
                  f"bin_mismatch={d['n_exec_state_bin_mismatch']:6,} "
                  f"shape={d['count']['shape_ratio']:.2f}")
            print("   dwell_s " + " ".join(f"{x:7.0f}" for x in d["dwell_s"]))
            print("   rate/s  " + " ".join(f"{x:7.3f}" for x in d["count"]["profile"]))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
