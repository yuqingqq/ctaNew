#!/usr/bin/env python3
"""Preflight for a GOVERNED day verdict + its mask. READ-ONLY, predicates only.

Tonight's 00:06Z run writes the first GOVERNED verdict and the first canonical
mask. This exists so that verdict is checked by COMPUTED PREDICATES rather
than by someone reading JSON at 00:14Z (rule 10), and so the check itself has
shown it can fire (rule 15).

WHAT IT WILL NOT DO.
  * It writes NOTHING. Not under `data/`, not anywhere. The selftest proves
    this at runtime by snapshotting the derived directory around a real run.
  * It carries NO threshold, NO minimum complement, and NO field that reads as
    an entitlement (rule 14). R-411(i) and R-411(ii) are the USER's; this tool
    reports the complement and stops.
  * It does not re-implement the mask contract. It calls
    `harmful_forward_scorer.load_blackout_mask`, which already asserts DA's
    schema and refuses on drift -- a third implementation would be a third
    thing to disagree.
  * It does not restate the frozen rule's status vocabulary. The set is
    EXTRACTED from the frozen module itself, so it cannot drift, and an empty
    extraction REFUSES rather than passing everything.

A predicate that cannot be evaluated is a STATUS with its reason, never a
pass. The exit code is 0 only when every predicate PASSED.

    python3 live/pm_research/da_governed_verdict_preflight.py --day 20260902
    python3 -m live.pm_research.da_governed_verdict_preflight --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import da_content_liveness_rule as CLR                          # noqa: E402
import harmful_forward_scorer as HFS                            # noqa: E402

# RR12-1 -- the SPLIT. This tool READS data, so its root is the tree holding
# the tape, imported from the lowest-level reader rather than restated.
CODE_ROOT = Path(__file__).resolve().parents[2]
import pm_tape_density as _TDROOT                                # noqa: E402
DATA_ROOT = _TDROOT.DATA_ROOT
REPO = DATA_ROOT
DERIVED = DATA_ROOT / "data/pm_5min/derived"

#: The scheduled unit's identity string, as the launcher composes it. Matched
#: by PREFIX on the whole field, not by `in` -- a substring test would accept
#: "UNATTRIBUTED hand run ... scheduled unit run" and any other text that
#: merely mentions it.
SCHEDULED_PREFIX = "scheduled unit run, da-midnight-verify.service"

#: R-405's established blackout interval on 2026-09-02, carried as a FACT to
#: intersect against, never as a bar.
R405_DAY = "20260902"
R405_FROM_UTC = "2026-09-02T01:35:00Z"
R405_TO_UTC = "2026-09-02T04:55:00Z"


#: CO-R4 -- EXIT CODES ARE THREE DIFFERENT STATEMENTS, not two.
#:   0  every predicate passed
#:   1  the artifacts were read and a PREDICATE DID NOT PASS -- a real result
#:   3  REFUSED: the artifacts could not be read as this day's at all, so no
#:      predicate was evaluated. Sharing rc 1 with a failing predicate would
#:      tell a reader "the day failed" when nothing was checked, and the
#:      timer's redirect would leave a ZERO-BYTE file with the reason only on
#:      stderr. The refusal is therefore emitted as JSON on STDOUT.
RC_ALL_PASSED = 0
RC_PREDICATE_DID_NOT_PASS = 1
RC_REFUSED = 3

class PreflightRefused(Exception):
    """An input this preflight must not summarise."""


def frozen_status_set() -> set[str]:
    """The statuses the FROZEN rule can emit, EXTRACTED from that module.

    The frozen file exports no vocabulary constant and may not be edited
    (rule 4), so the set is read out of its own compiled constants. That is
    importing the set rather than restating it: a status added or renamed in
    the frozen module moves this set with it.

    AN EMPTY EXTRACTION REFUSES. A checker whose membership test is against an
    empty set passes nothing and fails nothing.
    """
    seen: set[str] = set()

    def walk(code):
        for c in code.co_consts:
            if isinstance(c, str) and c.startswith("CONTENT_"):
                seen.add(c)
            elif hasattr(c, "co_consts"):
                walk(c)
    import types
    for obj in vars(CLR).values():
        if isinstance(obj, types.FunctionType):
            walk(obj.__code__)
        elif isinstance(obj, str) and obj.startswith("CONTENT_"):
            seen.add(obj)
    if len(seen) < 2:
        raise PreflightRefused(
            f"REFUSED: extracted {len(seen)} status name(s) from the frozen "
            f"rule. A membership test against an (almost) empty set is not a "
            f"test -- the extraction is broken, not the artifact.")
    return seen


#: Wording that was true before R-442 and is false after it. Kept as DATA so
#: the check can NAME what it found rather than describe it.
STALE_DECISION_PHRASES = ("awaiting the USER's word", "NOT reached by the")


def _assert_decisions_coherent(esc: dict) -> None:
    """A decision cannot be both settled and open, and settled wording must move.

    Two failures this makes impossible, both one careless edit away: (1) a key
    in `ruled` AND `still_open` -- a contradiction inside one block, which a
    reader can only resolve by guessing; (2) the pre-ruling wording surviving
    after the entry that settled the question, which is how
    `freeze_disposition` read as "awaiting the USER's word" for the whole of
    R-442's afternoon.
    """
    ruled = esc.get("ruled") or {}
    still = esc.get("still_open") or {}
    both = sorted(set(ruled) & set(still))
    if both:
        raise PreflightRefused(
            f"REFUSED: {both} appear in BOTH `ruled` and `still_open`. A "
            f"decision cannot be settled and open at once, and a reader "
            f"cannot resolve the contradiction from the artifact.")
    blob = json.dumps(esc)
    hit = [ph for ph in STALE_DECISION_PHRASES if ph in blob]
    if hit:
        raise PreflightRefused(
            f"REFUSED: the decisions block still carries pre-ruling wording "
            f"{hit}. After the entry that settled a question, that phrasing "
            f"makes a settled thing read as open -- which is the failure this "
            f"check exists to catch, not a style point.")


def _iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> float:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def day_bounds(day: str) -> tuple[int, int]:
    d = dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()), int(d.timestamp()) + 86400


def _p(out: list, name: str, state: str, detail: str, computed_from: str,
       ev: Any = None):
    """state is PASS / FAIL / UNEVALUABLE / ABSENT -- only PASS passes."""
    out.append({"predicate": name, "state": state, "pass": state == "PASS",
                "detail": detail, "computed_from": computed_from,
                "evidence": ev})


def preflight(day: str, derived: Path | None = None) -> dict[str, Any]:
    d = DERIVED if derived is None else derived
    vpath = d / f"da_dayverdict_{day}.json"
    mpath = d / f"da_blackout_mask_{day}.json"
    res: list[dict] = []

    if not vpath.exists():
        raise PreflightRefused(
            f"REFUSED: no verdict at {vpath}. An absent verdict is not a "
            f"failing day -- nothing was verified, and reporting predicates "
            f"over a file that does not exist would be the empty-set trap.")
    raw = vpath.read_bytes()
    try:
        v = json.loads(raw)
    except ValueError as e:
        raise PreflightRefused(f"REFUSED: {vpath.name} does not parse ({e}).")
    vsha = hashlib.sha256(raw).hexdigest()
    if str(v.get("day_token")) != str(day):
        raise PreflightRefused(
            f"REFUSED: {vpath.name} is for day {v.get('day_token')!r}, not "
            f"{day!r}. Checking one day's predicates against another's "
            f"artifact is worse than not checking.")

    lo, hi = day_bounds(day)

    # (a) identity of the writer, and the write is AFTER the day closed -------
    wr = v.get("write_reason")
    _p(res, "write_reason_is_the_scheduled_unit",
       "PASS" if isinstance(wr, str) and wr.startswith(SCHEDULED_PREFIX)
       else "FAIL",
       "the verdict was written by the scheduled unit, matched as a PREFIX of "
       "the whole field. A substring test would accept an UNATTRIBUTED hand "
       "run whose text merely mentions the unit -- which is the shape that "
       "replaced two 00:06Z verdicts on 2026-09-02",
       "verdict.write_reason", {"write_reason": wr,
                                "required_prefix": SCHEDULED_PREFIX})
    asof = v.get("as_of_utc")
    try:
        asof_ts = _parse(asof)
        state = "PASS" if asof_ts >= hi else "FAIL"
    except Exception:
        asof_ts, state = None, "UNEVALUABLE"
    _p(res, "as_of_is_after_the_day_closed", state,
       "a governed verdict is written after its UTC day ends; an earlier "
       "as-of is a mid-day snapshot, whatever else it says",
       "verdict.as_of_utc vs the day's UTC close",
       {"as_of_utc": asof, "day_close_utc": _iso(hi),
        "seconds_after_close": None if asof_ts is None
        else round(asof_ts - hi, 1)})

    # (b) the four conjuncts, as the verdict states them ---------------------
    vs = v.get("verdict_split") or {}
    names = {"FINISHED": "day_closed", "AFTER": "post_freeze_pass",
             "ADMISSIBLE": "era_admissible", "HEALTHY": "day_quality_pass"}
    vals = {k: vs.get(f) for k, f in names.items()}
    missing = [k for k, x in vals.items() if not isinstance(x, bool)]
    if missing:
        _p(res, "four_forward_race_conjuncts_present", "UNEVALUABLE",
           f"conjunct(s) {missing} are absent or not boolean; eligibility "
           f"must never be obtainable by omitting a question",
           "verdict.verdict_split", vals)
    else:
        stated = vs.get("race_accrual_eligible")
        recomputed = all(vals.values())
        _p(res, "four_forward_race_conjuncts_present", "PASS",
           "FINISHED / AFTER / ADMISSIBLE / HEALTHY are each present as "
           "booleans. ADMISSIBLE is an INTERLOCK against an unruled "
           "collector, not a quality grade (R-409)",
           "verdict.verdict_split", vals)
        _p(res, "eligibility_equals_its_own_conjunction",
           "PASS" if stated == recomputed else "FAIL",
           "race_accrual_eligible recomputed here from the four conjuncts, "
           "not read from the headline",
           "verdict.verdict_split", {"stated": stated,
                                     "recomputed": recomputed})

    # (c) the frozen rule's block, against the frozen rule's OWN vocabulary ---
    clr = v.get("content_liveness_rule")
    try:
        vocab = frozen_status_set()
    except PreflightRefused as e:
        vocab, vocab_err = set(), str(e)
    else:
        vocab_err = None
    if not isinstance(clr, dict):
        _p(res, "content_liveness_block_present_with_a_frozen_status",
           "ABSENT", "the verdict carries no content_liveness_rule block",
           "verdict.content_liveness_rule", None)
    elif vocab_err:
        _p(res, "content_liveness_block_present_with_a_frozen_status",
           "UNEVALUABLE", vocab_err, "frozen module extraction", None)
    else:
        st = clr.get("status")
        _p(res, "content_liveness_block_present_with_a_frozen_status",
           "PASS" if st in vocab else "FAIL",
           "the block's status is a member of the vocabulary EXTRACTED from "
           "the frozen module, so a renamed or invented status fails here "
           "rather than reading as fine",
           "verdict.content_liveness_rule.status vs da_content_liveness_rule",
           {"status": st, "frozen_vocabulary": sorted(vocab),
            "governs": clr.get("governs")})

    # (d) the verdict's record of its own mask -------------------------------
    bma = v.get("blackout_mask_artifact")
    if bma is None:
        _p(res, "verdict_records_its_own_mask_artifact", "ABSENT",
           "the verdict carries no blackout_mask_artifact field. R-412's "
           "producer obligation postdates artifacts written before it, so "
           "this is a STATUS about the artifact's vintage, not a claim that "
           "the day was bad",
           "verdict.blackout_mask_artifact",
           {"verdict_as_of": asof})
    else:
        _p(res, "verdict_records_its_own_mask_artifact",
           "PASS" if bma.get("status") == "WRITTEN" else "FAIL",
           "the verdict states that its mask was WRITTEN; a NOT_WRITTEN "
           "status is a refusal a scorer must honour, never an empty mask",
           "verdict.blackout_mask_artifact.status",
           {k: bma.get(k) for k in ("status", "path", "total_masked_windows",
                                    "n_coins", "day_closed_calendar",
                                    "carrying_commit", "why")})

    # (e) the mask itself, through the EXISTING contract ----------------------
    mask = None
    if not mpath.exists():
        _p(res, "mask_artifact_present", "ABSENT",
           f"no mask at {mpath.name}. Absent is not empty: a scorer must "
           f"refuse rather than assume nothing was dark (R-412)",
           "filesystem", {"path": str(mpath)})
    else:
        try:
            mask = HFS.load_blackout_mask(day, mpath)
            _p(res, "mask_artifact_present", "PASS",
               "the mask parses and satisfies the schema the SCORER asserts "
               "-- reused, not re-implemented, so producer and consumer "
               "cannot drift apart behind two green suites",
               "harmful_forward_scorer.load_blackout_mask",
               {"path": str(mpath), "artifact": mask.get("artifact"),
                "schema_asserted_by": mask.get("schema_asserted_by"),
                "total_masked_windows": mask.get("total_masked_windows")})
        except Exception as e:
            _p(res, "mask_artifact_present", "FAIL",
               "the mask failed the scorer's own schema contract",
               "harmful_forward_scorer.load_blackout_mask", {"error": str(e)})

    if mask is not None:
        _p(res, "mask_day_closed_calendar_true",
           "PASS" if mask.get("day_closed_calendar") is True else "FAIL",
           "a mid-day mask is a diagnostic, not a scoring input; the "
           "producer's own consumer_note makes this the separating field",
           "mask.day_closed_calendar",
           {"day_closed_calendar": mask.get("day_closed_calendar")})
        raw_mask = json.loads(mpath.read_text())
        bad = [c for c, m in (raw_mask.get("coins") or {}).items()
               if m.get("agrees_with_frozen_L1_numerator") is not True]
        _p(res, "mask_agrees_with_the_frozen_L1_numerator",
           "PASS" if not bad else "FAIL",
           "every coin's mask is the population the frozen L1 already counts; "
           "a mask that disagrees would mask windows the bars still charge for",
           "mask.coins[*].agrees_with_frozen_L1_numerator",
           {"coins_not_agreeing": bad,
            "n_coins": len(raw_mask.get("coins") or {})})
        if isinstance(bma, dict) and bma.get("status") == "WRITTEN":
            same_path = str(Path(bma.get("path", "")).resolve()) == \
                str(mpath.resolve())
            same_n = bma.get("total_masked_windows") == \
                mask.get("total_masked_windows")
            same_cc = bma.get("carrying_commit") == \
                (raw_mask.get("producer") or {}).get("carrying_commit")
            _p(res, "verdict_and_mask_on_disk_agree",
               "PASS" if (same_path and same_n and same_cc) else "FAIL",
               "the mask the verdict NAMES is the mask on disk: same path, "
               "same masked count, same carrying_commit. A verdict pointing "
               "at a different artifact than the one that exists is the "
               "hardest kind of wrong to see by reading",
               "verdict.blackout_mask_artifact vs the mask file",
               {"path_matches": same_path, "count_matches": same_n,
                "carrying_commit_matches": same_cc,
                "verdict_says": {"path": bma.get("path"),
                                 "n": bma.get("total_masked_windows"),
                                 "cc": bma.get("carrying_commit")},
                "mask_says": {"path": str(mpath),
                              "n": mask.get("total_masked_windows"),
                              "cc": (raw_mask.get("producer") or {}
                                     ).get("carrying_commit")}})

    # (f) FACTS -- per coin, and the R-405 interval. No bars here. -----------
    facts: dict[str, Any] = {}
    if mask is not None:
        rawc = json.loads(mpath.read_text()).get("coins") or {}
        for c, m in sorted(rawc.items()):
            tot, nm = m.get("n_windows_total"), m.get("n_masked")
            facts[c] = {
                "n_windows_total": tot, "n_masked": nm,
                "complement_windows": (None if tot is None or nm is None
                                       else tot - nm),
                "longest_run_windows": m.get("longest_run_windows"),
            }
        if day == R405_DAY:
            a, b = _parse(R405_FROM_UTC), _parse(R405_TO_UTC)
            for c, wins in (mask.get("per_coin") or {}).items():
                inside = [w for w in wins if a <= w < b]
                facts.setdefault(c, {})["r405_interval_overlap"] = {
                    "interval": [R405_FROM_UTC, R405_TO_UTC],
                    "n_masked_inside": len(inside),
                    "share_of_masked": (round(len(inside) / len(wins), 4)
                                        if wins else None),
                    "note": ("a FACT about where this day's masked windows "
                             "fall relative to the R-405 blackout. It is not "
                             "a predicate and nothing passes or fails on it"),
                }

    # (g) whose decisions are still open ------------------------------------
    esc = {}
    cq = ((v.get("blackout_mask_and_complement") or {})
          .get("complement_quality") or {})
    for k in cq:
        if k.startswith("ESCALATION"):
            esc[k] = cq[k]
    # RULED vs STILL OPEN, kept apart -- and an answered question must MOVE.
    # This block listed `freeze_disposition` as "awaiting the USER's word"
    # after R-442 had ruled it, which is the same failure R-434 §2's mirror
    # rule exists to prevent, one entry later: an artifact that keeps asking a
    # settled thing asks it forever. `_assert_decisions_coherent` below makes
    # the two halves unable to contradict each other or to carry the stale
    # wording, so this cannot recur by editing one half and forgetting the
    # other.
    esc["ruled"] = {
        "R-411(i)": "minimum complement for G-COUNTING -- RULED at R-424 §4: "
                    ">= 144 of 288 windows; every good window is scored "
                    "regardless. Emitted per coin-day as `counts_toward_G`.",
        "R-411(ii)": "the P1 denominator on a complement -- RULED at R-424 "
                     "§4: per UNMASKED hour governs, calendar-24h reported "
                     "beside it.",
        "R-408(3)": "the v2 absolute-floor freeze -- RULED at R-424 §2: "
                    "FROZEN, governing from 2026-09-03.",
        # R-434 §2: mirror ALL FOUR of R-424 §7's rulings, not only the three
        # that came from this instrument's own escalations. A reader of this
        # block should see the same count the register does.
        "R-408(2)": "the Phase-2 winner -- RULED at R-424 §3: the composed "
                    "candidate DOES NOT ADVANCE, Q1_arrival is the surviving "
                    "component of record, and there is NO race admission.",
    }
    esc["ruled"]["freeze_disposition"] = (
        "RULED at R-442: race on the frozen bytes at 1b53929 (R-424 §6 "
        "adopted verbatim); no re-freeze; multiplicity 2.")
    # EMPTY, and it SAYS it is empty. A bare {} reads as "not computed" to a
    # reader who does not know the schema; the note is what makes the absence
    # a statement.
    esc["still_open"] = {}
    esc["note"] = ("`ruled` entries are settled and cite the entry that "
                   "settled them; `still_open` is what remains the USER's, "
                   "and it is EMPTY: nothing remains the USER's as of R-442. "
                   "The 09-02 accrual call is R-409's principle applied at "
                   "scoring, not a decision listed here. Any key beginning "
                   "ESCALATION_ found in the artifact is carried above, "
                   "unchanged, as the producer wrote it.")
    _assert_decisions_coherent(esc)
    mseam = (json.loads(mpath.read_text()).get("v2_seam")
             if mpath.exists() else None)

    failing = [r["predicate"] for r in res if not r["pass"]]
    n_absent = sum(1 for r in res if r["state"] == "ABSENT")
    return {
        "tool": "da_governed_verdict_preflight",
        "day": day,
        "read_only": True,
        # DA10-R2: this is what the coordinator reads at 00:14Z, so it says
        # which tree it read from and by which resolver branch.
        "roots": _TDROOT.data_root_provenance(),
        "verdict_path": str(vpath), "verdict_sha256": vsha,
        "verdict_as_of_utc": asof,
        "mask_path": str(mpath) if mpath.exists() else None,
        "mask_sha256": (hashlib.sha256(mpath.read_bytes()).hexdigest()
                        if mpath.exists() else None),
        "checked_at_utc": _iso(dt.datetime.now(dt.timezone.utc).timestamp()),
        "predicates": res,
        "n_predicates": len(res),
        "n_failing": len(failing),
        "failing": failing,
        "facts_per_coin": facts,
        "v2_seam": mseam,
        "open_decisions": esc,
        "classification": (
            "GOVERNED_VERDICT_COMPLETE" if not failing else
            "PRE_GOVERNED_ARTIFACT"
            if n_absent and n_absent == len(failing) else "FAILED"),
        "classification_note": (
            "PRE_GOVERNED_ARTIFACT means every non-passing predicate is an "
            "ABSENT field on an artifact written before the obligation "
            "existed -- a statement about vintage, not about the day. The "
            "exit code is still non-zero, because a predicate that did not "
            "pass did not pass."),
        "decides_nothing": (
            "This tool computes predicates and reports facts. It carries no "
            "threshold and no minimum complement; R-411(i) and R-411(ii) are "
            "the USER's (rule 14)."),
    }


# --------------------------------------------------------------------------
def selftest() -> int:
    import copy
    import tempfile
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)

    vocab = frozen_status_set()
    ok({"CONTENT_LIVE", "CONTENT_THIN", "CONTENT_LIVENESS_UNJUDGEABLE",
        "CONTENT_LIVENESS_UNRESOLVED"} <= vocab,
       f"the frozen vocabulary is EXTRACTED from the frozen module and holds "
       f"its four statuses (got {len(vocab)})")

    # THE EMPTY-EXTRACTION REFUSAL, EXERCISED. A mutation deleting it
    # survived the whole suite, because the real frozen module never yields
    # an empty set -- a guard that cannot fire is not a guard (rule 16). This
    # drives it with a module that exports no statuses at all.
    import types as _t
    _saved_clr = globals()["CLR"]
    try:
        _empty = _t.ModuleType("empty_rule")
        _empty.measure_day = lambda *a, **k: None
        globals()["CLR"] = _empty
        try:
            frozen_status_set()
            ok(False, "an empty vocabulary extraction must REFUSE")
        except PreflightRefused as _e:
            ok("is not a test" in str(_e),
               "KNOWN-BAD: a frozen module yielding NO statuses REFUSES -- a "
               "membership test against an empty set passes nothing and "
               "fails nothing, so it must never be mistaken for a check")
    finally:
        globals()["CLR"] = _saved_clr
    ok(globals()["CLR"] is _saved_clr,
       "and the frozen module reference is restored after the control")

    day = "20260902"
    lo, hi = day_bounds(day)
    base = lo

    def fixture(tmp: Path):
        # 01:35Z is window index 19 and 04:55Z is the end of index 58,
        # so these 40 windows ARE the R-405 interval exactly -- the
        # overlap fact is then a real check rather than a coincidence.
        wins = [base + i * 300 for i in range(19, 59)]
        mask = {
            "artifact": "da_blackout_mask_v1", "day": day,
            "as_of_utc": "2026-09-03T00:06:05Z",
            "day_closed_calendar": True,
            "total_masked_windows": len(wins),
            "producer": {"carrying_commit": "deadbeef" * 5,
                         "module": "da_blackout_mask"},
            "v2_seam": {"status": "INERT_PENDING_USER_FREEZE",
                        "refuses": True},
            "coins": {"btc": {"n_windows_total": 288, "n_masked": len(wins),
                              "masked_windows": wins,
                              "longest_run_windows": len(wins),
                              "agrees_with_frozen_L1_numerator": True,
                              "status": "CONTENT_THIN"}},
        }
        verdict = {
            "day_token": day,
            "write_reason": SCHEDULED_PREFIX + " (INVOCATION_ID=abc)",
            "as_of_utc": "2026-09-03T00:06:01+00:00",
            "verdict_split": {"day_closed": True, "post_freeze_pass": True,
                              "era_admissible": True,
                              "day_quality_pass": True,
                              "race_accrual_eligible": True},
            "content_liveness_rule": {"status": "CONTENT_THIN",
                                      "governs": True},
            "blackout_mask_artifact": {
                "status": "WRITTEN",
                "path": str(tmp / f"da_blackout_mask_{day}.json"),
                "total_masked_windows": len(wins), "n_coins": 1,
                "day_closed_calendar": True,
                "carrying_commit": "deadbeef" * 5},
            "blackout_mask_and_complement": {"complement_quality": {
                "ESCALATION_no_minimum_complement_size": "the USER's"}},
        }
        (tmp / f"da_blackout_mask_{day}.json").write_text(json.dumps(mask))
        (tmp / f"da_dayverdict_{day}.json").write_text(json.dumps(verdict))
        return verdict, mask

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        v0, m0 = fixture(tmp)
        r = preflight(day, tmp)
        ok(r["n_failing"] == 0
           and r["classification"] == "GOVERNED_VERDICT_COMPLETE",
           f"POSITIVE CONTROL: a well-formed governed verdict+mask passes "
           f"every predicate ({r['n_predicates']} of them)")
        ok(r["facts_per_coin"]["btc"]["complement_windows"] == 288 - 40
           and r["facts_per_coin"]["btc"]["longest_run_windows"] == 40,
           "and the per-coin FACTS are computed: complement = total - masked")
        ok(r["facts_per_coin"]["btc"]["r405_interval_overlap"]["n_masked_inside"]
           == 40,
           "and the R-405 overlap is reported as a FACT for 09-02 -- all 40 "
           "fixture windows sit inside 01:35-04:55Z")
        _od = r["open_decisions"]
        ok(len(_od["ruled"]) == 5
           and set(_od["ruled"]) == {"R-411(i)", "R-411(ii)", "R-408(3)",
                                     "R-408(2)", "freeze_disposition"}
           and "NO race admission" in _od["ruled"]["R-408(2)"]
           and "R-424" in _od["ruled"]["R-411(i)"]
           and "R-442" in _od["ruled"]["freeze_disposition"]
           and "1b53929" in _od["ruled"]["freeze_disposition"]
           and _od["still_open"] == {}
           and "EMPTY" in _od["note"],
           "R-442: all FIVE decisions read as RULED -- the same count the "
           "register carries for R-424 plus R-442 -- and `still_open` is "
           "EMPTY and says so. An artifact that lists an answered question as "
           "open asks it forever")
        # ---- R-442: the block cannot contradict itself, driven both ways --
        _base_esc = {"ruled": dict(_od["ruled"]), "still_open": {},
                     "note": "EMPTY"}
        _assert_decisions_coherent(_base_esc)
        ok(True, "R-442 POSITIVE CONTROL: the real decisions block is "
                 "coherent -- a guard that only ever refuses passes nothing")
        try:
            _assert_decisions_coherent(
                {"ruled": dict(_od["ruled"]),
                 "still_open": {"freeze_disposition": "still open, somehow"},
                 "note": "EMPTY"})
            ok(False, "a key in BOTH halves must REFUSE")
        except PreflightRefused as _e:
            ok("appear in BOTH" in str(_e)
               and "freeze_disposition" in str(_e),
               "R-442 KNOWN-BAD: a key in BOTH `ruled` and `still_open` "
               "REFUSES and NAMES it -- a decision cannot be settled and open "
               "at once, and a reader cannot resolve that from the artifact")
        try:
            _assert_decisions_coherent(
                {"ruled": dict(_od["ruled"]),
                 "still_open": {"x": "a recommendation is on record awaiting "
                                     "the USER's word."},
                 "note": "EMPTY"})
            ok(False, "the pre-ruling wording must REFUSE")
        except PreflightRefused as _e:
            ok("pre-ruling wording" in str(_e)
               and "awaiting the USER's word" in str(_e),
               "R-442 KNOWN-BAD: the pre-ruling phrasing ANYWHERE in the "
               "block REFUSES and quotes it -- that exact sentence made a "
               "settled question read as open for an afternoon")
        # AND THE GUARD IS ON THE PRODUCTION PATH, not merely proven. A
        # mutation deleting the call from `preflight()` survived every check
        # above, because they all drive the function directly -- rule 17's
        # class, which this file has now met three times. Poisoning the phrase
        # list with a string the REAL block contains makes the real call the
        # only thing that can refuse.
        _saved_ph = globals()["STALE_DECISION_PHRASES"]
        try:
            # THE FORM, NOT ONE CITATION. Poisoning with "RULED at R-442"
            # worked only while that entry was the newest: a legitimate
            # re-ruling in band (rule 13) changes the citation, the poisoned
            # run stops raising, and this control goes red for a change that
            # is not a defect. The block's own discipline is that every ruled
            # entry CITES the entry that settled it, so the form is what is
            # invariant.
            globals()["STALE_DECISION_PHRASES"] = ("RULED at ",)
            try:
                preflight(day, tmp)
                ok(False, "the coherence guard is NOT called by preflight()")
            except PreflightRefused as _e:
                ok("RULED at " in str(_e),
                   "R-442 WIRING: `preflight()` itself runs the coherence "
                   "guard -- poisoned with the citation FORM every ruled "
                   "entry must carry, so the production path is the only "
                   "thing that can raise, and it survives any future "
                   "re-ruling")
        finally:
            globals()["STALE_DECISION_PHRASES"] = _saved_ph
        ok(globals()["STALE_DECISION_PHRASES"] is _saved_ph,
           "R-442: the phrase list is restored after the wiring control")

        # ---- PER-PREDICATE MUTANTS: each must FAIL BY NAME ---------------
        def remut(mutate) -> dict:
            v = copy.deepcopy(v0)
            m = copy.deepcopy(m0)
            mutate(v, m)
            (tmp / f"da_dayverdict_{day}.json").write_text(json.dumps(v))
            (tmp / f"da_blackout_mask_{day}.json").write_text(json.dumps(m))
            return preflight(day, tmp)

        for label, mut, pred in (
            ("write_reason becomes an unattributed hand run",
             lambda v, m: v.update(
                 write_reason="UNATTRIBUTED hand run of da_midnight_verify.sh"),
             "write_reason_is_the_scheduled_unit"),
            ("write_reason merely MENTIONS the unit (substring trap)",
             lambda v, m: v.update(
                 write_reason="UNATTRIBUTED hand run; cf " + SCHEDULED_PREFIX),
             "write_reason_is_the_scheduled_unit"),
            ("as_of moves before the day closed",
             lambda v, m: v.update(as_of_utc="2026-09-02T12:00:00+00:00"),
             "as_of_is_after_the_day_closed"),
            ("the headline disagrees with its conjuncts",
             lambda v, m: v["verdict_split"].update(day_quality_pass=False),
             "eligibility_equals_its_own_conjunction"),
            ("an invented liveness status",
             lambda v, m: v["content_liveness_rule"].update(status="FINE"),
             "content_liveness_block_present_with_a_frozen_status"),
            ("the verdict says its mask was NOT written",
             lambda v, m: v["blackout_mask_artifact"].update(
                 status="NOT_WRITTEN"),
             "verdict_records_its_own_mask_artifact"),
            ("a mid-day mask",
             lambda v, m: m.update(day_closed_calendar=False),
             "mask_day_closed_calendar_true"),
            ("a coin whose mask disagrees with the frozen L1 numerator",
             lambda v, m: m["coins"]["btc"].update(
                 agrees_with_frozen_L1_numerator=False),
             "mask_agrees_with_the_frozen_L1_numerator"),
            ("the verdict names a different carrying_commit",
             lambda v, m: v["blackout_mask_artifact"].update(
                 carrying_commit="0" * 40),
             "verdict_and_mask_on_disk_agree"),
        ):
            rr = remut(mut)
            ok(pred in rr["failing"],
               f"MUTANT ({label}) fails BY NAME at {pred}")
            ok(rr["classification"] == "FAILED",
               f"MUTANT ({label}) classifies as FAILED, not pre-governed")
        fixture(tmp)   # restore

        # ---- KNOWN-BADS it must REFUSE -----------------------------------
        (tmp / f"da_blackout_mask_{day}.json").unlink()
        r_nomask = preflight(day, tmp)
        ok(any(p["predicate"] == "mask_artifact_present"
               and p["state"] == "ABSENT" for p in r_nomask["predicates"])
           and r_nomask["n_failing"] > 0,
           "KNOWN-BAD: an ABSENT mask is a STATUS that does not pass -- "
           "absent is not empty")
        fixture(tmp)

        v_other = copy.deepcopy(v0)
        v_other["day_token"] = "20260901"
        (tmp / f"da_dayverdict_{day}.json").write_text(json.dumps(v_other))
        try:
            preflight(day, tmp)
            ok(False, "a verdict for another day must REFUSE")
        except PreflightRefused as e:
            ok("is for day" in str(e),
               "KNOWN-BAD: a verdict for a DIFFERENT day REFUSES -- checking "
               "one day's predicates against another's artifact is worse "
               "than not checking")
        fixture(tmp)

        m_bad = copy.deepcopy(m0)
        m_bad["coins"]["btc"]["n_masked"] = 39      # disagrees with its list
        (tmp / f"da_blackout_mask_{day}.json").write_text(json.dumps(m_bad))
        r_bad = preflight(day, tmp)
        ok(any(p["predicate"] == "mask_artifact_present"
               and p["state"] == "FAIL" for p in r_bad["predicates"]),
           "KNOWN-BAD: a mask whose count disagrees with its own window list "
           "FAILS the scorer's contract -- reused, so producer and consumer "
           "cannot drift apart behind two green suites")
        fixture(tmp)

        (tmp / f"da_dayverdict_{day}.json").unlink()
        try:
            preflight(day, tmp)
            ok(False, "an absent verdict must REFUSE")
        except PreflightRefused as e:
            ok("An absent verdict is not a failing day" in str(e),
               "KNOWN-BAD: an ABSENT verdict REFUSES and says nothing was "
               "verified, rather than reporting predicates over a file that "
               "does not exist")

    # ---- CO-R4: A REFUSAL IS MACHINE-READABLE, ON STDOUT, WITH ITS OWN RC
    import subprocess as _sp
    with tempfile.TemporaryDirectory() as _td2:
        _empty = Path(_td2)
        _prog = (f"import sys; sys.path.insert(0, {str(_HERE)!r});\n"
                 "import da_governed_verdict_preflight as P;\n"
                 f"P.DERIVED = __import__('pathlib').Path({str(_empty)!r});\n"
                 "raise SystemExit(P.main(['--day','20260902']))\n")
        _r = _sp.run([sys.executable, "-c", _prog], capture_output=True,
                     text=True)
        ok(_r.returncode == RC_REFUSED,
           f"CO-R4: a refusal exits {RC_REFUSED}, DISTINCT from "
           f"{RC_PREDICATE_DID_NOT_PASS} (a predicate that did not pass). "
           f"Sharing one code would report 'the day failed' when nothing was "
           f"checked (got {_r.returncode})")
        ok(_r.stdout.strip() != "",
           "CO-R4: stdout is NOT EMPTY on a refusal -- the timer redirects "
           "stdout to preflight_<day>.json, and a zero-byte file cannot be "
           "told from 'the tool never ran'")
        _j = json.loads(_r.stdout)
        ok(_j["classification"] == "REFUSED" and _j["exit_code"] == RC_REFUSED
           and "REFUSED" in _j["refusal"] and _j["day"] == "20260902",
           "CO-R4: and it PARSES as JSON carrying the refusal, the day and "
           "the exit code -- machine-readable, not a traceback on stderr")
        ok(len(_r.stdout.encode()) > 0 and _r.stdout.lstrip().startswith("{"),
           "CO-R4: the redirected file would hold a JSON object, never zero "
           "bytes")

    # ---- READ-ONLY, proven at RUNTIME, not by reading the source ---------
    if DERIVED.is_dir():
        before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                  for p in DERIVED.iterdir() if p.is_file()}
        try:
            preflight("20260901")
        except PreflightRefused:
            pass
        after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                 for p in DERIVED.iterdir() if p.is_file()}
        ok(before == after,
           f"READ-ONLY, MEASURED: a real run over {len(before)} files in the "
           f"derived directory changes no name, mtime or size. A source grep "
           f"for `write` would be vocabulary; this is the property")

    print(f"da_governed_verdict_preflight selftests: {checks} checks passed")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.day:
        raise SystemExit("REFUSED: --day YYYYMMDD")
    try:
        r = preflight(a.day)
    except PreflightRefused as e:
        # ON STDOUT, ALWAYS, AND AS JSON. A refusal that reaches only stderr
        # leaves the timer's `> preflight_<day>.json` holding zero bytes --
        # which a later reader cannot distinguish from "the tool never ran".
        print(json.dumps({
            "tool": "da_governed_verdict_preflight",
            "day": a.day,
            "read_only": True,
            "classification": "REFUSED",
            # DA10-R2: the REFUSED shape carries them too -- a refusal is
            # exactly when "which tree did you look in?" is the question.
            "roots": _TDROOT.data_root_provenance(),
            "refusal": str(e),
            "checked_at_utc": _iso(
                dt.datetime.now(dt.timezone.utc).timestamp()),
            "exit_code": RC_REFUSED,
            "note": ("REFUSED means NO predicate was evaluated -- it is not a "
                     "failing day. rc 3 is distinct from rc 1 for exactly "
                     "that reason."),
        }, indent=1))
        return RC_REFUSED
    r = r
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"day {r['day']}  verdict {r['verdict_sha256'][:16]}  "
              f"mask {(r['mask_sha256'] or 'ABSENT')[:16]}  "
              f"as_of {r['verdict_as_of_utc']}")
        for p in r["predicates"]:
            print(f"  [{p['state']:11s}] {p['predicate']}")
            if p["state"] != "PASS":
                print(f"                {p['detail'][:100]}")
        print(f"\nFACTS per coin:")
        for c, f in sorted(r["facts_per_coin"].items()):
            ov = f.get("r405_interval_overlap")
            print(f"  {c}: masked {f['n_masked']}/{f['n_windows_total']}  "
                  f"complement {f['complement_windows']}  longest run "
                  f"{f['longest_run_windows']}"
                  + (f"  R-405 overlap {ov['n_masked_inside']}"
                     f" ({ov['share_of_masked']})" if ov else ""))
        print(f"\nOPEN DECISIONS (USER's): "
              f"{sorted(r['open_decisions'].get('register_ids_transcribed', {}))}")
        print(f"CLASSIFICATION: {r['classification']}  "
              f"({r['n_predicates'] - r['n_failing']}/{r['n_predicates']} "
              f"predicates passed)")
    return (RC_ALL_PASSED if r["n_failing"] == 0
            else RC_PREDICATE_DID_NOT_PASS)


if __name__ == "__main__":
    raise SystemExit(main())
