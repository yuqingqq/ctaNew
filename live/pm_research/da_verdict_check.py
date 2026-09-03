#!/usr/bin/env python3
"""Grade a CLOSED-DAY verdict artifact against what a forward day must carry.

The standing 00:06Z cadence, mechanized. It exists because the check was
otherwise going to be typed out live at midnight against a dispatch's prose,
and a check improvised at the moment it matters is the one that misses a field.

WHAT IT ASSERTS, all recomputed rather than read:
  * the artifact is about the day it is being graded for, and the day is
    CLOSED -- an OPEN-day artifact is REFUSED, not graded leniently;
  * all four ACCRUAL_RULE conjuncts are PRESENT as bools, and
    `race_accrual_eligible` EQUALS their conjunction (recomputed here, so a
    headline that disagrees with its own inputs cannot pass);
  * `rule`, `era_role` and `selector_era` are carried, and `selector_era`
    matches the era DERIVED for that day from the ledger -- not the era the
    artifact says it used;
  * the 0h breadth disclosure is present in EVERY scope with BOTH
    denominators, and on a closed day the two denominators COINCIDE.

IT DECIDES NOTHING. A day that fails HEALTHY is a VALID OUTCOME and this
checker says so in those words: it grades the ARTIFACT's completeness, never
the day's quality (rule 14). `accrues` is reported as the artifact's own
computed conjunction, with n and as-of beside it.

THE READ PROVES ITSELF (R-289): a population counter and a typed-field
counter, and `assert_read()` REFUSES when either is short. A checker that
found nothing because it read nothing is the empty-set trap in the checker's
chair, and this programme has filed two of those an hour apart.

    python3 live/pm_research/da_verdict_check.py --selftest
    python3 live/pm_research/da_verdict_check.py --day 20260901
    python3 live/pm_research/da_verdict_check.py --artifact /path/to/verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import da_forward_day_verify as D
import da_race_withdrawals as RW                              # noqa: E402

# RR12-1: the DATA root, resolved once in `pm_tape_density`.
import pm_tape_density as _TDROOT  # noqa: E402
DERIVED = _TDROOT.DATA_ROOT / "data/pm_5min/derived"

CONJUNCTS = ("day_closed", "post_freeze_pass", "era_admissible",
             "day_quality_pass")
DISCLOSURE_FIELDS = ("role", "windows_affected_COIN_LEVEL",
                     "windows_complete_elapsed", "affected_over_elapsed",
                     "affected_over_288")


class Reads:
    def __init__(self) -> None:
        self.scopes = 0
        self.fields: dict[str, int] = {}
        self.errors: list[str] = []

    def typed(self, obj: dict, key: str, types, ctx: str):
        if not isinstance(obj, dict) or key not in obj:
            self.errors.append(f"{ctx}: {key!r} ABSENT")
            return None
        v = obj[key]
        if not isinstance(v, types):
            self.errors.append(
                f"{ctx}: {key!r} is {type(v).__name__}, not "
                f"{getattr(types, '__name__', types)}")
            return None
        self.fields[key] = self.fields.get(key, 0) + 1
        return v

    def assert_read(self, min_scopes: int, required: dict[str, int]) -> None:
        if self.scopes < min_scopes:
            raise SystemExit(
                f"REFUSED: {self.scopes} scope(s) read, expected at least "
                f"{min_scopes}. A verdict graded over no coins is the "
                f"empty-set trap (R-289).")
        short = {k: (self.fields.get(k, 0), n) for k, n in required.items()
                 if self.fields.get(k, 0) < n}
        if short:
            raise SystemExit(
                "REFUSED: fields not read as typed -- "
                + "; ".join(f"{k}: {g} of {w}" for k, (g, w)
                            in sorted(short.items())))


def _add(out: list, name: str, ok: bool, detail: str, ev: Any = None):
    out.append({"check": name, "pass": bool(ok), "detail": detail,
                "evidence": ev})


def check_verdict(rep: dict, day_token: str,
                  era_ledger: Path | None = None) -> dict:
    r = Reads()
    res: list[dict] = []

    tok = r.typed(rep, "day_token", str, "root")
    closed = r.typed(rep, "day_closed_calendar", bool, "root")
    _add(res, "artifact_is_the_named_CLOSED_day",
         tok == day_token and closed is True,
         "the artifact is about the day being graded AND the calendar day had "
         "finished when it was written. An open-day artifact is a different "
         "object and is refused rather than graded",
         {"day_token": tok, "asked_for": day_token,
          "day_closed_calendar": closed, "as_of_utc": rep.get("as_of_utc")})

    split = r.typed(rep, "verdict_split", dict, "root") or {}
    vals = {c: r.typed(split, c, bool, "verdict_split") for c in CONJUNCTS}
    elig = r.typed(split, "race_accrual_eligible", bool, "verdict_split")
    recomputed = all(v is True for v in vals.values())
    _add(res, "four_ACCRUAL_RULE_conjuncts_present_as_bools",
         all(isinstance(v, bool) for v in vals.values()),
         "FINISHED / AFTER / ADMISSIBLE / HEALTHY are each present and each a "
         "bool -- a missing conjunct is eligibility obtained by not asking",
         vals)
    _add(res, "eligibility_EQUALS_its_own_conjunction",
         elig is not None and elig == recomputed,
         "race_accrual_eligible recomputed from the four conjuncts here, not "
         "taken from the headline -- a headline that disagrees with its own "
         "inputs has contradicted a table three times in this programme",
         {"stated": elig, "recomputed": recomputed, "conjuncts": vals})

    # DA24-R1. THE FIELD THE RACE IS COUNTED BY MUST REFLECT THE WITHDRAWAL.
    # `race_accrual_eligible` is the pure four-conjunct computation and reads
    # TRUE for a withdrawn day -- correctly, because the day IS eligible and
    # is deliberately not entered. Until this check existed, the independent
    # verifier validated the field that says the day counts, and anyone
    # counting G by it got 3 of 5 instead of 2.
    #
    # `withdrawn` is recomputed FROM THE REGISTRY, not read from the artifact:
    # a checker that takes the producer's word for the policy fact is checking
    # nothing. The artifact must then AGREE with the registry.
    _wd_reg = bool(RW.withdrawal_for(day_token))
    _wd_art = r.typed(split, "withdrawn_from_race", bool, "verdict_split")
    _cnt_art = r.typed(split, "counts_toward_race", bool, "verdict_split")
    _cnt_recomputed = recomputed and not _wd_reg
    _add(res, "withdrawal_matches_the_registry",
         _wd_art is not None and _wd_art == _wd_reg,
         "withdrawn_from_race recomputed from da_race_withdrawals here, not "
         "taken from the artifact -- an artifact that disagrees with the "
         "registry about a USER ruling is the one a reader must not believe",
         {"stated": _wd_art, "registry": _wd_reg,
          "authority": (RW.withdrawal_for(day_token) or {}).get("authority")})
    _add(res, "counts_toward_race_EQUALS_conjuncts_AND_not_withdrawn",
         _cnt_art is not None and _cnt_art == _cnt_recomputed,
         "counts_toward_race recomputed from the four conjuncts AND the "
         "registry. THIS is the field G is counted by (D.ACCRUAL_RULE says "
         "so); race_accrual_eligible answers a different question and reads "
         "TRUE for a withdrawn day",
         {"stated": _cnt_art, "recomputed": _cnt_recomputed,
          "eligible": recomputed, "withdrawn": _wd_reg})

    rule = r.typed(split, "rule", str, "verdict_split")
    role = r.typed(split, "era_role", str, "verdict_split")
    _add(res, "rule_text_and_era_role_carried",
         rule == D.ACCRUAL_RULE and bool(role)
         and "INTERLOCK, NOT A QUALITY GRADE" in (role or ""),
         "the verdict carries the rule it was judged under and states that "
         "era is an interlock rather than a quality grade (USER 2026-09-01)",
         {"rule_matches_module": rule == D.ACCRUAL_RULE,
          "era_role_prefix": (role or "")[:48]})

    sel = rep.get("selector_era")
    r.fields["selector_era"] = r.fields.get("selector_era", 0) + 1
    era = D.day_era_admission(day_token, era_ledger)
    touched = era.get("eras_touched") or []
    derived = touched[0] if (era.get("era_pure") and len(touched) == 1) else None
    _add(res, "selector_era_matches_the_DERIVED_era",
         sel == derived,
         "the era the accrual selector loaded equals the era derived from the "
         "ledger for this day -- recomputed from the ledger, never read back "
         "from the artifact that is being graded",
         {"selector_era_in_artifact": sel, "derived_from_ledger": derived,
          "eras_touched": touched, "era_pure": era.get("era_pure"),
          "race_admissible_by_era": era.get("race_admissible_by_era")})

    # ---- the 0h disclosure, in EVERY scope --------------------------------
    bars = r.typed(rep, "day_bar_v2", dict, "root") or {}
    per_coin = r.typed(rep, "per_coin", dict, "root") or {}
    missing, mismatched, seen = [], [], []
    for scope, blocks in (("day_bar_v2", bars),
                          ("per_coin", {c: (v or {}).get("day_bar_v2")
                                        for c, v in per_coin.items()
                                        if isinstance(v, dict)})):
        for coin, b in sorted(blocks.items()):
            if not isinstance(b, dict):
                missing.append(f"{scope}/{coin}: no bar block")
                continue
            r.scopes += 1
            seen.append(f"{scope}/{coin}")
            d = b.get("windows_affected_disclosure")
            if not isinstance(d, dict):
                missing.append(f"{scope}/{coin}: disclosure absent")
                continue
            r.fields["windows_affected_disclosure"] = \
                r.fields.get("windows_affected_disclosure", 0) + 1
            gone = [k for k in DISCLOSURE_FIELDS if k not in d]
            if gone:
                missing.append(f"{scope}/{coin}: missing {gone}")
                continue
            if d.get("affected_over_elapsed") != d.get("affected_over_288"):
                mismatched.append(
                    f"{scope}/{coin}: {d['affected_over_elapsed']} vs "
                    f"{d['affected_over_288']}")
    # `bool(seen)` is REDUNDANT behind `assert_read`'s min_scopes refusal and
    # is kept anyway. Recorded rather than hidden: a mutation removing it
    # SURVIVES the suite, because the zero-scope case is refused at the read
    # layer before this predicate is ever evaluated. It is belt-and-braces for
    # the day that threshold changes -- not a control that has shown it can
    # fire, and rule 16 says the difference must be visible.
    _add(res, "breadth_disclosure_carried_in_every_scope",
         bool(seen) and not missing,
         "the 0h disclosure is present with both denominators in the whole-day "
         "bars AND every per-coin block. An empty scope list is a refusal, not "
         "a pass",
         {"scopes_checked": len(seen), "scopes": seen, "missing": missing})
    _add(res, "on_a_closed_day_the_two_denominators_COINCIDE",
         bool(seen) and not mismatched,
         "affected/elapsed and affected/288 must be equal once the day is "
         "complete -- a disagreement means the elapsed count did not reach 288 "
         "and the artifact is not a closed-day report",
         {"mismatched": mismatched})

    per_ok, per_bad = 0, []
    for coin, v in sorted(per_coin.items()):
        cs = (v or {}).get("verdict_split")
        if not isinstance(cs, dict):
            per_bad.append(f"{coin}: no verdict_split")
            continue
        gone = [c for c in CONJUNCTS if not isinstance(cs.get(c), bool)]
        if gone or "era_role" not in cs or "rule" not in cs:
            per_bad.append(f"{coin}: missing {gone or ['era_role/rule']}")
            continue
        if cs.get("race_accrual_eligible") != all(cs[c] for c in CONJUNCTS):
            per_bad.append(f"{coin}: headline disagrees with its conjuncts")
            continue
        per_ok += 1
    _add(res, "per_coin_splits_carry_the_same_four_conjuncts",
         per_ok > 0 and not per_bad,
         "each coin-day carries its own four conjuncts, rule and era_role, and "
         "its own headline agrees with them (R-211(3): coin-days pass or fail "
         "independently)",
         {"coins_ok": per_ok, "problems": per_bad})

    # ---- the USER-FROZEN content-liveness rule (R-386), wired R-402 -----
    clr = r.typed(rep, "content_liveness_rule", dict, "root") or {}
    clr_status = r.typed(clr, "status", str, "content_liveness_rule")
    clr_governs = clr.get("governs")
    r.fields["clr_governs"] = r.fields.get("clr_governs", 0) + 1
    frozen = r.typed(clr, "frozen_rule", dict, "content_liveness_rule") or {}
    comp = r.typed(clr, "composition_with_HEALTHY", dict,
                   "content_liveness_rule") or {}
    try:
        import da_content_liveness_rule as CLR
        want_gov = CLR.governs(day_token)
        want_bars = (CLR.L1_SEVERITY_MAX, CLR.L2_RUN_WINDOWS_MAX)
    except Exception:
        want_gov, want_bars = None, (None, None)
    _add(res, "frozen_content_liveness_rule_is_carried",
         bool(clr_status) and bool(frozen) and bool(comp)
         and clr_governs == want_gov,
         "the verdict carries the USER-frozen rule's status, the bars that "
         "judged it, and its composition -- and its governs field agrees with "
         "what the frozen module answers for this day, recomputed here rather "
         "than read back",
         {"status": clr_status, "governs_in_artifact": clr_governs,
          "governs_per_module": want_gov,
          "bars_in_artifact": (frozen.get("L1_severity_max"),
                               frozen.get("L2_run_windows_max")),
          "bars_per_module": want_bars,
          "effective_from_day": frozen.get("effective_from_day")})
    _add(res, "CONTENT_THIN_does_not_silently_veto_HEALTHY",
         comp.get("content_thin_vetoes_HEALTHY") is False,
         "the frozen rule's SS7 reserves exclusion to the coordinator and its "
         "SS8 leaves the composition open; a verdict that had adopted it "
         "would be a worker ruling (rules 11/14). If this ever reads True it "
         "must cite the ruling that made it so",
         {"content_thin_vetoes_HEALTHY":
          comp.get("content_thin_vetoes_HEALTHY"),
          "escalated": comp.get("escalated"),
          "day_is_CONTENT_THIN": comp.get("day_is_CONTENT_THIN"),
          "would_flip_under_worst_coin":
              comp.get("would_flip_HEALTHY_under_worst_coin_composition")})

    r.assert_read(2, {"content_liveness_rule": 1, "clr_governs": 1,
                      "frozen_rule": 1, "composition_with_HEALTHY": 1,
                      "day_token": 1, "day_closed_calendar": 1,
                      "verdict_split": 1, "race_accrual_eligible": 1,
                      "rule": 1, "era_role": 1, "day_bar_v2": 1,
                      "windows_affected_disclosure": 2,
                      "day_closed": 1, "post_freeze_pass": 1,
                      "era_admissible": 1, "day_quality_pass": 1})

    fails = [c for c in res if not c["pass"]]
    return {
        "day": day_token,
        "as_of_utc": rep.get("as_of_utc"),
        "checks": res, "n_checks": len(res), "n_failing": len(fails),
        "artifact_complete": not fails,
        # THE OUTCOME, reported and never decided here. DA24-R1: `accrues` is
        # now the WITHDRAWAL-AWARE value, so every consumer of this checker
        # gets the number G is counted by without having to know that a second
        # field exists. The four-conjunct answer is carried beside it under a
        # name that says which question it answers.
        "accrues": _cnt_recomputed,
        "eligible_by_four_conjuncts": recomputed,
        "withdrawn_from_race": _wd_reg,
        "accrues_note": (
            "`accrues` = the four conjuncts AND not withdrawn. A withdrawn "
            "day is ELIGIBLE and does not COUNT; G is counted by this field"),
        "conjuncts": vals,
        "all_pass_quality": rep.get("all_pass"),
        "outcome_note": (
            "A day that fails HEALTHY is a VALID OUTCOME and is recorded as "
            "one; this checker grades whether the ARTIFACT carries what a "
            "forward day must carry, never whether the feed was good. "
            "Excluding a day is the coordinator's ruling (rule 14)."),
        "reads": {"scopes": r.scopes, "fields": dict(sorted(r.fields.items()))},
    }


# --------------------------------------------------------------------------
def _fixture(day="20260829", era="clob_v3_1", closed=True,
             elapsed=288, affected=95) -> dict:
    disc = {"role": "REPORTED_NOT_GOVERNING", "is_a_gate": False,
            "windows_affected_COIN_LEVEL": affected,
            "windows_complete_elapsed": elapsed,
            "affected_over_elapsed": round(affected / elapsed, 4),
            "affected_over_288": round(affected / 288, 4)}
    # DA24-R1: the fixture's day is 2026-08-29, which the USER has WITHDRAWN
    # (R-500). So the fixture carries the withdrawn shape -- eligible by the
    # four conjuncts and NOT counting -- and the checks below read
    # `eligible_by_four_conjuncts` where they are about eligibility and
    # `accrues` where they are about the number G is counted by. A fixture
    # that pretended the day were ordinary would test the wrong artifact.
    split = {"day_closed": True, "post_freeze_pass": True,
             "era_admissible": True, "day_quality_pass": True,
             "race_accrual_eligible": True,
             "withdrawn_from_race": bool(RW.withdrawal_for(day)),
             "counts_toward_race": not bool(RW.withdrawal_for(day)),
             "rule": D.ACCRUAL_RULE,
             "era_role": "INTERLOCK, NOT A QUALITY GRADE (USER 2026-09-01)."}
    bar = {"evaluable": True, "P1_pass": True, "P2_pass": True,
           "P3_pass": True, "windows_affected_disclosure": dict(disc)}
    clr = {"status": "CONTENT_LIVE", "governs": False,
           "frozen_rule": {"module": "da_content_liveness_rule",
                           "effective_from_day": "20260902",
                           "L1_severity_max": 0.08, "L2_run_windows_max": 12},
           "composition_with_HEALTHY": {"content_thin_vetoes_HEALTHY": False,
                                        "escalated": "ESCALATION-FOR-USER"}}
    return {"day_token": day, "day_closed_calendar": closed,
            "content_liveness_rule": clr,
            "as_of_utc": "2026-09-02T00:06:00+00:00", "all_pass": True,
            "selector_era": era, "verdict_split": dict(split),
            "day_bar_v2": {"btc": dict(bar), "eth": dict(bar)},
            "per_coin": {"btc": {"day_bar_v2": dict(bar),
                                 "verdict_split": dict(split)},
                         "eth": {"day_bar_v2": dict(bar),
                                 "verdict_split": dict(split)}}}


def selftest() -> int:
    import copy
    checks = 0

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            print(f"FAIL: {label}")
            raise SystemExit(1)

    def named(rep, n):
        # A MISSING CHECK MUST FAIL BY NAME, NOT RAISE. `next(...)` raised
        # StopIteration when the named check had been deleted, so the suite
        # went red with a TRACEBACK instead of saying which check was gone
        # -- the refusal-must-be-by-name class, third instance in this lane.
        # Every known-bad below asserts `not named(...)["pass"]`, so a
        # sentinel with pass=False makes a DELETED CONSUMER fail at the
        # check that names it.
        return next((c for c in rep["checks"] if c["check"] == n),
                    {"check": n, "pass": False,
                     "detail": f"ABSENT: no check named {n!r} was produced"
                               f" -- the consumer that emits it is gone",
                     "evidence": {"present": sorted(
                         c["check"] for c in rep["checks"])}})

    base = _fixture()
    good = check_verdict(base, "20260829")
    # POSITIVE CONTROL -- a well-formed closed-day verdict must ADMIT, or
    # every refusal below proves nothing (rule 16).
    ok(good["artifact_complete"] and good["n_failing"] == 0
       and good["eligible_by_four_conjuncts"] is True
       and good["accrues"] is False and good["withdrawn_from_race"] is True
       and good["reads"]["scopes"] == 4,
       "POSITIVE CONTROL: a complete closed-day verdict passes all checks "
       "across 4 scopes")

    # KNOWN-BADS, one per assertion, each mutated from the SAME fixture.
    for c in CONJUNCTS:
        m = copy.deepcopy(base)
        m["verdict_split"].pop(c)
        try:
            r = check_verdict(m, "20260829")
            ok(not named(r, "four_ACCRUAL_RULE_conjuncts_present_as_bools")["pass"],
               f"KNOWN-BAD: a missing {c} conjunct must fail")
        except SystemExit:
            ok(True, f"KNOWN-BAD: a missing {c} conjunct refuses the read")

    m = copy.deepcopy(base)
    m["verdict_split"]["day_quality_pass"] = False
    r = check_verdict(m, "20260829")
    ok(not named(r, "eligibility_EQUALS_its_own_conjunction")["pass"]
       and r["accrues"] is False,
       "KNOWN-BAD: a headline claiming eligible while HEALTHY is false fails "
       "the recomputation -- and the recomputed answer is the one reported")

    m = copy.deepcopy(base)
    m["verdict_split"]["day_quality_pass"] = False
    m["verdict_split"]["race_accrual_eligible"] = False
    r = check_verdict(m, "20260829")
    ok(r["artifact_complete"] is True and r["accrues"] is False,
       "DISCRIMINATION: a day that HONESTLY fails HEALTHY is a COMPLETE "
       "artifact reporting a valid outcome -- failing quality is not a "
       "failing artifact, and the checker must not conflate them")

    m = copy.deepcopy(base)
    m["day_closed_calendar"] = False
    r = check_verdict(m, "20260829")
    ok(not named(r, "artifact_is_the_named_CLOSED_day")["pass"],
       "KNOWN-BAD: an OPEN-day artifact is refused rather than graded")

    m = copy.deepcopy(base)
    m["selector_era"] = None
    r = check_verdict(m, "20260829")
    ok(not named(r, "selector_era_matches_the_DERIVED_era")["pass"],
       "KNOWN-BAD: a null selector_era fails against the ledger-derived era")
    m = copy.deepcopy(base)
    m["selector_era"] = "clob_v4_1"
    r = check_verdict(m, "20260829")
    ok(not named(r, "selector_era_matches_the_DERIVED_era")["pass"],
       "KNOWN-BAD: a WRONG-but-plausible era fails too -- the check compares "
       "against the ledger, not against non-emptiness")

    for scope in ("day_bar_v2", "per_coin"):
        m = copy.deepcopy(base)
        (m["day_bar_v2"]["eth"] if scope == "day_bar_v2"
         else m["per_coin"]["eth"]["day_bar_v2"]).pop(
            "windows_affected_disclosure")
        r = check_verdict(m, "20260829")
        ok(not named(r, "breadth_disclosure_carried_in_every_scope")["pass"]
           and scope in str(named(r, "breadth_disclosure_carried_in_every_scope")
                            ["evidence"]),
           f"KNOWN-BAD: a disclosure missing from {scope} fails and names the "
           f"scope -- one coin out of four is enough")

    m = copy.deepcopy(base)
    m["day_bar_v2"]["btc"]["windows_affected_disclosure"][
        "windows_complete_elapsed"] = 113
    m["day_bar_v2"]["btc"]["windows_affected_disclosure"][
        "affected_over_elapsed"] = round(95 / 113, 4)
    r = check_verdict(m, "20260829")
    ok(not named(r, "on_a_closed_day_the_two_denominators_COINCIDE")["pass"],
       "KNOWN-BAD: an open-day elapsed count (113) inside a closed-day "
       "artifact is caught by the denominators disagreeing")

    m = copy.deepcopy(base)
    m["per_coin"]["btc"]["verdict_split"]["race_accrual_eligible"] = False
    r = check_verdict(m, "20260829")
    ok(not named(r, "per_coin_splits_carry_the_same_four_conjuncts")["pass"],
       "KNOWN-BAD: a per-coin headline disagreeing with its own conjuncts "
       "fails -- the whole-day split passing does not cover the coin splits")

    m = copy.deepcopy(base)
    m["day_bar_v2"] = {}
    m["per_coin"] = {}
    try:
        check_verdict(m, "20260829")
        ok(False, "KNOWN-BAD: a verdict with NO coins must REFUSE")
    except SystemExit as e:
        ok("empty-set trap" in str(e),
           "KNOWN-BAD: zero scopes REFUSES -- 'nothing missing' from nothing "
           "is the empty-set trap in the checker's chair")

    m = copy.deepcopy(base)
    m["verdict_split"]["rule"] = "a day accrues iff it looks fine"
    r = check_verdict(m, "20260829")
    ok(not named(r, "rule_text_and_era_role_carried")["pass"],
       "KNOWN-BAD: a rule text that is not ACCRUAL_RULE fails -- the artifact "
       "must carry the rule it was actually judged under")

    # END-TO-END POSITIVE CONTROL ON REAL DATA. The fixture above is my own
    # construction, so it proves the checker's logic and NOTHING about whether
    # the producer emits what the checker demands -- the two could drift apart
    # and both suites stay green (rule 17). This grades a verdict the real
    # `verify_day` just produced, on a real closed day.
    if D.PM_GAPS.exists():
        real = D.verify_day("20260829", 1787897340.0)
        rr = check_verdict(real, "20260829")
        ok(rr["artifact_complete"] and rr["reads"]["scopes"] >= 2,
           "END-TO-END: a verdict produced by the REAL verify_day on 08-29 "
           f"carries everything this checker demands "
           f"({rr['n_checks'] - rr['n_failing']}/{rr['n_checks']}, "
           f"{rr['reads']['scopes']} scopes) -- producer and checker are not "
           f"drifting apart behind two green suites")
        # SUPERSEDED BY R-497 (F)(1), AND THE CHECK MOVES WITH THE RULING.
        # This asserted `era_admissible is False` and `accrues is False` for
        # 08-29 -- true under the unruled `# pre-O1` default that the USER has
        # now replaced ("We check the data quality and only use qualifiable
        # data"). Asserting the old value would pin a ruling that no longer
        # exists. What the check is FOR survives unchanged: that this checker
        # discriminates rather than grading every day the same way, which is
        # why the discriminating leg is kept and driven on the SAME day.
        # WHY THIS ASSERTS THE ERA CONJUNCT AND NOT `accrues`. The composed
        # verdict is NOT tree-independent: `entirely_post_freeze` loads its
        # windows through `warning_window.select_holdout` -> `flow_intensity`,
        # whose `REPO` is `Path(__file__).resolve().parents[2]` -- RR12-1's
        # class, unfixed in that module. Run from a worktree the selector is
        # EMPTY, every day reads `post_freeze_pass` False, and the message
        # blames the data ("absent from the selector"). Measured: 0 archive
        # paths from a bare worktree, 29,438 with the tape mirrored, and the
        # SAME landed code then returns post_freeze True for 09-01/09-02.
        # Asserting `accrues` here would make this check pass or fail on which
        # tree ran it. The ERA conjunct is tree-independent (the ledger path is
        # canonical), and it is the one the ruling moves.
        ok(rr["conjuncts"]["day_quality_pass"] is True
           and rr["conjuncts"]["era_admissible"] is True,
           f"END-TO-END (R-497 (F)(1)): 08-29 reports quality PASS and era "
           f"ADMISSIBLE -- the conjunct the USER ruling moves, read off a "
           f"verdict the REAL verify_day just produced rather than off a "
           f"fixture. It does NOT say the day accrues: the artifact on disk "
           f"carries the pre-ruling answer and cannot be regenerated with the "
           f"attribution a score requires (DA22-B)")
        _r30 = check_verdict(D.verify_day("20260830", 1787897340.0),
                             "20260830")
        ok(_r30["accrues"] is False
           and _r30["conjuncts"]["era_admissible"] is False,
           f"END-TO-END DISCRIMINATION CONTROL, on the day the ruling does "
           f"NOT reach: 08-30 still reports era INADMISSIBLE and does not "
           f"accrue (mid-day boundary). A checker that graded every day "
           f"'complete and accruing' would pass the 08-29 leg above; it "
           f"cannot pass both")

    # KNOWN-BADS for the frozen-rule legs.
    m = copy.deepcopy(base)
    m.pop("content_liveness_rule")
    try:
        r = check_verdict(m, "20260829")
        ok(not named(r, "frozen_content_liveness_rule_is_carried")["pass"],
           "KNOWN-BAD: a verdict with no frozen-rule block must fail")
    except SystemExit:
        ok(True, "KNOWN-BAD: a verdict with no frozen-rule block refuses")
    m = copy.deepcopy(base)
    m["content_liveness_rule"]["governs"] = True
    r = check_verdict(m, "20260829")
    ok(not named(r, "frozen_content_liveness_rule_is_carried")["pass"],
       "KNOWN-BAD: an artifact claiming it was JUDGED on a day the frozen "
       "module says it was not fails -- governs is recomputed, not read")
    m = copy.deepcopy(base)
    m["content_liveness_rule"]["composition_with_HEALTHY"][
        "content_thin_vetoes_HEALTHY"] = True
    r = check_verdict(m, "20260829")
    ok(not named(r, "CONTENT_THIN_does_not_silently_veto_HEALTHY")["pass"],
       "KNOWN-BAD: a verdict that adopted the composition without a ruling "
       "fails -- the checker refuses a worker-made policy change")

    # ----------------------------------------------------------------------
    # DA24-R1: THE FIELD THE RACE IS COUNTED BY, AND ITS CONSUMER.
    # `counts_toward_race` had no consumer -- a field written and never read
    # is a comment with a JSON key. These drive the consumer in BOTH
    # directions, on the withdrawn day and on an ordinary one.
    # ----------------------------------------------------------------------
    m = copy.deepcopy(base)
    m["verdict_split"]["counts_toward_race"] = True          # the doctored day
    r = check_verdict(m, "20260829")
    ok(not named(r, "counts_toward_race_EQUALS_conjuncts_AND_not_withdrawn"
                 )["pass"] and r["artifact_complete"] is False,
       "DA25-1 KNOWN-BAD: an artifact claiming the WITHDRAWN day counts is "
       "REFUSED -- recomputed here from the four conjuncts AND the registry, "
       "never taken from the producer. Delete this consumer and that artifact "
       "is accepted, which is what `counts_toward_race` having no reader "
       "meant for four rounds")
    m = copy.deepcopy(base)
    m["verdict_split"]["withdrawn_from_race"] = False
    r = check_verdict(m, "20260829")
    ok(not named(r, "withdrawal_matches_the_registry")["pass"],
       "DA25-1b KNOWN-BAD: an artifact that disagrees with the REGISTRY about "
       "a USER ruling is refused -- a checker that takes the producer's word "
       "for the policy fact is checking nothing")
    for _f in ("counts_toward_race", "withdrawn_from_race"):
        m = copy.deepcopy(base)
        m["verdict_split"].pop(_f)
        r = check_verdict(m, "20260829")
        ok(r["artifact_complete"] is False,
           f"DA25-1c KNOWN-BAD: a verdict produced WITHOUT {_f} is refusable "
           f"-- an artifact made by a path that never learned about the "
           f"withdrawal must not grade as complete")
    _ord = _fixture(day="20260901", era="clob_v4_1")  # that day's own era
    _ro = check_verdict(_ord, "20260901")
    ok(_ro["artifact_complete"] and _ro["accrues"] is True
       and _ro["withdrawn_from_race"] is False
       and _ro["eligible_by_four_conjuncts"] is True,
       "DA25-1d POSITIVE CONTROL: an ORDINARY day still accrues -- the "
       "consumer discriminates between the two days rather than refusing "
       "everything, which is what makes the False above mean something")
    ok("counts_toward_race" in D.ACCRUAL_RULE
       and "G IS COUNTED BY" in D.ACCRUAL_RULE,
       "DA25-1e AND THE RULE SAYS SO: ACCRUAL_RULE names `counts_toward_race` "
       "as the field G is counted by, so the next reader does not have to "
       "infer which of two booleans is the one")

    print(f"da_verdict_check selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day")
    ap.add_argument("--artifact")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.day or a.artifact):
        raise SystemExit("REFUSED: --day YYYYMMDD or --artifact PATH")
    path = Path(a.artifact) if a.artifact else \
        DERIVED / f"da_dayverdict_{a.day}.json"
    day = a.day or json.loads(path.read_text()).get("day_token")
    if not path.exists():
        raise SystemExit(f"REFUSED: {path} does not exist. An absent verdict "
                         f"is not a passing day (rule 11: silence is not "
                         f"success).")
    rep = check_verdict(json.loads(path.read_text()), day)
    print(json.dumps(rep, indent=1))
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}")
    print(f"\nartifact_complete={rep['artifact_complete']} "
          f"({rep['n_checks'] - rep['n_failing']}/{rep['n_checks']}) "
          f"accrues={rep['accrues']} as_of={rep['as_of_utc']}")
    return 0 if rep["artifact_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
