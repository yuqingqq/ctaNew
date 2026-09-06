"""THE G = 5 DIRECTIONAL RACE READ, DECLARED BEFORE ANYTHING IS OPENED.

Declared with the SAME instrument the interim read was declared with:
`be_read_declaration`'s estimand, frozen decisions, null, masking, scoring
stack, era and reconciliation caveats are CALLED, not restated, so the two
declarations are comparable field-by-field and cannot drift. Only what is
genuinely race-specific is overridden -- the cluster disclosure (G = 5, and
the ruled unit is finally AVAILABLE) and rule 11's scope (five days, two of
them already opened).

R-529(A), THE USER'S RULING, VERBATIM AND UP FRONT because the ruling says
it must be up front: *"every future statement of a race result must say up
front that it establishes DIRECTION AND CONSISTENCY and never a
Holm-clearing verdict."* The arithmetic behind it: the ceiling of a
clustered permutation test is its floor, 1/2^G; at G = 5 with m = 2 the best
possible adjusted p is 0.0625 > 0.05, and the smallest G that clears is 6.
This bounds the SIGNIFICANCE the race can establish, not its VALUE.

THIS DECLARATION OPENS NOTHING. Filing it is not reading. The opening is the
coordinator's or the USER's act, after the reviewer has seen this.

TWO OF THE FIVE DAYS ARE NOT FIRST READS. 09-01 and 09-02 were opened under
the interim read. This read RE-READS them and reads 09-03..09-05 for the
first time -- so their rows are not new evidence in the same sense, and the
declaration says so rather than letting five days look alike.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
DERIVED = ROOT / "data/pm_5min/derived"

RACE_DAYS = ("20260901", "20260902", "20260903", "20260904", "20260905")
ALREADY_OPENED_UNDER_THE_INTERIM = ("20260901", "20260902")


class RaceReadRefused(RuntimeError):
    """A named refusal."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def members() -> dict:
    """m = 2, READ from the freeze audit rather than typed."""
    # THE KEYS ARE NESTED, AND THE FIRST VERSION OF THIS FUNCTION READ THE
    # TOP LEVEL AND REPORTED `read_not_typed: True` BESIDE TWO Nones. A field
    # that announces a successful read must be capable of announcing a failed
    # one, so the presence of the values is now the condition for claiming
    # the read -- not the absence of an exception.
    try:
        import be_freeze_audit as FA
        blk = FA.rule12_conjuncts().get("d_multiplicity_at_freeze") or {}
        m, mem = blk.get("race_multiplicity_at_freeze"), blk.get("race_members")
        if m is None or not mem:
            raise RaceReadRefused(
                f"REFUSED: `d_multiplicity_at_freeze` carried "
                f"multiplicity={m!r} members={mem!r}. A multiplicity that "
                f"reads as None is not a multiplicity that was read.")
        return {"source": "be_freeze_audit.rule12_conjuncts()"
                          "['d_multiplicity_at_freeze']",
                "read_not_typed": True,
                "race_multiplicity_at_freeze": m,
                "members": list(mem),
                "recorded_in_the_frozen_bytes": blk.get(
                    "recorded_in_the_frozen_bytes"),
                "conjunct_holds": blk.get("holds")}
    except Exception as e:                               # noqa: BLE001
        return {"source": "be_freeze_audit.rule12_conjuncts()",
                "read_not_typed": False,
                "status": f"UNREADABLE_AT_DECLARATION_TIME: "
                          f"{type(e).__name__}",
                "declared_from_R_529_A": {
                    "race_multiplicity_at_freeze": 2,
                    "members": ["PM_PLUS_FINE (PRIMARY)",
                                "PM_FINE_EXTENDED (HELD)"]},
                "and_that_is_a_weakness_not_a_fallback":
                    "a number that could not be read is RELAYED from the "
                    "register; it is marked, not silently substituted"}


def cluster_disclosure_race() -> dict:
    """The one block the interim read could not have: G is finally 5."""
    return {
        "ruled_cluster_unit": "UTC day",
        "G_complete_days": 5,
        "unit_actually_used": "UTC day",
        "weaker_than_ruled": False,
        "why_this_differs_from_the_interim_read": (
            "the interim read had G = 1 and had to disclose the WINDOW as a "
            "weaker substitute. Here the ruled unit is available and is the "
            "unit used -- which is the one thing five days bought."),
        "intervals_claimable": "NOT ON THIS READ, and not because of G. "
                               "R-529(A) makes the read DIRECTIONAL; an "
                               "interval would imply an inferential claim "
                               "the ruling forbids.",
        "permutation_floor": {
            "assignments": "2^G = 32 sign assignments over 5 day-clusters",
            "smallest_achievable_one_sided_p": 1.0 / 32,
            "with_multiplicity_2": 2.0 / 32,
            "best_possible_adjusted_p": 0.0625,
            "clears_0_05": False,
            "smallest_G_that_clears": 6,
            "computed_here_not_quoted": True,
        },
        "two_of_the_five_are_re_reads": list(ALREADY_OPENED_UNDER_THE_INTERIM),
        "why_that_matters": (
            "09-01 and 09-02 were opened under the interim read. Their "
            "contribution to a 5-cluster sign test is not a fresh draw in "
            "the sense 09-03..09-05 are, and a reader counting five "
            "independent days would be over-counting. It is DISCLOSED; this "
            "declaration does not resolve how to weight it -- that is the "
            "coordinator's and the USER's."),
    }


def rule_11_race() -> dict:
    return {
        "days_consumed_by_this_read": list(RACE_DAYS),
        "first_opening": [d for d in RACE_DAYS
                          if d not in ALREADY_OPENED_UNDER_THE_INTERIM],
        "already_consumed_before_this_read":
            list(ALREADY_OPENED_UNDER_THE_INTERIM),
        "consumed_the_moment_they_are_opened": True,
        "may_not_be_chosen_on_what_is_seen": [
            "parameter", "threshold", "horizon", "budget", "candidate",
            "the winner criterion", "the day set", "G"],
        "consequence_if_one_moves": (
            "the read is consumed and the moved quantity is selected on seen "
            "data -- rule 11 voids the test, and no later day repairs it"),
        "the_race_is_finished_after_this": (
            "there is no sixth day to add afterwards that would rescue "
            "significance: adding days AFTER seeing this read is selection "
            "on the outcome. If G = 6 is ever wanted it must be declared "
            "BEFORE this read is opened."),
    }


def what_is_opened() -> dict:
    """WHICH FILES, WHICH FIELDS. And what stays shut."""
    rows = []
    for d in RACE_DAYS:
        rec = DERIVED / f"be_forward_day_receipt_{d}.json"
        rows.append({
            "day": d,
            "receipt": str(rec.relative_to(ROOT)) if rec.exists() else None,
            "receipt_sha256": _sha(rec) if rec.exists() else None,
            "receipt_present": rec.exists(),
            "sealed_scores_file":
                f"be_forward_day_SEALED_scores_{d}.json",
            "first_opening": d not in ALREADY_OPENED_UNDER_THE_INTERIM,
        })
    return {
        "per_day": rows,
        "OPENED_BY_THE_READ": {
            "file": "be_forward_day_SEALED_scores_<DAY>.json, one per day",
            "fields": ["per_coin_scores", "report"],
            "field_shapes": {
                "per_coin_scores": "coin -> list of per-action rows",
                "report": "the full complement report",
            },
            "source_of_this_field_list": (
                "be_forward_day.seal(), read at the WRITER rather than by "
                "opening a sealed file -- the field names come from the code "
                "that writes them"),
        },
        "STAYS_SEALED": {
            "file": "be_forward_day_SEALED_feed_<DAY>.jsonl, one per day",
            "why": "the feed carries the per-row key, times, score and "
                   "latency-resolved preventable value. The read needs the "
                   "per-action scores, not the feed, and opening more than "
                   "the estimand needs is consumption without purpose "
                   "(rule 11).",
            "not_opened_by_this_read": True,
        },
        "NOT_OPENED_BY_FILING_THIS": (
            "filing a declaration is not reading. Nothing above is opened by "
            "this artifact; the opening is the coordinator's or the USER's "
            "act after the reviewer has seen this."),
    }


def statistic() -> dict:
    import be_read_declaration as RD
    est = RD.estimand()
    return {
        "per_day_quantity": est["quantity"],
        "per_day_unit_of_analysis": est["unit_of_analysis"],
        "inherited_verbatim_from": "be_read_declaration.estimand()",
        "estimand": est,
        "THE_RACE_STATISTIC": {
            "form": "the SIGN of the per-day quantity, per UTC day, five days",
            "why_the_sign_and_not_the_magnitude": (
                "R-529(A) makes this read DIRECTIONAL. Direction and "
                "consistency is what a 5-cluster sign pattern supports; a "
                "magnitude aggregated across days would invite the "
                "inferential reading the ruling forbids."),
            "reported": ["the per-day sign for each of the five days",
                         "the count of days in each direction",
                         "the per-day magnitude, REPORTED BESIDE the sign "
                         "and not aggregated into a test statistic"],
            "NOT_reported": ["any adjusted p presented as clearing a bar",
                             "any interval",
                             "any statement that the race establishes a "
                             "winner"],
            "multiplicity": members(),
        },
        "R_529_A_VERBATIM_AND_UP_FRONT": (
            "every future statement of a race result must say up front that "
            "it establishes DIRECTION AND CONSISTENCY and never a "
            "Holm-clearing verdict"),
    }


def build() -> dict:
    import be_read_declaration as RD
    return {
        "protocol": "BE_RACE_READ_DECLARATION_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "R_529_A_UP_FRONT": (
            "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY AND NEVER A "
            "HOLM-CLEARING VERDICT. USER ruling R-529(A). At G = 5 with "
            "m = 2 the best possible adjusted p is 0.0625 > 0.05; the "
            "smallest G that clears is 6. This bounds the SIGNIFICANCE the "
            "race can establish, not its VALUE."),
        "AND_THE_SECOND_LIMIT_WHICH_IS_INDEPENDENT": (
            "V2's own HANDOFF: the prior race cannot validate the CHANGED "
            "pipeline, because the pipeline the race scores is not the one "
            "V2 changed. Both limits stand together and neither implies the "
            "other."),
        "read_days": list(RACE_DAYS),
        "G": 5,
        "candidate": "the FROZEN candidate",
        "members": members(),
        "statistic": statistic(),
        "cluster_disclosure": cluster_disclosure_race(),
        "rule_11_in_force": rule_11_race(),
        "what_is_opened": what_is_opened(),
        # REUSED, NOT RESTATED -- the two declarations must not drift.
        "inherited_from_the_interim_declaration": {
            "instrument": "be_read_declaration",
            "why": "the interim read declared these and they have not "
                   "changed; restating them would create a second copy that "
                   "can disagree with the first",
            "frozen_decisions": RD.frozen_decisions(),
            "null_declaration": RD.null_declaration(),
            "masking_treatment": RD.masking_treatment(),
            "scoring_stack": RD.scoring_stack(),
            "era_caveat": RD.era_caveat(),
            "reconciliation_caveat": RD.reconciliation_caveat(),
            "exclusion_vocabulary": {
                "known_at_declaration": list(RD.EXCLUSION_VOCABULARY),
                "set_is_closed": False,
            },
        },
        "declared_in_commit": RD._git("rev-parse", "HEAD"),
        "opens_nothing": True,
        "selects_nothing": True,
        "adjudicates": None,
        "who_decides": "the USER (rule 14); the coordinator authorises the "
                       "read, after the reviewer has seen this declaration",
    }


EXPECTED_CHECKS = 10


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    d = build()
    ok(d["R_529_A_UP_FRONT"].startswith("THIS READ ESTABLISHES DIRECTION"),
       "R-529(A) IS THE FIRST SUBSTANTIVE FIELD -- the ruling says the "
       "statement must be up front, so it is, not in a caveats block")
    cl = d["cluster_disclosure"]["permutation_floor"]
    ok(cl["best_possible_adjusted_p"] == 0.0625
       and cl["with_multiplicity_2"] == 0.0625
       and not cl["clears_0_05"] and cl["smallest_G_that_clears"] == 6,
       f"and its ARITHMETIC IS COMPUTED, not quoted: 2/2^5 = "
       f"{cl['with_multiplicity_2']}, clears 0.05: {cl['clears_0_05']}, "
       f"smallest G that clears: {cl['smallest_G_that_clears']}")
    ok(d["cluster_disclosure"]["G_complete_days"] == 5
       and d["cluster_disclosure"]["unit_actually_used"] == "UTC day"
       and not d["cluster_disclosure"]["weaker_than_ruled"],
       "the ruled cluster unit is finally the unit USED (G=5), where the "
       "interim read had to disclose the window as a weaker substitute")
    ok(sorted(d["cluster_disclosure"]["two_of_the_five_are_re_reads"])
       == ["20260901", "20260902"]
       and d["rule_11_in_force"]["first_opening"]
       == ["20260903", "20260904", "20260905"],
       "TWO OF FIVE ARE RE-READS and three are first openings -- disclosed "
       "so a reader does not count five independent days")
    w = d["what_is_opened"]
    ok(w["OPENED_BY_THE_READ"]["fields"] == ["per_coin_scores", "report"]
       and "seal()" in w["OPENED_BY_THE_READ"]["source_of_this_field_list"],
       "WHAT IS OPENED names the FIELDS, and the field list is read at the "
       "WRITER -- no sealed file was opened to produce this declaration")
    ok(w["STAYS_SEALED"]["not_opened_by_this_read"]
       and "feed" in w["STAYS_SEALED"]["file"],
       "and WHAT STAYS SEALED is named too: the feed, because the estimand "
       "does not need it and opening more than it needs is consumption "
       "without purpose")
    ok(all(r["receipt_present"] for r in w["per_day"]),
       f"all {len(w['per_day'])} race days have a receipt present, each "
       f"pinned by sha256 in this declaration")
    mm = d["members"]
    ok(mm["read_not_typed"] is True
       and mm["race_multiplicity_at_freeze"] == 2
       and len(mm["members"]) == 2
       and any("PM_PLUS_FINE" in x for x in mm["members"])
       and any("PM_FINE_EXTENDED" in x for x in mm["members"]),
       f"m = {mm['race_multiplicity_at_freeze']} and BOTH MEMBERS ARE READ "
       f"from the freeze audit, not typed: {mm['members']}. The first "
       f"version of this reader looked at the top level, got None, and still "
       f"said read_not_typed -- so this check is aimed at that")
    ok(d["opens_nothing"] and d["selects_nothing"] and d["adjudicates"] is None,
       "the declaration OPENS NOTHING and ADJUDICATES NOTHING -- filing is "
       "not reading (rule 14)")
    inh = d["inherited_from_the_interim_declaration"]
    import be_read_declaration as RD
    ok(inh["frozen_decisions"] == RD.frozen_decisions()
       and inh["scoring_stack"] == RD.scoring_stack(),
       "and the shared blocks are the INTERIM INSTRUMENT'S OWN OUTPUT, "
       "called not restated -- two declarations that could drift are one "
       "declaration that cannot")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        out = build()
        dst = (HERE / "declarations" / "be_race_read_declaration_v1.json")
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst), "G": out["G"],
                          "opens_nothing": out["opens_nothing"]}))
        return 0
    print("usage: be_race_read_declaration.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
