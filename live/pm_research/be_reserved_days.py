"""A RESERVED DAY IS PROTECTED BY ITS NAME, NOT BY AN ABSENCE (BE 160/161).

WHAT WAS WRONG. 09-07's reservation rested on
`settlement_endpoint.admissible_days` being **None** in params v29. That
held -- `be_0907_reservation_finding_v1.json` shows a real day run on
2026-09-08 recorded NOT_VALUED_DAY_NOT_ADMISSIBLE on both arms -- but it
held for a reason nobody can read. An absence cannot be reviewed in a diff,
cannot be cited, and cannot be argued with. Worse, it is SILENTLY
REVERSIBLE: the moment anyone sets `admissible_days` for an unrelated
reason -- naming 09-03..09-06 explicitly is the obvious one -- 09-07 opens
as a SIDE EFFECT, and no line of that diff mentions 09-07.

THE SHAPE HERE. A reserved day is NAMED in `settlement_endpoint.
reserved_days`, and admissibility requires BOTH tests: the day is named
admissible AND it is not named reserved. Opening a reserved day then
requires REMOVING ITS NAME -- a deliberate act, visible in a diff, arguable
in review. That is rule 35's shape: the limit stops living in the absence of
a value and becomes a field the guard tests as a PROPERTY.

ABSENCE IS LOUD, NOT PERMISSIVE. If `reserved_days` is missing this REFUSES
by name rather than treating "no list" as "nothing is reserved". A default
that answers for something never given is the rule-28 defect that this
module exists to remove; reintroducing it here would be the same bug with a
new spelling.

IT DECIDES NOTHING (rule 14). `admissibility` COMPUTES a verdict and never
raises, so a caller that must RECORD the verdict can. `assert_not_reserved`
is the entry that REFUSES, for a caller asking to VALUE a day. Both exist
because the programme already has this pair one layer down and learned the
hard way that only one of them was on the path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: The field that names days held back from an endpoint.
RESERVED_FIELD = "reserved_days"
#: The field that names days a ruled endpoint may be computed for.
ADMISSIBLE_FIELD = "admissible_days"
ENDPOINT = "settlement_endpoint"

RESERVED = "DAY_IS_RESERVED_BY_NAME"
NO_RESERVED_FIELD = "RESERVED_DAYS_FIELD_ABSENT_NO_VERDICT_POSSIBLE"
NOT_ADMISSIBLE = "SETTLEMENT_DAY_NOT_ADMISSIBLE"


class ReservationRefused(RuntimeError):
    """The day may not be valued, or the question cannot be answered."""



#: WHAT DE MUST CHANGE FOR THIS TO BE LOAD-BEARING (BE 161).
#: A params field nothing consults is the rule-28 defect this module exists
#: to remove, so the field and the call site land together or neither does.
DE_CALL_SITE = {
    "module": "de_multiday_gate1_runner.py",
    "function": "settlement_admissibility",
    "change": ("after computing its own verdict, apply the reservation test: "
               "a day NAMED in `settlement_endpoint.reserved_days` is NOT "
               "admissible whatever `admissible_days` says"),
    "suggested": ("import be_reserved_days as RD; "
                  "then `admissible = admissible and not "
                  "RD.admissibility(day, params)['is_reserved']`, and carry "
                  "`reserved_days` and `is_reserved` onto the verdict dict "
                  "so they reach the artifact as required fields (rule 35)"),
    "and_the_raising_twin": (
        "`assert_settlement_day_admissible` is currently called ONLY from "
        "selftest (be_offpath_guards_v1.json). A caller reaching the "
        "endpoint through `run_day` is gated by the COMPUTING twin at "
        "9131/9133; a caller invoking the estimator DIRECTLY meets nothing. "
        "Putting the raising twin on the direct path is what closes that."),
    "params": ("de_multiday_gate1_params_v30.json adds "
               "`settlement_endpoint.reserved_days: ['2026-09-07']`; DE must "
               "pair a design version, as any params bump does"),
    "falsifier_that_must_keep_passing": (
        "be_reserved_days --selftest, check 3: naming 09-03..09-06 "
        "admissible must NOT open 09-07"),
}


def _endpoint(params: dict) -> dict:
    return (params or {}).get(ENDPOINT) or {}


def reserved_days(params: dict) -> tuple:
    """The NAMED reserved days. An absent field REFUSES."""
    ep = _endpoint(params)
    if RESERVED_FIELD not in ep:
        raise ReservationRefused(
            f"REFUSED {NO_RESERVED_FIELD}: `{ENDPOINT}.{RESERVED_FIELD}` is "
            f"not present in the parameter file. An absent list is not an "
            f"empty one -- treating it as 'nothing is reserved' is exactly "
            f"the protection-by-absence this field replaces.")
    got = ep[RESERVED_FIELD]
    if not isinstance(got, (list, tuple)):
        raise ReservationRefused(
            f"REFUSED {NO_RESERVED_FIELD}: `{RESERVED_FIELD}` is "
            f"{type(got).__name__}, not a list of day strings.")
    return tuple(str(d) for d in got)


def admissibility(day: str, params: dict) -> dict:
    """COMPUTE the verdict. Never raises on the verdict itself (rule 14).

    Both tests must pass: NAMED admissible AND NOT NAMED reserved. The
    second is what stops a day opening as a side effect of the first."""
    ep = _endpoint(params)
    res = reserved_days(params)          # refuses if the field is absent
    adm = ep.get(ADMISSIBLE_FIELD)
    named_admissible = bool(adm) and str(day) in {str(d) for d in adm}
    is_reserved = str(day) in set(res)
    admissible = named_admissible and not is_reserved
    refusal = None
    if is_reserved:
        refusal = RESERVED
    elif not named_admissible:
        refusal = NOT_ADMISSIBLE
    return {
        "day": str(day),
        "admissible": admissible,
        "named_admissible": named_admissible,
        "is_reserved": is_reserved,
        "refusal_name": refusal,
        "reserved_days": list(res),
        "admissible_days": list(adm) if adm else adm,
        "why": ("named admissible and not reserved" if admissible else
                (f"{day} is RESERVED by name; opening it requires removing "
                 f"it from `{RESERVED_FIELD}`, which is a deliberate act "
                 f"visible in a diff" if is_reserved else
                 f"{day} is not named in `{ADMISSIBLE_FIELD}`")),
        "BOTH_TESTS_APPLIED": True,
    }


def assert_not_reserved(day: str, params: dict) -> dict:
    """The entry that REFUSES, for a caller asking to VALUE a day."""
    v = admissibility(day, params)
    if not v["admissible"]:
        raise ReservationRefused(f"REFUSED {v['refusal_name']}: {v['why']}")
    return v


def falsify() -> int:
    checks = []

    def note(name, ok):
        checks.append((name, bool(ok)))

    def refuses(fn, token):
        try:
            fn()
            return False
        except ReservationRefused as e:
            return token in str(e)

    DAYS4 = ["2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"]
    P = {ENDPOINT: {ADMISSIBLE_FIELD: None, RESERVED_FIELD: ["2026-09-07"]}}

    # 0. NEGATIVE CONTROL -- a day that IS named and NOT reserved opens.
    note("a named, unreserved day is admissible",
         admissibility("2026-09-03", {ENDPOINT: {
             ADMISSIBLE_FIELD: DAYS4,
             RESERVED_FIELD: ["2026-09-07"]}})["admissible"] is True)

    # 1. POSITIVE CONTROL -- the reserved day refuses BY NAME.
    note("the reserved day refuses by name",
         refuses(lambda: assert_not_reserved("2026-09-07", P), RESERVED))

    # 2. **THE ONE THE COORDINATOR ASKED FOR.** Setting `admissible_days`
    #    for an UNRELATED reason must NOT open the reserved day.
    P2 = {ENDPOINT: {ADMISSIBLE_FIELD: DAYS4, RESERVED_FIELD: ["2026-09-07"]}}
    note("naming 09-03..09-06 admissible does NOT open 09-07",
         admissibility("2026-09-07", P2)["admissible"] is False
         and refuses(lambda: assert_not_reserved("2026-09-07", P2), RESERVED))

    # 3. Even naming the reserved day admissible does not open it -- the
    #    reservation must be LIFTED, not out-voted.
    P3 = {ENDPOINT: {ADMISSIBLE_FIELD: DAYS4 + ["2026-09-07"],
                     RESERVED_FIELD: ["2026-09-07"]}}
    note("naming it admissible while still reserved does NOT open it",
         admissibility("2026-09-07", P3)["admissible"] is False
         and admissibility("2026-09-07", P3)["refusal_name"] == RESERVED)

    # 4. It CAN be opened deliberately: remove the name, then name it.
    P4 = {ENDPOINT: {ADMISSIBLE_FIELD: DAYS4 + ["2026-09-07"],
                     RESERVED_FIELD: []}}
    note("removing the name AND naming it admissible opens it",
         admissibility("2026-09-07", P4)["admissible"] is True)

    # 5. KNOWN-BAD -- an ABSENT reserved list must REFUSE, never default.
    note("an absent reserved_days REFUSES rather than defaulting to empty",
         refuses(lambda: admissibility("2026-09-07",
                                       {ENDPOINT: {ADMISSIBLE_FIELD: DAYS4}}),
                 NO_RESERVED_FIELD))

    # 6. KNOWN-BAD -- a non-list reserved field refuses.
    note("a non-list reserved_days REFUSES",
         refuses(lambda: admissibility("2026-09-07",
                                       {ENDPOINT: {ADMISSIBLE_FIELD: DAYS4,
                                                   RESERVED_FIELD: "2026-09-07"}}),
                 NO_RESERVED_FIELD))

    # 7. The unrelated-day case still refuses for the RIGHT reason.
    note("an unnamed, unreserved day refuses as NOT_ADMISSIBLE",
         admissibility("2026-09-02", P2)["refusal_name"] == NOT_ADMISSIBLE)

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    bad = [n for n, ok in checks if not ok]
    print(json.dumps({"falsifier": "be_reserved_days", "n": len(checks),
                      "n_failed": len(bad), "failed": bad}))
    return 1 if bad else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return falsify()
    if len(argv) >= 2:
        params = json.loads(Path(argv[1]).read_text())
        print(json.dumps(admissibility(argv[0], params), indent=1))
        return 0
    print("usage: be_reserved_days.py <day> <params.json> | --selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
