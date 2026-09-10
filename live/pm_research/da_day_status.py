#!/usr/bin/env python3
"""DAY STATUS AS FIELDS, PER ENDPOINT -- and the guard that stops the short
form being quoted.

Plan v2 step 1, DA 195. The coordinator relayed DA 194 to the user as
"09-07 is clean". The plan is more precise: no fitted component or threshold
ever saw it, BUT its 5-second diagnostic was computed with 1,000 null draws
while its settlement endpoint was deliberately not computed. So it is
ENDPOINT-SPECIFICALLY clean.

"clean" and "clean for one endpoint" are different claims and the shorter one
is the one that gets quoted, so this module makes the short one UNAVAILABLE:
the record carries no broad cleanliness field, and `require_endpoint()`
REFUSES any claim about the day that does not name an endpoint.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECL_0907 = HERE / "declarations" / "p003_day_status_20260907_v1.json"

NO_ENDPOINT = "DAY_STATUS_CLAIMED_WITHOUT_AN_ENDPOINT"
BROAD_CLAIM = "BROAD_CLEANLINESS_CLAIMED"

#: Words that assert cleanliness. A claim using one of these about a day must
#: also name an endpoint -- the property is the PAIRING, not the vocabulary.
_CLEAN = re.compile(r"\b(clean|untouched|unconsumed|not consumed|virgin|"
                    r"pristine|unspent)\b", re.I)


def load(path=None) -> dict:
    return json.loads(Path(path or DECL_0907).read_text())


def endpoints(d: dict | None = None) -> dict:
    return (d if d is not None else load()).get("endpoint_status") or {}


def require_endpoint(claim: str, endpoint: str | None,
                     d: dict | None = None) -> dict:
    """A cleanliness claim about this day must NAME a declared endpoint.

    REFUSES rather than reports: a report beside a sentence is what let
    "09-07 is clean" reach the user in the first place."""
    d = d if d is not None else load()
    eps = endpoints(d)
    asserts_clean = bool(_CLEAN.search(str(claim or "")))
    if asserts_clean and not endpoint:
        raise ValueError(
            f"REFUSED {NO_ENDPOINT}: {claim!r} asserts cleanliness about "
            f"{d.get('day')} and names no endpoint. The declared endpoints "
            f"are {sorted(eps)}, and their statuses DIFFER -- "
            f"{ {k: v.get('status') for k, v in eps.items()} }.")
    if endpoint and endpoint not in eps:
        raise ValueError(
            f"REFUSED {NO_ENDPOINT}: {endpoint!r} is not a declared endpoint "
            f"of {d.get('day')}. Declared: {sorted(eps)}.")
    if not asserts_clean:
        return {"status": "NO_CLEANLINESS_CLAIMED", "endpoint": endpoint}
    ep = eps[endpoint]
    return {"status": "ENDPOINT_NAMED", "endpoint": endpoint,
            "endpoint_status": ep.get("status"),
            "usable_as_PRIMARY_validation": ep.get(
                "usable_as_PRIMARY_validation"),
            "the_claim_is_only_about": endpoint}


def verify_record(d: dict | None = None) -> dict:
    """PROPERTIES of the record, driven -- not the presence of its words."""
    d = d if d is not None else load()
    eps = endpoints(d)
    broad = d.get("NO_BROAD_CLEAN_FIELD_EXISTS") or {}
    statuses = {k: v.get("status") for k, v in eps.items()}
    # the whole point: the endpoints must not agree, or the distinction is
    # decorative and a broad claim would be harmless.
    props = {
        "at_least_two_endpoints_declared": len(eps) >= 2,
        "THE_ENDPOINTS_DISAGREE": len(set(statuses.values())) >= 2,
        "no_top_level_field_asserts_broad_cleanliness": not any(
            _CLEAN.search(k) for k in d),
        "the_broad_refusal_is_named":
            broad.get("refusal_name") == NO_ENDPOINT,
        "COMPUTED_exactly_one_endpoint_was_ever_computed": sum(
            1 for v in eps.values() if v.get("ever_computed_for_this_day")) == 1,
        "COMPUTED_the_uncomputed_endpoint_has_zero_draws": all(
            v.get("null_draws_taken") == 0 for v in eps.values()
            if not v.get("ever_computed_for_this_day")),
        "COMPUTED_the_computed_endpoint_has_draws_at_or_above_the_floor": all(
            isinstance(v.get("null_draws_taken"), int)
            and v["null_draws_taken"] > 0 for v in eps.values()
            if v.get("ever_computed_for_this_day")),
        "NO_endpoint_is_usable_as_PRIMARY_validation": all(
            v.get("usable_as_PRIMARY_validation") is False
            for v in eps.values()),
        "the_weaker_guarantee_is_stated": bool(
            d.get("the_limit_DA_194_named_and_will_not_soften")),
        "the_identity_trap_is_recorded_with_its_operation": bool(
            (d.get("the_identity_trap_found_mid_audit") or {}).get(
                "the_operation_that_distinguishes_them")),
        "no_protected_day_was_touched": (
            d.get("no_protected_day_was_touched") or {}).get("value") is True,
    }
    failed = [k for k, v in props.items() if not v]
    return {"verdict": "HOLDS" if not failed else "FAILS",
            "failed_properties": failed, "n_properties_driven": len(props),
            "endpoint_statuses": statuses}


# ------------------------------------------------------------- selftest

_N = {"n": 0, "bad": 0}


def _ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def selftest(quiet: bool = False) -> int:
    import copy
    d = load()
    r = verify_record(d)
    _ok(r["verdict"] == "HOLDS",
        f"the 09-07 record holds on all {r['n_properties_driven']} properties "
        f"(failed: {r['failed_properties']})")
    _ok(r["endpoint_statuses"]["SETTLEMENT_R801"] == "NOT_CONSUMED_FOR_THIS_ENDPOINT"
        and r["endpoint_statuses"]["MARKOUT_5S_D_E0"] == "COMPUTED_AND_NULLED",
        f"the two endpoints carry DIFFERENT statuses: {r['endpoint_statuses']} "
        f"-- which is exactly why an endpoint-free claim is inadmissible")

    # POSITIVE CONTROL: an endpoint-named claim passes and reports only that endpoint.
    got = require_endpoint("09-07 is clean for this purpose", "SETTLEMENT_R801", d)
    _ok(got["status"] == "ENDPOINT_NAMED"
        and got["endpoint_status"] == "NOT_CONSUMED_FOR_THIS_ENDPOINT"
        and got["usable_as_PRIMARY_validation"] is False,
        "POSITIVE CONTROL: a claim naming SETTLEMENT_R801 passes and carries "
        "usable_as_PRIMARY_validation False with it")

    # KNOWN-BAD 1: the exact sentence that reached the user.
    for claim in ("09-07 is clean",
                  "09-07 is untouched",
                  "2026-09-07 was not consumed",
                  "we have a pristine day in hand"):
        try:
            require_endpoint(claim, None, d)
            fired = False
        except ValueError as e:
            fired = NO_ENDPOINT in str(e)
        _ok(fired, f"KNOWN-BAD: {claim!r} with no endpoint REFUSES {NO_ENDPOINT}")

    # KNOWN-BAD 2: an endpoint that was never declared.
    try:
        require_endpoint("09-07 is clean", "SOME_OTHER_ENDPOINT", d)
        fired = False
    except ValueError as e:
        fired = NO_ENDPOINT in str(e)
    _ok(fired, "KNOWN-BAD: an undeclared endpoint REFUSES rather than passing")

    # NOT A CLAIM: a sentence asserting nothing about cleanliness is not blocked.
    _ok(require_endpoint("09-07 has 288 windows", None, d)["status"]
        == "NO_CLEANLINESS_CLAIMED",
        "the guard fires on CLEANLINESS CLAIMS, not on every mention of the day")

    # KNOWN-BAD 3: the record itself weakened.
    bad = copy.deepcopy(d)
    bad["endpoint_status"]["MARKOUT_5S_D_E0"]["status"] = "NOT_CONSUMED_FOR_THIS_ENDPOINT"
    _ok(verify_record(bad)["verdict"] == "FAILS",
        "KNOWN-BAD: if both endpoints were given the SAME status the record "
        "FAILS -- a distinction that does not distinguish is decoration")
    bad2 = copy.deepcopy(d)
    bad2["clean"] = True
    _ok(verify_record(bad2)["verdict"] == "FAILS",
        "KNOWN-BAD: adding a top-level `clean` field FAILS -- the short claim "
        "must stay unavailable, because it is the one that gets quoted")
    bad3 = copy.deepcopy(d)
    bad3["endpoint_status"]["SETTLEMENT_R801"]["usable_as_PRIMARY_validation"] = True
    _ok(verify_record(bad3)["verdict"] == "FAILS",
        "KNOWN-BAD: promoting either endpoint to PRIMARY validation FAILS -- "
        "the plan forbids it and the record must too")

    if not quiet:
        print(f"[da_day_status] {_N['n'] - _N['bad']}/{_N['n']} checks, "
              f"{_N['bad']} failures | 09-07 endpoints: {r['endpoint_statuses']}")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(verify_record(), indent=1))
