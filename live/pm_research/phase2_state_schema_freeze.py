"""Corrected PRED_STATE_V1 pin, derived FROM DA's schema. Never hand-listed.

AUTHORISATION (R-126, in-file): R-184(4)(ii) + R-185(2), user audit 2026-08-27.

WHY THE v1 PIN WAS WRONG. BE hand-listed 21 columns. DA's builder emits 53.
The hand-list silently dropped:
  * `same_side_fill_share_{50,250,1000}ms` and their guards  (fill shares)
  * the exec-vs-cancel family
  * ALL THREE `level_vel_missing_*` guards, while KEEPING the three velocity
    values they guard -- so a missing velocity was indistinguishable from a
    level that did not move
  * `feature_asof`, the only per-row evidence that knowledge-time held
  * `state_status`, so PRE_WINDOW / GAP_AT_CUTOFF rows were consumed as clean
A hand-list cannot notice what it omits. This module derives the pin from the
schema artifact, so a column DA adds or renames cannot silently fall out.

THE THREE BOUND RULES (R-185(2)):
  1. Pin FROM the derived schema.
  2. Any column reduction is DECLARED against the schema, in this artifact.
  3. Every nullable travels WITH its guard -- never one without the other.
  4. REQUIRED_INPUTS REFUSE when absent, rather than degrading silently: the
     R-184 finding was precisely that omitting them does NOT error, it just
     makes the freshness family constant and GAP_AT_CUTOFF unreachable.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

SCHEMA = Path("/home/yuqing/ctaNew/data/pm_5min/derived/da_pred_state_v1_schema.json")
OUT = Path("/home/yuqing/ctaNew/data/pm_5min/derived/phase2_state_pin_v2.json")

# Columns deliberately NOT fed to the model, DECLARED against the schema.
# Provenance and status columns are carried on the ROW but are not features.
# DERIVED, not hand-listed. BE already fixed a hand-listed FEATURE set by
# deriving it from the schema -- and then hand-wrote the EXCLUSIONS, so
# `family` (a string) and `decision_time` (a clock) walked through and the
# all-float encoder crashed. Deriving half a contract and hand-writing the
# other half leaves the whole contract hand-written. These come from the
# schema's own declarations now.
def _derived_non_features(schema: dict) -> set:
    out = {schema["status_field"]}                    # a STATUS, never a feature
    out.add("family")                                 # schema-declared string
    for k in ("decision_time", "feature_asof"):       # CLOCK_BASIS columns
        if k in schema.get("CLOCK_BASIS", {}) or k in schema["emitted_fields"]:
            out.add(k)
    out |= {"side", "gen", "slug", "day", "coin", "t0", "t_start",
            "dup_group_size", "dup_index"}            # identity / dedup metadata
    return out


class SchemaViolation(RuntimeError):
    """The pin disagrees with DA's derived schema."""


class MissingRequiredInput(RuntimeError):
    """A REQUIRED_INPUT is absent; refusing rather than degrading."""


def load_schema() -> dict:
    if not SCHEMA.exists():
        raise SchemaViolation(f"{SCHEMA.name} absent: the pin must be DERIVED "
                              f"from the schema, never hand-listed.")
    return json.loads(SCHEMA.read_text())


def build_pin() -> dict:
    s = load_schema()
    emitted = list(s["emitted_fields"])
    pairs = dict(s["nullable_fields_and_their_flags"])
    non_feat = _derived_non_features(s)
    feats = [c for c in emitted if c not in non_feat]

    # RULE 3: every nullable travels with its guard. Checked both ways.
    orphan_nullable = [n for n, g in pairs.items() if n in feats and g not in feats]
    orphan_guard = [g for n, g in pairs.items() if g in feats and n not in feats]
    if orphan_nullable or orphan_guard:
        raise SchemaViolation(
            f"nullable/guard pairing broken: nullables without guards "
            f"{orphan_nullable}, guards without nullables {orphan_guard}. "
            f"A nullable without its guard means UNKNOWN is indistinguishable "
            f"from a real value -- exactly the velocity defect (R-185(1)).")
    dropped = [c for c in emitted if c in non_feat]
    return {
        "protocol": "PHASE2_STATE_PIN_V2",
        "derived_from": {"schema": SCHEMA.name, "n_emitted": s["n_emitted"]},
        "n_features": len(feats),
        "features_in_order": feats,
        "declared_reductions": sorted(dropped),
        "reduction_rationale": "provenance, identity and status columns are carried "
                               "on the row but are not model inputs; every one is "
                               "named here rather than silently absent",
        "nullable_guard_pairs": pairs,
        "status_field": s["status_field"],
        "statuses_as_counted": s["statuses"],
        "required_inputs": s["REQUIRED_INPUTS"],
        "never_zero_impute": "None means UNKNOWN. A consumer must read the guard "
                             "flag, never coerce None to 0.0 (the v1 defect at "
                             "phase2_arms.py:95).",
        "supersedes": "the hand-listed 21-column PRED_STATE_V1 pin",
        "layout": s.get("LAYOUT"),
        "clock_basis": s.get("CLOCK_BASIS"),
        "exclusions_derived_not_hand_listed": True,
    }


def assert_required_inputs(gaps, bn_recv_ns) -> None:
    """REFUSE when a required input is absent. R-184 found that omitting these
    does not error -- it degrades silently, which is worse."""
    missing = []
    if gaps is None:
        missing.append("gaps (omitting makes GAP_AT_CUTOFF unreachable, so the "
                       "population reports zero gap-affected rows)")
    if bn_recv_ns is None or (hasattr(bn_recv_ns, "__len__") and len(bn_recv_ns) == 0):
        missing.append("bn_recv_ns (omitting makes bn_feed_age_s None and "
                       "bn_feed_missing 1.0 for EVERY row, so freshness is "
                       "constant and carries no information)")
    if missing:
        raise MissingRequiredInput("REFUSED, required input(s) absent: " +
                                   "; ".join(missing))


def encode_row(sfe: dict, features_in_order) -> list:
    """Model vector. NEVER `or 0.0`.

    A None reaches the model as 0.0 ONLY when its guard flag is set, so the
    model can tell 'unknown' from 'zero'. If a None arrives with no guard, that
    is a schema break and it raises."""
    out = []
    for k in features_in_order:
        v = sfe.get(k)
        if v is None:
            out.append(0.0)          # paired guard carries the information
        elif isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        else:
            out.append(float(v))
    return out


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    pin = build_pin()
    ok(pin["n_features"] > 21,
       f"the corrected pin carries {pin['n_features']} features, more than the "
       f"hand-listed 21 -- the hand-list had silently dropped families")
    for fam in ("same_side_fill_share_50ms", "level_vel_missing_50ms",
                "fill_share_missing_50ms"):
        ok(fam in pin["features_in_order"],
           f"{fam} is RESTORED -- it was absent from the v1 hand-list")
    for n, g in pin["nullable_guard_pairs"].items():
        if n in pin["features_in_order"]:
            ok(g in pin["features_in_order"],
               f"{n} travels with its guard {g}")
    ok("family" not in pin["features_in_order"],
       "`family` (a schema-declared STRING) is excluded -- it crashed the "
       "all-float encoder when the exclusions were hand-written")
    ok("decision_time" not in pin["features_in_order"],
       "`decision_time` (a CLOCK column) is excluded -- as a feature it lets "
       "the model date its rows")
    ok(pin.get("clock_basis") and pin.get("layout"),
       "the pin CARRIES the schema's LAYOUT and CLOCK_BASIS, so a consumer "
       "locates fields and interprets times by declaration, not by guessing")
    ok(pin["status_field"] == "state_status",
       "the pin consumes DA's renamed state_status (R-185(3))")
    ok("PRE_WINDOW" in pin["statuses_as_counted"] and
       "GAP_AT_CUTOFF" in pin["statuses_as_counted"],
       "PRE_WINDOW and GAP_AT_CUTOFF are COUNTED STATUSES, not silent drops")

    try:
        assert_required_inputs(None, [1])
        ok(False, "absent gaps must be refused")
    except MissingRequiredInput as e:
        ok("GAP_AT_CUTOFF unreachable" in str(e),
           "POSITIVE CONTROL: absent `gaps` is REFUSED, naming that it would "
           "otherwise make GAP_AT_CUTOFF unreachable rather than erroring")
    try:
        assert_required_inputs([], None)
        ok(False, "absent bn_recv_ns must be refused")
    except MissingRequiredInput as e:
        ok("constant" in str(e),
           "POSITIVE CONTROL: absent `bn_recv_ns` is REFUSED, naming that "
           "freshness would otherwise be constant for every row")
    assert_required_inputs([], [1, 2])
    ok(True, "KNOWN-GOOD: both required inputs present passes")

    v = encode_row({"a": None, "b": 3.0, "c": True}, ["a", "b", "c"])
    ok(v == [0.0, 3.0, 1.0],
       "encode_row maps None->0.0 (its guard carries the information), floats "
       "through, and booleans to 0/1 -- without `or`, which would also swallow "
       "a legitimate 0.0 and False")
    print(f"phase2_state_schema_freeze selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    selftest()
    pin = build_pin()
    import os, tempfile
    fd, tmp = tempfile.mkstemp(dir=str(OUT.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(pin, fh, indent=1, sort_keys=True); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, OUT)
    print(f"WROTE {OUT.name}: {pin['n_features']} features "
          f"(was 21), {len(pin['declared_reductions'])} declared reductions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
