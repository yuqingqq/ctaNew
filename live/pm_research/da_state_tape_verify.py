"""Gate BE's rebuilt state tape against DA's DECLARED PRED_STATE_V1 schema.

SURFACE AUTHORISATION (R-126, in-file): R-185 accepts DA's review offer AS A
GATE -- the corrected Phase-2 freeze waits on this check.

PRE-REGISTERED: written, and its falsifiers RUN, BEFORE BE's rebuilt tape
exists. The population verifier earned this pattern (Q-DA-77): a checker
authored after seeing the artifact can always be shaped to it, however
honestly. This one cannot be.

EXPECTATIONS COME FROM THE SCHEMA ARTIFACT, NOT FROM THIS FILE. It reads
`da_pred_state_v1_schema.json`, which is itself emitted by running the builder.
So the chain is: builder -> schema -> gate. Nothing in the middle is
transcribed, and a change to the family propagates to this gate without anyone
remembering to update it.

WHAT IT IS LOOKING FOR -- the five ways the FIRST tape went wrong (R-184/R-185),
each of which passed silently at the time:
  1. REQUIRED_INPUTS omitted. `build_tape` without `gaps` makes GAP_AT_CUTOFF
     unreachable; without `bn_recv_ns` the freshness family is CONSTANT.
     Neither errors, so the tape looked fine.
  2. A guardless pin. `level_size_vel_*` were pinned WITHOUT their
     `level_vel_missing_*` flags.
  3. Zero-imputation. `float(x or 0.0)` maps None -> 0.0, so a missing velocity
     is indistinguishable from a genuinely flat book -- the exact distinction
     the family exists to preserve.
  4. Undeclared reduction. 21 of 53 fields, discovered later as "the tested
     family was not the planned family".
  5. state_status ignored -- 27,552 PRE_WINDOW rows consumed as if clean.

IT REFUSES RATHER THAN PASSES when it cannot tell. An unreadable tape, an
unrecognised layout, or a missing schema is a REFUSAL, never a green check.

    python3 live/pm_research/da_state_tape_verify.py --selftest
    python3 live/pm_research/da_state_tape_verify.py verify --tape PATH
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
DERIVED = REPO / "data/pm_5min/derived"
SCHEMA = DERIVED / "da_pred_state_v1_schema.json"
EMBARGO_S = 60.0          # the manifest's declared split_embargo_s


class GateRefused(RuntimeError):
    """The gate could not evaluate. NOT a pass."""


def load_schema(path: Path = SCHEMA) -> dict[str, Any]:
    if not path.exists():
        raise GateRefused(
            f"{path} absent. The gate's expectations come from the builder's "
            f"own schema; without it there is nothing to check against, and a "
            f"check against nothing is not a pass. Regenerate with "
            f"`harmful_state_features.py --schema`.")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_tape(path: Path) -> Iterator[dict[str, Any]]:
    """Accept JSONL or a JSON object with a rows list. REFUSE anything else."""
    if not path.exists():
        raise GateRefused(f"{path} does not exist -- an absent tape is not a "
                          f"passing tape.")
    head = path.open("rb").read(4096).lstrip()
    if head.startswith(b"{") and b'"rows"' in head[:2048]:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj.get("rows")
        if not isinstance(rows, list):
            raise GateRefused("object has no `rows` list")
        yield from rows
        return
    if head.startswith(b"{"):
        n = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                n += 1
        if n == 0:
            raise GateRefused("no rows parsed from a JSONL-looking file")
        return
    raise GateRefused(
        f"unrecognised tape layout (first bytes {head[:40]!r}). REFUSING: a "
        f"gate that guesses a layout can silently check the wrong thing.")


def locate_features(rows: list[dict[str, Any]], schema: dict[str, Any]
                    ) -> tuple[str | None, list[dict[str, Any]]]:
    """Find the feature dicts by DECLARATION, never by guessing (R-187).

    The schema declares the builder's NATIVE layout (flat). A consumer that
    wraps the fields must say so on the tape as `features_under`. If the fields
    are in neither place this REFUSES -- because the alternative is what
    happened before: a checker searching top level while the builder nested
    under `state`, whereupon predicates that iterate over "present" fields
    found none, iterated over nothing, and PASSED. Three of them did.
    """
    # Locate on FEATURES, never on identity. `slug` is an emitted field, so a
    # tape carrying only slug+t0 scored a "hit" and the gate declined to
    # refuse -- one identity field masking a wholly absent feature set.
    declared = set(schema["emitted_fields"]) - set(
        schema.get("identity_fields", []))
    probe = rows[:200]
    flat_hits = len(declared & set().union(*(set(r) for r in probe)))
    under = None
    for r in probe:
        if isinstance(r.get("features_under"), str):
            under = r["features_under"]
            break
    if under is None:
        for cand in ("state", "features", "pred_state"):
            if any(isinstance(r.get(cand), dict) for r in probe):
                under = cand
                break
    nest_hits = 0
    if under:
        nest_hits = len(declared & set().union(
            *(set(r.get(under) or {}) for r in probe)))
    if flat_hits == 0 and nest_hits == 0:
        raise GateRefused(
            f"cannot locate ANY of the {len(declared)} declared FEATURE fields "
            f"(identity fields excluded on purpose), flat "
            f"or under {under!r}. REFUSING rather than checking an empty "
            f"intersection -- an unreadable layout must not yield passes.")
    if nest_hits > flat_hits:
        return under, [dict(r.get(under) or {}, **{k: v for k, v in r.items()
                                                   if k != under}) for r in rows]
    return None, rows


def clock_of(r: dict[str, Any], schema: dict[str, Any]) -> float | None:
    """Absolute decision epoch, honouring the DECLARED clock basis.

    Adding t0 to an already-absolute decision_time double-counts the window
    start. The basis is read from the tape when it declares one, else from the
    schema; it is never assumed.
    """
    basis = r.get("clock_basis") or schema["CLOCK_BASIS"]["decision_time"]
    for k in ("decision_epoch", "abs_decision_time"):
        if r.get(k) is not None:
            return float(r[k])
    dt = r.get("decision_time")
    if dt is None:
        return None
    if str(basis).startswith("absolute"):
        return float(dt)
    if r.get("t0") is None:
        return None
    return float(r["t0"]) + float(dt)


def gate(rows: list[dict[str, Any]], schema: dict[str, Any],
         gapped_slugs_expected: int | None = None) -> list[dict[str, Any]]:
    """Every predicate COMPUTED (rule 10); each names what it would miss."""
    out: list[dict[str, Any]] = []

    def p(name, ok, detail):
        out.append({"predicate": name, "pass": bool(ok), "detail": detail})

    if not rows:
        p("tape_non_empty", False, "0 rows -- an empty tape never passes")
        return out
    p("tape_non_empty", True, f"{len(rows)} rows")

    under, rows = locate_features(rows, schema)   # REFUSES if unlocatable
    p("layout_matches_declaration", True,
      f"features located {'under ' + repr(under) if under else 'at top level'}"
      f"; schema declares native layout "
      f"{schema.get('LAYOUT', {}).get('native', 'flat')!r}")

    present = set().union(*(set(r) for r in rows[:200]))
    declared = set(schema["emitted_fields"])
    reductions = set()
    for r in rows[:1]:
        reductions = set(r.get("declared_reductions", []) or [])
    missing = sorted(declared - present - reductions)
    p("no_undeclared_reduction", not missing,
      f"{len(present & declared)} of {len(declared)} schema fields present; "
      f"{len(reductions)} declared reductions; "
      + (f"UNDECLARED ABSENT: {missing[:8]}" if missing
         else "every absence is declared"))

    # --- guard beside nullable -------------------------------------------
    pairs = schema["nullable_fields_and_their_flags"]
    checked_pairs = [n for n in pairs if n in present]
    orphans = [n for n, f in pairs.items() if n in present and f not in present]
    p("guard_beside_every_nullable", bool(checked_pairs) and not orphans,
      f"{len(checked_pairs)} of {len(pairs)} declared nullables present; "
      f"orphaned (present without their flag): {orphans or 'none'}"
      + ("  <-- NONE of the declared nullables are present: nothing was "
         "checked, so this CANNOT pass" if not checked_pairs else "")
      + ("  <-- this is the R-185 guardless-pin defect" if orphans else ""))

    # --- zero-imputation, tested EXACTLY rather than by signature ---------
    # The first version flagged any column that was 100% finite with an
    # all-zero guard. The seam test killed it: `gen_age_s`, `pm_feed_age_s` and
    # others are LEGITIMATELY always computable, so that pattern is normal and
    # the check would have fired on every real tape. Hand-made fixtures hid
    # this because I only ever invented rows where things were missing.
    #
    # The exact discriminator: a guard flag that says MISSING beside a value
    # that is PRESENT. `float(x or 0.0)` produces precisely that -- the flag
    # still reports the miss while the value has become 0.0. No population
    # minimum is needed, because this is evidence rather than a pattern.
    imputed = {}
    for n, f in pairs.items():
        if n not in present or f not in present:
            continue
        bad = sum(1 for r in rows
                  if float(r.get(f) or 0.0) != 0.0 and r.get(n) is not None)
        if bad:
            imputed[n] = bad
    p("no_zero_imputation", bool(checked_pairs) and not imputed,
      (f"{len(checked_pairs)} nullable/guard pairs checked; "
       f"rows whose guard says MISSING while the value is PRESENT: "
       f"{imputed or 'none'}")
      + ("  <-- a missing value was imputed; float(x or 0.0) leaves exactly "
         "this" if imputed else "")
      + ("  <-- NONE of the declared nullables are present: nothing was "
         "checked, so this CANNOT pass" if not checked_pairs else ""))

    # --- REQUIRED_INPUTS actually supplied --------------------------------
    bn = [r.get("bn_feed_age_s") for r in rows]
    bn_distinct = len({x for x in bn if x is not None})
    p("bn_recv_ns_was_supplied", bn_distinct > 1,
      f"bn_feed_age_s distinct non-null values: {bn_distinct}"
      + ("  <-- CONSTANT: build_tape was called without bn_recv_ns, so the "
         "whole freshness family carries no information" if bn_distinct <= 1
         else ""))

    statuses = collections.Counter(
        str(r.get(schema["status_field"], "__ABSENT__")) for r in rows)
    p("state_status_present", "__ABSENT__" not in statuses,
      f"{schema['status_field']} counts: {dict(statuses)}")
    gap_seen = statuses.get("GAP_AT_CUTOFF", 0)
    if gapped_slugs_expected is None:
        p("gaps_were_supplied", gap_seen > 0,
          f"GAP_AT_CUTOFF rows: {gap_seen}"
          + ("  <-- ZERO: with gapped slugs in the population this means "
             "build_tape was called without `gaps`" if gap_seen == 0 else ""))
    else:
        okg = gap_seen > 0 or gapped_slugs_expected == 0
        p("gaps_were_supplied", okg,
          f"GAP_AT_CUTOFF rows: {gap_seen}; population carries "
          f"{gapped_slugs_expected} gapped slugs per DA's receipt")

    # --- statuses honoured, not silently consumed -------------------------
    bad_status = sum(v for k, v in statuses.items()
                     if k not in set(schema["statuses"]) | {"__ABSENT__"})
    have_status = "__ABSENT__" not in statuses
    p("statuses_are_declared_values", have_status and bad_status == 0,
      f"rows carrying an undeclared status: {bad_status}")

    # --- knowledge time ---------------------------------------------------
    viol = 0
    checked = 0
    for r in rows:
        a, t = r.get("feature_asof"), r.get("decision_time")
        if a is None or t is None:
            continue
        checked += 1
        if a > t + 1e-9:
            viol += 1
    p("feature_asof_never_after_decision", checked > 0 and viol == 0,
      f"{checked} rows carried both fields; {viol} violations"
      + ("  <-- feature_asof/decision_time NOT CARRIED: the knowledge-time "
         "invariant cannot be checked on this tape" if checked == 0 else ""))

    # --- embargo ----------------------------------------------------------
    tr = [r for r in rows if str(r.get("split", "")).lower().startswith("train")]
    sc = [r for r in rows if str(r.get("split", "")).lower().startswith(("score", "test", "hold"))]
    if tr and sc:
        trc = [c for c in (clock_of(r, schema) for r in tr) if c is not None]
        scc = [c for c in (clock_of(r, schema) for r in sc) if c is not None]
        if trc and scc:
            margin = min(scc) - max(trc)
            p("embargo_respected", margin >= EMBARGO_S,
              f"score starts {margin:.3f}s after training ends; declared "
              f"embargo {EMBARGO_S:.0f}s"
              + ("  <-- CONTAMINATED, the R-184 blocker" if margin < EMBARGO_S
                 else ""))
        else:
            p("embargo_respected", False,
              "split labels present but no usable decision clock -- REFUSING "
              "to certify an embargo it cannot measure")
    else:
        p("embargo_respected", False,
          "no train/score split labels on the tape -- the embargo cannot be "
          "checked here and is NOT certified by this gate")
    return out


def verify(tape: Path, schema_path: Path = SCHEMA,
           gapped_slugs_expected: int | None = None) -> dict[str, Any]:
    schema = load_schema(schema_path)
    rows = list(iter_tape(tape))
    preds = gate(rows, schema, gapped_slugs_expected)
    return {"gate": "da_state_tape_verify_v1", "tape": str(tape),
            "schema_family": schema["family"], "n_rows": len(rows),
            "predicates": preds,
            "all_pass": all(x["pass"] for x in preds)}


def _selftests() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    # A MINIATURE schema for the unit cases -- but it now carries the LAYOUT
    # and CLOCK_BASIS pins, because a test schema that omits them is how the
    # gate came to assume both (R-187).
    schema = {
        "family": "PRED_STATE_V1",
        "emitted_fields": ["a", "a_missing", "bn_feed_age_s", "state_status",
                           "feature_asof", "decision_time"],
        "statuses": ["OK", "PRE_WINDOW", "GAP_AT_CUTOFF"],
        "status_field": "state_status",
        "nullable_fields_and_their_flags": {"a": "a_missing"},
        "LAYOUT": {"native": "flat", "features_under": None},
        "CLOCK_BASIS": {"decision_time": "window_relative_seconds"},
    }

    def R(**kw):
        base = {"a": 1.0, "a_missing": 0.0, "bn_feed_age_s": 0.1,
                "state_status": "OK", "feature_asof": 1.0,
                "decision_time": 2.0}
        base.update(kw)
        return base

    def res(rows, **kw):
        return {p["predicate"]: p["pass"] for p in gate(rows, schema, **kw)}

    good = ([R(bn_feed_age_s=0.1), R(bn_feed_age_s=0.2, a=None, a_missing=1.0),
             R(bn_feed_age_s=0.3, state_status="GAP_AT_CUTOFF"),
             R(bn_feed_age_s=0.4, split="train", t0=0, decision_time=1.0),
             R(bn_feed_age_s=0.5, split="score", t0=0, decision_time=100.0)]
            * 12)          # clear the 50-row signature minimum
    g = res(good)
    for k in ("tape_non_empty", "no_undeclared_reduction",
              "guard_beside_every_nullable", "no_zero_imputation",
              "bn_recv_ns_was_supplied", "state_status_present",
              "gaps_were_supplied", "statuses_are_declared_values",
              "feature_asof_never_after_decision", "embargo_respected"):
        ok(g[k], f"a clean tape passes {k}")

    # FALSIFIERS -- each must FIRE (rule 15)
    ok(not res([])["tape_non_empty"], "an EMPTY tape fails, never passes")
    ok(not res([{k: v for k, v in R().items() if k != "a_missing"}])
       ["guard_beside_every_nullable"],
       "a nullable present WITHOUT its guard flag is caught (guardless pin)")
    ok(not res([R(a=0.0, a_missing=1.0)])["no_zero_imputation"],
       "guard says MISSING while the value is PRESENT -- caught exactly; this "
       "is the fingerprint float(x or 0.0) leaves")
    ok(res([R(a=None, a_missing=1.0), R()])["no_zero_imputation"],
       "a genuine None beside a hot flag is CLEAN")
    ok(res([R()] * 3)["no_zero_imputation"],
       "a column that is LEGITIMATELY always computable is NOT flagged -- the "
       "false positive the seam test exposed in the signature version, which "
       "would have fired on gen_age_s and pm_feed_age_s on every real tape")
    ok(not res([R(bn_feed_age_s=0.1), R(bn_feed_age_s=0.1)])
       ["bn_recv_ns_was_supplied"],
       "a CONSTANT freshness family is caught (bn_recv_ns omitted)")
    ok(not res([R()], gapped_slugs_expected=5)["gaps_were_supplied"],
       "zero GAP_AT_CUTOFF with gapped slugs expected is caught")
    ok(res([R()], gapped_slugs_expected=0)["gaps_were_supplied"],
       "zero GAP_AT_CUTOFF is fine when the population HAS no gapped slugs")
    ok(not res([{k: v for k, v in R().items() if k != "state_status"}])
       ["state_status_present"], "a tape without state_status is caught")
    ok(not res([R(state_status="INVENTED")])["statuses_are_declared_values"],
       "an undeclared status value is caught")
    ok(not res([R(feature_asof=9.0, decision_time=2.0)])
       ["feature_asof_never_after_decision"],
       "a knowledge-time violation is caught")
    ok(not res([{k: v for k, v in R().items()
                 if k not in ("feature_asof", "decision_time")}])
       ["feature_asof_never_after_decision"],
       "a tape that does not CARRY the fields cannot pass the invariant "
       "vacuously -- absence is failure, not silence")
    near = [R(split="train", t0=0, decision_time=100.0),
            R(split="score", t0=0, decision_time=110.0)]
    ok(not res(near)["embargo_respected"],
       "a 10s gap against a 60s declared embargo is caught (R-184 blocker)")
    ok(not res([R()])["embargo_respected"],
       "no split labels -> embargo NOT certified, rather than assumed clean")
    # declared reductions are honoured
    lean = [{**{k: v for k, v in R().items() if k != "a"},
             "declared_reductions": ["a"]}]
    ok(res(lean)["no_undeclared_reduction"],
       "a DECLARED reduction is accepted; only undeclared absence fails")

    # refusals, never passes
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            load_schema(Path(td) / "nope.json")
        except GateRefused:
            ok(True, "a missing schema REFUSES rather than passing")
        else:
            ok(False, "must refuse without a schema")
        try:
            list(iter_tape(Path(td) / "absent.jsonl"))
        except GateRefused:
            ok(True, "an absent tape REFUSES")
        else:
            ok(False, "must refuse an absent tape")
        weird = Path(td) / "w.csv"
        weird.write_text("a,b\n1,2\n", encoding="utf-8")
        try:
            list(iter_tape(weird))
        except GateRefused:
            ok(True, "an unrecognised layout REFUSES rather than guessing")
        else:
            ok(False, "must refuse an unrecognised layout")

    # ---- THE SEAM TEST (R-187) -------------------------------------------
    # Every case above feeds the gate dicts I INVENTED. That is precisely how
    # this gate came to search top level while the builder nested, and to add
    # t0 to a clock that might already be absolute: per-module selftests,
    # however falsified, cannot certify a seam. So round-trip a REAL builder
    # row through the REAL schema.
    import harmful_state_features as SF
    real_schema = SF.declared_schema()
    k2 = ("BUY", 0.5)
    tp = SF._synth_tape(level_t={k2: [10.0, 20.0]}, level_v={k2: [100.0, 60.0]},
                        pm_event_t=[10.0, 20.0])
    real_row = SF.features_at(tp, SF._row(25.0))
    flat_tape = [dict(real_row, slug="s", t0=1787650500)]
    under, got = locate_features(flat_tape, real_schema)
    ok(under is None and set(real_schema["emitted_fields"]) <= set(got[0]),
       "SEAM: a real builder row is located at top level and complete")

    nested_tape = [{"slug": "s", "t0": 1787650500, "state": dict(real_row)}]
    under_n, got_n = locate_features(nested_tape, real_schema)
    ok(under_n == "state"
       and set(real_schema["emitted_fields"]) <= set(got_n[0]),
       "SEAM: the SAME row nested under `state` is still located -- the exact "
       "layout mismatch that made three predicates pass on nothing")
    # Two real rows, the second at a level the tape never saw, so the
    # velocity nullables genuinely carry None with their flags set.
    sparse = SF.features_at(tp, SF._row(25.0, level=0.99))
    nested_many = [{"slug": "s", "t0": 1787650500, "state": dict(r)}
                   for r in ([real_row, sparse] * 30)]
    gn = {q["predicate"]: q["pass"] for q in gate(nested_many, real_schema)}
    ok(gn["guard_beside_every_nullable"],
       "SEAM: nested rows are now CHECKED for guard-beside-nullable, not "
       "skipped-and-passed -- the defect that made three predicates pass on "
       "an unreadable layout")
    ok(gn["no_zero_imputation"],
       "SEAM: REAL builder rows pass the exact zero-imputation test -- "
       "including the always-computable columns the signature version would "
       "have falsely flagged")
    forged = [{"slug": "s", "t0": 1,
               "state": dict(real_row, gen_age_s=0.0, gen_age_missing=1.0)}]
    gf = {q["predicate"]: q["pass"] for q in gate(forged, real_schema)}
    ok(not gf["no_zero_imputation"],
       "SEAM: a real row with ONE value imputed is caught -- so the exact test "
       "still fires on the defect it exists for")

    opaque = [{"slug": "s", "t0": 1, "payload": {"nothing": 1}}]
    try:
        locate_features(opaque, real_schema)
    except GateRefused:
        ok(True, "SEAM: an unlocatable layout REFUSES instead of passing on an "
                 "empty intersection")
    else:
        ok(False, "must refuse when no declared field can be found")

    # clock basis is READ, not assumed
    rel = {"t0": 1000.0, "decision_time": 25.0}
    ab = {"t0": 1000.0, "decision_time": 1787650525.0,
          "clock_basis": "absolute_epoch"}
    ok(clock_of(rel, real_schema) == 1025.0,
       "window-relative decision_time is made absolute by adding t0")
    ok(clock_of(ab, real_schema) == 1787650525.0,
       "an ALREADY-ABSOLUTE decision_time is not double-counted -- the basis "
       "is read from the tape, never assumed")
    ok(clock_of(rel, real_schema) != clock_of(
        dict(rel, clock_basis="absolute_epoch"), real_schema),
       "the two bases give DIFFERENT answers, so a rule that ignores the pin "
       "cannot pass (R-42 mirror on the clock)")

    print(f"da_state_tape_verify selftests: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", nargs="?", choices=["verify"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tape", default=None)
    ap.add_argument("--gapped-slugs", type=int, default=None)
    a = ap.parse_args()
    if a.selftest or not a.cmd:
        return _selftests()
    if not a.tape:
        raise SystemExit("--tape PATH required; refusing to guess a tape")
    rep = verify(Path(a.tape), gapped_slugs_expected=a.gapped_slugs)
    print(json.dumps({k: v for k, v in rep.items() if k != "predicates"},
                     indent=2, sort_keys=True))
    print("\nPREDICATES")
    for x in rep["predicates"]:
        print(f"  [{'PASS' if x['pass'] else 'FAIL'}] {x['predicate']}: {x['detail']}")
    print(f"\nALL PASS: {rep['all_pass']}")
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
