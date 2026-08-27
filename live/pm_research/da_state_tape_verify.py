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
    orphans = [n for n, f in pairs.items() if n in present and f not in present]
    p("guard_beside_every_nullable", not orphans,
      f"orphaned nullables (present without their flag): {orphans or 'none'}"
      + ("  <-- this is the R-185 guardless-pin defect" if orphans else ""))

    # --- zero-imputation signature ---------------------------------------
    # A nullable that is 100% finite while its flag is 100% zero is what
    # `float(x or 0.0)` leaves behind. It is a SIGNATURE, not a proof, so it is
    # reported per column rather than collapsed into one verdict.
    suspects = []
    for n, f in pairs.items():
        if n not in present or f not in present:
            continue
        nulls = sum(1 for r in rows if r.get(n) is None)
        flag_hot = sum(1 for r in rows if float(r.get(f) or 0.0) != 0.0)
        if nulls == 0 and flag_hot == 0:
            suspects.append(n)
    p("missingness_is_representable", not suspects,
      f"columns with ZERO nulls AND an all-zero guard flag: "
      f"{suspects or 'none'}"
      + ("  <-- signature of zero-imputation: a missing value became 0.0 with "
         "nothing to say so" if suspects else ""))

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
    p("statuses_are_declared_values", bad_status == 0,
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
        def clock(r):
            for k in ("decision_epoch", "t0_plus_t_start", "abs_decision_time"):
                if r.get(k) is not None:
                    return float(r[k])
            if r.get("t0") is not None and r.get("decision_time") is not None:
                return float(r["t0"]) + float(r["decision_time"])
            return None
        trc = [c for c in map(clock, tr) if c is not None]
        scc = [c for c in map(clock, sc) if c is not None]
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

    schema = {
        "family": "PRED_STATE_V1",
        "emitted_fields": ["a", "a_missing", "bn_feed_age_s", "state_status",
                           "feature_asof", "decision_time"],
        "statuses": ["OK", "PRE_WINDOW", "GAP_AT_CUTOFF"],
        "status_field": "state_status",
        "nullable_fields_and_their_flags": {"a": "a_missing"},
    }

    def R(**kw):
        base = {"a": 1.0, "a_missing": 0.0, "bn_feed_age_s": 0.1,
                "state_status": "OK", "feature_asof": 1.0,
                "decision_time": 2.0}
        base.update(kw)
        return base

    def res(rows, **kw):
        return {p["predicate"]: p["pass"] for p in gate(rows, schema, **kw)}

    good = [R(bn_feed_age_s=0.1), R(bn_feed_age_s=0.2, a=None, a_missing=1.0),
            R(bn_feed_age_s=0.3, state_status="GAP_AT_CUTOFF"),
            R(bn_feed_age_s=0.4, split="train", t0=0, decision_time=1.0),
            R(bn_feed_age_s=0.5, split="score", t0=0, decision_time=100.0)]
    g = res(good)
    for k in ("tape_non_empty", "no_undeclared_reduction",
              "guard_beside_every_nullable", "missingness_is_representable",
              "bn_recv_ns_was_supplied", "state_status_present",
              "gaps_were_supplied", "statuses_are_declared_values",
              "feature_asof_never_after_decision", "embargo_respected"):
        ok(g[k], f"a clean tape passes {k}")

    # FALSIFIERS -- each must FIRE (rule 15)
    ok(not res([])["tape_non_empty"], "an EMPTY tape fails, never passes")
    ok(not res([{k: v for k, v in R().items() if k != "a_missing"}])
       ["guard_beside_every_nullable"],
       "a nullable present WITHOUT its guard flag is caught (guardless pin)")
    ok(not res([R(), R(), R()])["missingness_is_representable"],
       "all-finite column with an all-zero flag is caught (zero-imputation)")
    ok(res([R(a=None, a_missing=1.0), R()])["missingness_is_representable"],
       "a column that does carry nulls is NOT flagged (no false positive)")
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
