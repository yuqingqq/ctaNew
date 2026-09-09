#!/usr/bin/env python3
"""DA 152: DRIVEN CASES FOR THE USER'S FIVE DEFECTS, BUILT BEFORE THE FIXES.

Each case answers one question about DE's code and answers it the way rule
33 demands: it must PASS on the real thing, FAIL on a known-bad, and REFUSE
a partial input. A case that only fires on today's broken code proves
nothing about tomorrow's fix.

WHY THESE ARE WRITTEN FROM THE ENTRY POINT. Defect (1) -- the cancel-matched
null being unreachable -- survived because the branch was unit-tested and
the CALL SITE was not. So case 1 reads `run_day`'s OWN call, not
`null_draws_valued`'s signature. That is the lesson of my own DA 146 green
too: I drove that the book-code predicate CAN pass and never drove that it
can pass ON A SUBSET, which is case 4 and is the direct heir of that miss.

NOTHING HERE FIXES DE'S FILES. Each case reports PRESENT or FIXED.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PRESENT, FIXED, REFUSED = "DEFECT_PRESENT", "DEFECT_FIXED", "REFUSED"


class CaseRefused(RuntimeError):
    """The case cannot be evaluated on the input it was given."""


# ---------------------------------------------------------------- case 1
NULL_KWARGS = ("arm_cancels", "control_set_path")


def _call_kwargs_at(func, callee: str) -> list:
    """Every keyword NAME passed to `callee` inside `func`'s own body.

    Reads the CALLER, which is the whole point: the branch inside the
    callee was tested and the call site was not.
    """
    src = inspect.getsource(func)
    tree = ast.parse(src[src.index("def "):])
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if nm == callee:
                out.append(sorted(k.arg for k in n.keywords if k.arg))
    return out


def case_1_cancel_matched_null_reachable(func=None, callee="null_draws_valued"):
    """(1) Is the CANCEL-matched null reachable from the entry point?

    `null_draws_valued` runs the cancel-matched control only when
    `arm_cancels is not None`. If `run_day` never passes it, the historical
    row-matched null runs on every real day and the branch is dead.
    """
    if func is None:
        import de_multiday_gate1_runner as R
        func = R.run_day
    calls = _call_kwargs_at(func, callee)
    if not calls:
        raise CaseRefused(
            f"PARTIAL INPUT: {getattr(func, '__name__', func)!r} contains no "
            f"call to {callee!r}, so this case has nothing to read. A case "
            f"that returns FIXED here would be reporting its own blindness.")
    reaching = [c for c in calls if any(k in c for k in NULL_KWARGS)]
    return {"case": "cancel_matched_null_reachable",
            "verdict": FIXED if reaching else PRESENT,
            "n_call_sites": len(calls),
            "kwargs_at_each_call_site": calls,
            "kwargs_that_would_reach_the_cancel_matched_branch":
                list(NULL_KWARGS),
            "n_call_sites_that_reach_it": len(reaching),
            "why": ("the branch is guarded by `arm_cancels is not None`; a "
                    "call site that passes neither kwarg runs the historical "
                    "row-matched null however well the branch itself tests")}


# ---------------------------------------------------------------- case 2
def case_2_decision_count(stream, theta: float):
    """(2) Is the decision count ROWS or ACTIONS?

    `decisions = [r for r in stream if score >= theta]; len(decisions)`
    counts every above-threshold ROW. Under a per-row stream one generation
    contributes several, and that number feeds admissibility and the null's
    size. The action is the cancellable GENERATION.
    """
    if not stream:
        raise CaseRefused(
            "PARTIAL INPUT: an empty stream cannot distinguish rows from "
            "actions -- both counts are 0 and the case would read FIXED.")
    for r in stream:
        for k in ("slug", "side", "gen", "score"):
            if k not in r:
                raise CaseRefused(
                    f"PARTIAL INPUT: a stream row is missing {k!r}; without "
                    f"`gen` there is no action to de-duplicate to.")
    above = [r for r in stream if float(r["score"]) >= theta]
    actions = {(r["slug"], r["side"], r["gen"]) for r in above}
    return {"case": "decision_count_is_actions",
            "verdict": FIXED if len(above) == len(actions) else PRESENT,
            "n_rows_above_theta": len(above),
            "n_distinct_actions": len(actions),
            "inflation": len(above) - len(actions),
            "inflation_ratio": (round(len(above) / len(actions), 4)
                                if actions else None),
            "why": ("rule 2: rows are actions. A count of rows feeding "
                    "`min_decisions_per_arm_day` and the null's size is a "
                    "different population from the one being decided over")}


# ---------------------------------------------------------------- case 3
def case_3_cancel_per_reference_generation(cancels):
    """(3) Is the one-cancel-per-generation invariant true of the REFERENCE
    id space, or only of the POLICY id space?

    A repost gives one reference generation a second policy generation
    (`7` -> `7.r1`). The invariant computed over `policy_gen` passes while
    the reference generation is cancelled twice -- and the cancel-matched
    control matches on REFERENCE generations, so the premise it rests on is
    the one that fails.
    """
    if not cancels:
        raise CaseRefused(
            "PARTIAL INPUT: no cancel records -- both id spaces are trivially "
            "unique over an empty set and the case would read FIXED.")
    pol, ref = {}, {}
    for c in cancels:
        if "policy_gen" not in c or "ref_gen" not in c:
            raise CaseRefused(
                "PARTIAL INPUT: a cancel record carries only one of "
                "`policy_gen` / `ref_gen`; the two id spaces cannot be "
                "compared from it, which is the whole question.")
        k = (c.get("slug"), c.get("side"))
        pol.setdefault((k, str(c["policy_gen"])), 0)
        pol[(k, str(c["policy_gen"]))] += 1
        ref.setdefault((k, c["ref_gen"]), 0)
        ref[(k, c["ref_gen"])] += 1
    pol_dupes = {k: v for k, v in pol.items() if v > 1}
    ref_dupes = {k: v for k, v in ref.items() if v > 1}
    return {"case": "one_cancel_per_REFERENCE_generation",
            "verdict": FIXED if not ref_dupes else PRESENT,
            "n_cancels": len(cancels),
            "unique_in_POLICY_id_space": not pol_dupes,
            "unique_in_REFERENCE_id_space": not ref_dupes,
            "reference_generations_cancelled_more_than_once":
                [{"slug_side": list(k[0]), "ref_gen": k[1], "n": v}
                 for k, v in sorted(ref_dupes.items(), key=str)],
            "policy_ids_seen": sorted({str(c["policy_gen"]) for c in cancels}),
            "why": ("the invariant passes on POLICY ids because a repost "
                    "MINTS a new one; the control matches on REFERENCE "
                    "generations, so a reference generation cancelled twice "
                    "breaks the premise that the cancel is the action")}


# ---------------------------------------------------------------- case 4
def case_4_book_code_predicate_requires_the_whole_set(
        predicate, expected_modules, digests):
    """(4) Does the book-code predicate require the WHOLE recorded set?

    THE DIRECT HEIR OF MY OWN DA 146 GREEN. I drove that the predicate CAN
    pass -- a receipt carrying all five current digests returns
    `BOOK_SCORING_CODE_MATCHES` -- and I never drove that it can pass on a
    SUBSET. The user's probe did: a receipt carrying ONE of the five
    returns `BOOK_SCORING_CODE_MATCHES, n_checked: 1`.
    """
    expected = sorted(expected_modules)
    if len(expected) < 2:
        raise CaseRefused(
            "PARTIAL INPUT: a set of fewer than two modules cannot have a "
            "proper subset, so this case could not fail and would report "
            "FIXED for a predicate that never checks membership.")
    missing = [m for m in expected if m not in digests]
    if missing:
        raise CaseRefused(
            f"PARTIAL INPUT: no current digest supplied for {missing}; the "
            f"case cannot build the full-set control it needs.")

    def _run(mods):
        try:
            return True, predicate(
                {"producing_code": {"import_closure": {"modules": mods}}},
                where="da_de_fix_cases")
        except Exception as e:                                # noqa: BLE001
            return False, e
    ok_full, full = _run({m: digests[m] for m in expected})
    subsets = {}
    for m in expected:
        ok_one, _ = _run({m: digests[m]})
        subsets[m] = ok_one
    accepted = sorted(m for m, ok in subsets.items() if ok)
    return {"case": "book_code_predicate_requires_the_whole_set",
            "verdict": PRESENT if accepted else (FIXED if ok_full else PRESENT),
            "full_set_accepted": ok_full,
            "n_expected_modules": len(expected),
            "single_module_receipts_accepted": accepted,
            "n_single_module_receipts_accepted": len(accepted),
            "full_set_n_checked": (full.get("n_checked")
                                   if ok_full and isinstance(full, dict)
                                   else None),
            "why": ("a predicate that accepts a subset of the set it names "
                    "checks a sample of a sample: REV 123 established the "
                    "five are a hand-typed slice of a recorded 49, and this "
                    "shows the five are not even required in full"),
            "provenance": ("DA 146 drove `can it pass` and not `can it pass "
                           "on a subset`; that gap is this case")}


# --------------------------------------------------------------- case 4b
def case_4b_params_protocol_matches_version(payload, *, loader=None):
    """(4b) Does a params payload's protocol string match its own version?"""
    if not isinstance(payload, dict):
        raise CaseRefused("PARTIAL INPUT: the params payload is not a mapping.")
    proto, ver = payload.get("protocol"), payload.get("version")
    if proto is None or ver is None:
        raise CaseRefused(
            f"PARTIAL INPUT: payload carries protocol={proto!r} "
            f"version={ver!r}; a version claim needs both.")
    import re
    m = re.search(r"V(\d+)\s*$", str(proto))
    pv = m.group(1) if m else None
    vv = re.sub(r"[^0-9]", "", str(ver))
    agree = pv is not None and vv != "" and pv == vv
    out = {"case": "params_protocol_matches_version",
           "verdict": FIXED if agree else PRESENT,
           "protocol": proto, "version": ver,
           "version_in_the_protocol_string": pv,
           "version_field": vv,
           "why": ("a payload whose protocol string names one version while "
                   "its version field names another is accepted by a loader "
                   "that checks only the string")}
    if loader is not None:
        try:
            loader(payload)
            out["loader_accepts_it"] = True
        except Exception as e:                                # noqa: BLE001
            out["loader_accepts_it"] = False
            out["loader_refusal"] = type(e).__name__
    return out


CASES = ("cancel_matched_null_reachable", "decision_count_is_actions",
         "one_cancel_per_REFERENCE_generation",
         "book_code_predicate_requires_the_whole_set",
         "params_protocol_matches_version")


# ======================================================================
# THE CASES' OWN FALSIFIERS. A case that only fires on today's broken code
# proves nothing about tomorrow's fix, so each is driven THREE ways: on a
# construction where the defect is PRESENT, on one where it is FIXED, and
# on a PARTIAL input it must refuse rather than answer.
# ======================================================================
def selftest() -> tuple:
    checks: list = []

    def ck(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed),
                       "detail": detail})

    def refuses(fn, tag):
        try:
            fn()
            return f"{tag}: ADMITTED (should have refused)"
        except CaseRefused as e:
            return f"{tag}: REFUSED -- {str(e).split(':')[0]}"

    # ---- case 1 -------------------------------------------------------
    def _broken():
        nul = null_draws_valued(mod, bk, base, by_side, n_draws=1, seed=2)
    def _fixed():
        nul = null_draws_valued(mod, bk, base, by_side, n_draws=1, seed=2,
                                arm_cancels=ac, control_set_path=p)
    def _none():
        x = 1
    b1 = case_1_cancel_matched_null_reachable(_broken)
    f1 = case_1_cancel_matched_null_reachable(_fixed)
    ck("CASE 1 fires on a call site that passes NEITHER kwarg and clears on "
       "one that passes both -- read at the CALLER, which is where this "
       "defect lived while the callee's branch was tested",
       b1["verdict"] == PRESENT and b1["n_call_sites_that_reach_it"] == 0
       and f1["verdict"] == FIXED and f1["n_call_sites_that_reach_it"] == 1,
       f"no-kwarg call -> {b1['verdict']} ({b1['kwargs_at_each_call_site']}); "
       f"both-kwarg call -> {f1['verdict']} "
       f"({f1['kwargs_at_each_call_site']})")
    ck("CASE 1 REFUSES a function containing no such call at all, rather "
       "than reporting FIXED for code it never examined",
       "REFUSED" in refuses(
           lambda: case_1_cancel_matched_null_reachable(_none), "x"),
       refuses(lambda: case_1_cancel_matched_null_reachable(_none),
               "a caller with no call to the null"))

    # ---- case 2 -------------------------------------------------------
    _rows = [{"slug": "s", "side": "BUY_UP", "gen": 0, "score": 0.9},
             {"slug": "s", "side": "BUY_UP", "gen": 0, "score": 0.95},
             {"slug": "s", "side": "BUY_UP", "gen": 1, "score": 0.8},
             {"slug": "s", "side": "BUY_UP", "gen": 2, "score": 0.1}]
    b2 = case_2_decision_count(_rows, 0.5)
    _one = [{"slug": "s", "side": "BUY_UP", "gen": g, "score": 0.9}
            for g in (0, 1, 2)]
    f2 = case_2_decision_count(_one, 0.5)
    ck("CASE 2 shows the count BEFORE and AFTER de-duplication to actions: "
       "3 rows above theta over 2 distinct generations is an inflation of 1 "
       "(ratio 1.5); a stream with one row per generation is clean",
       b2["verdict"] == PRESENT and b2["n_rows_above_theta"] == 3
       and b2["n_distinct_actions"] == 2 and b2["inflation"] == 1
       and b2["inflation_ratio"] == 1.5
       and f2["verdict"] == FIXED and f2["inflation"] == 0,
       f"inflated -> rows {b2['n_rows_above_theta']} vs actions "
       f"{b2['n_distinct_actions']}, ratio {b2['inflation_ratio']}; "
       f"clean -> {f2['verdict']}")
    ck("CASE 2 REFUSES an empty stream and a row with no `gen` -- on both, "
       "rows and actions agree trivially and the case would read FIXED",
       "REFUSED" in refuses(lambda: case_2_decision_count([], 0.5), "x")
       and "REFUSED" in refuses(
           lambda: case_2_decision_count(
               [{"slug": "s", "side": "BUY_UP", "score": 0.9}], 0.5), "y"),
       refuses(lambda: case_2_decision_count([], 0.5), "empty stream") + "; "
       + refuses(lambda: case_2_decision_count(
           [{"slug": "s", "side": "BUY_UP", "score": 0.9}], 0.5),
           "row with no gen"))

    # ---- case 3 -------------------------------------------------------
    _rep = [{"slug": "s", "side": "BUY_UP", "ref_gen": 7, "policy_gen": "7"},
            {"slug": "s", "side": "BUY_UP", "ref_gen": 7,
             "policy_gen": "7.r1"}]
    b3 = case_3_cancel_per_reference_generation(_rep)
    _ok = [{"slug": "s", "side": "BUY_UP", "ref_gen": 7, "policy_gen": "7"},
           {"slug": "s", "side": "BUY_UP", "ref_gen": 8, "policy_gen": "8"}]
    f3 = case_3_cancel_per_reference_generation(_ok)
    ck("CASE 3 SEPARATES THE TWO ID SPACES EXPLICITLY: the repost pair "
       "(`7`, `7.r1`) is UNIQUE in the policy space -- which is why the "
       "shipped invariant passes -- and NOT unique in the reference space, "
       "where reference generation 7 is cancelled twice. That is the "
       "premise the cancel-matched control rests on",
       b3["verdict"] == PRESENT
       and b3["unique_in_POLICY_id_space"] is True
       and b3["unique_in_REFERENCE_id_space"] is False
       and b3["reference_generations_cancelled_more_than_once"][0]["n"] == 2
       and f3["verdict"] == FIXED
       and f3["unique_in_REFERENCE_id_space"] is True,
       f"repost pair -> policy-unique {b3['unique_in_POLICY_id_space']}, "
       f"reference-unique {b3['unique_in_REFERENCE_id_space']}, "
       f"{b3['reference_generations_cancelled_more_than_once']}; "
       f"distinct generations -> {f3['verdict']}")
    ck("CASE 3 REFUSES an empty cancel list and a record carrying only one "
       "of the two id fields -- the comparison IS the case",
       "REFUSED" in refuses(
           lambda: case_3_cancel_per_reference_generation([]), "x")
       and "REFUSED" in refuses(
           lambda: case_3_cancel_per_reference_generation(
               [{"slug": "s", "side": "BUY_UP", "ref_gen": 7}]), "y"),
       refuses(lambda: case_3_cancel_per_reference_generation([]),
               "no cancels") + "; "
       + refuses(lambda: case_3_cancel_per_reference_generation(
           [{"slug": "s", "side": "BUY_UP", "ref_gen": 7}]),
           "record with no policy_gen"))

    # ---- case 4 -------------------------------------------------------
    _EXP = ("a.py", "b.py", "c.py")
    _DIG = {m: f"{i}" * 64 for i, m in enumerate(_EXP)}

    def _subset_ok(receipt, *, where, root=None):
        rec = {m: d for m, d in ((receipt.get("producing_code") or {})
                                 .get("import_closure") or {})
               .get("modules", {}).items() if m in _EXP}
        if not rec:
            raise RuntimeError("NOT_RECORDED")
        for m, d in rec.items():
            if d != _DIG[m]:
                raise RuntimeError("DIFFERS")
        return {"status": "MATCHES", "n_checked": len(rec)}

    def _whole_only(receipt, *, where, root=None):
        rec = ((receipt.get("producing_code") or {})
               .get("import_closure") or {}).get("modules", {})
        missing = [m for m in _EXP if m not in rec]
        if missing:
            raise RuntimeError(f"INCOMPLETE {missing}")
        return _subset_ok(receipt, where=where)
    b4 = case_4_book_code_predicate_requires_the_whole_set(
        _subset_ok, _EXP, _DIG)
    f4 = case_4_book_code_predicate_requires_the_whole_set(
        _whole_only, _EXP, _DIG)
    ck("CASE 4 IS THE PARTIAL-INPUT REFUSAL ITSELF: a predicate that "
       "accepts any single-module receipt reads DEFECT_PRESENT naming every "
       "module it accepted alone; one that demands the whole recorded set "
       "reads FIXED. ***This is the control DA 146 did not run: I drove "
       "that the predicate CAN pass, not that it can pass on a SUBSET***",
       b4["verdict"] == PRESENT
       and b4["n_single_module_receipts_accepted"] == 3
       and b4["full_set_accepted"] is True
       and f4["verdict"] == FIXED
       and f4["n_single_module_receipts_accepted"] == 0
       and f4["full_set_accepted"] is True,
       f"subset-accepting predicate -> {b4['verdict']}, accepted alone "
       f"{b4['single_module_receipts_accepted']}; whole-set predicate -> "
       f"{f4['verdict']}, full set still accepted "
       f"{f4['full_set_accepted']}")
    ck("CASE 4 REFUSES a one-module expectation (no proper subset exists, "
       "so it could not fail) and a digest map missing a module it must "
       "build the full-set control from",
       "REFUSED" in refuses(
           lambda: case_4_book_code_predicate_requires_the_whole_set(
               _subset_ok, ("a.py",), _DIG), "x")
       and "REFUSED" in refuses(
           lambda: case_4_book_code_predicate_requires_the_whole_set(
               _subset_ok, _EXP, {"a.py": _DIG["a.py"]}), "y"),
       refuses(lambda: case_4_book_code_predicate_requires_the_whole_set(
           _subset_ok, ("a.py",), _DIG), "one-module expectation") + "; "
       + refuses(lambda: case_4_book_code_predicate_requires_the_whole_set(
           _subset_ok, _EXP, {"a.py": _DIG["a.py"]}), "incomplete digests"))

    # ---- case 4b ------------------------------------------------------
    b5 = case_4b_params_protocol_matches_version(
        {"protocol": "P003_DE_MULTIDAY_GATE1_PARAMS_V20", "version": "v26"})
    f5 = case_4b_params_protocol_matches_version(
        {"protocol": "P003_DE_MULTIDAY_GATE1_PARAMS_V26", "version": "v26"})
    ck("CASE 4b fires where the protocol string names a different version "
       "from the version field and clears where they agree",
       b5["verdict"] == PRESENT and b5["version_in_the_protocol_string"] == "20"
       and b5["version_field"] == "26" and f5["verdict"] == FIXED,
       f"V20 protocol on a v26 payload -> {b5['verdict']} "
       f"({b5['version_in_the_protocol_string']} vs {b5['version_field']}); "
       f"matching -> {f5['verdict']}")
    ck("CASE 4b REFUSES a payload missing either half -- a version claim "
       "needs both",
       "REFUSED" in refuses(
           lambda: case_4b_params_protocol_matches_version(
               {"version": "v26"}), "x"),
       refuses(lambda: case_4b_params_protocol_matches_version(
           {"version": "v26"}), "no protocol string"))

    fails = sum(1 for c in checks if not c["passed"])
    for c in checks:
        print(f"{'ok  ' if c['passed'] else 'FAIL'} {c['check']}")
        print(f"       {c['detail']}")
    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main(argv=None) -> int:
    a = (argv if argv is not None else sys.argv[1:])
    if "--selftest" in a:
        return 1 if selftest()[1] else 0
    import de_multiday_gate1_runner as R
    import hashlib
    here = Path(R.__file__).resolve().parent
    dig = {m: hashlib.sha256((here / m).read_bytes()).hexdigest()
           for m in R.SCORING_PATH_MODULES}
    out = {"case_1": case_1_cancel_matched_null_reachable(),
           "case_4": case_4_book_code_predicate_requires_the_whole_set(
               R.assert_book_scoring_code, R.SCORING_PATH_MODULES, dig)}
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
