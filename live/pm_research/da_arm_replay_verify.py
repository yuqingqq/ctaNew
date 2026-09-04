#!/usr/bin/env python3
"""Independent verification of a seven-arm replay's OUTPUT. DA's instrument.

WHAT THIS IS FOR, in one sentence: **an arm that produces a PLAUSIBLE NUMBER
from a path that did not really run** is the failure this programme keeps
paying for, and a checker that reads the producer's own summary cannot see it.

INDEPENDENT BY CONSTRUCTION (R-235, do-not-harmonize). This module imports
NOTHING from `de_lane4_real_parity` or any arm runner. It consumes the emitted
ARTIFACT as a DATA CONTRACT and recomputes what it can from the artifact's own
numbers and from the tape. The moment it imported the producer's code,
agreement would stop being evidence -- a checker that shares an expression
with the thing it checks agrees with it by construction.

THREE QUESTIONS, AND EACH HAS A NAMED REFUSAL RATHER THAN A DEFAULT.

  1. IS THIS A REAL ARM OR A STUB? The producer's own declaration is read but
     never trusted alone -- `declared_parameters.stub` is the producer's word.
     What counts as EVIDENCE is computable: how many distinct predictors the
     submissions carry, whether the runnability table blocks arms for want of
     a released predictor, and -- when per-generation scores are supplied --
     whether the scores are reproducible from a pure hash of the identifiers,
     which a real predictor's cannot be because a real predictor reads the
     market. **A REAL CLAIM WITH NO EVIDENCE DOES NOT READ AS REAL.**

  2. IS THE OUTPUT COMPLETE, OR PARTIAL AND REPORTED AS COMPLETE? Every
     population count must RECONCILE: the status histogram must sum to the
     population it claims, and the headline admitted count must equal the
     histogram's own ADMITTED cell. And every gate that reports failures must
     report a DENOMINATOR -- "0 failing" with no count of what was checked is
     the empty-set trap, and it is the shape that has passed three times in
     this programme.

  3. DID IT CONSUME THE POPULATION IT CLAIMS? A named population is not a
     consumed one. The artifact must carry a DIGEST of its input that this
     module can recompute from the tape. Absent one, the answer is
     UNVERIFIABLE -- never a pass.

WHAT IT DOES NOT DO: it does not score, rank or adjudicate arms, and it never
decides whether an arm may be used (rule 14). It reports whether the OUTPUT
can be believed.

    python3 live/pm_research/da_arm_replay_verify.py --artifact <path>
    python3 live/pm_research/da_arm_replay_verify.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pm_tape_density as _TDROOT                              # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = _TDROOT.DATA_ROOT
DERIVED = DATA_ROOT / "data/pm_5min/derived"

EXPECTED_CHECKS = 26

#: §8.1's REQUIRED OUTPUT, ENUMERATED HERE INDEPENDENTLY of the producer's own
#: list. Written from the plan's eleven named quantities, NOT copied from
#: `de_lane4_real_parity.SECTION_8_1_FIELDS` -- the point of the audit is that
#: two enumerations made separately agree, and an enumeration read off the
#: thing it audits agrees by construction.
SECTION_8_1_REQUIRED: dict[str, tuple[str, ...]] = {
    "maker_pnl": ("maker_pnl_cents",),
    "spread_capture": ("spread_capture_cents",),
    "post_fill_markout": ("post_fill_markout_cents",),
    "fill_share_retention": ("fill_share_retention",),
    "rho": ("rho",),
    "cancels": ("cancels_effective", "cancels_stale", "cancels_unresolved"),
    "traffic": ("holds_total", "hold_seconds_total", "reposts",
                "queue_reset_cost_cents_total"),
    "inventory_terminal_peak": ("terminal_inventory", "peak_inventory"),
    "inventory_loss": ("inventory_loss_cents",),
}

#: A counter whose ZERO is ambiguous unless something says it was evaluated.
#: This is the class fixed on the blackout mask on 2026-09-04: an exclusion
#: with no status is indistinguishable from an exclusion that did not happen.
COUNTER_FIELDS = ("cancels_effective", "cancels_stale", "cancels_unresolved",
                  "holds_total", "hold_seconds_total", "reposts",
                  "queue_reset_cost_cents_total")


class ArmVerifyRefused(Exception):
    """An artifact this instrument must not summarise."""


def load_artifact(path: Path) -> dict[str, Any]:
    """Read and REFUSE anything this cannot grade as an arm replay output."""
    if not path.is_file():
        raise ArmVerifyRefused(
            f"REFUSED: no artifact at {path}. An absent artifact is not an "
            f"unverified one -- there is nothing to grade, and reporting a "
            f"clean verification for a file that does not exist is the "
            f"empty-set trap on this whole instrument.")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ArmVerifyRefused(
            f"REFUSED: {path} is not readable JSON ({e}). A file this cannot "
            f"parse is NOT the same as a file that failed verification, and "
            f"the two must not share an answer.") from None
    if not isinstance(doc, dict):
        raise ArmVerifyRefused(
            f"REFUSED: {path} is a {type(doc).__name__}, not an object.")
    return doc


# ------------------------------------------------------------ question 1
def stub_or_real(doc: dict[str, Any],
                 scores: list[dict] | None = None) -> dict[str, Any]:
    """Real arm or stub? The declaration is read; the EVIDENCE is computed.

    THE SCORE TEST IS THE STRONG ONE and it is the reason this takes an
    optional `scores` argument. A stub score in this programme is a
    sha256-derived function of the identifiers; a real predictor reads the
    market. So if every score is reproducible from `(slug, side, gen)` alone,
    the arm did not consume the market -- whatever it declares. That is the
    difference between a plausible number and a produced one.
    """
    declared = doc.get("declared_parameters") or {}
    stub_block = declared.get("stub")
    stub_declared = isinstance(stub_block, dict)

    inert = ((doc.get("contract_leg") or {}).get("inert_check") or {})
    preds = inert.get("predictors_seen")
    arms = inert.get("arms_seen") or []
    n_pred = len(preds) if isinstance(preds, list) else None
    run = doc.get("arm_runnability") or {}
    blocked_no_predictor = sorted(
        a for a, v in run.items()
        if isinstance(v, str) and "NO_RELEASED_PREDICTOR" in v)

    evidence: list[str] = []
    if n_pred is not None and n_pred <= 1 and len(arms) > 1:
        evidence.append(
            f"{len(arms)} arms submitted but {n_pred} distinct predictor(s) "
            f"seen ({preds}) -- arms that differ by predictor cannot differ "
            f"if there is only one")
    if blocked_no_predictor:
        evidence.append(
            f"the runnability table blocks {blocked_no_predictor} for want of "
            f"a released predictor")

    score_verdict = "NOT_SUPPLIED"
    n_reproduced = None
    if scores:
        salt = (stub_block or {}).get("salt", "")
        n_reproduced = sum(1 for s in scores
                           if _hash_score_matches(s, salt))
        if n_reproduced == len(scores):
            score_verdict = "EVERY_SCORE_REPRODUCED_FROM_IDENTIFIERS"
            evidence.append(
                f"all {len(scores)} scores reproduce from a hash of "
                f"(slug, side, gen) -- the arm did not read the market")
        elif n_reproduced == 0:
            score_verdict = "NO_SCORE_REPRODUCED_FROM_IDENTIFIERS"
        else:
            score_verdict = "MIXED"

    if stub_declared and evidence:
        verdict = "STUB_DECLARED_AND_EVIDENCED"
    elif stub_declared:
        verdict = "STUB_DECLARED_UNEVIDENCED"
    elif evidence:
        verdict = "REAL_CLAIMED_BUT_EVIDENCE_SAYS_STUB"
    elif score_verdict == "NO_SCORE_REPRODUCED_FROM_IDENTIFIERS":
        verdict = "REAL_EVIDENCED"
    else:
        # THE DEFAULT IS NOT "REAL". An arm that declares nothing and supplies
        # nothing has not shown it ran.
        verdict = "UNVERIFIABLE_NO_EVIDENCE_EITHER_WAY"

    return {
        "verdict": verdict,
        "stub_declared_by_producer": stub_declared,
        "stub_declaration": stub_block,
        "n_arms_submitted": len(arms),
        "n_distinct_predictors": n_pred,
        "predictors_seen": preds,
        "arms_blocked_for_no_released_predictor": blocked_no_predictor,
        "score_test": score_verdict,
        "n_scores_reproduced_from_identifiers": n_reproduced,
        "evidence": evidence,
        "note": ("the producer's declaration is READ and never trusted "
                 "alone; `evidence` is what this module computed. A REAL "
                 "claim with no evidence reads UNVERIFIABLE, never real"),
    }


def _hash_score_matches(s: dict, salt: str) -> bool:
    """Is this score reproducible from its identifiers alone?

    The form is the one this programme's stubs use: a sha256 over
    `salt|slug|side|gen` mapped into [0, 1). Implemented HERE rather than
    imported -- if it came from the producer it would agree by construction.
    """
    try:
        raw = f"{salt}|{s['slug']}|{s['side']}|{int(s['gen'])}"
        h = hashlib.sha256(raw.encode()).digest()
        want = int.from_bytes(h[:8], "big") / float(1 << 64)
        return abs(float(s["score"]) - want) < 1e-12
    except Exception:
        return False


# ------------------------------------------------------------ question 2
def completeness(doc: dict[str, Any]) -> dict[str, Any]:
    """Complete, or partial and reported as complete? Every count reconciles.

    TWO DIFFERENT FAILURES ARE SEPARATED HERE. A histogram that does not sum
    to its population is an arithmetic contradiction. A GATE that reports
    zero failures without reporting how many things it checked is the
    EMPTY-SET TRAP -- "0 failing" and "0 checked" are the same bytes, and a
    green board built from them says nothing.
    """
    findings: list[dict] = []
    pop = doc.get("population") or {}
    wsc = doc.get("window_status_counts") or {}
    gsc = doc.get("generation_status_counts") or {}

    n_sel = pop.get("n_selected_windows")
    w_sum = sum(v for v in wsc.values() if isinstance(v, (int, float)))
    windows_reconcile = (isinstance(n_sel, int) and w_sum == n_sel)
    if not windows_reconcile:
        findings.append({"kind": "WINDOW_COUNTS_DO_NOT_RECONCILE",
                         "n_selected_windows": n_sel,
                         "status_histogram_sum": w_sum})

    g_sum = sum(v for v in gsc.values() if isinstance(v, (int, float)))
    n_adm_head = doc.get("n_admitted_generations")
    n_adm_hist = gsc.get("ADMITTED")
    gens_reconcile = (n_adm_head is not None and n_adm_head == n_adm_hist)
    if not gens_reconcile:
        findings.append({"kind": "ADMITTED_HEADLINE_DISAGREES_WITH_HISTOGRAM",
                         "headline": n_adm_head, "histogram": n_adm_hist})

    # THE GATES. A pass needs a denominator.
    gates = doc.get("gates") or {}
    gates_without_denominator = sorted(
        g for g, v in gates.items()
        if isinstance(v, dict)
        and not any(k for k in v
                    if re.search(r"n_(windows|slugs|cases|checked)"
                                 r"_?(checked|examined|total|seen)?$", k)
                    and k not in ("n_failing_windows",))
    )
    if gates_without_denominator:
        findings.append({
            "kind": "GATE_PASSES_WITH_NO_DENOMINATOR",
            "gates": gates_without_denominator,
            "why": ("each of these reports failures but never how many "
                    "things it checked, so `pass: true` on `0 failing` is "
                    "indistinguishable from a gate that examined nothing "
                    "(rule 11). `all_gates_pass` rests on them")})

    return {
        "windows_reconcile": windows_reconcile,
        "n_selected_windows": n_sel, "window_histogram_sum": w_sum,
        "generation_histogram_sum": g_sum,
        "admitted_headline": n_adm_head, "admitted_in_histogram": n_adm_hist,
        "admitted_reconciles": gens_reconcile,
        "n_gates": len(gates),
        "n_gates_without_denominator": len(gates_without_denominator),
        "gates_without_denominator": gates_without_denominator,
        "all_gates_pass_claimed": doc.get("all_gates_pass"),
        "findings": findings,
        "complete_and_reconciled": not findings,
    }


# ------------------------------------------------------------ question 3
def population_consumed(doc: dict[str, Any],
                        recompute: bool = True) -> dict[str, Any]:
    """Did it consume the population it NAMES? A name is not a consumption.

    The artifact must carry a DIGEST of its input that this module can
    recompute. Absent one the answer is UNVERIFIABLE and says so; it is never
    a pass, because "the producer named a population" and "the producer read
    that population" are different claims and only the second is evidence.
    """
    pop = doc.get("population") or {}
    declared = None
    for k in ("digest", "sha256", "population_digest", "input_digest"):
        if isinstance(pop.get(k), str):
            declared = pop[k]
            break
    days = pop.get("days") or []
    coins = pop.get("coins") or []
    recomputed = None
    if recompute and days and coins:
        recomputed = _tape_digest(days, coins)
    if declared is None:
        return {
            "verdict": "UNVERIFIABLE_NO_DECLARED_DIGEST",
            "declared_digest": None,
            "recomputed_digest": recomputed,
            "population_named": {"days": days, "coins": coins,
                                 "era": pop.get("era"),
                                 "n_selected_windows":
                                     pop.get("n_selected_windows")},
            "why": ("the artifact NAMES a population and carries no digest of "
                    "it, so nothing here can distinguish an arm that read it "
                    "from one that read something else. This is a STATUS, not "
                    "a failure and not a pass"),
            "what_would_close_it": ("a `population.digest` the producer "
                                    "computes over the rows it actually "
                                    "consumed, recomputable from the tape"),
        }
    return {
        "verdict": ("POPULATION_DIGEST_MATCHES" if declared == recomputed
                    else "POPULATION_DIGEST_MISMATCH"),
        "declared_digest": declared, "recomputed_digest": recomputed,
        "population_named": {"days": days, "coins": coins},
    }


def _tape_digest(days: list, coins: list) -> str | None:
    """A digest over the tape the named population would have read.

    Computed from FILE IDENTITY (name + uncompressed size), which is cheap and
    is enough to detect a different population; it is deliberately NOT the
    producer's own digest form, because two implementations agreeing is the
    evidence and one implementation copied twice is not.
    """
    try:
        import pm_tape_density as TD
        h = hashlib.sha256()
        for d in sorted(days):
            tok = d.replace("-", "")
            try:
                agg = TD.scan_day(tok)
            except Exception:
                return None
            for (c, w), b in sorted(agg.items()):
                if coins and c not in coins:
                    continue
                h.update(f"{tok}|{c}|{w}|{b}\n".encode())
        return h.hexdigest()
    except Exception:                                        # pragma: no cover
        return None


# ------------------------------------------- the §8.1 output-field audit
def section_8_1_audit(fields: dict[str, dict] | None = None
                      ) -> dict[str, Any]:
    """Is every §8.1 field genuinely PRODUCED or genuinely NAMED ABSENT?

    `fields` is the producer's enumeration, passed IN as data. This module
    carries its own list of what §8.1 requires (`SECTION_8_1_REQUIRED`),
    written from the plan rather than copied from the producer, so the audit
    compares two enumerations made separately.

    THE QUESTION THAT MATTERS IS THE THIRD ONE: can a field silently report a
    DEFAULT that looks like a measurement? A counter initialised to 0 and
    never written reads exactly like a counter that counted nothing, and that
    is the defect fixed on the blackout mask on 2026-09-04 one layer down.
    """
    if fields is None:
        fields = _producer_fields()
    required = sorted({f for g in SECTION_8_1_REQUIRED.values() for f in g})
    present = sorted(fields)
    missing = [f for f in required if f not in fields]
    extra = [f for f in present if f not in required]

    neither, both, sourced, absent_named = [], [], [], []
    for name, spec in sorted(fields.items()):
        if not isinstance(spec, dict):
            neither.append(name)
            continue
        src = spec.get("source")
        why = spec.get("why")
        if src is None and not why:
            neither.append(name)
        elif src is not None and why:
            both.append(name)
        elif src is None:
            absent_named.append(name)
        else:
            sourced.append(name)

    dupes: dict[str, list] = {}
    for name, spec in fields.items():
        if isinstance(spec, dict) and isinstance(spec.get("source"), str):
            dupes.setdefault(spec["source"], []).append(name)
    shared_sources = {k: sorted(v) for k, v in dupes.items() if len(v) > 1}

    ambiguous_zero = [f for f in COUNTER_FIELDS
                      if f in fields
                      and isinstance(fields[f], dict)
                      and isinstance(fields[f].get("source"), str)
                      and not any(k in fields[f] for k in
                                  ("evaluated_flag", "denominator",
                                   "absent_status"))]
    return {
        "n_required_by_this_module": len(required),
        "n_in_producer_enumeration": len(present),
        "missing_from_producer": missing,
        "extra_beyond_this_module": extra,
        "n_sourced": len(sourced), "sourced": sourced,
        "n_absent_with_a_reason": len(absent_named),
        "absent_with_a_reason": absent_named,
        "entries_with_NEITHER_source_nor_reason": neither,
        "entries_with_BOTH_source_and_reason": both,
        "shared_sources": shared_sources,
        "counters_whose_zero_is_ambiguous": ambiguous_zero,
        "every_entry_is_one_or_the_other": not neither and not both,
        "note": ("a field with neither a source nor a reason is an "
                 "unanswered requirement wearing an answer's shape; a field "
                 "with both is a contradiction; two fields sharing a source "
                 "are one quantity under two names"),
    }


def _producer_fields() -> dict[str, dict]:
    """The producer's enumeration, read as DATA from its source.

    Parsed with `ast`, never imported: importing the producer to audit the
    producer runs its code and couples the two, which is the separation this
    whole module exists to keep.
    """
    import ast
    src = CODE_ROOT / "live/pm_research/de_lane4_real_parity.py"
    if not src.is_file():
        raise ArmVerifyRefused(
            f"REFUSED: no producer source at {src}; an audit that cannot read "
            f"the enumeration must not report one.")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        tgts = (node.targets if isinstance(node, ast.Assign)
                else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in tgts:
            if isinstance(t, ast.Name) and t.id == "SECTION_8_1_FIELDS":
                return ast.literal_eval(node.value)
    raise ArmVerifyRefused(
        "REFUSED: SECTION_8_1_FIELDS not found in the producer source as a "
        "literal. A computed enumeration cannot be audited as data.")


def verify(path: Path, scores: list[dict] | None = None) -> dict[str, Any]:
    doc = load_artifact(path)
    return {
        "instrument": "da_arm_replay_verify",
        "artifact": str(path),
        "artifact_sha256": hashlib.sha256(
            path.read_bytes()).hexdigest()[:16],
        "protocol_claimed": doc.get("protocol"),
        "status_claimed": doc.get("status"),
        "stub_or_real": stub_or_real(doc, scores),
        "completeness": completeness(doc),
        "population_consumed": population_consumed(doc),
        "decides_nothing": ("REPORTED. This says whether the OUTPUT can be "
                            "believed; whether an arm may be used is the "
                            "policy layer's (rule 14)"),
    }


# --------------------------------------------------------------- falsifier
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
        print(f"PASS: {label}")

    def art(tmp: Path, doc: dict) -> Path:
        p = tmp / "a.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    REAL = DERIVED / "de_lane4_real_parity_v1.json"

    # ---- refusals first: nothing may be graded that cannot be read --------
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        try:
            load_artifact(tmp / "nope.json")
            ok(False, "an absent artifact must REFUSE")
        except ArmVerifyRefused as e:
            ok("empty-set trap" in str(e),
               "REFUSAL-1 an ABSENT artifact refuses by name -- 'no file' and "
               "'verified clean' must never be the same answer")
        (tmp / "bad.json").write_text("{not json", encoding="utf-8")
        try:
            load_artifact(tmp / "bad.json")
            ok(False, "unparseable must REFUSE")
        except ArmVerifyRefused as e:
            ok("must not share an answer" in str(e),
               "REFUSAL-2 an UNPARSEABLE artifact refuses -- distinct from a "
               "failed verification")
        (tmp / "list.json").write_text("[1,2]", encoding="utf-8")
        try:
            load_artifact(tmp / "list.json")
            ok(False, "a non-object must REFUSE")
        except ArmVerifyRefused:
            ok(True, "REFUSAL-3 a JSON array is refused: an arm output is an "
                     "object and a list cannot be graded as one")

    # ---- Q1: stub or real, driven to EVERY verdict it can return ---------
    _stub_doc = {"declared_parameters": {"stub": {"salt": "s"}},
                 "contract_leg": {"inert_check": {
                     "predictors_seen": ["none"],
                     "arms_seen": ["A", "B", "C"]}},
                 "arm_runnability": {"C": "NO_RELEASED_PREDICTOR"}}
    _r = stub_or_real(_stub_doc)
    ok(_r["verdict"] == "STUB_DECLARED_AND_EVIDENCED" and len(_r["evidence"]) == 2,
       f"Q1-a a declared stub WITH computed evidence reads "
       f"STUB_DECLARED_AND_EVIDENCED ({len(_r['evidence'])} independent "
       f"signals), not merely 'the producer said so'")
    _d2 = copy.deepcopy(_stub_doc)
    _d2["contract_leg"]["inert_check"]["predictors_seen"] = ["p1", "p2", "p3"]
    _d2["arm_runnability"] = {}
    ok(stub_or_real(_d2)["verdict"] == "STUB_DECLARED_UNEVIDENCED",
       "Q1-b a declared stub with NO corroborating evidence is reported as "
       "unevidenced -- the declaration alone does not close the question in "
       "either direction")
    _d3 = copy.deepcopy(_stub_doc)
    _d3["declared_parameters"] = {}
    ok(stub_or_real(_d3)["verdict"] == "REAL_CLAIMED_BUT_EVIDENCE_SAYS_STUB",
       "Q1-c THE ONE THAT MATTERS: an arm declaring NO stub while one "
       "predictor serves three arms is flagged, not believed. A plausible "
       "number from a path that did not run is what this exists to catch")
    _d4 = {"declared_parameters": {}, "contract_leg": {}, "arm_runnability": {}}
    ok(stub_or_real(_d4)["verdict"] == "UNVERIFIABLE_NO_EVIDENCE_EITHER_WAY",
       "Q1-d AND THE DEFAULT IS NOT 'REAL': an arm that declares nothing and "
       "supplies nothing reads UNVERIFIABLE. Absence of evidence never "
       "becomes evidence of a real run")
    # the score test, both directions, on scores this module hashes itself
    _sc = [{"slug": "btc-1", "side": "up", "gen": i} for i in range(5)]
    for s in _sc:
        s["score"] = int.from_bytes(hashlib.sha256(
            f"s|{s['slug']}|{s['side']}|{s['gen']}".encode()).digest()[:8],
            "big") / float(1 << 64)
    _r5 = stub_or_real(_stub_doc, _sc)
    ok(_r5["score_test"] == "EVERY_SCORE_REPRODUCED_FROM_IDENTIFIERS"
       and _r5["n_scores_reproduced_from_identifiers"] == 5,
       "Q1-e THE STRONG TEST: every score reproduces from a hash of "
       "(slug, side, gen), so the arm did not read the market -- computed "
       "here, not read from a flag")
    _sc2 = [dict(s, score=s["score"] + 0.5) for s in _sc]
    ok(stub_or_real(_stub_doc, _sc2)["score_test"]
       == "NO_SCORE_REPRODUCED_FROM_IDENTIFIERS",
       "Q1-f and scores that do NOT reproduce read as such -- the test "
       "discriminates rather than always crying stub")

    # ---- Q2: complete vs partial-reported-as-complete ---------------------
    _good = {"population": {"n_selected_windows": 10},
             "window_status_counts": {"ADMITTED": 8, "REPLAY_NONE": 2},
             "generation_status_counts": {"ADMITTED": 100, "ZERO_LENGTH": 3},
             "n_admitted_generations": 100,
             "gates": {"g1": {"pass": True, "n_failing_windows": 0,
                              "n_windows_checked": 10}},
             "all_gates_pass": True}
    _c = completeness(_good)
    ok(_c["complete_and_reconciled"] is True and _c["windows_reconcile"]
       and _c["admitted_reconciles"],
       "Q2-a POSITIVE CONTROL: a reconciling artifact whose gates carry a "
       "denominator passes -- the check admits, so its refusals mean "
       "something")
    _bad1 = copy.deepcopy(_good)
    _bad1["window_status_counts"]["ADMITTED"] = 7
    ok(any(f["kind"] == "WINDOW_COUNTS_DO_NOT_RECONCILE"
           for f in completeness(_bad1)["findings"]),
       "Q2-b KNOWN-BAD: a status histogram that does not sum to the "
       "population it claims is an arithmetic contradiction and is named")
    _bad2 = copy.deepcopy(_good)
    _bad2["n_admitted_generations"] = 999
    ok(any(f["kind"] == "ADMITTED_HEADLINE_DISAGREES_WITH_HISTOGRAM"
           for f in completeness(_bad2)["findings"]),
       "Q2-c KNOWN-BAD: a headline count that disagrees with its own "
       "histogram -- the shape that has contradicted a table three times "
       "in this programme")
    _bad3 = copy.deepcopy(_good)
    _bad3["gates"]["g1"].pop("n_windows_checked")
    _f3 = completeness(_bad3)["findings"]
    ok(any(f["kind"] == "GATE_PASSES_WITH_NO_DENOMINATOR" for f in _f3),
       "Q2-d KNOWN-BAD: a gate reporting `0 failing` with no count of what "
       "it checked is the EMPTY-SET TRAP -- '0 failing' and '0 checked' are "
       "the same bytes")

    # ---- Q3: consumed the population it claims ---------------------------
    _p = population_consumed({"population": {"days": [], "coins": []}})
    ok(_p["verdict"] == "UNVERIFIABLE_NO_DECLARED_DIGEST"
       and "not a pass" in _p["why"],
       "Q3-a a NAMED population with no digest reads UNVERIFIABLE and says "
       "in the artifact that it is not a pass")
    _p2 = population_consumed(
        {"population": {"days": [], "coins": [], "digest": "abc"}},
        recompute=False)
    ok(_p2["verdict"] == "POPULATION_DIGEST_MISMATCH",
       "Q3-b a declared digest that does not match what this module "
       "recomputes is a MISMATCH, not an absence")

    # ---- the §8.1 audit, against the PRODUCER'S OWN enumeration ----------
    _a = section_8_1_audit()
    ok(_a["n_in_producer_enumeration"] >= _a["n_required_by_this_module"]
       and _a["missing_from_producer"] == [],
       f"8.1-a INDEPENDENT ENUMERATIONS AGREE: every one of the "
       f"{_a['n_required_by_this_module']} fields this module derives from "
       f"the plan is present in the producer's list "
       f"({_a['n_in_producer_enumeration']} entries), missing "
       f"{_a['missing_from_producer']}")
    ok(_a["every_entry_is_one_or_the_other"] is True
       and _a["entries_with_NEITHER_source_nor_reason"] == []
       and _a["entries_with_BOTH_source_and_reason"] == [],
       "8.1-b EVERY ENTRY IS EITHER PRODUCED OR NAMED ABSENT: none has "
       "neither (an unanswered requirement wearing an answer's shape) and "
       "none has both (a contradiction)")
    ok(_a["shared_sources"] == {},
       f"8.1-c no two fields name the SAME producer path "
       f"({_a['shared_sources']}) -- two names for one quantity would make a "
       f"table look twice as measured as it is")
    ok(len(_a["counters_whose_zero_is_ambiguous"]) == len(
        [f for f in COUNTER_FIELDS if f in _producer_fields()]),
       f"8.1-d THE FINDING: every counter in the enumeration "
       f"({_a['counters_whose_zero_is_ambiguous']}) carries a source and NO "
       f"companion evaluated-flag or denominator, so a ZERO from a path that "
       f"never ran is indistinguishable from a counted zero. This is the "
       f"class fixed on the blackout mask on 2026-09-04, one layer up")
    _synth = {"x": {"source": "a.b"}, "y": {"source": "a.b"},
              "z": {}, "w": {"source": "c.d", "why": "and also absent"}}
    _sa = section_8_1_audit(_synth)
    ok(_sa["shared_sources"] == {"a.b": ["x", "y"]}
       and _sa["entries_with_NEITHER_source_nor_reason"] == ["z"]
       and _sa["entries_with_BOTH_source_and_reason"] == ["w"]
       and _sa["every_entry_is_one_or_the_other"] is False,
       "8.1-e FALSIFIER: on a planted enumeration the audit names the shared "
       "source, the entry with neither, and the entry with both -- so its "
       "clean answer on the real one is a measurement")
    try:
        section_8_1_audit(_producer_fields())
        ok(True, "8.1-f the producer's enumeration is read as DATA via `ast`, "
                 "never imported -- auditing by importing runs the code under "
                 "audit and couples the two")
    except ArmVerifyRefused:
        ok(False, "8.1-f the producer enumeration must be readable")

    # ---- END TO END ON THE REAL ARTIFACT ---------------------------------
    ok(REAL.is_file(),
       f"REAL-0 PRECONDITION: the produced artifact exists at {REAL}. This "
       f"suite ships no skip for the real-artifact leg -- a named SKIP "
       f"standing in for a positive control was ruled out in this lane")
    _v = verify(REAL)
    ok(_v["stub_or_real"]["verdict"] == "STUB_DECLARED_AND_EVIDENCED"
       and _v["stub_or_real"]["n_distinct_predictors"] == 1
       and _v["stub_or_real"]["n_arms_submitted"] == 7,
       f"REAL-1 the produced artifact is a STUB run and this module says so "
       f"from its own evidence: 7 arms submitted, "
       f"{_v['stub_or_real']['n_distinct_predictors']} distinct predictor "
       f"({_v['stub_or_real']['predictors_seen']}). Its own "
       f"`declared_parameters.stub` agrees, and agreement of two independent "
       f"readings is the point")
    _cc = _v["completeness"]
    ok(_cc["windows_reconcile"] is True and _cc["admitted_reconciles"] is True,
       f"REAL-2 the produced artifact's counts RECONCILE: "
       f"{_cc['window_histogram_sum']} windows against "
       f"{_cc['n_selected_windows']} selected, and admitted "
       f"{_cc['admitted_headline']} == {_cc['admitted_in_histogram']}")
    ok(_cc["n_gates_without_denominator"] == 9 and len(_cc["findings"]) == 1,
       f"REAL-3 AND THE FINDING: all "
       f"{_cc['n_gates_without_denominator']} of its gates report failures "
       f"with NO denominator, so `all_gates_pass: "
       f"{_cc['all_gates_pass_claimed']}` rests on nine counts that cannot "
       f"distinguish '0 failing' from '0 checked'")
    ok(_v["population_consumed"]["verdict"]
       == "UNVERIFIABLE_NO_DECLARED_DIGEST",
       "REAL-4 and whether it consumed the population it names is "
       "UNVERIFIABLE -- the artifact names days, coins and an era, and "
       "carries no digest of what was read")

    print(f"\nda_arm_replay_verify selftest: {checks} checks PASSED")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: EXPECTED_CHECKS={EXPECTED_CHECKS} but {checks} ran.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    p = Path(a.artifact) if a.artifact else (
        DERIVED / "de_lane4_real_parity_v1.json")
    try:
        print(json.dumps(verify(p), indent=1, sort_keys=True))
    except ArmVerifyRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
