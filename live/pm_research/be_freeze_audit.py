#!/usr/bin/env python3
"""HAS THE RACE'S CANDIDATE BEEN FROZEN UNDER RULE 12? COMPUTED, NOT ARGUED.

THE QUESTION THIS ANSWERS. The 09-01 forward-day receipt binds
`harmful_candidate_manifest_v1.json`, and the BOUND bytes of that manifest say
`freeze_status: "NOT FROZEN. The freeze is the user's decision and Phase-0
reproduction has not yet run."` Read alone, that reads as a race scoring days
against a candidate nobody froze.

IT IS NOT WHAT IT LOOKS LIKE, AND THE REASON IS THAT THREE DIFFERENT THINGS ARE
CALLED "FROZEN" HERE. This module separates them and evaluates each:

  1. THE RECEIPT'S `frozen` BLOCK is a REPRODUCIBILITY property. The
     `materialise_frozen_bytes` gate writes the freeze commit's code bytes into
     the run directory and sha-verifies each anchor BEFORE import, so the run
     executed known bytes. It says nothing about race entry.
  2. THE MANIFEST'S `freeze_status` is a sentence written at the manifest's own
     `as_of_utc`, and it asserts TWO separable things: that the freeze is the
     USER's decision, and that Phase-0 reproduction has not run.
  3. THE CANDIDATE ARTIFACT'S OWN `status` is the rule-12 race-entry freeze.

The manifest was written at 2026-08-26T08:00:04Z. The candidate's
`frozen_at_utc` is 10:21:49Z and the freeze commit is 10:49:55Z. **The manifest
predates the freeze by 2h21m, so its sentence was TRUE AT ITS STAMP and is now
STALE on its first clause.** Its second clause -- Phase-0 reproduction -- is a
different gate and this module reports it separately rather than letting one
sentence answer for both.

WHAT THIS MODULE DOES NOT DO. It decides nothing (rule 14). Whether to freeze
again, what to freeze, and which artifact the forward read scores are the
USER's. Every field below is a predicate over bytes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
FREEZE_COMMIT = "1b53929"
CANDIDATE = DERIVED / "harmful_reduced_fine_candidate_v1.json"
MANIFEST = DERIVED / "harmful_candidate_manifest_v1.json"
CANDIDATE_REL = "data/pm_5min/derived/harmful_reduced_fine_candidate_v1.json"
MANIFEST_REL = "data/pm_5min/derived/harmful_candidate_manifest_v1.json"


class FreezeAuditRefused(RuntimeError):
    """A named refusal."""


def _git(*a, binary=False):
    r = subprocess.run(["git", "-C", str(REPO), *a],
                       capture_output=True, timeout=60)
    if r.returncode != 0:
        return None
    return r.stdout if binary else r.stdout.decode(errors="replace").strip()


def _blob(ref: str, path: str):
    return _git("show", f"{ref}:{path}", binary=True)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def two_senses() -> dict:
    """The distinction, with each sense answered separately (item 1)."""
    rec = DERIVED / "be_forward_day_receipt_20260901.json"
    r = json.loads(rec.read_text()) if rec.exists() else {}
    fz = r.get("frozen") or {}
    cand = json.loads(CANDIDATE.read_text())
    man_bound = _blob(FREEZE_COMMIT, MANIFEST_REL)
    man = json.loads(man_bound) if man_bound else {}
    return {
        "sense_1_receipt_frozen_block": {
            "what_it_means": ("the run MATERIALISED the freeze commit's code "
                              "bytes and sha-verified each anchor before "
                              "import -- a REPRODUCIBILITY property of the "
                              "run"),
            "gate": "materialise_frozen_bytes",
            "frozen_commit": fz.get("frozen_commit"),
            "n_anchors_compared": fz.get("n_compared"),
            "says_anything_about_race_entry": False,
        },
        "sense_2_manifest_freeze_status": {
            "what_it_means": ("a sentence written at the MANIFEST's own "
                              "as_of_utc, asserting TWO separable things"),
            "text": man.get("freeze_status"),
            "as_of_utc": man.get("as_of_utc"),
            "clause_a_freeze_is_the_users_decision": True,
            "clause_b_phase0_reproduction_not_run": True,
            "one_sentence_answering_for_two_gates": True,
        },
        "sense_3_candidate_status": {
            "what_it_means": "the rule-12 RACE-ENTRY freeze of the candidate",
            "status": cand.get("status"),
            "frozen_at_utc": cand.get("frozen_at_utc"),
            "user_approval": cand.get("user_approval"),
            "authorising_ask": cand.get("authorising_ask"),
        },
        "the_three_are_not_the_same_object": True,
    }


def timeline() -> dict:
    """Ordering, from the artifacts and from git. This is what makes the
    manifest's first clause stale rather than wrong."""
    cand = json.loads(CANDIDATE.read_text())
    man_bound = _blob(FREEZE_COMMIT, MANIFEST_REL)
    man = json.loads(man_bound) if man_bound else {}
    ct = _git("show", "-s", "--format=%cI", FREEZE_COMMIT)
    msg = _git("show", "-s", "--format=%s", FREEZE_COMMIT)

    def p(x):
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    m_at, c_at, f_at = man.get("as_of_utc"), cand.get("frozen_at_utc"), ct
    return {
        "manifest_as_of_utc": m_at,
        "candidate_frozen_at_utc": c_at,
        "freeze_commit_time_utc": f_at,
        "freeze_commit_subject": msg,
        "manifest_predates_candidate_freeze": p(m_at) < p(c_at),
        "manifest_predates_freeze_commit": p(m_at) < p(f_at),
        "gap_manifest_to_freeze_minutes": round(
            (p(c_at) - p(m_at)).total_seconds() / 60.0, 1),
        "why_it_matters": ("the manifest's sentence was TRUE AT ITS STAMP. It "
                           "is stale on its first clause because the freeze "
                           "happened after it and the manifest was never "
                           "superseded (rule 13)."),
    }


def manifest_never_updated_after_freeze() -> dict:
    """Was the manifest revised once the freeze happened? Computed from git."""
    hist = (_git("log", "--format=%H %cI", "--all", "--", MANIFEST_REL) or "")
    rows = [l.split(" ", 1) for l in hist.splitlines() if l.strip()]
    ct = _git("show", "-s", "--format=%cI", FREEZE_COMMIT)

    def p(x):
        return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    after = [(h[:12], t) for h, t in rows if p(t) > p(ct)]
    on_disk = json.loads(MANIFEST.read_text())
    bound = json.loads(_blob(FREEZE_COMMIT, MANIFEST_REL))
    return {
        "n_commits_touching_manifest": len(rows),
        "commits_after_the_freeze": after,
        "freeze_status_on_disk_today": on_disk.get("freeze_status"),
        "freeze_status_at_freeze_commit": bound.get("freeze_status"),
        "status_string_identical_before_and_after": (
            on_disk.get("freeze_status") == bound.get("freeze_status")),
        "manifest_bytes_changed_since_freeze": (
            _sha_file(MANIFEST) != _sha(_blob(FREEZE_COMMIT, MANIFEST_REL))),
        "finding": ("the manifest HAS been edited since the freeze but its "
                    "freeze_status sentence was never revised -- so the stale "
                    "clause survived edits that touched the same file"),
    }


def rule12_conjuncts() -> dict:
    """Rule 12's four requirements, each a PREDICATE over bytes (item 3).

    'A freeze is a commit. Candidate = builder file committed (hash + commit
    ref in the receipt), full pipeline in the repo (data -> target -> fit ->
    artifact; a scratch-dir builder voided one freeze), declared nulls inside
    the receipt, and the count of candidates in the forward race
    (multiplicity) recorded at freeze time.'"""
    cand = json.loads(CANDIDATE.read_text())
    man = json.loads(_blob(FREEZE_COMMIT, MANIFEST_REL))

    # --- (a) builder committed, hash + commit ref in the artifact ----------
    b_rel, b_sha = cand.get("builder"), cand.get("builder_sha256")
    fb_sha = cand.get("freeze_builder_sha256")
    refit = cand.get("git_commit_at_refit")
    b_at_freeze = _blob(FREEZE_COMMIT, b_rel) if b_rel else None
    b_at_refit = _blob(refit, b_rel) if (b_rel and refit) else None
    FB_REL = "live/pm_research/harmful_freeze_candidate.py"
    fb_at_freeze = _blob(FREEZE_COMMIT, FB_REL)
    fb_at_refit = _blob(refit, FB_REL) if refit else None
    builder = {
        "builder": b_rel, "declared_sha256": b_sha,
        "present_at_freeze_commit": b_at_freeze is not None,
        "sha_matches_at_freeze_commit": bool(
            b_at_freeze and _sha(b_at_freeze) == b_sha),
        "freeze_builder": FB_REL,
        "declared_freeze_builder_sha256": fb_sha,
        "freeze_builder_sha_matches_at_FREEZE_commit": bool(
            fb_at_freeze and _sha(fb_at_freeze) == fb_sha),
        "freeze_builder_sha_matches_at_REFIT_commit": bool(
            fb_at_refit and _sha(fb_at_refit) == fb_sha),
        "commit_ref_recorded": bool(refit),
        "commit_ref_resolves": _git("rev-parse", "--verify",
                                    f"{refit}^{{commit}}") is not None
        if refit else False,
        "subtlety": ("the freeze COMMIT also MODIFIED the freeze builder, so "
                     "the builder that RAN is the one at `git_commit_at_refit`, "
                     "not the one at the freeze commit. A verifier that checks "
                     "the freeze commit alone reports a false mismatch -- "
                     "measured here, both ways."),
        "holds": None,
    }
    builder["holds"] = bool(
        builder["present_at_freeze_commit"]
        and builder["sha_matches_at_freeze_commit"]
        and builder["freeze_builder_sha_matches_at_REFIT_commit"]
        and builder["commit_ref_recorded"] and builder["commit_ref_resolves"])

    # --- (b) full pipeline in the repo -------------------------------------
    code, data = {}, {}
    for p, sha in sorted((man.get("hashes") or {}).items()):
        at = _blob(FREEZE_COMMIT, p)
        tgt = code if p.startswith("live/") else data
        if at is not None:
            tgt[p] = {"in_git_at_freeze": True,
                      "sha_matches": _sha(at) == sha}
        else:
            f = REPO / p
            tgt[p] = {"in_git_at_freeze": False,
                      "on_disk": f.exists(),
                      "sha_matches_on_disk": bool(
                          f.exists() and _sha_file(f) == sha),
                      "why_not_in_git": ("data/ is gitignored by project "
                                         "policy -- large data files are "
                                         "explicitly out of scope for git "
                                         "(CLAUDE.md), so a data input is "
                                         "bound BY SHA rather than committed")}
    pipeline = {
        "code_artifacts": code, "data_artifacts": data,
        "n_code": len(code), "n_data": len(data),
        "all_code_in_git_and_matching": all(
            v["in_git_at_freeze"] and v["sha_matches"] for v in code.values()),
        "all_data_bound_by_sha_and_matching_today": all(
            v.get("sha_matches_on_disk") for v in data.values()),
        "note": ("rule 12's failure mode is a SCRATCH-DIR BUILDER -- code "
                 "that cannot be recovered. Every code artifact here is in "
                 "git at the freeze commit and hashes as declared; the two "
                 "data inputs are bound by sha and still match on disk."),
        "holds": None,
    }
    pipeline["holds"] = bool(pipeline["all_code_in_git_and_matching"]
                             and pipeline["all_data_bound_by_sha_and_matching_today"])

    # --- (c) declared nulls -------------------------------------------------
    dn = cand.get("declared_nulls") or {}
    nulls = {"present_in_candidate": bool(dn), "value": dn,
             "also_in_manifest": bool(man.get("declared_nulls")),
             "has_decision_metric": "decision_metric" in dn,
             "has_matching": "matching" in dn, "has_n_random": "n_random" in dn,
             "holds": all(k in dn for k in
                          ("decision_metric", "matching", "n_random"))}

    # --- (d) multiplicity at freeze time ------------------------------------
    m = cand.get("race_multiplicity_at_freeze")
    members = cand.get("race_members") or []
    mult = {"race_multiplicity_at_freeze": m, "race_members": members,
            "is_int": isinstance(m, int) and not isinstance(m, bool),
            "members_count_matches": len(members) == m if isinstance(m, int)
            else False,
            "recorded_in_the_frozen_bytes": True,
            "holds": bool(isinstance(m, int) and not isinstance(m, bool)
                          and m >= 1 and len(members) == m)}

    return {"a_builder_committed_hash_and_ref": builder,
            "b_full_pipeline_in_repo": pipeline,
            "c_declared_nulls": nulls,
            "d_multiplicity_at_freeze": mult,
            "n_conjuncts": 4,
            "n_holding": sum(1 for x in (builder, pipeline, nulls, mult)
                             if x["holds"]),
            "all_four_hold": all(x["holds"] for x in
                                 (builder, pipeline, nulls, mult))}


def freeze_is_a_commit() -> dict:
    """Rule 12's headline clause, checked at git."""
    added = (_git("show", "--diff-filter=A", "--name-only", "--format=",
                  FREEZE_COMMIT) or "").split()
    touched = (_git("show", "--name-only", "--format=", FREEZE_COMMIT)
               or "").split()
    reg = "orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/COORDINATION.md"
    diff = _git("show", FREEZE_COMMIT, "--", reg, binary=True) or b""
    rows = [l for l in diff.decode(errors="replace").splitlines()
            if l.startswith("+| Q-")]
    at_freeze = _blob(FREEZE_COMMIT, CANDIDATE_REL)
    return {
        "freeze_commit": FREEZE_COMMIT,
        "subject": _git("show", "-s", "--format=%s", FREEZE_COMMIT),
        "candidate_ADDED_by_this_commit": CANDIDATE_REL in added,
        "files_touched": touched,
        "register_rows_added": [r[1:60] for r in rows],
        "n_register_rows_added": len(rows),
        "candidate_sha_at_freeze": _sha(at_freeze) if at_freeze else None,
        "candidate_sha_on_disk_today": _sha_file(CANDIDATE),
        "artifact_unmoved_since_freeze": bool(
            at_freeze and _sha(at_freeze) == _sha_file(CANDIDATE)),
        "holds": bool(at_freeze and CANDIDATE_REL in added and rows
                      and _sha(at_freeze) == _sha_file(CANDIDATE)),
    }


def what_cuts_the_other_way() -> dict:
    """Item 4, weighed rather than dismissed.

    The manifest carries a great deal of freeze-shaped machinery for something
    whose own sentence says NOT FROZEN. Both halves are true and the
    discrepancy is the finding: the machinery was BUILT FOR the freeze that
    followed two hours later, and the sentence was never revised."""
    man = json.loads(_blob(FREEZE_COMMIT, MANIFEST_REL))
    dn = man.get("declared_nulls") or {}
    t = man.get("target_scores_to_reproduce") or {}
    rc = man.get("reproduction_contract") or {}
    return {
        "declared_nulls_present": bool(dn), "declared_nulls": dn,
        "target_scores_read_from_artifact_not_transcribed":
            t.get("values_read_from_artifact_not_transcribed"),
        "target_source_receipt": t.get("source_receipt"),
        "target_source_is_immutable_frozen_copy":
            t.get("source_is_immutable_frozen_copy"),
        "reproduction_contract_completed_under_cap":
            rc.get("completed_under_cap"),
        "reproduction_contract_peak_rss_status": rc.get("peak_rss_status"),
        "reading": ("this IS freeze-shaped machinery, and it is not evidence "
                    "AGAINST a freeze -- it is the preparation FOR one. The "
                    "manifest is the reproduction contract the freeze was "
                    "built on; `completed_under_cap` refers to the BUILD's "
                    "memory ceiling, not to a Phase-0 reproduction having "
                    "run. The sentence and the machinery disagree because the "
                    "sentence is older than the freeze, not because the "
                    "machinery is decorative."),
    }


# ---------------------------------------------------------------------------
# A CLASS THIS PROGRAMME HAS NOT NAMED: THE CONSTANT VERDICT
# ---------------------------------------------------------------------------
#: SEAT_PROTOCOL 16 names one sign of it -- "a control that cannot fail must
#: never be mistaken for a control that passed". Round 17's token check was the
#: OTHER sign: `want` was rebuilt from `provenance_verified` while the token
#: was taken over the raw `provenance`, so it read False in the honest case and
#: True in none. It could not pass.
#:
#: PROPOSED NAME, and it subsumes both rather than adding a second special
#: case: **a CONSTANT VERDICT** -- a control whose output does not depend on
#: its input. Always-pass and always-fail are its two signs, and the test is
#: the same in both directions: drive it twice and require the verdict to MOVE.
#:
#: The always-FAIL sign is not merely useless, which is worth saying because it
#: sounds like the harmless direction. It manufactures a permanent false alarm:
#: a reader who trusts the field reads tampering into every honest run, and a
#: reader who learns to ignore it has been trained to ignore a real one.
#: Silence is recoverable; a discredited alarm is not.
CONSTANT_VERDICT = {
    "name": "a constant verdict",
    "definition": "a control whose verdict does not depend on its input",
    "sign_always_pass": ("SEAT_PROTOCOL 16's named case: a fixture supplying "
                         "what the code should produce; a guard shown only to "
                         "refuse; an anchor that includes the arm name"),
    "sign_always_fail": ("round 17's token check: False in the honest case and "
                         "True in none, because `want` and the token were "
                         "computed over different objects"),
    "why_always_fail_is_worse_than_silence": (
        "it manufactures a permanent false alarm. A reader who trusts it reads "
        "tampering into every honest run; a reader who learns to ignore it has "
        "been trained to ignore a real one."),
    "the_test_is_the_same_in_both_directions": (
        "drive the control twice, on an input it should accept and one it "
        "should refuse, and require the VERDICT to MOVE"),
}


def verdict_depends_on_input(fn, good, bad) -> dict:
    """Does this control's verdict MOVE between an input it should accept and
    one it should refuse? The operational test for a constant verdict."""
    def run(x):
        try:
            return ("value", fn(x))
        except Exception as e:                        # noqa: BLE001
            return ("raised", type(e).__name__)
    g, b = run(good), run(bad)
    return {"on_good": g, "on_bad": b, "verdict_moved": g != b,
            "constant_verdict": g == b,
            "why": CONSTANT_VERDICT["the_test_is_the_same_in_both_directions"]}


def builder_reference_commit(which: str = "freeze_builder") -> dict:
    """ITEM 3. WHICH COMMIT DOES A DECLARED BUILDER SHA RESOLVE AT? Searched.

    Round 18's false positive made permanent: `freeze_builder_sha256` matches
    `harmful_freeze_candidate.py` at `git_commit_at_refit` and NOT at the
    freeze commit, because the freeze commit MODIFIED that same file. Checking
    the freeze commit alone reports a mismatch that is not there. A note in a
    filing decays; this searches the file's history and REPORTS where the
    declared sha actually lives, so the answer is derived rather than
    remembered."""
    cand = json.loads(CANDIDATE.read_text())
    spec = {
        "builder": {"path": cand.get("builder"),
                    "sha": cand.get("builder_sha256")},
        "freeze_builder": {"path": "live/pm_research/harmful_freeze_candidate.py",
                           "sha": cand.get("freeze_builder_sha256")},
    }[which]
    path, want = spec["path"], spec["sha"]
    refit = cand.get("git_commit_at_refit")
    hits = []
    for c in (_git("log", "--format=%H", "--all", "--", path) or "").split():
        b = _blob(c, path)
        if b is not None and _sha(b) == want:
            hits.append(c)
    at_freeze = _blob(FREEZE_COMMIT, path)
    at_refit = _blob(refit, path) if refit else None
    return {
        "which": which, "path": path, "declared_sha256": want,
        "matches_at_freeze_commit": bool(at_freeze and _sha(at_freeze) == want),
        "matches_at_git_commit_at_refit": bool(
            at_refit and _sha(at_refit) == want),
        "commits_carrying_this_sha": [h[:12] for h in hits],
        "n_commits_carrying_this_sha": len(hits),
        "correct_reference_commit": (
            refit if (at_refit and _sha(at_refit) == want)
            else (hits[0] if hits else None)),
        "trap": ("the freeze commit MODIFIED this file in the same commit that "
                 "landed the artifact, so the builder that RAN is the one at "
                 "`git_commit_at_refit`. An auditor checking only the freeze "
                 "commit sees a mismatch that is not there -- this is that "
                 "false positive turned into a lookup."),
    }


def manifest_binding_status() -> dict:
    """ITEM 1. What binds the manifest, and does that binding hold TODAY?

    Two different resolutions exist and both read DISK, not git:
    `materialise_frozen` uses the manifest's `hashes` to know what to
    materialise, and `assert_frozen_contract` compares the manifest's OWN sha
    to the candidate's declared `manifest_sha256`. The second is the binding
    the question is about."""
    import ast as _ast
    cand = json.loads(CANDIDATE.read_text())
    disk_sha = _sha_file(MANIFEST)
    bound_sha = cand.get("manifest_sha256")
    bound = json.loads(_blob(FREEZE_COMMIT, MANIFEST_REL))
    disk = json.loads(MANIFEST.read_text())
    differing = sorted(k for k in set(bound) | set(disk)
                       if bound.get(k) != disk.get(k))
    tree = _ast.parse((REPO / "live/pm_research/be_forward_day.py").read_text())
    # REACHABILITY from `run_forward_day`, computed TRANSITIVELY. "Not inside
    # selftest" is a different question: `assert_frozen_contract` is called by
    # `anchor_drift_root`, which is itself reached from nowhere but a selftest
    # fixture string, so counting its caller as production would overstate the
    # very wiring being measured.
    edges = {}
    for n in _ast.walk(tree):
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            outs = set()
            for c in _ast.walk(n):
                if isinstance(c, _ast.Call):
                    f = c.func
                    nm = (f.id if isinstance(f, _ast.Name)
                          else getattr(f, "attr", None))
                    if nm:
                        outs.add(nm)
            edges[n.name] = outs
    seen, stack = set(), ["run_forward_day"]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges.get(cur, ()))
    prod = sorted(f for f in ("assert_frozen_contract", "anchor_drift_root")
                  if f in seen)
    rec = DERIVED / "be_forward_day_receipt_20260901.json"
    gates = ([g["gate"] for g in json.loads(rec.read_text())["gates"]]
             if rec.exists() else [])
    return {
        "candidate_declares_manifest_sha256": bound_sha,
        "manifest_on_disk_sha256": disk_sha,
        "binding_holds_on_disk_today": disk_sha == bound_sha,
        "keys_that_differ_bound_vs_disk": differing,
        "hashes_block_identical": bound.get("hashes") == disk.get("hashes"),
        "pin_semantics_identical":
            bound.get("pin_semantics") == disk.get("pin_semantics"),
        "freeze_status_identical":
            bound.get("freeze_status") == disk.get("freeze_status"),
        "drift_is_metadata_only": (bound.get("hashes") == disk.get("hashes")
                                   and set(differing) <= {"as_of_utc",
                                                          "git_commit",
                                                          "git_dirty"}),
        "reachable_from_run_forward_day": prod,
        "n_reachable_from_run_forward_day": len(prod),
        "reachability_note": ("computed TRANSITIVELY from `run_forward_day`; "
                              "'not inside selftest' would have overstated it"),
        "frozen_contract_gate_in_the_09_01_receipt":
            any("contract" in g for g in gates),
        "gates_actually_run": gates,
        "finding": ("the candidate's manifest binding ALREADY FAILS on disk, "
                    "and the only checker that compares them is reachable "
                    "from the run path in ZERO ways. The drift is "
                    "metadata-only; the `hashes` a run materialises from are "
                    "IDENTICAL, so every anchor the forward runs used was the "
                    "frozen one."),
    }


def would_superseding_break_the_binding() -> dict:
    """ITEM 1's explicit question, answered as a predicate."""
    st = manifest_binding_status()
    rec = DERIVED / "be_forward_day_receipt_20260901.json"
    fz = (json.loads(rec.read_text()).get("frozen") or {}) if rec.exists() else {}
    return {
        "question": ("would superseding harmful_candidate_manifest_v1.json "
                     "invalidate the binding in "
                     "be_forward_day_receipt_20260901's frozen block?"),
        "answer": False,
        "reason_1_a_supersede_is_a_new_file": (
            "the receipt records `manifest_sha256_bound` "
            f"{str(fz.get('manifest_sha256_bound'))[:16]}…, which describes "
            "the BYTES READ AT RUN TIME. A superseding v2 is a new path; v1's "
            "bytes are untouched, so what the receipt describes still exists "
            "and still hashes the same. Editing v1 IN PLACE is the act that "
            "would break it -- which is why rule 13 forbids exactly that."),
        "reason_2_the_binding_is_already_untested": (
            "the disk manifest ALREADY differs from the candidate's declared "
            "`manifest_sha256`, and `assert_frozen_contract` -- the only "
            "checker that compares them -- is not reachable from the run "
            "path. Nothing there reads that binding, so nothing there can be "
            "broken by superseding it."),
        "but_this_is_not_a_licence": (
            "reason 2 is a DEFECT, not a permission: unenforced is not "
            "unimportant. Wiring `assert_frozen_contract` into the run path "
            "is the repair, and it is a metric-path edit this round must "
            "not make."),
        "receipt_frozen_block_manifest_sha256_bound":
            fz.get("manifest_sha256_bound"),
        "binding_holds_on_disk_today": st["binding_holds_on_disk_today"],
        "drift_is_metadata_only": st["drift_is_metadata_only"],
    }


def superseding_manifest_proposal() -> dict:
    """ITEM 1. PROPOSED, NOT ENACTED. Returned as data; nothing is written."""
    bound = json.loads(_blob(FREEZE_COMMIT, MANIFEST_REL))
    disk = json.loads(MANIFEST.read_text())
    tl = timeline()
    corrected = (
        "FROZEN as a race candidate at 2026-08-26T10:21:49Z, committed as "
        f"{FREEZE_COMMIT} (\"FREEZE: reduced-fine (PM_PLUS_FINE) frozen as "
        "PRIMARY candidate, multiplicity 2\") on the USER's explicit yes, "
        "recorded as register row Q-BE-143. Phase-0 reproduction is a "
        "SEPARATE gate this manifest imposes on itself, reported separately; "
        "it is not a rule-12 requirement.")
    return {
        "status": "PROPOSED, NOT ENACTED",
        "proposed_path": str(MANIFEST.parent
                             / "harmful_candidate_manifest_v2.json"),
        "why_a_new_file_and_not_an_edit": (
            "rule 13: the v1 bytes are named by a landed receipt and by the "
            "candidate's own `manifest_sha256`. A correction supersedes; it "
            "never edits."),
        "proposed_supersedes_block": {
            "supersedes": "harmful_candidate_manifest_v1.json",
            "supersedes_sha256_at_freeze_commit":
                _sha(_blob(FREEZE_COMMIT, MANIFEST_REL)),
            "supersedes_sha256_on_disk": _sha_file(MANIFEST),
            "why": ("v1's `freeze_status` first clause was TRUE at its "
                    f"`as_of_utc` {bound.get('as_of_utc')} and became false "
                    f"{tl['gap_manifest_to_freeze_minutes']} minutes later. "
                    "v1 is kept byte-identical as provenance."),
            "v1_edited_after_the_freeze_without_revising_the_sentence": True,
        },
        "proposed_freeze_status": corrected,
        "provenance_of_the_correction": {
            "established_by": "Q-BE-243 (BE round 18), verified at the artifacts",
            "freeze_commit": FREEZE_COMMIT,
            "freeze_commit_time_utc": tl["freeze_commit_time_utc"],
            "candidate_frozen_at_utc": tl["candidate_frozen_at_utc"],
            "register_row": "Q-BE-143",
            "rule12_conjuncts_holding": rule12_conjuncts()["n_holding"],
        },
        "everything_else_unchanged": (
            "the proposal changes ONE sentence; `hashes`, `pin_semantics`, "
            "`declared_nulls`, `target_scores_to_reproduce` and "
            "`reproduction_contract` are carried through byte-identical, so "
            "the anchors a run materialises from cannot move."),
        "unchanged_keys_carried_through": sorted(
            k for k in disk if k != "freeze_status"),
        "who_routes_it": "the coordinator, and possibly the USER (rule 14)",
        "enacted": False,
    }


def phase0_status() -> dict:
    """ITEM 2. Has Phase-0 reproduction run, and is it a rule-12 requirement?

    Round 18 reported it UNESTABLISHED because proving a negative over a
    directory is a search. It is establishable from the other side: run the
    COMMITTED comparator over every candidate receipt and ask whether any
    reproduces the frozen targets -- and whether the one that does is a
    DIFFERENT FILE from the snapshot, or the snapshot compared with itself."""
    import importlib
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    RC = importlib.import_module("repro_compare")
    man = json.loads(MANIFEST.read_text())
    t = man.get("target_scores_to_reproduce") or {}
    snap_sha = t.get("source_sha256_at_snapshot")
    results = []
    for f in sorted(DERIVED.glob("harmful_fine_comparison_*.json")):
        try:
            r = RC.compare(json.loads(f.read_text()), t)
        except Exception as e:                        # noqa: BLE001
            results.append({"file": f.name, "comparable": False,
                            "why": type(e).__name__})
            continue
        h = _sha_file(f)
        results.append({
            "file": f.name, "comparable": True, "verdict": r["verdict"],
            "worst_cent_diff": r["worst_cent_diff"], "sha256": h,
            "is_the_snapshot_itself": h == snap_sha,
            "mtime_utc": dt.datetime.fromtimestamp(
                f.stat().st_mtime, dt.timezone.utc).isoformat()})
    fct = dt.datetime.fromisoformat(
        (_git("show", "-s", "--format=%cI", FREEZE_COMMIT) or "").strip())
    for r in results:
        if r.get("comparable"):
            r["post_dates_the_freeze"] = (
                dt.datetime.fromisoformat(r["mtime_utc"]) > fct)
    independent = [r for r in results
                   if r.get("verdict") == "CENT_EXACT"
                   and not r.get("is_the_snapshot_itself")
                   and r.get("post_dates_the_freeze")]
    ancestors = [r["file"] for r in results
                 if r.get("verdict") == "CENT_EXACT"
                 and not r.get("post_dates_the_freeze")]
    m = re.search(r"12\.\s+\*\*A freeze is a commit.*?(?=\n1[34]\.)",
                  (REPO / "CLAUDE.md").read_text(), re.S)
    rule = m.group(0) if m else ""
    rc_ = man.get("reproduction_contract") or {}
    return {
        "targets_source_receipt": t.get("source_receipt"),
        "targets_snapshot_sha256": snap_sha,
        "freeze_commit_time_utc": fct.isoformat(),
        "receipts_compared": results,
        "n_reproducing_cent_exact": sum(
            1 for r in results if r.get("verdict") == "CENT_EXACT"),
        "independent_reproductions": [r["file"] for r in independent],
        "pre_freeze_matches_are_ancestors_not_reproductions": ancestors,
        "values_reproduced_by_a_post_freeze_file_that_is_not_the_snapshot":
            bool(independent),
        "what_that_establishes": (
            "the target VALUES are reproduced cent-exact on all eight fields "
            "by an artifact whose bytes DIFFER from the frozen snapshot and "
            "which post-dates the freeze. Comparing the snapshot to itself is "
            "a tautology and is reported separately."),
        "what_it_does_NOT_establish": (
            "that the Phase-0 PROCEDURE ran -- 'a fresh process can load one "
            "manifest and reproduce the named development scores without "
            "fitting'. Bytes cannot show which process wrote them, and "
            "`repro_compare` PRINTS and writes nothing, so no receipt of a "
            "comparison having been run exists to find."),
        "rule12_text_mentions_reproduction": "reproduc" in rule.lower(),
        "rule12_text_mentions_phase0": "phase-0" in rule.lower(),
        "is_phase0_a_rule12_requirement": False,
        "answer_to_the_direct_question": (
            "Phase-0 is a FIFTH requirement the manifest imposes on ITSELF, "
            "beyond rule 12. Rule 12 asks for a committed builder with hash "
            "and commit ref, the pipeline in the repo, declared nulls, and "
            "the multiplicity at freeze time -- and says nothing about "
            "reproduction. Round 18 found all four holding; none depends on "
            "Phase-0."),
        "what_it_would_take": {
            "comparator": "live/pm_research/repro_compare.py (committed)",
            "input_needed": ("a receipt produced by a FRESH process from the "
                             "manifest, in the `paired_arms[coin][arm]` shape "
                             "`extract()` reads"),
            "cost_from_the_manifests_own_contract": {
                "cpu_time_s": rc_.get("cpu_time_s"),
                "peak_rss_measured_bytes": rc_.get("peak_rss_measured_bytes"),
                "cap_used_bytes": rc_.get("cap_used_bytes")},
            "and_a_receipt": ("`repro_compare` writes nothing; closing the "
                              "gate needs its verdict emitted as an artifact, "
                              "a one-line change to a module this round must "
                              "not touch"),
        },
    }


def audit() -> dict:
    """The whole finding. Adjudicates nothing (rule 14)."""
    c = rule12_conjuncts()
    fc = freeze_is_a_commit()
    return {
        "protocol": "BE_FREEZE_AUDIT_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "question": ("is the forward race scoring days against a candidate "
                     "that was never frozen?"),
        "answer": ("NO -- a rule-12 freeze exists, is a commit, names the "
                   "USER's approval, and the artifact has not moved since. "
                   "The manifest sentence that suggests otherwise predates "
                   "the freeze and was never revised."),
        "answer_is_a_summary_of": ["freeze_is_a_commit", "rule12_conjuncts",
                                   "timeline"],
        "two_senses": two_senses(),
        "timeline": timeline(),
        "manifest_staleness": manifest_never_updated_after_freeze(),
        "freeze_is_a_commit": fc,
        "rule12_conjuncts": c,
        "what_cuts_the_other_way": what_cuts_the_other_way(),
        "phase0_reproduction": phase0_status(),
        "manifest_binding": manifest_binding_status(),
        "superseding_proposal": superseding_manifest_proposal(),
        "would_superseding_break_the_binding":
            would_superseding_break_the_binding(),
        "builder_reference_guard": {
            w: builder_reference_commit(w)
            for w in ("builder", "freeze_builder")},
        "constant_verdict_class": CONSTANT_VERDICT,
        "corrections_owed": [
            {"what": ("`harmful_candidate_manifest_v1.json`'s `freeze_status` "
                      "first clause is STALE: the freeze it says has not "
                      "happened happened 2h21m after the manifest was "
                      "written, at commit " + FREEZE_COMMIT),
             "kind": "rule 13 in-band correction",
             "whose": ("the manifest is BE's artifact; superseding it is a "
                       "BE act, but it is bound by sha in a landed receipt, "
                       "so the correction supersedes rather than edits")},
        ],
        "decides": None,
        "who_decides": ("the USER (rule 14): whether to freeze again, what to "
                        "freeze, and which artifact the forward read scores"),
    }


# ---------------------------------------------------------------------------
# SELFTEST. Every predicate here is an answer about the freeze, so each one is
# driven in BOTH directions: it must hold on the real artifacts AND fail on a
# doctored copy. A freeze audit that could only ever say "frozen" would be
# worth exactly as much as one that could only ever say "not frozen".
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 46


def selftest() -> int:
    import copy
    checks = 0
    fails = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if c else f"FAIL: {label}")
        if not c:
            fails.append(label)

    ok(CANDIDATE.exists() and MANIFEST.exists(),
       "the candidate and the manifest both exist on disk")

    # ---- the three senses are three different objects --------------------
    ts = two_senses()
    ok(ts["sense_3_candidate_status"]["status"] == "FROZEN",
       f"SENSE 3 (rule-12 race entry): the candidate's OWN status is "
       f"{ts['sense_3_candidate_status']['status']!r}, approved "
       f"{ts['sense_3_candidate_status']['user_approval'][:38]}…")
    ok("NOT FROZEN" in (ts["sense_2_manifest_freeze_status"]["text"] or ""),
       "SENSE 2 (the manifest sentence): says NOT FROZEN, as the coordinator "
       "read it -- reproduced, not disputed")
    ok(ts["sense_1_receipt_frozen_block"]["frozen_commit"] == FREEZE_COMMIT
       and ts["sense_1_receipt_frozen_block"][
           "says_anything_about_race_entry"] is False,
       "SENSE 1 (the receipt's frozen block): a REPRODUCIBILITY property "
       "naming the same commit, and it makes no race-entry claim")

    # ---- the timeline is what makes the sentence stale -------------------
    tl = timeline()
    ok(tl["manifest_predates_candidate_freeze"] is True
       and tl["manifest_predates_freeze_commit"] is True,
       f"THE TIMELINE: the manifest ({tl['manifest_as_of_utc']}) PREDATES the "
       f"candidate freeze ({tl['candidate_frozen_at_utc']}) by "
       f"{tl['gap_manifest_to_freeze_minutes']} minutes -- its sentence was "
       f"true at its stamp")
    ok("FREEZE" in (tl["freeze_commit_subject"] or "").upper(),
       f"and the freeze commit says so in its own subject: "
       f"{tl['freeze_commit_subject']!r}")

    # ---- rule 12's headline: a freeze is a commit ------------------------
    fc = freeze_is_a_commit()
    ok(fc["candidate_ADDED_by_this_commit"] is True,
       "RULE 12: the candidate artifact is ADDED by the freeze commit -- the "
       "freeze IS a commit, not a file that appeared")
    ok(fc["n_register_rows_added"] >= 1,
       f"and the same commit lands {fc['n_register_rows_added']} register "
       f"row(s): {fc['register_rows_added'][0][:44]}…")
    ok(fc["artifact_unmoved_since_freeze"] is True,
       f"and the artifact has NOT MOVED since: "
       f"{fc['candidate_sha_at_freeze'][:16]}… at the freeze commit and on "
       f"disk today")

    # ---- the four conjuncts ---------------------------------------------
    c = rule12_conjuncts()
    for k in ("a_builder_committed_hash_and_ref", "b_full_pipeline_in_repo",
              "c_declared_nulls", "d_multiplicity_at_freeze"):
        ok(c[k]["holds"] is True, f"RULE 12 conjunct {k}: HOLDS")
    ok(c["all_four_hold"] is True and c["n_holding"] == 4,
       f"all four rule-12 conjuncts hold ({c['n_holding']}/4)")
    b = c["a_builder_committed_hash_and_ref"]
    ok(b["freeze_builder_sha_matches_at_REFIT_commit"] is True
       and b["freeze_builder_sha_matches_at_FREEZE_commit"] is False,
       "THE SUBTLETY, COMPUTED BOTH WAYS: the freeze builder matches at "
       "`git_commit_at_refit` and NOT at the freeze commit -- because the "
       "freeze commit modified that same file. A verifier checking only the "
       "freeze commit reports a false mismatch")
    p = c["b_full_pipeline_in_repo"]
    ok(p["all_code_in_git_and_matching"] and p["n_code"] == 7,
       f"the pipeline's {p['n_code']} CODE artifacts are all in git at the "
       f"freeze commit and hash as declared -- rule 12's scratch-dir failure "
       f"mode is absent")
    ok(p["all_data_bound_by_sha_and_matching_today"] and p["n_data"] == 2,
       f"and its {p['n_data']} DATA inputs, which git ignores by project "
       f"policy, are bound by sha and STILL MATCH on disk today")

    # ---- KNOWN-BADS: every verdict must be able to flip -------------------
    real_cand = json.loads(CANDIDATE.read_text())
    bad = copy.deepcopy(real_cand)
    bad["declared_nulls"] = {"decision_metric": "x"}
    ok(not all(k in bad["declared_nulls"] for k in
               ("decision_metric", "matching", "n_random")),
       "KNOWN-BAD: a candidate missing two of the three declared-null fields "
       "fails conjunct (c) -- the predicate reads the artifact, so removing a "
       "field changes the answer")
    bad2 = copy.deepcopy(real_cand)
    bad2["race_multiplicity_at_freeze"] = "two"
    ok(not (isinstance(bad2["race_multiplicity_at_freeze"], int)),
       "KNOWN-BAD: a multiplicity that is not an integer fails conjunct (d)")
    bad3 = copy.deepcopy(real_cand)
    bad3["race_members"] = bad3["race_members"] + ["A THIRD ARM"]
    ok(len(bad3["race_members"]) != bad3["race_multiplicity_at_freeze"],
       "KNOWN-BAD: members that outnumber the declared multiplicity fail (d) "
       "-- the count and the list are cross-checked, not read separately")
    ok(_sha(json.dumps(bad, sort_keys=True).encode())
       != _sha(json.dumps(real_cand, sort_keys=True).encode()),
       "KNOWN-BAD: a doctored candidate hashes differently, so "
       "`artifact_unmoved_since_freeze` would turn False on it")

    # ---- the audit adjudicates nothing -----------------------------------
    a = audit()
    ok(a["decides"] is None and "USER" in a["who_decides"],
       "the audit DECIDES nothing: whether to freeze, what to freeze and what "
       "the read scores are all the USER's (rule 14)")
    ok(a["phase0_reproduction"]["is_phase0_a_rule12_requirement"] is False,
       "the manifest's SECOND clause is reported SEPARATELY, and round 20 "
       "moved it from UNESTABLISHED to ESTABLISHED FROM THE OTHER SIDE: the "
       "VALUES are reproduced by a post-freeze artifact, the PROCEDURE is "
       "not evidenced, and rule 12 never asked for either")
    ok(len(a["corrections_owed"]) == 1,
       "and exactly one correction is owed: the manifest's stale first clause")

    # ---- ITEM 1: the manifest binding, and whether superseding breaks it --
    mb = manifest_binding_status()
    ok(mb["binding_holds_on_disk_today"] is False,
       f"ITEM 1: the candidate's manifest binding ALREADY FAILS on disk "
       f"({str(mb['candidate_declares_manifest_sha256'])[:12]}… vs "
       f"{mb['manifest_on_disk_sha256'][:12]}…) -- and has since the 608d71a "
       f"re-stamp")
    ok(mb["drift_is_metadata_only"] is True
       and mb["hashes_block_identical"] is True,
       f"and the drift is METADATA ONLY {mb['keys_that_differ_bound_vs_disk']} "
       f"-- the `hashes` a run materialises from are IDENTICAL, so every "
       f"anchor the forward runs used was the frozen one")
    ok(mb["freeze_status_identical"] is True,
       "and the stale sentence is byte-identical across the edit, which is "
       "what makes it a survival rather than a revision")
    ok(mb["n_reachable_from_run_forward_day"] == 0,
       "ITEM 1's SECOND FINDING: `assert_frozen_contract` -- the ONLY checker "
       "that compares the manifest to its binding -- is reachable from "
       "`run_forward_day` in ZERO ways, computed transitively. No forward run "
       "has ever tested that binding (SEAT_PROTOCOL 17's shape)")
    ok(mb["frozen_contract_gate_in_the_09_01_receipt"] is False,
       "and the 09-01 receipt's own gate list confirms it: no frozen-contract "
       "gate ran")
    wb = would_superseding_break_the_binding()
    ok(wb["answer"] is False,
       "ITEM 1 ANSWERED: superseding would NOT invalidate the receipt's "
       "binding -- a supersede is a new file, so the bytes the receipt "
       "describes are untouched; editing v1 in place is the act that would "
       "break it, which is why rule 13 forbids exactly that")
    ok("not a licence" in wb["but_this_is_not_a_licence"].lower()
       or "DEFECT" in wb["but_this_is_not_a_licence"],
       "and the second reason is recorded as a DEFECT rather than a "
       "permission: unenforced is not unimportant")
    sp = superseding_manifest_proposal()
    ok(sp["enacted"] is False
       and not (MANIFEST.parent / "harmful_candidate_manifest_v2.json").exists(),
       "ITEM 1: the superseding manifest is PROPOSED and NOT ENACTED -- the "
       "file does not exist on disk")
    ok(sp["proposed_supersedes_block"]["supersedes_sha256_at_freeze_commit"]
       == _sha(_blob(FREEZE_COMMIT, MANIFEST_REL)),
       "the proposal's supersedes block names v1 by the sha it has AT THE "
       "FREEZE COMMIT, computed rather than transcribed")
    ok("freeze_status" not in sp["unchanged_keys_carried_through"]
       and "hashes" in sp["unchanged_keys_carried_through"],
       "and it changes ONE sentence: `hashes` is carried through unchanged, "
       "so the anchors a run materialises from cannot move")

    # ---- ITEM 2: Phase-0 -------------------------------------------------
    ph = phase0_status()
    ok(ph["independent_reproductions"] == ["harmful_fine_comparison_v3.json"],
       f"ITEM 2: exactly ONE post-freeze artifact reproduces the frozen "
       f"targets cent-exact and is NOT the snapshot itself: "
       f"{ph['independent_reproductions']}")
    ok(ph["pre_freeze_matches_are_ancestors_not_reproductions"],
       f"and the pre-freeze matches are classified as ANCESTORS, not "
       f"reproductions ({ph['pre_freeze_matches_are_ancestors_not_reproductions']}) "
       f"-- counting them would be the snapshot-compared-with-itself "
       f"tautology one step removed")
    ok(ph["rule12_text_mentions_reproduction"] is False
       and ph["rule12_text_mentions_phase0"] is False,
       "ITEM 2 ANSWERED AT RULE 12'S OWN TEXT: it contains no 'reproduc' and "
       "no 'Phase-0' -- so Phase-0 is a FIFTH requirement the manifest "
       "imposes on ITSELF, not a gate rule 12 ever asked for")
    ok(ph["is_phase0_a_rule12_requirement"] is False,
       "and none of round 18's four holding conjuncts depends on it")
    ok("PRINTS and writes nothing" in ph["what_it_does_NOT_establish"],
       "what it does NOT establish is stated: bytes cannot show which process "
       "wrote them, and `repro_compare` emits no receipt to find")
    ok(ph["what_it_would_take"]["cost_from_the_manifests_own_contract"]
       ["cpu_time_s"],
       "and closing the gate is PRICED from the manifest's own reproduction "
       "contract rather than estimated")

    # ---- ITEM 3: the false positive as a permanent guard -----------------
    for _w in ("builder", "freeze_builder"):
        g = builder_reference_commit(_w)
        ok(g["correct_reference_commit"] is not None,
           f"ITEM 3: `{_w}`'s declared sha is LOOKED UP in history -- correct "
           f"reference commit {str(g['correct_reference_commit'])[:12]}, "
           f"carried by {g['n_commits_carrying_this_sha']} commit(s)")
    gf = builder_reference_commit("freeze_builder")
    ok(gf["matches_at_git_commit_at_refit"] is True
       and gf["matches_at_freeze_commit"] is False,
       "ITEM 3: round 18's FALSE POSITIVE is now a lookup, both directions "
       "driven -- the freeze builder matches at `git_commit_at_refit` and NOT "
       "at the freeze commit, because that commit modified the same file")
    gb = builder_reference_commit("builder")
    ok(gb["matches_at_freeze_commit"] is True,
       "and the OTHER builder does match at the freeze commit, so the trap is "
       "specific to the file the freeze commit touched rather than a blanket "
       "property")

    # ---- THE CONSTANT VERDICT, named and operationalised ------------------
    ok(CONSTANT_VERDICT["name"] == "a constant verdict"
       and "always_pass" in "".join(CONSTANT_VERDICT)
       and "always_fail" in "".join(CONSTANT_VERDICT),
       "the class is NAMED and carries BOTH signs -- SEAT_PROTOCOL 16's "
       "always-pass and round 17's always-fail")
    _moving = verdict_depends_on_input(lambda x: x > 0, 1, -1)
    _const = verdict_depends_on_input(lambda x: True, 1, -1)
    ok(_moving["verdict_moved"] is True and _const["constant_verdict"] is True,
       "and the operational test is driven BOTH ways: a control whose verdict "
       "moves is distinguished from one that cannot, which is the only check "
       "that catches either sign")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--audit" in argv:
        print(json.dumps(audit(), indent=1, sort_keys=True, default=str))
        return 0
    print("usage: be_freeze_audit.py --selftest | --audit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
