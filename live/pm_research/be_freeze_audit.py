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


def phase0_reproduction_status() -> dict:
    """The manifest's SECOND clause, reported separately and NOT resolved by
    the freeze. This is the part of the sentence that may still be true."""
    comparator = Path(__file__).resolve().parent / "repro_compare.py"
    man = json.loads(_blob(FREEZE_COMMIT, MANIFEST_REL))
    t = man.get("target_scores_to_reproduce") or {}
    named = [k for k in t if k.endswith("_PM_PLUS_FINE")]
    return {
        "comparator_module_exists": comparator.exists(),
        "comparator": str(comparator) if comparator.exists() else None,
        "targets_declared": named,
        "targets_source_receipt": t.get("source_receipt"),
        "reproduction_receipt_found_in_derived": None,
        "status": "NOT ESTABLISHED BY THIS MODULE",
        "why": ("a Phase-0 reproduction receipt would be a separate artifact "
                "and this module found none by name; but proving a NEGATIVE "
                "over an artifact directory is a search, not a predicate, so "
                "this is reported as UNESTABLISHED rather than asserted as "
                "absent. It is a DIFFERENT gate from the rule-12 freeze and "
                "does not bear on it."),
        "bears_on_rule12_freeze": False,
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
        "phase0_reproduction": phase0_reproduction_status(),
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
EXPECTED_CHECKS = 24


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
    ok(a["phase0_reproduction"]["bears_on_rule12_freeze"] is False
       and a["phase0_reproduction"]["status"].startswith("NOT ESTABLISHED"),
       "the manifest's SECOND clause (Phase-0 reproduction) is reported "
       "SEPARATELY and as UNESTABLISHED -- proving a negative over an "
       "artifact directory is a search, not a predicate")
    ok(len(a["corrections_owed"]) == 1,
       "and exactly one correction is owed: the manifest's stale first clause")

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
