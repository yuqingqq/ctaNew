#!/usr/bin/env python3
"""Days the USER has WITHDRAWN from the forward race — recorded, and one-way.

WHY THIS IS A MODULE AND NOT A CONSTANT IN THE VERDICT FILE. A withdrawal is
only worth anything if it cannot be quietly undone, and "cannot be undone" is a
property of a FILE'S HISTORY. Put in `da_forward_day_verify.py` the check would
have to walk 5,700 lines of unrelated history on every run; here the file's
whole reason to exist is the registry, so the monotonicity check is exact,
cheap, and obviously scoped to the thing it protects.

THE TWO FACTS THIS SEPARATES, and running them together is what R-500 (C)
corrects. 2026-08-29's era IS admissible -- R-497 (F)(1) ruled it, and the
verdict that said otherwise was asserting something the USER had made false.
The day is nevertheless NOT ENTERED in the race, because the USER elected it as
a development read. The first is a computation and belongs to the four accrual
conjuncts; the second is a POLICY FACT and belongs here. A verdict that encodes
the policy by falsifying the computation is the `ERA_ADMISSIBLE` defect in a
new coat, and this module exists so that never has to happen again.

THE ONE-WAY PROPERTY, AND WHAT IT CAN AND CANNOT DO.
  * A day may be ADDED only while NO FORWARD READ OF IT EXISTS. After its
    economics have been seen, entering or re-entering it is selection on the
    outcome (CLAUDE.md rule 11). `assert_withdrawal_admissible` computes that
    from the forward artifacts, not from a promise.
  * A day may NEVER be REMOVED, and an entry's authority may never change.
    `assert_withdrawals_monotone` reads THIS FILE'S OWN COMMITTED HISTORY and
    REFUSES BY NAME if any previously recorded day is missing or re-cited.
  * WHAT IT CANNOT DO, stated rather than implied: nothing here stops a commit
    from editing this file. What it stops is that edit passing unnoticed --
    the suite goes red and every verdict emission refuses, by name, naming the
    day that was dropped. A guard that claims to make a file immutable would be
    lying; this one makes an undo LOUD, which is the property actually
    available.

    python3 live/pm_research/da_race_withdrawals.py            # report
    python3 live/pm_research/da_race_withdrawals.py --selftest  # rule 15
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
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

#: This suite's own total, asserted over ran + skipped.
EXPECTED_CHECKS = 66

#: DA39-R1 (MEM): A COUNT CANNOT TELL YOU WHICH CHECKS RAN. A hand-maintained
#: tally catches an ADDITION and a REMOVAL and is blind to a REPLACEMENT --
#: swap one check for another and the suite still reads 59, so "59 checks
#: passed" is evidence about the arithmetic and not about the coverage. And
#: this module is the one whose "52 checks passed" was quoted as verification.
#:
#: THE FIX IS TO DIGEST THE CHECK IDENTITIES, NOT TO COUNT THEM. Each label's
#: leading token is its stable id (`DA35-R1`, `WALK-S9`, `PARSER`); the suite
#: digests the ORDERED id list and asserts it. A replacement changes an id and
#: therefore the digest; a prose edit to a label's wording does not, so the pin
#: is not noise. On mismatch the ADDED and REMOVED ids are printed by name --
#: the "which" a count could never give.
#:
#: WHAT NO RUNTIME MECHANISM CAN DO, said plainly rather than left implied: no
#: check inside the suite can distinguish an INTENDED replacement from an
#: unintended one. The digest makes the swap visible in a diff and hands the
#: judgement to review, which is where it belongs.
EXPECTED_CHECK_IDS_SHA = "c59445595399fb4b"

#: THE REGISTRY. One entry per day the USER has taken out of the race.
#: EVERY entry carries its authority as DATA -- the same discipline round 22
#: put on `ERA_ADMISSIBLE`, and for the same reason: a policy value with no
#: cite is indistinguishable from a seat's default, and this programme has
#: already paid twice for that (`# pre-O1`, and `clob_v5`).
RACE_WITHDRAWALS: dict[str, dict[str, Any]] = {
    "20260829": {
        "authority": "USER RULING 2026-09-03, R-500 (B)",
        "reason": ("elected as a DEVELOPMENT READ. The day is not excluded "
                   "and it is not unusable -- it is deliberately not entered "
                   "in the forward race, so that reading its economics costs "
                   "the race nothing by a RECORDED DECISION rather than by an "
                   "accident of file state"),
        "recorded_at": "R-500",
        "era_admissible_at_withdrawal": True,
        "era_authority": "USER RULING 2026-09-03, R-497 (F)(1)",
        "note": ("THE DAY IS REACHABLE, and that is the point of recording "
                 "this. A fully attributed, closed, race-eligible verdict for "
                 "this day exists in git (blob 79767ca38 at commit 4e1133c, "
                 "as_of 2026-08-30T00:06:01.246972Z, written inside the "
                 "scheduled unit with a real INVOCATION_ID). DA's earlier "
                 "`UNREACHABLE_BY_ANY_HONEST_ROUTE` overstated a statement "
                 "about one file as a statement about the day; the reviewer's "
                 "DA22-R1 is upheld. What keeps 08-29 out of the race is this "
                 "ruling, not impossibility"),
    },
}


#: DA24-R4. WHICH FIELDS OF AN ENTRY MAY NEVER MOVE, DECLARED RATHER THAN
#: IMPLIED. The first version of the monotonicity check compared `authority`
#: alone, so `reason` -- the field a reader actually quotes -- was silently
#: rewritable: an entry could be re-labelled "re-admitted after review" with
#: `monotone: True`. It could not re-admit the day, so it was not a defeat,
#: but a withdrawal whose stated reason can be edited without a trace is
#: weaker than one whose cite cannot.
#:
#: Comparing the WHOLE entry was the other option and it is worse: it would
#: make an honest clarification of DA's own commentary a "violation", and a
#: guard that fires on legitimate maintenance is a guard people learn to work
#: around. So the split is declared, and the DEFAULT IS IMMUTABLE:
#: `_entry_fields_are_classified` refuses an entry carrying a key in neither
#: set, so a field added later is load-bearing until somebody says otherwise
#: rather than unguarded until somebody notices.
IMMUTABLE_FIELDS = ("authority", "reason", "recorded_at",
                    "era_admissible_at_withdrawal", "era_authority")
ANNOTATABLE_FIELDS = ("note",)


def _entry_fields_are_classified(entry: dict) -> list[str]:
    """Keys in neither set. Non-empty means the classification is stale."""
    known = set(IMMUTABLE_FIELDS) | set(ANNOTATABLE_FIELDS)
    return sorted(k for k in entry if k not in known)


class WithdrawalRefused(Exception):
    """A withdrawal this module must not record or must not let stand."""


def _git(args: list[str], cwd: Path | None = None
         ) -> tuple[int | None, str]:
    """`(returncode, stdout)`; **rc is None when git never ran.**

    It used to return `(127, str(e))`. 127 is a returncode, and a consumer
    reading one cannot tell "git exited non-zero" from "git was never
    executed" -- the DA32-R1 codomain predicate, one layer down from the
    monotonicity guarantee this module exists to make. `None` is not a
    returncode, so `rc != 0` still refuses (correctly) and a caller that
    wants the distinction can have it.
    """
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd=str(cwd or CODE_ROOT), timeout=120)
        return r.returncode, r.stdout
    except Exception as e:                                   # pragma: no cover
        return None, str(e)


def withdrawal_for(day_token: str, table: dict | None = None
                   ) -> dict[str, Any] | None:
    """The USER's withdrawal for this day, or None. Callable."""
    return (RACE_WITHDRAWALS if table is None else table).get(day_token)


def forward_read_exists(day_token: str, derived: Path | None = None,
                        extra_dirs: list[Path] | None = None
                        ) -> dict[str, Any]:
    """Has any forward READ of this day happened? Computed, never assumed.

    A REFUSAL receipt is NOT a read: BE's 08-29 receipt carries
    `outcome: REFUSED`, `refused_at: day_closed_and_attributed`, no
    `sealed_file`, no rows and no metric. Reading a refusal tells you the gate
    fired, which is the opposite of seeing the day's economics.

    ABSENCE IS REPORTED, NOT ASSUMED. If no directory can be read at all the
    result says so and `conclusive` is False -- "I found no receipt" and "I
    could not look" must never be the same answer (standing rule 11).
    """
    dirs = [DERIVED if derived is None else Path(derived)]
    dirs += [Path(x) for x in (extra_dirs or [])]
    seen, reads, refusals, unreadable = [], [], [], []
    for d in dirs:
        if not d.is_dir():
            unreadable.append(str(d))
            continue
        seen.append(str(d))
        for f in sorted(d.rglob(f"be_forward_day_receipt_{day_token}*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                reads.append({"path": str(f), "outcome": "UNPARSEABLE",
                              "counts_as_read": True,
                              "why": "a receipt this cannot read must count "
                                     "as a read: absence of evidence here is "
                                     "not evidence of absence"})
                continue
            outcome = doc.get("outcome")
            # THE NAME IS NOT THE DEFINITION, and this cost a wrong refusal
            # the first time it ran against the real artifact. BE's 08-29
            # receipt carries `"sealed": true` -- a PROTOCOL FLAG meaning the
            # receipt follows the sealing discipline -- while its `outcome` is
            # REFUSED, it names no score file, and it holds no rows and no
            # metric. Reading that boolean as evidence of a read classified a
            # gate firing as a look at the day's economics and REFUSED the
            # withdrawal for a reason that was not true.
            #
            # Evidence of a READ is a PATH to scores, never a boolean: a
            # non-empty `sealed_file`, or a positive scored-row/action count.
            _sf = doc.get("sealed_file")
            _sealed_path = _sf if isinstance(_sf, str) and _sf.strip() else None
            _n = doc.get("n_actions_scored") or doc.get("rows") or 0
            _scored = isinstance(_n, (int, float)) and _n > 0
            sealed = bool(_sealed_path) or _scored
            rec = {"path": str(f), "outcome": outcome,
                   "refused_at": doc.get("refused_at"),
                   "sealed_file": _sealed_path,
                   "n_scored": _n if isinstance(_n, (int, float)) else None,
                   "sealed_flag_present": "sealed" in doc,
                   "sealed_flag_is_not_evidence": True,
                   "as_of": doc.get("as_of_utc")}
            if outcome == "REFUSED" and not sealed:
                refusals.append(rec)
            else:
                rec["counts_as_read"] = True
                reads.append(rec)
    return {
        "day": day_token, "dirs_read": seen, "dirs_unreadable": unreadable,
        "n_reads": len(reads), "reads": reads,
        "n_refusals": len(refusals), "refusals": refusals,
        "a_read_exists": bool(reads),
        "conclusive": bool(seen),
        "why": ("a REFUSED receipt with no sealed file is a gate firing, not "
                "a look at the day's economics"),
    }


def assert_withdrawal_admissible(day_token: str, derived: Path | None = None,
                                 extra_dirs: list[Path] | None = None
                                 ) -> dict[str, Any]:
    """A day may enter the registry ONLY while no forward read of it exists.

    REFUSES BY NAME otherwise. This is the half of the one-way property that
    protects the FRONT door: withdrawing a day after seeing what it pays is
    selection on the outcome exactly as re-admitting one is.
    """
    fr = forward_read_exists(day_token, derived, extra_dirs)
    if not fr["conclusive"]:
        raise WithdrawalRefused(
            f"REFUSED: could not read any forward-artifact directory for "
            f"{day_token} ({fr['dirs_unreadable']}). 'I could not look' must "
            f"never be recorded as 'nothing was read'.")
    if fr["a_read_exists"]:
        raise WithdrawalRefused(
            f"REFUSED: {day_token} already has {fr['n_reads']} forward "
            f"READ(S) ({[r['path'] for r in fr['reads']]}). A day may be "
            f"withdrawn only BEFORE its economics are seen; withdrawing it "
            f"afterwards is selection on the outcome (rule 11) in the "
            f"opposite direction from re-admitting it.")
    return {"day": day_token, "admissible_to_withdraw": True, **fr}


class BlobUnparseable(WithdrawalRefused):
    """A version of the registry file that could not be read AS a registry.

    Outside the codomain of `dict | None` deliberately: `None` means the name
    is ABSENT from that version, which is a fact about the history, and this
    means the version could not be evaluated, which is a fact about the
    reader. Collapsing them drops a version from the monotonicity walk.
    """


def _registry_in_blob(text: str) -> dict[str, dict] | None:
    """`RACE_WITHDRAWALS` as data, parsed and never executed.

    None means the name is absent from that version -- which is different from
    an empty registry, and the caller must not collapse them.

    **DA32-R1, THE OTHER DOOR.** Round 32 stopped a blob git could not READ
    from being skipped. A blob that reads fine and does not PARSE, or whose
    value is not a literal, still returned `None` -- and `None` is
    "the registry did not exist in this version", so the version was dropped
    from the walk and `monotone: True` was reported over a smaller history
    for a second reason. Same guarantee, same defect, different route: these
    now RAISE.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise BlobUnparseable(
            f"REFUSED: a version of the registry file does not parse ({e}). "
            f"That is not the same as the registry being ABSENT in it, and "
            f"skipping it would evaluate a one-way guarantee over fewer "
            f"versions than exist.") from e
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign)
                   else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == "RACE_WITHDRAWALS":
                value = node.value if isinstance(node, ast.Assign) \
                    else node.value
                try:
                    return ast.literal_eval(value)
                except Exception as e:
                    raise BlobUnparseable(
                        f"REFUSED: `RACE_WITHDRAWALS` is present in this "
                        f"version but is not a literal ({e}); it cannot be "
                        f"evaluated as data and must not be read as absent."
                    ) from e
    return None


def assert_withdrawals_monotone(repo: Path | None = None,
                                path: str | None = None,
                                current: dict | None = None
                                ) -> dict[str, Any]:
    """A recorded withdrawal may never be removed or re-cited. REFUSES by name.

    Reads THIS FILE'S OWN committed history and compares every prior version's
    registry with the current one. A day that was withdrawn and is now absent,
    or whose authority has changed, is a quiet undo -- and this is what makes
    it loud.

    THE VACUOUS PASS IS REPORTED, NEVER HIDDEN. At the commit that introduces
    the registry there is no prior version carrying it, so the comparison has
    nothing to compare; `n_prior_versions_with_registry` says so, and a caller
    that wants a non-vacuous guarantee can read that number rather than the
    boolean.
    """
    # THE FLOOR IS A FACT ABOUT THIS FILE'S HISTORY, not about any repository
    # this walk is pointed at. A fixture repo with one commit is legitimate;
    # THIS module's history having shrunk to one is a rewrite. Captured before
    # the defaults are substituted, so "the caller asked for the canonical
    # walk" is what is recorded rather than "the arguments happen to match".
    is_canonical = repo is None and path is None
    repo = CODE_ROOT if repo is None else Path(repo)
    path = ("live/pm_research/da_race_withdrawals.py" if path is None
            else path)
    cur = dict(RACE_WITHDRAWALS if current is None else current)
    # NOT A REPOSITORY IS NOT AN UNREADABLE HISTORY, AND THE TWO MUST NOT
    # SHARE AN ANSWER. A copy of this module tree in a temp directory -- which
    # the wiring probes legitimately make -- has no history to compare, and
    # refusing there would block honest out-of-tree runs while proving
    # nothing. It is reported as a STATUS with `monotone: None`, never as a
    # pass. The CANONICAL write path refuses on it separately, because that is
    # where the guarantee has to hold.
    rc0, _ = _git(["rev-parse", "--is-inside-work-tree"], repo)
    if rc0 != 0:
        return {
            "repo": str(repo), "path": path, "status": "NO_REPOSITORY",
            "monotone": None, "vacuous": True,
            "n_commits_touching_file": None,
            "n_prior_versions_with_registry": None,
            "n_days_now": len(cur), "days_now": sorted(cur),
            "why": ("this tree is not a git work tree, so the one-way "
                    "guarantee has NO EVIDENCE here. `monotone` is None, not "
                    "True: a canonical write must refuse on this, and does"),
        }
    rc, out = _git(["log", "--format=%H", "--follow", "--", path], repo)
    if rc != 0:
        raise WithdrawalRefused(
            f"REFUSED: cannot read the history of {path} in {repo} "
            f"(git rc {rc}). A one-way guarantee that cannot read its own "
            f"history has no evidence for the claim it makes, and reporting "
            f"'monotone' from an unreadable history is the empty-set trap on "
            f"the check that exists to prevent a quiet undo.")
    commits = [c for c in out.split() if c]
    # ---------------------------------------------------------------------
    # THE WALK MUST BE ANCHORED AT THE FILE'S CREATION (round 35).
    #
    # Every route found so far shortened the walk from INSIDE the loop, and
    # each was closed where it was found. This closes the class from the
    # other end: whatever git hands us, the walk is only over the whole
    # history if it REACHES THE COMMIT THAT ADDED THE FILE. One predicate
    # subsumes the routes no scan of this function could see, because they
    # shorten the INPUT rather than the walk --
    #   * a SHALLOW clone, where `git log` stops at the graft boundary and
    #     exits 0;
    #   * a rename `--follow` failed to detect, where history silently
    #     begins mid-life;
    #   * a `path` that no longer names this file (the default is a string,
    #     and a moved module makes `git log` return zero commits, rc 0);
    #   * an empty repository, or any other reason the list comes back short
    #     without an error.
    # All of them present as: the oldest commit we walked is not the one that
    # created the file. Checked positively, against git's own answer.
    rc_add, add_out = _git(
        ["log", "--format=%H", "--diff-filter=A", "--follow", "--", path],
        repo)
    add_commits = [c for c in add_out.split() if c]
    if rc_add != 0 or not add_commits or not commits \
            or commits[-1] != add_commits[0]:
        raise WithdrawalRefused(
            f"REFUSED: the history walk for {path} is not anchored at the "
            f"file's creation (walked {len(commits)} commit(s), oldest "
            f"{(commits[-1][:9] if commits else None)!r}; git reports the "
            f"adding commit as {(add_commits[0][:9] if add_commits else None)!r}"
            f"). A walk over a SUFFIX of history is not a walk over history: "
            f"a shallow clone, a rename `--follow` did not detect, or a path "
            f"that no longer names this file all end here, all exit 0, and "
            f"all would report `monotone` over the versions that happen to "
            f"remain visible. This is the route class round 32 and round 34 "
            f"each closed one member of, closed from the supply side.")
    shallow_rc, shallow = _git(["rev-parse", "--is-shallow-repository"], repo)
    rep_rc, replaced = _git(["replace", "-l"], repo)
    if shallow.strip() == "true":
        raise WithdrawalRefused(
            f"REFUSED: {repo} is a SHALLOW repository. Its history is "
            f"truncated by construction, so a one-way guarantee proved over "
            f"it is proved over the part that was cloned.")
    if replaced.strip():
        raise WithdrawalRefused(
            f"REFUSED: {repo} carries git REPLACE refs "
            f"({replaced.split()[:3]}). They rewrite what `git log` reports "
            f"without changing any commit, which is precisely the quiet undo "
            f"this guarantee exists to make loud.")
    versions, violations, annotations_changed = [], [], []
    for c in commits:
        rc2, blob = _git(["show", f"{c}:{path}"], repo)
        if rc2 != 0:
            # PHANTOM PASS, the classic direction, on the guarantee that must
            # not weaken: a blob this cannot read used to `continue`, so the
            # version vanished from the walk and `monotone: True` was reported
            # over a SMALLER set. A one-way guarantee evaluated over fewer
            # versions than exist is not the guarantee.
            raise WithdrawalRefused(
                f"REFUSED: commit {c[:9]} touches {path} but its blob could "
                f"not be read (git rc {rc2}). Skipping it would report "
                f"`monotone` over a smaller history than exists, which is the "
                f"empty-set trap on the guard against a quiet undo.")
        try:
            reg = _registry_in_blob(blob)
        except BlobUnparseable as e:
            raise WithdrawalRefused(
                f"REFUSED at commit {c[:9]} of {path}: {e}") from e
        if reg is None:
            # THE ONE LEGITIMATE SHORTENING: this version predates the
            # registry, so there is nothing in it to compare. It is the only
            # `continue` in the walk and `walk_termination_census` accounts
            # for it by name.
            continue
        if not isinstance(reg, dict):
            raise WithdrawalRefused(
                f"REFUSED: at commit {c[:9]} `RACE_WITHDRAWALS` is a "
                f"{type(reg).__name__}, not a mapping. Iterating it would "
                f"raise AttributeError -- a traceback rather than a named "
                f"refusal -- and a guarantee that dies by traceback tells a "
                f"reader nothing about what it was checking.")
        versions.append(c)
        for day, entry in reg.items():
            if not isinstance(entry, dict):
                raise WithdrawalRefused(
                    f"REFUSED: at commit {c[:9]} the entry for {day!r} is a "
                    f"{type(entry).__name__}, not a mapping. Its fields "
                    f"cannot be compared, and a day whose record cannot be "
                    f"compared is not a day shown intact.")
            # `cur.get`, not `cur[day]`: the two branches must not depend on
            # each other for their safety. Written with an indexed `elif` a
            # mutation of the first branch raised KeyError instead of failing
            # by name -- the traceback-not-a-name shape again, found by
            # driving the mutant rather than by reading.
            now_entry = cur.get(day)
            if not isinstance(now_entry, dict):
                violations.append({
                    "commit": c, "day": day, "kind": "REMOVED",
                    "was": entry.get("authority")})
                continue
            # DA24-R4: EVERY load-bearing field, not `authority` alone.
            for _f in IMMUTABLE_FIELDS:
                if now_entry.get(_f) != entry.get(_f):
                    violations.append({
                        "commit": c, "day": day, "kind": "RE_CITED",
                        "field": _f,
                        "was": entry.get(_f), "now": now_entry.get(_f)})
            # An UNCLASSIFIED key is load-bearing by default; a field nobody
            # has classified must not be silently unguarded.
            for _f in set(entry) | set(now_entry):
                if _f in IMMUTABLE_FIELDS or _f in ANNOTATABLE_FIELDS:
                    continue
                if now_entry.get(_f) != entry.get(_f):
                    violations.append({
                        "commit": c, "day": day, "kind": "RE_CITED_UNCLASSIFIED",
                        "field": _f, "was": entry.get(_f),
                        "now": now_entry.get(_f),
                        "why": "a key in neither IMMUTABLE_FIELDS nor "
                               "ANNOTATABLE_FIELDS is treated as immutable"})
            for _f in ANNOTATABLE_FIELDS:
                if now_entry.get(_f) != entry.get(_f):
                    annotations_changed.append({
                        "commit": c, "day": day, "field": _f})
    # AND IT MUST BE *THIS* REPOSITORY, not merely a default-argument call.
    # `is_canonical` alone fired inside the DA20-R2 mutant tree -- a one-
    # commit repo the wiring probe builds in /tmp and emits through the
    # PRODUCTION path, so it calls with defaults too and its single version
    # is legitimate. The discriminator that survives a rebase (which changes
    # every hash) and distinguishes a scratch tree is the REMOTE: the
    # canonical repository has one, a probe tree has none. A fork under a
    # different remote name does not get the floor, and `floor_applies` says
    # so rather than leaving it to be discovered.
    _, _origin = _git(["config", "--get", "remote.origin.url"], repo)
    floor_applies = is_canonical and CANONICAL_REMOTE in _origin
    _reg_floor = floor_from_register(repo if not is_canonical else None)
    # DA35-R1: the register PIN beats the module literal, and a disagreement
    # is the coherent-rewrite signature -- lowering the literal now has to
    # lower a line in another file, in another seat's document, to pass.
    _floor = MIN_PRIOR_VERSIONS
    if floor_applies and _reg_floor["status"] == "PINNED":
        if _reg_floor["pinned"] != MIN_PRIOR_VERSIONS:
            raise WithdrawalRefused(
                f"REFUSED: the register pins the walk floor at "
                f"{_reg_floor['pinned']} and this module's literal says "
                f"{MIN_PRIOR_VERSIONS}. A floor that disagrees with its own "
                f"pin is the COHERENT-REWRITE signature: history dropped and "
                f"the floor lowered in one pass reads as a clean walk unless "
                f"the pin lives where this walk does not count it.")
        _floor = _reg_floor["pinned"]
    if floor_applies and versions and len(versions) < _floor:
        raise WithdrawalRefused(
            f"REFUSED: the walk found {len(versions)} version(s) of the "
            f"registry but MIN_PRIOR_VERSIONS pins {MIN_PRIOR_VERSIONS}. "
            f"History was rewritten under a one-way guarantee: a rebase that "
            f"dropped or squashed a commit touching {path} shortens this "
            f"walk with no error, no unreadable blob and no anchor "
            f"violation, because the replayed add is still an add. The floor "
            f"is committed in the file whose history is counted and may only "
            f"be RAISED.")
    if violations:
        raise WithdrawalRefused(
            f"REFUSED: a recorded race withdrawal has been REMOVED or "
            f"RE-CITED, which the ruling that made it forecloses. "
            f"{violations}. Re-admitting a day whose economics may have been "
            f"seen is selection on the outcome (rule 11); a withdrawal that a "
            f"later commit can quietly undo buys nothing.")
    return {
        "repo": str(repo), "path": path,
        "status": "READ" if versions else "NO_PRIOR_VERSION_WITH_REGISTRY",
        "anchored_at_creation": True,
        "min_prior_versions_pinned": (_floor if floor_applies else None),
        "floor_pin": _reg_floor,
        "floor_is_load_bearing": (floor_applies
                                  and _reg_floor["status"] == "PINNED"),
        "floor_applies": floor_applies,
        "origin_seen": _origin.strip()[:80] or None,
        "adding_commit": add_commits[0],
        "shallow": shallow.strip() == "true",
        "n_replace_refs": len(replaced.split()),
        "n_commits_touching_file": len(commits),
        "n_prior_versions_with_registry": len(versions),
        "prior_versions": versions,
        "n_days_now": len(cur), "days_now": sorted(cur),
        # `True` OVER ZERO COMPARISONS IS THE CODOMAIN DEFECT AGAIN (round
        # 34's predicate, applied to this module's own verdict). It used to
        # report True with `vacuous: True` beside it and trust the reader to
        # look; `da_forward_day_verify` reads `monotone is not True` and
        # would have emitted on a walk that compared nothing. None is not a
        # failure -- it is "not evaluated" -- and the caller already refuses
        # to emit on it.
        "monotone": True if versions else None,
        # REPORTED, not refused: prose may be clarified, and the report says
        # when it was, so "nothing changed" is never inferred from silence.
        "annotations_changed": annotations_changed,
        "immutable_fields": list(IMMUTABLE_FIELDS),
        "annotatable_fields": list(ANNOTATABLE_FIELDS),
        "vacuous": not versions,
        "why": ("no prior committed version carries the registry, so this "
                "pass compares nothing -- read n_prior_versions_with_registry, "
                "not the boolean" if not versions else
                f"every day in {len(versions)} prior version(s) is still "
                f"present with the same authority"),
    }


#: THE FLOOR ON THE WALK ITSELF, and the answer to what a REBASE means here.
#:
#: `git log --follow` walks HEAD's ANCESTRY ONLY. A rebase replays commits
#: into new objects and the originals become unreachable, so the file's
#: history survives only as replayed -- and a rebase that DROPS or SQUASHES a
#: commit touching this file shortens the walk with no error, no unreadable
#: blob, and no anchor violation (the replayed add is still an add). Nothing
#: inside the walk can see it, because the walk is over exactly what git
#: hands it. This programme rebases held chains routinely.
#:
#: So the count is pinned in the file whose history is being counted. It may
#: only ever be RAISED, by a commit, and a walk that sees fewer versions than
#: this refuses by name. That is the same one-way property the registry has,
#: applied to the evidence FOR it.
MIN_PRIOR_VERSIONS = 5

#: DA35-R1. THE LITERAL ABOVE IS NOT A FLOOR, IT IS A NUMBER IN THE FILE THE
#: FLOOR PROTECTS. "May only be RAISED" was prose and nothing enforced it, so
#: the guard caught an ACCIDENTAL rewrite and not a COHERENT one: a rewrite
#: that drops history AND lowers this literal in the same pass presents as a
#: clean walk with a satisfied floor -- and that is exactly the rewrite class
#: WALK-S9 exists for, because it is the one that can carry the file.
#:
#: So the floor is pinned WHERE THIS WALK DOES NOT COUNT: the register. That
#: is the honest home for the same reason deriving it from the module's own
#: history is circular. The reader below is DEGRADED-SAFE by design -- while
#: the register carries no pin it REPORTS `FLOOR_NOT_PINNED_IN_REGISTER` and
#: enforces nothing, because a guard that refuses the moment it is installed
#: takes the emission path down for a line that has not landed yet. It
#: becomes load-bearing when the coordinator places the marker, and a
#: DISAGREEMENT between register and module refuses by name from that moment.
#: R-529(A), USER RULING 2026-09-04: **THE FORWARD RACE IS DIRECTIONAL, NOT
#: SIGNIFICANCE-BEARING.** G=5 stands and the recorded multiplicity of 2 stands
#: and is not decorative -- but at G=5 with m=2 the best achievable adjusted p
#: is 1/2^G x m = 0.0625 > 0.05, so a clustered permutation test's CEILING is
#: its floor and the bar was one day short of what its own multiplicity
#: requires. The USER chose honesty about what five days of one-hour-scale data
#: support over extending the clock.
#:
#: SO EVERY EMISSION OF A RACE FIELD FROM THIS SEAT CARRIES THE FRAME. This
#: module's race fields are ADMISSIBILITY COUNTS -- `withdrawn_from_race`
#: answers a policy question and `counts_toward_race` counts days -- and none
#: of them was ever a p-value. That is exactly why the caveat has to be
#: EMITTED rather than assumed: a reader who finds a day count beside a
#: verdict is one step from reading it as evidence, and the ruling says the
#: statement must come first rather than be available on request.
RACE_READING = ("DIRECTIONAL AND CONSISTENCY ONLY, NEVER A HOLM-CLEARING "
                "VERDICT (R-529(A), USER 2026-09-04). At G=5 with the "
                "recorded multiplicity 2 the best achievable adjusted p is "
                "0.0625 > 0.05: the ceiling of a clustered permutation test "
                "is its floor, 1/2^G. A race result establishes DIRECTION "
                "and CONSISTENCY and cannot establish significance.")

#: DA37-R1 -- THE DOCUMENTATION OF A CONTROL SILENTLY BECAME THE CONTROL.
#: My own filing quoted the marker verbatim while ASKING for it to be placed,
#: so this reader returned PINNED before anyone wrote a pin: the register is
#: DATA READ BY AN INSTRUMENT, and a filing that names the token an
#: instrument scans for is a WRITE to that instrument's input. It was
#: harmless only by luck of the value -- a different number in the
#: illustration would have made the register carry two differing pins and
#: read UNPINNED, and a WRONG number would have guarded the walk at a floor
#: nobody decided.
#:
#: SO THE PIN NOW HAS A FORM PROSE CANNOT PRODUCE BY ACCIDENT: three
#: consecutive LINES -- an opening marker alone on its line, the value alone
#: on its line in strict form, a closing marker alone on its line. Every
#: entry in this register is a SINGLE LINE, so no filing, quotation or table
#: row can produce it; only a deliberate multi-line edit can. And a marker
#: appearing ANYWHERE outside such a block is reported by name rather than
#: ignored, because the next accidental quote should be loud.
FLOOR_BLOCK_BEGIN = "<!-- " + "DA-WALK-FLOOR-PIN:BEGIN" + " -->"
FLOOR_BLOCK_END = "<!-- " + "DA-WALK-FLOOR-PIN:END" + " -->"
REGISTER_FLOOR_MARKER = "DA-WALK-FLOOR: min_prior_versions="
REGISTER_PATH = ("orchestrator/PROGRAMS/P-2026-003-polymarket-5min/"
                 "workspace/COORDINATION.md")


def floor_from_register(repo: Path | None = None) -> dict[str, Any]:
    """The pinned floor, read from the register. REPORTS; never synthesises.

    A missing register, an unreadable one and an unpinned one are THREE
    different facts and none of them is a number.
    """
    root = CODE_ROOT if repo is None else Path(repo)
    reg = root / REGISTER_PATH
    if not reg.is_file():
        return {"status": "REGISTER_NOT_FOUND", "pinned": None,
                "path": str(reg)}
    try:
        text = reg.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"status": "REGISTER_UNREADABLE", "pinned": None,
                "error": repr(e), "path": str(reg)}
    lines = text.splitlines()
    # R512-R1 / DE16-R1: OWNERSHIP, NOT PRESENCE. A block shown INSIDE a
    # fenced code sample is a quotation of the control -- which is exactly
    # where a control gets documented -- and `de_ratification_check` learned
    # this before the floor marker existed. Fenced regions are excluded so a
    # filing may SHOW the pin form without becoming a pin.
    fenced = set()
    infence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            infence = not infence
            fenced.add(i)
        elif infence:
            fenced.add(i)
    hits, blocks = [], 0
    for i, ln in enumerate(lines):
        if i in fenced or ln.strip() != FLOOR_BLOCK_BEGIN:
            continue
        for j in range(i + 1, min(i + 6, len(lines))):
            if j in fenced:
                break
            cand = lines[j].strip()
            if cand == FLOOR_BLOCK_END:
                blocks += 1
                break
            if cand.startswith(REGISTER_FLOOR_MARKER):
                num = cand[len(REGISTER_FLOOR_MARKER):].strip()
                if num.isdigit():
                    hits.append(int(num))
    # EVERY OTHER MENTION IS REPORTED, so the next accidental quote is loud
    # rather than silent -- including this module's own docstrings.
    stray = [i + 1 for i, ln in enumerate(lines)
             if REGISTER_FLOOR_MARKER in ln
             and ln.strip() != REGISTER_FLOOR_MARKER + str(
                 hits[0] if hits else "")]
    if not hits:
        return {"status": ("FLOOR_MARKER_OUTSIDE_PIN_BLOCK" if stray
                           else "FLOOR_NOT_PINNED_IN_REGISTER"),
                "pinned": None, "path": str(reg),
                "n_blocks": blocks, "stray_marker_lines": stray,
                "why": (("the marker appears at line(s) " + str(stray) +
                         " but NOT inside a pin block, so it is prose "
                         "quoting a control rather than a control. Nothing "
                         "is pinned and nothing is enforced")
                        if stray else
                        "the register carries no pin block, so nothing "
                        "outside this file pins the floor and a coherent "
                        "rewrite is still unguarded")}
    if len(set(hits)) > 1:
        return {"status": "FLOOR_PINNED_INCONSISTENTLY", "pinned": None,
                "values": sorted(set(hits)), "path": str(reg),
                "n_blocks": blocks, "stray_marker_lines": stray,
                "why": "two different pins is not a pin"}
    return {"status": "PINNED", "pinned": hits[0], "path": str(reg),
            "n_blocks": blocks, "stray_marker_lines": stray}

#: The floor is a claim about THIS repository's history. A probe tree built
#: in /tmp calls the walk with the same default arguments and its one commit
#: is legitimate, so the remote -- which a rebase does not change and a
#: scratch tree does not have -- is what identifies the repository the claim
#: is about.
CANONICAL_REMOTE = "ctaNew"

#: How each SOURCE-LEVEL route out of the walk is accounted for. Keyed by
#: (kind, the nearest enclosing condition as source text) so a line moving
#: does not retire an entry and a NEW route appears as UNACCOUNTED.
WALK_ROUTES: dict[tuple[str, str], str] = {
    ("Return", "rc0 != 0"):
        "STATUS, NOT A PASS: not a repository -> monotone None. The caller "
        "refuses to emit on anything that is not True.",
    ("Raise", "rc != 0"):
        "REFUSES: the history itself is unreadable.",
    ("Raise", "rc_add != 0 or not add_commits or (not commits) or "
              "(commits[-1] != add_commits[0])"):
        "REFUSES: the walk is not anchored at the file's creation, which is "
        "how every supply-side truncation presents.",
    ("Raise", "shallow.strip() == 'true'"):
        "REFUSES: a shallow clone is truncated by construction.",
    ("Raise", "replaced.strip()"):
        "REFUSES: replace refs rewrite what `git log` reports.",
    ("Raise", "rc2 != 0"):
        "REFUSES: a blob that cannot be read (closed round 32).",
    ("Raise", "<except BlobUnparseable>"):
        "REFUSES: a version that does not parse, or whose registry is not a "
        "literal (closed round 34).",
    ("Continue", "reg is None"):
        "THE ONE LEGITIMATE SHORTENING: this version predates the registry, "
        "so it holds nothing to compare. Counted: the walked versions are "
        "reported as n_prior_versions_with_registry.",
    ("Raise", "not isinstance(reg, dict)"):
        "REFUSES: a registry that is not a mapping would die by "
        "AttributeError instead of by name.",
    ("Raise", "not isinstance(entry, dict)"):
        "REFUSES: an entry that is not a mapping cannot be compared.",
    ("Continue", "_f in IMMUTABLE_FIELDS or _f in ANNOTATABLE_FIELDS"):
        "NOT A SHORTENING OF THE WALK: it skips a field already compared by "
        "one of the two classified loops, inside the UNCLASSIFIED-key sweep. "
        "FOUND BY THIS CENSUS, not by me -- I wrote the accounting by hand "
        "and it was already incomplete, which is the argument for deriving "
        "the routes rather than listing them.",
    ("Continue", "not isinstance(now_entry, dict)"):
        "NOT A SHORTENING: the violation is RECORDED first; this only skips "
        "the field-by-field comparison of a day already found REMOVED.",
    ("Raise", "_reg_floor['pinned'] != MIN_PRIOR_VERSIONS"):
        "REFUSES: the register pin and the module literal disagree, which is "
        "the coherent-rewrite signature (DA35-R1).",
    ("Raise", "floor_applies and versions and (len(versions) < _floor)"):
        "REFUSES: fewer versions than the committed floor -- history was "
        "rewritten under the guarantee (a dropped or squashed commit in a "
        "rebase), which no check inside the walk can see.",
    ("Raise", "violations"):
        "REFUSES: the finding itself.",
    ("Return", "<unconditional>"):
        "THE VERDICT: monotone True over >=1 compared version, None over "
        "zero (a pass over nothing is not a pass).",
}

#: Routes that shorten the INPUT rather than the walk. **No scan of this
#: function can find these** -- they happen inside git and exit 0 -- which is
#: why the census carries them as a declared table with a named guard and a
#: named driver, rather than deriving them.
SUPPLY_ROUTES: tuple[dict[str, str], ...] = (
    {"id": "S1", "route": "the path names no file in this history (a moved "
                          "or renamed module; the default path is a string)",
     "presents_as": "git log exits 0 with NO commits",
     "guard": "anchor: commits[-1] == the adding commit",
     "driven_by": "WALK-S1"},
    {"id": "S2", "route": "a SHALLOW clone: history stops at the graft",
     "presents_as": "git log exits 0 with a truncated list, and the boundary "
                    "commit reports every file as ADDED, so the anchor check "
                    "alone PASSES -- the two guards are not redundant",
     "guard": "rev-parse --is-shallow-repository",
     "driven_by": "WALK-S2 (and WALK-S2b proves the anchor does not fire)"},
    {"id": "S3", "route": "a rename `--follow` did not detect",
     "presents_as": "history silently begins mid-life",
     "guard": "anchor: the oldest walked commit is not an ADD",
     "driven_by": "WALK-S3"},
    {"id": "S4", "route": "git REPLACE refs rewrite what `git log` reports",
     "presents_as": "a plausible, complete-looking, different history",
     "guard": "replace -l must be empty",
     "driven_by": "WALK-S4"},
    {"id": "S5", "route": "a version that genuinely predates the registry",
     "presents_as": "the name is absent from the blob",
     "guard": "NONE -- this one is legitimate and is COUNTED, not guarded",
     "driven_by": "WALK-S5"},
    {"id": "S6", "route": "a REBASE that drops or squashes a commit touching "
                          "the file (this programme rebases held chains)",
     "presents_as": "a shorter walk with no error, no unreadable blob and no "
                    "anchor violation -- the replayed add is still an add",
     "guard": "MIN_PRIOR_VERSIONS, a committed floor that may only be raised",
     "driven_by": "WALK-S9"},
)


def walk_termination_census(fn_name: str = "assert_withdrawals_monotone"
                            ) -> dict[str, Any]:
    """Every way the monotonicity walk can end or shorten, and its method.

    THE GUARANTEE HAS NOW FAILED BY TWO ROUTES NOBODY CLOSED IN ADVANCE
    (round 32: an unreadable blob was skipped; round 34: an unparseable one
    was read as "the registry was absent"). Both were found by looking at a
    defect, not by enumerating the space -- so this enumerates the space, and
    states how, because "these are all of them" is a claim that needs a
    method rather than a reading.

    THE METHOD IS TWO-PART, AND NEITHER PART ALONE IS SOUND:

      1. **Source routes, derived.** Every `return`, `raise`, `continue` and
         `break` in the walk's own body is enumerated from the AST and keyed
         by its nearest enclosing condition. Each must appear in
         `WALK_ROUTES` with a written judgement; one that does not is
         reported as **UNACCOUNTED**. This part cannot be forgotten, because
         it is derived from the source rather than remembered.

      2. **Supply routes, declared and driven.** A scan of this function can
         NEVER find a route that shortens its INPUT: a shallow clone, a lost
         rename, a path that no longer names the file. Those happen inside
         git, exit 0, and look exactly like a short history. They are carried
         as `SUPPLY_ROUTES`, each with the guard that catches it and the
         selftest that drives it.

    WHAT NEITHER PART COVERS, stated rather than implied: a git defect that
    misreports its own history; a filesystem or index race between the
    `log` and the `show`; and a version whose registry is syntactically a
    literal but semantically wrong (a fabricated authority string is
    monotone-clean and this check will pass it -- that is R-500's business,
    not this walk's).
    """
    import ast
    src = Path(__file__).read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:                                 # pragma: no cover
        raise WithdrawalRefused(
            f"REFUSED: this module does not parse ({e}); an enumeration that "
            f"cannot read the walk must not report a complete one.")
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        raise WithdrawalRefused(
            f"REFUSED: no function {fn_name!r} in this module. An "
            f"enumeration of a walk that is not there is not an empty "
            f"enumeration.")

    parent: dict[int, Any] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def _cond(node) -> str:
        cur = node
        while id(cur) in parent:
            up = parent[id(cur)]
            if isinstance(up, ast.If) and cur in up.body:
                return ast.unparse(up.test)
            if isinstance(up, ast.ExceptHandler):
                t = ast.unparse(up.type) if up.type is not None else "Exception"
                return f"<except {t}>"
            cur = up
        return "<unconditional>"

    found = []
    for node in ast.walk(fn):
        kind = type(node).__name__
        if kind not in ("Return", "Raise", "Continue", "Break"):
            continue
        cond = _cond(node).replace('"', "'")
        classification = (
            "REFUSES" if kind == "Raise" else
            "TERMINATES_THE_WALK" if kind == "Break" else
            "SHORTENS_THE_WALK" if kind == "Continue" else
            "EXITS")
        found.append({"kind": kind, "condition": cond, "line": node.lineno,
                      "classification": classification,
                      "accounted": (kind, cond) in WALK_ROUTES,
                      "judgement": WALK_ROUTES.get((kind, cond))})
    # DA35-R2: A ROUTE OUT OF THE WALK NEED NOT BE IN THE WALK. Part 1
    # enumerated this function's OWN exits, so an exception raised by a
    # callee -- `_registry_in_blob` raising BlobUnparseable, by design since
    # round 34 -- was neither a source route nor a supply route. It is safe
    # today only because the single production caller does not swallow it,
    # which rests on a CALLER rather than on the enumeration; the next callee
    # that raises need not be so lucky. The callees the walk names are
    # followed one level and their raises enumerated, each with whether the
    # walk HANDLES it (an except naming that type) or lets it PROPAGATE.
    callee_routes = []
    handled_types = set()
    for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
        if h.type is None:
            handled_types.add("BareExcept")
        else:
            handled_types.add(ast.unparse(h.type).split(".")[-1])
    called = {getattr(c.func, "id", None) or getattr(c.func, "attr", None)
              for c in ast.walk(fn) if isinstance(c, ast.Call)}
    bodies = {nd.name: nd for nd in tree.body
              if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name_ in sorted(x for x in called if x in bodies):
        for nd in ast.walk(bodies[name_]):
            if not isinstance(nd, ast.Raise) or nd.exc is None:
                continue
            exc = ast.unparse(nd.exc).split("(")[0].split(".")[-1]
            callee_routes.append({
                "callee": name_, "raises": exc, "line": nd.lineno,
                "handled_by_the_walk": exc in handled_types,
                "effect": ("CAUGHT AND RE-RAISED WITH CONTEXT"
                           if exc in handled_types
                           else "PROPAGATES OUT OF THE WALK")})
    unaccounted = [f for f in found if not f["accounted"]]
    stale = [list(k) for k in WALK_ROUTES
             if k not in {(f["kind"], f["condition"]) for f in found}]
    silent = [f for f in found
              if f["classification"] in ("SHORTENS_THE_WALK",
                                         "TERMINATES_THE_WALK")]
    return {
        "function": fn_name,
        "method": ("source routes DERIVED from the AST and reconciled "
                   "against a written accounting; CALLEE raises followed one "
                   "level and marked handled or propagating (DA35-R2); "
                   "supply routes DECLARED "
                   "with a guard and a driver, because no scan of this "
                   "function can see them"),
        "n_source_routes": len(found),
        "n_refusing": sum(1 for f in found if f["classification"] == "REFUSES"),
        "n_shortening": len(silent),
        "n_exits": sum(1 for f in found if f["classification"] == "EXITS"),
        "n_UNACCOUNTED": len(unaccounted),
        "UNACCOUNTED": unaccounted,
        "n_stale_accountings": len(stale), "stale_accountings": stale,
        "source_routes": found,
        "n_callee_raise_routes": len(callee_routes),
        "callee_raise_routes": callee_routes,
        "n_callee_raises_unhandled": len(
            [c for c in callee_routes if not c["handled_by_the_walk"]]),
        "n_supply_routes": len(SUPPLY_ROUTES),
        "supply_routes": list(SUPPLY_ROUTES),
        "n_shortenings_that_are_silent": len(
            [f for f in silent if not f["accounted"]]),
        "complete_for_this_method": not unaccounted and not stale,
        "residue": ("a callee TWO levels down, since callees are followed "
                    "one level only; a git defect misreporting its own "
                    "history; a race "
                    "between the `log` and the `show`; and a registry that "
                    "is a valid literal but a false one -- monotone-clean by "
                    "construction, and R-500's business rather than this "
                    "walk's"),
        "decides_nothing": "REPORTED (rule 14).",
    }


def withdrawal_block(day_token: str, table: dict | None = None,
                     reachability: dict | None = None) -> dict[str, Any]:
    """What a day verdict carries about the race. REPORTS; decides nothing.

    `counts_toward_race` is the number a reader should count, and it is
    COMPUTED from two inputs that stay separate: the four accrual conjuncts,
    and this policy fact. Neither is falsified to encode the other.
    """
    w = withdrawal_for(day_token, table)
    out = {
        "withdrawn_from_race": bool(w),
        "authority": (w or {}).get("authority"),
        "reason": (w or {}).get("reason"),
        "recorded_at": (w or {}).get("recorded_at"),
        "era_admissible_at_withdrawal": (w or {}).get(
            "era_admissible_at_withdrawal"),
        "one_way": ("a recorded withdrawal is never removed and never "
                    "re-cited; `assert_withdrawals_monotone` reads this "
                    "module's own committed history and refuses by name if "
                    "one is dropped"),
        "separation": ("the era's admissibility is a COMPUTATION and stays "
                       "whatever it computes; the withdrawal is a POLICY "
                       "FACT. `counts_toward_race` is derived from both and "
                       "neither is falsified to encode the other"),
        # FIRST FIELD A READER MEETS AFTER THE COUNTS, by ruling.
        "race_reading": RACE_READING,
    }
    if reachability is not None:
        out["reachability"] = {
            "verdict": reachability.get("verdict"),
            "attribution_reachable": reachability.get(
                "attribution_reachable"),
            "prior_attributed_versions": (reachability.get("achievable") or {}
                                          ).get("prior_attributed_versions"),
            "note": ("the day is REACHABLE -- what keeps it out of the race "
                     "is the ruling above, not impossibility"),
        }
    return out


def report() -> dict[str, Any]:
    out = {"registry": RACE_WITHDRAWALS,
           "n_withdrawn": len(RACE_WITHDRAWALS)}
    try:
        out["monotone"] = assert_withdrawals_monotone()
    except WithdrawalRefused as e:
        out["monotone"] = {"REFUSED": str(e)}
    for d in sorted(RACE_WITHDRAWALS):
        out.setdefault("forward_reads", {})[d] = forward_read_exists(d)
    return out


# --------------------------------------------------------------- falsifier
def selftest() -> int:
    import tempfile

    checks = 0
    ran_ids: list[str] = []

    def _check_id(label: str) -> str:
        """The stable identity of a check: its leading token."""
        head = str(label).split(":")[0].split(" -- ")[0].strip()
        return head[:32]

    def ok(c, label):
        nonlocal checks
        checks += 1
        ran_ids.append(_check_id(label))
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)
        print(f"PASS: {label}")

    def skip(label):
        ran_ids.append(_check_id(label))
        # A SKIP IS COUNTED AND NAMED. A check that silently vanishes when
        # its precondition is absent makes the total agree while the coverage
        # falls -- the empty-set trap on the suite itself.
        nonlocal checks
        checks += 1
        print(f"SKIP: {label}")

    # ---- the registry itself -------------------------------------------
    ok(set(RACE_WITHDRAWALS) == {"20260829"},
       f"REGISTRY: exactly one day is withdrawn ({sorted(RACE_WITHDRAWALS)}) "
       f"-- 2026-08-29, by R-500 (B)")
    _e = RACE_WITHDRAWALS["20260829"]
    ok("R-500" in _e["authority"] and "USER" in _e["authority"]
       and _e["era_admissible_at_withdrawal"] is True,
       "REGISTRY: the entry carries its authority as DATA and records that "
       "the era IS admissible -- the two facts stay separate, which is what "
       "R-500 (C) corrects")
    ok(all(isinstance(v.get("authority"), str) and v["authority"].strip()
           for v in RACE_WITHDRAWALS.values()),
       "REGISTRY: every entry has a non-empty authority -- the same invariant "
       "round 22 put on ERA_ADMISSIBLE, applied before this table has a "
       "second row to get it wrong on")

    # ---- the FRONT door: a read forecloses a withdrawal ------------------
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        ok(assert_withdrawal_admissible("20260829", d)[
               "admissible_to_withdraw"] is True,
           "FRONT DOOR: with no receipt at all the day may be withdrawn")
        (d / "be_forward_day_receipt_20260829.json").write_text(json.dumps(
            {"outcome": "REFUSED", "refused_at": "day_closed_and_attributed",
             "as_of_utc": "2026-09-03T04:11:16Z"}), encoding="utf-8")
        _r = assert_withdrawal_admissible("20260829", d)
        ok(_r["admissible_to_withdraw"] is True and _r["n_refusals"] == 1
           and _r["n_reads"] == 0,
           "FRONT DOOR: a REFUSED receipt with no sealed file is a GATE "
           "FIRING, not a read -- 08-29's real receipt is exactly this, so "
           "the withdrawal is recorded before any economics are seen")
        # THE REAL RECEIPT'S SHAPE, and the defect it caught. BE's 08-29
        # receipt is `outcome: REFUSED` WITH `"sealed": true` -- a protocol
        # flag, not a score file. Reading that boolean as evidence of a read
        # refused the withdrawal for a reason that was not true, on the first
        # run against the real artifact.
        (d / "be_forward_day_receipt_20260829.v1b.json").write_text(json.dumps(
            {"outcome": "REFUSED", "refused_at": "day_closed_and_attributed",
             "sealed": True}), encoding="utf-8")
        _rb = assert_withdrawal_admissible("20260829", d)
        ok(_rb["admissible_to_withdraw"] is True and _rb["n_reads"] == 0
           and _rb["n_refusals"] == 2
           and all(r["sealed_flag_present"] for r in _rb["refusals"][-1:]),
           "FRONT DOOR: a REFUSED receipt carrying `\"sealed\": true` is "
           "STILL a refusal -- that boolean is BE's protocol flag, not a "
           "score file. Evidence of a read is a PATH or a positive scored "
           "count; the name is not the definition, and reading it as one "
           "refused this very withdrawal on its first real run")
        (d / "be_forward_day_receipt_20260829.v2.json").write_text(json.dumps(
            {"outcome": "SCORED", "sealed_file": "/x/scores.json"}),
            encoding="utf-8")
        try:
            assert_withdrawal_admissible("20260829", d)
            ok(False, "FRONT DOOR: a SCORED receipt must foreclose it")
        except WithdrawalRefused as e:
            ok("already has 1 forward READ" in str(e)
               and "selection on the outcome" in str(e),
               f"FRONT DOOR FALSIFIER: a SCORED receipt REFUSES the "
               f"withdrawal by name ({str(e)[:70]}...) -- withdrawing after "
               f"seeing the economics is selection in the other direction")
        (d / "be_forward_day_receipt_20260829.v3.json").write_text(
            "{not json", encoding="utf-8")
        try:
            assert_withdrawal_admissible("20260829", d)
            ok(False, "an unparseable receipt must count as a read")
        except WithdrawalRefused as e:
            ok("READ" in str(e),
               "FRONT DOOR: an UNPARSEABLE receipt counts as a read -- a file "
               "this cannot interpret is not evidence that nothing happened")
    try:
        assert_withdrawal_admissible("20260829", Path("/nonexistent/derived"))
        ok(False, "an unreadable directory must REFUSE")
    except WithdrawalRefused as e:
        ok("could not look" in str(e),
           "FRONT DOOR: an unreadable directory REFUSES -- 'I could not look' "
           "is not 'nothing was read'")

    # ---- the ONE-WAY property, driven on a real git history --------------
    def _repo(versions: list[str]) -> Path:
        """A throwaway repo whose only file is a registry, committed N times."""
        r = Path(tempfile.mkdtemp())
        _git(["init", "-q", "."], r)
        _git(["config", "user.email", "t@t"], r)
        _git(["config", "user.name", "t"], r)
        (r / "reg.py").parent.mkdir(parents=True, exist_ok=True)
        for i, body in enumerate(versions):
            (r / "reg.py").write_text(body, encoding="utf-8")
            _git(["add", "reg.py"], r)
            _git(["commit", "-q", "-m", f"v{i}"], r)
        return r

    _V1 = ('RACE_WITHDRAWALS = {"20260829": {"authority": "R-500"}}\n')
    _V2_ADD = ('RACE_WITHDRAWALS = {"20260829": {"authority": "R-500"},\n'
               '                    "20260830": {"authority": "R-501"}}\n')
    _V2_DROP = 'RACE_WITHDRAWALS = {}\n'
    _V2_RECITE = ('RACE_WITHDRAWALS = {"20260829": {"authority": "R-999"}}\n')

    _r1 = _repo([_V1])
    _m = assert_withdrawals_monotone(_r1, "reg.py",
                                     {"20260829": {"authority": "R-500"}})
    ok(_m["monotone"] is True and _m["n_prior_versions_with_registry"] == 1
       and _m["vacuous"] is False,
       f"ONE-WAY: an unchanged registry passes against its own history "
       f"({_m['n_prior_versions_with_registry']} prior version(s))")

    _r2 = _repo([_V1, _V2_ADD])
    _m2 = assert_withdrawals_monotone(
        _r2, "reg.py", {"20260829": {"authority": "R-500"},
                        "20260830": {"authority": "R-501"}})
    ok(_m2["monotone"] is True and _m2["n_days_now"] == 2,
       "ONE-WAY: ADDING a day is allowed -- the property is one-way, not "
       "frozen; a later USER ruling can withdraw another day")

    _r3 = _repo([_V1, _V2_DROP])
    try:
        assert_withdrawals_monotone(_r3, "reg.py", {})
        ok(False, "ONE-WAY: a REMOVED withdrawal must refuse")
    except WithdrawalRefused as e:
        ok("REMOVED" in str(e) and "20260829" in str(e)
           and "quietly undo buys nothing" in str(e),
           f"ONE-WAY FALSIFIER -- REMOVAL: dropping a recorded day REFUSES BY "
           f"NAME and names the day ({str(e)[:80]}...)")

    _r4 = _repo([_V1, _V2_RECITE])
    try:
        assert_withdrawals_monotone(_r4, "reg.py",
                                    {"20260829": {"authority": "R-999"}})
        ok(False, "ONE-WAY: a RE-CITED withdrawal must refuse")
    except WithdrawalRefused as e:
        ok("RE_CITED" in str(e) and "R-500" in str(e) and "R-999" in str(e),
           f"ONE-WAY FALSIFIER -- RE-CITE: changing an entry's authority "
           f"REFUSES and prints BOTH cites ({str(e)[:80]}...). Swapping the "
           f"ruling behind a withdrawal is an undo with the row left in place")

    _r5 = _repo(["X = 1\n"])
    _m5 = assert_withdrawals_monotone(_r5, "reg.py", {"20260829": {}})
    # AND THIS CHECK ALSO ASSERTED A PASS OVER ZERO COMPARISONS. It read
    # `monotone is True and vacuous is True` and defended it as "the number
    # says so instead of the boolean implying otherwise" -- but the consumer
    # reads the BOOLEAN (`da_forward_day_verify`: `monotone is not True`), so
    # the number said so to nobody. Second falsifier today found enshrining
    # the defect it was written to catch.
    ok(_m5["monotone"] is None and _m5["vacuous"] is True
       and _m5["n_prior_versions_with_registry"] == 0,
       "ONE-WAY: a history with NO prior registry reports `vacuous: True` "
       "AND `monotone: None` rather than passing -- the introducing commit compares "
       "nothing, and the number says so instead of the boolean implying "
       "otherwise")
    # A REPO WHOSE HISTORY CANNOT BE READ (a git that errors for any other
    # reason -- permissions, a timeout, a corrupt object) is a THIRD state,
    # and it must refuse rather than report NO_REPOSITORY. It is driven by
    # making the `log` call fail while the work-tree test succeeds, because
    # constructing a repo that answers one and not the other by hand would be
    # a fixture pretending to be a condition.
    _real_git = globals()["_git"]
    try:
        globals()["_git"] = (lambda args, cwd=None:
                             (0, "true") if args[:1] == ["rev-parse"]
                             else (128, ""))
        try:
            assert_withdrawals_monotone(_r1, "reg.py", {})
            ok(False, "an unreadable history must REFUSE")
        except WithdrawalRefused as e:
            ok("empty-set trap" in str(e) and "git rc 128" in str(e),
               f"ONE-WAY: a REPOSITORY whose history cannot be read REFUSES "
               f"-- 'monotone' from a history nobody could read is the "
               f"empty-set trap on the guard itself ({str(e)[:70]}...)")
    finally:
        globals()["_git"] = _real_git
    ok(globals()["_git"] is _real_git,
       "ONE-WAY: and the git shim is restored after that falsifier")

    _nr = assert_withdrawals_monotone(Path(tempfile.mkdtemp()), "reg.py",
                                      {"20260829": {}})
    ok(_nr["status"] == "NO_REPOSITORY" and _nr["monotone"] is None,
       "ONE-WAY: a tree that is NOT A GIT REPOSITORY returns status "
       "NO_REPOSITORY with `monotone: None` -- not True. 'No history to "
       "compare' and 'compared and clean' are different answers, and only the "
       "second is a guarantee")
    ok(_nr["monotone"] is not True and _nr["vacuous"] is True,
       "ONE-WAY: and it is explicitly NOT a pass -- the canonical write path "
       "refuses on this status, which is where the guarantee has to hold; an "
       "out-of-tree probe is merely not blocked by it")

    # ---- DA24-R4: WHICH FIELDS MAY MOVE, AND WHICH MAY NOT ---------------
    ok(_entry_fields_are_classified(RACE_WITHDRAWALS["20260829"]) == [],
       "DA24-R4: every key of the shipped entry is classified as either "
       "load-bearing or prose -- an unclassified key is treated as immutable, "
       "so a field added later is guarded until somebody says otherwise "
       "rather than unguarded until somebody notices")
    _V1R = ('RACE_WITHDRAWALS = {"20260829": {"authority": "R-500",\n'
            '                                 "reason": "development read",\n'
            '                                 "note": "first wording"}}\n')
    _r6 = _repo([_V1R])
    _CUR_OK = {"20260829": {"authority": "R-500",
                            "reason": "development read",
                            "note": "first wording"}}
    ok(assert_withdrawals_monotone(_r6, "reg.py", _CUR_OK)["monotone"] is True,
       "DA24-R4 POSITIVE CONTROL: an unchanged entry passes")
    try:
        assert_withdrawals_monotone(_r6, "reg.py", {"20260829": {
            "authority": "R-500", "reason": "re-admitted after review",
            "note": "first wording"}})
        ok(False, "DA24-R4: a rewritten `reason` must refuse")
    except WithdrawalRefused as e:
        ok("RE_CITED" in str(e) and "'field': 'reason'" in str(e)
           and "re-admitted after review" in str(e),
           f"DA24-R4 FALSIFIER: rewriting `reason` -- the field a reader "
           f"QUOTES -- now REFUSES and names the field. It used to pass, "
           f"because the comparison covered `authority` alone, so a "
           f"withdrawal could be silently re-labelled 're-admitted after "
           f"review' with monotone True ({str(e)[:60]}...)")
    _ann = assert_withdrawals_monotone(_r6, "reg.py", {"20260829": {
        "authority": "R-500", "reason": "development read",
        "note": "clarified wording"}})
    ok(_ann["monotone"] is True and len(_ann["annotations_changed"]) == 1
       and _ann["annotations_changed"][0]["field"] == "note",
       "DA24-R4 AND THE OTHER DIRECTION: rewriting `note` -- DA's own "
       "commentary -- does NOT refuse and IS REPORTED. Comparing the whole "
       "entry would make an honest clarification a violation, and a guard "
       "that fires on legitimate maintenance is one people learn to work "
       "around; 'nothing changed' is still never inferred from silence")
    try:
        assert_withdrawals_monotone(_r6, "reg.py", {"20260829": {
            "authority": "R-500", "reason": "development read",
            "note": "first wording", "invented_later": "quietly added"}})
        ok(False, "DA24-R4: an unclassified field must refuse")
    except WithdrawalRefused as e:
        ok("RE_CITED_UNCLASSIFIED" in str(e) and "invented_later" in str(e),
           "DA24-R4b FALSIFIER: a key in NEITHER set is treated as immutable "
           "and refuses -- the default is guarded, so the classification "
           "going stale is loud rather than a silent hole")

    # ---- PHANTOM PASS on the one-way guarantee ---------------------------
    _real_git2 = globals()["_git"]
    try:
        def _shim(args, cwd=None):
            # ANSWERS THE QUESTION IT WAS ASKED, not the family. It used to
            # reply "true" to any `rev-parse`, which was fine while the walk
            # asked only `--is-inside-work-tree`; the moment it also asked
            # `--is-shallow-repository` the fixture started answering "yes,
            # shallow" and the shallow guard fired before the blob ever
            # failed -- a fixture supplying a shape the production path never
            # emits, which is the hazard I was warned about in round 29.
            if args[:2] == ["rev-parse", "--is-inside-work-tree"]:
                return (0, "true")
            if args[:2] == ["rev-parse", "--is-shallow-repository"]:
                return (0, "false")
            if args[:1] == ["replace"]:
                return (0, "")
            if args[:1] == ["log"]:
                return (0, "aaaaaaaaaaaa\n")
            return (128, "")          # every `show` fails
        globals()["_git"] = _shim
        try:
            assert_withdrawals_monotone(_r1, "reg.py", {})
            ok(False, "an unreadable blob must REFUSE, not vanish")
        except WithdrawalRefused as e:
            ok("could not be read" in str(e) and "smaller history" in str(e),
               f"PHANTOM-3 A BLOB THIS CANNOT READ NOW REFUSES: it used to "
               f"`continue`, so the version vanished from the walk and "
               f"`monotone: True` was reported over a SMALLER history than "
               f"exists. A one-way guarantee evaluated over fewer versions "
               f"than exist is not the guarantee ({str(e)[:60]}...)")
    finally:
        globals()["_git"] = _real_git2
    ok(globals()["_git"] is _real_git2,
       "PHANTOM-3b and the git shim is restored")

    # ---- the blob parser, both directions --------------------------------
    ok(_registry_in_blob(_V1) == {"20260829": {"authority": "R-500"}},
       "PARSER: the registry is read from source as DATA (`ast.literal_eval`) "
       "and never executed -- a history walk that imports old versions runs "
       "old code")
    ok(_registry_in_blob("X = 1\n") is None
       and _registry_in_blob("RACE_WITHDRAWALS = {}\n") == {},
       "PARSER: ABSENT and EMPTY are different answers -- None means the name "
       "is not there, {} means it is there and holds nothing, and collapsing "
       "them would make the introducing commit look like a removal")
    # DA32-R1, AND THIS CHECK USED TO ASSERT THE DEFECT. It read "an
    # unparseable blob returns None rather than raising -- a syntax error
    # somewhere in history must not break the guard". But `None` is
    # "the registry is ABSENT from this version", so the version was dropped
    # from the walk and the guarantee was reported over a smaller history.
    # Not breaking the guard is not the same as evaluating it.
    try:
        _registry_in_blob("def f(:\n")
        ok(False, "PARSER: an unparseable blob must REFUSE")
    except BlobUnparseable as e:
        ok("not the same as the registry being ABSENT" in str(e),
           "PARSER (DA32-R1): an unparseable version REFUSES. Returning None "
           "put a fact about the READER inside the codomain of a fact about "
           "the HISTORY, and the version silently left the monotonicity walk")
    try:
        _registry_in_blob("RACE_WITHDRAWALS = f()\n")
        ok(False, "PARSER: a computed registry must REFUSE")
    except BlobUnparseable as e2:
        ok("must not be read as absent" in str(e2),
           "PARSER: a computed registry is still not ACCEPTED -- no "
           "expression smuggles a value past the walk -- but it is now "
           "refused rather than read as a version that had no registry")
    with tempfile.TemporaryDirectory() as _t:
        _r = Path(_t)
        _git(["init", "-q", "."], _r)
        _git(["config", "user.email", "t@t"], _r)
        _git(["config", "user.name", "t"], _r)
        (_r / "reg.py").write_text("RACE_WITHDRAWALS = {}\n", encoding="utf-8")
        _git(["add", "reg.py"], _r)
        _git(["commit", "-q", "-m", "v1"], _r)
        (_r / "reg.py").write_text("RACE_WITHDRAWALS = {\n", encoding="utf-8")
        _git(["add", "reg.py"], _r)
        _git(["commit", "-q", "-m", "v2-broken"], _r)
        try:
            assert_withdrawals_monotone(repo=_r, path="reg.py")
            ok(False, "PARSER: the WALK must refuse on an unparseable version")
        except WithdrawalRefused as e3:
            ok("does not parse" in str(e3) and "commit" in str(e3),
               "PARSER: and THE WALK refuses by commit -- the fix is checked "
               "where the guarantee is made, not only at the parser")
    _rc_none, _msg = _git(["rev-parse", "HEAD"], Path("/nonexistent/repo/x"))
    ok(_rc_none is None,
       "GIT-CODOMAIN (DA32-R1): git that never RAN reports rc None, not 127. "
       "127 is a returncode, and a consumer could not tell a failed git from "
       "an absent one -- the same predicate, one layer under the guarantee")

    # ---- the block a verdict carries -------------------------------------
    _b = withdrawal_block("20260829")
    ok(_b["withdrawn_from_race"] is True
       and "R-500" in _b["authority"]
       and _b["era_admissible_at_withdrawal"] is True,
       "BLOCK: 08-29 carries withdrawn_from_race TRUE with its cite, and "
       "records the era as ADMISSIBLE beside it -- the verdict stops "
       "asserting the thing R-497 (F)(1) made false and carries the true "
       "fact instead")
    _b2 = withdrawal_block("20260901")
    ok(_b2["withdrawn_from_race"] is False and _b2["authority"] is None,
       "BLOCK: an ordinary day carries withdrawn_from_race FALSE -- the "
       "block is present on EVERY day, so absence never has to be "
       "interpreted (rule 4)")
    _b3 = withdrawal_block("20260829", reachability={
        "verdict": "ATTRIBUTED_ALREADY", "attribution_reachable": True,
        "achievable": {"prior_attributed_versions": [{"commit": "4e1133c"}]}})
    ok(_b3["reachability"]["attribution_reachable"] is True
       and "not impossibility" in _b3["reachability"]["note"],
       "BLOCK: when reachability is supplied the block says the day IS "
       "reachable -- so a reader sees the race stayed at 2 by the USER's "
       "CHOICE and not by an accident of file state")

    # ---- round 35: EVERY WAY THE WALK CAN END OR SHORTEN, DRIVEN ---------
    def _repo(d: Path, versions: list[str], name: str = "reg.py") -> Path:
        d.mkdir(parents=True, exist_ok=True)
        _git(["init", "-q", "."], d)
        _git(["config", "user.email", "t@t"], d)
        _git(["config", "user.name", "t"], d)
        for i, body in enumerate(versions):
            (d / name).write_text(body, encoding="utf-8")
            _git(["add", name], d)
            _git(["commit", "-q", "-m", f"v{i}"], d)
        return d

    _cen = walk_termination_census()
    ok(_cen["n_UNACCOUNTED"] == 0 and _cen["n_stale_accountings"] == 0
       and _cen["complete_for_this_method"] is True,
       f"WALK-CENSUS: all {_cen['n_source_routes']} source routes out of the "
       f"walk are DERIVED from the AST and reconciled against a written "
       f"accounting -- {_cen['n_refusing']} refuse, {_cen['n_shortening']} "
       f"shorten (each legitimate and named), {_cen['n_exits']} exit -- with "
       f"{_cen['n_supply_routes']} supply routes declared separately because "
       f"no scan of this function can see them")
    _fake = dict(WALK_ROUTES)
    _fake.pop(("Continue", "reg is None"))
    _saved_routes = dict(WALK_ROUTES)
    try:
        WALK_ROUTES.clear()
        WALK_ROUTES.update(_fake)
        _c2 = walk_termination_census()
        ok(_c2["n_UNACCOUNTED"] == 1
           and _c2["UNACCOUNTED"][0]["condition"] == "reg is None"
           and _c2["complete_for_this_method"] is False,
           "WALK-CENSUS FIRES: drop one route from the accounting and it is "
           "reported UNACCOUNTED. The reconciliation is what makes "
           "'these are all of them' a computed claim rather than a reading "
           "-- and it already caught a `continue` I had not listed")
    finally:
        WALK_ROUTES.clear()
        WALK_ROUTES.update(_saved_routes)
    ok(walk_termination_census()["n_UNACCOUNTED"] == 0,
       "WALK-CENSUS and the accounting is restored")
    try:
        walk_termination_census("no_such_function")
        ok(False, "WALK-CENSUS: an absent walk must REFUSE")
    except WithdrawalRefused as e:
        ok("is not an empty enumeration" in str(e),
           "WALK-CENSUS refuses to enumerate a function that is not there: "
           "zero routes out of a walk that does not exist is not a clean "
           "walk")

    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), [_V1])
        try:
            assert_withdrawals_monotone(repo=_d, path="nosuch.py")
            ok(False, "WALK-S1: a path with no history must REFUSE")
        except WithdrawalRefused as e:
            ok("not anchored at the file's creation" in str(e),
               "WALK-S1 (supply route): a path that names NO file in this "
               "history -- a moved or renamed module, and the default path "
               "is a hardcoded string -- makes `git log` exit 0 with zero "
               "commits. That used to be `monotone: True, vacuous: True`")
    with tempfile.TemporaryDirectory() as t:
        _src = _repo(Path(t) / "src", [_V1, _V1 + "# v2\n", _V1 + "# v3\n"])
        _dst = Path(t) / "shallow"
        _rc_c, _ = _git(["clone", "-q", "--depth", "1",
                         f"file://{_src}", str(_dst)], Path(t))
        if _rc_c != 0 or not (_dst / ".git").exists():
            skip("WALK-S2 needs a working `git clone --depth 1`")
            skip("WALK-S2b needs a working `git clone --depth 1`")
        else:
            _, _n_follow = _git(["log", "--format=%H", "--follow", "--",
                                 "reg.py"], _dst)
            _, _n_add = _git(["log", "--format=%H", "--diff-filter=A",
                              "--follow", "--", "reg.py"], _dst)
            _walked = [c for c in _n_follow.split() if c]
            _added = [c for c in _n_add.split() if c]
            ok(len(_walked) == 1 and _added and _walked[-1] == _added[0],
               f"WALK-S2b THE TWO GUARDS ARE NOT REDUNDANT, shown rather "
               f"than assumed: in a depth-1 clone the graft boundary reports "
               f"the file as ADDED, so the anchor check PASSES over "
               f"{len(_walked)} of 3 commits. The anchor alone would have "
               f"certified a walk over a third of the history")
            try:
                assert_withdrawals_monotone(repo=_dst, path="reg.py")
                ok(False, "WALK-S2: a shallow repository must REFUSE")
            except WithdrawalRefused as e:
                ok("SHALLOW repository" in str(e),
                   "WALK-S2 (supply route): and the shallow guard catches "
                   "what the anchor cannot -- a history truncated by "
                   "construction, proving the guarantee over the part that "
                   "was cloned")
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), [_V1, _V1 + "# later\n"])
        _real3 = globals()["_git"]
        _, _true = _real3(["log", "--format=%H", "--follow", "--", "reg.py"],
                          _d)
        _all = [c for c in _true.split() if c]

        def _lost_rename(args, cwd=None):
            # STANDS IN FOR A RENAME `--follow` DID NOT DETECT: the walk sees
            # a history that begins mid-life, and every commit in it is real.
            if args[:2] == ["log", "--format=%H"] and "--diff-filter=A" \
                    not in args:
                return 0, "\n".join(_all[:-1])
            return _real3(args, cwd)
        try:
            globals()["_git"] = _lost_rename
            assert_withdrawals_monotone(repo=_d, path="reg.py")
            ok(False, "WALK-S3: a truncated walk must REFUSE")
        except WithdrawalRefused as e:
            ok("not anchored at the file's creation" in str(e)
               and "rename `--follow` did not detect" in str(e),
               "WALK-S3 (supply route): a history that silently BEGINS "
               "MID-LIFE is refused. Every commit in it is real and every "
               "blob reads, so nothing inside the loop could have caught it "
               "-- this is the class round 32 and round 34 each closed one "
               "member of, closed from the supply side")
        finally:
            globals()["_git"] = _real3
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), [_V1, _V1 + "# two\n"])
        _rc_r, _ = _git(["replace", "--graft", "HEAD"], _d)
        _, _lst = _git(["replace", "-l"], _d)
        if _rc_r != 0 or not _lst.strip():
            skip("WALK-S4 needs `git replace --graft`")
        else:
            try:
                assert_withdrawals_monotone(repo=_d, path="reg.py")
                ok(False, "WALK-S4: a replace ref must REFUSE")
            except WithdrawalRefused as e:
                ok("REPLACE refs" in str(e),
                   "WALK-S4 (supply route): replace refs rewrite what "
                   "`git log` reports without changing a single commit, "
                   "which is the quiet undo this guarantee exists to make "
                   "loud. The history it shows is complete-looking and "
                   "different")
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), ["# no registry yet\n", _V1])
        _r5 = assert_withdrawals_monotone(
            repo=_d, path="reg.py", current={"20260829": {"authority":
                                                          "R-500"}})
        ok(_r5["monotone"] is True and _r5["n_commits_touching_file"] == 2
           and _r5["n_prior_versions_with_registry"] == 1,
           "WALK-S5 ADMITS the one legitimate shortening: a version that "
           "PREDATES the registry is skipped and COUNTED -- 2 commits "
           "walked, 1 carrying the registry. A guard here would refuse every "
           "repository whose history starts before the ruling")
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), ["RACE_WITHDRAWALS = [1, 2]\n"])
        try:
            assert_withdrawals_monotone(repo=_d, path="reg.py", current={})
            ok(False, "WALK-S6: a non-mapping registry must REFUSE")
        except WithdrawalRefused as e:
            ok("not a mapping" in str(e) and "AttributeError" in str(e),
               "WALK-S6: a registry that is a valid LITERAL but not a "
               "mapping would have died by AttributeError -- a traceback "
               "rather than a named refusal, which tells a reader nothing "
               "about what was being checked")
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), ['RACE_WITHDRAWALS = {"20260829": "R-500"}\n'])
        try:
            assert_withdrawals_monotone(repo=_d, path="reg.py", current={})
            ok(False, "WALK-S7: a non-mapping entry must REFUSE")
        except WithdrawalRefused as e:
            ok("not a mapping" in str(e) and "shown intact" in str(e),
               "WALK-S7: and an ENTRY that is not a mapping is refused by "
               "name for the same reason, one level down")
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), ["# nothing here\n"])
        _r8 = assert_withdrawals_monotone(repo=_d, path="reg.py", current={})
        ok(_r8["monotone"] is None
           and _r8["status"] == "NO_PRIOR_VERSION_WITH_REGISTRY",
           "WALK-S8 (round 34's predicate, applied to this module's OWN "
           "verdict): a walk that compared NOTHING reports `monotone: None`, "
           "not True. `True` was inside the codomain of the measurement "
           "while meaning 'not evaluated', and `da_forward_day_verify` reads "
           "`monotone is not True` -- so it would have emitted a verdict on "
           "a guarantee proved over zero versions")
    _floor = MIN_PRIOR_VERSIONS
    _real_ffr = globals()["floor_from_register"]
    try:
        # DRIVEN ON THE CANONICAL PATH, because that is the only place the
        # floor applies -- raising the pin above the real history is the same
        # shape as the history shrinking below the pin.
        #
        # R512-R1: THE LITERAL AND THE PIN MOVE TOGETHER. This mutated only
        # the literal, which was harmless while the register carried no pin
        # and became wrong the moment one landed: the walk then refused with
        # the DISAGREEMENT message instead of the FLOOR one, so the test
        # failed on a guard that did not exist when it was written. The suite
        # went red 48 seconds after the code landed with no code change in
        # between -- a FILING did it. Mutating both keeps this testing the
        # property it was written for; the disagreement has its own test
        # below, and neither guard is weakened to make a needle match.
        globals()["MIN_PRIOR_VERSIONS"] = 99
        globals()["floor_from_register"] = lambda repo=None: {
            "status": "PINNED", "pinned": 99, "path": "<mutated>",
            "n_blocks": 1, "stray_marker_lines": []}
        assert_withdrawals_monotone()
        ok(False, "WALK-S9: a walk under the floor must REFUSE")
    except WithdrawalRefused as e:
        ok("History was rewritten under a one-way guarantee" in str(e)
           and "pins 99" in str(e),
               f"WALK-S9 (supply route, and the REBASE answer): the "
               f"canonical walk sees fewer versions than the pin and "
               f"REFUSES. `--follow` walks HEAD's "
               f"ancestry only, so a rebase that drops or squashes a commit "
               f"touching this file shortens the walk with NO error, NO "
               f"unreadable blob and NO anchor violation -- the replayed add "
           f"is still an add. The floor is the one thing that survives a "
           f"rewrite, because it is committed in the file being counted")
    finally:
        globals()["MIN_PRIOR_VERSIONS"] = _floor
        globals()["floor_from_register"] = _real_ffr
    # R512-R1: AND THE DISAGREEMENT GETS ITS OWN TEST, by name. It is the
    # MORE important refusal -- a coherent rewrite lowers the literal, and
    # the pin living in another seat's document is what catches it.
    try:
        globals()["MIN_PRIOR_VERSIONS"] = 99
        assert_withdrawals_monotone()
        ok(False, "WALK-S9d: a literal disagreeing with the pin must REFUSE")
    except WithdrawalRefused as e:
        ok("COHERENT-REWRITE signature" in str(e)
           and "the register pins the walk floor at" in str(e),
           f"WALK-S9-DISAGREE (R512-R1): with the register PINNED, moving "
           f"the module literal alone refuses as a COHERENT REWRITE rather "
           f"than as a short walk -- the guard the round-38 pin exists for, "
           f"and the one WALK-S9 was accidentally tripping. It fires only "
           f"because the pin is real: {str(e)[:70]}...")
    finally:
        globals()["MIN_PRIOR_VERSIONS"] = _floor
    ok(assert_withdrawals_monotone()["monotone"] is True,
       "WALK-S9b ADMITS: with the real floor restored the canonical walk "
       "passes -- a guard that only ever refuses is not a guard")
    ok(assert_withdrawals_monotone(repo=CODE_ROOT, path="live/pm_research/"
                                   "da_race_withdrawals.py")["floor_applies"]
       is False,
       "WALK-S9c and the floor does NOT apply to an explicitly-addressed "
       "walk, even one naming the same repo and path: a fixture repository "
       "with one commit is legitimate, and only the CANONICAL walk is a "
       "claim about this file's own history")
    with tempfile.TemporaryDirectory() as t:
        _d = _repo(Path(t), [_V1])
        _real9 = globals()["_git"]

        def _no_remote(args, cwd=None):
            if args[:2] == ["config", "--get"]:
                return 1, ""
            return _real9(args, cwd)
        try:
            globals()["_git"] = _no_remote
            _r9 = assert_withdrawals_monotone(
                repo=_d, path="reg.py",
                current={"20260829": {"authority": "R-500"}})
            ok(_r9["floor_applies"] is False and _r9["origin_seen"] is None,
               "WALK-S9d and a tree with NO REMOTE is not this repository: "
               "the DA20-R2 wiring probe builds a one-commit repo in /tmp "
               "and emits through the PRODUCTION path, so it calls with the "
               "same defaults. A floor keyed on 'the caller used defaults' "
               "fired there and broke a probe that was right -- caught by "
               "running the dependent suite, not by reading")
        finally:
            globals()["_git"] = _real9
    # ---- DA35-R1: the floor pinned where this walk does not count -------
    _fr = floor_from_register()
    ok(_fr["status"] in ("PINNED", "FLOOR_NOT_PINNED_IN_REGISTER",
                         "FLOOR_MARKER_OUTSIDE_PIN_BLOCK",
                         "REGISTER_NOT_FOUND", "REGISTER_UNREADABLE",
                         "FLOOR_PINNED_INCONSISTENTLY"),
       f"DA35-R1 the register pin READS: {_fr['status']}. A missing "
       f"register, an unreadable one and an unpinned one are three "
       f"different facts and none of them is a number")
    with tempfile.TemporaryDirectory() as t:
        _d = Path(t)
        _reg = _d / REGISTER_PATH
        _reg.parent.mkdir(parents=True, exist_ok=True)
        _blk = (FLOOR_BLOCK_BEGIN + "\n" + REGISTER_FLOOR_MARKER + "7\n"
                + FLOOR_BLOCK_END + "\n")
        _reg.write_text(_blk, encoding="utf-8")
        _p1 = floor_from_register(_d)
        ok(_p1["status"] == "PINNED" and _p1["pinned"] == 7
           and _p1["n_blocks"] == 1,
           "DA35-R1 FIRES: a DELIBERATE pin block pins the floor from "
           "OUTSIDE this file, so lowering the module literal in a coherent "
           "rewrite must also alter another seat's document to pass")
        # DA37-R1: THE SHAPE THAT ACTUALLY HAPPENED. My own filing quoted the
        # marker while ASKING for it to be placed, and the first reader
        # returned PINNED before anyone wrote a pin -- documentation of a
        # control silently becoming the control.
        _reg.write_text(f"| Q-DA-235 | DA | place `{REGISTER_FLOOR_MARKER}5` "
                        f"somewhere in this register | OPEN |\n",
                        encoding="utf-8")
        _p2 = floor_from_register(_d)
        ok(_p2["status"] == "FLOOR_MARKER_OUTSIDE_PIN_BLOCK"
           and _p2["pinned"] is None and _p2["stray_marker_lines"] == [1],
           "DA37-R1 FIRES: a FILING that quotes the marker while asking for "
           "it to be placed does NOT pin anything. The register is data read "
           "by an instrument, so naming the token it scans for is a WRITE to "
           "its input -- harmless last round only by luck of the value, and "
           "a wrong number would have guarded the walk at a floor nobody "
           "decided. The stray is reported by LINE so the next one is loud")
        _reg.write_text(FLOOR_BLOCK_BEGIN + "\n" + REGISTER_FLOOR_MARKER
                        + "7\n" + FLOOR_BLOCK_END + "\n"
                        + FLOOR_BLOCK_BEGIN + "\n" + REGISTER_FLOOR_MARKER
                        + "9\n" + FLOOR_BLOCK_END + "\n", encoding="utf-8")
        ok(floor_from_register(_d)["status"] == "FLOOR_PINNED_INCONSISTENTLY",
           "DA35-R1 two different pins is NOT a pin -- it reads as unpinned "
           "with both values named, never as the first one found")
        _reg.write_text(f"| a table row mentioning {FLOOR_BLOCK_BEGIN} and "
                        f"{REGISTER_FLOOR_MARKER}5 and {FLOOR_BLOCK_END} all "
                        f"on ONE line |\n", encoding="utf-8")
        ok(floor_from_register(_d)["status"]
           == "FLOOR_MARKER_OUTSIDE_PIN_BLOCK",
           "DA37-R1b AND THE FORM IS ONE PROSE CANNOT PRODUCE: every entry "
           "in this register is a SINGLE LINE, and a row naming all three "
           "tokens inline pins nothing. Only a deliberate multi-line edit "
           "can, which is what makes the control distinguishable from a "
           "quotation of it")
        _reg.write_text("nothing here\n", encoding="utf-8")
        ok(floor_from_register(_d)["status"] == "FLOOR_NOT_PINNED_IN_REGISTER"
           and floor_from_register(_d)["pinned"] is None,
           "DA35-R1 DEGRADED-SAFE: with no marker the reader returns None "
           "and the walk enforces NOTHING. A guard that refused the moment "
           "it was installed would take the emission path down for a line "
           "that has not landed")
    # ---- DA35-R2: a route out of the walk need not be IN the walk --------
    _cc = walk_termination_census()
    _br = [r for r in _cc["callee_raise_routes"]
           if r["raises"] == "BlobUnparseable"]
    ok(len(_br) >= 1 and all(r["handled_by_the_walk"] for r in _br)
       and _cc["n_callee_raises_unhandled"] == 0,
       f"DA35-R2 the census now FOLLOWS THE CALLEES: "
       f"{_cc['n_callee_raise_routes']} raise route(s) reached through the "
       f"functions the walk calls, including `_registry_in_blob` raising "
       f"BlobUnparseable by design since round 34 -- neither a source route "
       f"nor a supply route, and safe only because the one production "
       f"caller does not swallow it. That safety now rests on the "
       f"enumeration rather than on a caller, and the residue says callees "
       f"are followed ONE level")

    _rr = assert_withdrawals_monotone()
    ok(_rr["monotone"] is True and _rr["anchored_at_creation"] is True
       and _rr["shallow"] is False and _rr["n_replace_refs"] == 0,
       f"WALK-REAL: and on THIS repository the walk is anchored at the "
       f"adding commit {_rr['adding_commit'][:9]}, over "
       f"{_rr['n_commits_touching_file']} commits and "
       f"{_rr['n_prior_versions_with_registry']} versions carrying the "
       f"registry, not shallow, no replace refs -- so the guards changed "
       f"nothing in production, which is how a guard should land")

    # ---- R-529(A): the race frame travels WITH the race fields ----------
    _wb = withdrawal_block("20260829")
    ok("race_reading" in _wb
       and "NEVER A HOLM-CLEARING VERDICT" in _wb["race_reading"]
       and "0.0625" in _wb["race_reading"],
       "R-529(A): every race block this seat emits carries the reading with "
       "it -- DIRECTIONAL AND CONSISTENCY ONLY, with the arithmetic that "
       "makes it so (at G=5, m=2 the best adjusted p is 0.0625 > 0.05). The "
       "fields were never p-values, which is precisely why the frame must be "
       "EMITTED and not assumed: a day count sitting beside a verdict is one "
       "step from being read as evidence")
    ok("race_reading" in withdrawal_block("20260901"),
       "R-529(A) and it travels on a day that was NOT withdrawn too -- the "
       "frame is a property of the race, not of the withdrawal")

    # ---- R512-R1: a pin SHOWN in a fence is a quotation, not a pin ------
    with tempfile.TemporaryDirectory() as t:
        _d2 = Path(t)
        _r2 = _d2 / REGISTER_PATH
        _r2.parent.mkdir(parents=True, exist_ok=True)
        _r2.write_text("```\n" + FLOOR_BLOCK_BEGIN + "\n"
                       + REGISTER_FLOOR_MARKER + "9\n"
                       + FLOOR_BLOCK_END + "\n```\n", encoding="utf-8")
        ok(floor_from_register(_d2)["status"]
           == "FLOOR_MARKER_OUTSIDE_PIN_BLOCK",
           "R512-R1 (DE16-R1 carried over): a pin block SHOWN INSIDE A FENCE "
           "pins nothing -- a fenced sample is exactly where a control gets "
           "documented, which `de_ratification_check` learned before this "
           "marker existed. Ownership, not presence")
        _r2.write_text(FLOOR_BLOCK_BEGIN + "\n" + REGISTER_FLOOR_MARKER
                       + "9\n" + FLOOR_BLOCK_END + "\n", encoding="utf-8")
        ok(floor_from_register(_d2)["pinned"] == 9,
           "R512-R1b ADMITS: the same three lines OUTSIDE a fence still pin, "
           "so the fence rule did not buy immunity by disabling the reader")

    # ---- DA39-R1 (MEM): a COUNT cannot say WHICH checks ran -------------
    import hashlib as _hl0

    def _dig0(ids):
        return _hl0.sha256("\n".join(ids).encode()).hexdigest()[:16]

    _swapped = list(ran_ids)
    _swapped[0] = "SOME-OTHER-CHECK"
    ok(len(_swapped) == len(ran_ids) and _dig0(_swapped) != _dig0(ran_ids),
       f"DA39-R1 (MEM) FIRES: swapping one check id for another leaves the "
       f"COUNT identical ({len(_swapped)} either way) and moves the digest "
       f"-- exactly the case a hand-maintained tally is blind to, in the "
       f"module whose '52 checks passed' was quoted as verification")
    ok(_dig0(ran_ids[:-1]) != _dig0(ran_ids)
       and _dig0(ran_ids + ["X"]) != _dig0(ran_ids),
       "DA39-R1b and it still catches the two a count DID catch -- a "
       "removal and an addition -- so nothing was traded away for it")

    print(f"\nda_race_withdrawals selftest: {checks} checks PASSED")
    import hashlib as _hl

    def _dig(ids):
        return _hl.sha256("\n".join(ids).encode()).hexdigest()[:16]

    _sha = _dig(ran_ids)
    print(f"check-id digest: {_sha} over {len(ran_ids)} ids")
    if _sha != EXPECTED_CHECK_IDS_SHA:
        # THE "WHICH", which a tally cannot give.
        print(f"FAIL: check-id digest {_sha} != pinned "
              f"{EXPECTED_CHECK_IDS_SHA}. A REPLACED check keeps the count "
              f"identical and changes this. Current ids in order:")
        for i, cid in enumerate(ran_ids, 1):
            print(f"   {i:3d}. {cid}")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: EXPECTED_CHECKS={EXPECTED_CHECKS} but {checks} ran.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    r = report()
    print(json.dumps(r, indent=2, sort_keys=True) if a.json else
          json.dumps(r, indent=1, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
