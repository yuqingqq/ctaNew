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
EXPECTED_CHECKS = 32

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


def _git(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd=str(cwd or CODE_ROOT), timeout=120)
        return r.returncode, r.stdout
    except Exception as e:                                   # pragma: no cover
        return 127, str(e)


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


def _registry_in_blob(text: str) -> dict[str, dict] | None:
    """`RACE_WITHDRAWALS` as data, parsed and never executed.

    None means the name is absent from that version -- which is different from
    an empty registry, and the caller must not collapse them.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
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
                except Exception:
                    return None
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
        reg = _registry_in_blob(blob)
        if reg is None:
            continue
        versions.append(c)
        for day, entry in reg.items():
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
    if violations:
        raise WithdrawalRefused(
            f"REFUSED: a recorded race withdrawal has been REMOVED or "
            f"RE-CITED, which the ruling that made it forecloses. "
            f"{violations}. Re-admitting a day whose economics may have been "
            f"seen is selection on the outcome (rule 11); a withdrawal that a "
            f"later commit can quietly undo buys nothing.")
    return {
        "repo": str(repo), "path": path, "status": "READ",
        "n_commits_touching_file": len(commits),
        "n_prior_versions_with_registry": len(versions),
        "prior_versions": versions,
        "n_days_now": len(cur), "days_now": sorted(cur),
        "monotone": True,
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

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)
        print(f"PASS: {label}")

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
    ok(_m5["monotone"] is True and _m5["vacuous"] is True
       and _m5["n_prior_versions_with_registry"] == 0,
       "ONE-WAY: a history with NO prior registry reports `vacuous: True` "
       "rather than passing silently -- the introducing commit compares "
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
            if args[:1] == ["rev-parse"]:
                return (0, "true")
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
    ok(_registry_in_blob("def f(:\n") is None,
       "PARSER: an unparseable blob returns None rather than raising -- a "
       "syntax error somewhere in history must not break the guard")
    ok(_registry_in_blob("RACE_WITHDRAWALS = f()\n") is None,
       "PARSER: a computed registry is NOT accepted -- if it cannot be read "
       "as a literal it is not read at all, so no expression can smuggle a "
       "value past the history walk")

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

    print(f"\nda_race_withdrawals selftest: {checks} checks PASSED")
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
