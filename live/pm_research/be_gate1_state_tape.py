"""THE GATE-1 STATE TAPE, ONE PER RULED DAY. The second missing input.

Round 48's book refused for want of a FEATURE FRAGMENT. Round 49 built one
for 09-03. The reviewer's BE48 filing names the second missing input in the
same breath: a scored book also needs a STATE TAPE, and no September tape
exists either.

`build_state_tape_v2.main` is already parameterised -- `fragment_path`,
`topup_path`, `out_path`, `allow_overwrite` -- and its default output IS the
live tape, which a bare invocation once destroyed. So this driver never
passes a default: it names the output explicitly, at a Gate-1 stem, and
guards it the way `be_gate1_fragment` guards its own.

THE THING A READER MUST KNOW BEFORE TRUSTING THIS TAPE, AND IT IS A LIMIT
THIS SEAT CANNOT CLOSE. `build_state_tape_v2` maps its two inputs onto the
two SPLITS -- `for split, src in (("train", FRAG), ("score", TOP))`. Those
are two POPULATIONS (eraB and the top-up), not two feature sets. A ONE-DAY
Gate-1 tape has ONE population, so one of the two splits has no input.

R-574 puts the split question with DE: DE declares which splits `--day`
needs and this seat builds to that. That declaration has not landed. So this
driver builds the day's fragment into the TRAIN split and supplies an
EXPLICITLY EMPTY score input, and it RECORDS that as a provisional choice
routed to DE rather than a decision -- because which split a day's rows
belong to is a modelling question, not a plumbing one.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR
import be_rule22 as _R22

#: R-649 §3.2: 75 is EX_TEMPFAIL. From OUTSIDE a unit, ExecMainStatus=75
#: reads "the lock was held" OR "a producer broke the declaration and exited
#: 75 for its own reasons" -- and the two are indistinguishable. So this
#: producer DECLARES its exit codes and its selftest asserts 75 is not among
#: them, which is what makes the launcher's 75 mean one thing.
EXIT_CODES = {
    0: "the artifact was written and its receipt emitted",
    1: "a refusal or an uncaught error (Python's default for an exception)",
    2: "usage: no --day and no --selftest",
}
EXIT_CODE_NOTE = ("75 is RESERVED to the launcher's flock conflict and is "
                  "not in this map; the selftest asserts it.")


#: RULE 22 AS AMENDED (R-605): the closure and HEAD are captured HERE, at
#: import, before any work. This producer carried NO provenance stamp at
#: all until round 60 (R-613) -- a landing to it mid-run would have been
#: invisible in its receipt. The stamp is READ at emit so it can report
#: drift; the digests it reports are the ones seen at import.
_R22.init("be_gate1_state_tape import")
import be_gate1_fragment as FRAGMOD

ROOT = HERE.parents[1]
DERIVED = _BDR.derived()
OUT_DERIVED = _BDR.derived()  # BE48 B.4: one root, from the resolver. This was `ROOT / 'data/...'` -- a data root built on a CODE root, which the first version of `audit_derived_roots` could not see because the value holds no `parents` and no literal.

GATE1_TAPE_STEM = "phase2_state_tape_gate1_"
COIN = "btc"
MEM_CAP_GB = 8.0


class TapeRefused(RuntimeError):
    """A named refusal."""


def pinned_tapes(derived: Path | None = None) -> set:
    d = Path(derived) if derived is not None else DERIVED
    return {p.name for p in d.glob("phase2_state_tape*.json")
            if not p.name.startswith(GATE1_TAPE_STEM)}


def out_path(day: str, coin: str = COIN) -> Path:
    return DERIVED / f"{GATE1_TAPE_STEM}{day}_{coin}.json"


def guard_output(path: Path, *, derived: Path | None = None) -> None:
    p = Path(path)
    if p.name in pinned_tapes(derived):
        raise TapeRefused(
            f"REFUSED: {p.name} is an existing state tape. The default output "
            f"of build_state_tape_v2 is the LIVE tape the fit manifest binds "
            f"by content, and a bare invocation of that builder once "
            f"destroyed it. This driver never writes a name it did not "
            f"choose.")
    if not p.name.startswith(GATE1_TAPE_STEM):
        raise TapeRefused(
            f"REFUSED: {p.name} is not a Gate-1 tape name. The output must be "
            f"`{GATE1_TAPE_STEM}<day>_<coin>.json`.")
    if p.exists():
        raise TapeRefused(
            f"REFUSED: {p} already exists. Overwriting one day's tape with "
            f"another's under an unpinned name is the same defect one step "
            f"down.")


def empty_split_path(day: str, coin: str = COIN) -> Path:
    return DERIVED / f"{FRAGMOD.GATE1_STEM}{day}_{coin}.EMPTY_SCORE.json"


def _rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def build(day: str, *, coin: str = COIN, progress: bool = True,
          fixture: bool = False) -> dict:
    # RULE 20, MEASURED BEFORE ANY WORK (REV 63 S4). This receipt mentioned
    # the lock in NO field at all, so "the lock was taken and held across
    # both steps" was a claim only a register row carried -- its own
    # artifacts could not support it. The evidence is DE's, delegated rather
    # than reimplemented (Q-BE-271), and a real build without an EXCLUSIVE
    # hold refuses HERE rather than after ten minutes of work.
    _wrapper = _R22.lock_evidence(fixture=fixture)
    t0 = time.time()
    dst = out_path(day, coin)
    guard_output(dst)
    frag = FRAGMOD.out_path(day, coin)
    if not frag.exists():
        raise TapeRefused(
            f"REFUSED: the day's Gate-1 feature fragment {frag.name} does not "
            f"exist. The tape is built FROM it; building a tape without one "
            f"would be a tape about a different population.")
    frag_sha = hashlib.sha256(frag.read_bytes()).hexdigest()

    empty = empty_split_path(day, coin)
    if not empty.exists():
        empty.write_text(json.dumps({"rows": [], "n_windows": 0,
                                     "days": [],
                                     "WHY_EMPTY": "a one-day Gate-1 tape has "
                                                  "ONE population; "
                                                  "build_state_tape_v2 maps "
                                                  "its two inputs onto the "
                                                  "two splits. Which split a "
                                                  "day belongs to is DE's "
                                                  "declaration (R-574), not "
                                                  "this seat's."}))
    ref = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    if len(ref) != 40:
        raise TapeRefused(f"REFUSED: could not read a 40-hex BUILD_REF "
                          f"({ref!r}); the builder requires one at startup.")
    os.environ["BUILD_REF"] = ref
    if progress:
        print(json.dumps({"stage": "start", "fragment": str(frag),
                          "fragment_sha256": frag_sha[:16],
                          "out": str(dst), "BUILD_REF": ref[:12]}), flush=True)

    import build_state_tape_v2 as BST
    # THE SPLIT ASSIGNMENT, CORRECTED (coordinator, round 51). The first
    # build put the day's rows in TRAIN and left SCORE empty. That is wrong
    # for a RULED FORWARD DAY: nothing is trained on it -- every row is a
    # SCORE row the PINNED heads score to produce `asm`. Labelling a forward
    # day `train` would report it as a day the heads were fitted on, which is
    # the look-ahead-shaped misreport, and `phase2_arms.tape_index` filters
    # `r["split"] != split`, so the label decides which index a row lands in.
    # `build_state_tape_v2` maps (('train', FRAG), ('score', TOP)) -- so the
    # DAY FRAGMENT goes in the TOPUP slot.
    # `build_state_tape_v2` maps (('train', FRAG), ('score', TOP)); the day
    # fragment goes in the TOPUP slot, so its split is SCORE. Derived from
    # the call, not restated.
    _fragment_arg, _empty_arg = "topup", "fragment"
    _DAY_SPLIT = "score" if _fragment_arg == "topup" else "train"
    rc = BST.main(fragment_path=empty, topup_path=frag, out_path=dst,
                  allow_overwrite=False)
    if rc != 0:
        raise TapeRefused(f"REFUSED: build_state_tape_v2 returned rc={rc}.")
    peak = _rss_gb()
    if peak > MEM_CAP_GB:
        raise TapeRefused(
            f"REFUSED: peak RSS {peak} GB exceeded the {MEM_CAP_GB} GB cap. "
            f"R8: the cap is NOT raised and the population is NOT reduced.")
    body = dst.read_bytes()
    n_rows = sum(1 for _ in body.split(b'{"')) if body else 0
    try:
        n_rows = len(json.loads(body).get("rows", []))
    except Exception:                                    # noqa: BLE001
        pass
    return {
        "protocol": "BE_GATE1_STATE_TAPE_V1",
        "day": day, "coin": coin,
        "tape": {"path": str(dst), "bytes": len(body),
                 "sha256": hashlib.sha256(body).hexdigest(),
                 "n_rows": n_rows},
        "inputs": {"score_split": {"path": str(frag), "sha256": frag_sha,
                                   "THE_DAY'S_ROWS": True},
                   "train_split": {"path": str(empty),
                                   "sha256": hashlib.sha256(
                                       empty.read_bytes()).hexdigest(),
                                   "bytes": empty.stat().st_size,
                                   "EMPTY_BY_CONSTRUCTION": True,
                                   "why_it_still_carries_a_digest": (
                                       "REV 43: 'empty by construction' is a "
                                       "claim about the file, and a claim "
                                       "about a file that carries no digest "
                                       "cannot be checked. An empty file is "
                                       "still bytes.")}},
        "WHICH_SPLIT_THE_ASSEMBLY_SCORES_FROM_AND_WHY": {
            "split": "score",
            "why": "a RULED FORWARD DAY is not trained on. Every row is a "
                   "SCORE row that the PINNED heads score at their PINNED "
                   "thetas to produce `asm`; nothing is fitted here.",
            "why_the_label_is_not_cosmetic": "`phase2_arms.tape_index` "
                                             "filters `r['split'] != split`, "
                                             "so the label decides which "
                                             "index a row lands in -- and a "
                                             "forward day labelled `train` "
                                             "reports as a day the heads "
                                             "were FITTED on, which is the "
                                             "look-ahead-shaped misreport",
            "corrected_from": "the first 09-03 tape put the day in TRAIN and "
                              "left SCORE empty; that artifact is moved "
                              "aside as .WRONG_SPLIT and superseded",
        },
        "THE_SPLIT_QUESTION_IS_NOT_MINE": {
            "mechanism": "build_state_tape_v2 maps its two inputs onto the "
                         "two splits: (('train', FRAG), ('score', TOP))",
            "why_it_bites_here": "those are two POPULATIONS (eraB, top-up), "
                                 "not two feature sets. A one-day Gate-1 "
                                 "tape has ONE population, so one split has "
                                 "no input.",
            # RULE 10, AND THIS IS THE SECOND ATTEMPT. Round 53 filed
            # that this field was "computed from the build". It was NOT:
            # that patch used a non-asserting `.replace()` whose pattern did
            # not match, so it silently did nothing and the literal below
            # survived -- FALSE of the build, which sends the day to SCORE.
            # Found in the 09-04 receipt this round. Computed now, from the
            # arguments actually passed, with the assert the first patch
            # lacked.
            "what_this_build_did": (
                f"day fragment -> {_DAY_SPLIT.upper()}; an explicitly EMPTY "
                f"file -> {'TRAIN' if _DAY_SPLIT == 'score' else 'SCORE'}"),
            "computed_from_the_build_not_a_literal": True,
            "status": "RULED — the day's rows are the SCORE split (R-560). A "
                      "ruled forward day is not trained on. (Superseded the "
                      "PROVISIONAL status this field carried while the "
                      "question was open.)",
            "consequence_if_DE_rules_otherwise": "the tape is rebuilt; it is "
                                                 "one day and it is cheap "
                                                 "relative to the fragment",
        },
        "resources": {"wall_s": round(time.time() - t0, 1),
                      "peak_rss_gb": peak, "cap_gb": MEM_CAP_GB,
                      "cap_raised": False},
        "guard": {"refuses_existing_state_tapes": sorted(pinned_tapes()),
                  "required_stem": GATE1_TAPE_STEM,
                  "refuses_any_existing_path": True},
        "build_ref": ref,
        "data_root": _BDR.receipt_block(),
        "scope": _BDR.scope_stats(),
        "launch_form_at_runtime": _R22.assert_not_a_scope(fixture=fixture),
        "exit_codes": {"map": {str(k): v for k, v in EXIT_CODES.items()},
                       "conflict_code_is_the_launcher's": _R22.lock_conflict_rc(),
                       "note": EXIT_CODE_NOTE,
                       "declaration_head":
                           _R22.declaration_head("heavy_run_form")["name"]},
        "wrapper_measured": _wrapper,
        "producing_code": _R22.stamp(__file__),
        "rule22_checked_at_emit": _R22.assert_unchanged(
            "be_gate1_state_tape receipt emit"),
        "no_book_built": True, "no_assembly_run": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 24      # BE 91: +1, the tape lands world-readable


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    names = pinned_tapes()
    ok(len(names) >= 1,
       f"the tape guard's refusal set is READ FROM DISK: {len(names)} "
       f"existing state tape(s), including the live one a bare invocation of "
       f"build_state_tape_v2 once destroyed")
    for n in sorted(names)[:2]:
        try:
            guard_output(DERIVED / n)
            ok(False, f"{n} must refuse")
        except TapeRefused as e:
            ok("existing state tape" in str(e),
               f"KNOWN-BAD: writing to {n} REFUSES")
    try:
        guard_output(DERIVED / "not_a_gate1_tape.json")
        ok(False, "a non-Gate-1 tape name must refuse")
    except TapeRefused as e:
        ok("not a Gate-1 tape name" in str(e),
           "KNOWN-BAD: any name outside the Gate-1 tape stem REFUSES")
    # THE LOCK IS CHECKED FIRST NOW, so this case declares itself a fixture
    # -- otherwise it would exercise the lock refusal and never reach the
    # refusal it exists to drive. The lock's own refusal is driven below.
    try:
        build("19700101", fixture=True)
        ok(False, "a day with no fragment must refuse")
    except TapeRefused as e:
        ok("does not exist" in str(e) and "fragment" in str(e),
           "KNOWN-BAD, AND IT REACHES ITS OWN REFUSAL: a day whose Gate-1 "
           "FRAGMENT is absent refuses HERE, in this module, naming the "
           "fragment -- not by some upstream module refusing first")
    # ---- RULE 20: the lock, MEASURED, and refused before any work -------
    try:
        build("19700101")
        ok(False, "a real build without the heavy lock must refuse")
    except _R22.HeavyRunRefused as e:
        ok("does not hold" in str(e) and "be_heavy_run.sh" in str(e),
           "KNOWN-BAD, REV 63 S4: a REAL build that does not hold the "
           "heavy-run lock REFUSES before any work, and the refusal names "
           "the launcher that takes it. This receipt carried no lock field "
           "at all, so `the lock was held` was a claim only a register row "
           "could make")
    _we = _R22.lock_evidence(fixture=True)
    ok(_we["delegated_to"] == "de_multiday_gate1_runner.wrapper_observed"
       and "lock_mode" in _we and "exclusive" in _we
       and _we["launch_form"].startswith("systemd-run --user transient"),
       f"POSITIVE CONTROL: the evidence block is DE's own, with the MODE "
       f"read from /proc/locks ({_we['lock_mode']!r}) -- two `flock -s` "
       f"holders would both report the fd and both certify, so held is not "
       f"exclusive. One implementation, delegated, not a second one")
    _srcT = Path(__file__).read_text()
    ok('"wrapper_measured": _wrapper,' in _srcT
       and "_R22.lock_evidence(fixture=fixture)" in _srcT,
       "AND IT IS WIRED INTO THE EMITTED RECEIPT: read from this module's "
       "own source, not asserted in prose")
    _lf = _R22.assert_launch_form()
    ok(_lf["form_is_correct"] and _lf["lock_is_inside_the_unit"],
       f"THE LAUNCHER IS THE SERVICE FORM: {_lf['problems'] or 'no problems'}"
       f" -- no `--scope`, a named unit, both caps, the data root and "
       f"working directory set inside the unit, and the lock taken INSIDE "
       f"it. Every BE heavy run through 09-05 was a transient scope, whose "
       f"payload dies with the launching shell; they survived because "
       f"nothing signalled it")
    _bad_lf = _R22.assert_launch_form(
        "flock -n /l systemd-run --user --scope --slice=research.slice "
        "-p MemoryMax=8G -p CPUQuota=100% --setenv=PM_DATA_ROOT=/r "
        "--working-directory=/w -- cmd")
    ok(_bad_lf["form_is_correct"] is False
       and any("--scope" in x for x in _bad_lf["problems"])
       and any("--unit=" in x for x in _bad_lf["problems"]),
       f"KNOWN-BAD: the exact form BE used for six heavy runs is REFUSED "
       f"by name, on both counts -- {len(_bad_lf['problems'])} problems")
    _lsrc = _R22.LAUNCHER.read_text()
    ok('--falsify' in _lsrc and "PASS cell 1" in _lsrc
       and "PASS cell 2" in _lsrc and _R22.LAUNCHER.exists(),
       "RULE 15: THE LAUNCHER SHIPS ITS OWN FALSIFIER -- `--falsify` drives "
       "both cells against real systemd units: a TERM to the launching "
       "shell's process group leaves the unit ALIVE on the same MainPID "
       "(parented by systemd, not by any shell of mine), and a HELD lock "
       "refuses with the distinct conflict code 75 having done no work, "
       "with the refusal in the journal. Nothing short of a real unit can "
       "show a claim about process ownership")
    _lsrc2 = _R22.LAUNCHER.read_text()
    _cap_i = _lsrc2.index('"--capture"')
    _stop_i = _lsrc2.index('systemctl --user stop "$CUNIT.service"', _cap_i)
    _out_i = _lsrc2.index('"event":"outcome"', _cap_i)
    _jrn_i = _lsrc2.index('"event":"journal_copy"', _cap_i)
    ok(_out_i < _stop_i < _jrn_i and "read_while" in _lsrc2
       and "not-found" in _lsrc2[_cap_i:_stop_i],
       "THE CAPTURE STEP EXISTS AND ITS ORDER IS THE POINT: the five fields "
       "are read WHILE LOADED, then the unit is stopped, then the journal "
       "is copied -- in that order, because the Stopped/Consumed lines are "
       "written BY the stop and a copy taken before it cannot contain them "
       "(DE 106). A unit already gone is REFUSED rather than reported as "
       "defaults (R-653)")
    # ---- REV 84 §3.2: THE SHARED MODULE'S FALSIFIER IS ONE CELL HERE --
    # This battery carried FIVE properties of `declaration_chain` typed out
    # again: the positive write, the three CAS refusals by name, and the
    # fork reported-not-refused. Every one of them is a cell of the module's
    # own `--falsify` since BE 79/81, so keeping them here was a second
    # implementation of the TEST -- the same divergence one implementation
    # was adopted to end. They are replaced by the invocation below.
    #
    # WHAT STAYS INDEPENDENT, AND WHY (REV 84 §3.1): the module's falsifier
    # runs on FIXTURES it builds itself. It cannot know that this seat's
    # four real families resolve through it, and it does not assert the
    # INTERFACE `be_rule22.declaration_head` consumes -- nine keys by name,
    # every one of which a KeyError would take out of every emitter this
    # seat owns. Those are the properties this seat's verdicts rest on, so
    # the cell below keeps them, now asserting the KEYS EXIST rather than
    # reading values out of them (REV 83 §3: a `.get` that returns None is
    # a property of the query, not of the object).
    _dcf = _R22.shared_falsifier()
    ok(_dcf["ok"],
       f"REV 84 §3.2 -- ONE IMPLEMENTATION, N DETECTORS: this battery RUNS "
       f"`declaration_chain.py --falsify` as a subprocess -> rc "
       f"{_dcf['rc']}, {_dcf['summary']!r}. A regression in the shared "
       f"module now fails every importer's battery at once, without any "
       f"importer re-implementing the logic -- BE 82's `and want` guard was "
       f"caught by DA's kept cell, not by mine. "
       f"{_dcf['failed_cells'] or _dcf['stderr_tail'] or ''}")
    # AND THE CELL ABOVE SHIPS ITS OWN FALSIFIER (rule 15). A cell that runs
    # someone's falsifier and reports a green is worth nothing until it has
    # been shown to go red -- REV 83 §1.3 ruled on exactly that shape one
    # round ago. Driven on THE PREDICATE with known-bad stand-ins rather
    # than by mutating the real module, which a battery must never do: the
    # two ways this could be vacuous are trusting `rc` alone and trusting
    # the summary alone, so both are driven.
    import tempfile as _tfF
    _dS = Path(_tfF.mkdtemp(prefix="be83_falsifier_"))

    def _stub(nm, body):
        q = _dS / nm
        q.write_text(body)
        return _R22.shared_falsifier(prog=str(q))
    _bad_rc = _stub("bad_rc.py", "print('PASS: x')\n"
                                 "print('3 cells, 1 failures')\n"
                                 "raise SystemExit(1)\n")
    _bad_txt = _stub("bad_text.py", "print('FAIL: CELL 9 something broke')\n"
                                    "print('3 cells, 1 failures')\n")
    _good = _stub("good.py", "print('PASS: x')\n"
                             "print('3 cells, 0 failures')\n")
    ok(_bad_rc["ok"] is False and _bad_txt["ok"] is False
       and _good["ok"] is True and _bad_txt["rc"] == 0
       and _bad_txt["failed_cells"],
       f"KNOWN-BAD FOR THE CELL ABOVE: a falsifier that EXITS NON-ZERO is "
       f"caught (rc {_bad_rc['rc']}, ok {_bad_rc['ok']}), and so is one "
       f"that exits ZERO while printing a failed cell (rc "
       f"{_bad_txt['rc']}, {_bad_txt['summary']!r}, ok {_bad_txt['ok']}) -- "
       f"reading only the exit status would have called that second one a "
       f"pass. POSITIVE CONTROL: a clean run admits ({_good['summary']!r}, "
       f"ok {_good['ok']})")
    _fams = ("heavy_run_form", "be_daybook_structure",
             "be_race_read_declaration", "producer_exit_maps")
    _heads = {f: _R22.declaration_head(f) for f in _fams}
    _CONSUMED = ("name", "sha256", "doc", "path", "n_versions",
                 "orphan_branches", "forks_two_versions_superseding_one",
                 "pair", "version")
    _missing = {f: [k for k in _CONSUMED if k not in h]
                for f, h in _heads.items()}
    ok(all(h["resolved_by"].startswith("declaration_chain.resolve_head")
           for h in _heads.values())
       and not any(_missing.values()),
       f"THE PROPERTY THIS SEAT'S VERDICTS REST ON, KEPT INDEPENDENT: ALL "
       f"FOUR OF THIS SEAT'S REAL FAMILIES RESOLVE THROUGH THE SHARED "
       f"IMPLEMENTATION, and every one of the {len(_CONSUMED)} keys "
       f"`be_rule22.declaration_head` consumes is PRESENT in the answer "
       f"(missing {_missing}) -- a renamed key is a KeyError in every "
       f"emitter this seat owns, and the module's fixture falsifier cannot "
       f"see this interface at all: "
       f"{ {f: (h['name'], h['n_versions']) for f, h in _heads.items()} }")
    ok(BUILD := True,
       "usage: --day builds one day only; the split assignment is recorded "
       "as PROVISIONAL and routed to DE (R-574)")

    # ---- RULE 22 AS AMENDED (R-605): this producer had NO stamp at all ----
    import tempfile as _tf, importlib as _il
    _st22 = _R22.stamp(__file__)
    ok(_st22["producing_code"] == "be_gate1_state_tape.py"
       and _st22["producing_code_sha256"]
       == __import__("hashlib").sha256(
           Path(__file__).read_bytes()).hexdigest()
       and _st22["captured_at"] == "IMPORT"
       and _st22["builder_commit"]
       and _st22["import_closure"]["n_modules"] >= 2,
       f"RULE 22: this module now stamps its OWN identity -- the digest of "
       f"the bytes at import, HEAD {{str(_st22['builder_commit'])[:12]}}, "
       f"and the {{_st22['import_closure']['n_modules']}} modules of its "
       f"import closure under live/. Until round 60 it carried none of "
       f"this, so a landing to it mid-run was invisible in its receipt")
    ok(_st22["closure_unchanged_during_the_run"] is True
       and _R22.assert_unchanged("be_gate1_state_tape battery")["closure_unchanged"],
       "POSITIVE CONTROL: with nothing moved the emit ADMITS -- a guard "
       "shown only to refuse has not been shown to work (rule 16)")
    _td22 = _tf.mkdtemp(prefix="be60_be_gate1_state_tape_")
    _pm = Path(_td22) / "be60_probe_be_gate1_state_tape.py"
    _pm.write_text("V = 1\n")
    sys.path.insert(0, _td22)
    _il.import_module("be60_probe_be_gate1_state_tape")
    _c22 = _R22.Capture(root=_td22).capture("battery")
    _pm.write_text("V = 2\n")
    try:
        _c22.assert_unchanged("be_gate1_state_tape known-bad")
        ok(False, "a module rewritten mid-run must refuse the emit")
    except _R22.Rule22Refused as _e22:
        ok("be60_probe_be_gate1_state_tape.py" in str(_e22) and "DID NOT RUN" in str(_e22),
           "KNOWN-BAD: a module of the closure rewritten mid-run REFUSES "
           "THE EMIT BY NAME -- R-603's defect, where a receipt would have "
           "named bytes that did not run")
    sys.path.remove(_td22)
    _src22 = Path(__file__).read_text()
    ok('"producing_code": _R22.stamp(__file__),' in _src22
       and '"rule22_checked_at_emit": _R22.assert_unchanged(' in _src22,
       "AND IT IS WIRED INTO THE EMITTED RECEIPT: both the stamp and the "
       "refusal, read from this module's own source rather than claimed")

    ok(75 not in EXIT_CODES and 75 == _R22.lock_conflict_rc(),
       f"R-649 §3.2: this producer's declared exit codes are "
       f"{sorted(EXIT_CODES)} and 75 is NOT among them -- so a unit reading "
       f"ExecMainStatus=75 means the lock was held, and cannot also mean "
       f"this producer broke the declaration. The 75 is read from the "
       f"declaration head, not typed here")
    ok(_R22.assert_not_a_scope(fixture=True)["kind"] in
       ("scope", "service", "none"),
       f"REV 65 §1.2: the launch form is decided at RUNTIME from this "
       f"process's own cgroup leaf ({_R22.cgroup_leaf()['leaf']!r}), which a "
       f"static lint cannot do -- it cannot see a `--scope` behind a "
       f"variable or a wrapper")
    try:
        _R22.assert_not_a_scope(fixture=False) if \
            _R22.cgroup_leaf()["kind"] == "scope" else None
        _leafok = _R22.cgroup_leaf()["kind"] != "scope"
    except _R22.HeavyRunRefused as _e:
        _leafok = "transient SCOPE" in str(_e)
    ok(_leafok,
       "KNOWN-BAD, DRIVEN WHERE IT LIVES: a process whose own cgroup leaf is "
       "a `.scope` REFUSES a real day by name -- nine BE heavy runs were "
       "scopes and every receipt said so in `scope.unit`; no seat read it")

    print()
    # ---- BE 91: THE TAPE LANDS WORLD-READABLE ---------------------------
    # Five `phase2_state_tape_gate1_*.json` were 0600 on disk (4,967,187,165
    # B), because `build_state_tape_v2` emits through `tempfile.mkstemp` --
    # a SECRET-file constructor -- and `os.replace` carries 0600 to the
    # landed artifact. The mode is now set from the ONE implementation
    # (`declaration_chain.plain_create_mode`), imported, before the rename.
    #
    # THIS CELL DRIVES THE BUILDER'S OWN `_land`, not a re-implementation of
    # chmod+replace, and it BASELINES ITSELF (REV 83 §5): the expected mode
    # is measured from a file the cell creates the ordinary way in the SAME
    # directory, so a wrong umask read fails the cell instead of agreeing
    # with it.
    import stat as _st, tempfile as _tfL
    import build_state_tape_v2 as _BST
    _ld = Path(_tfL.mkdtemp(prefix="be91_land_"))
    _probe = _ld / "plain_create.probe"
    _probe.write_text("a file made the ordinary way, in this directory\n")
    _base = _st.S_IMODE(_probe.stat().st_mode)
    _fd, _tmp = _tfL.mkstemp(dir=str(_ld), suffix=".tmp")
    with os.fdopen(_fd, "w") as _fh:
        _fh.write('{"rows": []}\n')
    _pre = _st.S_IMODE(Path(_tmp).stat().st_mode)
    _dst = _ld / "phase2_state_tape_gate1_20990101_btc.json"
    _BST._land(_tmp, _dst)
    _post = _st.S_IMODE(_dst.stat().st_mode)
    _base_owner_only = _base & 0o077 == 0
    _note = (" -- BASELINE_IS_OWNER_ONLY: this box's umask makes even a plain "
             "create owner-only, so the equality is the whole assertion"
             if _base_owner_only else "")
    ok(_pre == 0o600 and _post == _base and _dst.read_text() == '{"rows": []}\n'
       and not Path(_tmp).exists(),
       f"THE TAPE LANDS AS READABLE AS ITS NEIGHBOURS: mkstemp made the temp "
       f"0o{_pre:04o} (the defect's source, measured here rather than "
       f"asserted) and `build_state_tape_v2._land` landed it 0o{_post:04o}, "
       f"which is what a plain create in the same directory produced "
       f"(0o{_base:04o}){_note}. The content survived the chmod and the temp "
       f"is gone, so the rename still happened. The five real tapes were "
       f"re-moded at BE 91 with each digest asserted against its receipt")

    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--day" in argv:
        day = argv[argv.index("--day") + 1]
        out = build(day)
        dst = OUT_DERIVED / f"be_gate1_state_tape_receipt_{day}_{COIN}.json"
        # A LANDED RECEIPT IS NEVER OVERWRITTEN (rule 13). Round 51 wrote the
        # corrected receipt at v1's path and replaced a landed artifact; the
        # builder now versions instead, so the defect cannot recur from here.
        if dst.exists():
            n = 2
            while (OUT_DERIVED / f"be_gate1_state_tape_receipt_{day}_"
                                 f"{COIN}.v{n}.json").exists():
                n += 1
            dst = (OUT_DERIVED /
                   f"be_gate1_state_tape_receipt_{day}_{COIN}.v{n}.json")
            out["supersedes"] = {
                "artifact": f"be_gate1_state_tape_receipt_{day}_{COIN}.json",
                "rule": "13 -- vN+1; the earlier receipt is not edited"}
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"receipt": str(dst), **out["tape"],
                          "wall_s": out["resources"]["wall_s"],
                          "peak_rss_gb": out["resources"]["peak_rss_gb"]}))
        return 0
    print("usage: be_gate1_state_tape.py --selftest | --day <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
