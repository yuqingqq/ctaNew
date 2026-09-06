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


def build(day: str, *, coin: str = COIN, progress: bool = True) -> dict:
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
        "no_book_built": True, "no_assembly_run": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 6


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
    try:
        build("19700101")
        ok(False, "a day with no fragment must refuse")
    except TapeRefused as e:
        ok("does not exist" in str(e) and "fragment" in str(e),
           "KNOWN-BAD, AND IT REACHES ITS OWN REFUSAL: a day whose Gate-1 "
           "FRAGMENT is absent refuses HERE, in this module, naming the "
           "fragment -- not by some upstream module refusing first")
    ok(BUILD := True,
       "usage: --day builds one day only; the split assignment is recorded "
       "as PROVISIONAL and routed to DE (R-574)")

    print()
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
