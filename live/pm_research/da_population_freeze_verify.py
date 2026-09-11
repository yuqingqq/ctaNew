"""Recompute every digest in da_population_freeze_v1 FROM DISK and refuse on drift.

THE PROPERTY: a freeze that is only a list is a promise. This reads the bytes
each entry names, hashes them, and REFUSES BY FILENAME when they differ -- so
"nothing changed" is a measurement and not an assurance.

Rule 15: `--falsify` plants a ONE-BYTE change in a scratch copy of a listed
file and requires the refusal to NAME that file. A verifier that has never
been seen to fire is not a control.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

#: v2 is the UNION of DA's list and DE's generator output. v1 stays on disk as
#: provenance (rule 13) and is NOT the freeze in force.
DECL = Path(__file__).resolve().parent / "declarations" / "da_population_freeze_v4.json"
ROOTS = {"main": Path("/home/yuqing/ctaNew"),
         "wt-fwd": Path("/home/yuqing/ctaNew-wt-fwd"),
         "wt-deval": Path("/home/yuqing/ctaNew-wt-deval"),
         #: the emit waiter runs the two RESULT-BEARING modules from a FOURTH
         #: worktree. Classing them by a tree they were absent from was the gap.
         "wt-de2": Path("/home/yuqing/ctaNew-wt-de2")}

DRIFT = "POPULATION_FREEZE_FILE_DRIFTED"
#: R-900(7): two classes, two consequences. A PIPELINE byte moving means the
#: OBJECT UNDER TEST moved -- refuse. An INSTRUMENT byte moving means the thing
#: MEASURING it improved -- report and pass, until the instrument freeze is
#: called. v2 refused on an instrument landing within minutes, correctly on its
#: own terms, and that was the defect.
INSTRUMENT_DRIFT = "INSTRUMENT_DRIFTED"
ABSENT = "POPULATION_FREEZE_FILE_ABSENT"
NO_DECL = "POPULATION_FREEZE_DECLARATION_ABSENT"
#: An entry carrying `asserted_against` is checked TWICE: its executed bytes
#: against the declared digest (DRIFT), and the declared digest against what the
#: named origin ref actually carries (ASSERT_MISMATCH). The second is the emit
#: waiter's own expectation, checked WITHOUT trusting the waiter -- the ref it
#: names is the one that matters, and pointing at the wrong ref would bless
#: stale bytes silently.
ASSERT_MISMATCH = "EMIT_ASSERTION_DIGEST_MISMATCH"


class FreezeRefused(RuntimeError):
    pass


def _sha(p: Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return None


def verify(decl_path: Path = DECL, roots=None) -> dict:
    roots = ROOTS if roots is None else roots
    if not decl_path.is_file():
        raise FreezeRefused(f"REFUSED {NO_DECL}: {decl_path}")
    d = json.loads(decl_path.read_text())
    drifted, absent, ok, instr = [], [], 0, []
    for e in d["files"]:
        p = roots[e["root"]] / e["path"]
        got = _sha(p)
        row = {"path": e["path"], "root": e["root"],
               "declared": e["sha256"][:16],
               "on_disk": (got or "")[:16], "CLASS": e.get("CLASS", "PIPELINE")}
        if got is None:
            (instr if row["CLASS"] == "INSTRUMENT" else absent).append(
                row if row["CLASS"] == "INSTRUMENT" else e["path"])
        elif got != e["sha256"]:
            (instr if row["CLASS"] == "INSTRUMENT" else drifted).append(row)
        else:
            ok += 1
    if absent:
        raise FreezeRefused(
            f"REFUSED {ABSENT}: {len(absent)} declared file(s) are gone -- {absent[:4]}")
    bad_assert = []
    for e in d["files"]:
        ref = e.get("asserted_against")
        if not ref:
            continue
        gitref, _, gpath = ref.partition(":")
        # THE REF IS A REPOSITORY FACT, NOT A FILE IN THE MIRROR. Resolve it
        # against the REAL worktree even when `roots` points at a scratch copy:
        # a mirror has no .git, so looking the ref up there would report ABSENT
        # for every entry and turn the whole branch into a false alarm.
        out = subprocess.run(["git", "-C", str(ROOTS[e["root"]]), "show", f"{gitref}:{gpath}"],
                             capture_output=True)
        got = hashlib.sha256(out.stdout).hexdigest() if out.returncode == 0 else None
        if got != e.get("asserted_digest"):
            bad_assert.append({"path": e["path"], "ref": ref,
                               "declared": (e.get("asserted_digest") or "")[:16],
                               "on_ref": (got or "ABSENT")[:16]})
    if bad_assert:
        names = ", ".join(x["path"] for x in bad_assert)
        raise FreezeRefused(
            f"REFUSED {ASSERT_MISMATCH}: {len(bad_assert)} entr(ies) no longer match "
            f"the origin ref they are asserted against -- {names}. First: {bad_assert[0]}")
    if drifted:
        names = ", ".join(x["path"] for x in drifted[:4])
        raise FreezeRefused(
            f"REFUSED {DRIFT}: {len(drifted)} of {len(d['files'])} declared file(s) "
            f"changed on disk -- {names}. First: {drifted[0]}")
    return {"status": "POPULATION_FREEZE_HOLDS", "n_files": len(d["files"]),
            "n_verified": ok, "declared_at_utc": d["declared_at_utc"],
            "n_PIPELINE": sum(1 for e in d["files"] if e.get("CLASS") != "INSTRUMENT"),
            "n_INSTRUMENT": sum(1 for e in d["files"] if e.get("CLASS") == "INSTRUMENT"),
            "INSTRUMENT_DRIFTED": [f"{INSTRUMENT_DRIFT}:{r['path']}"
                                   f" {r['declared']}->{r['on_disk'] or 'ABSENT'}"
                                   for r in instr]}


def falsify() -> int:
    checks, fails = [], 0

    def ck(label, cond):
        nonlocal fails
        checks.append(label)
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    ck("the real freeze VERIFIES", verify()["status"] == "POPULATION_FREEZE_HOLDS")

    d = json.loads(DECL.read_text())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # mirror only the declared files into a scratch root set
        sroots = {k: td / k for k in ROOTS}
        for e in d["files"]:
            src = ROOTS[e["root"]] / e["path"]
            dst = sroots[e["root"]] / e["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        ck("the MIRROR verifies before any plant", verify(DECL, sroots)["status"] == "POPULATION_FREEZE_HOLDS")

        # ONE BYTE, in each role, must refuse BY NAME
        seen = set()
        for e in d["files"]:
            if e["role"] in seen:
                continue
            seen.add(e["role"])
            tgt = sroots[e["root"]] / e["path"]
            orig = tgt.read_bytes()
            tgt.write_bytes(orig + b" ")          # exactly one byte
            # THE EXPECTATION IS A FUNCTION OF THE CLASS, NOT THE ROLE. This loop
            # predated R-900(7)'s split and asserted that EVERY role refuses; it
            # then failed on LAUNCHER and INSTRUMENT_FILE -- correctly, because
            # those are INSTRUMENT and must report instead. A falsifier whose
            # expectations go stale is the defect it exists to catch.
            is_instr = e.get("CLASS") == "INSTRUMENT"
            try:
                r = verify(DECL, sroots)
                named = any(e["path"] in x for x in r["INSTRUMENT_DRIFTED"])
                ck(f"ONE BYTE in a {e['role']} file ({e.get('CLASS')}) "
                   f"REPORTS and passes ({Path(e['path']).name})", is_instr and named)
            except FreezeRefused as ex:
                ck(f"ONE BYTE in a {e['role']} file ({e.get('CLASS')}) "
                   f"REFUSES by name ({Path(e['path']).name})",
                   (not is_instr) and DRIFT in str(ex) and e["path"] in str(ex))
            tgt.write_bytes(orig)
            ck(f"  and RESTORING those bytes verifies again ({e['role']})",
               verify(DECL, sroots)["status"] == "POPULATION_FREEZE_HOLDS")

        # ---- R-900(7): BOTH BRANCHES, explicitly ----
        pipe = next(e for e in d["files"] if e.get("CLASS") != "INSTRUMENT")
        inst = next(e for e in d["files"] if e.get("CLASS") == "INSTRUMENT")
        for e, expect_refuse in ((pipe, True), (inst, False)):
            tgt = sroots[e["root"]] / e["path"]
            orig = tgt.read_bytes()
            tgt.write_bytes(orig + b" ")
            cls = e.get("CLASS", "PIPELINE")
            try:
                r = verify(DECL, sroots)
                named = any(e["path"] in x for x in r["INSTRUMENT_DRIFTED"])
                ck(f"an {cls} byte REPORTS and PASSES, naming it "
                   f"({Path(e['path']).name})", (not expect_refuse) and named)
            except FreezeRefused as ex:
                ck(f"a {cls} byte REFUSES by filename ({Path(e['path']).name})",
                   expect_refuse and DRIFT in str(ex) and e["path"] in str(ex))
            tgt.write_bytes(orig)
        ck("  and both restorations verify clean",
           verify(DECL, sroots)["status"] == "POPULATION_FREEZE_HOLDS")

        # the ASSERTED-DIGEST branch: point the expectation at the wrong bytes
        import copy as _c
        d2 = _c.deepcopy(d)
        tgt2 = next(e for e in d2["files"] if e.get("asserted_against"))
        tgt2["asserted_digest"] = "0" * 64
        dp = td / "bad_assert.json"; dp.write_text(json.dumps(d2))
        try:
            verify(dp, sroots)
            ck("a WRONG asserted digest REFUSES", False)
        except FreezeRefused as ex:
            ck(f"a WRONG asserted digest REFUSES by filename "
               f"({Path(tgt2['path']).name})",
               "EMIT_ASSERTION_DIGEST_MISMATCH" in str(ex) and tgt2["path"] in str(ex))
        ck("  and the UNMODIFIED declaration still verifies",
           verify(DECL, sroots)["status"] == "POPULATION_FREEZE_HOLDS")

        # a DELETED file refuses under a different name
        e = next(x for x in d["files"] if x.get("CLASS") != "INSTRUMENT")
        tgt = sroots[e["root"]] / e["path"]
        orig = tgt.read_bytes(); tgt.unlink()
        try:
            verify(DECL, sroots); ck("a DELETED declared file REFUSES", False)
        except FreezeRefused as ex:
            ck("a DELETED declared file REFUSES with ABSENT, not DRIFT",
               ABSENT in str(ex) and DRIFT not in str(ex))
        tgt.write_bytes(orig)

    print(json.dumps({"falsifier": "da_population_freeze_verify",
                      "n": len(checks), "n_failed": fails}))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args()
    if a.falsify:
        sys.exit(falsify())
    try:
        print(json.dumps(verify(), indent=1))
    except FreezeRefused as e:
        print(str(e)); sys.exit(3)
