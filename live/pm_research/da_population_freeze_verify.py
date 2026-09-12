"""Recompute every digest in da_population_freeze_v1 FROM DISK and refuse on drift.

THE PROPERTY: a freeze that is only a list is a promise. This reads the bytes
each entry names, hashes them, and REFUSES BY FILENAME when they differ -- so
"nothing changed" is a measurement and not an assurance.

Rule 15: `--falsify` plants a ONE-BYTE change in a scratch copy of a listed
file and requires the refusal to NAME that file. A verifier that has never
been seen to fire is not a control.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

#: v2 is the UNION of DA's list and DE's generator output. v1 stays on disk as
#: provenance (rule 13) and is NOT the freeze in force.
DECL = Path(__file__).resolve().parent / "declarations" / "da_population_freeze_v19.json"
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
#: DA 260: the guard register's gate. A real run is allowed only when EVERY row
#: it can hit is exercised. COMPUTED, never prose -- the register lists the
#: sites and this counts them.
GUARD_GATE = "GUARD_ROW_NOT_EXERCISED_BEFORE_A_REAL_RUN"
GUARD_DECL = Path(__file__).resolve().parent / "declarations" / "da_guard_register_v2.json"
#: REVIEW 168's gap: the register could not refuse an UNREGISTERED site. The
#: AST enumerator saw a new `raise ...Refused` and the register stayed at 167,
#: because NOTHING JOINED CODE TO ROWS -- a list of guards is not a guard over
#: the list. v2 re-enumerates AT VERIFY TIME and compares.
NOT_HIGHEST = "POPULATION_FREEZE_NOT_THE_HIGHEST"


def assert_decl_is_the_highest(decl: Path = None) -> dict:
    """DECL is a FILENAME LITERAL, and twice now it has gone stale in place.

    At DA 258 it was moved v5 -> v6 and never moved again, so the drift
    reported at Q-DA-450 and the "v7 landed" claim at Q-DA-451 were both
    measured against SUPERSEDED baselines -- a QUIET error that reached
    counts. It went stale a second time at DA 247 (pointing at v7 while v9
    was landed), which is why this exists.

    The fix is not to glob for the newest: "a file that merely sorts last
    is not the one the freeze names". It is to keep the literal AND REFUSE
    when a higher-numbered freeze is on disk beside it -- turning a silent
    stale baseline into a loud refusal, which is the direction that
    self-corrects.
    """
    d = (decl or DECL)
    import re as _re
    here = int(_re.search(r"_v(\d+)\.json$", d.name).group(1))
    found = {}
    for f in d.parent.glob("da_population_freeze_v*.json"):
        m = _re.search(r"_v(\d+)\.json$", f.name)
        if m:
            found[int(m.group(1))] = f.name
    top = max(found) if found else here
    if top > here:
        raise FreezeRefused(
            f"REFUSED {NOT_HIGHEST}: this verifier reads {d.name} but "
            f"{found[top]} is landed beside it. Every digest it compares "
            f"against is a SUPERSEDED baseline, and a clean answer from it "
            f"means nothing. Repoint DECL.")
    return {"decl": d.name, "version": here, "highest_on_disk": top,
            "versions_present": [found[k] for k in sorted(found)]}


GUARD_INCOMPLETE = "GUARD_REGISTER_INCOMPLETE"
GUARD_STALE = "GUARD_REGISTER_STALE"
_REFUSAL_NAME = re.compile(r"REFUSED\s*-{0,2}\s*\{?([A-Z][A-Z0-9_]{3,})")


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


def enumerate_sites(rows, src_for) -> set:
    """Every `raise` carrying a REFUSED <NAME>, re-derived from the SOURCE.

    THE JOIN REVIEW 168 FOUND MISSING. The register lists sites; this reads the
    code the register claims to cover and returns what is actually there, so
    the two can be compared instead of trusted."""
    out = set()
    for path in sorted({r["file"] for r in rows}):
        src = src_for(path)
        if src is None:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        consts = {}
        for n in ast.walk(tree):
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str)):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = n.value.value
        for n in ast.walk(tree):
            if not isinstance(n, ast.Raise):
                continue
            m = _REFUSAL_NAME.search(ast.unparse(n))
            if not m:
                continue
            tok = m.group(1)
            out.add((path, n.lineno, consts.get(tok, tok)))
    return out


def _src_from_ref(path, ref_of):
    r = ref_of(path)
    o = subprocess.run(["git", "-C", str(ROOTS["main"]), "show", f"{r}:{path}"],
                       capture_output=True, text=True)
    return o.stdout if o.returncode == 0 else None


def guard_gate(lanes=None, decl: Path = GUARD_DECL, src_for=None) -> dict:
    """REFUSE a real run while any row it can hit is unexercised (DA 260)."""
    if not decl.is_file():
        raise FreezeRefused(f"REFUSED {GUARD_GATE}: no guard register at {decl}")
    d = json.loads(decl.read_text())
    rows = d["rows"] if lanes is None else [
        r for r in d["rows"] if r["lane"] in set(lanes)]
    if not rows:
        raise FreezeRefused(
            f"REFUSED {GUARD_GATE}: no rows for lane(s) {lanes} -- an empty "
            f"lane cannot exonerate a run (the aggregate-only trap)")
    # ---- REVIEW 168: the register must be COMPLETE before it can gate ----
    ref_of = {r["file"]: r["ref"] for r in rows}
    getsrc = src_for or (lambda p: _src_from_ref(p, lambda x: ref_of[x]))
    enumerated = enumerate_sites(rows, getsrc)
    declared = {(r["file"], r["line"], r["refusal"]) for r in rows}
    missing = sorted(enumerated - declared)
    gone = sorted(declared - enumerated)
    if missing:
        first = missing[0]
        raise FreezeRefused(
            f"REFUSED {GUARD_INCOMPLETE}:{Path(first[0]).name}:{first[1]}:{first[2]} "
            f"-- {len(missing)} enumerated refusal site(s) have NO ROW. A "
            f"register that cannot see a new guard cannot gate a run on it.")
    if gone:
        first = gone[0]
        raise FreezeRefused(
            f"REFUSED {GUARD_STALE}:{Path(first[0]).name}:{first[1]}:{first[2]} "
            f"-- {len(gone)} row(s) name a site that no longer exists.")
    bad = [r for r in rows if not r.get("exercised_before_real_run")]
    if bad:
        names = ", ".join(f"{r['site']}:{r['refusal']}" for r in bad[:4])
        raise FreezeRefused(
            f"REFUSED {GUARD_GATE}: {len(bad)} of {len(rows)} guard row(s) on "
            f"lane(s) {lanes or 'ALL'} are NOT exercised -- {names}. A real run "
            f"is allowed only when every row it can hit has been driven.")
    return {"status": "EVERY_GUARD_ROW_ON_THESE_LANES_IS_EXERCISED",
            "lanes": lanes or "ALL", "n_rows": len(rows),
            #: DERIVED AT VERIFY TIME, NEVER RECORDED -- the v1 register carried
            #: POPULATION 217 and it was stale by measurement the same day.
            "POPULATION_derived_now": len(enumerated)}



# ---- THE EXECUTING TREE (DA 272) ---------------------------------------
#
# THE SHARED TREE IS NON-EXECUTING FOR BOTH LANES. `/home/yuqing/ctaNew` is
# the user's working fork; measured 2026-09-11T19:08Z it is 205 commits behind
# origin/mm-research and 100 behind the chain, and it carries PRE-FREEZE copies
# of FOUR frozen closure modules. A result produced from it is INADMISSIBLE
# REGARDLESS OF ITS CONTENT -- not because the numbers would be wrong, but
# because nothing about them would be attributable to the frozen code.
#
# THE CHECK READS THE PAYLOAD, NOT ITS ENVIRONMENT. An env var says what
# someone INTENDED; the resolved `__file__` of a module the payload actually
# imported, and the process's own cwd, say what it WILL IMPORT. The running
# 09-10 build is the case in point: its argv carries
# `/home/yuqing/ctaNew/live/pm_research/be_heavy_run.sh` and a RELATIVE
# `live/pm_research/be_daybook_build.py`, so the launcher path says "shared
# tree" and the cwd (/home/yuqing/ctaNew-wt-fwd) says otherwise. The cwd is
# right and the launcher path is decoration.
EXECUTING_TREES = {
    "build": Path("/home/yuqing/ctaNew-wt-fwd"),
    "valuation": Path("/home/yuqing/ctaNew-wt-deval"),
    "emit": Path("/home/yuqing/ctaNew-wt-de2"),
}
NON_EXECUTING_TREE = Path("/home/yuqing/ctaNew")
UNDECLARED_TREE = "EXECUTED_FROM_AN_UNDECLARED_TREE"


def executing_tree_of(module=None, cwd=None) -> Path:
    """The tree a payload WILL import from -- from the artifact, not the env."""
    if module is not None:
        f = getattr(module, "__file__", None)
        if f:
            return Path(f).resolve().parents[2]
    return Path(cwd if cwd is not None else Path.cwd()).resolve()


def assert_executing_tree(module=None, cwd=None, lane=None) -> dict:
    """REFUSE a run from any tree the freeze does not declare as executing."""
    tree = executing_tree_of(module, cwd)
    declared = {k: v.resolve() for k, v in EXECUTING_TREES.items()}
    lanes = [k for k, v in declared.items() if v == tree]
    if not lanes:
        why = ("it is the SHARED tree, which is NON-EXECUTING for both lanes"
               if tree == NON_EXECUTING_TREE.resolve()
               else "it is not a declared executing tree")
        raise FreezeRefused(
            f"REFUSED {UNDECLARED_TREE}:{tree} -- {why}. Declared: "
            f"{ {k: str(v) for k, v in declared.items()} }. A result produced "
            f"from an undeclared tree is INADMISSIBLE regardless of its "
            f"content.")
    if lane is not None and lane not in lanes:
        raise FreezeRefused(
            f"REFUSED {UNDECLARED_TREE}:{tree} -- it is the {lanes[0]!r} tree "
            f"and this payload declared lane {lane!r}.")
    return {"tree": str(tree), "lane": lanes[0], "declared_executing": True,
            "read_from": "module.__file__" if module is not None else "cwd",
            "NOT_read_from": "any environment variable"}


def falsify_executing_tree() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    for lane, t in EXECUTING_TREES.items():
        if not t.exists():
            continue
        r = assert_executing_tree(cwd=t)
        ck(f"a DECLARED executing tree passes and names its lane ({lane})",
           r["lane"] == lane and r["declared_executing"], r["tree"])
    try:
        assert_executing_tree(cwd=NON_EXECUTING_TREE)
        ck("the SHARED tree REFUSES", False, "admitted")
    except FreezeRefused as exc:
        ck("the SHARED tree REFUSES by name, saying it is NON-EXECUTING",
           UNDECLARED_TREE in str(exc) and "NON-EXECUTING" in str(exc))
    try:
        assert_executing_tree(cwd="/tmp")
        ck("an unrelated tree REFUSES", False)
    except FreezeRefused as exc:
        ck("an unrelated tree REFUSES by name", UNDECLARED_TREE in str(exc))
    try:
        assert_executing_tree(cwd=EXECUTING_TREES["build"], lane="valuation")
        ck("the RIGHT tree for the WRONG lane REFUSES", False)
    except FreezeRefused as exc:
        ck("the RIGHT tree for the WRONG lane REFUSES", UNDECLARED_TREE in str(exc))
    import types
    m = types.SimpleNamespace(
        __file__=str(EXECUTING_TREES["valuation"] / "live/pm_research/x.py"))
    r = assert_executing_tree(module=m)
    ck("it reads a MODULE'S RESOLVED __file__, not an env var",
       r["lane"] == "valuation" and r["read_from"] == "module.__file__",
       r["NOT_read_from"])
    import os
    os.environ["PM_TREE"] = str(NON_EXECUTING_TREE)
    r2 = assert_executing_tree(cwd=EXECUTING_TREES["valuation"])
    ck("...and an env var claiming the shared tree cannot change the answer",
       r2["lane"] == "valuation")
    os.environ.pop("PM_TREE", None)
    print(f"\n  {'EXECUTING-TREE CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


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

        # ---- REVIEW 168: an UNREGISTERED site must refuse BY NAME ----
        gdoc = json.loads(GUARD_DECL.read_text())
        grows = gdoc["rows"]
        real = {}
        for _p in sorted({r["file"] for r in grows}):
            real[_p] = _src_from_ref(_p, lambda x: {r["file"]: r["ref"]
                                                    for r in grows}[x])
        tgt = grows[0]["file"]

        def planted(path):
            """REV's scratch copy: one NEW raise, in memory, nothing on disk."""
            src0 = real.get(path)
            if path != tgt or src0 is None:
                return src0
            return src0 + (
                '\n\ndef _rev168_planted():\n'
                '    raise RuntimeError("REFUSED REV168_PLANTED_SITE: a site '
                'no row names")\n')
        try:
            guard_gate(None, GUARD_DECL, src_for=planted)
            ck("an UNREGISTERED site REFUSES", False)
        except FreezeRefused as ex:
            ck("an UNREGISTERED site REFUSES GUARD_REGISTER_INCOMPLETE by name",
               GUARD_INCOMPLETE in str(ex) and "REV168_PLANTED_SITE" in str(ex))

        def deleted(path):
            """The other direction: a row whose site is gone."""
            src0 = real.get(path)
            if path != tgt or src0 is None:
                return src0
            return "\n".join(
                ln for ln in src0.split("\n") if "REFUSED" not in ln)
        try:
            guard_gate(None, GUARD_DECL, src_for=deleted)
            ck("a VANISHED site REFUSES", False)
        except FreezeRefused as ex:
            ck("a VANISHED site REFUSES GUARD_REGISTER_STALE by name",
               GUARD_STALE in str(ex))

        # ---- DA 260: the guard gate, both directions ----
        try:
            guard_gate(["VALUATION"])
            ck("the guard gate on a lane with unexercised rows REFUSES", False)
        except FreezeRefused as ex:
            ck("the guard gate REFUSES an unexercised lane, naming sites",
               GUARD_GATE in str(ex) and ":" in str(ex))
        gd = json.loads(GUARD_DECL.read_text())
        allok = td / "all_exercised.json"
        for r in gd["rows"]:
            r["exercised_before_real_run"] = True
        allok.write_text(json.dumps(gd))
        ck("  and ADMITS when every row is exercised",
           guard_gate(None, allok)["status"].startswith("EVERY_GUARD_ROW"))
        empty = td / "empty_lane.json"
        empty.write_text(json.dumps({"rows": []}))
        try:
            guard_gate(None, empty)
            ck("  an EMPTY register does not exonerate a run", False)
        except FreezeRefused as ex:
            ck("  an EMPTY register REFUSES rather than passing vacuously",
               GUARD_GATE in str(ex))

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
    ap.add_argument("--guard-gate", nargs="*", default=None,
                    help="lane(s) a real run can hit; omit the value for ALL")
    a = ap.parse_args()
    if a.guard_gate is not None:
        try:
            print(json.dumps(guard_gate(a.guard_gate or None), indent=1))
        except FreezeRefused as e:
            print(str(e)); sys.exit(4)
        sys.exit(0)
    if a.falsify:
        sys.exit(falsify())
    try:
        print(json.dumps(verify(), indent=1))
    except FreezeRefused as e:
        print(str(e)); sys.exit(3)
