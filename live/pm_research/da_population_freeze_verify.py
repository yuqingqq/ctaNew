"""Recompute every digest in da_population_freeze_v1 FROM DISK and refuse on drift.

THE PROPERTY: a freeze that is only a list is a promise. This reads the bytes
each entry names, hashes them, and REFUSES BY FILENAME when they differ -- so
"nothing changed" is a measurement and not an assurance.

Rule 15: `--falsify` plants a ONE-BYTE change in a scratch copy of a listed
file and requires the refusal to NAME that file. A verifier that has never
been seen to fire is not a control.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

DECL = Path(__file__).resolve().parent / "declarations" / "da_population_freeze_v1.json"
ROOTS = {"main": Path("/home/yuqing/ctaNew"),
         "wt-fwd": Path("/home/yuqing/ctaNew-wt-fwd"),
         "wt-deval": Path("/home/yuqing/ctaNew-wt-deval")}

DRIFT = "POPULATION_FREEZE_FILE_DRIFTED"
ABSENT = "POPULATION_FREEZE_FILE_ABSENT"
NO_DECL = "POPULATION_FREEZE_DECLARATION_ABSENT"


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
    drifted, absent, ok = [], [], 0
    for e in d["files"]:
        p = roots[e["root"]] / e["path"]
        got = _sha(p)
        if got is None:
            absent.append(e["path"])
        elif got != e["sha256"]:
            drifted.append({"path": e["path"], "root": e["root"],
                            "declared": e["sha256"][:16], "on_disk": got[:16]})
        else:
            ok += 1
    if absent:
        raise FreezeRefused(
            f"REFUSED {ABSENT}: {len(absent)} declared file(s) are gone -- {absent[:4]}")
    if drifted:
        names = ", ".join(x["path"] for x in drifted[:4])
        raise FreezeRefused(
            f"REFUSED {DRIFT}: {len(drifted)} of {len(d['files'])} declared file(s) "
            f"changed on disk -- {names}. First: {drifted[0]}")
    return {"status": "POPULATION_FREEZE_HOLDS", "n_files": len(d["files"]),
            "n_verified": ok, "declared_at_utc": d["declared_at_utc"]}


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
            try:
                verify(DECL, sroots)
                ck(f"ONE BYTE in a {e['role']} file REFUSES", False)
            except FreezeRefused as ex:
                ck(f"ONE BYTE in a {e['role']} file REFUSES by name ({Path(e['path']).name})",
                   DRIFT in str(ex) and e["path"] in str(ex))
            tgt.write_bytes(orig)
            ck(f"  and RESTORING those bytes verifies again ({e['role']})",
               verify(DECL, sroots)["status"] == "POPULATION_FREEZE_HOLDS")

        # a DELETED file refuses under a different name
        e = d["files"][0]
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
