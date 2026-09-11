"""STAGE 0: the frozen modules, checked from OUTSIDE every module.

DA 255 / REVIEW 221: `de_forward_value_day.py` carries the valuation's own
freeze check, so it is the one module that still vouches for itself -- the
check would have to be edited to lie about itself, but nothing external
would notice if it were. DE 331 narrowed that gap (V2's digest moved out
to the launcher's matrix row); it did not close it.

This closes it from outside: DA's `da_population_freeze_verify.py`
recomputes every declared digest FROM DISK, and it is external to every
module it checks -- including this one and including the valuation's own
self-check. A PIPELINE-class drift REFUSES, by filename; an INSTRUMENT
drift REPORTS and passes, because the thing MEASURING moving is not the
thing MEASURED moving.

Usage:
  de_stage0_freeze_gate.py                      -> gate the real roots
  de_stage0_freeze_gate.py --root wt-deval=DIR  -> gate a scratch mirror
  de_stage0_freeze_gate.py --falsify            -> its own controls
Exit: 0 holds (instrument drift reported), 3 PIPELINE drift, 4 input absent.
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# THE VERIFIER IS DA'S, AND IT LIVES IN DA'S TREE. It is not on the freeze
# chain branch, and copying it here would make a second copy that could
# drift from the one DA maintains -- the defect REVIEW 204 found in
# be_heavy_run.sh, where one of three copies carried the fix. So it is
# imported BY PATH from the shared tree, and the report records WHICH copy
# ran, by path and digest. Its roots are absolute, so it measures the same
# files whichever tree invokes it.
DA_TREE = Path("/home/yuqing/ctaNew/live/pm_research")
if not (HERE / "da_population_freeze_verify.py").is_file():
    sys.path.append(str(DA_TREE))
try:
    import da_population_freeze_verify as V      # noqa: E402
except ImportError as exc:                       # pragma: no cover
    raise SystemExit(
        f"REFUSED STAGE0_VERIFIER_ABSENT: da_population_freeze_verify is "
        f"not importable from {HERE} or {DA_TREE} ({exc}). Stage 0 without "
        f"it is the absence of the check, not a pass.")

DRIFTED = "FROZEN_MODULE_DRIFTED"
GATE_ABSENT = "STAGE0_FREEZE_DECLARATION_ABSENT"


def gate(roots=None, decl=None) -> tuple:
    """(exit_code, report). The refusal NAMES the files, never a count."""
    decl = Path(decl) if decl else V.DECL
    if not decl.is_file():
        return 4, {"status": f"REFUSED {GATE_ABSENT}", "declaration": str(decl)}
    try:
        r = V.verify(decl, roots=roots)
    except V.FreezeRefused as exc:
        text = str(exc)
        # THE NAMES, FROM THE ROWS -- not from parsing a sentence. A
        # refusal that says "3 files changed" cannot be acted on.
        names = _drifted_names(decl, roots)
        return 3, {"status": " ".join(f"REFUSED {DRIFTED}:{n}" for n in names)
                   or f"REFUSED {DRIFTED}:{text[:120]}",
                   "n_drifted": len(names), "files": names,
                   "verifier_said": text[:300]}
    r = dict(r, roots_used={k: str(v) for k, v in
                            (roots or V.ROOTS).items()},
             roots_are_the_declared_defaults=(roots is None
                                              or roots == V.ROOTS),
             verifier={"path": str(Path(V.__file__).resolve()),
                          "sha256": V._sha(Path(V.__file__))[:16],
                          "declaration": str(decl)})
    return 0, r


def _drifted_names(decl: Path, roots=None) -> list:
    """Which PIPELINE files differ, by path -- recomputed, not parsed."""
    roots = V.ROOTS if roots is None else roots
    d = json.loads(Path(decl).read_text())
    out = []
    for e in d["files"]:
        if e.get("CLASS", "PIPELINE") == "INSTRUMENT":
            continue
        p = roots[e["root"]] / e["path"]
        got = V._sha(p)
        if got != e["sha256"]:
            out.append(e["path"])
    return out


def _mirror(root_name: str, decl: Path, dst: Path) -> Path:
    """A scratch copy of every file the declaration names under one root."""
    d = json.loads(Path(decl).read_text())
    for e in d["files"]:
        if e["root"] != root_name:
            continue
        src = V.ROOTS[e["root"]] / e["path"]
        if not src.is_file():
            continue
        out = dst / e["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
    return dst


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    decl = V.DECL
    target = "live/pm_research/de_forward_value_day.py"
    with tempfile.TemporaryDirectory() as td:
        mir = _mirror("wt-deval", decl, Path(td) / "wt-deval")
        roots = dict(V.ROOTS, **{"wt-deval": mir})

        rc_clean, rep_clean = gate(roots, decl)
        ck("an UNCHANGED mirror holds (the gate is not stuck refusing)",
           rc_clean == 0 and rep_clean.get("status")
           == "POPULATION_FREEZE_HOLDS",
           f"rc={rc_clean} {str(rep_clean.get('status'))[:40]}")

        f = mir / target
        b = bytearray(f.read_bytes())
        b[len(b) // 2] ^= 0x20                  # ONE byte
        f.write_bytes(bytes(b))
        rc_bad, rep_bad = gate(roots, decl)
        ck("ONE byte in de_forward_value_day.py REFUSES by name",
           rc_bad == 3 and target in rep_bad.get("files", [])
           and f"{DRIFTED}:{target}" in rep_bad["status"],
           f"rc={rc_bad} {str(rep_bad.get('status'))[:72]}")
        ck("the refusal NAMES the file rather than counting files",
           rep_bad.get("n_drifted") == 1
           and rep_bad.get("files") == [target],
           str(rep_bad.get("files")))
        shutil.copy2(V.ROOTS["wt-deval"] / target, f)

        d = json.loads(Path(decl).read_text())
        instr = next((e for e in d["files"]
                      if e.get("CLASS") == "INSTRUMENT"
                      and e["root"] == "wt-deval"
                      and (mir / e["path"]).is_file()), None)
        if instr:
            g = mir / instr["path"]
            bb = bytearray(g.read_bytes())
            bb[len(bb) // 2] ^= 0x20
            g.write_bytes(bytes(bb))
            rc_i, rep_i = gate(roots, decl)
            ck("an INSTRUMENT byte REPORTS and passes (it is not the "
               "object under test)",
               rc_i == 0 and any(instr["path"] in x for x in
                                 rep_i.get("INSTRUMENT_DRIFTED", [])),
               f"rc={rc_i} {instr['path']}")
        else:
            ck("an INSTRUMENT byte REPORTS and passes", False,
               "NO INSTRUMENT ENTRY UNDER wt-deval TO DRIVE")

    rc_missing, rep_missing = gate(None, Path(td) / "no_such_declaration.json")
    ck("an ABSENT declaration REFUSES rather than passing",
       rc_missing == 4 and GATE_ABSENT in rep_missing["status"],
       f"rc={rc_missing}")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsify", action="store_true")
    ap.add_argument("--root", action="append", default=[],
                    help="NAME=DIR, to gate a scratch mirror")
    ap.add_argument("--declaration", default=None)
    a = ap.parse_args(argv)
    if a.falsify:
        return falsify()
    roots = None
    if a.root:
        roots = dict(V.ROOTS)
        for spec in a.root:
            k, _, val = spec.partition("=")
            roots[k] = Path(val)
    rc, report = gate(roots, a.declaration)
    print(json.dumps(report, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
