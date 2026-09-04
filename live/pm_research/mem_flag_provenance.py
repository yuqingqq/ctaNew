#!/usr/bin/env python3
"""Separate the flags MEM CHECKED at an artifact from the ones it RELAYED.

WHY THIS EXISTS. `STATUS.yml` is a summary of the register, which is a summary
of other seats' reports. Its 458 flags all read as established fact in the same
confident capitals whether MEM verified them at an artifact or swept them out of
a row, and on 2026-09-04 MEM could not say which were which. The programme has a
rule that no number reaches the USER before review; it had no counterpart for
*no claim reaches the state files before it is checked at an artifact*, and MEM
is the enforcement point. This is that instrument (dispatched at R-512).

THE CLASSES, AND WHY ABSENCE IS ITS OWN CLASS. `CHECKED` is authoritative;
nothing else is. `RELAYED` and `UNMARKED` are equally non-authoritative to a
reader deciding whether to rely on a flag -- which is the ruling MEM was given.
They are nevertheless kept DISTINCT, because a second consumer (MEM, choosing
what to spend a round checking) needs to tell "assessed and relayed" from "never
assessed", and collapsing them would put an absence inside the codomain of the
measurement -- the predicate R-505 adopted after `uncompressed_size` returned 0
for a file it could not read. Authority collapses the two; provenance does not.

THE MARKER CANNOT BE PRODUCED BY PROSE, WHICH IS DELIBERATE. R-511 recorded a
guard that armed itself because a filing quoted the token its reader scanned
for: the register is DATA, so naming a marker inside it is a WRITE. Provenance
therefore lives in its own top-level `flag_provenance:` mapping, never in a
flag's own body -- a flag's prose cannot reach it, and neither can another
seat's filing, because this instrument reads STATUS.yml and nothing else.

AND `CHECKED` MUST CARRY WHAT THE ARTIFACT SAID. A bare "VERIFIED AT THE
ARTIFACT" is a token that would look identical had the check never happened
(R-509's counterfactual question), and MEM's own audit found its habit was
confirming that a string exists at a line -- a citation check, not a claim
check. So `CHECKED` REFUSES without both `artifact:` and `said:`, and refuses a
`said:` that is one of the bare confirmations that carry no content.

    python3 mem_flag_provenance.py --selftest
    python3 mem_flag_provenance.py --audit [--status PATH] [--json]

Exit: 0 clean, 1 findings (MALFORMED / ORPHAN / a CHECKED artifact that is
gone), 2 refusal (input unreadable or unparseable). UNMARKED is a counted
status and never fails the run -- a check that fires on all 458 is a check that
gets turned off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# The state file this instrument governs. Resolved from THIS file's location so
# it is the same artifact whichever tree the run happens in.
_REPO = Path(__file__).resolve().parents[2]
STATUS = (_REPO / "orchestrator" / "PROGRAMS" / "P-2026-003-polymarket-5min"
          / "STATUS.yml")

PROV_KEY = "flag_provenance"
FLAGS_KEY = "flags"

AUTHORITATIVE = "CHECKED"

# `said:` values that assert the check happened without saying what it found.
# This is the exact shape MEM found in its own back-catalogue; it is a refusal
# list of CONTENT-FREE confirmations, not a style rule.
_EMPTY_SAID = {
    "", "-", "n/a", "na", "none", "true", "yes", "ok", "okay",
    "verified", "checked", "confirmed", "correct", "as stated", "it does",
    "verified at the artifact", "verified by me at the artifact",
    "checked at the artifact", "the citation resolves", "resolves",
    "verified by mem at the artifact", "i verified it", "reproduced",
}


def _norm(s: object) -> str:
    return " ".join(str(s).strip().lower().split()).rstrip(".")


def classify(entry: object) -> tuple[str, str]:
    """Return (class, reason) for one `flag_provenance` entry.

    Never raises on bad input: a malformed entry is a FINDING to report, not an
    exception to swallow. The class set is closed and every non-conforming shape
    lands on MALFORMED with a reason naming what is wrong.
    """
    if not isinstance(entry, dict):
        return "MALFORMED", f"entry is {type(entry).__name__}, not a mapping"
    prov = entry.get("prov")
    if prov is None:
        return "MALFORMED", "no `prov:` key"
    if prov == "RELAYED":
        src = entry.get("from")
        if not str(src or "").strip():
            return "MALFORMED", "RELAYED without `from:` (name the row or entry)"
        return "RELAYED", f"relayed from {src}"
    if prov == AUTHORITATIVE:
        art = str(entry.get("artifact") or "").strip()
        said = entry.get("said")
        if not art:
            return "MALFORMED", "CHECKED without `artifact:`"
        if said is None or not str(said).strip():
            return "MALFORMED", "CHECKED without `said:`"
        if _norm(said) in _EMPTY_SAID:
            return ("MALFORMED",
                    f"`said:` is a bare confirmation ({str(said).strip()!r}) and "
                    "carries no content: it would read identically had the check "
                    "not happened")
        return "CHECKED", f"checked at {art}"
    return "MALFORMED", f"unknown prov {prov!r} (expected CHECKED or RELAYED)"


def artifact_exists(entry: dict, repo: Path) -> bool | None:
    """True/False if decidable on this disk; None if the reference is not a path.

    None is deliberate: a `git:<ref>:<path>` or an off-disk reference is NOT
    absent, and returning False for it would put "cannot be checked here" inside
    the codomain of "was checked and is gone".
    """
    art = str(entry.get("artifact") or "").strip()
    if not art or art.startswith(("git:", "http:", "https:")):
        return None
    p = Path(art)
    for cand in ((p,) if p.is_absolute() else (repo / p, p)):
        if cand.exists():
            return True
    return False


def audit(status_path: Path = STATUS) -> dict:
    """Census every flag against the provenance map. Reconciles BOTH ways."""
    try:
        doc = yaml.safe_load(status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"REFUSED: no such status file: {status_path}")
    except yaml.YAMLError as e:
        raise SystemExit(f"REFUSED: {status_path} did not parse: {e}")
    if not isinstance(doc, dict) or not isinstance(doc.get(FLAGS_KEY), dict):
        raise SystemExit(f"REFUSED: {status_path} has no `{FLAGS_KEY}:` mapping")

    flags = doc[FLAGS_KEY]
    prov = doc.get(PROV_KEY) or {}
    if not isinstance(prov, dict):
        raise SystemExit(f"REFUSED: `{PROV_KEY}:` is not a mapping")

    by_class: dict[str, list[str]] = {}
    reasons: dict[str, str] = {}
    by_name: dict[str, str] = {}
    missing_artifact: list[str] = []

    for name in flags:
        if name not in prov:
            by_class.setdefault("UNMARKED", []).append(name)
            by_name[name] = "UNMARKED"
            continue
        cls, why = classify(prov[name])
        by_class.setdefault(cls, []).append(name)
        by_name[name] = cls
        reasons[name] = why
        if cls == AUTHORITATIVE and artifact_exists(prov[name], _REPO) is False:
            missing_artifact.append(name)

    # The hand-maintained-map hazard, guarded in the direction that bites: an
    # entry naming a flag that no longer exists is drift, not provenance.
    orphans = [k for k in prov if k not in flags]

    n = len(flags)
    checked = by_class.get(AUTHORITATIVE, [])
    findings = (len(by_class.get("MALFORMED", [])) + len(orphans)
                + len(missing_artifact))
    return {
        "status_file": str(status_path),
        "n_flags": n,
        "counts": {k: len(v) for k, v in sorted(by_class.items())},
        "orphan_entries": sorted(orphans),
        "checked_artifact_missing": sorted(missing_artifact),
        "authoritative": sorted(checked),
        "n_authoritative": len(checked),
        "n_non_authoritative": n - len(checked),
        "authority_rule": (
            "only CHECKED is authoritative; RELAYED and UNMARKED are equally "
            "non-authoritative to a reader, and are kept distinct so the "
            "unassessed backlog stays countable"),
        "reasons": reasons,
        "class_by_flag": by_name,
        "findings": findings,
    }


def render(rep: dict) -> None:
    print(f"flags: {rep['n_flags']}   authoritative (CHECKED): "
          f"{rep['n_authoritative']}   non-authoritative: "
          f"{rep['n_non_authoritative']}")
    for k, v in rep["counts"].items():
        print(f"  {k:<10} {v}")
    for name in rep["authoritative"]:
        print(f"  CHECKED   {name}: {rep['reasons'][name]}")
    for name in rep.get("checked_artifact_missing", []):
        print(f"  FINDING   {name}: CHECKED, but its artifact is not on disk")
    for name, cls in rep["class_by_flag"].items():
        if cls == "MALFORMED":
            print(f"  MALFORMED {name}: {rep['reasons'][name]}")
    for name in rep["orphan_entries"]:
        print(f"  ORPHAN    {name}: provenance for a flag that does not exist")
    print(f"findings: {rep['findings']}")


def selftest() -> int:
    """Falsifiers, both directions (SEAT_PROTOCOL 16).

    No expected-total literal: a hardcoded tally cannot tell you WHICH checks
    ran, and is the defect MEM filed against `da_race_withdrawals.py:59`. This
    counts what it ran and names each one.
    """
    ran: list[str] = []

    def ok(cond: bool, label: str) -> None:
        if not cond:
            raise AssertionError(label)
        ran.append(label)

    # --- it must ADMIT the good case (a control that only refuses proves nothing)
    ok(classify({"prov": "CHECKED", "artifact": "data/x.json",
                 "said": "declared_gate_outcome.passed is null on all six"})[0]
       == "CHECKED", "ADMITS a CHECKED entry carrying an artifact and its words")
    ok(classify({"prov": "RELAYED", "from": "R-512"})[0] == "RELAYED",
       "ADMITS a RELAYED entry that names its source")

    # --- the positive control it MUST flag: CHECKED without what was found
    ok(classify({"prov": "CHECKED", "artifact": "data/x.json"})[0] == "MALFORMED",
       "REFUSES CHECKED with no `said:` -- the citation-not-claim shape")
    ok(classify({"prov": "CHECKED", "artifact": "data/x.json",
                 "said": "VERIFIED BY ME AT THE ARTIFACT"})[0] == "MALFORMED",
       "REFUSES a bare confirmation as `said:` -- the token that reads the same "
       "whether or not the check happened")
    ok(classify({"prov": "CHECKED",
                 "said": "the file says 12 OK + 6 + 6"})[0] == "MALFORMED",
       "REFUSES CHECKED with no artifact named")
    ok(classify({"prov": "RELAYED"})[0] == "MALFORMED",
       "REFUSES RELAYED that does not name where it came from")
    ok(classify({"prov": "AGREED"})[0] == "MALFORMED",
       "REFUSES an unknown class rather than passing it through")
    ok(classify({})[0] == "MALFORMED", "REFUSES an entry with no `prov:`")
    ok(classify("CHECKED")[0] == "MALFORMED",
       "REFUSES a bare string where a mapping is required")

    # --- absence must not be answered from inside the measurement's codomain
    ok(artifact_exists({"artifact": "git:abc123:live/x.py"}, _REPO) is None,
       "an off-disk reference is None, never False -- 'cannot decide here' is "
       "not 'was checked and is gone'")
    ok(artifact_exists({"artifact": "does/not/exist.json"}, _REPO) is False,
       "a repo-relative path that is absent reads False")

    # --- end to end, on files, both directions
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "good.yml").write_text(
        "flags:\n  a: X\n  b: Y\n"
        f"{PROV_KEY}:\n"
        "  a:\n    prov: CHECKED\n    artifact: live/pm_research/mem_flag_provenance.py\n"
        "    said: the module defines classify() with a closed class set\n"
        "  b:\n    prov: RELAYED\n    from: R-512\n")
    r = audit(d / "good.yml")
    ok(r["findings"] == 0 and r["n_authoritative"] == 1,
       "a clean file reports 0 findings and exactly one authoritative flag")
    ok(r["counts"].get("RELAYED") == 1 and "UNMARKED" not in r["counts"],
       "and the relayed flag is counted as relayed, not as unmarked")

    (d / "unmarked.yml").write_text("flags:\n  a: X\n  b: Y\n")
    r = audit(d / "unmarked.yml")
    ok(r["counts"].get("UNMARKED") == 2 and r["findings"] == 0,
       "UNMARKED is a counted status and does NOT fail the run -- a check that "
       "fires on everything is a check that gets turned off")
    ok(r["n_authoritative"] == 0,
       "and an unmarked file has no authoritative flags, which is the ruling")

    (d / "orphan.yml").write_text(
        "flags:\n  a: X\n"
        f"{PROV_KEY}:\n  zzz:\n    prov: RELAYED\n    from: R-1\n")
    ok(audit(d / "orphan.yml")["orphan_entries"] == ["zzz"],
       "REFUSES an entry for a flag that does not exist -- the hand-maintained "
       "map drifting from what it describes")

    (d / "gone.yml").write_text(
        "flags:\n  a: X\n"
        f"{PROV_KEY}:\n  a:\n    prov: CHECKED\n    artifact: data/vanished.json\n"
        "    said: it read 333 of 1154\n")
    r = audit(d / "gone.yml")
    ok(r["checked_artifact_missing"] == ["a"] and r["findings"] == 1,
       "a CHECKED flag whose artifact is GONE is a finding -- three arms "
       "artifacts vanished on 2026-09-04 and a flag citing one still read as "
       "established")

    (d / "noflags.yml").write_text("focus: nothing here\n")
    try:
        audit(d / "noflags.yml")
        ok(False, "unreachable")
    except SystemExit as e:
        ok("REFUSED" in str(e), "REFUSES a status file with no flags mapping")
    try:
        audit(d / "absent.yml")
        ok(False, "unreachable")
    except SystemExit as e:
        ok("REFUSED" in str(e), "REFUSES a missing status file rather than "
                                "reporting a clean census of nothing")

    # --- the R-511 hazard, instrumented: prose cannot arm this instrument
    (d / "quoted.yml").write_text(
        "flags:\n"
        "  a: >-\n"
        "    this flag's own body quotes the format: prov: CHECKED artifact: x\n"
        "    said: something -- and it must NOT make this flag authoritative\n")
    r = audit(d / "quoted.yml")
    ok(r["n_authoritative"] == 0 and r["counts"].get("UNMARKED") == 1,
       "a flag whose PROSE quotes the marker stays UNMARKED -- R-511's pin that "
       "armed itself because a filing named the token its reader scanned for")

    for label in ran:
        print(f"  ok  {label}")
    print(f"mem_flag_provenance selftest: {len(ran)} checks ran, all passed "
          f"(count is reported, never asserted against a literal)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--status", type=Path, default=STATUS)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.audit:
        ap.print_help()
        return 2
    rep = audit(a.status)
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    else:
        render(rep)
    return 1 if rep["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
