#!/usr/bin/env python3
"""DA -- THE DEPLOY PIN AS A DECLARATION (R-741, REV 88 section 3).

***THE UNIT DRIFTED ON MY OWN LANDINGS AND NOBODY KNEW UNTIL 00:06Z.***
`da-midnight-verify.service` refused at rc 7 on 2026-09-07 because files it
executes had moved after the deploy act. The refusal was correct and it was
also the FIRST time anyone learned of it -- a night's verdict was already
gone. Both halves of REV 88 section 3 are the rule now:

  * the deploy pin is a DECLARATION with a version chain, written through
    the shared compare-and-swap (`declaration_chain.write_next_version`),
    never edited in place -- so what the unit was deployed at is a
    resolvable artifact and not a file somebody overwrote;
  * a landing that touches a pinned file RE-PINS IN THE SAME ROUND, and
    the non-head census MARKS any pinned file whose digest moved without
    a new pin (`DEPLOY_PIN_STALE`, naming the file). ***The drift becomes
    visible at the census, in daylight, instead of at 00:06Z as a
    refusal.***

The pin carries {unit, commit, files: [{path, sha256}], deployed_at}: the
unit it deploys, the commit it was taken at, and every file whose bytes the
unit's own drift guard compares.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "DA_MIDNIGHT_DEPLOY_PIN_V1"
FAMILY = "da_midnight_deploy_pin"
UNIT = "da-midnight-verify.service"
RECORD = HERE / "systemd" / "da_deploy_record.json"
DECL_DIR = HERE / "declarations"


class PinRefused(RuntimeError):
    """The pin cannot be written or read honestly."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree() -> Path:
    return HERE.parent.parent


def payload_from_record(record: Path | None = None) -> dict:
    """The pin's content, taken from the deploy act's own record."""
    r = Path(record) if record else RECORD
    if not r.is_file():
        raise PinRefused(
            f"REFUSED: DEPLOY_RECORD_ABSENT -- {r}. The pin is written FROM "
            f"the deploy act's record; inventing one here would make the "
            f"pin a claim about a deploy that never happened.")
    rec = json.loads(r.read_text())
    files = [{"path": f["path"], "sha256": f["sha256"],
              "tier": f.get("tier")}
             for f in (rec.get("files") or []) if f.get("path")
             and f.get("sha256")]
    if not files:
        raise PinRefused(
            "REFUSED: DEPLOY_RECORD_PINS_NO_FILES -- a pin over an empty "
            "file set would admit every drift there is.")
    #: THE UNIT IS A NAME. v1 took `installed_units` -- which is the LIST
    #: OF INSTALLED UNIT FILES, not the unit -- so the pin's own `unit`
    #: field read as an array of paths. Caught by reading the artifact
    #: after writing it. The installed files keep their own key.
    return {"protocol": PROTOCOL,
            "unit": UNIT,
            "installed_unit_files": rec.get("installed_units"),
            "commit": rec.get("deployed_commit"),
            "deployed_at": rec.get("deployed_at_utc"),
            "deployed_by": rec.get("deployed_by"),
            "files": sorted(files, key=lambda f: f["path"]),
            "n_files": len(files),
            "what_it_is": (
                "what the unit was DEPLOYED at: the commit, and every file "
                "whose bytes its drift guard compares. Written by the "
                "deploy act, never edited -- a new deploy is a new "
                "version, by the pair"),
            "the_other_half_of_the_rule": (
                "a landing that touches a pinned file RE-PINS in the same "
                "round; the non-head census marks a moved digest as "
                "DEPLOY_PIN_STALE so the drift is seen in daylight rather "
                "than at 00:06Z as a refusal")}


def head_pin(decl_dir: Path | None = None) -> dict:
    """The pin in force, resolved by the shared chain resolver."""
    import declaration_chain as DC                            # noqa: PLC0415
    d = Path(decl_dir) if decl_dir else DECL_DIR
    try:
        r = DC.resolve_head(d, FAMILY)
    except DC.ChainRefused as e:
        raise PinRefused(f"REFUSED: {e}") from e
    if r["orphan_branches"]:
        raise PinRefused(
            f"REFUSED: DEPLOY_PIN_DOES_NOT_RESOLVE_TO_ONE_HEAD -- orphan "
            f"branch(es) {[o['version'] for o in r['orphan_branches']]}. "
            f"Which bytes the unit was deployed at would have no answer.")
    return {"name": r["name"], "path": r["path"], "sha256": r["sha256"],
            "doc": r["doc"], "version": r["version"]}


def write_pin(decl_dir: Path | None = None,
              record: Path | None = None) -> dict:
    """The next pin version, through the shared compare-and-swap."""
    import declaration_chain as DC                            # noqa: PLC0415
    d = Path(decl_dir) if decl_dir else DECL_DIR
    payload = payload_from_record(record)
    payload["as_of_utc"] = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        head = DC.resolve_head(d, FAMILY)
    except DC.ChainRefused:
        #: FIRST OF FAMILY, declared as such -- never an omission.
        payload["version"] = 1
        payload["first_of_family"] = True
        payload["why_first_of_family"] = (
            "no prior pin exists; the link is DECLARED rather than left out")
        p = d / f"{FAMILY}_v1.json"
        if p.exists():
            raise PinRefused(
                f"REFUSED: VERSION_PATH_EXISTS -- {p.name} is there and the "
                f"family did not resolve; that is a repair, not a deploy.")
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {"path": str(p), "name": p.name, "sha256": _sha(p),
                "version": 1, "first_of_family": True}
    payload["version"] = head["version"] + 1
    payload["supersedes"] = {"path": head["path"], "sha256": head["sha256"],
                             "rule": "13 -- vN+1 by the {path, sha256} PAIR"}
    return DC.write_next_version(d, FAMILY, payload, head)


def stale_pins(root: Path | None = None,
               decl_dir: Path | None = None) -> dict:
    """Pinned files whose bytes on disk are not what the head pin says.

    ***THIS IS THE HALF THAT RUNS IN DAYLIGHT.*** The unit's guard asks
    the same question at 00:06Z and answers it by refusing; the census
    asks it whenever it runs, and a landing that moved a pinned file is a
    named mark instead of a lost night."""
    r = Path(root) if root else _tree()
    try:
        head = head_pin(decl_dir)
    except PinRefused as e:
        return {"status": "NO_PIN", "why": str(e)[:200], "n_stale": 0,
                "stale": []}
    stale, missing = [], []
    for f in head["doc"].get("files") or []:
        q = r / f["path"]
        if not q.is_file():
            missing.append({"path": f["path"], "status": "PINNED_FILE_ABSENT"})
            continue
        now = _sha(q)
        if now != f["sha256"]:
            stale.append({"path": f["path"], "tier": f.get("tier"),
                          "pinned_sha256": f["sha256"][:16],
                          "on_disk_sha256": now[:16],
                          "status": "DEPLOY_PIN_STALE"})
    return {"status": ("DEPLOY_PIN_STALE" if stale else
                       "PINNED_FILE_ABSENT" if missing else "CLEAN"),
            "pin": {"name": head["name"], "sha256": head["sha256"],
                    "commit": head["doc"].get("commit"),
                    "n_files": len(head["doc"].get("files") or [])},
            "n_stale": len(stale), "stale": stale,
            "n_missing": len(missing), "missing": missing,
            "why": ("a landing that changes a pinned file without a new pin "
                    "is drift the unit will refuse at 00:06Z; here it is a "
                    "MARK, in daylight, naming the file")}


def selftest() -> tuple:
    import tempfile
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond)})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    t = Path(tempfile.mkdtemp(prefix="da115pin_"))
    (t / "live" / "pm_research").mkdir(parents=True)
    d = t / "decl"
    d.mkdir()
    pinned = t / "live" / "pm_research" / "x.sh"
    pinned.write_text("#!/bin/sh\necho one\n")
    rec = t / "rec.json"
    rec.write_text(json.dumps({
        "installed_units": UNIT, "deployed_commit": "c" * 40,
        "deployed_at_utc": "2026-09-07T02:00:00Z", "deployed_by": "test",
        "files": [{"path": "live/pm_research/x.sh",
                   "sha256": _sha(pinned), "tier": "REFUSE"}]}))
    v1 = write_pin(d, rec)
    clean = stale_pins(t, d)
    pinned.write_text("#!/bin/sh\necho two\n")            # a landing moves it
    dirty = stale_pins(t, d)
    rec.write_text(json.dumps({
        "installed_units": UNIT, "deployed_commit": "d" * 40,
        "deployed_at_utc": "2026-09-07T02:10:00Z", "deployed_by": "test",
        "files": [{"path": "live/pm_research/x.sh",
                   "sha256": _sha(pinned), "tier": "REFUSE"}]}))
    v2 = write_pin(d, rec)                                 # re-pinned
    after = stale_pins(t, d)
    ck("R-741 / REV 88 S3 -- ***A LANDING THAT MOVES A PINNED FILE IS SEEN "
       "IN DAYLIGHT, NOT AT 00:06Z.*** The unit drifted on my own landings "
       "and the FIRST anyone knew was a refusal that had already cost a "
       "night's verdict. The pin is a DECLARATION with a version chain "
       "(v1 first-of-family, DECLARED as such; every deploy after it "
       "through the shared compare-and-swap, never an edit), and the "
       "census asks the guard's own question whenever it runs: a pinned "
       "file whose digest moved without a new pin is `DEPLOY_PIN_STALE`, "
       "***naming the file***. Driven: clean -> CLEAN; the file edited -> "
       "STALE naming it; RE-PINNED -> clean again",
       v1["version"] == 1 and v1.get("first_of_family") is True
       and clean["status"] == "CLEAN" and clean["n_stale"] == 0
       and dirty["status"] == "DEPLOY_PIN_STALE" and dirty["n_stale"] == 1
       and dirty["stale"][0]["path"] == "live/pm_research/x.sh"
       and v2["version"] == 2
       and after["status"] == "CLEAN",
       f"v1 first-of-family; clean -> {clean['status']}; after an edit -> "
       f"{dirty['status']} at {dirty['stale'][0]['path']}; after re-pin "
       f"(v{v2['version']}) -> {after['status']}")
    (t / "live" / "pm_research" / "x.sh").unlink()
    gone = stale_pins(t, d)
    ck("AND A PINNED FILE THAT IS GONE IS ITS OWN STATUS, never a silent "
       "pass: `PINNED_FILE_ABSENT`, naming it -- an absent file has no "
       "digest to compare, and reading that as CLEAN is how a deleted "
       "guard would deploy itself",
       gone["status"] == "PINNED_FILE_ABSENT" and gone["n_missing"] == 1,
       f"the pinned file removed -> {gone['status']} "
       f"({gone['missing'][0]['path']})")
    try:
        payload_from_record(t / "no_such_record.json")
        absent = "ADMITTED"
    except PinRefused as e:
        absent = str(e).split(" -- ")[0].replace("REFUSED: ", "")
    ck("AND THE PIN IS WRITTEN **FROM THE DEPLOY ACT'S OWN RECORD**: with "
       "no record there is nothing to pin, and inventing one would make "
       "the pin a claim about a deploy that never happened",
       absent == "DEPLOY_RECORD_ABSENT", f"no record -> {absent}")
    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write", action="store_true",
                    help="write the next pin version from the deploy record")
    ap.add_argument("--check", action="store_true",
                    help="are the pinned files still what the pin says?")
    a = ap.parse_args()
    if a.selftest:
        return 1 if selftest()[1] else 0
    if a.write:
        r = write_pin()
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0
    if a.check:
        r = stale_pins()
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0 if r["status"] == "CLEAN" else 7
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
