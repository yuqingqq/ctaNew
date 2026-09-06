"""THE ONE DECLARATION-CHAIN IMPLEMENTATION. Every seat's emitter imports it.

WHY SHARED (R-711, REV 81 §1.4, and the porcelain-parser lesson): three
implementations of one way of reading a tool's output disagreed on four of
twelve lines and no two were wrong in the same place. A chain resolver and a
version writer are infrastructure, not a statistic -- R-235's do-not-harmonize
does not bite here, and three copies would drift the same way.

WHY A COMPARE-AND-SWAP AT THE WRITE, not a re-read before it: on 2026-09-06
the exit-map chain lost two seats' blocks because each seat read v1, composed
its own v2, and landed it -- the second landing overwrote the first in a shared
tree, and a re-read at landing time would not have caught it either, because
both writes were already on disk. The closure has to be AT THE MOMENT OF THE
WRITE: refuse if anything already occupies the next path, and refuse if the
head has moved since the caller read it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


class ChainRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


#: `<family>_v<N>.json` AND `<family>_v<N>__<stamp>.json`. The stamped form
#: is what the design family actually uses (22 of its 23 files); a pattern
#: anchored on `_v<N>.json$` parsed EVERY one of them as version 0.
VERSION_RE = re.compile(r"_v(\d+)(?:__[^/]*)?\.json$")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _version_of(p: Path) -> int:
    m = VERSION_RE.search(Path(p).name)
    return int(m.group(1)) if m else 0


def resolve_head(directory, family: str) -> dict:
    """The chain's head, its pairs, and any ORPHAN BRANCHES.

    REV 81 §5: a fork is REPORTED, not refused. A version that nobody
    supersedes and that is not the head is an orphan branch -- the chain
    still resolves, and whether a fork is a defect under rule 13 is a
    ruling, not this function's to make.

    The head, when more than one version is unsuperseded, is the HIGHEST
    version number among them. That rule is stated here rather than left to
    sort order, because a resolver that picks silently is a resolver whose
    answer nobody can check.
    """
    d = Path(directory)
    files = sorted(d.glob(f"{family}_v*.json"), key=_version_of)
    if not files:
        raise ChainRefused(
            f"DECLARATION_ABSENT: no file matching {family}_v*.json under {d}. A "
            f"check that depends on a declaration FAILS when it is gone; it "
            f"does not skip (R-649).")
    loaded, unparseable = {}, []
    for q in files:
        b = q.read_bytes()
        try:
            loaded[q.name] = {"path": q, "sha256": hashlib.sha256(b).hexdigest(),
                              "doc": json.loads(b), "version": _version_of(q)}
        except json.JSONDecodeError as e:
            unparseable.append({"file": q.name, "error": str(e)})
    if unparseable:
        raise ChainRefused(
            f"DECLARATION_UNPARSEABLE: {[u['file'] for u in unparseable]} "
            f"match the {family} glob and are not valid JSON. PRESENT and "
            f"unreadable is not the same as absent: the file is there to be "
            f"fixed. ({unparseable[0]['error']})")
    superseded, broken, shapes = {}, [], {}

    def _predecessor(sup):
        """(name, sha256, shape) of the DIRECT predecessor, or (None, …).

        Two link shapes exist in this programme and BOTH must be read. The
        `{path, sha256}` pair is the ruled form. The EARLY form -- a dict
        carrying `chain: [[path, sha256], …]` -- is what the design family
        used for 22 of its 23 versions, and a resolver that does not follow
        it reports those versions as ORPHAN BRANCHES: an unread link
        presented as a missing one. That is what this resolver did, and it
        disagreed with DE's on four early design versions (Q-MEM-211,
        reproduced by the coordinator). Never report an unread link as an
        orphan: follow it, or refuse it by name.
        """
        if sup is None:
            return None, None, "root"
        if isinstance(sup, dict) and sup.get("path"):
            return Path(sup["path"]).name, sup.get("sha256"), "pair"
        if isinstance(sup, dict) and isinstance(sup.get("chain"), list) \
                and sup["chain"]:
            last = sup["chain"][-1]
            if isinstance(last, (list, tuple)) and len(last) >= 2:
                return Path(str(last[0])).name, str(last[1]), "chain"
            return None, None, "chain-malformed"
        if isinstance(sup, str) and sup:
            return None, None, "bare-string"
        return None, None, "unknown"

    for name, e in loaded.items():
        sup = (e["doc"] or {}).get("supersedes")
        prev, want, shape = _predecessor(sup)
        shapes[name] = shape
        if shape in ("bare-string", "chain-malformed", "unknown") and sup:
            broken.append({"version": name, "shape": shape,
                           "supersedes": str(sup)[:120],
                           "why": "LINK_SHAPE_UNSUPPORTED: this resolver "
                                  "cannot follow that shape, and reporting "
                                  "the version as an orphan would present "
                                  "an UNREAD link as a MISSING one"})
            continue
        if prev is None:
            continue
        if prev in loaded and want and loaded[prev]["sha256"] != want:
            broken.append({"version": name, "names": prev,
                           "shape": shape,
                           "pair_sha256": want,
                           "on_disk_sha256": loaded[prev]["sha256"]})
        superseded.setdefault(prev, []).append(name)
    unsupported = [b for b in broken if b.get("shape") in
                   ("bare-string", "chain-malformed", "unknown")]
    if unsupported:
        raise ChainRefused(
            f"LINK_SHAPE_UNSUPPORTED: {[(b['version'], b['shape']) for b in unsupported]}. "
            f"This resolver reads the {{path, sha256}} pair and the early "
            f"`chain: [[path, sha256], …]` form; it will not guess at "
            f"another, and it will not report an unread link as an orphan.")
    if broken:
        raise ChainRefused(
            f"DECLARATION_LINK_CORRUPTED: {broken}. Every version is present "
            f"and readable; it is the LINK that is wrong, so the repair is "
            f"the link, not the files.")
    tips = [n for n in loaded if n not in superseded]
    tips.sort(key=lambda n: loaded[n]["version"])
    head = loaded[tips[-1]]
    orphans = [{"version": n, "sha256": loaded[n]["sha256"],
                "supersedes": (loaded[n]["doc"] or {}).get("supersedes")}
               for n in tips[:-1]]
    forks = {prev: names for prev, names in superseded.items() if len(names) > 1}
    return {"family": family, "dir": str(d),
            "name": head["path"].name, "path": str(head["path"]),
            "sha256": head["sha256"], "doc": head["doc"],
            "version": head["version"], "n_versions": len(loaded),
            "orphan_branches": orphans,
            "link_shapes": shapes,
            "forks_two_versions_superseding_one": forks,
            "head_rule": "the highest version number among the versions "
                         "nobody supersedes",
            "pair": {"path": str(head["path"]), "sha256": head["sha256"]}}


def next_version_path(directory, family: str, head: dict) -> Path:
    return Path(directory) / f"{family}_v{head['version'] + 1}.json"


def write_next_version(directory, family: str, payload: dict,
                       head_read: dict) -> dict:
    """WRITE the next version, or REFUSE by name. The closure is HERE.

    `head_read` is the {path, sha256} the caller resolved BEFORE composing
    its payload. Three refusals, each named:

      VERSION_PATH_EXISTS  something already occupies the next path -- a
                           landed version is immutable and this write would
                           be the in-place edit that cost two seats their
                           blocks;
      HEAD_MOVED           the head's digest at WRITE time differs from the
                           one the caller read, so another writer landed in
                           between and this payload supersedes a version
                           that is no longer the head;
      PAIR_MISMATCH        the payload's `supersedes` is not exactly the
                           current head's pair, so the chain it claims is
                           not the chain that exists.

    The write itself is a temp file plus a rename IN THE SAME DIRECTORY, so
    a reader never sees a half-written version.
    """
    d = Path(directory)
    # THE TARGET IS THE CALLER'S INTENT: read+1, not (re-resolved head)+1.
    # Computing it from a fresh resolve made VERSION_PATH_EXISTS
    # unreachable -- the path would always be free by construction and
    # HEAD_MOVED shadowed it in every case. Two writers from v1 both intend
    # v2, and the second must be told THAT, in rule 20's own order: the
    # path first, then the head.
    dst = d / f"{family}_v{_version_of(Path(str(head_read.get('path')))) + 1}.json"
    if dst.exists():
        raise ChainRefused(
            f"VERSION_PATH_EXISTS: {dst.name} already exists. A landed "
            f"version is IMMUTABLE (rule 20): writing here would be the "
            f"in-place edit that removed two seats' blocks from the exit-map "
            f"chain on 2026-09-06. Re-read the head and recompose from it.")
    now = resolve_head(d, family)
    if str(now["sha256"]) != str(head_read.get("sha256")):
        raise ChainRefused(
            f"HEAD_MOVED: the caller read {Path(str(head_read.get('path'))).name} "
            f"at {str(head_read.get('sha256'))[:16]}… but the head at write "
            f"time is {now['name']} at {now['sha256'][:16]}…. Another writer "
            f"landed in between; this payload supersedes a version that is "
            f"no longer the head. Re-read the head and recompose.")
    sup = (payload or {}).get("supersedes")
    if not isinstance(sup, dict) or sup.get("sha256") != now["sha256"] \
            or Path(str(sup.get("path", ""))).name != now["name"]:
        raise ChainRefused(
            f"PAIR_MISMATCH: the payload's `supersedes` is {sup!r}, which is "
            f"not the current head's pair ({now['name']}, "
            f"{now['sha256'][:16]}…). The chain it claims is not the chain "
            f"that exists.")
    fd, tmp = tempfile.mkstemp(dir=str(d), prefix=f".{dst.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, dst)          # atomic within the directory
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"path": str(dst), "name": dst.name, "sha256": _sha(dst),
            "version": now["version"] + 1,
            "superseded": {"path": now["path"], "sha256": now["sha256"]},
            "pair": {"path": str(dst), "sha256": _sha(dst)},
            "written": "temp file + rename in the same directory"}


# --------------------------------------------------------------------------
# RULE 15 AT THE IMPORT SURFACE (Q-MEM-211). This module is imported by other
# seats' emitters, and a falsifier that lives only in one caller's battery
# cannot be fired by a caller from outside. `python3 declaration_chain.py
# --falsify` drives every cell here.
# --------------------------------------------------------------------------
def _falsify() -> int:
    import tempfile
    ok_n, fails = 0, []

    def ok(cond, label):
        nonlocal ok_n
        ok_n += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    FAM = "fixture_family"

    def pl(sup=None, note=""):
        return {"protocol": "FIXTURE", "note": note, "supersedes": sup}

    d = Path(tempfile.mkdtemp(prefix="dc_falsify_"))
    (d / f"{FAM}_v1.json").write_text(json.dumps(pl(None, "first"), indent=1,
                                                 sort_keys=True) + "\n")
    h = resolve_head(d, FAM)
    ok(h["name"] == f"{FAM}_v1.json" and h["version"] == 1
       and h["orphan_branches"] == [],
       f"CELL 0 resolve: head {h['name']} v{h['version']}, no orphans")
    w = write_next_version(d, FAM, pl({"path": h["path"],
                                       "sha256": h["sha256"]}, "second"),
                           h["pair"])
    h2 = resolve_head(d, FAM)
    ok(h2["sha256"] == w["sha256"] and w["version"] == 2,
       f"CELL 1 positive: wrote {w['name']} and the resolver returns it")
    codes = []
    for pay, hr in ((pl({"path": h["path"], "sha256": h["sha256"]}), h["pair"]),
                    (pl({"path": h2["path"], "sha256": h2["sha256"]}),
                     {"path": str(d / f"{FAM}_v2.json"), "sha256": "0" * 64}),
                    (pl({"path": h2["path"], "sha256": "b" * 64}), h2["pair"])):
        try:
            write_next_version(d, FAM, pay, hr)
            codes.append("NOT REFUSED")
        except ChainRefused as e:
            codes.append(str(e).split(":")[0])
    ok(codes == ["VERSION_PATH_EXISTS", "HEAD_MOVED", "PAIR_MISMATCH"],
       f"CELLS 2-4 the three refusals, distinct and by name: {codes}")
    df = Path(tempfile.mkdtemp(prefix="dc_fork_"))
    (df / f"{FAM}_v1.json").write_text(json.dumps(pl(None, "base"), indent=1,
                                                  sort_keys=True) + "\n")
    b = {"path": str(df / f"{FAM}_v1.json"), "sha256": _sha(df / f"{FAM}_v1.json")}
    for n in (2, 3):
        (df / f"{FAM}_v{n}.json").write_text(
            json.dumps(pl(b, f"branch {n}"), indent=1, sort_keys=True) + "\n")
    hf = resolve_head(df, FAM)
    ok(hf["name"] == f"{FAM}_v3.json"
       and [o["version"] for o in hf["orphan_branches"]] == [f"{FAM}_v2.json"],
       f"CELL 5 fork REPORTED not refused: head {hf['name']}, orphans "
       f"{[o['version'] for o in hf['orphan_branches']]}")
    # THE EARLY-SHAPE CELL (Q-MEM-211): a `chain: [[path, sha256], …]` link
    # must be FOLLOWED, not reported as an orphan.
    de = Path(tempfile.mkdtemp(prefix="dc_early_"))
    (de / f"{FAM}_v1__20260101T000000Z.json").write_text(
        json.dumps(pl(None, "root"), indent=1, sort_keys=True) + "\n")
    r1 = de / f"{FAM}_v1__20260101T000000Z.json"
    (de / f"{FAM}_v2__20260102T000000Z.json").write_text(json.dumps(
        pl({"chain": [[r1.name, _sha(r1)]]}, "early form"), indent=1,
        sort_keys=True) + "\n")
    he = resolve_head(de, FAM)
    ok(he["name"] == f"{FAM}_v2__20260102T000000Z.json" and he["version"] == 2
       and he["orphan_branches"] == []
       and he["link_shapes"][he["name"]] == "chain",
       f"CELL 6 the EARLY `chain` form is FOLLOWED: head {he['name']} v"
       f"{he['version']}, orphans {[o['version'] for o in he['orphan_branches']]} "
       f"-- an unread link reported as an orphan is a MISSING link claimed "
       f"where an UNREAD one exists (Q-MEM-211), and the stamped name parses "
       f"as version {he['version']}, not 0")
    du = Path(tempfile.mkdtemp(prefix="dc_unsup_"))
    (du / f"{FAM}_v1.json").write_text(json.dumps(pl(None), indent=1,
                                                  sort_keys=True) + "\n")
    (du / f"{FAM}_v2.json").write_text(json.dumps(pl("just_a_name.json"),
                                                  indent=1, sort_keys=True) + "\n")
    try:
        resolve_head(du, FAM)
        unsup = "NOT REFUSED"
    except ChainRefused as e:
        unsup = str(e).split(":")[0]
    ok(unsup == "LINK_SHAPE_UNSUPPORTED",
       f"CELL 7 an unfollowable shape is REFUSED BY NAME ({unsup}), never "
       f"reported as an orphan")
    print()
    print(f"{ok_n} cells, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys as _sys
    if "--falsify" in _sys.argv:
        raise SystemExit(_falsify())
    print("usage: declaration_chain.py --falsify")
    raise SystemExit(2)
