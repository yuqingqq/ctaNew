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


VERSION_RE = re.compile(r"_v(\d+)\.json$")


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
    superseded, broken = {}, []
    for name, e in loaded.items():
        sup = (e["doc"] or {}).get("supersedes")
        if isinstance(sup, dict) and sup.get("path"):
            prev = Path(sup["path"]).name
            if prev in loaded and loaded[prev]["sha256"] != sup.get("sha256"):
                broken.append({"version": name, "names": prev,
                               "pair_sha256": sup.get("sha256"),
                               "on_disk_sha256": loaded[prev]["sha256"]})
            superseded.setdefault(prev, []).append(name)
        elif isinstance(sup, str) and sup:
            broken.append({"version": name, "names": sup,
                           "why": "supersedes is a bare string, not the "
                                  "{path, sha256} pair -- unfollowable"})
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
