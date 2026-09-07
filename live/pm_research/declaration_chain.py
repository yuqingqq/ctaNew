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
import stat
import tempfile
from pathlib import Path


class ChainRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


#: `<family>_v<N>.json` AND `<family>_v<N>__<stamp>.json`. The stamped form
#: is what the design family actually uses (22 of its 23 files); a pattern
#: anchored on `_v<N>.json$` parsed EVERY one of them as version 0.
VERSION_RE = re.compile(r"_v(\d+)(?:__[^/]*)?\.json$")

#: A LINK IS A PAIR (R-608), and a digest that is not 64 lowercase hex is
#: not a digest. Used by every link shape this resolver follows.
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _version_of(p: Path) -> int:
    m = VERSION_RE.search(Path(p).name)
    return int(m.group(1)) if m else 0


def plain_create_mode() -> int:
    """The mode a PLAIN create would produce here: `0o666 & ~umask`.

    Reading the umask means SETTING it -- POSIX offers no read-only call --
    so it is set to 0 and restored on the next line. The window is two
    syscalls wide and process-global; this module writes one declaration per
    landing, from short-lived single-threaded producers. The alternative, a
    hard-coded 0o644, is the literal that goes wrong on the first box whose
    umask is not 0o022 -- and it is wrong on THIS one (umask 0o002, so a
    plain create is 0o664).

    The falsifier does not trust this function. CELL 16 compares a written
    version's mode against a file the cell creates ITSELF, the ordinary way,
    in the same directory -- so a wrong umask read fails the cell rather
    than agreeing with it.
    """
    um = os.umask(0)
    os.umask(um)
    return 0o666 & ~um


#: BE 91: kept so the name this landed under still resolves. One
#: implementation, two names -- never two implementations.
_plain_create_mode = plain_create_mode


def resolve_head(directory, family: str) -> dict:
    """The chain's head, its pairs, and any ORPHAN BRANCHES.

    A MERGE VERSION carries `also_supersedes`: a list of {path, sha256}
    naming other tips it supersedes. Those tips become REACHABLE and stop
    being orphans -- a fork that has been merged is history, not an open
    branch. Each merge pair is verified against the file on disk
    (`MERGE_PAIR_MISMATCH`, `MERGE_TARGET_ABSENT`).

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

    def _digest_fault(want):
        """None if `want` is a usable digest; else WHAT IT HAS, said plainly.

        R-608: the link IS the pair. A digest that is not 64 lowercase hex
        verifies nothing -- `None`, an empty string, a truncated value, a
        placeholder -- and a link that cannot verify is HALF-WRITTEN.
        """
        if want is None:
            return "no `sha256` at all"
        if not isinstance(want, str):
            return f"a {type(want).__name__} where the digest belongs"
        if not want:
            return "an empty `sha256`"
        if not DIGEST_RE.match(want):
            return (f"`sha256` = {want[:16]!r}... ({len(want)} chars), which "
                    f"is not 64 lowercase hex")
        return None

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
            if isinstance(last, (list, tuple)) and len(last) >= 1:
                # THE DIGEST IS RETURNED RAW, never `str()`-ed: `str(None)`
                # is the string "None", which would have been compared as a
                # digest and reported as a MISMATCH -- the wrong name for a
                # link that simply has none. A chain element is a pair like
                # any other, so an element without a digest is HALF-WRITTEN
                # and refuses under that name (BE 82).
                return (Path(str(last[0])).name,
                        last[1] if len(last) >= 2 else None, "chain")
            return None, None, "chain-malformed"
        if isinstance(sup, str) and sup:
            return None, None, "bare-string"
        return None, None, "unknown"

    merges = {}
    for name, e in loaded.items():
        # A MERGE VERSION closes a fork. Besides its own `supersedes` -- the
        # single predecessor on its own path -- it may carry
        # `also_supersedes`: a list of {path, sha256} naming OTHER TIPS it
        # supersedes. A tip named there is REACHABLE from the head and stops
        # being an orphan; no content is taken from it. A resolver that does
        # not read the field reports the merged tips as orphans forever, so
        # its after-the-merge answer is vacuous -- which is what this one
        # did to DE's v25 (Q-BE-322).
        also = (e["doc"] or {}).get("also_supersedes") or []
        if isinstance(also, list) and also:
            named = []
            for item in also:
                if not isinstance(item, dict) or not item.get("path"):
                    broken.append({"version": name, "shape": "merge-malformed",
                                   "supersedes": str(item)[:120],
                                   "why": "an `also_supersedes` entry is not "
                                          "a {path, sha256} pair"})
                    continue
                tname = Path(str(item["path"])).name
                if tname not in loaded:
                    broken.append({"version": name, "shape": "merge-absent",
                                   "names": tname,
                                   "why": "MERGE_TARGET_ABSENT: the merge "
                                          "names a version that is not in "
                                          "this family on disk"})
                    continue
                _f = _digest_fault(item.get("sha256"))
                if _f is not None:
                    broken.append({"version": name, "shape": "half-written",
                                   "names": tname,
                                   "link_shape": "also_supersedes",
                                   "has": _f,
                                   "why": "HALF_WRITTEN_LINK: a merge link "
                                          "is a {path, sha256} pair like any "
                                          "other, and this one has no usable "
                                          "digest -- it closes nothing"})
                    continue
                if loaded[tname]["sha256"] != item.get("sha256"):
                    broken.append({"version": name, "shape": "merge-mismatch",
                                   "names": tname,
                                   "pair_sha256": item.get("sha256"),
                                   "on_disk_sha256": loaded[tname]["sha256"],
                                   "why": "MERGE_PAIR_MISMATCH: the merge "
                                          "names a tip whose digest differs "
                                          "from the file on disk"})
                    continue
                superseded.setdefault(tname, []).append(name)
                named.append(tname)
            if named:
                merges[name] = named
        sup = (e["doc"] or {}).get("supersedes")
        prev, want, shape = _predecessor(sup)
        shapes[name] = ("merge+" + shape) if name in merges else shape
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
        # THE REGRESSION THIS REPLACES (DA 106's own cell, R-718, MEM 228).
        # `_predecessor` returned `(path, None, "pair")` for a `{path}`-only
        # `supersedes`, and the comparison below was guarded by `and want`
        # -- so a missing digest SKIPPED the check, the link was followed
        # unverified, and `link_shapes` even called it a `pair`. The
        # coordinator's drive: `fam_v2 {supersedes: {path: fam_v1.json}}`
        # resolved to head `fam_v2.json`, orphans [], shapes `{'fam_v2':
        # 'pair'}`. A guard that skips its own check when the input is
        # missing is not a guard; it is the input deciding whether to be
        # checked.
        _fault = _digest_fault(want)
        if _fault is not None:
            shapes[name] = shapes[name] + "-HALF_WRITTEN"
            broken.append({"version": name, "shape": "half-written",
                           "names": prev, "link_shape": shape,
                           "has": _fault,
                           "why": "HALF_WRITTEN_LINK: the supersession "
                                  "carries a path and no usable digest, so "
                                  "it verifies nothing"})
            continue
        if prev in loaded and loaded[prev]["sha256"] != want:
            broken.append({"version": name, "names": prev,
                           "shape": shape,
                           "pair_sha256": want,
                           "on_disk_sha256": loaded[prev]["sha256"]})
        superseded.setdefault(prev, []).append(name)
    m_absent = [b for b in broken if b.get("shape") == "merge-absent"]
    if m_absent:
        raise ChainRefused(
            f"MERGE_TARGET_ABSENT: "
            f"{[(b['version'], b['names']) for b in m_absent]}. A merge that "
            f"names a version not present in this family cannot make it "
            f"reachable, and a tip that does not exist is not a tip that "
            f"was closed.")
    # ORDER. After MERGE_TARGET_ABSENT (a named file that is not there is a
    # stronger fact than a link that cannot verify) and BEFORE the mismatch
    # refusals, because "the bytes moved" MISSTATES a link that never
    # carried a digest -- and a message that misstates the cause sends the
    # reader to repair the wrong thing (REV 54 §0).
    half = [b for b in broken if b.get("shape") == "half-written"]
    if half:
        raise ChainRefused(
            "HALF_WRITTEN_LINK: "
            + "; ".join(f"{b['version']} names {b['names']} in its "
                        f"`{b['link_shape']}` link with {b['has']}"
                        for b in half)
            + ". R-608: the link IS the pair. A supersession carrying a path "
              "and no usable digest verifies nothing, so it is never "
              "followed, never counted as a `pair`, and never reported as "
              "an orphan -- the repair is the link, and it belongs to the "
              "seat that wrote it.")
    m_bad = [b for b in broken if b.get("shape") in ("merge-mismatch",
                                                     "merge-malformed")]
    if m_bad:
        raise ChainRefused(
            f"MERGE_PAIR_MISMATCH: "
            f"{[(b['version'], b.get('names'), b.get('why', '')[:40]) for b in m_bad]}. "
            f"A merge link is a {{path, sha256}} pair like any other: naming "
            f"a tip whose bytes have moved closes nothing.")
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
    merged_tips = {t for ts in merges.values() for t in ts}

    def _fork_status(branches):
        """A fork is MERGED when none of its BRANCHES is still an open tip.

        The first form of this asked whether the fork's PREDECESSOR was in
        `merged_tips` -- the wrong end of the link entirely, and it reported
        OPEN for a fork whose every branch had just been merged. A branch is
        open if nothing supersedes it and no merge names it.
        """
        open_branches = [b for b in branches
                         if b not in superseded and b not in merged_tips]
        return ("MERGED" if not open_branches else "OPEN"), open_branches
    return {"family": family, "dir": str(d),
            "name": head["path"].name, "path": str(head["path"]),
            "sha256": head["sha256"], "doc": head["doc"],
            "version": head["version"], "n_versions": len(loaded),
            "orphan_branches": orphans,
            "link_shapes": shapes,
            "merges": merges,
            "n_merge_links": sum(len(v) for v in merges.values()),
            "forks_two_versions_superseding_one": forks,
            "fork_status": {p: _fork_status(bs)[0]
                            for p, bs in forks.items()},
            "fork_open_branches": {p: _fork_status(bs)[1]
                                   for p, bs in forks.items()
                                   if _fork_status(bs)[1]},
            "merged_tips": sorted(merged_tips),
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
        # THE MODE (DE's finding at the 2026-09-07 reset, R-746). `mkstemp`
        # is a SECRET-file constructor: it creates 0600 by design, and
        # `os.replace` carries that mode to the landed version -- so every
        # version this module has ever written was readable only by the seat
        # that wrote it. A declaration is the opposite of a secret: five
        # seats and the reviewer resolve these chains, and rule 20 makes
        # them the shared contract. Set the mode a plain create in this
        # directory would have produced, so a landed version is exactly as
        # readable as its neighbours.
        #
        # BEFORE the rename, not after: the destination then never exists at
        # the wrong mode, not even for the width of one syscall.
        os.chmod(tmp, plain_create_mode())
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
    # THE MERGE CELLS (Q-BE-322). A fork that has been merged is HISTORY,
    # not an open branch: a resolver that does not read `also_supersedes`
    # reports the merged tips as orphans forever, and its after-the-merge
    # answer is vacuous.
    dm = Path(tempfile.mkdtemp(prefix="dc_merge_"))
    (dm / f"{FAM}_v1.json").write_text(json.dumps(pl(None, "base"), indent=1,
                                                  sort_keys=True) + "\n")
    b1 = {"path": str(dm / f"{FAM}_v1.json"), "sha256": _sha(dm / f"{FAM}_v1.json")}
    for n in (2, 3):
        (dm / f"{FAM}_v{n}.json").write_text(
            json.dumps(pl(b1, f"branch {n}"), indent=1, sort_keys=True) + "\n")
    pre = resolve_head(dm, FAM)
    v4 = pl({"path": str(dm / f"{FAM}_v3.json"),
             "sha256": _sha(dm / f"{FAM}_v3.json")}, "the merge")
    v4["also_supersedes"] = [
        {"path": str(dm / f"{FAM}_v2.json"),
         "sha256": _sha(dm / f"{FAM}_v2.json"),
         "what_this_link_is": "a MERGE link: the tip becomes reachable"}]
    (dm / f"{FAM}_v4.json").write_text(json.dumps(v4, indent=1, sort_keys=True) + "\n")
    hm = resolve_head(dm, FAM)
    ok([o["version"] for o in pre["orphan_branches"]] == [f"{FAM}_v2.json"]
       and hm["name"] == f"{FAM}_v4.json"
       and hm["orphan_branches"] == []
       and hm["n_merge_links"] == 1
       and list(hm["fork_status"].values()) == ["MERGED"],
       f"CELL 8 A MERGE CLOSES A FORK: before, orphans "
       f"{[o['version'] for o in pre['orphan_branches']]}; after v4 merges "
       f"v2, head {hm['name']}, orphans {hm['orphan_branches']}, "
       f"fork_status {list(hm['fork_status'].values())} -- a fork that was "
       f"merged is history, not an orphan")
    dmm = Path(tempfile.mkdtemp(prefix="dc_mergebad_"))
    for n, sup in ((1, None), (2, None)):
        (dmm / f"{FAM}_v{n}.json").write_text(
            json.dumps(pl(sup, f"v{n}"), indent=1, sort_keys=True) + "\n")
    bad = pl({"path": str(dmm / f"{FAM}_v2.json"),
              "sha256": _sha(dmm / f"{FAM}_v2.json")}, "bad merge")
    bad["also_supersedes"] = [{"path": str(dmm / f"{FAM}_v1.json"),
                               "sha256": "e" * 64}]
    (dmm / f"{FAM}_v3.json").write_text(json.dumps(bad, indent=1, sort_keys=True) + "\n")
    try:
        resolve_head(dmm, FAM); mm = "NOT REFUSED"
    except ChainRefused as e:
        mm = str(e).split(":")[0]
    ok(mm == "MERGE_PAIR_MISMATCH",
       f"CELL 9 a merge naming a tip whose digest has MOVED is REFUSED by "
       f"name ({mm}) -- a merge link is a pair like any other and naming "
       f"moved bytes closes nothing")
    dma = Path(tempfile.mkdtemp(prefix="dc_mergeabs_"))
    (dma / f"{FAM}_v1.json").write_text(json.dumps(pl(None), indent=1,
                                                   sort_keys=True) + "\n")
    ab = pl({"path": str(dma / f"{FAM}_v1.json"),
             "sha256": _sha(dma / f"{FAM}_v1.json")}, "absent merge")
    ab["also_supersedes"] = [{"path": str(dma / f"{FAM}_v99.json"),
                              "sha256": "f" * 64}]
    (dma / f"{FAM}_v2.json").write_text(json.dumps(ab, indent=1, sort_keys=True) + "\n")
    try:
        resolve_head(dma, FAM); ma = "NOT REFUSED"
    except ChainRefused as e:
        ma = str(e).split(":")[0]
    ok(ma == "MERGE_TARGET_ABSENT",
       f"CELL 10 a merge naming a version NOT PRESENT is REFUSED by name "
       f"({ma}) -- a tip that does not exist is not a tip that was closed")

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

    # THE HALF-WRITTEN LINK (R-718; DA 106's cell, the coordinator's drive,
    # MEM 228). THREE OUTCOMES ON ONE FIXTURE, so the cells distinguish a
    # link with NO digest from one with a WRONG digest from one that is
    # right: before BE 82 the first two were not distinguished -- the digest
    # comparison was guarded by `and want`, so an absent digest skipped the
    # check entirely and the link was FOLLOWED and counted as a `pair`.
    dh = Path(tempfile.mkdtemp(prefix="dc_half_"))
    (dh / f"{FAM}_v1.json").write_text(
        json.dumps(pl(None, "root"), indent=1, sort_keys=True) + "\n")
    v1p = dh / f"{FAM}_v1.json"
    v2p = dh / f"{FAM}_v2.json"

    def _v2(sup):
        v2p.write_text(json.dumps(pl(sup, "second"), indent=1,
                                  sort_keys=True) + "\n")

    _v2({"path": f"{FAM}_v1.json"})            # the coordinator's drive
    try:
        rh = resolve_head(dh, FAM)
        half = (f"NOT REFUSED -- head {rh['name']}, orphans "
                f"{[o['version'] for o in rh['orphan_branches']]}, shapes "
                f"{rh['link_shapes']}")
    except ChainRefused as e:
        half = str(e)
    ok(half.startswith("HALF_WRITTEN_LINK:") and f"{FAM}_v2.json" in half
       and "no `sha256` at all" in half,
       f"CELL 11 A PATH-ONLY LINK IS REFUSED BY NAME, naming the file and "
       f"WHAT IT HAS: {half[:200]!r}. This is the coordinator's drive "
       f"(R-718): it used to answer `head {FAM}_v2.json, orphans [], "
       f"link_shapes {{'{FAM}_v2.json': 'pair'}}` -- an unverified link "
       f"followed, and counted as a pair")
    _v2({"path": f"{FAM}_v1.json", "sha256": "0" * 64})
    try:
        resolve_head(dh, FAM)
        wrong = "NOT REFUSED"
    except ChainRefused as e:
        wrong = str(e).split(":")[0]
    ok(wrong == "DECLARATION_LINK_CORRUPTED",
       f"CELL 12 and a link whose digest is WELL-FORMED BUT WRONG is still "
       f"refused under its own name ({wrong}) -- the two faults are not the "
       f"same fault, and `the bytes moved` misstates a link that never "
       f"carried a digest")
    _v2({"path": f"{FAM}_v1.json", "sha256": _sha(v1p)})
    rok = resolve_head(dh, FAM)
    ok(rok["name"] == f"{FAM}_v2.json" and rok["version"] == 2
       and rok["orphan_branches"] == []
       and rok["link_shapes"][f"{FAM}_v2.json"] == "pair",
       f"CELL 13 POSITIVE CONTROL ON THE SAME FIXTURE: with the RIGHT "
       f"digest the link is followed, head {rok['name']}, orphans [], shape "
       f"`pair` -- the refusal is about the missing digest, not about the "
       f"link")
    _v2({"chain": [[f"{FAM}_v1.json"]]})
    try:
        resolve_head(dh, FAM)
        chain_half = "NOT REFUSED"
    except ChainRefused as e:
        chain_half = str(e)
    ok(chain_half.startswith("HALF_WRITTEN_LINK:")
       and "no `sha256` at all" in chain_half and "`chain`" in chain_half,
       f"CELL 14 THE EARLY `chain` FORM KEEPS THE SAME RULE -- its elements "
       f"are pairs too, so an element without a digest is HALF-WRITTEN and "
       f"refuses under that name, not as a broken shape: {chain_half[:160]!r}")
    _v2({"path": f"{FAM}_v1.json", "sha256": _sha(v1p)})
    (dh / f"{FAM}_v3.json").write_text(json.dumps(
        dict(pl({"path": f"{FAM}_v2.json", "sha256": _sha(v2p)}, "merge"),
             also_supersedes=[{"path": f"{FAM}_v1.json"}]), indent=1,
        sort_keys=True) + "\n")
    try:
        resolve_head(dh, FAM)
        merge_half = "NOT REFUSED"
    except ChainRefused as e:
        merge_half = str(e)
    ok(merge_half.startswith("HALF_WRITTEN_LINK:")
       and "also_supersedes" in merge_half,
       f"CELL 15 AND A MERGE LINK IS A PAIR LIKE ANY OTHER: an "
       f"`also_supersedes` entry with no digest refuses as HALF_WRITTEN_LINK "
       f"-- it used to be reported as MERGE_PAIR_MISMATCH, which says the "
       f"bytes moved when there was nothing to compare: {merge_half[:160]!r}")
    # THE MODE CELL (CELL 16; BE 90, on DE's finding at the second reset).
    # Every version this module wrote before today landed at 0600 -- the
    # mode `mkstemp` creates and `os.replace` carries -- so the shared
    # contract was owner-only on disk.
    #
    # THE CELL ESTABLISHES ITS OWN BASELINE INSIDE ITSELF (REV 83 §5): it
    # creates a file the ordinary way, in the SAME directory, in the SAME
    # process, and asserts the written version's mode EQUALS that. A literal
    # (0o644) would be a literal agreeing with a literal, and would be wrong
    # on this box, whose umask is 0o002.
    #
    # THIRD OUTCOME, NAMED, NEVER SKIPPED (R-649): on a box whose umask is
    # itself owner-only, a CORRECT implementation also yields 0600. The
    # equality is then the whole assertion and the cell SAYS SO
    # (`BASELINE_IS_OWNER_ONLY`) instead of quietly passing on a readability
    # property it did not demonstrate.
    dmo = Path(tempfile.mkdtemp(prefix="dc_mode_"))
    (dmo / f"{FAM}_v1.json").write_text(json.dumps(pl(None, "root"), indent=1,
                                                   sort_keys=True) + "\n")
    hmo = resolve_head(dmo, FAM)
    probe = dmo / "plain_create.probe"
    probe.write_text("a file made the ordinary way, in this directory\n")
    base_mode = stat.S_IMODE(probe.stat().st_mode)
    wmo = write_next_version(dmo, FAM, pl({"path": hmo["path"],
                                           "sha256": hmo["sha256"]}, "second"),
                             hmo["pair"])
    got_mode = stat.S_IMODE(Path(wmo["path"]).stat().st_mode)
    base_owner_only = base_mode & 0o077 == 0
    note = (" -- BASELINE_IS_OWNER_ONLY: this box's umask makes even a plain "
            "create owner-only, so the equality is the whole assertion and no "
            "readability property is demonstrated here"
            if base_owner_only else "")
    ok(got_mode == base_mode and _sha(Path(wmo["path"])) == wmo["sha256"],
       f"CELL 16 A LANDED VERSION IS AS READABLE AS ITS NEIGHBOURS: "
       f"{wmo['name']} is 0o{got_mode:04o}; a file created the ordinary way "
       f"in the same directory is 0o{base_mode:04o}; owner-only: version "
       f"{got_mode & 0o077 == 0}, plain create {base_owner_only}{note}. "
       f"PRE-FIX THIS CELL READ 0o0600 AGAINST A 0o0664 BASELINE and failed. "
       f"The cell re-digests the file AFTER the chmod and compares it to "
       f"what the write returned ({wmo['sha256'][:16]}...), so `a mode is "
       f"not content` is measured here rather than assumed -- which is what "
       f"the once-off re-mode of the 23 already-landed files leaned on for "
       f"the 9 too large to re-digest under a light GO.")

    print()
    print(f"{ok_n} cells, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys as _sys
    if "--falsify" in _sys.argv:
        raise SystemExit(_falsify())
    print("usage: declaration_chain.py --falsify")
    raise SystemExit(2)
