#!/usr/bin/env python3
"""THE PRODUCTION FORWARD-DAY RUN PATH. Scores stay SEALED.

§10 step 9: score the frozen set UNCHANGED on >=5 later complete UTC days.
`forward_dry_run.py` proves the wiring on synthetic rows and says so; this is
the other half -- the same wiring on REAL inputs, end to end.

A SIBLING DRIVER, NOT A FLAG ON THE SCORER, and the reason is a boundary
worth keeping. `harmful_forward_scorer` applies the frozen artifact and
consumes the mask; that is all it does, and its small dependency set is what
makes "the frozen artifact is applied, never refitted" checkable by reading
it. The run path needs the market ledger, the admissible-window supply, the
replay bridge, the v3 exposure builder, sealing and receipts. Putting those
behind a flag in the scorer would make the scorer's own claim harder to see.
The scorer is IMPORTED here, so there is still exactly one implementation of
scoring.

SEALED (rule 11). Per-action scores and every value metric go to an OUTDIR the
caller names. Nothing is written under data/pm_5min/derived/. The receipt
carries counts, identities and hashes -- never a metric. UNSEALING IS THE
COORDINATOR'S OR THE USER'S ACT, and this file cannot do it: it has no code
that prints a score.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path("/home/yuqing/ctaNew")


def _exec_tree_is_repo(tree: Path = None) -> bool:
    """BE10-R4: does the EXECUTING tree IS `REPO`? Named, never implied.

    From the main tree `EXEC_TREE()` and `REPO` are the same path, so a
    reader of a receipt cannot tell whether a root resolved BY RULE or by
    coincidence. The receipt now carries the answer as a field, and both
    directions of this predicate are driven in the selftest -- a constant
    here would otherwise be invisible from whichever tree agrees with it."""
    return Path(tree if tree is not None else EXEC_TREE()) == REPO


def EXEC_TREE() -> Path:
    """The tree THIS FILE is in — the root for CODE and ANCHORS.

    BE9-C2: BE7-R4 fixed the receipt's commit and dirt but left every other
    root absolute, so a receipt written from a worktree named that worktree's
    HEAD while its anchors were compared against the MAIN tree's files and
    the audit's children read the main tree's entries. Code and anchors now
    resolve in the tree that executes; `data/` deliberately does NOT (a bare
    worktree has no `data/`), and the receipt SAYS SO in `roots`."""
    return Path(__file__).resolve().parents[2]


# DELIBERATELY ABSOLUTE, and published as such: a worktree carries no data.
DERIVED = REPO / "data/pm_5min/derived"
MARKETS = REPO / "data/pm_5min/markets.jsonl"
RATIFICATION_REF = "R-419"          # R-419 supersedes R-418 (DE round 9)

#: R-502: the USER ratified 08-29 for a DEVELOPMENT READ. It does NOT
#: supersede R-419, which stands unchanged and still ratifies the forward
#: race from 20260901 -- the register now carries two ratification fences.
#:
#: KEYED BY DAY, exactly as `USER_ADMISSIONS_BY_DAY` is, and for the same
#: reason: a day not in this table gets the race ref and the race gate, so
#: the development ratification cannot become the driver's mode.
#:
#: AND THE GATE IS CHOSEN WITH IT, NEVER A FLAG. DE round 51 made
#: `race_admissible` load-bearing IN THE REFUSING DIRECTION: the default
#: `require_verified` REFUSES a non-race block BY NAME with
#: `NotForRaceAdmission`, and a development read must reach for a
#: DIFFERENTLY-NAMED function. A flag on the default gate would mean a
#: caller who forgets it gets an admission; a separate name means a caller
#: who forgets it gets a refusal. That asymmetry is the whole design, so
#: this selector returns the ref AND the gate together and there is no
#: parameter anywhere that turns one into the other.
DEVELOPMENT_READ_RATIFICATIONS = {
    "20260829": {
        "ref": "R-502",
        "ratified_by": "USER",
        "recorded_at": "R-502",
        "scope_days": "DEVELOPMENT_READ_DAYS",
        "scope_from": "20260829", "scope_to": "20260829",
        "does_not_supersede": ("R-419 -- the race ratification stands "
                               "unchanged and still ratifies from 20260901"),
        "why_this_day": ("the USER withdrew 08-29 from the race at R-500 and "
                         "kept it readable; this ratifies the POPULATION for "
                         "that read and for nothing else"),
    },
}


def ratification_for(day: str) -> dict:
    """The ref and the GATE for this day, chosen together.

    Returns the race pair for every day but the ones the USER has ratified
    for a development read -- so the race path is what a caller gets by
    default and by omission, and the development path takes naming it."""
    d = DEVELOPMENT_READ_RATIFICATIONS.get(day)
    if d is None:
        return {"ref": RATIFICATION_REF,
                "gate": RAT.require_verified,
                "gate_name": "de_ratification_check.require_verified",
                "kind": "RACE",
                "why": "no development ratification covers this day"}
    return {"ref": d["ref"],
            "gate": RAT.require_verified_for_development_read,
            "gate_name": "de_ratification_check."
                         "require_verified_for_development_read",
            "kind": "DEVELOPMENT_READ",
            "why": d["why_this_day"],
            "record": d}
#: The USER-authorised freeze commit (R-421 §2). Every sha the candidate and
#: its manifest bind equals the blob HERE; the tree moved the anchors in nine
#: commits afterwards. Rule 12: the frozen set is the commit's bytes.
FROZEN_COMMIT = "1b53929"

#: The feed the action-level estimand consumes. Named here so a checker can
#: assert the driver emits it without importing the metric module.
FEED_PROTOCOL = "BE_FORWARD_METRIC_FEED_V1"
FEED_FIELDS = ("slug", "side", "gen", "t0", "t_start", "score",
               "score_incumbent", "any_fill_ahead", "value_cents",
               "preventable_shares", "level")

#: ROUND 28: THE DENOMINATOR. Every number reported so far has been an
#: absolute (cents) or a ratio (rho) with NO SCALE beside it, and "$206 a day"
#: against an unstated notional is not a profitability answer. `shares` and
#: `level` are both on the row the frozen builder produces -- `latency[L]
#: ["preventable_shares"]` and the quote `level` -- and neither reached the
#: feed, so notional could not be formed downstream. They are ADDED HERE, in
#: the driver, because `harmful_exposure_rows.py` is a FROZEN ANCHOR and the
#: run executes the freeze commit's bytes: editing the builder would change
#: nothing at run time and would break the frozen contract for no gain.

#: ROUND 25: THE SECOND ARM. The declared estimand is net cents AGAINST THE
#: INCUMBENT; `increment()` takes TWO score vectors and the driver emitted
#: ONE. That is the round-13 gap in its last form -- capability built, wiring
#: absent -- and it is closed by scoring both arms IN THE SAME REPLAY PASS.
#:
#: CHOSEN ON MEASURED COST, not preference. The incumbent consumes the SAME
#: 60-feature vector the candidate does (the candidate's own
#: `feature_vector_contract` says "54 PM features, then 6 reduced-fine
#: features", and both fits carry len(norm_mu) == 60, len(weights) == 61),
#: and `expected_cancel_value` is the identical p x v construction
#: `phase2_increment_null` uses for INCUMBENT_REWEIGHTED_ONLY. So the second
#: arm costs ONE extra 61-dim dot product pair per row and ONE float per feed
#: row (~+20 MB). Carrying the raw blocks instead would have added
#: 880,766 x 60 floats -- about 1.0 GB of JSON, a 7x blow-up of a sealed
#: artifact, and it would put model INPUTS in it for no gain.
#:
#: BOUND TO THE DECLARED IDENTITY, never to whatever the path holds (BEM-R3):
#: the shas are the ones `be_read_declaration_v1.json` froze before the read.
INCUMBENT_FITS = {
    "btc": {"path": "data/pm_5min/derived/phase2_fits/linear_d_btc.json",
            "sha256": "18701008c2bd18c68dc8c8ba38f49ca34ac83a0d"
                      "9d7587f139f8a03848a6980c"},
    "eth": {"path": "data/pm_5min/derived/phase2_fits/linear_d_eth.json",
            "sha256": "fb371f6352214a9245523a5724e7f645b01c7835"
                      "1f728bb82cba27cc2d704797"},
}


def load_incumbent_fits() -> dict:
    """The per-coin incumbent, verified against the sha the declaration froze.

    A mismatch REFUSES: the estimand is defined against a NAMED incumbent, and
    scoring against a different one would answer a different question while
    looking identical in every count."""
    out = {}
    for coin, d in sorted(INCUMBENT_FITS.items()):
        f = REPO / d["path"]
        if not f.exists():
            raise ForwardDayRefused(
                f"REFUSED: no incumbent fit for {coin} at {f}. The declared "
                f"estimand is an increment OVER the incumbent; without it "
                f"there is no increment to compute.")
        raw = f.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != d["sha256"]:
            raise ForwardDayRefused(
                f"REFUSED: the {coin} incumbent at {f} hashes {got[:16]}, not "
                f"the {d['sha256'][:16]} the read declaration froze. This is "
                f"not the incumbent the estimand names.")
        fit = json.loads(raw)
        if len(fit.get("norm_mu") or ()) != 60:
            raise ForwardDayRefused(
                f"REFUSED: the {coin} incumbent takes "
                f"{len(fit.get('norm_mu') or ())} features, not the 60 the "
                f"candidate's feature_vector_contract emits. The two arms "
                f"must consume the SAME vector or they are not comparable.")
        out[coin] = fit
    return out

#: The latency the feed resolves its value at. Taken from the FROZEN
#: candidate's own declaration, never chosen here.
def _latency_of_record() -> int:
    return int(json.loads(
        (DERIVED / "harmful_reduced_fine_candidate_v1.json").read_text()
    )["target_latency_ms"])

import de_admissible_windows as AW
import de_ratification_check as RAT
import ev_replay_seam as SEAM
import harmful_forward_scorer as FS
# The scheduled-unit prefix is IMPORTED, never restated: DA's preflight
# matches `write_reason` the same way, and two copies of a matching string
# drift apart silently.
from da_governed_verdict_preflight import SCHEDULED_PREFIX


class ForwardDayRefused(RuntimeError):
    """A named refusal. Every gate below refuses by name, never by absence."""


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _as_of() -> str:
    """This run's UTC instant. Rule 8: every quoted population carries its n
    AND its as-of."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _provenance(tree: Path = None) -> dict:
    """The carrying commit and this driver's own bytes.

    A ref alone is not an identity — the dirty flag and the file hash travel
    with it, the same shape the scorer's provenance block uses.

    BE7-R4: the commit and the dirt are read from THE TREE THIS FILE IS IN,
    never from a hardcoded root. With `cwd=REPO` a receipt written from a
    worktree named the MAIN tree's HEAD and inherited its dirtiness while
    `driver_sha256_prefix` correctly named the file that ran — so the receipt
    claimed a commit it did not carry. BE builds in a worktree, so that was
    the normal case, not the exotic one; the programme ruled the same point
    for DA at CO-10. `_git_blob`'s freeze reads are UNAFFECTED: the object
    store is shared, so `git show <ref>:<path>` resolves from any worktree."""
    import subprocess
    root = Path(__file__).resolve().parents[2] if tree is None else Path(tree)

    def git(*a):
        try:
            r = subprocess.run(("git", *a), cwd=str(root), capture_output=True,
                               text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                            # noqa: BLE001
            return None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    me = Path(__file__).resolve()
    return {"carrying_commit": head or "UNAVAILABLE",
            "carrying_commit_resolved": bool(head),
            "working_tree_dirty": ("unknown" if status is None
                                   else bool(status.strip())),
            "provenance_root": str(root),
            "roots": {
                "code_and_anchors": str(EXEC_TREE()),
                "data": str(REPO / "data"),
                "data_is_absolute_because":
                    "a bare worktree carries no data/; the tape and the "
                    "market ledger are the repo's, not the checkout's, and "
                    "this receipt says so rather than implying one root",
                "git_objects": "shared object store — `git show <ref>:<path>` "
                               "resolves the freeze blobs from any worktree, "
                               "so `_git_blob` needs no per-tree root",
                # BE10-R4: from the main tree `code_and_anchors` and the
                # `data` root share a prefix and a reader cannot tell rule
                # from coincidence. This says which.
                "exec_tree_is_repo": _exec_tree_is_repo(),
                "exec_tree_is_repo_means":
                    "true when this run's EXECUTING tree IS the repo root, "
                    "so the code/anchor root and the data root coincide by "
                    "COINCIDENCE; false from a worktree, where they differ "
                    "by RULE and only `data/` stays absolute"},
            "driver": me.name, "driver_sha256_prefix": _sha_file(me)[:16]}


def day_bounds(day: str) -> tuple:
    d = dt.datetime.strptime(day, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    lo = int(d.timestamp())
    return lo, lo + 86400


def _git_blob(ref: str, path: str) -> bytes | None:
    import subprocess
    r = subprocess.run(("git", "show", f"{ref}:{path}"), cwd=str(REPO),
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def materialise_frozen(outdir: Path) -> dict:
    """Write the FROZEN BYTES into the run dir and verify them BEFORE import.

    R-421 §2: the frozen set is the freeze commit's bytes, not the tree's.
    Round 3 refused because the tree moved; this executes what was frozen.

    THE SHA IS CHECKED BEFORE ANYTHING IS IMPORTED. A byte verified after
    import has already run. Each anchor's source is NAMED, because they are
    not all obtainable the same way: code anchors come from
    `git show <freeze>:<path>`; `data/` is gitignored, so a DATA anchor is
    absent from the commit and its only source is disk — where it must still
    equal the manifest's sha, which is the binding either way. A CODE anchor
    absent from the freeze commit REFUSES: that would mean the freeze does not
    contain the code it claims to bind."""
    c = json.loads(FS.CANDIDATE.read_text())
    mp = FS.CANDIDATE.parent / c["manifest"]
    m = json.loads(mp.read_text())
    ps = m.get("pin_semantics") or {}
    default = ps.get("_default", "reproducibility_anchor")
    root = outdir / "frozen"
    per: dict[str, dict] = {}
    for k, want in sorted((m.get("hashes") or {}).items()):
        if ps.get(k, default) != "reproducibility_anchor":
            per[k] = {"source": "state_at_build",
                      "compared": False,
                      "why": "pin_semantics marks this state_at_build; it MUST "
                             "NOT be compared for equality when validating a "
                             "reproduction"}
            continue
        blob = _git_blob(FROZEN_COMMIT, k)
        source = f"git:{FROZEN_COMMIT}"
        if blob is None:
            dp = REPO / k
            if k.endswith(".py"):
                raise ForwardDayRefused(
                    f"REFUSED: CODE anchor {k} is absent from the freeze "
                    f"commit {FROZEN_COMMIT}. A freeze that does not contain "
                    f"the code it binds cannot be executed as frozen.")
            if not dp.exists():
                raise ForwardDayRefused(
                    f"REFUSED: anchor {k} is absent from {FROZEN_COMMIT} and "
                    f"from disk. There is nowhere to obtain the frozen bytes.")
            blob = dp.read_bytes()
            source = ("disk (not tracked at the freeze commit; data/ is "
                      "gitignored, so the manifest's sha is the only binding "
                      "and it is checked here)")
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            raise ForwardDayRefused(
                f"REFUSED: {k} from {source} hashes {got[:16]} but the "
                f"manifest binds {want[:16]}. The bytes are checked BEFORE "
                f"import — a byte verified after import has already run.")
        # ONLY CODE IS MATERIALISED. Writing the DATA anchor into the run
        # dir created a real `frozen/data/` that SHADOWED the symlink below,
        # so the frozen modules resolved their root to an empty tree and the
        # archive index came back 0 — measured, not feared. Data anchors are
        # verified by CONTENT where they live; the symlink points at them.
        if k.endswith(".py"):
            dest = root / k
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            per[k] = {"source": source, "compared": True, "sha256": got,
                      "materialised_to": str(dest)}
        else:
            per[k] = {"source": source, "compared": True, "sha256": got,
                      "materialised_to": None,
                      "why_not_materialised": "a DATA anchor is verified by "
                                              "content where it lives; "
                                              "copying it into the run dir "
                                              "would shadow the data-root "
                                              "symlink the frozen code needs"}
    # THE FROZEN CODE DERIVES ITS DATA ROOT FROM `__file__`. Measured: with
    # the anchors materialised into the run dir, `flow_intensity.PM` resolved
    # to OUTDIR/frozen/data/pm_5min and the archive index came back EMPTY --
    # 0 slugs, silently. Executing frozen bytes out of tree repoints the data
    # root, and an empty index reads as "no windows" rather than as a broken
    # path. The DATA is not frozen (it is the day being scored), so the run
    # dir mirrors the repo's data root by symlink: FREEZE's code, TODAY's
    # data, each named in the receipt.
    dlink = root / "data"
    if not dlink.exists():
        dlink.parent.mkdir(parents=True, exist_ok=True)
        dlink.symlink_to(REPO / "data", target_is_directory=True)
    checked = sum(1 for v in per.values() if v.get("compared"))
    if checked == 0:
        raise ForwardDayRefused(
            "REFUSED: materialisation compared ZERO anchors; a step that "
            "verified nothing must not report a pass (R-289).")
    return {"frozen_commit": FROZEN_COMMIT, "root": str(root),
            "anchors": per, "n_compared": checked,
            "data_root": {"path": str(dlink), "symlink_to": str(REPO / "data"),
                          "why": "the frozen code derives its data root from "
                                 "__file__; the DATA is not frozen (it is the "
                                 "day being scored), so the run dir mirrors "
                                 "the repo's. Code from the freeze, data from "
                                 "today, and the receipt says which is which"},
            "manifest": c["manifest"], "manifest_sha256_bound":
                c.get("manifest_sha256")}


#: The anchor paths the LAST contract check actually read, recorded at the
#: site so the root can be reported even when the contract REFUSES -- which
#: it does today, on the known freeze drift (R-424 §6).
_LAST_ANCHOR_PATHS: list = []
_LAST_CONTRACT_REFUSAL = None


def anchor_drift_root() -> str:
    """The root the anchor-drift comparison ACTUALLY read from.

    BE9-C2 needs this checkable from another tree, and it must be derived
    from the paths the comparison used -- returning `EXEC_TREE()` directly
    was a restatement, and the mutant that points the comparison back at a
    fixed root survived it. This runs the contract and reports the common
    prefix of the anchor paths it examined."""
    import os as _os
    global _LAST_CONTRACT_REFUSAL
    _LAST_CONTRACT_REFUSAL = None
    try:
        assert_frozen_contract()
    except ForwardDayRefused as e:
        # ROUND 21: this used to be `except Exception: pass` with the comment
        # "the refusal is expected; the PATHS are the point". The paths ARE
        # the point here -- but discarding the exception meant the single
        # invocation of the contract in the whole file threw its answer away.
        # It is RECORDED now, and narrowed to the named refusal so an
        # unexpected error still propagates.
        _LAST_CONTRACT_REFUSAL = str(e)
    ex = list(_LAST_ANCHOR_PATHS)
    return _os.path.commonpath(ex) if ex else ""


def roots_data_is_absolute() -> bool:
    """The data root is REPO's on purpose, and that is a checkable claim."""
    return str(DERIVED).startswith(str(REPO)) and not str(
        DERIVED).startswith(str(EXEC_TREE())) or EXEC_TREE() == REPO


def _repo_module_path(name: str) -> Path | None:
    p = EXEC_TREE() / "live/pm_research" / f"{name}.py"
    return p if p.exists() else None


def import_closure(frozen_root: Path, anchors: list) -> dict:
    """Every repo module the anchors reach, and which of them are NOT frozen.

    The anchors run from the run dir; everything they import that is not
    itself an anchor loads from HEAD. That is a fact about this run and the
    receipt states it by name rather than leaving a reader to assume the
    whole closure was frozen."""
    import ast
    seen: set[str] = set()
    stack = [Path(a).stem for a in anchors if a.endswith(".py")]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        fp = (frozen_root / "live/pm_research" / f"{mod}.py")
        if not fp.exists():
            fp = _repo_module_path(mod)
        if fp is None:
            continue
        try:
            tree = ast.parse(fp.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    stack.append(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                stack.append(node.module.split(".")[0])
    anchor_stems = {Path(a).stem for a in anchors if a.endswith(".py")}
    closure, moved = [], []
    for mod in sorted(seen):
        rp = _repo_module_path(mod)
        if rp is None:
            continue
        rel = f"live/pm_research/{mod}.py"
        closure.append(rel)
        if mod in anchor_stems:
            continue
        at_head = hashlib.sha256(rp.read_bytes()).hexdigest()
        blob = _git_blob(FROZEN_COMMIT, rel)
        at_freeze = hashlib.sha256(blob).hexdigest() if blob else None
        if at_freeze != at_head:
            moved.append({"path": rel, "sha256_at_HEAD": at_head[:16],
                          "sha256_at_freeze": (at_freeze or "ABSENT")[:16]})
    return {"closure": closure, "n_in_closure": len(closure),
            "not_frozen_in_closure": moved,
            "n_not_frozen": len(moved),
            "closure_method": "STATIC import walk (ast Import/ImportFrom, "
                              "transitive) over the anchors' source. It is "
                              "what the anchors can REACH, not what this run "
                              "observed executing: a module reached only on a "
                              "branch not taken is still listed, and a "
                              "dynamic import (importlib, __import__) is NOT "
                              "seen at all.",
            "why": f"anchors are imported from the run dir at "
                   f"{FROZEN_COMMIT}; every module STATICALLY REACHABLE from "
                   f"them that is NOT an anchor resolves at HEAD. Those whose "
                   f"bytes differ between HEAD and the freeze are named here "
                   f"— this run is not claiming they were frozen."}


def import_frozen_anchors(frozen_root: Path, anchors: list) -> dict:
    """Import the anchors FROM THE RUN DIR, and prove it by `__file__`."""
    import importlib
    fdir = frozen_root / "live/pm_research"
    sys.path.insert(0, str(fdir))
    stems = [Path(a).stem for a in anchors if a.endswith(".py")]
    for st in stems:
        sys.modules.pop(st, None)
    where = {}
    for st in stems:
        mod = importlib.import_module(st)
        f = Path(getattr(mod, "__file__", "") or "").resolve()
        if frozen_root.resolve() not in f.parents:
            raise ForwardDayRefused(
                f"REFUSED: {st} imported from {f}, which is NOT under the "
                f"frozen run dir {frozen_root}. The anchors must execute the "
                f"freeze's bytes, not the tree's.")
        where[st] = str(f)
    # AND THE DATA ROOT MUST RESOLVE. An empty archive index is
    # indistinguishable from a day with no windows, so this is checked rather
    # than assumed: the frozen module's own root must resolve to the repo's.
    probe = {}
    try:
        import flow_intensity as _fi
        pm = Path(str(getattr(_fi, "PM", ""))).resolve()
        probe = {"frozen_flow_intensity_PM": str(pm),
                 "resolves_to_repo_data": str(pm).startswith(
                     str((REPO / "data").resolve())),
                 "n_archive_slugs": len(_fi._archive_paths())}
        if not probe["resolves_to_repo_data"] or probe["n_archive_slugs"] == 0:
            raise ForwardDayRefused(
                f"REFUSED: the frozen anchors' data root resolves to "
                f"{pm} with {probe['n_archive_slugs']} archive slugs. "
                f"Executing frozen bytes out of tree repoints the data root, "
                f"and an EMPTY index reads as 'no windows' rather than as a "
                f"broken path — which is why it is checked here.")
    except ForwardDayRefused:
        raise
    except Exception as e:                           # noqa: BLE001
        raise ForwardDayRefused(
            f"REFUSED: could not probe the frozen data root ({e}).")
    return {"imported_from": where, "n_imported": len(where),
            "sys_path_head": sys.path[0], "data_root_probe": probe}


# ---------------------------------------------------------------------------
# gate 1 -- the frozen candidate's own reproduction contract (§10 step 1)
# ---------------------------------------------------------------------------
def manifest_keys_read_by_run_path() -> dict:
    """WHICH MANIFEST KEYS DOES THE RUN PATH ACTUALLY READ? Derived from code.

    The question "is this drift load-bearing" must not be answered from a list
    someone typed. Both functions that parse the manifest bind it to `m`, so
    this walks their ASTs and collects every string used to index `m` -- a key
    the run path never reads cannot change what the run does, and a key it
    reads can."""
    import ast as _ast
    tree = _ast.parse(Path(__file__).read_text())
    # REACHABLE FROM `run_forward_day`, computed. My first version scanned
    # every function and picked up `emits_feed` out of the SELFTEST -- a key
    # of a dict that has nothing to do with the manifest. "The run path" has
    # to mean the run path, or the derivation is just a wider list.
    # BE21-R2: a `Call.func`-only walk cannot see a function passed BY
    # REFERENCE -- which is exactly how `frozen_contract_gate` is wired, by
    # `gate("frozen_contract", frozen_contract_gate)`. So the gate, a manifest
    # reader, was invisible to the derivation that decides which manifest keys
    # are load-bearing. The key SET happens to be identical today because the
    # missed reader reads the same two keys; that is luck. A future reader
    # wired the same way and reading a THIRD key would be silently omitted,
    # and a drift in it would be DISCLOSED as survivable instead of refusing
    # -- the one direction this gate must never get wrong. Every Name and
    # Attribute in a function body now counts as an edge, called or not.
    defined = {n.name for n in _ast.walk(tree)
               if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
    edges: dict = {}
    for n in _ast.walk(tree):
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            outs = set()
            for c in _ast.walk(n):
                nm = None
                if isinstance(c, _ast.Name):
                    nm = c.id
                elif isinstance(c, _ast.Attribute):
                    nm = c.attr
                if nm and nm in defined and nm != n.name:
                    outs.add(nm)
            edges[n.name] = outs
    reachable, stack = set(), ["run_forward_day"]
    while stack:
        cur = stack.pop()
        if cur in reachable:
            continue
        reachable.add(cur)
        stack.extend(edges.get(cur, ()))
    reads: dict = {}
    for fn in _ast.walk(tree):
        if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        if fn.name not in reachable:
            continue
        binds = set()
        for n in _ast.walk(fn):
            if (isinstance(n, _ast.Assign) and len(n.targets) == 1
                    and isinstance(n.targets[0], _ast.Name)
                    and isinstance(n.value, _ast.Call)
                    and "loads" in _ast.dump(n.value)
                    and "mp" in _ast.dump(n.value)):
                binds.add(n.targets[0].id)
        if not binds:
            continue
        for n in _ast.walk(fn):
            key = None
            if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
                    and n.func.attr == "get"
                    and isinstance(n.func.value, _ast.Name)
                    and n.func.value.id in binds and n.args
                    and isinstance(n.args[0], _ast.Constant)):
                key = n.args[0].value
            elif (isinstance(n, _ast.Subscript)
                  and isinstance(n.value, _ast.Name)
                  and n.value.id in binds
                  and isinstance(n.slice, _ast.Constant)):
                key = n.slice.value
            if isinstance(key, str):
                reads.setdefault(key, set()).add(fn.name)
    return {"load_bearing_keys": sorted(reads),
            "read_by": {k: sorted(v) for k, v in sorted(reads.items())},
            "n_functions_reachable_from_run_forward_day": len(reachable),
            "derived_from": ("an AST walk of the functions REACHABLE FROM "
                             "`run_forward_day` that parse the manifest -- not "
                             "from a list, not from every function in the "
                             "file, and counting a function passed BY "
                             "REFERENCE as an edge (BE21-R2)"),
            "why": ("a manifest key the run path never reads cannot change "
                    "what the run does; a key it reads can. That is the only "
                    "defensible line between drift that must refuse and drift "
                    "that may be disclosed.")}


def drift_is_fatal(differing, load_bearing) -> bool:
    """THE DECISION RULE, as a pure function so it can be driven both ways.

    Fatal iff the drift touches a key the run path reads. Separated out
    because a rule embedded in the gate can only be tested by provoking the
    gate, and a rule nobody can drive is how a waiver gets written."""
    return bool(set(differing) & set(load_bearing))


def manifest_drift_detail(candidate: Path = None) -> dict:
    """WHAT drifted between the BOUND manifest and the one on disk, and does
    any of it touch a key the run path reads?

    The bound bytes are recoverable: the candidate's `manifest_sha256` is the
    blob at FROZEN_COMMIT, so the diff is computable rather than inferable
    from a sha mismatch alone."""
    cp = Path(candidate or FS.CANDIDATE)
    c = json.loads(cp.read_text())
    mname = c.get("manifest")
    mp = cp.parent / mname
    disk_sha = _sha_file(mp)
    want = c.get("manifest_sha256")
    if disk_sha == want:
        return {"drifted": False, "manifest": mname, "sha256": disk_sha}
    rel = f"data/pm_5min/derived/{mname}"
    raw = _git_blob(FROZEN_COMMIT, rel)
    if raw is None or hashlib.sha256(raw).hexdigest() != want:
        raise ForwardDayRefused(
            f"REFUSED: {mname} on disk hashes {disk_sha[:16]} against the "
            f"bound {str(want)[:16]}, and the BOUND BYTES are not recoverable "
            f"from {FROZEN_COMMIT}. The drift cannot be characterised, so it "
            f"cannot be excused.")
    bound = json.loads(raw)
    disk = json.loads(mp.read_text())
    differing = sorted(k for k in set(bound) | set(disk)
                       if bound.get(k) != disk.get(k))
    lb = manifest_keys_read_by_run_path()["load_bearing_keys"]
    hit = [k for k in differing if k in lb]
    fatal = drift_is_fatal(differing, lb)
    return {"drifted": True, "manifest": mname,
            "bound_sha256": want, "disk_sha256": disk_sha,
            "keys_that_differ": differing,
            "load_bearing_keys": lb,
            "load_bearing_keys_that_drifted": hit,
            "drift_touches_the_run_path": fatal,
            "disclosure": ("this manifest has drifted from the sha the frozen "
                           "candidate binds. The keys above are the whole of "
                           "the difference, and the load-bearing set is "
                           "DERIVED from the code that reads the manifest."),
            }


def frozen_contract_gate(candidate: Path = None) -> dict:
    """THE GATE round 20 said was missing. It refuses what the run DEPENDS on
    and discloses what it does not -- and the difference is computed.

    WHY NOT `assert_frozen_contract` VERBATIM. It compares every bound anchor
    against `EXEC_TREE()/k` -- the WORKING TREE -- and today five of nine have
    moved. Wiring it as-is would refuse every forward run. But the run does
    not execute the working tree: `materialise_frozen` sources each CODE
    anchor from `_git_blob(FROZEN_COMMIT, k)` and refuses a mismatch there, so
    the bytes that run are the freeze's whatever the tree holds. Refusing on
    tree drift would block the race on a question the run does not depend on;
    IGNORING it would hide a real fact. So it is DISCLOSED, in every receipt,
    with names and counts, and the reason it is not fatal is asserted from the
    source of the function that does the sourcing.

    What IS fatal here: the manifest binding when the drift touches a key the
    run path reads, and any bound CODE anchor whose bytes at the freeze commit
    do not match what the manifest binds -- that is the freeze contradicting
    itself and no run should proceed through it."""
    import inspect
    cp = Path(candidate or FS.CANDIDATE)
    c = json.loads(cp.read_text())
    mname = c.get("manifest")
    mp = cp.parent / mname
    m = json.loads(mp.read_text())
    ps = m.get("pin_semantics") or {}
    default = ps.get("_default", "reproducibility_anchor")

    # (1) the manifest's own binding, partitioned by what the run path reads
    md = manifest_drift_detail(cp)
    if md.get("drifted") and md.get("drift_touches_the_run_path"):
        raise ForwardDayRefused(
            f"REFUSED: the manifest {mname} has drifted from the sha the "
            f"frozen candidate binds, and the difference touches "
            f"{md['load_bearing_keys_that_drifted']} -- keys the run path "
            f"READS. The run would materialise from a contract the freeze did "
            f"not bind.")

    # (2) every bound CODE anchor, AT THE FREEZE COMMIT -- the bytes that run
    at_freeze, bad = {}, []
    for k, want in sorted((m.get("hashes") or {}).items()):
        if ps.get(k, default) != "reproducibility_anchor":
            continue
        blob = _git_blob(FROZEN_COMMIT, k)
        if blob is None:
            if k.endswith(".py"):
                bad.append(f"{k}: absent from {FROZEN_COMMIT}")
            continue
        got = hashlib.sha256(blob).hexdigest()
        at_freeze[k] = got == want
        if got != want:
            bad.append(f"{k}: freeze commit has {got[:16]}, manifest binds "
                       f"{want[:16]}")
    if not at_freeze:
        raise ForwardDayRefused(
            "REFUSED: the frozen-contract gate compared ZERO anchors at the "
            "freeze commit. A gate that reads nothing must not report a pass "
            "(R-289).")
    if bad:
        raise ForwardDayRefused(
            f"REFUSED: the freeze contradicts itself -- {len(bad)} bound "
            f"anchor(s) at {FROZEN_COMMIT} do not match what the manifest "
            f"binds: {bad}. These are the bytes the run would execute.")

    # (3) the WORKING-TREE drift: disclosed, never fatal, reason asserted
    tree_drift = []
    for k, want in sorted((m.get("hashes") or {}).items()):
        if ps.get(k, default) != "reproducibility_anchor":
            continue
        tp = EXEC_TREE() / k
        now = _sha_file(tp) if tp.exists() else None
        if now != want:
            tree_drift.append({"anchor": k, "bound": want[:16],
                               "in_tree": (now or "MISSING")[:16]})
    src = inspect.getsource(materialise_frozen)
    sources_from_freeze = "_git_blob(FROZEN_COMMIT" in src
    if tree_drift and not sources_from_freeze:
        raise ForwardDayRefused(
            "REFUSED: anchors have drifted in the working tree AND "
            "`materialise_frozen` no longer sources them from the freeze "
            "commit. The reason tree drift was survivable has gone.")
    # R39, found by `be_prose_audit`: this was the literal "HOLDS". It is
    # reached only when the checks above have passed, so it was never WRONG --
    # but the STRING asserted nothing, and a check added later whose failure
    # did not raise would leave it still reading HOLDS. It is now DERIVED from
    # the conjuncts in this same dict, which is rule 10 applied to my own
    # emission rather than to someone else's.
    _conj = {"all_anchors_match_at_freeze_commit": all(at_freeze.values()),
             "at_least_one_anchor_verified": len(at_freeze) > 0,
             "drift_is_survivable": (not tree_drift) or sources_from_freeze}
    return {
        "contract": "HOLDS" if all(_conj.values()) else "DOES NOT HOLD",
        "contract_conjuncts": _conj,
        "contract_is_derived_not_asserted": True,
        "manifest": mname, "manifest_drift": md,
        "load_bearing_keys": manifest_keys_read_by_run_path(),
        "anchors_verified_at_freeze_commit": len(at_freeze),
        "all_anchors_match_at_freeze_commit": all(at_freeze.values()),
        "working_tree_drift": tree_drift,
        "n_working_tree_drift": len(tree_drift),
        "working_tree_drift_is_not_fatal_because": (
            "`materialise_frozen` sources every CODE anchor from "
            "`_git_blob(FROZEN_COMMIT, k)` and refuses a mismatch there, so "
            "the bytes the run executes are the freeze's whatever the tree "
            "holds. Asserted from that function's own source, not stated: if "
            "it ever stops doing so, this gate refuses instead of disclosing."),
        # the prose above is BACKED by this computed field, which is read from
        # `materialise_frozen`'s own source at call time
        "materialise_frozen_sources_from_the_freeze_commit": sources_from_freeze,
        "disclosed_not_waived": True,
    }


def assert_frozen_contract(candidate: Path = None) -> dict:
    """The artifact must BE the frozen one, and its bound inputs must still be
    the bytes it was frozen against.

    §10 step 1: "require every conditional-model artifact in the hash set,
    cover every bound input in fit-side drift detection". The candidate names
    its manifest by sha and its builder by sha; the manifest's `pin_semantics`
    says `reproducibility_anchor` entries MUST be compared for equality when
    validating a reproduction, and marks the one entry that must not be.

    This gate does not decide what to do about drift -- it refuses and names
    it. Re-stamping a frozen contract to make a run pass would be editing the
    thing being validated (rule 13)."""
    cp = candidate or FS.CANDIDATE
    c = json.loads(cp.read_text())
    if c.get("status") != "FROZEN":
        raise ForwardDayRefused(
            f"REFUSED: {cp.name} status is {c.get('status')!r}, not FROZEN.")
    drift: list[str] = []
    mname = c.get("manifest")
    mp = cp.parent / mname if mname else None
    if not mp or not mp.exists():
        raise ForwardDayRefused(
            f"REFUSED: {cp.name} names manifest {mname!r}, which is absent.")
    msha = _sha_file(mp)
    # THE MANIFEST'S OWN DRIFT IS PARTITIONED, NOT WAIVED. A key the run path
    # READS is fatal; a key it never reads is DISCLOSED in the receipt of every
    # run and is not. The set is derived from the code, so this cannot become a
    # list somebody widens.
    mdrift = manifest_drift_detail(cp)
    if mdrift.get("drifted") and mdrift.get("drift_touches_the_run_path"):
        drift.append(
            f"manifest {mname}: bound {str(c.get('manifest_sha256'))[:16]} "
            f"now {msha[:16]} — and the difference touches "
            f"{mdrift['load_bearing_keys_that_drifted']}, which the run path "
            f"READS")
    m = json.loads(mp.read_text())
    ps = m.get("pin_semantics") or {}
    default = ps.get("_default", "reproducibility_anchor")
    anchors, checked = [], 0
    examined: list = []
    _LAST_ANCHOR_PATHS.clear()
    for k, want in sorted((m.get("hashes") or {}).items()):
        if ps.get(k, default) != "reproducibility_anchor":
            continue
        anchors.append(k)
        # BE9-C2: the anchor drift is measured against the tree the receipt
        # names, not a fixed one.
        p = EXEC_TREE() / k
        examined.append(str(p))
        _LAST_ANCHOR_PATHS.append(str(p))
        now = _sha_file(p) if p.exists() else None
        checked += 1
        if now != want:
            drift.append(f"{k}: bound {want[:16]} now "
                         f"{(now or 'MISSING')[:16]}")
    bsha = c.get("builder_sha256")
    bp = EXEC_TREE() / "live/pm_research/harmful_hazard_model.py"
    if bsha and bp.exists() and _sha_file(bp) != bsha:
        drift.append(f"builder harmful_hazard_model.py: bound {bsha[:16]} "
                     f"now {_sha_file(bp)[:16]}")
    if checked == 0:
        raise ForwardDayRefused(
            "REFUSED: the frozen contract check compared ZERO anchors. A gate "
            "that reads nothing must not report a pass (R-289).")
    if drift:
        raise ForwardDayRefused(
            f"REFUSED: the frozen candidate's reproduction contract does not "
            f"hold against the working tree — {len(drift)} of "
            f"{checked + 1} bound inputs have moved. The forward rows would "
            f"be produced by code the freeze did not bind, which is a "
            f"different program from the one being raced (§10 step 1). "
            f"Drift: {drift}. Re-stamping the contract to make this pass "
            f"would edit the thing being validated; that is the "
            f"coordinator's or the USER's act, not this driver's.")
    return {"candidate": cp.name, "candidate_sha256": _sha_file(cp),
            "anchors_examined": examined,
            "manifest": mname, "manifest_sha256": msha,
            "manifest_drift": mdrift,
            "anchors_checked": checked, "anchor_keys": anchors,
            "builder_sha256": bsha, "contract": "HOLDS"}


# ---------------------------------------------------------------------------
# gate 2 -- the day is closed, and its verdict was written by the scheduled unit
# ---------------------------------------------------------------------------
#: THE ADMISSION, AND IT IS A RECORD RATHER THAN A RELAXED GATE.
#:
#: 08-29's verdict was superseded by BE (Q-DA-180 item 2) to carry the
#: era-admission guard. The supersede is CORRECT and it is also
#: unattributable: its `write_reason` is BE's, so `day_closed_and_attributed`
#: refuses it -- while the bytes it REPLACED, which do carry genuine
#: scheduled-unit attribution, are the pre-era-guard ones. The gate therefore
#: admits the verdict known to be wrong and refuses the one known to be right.
#:
#: The USER ruled the narrowest of the three options: admit the SUPERSEDED
#: ATTRIBUTED BYTES for THIS READ. The ruling turns on a computable fact --
#: this driver reads exactly TWO fields from a verdict, `day_closed_calendar`
#: and `write_reason`, and both are GENUINE in those bytes; the fields that
#: are stale there (`era_admissible`, `race_accrual_eligible`) are fields the
#: driver never reads. That is why it is admissible, and it would not be if
#: the driver read them.
#:
#: So the admission is SCOPED TO ONE DAY and carries who ruled, the blob it
#: admits, and the two field values it relies on -- and every one of those is
#: RE-VERIFIED AT THE BLOB AT RUN TIME. Nothing here is trusted from the
#: dispatch that granted it (rule 16). If either field is not what the ruling
#: says, the admission is not in force and the run REFUSES, the ruling
#: notwithstanding, because the ruling was granted on a condition.
USER_ADMISSIONS_BY_DAY = {
    "20260829": {
        "admitted_by": "USER",
        "relayed_by": "coordinator (dispatch of BE round 23)",
        "filed_at": "Q-BE-248",
        "depends_on": ("R-500 -- the USER WITHDREW 08-29 from the race and "
                       "kept it readable; this admission is for a READ of a "
                       "withdrawn day, never for a race day"),
        "blob_commit": "4e1133c",
        "blob_path": "data/pm_5min/derived/da_dayverdict_20260829.json",
        "blob_sha256": ("b808e603f3448a503d8ef72f8d1713bece20eda1d67a1e8f"
                        "03162a59ecc0709e"),
        "invocation_id": "142596744fc3492283df4f1ceb3be3b2",
        "as_of_utc": "2026-08-30T00:06:01.246972+00:00",
        "fields_relied_on": ("day_closed_calendar", "write_reason"),
        "why_the_stale_fields_do_not_matter": (
            "`era_admissible` and `race_accrual_eligible` are stale in these "
            "bytes and are read NOWHERE in this driver -- established by "
            "`driver_reads_no_era_field()` against this file's own source, "
            "not asserted. If a future edit makes the driver read one, that "
            "predicate goes False and the admission REFUSES rather than "
            "silently covering a field it was never granted for"),
        "scope": ("THIS READ, on 20260829 ONLY. It is not a widening of "
                  "`day_closed_and_attributed`, which is unchanged for every "
                  "other day, race days included"),
        "kind": "SUPERSEDED_ATTRIBUTED_BLOB",
    },
    #: R-503 + the USER's directive ("dont waste time in ruling, just mark it
    #: missing a window"). DA re-verdicted 09-03 under R-503 so the day
    #: accrues on its COVERED COMPLEMENT -- and the scheduled-unit prefix was
    #: LOST on the supersede, which DA flagged rather than papered over. So
    #: the gate-1 shape of 08-29 recurs exactly: the ordinary gate would admit
    #: the 00:06Z verdict that says the day does NOT accrue, and refuse the
    #: 09:36Z one that says it does.
    #:
    #: The condition here is STRONGER than 08-29's, because the risk is
    #: different. 08-29's question was "are these bytes genuinely attributed";
    #: this one's is "did the RULE change, or did the DAY change". So the
    #: admission verifies, at run time, that every raw measurement is
    #: IDENTICAL between the re-verdict and its predecessor and that only
    #: rule-derived fields moved. If the day's data moved, the admission
    #: REFUSES -- a re-verdict is allowed to reinterpret evidence, never to
    #: restate it.
    "20260903": {
        "admitted_by": "USER",
        "relayed_by": "coordinator (dispatch of BE round 29)",
        "filed_at": "Q-BE-255",
        "depends_on": ("R-503 -- a day accrues on its COVERED COMPLEMENT, "
                       "287 of 288 with the uncovered window marked and "
                       "counted, built on R-409's accounting"),
        "kind": "RE_VERDICT_UNDER_RULING",
        "verdict_path": "data/pm_5min/derived/da_dayverdict_20260903.json",
        "verdict_sha256": ("727a4dcd5e2d4c4ded866f5e4f1977a22146d280db6b"
                           "fdbe42e38732f84ac8a1"),
        "verdict_as_of": "2026-09-04T09:36:44.731072+00:00",
        "predecessor_path": ("data/pm_5min/derived/da_dayverdict_20260903."
                             "superseded_20260904T000601.304982+0000.json"),
        "predecessor_sha256": ("89a70af147c1226dd2bea969af7836737ab7f7fae9"
                               "7ea9895fc2bec29a040e0d"),
        "ruling_token": "R-503",
        #: Raw MEASUREMENTS of the day. These must not move.
        "data_fields_that_must_be_identical": (
            "windows_gap_affected", "tape_density"),
        "per_coin_measurements_that_must_be_identical": (
            "lost_seconds", "coin_level_gap_intervals"),
        #: The ONE difference allowed inside `gap_series`, with its reason.
        "gap_series_fields_allowed_to_differ": ("ledger_lines",),
        "why_ledger_lines_may_differ": (
            "it is the LINE COUNT of a live append-only gap ledger read at a "
            "later instant (11,545 at 00:06Z, 11,702 at 09:36Z), not a "
            "measurement of 09-03. Every derived gap quantity in the same "
            "block -- causes, gaps_per_hour, hours_over_bar -- is identical, "
            "and so are lost_seconds and coin_level_gap_intervals per coin"),
        "why_this_day": ("R-503 changed the RULE by which coverage is "
                         "judged; the day's data is unchanged and that is "
                         "checked here rather than inherited from the "
                         "dispatch"),
        # The prose here used to ASSERT that the ordinary gate still refuses
        # this verdict. That is a verdict about code behaviour that the
        # artifact did not compute, so it would have kept reading true after
        # any widening of the gate. It is now MEASURED at emission time in
        # `ordinary_gate_without_this_admission` and the claim is gone.
        "scope": "THIS DAY ONLY",
    },
}

#: The fields whose staleness the ruling tolerated, BECAUSE the driver does
#: not read them. That premise is checked, not trusted.
_ADMISSION_STALE_FIELDS = ("era_admissible", "race_accrual_eligible",
                           "era_admission", "day_quality_pass")


def driver_reads_no_era_field(src: str = None) -> dict:
    """The premise the USER's ruling rests on, computed from this source.

    The ruling holds because the driver reads only `day_closed_calendar` and
    `write_reason` from a verdict. If an edit ever makes it read a field that
    is STALE in the admitted bytes, the ruling's ground is gone -- so this is
    a predicate, evaluated at run time, not a sentence in a comment.

    It looks for READS -- `x["era_admissible"]`, `x.get("era_admissible")` --
    and not for mentions, because the admission machinery below necessarily
    NAMES these fields in order to record them. A predicate that counted its
    own record would be False by construction, which is a check that cannot
    pass rather than a check that failed."""
    import ast as _ast
    src = Path(__file__).read_text() if src is None else src
    tree = _ast.parse(src)
    exempt = {"driver_reads_no_era_field", "admitted_verdict", "selftest"}
    reads: dict = {}

    def _scan(node, fname):
        for n in _ast.walk(node):
            key = None
            if isinstance(n, _ast.Subscript) and isinstance(
                    n.slice, _ast.Constant) and isinstance(n.slice.value, str):
                key = n.slice.value
            elif (isinstance(n, _ast.Call)
                  and isinstance(n.func, _ast.Attribute)
                  and n.func.attr == "get" and n.args
                  and isinstance(n.args[0], _ast.Constant)
                  and isinstance(n.args[0].value, str)):
                key = n.args[0].value
            if key in _ADMISSION_STALE_FIELDS:
                reads.setdefault(key, []).append(fname)

    for n in tree.body:
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if n.name not in exempt:
                _scan(n, n.name)
        else:
            _scan(n, "<module>")
    return {"stale_fields_checked": list(_ADMISSION_STALE_FIELDS),
            "stale_fields_the_driver_reads": sorted(reads),
            "read_by": {k: sorted(set(v)) for k, v in reads.items()},
            "exempt_because_they_RECORD_rather_than_READ": sorted(exempt),
            "premise_holds": not reads}


def admission_bytes_ok(day: str, a: dict, raw: bytes) -> dict:
    """Every condition the ruling rests on, checked ON THE BYTES.

    Split out so each refusal can be DRIVEN with fabricated bytes rather than
    only provoked through the one real blob -- a condition that can only be
    tested by the input that satisfies it is a condition nobody has tested."""
    got = hashlib.sha256(raw).hexdigest()
    if got != a["blob_sha256"]:
        raise ForwardDayRefused(
            f"REFUSED: the admitted blob for {day} hashes {got[:16]}, not the "
            f"{a['blob_sha256'][:16]} the ruling names. These are not the "
            f"bytes the USER verified.")
    try:
        v = json.loads(raw)
    except ValueError as e:
        raise ForwardDayRefused(
            f"REFUSED: the admitted bytes for {day} are not parseable JSON "
            f"({type(e).__name__}). Their sha matched, so they ARE the "
            f"declared bytes -- they are simply not a verdict.") from None
    closed = v.get("day_closed_calendar")
    if closed is not True:
        raise ForwardDayRefused(
            f"REFUSED: the admitted bytes for {day} carry "
            f"day_closed_calendar={closed!r}, not True. The ruling relies on "
            f"that field being genuine; it is not what the ruling says.")
    wr = v.get("write_reason")
    if not (isinstance(wr, str) and wr.startswith(SCHEDULED_PREFIX)):
        raise ForwardDayRefused(
            f"REFUSED: the admitted bytes for {day} carry "
            f"write_reason={wr!r}, which does not start with the scheduled "
            f"prefix. The whole ground of the admission is that THESE bytes "
            f"are genuinely attributed; they are not.")
    if a["invocation_id"] not in wr:
        raise ForwardDayRefused(
            f"REFUSED: the admitted bytes for {day} do not carry the "
            f"INVOCATION_ID {a['invocation_id']} the USER verified. A "
            f"scheduled-unit prefix with a different invocation is a "
            f"different run.")
    return v


def reverdict_data_unchanged(a: dict) -> dict:
    """R-503 ADMISSION: did the RULE change, or did the DAY change?

    A re-verdict may reinterpret evidence. It may NOT restate it. So this
    compares the re-verdict against the predecessor it superseded and refuses
    unless every raw measurement is IDENTICAL -- with exactly one exemption,
    named in the admission with its reason, for a live ledger's line count.

    It also requires that something rule-derived DID move: a re-verdict that
    changes nothing is not the re-verdict the admission was granted for."""
    cur = REPO / a["verdict_path"]
    pre = REPO / a["predecessor_path"]
    for label, f, want in (("re-verdict", cur, a["verdict_sha256"]),
                           ("predecessor", pre, a["predecessor_sha256"])):
        if not f.exists():
            raise ForwardDayRefused(
                f"REFUSED: the {label} named by the 20260903 admission is not "
                f"at {f}. An admission whose artifact nobody can read is not "
                f"evidence.")
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            raise ForwardDayRefused(
                f"REFUSED: the {label} at {f} hashes {got[:16]}, not the "
                f"{want[:16]} the admission names. These are not the bytes "
                f"the USER's directive was given on.")
    new = json.loads(cur.read_text())
    old = json.loads(pre.read_text())

    moved = []
    for k in a["data_fields_that_must_be_identical"]:
        if json.dumps(new.get(k), sort_keys=True) != json.dumps(
                old.get(k), sort_keys=True):
            moved.append(k)
    for coin in sorted(set(new.get("per_coin") or {})
                       & set(old.get("per_coin") or {})):
        nb = (new["per_coin"][coin].get("day_bar_v2") or {})
        ob = (old["per_coin"][coin].get("day_bar_v2") or {})
        for f in a["per_coin_measurements_that_must_be_identical"]:
            if nb.get(f) != ob.get(f):
                moved.append(f"per_coin.{coin}.day_bar_v2.{f}")
    ng, og = new.get("gap_series") or {}, old.get("gap_series") or {}
    allowed = set(a["gap_series_fields_allowed_to_differ"])
    for f in sorted(set(ng) | set(og)):
        if f in allowed:
            continue
        if json.dumps(ng.get(f), sort_keys=True) != json.dumps(
                og.get(f), sort_keys=True):
            moved.append(f"gap_series.{f}")
    if moved:
        raise ForwardDayRefused(
            f"REFUSED: the 20260903 re-verdict differs from its predecessor "
            f"in MEASUREMENTS, not only in rule: {sorted(set(moved))}. The "
            f"admission was granted for a RULE change under "
            f"{a['ruling_token']}; a re-verdict that restates the day's data "
            f"is a different act and is not admitted.")

    # And it must actually have DONE something.
    rule_moved = {}
    for coin in sorted(set(new.get("per_coin") or {})):
        nb = (new["per_coin"][coin].get("day_bar_v2") or {})
        ob = ((old.get("per_coin") or {}).get(coin, {}).get("day_bar_v2") or {})
        if nb.get("evaluable") != ob.get("evaluable"):
            rule_moved[coin] = {"evaluable": [ob.get("evaluable"),
                                              nb.get("evaluable")]}
    if not rule_moved:
        raise ForwardDayRefused(
            f"REFUSED: the 20260903 re-verdict moved NO rule-derived field. "
            f"An admission for a rule change that changed nothing is an "
            f"admission for nothing.")
    return {"data_identical": True,
            "measurements_compared": sorted(
                list(a["data_fields_that_must_be_identical"])
                + [f"per_coin.*.day_bar_v2.{f}" for f in
                   a["per_coin_measurements_that_must_be_identical"]]
                + ["gap_series.* (except "
                   + ",".join(a["gap_series_fields_allowed_to_differ"]) + ")"]),
            "exempted_with_reason": {
                f: a["why_ledger_lines_may_differ"]
                for f in a["gap_series_fields_allowed_to_differ"]},
            "rule_derived_fields_that_moved": rule_moved,
            "predecessor": a["predecessor_path"],
            "predecessor_sha256": a["predecessor_sha256"]}


def _ordinary_gate_outcome(day: str) -> dict:
    """What `assert_day_closed_and_attributed` does with NO admission, run.

    Not asserted in prose: called. An artifact that SAYS the ordinary gate
    still refuses would go on saying it after the gate stopped refusing."""
    try:
        assert_day_closed_and_attributed(day)
        return {"outcome": "ADMITTED",
                "measured_at_emission": True,
                "meaning": ("the ordinary gate accepts this day WITHOUT the "
                            "admission, so the admission is not what makes it "
                            "scoreable")}
    except ForwardDayRefused as e:
        return {"outcome": "REFUSED", "measured_at_emission": True,
                "refusal": str(e)[:180],
                "meaning": ("the ordinary gate refuses this day without the "
                            "admission, so the admission is load-bearing and "
                            "is not decoration")}


def admitted_verdict(day: str) -> dict | None:
    """Recover and VERIFY the admitted bytes at the blob, now.

    Returns None when no admission covers the day -- which is every day but
    one, and is why this cannot become a general bypass. Raises when an
    admission exists but its condition fails: a granted ruling whose
    condition no longer holds is a refusal, not a waiver."""
    a = USER_ADMISSIONS_BY_DAY.get(day)
    if a is None:
        return None
    prem = driver_reads_no_era_field()
    if not prem["premise_holds"]:
        raise ForwardDayRefused(
            f"REFUSED: the {day} admission was granted because this driver "
            f"reads no field that is stale in the admitted bytes, and it now "
            f"reads {prem['stale_fields_the_driver_reads']}. The ruling's "
            f"ground is gone, so the admission is not in force.")
    if a.get("kind") == "RE_VERDICT_UNDER_RULING":
        ev = reverdict_data_unchanged(a)
        v = json.loads((REPO / a["verdict_path"]).read_text())
        closed, wr = v.get("day_closed_calendar"), v.get("write_reason")
        if closed is not True:
            raise ForwardDayRefused(
                f"REFUSED: the {day} re-verdict carries "
                f"day_closed_calendar={closed!r}, not True.")
        if v.get("race_accrual_eligible") is not True:
            raise ForwardDayRefused(
                f"REFUSED: the {day} re-verdict does not accrue "
                f"(race_accrual_eligible={v.get('race_accrual_eligible')!r}). "
                f"The admission was granted for a day that DOES.")
        if not (isinstance(wr, str) and a["ruling_token"] in wr):
            raise ForwardDayRefused(
                f"REFUSED: the {day} re-verdict's write_reason does not name "
                f"{a['ruling_token']}, so it is not the re-verdict the "
                f"admission was granted for. Got {wr!r}.")
        return {"verdict": v, "record": {
            "ADMISSION": "RE-VERDICT UNDER A USER RULING, ADMITTED BY THE USER",
            "admitted_by": a["admitted_by"], "relayed_by": a["relayed_by"],
            "filed_at": a["filed_at"], "depends_on": a["depends_on"],
            "scope": a["scope"], "kind": a["kind"],
            # rule 14, THIRD time this guard has caught this record's
            # NAMING and third time it was right: `verdict` reads as
            # decision-shaped. The path is a path; say so.
            "artifact_path": a["verdict_path"],
            "artifact_sha256": a["verdict_sha256"],
            "artifact_as_of": a["verdict_as_of"],
            "verified_at_run_time": True,
            "fields_relied_on": {"day_closed_calendar": closed,
                                 "write_reason": wr},
            "the_lost_prefix": (
                "the scheduled-unit prefix was LOST on DA's supersede and DA "
                "flagged it rather than papering over it; the ordinary gate "
                "would therefore admit the 00:06Z verdict that says the day "
                "does NOT accrue and refuse the 09:36Z one that says it does"),
            "rule_change_not_data_change": ev,
            "premise_checked_not_trusted": driver_reads_no_era_field(),
            # RULE 10, and the shape that put three of today's errors in front
            # of the USER: do not write a verdict beside a computed field.
            # This RUNS the ordinary gate without the admission and records
            # what it actually did, so the artifact cannot keep claiming a
            # refusal that a later edit removed.
            "ordinary_gate_without_this_admission":
                _ordinary_gate_outcome(day),
        }}

    raw = _git_blob(a["blob_commit"], a["blob_path"])
    if not raw:
        raise ForwardDayRefused(
            f"REFUSED: the {day} admission names blob "
            f"{a['blob_commit']}:{a['blob_path']}, which this repo cannot "
            f"produce. An admission whose bytes nobody can recover is not "
            f"evidence.")
    v = admission_bytes_ok(day, a, raw)
    closed, wr = v["day_closed_calendar"], v["write_reason"]
    return {
        "verdict": v,
        "record": {
            "ADMISSION": "SUPERSEDED-BUT-GENUINE BYTES, ADMITTED BY THE USER",
            "admitted_by": a["admitted_by"], "relayed_by": a["relayed_by"],
            "filed_at": a["filed_at"], "depends_on": a["depends_on"],
            "scope": a["scope"],
            "blob": f"{a['blob_commit']}:{a['blob_path']}",
            "blob_sha256": a["blob_sha256"],
            "verified_at_run_time": True,
            "fields_relied_on": {"day_closed_calendar": closed,
                                 "write_reason": wr},
            "invocation_id": a["invocation_id"],
            # rule 14: the NAMES, never the values. Echoing
            # `race_accrual_eligible` into a receipt would put an
            # ENTITLEMENT in a worker's output -- `assert_no_decision_field`
            # refused exactly that, and it was right to.
            "stale_fields_PRESENT_in_these_bytes_and_read_by_nothing": sorted(
                f for f in _ADMISSION_STALE_FIELDS if f in v),
            "stale_field_values_deliberately_not_echoed": (
                "rule 14 -- these are entitlement fields; this driver "
                "supplies counts and refusals, and reproducing their values "
                "in a receipt would encode a decision it does not own"),
            "why_the_stale_fields_do_not_matter":
                a["why_the_stale_fields_do_not_matter"],
            "premise_checked_not_trusted": prem,
            "the_superseding_verdict_is_not_repudiated": (
                "the current verdict at that path remains the correct one on "
                "era; it is simply unattributable, and this read needs "
                "attribution rather than era"),
        },
    }


def assert_day_closed_and_attributed(day: str, verdict: dict = None,
                                     admission: dict = None) -> dict:
    v = FS.read_day_verdict(day) if verdict is None else verdict
    if not v:
        raise ForwardDayRefused(
            f"REFUSED: {day} has no day verdict at "
            f"{DERIVED / f'da_dayverdict_{day}.json'}. A forward day is scored "
            f"only after DA has verified it (R-153(3)); absence is not a pass.")
    closed = v.get("day_closed_calendar")
    if closed is not True:
        raise ForwardDayRefused(
            f"REFUSED: {day} is not closed by calendar "
            f"(day_closed_calendar={closed!r}). Scoring an OPEN day scores a "
            f"population that is still growing.")
    wr = v.get("write_reason")
    if isinstance(wr, str) and wr.startswith(SCHEDULED_PREFIX):
        return {"day_closed_calendar": True, "write_reason": wr,
                "attribution": "SCHEDULED UNIT",
                "write_reason_prefix_source": "da_governed_verdict_preflight."
                                              "SCHEDULED_PREFIX"}
    # ROUND 29: THE ONE OTHER WAY A DAY MAY BE ATTRIBUTED, AND IT IS NOT A
    # WEAKENING BECAUSE IT CANNOT BE REACHED BY OMISSION.
    #
    # DA's R-503 re-verdict LOST the scheduled prefix on the supersede. The
    # prefix check is therefore satisfied by the verdict that says the day
    # does NOT accrue and refused by the one that says it does -- the 08-29
    # shape again. The USER admitted it rather than re-ruling.
    #
    # `admission` DEFAULTS TO None, so every caller that does not name one
    # gets EXACTLY the old behaviour: the refusal below. It is not a flag on
    # a path that would otherwise pass; it is a second, narrower door that a
    # caller must hold a VERIFIED admission record to open.
    if admission is not None:
        if admission.get("kind") != "RE_VERDICT_UNDER_RULING":
            raise ForwardDayRefused(
                f"REFUSED: {day} was given an admission of kind "
                f"{admission.get('kind')!r}, which does not attest "
                f"attribution. Only a verified RE_VERDICT_UNDER_RULING "
                f"record may stand in for the scheduled-unit prefix.")
        if admission.get("verified_at_run_time") is not True:
            raise ForwardDayRefused(
                f"REFUSED: {day}'s admission record was not verified at run "
                f"time. An unverified attestation is a claim, not evidence.")
        if str(admission.get("artifact_sha256")) != hashlib.sha256(
                (REPO / admission["artifact_path"]).read_bytes()).hexdigest():
            raise ForwardDayRefused(
                f"REFUSED: {day}'s admission attests bytes that are not the "
                f"bytes now on disk at {admission['artifact_path']}.")
        return {"day_closed_calendar": True, "write_reason": wr,
                "attribution": "USER ADMISSION, NOT THE SCHEDULED UNIT",
                "why_the_prefix_is_absent": (
                    "DA's supersede under R-503 lost the scheduled-unit "
                    "prefix and flagged it; the prefix check would otherwise "
                    "admit the superseded verdict that says this day does not "
                    "accrue and refuse the re-verdict that says it does"),
                "admission": {k: admission.get(k) for k in
                              ("admitted_by", "filed_at", "depends_on",
                               "artifact_path", "artifact_sha256",
                               "artifact_as_of")},
                "write_reason_prefix_source": "NOT USED -- see `attribution`"}
    raise ForwardDayRefused(
        f"REFUSED: {day}'s verdict was not written by the scheduled unit "
        f"— write_reason={wr!r} does not start with the required prefix "
        f"(imported from da_governed_verdict_preflight, matched as a "
        f"PREFIX exactly as DA's preflight matches it; a substring test "
        f"would accept an unattributed hand run).")


# ---------------------------------------------------------------------------
# gate 3 -- the population: the day's own ledger, then supply, then bridge
# ---------------------------------------------------------------------------
def present_from_ledger(day: str, path: Path = None) -> dict:
    """The windows that EXISTED, read from the day's own market ledger.

    A FACT, NOT A GRID (R-418). `de_admissible_windows` refuses to derive this
    itself — deriving the calendar is selecting, and it supplies."""
    p = path or MARKETS
    if not p.exists():
        raise ForwardDayRefused(f"REFUSED: no market ledger at {p}.")
    lo, hi = day_bounds(day)
    per: dict[str, set] = collections.defaultdict(set)
    rows = 0
    with p.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            rows += 1
            ws = d.get("window_start")
            coin = d.get("coin")
            if ws is None or coin is None or not (lo <= int(ws) < hi):
                continue
            per[coin].add(int(ws))
    if not per:
        raise ForwardDayRefused(
            f"REFUSED: the ledger holds no window for {day} (read {rows} "
            f"rows). An empty present is not an empty day — it is a read that "
            f"found nothing, and scoring off it would score nothing and say "
            f"it succeeded.")
    return {c: sorted(v) for c, v in sorted(per.items())}


def archive_windows_for_day(day: str) -> dict:
    """The windows the TAPE/ARCHIVE index holds for `day`, per coin.

    A SECOND, INDEPENDENT reading of the same day. The ledger says which
    windows EXISTED; the archive index says which are REPLAYABLE. Read from
    the tree's `flow_intensity` because the frozen anchors are not imported
    until later in the sequence — and the receipt says so."""
    import flow_intensity as fi
    lo, hi = day_bounds(day)
    per: dict[str, set] = collections.defaultdict(set)
    for slug in fi._archive_paths():
        try:
            t0 = int(slug.rsplit("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if lo <= t0 < hi:
            per[slug.split("-")[0]].add(t0)
    return {c: sorted(v) for c, v in sorted(per.items())}


def assert_ledger_matches_archive(day: str, present: dict,
                                  archive: dict = None) -> dict:
    """The ledger and the archive index must agree, per coin. R-424-adjacent.

    An EMPTY index already refuses at the data-root probe; this is the
    NON-EMPTY-BUT-WRONG case, which that probe cannot see. The population is
    the ledger's, so a coin whose archive holds a different set is either a
    window that existed and cannot be replayed, or one replayable that the
    ledger never recorded — both are disagreements about what the day WAS,
    and R-418 scores the complement WHOLE, so neither may be resolved by
    quietly taking an intersection."""
    arc = archive_windows_for_day(day) if archive is None else archive
    rows, bad = {}, []
    for coin in sorted(set(present) | set(arc)):
        lw = set(present.get(coin) or ())
        aw = set(arc.get(coin) or ())
        rows[coin] = {"n_ledger": len(lw), "n_archive": len(aw),
                      "n_ledger_not_archive": len(lw - aw),
                      "n_archive_not_ledger": len(aw - lw)}
        if lw != aw:
            bad.append(
                f"{coin}: ledger {len(lw)} vs archive {len(aw)} "
                f"(ledger-only {len(lw - aw)}, archive-only {len(aw - lw)}; "
                f"e.g. {sorted(lw ^ aw)[:3]})")
    if bad:
        raise ForwardDayRefused(
            f"REFUSED: the market ledger and the archive index disagree about "
            f"{day} — {'; '.join(bad)}. The ledger is the population's "
            f"source and the archive is what can be replayed; a silent "
            f"intersection would re-select the population R-418 fixes, and a "
            f"silent union would score a window that was never recorded.")
    return {"per_coin": rows, "coins": sorted(rows),
            "archive_index_source": "flow_intensity (tree copy; the frozen "
                                    "anchors are imported later in the "
                                    "sequence)",
            "agree": True}


def population(day: str, present: dict = None) -> dict:
    pres = present_from_ledger(day) if present is None else present
    # The ledger is the population's SOURCE; the archive index is what can be
    # replayed. They must agree before either is used.
    ledger_vs_archive = assert_ledger_matches_archive(day, pres)
    supply = AW.supply(day, pres)                    # refusals propagate
    # CO-5 / R-421 §4: `verified` ALONE reads absence as a pass. The PAIR is
    # required -- verified AND nothing unverifiable -- because a field the
    # checker could not bind is not a field that passed.
    _sel = ratification_for(day)
    rat = RAT.check(supply, _sel["ref"])
    # The assertion's RESULT is recorded, so removing the call removes the
    # evidence: on a healthy day a bypassed pair-check changes no answer, and
    # a guard nothing can observe is a guard nothing protects.
    # THE CHECKER'S OWN FAIL-CLOSED GATE FIRST. `require_verified` is the
    # CONSUMER contract (verified AND nothing unverifiable AND not a
    # provenance result); asserting the pair myself left the third conjunct
    # unchecked and would let a FUTURE checker refusal be bypassed by a
    # reader of one field. The local pair assertion stays as the EVIDENCE
    # recorded in the receipt, not as the gate.
    # MEASURED (round-5 audit, mutant H4): a `require_verified_called: True`
    # flag written BESIDE the call survives deleting the call -- the same
    # forgeable-evidence defect I named in `pair_asserted` last round, made
    # again here. The evidence is now the checker's RETURN VALUE, so removing
    # the call leaves `_rv` unbound and the driver cannot run at all.
    _rv = _sel["gate"](rat)          # RAISES on a refusal; see below
    rat["pair_asserted"] = assert_ratification_pair(rat)
    # ONE place, and it is HERE. A second reading in `run_forward_day` could
    # not be reached by any affordable suite (gate 2 is 60 s in, the replay is
    # 26 min), so a mutant deleting it survived while the suite stayed green.
    # A guard nothing can drive is a guard nothing protects.
    rat["require_verified"] = {
        "checker": _sel["gate_name"],
        "ratification_ref_used": _sel["ref"],
        "ratification_kind": _sel["kind"],
        "why_this_gate": _sel["why"],
        "race_admissible_reported_by_the_checker": _rv.get("race_admissible"),
        "scope_days_reported_by_the_checker": _rv.get("scope_days"),
        "verified": bool(_rv.get("verified")),
        "unverifiable": list(_rv.get("unverifiable") or ()),
        "provenance_absent": not _rv.get("provenance"),
        "checks_seen": len(_rv.get("checks") or ())}
    # NO second gate here. `require_verified` checks all three conjuncts and
    # RAISES; a local re-check of the same three could only fire on something
    # the checker let through, which is nothing -- and the audit proved it,
    # by disabling it with no test able to notice (H5b).
    specs = SEAM.window_specs_from_supply(
        supply, ratification_ref=_sel["ref"])        # refusals propagate
    # MEASURED (mutant H6): stubbing this call with a literal `{"agree":
    # True}` survived, because the controls exercised the function and nobody
    # consumed its result. The record must reproduce the ledger it checked.
    # Driven by the suite with a stubbed checker: a guard that only fires
    # when another function is replaced must be shown to fire.
    _lva = ledger_vs_archive.get("per_coin") or {}
    _mismatch = sorted(c for c in pres
                       if (_lva.get(c) or {}).get("n_ledger")
                       != len(pres.get(c) or ()))
    if not _lva or _mismatch:
        raise ForwardDayRefused(
            f"REFUSED: the ledger/archive record does not reproduce the "
            f"ledger it claims to have checked (coins disagreeing or absent: "
            f"{_mismatch or 'the per-coin block is empty'}). A record that "
            f"cannot be tied back to the population is not evidence that the "
            f"population was checked.")
    return {"present": pres, "supply": supply, "specs": specs,
            "ratification": rat, "ledger_vs_archive": ledger_vs_archive,
            "r411_inputs": r411_inputs(supply)}


def assert_ratification_pair(rat: dict) -> bool:
    """verified AND unverifiable == []. CO-5.

    Explicit argument so the PREDICATE is drivable: read only through
    `population()`'s output, a control cannot tell a real check from a
    literal dict, and both a weakened predicate and a skipped call survived."""
    if not rat.get("verified") or rat.get("unverifiable"):
        raise ForwardDayRefused(
            f"REFUSED: ratification {RATIFICATION_REF} did not verify as a "
            f"PAIR — verified={rat.get('verified')!r}, "
            f"unverifiable={rat.get('unverifiable')!r}. `verified` alone "
            f"reads an unbindable field as a pass (CO-5); a field the checker "
            f"could not bind has not been ratified.")
    # RETURN WHAT WAS ASSERTED, not THAT it was. A boolean flag can be forged
    # by the same edit that bypasses the call -- measured: a mutant setting
    # `pair_asserted = True` survived an evidence check that only asked
    # whether the field was PRESENT.
    return {"asserted": True, "verified_seen": rat.get("verified"),
            "unverifiable_seen": list(rat.get("unverifiable") or []),
            "ref_seen": rat.get("ratification_ref")}


def r411_inputs(supply: dict) -> dict:
    """The INPUTS R-411 needs, per coin. ESTIMATES, never a decision.

    R-411(i) sets the G floor at >=144/288 unmasked; R-411(ii) makes P1 "per
    UNMASKED hour". Both are POLICY arithmetic, so this emits what they are
    computed FROM — unmasked windows per UTC hour, the hours that carry any,
    and the totals — and NOTHING that reads as a verdict. `counts_toward_G`
    is deliberately absent and a post-condition refuses it (rule 14)."""
    import da_blackout_mask as _DAM
    out = {}
    for coin, recs in sorted((supply.get("windows") or {}).items()):
        starts = sorted(int(r["start"]) for r in recs)
        by_hour: dict = collections.defaultdict(int)
        for t in starts:
            by_hour[dt.datetime.fromtimestamp(
                t, dt.timezone.utc).strftime("%H")] += 1
        cnt = (supply.get("counts") or {}).get(coin) or {}
        out[coin] = {
            "unmasked_windows_per_utc_hour": dict(sorted(by_hour.items())),
            "n_hours_with_any_unmasked_window": len(by_hour),
            "n_unmasked_windows": len(starts),
            "n_present_windows": cnt.get("n_present"),
            "n_masked_windows": cnt.get("n_masked_applied"),
            # NOT a literal: R-411(i)'s denominator is DA's, so it is BOUND to
            # DA's committed constant. A restated 288 could drift away from the
            # producer's silently, and the emission would disagree with the
            # artifact it is supposed to be read beside (RR3-1's discipline).
            "calendar_windows_per_day": _DAM.WINDOWS_PER_DAY,
            "calendar_windows_per_day_source":
                "da_blackout_mask.WINDOWS_PER_DAY"}
    return {"per_coin": out,
            "for": {"R-411(i)": "the G floor is >=144/288 unmasked windows; "
                                "the numerator and denominator are above",
                    "R-411(ii)": "P1 is per UNMASKED hour; the hours carrying "
                                 "any unmasked window are above"},
            "these_are_ESTIMATES": "inputs only. No field here says whether "
                                   "the day counts, is eligible or is "
                                   "admissible — that is the policy layer's "
                                   "act (rule 14), and a post-condition "
                                   "refuses any decision-shaped key."}


def counts_per_coin(pop: dict) -> dict:
    out = {}
    for coin, c in sorted((pop["supply"]["counts"] or {}).items()):
        out[coin] = {"n_present": c["n_present"],
                     "n_masked": c["n_masked_applied"],
                     "n_supplied": c["n_supplied"]}
    return out


# ---------------------------------------------------------------------------
# gate 4 -- rows, from the ACCEPTED v3 builder, over EXACTLY the bridged windows
# ---------------------------------------------------------------------------
def selected_from_specs(specs: list) -> tuple:
    """The builder's own selection tuple, for the bridged windows only.

    `harmful_exposure_rows.select_stratified` builds
    `(slug, path, up, down, gaps)` from `fi._archive_paths()`,
    `fi.token_map()` and `fi.gaps_by_slug(era)`. R-418 forbids that selector
    on a race day, so the SAME three lookups are used here over the windows
    the supply emitted. Nothing is chosen: the set is the bridge's, in its
    order. A window with no archive or no token map is REFUSED, not skipped —
    a race day is scored whole or not at all."""
    import harmful_exposure_rows as HER
    fi = HER.qr.base.fi
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    out, missing = [], []
    for spec in specs:
        slug = spec["slug"]
        if slug not in paths or slug not in tokens:
            missing.append(slug)
            continue
        up, down = tokens[slug]
        out.append((slug, paths[slug], up, down, gaps.get(slug, [])))
    if missing:
        raise ForwardDayRefused(
            f"REFUSED: {len(missing)} supplied windows have no archive or no "
            f"token map ({missing[:4]}). R-418 scores the complement WHOLE; "
            f"dropping windows here would silently re-select the population "
            f"the supply already fixed.")
    return out, {"n_specs": len(specs), "n_selected": len(out)}


def build_rows_over(selected: list) -> dict:
    """The v3 builder's per-window sequence, over the supplied windows.

    NOT A SECOND BUILDER. Every step is `harmful_exposure_rows`' own function
    in its own order — replay, join, boundary, clock, generation table,
    labels — and the STRICT failure condition is the module's, copied by
    reference rather than restated. `build_rows` itself cannot be reused
    because its window set comes from `select_stratified`/`select_v2_era` and
    it takes no injection point; `harmful_exposure_rows.py` is a
    reproducibility anchor of the frozen candidate, so this driver does not
    edit it to add one.

    THE RECONCILIATION IS THE GATE. A mismatch marks the rows and is reported
    as a FAILURE for the day; it is never absorbed."""
    import harmful_exposure_rows as HER
    qr = HER.qr
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    rows, per_window = [], {}
    recon_fail = unhooked = wrong_gen = boundary_bad = clock_bad = 0
    n_windows = 0
    for slug, path, up, down, wgaps in selected:
        out = HER.replay_with_recorder(path, up, down, wgaps, spec)
        if out is None:
            per_window[slug] = {"replayed": False, "n_rows": 0}
            continue
        arm, wf = out
        n_windows += 1
        t0 = int(slug.rsplit("-", 1)[1])
        day_s = dt.datetime.fromtimestamp(
            t0, dt.timezone.utc).strftime("%Y-%m-%d")
        joined, jrec = HER.join_fills(arm.fill_log, arm.fills)
        n_b = HER.verify_boundary_times(arm.segments, joined)
        ttimes = HER.trade_receipt_times(path, up, down)
        n_c = HER.verify_consume_clock(arm.consume_times, ttimes)
        gens, recon = HER.generation_table(arm.segments, joined, wf,
                                           qr.base.fi.WINDOW_S)
        wrows = HER.label_rows(arm.segments, gens, wf, qr.base.fi.WINDOW_S)
        bad = (jrec["count_mismatch"] or jrec["tuple_mismatches"]
               or recon["orphan_fills"]
               or recon["wrong_generation_assignments"]
               or arm.unhooked_changes or n_b or n_c)
        wrong_gen += recon["wrong_generation_assignments"]
        boundary_bad += n_b
        clock_bad += n_c
        if bad:
            recon_fail += 1
            unhooked += arm.unhooked_changes
            for r in wrows:
                r["status"] = "RECONCILIATION_FAILED"
        for r in wrows:
            r["slug"] = slug
            r["coin"] = slug.split("-")[0]
            r["day"] = day_s
            r["t0"] = t0
        rows.extend(wrows)
        per_window[slug] = {"replayed": True, "n_rows": len(wrows),
                            "reconciled": not bool(bad)}
    return {"rows": rows, "n_windows": n_windows,
            "reconciliation_failures": recon_fail,
            "unhooked_state_changes": unhooked,
            "wrong_generation_assignments": wrong_gen,
            "boundary_time_violations": boundary_bad,
            "consume_clock_violations": clock_bad,
            "per_window": per_window,
            "schema": "harmful_exposure_v3_4_fill_scoped_markout"}


def assert_window_sets_agree(specs: list, row_windows) -> dict:
    """The rows must cover EXACTLY the bridged windows. Neither side may
    silently be the other's subset."""
    bridged = {s["slug"] for s in specs}
    got = (set(row_windows) if not isinstance(row_windows, dict)
           else {r["slug"] for r in row_windows["rows"]})
    only_bridged = sorted(bridged - got)
    only_rows = sorted(got - bridged)
    if only_rows:
        raise ForwardDayRefused(
            f"REFUSED: rows carry {len(only_rows)} windows the bridge never "
            f"supplied ({only_rows[:4]}). A row outside the ratified "
            f"population is a window nobody admitted (R-418).")
    return {"n_bridged": len(bridged), "n_with_rows": len(got),
            "bridged_without_rows": len(only_bridged),
            "bridged_without_rows_examples": only_bridged[:8],
            "note": "a bridged window with no rows is a window that produced "
                    "no cancellable generation; it is COUNTED here, never "
                    "dropped from the denominator"}


def action_count(rows: list) -> int:
    """Rule 2: rows are actions; the evaluator de-duplicates to actions."""
    return len({(r.get("slug"), r.get("side"), r.get("gen")) for r in rows})


def build_and_score(selected: list, frozen: dict, feed=None,
                    inc_fits: dict = None) -> dict:
    """ONE STREAMING PASS: replay a window, label it, score it, DROP it.

    MEASURED, and this is the whole reason the shape changed: holding every
    window's rows AND its `window_streams` for all 1,875 supplied windows
    OOM-killed the run at exactly the 12 G cap after 21 minutes of CPU. The
    cap is not raised (R-174) — the run stops CARRYING what it does not
    need. Rows exist only while their window is being scored; what survives
    a window is its counters, its action keys and its per-action scores.

    Same functions, same order, same STRICT failure condition as the v3
    builder — the loop is streamed, not reimplemented."""
    import harmful_exposure_rows as HER
    import harmful_hazard_model as hm
    # ROUND 25: the second arm, loaded ONCE and verified against the sha the
    # read declaration froze before the read. INJECTABLE exactly as `frozen`
    # is, because the suite stubs the scorer with a fit shape of its own --
    # and a production default that a test cannot replace is a seam the
    # suite has to route around instead of exercising.
    inc_fits = load_incumbent_fits() if inc_fits is None else inc_fits
    qr = HER.qr
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    paths = hm.fi._archive_paths()
    tokens = hm.fi.token_map()
    scores: dict = collections.defaultdict(list)
    actions: set = set()
    n_rows = n_windows = 0
    recon_fail = unhooked = wrong_gen = boundary_bad = clock_bad = 0
    no_features = 0
    recon_bad: list = []
    windows_with_rows: set = set()
    for slug, path, up, down, wgaps in selected:
        out = HER.replay_with_recorder(path, up, down, wgaps, spec)
        if out is None:
            continue
        arm, wf = out
        n_windows += 1
        t0 = int(slug.rsplit("-", 1)[1])
        joined, jrec = HER.join_fills(arm.fill_log, arm.fills)
        n_b = HER.verify_boundary_times(arm.segments, joined)
        n_c = HER.verify_consume_clock(
            arm.consume_times, HER.trade_receipt_times(path, up, down))
        gens, recon = HER.generation_table(arm.segments, joined, wf,
                                           qr.base.fi.WINDOW_S)
        wrows = HER.label_rows(arm.segments, gens, wf, qr.base.fi.WINDOW_S)
        bad = (jrec["count_mismatch"] or jrec["tuple_mismatches"]
               or recon["orphan_fills"]
               or recon["wrong_generation_assignments"]
               or arm.unhooked_changes or n_b or n_c)
        wrong_gen += recon["wrong_generation_assignments"]
        boundary_bad += n_b
        clock_bad += n_c
        if bad:
            recon_fail += 1
            recon_bad.append(slug)
            unhooked += arm.unhooked_changes
        coin = slug.split("-")[0]
        fit = frozen["fits"].get(coin)
        if wrows:
            windows_with_rows.add(slug)
        n_rows += len(wrows)
        if fit is not None and not bad:
            stream = hm.window_streams(paths[slug], *tokens[slug])
            _feed_rows, _feed_scores = [], []
            for r in wrows:
                actions.add((slug, r.get("side"), r.get("gen")))
                fp = hm.features(stream, r["t_start"], r["side"],
                                 r.get("level"), r.get("resting"),
                                 r.get("qahead"))
                ff = hm.fine_feats(t0 + r["t_start"], r["side"], coin)
                if fp is None or ff is None:
                    no_features += 1
                    continue
                _vec = fp + ff
                _s = FS.expected_cancel_value(fit, _vec)
                # THE SECOND ARM, on the SAME vector, in the SAME pass. The
                # candidate's score above is untouched by this line -- it is
                # computed first and from its own fit.
                _si = (FS.expected_cancel_value(inc_fits[coin], _vec)
                       if coin in inc_fits else None)
                scores[coin].append((t0, _s))
                if feed is not None:
                    # `t0` is added to the row so the feed carries an ABSOLUTE
                    # time; `t_start` alone is relative to the window and the
                    # estimand's hour key needs both (harmful_action_eval:35).
                    _feed_rows.append(dict(r, slug=slug, t0=t0))
                    _feed_scores.append((_s, _si))
            if feed is not None and _feed_rows:
                feed.write_window(_feed_rows, _feed_scores)
            del _feed_rows, _feed_scores
            del stream                                # DROP the window
        else:
            for r in wrows:
                actions.add((slug, r.get("side"), r.get("gen")))
        del arm, wf, joined, gens, wrows              # DROP the rows
    return {"scores": dict(scores),
            "n_windows": n_windows, "n_rows": n_rows,
            "n_actions": len(actions),
            "n_windows_with_rows": len(windows_with_rows),
            "windows_with_rows": windows_with_rows,
            "rows_without_features": no_features,
            "reconciliation_failures": recon_fail,
            "reconciliation_failed_windows": recon_bad,
            "unhooked_state_changes": unhooked,
            "wrong_generation_assignments": wrong_gen,
            "boundary_time_violations": boundary_bad,
            "consume_clock_violations": clock_bad,
            "schema": "harmful_exposure_v3_4_fill_scoped_markout"}


def score_rows(rows: list) -> dict:
    """Per-action expected cancel value, through the FROZEN artifact's OWN
    feature_vector_contract.

    The frozen fit is APPLIED, never refitted: `harmful_forward_scorer`
    owns `design_row`/`expected_cancel_value` and this passes each row's
    PM+fine vector to them. Features come from `harmful_hazard_model`, the
    builder the candidate names — the same two calls `phase2_arms` makes,
    in the same order, so there is one feature construction and not a
    second."""
    import harmful_hazard_model as hm
    # BEM-R3: bound to the DECLARED identity, never to whatever the
    # module constant happens to name.
    frozen = FS.load_frozen(expect=FS.declared_candidate_identity())
    fi = hm.fi
    paths = fi._archive_paths()
    tokens = fi.token_map()
    streams: dict = {}
    out: dict = collections.defaultdict(list)
    skipped = 0
    for r in rows:
        coin, slug = r["coin"], r["slug"]
        fit = frozen["fits"].get(coin)
        if fit is None:
            skipped += 1
            continue
        if slug not in streams:
            if slug not in paths or slug not in tokens:
                raise ForwardDayRefused(
                    f"REFUSED: no archive for {slug} at scoring time.")
            up, dn = tokens[slug]
            streams[slug] = hm.window_streams(paths[slug], up, dn)
        fp = hm.features(streams[slug], r["t_start"], r["side"],
                         r.get("level"), r.get("resting"), r.get("qahead"))
        ff = hm.fine_feats(r["t0"] + r["t_start"], r["side"], coin)
        if fp is None or ff is None:
            skipped += 1
            continue
        out[coin].append((int(r["t0"]), FS.expected_cancel_value(fit, fp + ff)))
    if not out:
        raise ForwardDayRefused(
            f"REFUSED: zero actions scored across {len(rows)} rows "
            f"({skipped} lacked features or a fit). A forward report with no "
            f"scores is a FAILURE, not an empty result (R-141).")
    return dict(out)


class FeedWriter:
    """THE ACTION-KEYED FEED, WRITTEN AS THE DAY IS SCORED.

    The gap rounds 13-16 kept naming in its last remaining form: a scored
    forward day emitted `(window_start, value)` pairs and nothing the
    action-level estimand could consume. This writes the estimand's OWN input
    -- `slug`, `side`, `gen`, `t0`, `t_start`, `any_fill_ahead` and the
    latency-resolved value -- one JSONL line per row, AS each window is
    scored, so the 12 G cap is never approached: the rows are still dropped,
    only their six scalars survive to disk.

    SEALED LIKE THE SCORES (rule 11). It lands in the caller's outdir beside
    the sealed file, never under `data/pm_5min/derived/`, and the receipt
    records its path, sha256, byte count and row count and NO value from it.
    A feed is per-row model output; reading it is the coordinator's or the
    USER's act exactly as the scores are."""

    def __init__(self, day: str, outdir: Path, latency_ms: int):
        self.path = outdir / f"be_forward_day_SEALED_feed_{day}.jsonl"
        self.latency_ms = latency_ms
        self._fh = None
        self.n_rows = 0
        self.n_windows = 0

    def __enter__(self):
        self._fh = open(self.path, "w")
        return self

    def write_window(self, wrows, scores):
        import be_forward_metric as FM
        recs = FM.reduce_window(wrows, scores, self.latency_ms)
        for rec in recs:
            self._fh.write(json.dumps(rec, sort_keys=True,
                                      separators=(",", ":")) + "\n")
        self.n_rows += len(recs)
        self.n_windows += 1
        return len(recs)

    def __exit__(self, *a):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return False

    def manifest(self) -> dict:
        return {
            "protocol": FEED_PROTOCOL,
            "path": str(self.path),
            "sha256": _sha_file(self.path) if self.path.exists() else None,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
            "n_rows": self.n_rows, "n_windows": self.n_windows,
            "latency_ms_resolved": self.latency_ms,
            "fields": list(FEED_FIELDS),
            "contents": ("one action-keyed record per scored row: the action "
                         "key, the times, the score and the latency-resolved "
                         "preventable value"),
            "sealed": True,
            "not_in_receipt": ("no value from the feed appears outside it; "
                               "this block is paths, counts and hashes only "
                               "(rule 11)"),
        }


def seal(day: str, outdir: Path, scored: dict, report: dict) -> dict:
    """Scores OUT of the receipt and into a sealed file. Rule 11.

    The receipt records the sealed file's sha256 and nothing about its
    contents. Reading it is the coordinator's or the USER's act."""
    sp = outdir / f"be_forward_day_SEALED_scores_{day}.json"
    sp.write_text(json.dumps(
        {"protocol": "BE_FORWARD_DAY_SEALED_SCORES_V1", "day": day,
         "SEALED": "rule 11: not for the filing. Counts and refusals only.",
         "per_coin_scores": {c: [list(x) for x in v]
                             for c, v in sorted(scored.items())},
         "report": report}, indent=1, sort_keys=True, default=str))
    return {"path": str(sp), "sha256": _sha_file(sp),
            "bytes": sp.stat().st_size,
            "contents": "per-action scores and the full complement report",
            "not_in_receipt": "no metric, rho, net value or sign appears "
                              "outside this file"}


#: Decision-shaped vocabulary. Borrowed BY VALUE from
#: `de_admissible_windows.DECISION_VOCAB` rather than restated, so the two
#: cannot drift; the local additions are named.
def _decision_vocab() -> tuple:
    extra = ("counts_toward_g", "counts_toward_G", "g_contribution",
             "qualifies", "admitted", "advance")
    return tuple(AW.DECISION_VOCAB) + extra


def _walk_keys(obj):
    for _p, k, _v in _walk_paths(obj):
        yield k


def _walk_paths(obj, pre=""):
    """(path, key, value) for every key, lists collapsed to `[]`.

    PATHS, not bare keys, because the allowlist below has to bind to ONE
    place. A bare-key allowlist would excuse the same word wherever it
    appeared, which is precisely the smuggling route it must not open."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{pre}.{k}" if pre else str(k)
            yield path, str(k), v
            yield from _walk_paths(v, path)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_paths(v, pre + "[]")


# Decision-shaped words that are NOT entitlements at ONE path, each with the
# reason. MEASURED: the real receipt carries `gates[].gate` -- a gate's NAME
# -- and the borrowed vocabulary flagged it, so the post-condition would have
# refused every real run. Narrowing the vocabulary was the wrong repair: the
# word is fine THERE and nowhere else, and only a path can say that. The
# value must still be a string, or `gates[].gate: true` would smuggle a
# boolean through the exemption.
# PINNED, and the pin is asserted (BE5-R2). `excused_paths` reports what an
# emission USED, so a second entry that nothing happens to hit leaves every
# check green and the growth is already in the artifact by the time anything
# notices. An exemption from a rule-14 post-condition grows by a deliberate,
# visible act: adding a path here fails the membership assertion until this
# tuple is changed too. The idiom is `de_admissible_windows`'s
# BLIND_ENTRY_ASSERTIONS and `de_ratification_check`'s SCOPE_OPEN_TOKENS.
DECISION_ALLOWLIST_PINNED = ("gates[].gate",)

DECISION_ALLOWLIST = {
    "gates[].gate": ("the NAME of a gate this run ran (a string identifier), "
                     "not a right to anything. Any non-string here refuses, "
                     "and the same word at any other path refuses."),
}


def assert_no_decision_field(emission: dict) -> dict:
    """Nothing decision-shaped may leave this driver (rule 14).

    A POST-CONDITION on the emission, the idiom `de_admissible_windows` and
    the replay seam already carry: the R-411 inputs below are ESTIMATES for
    the policy layer to compute G from, and a driver that shipped
    `counts_toward_G` would be making the decision instead of supplying its
    inputs."""
    vocab = {v.lower() for v in _decision_vocab()}
    hits, excused = [], []
    for path, k, v in _walk_paths(emission):
        if k.lower() not in vocab:
            continue
        if path in DECISION_ALLOWLIST and isinstance(v, str):
            excused.append(path)
            continue
        hits.append(f"{path}={v!r}" if path in DECISION_ALLOWLIST else path)
    hits = sorted(set(hits))
    if hits:
        raise ForwardDayRefused(
            f"REFUSED: the receipt carries decision-shaped field(s) {hits}. "
            f"This driver SUPPLIES counts and refusals; whether a day counts "
            f"toward G is the policy layer's (rule 14, R-411(i)).")
    return {"checked_keys": len(set(_walk_keys(emission))),
            "excused_paths": sorted(set(excused)),
            "allowlist": DECISION_ALLOWLIST,
            "vocabulary_size": len(vocab),
            "vocabulary_source": "de_admissible_windows.DECISION_VOCAB, by "
                                 "value, plus named local additions"}


def receipt_path(outdir: Path, day: str) -> Path:
    return outdir / f"be_forward_day_receipt_{day}.json"


def _flush(rec: dict, outdir: Path, day: str) -> Path:
    """Write the receipt for THIS run, never over another run's.

    MEASURED: a second run into an outdir that already held a receipt
    OVERWROTE it -- the 12:49 receipt was lost that way, and the two runs
    differed in exactly the fields a reader would have wanted to compare. A
    receipt is evidence; evidence is not a scratch buffer. The FIRST write of
    a run claims the canonical name and every later write in the SAME run
    updates it (the durable-flush guarantee); a NEW run finding a receipt it
    did not write takes a numbered successor and leaves the old file
    byte-identical."""
    # The post-condition's RESULT is recorded, not discarded: a check whose
    # only trace is that nothing happened cannot be told apart from a check
    # that was deleted. Computed on the receipt WITHOUT this block, then
    # attached, so it describes what it actually examined.
    rec["decision_field_check"] = assert_no_decision_field(
        {k: v for k, v in rec.items()
         if k not in ("_receipt_path", "decision_field_check")})
    p = receipt_path(outdir, day)
    claimed = rec.get("_receipt_path")
    if claimed:
        q = Path(claimed)
        q.write_text(json.dumps(
            {k: v for k, v in rec.items() if k != "_receipt_path"},
            indent=1, sort_keys=True, default=str))
        return q
    if p.exists():
        # BE5-R1: the chain, not a star. Recording `p` whatever N is made
        # BOTH `.1` and `.2` claim to supersede the BASE, so two receipts
        # named the same predecessor and "which is current" was answerable
        # only by sorting filenames -- in a programme whose rule 13 is
        # precisely about what superseded what. The successor now names the
        # receipt it actually STANDS AFTER (the highest-numbered existing
        # one), and carries the whole chain beside it so the order is
        # readable from the receipts themselves.
        prior = [p]
        n = 1
        while True:
            q = outdir / f"be_forward_day_receipt_{day}.{n}.json"
            if not q.exists():
                break
            prior.append(q)
            n += 1
        stands_after = prior[-1]
        rec["supersedes_receipt"] = {
            "path": str(stands_after), "sha256": _sha_file(stands_after),
            "is_base": stands_after == p, "n_prior": len(prior),
            "why": "the receipt this run STANDS AFTER -- the highest-numbered "
                   "one already present, not the base. Every prior receipt is "
                   "KEPT byte-identical and this run takes the next number; "
                   "overwriting evidence loses the comparison a reader needs, "
                   "and naming the base from every successor loses the ORDER "
                   "(rule 13)."}
        rec["prior_receipts"] = [
            {"path": str(x), "sha256": _sha_file(x)} for x in prior]
        rec["_receipt_path"] = str(q)
        q.write_text(json.dumps(
            {k: v for k, v in rec.items() if k != "_receipt_path"},
            indent=1, sort_keys=True, default=str))
        return q
    rec["_receipt_path"] = str(p)
    p.write_text(json.dumps(
        {k: v for k, v in rec.items() if k != "_receipt_path"},
        indent=1, sort_keys=True, default=str))
    return p


def run_forward_day(day: str, outdir: Path) -> int:
    """THE SEQUENCE. Every gate refuses BY NAME; a refusal still writes a
    receipt carrying what was established before it, so a refused day is a
    reported fact rather than a silence."""
    import time
    t_start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    # R24: this hardcoded RATIFICATION_REF, so the FIRST 08-29 receipt says
    # `R-419` at the top while the population was admitted under `R-502` --
    # the headline ref contradicting the ref actually used. It is derived
    # from the same selector the gate uses, so the two cannot disagree.
    rec: dict = {"protocol": "BE_FORWARD_DAY_SEALED_V1", "day": day,
                 "ratification_ref": ratification_for(day)["ref"],
                 "as_of_utc": _as_of(),
                 "sealed": True,
                 "sealing_note": "per-action scores and every value metric are "
                                 "written to the sealed file only; this "
                                 "receipt carries counts, identities and "
                                 "hashes and NO metric (rule 11). Unsealing "
                                 "is the coordinator's or the USER's act.",
                 "producing_code": _provenance(),
                 "gates": []}

    def gate(name, fn):
        try:
            out = fn()
        except Exception as e:                       # noqa: BLE001
            rec["gates"].append({"gate": name, "result": "REFUSED",
                                 "why": str(e),
                                 "refusal_type": type(e).__name__})
            raise
        rec["gates"].append({"gate": name, "result": "PASS"})
        # DURABLE: an OOM kill bypasses every `finally`, and the 12 G run died
        # after 21 minutes leaving NO receipt at all. The receipt is written
        # after each gate so a killed run still says how far it got.
        _flush(rec, outdir, day)
        return out

    rc = 0
    try:
        _adm = gate("user_admission", lambda: admitted_verdict(day))
        # rule 14 again: `admitted` is a decision-shaped NAME, and
        # `assert_no_decision_field` refused it. The record says what is
        # true without wearing an entitlement's clothes.
        rec["user_admission"] = _adm["record"] if _adm else {
            "no_admission_covers_this_day": True,
            "note": "the ordinary gate applies, unchanged"}
        rec["day_verdict"] = gate(
            "day_closed_and_attributed",
            lambda: assert_day_closed_and_attributed(
                day, verdict=(_adm["verdict"] if _adm else None),
                admission=(_adm["record"] if _adm else None)))
        if _adm:
            rec["day_verdict"]["attribution_source"] = (
                "SUPERSEDED-BUT-GENUINE BYTES ADMITTED BY THE USER -- see "
                "`user_admission`")
        rec["frozen_contract"] = gate("frozen_contract", frozen_contract_gate)
        pop = gate("population_supply_and_bridge", lambda: population(day))
        rec["population"] = {
            "present_source": str(MARKETS),
            "mask_identity_hash": pop["supply"]["mask_identity_hash"],
            "mask_consumed": pop["supply"]["mask_consumed"],
            "governed": pop["supply"]["governed"],
            "counts_per_coin": counts_per_coin(pop),
            "n_supplied_total": pop["supply"]["n_supplied_total"],
            "n_bridged_specs": len(pop["specs"])}
        # THE FROZEN CONTRACT IS CHECKED AFTER THE POPULATION, DELIBERATELY.
        # The population is a property of the day's ledger and DA's mask —
        # both committed artifacts, neither touched by the candidate — so
        # establishing it costs one file read and no replay, and a refusal
        # here still tells the coordinator what the day's population WAS.
        _r = pop.get("ratification") or {}
        _pa = _r.get("pair_asserted")
        if (not isinstance(_pa, dict) or _pa.get("asserted") is not True
                or _pa.get("verified_seen") != _r.get("verified")
                or _pa.get("unverifiable_seen") != list(
                    _r.get("unverifiable") or [])
                or _pa.get("ref_seen") != _r.get("ratification_ref")):
            raise ForwardDayRefused(
                f"REFUSED: the ratification PAIR assertion left no evidence "
                f"that it RAN ON THIS CHECKER'S OUTPUT (CO-5). Its record "
                f"must reproduce the values it saw -- a bare flag can be "
                f"forged by the same edit that bypasses the call. Got "
                f"{_pa!r}.")
        rec["population"]["ratification"] = {
            k: v for k, v in _r.items()
            if k in ("verified", "unverifiable", "day_in_scope",
                     "ratification_ref", "pair_asserted",
                     "require_verified")}
        # The gate itself is in `population()` -- reachable, and driven by the
        # suite with a refusing checker. What remains here is the RECORD.
        if not isinstance(_r.get("require_verified"), dict):
            raise ForwardDayRefused(
                "REFUSED: `de_ratification_check.require_verified` left no "
                "evidence that it ran. It is the CONSUMER's fail-closed gate "
                "(verified AND nothing unverifiable AND not provenance); a "
                "local pair assertion is the evidence, not the gate.")
        rec["population"]["ledger_vs_archive"] = pop["ledger_vs_archive"]
        rec["r411_inputs"] = pop["r411_inputs"]
        # R-421 §2 / rule 12: EXECUTE THE FROZEN BYTES. Round 3 refused
        # because the TREE moved; the freeze's bytes exist at the commit and
        # this materialises them, verifies each sha BEFORE import, and
        # imports the anchors from the run dir.
        mat = gate("materialise_frozen_bytes",
                   lambda: materialise_frozen(outdir))
        rec["frozen"] = mat
        _anchor_keys = [k for k, v in mat["anchors"].items()
                        if v.get("compared")]
        rec["frozen"]["closure"] = gate(
            "import_closure_disclosure",
            lambda: import_closure(Path(mat["root"]), _anchor_keys))
        rec["frozen"]["imports"] = gate(
            "import_anchors_from_run_dir",
            lambda: import_frozen_anchors(Path(mat["root"]), _anchor_keys))
        sel, selc = gate("selection_from_specs",
                         lambda: selected_from_specs(pop["specs"]))
        rec["selection"] = selc
        _frozen = FS.load_frozen(expect=FS.declared_candidate_identity())
        _feed = FeedWriter(day, outdir, _latency_of_record())
        with _feed:
            built = gate("rows_and_scores_streamed",
                         lambda: build_and_score(sel, _frozen, feed=_feed))
        rec["feed"] = _feed.manifest()
        rec["rows"] = {k: v for k, v in built.items()
                       if k not in ("scores", "windows_with_rows")}
        if built["reconciliation_failures"]:
            # BE6-R1: the two bare raises here are NOT inside `gate()`, so
            # the generic handler blamed `gates[-1]` -- the gate that had
            # just PASSED, and the suite pinned that as expected. A reader,
            # or the automated resolver rule 13 exists for, was sent to a
            # gate that succeeded. Each bare raise now names its own check.
            rec["refused_at"] = "reconciliation"
            # BE6-R4: the count without the name. WHICH windows failed is
            # what an operator needs, bounded the way
            # `assert_window_sets_agree` already bounds its examples.
            _bad = sorted(built.get("reconciliation_failed_windows") or ())
            raise ForwardDayRefused(
                f"REFUSED: {built['reconciliation_failures']} of "
                f"{built['n_windows']} windows failed reconciliation "
                f"({', '.join(_bad[:8])}"
                f"{f' and {len(_bad) - 8} more' if len(_bad) > 8 else ''}). "
                f"The reconciliation selftest is the gate: a mismatch fails "
                f"the DAY and is never absorbed.")
        rec["gates"].append({"gate": "reconciliation", "result": "PASS"})
        rec["window_agreement"] = gate(
            "bridged_windows_equal_row_windows",
            lambda: assert_window_sets_agree(
                pop["specs"], built["windows_with_rows"]))
        scored = built["scores"]
        if not scored:
            rec["refused_at"] = "zero_actions_scored"
            raise ForwardDayRefused(
                f"REFUSED: zero actions scored across {built['n_rows']} rows. "
                f"A forward report with no scores is a FAILURE, not an empty "
                f"result (R-141).")
        rep = gate("mask_seam_and_complement_report",
                   lambda: FS.score_day(day, scored, da_verified=True))
        rec["blackout_accounting"] = rep["blackout_accounting"]
        rec["n_actions_scored"] = rep["n_actions_scored"]
        # TWO DISCLOSURES, because the counts above read wrongly without them.
        #
        # (1) THE COMPLEMENT WAS APPLIED AT SUPPLY, NOT AT SCORING. The mask
        # removed its windows before any row was built, so the scoring-stage
        # seam finds ZERO more to mask. `n_masked: 0` there does NOT mean
        # nothing was masked — it means nothing was left to mask.
        _sup = pop["supply"]["counts"]
        rec["masking"] = {
            "applied_at": "supply (de_admissible_windows), before rows",
            "n_masked_at_supply": {c: v["n_masked_applied"]
                                   for c, v in sorted(_sup.items())},
            "n_masked_total_at_supply": sum(v["n_masked_applied"]
                                            for v in _sup.values()),
            "n_masked_at_scoring": rep["blackout_accounting"]["n_masked"],
            "why": "R-418 supplies PRESENT minus MASKED, so the blackout is "
                   "already excluded when rows are built. A reader seeing 0 "
                   "at the scoring seam must not conclude the day had no "
                   "blackout."}
        # (2) THE FROZEN CANDIDATE DOES NOT COVER EVERY SUPPLIED COIN. R-418
        # supplies all seven; the frozen fits cover two. The other five
        # produced rows and NO score — that is the candidate's scope, not a
        # failure, and the receipt must not let "scored" read as "whole day".
        _fit_coins = sorted(_frozen["fits"])
        _sup_coins = sorted(_sup)
        rec["coin_coverage"] = {
            "coins_supplied": _sup_coins,
            "coins_with_a_frozen_fit": _fit_coins,
            "coins_supplied_without_a_fit": [c for c in _sup_coins
                                             if c not in _fit_coins],
            "n_windows_supplied_without_a_fit": sum(
                v["n_supplied"] for c, v in _sup.items()
                if c not in _fit_coins),
            "why": "the frozen candidate carries fits for "
                   f"{_fit_coins} only. Windows of the other coins are "
                   "supplied, replayed and counted, and produce NO score. "
                   "The day is not scored whole and this says so."}
        # READ-R3, CARRIED FROM THE START rather than added after a result
        # exists. The 08-29 artifacts promised a cluster disclosure in the
        # pre-declaration and then carried it in NEITHER the receipt nor the
        # cells, so the artifact understated its own limits -- the one thing
        # a result must never do. Both blocks are computed here, in the
        # receipt every run writes.
        rec["cluster_disclosure"] = {
            "ruled_cluster_unit": "UTC day",
            "G_complete_days_in_THIS_run": 1,
            "bar_for_an_interval": 5,
            "intervals_claimable_from_this_run_alone": False,
            "what_this_run_supports": "A POINT ESTIMATE AND NO INTERVAL",
            "authority": "CLAUDE.md rule 8 -- below G=5 complete days: point "
                         "estimate, no interval, and say so",
            "unit_available_below_the_ruled_unit": "window",
            "n_windows_with_rows": rec["rows"]["n_windows_with_rows"],
            "weaker_than_ruled": True,
            "why_a_window_level_p_is_OPTIMISTIC": (
                "a window-level null treats windows as exchangeable draws. "
                "Windows inside one UTC day share coin, regime and book "
                "state, so the null variance is understated and any p "
                "computed on them is SMALLER than a day-clustered p would "
                "be. One day cannot be permuted at the ruled unit at all."),
            "pooling_note": (
                "G counts COMPLETE UTC DAYS across the runs a claim rests "
                "on, not windows within one; this field describes THIS run "
                "and a pooled claim must recompute it over its own days"),
        }
        rec["reconciliation_caveat"] = {
            "claim": ("the decision metric has NEVER been reconciled against "
                      "any published number, and cannot be from existing "
                      "artifacts"),
            "why_not": (
                "`increment()` computes a BY_THRESHOLD estimand and iteration "
                "011's published cells are BY_COUNT -- different estimands, "
                "so there is no published number to reconcile against"),
            "what_the_36_of_36_did_validate": (
                "the BRIDGE arm and everything downstream of "
                "`increment_by_window`, not the primary estimand"),
            "for_the_reader": ("do not carry the reconciliation's authority "
                               "onto any number derived from this run"),
        }
        sealed = seal(day, outdir, scored, rep)
        rec["sealed_file"] = sealed
        rec["outcome"] = "SCORED"
    except Exception as e:                           # noqa: BLE001
        rec["outcome"] = "REFUSED"
        # A bare raise names its own check; only a failure INSIDE `gate()`
        # (whose entry is the REFUSED one) falls back to gates[-1].
        rec["refused_at"] = rec.get("refused_at") or (
            rec["gates"][-1]["gate"] if rec["gates"] else None)
        rec["refusal"] = str(e)
        rc = 1
    rec["wall_seconds"] = round(time.time() - t_start, 1)
    rp = _flush(rec, outdir, day)
    print(f"{'REFUSED' if rc else 'OK'} {day}: receipt {rp}")
    print(f"  receipt_sha256 {_sha_file(rp)[:16]}")
    if rec.get("refusal"):
        print(f"  {rec['refusal'][:400]}")
    return rc


# ---------------------------------------------------------------------------
# BE34-R1: ONE fixture, TWO consumers
# ---------------------------------------------------------------------------
# `build_and_score` replaces `score_rows` -- it replays, labels and scores in
# one streaming pass so the run stops carrying 1,875 windows of rows. Nothing
# asserted the two agreed, and `score_rows` had become dead code that still
# reads as the reference. The idiom is `v5_chain_equivalence_test.py`'s: ONE
# fixture through BOTH consumers, asserting each side AND their agreement --
# not two independently constructed approximations.
#
# Values are kept SMALL on purpose: a score of 1e9 cannot represent a 1e-9
# perturbation in a double, so a fixture with large numbers would silently
# pass the very mutant this exists to catch.

_R1_W = [1.0, 2.0, 0.5, 3.0, 0.25]          # 3 window feats + 2 fine feats
#: ROUND 25: the fixture's INCUMBENT, in the same stub shape as the
#: candidate. Deliberately DIFFERENT weights: an incumbent stub equal to the
#: candidate's would make every increment identically zero, and a cell that
#: is zero by construction cannot tell a working estimand from a broken one.
_R1_INC = {"btc": {"w": [x * 0.5 for x in _R1_W]},
           "eth": {"w": [x * 0.25 for x in _R1_W]}}
_R1_FROZEN = {"fits": {"btc": {"w": _R1_W},
                       "eth": {"w": [x * 1.5 for x in _R1_W]}}}


def _r1_windows(bad_window: str = None) -> tuple:
    """(selected, rows) for a handful of windows with KNOWN features."""
    selected, rows = [], []
    # BE6-R7: `sol` is present in the windows and ABSENT from the frozen
    # fits -- the real day's DOMINANT class (five of seven supplied coins
    # carry no fit, 1,344 of 1,875 windows). Without it the equality never
    # entered `build_and_score`'s no-fit branch, so the two consumers could
    # have stopped agreeing there and nothing would have noticed.
    for k, (coin, t0) in enumerate((("btc", 1788000000), ("eth", 1788000300),
                                    ("sol", 1788000600), ("btc", 1788000900),
                                    ("eth", 1788001200), ("sol", 1788001500))):
        slug = f"{coin}-updown-5m-{t0}"
        selected.append((slug, f"/fixture/{slug}", "U", "D", ()))
        for j in range(3):
            rows.append({
                "coin": coin, "slug": slug, "t0": t0,
                "t_start": 10 * (j + 1) + k, "side": "up" if j % 2 else "dn",
                "gen": j, "resting": 0.5 * j, "qahead": j,
                # level -1 is the row whose features come back None: BOTH
                # consumers must drop it, and neither may score it.
                "level": -1 if (k == 1 and j == 2) else j})
    return selected, rows


def _r1_fakes(rows: list, bad_window: str = None) -> tuple:
    """Fake HER/hm/FS collaborators. Deterministic, no data on disk."""
    import types
    by_slug: dict = collections.defaultdict(list)
    for r in rows:
        by_slug[r["slug"]].append(r)

    def features(stream, t_start, side, level, resting, qahead):
        if level == -1:
            return None
        return [float(t_start), 1.0 if side == "up" else -1.0,
                float(len(stream[1]) % 97)]

    def fine_feats(t_abs, side, coin):
        return [float(t_abs % 997), 1.0 if coin == "btc" else 2.0]

    def expected_cancel_value(fit, vec):
        return sum(v * w for v, w in zip(vec, fit["w"]))

    slugs = sorted(by_slug)
    fi = types.SimpleNamespace(
        _archive_paths=lambda: {sg: f"/fixture/{sg}" for sg in slugs},
        token_map=lambda: {sg: ("U", "D") for sg in slugs})
    hm = types.SimpleNamespace(
        fi=fi, features=features, fine_feats=fine_feats,
        window_streams=lambda path, up, dn: ("STREAM", path, up, dn))

    def replay(path, up, down, wgaps, spec):
        slug = path.rsplit("/", 1)[1]
        arm = types.SimpleNamespace(
            fill_log=slug, fills=slug, segments=slug,
            consume_times=slug, unhooked_changes=0)
        return arm, slug

    def join_fills(fill_log, fills):
        n = 1 if fill_log == bad_window else 0
        return fill_log, {"count_mismatch": n, "tuple_mismatches": 0}

    qr = types.SimpleNamespace(
        QR_SKEW=None, _qr_spec=lambda skew, latency_ms, cancel: "SPEC",
        base=types.SimpleNamespace(fi=types.SimpleNamespace(WINDOW_S=300)))
    HER = types.SimpleNamespace(
        qr=qr, replay_with_recorder=replay, join_fills=join_fills,
        verify_boundary_times=lambda seg, joined: 0,
        verify_consume_clock=lambda ct, trt: 0,
        trade_receipt_times=lambda path, up, dn: [],
        generation_table=lambda seg, joined, wf, w: (
            seg, {"orphan_fills": 0, "wrong_generation_assignments": 0}),
        label_rows=lambda seg, gens, wf, w: list(by_slug[seg]))
    return HER, hm, features, fine_feats, expected_cancel_value


class _r1_installed:
    """Install the fakes for BOTH consumers, and put everything back."""

    def __init__(self, rows, bad_window=None):
        self.HER, self.hm, _f, _ff, self.ecv = _r1_fakes(rows, bad_window)

    def __enter__(self):
        self._saved_mods = {n: sys.modules.get(n) for n in
                            ("harmful_exposure_rows", "harmful_hazard_model")}
        sys.modules["harmful_exposure_rows"] = self.HER
        sys.modules["harmful_hazard_model"] = self.hm
        self._saved_fs = (FS.expected_cancel_value, FS.load_frozen)
        FS.expected_cancel_value = self.ecv
        # BEM-R3: the production signature now REQUIRES an expectation,
        # so the stub takes it too -- a fixture whose signature drifts
        # from the thing it replaces stops testing the real call.
        FS.load_frozen = lambda path=None, expect=None: _R1_FROZEN
        return self

    def __exit__(self, *exc):
        for n, m in self._saved_mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        FS.expected_cancel_value, FS.load_frozen = self._saved_fs
        return False


# ---------------------------------------------------------------------------
# BE5-R3: the mutation audit SHIPS
# ---------------------------------------------------------------------------
# The audit was the substance of rounds 5 and 6 -- it killed two of my own
# evidence attempts and found four call-site survivors -- and none of it was
# in the module, so "N/M killed" was a claim in a filing that no reader could
# re-run. Rule 15 at the level of the harness: a checker ships its falsifier.
#
# Each case names THE EDIT and THE CHECK THAT MUST GO RED. A case passes only
# if the mutated module fails AND fails at that check; a mutant that dies for
# some other reason is NOT counted as killed, because then the named check is
# not what caught it.
#
# The edit is applied to a COPY in a temp tree, never to this file. Two runs
# of an earlier scratch harness were SIGKILLed mid-mutation and left a mutant
# in the working tree; a copy cannot do that. The importable siblings are
# COPIED, not symlinked -- `Path(__file__).resolve()` follows a symlink, so
# symlinked siblings put the real tree back at the front of `sys.path` and
# the copy stopped being what ran. `data/` and the other tree roots are
# linked and read through the real repo path, read-only.
#
# SCOPE, stated so the count is not mistaken for more than it is: these cases
# cover the round-5 and round-6 items and the two closures of round 7. They
# are not the whole 63-mutant sweep run out-of-tree, and three of that sweep's
# four survivors are recorded in `AUDIT_KNOWN_UNKILLABLE` below with the
# reason each cannot be killed from inside one tree.

AUDIT_KNOWN_UNKILLABLE = {
    "spawn root REPO vs parents[2]":
        "a no-op wherever the two names denote the same path, which is true "
        "in the shared tree; it is observable only from a worktree whose file "
        "differs, and it was executed and killed there",
    "a `want` in AUDIT_CASES is edited":
        "an anchor that exists ONLY inside the case table cannot be applied: "
        "`_audit_apply` requires it to be uniquely locatable OUTSIDE the "
        "table, precisely so that a case cannot rewrite the table. The "
        "BE34-R5 want mismatch this round found is therefore NOT expressible "
        "as a shipped mutant; standing in its place is the table-wide "
        "assertion that every case died at a line OPENING with its want -- "
        "which is the check that caught it",
    "a true conjunct is deleted":
        "removing a conjunct that holds whenever the code is correct cannot "
        "be seen by correct code; the conjunct's load-bearing-ness is shown "
        "instead by OTHER cases dying only once it exists",
}

# (name, old, new, the check whose text must appear in the failure)
AUDIT_CASES = (
    ("BE5-R1 the successor names the base again, not what it stands after",
     '        stands_after = prior[-1]', '        stands_after = prior[0]',
     "BE5-R1 the chain is a CHAIN"),
    ("BE5-R1 the prior chain is not carried",
     '        rec["prior_receipts"] = [', '        _unused_prior = [',
     "BE5-R1 and the whole chain is carried beside it"),
    ("BE5-R2 a second path is added to the allowlist",
     'DECISION_ALLOWLIST = {\n    "gates[].gate":',
     'DECISION_ALLOWLIST = {\n    "population.gate": "smuggled",\n'
     '    "gates[].gate":',
     "BE5-R2 the excused-path allowlist is PINNED"),
    ("BE5-R2 the pin is widened to match instead of the allowlist narrowed",
     'DECISION_ALLOWLIST_PINNED = ("gates[].gate",)',
     'DECISION_ALLOWLIST_PINNED = ("gates[].gate", "population.gate")',
     "BE5-R2 the excused-path allowlist is PINNED"),
    ("R5(1) a receipt overwrites an earlier run's",
     '    if p.exists():\n        # BE5-R1', '    if False:\n        # BE5-R1',
     "R5(1) KNOWN-BAD: a SECOND run into the same outdir"),
    ("R5(5) the decision-field post-condition is removed from the flush",
     '    rec["decision_field_check"] = assert_no_decision_field(',
     '    _skip = dict(',
     "R5(5) a decision-shaped key (counts_toward_G) must REFUSE"),
    ("R5(5) the post-condition looks at top-level keys only",
     '    for path, k, v in _walk_paths(emission):',
     '    for path, k, v in [(str(x), str(x), emission[x]) for x in emission]:',
     "R5(5) a decision-shaped key (counts_toward_G) must REFUSE"),
    ("R5(5) the excused path stops requiring a string",
     '        if path in DECISION_ALLOWLIST and isinstance(v, str):\n'
     '            excused.append(path)',
     '        if path in DECISION_ALLOWLIST:\n            excused.append(path)',
     "R5(5) must REFUSE (a boolean at the excused path)"),
    ("BE34-R5 the closure method is no longer declared",
     '            "closure_method": "STATIC import walk (ast Import/ImportFrom, "',
     '            "closure_method_UNUSED": "STATIC import walk (ast Import/'
     'ImportFrom, "',
     # BE8-R1: the label OPENS with `R-421(2)/`; as `BE34-R5 …` this want
     # was a substring of its label but NOT a prefix, and only the substring
     # test made it read as a kill.
     "R-421(2)/BE34-R5 the closure DECLARES its method"),
    ("CO-12 the attribution matches the TRANSCRIPT again, not the failure",
     '    line = _audit_failure_line(stderr)\n'
     '    return bool(line) and line.startswith(want)',
     '    line = _audit_failure_line(stderr)\n    return want in stderr',
     "CO-12 KNOWN-BAD: a `want` that appears only on a `  PASS  ` line"),
    ("CO-12 the failure line is taken from the FIRST raise, not the last",
     '    i = stderr.rfind(_AUDIT_RAISE)', '    i = stderr.find(_AUDIT_RAISE)',
     "CO-12 with a CHAINED traceback the attribution takes the LAST"),
    ("BE7-R4 the provenance root is hardcoded again",
     '    root = Path(__file__).resolve().parents[2] if tree is None else Path(tree)',
     '    root = REPO if tree is None else Path(tree)',
     "BE7-R4 a receipt written from a WORKTREE names THAT worktree's HEAD"),
    ("BE9-C1 the flip copies the running bytes again (a NO-OP whenever "
     "the branch already carries this driver)",
     '            _tgt.write_bytes(Path(__file__).read_bytes()\n'
     '                             + b"\\n# BE7-R4 planted difference\\n")',
     '            _tgt.write_bytes(Path(__file__).read_bytes())',
     "BE7-R4 and it FLIPS with that tree"),
    ("BE9-C1 the worktree's commit is read from the executing tree again",
     '            _wt_head = _git_in(_wt, "rev-parse", "HEAD")',
     '            _wt_head = _git_in(_me_tree, "rev-parse", "HEAD~1")',
     "BE7-R4 a CLEAN worktree reads clean and names ITS OWN HEAD"),
    ("BE9-C2 the anchors are compared against a fixed root again",
     '        p = EXEC_TREE() / k', '        p = REPO / k',
     "BE9-C2 code and ANCHORS resolve in the executing tree"),
    ("BE9-C2 the audit tree links a fixed root's entries again",
     '    for entry in EXEC_TREE().iterdir():', '    for entry in REPO.iterdir():',
     "BE9-C2 the audit copy links the EXECUTING tree's entries"),
    ("BE9-C3 the selftest prunes shared worktree state again",
     '            _sp.run(("git", "worktree", "remove", "--force", str(_wt)),\n'
     '                    cwd=str(_me_tree), capture_output=True, text=True)',
     '            _sp.run(("git", "worktree", "remove", "--force", str(_wt)),\n'
     '                    cwd=str(_me_tree), capture_output=True, text=True)\n'
     '            _sp.run(("git", "worktree", "prune"), cwd=str(REPO),\n'
     '                    capture_output=True, text=True)',
     "BE9-C3 a STALE worktree entry"),
    ("BE7-R2 n_prior is off by one and nothing notices",
     '"is_base": stands_after == p, "n_prior": len(prior),',
     '"is_base": stands_after == p, "n_prior": len(prior) - 1,',
     "BE5-R1 and the whole chain is carried beside it"),
    ("BE7-R3 a LEGITIMATE second exemption — table AND tuple both grow",
     'DECISION_ALLOWLIST_PINNED = ("gates[].gate",)',
     'DECISION_ALLOWLIST_PINNED = ("gates[].gate", "population.gate")\n'
     '_SECOND = {"population.gate": "a reason, stated"}',
     "BE5-R2 the excused-path allowlist is PINNED"),
    ("BE6-R1 the reconciliation refusal stops naming its own check",
     '            rec["refused_at"] = "reconciliation"', '            pass',
     "BE34-R1 KNOWN-BAD: the CALLER refuses the whole DAY by name"),
    ("BE6-R1 the zero-score refusal stops naming its own check",
     '            rec["refused_at"] = "zero_actions_scored"', '            pass',
     "BE6-R3 KNOWN-BAD: a day whose windows all reconcile but score"),
    ("BE6-R2 the launch parity ignores the sha again",
     '    return (rc == 0 and child_sha is not None and child_sha == expect_sha\n'
     '            and child == expect)',
     '    return rc == 0 and child == expect',
     "BE6-R2 the launch parity compares the SHA of the file that ran"),
    ("BE6-R4 the refusal drops WHICH windows failed",
     "                f\"({', '.join(_bad[:8])}\"",
     '                f"(",',
     "BE34-R1 KNOWN-BAD: the CALLER refuses the whole DAY by name"),
    ("BE6-R7 the fixture loses the coin with NO frozen fit",
     '("sol", 1788000600), ("btc", 1788000900),',
     '("btc", 1788000600), ("btc", 1788000900),',
     "BE34-R1 and BOTH drop the SAME rows"),
    ("BE6-R6 the restore evicts everything absent from the snapshot again",
     '            if _f.startswith(_tree) or (root and _f.startswith(str(root))):',
     '            if True:',
     "BE6-R6 the restore leaves the TREE's modules alone"),
    ("BE34-R4 the usage error returns success again",
     '        print("usage: be_forward_day.py --selftest | "\n'
     '              "--forward-day <YYYYMMDD> --outdir <dir>")\n        return 2',
     '        print("usage: be_forward_day.py --selftest | "\n'
     '              "--forward-day <YYYYMMDD> --outdir <dir>")\n        return 0',
     "BE34-R4 a usage error RETURNS 2"),
    # ---- round 12: BE8-R1/R2 and BE10-R1..R4 -------------------------
    ("BE8-R1 the attribution is a SUBSTRING test again",
     "    return bool(line) and line.startswith(want)",
     "    return bool(line) and want in line",
     "BE8-R1 attribution is a PREFIX match"),
    ("BE8-R2 a SECOND incrementer of `checks` reappears",
     "    # BE8-R2: NO increment here.",
     "    checks += 1\n    # BE8-R2: NO increment here.",
     "BE8-R2 `ok` is the ONLY thing that increments"),
    ("BE10-R1 the git read ignores its return code again",
     "        if r.returncode != 0 or not _outv:",
     "        if False:",
     "BE10-R1 KNOWN-BAD: a git read that FAILS refuses BY NAME"),
    ("BE10-R2 an empty --git-common-dir resolves again",
     "    if rc != 0 or not out:\n        raise ForwardDayRefused(\n"
     "            f\"REFUSED: `git rev-parse --git-common-dir` in {tree} "
     "exited \"",
     "    if False:\n        raise ForwardDayRefused(\n"
     "            f\"REFUSED: `git rev-parse --git-common-dir` in {tree} "
     "exited \"",
     "BE10-R2 KNOWN-BAD: an EMPTY or FAILED `--git-common-dir` REFUSES"),
    ("BE10-R3 the tampered byte is not actually tampered",
     '        _t.write_bytes(_t.read_bytes() + b"\\n# tampered\\n")',
     '        _t.write_bytes(_t.read_bytes())',
     "R-421(2)/BE10-R3 a tampered materialised byte is caught by the SHA"),
    # The mutant names a day that IS scorable. It dies at the guard, which
    # runs BEFORE `run_forward_day` -- so the case proves the guard without
    # any child ever scoring a real day (which is the whole point of it).
    ("BE12-S1 the real-emission control names a scorable day again",
     '        _day9 = "21000101"', '        _day9 = "20260902"',
     "BE12-S1 the real-emission control names a day that CANNOT"),
    ("BE10-R4 the receipt hardcodes whether the tree is REPO",
     '                "exec_tree_is_repo": _exec_tree_is_repo(),',
     '                "exec_tree_is_repo": True,',
     "BE10-R4 `roots` SAYS whether the executing tree IS `REPO`"),
)


def _audit_span(base: str) -> tuple:
    """Where the case table itself lives in this source.

    Every anchor below appears TWICE -- once at the site it names and once in
    the table that names it -- so a naive uniqueness test reports 2 for every
    case and the whole audit refuses to run. The table's own span is excluded
    and uniqueness is required OUTSIDE it, which is stricter than counting:
    an anchor that appears twice in real code still refuses."""
    a = base.index("AUDIT_CASES = (")
    b = base.index("\n)\n", a)
    return a, b


def _audit_apply(base: str, old: str, new: str) -> tuple:
    """(mutated source, n_outside). None when not uniquely locatable."""
    a, b = _audit_span(base)
    hits = []
    i = base.find(old)
    while i != -1:
        if not (a <= i < b):
            hits.append(i)
        i = base.find(old, i + 1)
    if len(hits) != 1:
        return None, len(hits)
    i = hits[0]
    return base[:i] + new + base[i + len(old):], 1


_AUDIT_RAISE = "AssertionError: "


def _checks_incrementers(src: str) -> list:
    """Every function that augments a name called `checks`, from the AST.

    BE8-R2: CO-13 was closed by REMOVAL and nothing said it must stay
    closed. Two increments existed -- `ok` (nonlocal) and `_selftest_launch`
    (on its parameter, returned and assigned back over `ok`'s) -- and they
    CANCELLED by arrangement, so the printed total was right by luck. A
    second `ok` in that function would have printed two PASS lines and added
    one. Attribution is to the INNERMOST enclosing function, or `ok`'s own
    increment would also be reported against `selftest`, which encloses it."""
    import ast
    found = []

    def _walk(node, fname):
        for ch in ast.iter_child_nodes(node):
            nf = (ch.name if isinstance(ch, (ast.FunctionDef,
                                             ast.AsyncFunctionDef))
                  else fname)
            if (isinstance(ch, ast.AugAssign)
                    and isinstance(ch.target, ast.Name)
                    and ch.target.id == "checks"):
                found.append(nf)
            _walk(ch, nf)

    _walk(ast.parse(src), "<module>")
    return sorted(found)


def _audit_failure_line(stderr: str) -> str:
    """The label the child DIED on, or None if it did not die at a check.

    `ok` raises `AssertionError(label)`, so the label is the tail of the
    traceback on STDERR. A mutant that dies some other way (a KeyError, a
    refusal, a non-zero return with no traceback) has NO failure label, and
    that is not a kill at a named check -- it is a survivor."""
    i = stderr.rfind(_AUDIT_RAISE)
    if i == -1:
        return None
    return stderr[i + len(_AUDIT_RAISE):].strip() or None


def _audit_attributed(want: str, stderr: str) -> bool:
    """Did the child die AT the named check? Matched against the failure
    line ALONE -- never against the transcript, where the same text appears
    on the `  PASS  ` line the check prints when it merely RAN.

    BE8-R1: the match is a PREFIX, not a substring. `want in line` credited
    a mutant to a case whose `want` merely APPEARED INSIDE another check's
    label -- the CO-12 control quotes the `died_at` it just measured, so its
    own failure line contains another case's `want` verbatim, and a mutant
    killing THAT control would have been credited to the other case. A
    prefix cannot collide that way, because every `want` is the OPENING of
    its target's label.

    MEASURED when this changed: one shipped `want` was NOT a prefix of its
    label (BE34-R5's, whose label opens `R-421(2)/`), so under `startswith`
    it became a SURVIVOR -- the substring test had been HIDING a mismatch
    between a case and the check it names. The `want` was corrected, and the
    prefix property is now asserted across the whole table."""
    line = _audit_failure_line(stderr)
    return bool(line) and line.startswith(want)


def _resolve_git_common_dir(out: str, rc: int, tree: Path) -> Path:
    """BE10-R2: an EMPTY `--git-common-dir` REFUSES, never resolves.

    `(tree / "").resolve()` IS `tree`, so an empty answer silently aimed the
    stale-entry cleanup at the executing tree: the `rmtree` missed,
    `ignore_errors=True` swallowed the miss, and the worktree entry this
    selftest PLANTED leaked into shared state -- in the one check whose
    whole subject is not leaving shared state behind."""
    out = (out or "").strip()
    if rc != 0 or not out:
        raise ForwardDayRefused(
            f"REFUSED: `git rev-parse --git-common-dir` in {tree} exited "
            f"{rc} with {out!r}. An empty answer resolves to the tree "
            f"ITSELF, which would aim this check's cleanup at the wrong "
            f"path and leak the admin entry it planted.")
    return Path(out) if out.startswith("/") else (tree / out).resolve()


def _audit_tree(src: str, root: Path) -> Path:
    """A temp tree holding the MUTATED copy and symlinks to its siblings."""
    here = Path(__file__).resolve().parent
    pkg = root / "live/pm_research"
    pkg.mkdir(parents=True, exist_ok=True)
    (root / "live/__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    me = Path(__file__).name
    for f in here.iterdir():
        if f.name in (me, "__init__.py") or f.name.startswith("__pycache__"):
            continue
        tgt = pkg / f.name
        if tgt.exists():
            continue
        # MEASURED: symlinking the siblings did not work. Nearly every module
        # here does `sys.path.insert(0, Path(__file__).resolve().parent)`, and
        # `resolve()` FOLLOWS a symlink -- so the first sibling imported put
        # the REAL directory at the front of sys.path and every later import
        # came from the tree, not the copy. Importable files are COPIED; the
        # rest (data, notebooks) are symlinked, since nothing resolves them.
        if f.is_file() and f.suffix == ".py":
            tgt.write_bytes(f.read_bytes())
        else:
            tgt.symlink_to(f)
    (pkg / me).write_text(src)
    # The copy's DATA and DOCUMENT roots are derived from `__file__` too
    # (round 4's lesson: materialising anchors into a run dir silently
    # emptied the archive index, and an empty index reads as "no windows"
    # rather than as a broken path). Every tree-root entry except `live` --
    # which the copy provides -- is linked, so `data/` and the ratification
    # register resolve. Linked for READING: the suite writes only into its
    # own temp dirs, and nothing here writes under `data/`.
    # BE9-C2: the audit's copy links the EXECUTING tree's entries, so a
    # child reads what this run reads. `data` is the exception below.
    for entry in EXEC_TREE().iterdir():
        # `data` is NEVER taken from the walk: in a copy it is itself a link
        # into the previous copy, and the chain drifts one hop per level.
        # It is linked from the absolute root below, uniformly, in any tree.
        if entry.name in ("live", "data"):
            continue
        lnk = root / entry.name
        if not lnk.exists():
            lnk.symlink_to(entry)
    # `data/` is absolute by design and a worktree may not have one, so it is
    # linked from REPO explicitly rather than inherited from the tree walk.
    if not (root / "data").is_symlink():
        (root / "data").symlink_to(REPO / "data")
    return pkg / me


def mutation_audit(cases=AUDIT_CASES) -> dict:
    """Every case driven against a COPY; survivors named, never counted away.

    A GREEN BASELINE IS REQUIRED. An earlier scratch harness mutated a
    RENAMED copy whose name-bound control failed on ANY copy, so every case
    'died' for the wrong reason and the audit proved nothing. If the
    unmutated copy is not green, this refuses rather than reporting kills.
    """
    import subprocess, tempfile as _tf, shutil as _sh
    base = Path(__file__).read_text(encoding="utf-8")
    env = dict(os.environ, BE_FORWARD_AUDIT="1", BE_FORWARD_LAUNCH_CHECK="1")
    per, survivors = {}, []

    def _run(text: str) -> tuple:
        with _tf.TemporaryDirectory() as td:
            mod = _audit_tree(text, Path(td))
            # R-446: a stale .pyc can answer for a source file that changed.
            _sh.rmtree(mod.parent / "__pycache__", ignore_errors=True)
            r = subprocess.run([sys.executable, str(mod), "--selftest"],
                               env=env, capture_output=True, text=True,
                               timeout=900)
            # CO-12 / R-454 §1: the two streams are kept SEPARATE. Joined,
            # a `want` that is a prefix of its check's own label matched the
            # `  PASS  {label}` line the check prints when it RUNS -- so
            # attribution said "died at the named check" for any mutant that
            # died anywhere AFTER it. 7 of 10 cases were in that class.
            return r.returncode, r.stdout, r.stderr

    rc0, out0, err0 = _run(base)
    if rc0 != 0:
        raise ForwardDayRefused(
            f"REFUSED: the UNMUTATED copy is not green (rc={rc0}), so no "
            f"case here could be said to die of its own edit. A harness "
            f"whose baseline is red proves nothing about its mutants. "
            f"Tail: {(out0 + err0).strip()[-400:]!r}")
    for name, old, new, want in cases:
        text, n_outside = _audit_apply(base, old, new)
        if text is None:
            survivors.append(name)
            per[name] = {"applied": False, "n_anchor_outside_table": n_outside,
                         "why": "the edit's anchor is not uniquely locatable "
                                "outside the case table, so this case did "
                                "NOT run and is counted as a survivor -- a "
                                "case that did not execute is never a kill"}
            continue
        rc, out, err = _run(text)
        died_at = _audit_failure_line(err)
        at_named = _audit_attributed(want, err)
        per[name] = {"applied": True, "rc": rc, "died_at_named_check": at_named,
                     "must_go_red": want, "died_at": died_at,
                     "attribution": "the AssertionError line on STDERR only; "
                                    "STDOUT (where every PASS is printed) is "
                                    "never searched"}
        if not (rc != 0 and at_named):
            survivors.append(name)
    return {"n_cases": len(cases), "survivors": sorted(survivors),
            "n_survivors": len(survivors), "baseline_green": True,
            "per_case": per,
            "known_unkillable_in_one_tree": dict(AUDIT_KNOWN_UNKILLABLE),
            "scope": "the round-5/6 items and round 7's two closures, each "
                     "driven against a COPY in a temp tree; a case counts as "
                     "killed only if the mutant fails AT ITS NAMED CHECK, so "
                     "dying for another reason is not a kill"}


def selftest() -> int:
    """Every named refusal, red-first, plus the launch-invariance check."""
    import os, subprocess, tempfile
    checks = 0

    def ok(cond, label):
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1
        print(f"  PASS  {label}")

    # ---- gate 1: the frozen contract, both directions -------------------
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # THE GATE MUST AGREE WITH AN INDEPENDENT COMPUTATION, not merely
        # return something. Written as "accept either answer", this control
        # COULD NOT FAIL — a mutant disabling the drift check survived it.
        # The expectation is now computed here from the same committed
        # artifacts, by a second reading, and the gate must match it.
        _c = json.loads(FS.CANDIDATE.read_text())
        _mp = FS.CANDIDATE.parent / _c["manifest"]
        _m = json.loads(_mp.read_text())
        _ps = _m.get("pin_semantics") or {}
        _df = _ps.get("_default", "reproducibility_anchor")
        _expect_drift = _sha_file(_mp) != _c.get("manifest_sha256")
        for _k, _w in (_m.get("hashes") or {}).items():
            if _ps.get(_k, _df) != "reproducibility_anchor":
                continue
            _f = REPO / _k
            if (_sha_file(_f) if _f.exists() else None) != _w:
                _expect_drift = True
        try:
            ev = assert_frozen_contract()
            ok(not _expect_drift and ev["contract"] == "HOLDS",
               f"§10(1) the gate says HOLDS and an INDEPENDENT re-reading of "
               f"the manifest agrees ({ev['anchors_checked']} anchors)")
        except ForwardDayRefused as e:
            ok(_expect_drift and "reproduction contract does not hold" in str(e),
               f"§10(1) the gate REFUSES and an INDEPENDENT re-reading agrees "
               f"that at least one bound input has moved — measured on the "
               f"real committed pair, so this control fails if the gate stops "
               f"noticing (drift expected: {_expect_drift})")
        # and the gate must REFUSE a non-frozen artifact
        cand = tdp / "c.json"
        cand.write_text(json.dumps({"status": "DRAFT"}))
        try:
            assert_frozen_contract(cand)
            ok(False, "a non-FROZEN candidate must REFUSE")
        except ForwardDayRefused as e:
            ok("not FROZEN" in str(e),
               "§10(1) KNOWN-BAD: a candidate whose status is not FROZEN is "
               "refused before anything is read")
        cand.write_text(json.dumps({"status": "FROZEN",
                                    "manifest": "nope.json"}))
        try:
            assert_frozen_contract(cand)
            ok(False, "a missing manifest must REFUSE")
        except ForwardDayRefused as e:
            ok("is absent" in str(e),
               "§10(1) KNOWN-BAD: a candidate naming a manifest that does not "
               "exist is refused, not treated as unbound")

        # a pair whose manifest binds NOTHING: the gate must refuse the
        # zero-anchor read rather than report a pass over an empty set.
        def _pair(hashes, pin=None, msha=None):
            mm = tdp / "m.json"
            mm.write_text(json.dumps({"hashes": hashes,
                                      "pin_semantics": pin or {}}))
            cc = tdp / "cand.json"
            cc.write_text(json.dumps({
                "status": "FROZEN", "manifest": "m.json",
                "manifest_sha256": msha or _sha_file(mm)}))
            return cc
        try:
            assert_frozen_contract(_pair({}))
            ok(False, "a zero-anchor contract must REFUSE")
        except ForwardDayRefused as e:
            ok("compared ZERO anchors" in str(e),
               "§10(1) KNOWN-BAD: a manifest binding NO anchors is refused — "
               "a gate that reads nothing must not report a pass (R-289)")
        # an anchor that MATCHES passes; the same file changed REFUSES.
        _a = tdp / "anchor_file.py"
        _a.write_text("# v1\n")
        _rel = str(_a)                    # absolute; REPO / abs == abs
        ok(assert_frozen_contract(_pair({_rel: _sha_file(_a)}))["contract"]
           == "HOLDS",
           "§10(1) POSITIVE CONTROL: a contract whose anchor matches disk "
           "HOLDS — the gate discriminates rather than refusing universally")
        _a.write_text("# v2 — moved\n")
        try:
            assert_frozen_contract(_pair({_rel: "0" * 64}))
            ok(False, "a moved anchor must REFUSE")
        except ForwardDayRefused as e:
            ok("does not hold" in str(e),
               "§10(1) KNOWN-BAD: an anchor whose bytes moved REFUSES by name")
        # state_at_build entries must NOT be compared (pin_semantics says so)
        _k = tdp / "keep.py"
        _k.write_text("# kept anchor\n")
        ok(assert_frozen_contract(_pair(
            {_rel: "0" * 64, str(_k): _sha_file(_k)},
            pin={"_default": "reproducibility_anchor",
                 _rel: "state_at_build"}))["anchors_checked"] == 1,
           "§10(1) a state_at_build entry is EXCLUDED from the equality "
           "check, as pin_semantics requires, and the anchor count says how "
           "many were actually compared")

    # ---- gate 2: day closed, and attributed to the scheduled unit -------
    ok(assert_day_closed_and_attributed(
        "20260101", {"day_closed_calendar": True,
                     "write_reason": SCHEDULED_PREFIX + " (X)"})[
           "day_closed_calendar"] is True,
       "gate-2 POSITIVE CONTROL: a closed day written by the scheduled unit "
       "passes")
    for lbl, v, want in (
            ("no verdict at all", {}, "has no day verdict"),
            ("day still open", {"day_closed_calendar": False,
                                "write_reason": SCHEDULED_PREFIX},
             "not closed by calendar"),
            ("unattributed hand run", {"day_closed_calendar": True,
                                       "write_reason": "ran it myself"},
             "not written by the scheduled unit"),
            ("prefix only mentioned, not leading",
             {"day_closed_calendar": True,
              "write_reason": "hand run mentioning " + SCHEDULED_PREFIX},
             "not written by the scheduled unit")):
        try:
            assert_day_closed_and_attributed("20260101", v)
            ok(False, f"gate-2 must REFUSE ({lbl})")
        except ForwardDayRefused as e:
            ok(want in str(e),
               f"gate-2 KNOWN-BAD ({lbl}): refused by name — the prefix is "
               f"IMPORTED from DA's preflight and matched as a PREFIX, so a "
               f"mention cannot pass as an attribution")

    # ---- gate 3: the ledger is read, and an empty read REFUSES ----------
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "m.jsonl"
        empty.write_text('{"coin":"btc","window_start":1}\n')
        try:
            present_from_ledger("20260901", empty)
            ok(False, "an out-of-day ledger must REFUSE")
        except ForwardDayRefused as e:
            ok("is not an empty day" in str(e),
               "gate-3 KNOWN-BAD: a ledger with no window for the day REFUSES "
               "— an empty present is a read that found nothing, and scoring "
               "off it would score nothing and say it succeeded")
    real = present_from_ledger("20260901")
    ok(real and all(len(v) > 0 for v in real.values()),
       f"gate-3 POSITIVE CONTROL: 09-01's present is read from the day's own "
       f"ledger — {len(real)} coins, {sum(len(v) for v in real.values())} "
       f"windows (a fact, not a grid)")

    # ---- window-set agreement, both directions -------------------------
    _specs = [{"slug": "a"}, {"slug": "b"}]
    ok(assert_window_sets_agree(_specs, {"rows": [{"slug": "a"}]})[
           "bridged_without_rows"] == 1,
       "agreement: a bridged window with NO rows is COUNTED, never dropped "
       "from the denominator")
    try:
        assert_window_sets_agree(_specs, {"rows": [{"slug": "zzz"}]})
        ok(False, "a row outside the bridge must REFUSE")
    except ForwardDayRefused as e:
        ok("never supplied" in str(e),
           "agreement KNOWN-BAD: a row from a window the bridge never "
           "supplied REFUSES — it is a window nobody admitted (R-418)")

    # ---- ROUND 4: the frozen bytes, verified BEFORE import --------------
    # THE IMPORT CONTROLS MUTATE PROCESS STATE (sys.path and sys.modules) and
    # must put it back: leaving the FROZEN anchors imported from a tmpdir that
    # is then deleted poisons every control after this one. Measured — the
    # selection control below started refusing on a stale frozen module.
    import tempfile as _tf
    _sp_saved = list(sys.path)
    _mods_saved = dict(sys.modules)

    def _restore_imports(root=None):
        """Undo the run-dir imports WITHOUT evicting the tree's modules.

        BE6-R6: evicting every name absent from the snapshot threw out
        packages that had nothing to do with the tmpdir — measured, numpy was
        first imported after the snapshot and its reload guard fired at the
        re-import below. Nothing here carries a numpy object across the
        boundary today, so it was latent; but the isolation this exists for
        is the ANCHORS imported from the run dir, and that is what it now
        removes: a module whose `__file__` lies under the run dir."""
        sys.path[:] = _sp_saved
        _tree = str(Path(__file__).resolve().parents[2])
        for _k in list(sys.modules):
            if _k in _mods_saved:
                continue
            _f = str(getattr(sys.modules.get(_k), "__file__", None) or "")
            # Evict what the run dir brought in AND the TREE's own modules --
            # the latter because one of them caches an archive index built
            # while the run dir's data root was in effect, and keeping it
            # empties the index two checks later (measured). THIRD-PARTY
            # packages are left alone: that is BE6-R6's point, and numpy was
            # the one being reloaded for no reason.
            if _f.startswith(_tree) or (root and _f.startswith(str(root))):
                sys.modules.pop(_k, None)
        for _k, _v in _mods_saved.items():
            sys.modules[_k] = _v

    with _tf.TemporaryDirectory() as td4:
        o4 = Path(td4)
        mat = materialise_frozen(o4)
        ok(mat["frozen_commit"] == FROZEN_COMMIT and mat["n_compared"] >= 7,
           f"R-421(2) POSITIVE CONTROL: every reproducibility_anchor is "
           f"materialised from the freeze commit and sha-checked against the "
           f"manifest BEFORE import ({mat['n_compared']} compared)")
        _cm = [k for k, v in mat["anchors"].items() if not v.get("compared")]
        ok(_cm and all("state_at_build" in str(mat["anchors"][k]) for k in _cm),
           f"R-421(2) collector_runs.jsonl stays state_at_build and is NOT "
           f"compared, as pin_semantics requires (not compared: {_cm})")
        ok(mat["data_root"]["symlink_to"].endswith("/data"),
           "R-421(2) the run dir mirrors the repo's DATA root by symlink — "
           "frozen code, today's data, each named")
        # A TAMPERED MATERIALISED BYTE MUST REFUSE, AND BEFORE IMPORT.
        _py = [k for k, v in mat["anchors"].items()
               if v.get("materialised_to")][0]
        _t = Path(mat["anchors"][_py]["materialised_to"])
        _t.write_bytes(_t.read_bytes() + b"\n# tampered\n")
        _stems = [Path(k).stem for k in mat["anchors"]
                  if mat["anchors"][k].get("materialised_to")]
        # BE10-R3. What stood here was a control that COULD NOT FAIL beside
        # a claim that is FALSE:
        #     try: import…; ok(False, "must not import cleanly")
        #     except (ForwardDayRefused, Exception): ok(True, "…")
        # `ok(False, …)` raises AssertionError; AssertionError IS an
        # Exception; so the handler caught the control's OWN RED and printed
        # a PASS for a refusal that never happened. MEASURED at these bytes:
        # the tampered file imports CLEANLY — appending a comment is valid
        # Python and `import_frozen_anchors` proves WHERE a module came from
        # (`__file__` under the run dir) and that the data root resolves; it
        # never reads the bytes. The claim is WITHDRAWN and replaced by what
        # is true and computed: the bytes are bound at MATERIALISATION, and
        # re-materialisation is the step that restores them (next check).
        _tampered_sha = _sha_file(_t)
        _bound = mat["anchors"][_py]["sha256"]
        try:
            import_frozen_anchors(Path(mat["root"]), [k for k in mat["anchors"]
                                                      if k.endswith(".py")])
            _imported_anyway = True
        except ForwardDayRefused:
            _imported_anyway = False
        ok(_tampered_sha != _bound and _imported_anyway,
           f"R-421(2)/BE10-R3 a tampered materialised byte is caught by the "
           f"SHA THE MANIFEST BINDS ({_tampered_sha[:12]} != "
           f"{_bound[:12]}) and NOT by the import — asserted here to SUCCEED "
           f"({_imported_anyway}), because the import proves provenance, "
           f"never content. The control this replaces asserted the opposite "
           f"and passed by catching its own AssertionError in an "
           f"`except Exception`")
        # re-materialising RESTORES and REFUSES if the source moved
        mat2 = materialise_frozen(o4)
        ok(_sha_file(Path(mat2["anchors"][_py]["materialised_to"]))
           == mat2["anchors"][_py]["sha256"],
           "R-421(2) re-materialisation restores the frozen bytes exactly, so "
           "a tampered run dir cannot persist into the next run")

    _np_before = sys.modules.get("numpy")
    _restore_imports(o4)
    ok(Path(getattr(__import__("harmful_exposure_rows"), "__file__", "")
            ).parent == Path(__file__).parent,
       "R-421(2) the import controls RESTORE sys.path and sys.modules — a "
       "suite that leaves frozen modules imported from a deleted tmpdir "
       "poisons every check after it")
    # BE6-R6, both directions: the tree's modules SURVIVE the restore, and
    # the run dir's do not. Evicting by absence-from-snapshot threw out numpy
    # -- first imported after the snapshot -- and its reload guard fired on
    # the re-import above.
    ok(_np_before is not None and sys.modules.get("numpy") is _np_before,
       "BE6-R6 the restore leaves the TREE's modules alone: numpy is the "
       "same object afterwards, so a C-extension package first imported "
       "after the snapshot is no longer evicted and reloaded")
    ok(not [k for k, m in sys.modules.items()
            if str(getattr(m, "__file__", "") or "").startswith(str(o4))],
       f"BE6-R6 and NOTHING imported from the run dir survives it — the "
       f"isolation this exists for is the anchors, and that is exactly what "
       f"it removes")

    # a CODE anchor absent from the freeze commit REFUSES
    _real_blob = _git_blob
    try:
        globals()["_git_blob"] = lambda ref, path: (
            None if path.endswith(".py") else _real_blob(ref, path))
        with _tf.TemporaryDirectory() as td5:
            try:
                materialise_frozen(Path(td5))
                ok(False, "a code anchor absent from the freeze must REFUSE")
            except ForwardDayRefused as e:
                ok("absent from the freeze commit" in str(e),
                   "R-421(2) KNOWN-BAD: a CODE anchor absent from the freeze "
                   "commit REFUSES — a freeze that does not contain the code "
                   "it binds cannot be executed as frozen")
    finally:
        globals()["_git_blob"] = _real_blob

    # ---- CO-5: the ratification PAIR, and R-418 as superseded -----------
    _pop = population("20260901")
    _rat = _pop["ratification"]
    ok(_rat.get("verified") and _rat.get("unverifiable") == [],
       f"CO-5 POSITIVE CONTROL: {RATIFICATION_REF} verifies as a PAIR — "
       f"verified={_rat.get('verified')}, unverifiable="
       f"{_rat.get('unverifiable')}")
    try:
        _r418 = RAT.check(_pop["supply"], "R-418")
        ok(not _r418.get("verified"),
           f"CO-5 KNOWN-BAD: the superseded R-418 stamp does NOT verify "
           f"({str(_r418)[:150]})")
    except RAT.RatificationRefused as e:
        ok("SUPERSEDED by R-419" in str(e),
           f"CO-5 KNOWN-BAD: the superseded R-418 stamp REFUSES for a NEW "
           f"run, quoting the checker's OWN emission: "
           f"{str(e)[:120]!r}")
    # G8/G9: the PAIR predicate and the CALL, both driven. A control reading
    # `population()`'s output cannot tell a real check from a literal dict.
    _pv = assert_ratification_pair({"verified": True, "unverifiable": [],
                                    "ratification_ref": "R-419"})
    ok(_pv["asserted"] is True and _pv["verified_seen"] is True
       and _pv["unverifiable_seen"] == [] and _pv["ref_seen"] == "R-419",
       "CO-5 POSITIVE CONTROL: the assertion returns WHAT IT SAW, so the "
       "receipt's evidence reproduces the checker's own values -- a bare flag "
       "could be forged by the edit that bypasses the call")
    for _lbl, _r in (("verified with an unbindable field",
                      {"verified": True, "unverifiable": ["day_in_scope"]}),
                     ("not verified", {"verified": False, "unverifiable": []}),
                     ("neither", {"verified": False,
                                  "unverifiable": ["x"]})):
        try:
            assert_ratification_pair(_r)
            ok(False, f"CO-5 must REFUSE ({_lbl})")
        except ForwardDayRefused as e:
            ok("did not verify as a PAIR" in str(e),
               f"CO-5 KNOWN-BAD ({_lbl}): REFUSED — `verified` alone reads an "
               f"unbindable field as a pass, which is the defect CO-5 names")
    # G8b: the EVIDENCE lives in `run_forward_day`, which the suite cannot
    # drive (26 minutes of replay), so a mutant bypassing the call survived.
    # `population()` is cheap and is already called here, so the shape is
    # asserted where it is produced.
    _pa = _rat.get("pair_asserted")
    ok(isinstance(_pa, dict) and _pa.get("asserted") is True
       and _pa.get("verified_seen") == _rat.get("verified")
       and _pa.get("unverifiable_seen") == list(_rat.get("unverifiable") or [])
       and _pa.get("ref_seen") == _rat.get("ratification_ref"),
       f"CO-5 `population()` records WHAT the pair assertion saw, reproducing "
       f"the checker's own values — a bare True (which is what bypassing the "
       f"call leaves behind) does not satisfy this (got {type(_pa).__name__})")
    ok(set(("ratification_ref", "bound_fields", "checks", "protocol"))
       <= set(_rat),
       "CO-5 the block in the receipt is the CHECKER's own emission (it "
       "carries ratification_ref/bound_fields/checks/protocol), not a literal "
       "the driver could have written")

    # ---- G1/G4/G5/G6/G7: the frozen-byte gates, driven -------------------
    with _tf.TemporaryDirectory() as td6:
        o6 = Path(td6)
        _real = _git_blob
        try:
            globals()["_git_blob"] = lambda ref, path: (
                (_real(ref, path) or b"") + b"\n# altered\n"
                if path.endswith(".py") else _real(ref, path))
            try:
                materialise_frozen(o6)
                ok(False, "an altered frozen blob must REFUSE")
            except ForwardDayRefused as e:
                ok("hashes" in str(e) and "manifest binds" in str(e),
                   "R-421(2) KNOWN-BAD: a blob whose sha does not match the "
                   "manifest REFUSES at materialisation — BEFORE import, "
                   "because a byte verified after import has already run")
        finally:
            globals()["_git_blob"] = _real
        m6 = materialise_frozen(o6)
        _data = [k for k, v in m6["anchors"].items()
                 if v.get("compared") and not k.endswith(".py")]
        ok(_data and all(m6["anchors"][k]["materialised_to"] is None
                         for k in _data),
           f"R-421(2) a DATA anchor is verified by content and NOT copied "
           f"into the run dir — copying it created a real frozen/data/ that "
           f"SHADOWED the symlink and the archive index came back empty "
           f"({_data})")
        _cl = import_closure(Path(m6["root"]),
                             [k for k in m6["anchors"] if k.endswith(".py")])
        ok(_cl["n_in_closure"] > 10 and _cl["n_not_frozen"] > 0
           and all("sha256_at_HEAD" in x for x in _cl["not_frozen_in_closure"]),
           f"R-421(2) the closure NAMES the modules that run at HEAD — "
           f"{_cl['n_not_frozen']} of {_cl['n_in_closure']}, each with its sha "
           f"at both commits; a closure that named none would let a reader "
           f"assume the whole of it was frozen")
        # BE34-R5: the words must say what is COMPUTED, and the field that
        # says it must EXIST. MEASURED: renaming `closure_method` away
        # survived the audit, because adding a disclosure and asserting
        # nothing about it leaves a reader trusting a key that can vanish.
        ok("STATIC import walk" in _cl.get("closure_method", "")
           and "not what this run observed executing" in _cl["closure_method"]
           and "dynamic import" in _cl["closure_method"]
           and "execute" not in _cl["why"] and "REACHABLE" in _cl["why"],
           f"R-421(2)/BE34-R5 the closure DECLARES its method — a static "
           f"ast walk, naming what it cannot see (a branch not taken is "
           f"still listed; a dynamic import is not listed at all) — and the "
           f"`why` no longer says these modules 'execute' at HEAD, which is "
           f"more than an import walk can know")
        # the data-root probe must REFUSE when the symlink is absent
        _lnk = Path(m6["root"]) / "data"
        _lnk.unlink()
        _sp_save, _mod_save = list(sys.path), dict(sys.modules)
        try:
            import_frozen_anchors(Path(m6["root"]),
                                  [k for k in m6["anchors"]
                                   if k.endswith(".py")])
            ok(False, "a run dir with no data root must REFUSE")
        except ForwardDayRefused as e:
            ok("data root resolves to" in str(e) or "could not probe" in str(e),
               "R-421(2) KNOWN-BAD: without the data-root symlink the frozen "
               "modules index ZERO archive slugs, and an EMPTY index reads as "
               "'no windows' rather than as a broken path — so it REFUSES")
        finally:
            sys.path[:] = _sp_save
            for _k in list(sys.modules):
                if _k not in _mod_save:
                    sys.modules.pop(_k, None)
            sys.modules.update(_mod_save)
        # ...and an anchor MISSING from the run dir refuses on __file__
        # The run dir EXISTS but holds no anchor, so the import succeeds FROM
        # THE TREE — and the `__file__` assertion is the only thing that can
        # notice. Catching a bare Exception passed either way; the refusal
        # type and its message are now required.
        _empty = o6 / "empty_run_dir"
        (_empty / "live/pm_research").mkdir(parents=True, exist_ok=True)
        try:
            import_frozen_anchors(_empty, [k for k in m6["anchors"]
                                           if k.endswith(".py")][:1])
            ok(False, "an anchor imported from the TREE must REFUSE")
        except ForwardDayRefused as e:
            ok("NOT under the frozen run dir" in str(e),
               "R-421(2) KNOWN-BAD: with the run dir empty the import "
               "succeeds FROM THE TREE, and `__file__` is what catches it — "
               "the tree's copy can never stand in for the freeze's")
        finally:
            sys.path[:] = _sp_save
            for _k in list(sys.modules):
                if _k not in _mod_save:
                    sys.modules.pop(_k, None)
            sys.modules.update(_mod_save)

        # G11: a manifest whose every entry is state_at_build compares ZERO
        # anchors; materialisation must refuse rather than report a pass.
        _m0 = o6 / "m0.json"
        _m0.write_text(json.dumps({
            "hashes": {"data/x": "0" * 64},
            "pin_semantics": {"_default": "state_at_build"}}))
        _c0 = o6 / "c0.json"
        _c0.write_text(json.dumps({"status": "FROZEN", "manifest": "m0.json",
                                   "manifest_sha256": _sha_file(_m0)}))
        _sv = FS.CANDIDATE
        try:
            FS.CANDIDATE = _c0
            materialise_frozen(o6 / "zero")
            ok(False, "a zero-anchor materialisation must REFUSE")
        except ForwardDayRefused as e:
            ok("materialisation compared ZERO anchors" in str(e),
               "R-421(2) KNOWN-BAD: a manifest binding NO anchors REFUSES at "
               "materialisation — a step that verified nothing must not "
               "report a pass (R-289)")
        finally:
            FS.CANDIDATE = _sv

    # ---- ROUND 5 (1): a receipt is EVIDENCE, never a scratch buffer -----
    # MEASURED: a second run into an outdir that already held a receipt
    # OVERWROTE it, and the two runs differed in exactly the fields a reader
    # would have compared.
    with _tf.TemporaryDirectory() as td7:
        o7 = Path(td7)
        r1 = {"protocol": "X", "day": "20260101", "n": 1}
        p1 = _flush(dict(r1), o7, "20260101")
        first = p1.read_bytes()
        ok(p1.name == "be_forward_day_receipt_20260101.json",
           "R5(1) the first run of a day claims the canonical receipt name")
        r2 = {"protocol": "X", "day": "20260101", "n": 2}
        p2 = _flush(r2, o7, "20260101")
        ok(p2 != p1 and p2.exists(),
           f"R5(1) KNOWN-BAD: a SECOND run into the same outdir takes a "
           f"NUMBERED SUCCESSOR ({p2.name}) instead of overwriting")
        ok(p1.read_bytes() == first,
           "R5(1) and the FIRST receipt is byte-identical afterwards — "
           "overwriting evidence loses the comparison a reader needs")
        # BE5-R1 driven: THREE runs -> base / .1 / .2, and the chain must
        # read .2 -> .1 -> base. Recording the base from both successors made
        # the graph a star; with same-second runs the receipts are otherwise
        # byte-identical, so the record IS the only order there is.
        r4 = {"protocol": "X", "day": "20260101", "n": 4}
        p4 = _flush(r4, o7, "20260101")
        ok(p4.name == "be_forward_day_receipt_20260101.2.json"
           and r4["supersedes_receipt"]["path"] == str(p2)
           and r4["supersedes_receipt"]["sha256"] == _sha_file(p2)
           and r4["supersedes_receipt"]["is_base"] is False
           and r2["supersedes_receipt"]["path"] == str(p1)
           and r2["supersedes_receipt"]["is_base"] is True,
           f"BE5-R1 the chain is a CHAIN: `.2` names `.1` and `.1` names the "
           f"base ({p4.name} -> {Path(r4['supersedes_receipt']['path']).name} "
           f"-> {Path(r2['supersedes_receipt']['path']).name}) — naming the "
           f"base from every successor made two receipts claim the same "
           f"predecessor and left the order readable only from `ls`")
        _chain = r4.get("prior_receipts") or []
        ok([Path(x["path"]).name for x in _chain] == [p1.name, p2.name]
           and all(x["sha256"] == _sha_file(Path(x["path"]))
                   for x in _chain)
           and r4["supersedes_receipt"]["n_prior"] == len(_chain),
           f"BE5-R1 and the whole chain is carried beside it "
           f"({[Path(x['path']).name for x in _chain]}), each "
           f"with its sha, so a reader reconstructs the order from the "
           f"receipts and never from the filenames")
        ok(p1.read_bytes() == first and p2.exists()
           and json.loads(p2.read_text())["n"] == 2,
           "BE5-R1 and every earlier receipt is still byte-identical after "
           "the third run — the chain is recorded by ADDING, never by "
           "touching what already stands")
        ok(r2["supersedes_receipt"]["sha256"] == _sha_file(p1),
           "R5(1) the successor NAMES the receipt it did not overwrite, by "
           "path and sha")
        # ...and the SAME run's later flushes still update ITS OWN file, or
        # the durable per-gate receipt would fan out into a numbered pile.
        r2["n"] = 3
        p3 = _flush(r2, o7, "20260101")
        ok(p3 == p2 and json.loads(p3.read_text())["n"] == 3,
           "R5(1) POSITIVE CONTROL: later flushes of the SAME run update its "
           "own file — the per-gate durability guarantee is preserved")

    # ---- ROUND 5 (3): require_verified is the GATE ----------------------
    for _lbl, _bad in (
            ("unverifiable remains", {"verified": True,
                                      "unverifiable": ["x"],
                                      "ratification_ref": "R-419"}),
            ("not verified", {"verified": False, "unverifiable": [],
                              "checks": {"a": False},
                              "ratification_ref": "R-419"}),
            ("a PROVENANCE result", {"verified": True, "unverifiable": [],
                                     "provenance": {"stamped_at": "x"},
                                     "ratification_ref": "R-418"})):
        try:
            RAT.require_verified(_bad)
            ok(False, f"R5(3) require_verified must REFUSE ({_lbl})")
        except Exception as e:                       # noqa: BLE001
            ok("REFUSED" in str(e),
               f"R5(3) KNOWN-BAD ({_lbl}): the CHECKER's own fail-closed gate "
               f"refuses — asserting the pair myself left the provenance "
               f"conjunct unchecked")
    _rvv = _rat.get("require_verified") or {}
    ok(_rvv.get("verified") is True and _rvv.get("unverifiable") == []
       and _rvv.get("provenance_absent") is True
       and _rvv.get("checks_seen", 0) > 0,
       f"R5(3) the receipt records what the checker SAW "
       f"({_rvv.get('checks_seen')} checks) — but the record is evidence, "
       f"never the gate: every field of it is reproducible from the "
       f"checker's INPUT, which is exactly why the audit killed two "
       f"successive attempts to make it one")

    # ---- ROUND 5 (4): the ledger and the archive must agree -------------
    _arc = archive_windows_for_day("20260901")
    ok(_arc and all(len(v) > 0 for v in _arc.values()),
       f"R5(4) the archive index is read independently of the ledger "
       f"({ {c: len(v) for c, v in _arc.items()} })")
    ok(assert_ledger_matches_archive("20260901", real, _arc)["agree"] is True,
       "R5(4) POSITIVE CONTROL: on 09-01 the ledger and the archive AGREE, "
       "per coin — the check discriminates rather than refusing universally")
    _short = {c: (v[:-1] if c == "btc" else list(v))
              for c, v in real.items()}
    try:
        assert_ledger_matches_archive("20260901", _short, _arc)
        ok(False, "R5(4) a ledger missing a window must REFUSE")
    except ForwardDayRefused as e:
        ok("btc: ledger 287 vs archive 288" in str(e),
           "R5(4) KNOWN-BAD: ONE window removed from the ledger REFUSES, "
           "naming the coin and BOTH counts — the empty-index probe cannot "
           "see this non-empty-but-wrong case")
    _extra = {c: (list(v) + [max(v) + 300] if c == "eth" else list(v))
              for c, v in real.items()}
    try:
        assert_ledger_matches_archive("20260901", _extra, _arc)
        ok(False, "R5(4) a ledger with an extra window must REFUSE")
    except ForwardDayRefused as e:
        ok("eth: ledger 289 vs archive 288" in str(e),
           "R5(4) and the OTHER direction too — a silent union would score a "
           "window the archive never recorded")

    # ---- ROUND 5 (5): R-411 inputs are ESTIMATES, never a decision ------
    _r411 = r411_inputs(_pop["supply"])
    _btc = _r411["per_coin"]["btc"]
    ok(sum(_btc["unmasked_windows_per_utc_hour"].values())
       == _btc["n_unmasked_windows"]
       and _btc["n_present_windows"] - _btc["n_masked_windows"]
       == _btc["n_unmasked_windows"],
       f"R5(5) the R-411 inputs are ARITHMETICALLY CLOSED: the per-hour "
       f"counts sum to the unmasked total, and present minus masked equals "
       f"it ({_btc['n_present_windows']} - {_btc['n_masked_windows']} = "
       f"{_btc['n_unmasked_windows']})")
    import da_blackout_mask as _DAM_T
    ok(_btc["calendar_windows_per_day"] == _DAM_T.WINDOWS_PER_DAY
       and _btc["calendar_windows_per_day_source"]
       == "da_blackout_mask.WINDOWS_PER_DAY",
       f"R5(5) the denominator is BOUND to the producer's constant "
       f"(da_blackout_mask.WINDOWS_PER_DAY = {_DAM_T.WINDOWS_PER_DAY}), not "
       f"restated here — a local copy can drift from DA's without either "
       f"side noticing")
    _hrs = _btc["unmasked_windows_per_utc_hour"]
    _per_h = _DAM_T.WINDOWS_PER_DAY // 24
    ok(len(_hrs) == _btc["n_hours_with_any_unmasked_window"]
       and all(k.isdigit() and len(k) == 2 and 0 <= int(k) <= 23
               for k in _hrs)
       and all(1 <= v <= _per_h for v in _hrs.values()),
       f"R5(5) the breakdown is genuinely PER UTC HOUR — {len(_hrs)} "
       f"two-digit hour keys, each holding 1..{_per_h} windows; collapsing it "
       f"to one total still satisfies a sum check, so the sum alone is not "
       f"evidence of a breakdown")
    ok(_btc["n_hours_with_any_unmasked_window"] <= 24
       and _btc["calendar_windows_per_day"] == _DAM_T.WINDOWS_PER_DAY,
       "R5(5) and it carries R-411(ii)'s denominator (hours with any unmasked "
       "window) beside R-411(i)'s (288 calendar windows)")
    ok(not (set(_walk_keys(_r411)) & {v.lower() for v in _decision_vocab()}),
       "R5(5) nothing decision-shaped is in the R-411 block — it supplies the "
       "INPUTS and the policy layer computes G (rule 14)")
    ok(assert_no_decision_field({"a": {"b": 1}})["checked_keys"] >= 2,
       "R5(5) POSITIVE CONTROL: a clean emission passes the post-condition")
    # MEASURED (mutant H10): exercising the function alone let the post-
    # condition be DELETED FROM `_flush` and survive. The falsifier drives the
    # real write path, so it fails if the emission stops being checked.
    with _tf.TemporaryDirectory() as td8:
        o8 = Path(td8)
        for _k in ("counts_toward_G", "eligible", "admissible", "verdict",
                   "qualifies"):
            try:
                _flush({"protocol": "X", "day": "20260101",
                        "r411_inputs": {"per_coin": {"btc": {_k: True}}}},
                       o8, "20260101")
                ok(False, f"R5(5) a decision-shaped key ({_k}) must REFUSE")
            except ForwardDayRefused as e:
                ok("decision-shaped field" in str(e),
                   f"R5(5) KNOWN-BAD: a receipt carrying `{_k}` REFUSES AT "
                   f"THE WRITE — nested arbitrarily deep, and checked where "
                   f"the emission actually leaves the driver (rule 14)")
        ok(not list(o8.iterdir()),
           "R5(5) and NOTHING was written — a receipt that refuses must not "
           "leave a partial file behind for a reader to find")
        # THE CONTROL THAT WAS MISSING. Fixtures said the post-condition
        # worked; the REAL receipt carries `gates[].gate` and would have
        # refused every run, so this check meets a real emission.
        #
        # BE12-S1 -- FOUND BY EXECUTING IT, NOT BY READING IT. This ran
        # `run_forward_day("20260902")` under the comment "09-02 refuses at
        # gate 1 in ~0 s, so a real emission is affordable here". That was a
        # property of the CALENDAR, not of the code. At 00:06Z on 09-03 the
        # scheduled unit wrote the 09-02 governed verdict; gate 1
        # (`day_closed_and_attributed`) began to PASS; and this control
        # started performing a FULL CLOSED-DAY SCORING RUN of 09-02 inside
        # the selftest -- the run R-486 (6) reserves for the USER. MEASURED
        # before it was killed: 14 min of full-pipeline execution, ~16 GB
        # read, in a batch whose dispatch forbids exactly that.
        #
        # The day is now one no later event can make scorable, and the
        # control PROVES that BEFORE it calls the driver: if the day it
        # names could be scored, it refuses here rather than scoring it.
        _day9 = "21000101"
        _v9 = DERIVED / f"da_dayverdict_{_day9}.json"
        ok(not _v9.exists(),
           f"BE12-S1 the real-emission control names a day that CANNOT "
           f"become scorable ({_day9}: no day verdict at {_v9.name}) — "
           f"checked BEFORE the driver is called, so this control can never "
           f"turn into a scoring run when a day closes. Its predecessor "
           f"named 09-02 and did exactly that once DA's unit attributed the "
           f"day: the refusal it asserted was the CALENDAR's, not the code's")
        _rc9 = run_forward_day(_day9, o8)
        _r9 = json.loads(receipt_path(o8, _day9).read_text())
        # R23: this asserted `gates[0]`, and inserting the `user_admission`
        # gate ahead of it turned a control about WHICH GATE REFUSED into a
        # control about POSITION -- it went red the moment a passing gate was
        # added in front, which is the right failure but the wrong reason.
        # It now finds the refusing gate BY NAME, so a later insertion cannot
        # move it and cannot silently satisfy it either.
        _ref9 = [g for g in _r9["gates"] if g["result"] == "REFUSED"]
        _pass9 = [g["gate"] for g in _r9["gates"] if g["result"] == "PASS"]
        ok(_rc9 != 0 and len(_ref9) == 1
           and _ref9[0]["gate"] == "day_closed_and_attributed"
           and isinstance(_ref9[0]["gate"], str)
           and _pass9 == ["user_admission"],
           f"R5(5) POSITIVE CONTROL ON A REAL EMISSION: the {_day9} run "
           f"refuses at `{_ref9[0]['gate']}` — located BY NAME, with "
           f"{_pass9} passing ahead of it — and its receipt WRITES; a "
           f"fixture-only control let the post-condition refuse every real "
           f"receipt without the suite noticing (rule 17)")
        ok(_r9["user_admission"] == {
               "no_admission_covers_this_day": True,
               "note": "the ordinary gate applies, unchanged"},
           f"R23 ON A REAL EMISSION: {_day9} carries NO admission, so the "
           f"receipt says so in words and the ordinary gate is what refused "
           f"it — the admission is one day's bytes, not a mode this driver "
           f"is now in")
        ok(_r9["decision_field_check"]["excused_paths"] == ["gates[].gate"],
           "R5(5) and exactly ONE path is excused, named in the receipt with "
           "its reason — an exemption a reader cannot see is not one")
    # ---- BE34-R1: the streaming pass EQUALS the reference, per score -----
    _sel, _rows = _r1_windows()
    with _r1_installed(_rows):
        _bs = build_and_score(_sel, _R1_FROZEN, inc_fits=_R1_INC)
        _sr = score_rows(_rows)
    ok(_bs["scores"] == _sr,
       f"BE34-R1 ONE fixture, TWO consumers: `build_and_score`'s streamed "
       f"scores are EQUAL to `score_rows`'s on the same windows "
       f"({ {c: len(v) for c, v in _sr.items()} }) — the streaming rewrite "
       f"replaced the reference and nothing had ever compared them")
    ok(sum(len(v) for v in _sr.values()) == 11 and _bs["n_rows"] == 18
       and _bs["rows_without_features"] == 1 and "sol" not in _bs["scores"]
       and "sol" not in _sr,
       f"BE34-R1 and BOTH drop the SAME rows: 18 rows in, 11 scored — 6 for "
       f"a coin with NO frozen fit (BE6-R7's dominant class) and 1 "
       f"without features (got {_bs['n_rows']}/"
       f"{sum(len(v) for v in _sr.values())}/"
       f"{_bs['rows_without_features']}) — an equality between two consumers "
       f"that both dropped everything would also be an equality")
    ok(_bs["n_windows"] == 6 and _bs["n_actions"] == 18
       and _bs["n_windows_with_rows"] == 6,
       f"BE34-R1 the counters the receipt publishes come from the same pass: "
       f"6 windows, 18 actions (got {_bs['n_windows']}/{_bs['n_actions']})")
    _vals = sorted(v for lst in _sr.values() for _, v in lst)
    ok(len(set(_vals)) == len(_vals) and max(abs(v) for v in _vals) < 1e5,
       f"BE34-R1 every fixture score is DISTINCT and small "
       f"(|max| {max(abs(v) for v in _vals):.1f}) — equal-valued or huge "
       f"scores would hide a perturbation instead of catching it")

    # ---- BE34-R1: a reconciliation failure fails the DAY, by name --------
    _selb, _rowsb = _r1_windows()
    with _r1_installed(_rowsb, bad_window="eth-updown-5m-1788000300"):
        _bad = build_and_score(_selb, _R1_FROZEN, inc_fits=_R1_INC)
    _eth_t0 = {t for t, _ in _bad["scores"].get("eth", ())}
    ok(_bad["reconciliation_failures"] == 1 and _bad["n_windows"] == 6
       and 1788000300 not in _eth_t0 and 1788001200 in _eth_t0
       and _bad["reconciliation_failed_windows"] == ["eth-updown-5m-1788000300"]
       and _bad["n_rows"] == 18 and _bad["n_actions"] == 18,
       f"BE34-R1 a window whose fills do not reconcile is COUNTED and ITS "
       f"rows are not scored, while the OTHER window of the same coin still "
       f"is ({_bad['reconciliation_failures']} of {_bad['n_windows']}; eth "
       f"t0s {sorted(_eth_t0)}) — the v3 builder's STRICT condition, and the "
       f"failure is scoped to the window, not spread to the coin")
    with _tf.TemporaryDirectory() as tdr1:
        _real_bas = globals()["build_and_score"]
        globals()["build_and_score"] = (
            lambda sel, fr, feed=None: dict(_bad))
        # MEASURED: driving the real chain IMPORTS THE FROZEN ANCHORS from the
        # run dir into sys.modules, and the run dir is a tmpdir that is about
        # to vanish. Left in place they shadow the tree's modules for every
        # later check -- which is how this control first broke a round-5 one.
        _mods = dict(sys.modules)
        try:
            _rc = run_forward_day("20260901", Path(tdr1))
        finally:
            globals()["build_and_score"] = _real_bas
            for _k in [k for k in sys.modules if k not in _mods]:
                del sys.modules[_k]
            for _k, _v in _mods.items():
                if sys.modules.get(_k) is not _v:
                    sys.modules[_k] = _v
        _recr = json.loads(receipt_path(Path(tdr1), "20260901").read_text())
        _why = _recr.get("refusal", "")
        ok(_rc != 0 and _recr["outcome"] == "REFUSED"
           and "failed reconciliation" in _why and "fails the DAY" in _why
           and "eth-updown-5m-1788000300" in _why
           and _recr["refused_at"] == "reconciliation",
           f"BE34-R1 KNOWN-BAD: the CALLER refuses the whole DAY by name "
           f"(rc={_rc}) — driven through the real gate chain, so disabling "
           f"the refusal is visible here; a mismatch is never absorbed")
        ok(not any(g["gate"] == "reconciliation" and g["result"] == "PASS"
                   for g in _recr["gates"])
           and _recr["gates"][-1]["result"] == "PASS"
           and _recr["gates"][-1]["gate"] != _recr["refused_at"],
           f"BE6-R1 the refusal names the CHECK that refused "
           f"({_recr['refused_at']!r}) and NOT the last gate to pass "
           f"({_recr['gates'][-1]['gate']!r}) — a bare raise sits outside "
           f"`gate()`, so the generic fallback blamed a gate that succeeded, "
           f"and the suite used to pin that as expected")
        # BE6-R3: the zero-score refusal, driven. It is the guard between an
        # empty book and a "scored" day when the frozen fits do not cover the
        # supply, and it had no falsifier.
        _empty = dict(_bad, reconciliation_failures=0,
                      reconciliation_failed_windows=[], scores={},
                      windows_with_rows=set())
        _mods2 = dict(sys.modules)
        globals()["build_and_score"] = lambda sel, fr, feed=None: dict(_empty)
        try:
            _rc0 = run_forward_day("20260901", Path(tdr1))
        finally:
            globals()["build_and_score"] = _real_bas
            for _k in [k for k in sys.modules if k not in _mods2]:
                del sys.modules[_k]
            for _k, _v in _mods2.items():
                if sys.modules.get(_k) is not _v:
                    sys.modules[_k] = _v
        # The SECOND run into this outdir takes a numbered successor, and
        # `.1.json` sorts BEFORE `.json`, so the newest is chosen by write
        # time rather than by name.
        _rp0 = max(Path(tdr1).glob("be_forward_day_receipt_20260901*.json"),
                   key=lambda x: x.stat().st_mtime)
        _rec0 = json.loads(_rp0.read_text())
        ok(_rc0 != 0 and _rec0["outcome"] == "REFUSED"
           and "zero actions scored" in _rec0.get("refusal", "")
           and _rec0["refused_at"] == "zero_actions_scored",
           f"BE6-R3 KNOWN-BAD: a day whose windows all reconcile but score "
           f"NOTHING is REFUSED, not published as an empty book (R-141), and "
           f"the receipt names {_rec0['refused_at']!r} — the check that "
           f"refused, not the gate before it")

    # ---- BE34-R4: a usage error is a refusal, not a success --------------
    # sys.argv is NEUTRALISED for the call, for two reasons. It stops the
    # check depending on how the suite was launched; and MEASURED, an early
    # version of this control re-entered `selftest()` under the mutant that
    # ignores `argv` (sys.argv still held --selftest), so each level re-ran
    # the whole suite AND spawned a child. The falsifier therefore asserts
    # the MESSAGE, not just the code: both paths return 2, and only the text
    # says which argv was read.
    import contextlib as _cl, io as _io
    _sv = list(sys.argv)
    try:
        sys.argv = ["be_forward_day.py"]
        _out = _io.StringIO()
        with _cl.redirect_stdout(_out):
            _rc_usage = main([])
            _rc_day = main(["--forward-day"])
        _txt = _out.getvalue()
    finally:
        sys.argv = _sv
    ok(_rc_usage == 2 and _rc_day == 2,
       f"BE34-R4 a usage error RETURNS 2, the code every other refusal here "
       f"uses (got {_rc_usage}/{_rc_day}) — returning 0 let a misspelled "
       f"flag look to a caller like a day that ran")
    ok("usage: be_forward_day.py" in _txt
       and "needs a day token" in _txt,
       f"BE34-R4 and the message matches the argv PASSED, not the process's "
       f"own — `main` that ignored its parameter would answer both calls "
       f"from sys.argv and print the same text twice")

    # ---- ROUND 5, the AUDIT's own finding: five mutants survived --------
    # Every one of them was an edit that leaves a HEALTHY day's numbers
    # untouched -- the checker not called (its input already satisfies the
    # evidence), the guard disabled (it only fires on a bad day), the
    # denominator restated (the literal equals the constant). A suite that
    # only ever runs a good day cannot kill any of them. Each now has a
    # KNOWN-BAD input driven through the REAL producer (rule 16).
    _real_check = RAT.check
    for _lbl, _res, _typ, _frag in (
            ("unverifiable remains",
             {"verified": True, "unverifiable": ["x"], "checks": {"a": True},
              "ratification_ref": RATIFICATION_REF},
             "NotVerified", "unverifiable checks remain"),
            ("not verified",
             {"verified": False, "unverifiable": [], "checks": {"a": False},
              "ratification_ref": RATIFICATION_REF},
             "NotVerified", "verified is False"),
            ("a PROVENANCE result",
             {"verified": True, "unverifiable": [], "checks": {"a": True},
              "provenance": {"stamped_at": "earlier"},
              "ratification_ref": RATIFICATION_REF},
             "NotVerified", "PROVENANCE")):
        RAT.check = (lambda *_a, _r=_res, **_k: dict(_r))
        try:
            population("20260901", present=real)
            ok(False, f"R5(3) population must REFUSE ({_lbl})")
        except Exception as e:                        # noqa: BLE001
            # WHICH guard fired is the whole question. Accepting any refusal
            # let `require_verified` be deleted and still pass: the pair
            # assertion refuses the first two cases by itself. Only the
            # PROVENANCE conjunct is the checker's alone, and only its
            # exception TYPE distinguishes the checker from my own guard.
            ok(type(e).__name__ == _typ and _frag in str(e),
               f"R5(3) KNOWN-BAD ({_lbl}): REFUSED by {_typ} — "
               f"{'the checker ALONE holds this conjunct, so deleting the '
                  'call is visible here and nowhere else'
                  if _typ == 'NotVerified' else
                  'the recorded pair assertion refuses it'}")
        finally:
            RAT.check = _real_check
    _real_lva = globals()["assert_ledger_matches_archive"]
    for _lbl, _stub in (
            ("an empty per-coin block", {"agree": True, "per_coin": {}}),
            ("a block that does not reproduce the ledger",
             {"agree": True, "per_coin": {"btc": {"n_ledger": 1}}})):
        globals()["assert_ledger_matches_archive"] = (
            lambda *_a, _s=_stub, **_k: dict(_s))
        try:
            population("20260901", present=real)
            ok(False, f"R5(4) population must REFUSE ({_lbl})")
        except ForwardDayRefused as e:
            ok("does not reproduce the ledger" in str(e),
               f"R5(4) KNOWN-BAD ({_lbl}): REFUSED. This guard only fires "
               f"when the cross-count is stubbed, so a good-day suite could "
               f"never kill its removal — it has to be driven")
        finally:
            globals()["assert_ledger_matches_archive"] = _real_lva
    _real_wpd = _DAM_T.WINDOWS_PER_DAY
    try:
        _DAM_T.WINDOWS_PER_DAY = 997
        ok(r411_inputs(_pop["supply"])["per_coin"]["btc"]
           ["calendar_windows_per_day"] == 997,
           "R5(5) KNOWN-BAD: moving the PRODUCER's constant moves the emitted "
           "denominator — a restated `288` equals the constant on every real "
           "day, so only changing it can tell a binding from a copy")
    finally:
        _DAM_T.WINDOWS_PER_DAY = _real_wpd
    ok(r411_inputs(_pop["supply"])["per_coin"]["btc"]
       ["calendar_windows_per_day"] == _real_wpd,
       "R5(5) POSITIVE CONTROL: and it is restored, so the mutation cannot "
       "leak into the checks that follow")

    # the exemption is a PATH, and it cannot be used to smuggle
    for _lbl, _bad, _want in (
            ("a boolean at the excused path",
             {"gates": [{"gate": True}]}, "gates[].gate=True"),
            ("an entitlement INSIDE the excused block",
             {"gates": [{"gate": "g", "eligible": True}]}, "gates[].eligible"),
            ("the same word at another path",
             {"r411_inputs": {"gate": "g"}}, "r411_inputs.gate"),
            ("the word at the top level",
             {"gate": "g"}, "gate")):
        try:
            assert_no_decision_field(_bad)
            ok(False, f"R5(5) must REFUSE ({_lbl})")
        except ForwardDayRefused as e:
            ok(_want in str(e),
               f"R5(5) KNOWN-BAD ({_lbl}): REFUSED naming `{_want}` — the "
               f"allowlist binds to ONE path and requires a string, so it "
               f"excuses the gate's NAME and nothing else")
    # ---- BE9-C3: this selftest does not touch shared repo state ---------
    # `git worktree prune` removes the admin entry of EVERY worktree whose
    # directory is gone -- other sessions' included. A STALE entry is PLANTED
    # HERE, BEFORE the block that creates and removes a worktree, and must
    # still be listed AFTER it. Planted afterwards it survived trivially: a
    # prune inside that block had already run, and the mutant lived.
    import shutil as _sh2
    import subprocess as _sp
    _me_tree = Path(__file__).resolve().parents[2]
    _gcd_r = _sp.run(("git", "rev-parse", "--git-common-dir"),
                     cwd=str(_me_tree), capture_output=True, text=True)
    _gcd = _resolve_git_common_dir(_gcd_r.stdout, _gcd_r.returncode, _me_tree)
    _gcd_refused = 0
    for _bad in (("", 0), ("", 128), ("   \n", 0), (None, 0)):
        try:
            _resolve_git_common_dir(_bad[0], _bad[1], _me_tree)
        except ForwardDayRefused:
            _gcd_refused += 1
    ok(_gcd_refused == 4 and _gcd.is_absolute() and _gcd.exists()
       and _resolve_git_common_dir(".git", 0, _me_tree)
       == (_me_tree / ".git").resolve(),
       f"BE10-R2 KNOWN-BAD: an EMPTY or FAILED `--git-common-dir` REFUSES "
       f"({_gcd_refused}/4 bad forms) instead of resolving to the executing "
       f"tree — which is how the planted stale entry leaked silently, since "
       f"`(tree / '').resolve()` IS the tree and the miss was swallowed by "
       f"`ignore_errors=True`; a real relative answer still resolves "
       f"({_gcd})")
    _std = _tf.mkdtemp()
    _stale = Path(_std) / "be-r10-c3-stale"
    _sa = _sp.run(("git", "worktree", "add", "--detach", str(_stale), "HEAD"),
                  cwd=str(_me_tree), capture_output=True, text=True)
    _sh2.rmtree(_stale, ignore_errors=True)          # now PRUNABLE

    # ---- BE7-R4: the receipt names THE TREE THAT EXECUTED ---------------
    # Driven against a real detached worktree, because in the main tree the
    # hardcoded root and this file's tree are the SAME path and no in-tree
    # comparison can tell them apart -- BE34-R3's lesson, one function over.
    def _git_in(root, *a):
        """Every read below is of the EXECUTING tree, never a fixed one.

        BE9-C3: reading HEAD from `REPO` made a selftest's verdict depend on
        a tree it was not running in.

        BE10-R1: the RETURN CODE is checked. Without it a tree with fewer
        than three commits, or a shallow clone, failed MUTELY two lines
        later as an unexplained `git worktree add` return code. Worse than
        empty: `git rev-parse HEAD~999999` exits 128 having printed the
        LITERAL `HEAD~999999` on stdout, so the old form passed a plausible
        non-commit on as a commit."""
        r = _sp.run(("git", *a), cwd=str(root), capture_output=True,
                    text=True)
        _outv = r.stdout.strip()
        if r.returncode != 0 or not _outv:
            raise ForwardDayRefused(
                f"REFUSED: `git {' '.join(a)}` in {root} exited "
                f"{r.returncode} with stdout {_outv[:40]!r} — this check "
                f"needs three commits of history IN THE EXECUTING TREE, and "
                f"a shallow clone or a young tree cannot supply HEAD~2. "
                f"{r.stderr.strip()[-160:]!r}")
        return _outv

    try:
        _git_in(_me_tree, "rev-parse", "HEAD~999999")
        _r1_refused = False
    except ForwardDayRefused as _e1:
        _r1_refused = ("exited 128" in str(_e1)
                       and "three commits of history" in str(_e1))
    ok(_r1_refused,
       "BE10-R1 KNOWN-BAD: a git read that FAILS refuses BY NAME, naming "
       "the command, the tree and the cause — `rev-parse HEAD~999999` exits "
       "128 having printed the literal ref on stdout, so the old "
       "code-ignoring form returned that string AS a commit and the suite "
       "went red two lines later at `git worktree add` with no cause")
    ok(len(_git_in(_me_tree, "rev-parse", "HEAD")) == 40,
       "BE10-R1 POSITIVE CONTROL: a git read that SUCCEEDS still returns "
       "its 40-char answer, so the refusal above is about the return code "
       "and not about refusing everything")

    _exec_head = _git_in(_me_tree, "rev-parse", "HEAD")
    _prev1 = _git_in(_me_tree, "rev-parse", "HEAD~1")
    # HEAD~2, NOT HEAD~1: at HEAD~1 the commit read FROM the worktree and the
    # commit read as HEAD~1 OF THE EXECUTING TREE are the same string, so a
    # mutant swapping one for the other is an equivalent substitution and the
    # check cannot see it. Measured: that mutant survived the audit.
    _prev = _git_in(_me_tree, "rev-parse", "HEAD~2")
    with _tf.TemporaryDirectory() as _wtd:
        _wt = Path(_wtd) / "wt"
        # BE9-C3: created from the EXECUTING tree, removed by path, and
        # NEVER pruned -- `git worktree prune` drops the admin entry of every
        # worktree whose directory is gone, other sessions' included, which
        # is not a selftest's business.
        _add = _sp.run(("git", "worktree", "add", "--detach", str(_wt), _prev),
                       cwd=str(_me_tree), capture_output=True, text=True)
        try:
            ok(_add.returncode == 0 and _wt.exists(),
               f"BE7-R4 a detached scratch worktree exists to drive this "
               f"against, made from the EXECUTING tree ({_add.returncode})")
            # The worktree's OWN HEAD, read from the worktree. BE9-C1: the
            # old check compared against a commit read from the main tree.
            _wt_head = _git_in(_wt, "rev-parse", "HEAD")
            _pv0 = _provenance(tree=_wt)
            ok(_pv0["carrying_commit"] == _wt_head and _wt_head != _exec_head
               and _pv0["working_tree_dirty"] is False,
               f"BE7-R4 a CLEAN worktree reads clean and names ITS OWN HEAD "
               f"({_wt_head[:12]}), which differs from the executing tree's "
               f"({_exec_head[:12]}) — commit and flag come from the tree "
               f"asked about, not from a fixed root")
            _tgt = _wt / "live/pm_research" / Path(__file__).name
            # BE9-C1: PLANT the difference. Copying the running bytes made
            # this check a function of the BRANCH'S GIT STATE -- it dirtied
            # the worktree only while the running bytes happened to differ
            # from the bytes at that commit, which was true at the moment I
            # wrote it and false from the first commit afterwards. Executed
            # at the tip it failed after 93 checks, in a scratch worktree AND
            # in the main tree. A marker line cannot be a no-op.
            _tgt.write_bytes(Path(__file__).read_bytes()
                             + b"\n# BE7-R4 planted difference\n")
            _pv1 = _provenance(tree=_wt)
            ok(_pv1["working_tree_dirty"] is True
               and _pv1["carrying_commit"] == _wt_head
               and _tgt.read_bytes().endswith(b"planted difference\n"),
               "BE7-R4 and it FLIPS with that tree: a PLANTED byte "
               "difference makes THAT worktree dirty while its commit is "
               "unchanged — the flag tracks the tree, and the difference is "
               "this check's to create, never the branch's to supply")
            _probe = (
                "import sys, json;"
                f"sys.path.insert(0, {str(_tgt.parent)!r});"
                "import be_forward_day as M;"
                "print(json.dumps(M._provenance()))")
            _r = _sp.run((sys.executable, "-c", _probe), cwd=str(_wt),
                         capture_output=True, text=True, timeout=300)
            _pv = json.loads(_r.stdout.strip().splitlines()[-1])
            ok(_pv["carrying_commit"] == _wt_head
               and _pv["carrying_commit"] != _exec_head
               and _pv["provenance_root"] == str(_wt.resolve()),
               f"BE7-R4 a receipt written from a WORKTREE names THAT "
               f"worktree's HEAD ({_pv['carrying_commit'][:12]}) and not the "
               f"executing tree's ({_exec_head[:12]}) — with a hardcoded "
               f"receipt claimed a commit it did not carry while its driver "
               f"sha correctly named the file that ran")
            ok(_pv["working_tree_dirty"] is True
               and _pv["driver_sha256_prefix"] == _sha_file(_tgt)[:16]
               and _pv["driver_sha256_prefix"]
               != _sha_file(Path(__file__).resolve())[:16],
               "BE7-R4 and the DEFAULT derivation is the RUNNING file's "
               "tree: the child was given no `tree`, and its driver sha is "
               "the PLANTED file's — not this one's — so the receipt names "
               "the bytes and the tree that actually ran, which is the case "
               "a hardcoded root gets wrong and one tree cannot distinguish")
            # BE9-C2 driven IN THE WORKTREE: the anchor comparison must
            # read the tree the receipt names. The worktree's copy of an
            # anchor is REPLACED with different bytes; a check rooted at the
            # main tree would not see it, and a check rooted at the executing
            # tree must.
            _anch = "live/pm_research/harmful_hazard_model.py"
            _probe2 = (
                "import sys, json;"
                f"sys.path.insert(0, {str(_tgt.parent)!r});"
                "import be_forward_day as M;"
                "print(json.dumps({'root': str(M.EXEC_TREE()),"
                f"   'anchor': str(M._repo_module_path('harmful_hazard_model')),"
                "    'drift_root': M.anchor_drift_root(),"
                "    'data': M.roots_data_is_absolute()}))")
            _r3 = _sp.run((sys.executable, "-c", _probe2), cwd=str(_wt),
                          capture_output=True, text=True, timeout=300)
            _pv3 = json.loads(_r3.stdout.strip().splitlines()[-1])
            ok(_pv3["root"] == str(_wt.resolve())
               and _pv3["anchor"].startswith(str(_wt.resolve()))
               and _pv3["drift_root"].startswith(str(_wt.resolve()))
               and _pv3["data"] is True,
               f"BE9-C2 code and ANCHORS resolve in the executing tree "
               f"({_pv3['root']}), while `data/` stays absolute by design — "
               f"a bare worktree carries no data/, and the receipt says so "
               f"in `roots` rather than implying a single root")
            _blob = _git_blob(FROZEN_COMMIT,
                              "live/pm_research/harmful_hazard_model.py")
            ok(_blob is not None and len(_blob) > 0,
               "BE7-R4 `_git_blob`'s freeze reads are UNAFFECTED — the object "
               "store is shared, so `git show <ref>:<path>` resolves the same "
               "bytes from any worktree")
        finally:
            # ONLY this worktree, by path, from the executing tree. No prune.
            _sp.run(("git", "worktree", "remove", "--force", str(_wt)),
                    cwd=str(_me_tree), capture_output=True, text=True)
    # ...and it is GONE. A removal that quietly failed would leave this
    # selftest littering the repo it runs in, one entry per run.
    ok(str(_wt) not in _sp.run(
        ("git", "worktree", "list", "--porcelain"), cwd=str(_me_tree),
        capture_output=True, text=True).stdout,
       f"BE9-C3 and the worktree this check made is REMOVED afterwards "
       f"({_wt.name}) — the only git writes this selftest performs are the "
       f"creation and removal of its own")

    # BE9-C3 verified AFTER the worktree block: the planted stale entry is
    # still registered, so nothing in it garbage-collected shared state.
    _listed = _sp.run(("git", "worktree", "list", "--porcelain"),
                      cwd=str(_me_tree), capture_output=True,
                      text=True).stdout
    _sh2.rmtree(_gcd / "worktrees" / _stale.name, ignore_errors=True)
    _sh2.rmtree(_std, ignore_errors=True)
    ok(_sa.returncode == 0 and str(_stale) in _listed,
       "BE9-C3 a STALE worktree entry — one whose directory is gone, exactly "
       "what `git worktree prune` collects — SURVIVES this selftest: nothing "
       "here garbage-collects another session's state, and the entry is "
       "planted BEFORE the worktree block so the claim can fail")

    # BE9-C2: the audit's copy links the EXECUTING tree's entries, so a
    # child reads what this run reads. Checked at the link's TARGET, since a
    # fixed root produces links that resolve elsewhere — invisible to any
    # assertion about EXEC_TREE() alone.
    with _tf.TemporaryDirectory() as _atd:
        _amod = _audit_tree(Path(__file__).read_text(), Path(_atd))
        _lnk = [x for x in Path(_atd).iterdir()
                if x.is_symlink() and x.name not in ("data",)]
        ok(_lnk and all(str(Path(x).readlink()).startswith(str(EXEC_TREE()))
                        for x in _lnk)
           and str(Path(Path(_atd) / "data").readlink()).startswith(str(REPO)),
           f"BE9-C2 the audit copy links the EXECUTING tree's entries "
           f"({len(_lnk)} of them) while `data` is linked from the absolute "
           f"root — a child of the audit therefore reads what this run "
           f"reads, and the one deliberate exception is the one published")

    # ---- BE6-R2: the parity predicate is an IDENTITY, driven -------------
    ok(_launch_parity(0, 5, 5, "aa", "aa") is True
       and _launch_parity(0, 5, 5, "aa", "bb") is False,
       "BE6-R2 the launch parity compares the SHA of the file that ran — a "
       "byte-different tree with the same count is refused, which a count "
       "alone let through")
    ok(_launch_parity(0, 5, 5, None, "aa") is False
       and _launch_parity(0, 5, 6, "aa", "aa") is False
       and _launch_parity(1, 5, 5, "aa", "aa") is False,
       "BE6-R2 and a child that prints NO sha (the stub tree) is refused by "
       "the same predicate, with the count kept as the second conjunct")

    # ---- CO-12: the attribution predicate, driven both directions -------
    # NOT gated behind BE_FORWARD_AUDIT, so these run inside the audit's own
    # children too -- which is what lets a mutation of the predicate be
    # caught by the shipped audit rather than only by a reader.
    _pass_txt = ("  PASS  BE34-R4 a usage error RETURNS 2, the code every "
                 "other refusal here uses\n")
    _died_txt = ('Traceback (most recent call last):\n  File "x", line 1\n'
                 "AssertionError: R5(1) KNOWN-BAD: a SECOND run into the "
                 "same outdir takes a NUMBERED SUCCESSOR\n")
    ok(_audit_attributed("R5(1) KNOWN-BAD: a SECOND run", _died_txt) is True
       and _audit_failure_line(_died_txt).startswith("R5(1) KNOWN-BAD"),
       "CO-12 the attribution reads the label the child DIED on, taken from "
       "the AssertionError line the raise leaves on stderr")
    ok(_audit_attributed("BE34-R4 a usage error RETURNS 2",
                         _pass_txt + _died_txt) is False,
       "CO-12 KNOWN-BAD: a `want` that appears only on a `  PASS  ` line is "
       "NOT attribution — matching the transcript made every case whose "
       "`want` prefixes its own label read as killed whenever that check "
       "merely RAN, which was 7 of the 10")
    # A CHAINED traceback is the real shape when a check fails inside an
    # `except`: Python prints the HANDLED exception first and the one that
    # actually ended the run last. Attributing to the first would name an
    # exception the code recovered from.
    _chain_txt = ('Traceback (most recent call last):\n  File "x", line 1\n'
                  "AssertionError: FIRST label, which was HANDLED\n\n"
                  "During handling of the above exception, another "
                  "exception occurred:\n\n"
                  'Traceback (most recent call last):\n  File "y", line 2\n'
                  "AssertionError: SECOND label, the one it DIED on\n")
    ok(_audit_failure_line(_chain_txt) == "SECOND label, the one it DIED on"
       and _audit_attributed("FIRST label", _chain_txt) is False
       and _audit_attributed("SECOND label", _chain_txt) is True,
       "CO-12 with a CHAINED traceback the attribution takes the LAST "
       "AssertionError — the one the run died on — not the first, which "
       "Python prints for an exception that was handled")
    ok(_audit_attributed("anything", "") is False
       and _audit_failure_line("boom\n") is None,
       "CO-12 and a child that dies WITHOUT an AssertionError — a KeyError, "
       "a refusal, a bare non-zero exit — has no named check to be "
       "attributed to, so it is a survivor and not a kill")

    # BE5-R2: the allowlist's MEMBERSHIP, not merely what an emission used.
    # BE7-R3: `len(...) == 1` is a THIRD pin, not decoration. The set
    # comparison alone passes when the table AND the tuple both grow — which
    # is exactly what a legitimate-looking second exemption does — and the
    # literal refuses that too, so growth takes THREE deliberate edits.
    ok(set(DECISION_ALLOWLIST) == set(DECISION_ALLOWLIST_PINNED)
       and len(DECISION_ALLOWLIST) == 1,
       f"BE5-R2 the excused-path allowlist is PINNED to "
       f"{sorted(DECISION_ALLOWLIST_PINNED)} — `excused_paths` reports what "
       f"THIS emission used, so a second entry nothing happens to hit left "
       f"every check green and the growth reached the artifact unannounced")
    ok(all(k in DECISION_ALLOWLIST and DECISION_ALLOWLIST[k].strip()
           for k in DECISION_ALLOWLIST_PINNED),
       "BE5-R2 and every pinned path carries its REASON in the table the "
       "receipt publishes — a pin with no stated ground is a list, not a "
       "justification")
    ok(assert_no_decision_field(
        {"gates": [{"gate": "g", "result": "PASS"}]})["excused_paths"]
       == ["gates[].gate"],
       "R5(5) POSITIVE CONTROL: the real receipt's own shape passes, and the "
       "excusing is reported rather than silent")

    # ---- F9: the prefix is the IMPORTED object, not a copy of its text --
    import da_governed_verdict_preflight as _PF
    ok(SCHEDULED_PREFIX is _PF.SCHEDULED_PREFIX,
       "gate-2 the scheduled prefix is the IMPORTED object (identity, not "
       "equality) — a restated copy would compare equal today and drift "
       "silently the day DA changes it")

    # ---- F12/F15/F16: the selection is the BRIDGE's, whole ---------------
    _sp = [{"slug": "no-such-slug-12345"}]
    try:
        selected_from_specs(_sp)
        ok(False, "a supplied window with no archive must REFUSE")
    except ForwardDayRefused as e:
        ok("no archive or no token map" in str(e),
           "selection KNOWN-BAD: a supplied window with no archive REFUSES — "
           "R-418 scores the complement WHOLE, so skipping one would "
           "re-select the population the supply already fixed")
    _pop = population("20260901")
    _sel, _selc = selected_from_specs(_pop["specs"][:6])
    ok(_selc["n_selected"] == _selc["n_specs"] == 6
       and [x[0] for x in _sel] == [sp["slug"] for sp in _pop["specs"][:6]],
       f"selection: the tuples are the BRIDGE's slugs in the bridge's order, "
       f"one for one ({_selc}) — not `select_stratified`, which R-418 forbids "
       f"on a race day")
    ok(all(sp["ratification_ref"] == RATIFICATION_REF
           and sp["mask_identity_hash"] == _pop["supply"]["mask_identity_hash"]
           for sp in _pop["specs"]),
       f"bridge: every spec carries ratification_ref {RATIFICATION_REF} and "
       f"the supply's mask_identity_hash, so the receipt names which windows "
       f"it ran over and under what ratification")

    # ---- rows are actions (rule 2) -------------------------------------
    ok(action_count([{"slug": "s", "side": "B", "gen": 1},
                     {"slug": "s", "side": "B", "gen": 1},
                     {"slug": "s", "side": "S", "gen": 1}]) == 2,
       "rule 2: the evaluator de-duplicates rows to ACTIONS "
       "(slug, side, gen), so two rows of one generation count once")

    # ---- sealing: the receipt carries no metric ------------------------
    with tempfile.TemporaryDirectory() as td:
        sealed = seal("20260101", Path(td), {"btc": [(1, 0.5)]},
                      {"n_actions_scored": {"btc": 1}})
        ok(Path(sealed["path"]).exists() and sealed["sha256"],
           "sealing: scores go to the SEALED file and the receipt keeps only "
           "its sha256 (rule 11)")
        body = Path(sealed["path"]).read_text()
        ok("0.5" in body and "0.5" not in json.dumps(sealed),
           "sealing KNOWN-BAD: the score VALUE appears in the sealed file and "
           "NOT in the receipt block — a filing built from the receipt cannot "
           "quote a metric")

    # ---- the driver refuses to write beside canonical artifacts ---------
    ok(str(DERIVED) not in str(Path(".").resolve()),
       "sealing: OUTDIR is a caller parameter and nothing is written under "
       "data/pm_5min/derived/ by this driver")

    # ---- BE8-R1: attribution is a PREFIX, driven both ways -------------
    # The REAL collision, reproduced: the CO-12 control's own label quotes
    # the `died_at` it measured, so its failure line CONTAINS another case's
    # `want`. Under `want in line` a mutant killing that control was
    # credited to case 12.
    _coll = ("Traceback (most recent call last):\n"
             "  File \"be_forward_day.py\", line 1, in <module>\n"
             "AssertionError: CO-12 the attribution HAS a falsifier: ONE "
             "edit, TWO names — ... ('BE34-R4 a usage error RETURNS 2')")
    ok(_audit_attributed("BE34-R4 a usage error RETURNS 2", _coll) is False
       and _audit_attributed("CO-12 the attribution HAS a falsifier",
                             _coll) is True,
       "BE8-R1 attribution is a PREFIX match: a `want` that appears INSIDE "
       "another check's label — the CO-12 control quotes the died_at it "
       "measured — no longer credits that check's death to it, while the "
       "label that OPENS with the want still does. The substring test "
       "matched BOTH, and the error direction was a false KILL")

    # ---- BE8-R2: `ok` is the ONLY incrementer, from this file's AST -----
    _inc = _checks_incrementers(Path(__file__).read_text(encoding="utf-8"))
    _inc_bad = _checks_incrementers(
        "def selftest():\n"
        "    checks = 0\n"
        "    def ok(c, l):\n        checks += 1\n"
        "    return checks\n\n"
        "def _selftest_launch(checks, ok):\n    checks += 1\n"
        "    return checks\n")
    ok(_inc == ["ok"] and _inc_bad == ["_selftest_launch", "ok"],
       f"BE8-R2 `ok` is the ONLY thing that increments `checks` — from this "
       f"file's own AST, attributed to the INNERMOST function ({_inc}) — "
       f"and the KNOWN-BAD source carrying the second incrementer this "
       f"round removed is reported as {_inc_bad}. CO-13 was closed by "
       f"removal; nothing until now said it must STAY closed, and the two "
       f"increments cancelled only by arrangement")

    # ---- BE10-R4: the receipt says whether the executing tree IS REPO ---
    _pv_roots = _provenance()["roots"]
    ok(_exec_tree_is_repo(REPO) is True
       and _exec_tree_is_repo(Path("/tmp")) is False
       and _pv_roots["exec_tree_is_repo"] is _exec_tree_is_repo(EXEC_TREE())
       and _pv_roots["code_and_anchors"] == str(EXEC_TREE())
       and _pv_roots["data"] == str(REPO / "data"),
       f"BE10-R4 `roots` SAYS whether the executing tree IS `REPO` "
       f"(exec_tree_is_repo={_pv_roots['exec_tree_is_repo']}: "
       f"{EXEC_TREE()} vs {REPO}) — from the main tree the two coincide and "
       f"a reader could not tell RULE from COINCIDENCE. Both directions of "
       f"the predicate are driven here, so a constant in it dies in any "
       f"tree rather than only in the one that disagrees")

    _before = checks
    # BE8-R2: the return value is NOT assigned back — `ok` increments the
    # nonlocal directly, and assigning here would restore the cancelling
    # pair this round removed.
    # ---- BE17: THE FEED IS EMITTED, AND THE CHECK FAILS ON THE UNWIRED
    # SOURCE. Rounds 13-16 named this gap; the driver emitted (t0, value)
    # pairs and nothing the action-level estimand could consume. The
    # emission is DRIVEN here, and the wiring predicate is driven against a
    # source with the call removed, so the check can fail.
    import ast as _ast
    import tempfile as _tfd

    def _emits_feed(src: str) -> dict:
        """SCOPED TO THE PRODUCTION FUNCTIONS, not to the file.

        The first version walked the whole module and so counted the
        SELFTEST's own `write_window` call -- the known-bad below stayed green
        with the production call deleted, which is the census-scope defect the
        reviewer found elsewhere in this batch, made here by me."""
        t = _ast.parse(src)
        want = {"build_and_score": "write_window",
                "run_forward_day": "FeedWriter"}
        found = {k: False for k in want}
        manifest = False
        for n in _ast.walk(t):
            if not isinstance(n, _ast.FunctionDef) or n.name not in want:
                continue
            for c in _ast.walk(n):
                if not isinstance(c, _ast.Call):
                    continue
                f = c.func
                if isinstance(f, _ast.Attribute) and f.attr == want[n.name]:
                    found[n.name] = True
                if isinstance(f, _ast.Name) and f.id == want[n.name]:
                    found[n.name] = True
                if (n.name == "run_forward_day" and isinstance(f, _ast.Attribute)
                        and f.attr == "manifest"):
                    manifest = True
        return {"writes_rows_in_build_and_score": found["build_and_score"],
                "constructs_writer_in_run_forward_day": found["run_forward_day"],
                "records_manifest_in_run_forward_day": manifest,
                "emits_feed": all(found.values()) and manifest}

    _src_now = Path(__file__).read_text()
    _now = _emits_feed(_src_now)
    ok(_now["emits_feed"] is True,
       f"BE17 POSITIVE CONTROL: this driver constructs a FeedWriter, writes "
       f"rows through it and records its manifest ({_now})")
    # BE17 (reviewer, LOW): the known-bad falsified only the `write_window`
    # conjunct, so two thirds of `emits_feed` were asserted and not driven.
    # Each conjunct now has its own mutation.
    for _mut, _key, _lab in (
        ("feed.write_window(_feed_rows, _feed_scores)",
         "writes_rows_in_build_and_score", "the row emission"),
        ("_feed = FeedWriter(day, outdir, _latency_of_record())",
         "constructs_writer_in_run_forward_day", "the writer construction"),
        ('rec["feed"] = _feed.manifest()',
         "records_manifest_in_run_forward_day", "the manifest record"),
    ):
        _m = _emits_feed(_src_now.replace(_mut, "pass  # unwired"))
        ok(_m[_key] is False and _m["emits_feed"] is False,
           f"BE17 KNOWN-BAD: removing {_lab} turns `{_key}` AND `emits_feed` "
           f"False -- each of the three conjuncts is driven, not asserted")
    with _tfd.TemporaryDirectory() as _fd:
        _L = _latency_of_record()
        _fr = [{"slug": "btc-updown-5m-1000", "side": "BUY_UP", "gen": 1,
                "t0": 1000, "t_start": 0.5, "any_fill_ahead": True,
                "latency": {str(_L): {"preventable_value_cents": 12.5}}},
               {"slug": "btc-updown-5m-1000", "side": "BUY_UP", "gen": 1,
                "t0": 1000, "t_start": 1.5, "any_fill_ahead": False,
                "latency": {str(_L): {"preventable_value_cents": 99.0}}}]
        with FeedWriter("20260101", Path(_fd), _L) as _w:
            _w.write_window(_fr, [0.9, 0.2])
            _man = None
        _man = _w.manifest()
        ok(_man["n_rows"] == 2 and _man["n_windows"] == 1
           and _man["protocol"] == FEED_PROTOCOL,
           f"BE17 POSITIVE CONTROL: the writer emits one record per scored "
           f"row ({_man['n_rows']}) and its manifest carries counts, a "
           f"sha256 and no value")
        _lines = [json.loads(x) for x in
                  Path(_man["path"]).read_text().splitlines()]
        ok(all(set(r) == set(FEED_FIELDS) for r in _lines),
           f"BE17: every emitted record carries exactly the declared feed "
           f"fields {list(FEED_FIELDS)}")
        ok(_lines[0]["value_cents"] == 12.5 and _lines[1]["value_cents"] == 0.0
           and _lines[1]["any_fill_ahead"] is False,
           "BE17: the latency value is RESOLVED at the row, and a row with no "
           "fill ahead resolves to 0.0 while KEEPING any_fill_ahead false -- "
           "the field `exclusions()` classifies on")
        import be_forward_metric as _FM
        _back = [_FM.feed_row_to_eval_row(r, _L) for r in _lines]
        _ak = _FM.assert_action_keys(_back)
        ok(_ak["n_rows"] == 2 and _ak["n_actions"] == 1,
           f"BE17 THE GAP CLOSED, DRIVEN END TO END: the emitted feed "
           f"re-inflates into rows the estimand's OWN action-key contract "
           f"ADMITS ({_ak['n_rows']} rows, {_ak['n_actions']} action) -- "
           f"which the sealed pair form could never do")
        ok(_FM.sealed_shape_is_unusable(
            {"btc": _lines})["usable_for_action_estimand"] is True,
           "BE17: and the shape checker, which answers False for today's "
           "sealed pairs, answers TRUE for this feed -- the same function, "
           "two inputs, two answers")

    # ---- ROUND 23: THE USER'S ADMISSION OF SUPERSEDED-BUT-GENUINE BYTES --
    def refuses(fn, want, label, exc=ForwardDayRefused):
        """A known-bad must refuse BY NAME. Passing for the wrong reason --
        a NameError, a KeyError -- is not a refusal, so the type is pinned
        and the message must carry `want`."""
        try:
            fn()
        except exc as e:
            ok(want in str(e),
               f"{label} [refused with: {str(e)[:90]}…]")
            return
        except Exception as e:                        # noqa: BLE001
            ok(False, f"{label} [WRONG EXCEPTION {type(e).__name__}: {e}]")
            return
        ok(False, f"{label} [DID NOT REFUSE]")

    _A = USER_ADMISSIONS_BY_DAY["20260829"]
    _adm = admitted_verdict("20260829")
    ok(_adm is not None
       and _adm["record"]["fields_relied_on"]["day_closed_calendar"] is True
       and _adm["record"]["fields_relied_on"]["write_reason"].startswith(
           SCHEDULED_PREFIX)
       and _A["invocation_id"] in _adm["record"]["fields_relied_on"][
           "write_reason"],
       f"R23 POSITIVE CONTROL: the admitted blob VERIFIES AT RUN TIME -- "
       f"{_A['blob_commit']} carries day_closed_calendar True and a genuine "
       f"scheduled-unit write_reason with INVOCATION_ID "
       f"{_A['invocation_id'][:12]}…, checked at the bytes and not taken "
       f"from the dispatch that granted it")
    # R29: this asserted the admission set was EXACTLY ["20260829"], so it
    # went red the moment a second day was legitimately admitted. Membership
    # is the R29 control's job; this one asserts the PROPERTY that makes an
    # admission an admission -- a day without one gets None and the ordinary
    # gate, however many days have one.
    ok(admitted_verdict("20260830") is None
       and admitted_verdict("20260901") is None
       and admitted_verdict("20260902") is None
       and "20260829" in USER_ADMISSIONS_BY_DAY,
       f"R23 THE ADMISSION CANNOT BECOME A BYPASS: it is keyed by DAY, and "
       f"days outside the {len(USER_ADMISSIONS_BY_DAY)}-day admission set -- "
       f"race days included -- get None and the ordinary gate")
    refuses(lambda: assert_day_closed_and_attributed("20260829"),
            "was not written by the scheduled unit",
            "R23 AND THE ORDINARY GATE IS UNCHANGED: called WITHOUT the "
            "admission it still refuses the current verdict, so the "
            "admission widened nothing -- it supplied one day's bytes",
            exc=ForwardDayRefused)
    _raw = _git_blob(_A["blob_commit"], _A["blob_path"])
    refuses(lambda: admission_bytes_ok("20260829",
                                       {**_A, "blob_sha256": "0" * 64}, _raw),
            "not the 0000000000000000 the ruling names",
            "R23 KNOWN-BAD: bytes whose sha is not the one the ruling names "
            "REFUSE -- the USER verified specific bytes, not a path",
            exc=ForwardDayRefused)
    def _mut(**kw):
        v = json.loads(_raw); v.update(kw)
        b = json.dumps(v).encode()
        return {**_A, "blob_sha256": hashlib.sha256(b).hexdigest()}, b
    _a2, _b2 = _mut(day_closed_calendar=False)
    refuses(lambda: admission_bytes_ok("20260829", _a2, _b2),
            "not True. The ruling relies on that field being genuine",
            "R23 KNOWN-BAD, DRIVEN ON FABRICATED BYTES: a verdict that is "
            "NOT closed by calendar refuses even under the admission -- the "
            "ruling admitted attribution, never an open day",
            exc=ForwardDayRefused)
    _a3, _b3 = _mut(write_reason="UNATTRIBUTED hand run")
    refuses(lambda: admission_bytes_ok("20260829", _a3, _b3),
            "does not start with the scheduled prefix",
            "R23 KNOWN-BAD: bytes WITHOUT genuine scheduled-unit attribution "
            "refuse -- the whole ground of the admission is that these "
            "particular bytes are attributed", exc=ForwardDayRefused)
    _a4, _b4 = _mut(write_reason=SCHEDULED_PREFIX + " (INVOCATION_ID=deadbeef)")
    refuses(lambda: admission_bytes_ok("20260829", _a4, _b4),
            "do not carry the INVOCATION_ID",
            "R23 KNOWN-BAD, AND THIS IS THE SUBTLE ONE: a CORRECT prefix "
            "with a DIFFERENT invocation refuses -- a prefix match alone "
            "would admit any other night's scheduled run",
            exc=ForwardDayRefused)
    _prem = driver_reads_no_era_field()
    ok(_prem["premise_holds"] is True
       and _prem["stale_fields_the_driver_reads"] == [],
       f"R23 THE RULING'S PREMISE IS COMPUTED, NOT QUOTED: this driver reads "
       f"none of {_prem['stale_fields_checked']} from a verdict, which is "
       f"the entire reason stale fields in the admitted bytes are harmless")
    ok(driver_reads_no_era_field(
           'def f(v):\n    return v["race_accrual_eligible"]\n'
       )["premise_holds"] is False,
       "R23 KNOWN-BAD: a source that DOES read a stale field turns the "
       "premise False, so the check above can fail -- and `admitted_verdict` "
       "refuses when it does")
    ok(driver_reads_no_era_field(
           '_ADMISSION_STALE = ("race_accrual_eligible",)\n'
       )["premise_holds"] is True,
       "R23 AND THE PREMISE COUNTS READS, NOT MENTIONS: a source that merely "
       "NAMES the field does not break it -- otherwise the admission record, "
       "which must name what it admits, would falsify its own ground")

    # ---- ROUND 29: THE R-503 RE-VERDICT ADMISSION ------------------------
    _A3 = USER_ADMISSIONS_BY_DAY["20260903"]
    _a3 = admitted_verdict("20260903")
    _ev3 = _a3["record"]["rule_change_not_data_change"]
    ok(_a3 is not None and _ev3["data_identical"] is True
       and _a3["record"]["fields_relied_on"]["day_closed_calendar"] is True,
       f"R29 POSITIVE CONTROL: the 09-03 re-verdict VERIFIES at run time -- "
       f"sha {_a3['record']['artifact_sha256'][:16]} matches, the day is "
       f"closed, it accrues, and its write_reason names "
       f"{_A3['ruling_token']}")
    ok(_ev3["rule_derived_fields_that_moved"]
       and all("evaluable" in v for v in
               _ev3["rule_derived_fields_that_moved"].values()),
       f"R29: and something rule-derived ACTUALLY MOVED "
       f"({_ev3['rule_derived_fields_that_moved']}) -- an admission for a "
       f"rule change that changed nothing is an admission for nothing")
    refuses(lambda: assert_day_closed_and_attributed("20260903"),
            "was not written by the scheduled unit",
            "R29 THE ORDINARY GATE IS UNCHANGED: without the admission it "
            "still refuses the re-verdict, because DA's supersede LOST the "
            "scheduled prefix -- this is an admission, not a weakened gate",
            exc=ForwardDayRefused)
    ok(sorted(USER_ADMISSIONS_BY_DAY) == ["20260829", "20260903"]
       and admitted_verdict("20260901") is None
       and admitted_verdict("20260902") is None,
       "R29 STILL KEYED BY DAY: two days carry an admission and every other "
       "day -- 09-01 and 09-02 included -- gets None and the ordinary gate")
    refuses(lambda: reverdict_data_unchanged(
                {**_A3, "verdict_sha256": "0" * 64}),
            "not the 0000000000000000 the admission names",
            "R29 KNOWN-BAD: a re-verdict whose bytes are not the admitted "
            "bytes REFUSES", exc=ForwardDayRefused)
    refuses(lambda: reverdict_data_unchanged(
                {**_A3, "gap_series_fields_allowed_to_differ": ()}),
            "differs from its predecessor in MEASUREMENTS",
            "R29 KNOWN-BAD, AND IT DRIVES THE EXEMPTION ITSELF: with the "
            "`ledger_lines` exemption REMOVED the comparison REFUSES -- so "
            "the exemption is load-bearing and is not a clause nobody tests",
            exc=ForwardDayRefused)
    refuses(lambda: reverdict_data_unchanged(
                {**_A3, "data_fields_that_must_be_identical":
                 tuple(_A3["data_fields_that_must_be_identical"])
                 + ("per_coin",)}),
            "differs from its predecessor in MEASUREMENTS",
            "R29 KNOWN-BAD: widening the must-be-identical set to a field "
            "the RULE legitimately changed (`per_coin`) REFUSES -- the "
            "comparison fires on real difference, not on a fixed list",
            exc=ForwardDayRefused)

    # ---- ROUND 21: the frozen-contract gate is ON THE RUN PATH -----------
    _fc = _emits_feed  # reuse the scoped-AST idiom
    import ast as _ast2

    def _gate_wired(src: str) -> bool:
        t = _ast2.parse(src)
        for n in _ast2.walk(t):
            if isinstance(n, _ast2.FunctionDef) and n.name == "run_forward_day":
                for c in _ast2.walk(n):
                    if (isinstance(c, _ast2.Call)
                            and isinstance(c.func, _ast2.Name)
                            and c.func.id == "gate"
                            and c.args
                            and isinstance(c.args[0], _ast2.Constant)
                            and c.args[0].value == "frozen_contract"):
                        return True
        return False

    _src21 = Path(__file__).read_text()
    ok(_gate_wired(_src21) is True,
       "R21 POSITIVE CONTROL: `frozen_contract` is a REAL gate inside "
       "`run_forward_day` -- the repair round 20 identified and deferred")
    ok(_gate_wired(_src21.replace(
        'rec["frozen_contract"] = gate("frozen_contract", frozen_contract_gate)',
        "pass  # unwired")) is False,
       "R21 KNOWN-BAD: with the gate call removed the same predicate returns "
       "False, so the check above can fail")
    _lb = manifest_keys_read_by_run_path()
    ok(_lb["load_bearing_keys"] == ["hashes", "pin_semantics"],
       f"R21: the load-bearing keys are DERIVED from the code reachable from "
       f"`run_forward_day` ({_lb['load_bearing_keys']}, read by "
       f"{sorted(set(sum(_lb['read_by'].values(), [])))}) -- not listed from "
       f"a dispatch")
    ok("emits_feed" not in _lb["load_bearing_keys"],
       "R21: and the derivation is SCOPED -- an unscoped walk picked up "
       "`emits_feed` out of this selftest, which has nothing to do with the "
       "manifest")
    ok(drift_is_fatal(["as_of_utc"], _lb["load_bearing_keys"]) is False
       and drift_is_fatal(["hashes"], _lb["load_bearing_keys"]) is True,
       "R21: the decision rule is DRIVEN BOTH WAYS -- metadata drift is not "
       "fatal, a `hashes` drift is")
    _md = manifest_drift_detail()
    ok(_md["drifted"] is True and _md["drift_touches_the_run_path"] is False
       and _md["keys_that_differ"] == ["as_of_utc", "git_commit", "git_dirty"],
       f"R21: the LIVE drift is real and characterised, not papered over -- "
       f"{_md['keys_that_differ']}, none load-bearing")
    _g = frozen_contract_gate()
    ok(_g["contract"] == "HOLDS"
       and _g["all_anchors_match_at_freeze_commit"] is True
       and _g["anchors_verified_at_freeze_commit"] > 0,
       f"R21 POSITIVE CONTROL: the gate HOLDS, having verified "
       f"{_g['anchors_verified_at_freeze_commit']} anchors AT THE FREEZE "
       f"COMMIT -- the bytes the run actually executes")
    ok(_g["n_working_tree_drift"] > 0 and _g["disclosed_not_waived"] is True,
       f"R21: and it DISCLOSES {_g['n_working_tree_drift']} working-tree "
       f"anchor drifts by name rather than hiding them "
       f"({[d['anchor'].split('/')[-1] for d in _g['working_tree_drift']]}) -- "
       f"wiring `assert_frozen_contract` verbatim would have refused every "
       f"run on these, which the run does not depend on")
    ok(_g["materialise_frozen_sources_from_the_freeze_commit"] is True,
       "R21: and the REASON tree drift is survivable is asserted from "
       "`materialise_frozen`'s own source, so it cannot quietly stop being "
       "true")
    _ = anchor_drift_root()
    ok(_LAST_CONTRACT_REFUSAL is not None,
       "R21: `anchor_drift_root` now RECORDS the contract's refusal instead "
       "of `except Exception: pass` -- the file's only invocation of the "
       "contract no longer throws its answer away")

    _selftest_launch(checks, ok)
    # BE5-R3: the audit is an ARTIFACT, not a report. Skipped in the audit's
    # own children and in the launch child (both carry BE_FORWARD_AUDIT=1),
    # or every case would re-run the whole sweep inside itself.
    if os.environ.get("BE_FORWARD_AUDIT") != "1":
        _aud = mutation_audit()
        # BE8-R1: the prefix property the attribution now REQUIRES, asserted
        # across the whole table instead of assumed. This is what caught the
        # one shipped `want` that was a substring of its label but not its
        # opening (BE34-R5's, label `R-421(2)/…`) — under `startswith` that
        # case became a survivor, and the substring test had hidden it.
        _applied = [d for d in _aud["per_case"].values() if d.get("applied")]
        _nonpref = [d["must_go_red"] for d in _applied
                    if not (d["died_at"] or "").startswith(d["must_go_red"])]
        ok(_applied and not _nonpref,
           f"BE8-R1 every one of the {len(_applied)} shipped cases died at a "
           f"line that OPENS with its `want` — the prefix property asserted "
           f"over the whole table, not assumed; non-prefix wants: "
           f"{_nonpref}")
        ok(_aud["survivors"] == [] and _aud["baseline_green"]
           and all(v.get("died_at_named_check")
                   for v in _aud["per_case"].values()),
           f"BE5-R3 the shipped mutation audit runs GREEN: {_aud['n_cases']} "
           f"cases, {_aud['n_survivors']} survivors, each mutant dying AT "
           f"ITS NAMED CHECK against a copy — the audit's result is now "
           f"re-runnable by a reader instead of a number in a filing")
        # CO-12 END TO END, both directions, through the REAL audit: the
        # SAME edit, named once wrongly and once rightly. Under the old
        # transcript test both read as killed; the wrong one must now be a
        # SURVIVOR, or the attribution is not attributing.
        _usage = [c for c in AUDIT_CASES if c[0].startswith("BE34-R4")][0]
        _mis = ("CO-12 CONTROL: the usage edit named to a check it does NOT "
                "die at", _usage[1], _usage[2],
                "R5(1) KNOWN-BAD: a SECOND run into the same outdir")
        _hit = ("CO-12 CONTROL: the same edit named to the check it DOES "
                "die at", _usage[1], _usage[2], _usage[3])
        _ctl = mutation_audit(cases=(_mis, _hit))
        ok(_ctl["survivors"] == [_mis[0]]
           and _ctl["per_case"][_hit[0]]["died_at_named_check"] is True
           and _ctl["per_case"][_mis[0]]["died_at"]
           == _ctl["per_case"][_hit[0]]["died_at"],
           f"CO-12 the attribution HAS a falsifier: ONE edit, TWO names — "
           f"named to a check it does not die at it is a SURVIVOR, named to "
           f"the one it does it is KILLED, and both died at the SAME line "
           f"({(_ctl['per_case'][_hit[0]]['died_at'] or '')[:60]!r}). Without "
           f"this, no case was ever driven to a survivor by dying at the "
           f"wrong check")
        # CO-13: `ok` already increments. The extra `checks += 1` that stood
        # here counted this ONE assertion TWICE, so the printed figure was
        # one more than the assertions that ran -- 101 PASS lines under a
        # summary saying 102.
    import os as _os
    if (_os.environ.get("BE_FORWARD_LAUNCH_CHECK") != "1"
            and checks == _before):
        raise AssertionError(
            "the launch-invariance check contributed NO checks, so it did not "
            "run — removing that call is a guard-removal nothing else "
            "notices, which is CO-1's own shape")
    # BE6-R2: the child announces WHICH FILE ran, not just how many checks
    # it counted. A count is not an identity.
    print(f"be_forward_day selftest: {checks} checks OK "
          f"[sha {_sha_file(Path(__file__).resolve())[:16]}]")
    return 0


def _child_count(r) -> int:
    """The count a spawned child printed, or None if it printed none."""
    m = re.search(r"be_forward_day selftest: (\d+) checks OK",
                  r.stdout + r.stderr)
    return int(m.group(1)) if m else None


def _child_sha(r) -> str:
    """The sha of the file the child actually RAN, or None if it said none.

    BE6-R2: the stub tree prints a bare count and no sha, so it is refused by
    the same predicate rather than by a special case."""
    m = re.search(r"be_forward_day selftest: \d+ checks OK \[sha ([0-9a-f]+)\]",
                  r.stdout + r.stderr)
    return m.group(1) if m else None


def _launch_parity(rc: int, child: int, expect: int,
                   child_sha: str = None, expect_sha: str = None) -> bool:
    """ONE predicate for both spawns, so weakening it fails the known-bad.

    BE6-R2: a COUNT is not an identity — a byte-different tree with the SAME
    number of checks passed, and the message claimed the tree while the
    predicate compared only the total (rule 10). The child now prints the sha
    of the file that ran; the shas are compared, with the count kept as the
    second conjunct. The stub tree prints no sha and stays refused."""
    return (rc == 0 and child_sha is not None and child_sha == expect_sha
            and child == expect)


def _selftest_launch(checks: int, ok) -> int:
    """Green under BOTH launchers, asserted rather than assumed."""
    import os, subprocess, tempfile as _tf
    if os.environ.get("BE_FORWARD_LAUNCH_CHECK") == "1":
        return checks
    # The child returns from HERE without adding a check, so its total is
    # exactly the count at this point -- captured, not re-derived from the
    # increment order below.
    at_entry = checks
    # The child skips BOTH the spawn and the shipped audit. Without the
    # second flag the child re-runs the whole audit inside the parent's --
    # doubling the suite -- and adds a check the parent's parity does not
    # expect, so the arithmetic that proves the child ran THIS file breaks.
    env = dict(os.environ, BE_FORWARD_LAUNCH_CHECK="1", BE_FORWARD_AUDIT="1")
    # BE34-R3: the tree of THIS FILE, never a hardcoded root. `cwd=REPO` made
    # every worktree spawn the SHARED tree's module, so the child checked a
    # file the parent was not running and the launcher silently stopped being
    # about the code under test.
    tree = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m",
                        "live.pm_research.be_forward_day", "--selftest"],
                       cwd=str(tree), env=env, capture_output=True,
                       text=True, timeout=900)
    child, child_sha = _child_count(r), _child_sha(r)
    me_sha = _sha_file(Path(__file__).resolve())[:16]
    # THE COMPARISON MUST BE ABLE TO FAIL. In a single tree the parent and the
    # child run the SAME file, so `child == checks` is a tautology there and
    # three mutations of it survived a full audit. A second spawn against a
    # DELIBERATELY DIVERGENT stub tree gives the predicate something it must
    # reject, and both spawns go through the same `_launch_parity`, so a
    # mutation that weakens the comparison fails the known-bad.
    with _tf.TemporaryDirectory() as _td:
        _stub = Path(_td) / "live/pm_research"
        _stub.mkdir(parents=True)
        (_stub / "__init__.py").write_text("")
        (Path(_td) / "live/__init__.py").write_text("")
        (_stub / "be_forward_day.py").write_text(
            "print('be_forward_day selftest: 4242 checks OK')\n")
        _rs = subprocess.run([sys.executable, "-m",
                              "live.pm_research.be_forward_day", "--selftest"],
                             cwd=_td, env=env, capture_output=True,
                             text=True, timeout=120)
        _cs, _cs_sha = _child_count(_rs), _child_sha(_rs)
    # ONE check holding BOTH directions. Written as two, the negative half
    # could be weakened to something that cannot fail and nothing would
    # notice -- which is what a mutation of it proved. Inside one condition,
    # weakening the predicate breaks the positive or the negative side.
    ok(_launch_parity(r.returncode, child, at_entry, child_sha, me_sha)
       and _cs == 4242 and _cs != child and _cs_sha is None
       and not _launch_parity(_rs.returncode, _cs, at_entry, _cs_sha, me_sha),
       f"launch: GREEN under the PACKAGE launch of {tree}, the child counted "
       f"{child} = this parent's count on entry ({at_entry}), and its sha "
       f"is this file's — AND a child spawned from another tree "
       f"reports {_cs} and the same predicate REFUSES it. Both halves are "
       f"needed: in one tree the parent and the child are the same file, so "
       f"the comparison alone is a tautology. Child tail: "
       f"{(r.stdout + r.stderr).strip()[-300:]!r}")
    # BE8-R2: NO increment here. `ok` above already counted this assertion
    # against the caller's `checks`; the `checks += 1` that stood here
    # incremented this function's PARAMETER, which the call site then
    # assigned back OVER `ok`'s increment. The two cancelled, so 121 = 121
    # by arrangement rather than by rule. `ok` is now the only incrementer
    # and an AST check asserts it.
    return checks


def main(argv: list = None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--forward-day" not in argv:
        # BE34-R4: 0 is what a SUCCESSFUL run returns. A caller that misspells
        # the flag got "usage" on stdout and a success code, so a scripted
        # invocation could record a day as run without running it. Every other
        # refusal here returns 2; so does this one.
        print("usage: be_forward_day.py --selftest | "
              "--forward-day <YYYYMMDD> --outdir <dir>")
        return 2
    i = argv.index("--forward-day")
    if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
        print("REFUSED: --forward-day needs a day token (YYYYMMDD)")
        return 2
    day = argv[i + 1]
    outdir = None
    if "--outdir" in argv:
        j = argv.index("--outdir")
        if j + 1 < len(argv) and not argv[j + 1].startswith("-"):
            outdir = Path(argv[j + 1])
    if outdir is None:
        print("REFUSED: --outdir is required. Scores are SEALED (rule 11) and "
              "this driver writes nothing under data/pm_5min/derived/.")
        return 2
    if str(DERIVED) in str(outdir.resolve()):
        print(f"REFUSED: --outdir {outdir} is inside {DERIVED}. Sealed output "
              f"must not land beside canonical artifacts.")
        return 2
    return run_forward_day(day, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
