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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
MARKETS = REPO / "data/pm_5min/markets.jsonl"
RATIFICATION_REF = "R-419"          # R-419 supersedes R-418 (DE round 9)
#: The USER-authorised freeze commit (R-421 §2). Every sha the candidate and
#: its manifest bind equals the blob HERE; the tree moved the anchors in nine
#: commits afterwards. Rule 12: the frozen set is the commit's bytes.
FROZEN_COMMIT = "1b53929"

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


def _provenance() -> dict:
    """The carrying commit and this driver's own bytes.

    A ref alone is not an identity — the dirty flag and the file hash travel
    with it, the same shape the scorer's provenance block uses."""
    import subprocess

    def git(*a):
        try:
            r = subprocess.run(("git", *a), cwd=str(REPO), capture_output=True,
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


def _repo_module_path(name: str) -> Path | None:
    p = REPO / "live/pm_research" / f"{name}.py"
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
    if msha != c.get("manifest_sha256"):
        drift.append(f"manifest {mname}: bound "
                     f"{str(c.get('manifest_sha256'))[:16]} now {msha[:16]}")
    m = json.loads(mp.read_text())
    ps = m.get("pin_semantics") or {}
    default = ps.get("_default", "reproducibility_anchor")
    anchors, checked = [], 0
    for k, want in sorted((m.get("hashes") or {}).items()):
        if ps.get(k, default) != "reproducibility_anchor":
            continue
        anchors.append(k)
        p = REPO / k
        now = _sha_file(p) if p.exists() else None
        checked += 1
        if now != want:
            drift.append(f"{k}: bound {want[:16]} now "
                         f"{(now or 'MISSING')[:16]}")
    bsha = c.get("builder_sha256")
    bp = REPO / "live/pm_research/harmful_hazard_model.py"
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
            "manifest": mname, "manifest_sha256": msha,
            "anchors_checked": checked, "anchor_keys": anchors,
            "builder_sha256": bsha, "contract": "HOLDS"}


# ---------------------------------------------------------------------------
# gate 2 -- the day is closed, and its verdict was written by the scheduled unit
# ---------------------------------------------------------------------------
def assert_day_closed_and_attributed(day: str, verdict: dict = None) -> dict:
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
    if not (isinstance(wr, str) and wr.startswith(SCHEDULED_PREFIX)):
        raise ForwardDayRefused(
            f"REFUSED: {day}'s verdict was not written by the scheduled unit "
            f"— write_reason={wr!r} does not start with the required prefix "
            f"(imported from da_governed_verdict_preflight, matched as a "
            f"PREFIX exactly as DA's preflight matches it; a substring test "
            f"would accept an unattributed hand run).")
    return {"day_closed_calendar": True, "write_reason": wr,
            "write_reason_prefix_source": "da_governed_verdict_preflight."
                                          "SCHEDULED_PREFIX"}


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
    rat = RAT.check(supply, RATIFICATION_REF)
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
    _rv = RAT.require_verified(rat)   # RAISES on a refusal; see below
    rat["pair_asserted"] = assert_ratification_pair(rat)
    # ONE place, and it is HERE. A second reading in `run_forward_day` could
    # not be reached by any affordable suite (gate 2 is 60 s in, the replay is
    # 26 min), so a mutant deleting it survived while the suite stayed green.
    # A guard nothing can drive is a guard nothing protects.
    rat["require_verified"] = {
        "checker": "de_ratification_check.require_verified",
        "verified": bool(_rv.get("verified")),
        "unverifiable": list(_rv.get("unverifiable") or ()),
        "provenance_absent": not _rv.get("provenance"),
        "checks_seen": len(_rv.get("checks") or ())}
    # NO second gate here. `require_verified` checks all three conjuncts and
    # RAISES; a local re-check of the same three could only fire on something
    # the checker let through, which is nothing -- and the audit proved it,
    # by disabling it with no test able to notice (H5b).
    specs = SEAM.window_specs_from_supply(
        supply, ratification_ref=RATIFICATION_REF)   # refusals propagate
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


def build_and_score(selected: list, frozen: dict) -> dict:
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
    qr = HER.qr
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    paths = hm.fi._archive_paths()
    tokens = hm.fi.token_map()
    scores: dict = collections.defaultdict(list)
    actions: set = set()
    n_rows = n_windows = 0
    recon_fail = unhooked = wrong_gen = boundary_bad = clock_bad = 0
    no_features = 0
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
            unhooked += arm.unhooked_changes
        coin = slug.split("-")[0]
        fit = frozen["fits"].get(coin)
        if wrows:
            windows_with_rows.add(slug)
        n_rows += len(wrows)
        if fit is not None and not bad:
            stream = hm.window_streams(paths[slug], *tokens[slug])
            for r in wrows:
                actions.add((slug, r.get("side"), r.get("gen")))
                fp = hm.features(stream, r["t_start"], r["side"],
                                 r.get("level"), r.get("resting"),
                                 r.get("qahead"))
                ff = hm.fine_feats(t0 + r["t_start"], r["side"], coin)
                if fp is None or ff is None:
                    no_features += 1
                    continue
                scores[coin].append(
                    (t0, FS.expected_cancel_value(fit, fp + ff)))
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
    frozen = FS.load_frozen()
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
        n = 1
        while True:
            q = outdir / f"be_forward_day_receipt_{day}.{n}.json"
            if not q.exists():
                break
            n += 1
        rec["supersedes_receipt"] = {
            "path": str(p), "sha256": _sha_file(p),
            "why": "an earlier run's receipt was already here. It is KEPT "
                   "byte-identical and this run takes a numbered successor "
                   "-- overwriting evidence loses the comparison a reader "
                   "needs (rule 13)."}
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
    rec: dict = {"protocol": "BE_FORWARD_DAY_SEALED_V1", "day": day,
                 "ratification_ref": RATIFICATION_REF,
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
        rec["day_verdict"] = gate("day_closed_and_attributed",
                                  lambda: assert_day_closed_and_attributed(day))
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
        _frozen = FS.load_frozen()
        built = gate("rows_and_scores_streamed",
                     lambda: build_and_score(sel, _frozen))
        rec["rows"] = {k: v for k, v in built.items()
                       if k not in ("scores", "windows_with_rows")}
        if built["reconciliation_failures"]:
            raise ForwardDayRefused(
                f"REFUSED: {built['reconciliation_failures']} of "
                f"{built['n_windows']} windows failed reconciliation. The "
                f"reconciliation selftest is the gate: a mismatch fails the "
                f"DAY and is never absorbed.")
        rec["gates"].append({"gate": "reconciliation", "result": "PASS"})
        rec["window_agreement"] = gate(
            "bridged_windows_equal_row_windows",
            lambda: assert_window_sets_agree(
                pop["specs"], built["windows_with_rows"]))
        scored = built["scores"]
        if not scored:
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
        sealed = seal(day, outdir, scored, rep)
        rec["sealed_file"] = sealed
        rec["outcome"] = "SCORED"
    except Exception as e:                           # noqa: BLE001
        rec["outcome"] = "REFUSED"
        rec["refused_at"] = rec["gates"][-1]["gate"] if rec["gates"] else None
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
_R1_FROZEN = {"fits": {"btc": {"w": _R1_W},
                       "eth": {"w": [x * 1.5 for x in _R1_W]}}}


def _r1_windows(bad_window: str = None) -> tuple:
    """(selected, rows) for a handful of windows with KNOWN features."""
    selected, rows = [], []
    for k, (coin, t0) in enumerate((("btc", 1788000000), ("eth", 1788000300),
                                    ("btc", 1788000600), ("eth", 1788000900))):
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
        FS.load_frozen = lambda: _R1_FROZEN
        return self

    def __exit__(self, *exc):
        for n, m in self._saved_mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        FS.expected_cancel_value, FS.load_frozen = self._saved_fs
        return False


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

    def _restore_imports():
        sys.path[:] = _sp_saved
        for _k in list(sys.modules):
            if _k not in _mods_saved:
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
        try:
            import_frozen_anchors(Path(mat["root"]), [k for k in mat["anchors"]
                                                      if k.endswith(".py")])
            ok(False, "a tampered materialised byte must not import cleanly")
        except (ForwardDayRefused, Exception):
            ok(True, "R-421(2) a TAMPERED materialised byte does not pass — "
                     "the sha is checked at materialisation, before any "
                     "import, so a byte verified after import has not run")
        # re-materialising RESTORES and REFUSES if the source moved
        mat2 = materialise_frozen(o4)
        ok(_sha_file(Path(mat2["anchors"][_py]["materialised_to"]))
           == mat2["anchors"][_py]["sha256"],
           "R-421(2) re-materialisation restores the frozen bytes exactly, so "
           "a tampered run dir cannot persist into the next run")

    _restore_imports()
    ok(Path(getattr(__import__("harmful_exposure_rows"), "__file__", "")
            ).parent == Path(__file__).parent,
       "R-421(2) the import controls RESTORE sys.path and sys.modules — a "
       "suite that leaves frozen modules imported from a deleted tmpdir "
       "poisons every check after it")

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
        # refused every run. 09-02 refuses at gate 1 in ~0 s, so a real
        # emission is affordable here and the check now meets one.
        _rc9 = run_forward_day("20260902", o8)
        _r9 = json.loads(receipt_path(o8, "20260902").read_text())
        ok(_rc9 != 0 and _r9["gates"][0]["result"] == "REFUSED"
           and isinstance(_r9["gates"][0]["gate"], str),
           f"R5(5) POSITIVE CONTROL ON A REAL EMISSION: the 09-02 run refuses "
           f"at `{_r9['gates'][0]['gate']}` and its receipt WRITES — a "
           f"fixture-only control let the post-condition refuse every real "
           f"receipt without the suite noticing (rule 17)")
        ok(_r9["decision_field_check"]["excused_paths"] == ["gates[].gate"],
           "R5(5) and exactly ONE path is excused, named in the receipt with "
           "its reason — an exemption a reader cannot see is not one")
    # ---- BE34-R1: the streaming pass EQUALS the reference, per score -----
    _sel, _rows = _r1_windows()
    with _r1_installed(_rows):
        _bs = build_and_score(_sel, _R1_FROZEN)
        _sr = score_rows(_rows)
    ok(_bs["scores"] == _sr,
       f"BE34-R1 ONE fixture, TWO consumers: `build_and_score`'s streamed "
       f"scores are EQUAL to `score_rows`'s on the same windows "
       f"({ {c: len(v) for c, v in _sr.items()} }) — the streaming rewrite "
       f"replaced the reference and nothing had ever compared them")
    ok(sum(len(v) for v in _sr.values()) == 11 and _bs["n_rows"] == 12
       and _bs["rows_without_features"] == 1,
       f"BE34-R1 and BOTH drop the SAME row: 12 rows in, 11 scored, 1 "
       f"without features (got {_bs['n_rows']}/"
       f"{sum(len(v) for v in _sr.values())}/"
       f"{_bs['rows_without_features']}) — an equality between two consumers "
       f"that both dropped everything would also be an equality")
    ok(_bs["n_windows"] == 4 and _bs["n_actions"] == 12
       and _bs["n_windows_with_rows"] == 4,
       f"BE34-R1 the counters the receipt publishes come from the same pass: "
       f"4 windows, 12 actions (got {_bs['n_windows']}/{_bs['n_actions']})")
    _vals = sorted(v for lst in _sr.values() for _, v in lst)
    ok(len(set(_vals)) == len(_vals) and max(abs(v) for v in _vals) < 1e5,
       f"BE34-R1 every fixture score is DISTINCT and small "
       f"(|max| {max(abs(v) for v in _vals):.1f}) — equal-valued or huge "
       f"scores would hide a perturbation instead of catching it")

    # ---- BE34-R1: a reconciliation failure fails the DAY, by name --------
    _selb, _rowsb = _r1_windows()
    with _r1_installed(_rowsb, bad_window="eth-updown-5m-1788000300"):
        _bad = build_and_score(_selb, _R1_FROZEN)
    _eth_t0 = {t for t, _ in _bad["scores"].get("eth", ())}
    ok(_bad["reconciliation_failures"] == 1 and _bad["n_windows"] == 4
       and 1788000300 not in _eth_t0 and 1788000900 in _eth_t0
       and _bad["n_rows"] == 12 and _bad["n_actions"] == 12,
       f"BE34-R1 a window whose fills do not reconcile is COUNTED and ITS "
       f"rows are not scored, while the OTHER window of the same coin still "
       f"is ({_bad['reconciliation_failures']} of {_bad['n_windows']}; eth "
       f"t0s {sorted(_eth_t0)}) — the v3 builder's STRICT condition, and the "
       f"failure is scoped to the window, not spread to the coin")
    with _tf.TemporaryDirectory() as tdr1:
        _real_bas = globals()["build_and_score"]
        globals()["build_and_score"] = lambda sel, fr: dict(_bad)
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
           and _recr["refused_at"] == "rows_and_scores_streamed",
           f"BE34-R1 KNOWN-BAD: the CALLER refuses the whole DAY by name "
           f"(rc={_rc}) — driven through the real gate chain, so disabling "
           f"the refusal is visible here; a mismatch is never absorbed")
        ok(not any(g["gate"] == "reconciliation" and g["result"] == "PASS"
                   for g in _recr["gates"]),
           "BE34-R1 and no `reconciliation: PASS` is recorded for a day that "
           "refused — a receipt that says PASS beside a refusal is the "
           "hardcoded-verdict shape (rule 10)")

    # ---- BE34-R4: a usage error is a refusal, not a success --------------
    ok(main([]) == 2 and main(["--forward-day"]) == 2,
       "BE34-R4 `main([])` returns 2, the code every other refusal here "
       "uses — returning 0 let a misspelled flag look like a day that ran")

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

    _before = checks
    checks = _selftest_launch(checks, ok)
    import os as _os
    if (_os.environ.get("BE_FORWARD_LAUNCH_CHECK") != "1"
            and checks == _before):
        raise AssertionError(
            "the launch-invariance check contributed NO checks, so it did not "
            "run — removing that call is a guard-removal nothing else "
            "notices, which is CO-1's own shape")
    print(f"be_forward_day selftest: {checks} checks OK")
    return 0


def _selftest_launch(checks: int, ok) -> int:
    """Green under BOTH launchers, asserted rather than assumed."""
    import os, subprocess
    if os.environ.get("BE_FORWARD_LAUNCH_CHECK") == "1":
        return checks
    env = dict(os.environ, BE_FORWARD_LAUNCH_CHECK="1")
    # BE34-R3: the tree of THIS FILE, never a hardcoded root. `cwd=REPO` made
    # every worktree spawn the SHARED tree's module, so the child checked a
    # file the parent was not running and the launcher silently stopped being
    # about the code under test.
    tree = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, "-m",
                        "live.pm_research.be_forward_day", "--selftest"],
                       cwd=str(tree), env=env, capture_output=True,
                       text=True, timeout=900)
    m = re.search(r"be_forward_day selftest: (\d+) checks OK",
                  r.stdout + r.stderr)
    child = int(m.group(1)) if m else None
    # ONE check, deliberately: rc AND parity together, so the child's count is
    # exactly the parent's minus this very check -- which is the parity the
    # guard is named for. Splitting it in two would make "minus one" false.
    ok(r.returncode == 0 and child == checks,
       f"launch: GREEN under the PACKAGE launch of {tree}, and the child "
       f"counted {child} = this parent's {checks} (its total is ours minus "
       f"this check) — an rc alone cannot tell a child running THIS file "
       f"from one running another tree's. Child tail: "
       f"{(r.stdout + r.stderr).strip()[-300:]!r}")
    checks += 1
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
