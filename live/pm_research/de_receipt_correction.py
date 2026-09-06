"""THE RECEIPT `.v2` FORM (REV 79 §3) -- this seat is the producer.

WHAT A CORRECTION MAY DO. It ADDS declared keys and it changes NOTHING
else. It never publishes an economic field, never recomputes a statistic,
never touches a sealed or frozen block, never puts a reconstruction under
a plain name, and never carries half a `supersedes` -- the pair or
nothing.

THE CENSUS IS BE'S. `be_race_reader.correction_census` is IMPORTED, never
re-implemented: one census for both seats, so a correction that passes
here passes there for the same reason (R-641's rule applied to a
predicate rather than a parser).

THE FROZEN SET IS READ, NEVER TYPED. Everything the v1 already carries is
frozen: the declared additions are the only keys that may appear, and
every pre-existing key must be byte-identical. The blocks REV 79 names --
`per_day_sealed_artifacts`, `decision_populations`, `work_counters`,
`battery`, `memory_plan`, `resources`, `source_identity`, the byte-level
provenance -- are then frozen BY CONSTRUCTION rather than by a list
somebody maintains, and this module asserts they are among them.

RECONSTRUCTIONS RIDE IN THE KEY. A value recovered rather than recorded is
named `..._RECONSTRUCTED`, and one that cannot be recovered is a NAMED
ABSENCE -- never an estimate, never a plain field a reader would take for
a measurement.

    python3 live/pm_research/de_receipt_correction.py --selftest
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import be_race_reader as BE                               # noqa: E402
import de_multiday_gate1_runner as RUNNER                 # noqa: E402

PROTOCOL = "P003_DE_RECEIPT_CORRECTION_V1"
EXPECTED_CHECKS = 27

#: The blocks REV 79 §3 names. NOT the frozen set -- the frozen set is
#: "everything v1 carries". These are asserted to be INSIDE it, so the
#: naming is a check on the construction rather than the construction.
REV79_NAMED_BLOCKS = (
    "per_day_sealed_artifacts", "decision_populations", "work_counters",
    "battery", "memory_plan", "resources", "source_identity",
    "provenance",
)

#: A reconstruction never sits under one of these.
PLAIN_NAMES_A_RECONSTRUCTION_MAY_NOT_USE = (
    "sha256_at_load", "sha256_at_emit", "sha256", "n_days_complete",
    "producing_code_sha256",
)

#: THE SAME LINE, DRAWN ONE STEP EARLIER (DE 109). The check above fires
#: only when the VALUE says RECONSTRUCTED -- so a recovered digest written
#: as a bare hex string under `sha256` passed it, which is the exact shape
#: REV 79 S3 forbids ("never as `sha256_at_load`, which nobody took").
#: Inside an ADDED block nothing was measured at run time, so a plain
#: digest name there is refused on the NAME, whatever the value looks
#: like. Pre-existing blocks are untouched: they carry real measurements.
PLAIN_DIGEST_NAMES = ("sha256", "sha256_at_load", "sha256_at_emit",
                      "digest", "producing_code_sha256")


class CorrectionRefused(RuntimeError):
    """The correction cannot be emitted honestly."""


def frozen_set_of(v1: dict, additions) -> dict:
    """EVERY KEY THE v1 CARRIES, read from the v1 (never typed)."""
    frozen = sorted(set(v1) - set(additions))
    named_inside = {b: (b in frozen) for b in REV79_NAMED_BLOCKS
                    if b in v1}
    missing = [b for b, ok in named_inside.items() if not ok]
    if missing:
        raise CorrectionRefused(
            f"REFUSED: {missing} are declared as ADDITIONS while the v1 "
            f"already carries them. A correction adds; it does not "
            f"redefine what is already recorded.")
    return {"frozen_keys": frozen, "n_frozen": len(frozen),
            "read_from": "the v1's own keys -- everything it carries is "
                         "frozen, and the declared additions are the only "
                         "keys that may appear",
            "rev79_named_blocks_present_and_frozen": named_inside}


def sealed_names_in_force_for(rec: dict) -> dict:
    """The sealed list in force WHEN THAT RECEIPT WAS EMITTED."""
    scope = RUNNER.design_version_of_receipt(rec)
    return {"design_version": scope["design_version"],
            "read_from": scope["read_from"],
            "sealed_names": list(scope["fields_in_force"]),
            "n": len(scope["fields_in_force"])}


def scope_at_this_correction() -> dict:
    """THE SEALED LIST IN FORCE **NOW** -- at the moment this correction is
    written, not when the artifact it corrects was produced.

    REV 82 §2.3: `seal_census_by_KEY_before_writing` was scoped entirely to
    the past. All three landed corrections resolve design v21 on BOTH
    halves of DE 109's union, so the union was EIGHT while the code
    writing them seals ELEVEN -- an addition carrying `n_fills_arm` would
    not have been flagged.

    The version is RESOLVED, not read from a literal: this module's
    `DESIGN_VERSION_IN_FORCE` is a constant that has to track a moving
    thing, so the chain head is resolved too and the WIDER of the two is
    used. A resolution failure is a STATUS and falls back to the constant
    -- never to a narrower list."""
    v_const = RUNNER.DESIGN_VERSION_IN_FORCE
    v_head, how = None, None
    try:
        v_head = RUNNER.design_chain().get("head_version")
        how = "design_chain() head, resolved at this emit"
    except Exception as exc:                     # a STATUS, never a skip
        how = f"UNRESOLVED: {type(exc).__name__}: {exc}"
    v = max([x for x in (v_const, v_head) if isinstance(x, int)])
    return {"design_version": v,
            "module_constant": v_const,
            "chain_head_version": v_head,
            "chain_head_read": how,
            "rule": "the WIDER of the module's own version and the "
                    "resolved chain head -- a constant alone is a literal "
                    "tracking a moving thing, and a failed resolution "
                    "never narrows the list",
            "sealed_names": list(RUNNER.economic_fields_in_force(v)),
            "n": len(RUNNER.economic_fields_in_force(v))}


def two_scope_seal_census(v1: dict, v2: dict, *, at_v1: dict,
                          now: dict) -> dict:
    """THE SEAL CENSUS BY KEY, UNDER **TWO** SCOPES (R-716 ruling).

    DA 105 forced the distinction and it is the whole of it: the 09-03
    `.v2` carries the three outcome counts as keys **inherited
    byte-identical from its v1**, open under that receipt's v21 eight and
    sealed under today's eleven. A plain union would refuse every
    correction of an eight-scope receipt -- because the frozen blocks MAY
    NOT CHANGE, so the correction cannot remove what it is forbidden to
    touch, and the only obedient act would be not correcting it at all.

    So:
      INHERITED keys (a sealed-name path the v1 already carries) are
        judged under **the v1's own scope**. One that is open there is
        REPORTED as inherited-open, with its paths -- never refused, and
        never silently dropped either.
      ADDED keys (a path the .vN introduces) are judged under
        **v1's scope UNION the scope in force at THIS correction's emit**.
        A correction is written today and may not carry a name that is
        sealed today, whatever the artifact it corrects predates.
    """
    w1, w2 = set(RUNNER.seal_key_walk(v1)), RUNNER.seal_key_walk(v2)
    inherited = sorted(p for p in w2 if p in w1)
    added = sorted(p for p in w2 if p not in w1)
    at_v1_names = set(at_v1["sealed_names"])
    added_names = at_v1_names | set(now["sealed_names"])

    def _leaf(path):
        return path.rsplit(".", 1)[-1]

    inherited_leaks = [p for p in inherited if _leaf(p) in at_v1_names]
    inherited_open = [p for p in inherited if _leaf(p) not in at_v1_names]
    added_leaks = [p for p in added if _leaf(p) in added_names]
    return {
        "rule": RUNNER.SEAL_RULE,
        "ruling": "R-716 (coordinator, disclosed to REV 83 for overrule) "
                  "on REV 82 §2.3 with DA 105's distinction",
        "inherited": {
            "judged_under": {
                "design_version": at_v1["design_version"],
                "n": at_v1["n"], "sealed_names": at_v1["sealed_names"],
                "read_from": at_v1["read_from"]},
            "n_paths_walked": len(inherited),
            "n_sealed_keys_found": len(inherited_leaks),
            "found": inherited_leaks,
            "n_open_under_v1_but_sealed_today": len(inherited_open),
            "open_under_v1_but_sealed_today": inherited_open,
            "why_these_are_not_a_refusal": (
                "the v1 already carried them and the frozen blocks may "
                "not change -- refusing here would forbid correcting an "
                "eight-scope receipt at all. They are REPORTED, with "
                "their paths, so a reader judging under today's list "
                "sees exactly what DA's independent census sees")},
        "added": {
            "judged_under": {
                "v1_scope_design_version": at_v1["design_version"],
                "correction_emit_design_version": now["design_version"],
                "n": len(added_names),
                "sealed_names": sorted(added_names)},
            "n_paths_walked": len(added),
            "n_sealed_keys_found": len(added_leaks),
            "found": added_leaks},
        "the_two_version_pair": f"v{at_v1['design_version']} (the "
                                f"artifact) / v{now['design_version']} "
                                f"(this correction's emit)",
        "n_sealed_keys_found": len(inherited_leaks) + len(added_leaks),
    }


def assert_no_reconstruction_under_a_plain_name(v2: dict) -> dict:
    """A RECOVERED VALUE RIDES IN THE KEY (REV 79 §3, BE's §1.2)."""
    bad = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if (k in PLAIN_NAMES_A_RECONSTRUCTION_MAY_NOT_USE
                        and isinstance(v, str)
                        and "RECONSTRUCT" in v.upper()):
                    bad.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(v2)
    if bad:
        raise CorrectionRefused(
            f"REFUSED: a reconstruction sits under a PLAIN name -- {bad}. "
            f"A reader keying it would get a value that looks recorded; "
            f"the status must ride in the KEY.")
    return {"checked": list(PLAIN_NAMES_A_RECONSTRUCTION_MAY_NOT_USE),
            "none_carry_a_reconstruction": True}


def assert_added_blocks_carry_no_plain_digest(v2: dict, added) -> dict:
    """A DIGEST INSIDE AN ADDED BLOCK WAS RECOVERED, NEVER TAKEN.

    An addition is written now, about a run that finished; no digest in it
    can be a reading. So the reconstruction must ride in the KEY there --
    on the name, not on whether the value happens to spell it out."""
    bad = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in PLAIN_DIGEST_NAMES:
                    bad.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    for k in added:
        if k == "supersedes":       # the PAIR's own halves, R-608's shape
            continue
        walk(v2.get(k), k)
    if bad:
        raise CorrectionRefused(
            f"REFUSED: an ADDED block carries a PLAIN digest name -- "
            f"{bad}. Nothing in an addition was measured at run time, so "
            f"a digest there is RECONSTRUCTED and must say so in its key.")
    return {"added_blocks_walked": [k for k in added if k != "supersedes"],
            "plain_digest_names_checked": list(PLAIN_DIGEST_NAMES),
            "none_present": True,
            "why_supersedes_is_exempt": "its {path, sha256} pair is R-608's "
                                        "own shape, recomputed here from "
                                        "the file it names"}


def assert_supersedes_is_a_pair(v2: dict, v1_path: Path) -> dict:
    """THE PAIR OR NOTHING (R-608), and the digest recomputed here."""
    sup = v2.get("supersedes")
    if not isinstance(sup, dict) or not sup.get("path") \
            or not sup.get("sha256"):
        raise CorrectionRefused(
            "REFUSED: `supersedes` is not the PAIR {path, sha256}. Half a "
            "link is not a link (R-608).")
    actual = hashlib.sha256(Path(v1_path).read_bytes()).hexdigest()
    if sup["sha256"] != actual:
        raise CorrectionRefused(
            f"REFUSED: `supersedes.sha256` is {sup['sha256'][:16]} and the "
            f"file it names hashes to {actual[:16]}.")
    return {"path": sup["path"], "sha256": sup["sha256"],
            "recomputed_here": actual, "agrees": True}


def _emit_stamp_of(rec: dict, path: Path) -> dict:
    """WHEN THE RUN THAT WROTE THIS RECEIPT EMITTED IT.

    From the receipt's OWN `emitted_at_utc` -- the field the emitting run
    wrote -- never from the filename, which is a NEARBY PROXY: a `.vN`
    correction is named with the CORRECTION's stamp while the run it
    describes emitted earlier (09-04: run 16:33:51Z, correction 17:11:44Z).
    The filename stamp is parsed too and carried beside it as a
    cross-check, so a disagreement is visible rather than silent."""
    iso = rec.get("emitted_at_utc")
    at = None
    if isinstance(iso, str):
        try:
            at = datetime.datetime.fromisoformat(iso)
            if at.tzinfo is None:
                at = at.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            at = None
    m = re.search(r"_SEALED__(\d{8}T\d{6}Z)", Path(path).name)
    fn = None
    if m:
        fn = datetime.datetime.strptime(
            m.group(1), "%Y%m%dT%H%M%SZ").replace(
                tzinfo=datetime.timezone.utc)
    return {"emitted_at_utc": iso, "parsed": at,
            "filename_stamp": m.group(1) if m else None,
            "filename_stamp_parsed": fn,
            "read_from": "the receipt's own `emitted_at_utc`",
            "filename_stamp_agrees_to_the_second": (
                None if (at is None or fn is None)
                else abs((at - fn).total_seconds()) < 1.0)}


def n_days_complete_true_at_emit(rec_path: Path, root: Path,
                                 rec: dict | None = None) -> dict:
    """HOW MANY DAYS WERE SEALED WHEN THIS RECEIPT'S RUN EMITTED IT.

    Counted from the sealed receipts whose OWN `emitted_at_utc` is at or
    before this one's -- the method named here, inside the artifact, so a
    reader need not reconstruct the reconstruction. Every stamp compared
    is read from the receipt that carries it (CLAUDE.md rule 3); the
    filenames are parsed only as a cross-check and any disagreement is
    reported.

    The DAY is the key, so a `.vN` correction of a day already counted
    does not count it twice -- and a correction cannot raise the count of
    the run it corrects, because it carries that run's `emitted_at_utc`."""
    d = Path(root) / "pm_5min/derived"
    rec = rec if rec is not None else json.loads(
        Path(rec_path).read_text())
    mine = _emit_stamp_of(rec, Path(rec_path))
    if mine["parsed"] is None:
        return {"status": "NOT_RECONSTRUCTABLE",
                "why": "this receipt carries no parseable "
                       "`emitted_at_utc`, and the filename stamp is a "
                       "proxy this method does not substitute"}
    seen, rows, unreadable, disagree = {}, [], [], []
    for f in sorted(d.glob("p003_de_gate1_day_run_*_SEALED__*.json")):
        m = re.match(r"p003_de_gate1_day_run_(\d{8})_SEALED__", f.name)
        if not m:
            continue
        try:
            other = json.loads(f.read_text())
        except (OSError, ValueError) as exc:
            # RULE 11: an unreadable candidate is a STATUS, never a skip --
            # a receipt that cannot be read is not a receipt that is late.
            unreadable.append({"file": f.name, "why": str(exc)})
            continue
        st = _emit_stamp_of(other, f)
        if st["parsed"] is None:
            unreadable.append({"file": f.name,
                               "why": "no parseable `emitted_at_utc`"})
            continue
        if st["filename_stamp_agrees_to_the_second"] is False:
            disagree.append({"file": f.name,
                             "emitted_at_utc": st["emitted_at_utc"],
                             "filename_stamp": st["filename_stamp"]})
        counted = st["parsed"] <= mine["parsed"]
        rows.append({"day": m.group(1), "file": f.name,
                     "emitted_at_utc": st["emitted_at_utc"],
                     "filename_stamp": st["filename_stamp"],
                     "counted": counted})
        if counted:
            seen.setdefault(m.group(1), st["emitted_at_utc"])
    if unreadable:
        raise CorrectionRefused(
            f"REFUSED: {len(unreadable)} sealed receipt(s) could not be "
            f"read for the count -- {unreadable[:3]}. A count taken over "
            f"the receipts that happened to parse is not a count.")
    return {"n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED": len(seen),
            "days_sealed_at_or_before_this_emit": sorted(seen),
            "this_receipt": mine["emitted_at_utc"],
            "this_receipt_filename_stamp": mine["filename_stamp"],
            "method": ("counted from the SEALED day receipts under the "
                       "ledger whose OWN `emitted_at_utc` is at or before "
                       "this receipt's own `emitted_at_utc`; the DAY is "
                       "the key, so a `.vN` correction of a day already "
                       "counted does not count it twice"),
            "why_not_the_filename": (
                "a `.vN` correction is named with the CORRECTION's stamp "
                "while it carries the RUN's `emitted_at_utc`; the stamp a "
                "receipt is judged by is the one the emitting event wrote"),
            "filename_stamp_cross_check": {
                "n_disagreeing_with_their_own_emitted_at_utc":
                    len(disagree),
                "disagreeing": disagree},
            "all_sealed_receipts_considered": rows,
            "why_the_key_says_RECONSTRUCTED": (
                "the emitted `n_days_complete` is left UNTOUCHED; this is "
                "recovered after the fact and says so in its own name")}


def provenance_reconstructed_at_the_carrying_commit(
        rec: dict, *, repo: Path) -> dict:
    """THE PARAMS AND DESIGN THE RECEIPT ITSELF NAMES, DIGESTED AT ITS
    CARRYING COMMIT (REV 79 S3(3)).

    Nothing here is measured: the run took no digest of either file, and
    this block never pretends it did. The PATHS come from the receipt's
    own fields; the DIGESTS come from git at the commit the receipt names
    as its own. Where a value cannot be recovered the field is a NAMED
    ABSENCE, never an estimate."""
    commit = (rec.get("source_identity") or {}).get("carrying_commit")
    lock = rec.get("fixture_day_lock") or {}
    opened = ((rec.get("before_work") or {}).get("residency") or {}
              ).get("data_paths_opened") or []
    out = {
        "WHAT_THIS_BLOCK_IS": (
            "RECONSTRUCTED, not stamped. The run wrote no params/design "
            "pin (that is why R-654's pre-read had to infer both). The "
            "PATHS below are read from this receipt's own fields and the "
            "DIGESTS from git at the commit this receipt names as its "
            "carrying commit -- no digest here was taken at load or at "
            "emit, and none is offered as one."),
        "carrying_commit_RECONSTRUCTED_FROM": (
            "source_identity.carrying_commit"),
        "carrying_commit": commit,
        "inputs": {},
    }
    if not commit:
        out["status"] = "NOT_RECONSTRUCTABLE"
        out["why"] = ("the receipt names no `carrying_commit`, so there "
                      "is no commit at which to resolve a digest")
        return out

    def _rel(pth: str) -> str:
        pth = str(pth)
        for pre in (str(Path(repo).resolve()) + "/", "/home/yuqing/ctaNew/"):
            if pth.startswith(pre):
                return pth[len(pre):]
        return pth.lstrip("/")

    named = {}
    if lock.get("ruled_day_set_read_from"):
        named["params"] = {
            "path": lock["ruled_day_set_read_from"],
            "named_in_this_receipt_at": "fixture_day_lock."
                                        "ruled_day_set_read_from"}
    designs = [q for q in opened if re.search(r"_design_v\d+\.json$",
                                              str(q))]
    if len(designs) == 1:
        named["design"] = {
            "path": _rel(designs[0]),
            "named_in_this_receipt_at": "before_work.residency."
                                        "data_paths_opened"}
    elif len(designs) > 1:
        # REV 73 S1.1(c): AN OPENED PATH IS NOT A PIN, and this run opened
        # a stale v10 beside v21. The bare `_design_v<N>.json` is the
        # DECLARATION; a stamped sibling is a snapshot of one. Both are
        # reported, and the choice is stated rather than taken silently.
        bare = [q for q in designs
                if re.match(r"^p003_de_multiday_gate1_design_v\d+\.json$",
                            Path(str(q)).name)]
        named["design"] = {
            "path": _rel(bare[0]) if len(bare) == 1 else None,
            "named_in_this_receipt_at": "before_work.residency."
                                        "data_paths_opened",
            "all_design_paths_this_run_opened": [_rel(q) for q in designs],
            "how_this_one_was_chosen": (
                "the only path whose name is the bare "
                "`_design_v<N>.json` DECLARATION; the other(s) are "
                "stamped snapshots. AN OPENED PATH IS STILL NOT A PIN "
                "(REV 73 S1.1(c)) -- this block reconstructs what the run "
                "READ, and does not claim the run pinned it"),
            "status": ("RECONSTRUCTED" if len(bare) == 1
                       else "NOT_RECONSTRUCTABLE")}
        if len(bare) != 1:
            named["design"]["why"] = (
                f"{len(bare)} bare declaration paths among the opened "
                f"design paths -- which one the run was judged under "
                f"cannot be recovered from this receipt")
    for k, blk in named.items():
        row = dict(blk)
        rel = row.get("path")
        if not rel:
            row.setdefault("status", "NOT_RECONSTRUCTABLE")
            out["inputs"][k] = row
            continue
        dig = RUNNER._blob_sha256_at(commit, rel, Path(repo))
        if dig is None:
            row["status"] = "NOT_RECONSTRUCTABLE"
            row["why"] = (f"`git show {commit[:8]}:{rel}` returns nothing "
                          f"-- the path is not tracked at the carrying "
                          f"commit, so no digest can be recovered")
        else:
            row.setdefault("status", "RECONSTRUCTED")
            row["sha256_AT_THE_CARRYING_COMMIT_RECONSTRUCTED"] = dig
            here = Path(repo) / rel
            now = (hashlib.sha256(here.read_bytes()).hexdigest()
                   if here.is_file() else None)
            row["and_the_file_on_disk_now"] = {
                "sha256_read_now": now,
                "equals_the_commit_bytes": (now == dig),
                "what_this_is": "a cross-check on whether the file has "
                                "MOVED since the carrying commit -- not a "
                                "second source for the reconstruction"}
        out["inputs"][k] = row
    missing = [k for k in ("params", "design") if k not in out["inputs"]]
    for k in missing:
        out["inputs"][k] = {
            "status": "NOT_RECONSTRUCTABLE",
            "why": f"this receipt names no {k} path in any field this "
                   f"reconstruction reads"}
    out["status"] = ("RECONSTRUCTED"
                     if all(v.get("status") == "RECONSTRUCTED"
                            for v in out["inputs"].values())
                     else "PARTIALLY_RECONSTRUCTED")
    return out


def build_correction(v1_path: Path, additions: dict, *,
                     root: Path, now=None) -> dict:
    """v1 + declared additions, censused BEFORE it is written."""
    v1_path = Path(v1_path)
    raw = v1_path.read_bytes()
    v1 = json.loads(raw)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    v2 = json.loads(json.dumps(v1))
    for k, val in additions.items():
        if k in v1:
            raise CorrectionRefused(
                f"REFUSED: {k!r} is declared as an addition and the v1 "
                f"already carries it.")
        v2[k] = val
    v2["supersedes"] = {"path": str(v1_path),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    declared = tuple(sorted(set(additions) | {"supersedes"}))
    frozen = frozen_set_of(v1, declared)
    # BE'S CENSUS, IMPORTED. One predicate for both seats.
    census = BE.correction_census(v1, v2, added=declared)
    sealed = sealed_names_in_force_for(v1)
    # THE SCOPE THE OUTPUT ITSELF RESOLVES TO. The list in force is the one
    # at this receipt's OWN emit -- a correction inherits `emitted_at_utc`
    # and does not become a later receipt. But an addition can move what a
    # READER resolves from the .vN (adding `provenance` moves the 09-03
    # receipt off positive recognition), so the census runs against the
    # UNION and both resolutions are recorded below: a widening is never
    # silently escaped, and a narrowing is never silently taken.
    sealed_out = sealed_names_in_force_for(v2)
    scope_now = scope_at_this_correction()
    seal = two_scope_seal_census(v1, v2, at_v1=sealed, now=scope_now)
    if seal["inherited"]["n_sealed_keys_found"]:
        raise CorrectionRefused(
            f"REFUSED before writing: the correction carries sealed names "
            f"as KEYS that are sealed under the CORRECTED ARTIFACT'S OWN "
            f"scope (v{sealed['design_version']}) -- "
            f"{seal['inherited']['found'][:6]}. {RUNNER.SEAL_RULE}")
    if seal["added"]["n_sealed_keys_found"]:
        raise CorrectionRefused(
            f"REFUSED before writing: an ADDED key carries a name sealed "
            f"under v{sealed['design_version']} (the artifact) or "
            f"v{scope_now['design_version']} (this correction's emit) -- "
            f"{seal['added']['found'][:6]}. A correction is written TODAY "
            f"and may not introduce a name that is sealed today, whatever "
            f"the artifact it corrects predates (REV 82 §2.3). "
            f"{RUNNER.SEAL_RULE}")
    plain = assert_no_reconstruction_under_a_plain_name(v2)
    digests = assert_added_blocks_carry_no_plain_digest(v2, declared)
    v2["correction_census"] = {
        "protocol": PROTOCOL,
        "as_of": now.isoformat(),
        "census_by": "be_race_reader.correction_census -- IMPORTED, never "
                     "re-implemented",
        "census": census,
        "frozen_set": frozen,
        "sealed_names_in_force_at_v1s_emit": sealed,
        "sealed_names_the_correction_itself_resolves_to": {
            **sealed_out,
            "same_names_as_the_v1_resolves": (
                sorted(sealed_out["sealed_names"])
                == sorted(sealed["sealed_names"])),
            "why_this_is_recorded": (
                "an addition can move which rule a READER resolves from "
                "the .vN. The census is run under the list in force at "
                "this receipt's own emit; if the two lists ever differ, "
                "the difference is here rather than in nobody's hands")},
        "seal_census_by_KEY_before_writing": {
            **seal,
            "walked": "every key at every depth of the OUTPUT, before it "
                      "was written; a leak REFUSES and is never repaired",
            "scope_at_this_corrections_emit": scope_now,
            "the_vNs_own_resolution": {
                "design_version": sealed_out["design_version"],
                "n": sealed_out["n"],
                "is_a_subset_of_the_correction_emit_scope": set(
                    sealed_out["sealed_names"])
                <= set(scope_now["sealed_names"]),
                "why_it_no_longer_widens_the_judgement": (
                    "DE 109 unioned it in; it resolves from the .vN's own "
                    "(inherited) emit stamp, so it can never exceed the "
                    "scope in force at this emit. Recorded as a fact, and "
                    "the predicate above says so rather than assuming it")},
        },
        "no_reconstruction_under_a_plain_name": plain,
        "no_plain_digest_in_an_added_block": digests,
        "what_a_correction_may_never_do": [
            "publish an economic field", "recompute a statistic",
            "change a sealed or frozen block",
            "put a reconstruction under a plain name",
            "carry half a `supersedes`"],
    }
    assert_supersedes_is_a_pair(v2, v1_path)
    return v2


#: THE THREE RECEIPTS THIS SEAT PRODUCED AND IS THE PRODUCER OF, each
#: named by its CHAIN HEAD -- the file nothing else supersedes -- never by
#: a v1 literal. `--emit` re-resolves each chain before it builds.
CORRECTION_TARGETS = ("2026-09-03", "2026-09-04", "2026-09-05")


def _next_suffix(head_name: str) -> str:
    """`.vN.json` FOLLOWING THE FAMILY'S OWN SUFFIX CONVENTION."""
    m = re.search(r"\.v(\d+)\.json$", head_name)
    n = int(m.group(1)) + 1 if m else 2
    stem = re.sub(r"(\.v\d+)?\.json$", "", head_name)
    return f"{stem}.v{n}.json", n


def emit_correction(day: str, *, root: Path, repo: Path,
                    now=None) -> dict:
    """ONE receipt's correction: resolve the head, build, census, write,
    then RE-RESOLVE and require the new file to BE the head."""
    d = Path(root) / "pm_5min/derived"
    files = sorted(d.glob(RUNNER.sealed_day_receipt_glob(day)))
    before = RUNNER.resolve_day_chain(files, kind="day receipt")
    if before["status"] not in RUNNER.CHAIN_RESOLVED_STATUSES:
        raise CorrectionRefused(
            f"REFUSED: {day}'s receipt chain does not resolve before the "
            f"correction -- {before['status']}. A correction of an "
            f"unresolved chain is a second head, not a correction.")
    head = Path(before["head"])
    rec = json.loads(head.read_text())
    out_name, vn = _next_suffix(head.name)
    out = d / out_name
    if out.exists():
        raise CorrectionRefused(
            f"REFUSED: {out.name} already exists. A correction is a NEW "
            f"version; rule 13 does not overwrite.")

    additions = {"n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED":
                 n_days_complete_true_at_emit(head, root, rec)}
    if "provenance" not in rec:
        additions["provenance"] = (
            provenance_reconstructed_at_the_carrying_commit(
                rec, repo=repo))

    v2 = build_correction(head, additions, root=root, now=now)
    v2["correction_census"]["emitted_by"] = {
        "seat": "DE -- the producer of these receipts (REV 79 S3(5))",
        "module": str(Path(__file__).resolve().relative_to(
            Path(repo).resolve())),
        "module_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "ruling": "REVIEW_BE73_DA99_V2_FORM_2026-09-06.md S3",
        "suffix_convention": (
            f"`.v{vn}.json` -- the family's own suffix, which this seat's "
            f"chain resolver already follows by the PAIR; the STAMP is "
            f"the one the file it supersedes carries, because a stamp in "
            f"a receipt name is the run's, not the correction's"),
    }
    out.write_text(json.dumps(v2, indent=2, sort_keys=True) + "\n")

    after = RUNNER.resolve_day_chain(
        sorted(d.glob(RUNNER.sealed_day_receipt_glob(day))),
        kind="day receipt")
    resolved_to_me = (after["status"] in RUNNER.CHAIN_RESOLVED_STATUSES
                      and Path(after["head"]).name == out.name)
    if not resolved_to_me:
        raise CorrectionRefused(
            f"REFUSED after writing: the chain for {day} resolves to "
            f"{after.get('status')} / "
            f"{Path(after['head']).name if after.get('head') else None}, "
            f"not to {out.name}. A correction a reader does not resolve "
            f"to is an orphan file; it is reported, never reported as a "
            f"head.")
    return {"day": day, "path": str(out),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "supersedes": v2["supersedes"],
            "added_keys": sorted(set(additions) | {"supersedes",
                                                   "correction_census"}),
            "census": v2["correction_census"]["census"],
            "resolver_before": {"status": before["status"],
                                "head": Path(before["head"]).name},
            "resolver_after": {"status": after["status"],
                               "head": Path(after["head"]).name,
                               "n_matches": after["n_matches"],
                               "chain_head_is_this_file": resolved_to_me},
            "seal_census_n_keys_found": 0}


def emit_all(*, root: Path, repo: Path) -> int:
    rows = []
    for day in CORRECTION_TARGETS:
        rows.append(emit_correction(day, root=root, repo=repo))
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            print(f"  FAIL  {label}")
            raise SystemExit(f"[de_receipt_correction] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(fn, label, needle):
        try:
            fn()
        except (CorrectionRefused, BE.ReadRefused) as exc:
            if needle.lower() not in str(exc).lower():
                raise SystemExit(
                    f"[de_receipt_correction] FAIL: {label} -- refused "
                    f"for the wrong reason: {exc}")
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_receipt_correction] FAIL: {label} -- "
                         f"ADMITTED")

    import tempfile
    td = Path(tempfile.mkdtemp(prefix="de108_"))
    (td / "pm_5min/derived").mkdir(parents=True)
    base = {"protocol": "P003_DE_MULTIDAY_GATE1_DAY_RUN_V1",
            "emitted_at_utc": "2026-09-06T14:01:55.557479+00:00",
            "n_days_complete": 1, "G": 6,
            "per_day_sealed_artifacts": [{"arm": "A", "sealed": True}],
            "decision_populations": {"A": {"decisions": 10}},
            "work_counters": {"draws_performed_by_this_day": 1000},
            "battery": {"outcome": "PASS"},
            "memory_plan": {"budget_mb": 4000.0},
            "resources": {"wall_seconds": 1.0},
            "source_identity": {"carrying_commit": None},
            "provenance": {"params": {"path": "p", "sha256": "a" * 64}}}
    v1p = td / "pm_5min/derived/p003_de_gate1_day_run_20260903_SEALED__20260906T140155Z.json"
    v1p.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")

    v2 = build_correction(v1p, {"note_RECONSTRUCTED": {"x": 1}},
                          root=td)
    ok(v2["supersedes"]["sha256"] ==
       hashlib.sha256(v1p.read_bytes()).hexdigest()
       and v2["supersedes"]["path"] == str(v1p),
       "the correction carries `supersedes` as the PAIR, digest "
       "recomputed from the file it names")
    ok(all(json.dumps(v2[k], sort_keys=True) ==
           json.dumps(base[k], sort_keys=True) for k in base),
       f"and every key the v1 carried is BYTE-IDENTICAL in the .v2 "
       f"({len(base)} keys) -- the frozen set is READ from the v1, not "
       f"typed, so REV 79's named blocks are frozen by construction")
    fz = v2["correction_census"]["frozen_set"]
    ok(all(fz["rev79_named_blocks_present_and_frozen"].values())
       and set(fz["rev79_named_blocks_present_and_frozen"])
       <= set(REV79_NAMED_BLOCKS),
       f"and REV 79's named blocks are asserted to be INSIDE that frozen "
       f"set ({sorted(fz['rev79_named_blocks_present_and_frozen'])}) -- "
       f"the naming checks the construction rather than being it")
    ok(v2["correction_census"]["census_by"].startswith(
           "be_race_reader.correction_census"),
       "the census is BE's, IMPORTED -- one predicate for both seats, so "
       "a correction that passes here passes there for the same reason")

    refuses(lambda: build_correction(v1p, {"battery": {"outcome": "X"}},
                                     root=td),
            "KNOWN-BAD: an 'addition' the v1 ALREADY CARRIES is refused -- "
            "a correction adds, it does not redefine what is recorded",
            "already carries it")
    refuses(lambda: build_correction(
                v1p, {"Z": 1.0}, root=td),
            "KNOWN-BAD: a correction carrying a SEALED NAME as a key is "
            "refused BEFORE it is written", "a name sealed under")

    # ===== DE 111 / R-716: THE CENSUS IS TWO-SCOPED ====================
    _now = scope_at_this_correction()
    ok(_now["n"] == 11 and _now["design_version"] >= 23
       and set(RUNNER.economic_fields_in_force(22)) < set(
           _now["sealed_names"]),
       f"the scope at THIS correction's emit is RESOLVED, not read from a "
       f"literal: v{_now['design_version']} ({_now['n']} names) -- the "
       f"wider of the module constant v{_now['module_constant']} and the "
       f"chain head v{_now['chain_head_version']}. REV 82 §2.3's finding "
       f"was that the census never looked here at all")
    # (a) AN ADDITION carrying one of the ELEVEN, under an EIGHT-scope v1.
    _v1scope = sealed_names_in_force_for(json.loads(v1p.read_text()))
    ok(_v1scope["n"] == 8,
       f"and the fixture v1 is an EIGHT-scope receipt "
       f"(design v{_v1scope['design_version']}), which is the case the "
       f"whole ruling is about: open under its own list, sealed under "
       f"today's")
    refuses(lambda: build_correction(
                v1p, {"note_RECONSTRUCTED": {"n_fills_arm": 7}}, root=td),
            "KNOWN-BAD (a): an ADDITION carrying `n_fills_arm` -- OPEN "
            "under the v1's eight, SEALED under today's eleven -- is "
            "REFUSED NAMING THE KEY. This is exactly what DE 109's "
            "past-scoped union would have admitted",
            "n_fills_arm")
    # (b) AN INHERITED outcome count under an EIGHT-scope v1: REPORTED.
    _inh = json.loads(v1p.read_text())
    _inh["per_day_sealed_artifacts"] = [
        {"arm": "A", "counts": {"n_fills_arm": 11, "n_cancels_issued": 3}}]
    # ITS OWN TEMP ROOT (DE 111 second item, applied to itself): these
    # fixtures are extra SEALED receipts, and dropped into `td` they moved
    # the day-count cells below -- a cell measuring against what ran
    # before it.
    _td2 = Path(tempfile.mkdtemp(prefix="de111_"))
    (_td2 / "pm_5min/derived").mkdir(parents=True)
    _ip = _td2 / ("pm_5min/derived/p003_de_gate1_day_run_20260903_SEALED__"
                  "20260906T140156Z.json")
    _ip.write_text(json.dumps(_inh, indent=2, sort_keys=True) + "\n")
    _iv2 = build_correction(_ip, {"note_RECONSTRUCTED": {"x": 1}},
                            root=_td2)
    _sc = _iv2["correction_census"]["seal_census_by_KEY_before_writing"]
    ok(_sc["inherited"]["n_sealed_keys_found"] == 0
       and _sc["inherited"]["n_open_under_v1_but_sealed_today"] == 2
       and sorted(_sc["inherited"]["open_under_v1_but_sealed_today"]) == [
           "per_day_sealed_artifacts[0].counts.n_cancels_issued",
           "per_day_sealed_artifacts[0].counts.n_fills_arm"]
       and _sc["added"]["n_sealed_keys_found"] == 0,
       f"POSITIVE CONTROL (b): the SAME two names INHERITED byte-identical "
       f"from an eight-scope v1 are NOT refused -- they are reported as "
       f"inherited-open with their paths "
       f"({_sc['inherited']['n_open_under_v1_but_sealed_today']} of them, "
       f"judged under v{_sc['inherited']['judged_under']['design_version']}"
       f"). DA 105's distinction: the frozen blocks may not change, so "
       f"refusing here would forbid correcting the receipt at all")
    # (c) AN ELEVEN-scope v1: an outcome count ANYWHERE refuses.
    _dg = (Path(RUNNER.DR.resolve()["data_root"]) / "pm_5min/derived"
           / "p003_de_multiday_gate1_design_v23.json")
    _e11 = dict(_inh, provenance={"design": {
        "path": "data/pm_5min/derived/p003_de_multiday_gate1_design_v23.json",
        "sha256": hashlib.sha256(_dg.read_bytes()).hexdigest()}})
    _ep = _td2 / ("pm_5min/derived/p003_de_gate1_day_run_20260903_SEALED__"
                  "20260906T140157Z.json")
    _ep.write_text(json.dumps(_e11, indent=2, sort_keys=True) + "\n")
    ok(sealed_names_in_force_for(_e11)["n"] == 11,
       "and a v1 whose `provenance.design` pair resolves v23 is an "
       "ELEVEN-scope receipt -- the other side of the ruling")
    refuses(lambda: build_correction(_ep, {"note_RECONSTRUCTED": {"x": 1}},
                                     root=_td2),
            "KNOWN-BAD (c): a correction of an ELEVEN-scope receipt "
            "carrying an outcome count ANYWHERE -- inherited or added -- "
            "is REFUSED. Under its own scope the name was never open, so "
            "there is nothing to inherit",
            "n_fills_arm")
    refuses(lambda: build_correction(
                v1p, {"sha256_at_load": "RECONSTRUCTED"}, root=td),
            "KNOWN-BAD: a reconstruction under a PLAIN name is refused -- "
            "the status rides in the KEY", "PLAIN name")

    nd = n_days_complete_true_at_emit(v1p, td)
    ok(nd["n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED"] == 1
       and nd["days_sealed_at_or_before_this_emit"] == ["20260903"]
       and "own `emitted_at_utc`" in nd["method"],
       f"the day count TRUE AT EMIT is reconstructed from the sealed "
       f"receipts whose OWN `emitted_at_utc` is at or before this one's "
       f"({nd['n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED']}), with the "
       f"method named INSIDE the artifact")
    later = td / "pm_5min/derived/p003_de_gate1_day_run_20260904_SEALED__20260906T163351Z.json"
    later.write_text(json.dumps(
        dict(base, day="2026-09-04",
             emitted_at_utc="2026-09-06T16:33:51.100000+00:00")) + "\n")
    nd2 = n_days_complete_true_at_emit(later, td)
    ok(nd2["n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED"] == 2
       and nd2["days_sealed_at_or_before_this_emit"] == ["20260903",
                                                         "20260904"],
       "and a LATER receipt reconstructs 2 -- the count is a function of "
       "the emit stamp, not of how many files exist now")
    # THE FILENAME IS A PROXY, AND THIS IS THE CASE THAT SEPARATES THEM: a
    # .vN correction is NAMED with the correction's stamp and CARRIES the
    # run's. Counting by the filename would let a correction of a LATER
    # day raise an EARLIER receipt's count.
    corr = (td / "pm_5min/derived/p003_de_gate1_day_run_20260904_SEALED"
                 "__20260906T171144Z.v2.json")
    corr.write_text(json.dumps(
        dict(base, day="2026-09-04",
             emitted_at_utc="2026-09-06T16:33:51.100000+00:00")) + "\n")
    nd3 = n_days_complete_true_at_emit(v1p, td)
    ok(nd3["n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED"] == 1
       and nd3["days_sealed_at_or_before_this_emit"] == ["20260903"]
       and any(r["file"].endswith(".v2.json") and r["counted"] is False
               for r in nd3["all_sealed_receipts_considered"])
       and nd3["filename_stamp_cross_check"][
           "n_disagreeing_with_their_own_emitted_at_utc"] == 1,
       f"KNOWN-BAD FOR THE PROXY: a 09-04 `.v2` stamped 171144Z but "
       f"carrying the RUN's 16:33:51Z does NOT raise the 09-03 "
       f"receipt's count ({nd3['n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED']}"
       f"), and the 1 filename/emit disagreement is REPORTED")
    nd4 = n_days_complete_true_at_emit(corr, td)
    ok(nd4["n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED"] == 2
       and nd4["days_sealed_at_or_before_this_emit"] == ["20260903",
                                                         "20260904"],
       "and the correction itself reconstructs 2, not 3: the DAY is the "
       "key, so a `.vN` of a day already counted does not count it twice")
    (td / "pm_5min/derived/p003_de_gate1_day_run_20260905_SEALED"
          "__20260906T175550Z.json").write_text("{not json")
    refuses(lambda: n_days_complete_true_at_emit(v1p, td),
            "KNOWN-BAD: an UNREADABLE sealed receipt REFUSES the count -- "
            "a count over the receipts that happened to parse is not a "
            "count (rule 11: absence must never read as a pass)",
            "could not be read")
    (td / "pm_5min/derived/p003_de_gate1_day_run_20260905_SEALED"
          "__20260906T175550Z.json").unlink()
    ok(v2["n_days_complete"] == base["n_days_complete"],
       "and the EMITTED `n_days_complete` is left UNTOUCHED: the "
       "reconstruction is a new key beside it, never a repair of it")
    sc = v2["correction_census"]["sealed_names_in_force_at_v1s_emit"]
    ok(sc["n"] == 8 and "POSITIVE" in sc["read_from"],
       f"the sealed list is the one IN FORCE AT THE v1'S EMIT "
       f"({sc['n']} names, design v{sc['design_version']}, recognised "
       f"POSITIVELY by its emit stamp) -- a receipt cannot have disobeyed "
       f"a rule that did not exist when it was written")

    # ---- THE TWO NEW REFUSALS (DE 109) ------------------------------
    refuses(lambda: build_correction(
                v1p, {"prov_RECONSTRUCTED": {"design": {
                    "path": "d", "sha256": "b" * 64}}}, root=td),
            "KNOWN-BAD: a PLAIN digest name inside an ADDED block is "
            "refused ON THE NAME -- nothing in an addition was measured, "
            "so a bare hex under `sha256` is a reconstruction in "
            "disguise (the value never says so)",
            "PLAIN digest name")
    v2ok = build_correction(
        v1p, {"prov_RECONSTRUCTED": {"design": {
            "path": "d",
            "sha256_AT_THE_CARRYING_COMMIT_RECONSTRUCTED": "b" * 64}}},
        root=td)
    ok(v2ok["correction_census"]["no_plain_digest_in_an_added_block"]
       ["none_present"] is True
       and v2ok["supersedes"]["sha256"],
       "POSITIVE CONTROL for it: the SAME block with the digest named "
       "`..._AT_THE_CARRYING_COMMIT_RECONSTRUCTED` is ADMITTED, and "
       "`supersedes`'s own pair is exempt by name -- the check fires on "
       "the bad case AND admits the good one")

    # ---- REV 79 S1.3, RE-DRIVEN AT BE'S LANDED CODE (DE 109) ---------
    # DE 108 pinned this cell to the ADMIT of the moment. BE 75 (f414f05)
    # landed the one-line fix; the cell now pins the BINDING gate, in both
    # directions, through THIS module's import of BE and on a RECEIPT-
    # shaped fixture with THIS module's own declared additions.
    g1 = {b: {"fixture": b} for b in BE.FROZEN_BLOCKS}
    g1.update({"protocol": base["protocol"], "n_days_complete": 1})
    g_added = ("provenance", "n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED",
               "supersedes")
    g_add = {"provenance": {"params": {"path": "p"}},
             "n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED": 1,
             "supersedes": {"path": "v1.json", "sha256": "a" * 64}}
    gc = BE.correction_census(g1, {**g1, **g_add}, added=g_added)
    ok(gc["difference_is_exactly_the_additions"] is True
       and gc["missing_declared_additions"] == []
       and sorted(gc["keys_changed_vs_v1"]) == sorted(g_added),
       f"REV 79 S1.3 POSITIVE CONTROL at BE's landed code, through this "
       f"module's import: the COMPLETE .v2 is ADMITTED "
       f"({sorted(gc['keys_changed_vs_v1'])}), difference_is_exactly_"
       f"the_additions True, missing []")
    for _m in g_added:
        _lack = {**g1, **{k: v for k, v in g_add.items() if k != _m}}
        try:
            BE.correction_census(g1, _lack, added=g_added)
            raise SystemExit(
                f"[de_receipt_correction] FAIL: REV 79 S1.3 KNOWN-BAD "
                f"({_m}) -- ADMITTED. The gate does not bind.")
        except BE.ReadRefused as _e:
            ok(_m in str(_e) and "missing declared addition" in str(_e),
               f"REV 79 S1.3 KNOWN-BAD: a `.v2` OMITTING the declared "
               f"addition {_m!r} is REFUSED, and the refusal NAMES it")
    ok(True,
       "so the gate BINDS at f414f05 in both directions and THIS SEAT "
       "MAY EMIT -- DE 108's cell pinned the ADMIT of that moment and is "
       "superseded by this drive, not deleted")

    print(f"[de_receipt_correction] PASS -- {n[0]} checks")
    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(
            f"[de_receipt_correction] FAIL: {n[0]} checks run against a "
            f"declared {EXPECTED_CHECKS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.emit:
        # THE BATTERY RUNS FIRST, ALWAYS. An emitter whose own checks have
        # not been driven at this tip is an emitter nobody has checked.
        rc = selftest()
        if rc != 0:
            return rc
        repo = Path(__file__).resolve().parents[2]
        return emit_all(root=Path(RUNNER.DR.resolve()["data_root"]),
                        repo=repo)
    ap.error("--selftest or --emit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
