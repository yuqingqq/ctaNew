#!/usr/bin/env python3
"""RELOCATE A SEALED FORWARD-DAY ARTIFACT TO A DURABLE PATH. THE SEAL IS NEVER
OPENED.

WHY THIS IS AN INSTRUMENT AND NOT A `cp` PLUS A HAND-EDITED JSON. The thing
being moved is the only copy of a scored race day. Every claim a relocation
receipt makes -- "byte-identical", "the old receipt is untouched", "no metric
crossed over" -- is a predicate, and rule 10 says a predicate is computed or it
is not claimed. Three of them are checkable only by comparing bytes that no
reader of the receipt will ever have, so the receipt must be written by the
thing that did the comparing.

WHAT IT REFUSES TO DO. It never reads the sealed file's CONTENT: the seal is
hashed and sized, never parsed. It never edits the receipt it supersedes
(rule 13). It never writes a destination that already holds DIFFERENT bytes.
And it refuses -- by name -- when the receipt's declared sha does not describe
the file at the receipt's declared path, because a receipt that does not
describe its own artifact is the one case where copying is the wrong move.

R-496 (B) is the finding this exists for: a sealed artifact whose only copy is
in a session-scoped temp directory is one /tmp sweep from voiding a race day.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import be_forward_day as BFD


class SealRelocateRefused(RuntimeError):
    """A named refusal. Absence and quiet are never a pass (rule 11)."""


CHUNK = 1 << 20


def sha_file(p: Path) -> str:
    """Streaming sha256. A 54 MB artifact is not read into memory to be
    hashed, and -- the point -- it is not read into memory at all: hashing is
    the only contact this module has with the seal."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _as_of() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def declared_seal(receipt: Path) -> dict:
    """The `sealed_file` block of a run receipt, as the RUN wrote it.

    Read from the receipt rather than passed in, so the relocation is anchored
    to what the producer declared and not to what the operator remembers."""
    if not receipt.exists():
        raise SealRelocateRefused(
            f"REFUSED: no receipt at {receipt}. The declared sha comes from "
            f"the producing run's own receipt; without it there is nothing to "
            f"verify the copy AGAINST, and a copy verified against itself "
            f"proves nothing.")
    rec = json.loads(receipt.read_text())
    sf = rec.get("sealed_file")
    if not isinstance(sf, dict) or not sf.get("sha256") or not sf.get("path"):
        raise SealRelocateRefused(
            f"REFUSED: {receipt} carries no usable `sealed_file` block "
            f"(got {sf!r}). Outcome was {rec.get('outcome')!r}: a REFUSED run "
            f"seals nothing, and there is no artifact to relocate.")
    return {"path": sf["path"], "sha256": sf["sha256"],
            "bytes": sf.get("bytes"), "outcome": rec.get("outcome"),
            "as_of_utc": rec.get("as_of_utc")}


def verify_against_declaration(src: Path, decl: dict) -> dict:
    """Does the file at the receipt's declared path match its declared sha?

    Checked BEFORE the copy. If a receipt does not describe its own artifact,
    relocating the artifact would propagate the disagreement to a second path
    and give it a fresh timestamp."""
    if not src.exists():
        raise SealRelocateRefused(
            f"REFUSED: the receipt declares {src} and no file is there.")
    got = sha_file(src)
    size = src.stat().st_size
    if got != decl["sha256"]:
        raise SealRelocateRefused(
            f"REFUSED: {src} hashes to {got} but its own receipt declares "
            f"{decl['sha256']}. The receipt does not describe the file; "
            f"relocating it would copy the disagreement, not the artifact.")
    if decl.get("bytes") is not None and size != decl["bytes"]:
        raise SealRelocateRefused(
            f"REFUSED: {src} is {size} B; the receipt declares "
            f"{decl['bytes']} B. Equal shas with unequal sizes is not a "
            f"disagreement this module resolves.")
    return {"path": str(src), "sha256": got, "bytes": size,
            "matches_declaration": True}


def place(src: Path, dst: Path, declared_sha: str) -> dict:
    """Put the bytes at `dst` and PROVE they are the same bytes.

    Idempotent BY COMPARISON, never by assumption: a destination that already
    exists is hashed, and it is accepted only if it already equals the
    declared sha. Different bytes at the destination refuse -- overwriting is
    how the 12:49 receipt was lost, and a seal is worth more than a receipt."""
    pre_existing = dst.exists()
    if pre_existing:
        got = sha_file(dst)
        if got != declared_sha:
            raise SealRelocateRefused(
                f"REFUSED: {dst} already exists and hashes to {got}, not the "
                f"declared {declared_sha}. This module never overwrites a "
                f"destination that holds different bytes.")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    after = sha_file(dst)
    if after != declared_sha:
        raise SealRelocateRefused(
            f"REFUSED: after placement {dst} hashes to {after}, not the "
            f"declared {declared_sha}. The copy is not the artifact.")
    return {
        "path": str(dst), "sha256": after, "bytes": dst.stat().st_size,
        "pre_existing": pre_existing,
        "verified": "sha256 recomputed at the destination AFTER placement, "
                    "compared to the sha the producing run declared",
        "seal_opened": False,
        "seal_contact": "hashed and sized only; the file is never parsed and "
                        "no field of it is read"}


def relocate(day: str, receipt_v1: Path, dst_dir: Path,
             durability: dict = None, note: str = None,
             also_supersedes: list = None) -> dict:
    """The act, and the receipt that supersedes (rule 13).

    `receipt_v1` is READ and never written. The v2 receipt is a NEW file; the
    v1 receipt's sha is recorded before and after so "left untouched" is a
    measurement rather than an intention.

    `also_supersedes` exists because a day can have TWO predecessors that are
    not the same document. 09-02's receipt at the canonical path is a REFUSAL
    written before the day closed; it declares no seal, so it cannot be the
    receipt this act verifies the copy against -- that must be the SCORED
    run's own receipt. Both are predecessors, and collapsing them into one
    field would make the v2 receipt claim the refusal declared a sealed file.
    Every receipt named here is read-only and its byte-identity is asserted
    after the act, exactly as `receipt_v1`'s is."""
    # ORDER MATTERS AND THE SUITE PROVED IT: hashing the v1 receipt before
    # establishing that it EXISTS died by `FileNotFoundError` rather than
    # refusing by name. A refusal must be by name and never a traceback
    # (DA20-R3's class, R-495 (D)). `declared_seal` is the existence gate, so
    # it runs first and the hash is taken only of a file already proved there.
    # NORMALISED ONCE, at the top. `dst_dir` was converted at two of its
    # three uses and left raw at the third, so a caller passing a string got
    # a `TypeError` from inside the chaining block -- a traceback where this
    # programme requires a refusal by name. Converting here removes the
    # inconsistency rather than adding a third conversion.
    dst_dir = Path(dst_dir)
    receipt_v1 = Path(receipt_v1)
    decl = declared_seal(receipt_v1)
    v1_before = sha_file(receipt_v1)
    src = Path(decl["path"])
    checked = verify_against_declaration(src, decl)
    dst = dst_dir / src.name
    placed = place(src, dst, decl["sha256"])

    rec = {
        "protocol": "BE_FORWARD_DAY_SEAL_RELOCATION_V2",
        "supersedes_version": 1,
        "day": day,
        "as_of_utc": _as_of(),
        "sealed": True,
        "seal_state": "SEALED -- NOT OPENED BY THIS ACT",
        "sealing_note": "this receipt carries paths, sizes and hashes and NO "
                        "metric (rule 11). It records a RELOCATION, not a "
                        "run: no scoring happened, no row was built, and the "
                        "sealed file was hashed but never parsed. Unsealing "
                        "remains the coordinator's or the USER's act.",
        "what_this_receipt_changes": "the DURABLE LOCATION of the sealed "
                                     "artifact, and nothing else. The scores, "
                                     "the run that produced them and the "
                                     "receipt that reported it are unchanged.",
        "sealed_file": {
            "path": placed["path"], "sha256": placed["sha256"],
            "bytes": placed["bytes"],
            "contents": "per-action scores and the full complement report",
            "not_in_receipt": "no metric, rho, net value or sign appears "
                              "outside this file"},
        "sealed_file_previous_location": {
            "path": checked["path"], "sha256": checked["sha256"],
            "bytes": checked["bytes"],
            "status": "still present; NOT deleted by this act",
            "why_not_deleted": "durability is achieved by the presence of the "
                               "durable copy, not by the absence of the "
                               "temporary one. Deleting is irreversible and "
                               "buys nothing; the authoritative path is the "
                               "one this receipt names."},
        "identity": {
            "declared_by_producing_run": decl["sha256"],
            "measured_at_source": checked["sha256"],
            "measured_at_destination": placed["sha256"],
            "all_three_equal": (decl["sha256"] == checked["sha256"]
                                == placed["sha256"]),
            "destination_pre_existing": placed["pre_existing"]},
        "supersedes_receipt": {
            "path": str(receipt_v1),
            "sha256_before_this_act": v1_before,
            "outcome": decl["outcome"],
            "as_of_utc": decl["as_of_utc"],
            "status": "UNTOUCHED, kept as provenance (rule 13). A frozen "
                      "artifact is never edited; the correction supersedes "
                      "IN-BAND as this vN+1 receipt, and the v1 receipt "
                      "remains the record of what the producing run wrote, "
                      "including the /tmp path it then named."},
        "producing_code": BFD._provenance(),
        "relocating_code": {
            "module": "be_seal_relocate.py",
            "sha256_prefix": sha_file(
                Path(__file__).resolve())[:16]},
    }
    _extra_before = {}
    if also_supersedes:
        extra = []
        for q in also_supersedes:
            q = Path(q)
            if not q.exists():
                raise SealRelocateRefused(
                    f"REFUSED: `also_supersedes` names {q} and no file is "
                    f"there. A predecessor that cannot be hashed cannot be "
                    f"recorded as one; absence is not a pass (rule 11).")
            sh = sha_file(q)
            _extra_before[str(q)] = sh
            body = {}
            try:
                body = json.loads(q.read_text())
            except ValueError:
                pass
            extra.append({
                "path": str(q), "sha256": sh,
                "outcome": body.get("outcome"),
                "as_of_utc": body.get("as_of_utc"),
                "declares_a_sealed_file": isinstance(
                    body.get("sealed_file"), dict),
                "status": "UNTOUCHED, kept as provenance (rule 13)"})
        rec["supersedes_receipts_additional"] = extra
    if durability:
        rec["durability_finding"] = durability
    if note:
        rec["note"] = note

    # Computed, then attached -- the check describes the receipt it examined,
    # not the receipt plus its own verdict. Borrowed from the driver rather
    # than restated, so the vocabulary cannot drift between the two.
    rec["decision_field_check"] = BFD.assert_no_decision_field(rec)

    out = dst_dir / f"be_forward_day_receipt_{day}.v2.json"
    if out.exists():
        # BE5-R1's class, REPEATED HERE AND CAUGHT BY USING IT: the first
        # version of this function OVERWROTE the v2 path and recorded only
        # the sha of what it had just destroyed -- a provenance pointer to
        # bytes that exist nowhere. The driver's `_flush` was fixed for
        # exactly this; a second implementation is not excused by the first
        # one's repair. The incumbent v2 is MOVED ASIDE to the next free
        # number and kept byte-identical, and the new receipt names the file
        # it actually stands after, so the chain is readable from the
        # receipts themselves rather than from a hash with no referent.
        n = 1
        while (dst_dir / f"be_forward_day_receipt_{day}.v2.{n}.json").exists():
            n += 1
        kept = dst_dir / f"be_forward_day_receipt_{day}.v2.{n}.json"
        prior_sha = sha_file(out)
        shutil.move(str(out), str(kept))
        moved_sha = sha_file(kept)
        if moved_sha != prior_sha:
            raise SealRelocateRefused(
                f"REFUSED: preserving the prior v2 changed it "
                f"({prior_sha} -> {moved_sha}). A predecessor kept as "
                f"provenance must be byte-identical (rule 13).")
        rec["replaces_prior_v2"] = {
            "path": str(kept), "sha256": prior_sha, "n_prior": n,
            "why": "the v2 this act STANDS AFTER, preserved under the next "
                   "free number rather than overwritten. A superseded "
                   "receipt is evidence; evidence is not a scratch buffer."}
    out.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))

    for q, before in _extra_before.items():
        after = sha_file(Path(q))
        if after != before:
            raise SealRelocateRefused(
                f"REFUSED: the superseded receipt {q} CHANGED during this act "
                f"({before} -> {after}). Rule 13: a predecessor kept as "
                f"provenance is never edited.")
    v1_after = sha_file(receipt_v1)
    if v1_after != v1_before:
        raise SealRelocateRefused(
            f"REFUSED: the superseded receipt {receipt_v1} CHANGED during "
            f"this act ({v1_before} -> {v1_after}). Rule 13: the old receipt "
            f"stays as provenance and is never edited.")
    return {"receipt_v2": str(out), "receipt_v2_sha256": sha_file(out),
            "v1_unchanged": True, "v1_sha256": v1_after,
            "sealed_file": rec["sealed_file"],
            "all_three_equal": rec["identity"]["all_three_equal"]}


# ---------------------------------------------------------------------------
# SELFTEST. Rule 15: every checker ships a falsifier -- a known-bad it must
# REFUSE and a positive control it must ADMIT. SEAT_PROTOCOL rule 16: a
# control that only ever refuses proves nothing about the good case, so every
# control below is driven in BOTH directions.
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 41


def _mk_pair(root: Path, day: str, body: bytes,
             declared_sha: str = None) -> tuple:
    """A synthetic (sealed file, v1 receipt) pair in a temp tree.

    The receipt is built from the file's REAL sha unless the caller overrides
    it -- the override is what makes the "receipt does not describe its file"
    known-bad expressible without hand-writing a hash."""
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    sealed = src_dir / f"be_forward_day_SEALED_scores_{day}.json"
    sealed.write_bytes(body)
    real = sha_file(sealed)
    rec = {"protocol": "BE_FORWARD_DAY_SEALED_V1", "day": day,
           "outcome": "SCORED", "as_of_utc": "2026-01-01T00:00:00Z",
           "sealed_file": {"path": str(sealed),
                           "sha256": declared_sha or real,
                           "bytes": len(body)}}
    r1 = src_dir / f"be_forward_day_receipt_{day}.json"
    r1.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return sealed, r1, real


def selftest() -> int:
    import tempfile
    import traceback
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        if cond:
            print(f"PASS: {label}")
        else:
            fails.append(label)
            print(f"FAIL: {label}")

    def refuses(fn, want_substr, label):
        """A known-bad must refuse BY NAME. A traceback is not a refusal, and
        a refusal whose message does not name the cause sends the reader to
        the wrong place (BE6-R1's class)."""
        nonlocal checks
        checks += 1
        try:
            fn()
        except SealRelocateRefused as e:
            if want_substr in str(e):
                print(f"PASS: {label}")
                return
            fails.append(f"{label} [refused, WRONG cause: {str(e)[:120]}]")
            print(f"FAIL: {label} -- refused but not for {want_substr!r}")
            return
        except Exception as e:                        # noqa: BLE001
            fails.append(f"{label} [{type(e).__name__}, not a named refusal]")
            print(f"FAIL: {label} -- {type(e).__name__}: {str(e)[:120]}")
            print(traceback.format_exc()[-400:])
            return
        fails.append(f"{label} [ACCEPTED the known-bad]")
        print(f"FAIL: {label} -- the known-bad was ACCEPTED")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # -- POSITIVE CONTROL: the good case must be ADMITTED. A module that
        # only ever refuses would pass every known-bad below and be useless.
        body = b'{"SEALED": "synthetic", "per_coin_scores": {"btc": [[1,2]]}}'
        sealed, r1, real = _mk_pair(root / "good", "20260101", body)
        dst = root / "good" / "dst"
        r1_before = sha_file(r1)
        out = relocate("20260101", r1, dst)
        ok(out["all_three_equal"] is True,
           "POSITIVE CONTROL: a good pair RELOCATES -- declared, source and "
           "destination shas all equal")
        placed = dst / sealed.name
        ok(placed.exists(), "POSITIVE CONTROL: the destination file exists")
        ok(sha_file(placed) == real,
           "POSITIVE CONTROL: the destination is byte-identical to the source")
        ok(placed.stat().st_size == len(body),
           "POSITIVE CONTROL: the destination size equals the source size")
        v2 = Path(out["receipt_v2"])
        ok(v2.exists(), "POSITIVE CONTROL: the v2 receipt was written")
        v2d = json.loads(v2.read_text())
        ok(v2d["protocol"] == "BE_FORWARD_DAY_SEAL_RELOCATION_V2",
           "the v2 receipt declares its own protocol")
        ok(v2d["sealed_file"]["path"] == str(placed),
           "the v2 receipt names the NEW path")
        ok(v2d["supersedes_receipt"]["path"] == str(r1),
           "the v2 receipt names the receipt it supersedes")
        ok(v2d["supersedes_receipt"]["sha256_before_this_act"] == r1_before,
           "the v2 receipt records the superseded receipt's sha AS MEASURED")

        # -- v1 UNTOUCHED, measured rather than intended (rule 13).
        ok(sha_file(r1) == r1_before,
           "rule 13: the superseded v1 receipt is byte-identical after the act")
        ok(json.loads(r1.read_text())["sealed_file"]["path"] == str(sealed),
           "rule 13: v1 still names the OLD path -- it is provenance, not a "
           "correction target")

        # -- NO METRIC CROSSED OVER (rule 11), and the check is the driver's
        # own, borrowed by import so the two vocabularies cannot drift.
        ok(isinstance(v2d.get("decision_field_check"), dict),
           "the v2 receipt carries the driver's decision-field post-condition")
        ok("per_coin_scores" not in v2.read_text(),
           "rule 11: no scored content appears in the v2 receipt")
        ok(v2d["sealed"] is True and "NOT OPENED" in v2d["seal_state"],
           "the v2 receipt states the seal was not opened")

        # -- THE SEAL IS HASHED, NEVER PARSED. The positive control for that
        # claim is a sealed file that is NOT valid JSON: if any code path
        # parsed it, this relocation would raise instead of succeeding.
        bad_json = b"\xff\xfe not json at all \x00 {{{"
        s2, r2, real2 = _mk_pair(root / "unparseable", "20260102", bad_json)
        out2 = relocate("20260102", r2, root / "unparseable" / "dst")
        ok(out2["all_three_equal"] is True,
           "POSITIVE CONTROL: an UNPARSEABLE sealed file relocates -- proving "
           "the seal is hashed and never parsed")
        ok(sha_file(root / "unparseable" / "dst" / s2.name) == real2,
           "the unparseable seal arrived byte-identical")

        # -- IDEMPOTENT BY COMPARISON, not by assumption.
        out3 = relocate("20260101", r1, dst)
        ok(out3["all_three_equal"] is True,
           "a second relocation of identical bytes SUCCEEDS")
        ok(json.loads(Path(out3["receipt_v2"]).read_text())
           ["identity"]["destination_pre_existing"] is True,
           "the second relocation RECORDS that the destination pre-existed")
        rec3 = json.loads(Path(out3["receipt_v2"]).read_text())
        ok("replaces_prior_v2" in rec3,
           "the second v2 receipt names the v2 it stands after")
        kept = Path(rec3["replaces_prior_v2"]["path"])
        ok(kept.exists() and kept != Path(out3["receipt_v2"]),
           "the superseded v2 STILL EXISTS at its own numbered path -- a "
           "provenance sha with no referent is not provenance")
        ok(sha_file(kept) == rec3["replaces_prior_v2"]["sha256"],
           "the preserved predecessor is byte-identical to what the new "
           "receipt says it was")
        ok(json.loads(kept.read_text())["sealed_file"]["path"]
           == str(dst / sealed.name),
           "POSITIVE CONTROL: the preserved predecessor is readable and is "
           "the earlier relocation receipt, not a stub")

        # -- also_supersedes: BOTH directions. A day can have a predecessor
        # that declares no seal (09-02's refusal), and recording it must not
        # let the v2 claim that predecessor declared one.
        s9, r9, _ = _mk_pair(root / "extra", "20260109", b"sealed bytes")
        ref = root / "extra" / "refusal.json"
        ref.write_text(json.dumps({"day": "20260109", "outcome": "REFUSED",
                                   "as_of_utc": "2026-01-09T00:00:00Z"}))
        ref_before = sha_file(ref)
        out9 = relocate("20260109", r9, root / "extra" / "dst",
                        also_supersedes=[ref])
        r9d = json.loads(Path(out9["receipt_v2"]).read_text())
        extra = r9d["supersedes_receipts_additional"]
        ok(len(extra) == 1 and extra[0]["path"] == str(ref),
           "POSITIVE CONTROL: `also_supersedes` records the extra predecessor")
        ok(extra[0]["sha256"] == ref_before and sha_file(ref) == ref_before,
           "the extra predecessor is recorded by sha and left byte-identical")
        ok(extra[0]["declares_a_sealed_file"] is False
           and extra[0]["outcome"] == "REFUSED",
           "the v2 records that the extra predecessor declares NO seal -- a "
           "refusal is never reported as having sealed something")
        refuses(lambda: relocate("20260109", r9, root / "extra" / "dst2",
                                 also_supersedes=[root / "extra" / "nope.json"]),
                "and no file is there",
                "KNOWN-BAD: an `also_supersedes` path with no file is REFUSED "
                "by name")

        # -- STRING PATHS: accepted, and the same receipt results. Both
        # directions -- a str `dst_dir` must WORK, not merely not-crash.
        s10, r10, real10 = _mk_pair(root / "strpath", "20260110", b"str bytes")
        d10 = root / "strpath" / "dst"
        out10 = relocate("20260110", str(r10), str(d10))
        ok(out10["all_three_equal"] is True,
           "POSITIVE CONTROL: string `dst_dir` and `receipt_v1` are accepted "
           "and relocate identically to Path arguments")
        ok(sha_file(d10 / s10.name) == real10,
           "the string-argument relocation placed the same bytes")

        # -- KNOWN-BAD 1: a destination holding DIFFERENT bytes.
        s4, r4, _ = _mk_pair(root / "clash", "20260103", b"original bytes")
        d4 = root / "clash" / "dst"
        d4.mkdir(parents=True)
        (d4 / s4.name).write_bytes(b"DIFFERENT bytes at the destination")
        refuses(lambda: relocate("20260103", r4, d4),
                "already exists and hashes to",
                "KNOWN-BAD: a destination holding different bytes is REFUSED, "
                "never overwritten")
        ok((d4 / s4.name).read_bytes() == b"DIFFERENT bytes at the destination",
           "the refused destination was left untouched")

        # -- KNOWN-BAD 2: the receipt does not describe its own file.
        s5, r5, _ = _mk_pair(root / "liar", "20260104", b"real bytes",
                             declared_sha="0" * 64)
        refuses(lambda: relocate("20260104", r5, root / "liar" / "dst"),
                "does not describe the file",
                "KNOWN-BAD: a receipt whose declared sha does not match its "
                "own file is REFUSED before any copy")
        ok(not (root / "liar" / "dst").exists(),
           "nothing was copied when the declaration failed")

        # -- KNOWN-BAD 3: the declared file is missing.
        s6, r6, _ = _mk_pair(root / "gone", "20260105", b"soon deleted")
        s6.unlink()
        refuses(lambda: relocate("20260105", r6, root / "gone" / "dst"),
                "and no file is there",
                "KNOWN-BAD: a declared path with no file is REFUSED by name")

        # -- KNOWN-BAD 4: a REFUSED run seals nothing.
        rr = root / "refused"
        rr.mkdir()
        r7 = rr / "be_forward_day_receipt_20260106.json"
        r7.write_text(json.dumps({"day": "20260106", "outcome": "REFUSED",
                                  "refused_at": "day_closed_and_attributed"}))
        refuses(lambda: relocate("20260106", r7, rr / "dst"),
                "carries no usable `sealed_file` block",
                "KNOWN-BAD: a REFUSED run's receipt has no artifact to "
                "relocate and is REFUSED by name")

        # -- KNOWN-BAD 5: no receipt at all. Absence is not a pass (rule 11).
        refuses(lambda: relocate("20260107", rr / "nope.json", rr / "dst2"),
                "no receipt at",
                "KNOWN-BAD: a missing receipt is REFUSED -- the copy is never "
                "verified against itself")

        # -- KNOWN-BAD 6: declared bytes disagree with the file's size while
        # the sha matches. Refused rather than silently preferred either way.
        s8, r8, real8 = _mk_pair(root / "size", "20260108", b"twelve bytes")
        d8 = json.loads(r8.read_text())
        d8["sealed_file"]["bytes"] = 999999
        r8.write_text(json.dumps(d8, indent=1, sort_keys=True))
        refuses(lambda: relocate("20260108", r8, root / "size" / "dst"),
                "Equal shas with unequal sizes",
                "KNOWN-BAD: a size that contradicts the receipt is REFUSED")

        # -- THE FALSIFIER FOR THE POST-CONDITION ITSELF: a receipt carrying a
        # decision-shaped field must be caught by the borrowed check. Without
        # this, `decision_field_check` could be a field that never fires.
        try:
            BFD.assert_no_decision_field({"x": {"counts_toward_G": True}})
            ok(False, "KNOWN-BAD: the borrowed decision-field check FIRES on "
                      "a decision-shaped field")
        except BFD.ForwardDayRefused as e:
            ok("counts_toward_G" in str(e),
               "KNOWN-BAD: the borrowed decision-field check FIRES on a "
               "decision-shaped field, naming it")
        ok(isinstance(BFD.assert_no_decision_field({"x": {"n_rows": 3}}), dict),
           "POSITIVE CONTROL: the same check ADMITS an ordinary receipt")

        # -- sha_file is the identity this module rests on: both directions.
        a = root / "a.bin"
        a.write_bytes(b"x" * (CHUNK + 7))     # spans the streaming boundary
        b = root / "b.bin"
        b.write_bytes(b"x" * (CHUNK + 7))
        c = root / "c.bin"
        c.write_bytes(b"x" * (CHUNK + 7) + b"!")
        ok(sha_file(a) == sha_file(b),
           "sha_file: identical multi-chunk files hash equal")
        ok(sha_file(a) != sha_file(c),
           "sha_file: a one-byte difference across the chunk boundary is seen")
        ok(sha_file(a) == hashlib.sha256(a.read_bytes()).hexdigest(),
           "sha_file: the streaming hash equals the whole-file hash")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}. "
              f"A suite that silently shrinks reports a pass it did not earn "
              f"(DA20-R3's class).")
        return 1
    return 1 if fails else 0


def main(argv: list = None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    need = ("--day", "--receipt", "--dst")
    if not all(f in argv for f in need):
        # BE34-R4: usage returns 2, never 0. A misspelled flag must not be
        # recordable as a completed relocation.
        print("usage: be_seal_relocate.py --selftest | "
              "--day <YYYYMMDD> --receipt <v1 receipt> --dst <dir> "
              "[--note <text>]")
        return 2
    g = {f: argv[argv.index(f) + 1] for f in need}
    note = argv[argv.index("--note") + 1] if "--note" in argv else None
    try:
        out = relocate(g["--day"], Path(g["--receipt"]), Path(g["--dst"]),
                       note=note)
    except SealRelocateRefused as e:
        print(str(e))
        return 1
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
