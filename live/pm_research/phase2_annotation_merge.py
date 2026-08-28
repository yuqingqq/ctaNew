#!/usr/bin/env python3
"""Receipt-side implementation of the annotation-survival contract (blocker 6).

CONTRACT OWNER: DA. Spec: plans/ANNOTATION_SURVIVAL_CONTRACT.md (9ae7358,
amended 417423a). This module implements the GENERATOR side only. It never
authors, edits, repairs, truncates or reconstructs annotator content -- it
validates identity and bindings, and merges verbatim or refuses.

WHY THE CANON IS PINNED. The first spec said "sha256 of content||binds_to" and
pinned no canonical form. Two correct implementations differing on key order,
whitespace, unicode escaping, float repr or what `||` concatenates would make
every VALID sidecar mismatch on recompute -- and that refusal is
indistinguishable from tampering. The mechanism built to preserve annotations
would drop them while reporting an attack. DA ruled `annotation_canon_v1`; this
implements it exactly, and an UNKNOWN canonical form refuses with a DISTINCT
cause so "we disagree about the recipe" is never reported as "someone tampered".

AGREEMENT IS PROVEN, NOT ASSUMED. merge() REFUSES TO RUN unless
assert_canon_agreement() has recomputed the owner's sha from the owner's REAL
COMMITTED SIDECAR BYTES and matched it. Two implementations never shown to agree
on bytes are not a mechanism.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
ANNOTATIONS = DERIVED / "annotations"

CANON = "annotation_canon_v1"
SCHEMA = "annotation_v1"
RESERVED = "RESERVED for Q-DA-79 post-gap queue-validity finding"

# binds_to keys that are NOT part of the population fingerprint
FINGERPRINT_IGNORE = ("population_independent", "note")


class MergeRefused(RuntimeError):
    """Refusal carrying a DISTINCT cause, so callers can tell WHY."""

    def __init__(self, cause: str, detail: str):
        self.cause = cause
        self.detail = detail
        super().__init__(f"[{cause}] {detail}")


def canonical_payload(content, binds_to) -> bytes:
    """annotation_canon_v1, exactly as the contract owner ruled it.

    allow_nan=False is not decoration: the default emits bare NaN/Infinity,
    which is not JSON, so some readers reject the payload and others parse it
    differently -- a divergence that surfaces as a signature mismatch."""
    return json.dumps({"content": content, "binds_to": binds_to},
                      sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def owner_sha256(content, binds_to) -> str:
    return hashlib.sha256(canonical_payload(content, binds_to)).hexdigest()


def assert_python_supported() -> None:
    """annotation_canon_v1 is defined for CPython 3.12+ on BOTH sides.

    json.dumps serialises floats with Python's shortest round-tripping repr:
    stable across CPython 3.x, NOT a cross-language guarantee. An implementation
    outside that must declare a NEW canonical_form rather than reinterpret this
    one."""
    if sys.implementation.name != "cpython" or sys.version_info < (3, 12):
        raise MergeRefused(
            "unsupported_runtime",
            f"annotation_canon_v1 is defined for CPython 3.12+; this is "
            f"{sys.implementation.name} {sys.version_info[:3]}. Reinterpreting "
            f"the canon on another runtime is how two sides silently disagree.")


def _stem(receipt_name: str) -> str:
    """Sidecar filenames use the receipt STEM; target.artifact uses the full
    filename. Normalising here means a caller may pass either and the two
    conventions cannot drift into a spurious wrong_target refusal."""
    return receipt_name[:-5] if receipt_name.endswith(".json") else receipt_name


def _artifact(receipt_name: str) -> str:
    return _stem(receipt_name) + ".json"


def sidecar_path(receipt_name: str, field: str) -> Path:
    return ANNOTATIONS / f"{_stem(receipt_name)}.{field}.json"


def assert_canon_agreement(path: Path = None) -> dict:
    """Recompute the owner's sha from the owner's REAL committed bytes.

    This is the gate on the whole merge path. If it is absent or failing the
    merge must not run at all -- not fall back, not warn."""
    assert_python_supported()
    path = path or sidecar_path("phase2_four_arm_v2", "da_caveat_field")
    if not path.exists():
        raise MergeRefused(
            "agreement_test_impossible",
            f"no owner sidecar at {path} to prove agreement against. The merge "
            f"path must not run on an unproven canonicalization.")
    sc = json.loads(path.read_text())
    if sc.get("canonical_form") != CANON:
        raise MergeRefused(
            "unrecognised_canonical_form",
            f"sidecar declares canonical_form {sc.get('canonical_form')!r}; "
            f"this implementation knows {CANON!r}. Refusing with a DISTINCT "
            f"cause: this is a recipe disagreement, NOT tampering.")
    got = owner_sha256(sc["content"], sc["binds_to"])
    if got != sc.get("owner_sha256"):
        raise MergeRefused(
            "canon_disagreement",
            f"recomputed {got} but the owner recorded {sc.get('owner_sha256')} "
            f"on their own committed bytes. The two sides do not agree on the "
            f"canonical form; merging would report every valid sidecar as "
            f"tampered.")
    return {"agreed": True, "sidecar": str(path), "sha256": got,
            "canonical_form": CANON}


def _fingerprint(binds_to: dict) -> dict:
    return {k: v for k, v in (binds_to or {}).items()
            if k not in FINGERPRINT_IGNORE}


def merge(receipt_name: str, field: str, actual: dict,
          sidecar: Path = None) -> tuple:
    """Return (field_value, evidence). Never raises for an ABSENT sidecar.

    `actual` is the fingerprint recomputed against the receipt being produced."""
    assert_canon_agreement()               # the gate; refuses the whole path
    path = sidecar or sidecar_path(receipt_name, field)
    if not path.exists():
        return RESERVED, {"merged": False, "cause": "sidecar_absent",
                          "note": "field stays RESERVED; never fabricated"}
    sc = json.loads(path.read_text())
    if sc.get("canonical_form") != CANON:
        raise MergeRefused("unrecognised_canonical_form",
                           f"{path.name} declares {sc.get('canonical_form')!r}")
    if sc.get("owner") != "DA":
        raise MergeRefused("wrong_owner", f"{path.name} owner={sc.get('owner')!r}")
    if sc.get("schema_version") != SCHEMA:
        raise MergeRefused("wrong_schema_version",
                           f"{path.name} schema_version={sc.get('schema_version')!r}")
    tgt = sc.get("target") or {}
    if tgt.get("artifact") != _artifact(receipt_name) or tgt.get("field") != field:
        raise MergeRefused(
            "wrong_target",
            f"{path.name} targets {tgt}, not "
            f"{_artifact(receipt_name)}::{field}")
    if owner_sha256(sc["content"], sc["binds_to"]) != sc.get("owner_sha256"):
        raise MergeRefused(
            "owner_sha_mismatch",
            f"{path.name} content/binds_to do not hash to the recorded "
            f"owner_sha256. REFUSING the merge and leaving the field RESERVED; "
            f"never repairing, truncating or partially merging.")

    binds = sc["binds_to"]
    if binds.get("population_independent") is True:
        return dict(sc["content"]), {
            "merged": True, "cause": "population_independent",
            "binding_stale": False}
    declared = _fingerprint(binds)
    actual_cmp = {k: (actual or {}).get(k) for k in declared}
    if declared == actual_cmp:
        return dict(sc["content"]), {"merged": True, "cause": "fingerprint_match",
                                     "binding_stale": False}
    # DIFFERS: carry the content VERBATIM and flag it. Dropping loses a caveat
    # someone relied on; silent carry is how one population's magnitudes end up
    # describing another's receipt.
    merged = dict(sc["content"])
    merged["BINDING_STALE"] = {
        "declared": declared, "actual": actual_cmp,
        "fields_affected": sorted(k for k in declared
                                  if declared[k] != actual_cmp.get(k)),
        "meaning": "the annotation is carried VERBATIM and is NOT dropped; its "
                   "declared population binding no longer holds for this "
                   "receipt, so magnitudes inside it describe a different "
                   "population",
        "policy": "a stale binding ANNOTATES; it does not refuse the receipt "
                  "(contract §'what this does not do'; rule 14)"}
    return merged, {"merged": True, "cause": "fingerprint_differs",
                    "binding_stale": True}


def selftest() -> int:
    """The contract's six falsifiers, behavioural. Rule 15."""
    import tempfile
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    ag = assert_canon_agreement()
    ok(ag["agreed"], "0 canon agreement PROVEN on DA's real committed bytes")

    real = sidecar_path("phase2_four_arm_v2", "da_caveat_field")
    sc = json.loads(real.read_text())
    d = Path(tempfile.mkdtemp())

    def write(mut):
        s2 = json.loads(real.read_text())
        mut(s2)
        p = d / f"t{abs(hash(json.dumps(s2, sort_keys=True))) % 10**9}.json"
        p.write_text(json.dumps(s2))
        return p

    # 1 absent sidecar -> RESERVED, nothing invented
    v, ev = merge("phase2_four_arm_v2", "da_caveat_field", {},
                  sidecar=d / "nope.json")
    ok(v == RESERVED and ev["cause"] == "sidecar_absent",
       "1 an ABSENT sidecar leaves the field RESERVED, nothing invented")

    # 2 valid + MATCHING fingerprint -> verbatim, owner's bytes unchanged
    declared = _fingerprint(sc["binds_to"])
    v, ev = merge("phase2_four_arm_v2", "da_caveat_field", dict(declared),
                  sidecar=real)
    ok(v == sc["content"] and not ev["binding_stale"],
       "2 matching fingerprint merges content VERBATIM (compared to the "
       "SIDECAR, not to the merged object)")

    # 3 valid + DIFFERING fingerprint -> content STILL present AND flagged
    v, ev = merge("phase2_four_arm_v2", "da_caveat_field",
                  {"n_rows": {"btc": 1}}, sidecar=real)
    content_intact = all(v.get(k) == sc["content"][k] for k in sc["content"])
    ok(content_intact and "BINDING_STALE" in v and ev["binding_stale"],
       "3 differing fingerprint carries content AND flags BINDING_STALE "
       "(asserting only 'content present' would pass silent-carry too)")
    ok(v["BINDING_STALE"]["fields_affected"] == ["n_rows"],
       "3b BINDING_STALE names WHICH field went stale")

    # 4 tampered owner_sha256 -> REFUSED
    try:
        merge("phase2_four_arm_v2", "da_caveat_field", {},
              sidecar=write(lambda s: s.update({"owner_sha256": "0" * 64})))
        ok(False, "4 a TAMPERED owner_sha256 is REFUSED")
    except MergeRefused as e:
        ok(e.cause == "owner_sha_mismatch", "4 a TAMPERED owner_sha256 is REFUSED")

    # 5 wrong owner / schema_version -> REFUSED
    for key, val, cause in (("owner", "BE", "wrong_owner"),
                            ("schema_version", "v9", "wrong_schema_version")):
        try:
            merge("phase2_four_arm_v2", "da_caveat_field", {},
                  sidecar=write(lambda s, k=key, x=val: s.update({k: x})))
            ok(False, f"5 wrong {key} is REFUSED")
        except MergeRefused as e:
            ok(e.cause == cause, f"5 wrong {key} is REFUSED")

    # unknown canonical form -> DISTINCT cause, never signature mismatch
    try:
        merge("phase2_four_arm_v2", "da_caveat_field", {},
              sidecar=write(lambda s: s.update({"canonical_form": "canon_v99"})))
        ok(False, "5b an UNKNOWN canonical_form is REFUSED")
    except MergeRefused as e:
        ok(e.cause == "unrecognised_canonical_form",
           "5b an UNKNOWN canonical_form refuses with a DISTINCT cause, not a "
           "signature mismatch")

    # wrong target -> REFUSED
    try:
        merge("some_other_receipt.json", "da_caveat_field", {}, sidecar=real)
        ok(False, "5c a sidecar aimed at ANOTHER artifact is REFUSED")
    except MergeRefused as e:
        ok(e.cause == "wrong_target",
           "5c a sidecar aimed at ANOTHER artifact is REFUSED")

    # 6 positive control: the merge path must not become load-bearing
    v, ev = merge("phase2_four_arm_v2", "nonexistent_field", {})
    ok(v == RESERVED and not ev["merged"],
       "6 with no sidecar the generator still produces a valid field value")

    print(f"\n{'ANNOTATION-MERGE SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
