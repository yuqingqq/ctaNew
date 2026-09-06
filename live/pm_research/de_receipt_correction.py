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
EXPECTED_CHECKS = 13

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


def n_days_complete_true_at_emit(rec_path: Path, root: Path) -> dict:
    """HOW MANY DAYS WERE SEALED WHEN THIS RECEIPT WAS EMITTED.

    Counted from the sealed receipts whose EMIT STAMPS PRECEDE this one --
    the method named here, inside the artifact, so a reader need not
    reconstruct the reconstruction."""
    d = Path(root) / "pm_5min/derived"
    mine = re.search(r"_SEALED__(\d{8}T\d{6}Z)", Path(rec_path).name)
    if not mine:
        return {"status": "NOT_RECONSTRUCTABLE",
                "why": "this receipt's own name carries no emit stamp"}
    stamp = mine.group(1)
    seen, rows = {}, []
    for f in sorted(d.glob("p003_de_gate1_day_run_*_SEALED__*.json")):
        m = re.match(r"p003_de_gate1_day_run_(\d{8})_SEALED__"
                     r"(\d{8}T\d{6}Z)", f.name)
        if not m:
            continue
        day, st = m.group(1), m.group(2)
        rows.append({"day": day, "emit_stamp": st, "file": f.name})
        if st <= stamp:
            seen.setdefault(day, st)
    return {"n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED": len(seen),
            "days_sealed_at_or_before_this_emit": sorted(seen),
            "this_receipt_emit_stamp": stamp,
            "method": ("counted from the SEALED receipts under the "
                       "ledger whose emit stamp is <= this receipt's, "
                       "read from their filenames; the day is the key so "
                       "a `.v2` of a day already counted does not count "
                       "it twice"),
            "all_sealed_receipts_considered": rows,
            "why_the_key_says_RECONSTRUCTED": (
                "the emitted `n_days_complete` is left UNTOUCHED; this is "
                "recovered after the fact and says so in its own name")}


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
    leaks = [k for k in RUNNER.seal_key_walk(v2)
             if k.rsplit(".", 1)[-1] in set(sealed["sealed_names"])]
    if leaks:
        raise CorrectionRefused(
            f"REFUSED before writing: the correction carries sealed names "
            f"as KEYS -- {leaks[:6]}. {RUNNER.SEAL_RULE}")
    assert_no_reconstruction_under_a_plain_name(v2)
    v2["correction_census"] = {
        "protocol": PROTOCOL,
        "as_of": now.isoformat(),
        "census_by": "be_race_reader.correction_census -- IMPORTED, never "
                     "re-implemented",
        "census": census,
        "frozen_set": frozen,
        "sealed_names_in_force_at_v1s_emit": sealed,
        "seal_census_by_KEY_before_writing": {
            "n_sealed_keys_found": 0, "rule": RUNNER.SEAL_RULE},
        "what_a_correction_may_never_do": [
            "publish an economic field", "recompute a statistic",
            "change a sealed or frozen block",
            "put a reconstruction under a plain name",
            "carry half a `supersedes`"],
    }
    assert_supersedes_is_a_pair(v2, v1_path)
    return v2


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
            "refused BEFORE it is written", "sealed names")
    refuses(lambda: build_correction(
                v1p, {"sha256_at_load": "RECONSTRUCTED"}, root=td),
            "KNOWN-BAD: a reconstruction under a PLAIN name is refused -- "
            "the status rides in the KEY", "PLAIN name")

    nd = n_days_complete_true_at_emit(v1p, td)
    ok(nd["n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED"] == 1
       and nd["days_sealed_at_or_before_this_emit"] == ["20260903"]
       and "counted from the SEALED receipts" in nd["method"],
       f"the day count TRUE AT EMIT is reconstructed from the sealed "
       f"receipts whose stamps precede this one "
       f"({nd['n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED']}), with the "
       f"method named INSIDE the artifact")
    later = td / "pm_5min/derived/p003_de_gate1_day_run_20260904_SEALED__20260906T163351Z.json"
    later.write_text(json.dumps(base) + "\n")
    nd2 = n_days_complete_true_at_emit(later, td)
    ok(nd2["n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED"] == 2
       and nd2["days_sealed_at_or_before_this_emit"] == ["20260903",
                                                         "20260904"],
       "and a LATER receipt reconstructs 2 -- the count is a function of "
       "the stamp, not of how many files exist now")
    ok(v2["n_days_complete"] == base["n_days_complete"],
       "and the EMITTED `n_days_complete` is left UNTOUCHED: the "
       "reconstruction is a new key beside it, never a repair of it")
    sc = v2["correction_census"]["sealed_names_in_force_at_v1s_emit"]
    ok(sc["n"] == 8 and "POSITIVE" in sc["read_from"],
       f"the sealed list is the one IN FORCE AT THE v1'S EMIT "
       f"({sc['n']} names, design v{sc['design_version']}, recognised "
       f"POSITIVELY by its emit stamp) -- a receipt cannot have disobeyed "
       f"a rule that did not exist when it was written")

    # THE GATE (REV 79 §1.3), driven at BE's landed code.
    frozen_stub = {b: {"x": 1} for b in BE.FROZEN_BLOCKS}
    o1 = {**frozen_stub, "other": 1}
    o2 = {**o1, "pinned_days_not_in_READABLE": ["d"],
          "producing_code": {"status": "X"}, "correction_census": {}}
    admitted = True
    try:
        BE.correction_census(o1, o2)
    except BE.ReadRefused:
        admitted = False
    ok(isinstance(admitted, bool),
       f"REV 79 S1.3 DRIVEN AT BE'S LANDED CODE: a `.v2` omitting the "
       f"declared addition `supersedes` is "
       f"{'ADMITTED -- the gate does not yet bind' if admitted else 'REFUSED'}"
       f". The comparison filters the declared set by what the .v2 "
       f"happens to carry, so an omitted addition drops out of BOTH sides")
    ok(admitted is True,
       "and it ADMITS at this tip, so THIS SEAT EMITS NO RECEIPT `.v2` "
       "until BE 75's one-line fix lands -- the emitter and its fixtures "
       "land now, the real emit follows")

    print(f"[de_receipt_correction] PASS -- {n[0]} checks")
    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(
            f"[de_receipt_correction] FAIL: {n[0]} checks run against a "
            f"declared {EXPECTED_CHECKS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ap.error("--selftest")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
