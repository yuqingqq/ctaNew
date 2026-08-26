"""Phase-0 deliverable 1 — build `harmful_candidate_manifest_v1.json`.

SURFACE AUTHORISATION (R-126, in-file): coordinator dispatch R-145, Phase 0,
BE assignment. New surface, authorisation carried here rather than assumed.

WHAT A MANIFEST IS FOR. Rule 12: a freeze is a commit. A candidate is frozen
only if a FRESH PROCESS can load ONE file and reproduce the named development
scores WITHOUT FITTING and WITHOUT READING GROWING RAW DATA. Everything a
reproducer needs is pinned here by content hash; nothing is derived at runtime.

THE ERA BOUNDARY IS A PINNED CONSTANT, NOT A LOOKUP.
`ERA_BOUNDARY_NS` is written into the manifest as a literal. It must never be
derived from "the latest collector restart", because that value MOVES: a
collector restart tomorrow would silently redefine which rows were admissible
yesterday, and a manifest that changes meaning after the fact pins nothing.

THE SPLIT CHECKER SHIPS A FALSIFIER (rule 15).
`check_split_matches_rows()` refuses a receipt whose DECLARED split disagrees
with its ROW TIMESTAMPS. Its selftest asserts both arms: a positive control it
must flag (the known-stale `harmful_hazard_model.py` docstring naming
2026-08-20/21/22, all three of which precede the era boundary) and a known-good
input it must pass. A checker that has never proved it can fire is not evidence.

ATOMIC WRITES. Manifest and score dumps are written to a temporary file in the
same directory, fsynced, then `os.replace`d. A half-written manifest that still
parses is worse than none, because it reproduces something.
"""
from __future__ import annotations

import hashlib, json, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"

# PINNED, never derived. hf_ws_v2 stamp boundary; see CLAUDE.md reliability rule 5.
ERA_BOUNDARY_NS = 1787579334881534478
ERA_BOUNDARY_UTC = "2026-08-24T13:48:54Z"
SPLIT_EMBARGO_S = 60

# Files whose content defines the candidate. Order is stable so the digest is.
HASHED = [
    "data/pm_5min/derived/harmful_exposure_rows_v3_eraB.json",   # dataset
    "data/mm_hf/collector_runs.jsonl",                           # raw-source ledger
    "live/pm_research/harmful_hazard_model.py",                  # builder
    "live/pm_research/flow_fill_development.py",                 # dep
    "live/pm_research/policy_bounds_v1.py",                      # dep
    "live/pm_research/flow_intensity.py",                        # dep
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def atomic_write_json(path: Path, obj: dict) -> None:
    """Write-then-rename. A half-written manifest that PARSES is worse than none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def day_of(slug_or_ts) -> str | None:
    try:
        ts = int(str(slug_or_ts).rsplit("-", 1)[-1])
    except (ValueError, AttributeError):
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def check_split_matches_rows(declared_days: list[str],
                             row_days: list[str],
                             era_boundary_ns: int = ERA_BOUNDARY_NS) -> dict:
    """REFUSE a receipt whose declared split disagrees with its row timestamps.

    Two independent failures, reported separately because they have different
    remedies:
      MISMATCH      declared days are not the days the rows are on
      ERA_VIOLATION a declared day precedes the era boundary, so it is
                    legacy-stamped and inadmissible for sub-second features
    """
    dset, rset = set(declared_days), set(row_days)
    # A day is an era violation only if it ENDS before the boundary -- i.e. NO
    # part of it is admissible. A day that STRADDLES the boundary is partly
    # admissible and must not be refused wholesale: rule 5 makes era purity a
    # PER-EVENT predicate on recv_ns, never a per-file or per-day one.
    # (BE's first version tested day START and refused 2026-08-24, the very day
    # the boundary falls on and the day the fragment is mostly drawn from. The
    # known-good arm of this selftest caught it.)
    pre_era, straddling = [], []
    for d in sorted(dset):
        start = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        end = start + 86400
        if end * 1e9 <= era_boundary_ns:
            pre_era.append(d)
        elif start * 1e9 < era_boundary_ns:
            straddling.append(d)
    verdict = "OK"
    if pre_era:
        verdict = "REFUSED_ERA_VIOLATION"
    elif dset != rset:
        verdict = "REFUSED_SPLIT_MISMATCH"
    return {"verdict": verdict, "declared": sorted(dset), "rows_on": sorted(rset),
            "declared_not_in_rows": sorted(dset - rset),
            "rows_not_declared": sorted(rset - dset),
            "declared_days_before_era_boundary": pre_era,
            "declared_days_straddling_boundary": straddling}


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    # POSITIVE CONTROL: the real, known-stale docstring at harmful_hazard_model.py:24
    r = check_split_matches_rows(["2026-08-20", "2026-08-21", "2026-08-22"],
                                 ["2026-08-24", "2026-08-25"])
    ok(r["verdict"] == "REFUSED_ERA_VIOLATION",
       "POSITIVE CONTROL: the known-stale split (08-20/21/22) is REFUSED")
    ok(r["declared_days_before_era_boundary"] == ["2026-08-20", "2026-08-21", "2026-08-22"],
       "and all three pre-era days are NAMED, not merely counted")

    # KNOWN-GOOD: the split the code actually reads
    g = check_split_matches_rows(["2026-08-24", "2026-08-25"], ["2026-08-24", "2026-08-25"])
    ok(g["verdict"] == "OK", "the true split passes")
    ok(g["declared_days_straddling_boundary"] == ["2026-08-24"],
       "and 08-24 is reported as STRADDLING, not refused — era purity is "
       "per-event on recv_ns, so part of that day is admissible")

    # a post-era mismatch is a DIFFERENT failure from an era violation
    m = check_split_matches_rows(["2026-08-24"], ["2026-08-24", "2026-08-25"])
    ok(m["verdict"] == "REFUSED_SPLIT_MISMATCH", "a post-era mismatch is refused")
    ok(m["rows_not_declared"] == ["2026-08-25"], "and the undeclared day is named")

    ok(ERA_BOUNDARY_NS == 1787579334881534478, "the era boundary is a pinned literal")
    ok(datetime.fromtimestamp(ERA_BOUNDARY_NS / 1e9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
       == ERA_BOUNDARY_UTC, "and its UTC rendering agrees with the literal")

    # atomic write leaves no partial file behind on failure
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / "m.json"
        atomic_write_json(t, {"a": 1})
        ok(json.loads(t.read_text()) == {"a": 1}, "atomic write round-trips")
        ok(not list(Path(td).glob("*.tmp")), "and leaves no .tmp behind")

    print(f"harmful_candidate_manifest selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    print("build the manifest with: python3 harmful_candidate_manifest.py build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
