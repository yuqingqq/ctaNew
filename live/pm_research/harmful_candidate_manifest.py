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
    "live/pm_research/harmful_exposure_rows.py",                 # era selector
    "live/pm_research/harmful_action_eval.py",                   # evaluator
    "live/pm_research/harmful_rows_loader.py",                   # streaming load path
]


# ---- reproduction resource contract (R-148(5)) --------------------------
# A reproduction that cannot RUN on the hardware is not reproducible. So the
# manifest states a peak-RSS bound alongside the hashes, and a reproducer that
# cannot meet it must say so rather than quietly succeed on a bigger box.
#
# WHAT IS ACTUALLY KNOWN (measured, 2026-08-26 -- and it is a LOWER BOUND, not
# a peak; the run has never been observed to completion):
#   attempt 1  MemoryMax=8G   -> cgroup OOM-killed AT 8.0G after 10m40s CPU
#   attempt 2  MemoryMax=8G   -> same
#   attempt 3  MemoryMax=14G  -> observed 8.6/8.8/8.5/8.3G at ~03:51-03:53,
#                                still running when the box died at ~03:55
# So: peak_rss > 8.8 GiB, CEILING UNKNOWN. The honest field is a bound with a
# status, never a number that implies a completed measurement.
# MEASURED 2026-08-26T07:57Z: the builder COMPLETED under a 14 G cap
# (be-ceiling5-1787726526, 1h15m26s CPU, exit 0, receipt written).
# Two peak figures disagree and BOTH are recorded rather than one chosen:
#   systemd unit accounting ....... 8.3 GiB  (Consumed line, rounded)
#   mem_trace cgroup sampling ..... 9.66 GiB (4,518 samples of memory.peak)
# The tracer's cgroup attribution CANNOT be re-verified: `--collect` removed
# the unit at exit, so `cgroup_dir` now resolves to None. It may have watched
# the enclosing research.slice rather than the unit. The slice held nothing
# else, so the two should coincide; they do not, and BE will not resolve that
# by preferring the convenient number. The CONSERVATIVE figure is recorded as
# the planning ceiling.
PEAK_RSS_MEASURED_BYTES = 9_660_000_000
PEAK_RSS_SYSTEMD_UNIT_BYTES = 8_300_000_000
PEAK_RSS_STATUS = ("MEASURED -- run completed under a 14 GiB cap. Conservative "
                   "ceiling 9.66 GiB (mem_trace); systemd unit accounting says "
                   "8.3 GiB. Discrepancy disclosed, not resolved: the tracer's "
                   "cgroup attribution cannot be re-verified after --collect.")
PEAK_RSS_SOURCE = ("be-ceiling5-1787726526, 2026-08-26T06:42-07:57Z, "
                   "MemoryMax=14G in research.slice, max PSI some_avg10 0.54")
PEAK_RSS_LOWER_BOUND_BYTES = PEAK_RSS_MEASURED_BYTES   # kept for readers


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

    ok(PEAK_RSS_STATUS.startswith("MEASURED"),
       "the peak-RSS field now reports a MEASURED ceiling: the builder "
       "completed under a 14 GiB cap")
    ok(PEAK_RSS_MEASURED_BYTES > PEAK_RSS_SYSTEMD_UNIT_BYTES,
       "the CONSERVATIVE of two disagreeing peak figures is the recorded "
       "ceiling -- the discrepancy is disclosed, not resolved by preference")
    ok(ERA_BOUNDARY_NS == 1787579334881534478, "the era boundary is a pinned literal")
    _m = build()["manifest"]
    ok(_m["pin_semantics"]["data/mm_hf/collector_runs.jsonl"] == "state_at_build",
       "the GROWING ledger is pinned as state_at_build, not as an anchor -- "
       "its hash drifted once already inside a single session")
    ok(_m["target_scores_to_reproduce"]["source_sha256_at_snapshot"].startswith(
           "3279e2aab3c3723e"),
       "targets come from the IMMUTABLE pre-probe frozen copy, so the gate can "
       "never degenerate into comparing a run against itself")
    ok(_m["target_scores_to_reproduce"]["btc_PM_PLUS_FINE"]["auc"]
       != round(_m["target_scores_to_reproduce"]["btc_PM_PLUS_FINE"]["auc"], 6),
       "targets are READ from the receipt at FULL precision, not transcribed "
       "at 6dp -- a rounded target reports a perfect reproduction as a failure")
    ok(_m["era_key_at_build"]["distinct_keys_in_ledger"] == 1,
       "and the ledger's ERA KEY -- the part that is actually invariant under "
       "restarts -- is pinned beside it")
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


def _read_targets() -> dict:
    """READ the targets from the frozen receipt. Never transcribe them.

    BE's first version hand-typed these and rounded the AUCs to six decimals
    (0.692310 for a true 0.6923099451399828). The comparator, run against the
    very receipt the targets came from, returned NOT_REPRODUCED -- i.e. a
    PERFECT reproduction would have been reported as a failure by a target
    that was never exact in the first place. Third time this session that
    transcribing a value instead of reading it from the artifact produced a
    wrong one; the rule is mechanical for the same reason the era end is."""
    # READ FROM THE IMMUTABLE FROZEN COPY, never the live receipt.
    # `run --fine` OVERWRITES harmful_fine_comparison_v3.json. Rebuilding the
    # manifest after a probe therefore re-read the targets from the file the
    # probe had just written, replacing the pre-probe provenance hash with the
    # post-probe one -- which would make every future gate run CIRCULAR
    # (comparing a run against itself). Caught immediately after the first
    # post-probe rebuild. The frozen copy is committed and never regenerated.
    src = DERIVED / "harmful_fine_comparison_v3_FROZEN_TARGETS.json"
    d = json.loads(src.read_text())
    out = {"source_receipt": src.name,
           "source_sha256_at_snapshot": sha256(src),
           "values_read_from_artifact_not_transcribed": True,
           "source_is_immutable_frozen_copy": True,
           "why": "the live receipt is overwritten by every run; reading "
                  "targets from it would compare a run against itself"}
    for coin in ("btc", "eth"):
        a = d["paired_arms"][coin]["PM_PLUS_FINE"]
        g = a.get("gate", {})
        b = (g.get("budgets") or {}).get("5%", {})
        out[f"{coin}_PM_PLUS_FINE"] = {
            "auc": a.get("auc"),
            "n_generations": g.get("n_generations"),
            "net_cents_5pct": b.get("net_cents"),
            "harm_avoided_cents_5pct": b.get("harm_avoided_cents")}
    return out


def build() -> dict:
    """Emit `harmful_candidate_manifest_v1.json`.

    Everything a fresh process needs to reproduce the named development
    scores WITHOUT FITTING and WITHOUT READING GROWING RAW DATA. Fields that
    genuinely do not exist yet (fitted weights) are marked PENDING with the
    reason -- never omitted, never faked, because an automated reader must be
    able to tell "not yet measured" from "measured as absent"."""
    import subprocess as _sp
    from harmful_exposure_rows import (DECLARED_ERA_KEY, DECLARED_ERA_END_S,
                                       ledger_era_keys)

    ds = REPO / "data/pm_5min/derived/harmful_exposure_rows_v3_eraB.json"
    hashes = {}
    for rel in HASHED:
        f = REPO / rel
        hashes[rel] = sha256(f) if f.exists() else "ABSENT"

    as_of = _sp.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                    capture_output=True, text=True).stdout.strip()
    m = {
        "protocol": "HARMFUL_CANDIDATE_MANIFEST_V1",
        "as_of_utc": as_of,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),

        "era": {
            "boundary_ns": ERA_BOUNDARY_NS,
            "boundary_utc": ERA_BOUNDARY_UTC,
            "pinned_literal": True,
            "why_pinned": "Deriving it from max(started_at_ns) let two "
                          "collector restarts walk the floor 39.6 h forward "
                          "and admit 0 of 926 windows (Q-DA-67). A derived "
                          "boundary redefines yesterday's admissibility.",
            "declared_era_key": list(DECLARED_ERA_KEY),
            "ledger_distinct_keys": len(set(ledger_era_keys())),
            "era_end_s": DECLARED_ERA_END_S["v3_4_consumed_fragment"],
            "era_end_basis": "last slug t0 1787650200 + WINDOW_S 300 + "
                             "MARKOUT_S 5 + 5; verified at the artifact",
        },
        "population": {
            "name": "v3_4_consumed_fragment",
            "schema": "harmful_exposure_v3_4_fill_scoped_markout",
            "n_windows": 471,
            "n_rows_total": 1135943,
            "n_rows_ok": 1125289,
            "statuses": {"OK": 1125289, "GAP_IN_HORIZON": 10456,
                         "TRUNCATED_HORIZON": 198},
            "by_coin": {"btc": 609775, "eth": 526168},
            "by_coin_day": {"btc|2026-08-24": 321770, "btc|2026-08-25": 288005,
                            "eth|2026-08-24": 265018, "eth|2026-08-25": 261150},
            "days": ["2026-08-24", "2026-08-25"],
            "first_slug_t0": 1787579400,
            "last_slug_t0": 1787650200,
            "split_embargo_s": SPLIT_EMBARGO_S,
            "correctness_counters": {"reconciliation_failures": 0,
                                     "boundary_time_violations": 0,
                                     "consume_clock_violations": 0,
                                     "unhooked_state_changes": 0},
            "consumed": True,
            "role": "DEVELOPMENT ONLY -- partial 08-24/25 fragment, five specs "
                    "already scored on it. Never forward validation.",
        },
        "split": {
            "train_days": ["2026-08-24"],
            "dev_day": "2026-08-25",
            "derived_from": "receipt days[:-1] / days[-1]",
            "supersedes_docstring_claim": "harmful_hazard_model.py:24 declares "
                "'train 2026-08-20/21, development 2026-08-22'. All three days "
                "END before the era boundary and NONE appear in this dataset; "
                "the docstring is stale and this field governs.",
        },
        "hashes": hashes,
        # R-160: a GROWING input's hash is provenance, not a reproducibility
        # anchor. collector_runs.jsonl gains a row on every collector restart
        # -- it went from 2 rows to 4 during the 08-26 recovery alone, so its
        # hash ALREADY drifted once inside a single session. A reproducer that
        # treats it as an anchor would declare a valid manifest stale; one that
        # treats it as provenance reads it as "this is what the ledger looked
        # like when the manifest was built", which is the only true claim.
        "pin_semantics": {
            "data/mm_hf/collector_runs.jsonl": "state_at_build",
            "_default": "reproducibility_anchor",
            "_note": "state_at_build entries MUST NOT be compared for equality "
                     "when validating a reproduction; compare only the "
                     "reproducibility_anchor entries.",
        },
        "era_key_at_build": {
            "declared": list(DECLARED_ERA_KEY),
            "distinct_keys_in_ledger": len(set(ledger_era_keys())),
            "why_this_and_not_the_hash": "the ledger's HASH moves with every "
                "restart, but its ERA KEY does not. The key is the invariant "
                "worth pinning; the hash is only provenance.",
        },
        "deps": {"python": _sp.run([sys.executable, "-c",
                    "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
                    capture_output=True, text=True).stdout.strip(),
                 "numpy": _sp.run([sys.executable, "-c",
                    "import numpy;print(numpy.__version__)"],
                    capture_output=True, text=True).stdout.strip(),
                 "sklearn": _sp.run([sys.executable, "-c",
                    "import sklearn;print(sklearn.__version__)"],
                    capture_output=True, text=True).stdout.strip()},
        "reproduction_contract": {
            "target_latency_ms": 50,
            "peak_rss_measured_bytes": PEAK_RSS_MEASURED_BYTES,
            "peak_rss_systemd_unit_bytes": PEAK_RSS_SYSTEMD_UNIT_BYTES,
            "peak_rss_headroom_under_14g": round(1 - 9.66 / 14.0, 3),
            "cap_used_bytes": 14 * 1024**3,
            "completed_under_cap": True,
            "cpu_time_s": 4526,
            "max_psi_some_avg10": 0.54,
            "peak_rss_status": PEAK_RSS_STATUS,
            "peak_rss_source": PEAK_RSS_SOURCE,
            "streaming_load_path": "live/pm_research/harmful_rows_loader.py",
            "streaming_projected_bytes": 1_060_000_000,
            "streaming_equivalence": "5000 rows field-by-field vs independent "
                                     "full parse: 0 differences",
            "launch": "systemd-run --user --slice=research.slice "
                      "-p MemoryMax=14G -p OOMScoreAdjust=1000 -- "
                      "/home/yuqing/pricer-sol/venv/bin/python3 ...",
        },
        "candidate": {
            "spec": "PM_PLUS_FINE (reduced fine)",
            "features_pm": "54 PM queue/flow features, scales 0.01-5.0s, LAG_S 0.001",
            "features_fine": ["bnf_midbps @10,25,50,100,250ms", "bnf_imb_now"],
            "fine_cutoff_s": 0.001,
            "normalization": "z-score on train rows, applied in place",
            "weights": "PENDING -- no fit has completed on this box; the "
                       "reproduction has never run to completion. Marked "
                       "PENDING rather than omitted so a reader cannot mistake "
                       "'not yet measured' for 'measured as absent'.",
            "thresholds": "per-generation max score, quantile at budget "
                          "{0.05,0.10,0.15}; first crossing cancels once",
        },
        "declared_nulls": {"n_random": 200,
                           "matching": "side x hour strata, matched action count",
                           "decision_metric": "net_cents (NOT harm share)"},
        "target_scores_to_reproduce": _read_targets(),
        "multiplicity_note": "adverse race = v2.1 alone (multiplicity 1) per "
                             "R-147(4); the harmful-fill line is separate.",
        "freeze_status": "NOT FROZEN. The freeze is the user's decision and "
                         "Phase-0 reproduction has not yet run.",
    }
    out = DERIVED / "harmful_candidate_manifest_v1.json"
    atomic_write_json(out, m)
    return {"path": str(out), "n_hashes": len(hashes), "manifest": m}


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "build" in sys.argv:
        r = build()
        print(f"WROTE {r['path']}")
        m = r["manifest"]
        print(f"  era floor pinned : {m['era']['boundary_ns']} ({m['era']['boundary_utc']})")
        print(f"  era end declared : {m['era']['era_end_s']}")
        print(f"  ledger era keys  : {m['era']['ledger_distinct_keys']} (1 = no transition)")
        print(f"  population       : {m['population']['n_windows']} windows, "
              f"{m['population']['n_rows_ok']:,} OK rows")
        print(f"  hashes pinned    : {r['n_hashes']}")
        print(f"  weights          : {m['candidate']['weights'][:38]}...")
        print(f"  freeze_status    : {m['freeze_status'][:34]}...")
        return 0
    print("usage: harmful_candidate_manifest.py [build|--selftest]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
