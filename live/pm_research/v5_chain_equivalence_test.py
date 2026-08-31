#!/usr/bin/env python3
"""ONE fixture set asserted through BOTH era-chain consumers (V5-R3A).

Codex: "Add one fixture asserted through both consumers, not two
independently constructed approximations." Each case runs through the
coordinator's `current_era_and_open_v5` AND DA's `day_era_admission` over the
same rows; the test asserts each side's expected verdict AND their agreement.
A later pre-arm can therefore never proceed from a ledger the eligibility
consumer refuses.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_boundary_preflight as P  # noqa: E402
import da_forward_day_verify as D  # noqa: E402

B = P.BOUNDARY_EPOCH
LEGACY = {"collector_schema_version": "clob_v4", "supersedes": "clob_v3_1",
          "boundary_utc": "2026-08-30T05:30:00Z"}
V5OK = {"collector_schema_version": "clob_v5", "supersedes": "clob_v4",
        "transitioned": True, "boundary_utc": P.BOUNDARY_UTC}
V5MAL = {**V5OK, "supersedes": "clob_v3_1"}
V5SELF = {**V5OK, "supersedes": "clob_v5"}
RB = {"collector_schema_version": "clob_v4", "supersedes": "clob_v5",
      "rollback": True, "closes_boundary_utc": P.BOUNDARY_UTC,
      "stage": "counters_refused",
      "collector_start_recv_ns": (B + 900) * 10**9,
      "boundary_utc": "2026-08-31T07:15:00Z"}

# (name, rows, expected: "ok" both accept / "refuse" both refuse)
CASES = [
    ("good open v5", [LEGACY, V5OK], "ok"),
    ("good rollback chain", [LEGACY, V5OK, RB], "ok"),
    ("malformed supersedes (round-3 executed)", [LEGACY, V5MAL, RB],
     "refuse"),
    ("self-supersede (DA 9ee4f44)", [LEGACY, V5SELF], "refuse"),
    ("rollback-only ledger", [LEGACY, RB], "refuse"),
    ("wrong closes_boundary", [LEGACY, V5OK,
     {**RB, "closes_boundary_utc": "2026-08-31T06:00:00Z"}], "refuse"),
    ("stage-less rollback", [LEGACY, V5OK, {**RB, "stage": ""}], "refuse"),
    ("no restoration receipt", [LEGACY, V5OK,
     {k: v for k, v in RB.items() if k != "collector_start_recv_ns"}],
     "refuse"),
    ("double-open second transition", [LEGACY, V5OK,
     {**V5OK, "boundary_utc": "2026-08-31T09:00:00Z"}], "refuse"),
    ("recovery bundle complete (DA 8bfcc9b shape)", [LEGACY,
     {**V5OK, "recovered": True, "stage": "stamp_unwritable_recovery",
      "collector_start_recv_ns": (B + 5) * 10**9}, RB], "ok"),
    ("UNCLOSED recovered transition (half-written bundle)", [LEGACY,
     {**V5OK, "recovered": True, "stage": "stamp_unwritable_recovery",
      "collector_start_recv_ns": (B + 5) * 10**9}], "refuse"),
    ("recovered without stage", [LEGACY,
     {**V5OK, "recovered": True,
      "collector_start_recv_ns": (B + 5) * 10**9}, RB], "refuse"),
    # Audit F1: three REALISTIC chains the fuzzer found where --pre-arm
    # proceeded on a ledger DA refuses. The 12 original fixtures missed all
    # three shapes; the walk now refuses them too.
    ("F1/D1 rollback copies the TRANSITION instant (zero-width era)",
     [LEGACY, V5OK, {**RB, "boundary_utc": V5OK["boundary_utc"]}], "refuse"),
    ("F1/D2 rollback restoration PREDATES the era it reverts",
     [LEGACY, V5OK, {**RB, "boundary_utc": "2026-08-31T06:00:00Z"}],
     "refuse"),
    ("F1/D3 aborted clob_v4 row while clob_v4 is LIVE",
     [LEGACY, {"collector_schema_version": "clob_v4", "aborted": True,
               "boundary_utc": "2026-08-31T07:00:00Z", "stage": "x"}],
     "refuse"),
    ("rollback with epoch-0 restoration recv_ns",
     [LEGACY, V5OK, {**RB, "collector_start_recv_ns": 0}], "refuse"),
    ("recovered as truthy 1, not bool (audit S7)",
     [LEGACY, {**V5OK, "recovered": 1}], "refuse"),
]


# V5-R4-1: the emitter's OWN output, retried — the case the 17-case set
# omitted and the mutation audit cannot see (it tests rows, not sequences).
REC = {**V5OK, "recovered": True, "stage": "stamp_unwritable_recovery",
       "collector_start_recv_ns": (B + 5) * 10**9, "stamp_written_ns": 3000}
RB_REC = {**RB, "stamp_written_ns": 4000}
CASES.append(("V5-R4-1 retried recovery bundle (emitter output, appended "
              "twice)", [LEGACY, REC, RB_REC,
                         {**REC, "stamp_written_ns": 3500}], "refuse"))
CASES.append(("V5-R4-1 single recovery bundle in order", [LEGACY, REC,
                                                          RB_REC], "ok"))


# Differential fuzz (17,729 ledgers, 735 disagreements at peak) — the classes
# the 19 canonical cases never reached. Ported so the seam test covers the
# space the fuzzer explored rather than the happy path with canonical stamps.
V6 = {"collector_schema_version": "clob_v6", "supersedes": "clob_v4",
      "transitioned": True, "boundary_utc": "2026-08-31T07:00:00Z",
      "stage": "post-restart",
      "collector_start_recv_ns": (B + 10) * 10**9}
RB_V6 = {"collector_schema_version": "clob_v4", "supersedes": "clob_v6",
         "rollback": True, "closes_boundary_utc": "2026-08-31T07:00:00Z",
         "boundary_utc": "2026-08-31T07:03:00Z", "stage": "stage-4",
         "collector_start_recv_ns": (B + 180) * 10**9}
FUZZ = [
    ("A8 return-transition bypassing the rollback contract",
     [LEGACY, V5OK, {"collector_schema_version": "clob_v4",
                     "supersedes": "clob_v5", "transitioned": True,
                     "boundary_utc": "2026-08-31T07:03:00Z"}], "refuse"),
    ("A9 unclosed recovered era of a NON-v5 version",
     [LEGACY, {**V6, "recovered": True}], "refuse"),
    ("A5 aborted row with no stage",
     [LEGACY, {"collector_schema_version": "clob_v5",
               "supersedes": "clob_v4", "aborted": True,
               "boundary_utc": "2026-08-31T07:00:00Z"}], "refuse"),
    ("A6 recovered:true on an aborted row",
     [LEGACY, {"collector_schema_version": "clob_v5",
               "supersedes": "clob_v4", "aborted": True, "recovered": True,
               "stage": "s", "boundary_utc": "2026-08-31T07:00:00Z"}],
     "refuse"),
    ("A3 row with an empty collector_schema_version",
     [LEGACY, {**V5OK, "collector_schema_version": ""}], "refuse"),
    ("A3b row with no boundary_utc",
     [LEGACY, {k: v for k, v in V5OK.items() if k != "boundary_utc"}],
     "refuse"),
    ("A7 first effective row with no supersedes",
     [{k: v for k, v in V5OK.items() if k != "supersedes"}], "refuse"),
    ("A10 recovered row with epoch-0 evidence",
     [LEGACY, {**V5OK, "recovered": True, "stage": "rec",
               "collector_start_recv_ns": 0}], "refuse"),
    ("A4 non-canonical boundary spelling (offset form)",
     [LEGACY, {**V5OK, "boundary_utc": "2026-08-31T07:00:00+00:00"}],
     "refuse"),
    ("A4b unparseable boundary",
     [LEGACY, {**V5OK, "boundary_utc": "not-a-time"}], "refuse"),
    ("C1 non-string boundary_utc",
     [LEGACY, {**V5OK, "boundary_utc": 20260831}], "refuse"),
    ("C2 non-string collector_schema_version",
     [LEGACY, {**V5OK, "collector_schema_version": 5}], "refuse"),
    ("A11 self-supersede on an aborted row",
     [LEGACY, {"collector_schema_version": "clob_v5",
               "supersedes": "clob_v5", "aborted": True, "stage": "s",
               "boundary_utc": "2026-08-31T07:00:00Z"}], "refuse"),
    ("B3 whitespace-only stage on a rollback",
     [LEGACY, V5OK, {**RB, "stage": "   "}], "refuse"),
    ("B1 rollback of a NON-v5 era (must be ACCEPTED — version-general)",
     [LEGACY, V6, RB_V6], "ok"),
    ("v4 -> v6 transition chain (no rollback, must be ACCEPTED)",
     [LEGACY, V6], "ok"),
]
CASES.extend(FUZZ)


def mine(rows):
    try:
        P.current_era_and_open_v5(rows)
        return "ok"
    except P.Refused:
        return "refuse"


def da(rows, path):
    """Only DA's DECLARED refusal (ValueError) counts as 'refuse'. Audit F3:
    a bare `except Exception` scored crashes, missing files and injected
    NameErrors as refusals, so two deliberately broken DA paths kept this
    test 12/12 green."""
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        # Admissibility (which eras are RULED usable) is a separate axis
        # from chain validity; this seam compares MALFORMEDNESS only, so a
        # permissive table is supplied. DA's own suite governs admissibility.
        D.day_era_admission("20260901", path,
                            admissible_table={"clob_v3_1": True,
                                              "clob_v4": True,
                                              "clob_v5": True,
                                              "clob_v6": True})
        return "ok"
    except ValueError:
        return "refuse"


def main() -> int:
    n_fail = 0
    with tempfile.TemporaryDirectory() as td:
        ledger = Path(td) / "eq_ledger.jsonl"
        for name, rows, want in CASES:
            m, d_ = mine(rows), da(rows, ledger)
            good = (m == d_ == want)
            n_fail += 0 if good else 1
            print(f"  {'PASS' if good else 'FAIL'}  {name}: mine={m} "
                  f"da={d_} expected={want}")
    if n_fail:
        print(f"CHAIN EQUIVALENCE: {n_fail} FAILURES")
        return 1
    print(f"chain equivalence: {len(CASES)} cases, both consumers agree on "
          f"every verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
