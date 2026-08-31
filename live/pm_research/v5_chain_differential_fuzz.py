#!/usr/bin/env python3
"""Differential fuzz of the two era-ledger consumers — RETAINED, not ad hoc.

Codex V5-P5-1: "Re-run the differential generator as an executable, retained
artifact; the reported 17,729-ledger fuzz itself is not committed here, only
selected repros are." A finding produced by a script nobody can re-run is a
claim, not a result — so the generator lives here and reruns on demand.

Consumer A: v5_boundary_preflight.current_era_and_open_v5 (raises Refused)
Consumer B: da_forward_day_verify.day_era_admission  (raises ValueError)

Admissibility is neutralised with a permissive table: this compares CHAIN
VALIDITY, which is the shared axis. Which eras are RULED usable is DA's own
axis and its own suite governs it.

Exit 0 = no disagreement. Exit 1 = disagreements, each printed with a minimal
reproducing ledger.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_boundary_preflight as P  # noqa: E402
import da_forward_day_verify as D  # noqa: E402

B = P.BOUNDARY_EPOCH
PERMISSIVE = {f"clob_v{n}": True for n in
              ("3_1", "4", "5", "6", "7")}

LEGACY = {"collector_schema_version": "clob_v4", "supersedes": "clob_v3_1",
          "boundary_utc": "2026-08-30T05:30:00Z"}


def transition(ver, sup, hh, recovered=False):
    row = {"collector_schema_version": ver, "supersedes": sup,
           "transitioned": True,
           "boundary_utc": f"2026-08-31T{hh:02d}:00:00Z",
           "stage": "post-restart",
           "collector_start_recv_ns": (B + hh * 3600 + 10) * 10**9}
    if recovered:
        row["recovered"] = True
    return row


def rollback(sup, closes_hh, hh):
    return {"collector_schema_version": "clob_v4", "supersedes": sup,
            "rollback": True,
            "closes_boundary_utc": f"2026-08-31T{closes_hh:02d}:00:00Z",
            "boundary_utc": f"2026-08-31T{hh:02d}:00:00Z",
            "stage": "counters_refused",
            "collector_start_recv_ns": (B + hh * 3600 + 20) * 10**9}


def abort(ver, hh):
    return {"collector_schema_version": ver, "supersedes": "clob_v4",
            "aborted": True, "stage": "restart_failed",
            "boundary_utc": f"2026-08-31T{hh:02d}:00:00Z"}


MUTATIONS = [
    ("none", lambda r: r),
    ("drop_supersedes", lambda r: {k: v for k, v in r.items()
                                   if k != "supersedes"}),
    ("drop_boundary", lambda r: {k: v for k, v in r.items()
                                 if k != "boundary_utc"}),
    ("drop_stage", lambda r: {k: v for k, v in r.items() if k != "stage"}),
    ("blank_stage", lambda r: {**r, "stage": "   "}),
    ("self_supersede", lambda r: {**r,
                                  "supersedes": r.get(
                                      "collector_schema_version")}),
    ("nonbool_flag", lambda r: {**r, **({"transitioned": 1}
                                        if r.get("transitioned") else
                                        {"aborted": 1}
                                        if r.get("aborted") else
                                        {"rollback": 1})}),
    ("two_flags", lambda r: {**r, "aborted": True, "transitioned": True}),
    ("int_version", lambda r: {**r, "collector_schema_version": 5}),
    ("empty_version", lambda r: {**r, "collector_schema_version": ""}),
    ("offset_instant", lambda r: {**r,
                                  "boundary_utc":
                                  "2026-08-31T07:00:00+00:00"}),
    ("bare_date", lambda r: {**r, "boundary_utc": "2026-08-31"}),
    ("nonstr_instant", lambda r: {**r, "boundary_utc": 20260831}),
    ("epoch0_recv", lambda r: {**r, "collector_start_recv_ns": 0}),
    ("float_recv", lambda r: {**r, "collector_start_recv_ns": 1.5}),
    ("recovered_qualifier", lambda r: {**r, "recovered": True}),
]

CHAINS = [
    ("v5 open", [LEGACY, transition("clob_v5", "clob_v4", 7)]),
    ("v5 rolled back", [LEGACY, transition("clob_v5", "clob_v4", 7),
                        rollback("clob_v5", 7, 8)]),
    ("v5 recovered bundle", [LEGACY,
                             transition("clob_v5", "clob_v4", 7, True),
                             rollback("clob_v5", 7, 8)]),
    ("v5 aborted", [LEGACY, abort("clob_v5", 7)]),
    ("v4->v5->v6", [LEGACY, transition("clob_v5", "clob_v4", 7),
                    transition("clob_v6", "clob_v5", 8)]),
    ("multi-hop return", [LEGACY, transition("clob_v5", "clob_v4", 7),
                          transition("clob_v6", "clob_v5", 8),
                          transition("clob_v4", "clob_v6", 9)]),
    ("one-hop return", [LEGACY, transition("clob_v5", "clob_v4", 7),
                        transition("clob_v4", "clob_v5", 8)]),
    ("retry after rollback", [LEGACY, transition("clob_v5", "clob_v4", 7),
                              rollback("clob_v5", 7, 8),
                              transition("clob_v5", "clob_v4", 9)]),
    ("retry after multi-hop rollback",
     [LEGACY, transition("clob_v5", "clob_v4", 7),
      transition("clob_v6", "clob_v5", 8), rollback("clob_v6", 8, 9),
      transition("clob_v5", "clob_v4", 10)]),
    # audit A3: emitter ACCEPTED and DA REFUSED here; unreachable with two
    # versions, arms on the third. The open era is v6, created by a plain
    # transition, so the retry exemption must NOT apply to the v5 return.
    ("return after an INTERVENING version",
     [LEGACY, transition("clob_v5", "clob_v4", 7), rollback("clob_v5", 7, 8),
      transition("clob_v6", "clob_v4", 9),
      transition("clob_v5", "clob_v6", 10)]),
    ("return after two intervening versions",
     [LEGACY, transition("clob_v5", "clob_v4", 7),
      transition("clob_v6", "clob_v5", 8),
      transition("clob_v7", "clob_v6", 9),
      transition("clob_v5", "clob_v7", 10)]),
    ("retried bundle", [LEGACY, transition("clob_v5", "clob_v4", 7, True),
                        rollback("clob_v5", 7, 8),
                        transition("clob_v5", "clob_v4", 7, True)]),
]


def verdict_a(rows):
    try:
        P.current_era_and_open_v5(rows)
        return "ok", ""
    except P.Refused as ex:
        return "refuse", str(ex)[:70]


def verdict_b(rows, path):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        D.day_era_admission("20260901", path, admissible_table=PERMISSIVE)
        return "ok", ""
    except ValueError as ex:
        return "refuse", str(ex)[:70]
    except Exception as ex:                                   # noqa: BLE001
        return "CRASH", f"{type(ex).__name__}: {str(ex)[:60]}"


def main() -> int:
    rng = random.Random(20260831)
    disagreements, tested = [], 0
    with tempfile.TemporaryDirectory() as td:
        ledger = Path(td) / "fuzz.jsonl"
        for (cname, chain), (mname, mut) in itertools.product(CHAINS,
                                                              MUTATIONS):
            for idx in range(len(chain)):
                rows = [mut(r) if i == idx else r
                        for i, r in enumerate(chain)]
                tested += 1
                a, a_why = verdict_a(rows)
                b, b_why = verdict_b(rows, ledger)
                if a != b:
                    disagreements.append((cname, mname, idx, a, a_why,
                                          b, b_why))
        # randomized shuffles and duplications on top of the grid
        for _ in range(600):
            cname, chain = rng.choice(CHAINS)
            rows = list(chain)
            if rng.random() < 0.5 and len(rows) > 2:
                i, j = rng.randrange(1, len(rows)), rng.randrange(1, len(rows))
                rows[i], rows[j] = rows[j], rows[i]
            if rng.random() < 0.5:
                rows.append(rng.choice(rows[1:]))
            tested += 1
            a, a_why = verdict_a(rows)
            b, b_why = verdict_b(rows, ledger)
            if a != b:
                disagreements.append((cname, "shuffle/dup", -1, a, a_why,
                                      b, b_why))
    seen = set()
    for d in disagreements:
        key = (d[0], d[1], d[3], d[5])
        if key in seen:
            continue
        seen.add(key)
        print(f"  DISAGREE [{d[0]} | {d[1]} | row {d[2]}] "
              f"A={d[3]} ({d[4]}) B={d[5]} ({d[6]})")
    print(f"differential fuzz: {tested} ledgers, "
          f"{len(disagreements)} disagreements "
          f"({len(seen)} distinct classes)")
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
