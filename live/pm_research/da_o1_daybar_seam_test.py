"""SEAM: the O1 producer's REAL emitted row, consumed by the day-bar.

DB2 closure. Both suites were green in isolation while their integration always
refused the moment O1d fired: the day-bar test used a HAND-BUILT missing-end
event, and the O1 test checked the producer's start stamp but never passed its
emitted row through the day-bar. Neither suite could see the disagreement,
because neither crossed the seam.

So this test does not construct a row. It DRIVES the committed v4 producer
against a fake socket, reads the row the producer ACTUALLY WROTE to its ledger,
and feeds that file to `day_bar_v2`. Nothing in the middle is hand-shaped.

Light class: fake sockets, compressed clock, scratch paths. No tape, no network.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PASS: list[str] = []
FAIL: list[str] = []


def ok(c: bool, label: str) -> None:
    print(f"  {'PASS' if c else 'FAIL'}  {label}")
    (PASS if c else FAIL).append(label)


def produce_real_open_at_exit_row() -> dict:
    """Drive the committed v4 producer until O1d fires; return its ledger row."""
    import collect_pm_v4_behavior_tests as H   # the coordinator's harness (c288ed1)
    cap: dict = {}
    rows = asyncio.run(H.run_market(["refuse_connect"] * 50, cap, window_s=2))
    opens = [r for r in rows if r.get("event") == "gap_open_at_exit"]
    if len(opens) != 1:
        raise AssertionError(
            f"REFUSED: expected exactly one gap_open_at_exit from the never-"
            f"connected path, got {len(opens)}. The seam cannot be tested "
            f"against a row the producer did not emit.")
    return opens[0]


def main() -> int:
    import da_forward_day_verify as F

    print("SEAM: O1 producer -> day-bar consumer\n")

    row = produce_real_open_at_exit_row()
    print(f"  producer emitted: event={row.get('event')} "
          f"gap_start_ns={row.get('gap_start_ns')} "
          f"gap_end_ns={row.get('gap_end_ns')} cause={row.get('cause')}")
    ok(row.get("gap_end_ns") is not None,
       "the REAL producer stamps a finite gap_end_ns on gap_open_at_exit -- "
       "the fact the day-bar's hand-built fixture never contained")
    ok(isinstance(row.get("gap_start_ns"), int)
       and row["gap_end_ns"] > row["gap_start_ns"],
       "and the emitted interval is ordered (end > start)")

    # ---- consume THAT row with the day-bar, over the producer's own day ----
    gs = row["gap_start_ns"] / 1e9
    import datetime as dt
    day = dt.datetime.fromtimestamp(gs, dt.timezone.utc).strftime("%Y%m%d")
    lo, hi = F.day_bounds(day)
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "producer_emitted.jsonl"
        led.write_text(json.dumps(row), encoding="utf-8")
        diag: dict = {}
        try:
            iv = F.coin_gap_intervals(lo, hi, row.get("coin", "btc"), led,
                                      diag=diag)
            consumed = True
        except ValueError as e:
            consumed = False
            print(f"  REFUSED: {str(e)[:110]}")
        ok(consumed,
           "the day-bar CONSUMES the producer's real row -- before DB2 this "
           "refused, so the integration failed the moment O1d fired while both "
           "suites stayed green")
        if consumed:
            ok(len(iv) == 1,
               "exactly one interval is charged from the emitted row")
            ok(diag.get("producer_supplied_ends_used") == 1
               and diag.get("synthesized_ends_charged_to_scope_end") == 0,
               "the PRODUCER's end is used and counted; synthesis did NOT fire "
               "-- the producer knows when its task exited, the consumer does not")
            dur = (row["gap_end_ns"] - row["gap_start_ns"]) / 1e9
            charged = iv[0][1] - iv[0][0]
            ok(abs(charged - dur) < 1e-6,
               f"and the charged duration equals the producer's own "
               f"({charged:.3f}s == {dur:.3f}s) -- not a scope-end synthesis "
               f"that would have silently overstated it")

        # ---- KNOWN-BAD: the reviewer's probe row shape must still refuse ----
        bad = dict(row)
        bad["gap_end_ns"] = row["gap_start_ns"] - 1     # end precedes start
        bad_led = Path(td) / "bad.jsonl"
        bad_led.write_text(json.dumps(bad), encoding="utf-8")
        refused = False
        try:
            F.coin_gap_intervals(lo, hi, row.get("coin", "btc"), bad_led)
        except ValueError:
            refused = True
        ok(refused,
           "KNOWN-BAD: a producer-shaped row whose end PRECEDES its start still "
           "REFUSES -- accepting the producer's end is not accepting any end")

    print(f"\n{'O1<->DAY-BAR SEAM GREEN' if not FAIL else 'SEAM RED'}: "
          f"{len(FAIL)} failing, {len(PASS) + len(FAIL)} checks")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
