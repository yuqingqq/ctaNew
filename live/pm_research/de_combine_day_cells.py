"""COMBINE A DAY'S TWO ARM CELLS UNDER A DECLARED, MEASURED WAIVER.

`de_forward_evaluator` refused FORWARD_ARM_CELLS_DO_NOT_SHARE_ONE_DAY_INPUT
_COHORT on 09-07: the arms read the live settlement ledger 40 minutes apart
and it grew by 55 records in between -- rule 8, the tape grows during
measurement. Everything else compared IDENTICAL.

THE WAIVER IS NOT A SUPPRESSION. It names ONE field, requires the day's own
SUBSET of the ledger to be byte-identical under both snapshots, and carries
both snapshots plus the subset digest as fields. If the subset differs the
waiver REFUSES: a cohort difference that touches the day is exactly what
the guard exists to stop.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DE_COMBINE_DAY_CELLS_V1"
WAIVABLE = "winner_source"
NOT_WAIVED = "COMBINE_FIELD_DIFFERS_AND_IS_NOT_THE_WAIVED_ONE"
SUBSET_DIFFERS = "COMBINE_WAIVER_REFUSED_THE_DAY_SUBSET_DIFFERS"
NO_WAIVER = "COMBINE_WAIVER_NOT_DECLARED"
COMPARED = ("book_sha256", "zero_model_cancel_baseline_total_cents",
            "book_receipt", "params_pin", "score_neutrality_certifications",
            "forward_book_margin_guard", WAIVABLE)


class CombineRefused(RuntimeError):
    """A named refusal."""


def day_subset_digest(ledger: Path, n_records: int, day_start: int,
                      day_end: int) -> dict:
    """The day's own records inside the FIRST n_records of the ledger.

    The ledger is append-only, so a prefix IS the earlier snapshot -- the
    old bytes need not be kept to be compared.
    """
    lines = [x for x in Path(ledger).read_text().splitlines() if x.strip()]
    if len(lines) < n_records:
        raise CombineRefused(
            f"REFUSED {SUBSET_DIFFERS}: the ledger holds {len(lines)} "
            f"records and a snapshot claimed {n_records}; a prefix that "
            f"does not exist cannot be compared.")
    sub = []
    for line in lines[:n_records]:
        slug = str(json.loads(line).get("slug") or "")
        try:
            t = int(slug.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if day_start <= t < day_end:
            sub.append(line)
    return {"n_records_in_snapshot": n_records, "n_day_records": len(sub),
            "sha256": hashlib.sha256(
                "\n".join(sorted(sub)).encode()).hexdigest()}


def combine(cells: dict, *, ledger, day_start: int, day_end: int,
            waive=None) -> dict:
    """Compare the cells, apply the declared waiver, return the evidence."""
    arms = sorted(cells)
    a, b = cells[arms[0]], cells[arms[1]]
    differing = [f for f in COMPARED
                 if json.dumps(a.get(f), sort_keys=True)
                 != json.dumps(b.get(f), sort_keys=True)]
    beyond = [f for f in differing if f != WAIVABLE]
    if beyond:
        raise CombineRefused(
            f"REFUSED {NOT_WAIVED}: {beyond} differ between the arms and "
            f"the waiver names only {WAIVABLE!r}.")
    if not differing:
        return {"protocol": PROTOCOL, "waiver_needed": False,
                "fields_differing": []}
    if waive != WAIVABLE:
        raise CombineRefused(
            f"REFUSED {NO_WAIVER}: {WAIVABLE} differs and no waiver was "
            f"declared. A cohort difference is not waived by being small.")
    subsets = {}
    for arm in arms:
        n = (cells[arm].get(WAIVABLE) or {}).get("n_records")
        subsets[arm] = day_subset_digest(ledger, int(n), day_start, day_end)
    shas = {v["sha256"] for v in subsets.values()}
    if len(shas) != 1:
        raise CombineRefused(
            f"REFUSED {SUBSET_DIFFERS}: the day's own records are NOT "
            f"identical under the two snapshots {sorted(shas)}. The growth "
            f"touched this day, which is what the guard exists to stop.")
    return {"protocol": PROTOCOL, "waiver_needed": True,
            "waived_field": WAIVABLE, "fields_differing": differing,
            "snapshots": {arm: {k: (cells[arm].get(WAIVABLE) or {}).get(k)
                                for k in ("sha256", "n_records",
                                          "n_closed_records")}
                          for arm in arms},
            "day_subset_identity": subsets,
            "day_subset_sha256": shas.pop(),
            "why_this_is_sound": (
                "the ledger is append-only, so each snapshot's prefix IS "
                "the earlier state; the day's own records are byte-"
                "identical under both, so no number either arm computed "
                "could have differed"),
            "what_would_refuse_it": (
                "any differing field other than winner_source, an "
                "undeclared waiver, or a day subset that differs")}


def falsify() -> int:
    cells = ok = 0

    def ck(n, c):
        nonlocal cells, ok
        cells += 1
        ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        led = Path(td) / "resolutions.jsonl"
        day_rows = [json.dumps({"slug": f"btc-updown-5m-{1788739200 + i*300}"})
                    for i in range(10)]
        other = [json.dumps({"slug": f"btc-updown-5m-{1789000000 + i*300}"})
                 for i in range(5)]
        led.write_text("\n".join(day_rows + other) + "\n")
        base = {"book_sha256": "a", "zero_model_cancel_baseline_total_cents":
                1.0, "book_receipt": {}, "params_pin": {},
                "score_neutrality_certifications": [],
                "forward_book_margin_guard": {}}
        c = {"A": dict(base, winner_source={"sha256": "x", "n_records": 10,
                                            "n_closed_records": 10}),
             "B": dict(base, winner_source={"sha256": "y", "n_records": 15,
                                            "n_closed_records": 15})}
        r = combine(c, ledger=led, day_start=1788739200, day_end=1788825600,
                    waive=WAIVABLE)
        ck("growth OUTSIDE the day -> waiver holds, subset identical",
           r["waiver_needed"] and len({v["sha256"] for v in
                                       r["day_subset_identity"].values()}) == 1)
        try:
            combine(c, ledger=led, day_start=1788739200, day_end=1788825600)
            ck("an UNDECLARED waiver REFUSES", False)
        except CombineRefused as e:
            ck("an UNDECLARED waiver REFUSES", NO_WAIVER in str(e))
        led.write_text("\n".join(day_rows + [json.dumps(
            {"slug": f"btc-updown-5m-{1788739200 + 99*300}"})] + other) + "\n")
        try:
            combine({"A": c["A"], "B": dict(c["B"], winner_source={
                "sha256": "y", "n_records": 11, "n_closed_records": 11})},
                ledger=led, day_start=1788739200, day_end=1788825600,
                waive=WAIVABLE)
            ck("growth INSIDE the day REFUSES the waiver", False)
        except CombineRefused as e:
            ck("growth INSIDE the day REFUSES the waiver",
               SUBSET_DIFFERS in str(e))
        try:
            combine({"A": c["A"], "B": dict(c["B"], book_sha256="zzz")},
                    ledger=led, day_start=1788739200, day_end=1788825600,
                    waive=WAIVABLE)
            ck("a DIFFERENT field differing REFUSES, waiver notwithstanding",
               False)
        except CombineRefused as e:
            ck("a DIFFERENT field differing REFUSES, waiver notwithstanding",
               NOT_WAIVED in str(e))
    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


if __name__ == "__main__":
    raise SystemExit(falsify() if "--falsify" in sys.argv[1:] else 2)
