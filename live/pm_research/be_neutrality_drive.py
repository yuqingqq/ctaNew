#!/usr/bin/env python3
"""BE 129 -- drive `be_score_neutrality` against TWO BOOKS THE PIPELINE MADE DIFFER.

WHY THIS EXISTS. The comparator's 18 falsifier cells all build their own
synthetic books, so every difference it has ever been shown was one the test
author wrote. CLAUDE.md rule 15 wants a known-bad it must refuse and a positive
control it must admit, and rule 16's reading is that a control which has only
ever seen manufactured input has not been shown to fire on the real thing.

THE KNOWN-BAD IS FREE AND ALREADY ON DISK. `be_daybook_20260903_btc__L250ms__EV20`
and `__EV21` are the same day, coin and placement latency, built hours apart,
and EV21 EXCLUDED NINE ZERO-LENGTH GENERATIONS THAT EV20 RETAINED -- 313,149
against 313,140 in their own receipts. Nobody constructed that difference for a
test; the builder made it.

FOUR CELLS.

  A  REFUSE   certify(EV20, EV21) must raise KEYS_DIFFER naming 9 only-in-old
              and 0 only-in-new -- the count from the receipts, not from the
              comparator, so the cell can disagree with it.
  B  ADMIT    certify(EV21, EV21) must return BIT_IDENTICAL on a REAL book and
              D must reconcile against the RECEIPT's generation count. This is
              the cell that tests my BE 128 change to `n_generations_in_book`
              on a real shape; the synthetic cells cannot.
  C  ARITHMETIC on real scores: on the 313,140 keys the two books SHARE, per
              arm, delta_max, the flip count at the FROZEN theta, m_min and the
              occupancy curve. If the nine dropped generations were the only
              difference, delta on the shared keys is exactly 0 and the cell
              SAYS SO rather than implying more.
  D  RECONCILE the comparator's denominator against both receipts.

The verdict is COMPUTED (rule 10). Cell A failing to refuse, or refusing under
a different name, is a FAILED DRIVE and this exits non-zero.

Exit codes -- declared, and 75 is not among them (rule 20):
  0  every cell passed
  1  a cell FAILED (the drive did its job and the comparator did not)
  2  usage
  3  an input could not be read
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from live.pm_research.be_score_neutrality import (  # noqa: E402
    NeutralityRefused, KEYS_DIFFER, _entries, arm_heads, certify, gen_max,
    load, n_generations_in_book,
)

DERIVED = Path("data/pm_5min/derived")
OLD_REV, NEW_REV = "EV20", "EV21"
DAY, COIN, L = "20260903", "btc", "L250ms"

EXIT_CODES = {0: "ALL_CELLS_PASSED", 1: "A_CELL_FAILED",
              2: "USAGE", 3: "INPUT_UNREADABLE"}
assert 75 not in EXIT_CODES, "75 is the wrapper's conflict code (rule 20)"


def book_path(rev: str) -> Path:
    return DERIVED / f"be_daybook_{DAY}_{COIN}__{L}__{rev}.pkl"


def receipt_path(rev: str) -> Path:
    return DERIVED / f"be_daybook_receipt_{DAY}_{COIN}__{L}__{rev}.json"


def receipt_generations(rev: str) -> int:
    """The count from the PRODUCER's own receipt -- an independent number."""
    d = json.loads(receipt_path(rev).read_text())
    return int((d.get("reference") or {})["generations"])


def main() -> int:
    t0 = time.time()
    checks: list[tuple[str, bool, object]] = []

    def note(name, ok, detail=None):
        checks.append((name, bool(ok), detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f"   {detail}" if detail is not None else ""), flush=True)

    try:
        g_old_receipt = receipt_generations(OLD_REV)
        g_new_receipt = receipt_generations(NEW_REV)
    except (OSError, KeyError, ValueError) as exc:
        print(json.dumps({"refused": f"receipts unreadable: {exc}"}))
        return 3

    print(f"receipts: {OLD_REV}={g_old_receipt} {NEW_REV}={g_new_receipt} "
          f"delta={g_old_receipt - g_new_receipt}", flush=True)
    note("the two books DIFFER in the producers' own receipts",
         g_old_receipt != g_new_receipt,
         f"{g_old_receipt} vs {g_new_receipt}")

    try:
        old = load(str(book_path(OLD_REV)))
        print(f"loaded {OLD_REV} at {time.time() - t0:.0f}s", flush=True)
        new = load(str(book_path(NEW_REV)))
        print(f"loaded {NEW_REV} at {time.time() - t0:.0f}s", flush=True)
    except NeutralityRefused as exc:
        print(json.dumps({"refused": str(exc)}))
        return 3

    arms = arm_heads()
    result: dict = {
        "protocol": "BE_NEUTRALITY_DRIVE_V1",
        "what_this_is": (
            "the comparator driven against two books THE PIPELINE made differ; "
            "every earlier cell used a difference the test author wrote"),
        "old": {"revision": OLD_REV, "path": str(book_path(OLD_REV)),
                "receipt_generations": g_old_receipt},
        "new": {"revision": NEW_REV, "path": str(book_path(NEW_REV)),
                "receipt_generations": g_new_receipt},
        "arms": {a: {"head": v["head"], "theta": v["theta"],
                     "params_file": v["params_file"]} for a, v in arms.items()},
    }

    # ---- CELL A: the REFUSE direction, on a real difference.
    refusal = None
    try:
        certify(old, new)
    except NeutralityRefused as exc:
        refusal = str(exc)
    note("cell A: certify REFUSES two books the pipeline made differ",
         refusal is not None)
    note("cell A: it refuses under KEYS_DIFFER, not some other name",
         refusal is not None and KEYS_DIFFER in refusal, refusal)
    expect = (f"{g_old_receipt - g_new_receipt} generation(s) only in old "
              f"and 0 only in new")
    note("cell A: the refusal names the count the RECEIPTS imply",
         refusal is not None and expect in refusal, f"expected: {expect}")
    result["cell_A_refusal"] = refusal

    # ---- CELL B: the ADMIT direction, on a REAL book.
    admitted = None
    try:
        admitted = certify(new, new)
    except NeutralityRefused as exc:
        note("cell B: certify ADMITS a real book against itself", False,
             f"refused: {exc}")
    if admitted is not None:
        note("cell B: verdict is SUPPORTED_ON_THIS_DAY",
             admitted["verdict"] == "SUPPORTED_ON_THIS_DAY",
             admitted["verdict"])
        note("cell B: every arm reads BIT_IDENTICAL",
             all(r["strength"] == "BIT_IDENTICAL"
                 for r in admitted["per_arm"].values()),
             {a: r["strength"] for a, r in admitted["per_arm"].items()})
        note("cell B: zero flips on a real book against itself",
             admitted["n_flips_overall"] == 0)
        result["cell_B_admit"] = {
            "verdict": admitted["verdict"],
            "per_arm": {a: {k: r[k] for k in
                            ("strength", "C_n_generations",
                             "D_n_generations_in_book", "D_reconciles",
                             "C_m_min", "C_n_exactly_at_theta")}
                        for a, r in admitted["per_arm"].items()},
        }

    # ---- SHAPE PROBE: what coverage evidence does the BOOK itself carry?
    # The receipts say coverage is 0.7418 -- 232,307 covered of 313,140
    # reference generations, 80,833 GENERATION_NOT_SCORED. If `gen_max`
    # returns the covered count and `n_generations_in_book` returns the
    # reference count, D's reconciliation refuses every real book. The probe
    # records what the book carries so the repair reads the PRODUCER's number
    # rather than the receipt's (rule 28: carry the evidence or refuse on it).
    asm = (new.get("asm") or {})
    probe = {"asm_top_level_keys": sorted(str(k) for k in asm),
             "by_arm_keys": [str(k) for k in (asm.get("by_arm") or {})]}
    cov = asm.get("coverage_by_head") or asm.get("coverage") or {}
    probe["coverage_present_in_the_BOOK"] = bool(cov)
    if cov:
        probe["coverage_keys"] = {str(h): sorted(str(k) for k in v)
                                  for h, v in cov.items()
                                  if isinstance(v, dict)}
    for arm, spec in sorted(arms.items()):
        head = spec["head"]
        try:
            ents = _entries(new, head)
        except NeutralityRefused as exc:
            probe[f"entries[{arm}]"] = f"REFUSED: {exc}"
            continue
        n_rows = len(ents)
        n_null = sum(1 for v in ents.values() if v.get("score") is None)
        gens = {(k[0], k[1], v.get("gen")) for k, v in ents.items()}
        gens_scored = {(k[0], k[1], v.get("gen")) for k, v in ents.items()
                       if v.get("score") is not None}
        probe[f"entries[{arm}]"] = {
            "head": head,
            "n_rows": n_rows,
            "n_rows_with_a_null_score": n_null,
            "n_distinct_generations_among_rows": len(gens),
            "n_distinct_generations_among_SCORED_rows": len(gens_scored),
            "n_generations_dropped_by_the_null_skip": len(gens) - len(gens_scored),
            "sample_entry_keys": sorted(str(x) for x in
                                        list(ents.values())[0])[:12],
        }
        del ents, gens, gens_scored
        gc.collect()
    result["shape_probe"] = probe
    print(json.dumps(probe, indent=1, default=str)[:2000], flush=True)

    # ---- CELL D: the denominator against BOTH receipts.
    in_old, in_new = n_generations_in_book(old), n_generations_in_book(new)
    note("cell D: the comparator's denominator equals the OLD receipt",
         in_old == g_old_receipt, f"{in_old} vs {g_old_receipt}")
    note("cell D: the comparator's denominator equals the NEW receipt",
         in_new == g_new_receipt, f"{in_new} vs {g_new_receipt}")
    result["cell_D"] = {"denominator_old": in_old, "denominator_new": in_new}

    # ---- CELL C: the arithmetic on the keys the two books SHARE.
    shared: dict = {}
    for arm, spec in sorted(arms.items()):
        head, theta = spec["head"], spec["theta"]
        A, B = gen_max(old, head), gen_max(new, head)
        common = set(A) & set(B)
        only_old, only_new = len(set(A) - common), len(set(B) - common)
        deltas = [abs(B[g] - A[g]) for g in common]
        d_max = max(deltas) if deltas else 0.0
        n_nonzero = sum(1 for d in deltas if d != 0.0)
        flips = [g for g in common if (A[g] >= theta) != (B[g] >= theta)]
        margins = [abs(A[g] - theta) for g in common]
        m_min = min(margins) if margins else None
        occ = {f"within_10^{k}_x_delta_max":
               (sum(1 for m in margins if m < (10 ** k) * d_max)
                if d_max > 0 else None) for k in range(4)}
        shared[arm] = {
            "head": head, "theta": theta,
            "n_common": len(common), "n_only_old": only_old,
            "n_only_new": only_new,
            "delta_max_on_shared_keys": d_max,
            "n_generations_whose_score_moved": n_nonzero,
            "n_flips_on_shared_keys": len(flips),
            "m_min_on_shared_keys": m_min,
            "occupancy_on_shared_keys": occ,
            "n_exactly_at_theta": sum(1 for g in common if A[g] == theta),
            "READING": (
                "the nine dropped generations are the ONLY difference; every "
                "shared score is bit-identical" if d_max == 0.0 else
                "the exclusion moved scores on generations it did not drop"),
        }
        note(f"cell C[{arm}]: the shared-key count is the NEW receipt's count",
             len(common) == g_new_receipt, f"{len(common)} vs {g_new_receipt}")
        note(f"cell C[{arm}]: exactly the receipt delta is only-in-old",
             only_old == g_old_receipt - g_new_receipt
             and only_new == 0, f"only_old={only_old} only_new={only_new}")
        del A, B, common, deltas, margins
        gc.collect()
    result["cell_C_shared_keys"] = shared

    failed = [n for n, ok, _ in checks if not ok]
    result["drive"] = {
        "n_cells": len(checks), "n_failed": len(failed), "failed": failed,
        "wall_s": round(time.time() - t0, 1),
        "verdict": ("DRIVEN_ON_A_REAL_DIFFERENCE" if not failed
                    else "DRIVE_FAILED"),
        "WHAT_THIS_DOES_NOT_LICENSE": (
            "This drives the comparator, NOT the pipeline. It says the "
            "instrument refuses a real pipeline difference under the right "
            "name and admits a real book against itself. It says NOTHING "
            "about whether 7ed5a90 is score-neutral -- that is the "
            "NEUTCHK-vs-EV22 comparison, a different artifact."),
    }
    out = DERIVED / f"be_neutrality_drive_{OLD_REV}_vs_{NEW_REV}_{DAY}.json"
    out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["drive"], indent=1))
    print(f"wrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
