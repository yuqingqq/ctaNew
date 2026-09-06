"""CAN ONE DAY'S ASSEMBLY FIT UNDER 8 GB? COMPUTED FROM MEASUREMENTS, NOT SIZED BY EYE.

Every component below was measured on 2026-09-03, btc, this seat, under the
8 GB cap. Nothing here is an estimate scaled from the consumed hour.

    reference                247 windows / 313,114 generations   2.008 GB
    reference + tape index   1,764,206 tape rows                  5.971 GB
    fragment, whole          576.6 MiB on disk, 545,240 rows      2.742 GB
    asm                      NOT MEASURED -- it does not exist yet

THE ANSWER IS NO, AND IT IS NOT CLOSE IN THE DIRECTION THAT MATTERS. The
resident floor before a single byte of `asm` is 5.971 GB, leaving 2.029 GB.
The fragment materialised whole needs 2.742 GB. That is 0.713 GB OVER the
cap with zero allocated for the thing the assembly exists to produce.

SO THE STREAMED ASSEMBLY IS REQUIRED, NOT PREFERRED -- and DE's own docstring
already recorded the consequence of learning this the other way: "the ruled
run of 2026-09-03 died because [the 3.9 GB index] was held alongside the
entire fragment."

R8 GOVERNS THE REMEDY. The cap is not raised and the population is not
reduced. What changes is the ORDER and the RESIDENCY, not the question.

THIS IS A DECLARATION. It ships no assembly code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR

ROOT = HERE.parents[1]
OUT_DERIVED = ROOT / "data/pm_5min/derived"

CAP_GB = 8.0

#: MEASURED, this seat, 2026-09-06, day 20260903 btc. Each carries where.
M = {
    "reference_gb": 2.008,
    "reference_and_tape_index_gb": 5.971,
    "tape_index_alone_gb": round(5.971 - 2.008, 3),
    "fragment_file_mib": 576.6,
    "fragment_materialised_whole_gb": 2.742,
    "fragment_materialisation_factor": 4.87,
    "n_windows": 247,
    "n_generations": 313114,
    "n_fragment_rows": 545240,
    "n_tape_rows": 1764206,
    "asm_gb": None,
}
SOURCES = {
    "reference_gb": "be_daybook_build_attempt_20260903_btc.json "
                    "(round 48, reference stage)",
    "reference_and_tape_index_gb": "same receipt, after_tape_peak_gb",
    "fragment_file_mib": "be_gate1_fragment_receipt_20260903_btc.json",
    "fragment_materialised_whole_gb": "measured this round: read + "
                                      "json.loads under the 8 GB scope",
    "asm_gb": "NOT MEASURED -- no assembly has completed on a September day",
}


def budget() -> dict:
    floor = M["reference_and_tape_index_gb"]
    head = round(CAP_GB - floor, 3)
    whole = M["fragment_materialised_whole_gb"]
    total_whole = round(floor + whole, 3)
    per_window_mb = round(whole * 1024 / M["n_windows"], 2)
    return {
        "cap_gb": CAP_GB,
        "resident_floor_gb": floor,
        "what_is_in_the_floor": ["the day reference (2.008 GB)",
                                 f"the tape index ({M['tape_index_alone_gb']} "
                                 f"GB, both ruled splits)"],
        "headroom_after_the_floor_gb": head,
        "fragment_whole_gb": whole,
        "total_if_fragment_is_whole_gb": total_whole,
        "FITS_WITH_WHOLE_FRAGMENT": total_whole <= CAP_GB,
        "over_by_gb": round(max(0.0, total_whole - CAP_GB), 3),
        "and_that_is_before_asm": True,
        "asm_budget_gb": "UNKNOWN — asm has never been built for a "
                         "September day, so the true requirement is "
                         "headroom MINUS asm, and asm is the term nobody "
                         "has measured",
        "fragment_resident_per_window_mb": per_window_mb,
        "max_windows_resident_within_headroom": int(head * 1024 //
                                                    per_window_mb),
        "computed_not_asserted": True,
    }


def streamed_plan() -> dict:
    b = budget()
    return {
        "REQUIRED_BECAUSE": f"{b['total_if_fragment_is_whole_gb']} GB > "
                            f"{CAP_GB} GB with 0 allocated for asm",
        "WHAT_STAYS_RESIDENT": [
            {"object": "tape index", "gb": M["tape_index_alone_gb"],
             "why": "it is the shared index every chunk is joined against; "
                    "rebuilding it per chunk would cost 387 s each time"},
            {"object": "day reference", "gb": M["reference_gb"],
             "why": "assemble_streaming takes {coin: ref} and the generation "
                    "table is the join key"},
            {"object": "asm, accumulating", "gb": None,
             "why": "it is the output; it grows as chunks land and is the "
                    "one term that has never been measured"},
        ],
        "WHAT_IS_STREAMED": {
            "object": "the day fragment",
            "unit": "windows",
            "mechanism": "de_phase4_diag_runner.assemble_streaming("
                         "chunk_windows=N, source=<fragment>)",
            "never_resident_whole": True,
            "measured_cost_per_window_mb": b["fragment_resident_per_window_mb"],
        },
        "ORDER": [
            "1. build the day reference (531 s) and keep it",
            "2. build the tape index ONCE (387 s) and keep it -- the floor is "
            "now 5.971 GB and is not revisited",
            "3. build the day fragment to disk (608 s) -- it is NOT loaded "
            "here; the builder streams it out and never holds it",
            "4. stream the fragment in chunks of N windows into "
            "assemble_streaming, against the resident index and reference",
            "5. emit asm with the book; measure asm's peak and publish it, "
            "because it is the term this budget cannot close",
        ],
        "CHUNK_SIZING": {
            "headroom_gb": b["headroom_after_the_floor_gb"],
            "per_window_mb": b["fragment_resident_per_window_mb"],
            "windows_that_fit_headroom_ignoring_asm":
                b["max_windows_resident_within_headroom"],
            "recommended_chunk_windows": 6,
            "why_6_and_not_the_maximum": "6 windows is ~68 MB resident, "
                                         "about 3% of headroom, leaving "
                                         "essentially all of it for asm -- "
                                         "the unmeasured term. Sizing the "
                                         "chunk to the maximum spends the "
                                         "budget on the term that is known "
                                         "and starves the one that is not.",
            "6_is_also_what_the_arms_run_used": True,
        },
        "THE_REFUSAL_IF_IT_STILL_DOES_NOT_FIT": {
            "rule": "R8 -- the cap is NOT raised and the population is NOT "
                    "reduced. A day that will not assemble is REPORTED with "
                    "its measured peak and REFUSED.",
            "not_permitted": ["a larger MemoryMax",
                              "fewer windows than the day supplies",
                              "dropping a ruled split"],
        },
        "THE_LARGEST_LEVER_IS_NOT_MINE_TO_PULL": {
            "observation": f"the tape index is {M['tape_index_alone_gb']} GB "
                           f"of the {b['resident_floor_gb']} GB floor, and it "
                           f"is built over BOTH ruled splits "
                           f"(MECHANICS_BOTH_SPLITS). DE's own docstring "
                           f"measures the score split alone at 1.42 GB "
                           f"against 3.90 GB cumulative for both.",
            "consequence": "if one split were sufficient the floor would "
                           "fall by roughly 2.5 GB and the whole question "
                           "would close.",
            "why_this_seat_does_not_do_it": "the split set is RULED. "
                                            "Changing it is a declaration "
                                            "change, not a memory "
                                            "optimisation, and it is the "
                                            "coordinator's and DE's.",
            "routed_not_taken": True,
        },
    }


def build() -> dict:
    return {
        "protocol": "BE_ASSEMBLY_BUDGET_DECLARATION_V1",
        "day_measured": "20260903", "coin": "btc",
        "scope": "one day's assembly under the 8 GB cap. A DECLARATION for "
                 "DE's runner and this seat's builder; it ships no assembly "
                 "code.",
        "measurements": M,
        "measurement_sources": SOURCES,
        "budget": budget(),
        "streamed_plan": streamed_plan(),
        "data_root": _BDR.receipt_block(),
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 7


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    b = budget()
    ok(b["FITS_WITH_WHOLE_FRAGMENT"] is False and b["over_by_gb"] > 0,
       f"THE COMPUTED ANSWER IS NO: floor {b['resident_floor_gb']} GB + "
       f"whole fragment {b['fragment_whole_gb']} GB = "
       f"{b['total_if_fragment_is_whole_gb']} GB, OVER the {CAP_GB} GB cap "
       f"by {b['over_by_gb']} GB -- and with ZERO allocated for asm")
    ok(b["headroom_after_the_floor_gb"] == round(CAP_GB - 5.971, 3)
       and b["headroom_after_the_floor_gb"] < b["fragment_whole_gb"],
       f"the headroom after the resident floor is "
       f"{b['headroom_after_the_floor_gb']} GB and the whole fragment needs "
       f"{b['fragment_whole_gb']} GB -- the shortfall is the finding, not "
       f"the total")
    ok(M["asm_gb"] is None and "UNKNOWN" in b["asm_budget_gb"],
       "and `asm` is declared UNMEASURED rather than given a number -- it "
       "has never been built for a September day, so the budget cannot be "
       "closed and says so")
    ok(abs(M["fragment_materialisation_factor"]
           - M["fragment_materialised_whole_gb"] * 1024
           / M["fragment_file_mib"]) < 0.05,
       f"the {M['fragment_materialisation_factor']}x materialisation factor "
       f"is CONSISTENT with its own components "
       f"({M['fragment_materialised_whole_gb']} GB from "
       f"{M['fragment_file_mib']} MiB) -- measured, not a rule of thumb")
    p = streamed_plan()
    ok(p["CHUNK_SIZING"]["recommended_chunk_windows"] <
       p["CHUNK_SIZING"]["windows_that_fit_headroom_ignoring_asm"],
       f"the recommended chunk ({p['CHUNK_SIZING']['recommended_chunk_windows']}"
       f" windows) is far BELOW what headroom alone would allow "
       f"({p['CHUNK_SIZING']['windows_that_fit_headroom_ignoring_asm']}) -- "
       f"because the budget must be left to the term nobody has measured")
    ok(any(x["object"] == "the day fragment" or
           x.get("object") == "the day fragment"
           for x in [p["WHAT_IS_STREAMED"]])
       and p["WHAT_IS_STREAMED"]["never_resident_whole"],
       "the plan names WHAT IS STREAMED (the fragment, by windows) and WHAT "
       "STAYS RESIDENT (index, reference, accumulating asm), in order")
    ok("NOT raised" in p["THE_REFUSAL_IF_IT_STILL_DOES_NOT_FIT"]["rule"]
       and p["THE_LARGEST_LEVER_IS_NOT_MINE_TO_PULL"]["routed_not_taken"],
       "R8's refusal is written down, and the one lever that would close the "
       "question -- the ruled split set -- is ROUTED rather than pulled")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        out = build()
        dst = (HERE / "declarations" / "be_assembly_budget_declaration_v1.json")
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst),
                          "fits_whole": out["budget"]["FITS_WITH_WHOLE_FRAGMENT"],
                          "over_by_gb": out["budget"]["over_by_gb"]}))
        return 0
    print("usage: be_assembly_budget.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
