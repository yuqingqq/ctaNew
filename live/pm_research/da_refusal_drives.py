#!/usr/bin/env python3
"""DA-OWNED TESTS THAT DRIVE OTHER SEATS' REFUSALS.

`p003_refusal_coverage` found 24 `REFUSED <NAME>` tokens that no test names.
Most live in DE's modules and DA does not edit another seat's module -- but
DA can DRIVE the refusal from here, which closes the coverage gap without
touching the code under test and, more importantly, actually observes the
branch fire.

The pattern: import the owning module, construct the input that SHOULD refuse,
observe the refusal BY NAME, and drive the positive control beside it so the
test cannot pass by refusing everything.

DA 199: the first entry is `SETTLEMENT_NULL_TOO_SMALL`, driven because step 2
was drawing behind it with no test naming it.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PARAMS = HERE / "declarations" / "de_multiday_gate1_params_v29.json"

_N = {"n": 0, "bad": 0}


def ok(cond, label):
    _N["n"] += 1
    if not cond:
        _N["bad"] += 1
        print(f"  FAIL {label}")
    return cond


def drive_settlement_null_too_small() -> dict:
    """DE's `settlement_arm_day` guard, DRIVEN.

    This is the bar between step 2's published p and an under-sampled null.
    It is the params value, read at run time, NOT the colliding `MIN_DRAWS`
    constant -- so it cannot be moved by renaming that constant."""
    import de_multiday_gate1_runner as R
    P = json.loads(PARAMS.read_text())
    bar = P["min_draws_per_arm_day"]
    fired = {}
    for n in (0, 1, 199, 200, bar - 1):
        try:
            R.settlement_arm_day(12345.0, [float(i) for i in range(n)], P)
            fired[n] = None
        except R.RunnerRefused as e:
            fired[n] = str(e)
    ok(all(v and "SETTLEMENT_NULL_TOO_SMALL" in v for v in fired.values()),
       f"REFUSED SETTLEMENT_NULL_TOO_SMALL fires at every short count "
       f"{sorted(fired)} against the declared bar {bar}")
    ok(fired.get(200) and "SETTLEMENT_NULL_TOO_SMALL" in fired[200],
       "and it fires at 200 -- the settlement path enforces the PARAMS bar "
       "of 500, NOT the rule-6 floor of 200. The stricter bar wins, which is "
       "fail-safe, and a seat reading only the floor would be surprised")
    ok(all(str(n) in fired[n] for n in fired),
       "the refusal NAMES THE COUNT it saw, so a short run says how short")
    # POSITIVE CONTROL: the test must not pass by refusing everything.
    random.seed(7)
    out = R.settlement_arm_day(12345.0,
                               [random.gauss(0, 1000.0) for _ in range(bar)],
                               P)
    ok(isinstance(out, dict) and out.get("p_location") is not None,
       f"POSITIVE CONTROL: at exactly {bar} draws the guard PASSES and a "
       f"p is formed (p={out.get('p_location')}) -- the test does not pass "
       f"by refusing everything")
    return {"bar": bar, "refused_at": sorted(k for k, v in fired.items() if v),
            "passes_at": bar}


def selftest(quiet: bool = False) -> int:
    res = drive_settlement_null_too_small()
    if not quiet:
        print(f"[da_refusal_drives] {_N['n'] - _N['bad']}/{_N['n']} checks, "
              f"{_N['bad']} failures | SETTLEMENT_NULL_TOO_SMALL refused at "
              f"{res['refused_at']}, passed at {res['passes_at']}")
    return 1 if _N["bad"] else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(drive_settlement_null_too_small(), indent=1))
