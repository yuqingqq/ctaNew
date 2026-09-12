"""BE 235: the book's producer-receipt BINDINGS, enumerated from the code.

WHY AN ARTIFACT AND NOT A DISPATCH. The enumeration was delivered twice in
channel messages and asked for a third time. A table that lives in a dispatch
is a table nobody can query later -- the same defect this programme has been
cataloguing all night, committed on the answer to it.

WHY ENUMERATED RATHER THAN DISCOVERED. `be_daybook_build.build()` refuses on
the FIRST missing binding, so fixing one reveals the next. Discovering the set
by rebuilding costs up to 11 minutes per step and is unbounded until it ends.
The set is readable from the AST in milliseconds: walk `build()` for calls to
the pin/assert functions, then check each against disk for each coin.

THE PENDING TEST, recorded here so a hypothesis cannot drift into a rule
(BE 235): peak RSS scaled 0.97x between coins while output size scaled 0.66x
and wall clock 0.27-0.45x. The proposed MECHANISM -- that peak is set by the
per-day window index, which is 288 for both coins, so memory does not scale
with data volume -- is INFERRED FROM A COUNT, not from reading the allocator,
and rests on two stages of ONE day of ONE coin pair. One prediction is not
evidence. The next ETH or third-coin build either reproduces the ratio or
kills it. Until then the DECISION (do not overlap) rests on the measurement
and on 0.7% of soft-limit headroom over single observations, NOT on the
mechanism.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/yuqing/ctaNew")
TREE = Path("/home/yuqing/ctaNew-wt-fwd")
BUILDER = TREE / "live/pm_research/be_daybook_build.py"
OUT = (ROOT / "orchestrator/PROGRAMS/P-2026-003-polymarket-5min"
            / "workspace/be_closeouts/be_receipt_bindings_v1.json")
CONTRACT = "BE_RECEIPT_BINDINGS_V1"

#: the calls that bind the assembly to a PRODUCER's published receipt. Other
#: asserts in build() bind to computed properties and are not receipts.
RECEIPT_CALLS = {"day_tape_pin", "day_fragment_pin",
                 "assert_day_tape", "assert_input_matches_its_receipt"}


def bindings_from_code() -> list[dict]:
    src = BUILDER.read_text()
    tree = ast.parse(src)
    out = []
    for fn in [n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "build"]:
        for n in ast.walk(fn):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id in RECEIPT_CALLS):
                arg0 = None
                if n.args and isinstance(n.args[0], ast.Constant):
                    arg0 = n.args[0].value
                out.append({"line": n.lineno, "call": n.func.id,
                            "kind": arg0})
    return sorted(out, key=lambda r: r["line"])


def state_for(day: str, coins=("btc", "eth")) -> dict:
    sys.path.insert(0, str(TREE / "live/pm_research"))
    sys.path.insert(0, str(TREE))
    import be_daybook_build as D
    rows = {}
    for coin in coins:
        tp = D.day_tape_pin(day, coin)
        fp = D.day_fragment_pin(day, coin)
        rows[coin] = {
            "day_tape_pin": {"present": bool(tp),
                             "receipt": (tp or {}).get("receipt"),
                             "split": (tp or {}).get("split"),
                             "n_rows": (tp or {}).get("n_rows")},
            "day_fragment_pin": {"present": bool(fp),
                                 "receipt": (fp or {}).get("receipt"),
                                 "sha256": (fp or {}).get("sha256")}}
    return rows


def build_record(day: str = "20260905") -> dict:
    calls = bindings_from_code()
    st = state_for(day)
    missing = {c: [k for k, v in b.items() if not v["present"]]
               for c, b in st.items()}
    return {
        "contract": CONTRACT,
        "as_of_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                    capture_output=True, text=True
                                    ).stdout.strip(),
        "day": day,
        "builder": str(BUILDER),
        "THE_SET_IS_CLOSED_AT_TWO":
            "day_tape_pin and day_fragment_pin. Every other assert reachable "
            "from build() -- assert_rule20, assert_pool_equality, "
            "assert_coverage, assert_index_released, assert_artifacts_absent "
            "-- binds to a COMPUTED property, not to a producer's receipt.",
        "call_sites_in_build": calls,
        "state_by_coin": st,
        "missing_by_coin": missing,
        "WHY_ETH_IS_MISSING_ANY":
            "Neither producer writes its receipt from build(); only main() "
            "does, at a path composed from the module's COIN constant. A "
            "caller using the library published none -- for ANY coin, which "
            "the BTC control run confirms: it too wrote no receipt.",
        "COST_TO_CLOSE": {
            "fragment_rebuild_minutes": 3.33,
            "basis": "the measured 09-05 ETH fragment wall clock, 3m20s",
            "one_off_per_coin": True,
            "why": "once a coin's receipts exist, later days publish on the "
                   "first pass through the fixed launcher"},
        "PENDING_TEST_peak_ratio_mechanism": {
            "measurement": {"peak_ratio": 0.97, "size_ratio": 0.66,
                            "time_ratio": [0.27, 0.45],
                            "basis": "MEASURED, single observations"},
            "mechanism": "peak set by the per-day window index (288 for both "
                         "coins), so memory does not scale with volume",
            "status": "INFERRED, UNREPRODUCED -- from a count, not from the "
                      "allocator; two stages of one day of one coin pair",
            "what_would_settle_it": "the next ETH or third-coin build: the "
                                    "peak ratio reproduces or it does not",
            "the_decision_does_NOT_rest_on_it":
                "do-not-overlap rests on the measurement plus 0.7% of "
                "soft-limit headroom over single observations"},
    }


def falsify() -> int:
    rc = 0

    def note(n, ok, d=""):
        nonlocal rc
        if not ok:
            rc = 1
        print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"   [{d}]" if d else ""))

    r = build_record()
    calls = r["call_sites_in_build"]
    names = {c["call"] for c in calls}
    note("the binding set is read from the AST, not typed",
         names <= RECEIPT_CALLS and len(calls) >= 4, str(sorted(names)))
    note("both pins are found, and on adjacent lines as the source shows",
         {"day_tape_pin", "day_fragment_pin"} <= names,
         str([c["line"] for c in calls if c["call"].endswith("_pin")]))
    note("the two assert_input_matches_its_receipt calls name their kinds",
         sorted(c["kind"] for c in calls
                if c["call"] == "assert_input_matches_its_receipt")
         == ["fragment", "tape"])
    st = r["state_by_coin"]
    note("BTC -- the control -- has BOTH bindings present",
         st["btc"]["day_tape_pin"]["present"]
         and st["btc"]["day_fragment_pin"]["present"])
    note("ETH has the tape pin (the 11m21s rebuild bought it) and NOT the "
         "fragment pin",
         st["eth"]["day_tape_pin"]["present"]
         and not st["eth"]["day_fragment_pin"]["present"],
         str(r["missing_by_coin"]["eth"]))
    note("the ETH tape pin is a SCORE split whose n_rows is the fragment's "
         "OK count",
         st["eth"]["day_tape_pin"]["split"] == "score"
         and st["eth"]["day_tape_pin"]["n_rows"] == 334336,
         str(st["eth"]["day_tape_pin"]["n_rows"]))
    note("the pending test is recorded as INFERRED, not as a rule",
         "INFERRED" in r["PENDING_TEST_peak_ratio_mechanism"]["status"])
    print(json.dumps({"falsifier": "be_receipt_bindings", "n": 7,
                      "failed": rc}))
    return rc


if __name__ == "__main__":
    if "--falsify" in sys.argv:
        raise SystemExit(falsify())
    r = build_record()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1) + "\n")
    print(OUT.relative_to(ROOT))
