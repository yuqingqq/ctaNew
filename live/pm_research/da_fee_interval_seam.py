"""DA: is the fee-INTERVAL re-run actually runnable, or does it need new code?

The reviewer's §4 says a bound-endpoint re-run -- value every maker leg twice,
once at fee = 0 and once at the taker-rate equivalent -- is "cheap,
precedented, and unrun", and the USER's ruling turns on whether that is true.
That is a claim about CODE, so it is settled by reading and driving the code,
not by agreeing with it.

WHAT THIS PROBES, and it modifies nothing:
  * `de_v2_lifecycle_economics.economic_arm` already takes `maker_fees` as a
    keyword; `_fee_ledger` validates it and computes `maker_fee_cents`
  * so both endpoints are reachable through the SHIPPED seam, on the real
    fill identities carried in the Gate-1e receipt
  * and the three PRODUCTION call sites omit the keyword, which is why the
    endpoints have never been run

THE ONE FRICTION, reported because "cheap" should carry its cost: the ledger
is keyed by EXACT fill ids and refuses unknown ones, so a caller must build
one dict PER ARM after that arm's fills are known -- 202 dicts for the Gate-1e
population, not one.

    python3 live/pm_research/da_fee_interval_seam.py --selftest
    python3 live/pm_research/da_fee_interval_seam.py --real --output P
"""
from __future__ import annotations

import argparse
import hashlib
import ast
import json
from pathlib import Path

#: ROUND 54, C-2: DERIVED, NEVER TYPED. v2 shipped still carrying the V1
#: string, so an automated reader resolving by `protocol` saw V1 for BOTH
#: files -- the supersession was legible to a human reading `supersedes` and
#: invisible to a resolver keying on the protocol. The version now comes from
#: `RECEIPT_VERSION` in one place, so bumping the receipt cannot leave the
#: protocol behind.
RECEIPT_VERSION = 4
PROTOCOL = f"P003_DA_FEE_INTERVAL_SEAM_V{RECEIPT_VERSION}"
REPO = Path("/home/yuqing/ctaNew")
ECON = (REPO / "data/pm_5min/derived"
        / "p003_v2_gate1_economics_smoke__20260905T052605Z.json")
MODULE = REPO / "live/pm_research/de_v2_lifecycle_economics.py"
#: WITHDRAWN AT v2, AND THE ROW THAT FORBIDS IT IS NAMED.
#:
#: v1 priced a lower endpoint at 1.75 c/share -- 0.07 * p(1-p) at p = 0.5 --
#: and SUBTRACTED it from a maker net. `FLOW_MODEL_STATE.md:79` is a row of
#: the frozen facts table and it forbids exactly that, in its own words:
#:
#:   "Crossing costs ~2.25 c/share ATM -- TAKER LEG ONLY | 0.50 c half-spread
#:    + 1.75 c fee ~= 225 bps on a $1 binary. BOTH TERMS ARE THE SAME SIDE.
#:    DO NOT SUBTRACT THIS FROM A MAKER NET."
#:
#: The 1.75 c IS that row's fee term. Charging it against a MAKER net prices
#: the counterparty's cost as if it were ours, so the quantity is not merely
#: conservative -- it is NOT AN ECONOMIC QUANTITY AT ALL, and an interval
#: whose lower end is not economic is not an interval. Two seats found this
#: independently and the reviewer confirmed it by execution (9e5d62f 3.2).
#:
#: The bracket therefore COLLAPSES TO ITS UPPER POINT, which is E0, the
#: zero-maker-fee endpoint -- the venue's default, our signed rate, and the
#: estimand V2 declares. `arms_whose_bracket_straddles_zero` is consequently
#: EMPTY: there is no bracket left to straddle anything.
#:
#: Kept as a named constant rather than deleted so the withdrawn number stays
#: legible to a reader of the v1 receipt, which stands as provenance (rule 13).
WITHDRAWN_LOWER_ENDPOINT_CENTS_PER_SHARE = 0.07 * 0.25 * 100.0
WITHDRAWAL_AUTHORITY = "FLOW_MODEL_STATE.md:79"
WITHDRAWAL_ROW = (
    "Crossing costs ~2.25 c/share ATM -- TAKER LEG ONLY | 0.50 c half-spread "
    "+ 1.75 c fee ~= 225 bps on a $1 binary. BOTH TERMS ARE THE SAME SIDE. "
    "DO NOT SUBTRACT THIS FROM A MAKER NET.")
#: v3 supersedes v2, which superseded v1. The chain is kept whole rather than
#: collapsed: v2 is what withdrew the endpoint, and v3 changes only the
#: protocol string, so a reader must be able to see both steps.
SUPERSEDES = {
    "path": "data/pm_5min/derived/"
            "p003_da_fee_interval_seam_v3__20260906T024354Z.json",
    "sha256": "PINNED_AT_EMIT",
    "which_superseded": {
        "path": "data/pm_5min/derived/"
                "p003_da_fee_interval_seam_v2__20260906T021955Z.json",
        "which_superseded": "p003_da_fee_interval_seam__20260905T155346Z.json",
        "sha256": "a7b562f0ab4673160aa8757083a721c9c90d8b317"
                  "a36beb538674cf22db624f8",
    },
    "v4_changes_only": (
        "the `correction_is_in_band` sentence, which v3 shipped still reading "
        "\"this is v2\". Derived from RECEIPT_VERSION now. No number, no "
        "predicate and no field other than the version strings differs from "
        "v3, and v3 is not edited."),
    "v3_changed_only": (
        "the `protocol` string, which v2 left reading "
        "P003_DA_FEE_INTERVAL_SEAM_V1 so an automated reader resolving by "
        "protocol saw V1 for both files (reviewer C-2). Every number, every "
        "predicate and the withdrawal itself are UNCHANGED from v2, and v2 "
        "is not edited"),
    "what_changed": (
        "the lower endpoint is WITHDRAWN as a non-economic quantity and "
        "`arms_whose_bracket_straddles_zero` is now EMPTY. The E0 endpoint, "
        "the call-site AST counts, the guard behaviour and every other field "
        "are unchanged -- v1's arithmetic was right and its ESTIMAND was "
        "wrong"),
    # ROUND 55: DERIVED, like the protocol. This sentence still read "this is
    # v2" inside the v3 artifact -- the C-2 defect one field over, in prose
    # instead of in `protocol`. It is NOT resolution-bearing (a reader follows
    # `path` and `sha256`), so it is a nit; but a shipped artifact asserting a
    # false statement about its own version is exactly what the last two
    # rounds were spent removing, and leaving one in to save a file is the
    # wrong trade here. Both the number and the predecessor's name come from
    # RECEIPT_VERSION now, so neither can be left behind again.
    "correction_is_in_band": (
        f"rule 13: this is v{RECEIPT_VERSION}, a superseding receipt. The "
        f"v{RECEIPT_VERSION - 1} artifact is not edited and stands as "
        f"provenance, as does every earlier link in the chain"),
}


class SeamRefused(RuntimeError):
    """The seam cannot be probed as claimed."""


def call_sites_passing_maker_fees(path: Path | None = None) -> dict:
    """Which `economic_arm` calls pass `maker_fees`, by AST, not by grep.

    A grep counts the word; this counts CALLS and reads their keywords, which
    is the difference between 'the token appears' and 'the argument is passed'
    -- the ownership distinction this programme keeps relearning."""
    p = Path(path) if path is not None else MODULE
    if not p.is_file():
        raise SeamRefused(f"REFUSED: no module at {p}")
    tree = ast.parse(p.read_text(encoding="utf-8"))
    # The selftest lives inside a function named `selftest`; production call
    # sites do not. Separating them is the whole point of the count.
    selftest_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "selftest":
            selftest_lines = set(range(node.lineno, (node.end_lineno or 0) + 1))
    prod, test = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "economic_arm":
            continue
        kw = {k.arg for k in node.keywords if k.arg}
        rec = {"line": node.lineno, "passes_maker_fees": "maker_fees" in kw}
        (test if node.lineno in selftest_lines else prod).append(rec)
    return {
        "production_call_sites": sorted(prod, key=lambda r: r["line"]),
        "selftest_call_sites": sorted(test, key=lambda r: r["line"]),
        "n_production": len(prod),
        "n_production_passing_maker_fees":
            sum(1 for r in prod if r["passes_maker_fees"]),
        "n_selftest_passing_maker_fees":
            sum(1 for r in test if r["passes_maker_fees"]),
    }


def drive_endpoints(fill_ids: list, gross_cents: float,
                    shares: float) -> dict:
    """Drive the SHIPPED ledger at both endpoints. Imports, never edits."""
    import sys
    sys.path.insert(0, str(REPO / "live/pm_research"))
    import de_v2_lifecycle_economics as L
    if not fill_ids:
        raise SeamRefused("REFUSED: no fill identities to price")
    zero = L._fee_ledger(fill_ids, {f: 0.0 for f in fill_ids})
    # v2: the lower endpoint is priced ONLY to show it still prices -- the
    # seam is the finding -- and its VALUE is reported as withdrawn, never as
    # an endpoint of a decision interval.
    per = WITHDRAWN_LOWER_ENDPOINT_CENTS_PER_SHARE * shares / len(fill_ids)
    upper = L._fee_ledger(fill_ids, {f: per for f in fill_ids})
    guards = {}
    inc = L._fee_ledger(fill_ids, {f: 0.0 for f in fill_ids[:-1]})
    guards["incomplete_ledger_status"] = inc["status"]
    guards["incomplete_ledger_fee_is_none"] = inc["maker_fee_cents"] is None
    for label, bad in (("unknown_fill_id", {"not-a-fill": 1.0}),
                       ("non_finite_fee",
                        {**{f: 0.0 for f in fill_ids},
                         fill_ids[0]: float("nan")})):
        try:
            L._fee_ledger(fill_ids, bad)
            guards[label] = "ACCEPTED"
        except L.LifecycleEconomicsRefused:
            guards[label] = "REFUSED"
    lo = gross_cents - upper["maker_fee_cents"]
    hi = gross_cents - zero["maker_fee_cents"]
    # THE ADMISSIBLE READING (v2): a point at E0, not an interval.
    return {
        "n_fills": len(fill_ids),
        "shares": shares,
        "gross_before_fees_cents": gross_cents,
        "endpoint_zero": {"status": zero["status"],
                          "maker_fee_cents": zero["maker_fee_cents"],
                          "strategy_net_cents": hi},
        "endpoint_WITHDRAWN_lower": {
            "status": upper["status"],
            "cents_per_share": WITHDRAWN_LOWER_ENDPOINT_CENTS_PER_SHARE,
            "maker_fee_cents": upper["maker_fee_cents"],
            "would_have_given_strategy_net_cents": lo,
            "WITHDRAWN": True,
            "authority": WITHDRAWAL_AUTHORITY,
            "row": WITHDRAWAL_ROW,
            "why": ("this charges the TAKER's fee against a MAKER net, which "
                    "the named row forbids in terms. It is not a conservative "
                    "endpoint; it is not an economic quantity"),
        },
        "admissible_reading_cents": hi,
        "admissible_reading_is": "A POINT AT E0, NOT AN INTERVAL",
        "computed_predicates": {
            "both_endpoints_priced": zero["status"] == "OK"
                                     and upper["status"] == "OK",
            "lower_endpoint_withdrawn": True,
            "bracket_straddles_zero": False,
            "why_not": ("there is no bracket: the lower endpoint is "
                        "withdrawn, so nothing straddles anything"),
            "E0_sign": (0 if hi == 0 else (1 if hi > 0 else -1)),
            "gross_equals_E0_because_the_maker_fee_is_zero":
                abs(hi - gross_cents) < 1e-12,
        },
        "guards_still_refuse": guards,
    }


def probe() -> dict:
    if not ECON.is_file():
        raise SeamRefused(f"REFUSED: no Gate-1e receipt at {ECON}")
    o = json.loads(ECON.read_text())
    la = o["lifecycle_economic_audit"]
    out = {"protocol": PROTOCOL, "source_receipt": str(ECON),
           "call_sites": call_sites_passing_maker_fees()}
    arms = {}
    for name in ("treatment", "baseline_qr_skew_only"):
        arm = la.get(name)
        if not arm:
            continue
        arms[name] = drive_endpoints(
            arm["received_fill_ids"],
            arm["gross_after_queue_reset_before_fees_cents"],
            arm["received_shares"])
        arms[name]["shipped_fee_status"] = arm["maker_fee_ledger"]["status"]
        arms[name]["shipped_strategy_net"] = arm[
            "fee_adjusted_strategy_net_cents"]
    out["arms"] = arms
    cs = out["call_sites"]
    out["computed_predicates"] = {
        "seam_exists_in_shipped_signature": True,
        "endpoints_run_without_modifying_any_module": all(
            a["computed_predicates"]["both_endpoints_priced"]
            for a in arms.values()),
        "precedented_selftest_already_drives_the_OK_path":
            cs["n_selftest_passing_maker_fees"] > 0,
        "unrun_no_production_call_site_passes_it":
            cs["n_production_passing_maker_fees"] == 0,
        # v2: EMPTY BY CONSTRUCTION, and that is the correction. v1 reported
        # baseline_qr_skew_only here on a bracket whose lower end charged a
        # taker fee to a maker net. With that endpoint withdrawn there is no
        # bracket, so nothing can straddle: the list is empty because the
        # QUANTITY is gone, not because a number moved.
        "arms_whose_bracket_straddles_zero": [],
        "every_arm_reads_as_a_POINT_at_E0": all(
            not a["computed_predicates"]["bracket_straddles_zero"]
            for a in arms.values()),
    }
    sup = dict(SUPERSEDES)
    v2 = REPO / sup["path"]
    sup["sha256"] = (hashlib.sha256(v2.read_bytes()).hexdigest()
                     if v2.is_file() else "V2_ARTIFACT_ABSENT_AT_EMIT")
    out["supersedes"] = sup
    out["role"] = ("REPORTED, NOT ENFORCED (rule 14). This says the "
                   "bound-endpoint re-run is runnable and what it yields on "
                   "the two arms the receipt carries. It clears no gate and "
                   "promotes nothing.")
    out["limits"] = [
        "the upper endpoint is the ABSOLUTE worst case (0.07*p(1-p) at "
        "p=0.5, on EVERY maker leg); measured incidence is 10 charged legs "
        "in 1,056, so it is a bound and never an estimate",
        "maker rebates and liquidity rewards are outside the ledger's own "
        "sign convention and are not netted here",
        "two arms, one window: the full 202-arm re-run needs the replay and "
        "one fee dict per arm, which is caller work this probe does not do",
        "a bracket that straddles zero does so AT THE UPPER BOUND, which "
        "charges 1.75 c/share on EVERY maker leg; measured incidence is 10 "
        "charged legs in 1,056, so the straddle says the interval is wide, "
        "NOT that the arm is likely negative",
    ]
    return out


def selftest() -> int:
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    cs = call_sites_passing_maker_fees()
    ok(cs["n_production"] >= 3 and cs["n_production_passing_maker_fees"] == 0,
       f"CALL SITES: {cs['n_production']} production calls to economic_arm, "
       f"{cs['n_production_passing_maker_fees']} pass maker_fees -- the "
       f"endpoints are UNRUN because the keyword is omitted, not absent")
    ok(cs["n_selftest_passing_maker_fees"] > 0,
       f"PRECEDENT: the module's OWN selftest drives the priced path at "
       f"{cs['n_selftest_passing_maker_fees']} call site(s), so the OK branch "
       f"is exercised code and not a dead limb")

    # POSITIVE CONTROL: the AST reader must SEE a passed keyword.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.py"
        p.write_text("def selftest():\n    economic_arm(1, maker_fees={})\n"
                     "economic_arm(2)\neconomic_arm(3, maker_fees={'a':1})\n")
        g = call_sites_passing_maker_fees(p)
    ok(g["n_production"] == 2 and g["n_production_passing_maker_fees"] == 1
       and g["n_selftest_passing_maker_fees"] == 1,
       f"AST POSITIVE CONTROL: on a planted file it separates production "
       f"from selftest and counts the passed keyword ({g['n_production']} "
       f"prod, 1 passing, 1 in selftest) -- the reader can fire")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.py"
        p.write_text("x = 'economic_arm(maker_fees=1)'  # a STRING\n"
                     "other_fn(maker_fees={})\n")
        g2 = call_sites_passing_maker_fees(p)
    ok(g2["n_production"] == 0,
       "AST KNOWN-BAD: the word inside a string literal, and the keyword on "
       "a DIFFERENT function, are not counted -- a grep would have said 2")
    try:
        call_sites_passing_maker_fees(Path("/nonexistent.py"))
        ok(False, "KNOWN-BAD: reported on an absent module -- must refuse")
    except SeamRefused:
        ok(True, "KNOWN-BAD: an absent module REFUSES")
    try:
        drive_endpoints([], 0.0, 0.0)
        ok(False, "KNOWN-BAD: priced an empty fill set -- must refuse")
    except SeamRefused:
        ok(True, "KNOWN-BAD: an empty fill set refuses rather than reporting "
                 "a zero fee")

    if ECON.is_file():
        r = probe()
        cp = r["computed_predicates"]
        ok(cp["endpoints_run_without_modifying_any_module"],
           "REAL: both endpoints price on the real Gate-1e fill identities "
           "through the SHIPPED seam, with no edit to any module")
        for name, a in r["arms"].items():
            g = a["guards_still_refuse"]
            ok(g["unknown_fill_id"] == "REFUSED"
               and g["non_finite_fee"] == "REFUSED"
               and a["guards_still_refuse"]["incomplete_ledger_fee_is_none"],
               f"REAL/{name}: the ledger's guards are UNWEAKENED by pricing "
               f"the endpoints -- unknown id and NaN still refuse, an "
               f"incomplete ledger still yields no fee")
            hi = a["admissible_reading_cents"]
            lo = a["endpoint_WITHDRAWN_lower"][
                "would_have_given_strategy_net_cents"]
            # v2, AND THIS IS A SECOND CORRECTION ON THE SAME CHECK. v1 had
            # already been fixed once here -- I had ASSERTED sign-invariance
            # and the falsifier caught it, so the check began MEASURING the
            # straddle instead. That fix was right about the arithmetic and
            # wrong about the estimand: there was never an admissible
            # interval to measure, because its lower end charged a TAKER fee
            # to a MAKER net. FLOW_MODEL_STATE.md:79 forbids that in terms.
            # So the straddle is not re-measured, it is GONE.
            ok(a["computed_predicates"]["both_endpoints_priced"]
               and a["computed_predicates"]["lower_endpoint_withdrawn"]
               and not a["computed_predicates"]["bracket_straddles_zero"],
               f"REAL/{name}: v2 reads as a POINT at E0 = {hi:.4f} cents. "
               f"The withdrawn lower endpoint would have given {lo:.1f} -- "
               f"a taker fee charged to a maker net, forbidden by "
               f"{WITHDRAWAL_AUTHORITY}. Nothing straddles zero because "
               f"there is no bracket left to straddle it.")
    else:
        ok(False, f"REAL: no Gate-1e receipt at {ECON}")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = probe()
        txt = json.dumps(out, indent=2, sort_keys=True)
        if a.output:
            a.output.write_text(txt)
        print(txt)
        return 0
    ap.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
