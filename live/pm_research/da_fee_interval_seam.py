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
import ast
import json
from pathlib import Path

PROTOCOL = "P003_DA_FEE_INTERVAL_SEAM_V1"
REPO = Path("/home/yuqing/ctaNew")
ECON = (REPO / "data/pm_5min/derived"
        / "p003_v2_gate1_economics_smoke__20260905T052605Z.json")
MODULE = REPO / "live/pm_research/de_v2_lifecycle_economics.py"
#: The absolute worst-case taker-rate equivalent: 0.07 * p(1-p) maximised at
#: p = 0.5, in cents per share. An UPPER BOUND on cost, never an estimate --
#: 99.05% of observed maker legs are charged exactly zero (da_onchain_fee_audit).
WORST_CASE_CENTS_PER_SHARE = 0.07 * 0.25 * 100.0


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
    per = WORST_CASE_CENTS_PER_SHARE * shares / len(fill_ids)
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
    return {
        "n_fills": len(fill_ids),
        "shares": shares,
        "gross_before_fees_cents": gross_cents,
        "endpoint_zero": {"status": zero["status"],
                          "maker_fee_cents": zero["maker_fee_cents"],
                          "strategy_net_cents": hi},
        "endpoint_worst_case": {
            "status": upper["status"],
            "cents_per_share": WORST_CASE_CENTS_PER_SHARE,
            "maker_fee_cents": upper["maker_fee_cents"],
            "strategy_net_cents": lo},
        "strategy_net_interval_cents": [lo, hi],
        "computed_predicates": {
            "both_endpoints_priced": zero["status"] == "OK"
                                     and upper["status"] == "OK",
            "interval_excludes_zero": (lo < 0 and hi < 0) or (lo > 0 and hi > 0),
            "sign_is_invariant_across_the_interval":
                (lo < 0) == (hi < 0),
            "gross_already_negative_before_any_fee": gross_cents < 0,
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
        "sign_invariant_on_every_arm_probed": all(
            a["computed_predicates"]["sign_is_invariant_across_the_interval"]
            for a in arms.values()),
        "arms_whose_bracket_straddles_zero": sorted(
            n for n, a in arms.items()
            if not a["computed_predicates"][
                "sign_is_invariant_across_the_interval"]),
    }
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
            lo, hi = a["strategy_net_interval_cents"]
            inv = a["computed_predicates"][
                "sign_is_invariant_across_the_interval"]
            # MEASURED, NEVER ASSERTED. The first cut of this check asserted
            # sign-invariance as a pass condition and went RED on
            # baseline_qr_skew_only -- whose interval genuinely straddles
            # zero. That was rule 10 on my own instrument: a conclusion
            # printed beside a table instead of a predicate evaluated from
            # it. The check now requires the interval to be COMPUTED and
            # ORDERED, and REPORTS which way the sign falls.
            ok(lo <= hi and a["computed_predicates"][
                   "both_endpoints_priced"],
               f"REAL/{name}: strategy net interval [{lo:.1f}, {hi:.1f}] "
               f"cents, ordered and priced at both ends -- sign invariant "
               f"across the interval: {inv}"
               + ("" if inv else "  <-- THE BRACKET STRADDLES ZERO: the fee "
                                 "term is DECISION-RELEVANT for this arm at "
                                 "the worst-case endpoint"))
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
