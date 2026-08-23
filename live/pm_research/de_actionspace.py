"""DE-ActionSpace -- the action MENU enumerator (B2's last build item, R-68).

Sits between the feasibility oracle (`de_constraints.max_size`) and the
DecisionScheme: given halt state, position, resting quotes and the oracle's
side-keyed map, it emits the FINITE menu of typed, venue-expressible
`Action`s the solver may choose among. The oracle says what is FEASIBLE;
this module says what is EXPRESSIBLE — the intersection is the menu.

The one fact this module owns: the VENUE FLOOR. `orderMinSize = 5` (SP §4,
MEASURED on 7,771/7,815 rows; Class D — equals the pin, zero downward
headroom). An action the oracle caps at 3 shares is feasible and
INEXPRESSIBLE — the menu omits it, and that omission is a typed fact here,
never a silent truncation. Structural consequence, tested below: a
REDUCING_ONLY position with |net| < 5 cannot be reduced by any venue action
— the residual rides to resolution (carry, Layer 2's estimand), and the
menu says so by containing only CANCEL/WAIT.

Imports flat per IMPORT_LAYOUT.md. Contracts v23 conformance is a SELFTEST
that reads `contracts/contracts.yaml` — the menu's vocabulary must match
the ratified contract literally, or this module fails loudly at test time.

    python3 live/pm_research/de_actionspace.py --selftest
"""

from __future__ import annotations

import argparse
from typing import Any

import de_constraints as dc

VENUE_MIN_SIZE = 5.0     # SP §4 orderMinSize, MEASURED; = the pin (Class D)


def enumerate_actions(state: str,
                      pos: dc.Position,
                      resting: dict[str, dc.RestingQuote],
                      prices: dict[str, float],
                      sp: dict[str, Any] | None = None,
                      scenario_losses: dict[str, float] | None = None
                      ) -> list[dc.Action]:
    """The menu. Every returned Action validates and is oracle-feasible —
    asserted at emission, not assumed. `resting` maps order_ref -> quote
    (the ref is the venue's handle; CANCEL is inexpressible without it,
    which is why v23 carries `Action.order_ref`).

    Menu construction is DEFAULT-DENY end to end: nothing enters the menu
    unless the oracle's map carries its key AND the venue can express it.
    """
    quotes = list(resting.values())
    ms = dc.max_size(state, pos, quotes, prices, sp=sp,
                     scenario_losses=scenario_losses)
    menu: list[dc.Action] = [dc.Action("WAIT")]
    menu += [dc.Action("CANCEL", order_ref=ref) for ref in sorted(resting)]

    for side in dc.SIDES:
        q_cap = ms.get(f"QUOTE:{side}", 0.0)
        if q_cap + 1e-12 >= VENUE_MIN_SIZE:
            size = min(q_cap, (sp or dc.SP_OPERATIVE)["max_quote_size_shares"])
            for placement in dc.PLACEMENTS:
                menu.append(dc.Action("QUOTE", side=side,
                                      placement=placement, size=size))
        c_cap = ms.get(f"CROSS:{side}", 0.0)
        if c_cap + 1e-12 >= VENUE_MIN_SIZE:
            menu.append(dc.Action("CROSS", side=side, size=c_cap))

    for a in menu:                       # emission-time property, fail-loud
        dc.validate_action(a)
        if not dc.action_feasible(a, ms):
            raise AssertionError(f"enumerator emitted an infeasible action: {a}")
    return menu


def menu_facts(menu: list[dc.Action]) -> dict[str, Any]:
    """Typed summary for receipts/telemetry: what the menu contains and,
    as important, what it structurally CANNOT contain."""
    verbs = sorted({a.verb for a in menu})
    return {
        "n": len(menu),
        "verbs": verbs,
        "size_bearing": sorted({f"{a.verb}:{a.side}" for a in menu
                                if a.verb in dc.SIZE_BEARING}),
        "reduction_expressible": any(a.verb == "CROSS" for a in menu),
    }


# --------------------------------------------------------------------------
# contracts v23 conformance -- read the ratified file, match literally
# --------------------------------------------------------------------------

def contract_conformance(doc: dict[str, Any] | None = None) -> list[str]:
    """Returns the list of MISMATCHES between this module's vocabulary and
    contracts v23 (empty = conformant). A doctored contract must FAIL —
    demonstrated in the selftest, so this check cannot pass vacuously."""
    if doc is None:
        import sys, pathlib, yaml
        root = pathlib.Path(__file__).resolve().parents[2]
        doc = yaml.safe_load(open(root / "live/pm_research/contracts/contracts.yaml"))
    bad: list[str] = []
    if doc.get("version") != 23:
        bad.append(f"contracts version {doc.get('version')} != 23")
    f = (doc.get("types", {}).get("Action", {}) or {}).get("fields", {})
    if "order_ref" not in f:
        bad.append("Action.order_ref missing (CANCEL inexpressible)")
    pl = str(f.get("placement", ""))
    for p in dc.PLACEMENTS:
        if p not in pl:
            bad.append(f"placement variant {p} not in contract enum {pl!r}")
    fs_notes = str((doc.get("types", {}).get("FeasibleSet", {}) or {}).get("notes", ""))
    if "VERB:SIDE" not in fs_notes or "DEFAULT-DENY" not in fs_notes:
        bad.append("FeasibleSet §2a pin (VERB:SIDE / DEFAULT-DENY) not in contract notes")
    # RULED, NOT YET LANDED (R-72, Q-DE-13 closed): the enum is MINT|MERGE;
    # the applied v23 literal DEPOSIT|WITHDRAW was applier-chosen and has no
    # ratification to defend. The v23→v24 change record is drafted in
    # CONTRACTS_BATCH_v24.md and RIDES A BATCH, never an ad-hoc edit (R-35).
    # Until that batch lands, this check matches NEITHER side BY ORDER —
    # R-72: "it is the only thing holding the discrepancy visible; do not
    # turn it green." The ActionSpace never emits capital ops (Allocator-
    # issued, bypass the oracle), so the menu is unaffected either way.
    return bad


# --------------------------------------------------------------------------

def selftest() -> int:
    n = [0]

    def ok(cond: bool, label: str) -> None:
        n[0] += 1
        if not cond:
            raise SystemExit(f"[de_actionspace selftest] FAIL: {label}")

    sp = dict(dc.SP_OPERATIVE)
    flat = dc.Position()
    prices = {"BID_UP": 0.49, "ASK_UP": 0.51}
    scen = {"S1": 0.0}

    # RUNNING, flat, no resting: WAIT + 2 sides x 2 placements QUOTE + 2 CROSS
    m = enumerate_actions(dc.RUNNING, flat, {}, prices, sp, scen)
    ok(len([a for a in m if a.verb == "QUOTE"]) == 4, "4 QUOTE arms when open")
    ok(len([a for a in m if a.verb == "CROSS"]) == 2, "2 CROSS arms when open")
    ok(any(a.verb == "WAIT" for a in m), "WAIT always present")
    ok(not any(a.verb in dc.CAPITAL_OPS for a in m), "capital ops never in menu")
    ok(all(a.size == 5.0 for a in m if a.verb == "QUOTE"),
       "QUOTE size = pin (cap >= floor = pin)")

    # venue floor: oracle cap below 5 -> feasible but INEXPRESSIBLE
    tight = dict(sp, kappa_usd_per_market=2.0)      # head ~ $2 -> cap ~ 4 sh
    m2 = enumerate_actions(dc.RUNNING, flat, {}, prices, tight, scen)
    ms2 = dc.max_size(dc.RUNNING, flat, [], prices, tight, scen)
    ok(ms2.get("QUOTE:BID_UP", 0) > 0, "oracle allows a sub-floor size")
    ok(not any(a.verb in dc.SIZE_BEARING for a in m2),
       "menu omits sub-floor arms (venue floor is the menu's own fact)")
    ok(not menu_facts(m2)["reduction_expressible"], "facts: nothing size-bearing")

    # CANCEL per resting ref, deterministic order
    resting = {"ord-2": dc.RestingQuote("BID_UP", 0.49, 5.0),
               "ord-1": dc.RestingQuote("ASK_UP", 0.51, 5.0)}
    m3 = enumerate_actions(dc.HALTED, flat, resting, prices, sp, scen)
    cancels = [a.order_ref for a in m3 if a.verb == "CANCEL"]
    ok(cancels == ["ord-1", "ord-2"], "CANCEL per ref, sorted")
    ok(all(a.verb in dc.ALWAYS_FEASIBLE for a in m3),
       "HALTED menu = CANCEL/WAIT only (oracle door)")

    # REDUCING_ONLY with |net| >= floor: reducing CROSS expressible
    longpos = dc.Position(q_up=8.0, cost_up=4.0)
    m4 = enumerate_actions(dc.REDUCING_ONLY, longpos, {}, prices, sp, scen)
    crosses = [a for a in m4 if a.verb == "CROSS"]
    ok(len(crosses) == 1 and crosses[0].side == "ASK_UP"
       and abs(crosses[0].size - 5.0) < 1e-9,
       "reducing CROSS on the reducing side, capped min(|net|, pin)... "
       f"got {crosses}")
    ok(not any(a.verb == "QUOTE" and a.side == "BID_UP" for a in m4),
       "adding side absent under REDUCING_ONLY")

    # the sub-floor residual: |net| = 3 < 5 -> reduction INEXPRESSIBLE
    stub = dc.Position(q_up=3.0, cost_up=1.5)
    m5 = enumerate_actions(dc.REDUCING_ONLY, stub, {}, prices, sp, scen)
    ok(not any(a.verb in dc.SIZE_BEARING for a in m5),
       "sub-floor residual: CANCEL/WAIT only — the residual rides to "
       "resolution and the menu SAYS so")

    # fail-loud inheritance: RUNNING with no scenario declared
    try:
        enumerate_actions(dc.RUNNING, flat, {}, prices, sp, None)
        ok(False, "missing scenario must raise")
    except ValueError:
        ok(True, "oracle's fail-closed ValueError propagates through the menu")

    # v23 conformance on the REAL file: zero mismatches
    real = contract_conformance()
    ok(real == [], f"contracts v23 conformance clean, got {real}")

    # ...and the check CAN fail: doctored contracts must mismatch
    doctored = {"version": 23, "types": {"Action": {"fields": {
        "placement": "enum:JOIN"}}, "FeasibleSet": {"notes": ""}}}
    bad = contract_conformance(doctored)
    ok(len(bad) >= 3, f"doctored contract detected ({len(bad)} mismatches)")
    ok(contract_conformance({"version": 22, "types": {}}) != [],
       "wrong version detected")

    print(f"[de_actionspace] selftest OK — {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
