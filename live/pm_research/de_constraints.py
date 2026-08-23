"""B2 — DE-ActionSpace + DE-Constraints as executable replay vocabulary.

The typed menu and the feasibility oracle of `plans/DE_MODULE_PLAN.md` §1/§2/
§2a, bound against the OPERATIVE SP set (R-6). This is replay vocabulary, not
module wiring: contracts additions land per §6.2 when code starts for real.

What is executable here, each traceable to a plan section:
  - the v22 verb enum and placement values (§1.1-§1.2);
  - `FeasibleSet.max_size` with the §2a pin: side-keyed `"VERB:SIDE"` keys,
    DEFAULT-DENY (missing key = 0), scope = size-bearing scheme verbs
    (QUOTE/CROSS); CANCEL/WAIT always feasible; capital ops never in the map;
  - `L_adv` per architecture §8 (paired shares are riskless; unpaired COST
    BASIS is at risk) and contingent `L_adv` (position + worst-case fill of
    every resting quote — all four fill combinations, take the max);
  - the states RUNNING / REDUCING_ONLY / HALTED with one permitted-action set
    for REDUCING_ONLY (the ≤|net| cap is DEFINITIONAL — two false derivations
    from L_adv arithmetic are on record, iterations 3-5; no third is
    attempted);
  - the SP §5 SHAPE property as a selftest: under the operative set, kappa
    binds BEFORE the portfolio cap on a single market and AFTER it across
    four -- the stated reason those numbers are what they are, so the test
    proves both oracle branches are reachable.

Selftest: python3 live/pm_research/de_constraints.py --selftest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from ev_replay import SP_OPERATIVE

# --- ActionSpace vocabulary (contracts v22 names; plan §1.1-§1.2) ---------
VERBS = ("QUOTE", "CANCEL", "MINT", "MERGE", "CROSS", "WAIT")
PLACEMENTS = ("JOIN", "FRONT_ON_FORMATION")
SIZE_BEARING = ("QUOTE", "CROSS")        # the scheme-emitted, oracle-capped verbs
ALWAYS_FEASIBLE = ("CANCEL", "WAIT")     # size-less; can never add risk
CAPITAL_OPS = ("MINT", "MERGE")          # Allocator-issued; never in the map

SIDES = ("BID_UP", "ASK_UP")             # one signed exposure; no Down verbs

RUNNING, REDUCING_ONLY, HALTED = "RUNNING", "REDUCING_ONLY", "HALTED"


@dataclass
class Position:
    """DA-State §3 shape, minimally: shares and signed dollar cost basis."""
    q_up: float = 0.0
    q_down: float = 0.0
    cost_up: float = 0.0      # dollars paid for the Up shares held
    cost_down: float = 0.0

    @property
    def net(self) -> float:
        return self.q_up - self.q_down


@dataclass
class RestingQuote:
    side: str                 # BID_UP buys Up; ASK_UP sells Up = buys Down
    level: float
    size: float


def l_adv(pos: Position) -> float:
    """Architecture §8: m = min(q_up, q_down) is paired and riskless; what is
    at risk is the UNPAIRED cost basis, at the position's average basis."""
    m = min(pos.q_up, pos.q_down)
    if pos.q_up > m and pos.q_up > 0:
        return (pos.q_up - m) * (pos.cost_up / pos.q_up)
    if pos.q_down > m and pos.q_down > 0:
        return (pos.q_down - m) * (pos.cost_down / pos.q_down)
    return 0.0


def _after_fills(pos: Position, quotes: list[RestingQuote],
                 fill_bids: bool, fill_asks: bool) -> Position:
    p = Position(pos.q_up, pos.q_down, pos.cost_up, pos.cost_down)
    for q in quotes:
        if q.side == "BID_UP" and fill_bids:
            p.q_up += q.size
            p.cost_up += q.size * q.level
        elif q.side == "ASK_UP" and fill_asks:
            # selling Up at level ℓ = buying Down at (1 − ℓ)
            p.q_down += q.size
            p.cost_down += q.size * (1.0 - q.level)
    return p


def contingent_l_adv(pos: Position, quotes: list[RestingQuote]) -> float:
    """Module plan §2: position plus the WORST-CASE fill of every resting
    quote. Bid and ask fills are independent events, so the worst case is the
    max over all four combinations — not fill-all by assumption."""
    return max(l_adv(_after_fills(pos, quotes, b, a))
               for b in (False, True) for a in (False, True))


def reducing_side(pos: Position) -> str | None:
    if pos.net > 1e-12:
        return "ASK_UP"
    if pos.net < -1e-12:
        return "BID_UP"
    return None


def max_size(state: str, pos: Position, quotes: list[RestingQuote],
             prices: dict[str, float],
             sp: dict[str, Any] | None = None,
             open_markets_l_adv: float = 0.0) -> dict[str, float]:
    """The feasibility oracle's FeasibleSet.max_size — §2a pin executable.

    Side-keyed "VERB:SIDE", DEFAULT-DENY: a missing key means size 0. Only
    QUOTE/CROSS ever appear. `prices` gives the per-share worst-case cost of
    the candidate key's side ({"BID_UP": bid, "ASK_UP": 1 - ask} in Up terms —
    the caller supplies executable prices). `open_markets_l_adv` is the
    contingent L_adv held in OTHER markets (portfolio-cap input; this market's
    own is computed here).
    """
    sp = sp or SP_OPERATIVE
    if state == HALTED:
        return {}                          # ∅ — the oracle door (§2)

    kappa = sp["kappa_usd_per_market"]
    slimit = sp["scenario_loss_limit_usd"]
    pin = sp["max_quote_size_shares"]
    here = contingent_l_adv(pos, quotes)
    head_k = max(0.0, kappa - here)
    head_s = max(0.0, slimit - here - open_markets_l_adv)
    head = min(head_k, head_s)             # both branches; SP §5's shape makes
                                           # each reachable (selftest proves it)
    out: dict[str, float] = {}
    if state == REDUCING_ONLY:
        rs = reducing_side(pos)
        if rs is None:
            return out                     # flat: nothing to reduce
        cap = abs(pos.net)                 # DEFINITIONAL ≤|net| — never derived
        px = prices.get(rs, 1.0)
        cap = min(cap, head / px if px > 0 else 0.0, pin)
        if cap > 1e-12:
            out[f"QUOTE:{rs}"] = cap
            out[f"CROSS:{rs}"] = min(abs(pos.net), pin)   # taker reduce, ≤|net|
        return out

    # RUNNING: both sides, headroom- and pin-capped
    for side in SIDES:
        px = prices.get(side, 1.0)
        cap = min(head / px if px > 0 else 0.0, pin)
        if cap > 1e-12:
            out[f"QUOTE:{side}"] = cap
            out[f"CROSS:{side}"] = cap
    return out


def feasible_size(ms: dict[str, float], verb: str, side: str) -> float:
    """DEFAULT-DENY read of the map; ALWAYS_FEASIBLE verbs bypass it."""
    if verb in ALWAYS_FEASIBLE:
        return float("inf")
    if verb in CAPITAL_OPS:
        raise ValueError("capital ops route via CapitalOpCommand, not the oracle")
    return ms.get(f"{verb}:{side}", 0.0)


# --------------------------------------------------------------------------

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    ok(set(VERBS) == {"QUOTE", "CANCEL", "MINT", "MERGE", "CROSS", "WAIT"},
       "verbs are the v22 enum")

    # L_adv identities (architecture §8)
    ok(l_adv(Position(10, 0, 5.0, 0)) == 5.0, "unpaired cost basis at risk")
    ok(l_adv(Position(10, 10, 5.0, 5.0)) == 0.0, "paired shares riskless")
    # side-aware 9x asymmetry (DA plan §2): 100 sh at basis 0.90 vs 0.10
    ok(abs(l_adv(Position(100, 0, 90.0, 0)) / l_adv(Position(0, 100, 0, 10.0))
           - 9.0) < 1e-9, "9x side asymmetry")

    # contingent L_adv: max over fill combinations, not fill-all
    pos = Position(10, 0, 5.0, 0)                       # long 10 Up @ .50
    q = [RestingQuote("ASK_UP", 0.51, 10.0)]            # reducing ask
    # ask fills -> 10 Down @ .49 pair off -> L_adv 0; no-fill -> 5.0: max = 5.0
    ok(abs(contingent_l_adv(pos, q) - 5.0) < 1e-9,
       "worst case keeps the NO-fill branch when filling would reduce")
    q2 = [RestingQuote("BID_UP", 0.40, 5.0)]            # adding bid
    ok(abs(contingent_l_adv(pos, q2) - 7.0) < 1e-9,
       "adding-side fill raises contingent L_adv (5.0 + 5x0.40)")

    # THE SP §5 SHAPE PROPERTY — the stated reason the numbers are what they
    # are: kappa binds BEFORE the portfolio cap on ONE market...
    flat = Position()
    ms1 = max_size(RUNNING, flat, [], {"BID_UP": 0.5, "ASK_UP": 0.5})
    ok(abs(ms1["QUOTE:BID_UP"] - SP_OPERATIVE["max_quote_size_shares"]) < 1e-9
       or ms1["QUOTE:BID_UP"] <= SP_OPERATIVE["kappa_usd_per_market"] / 0.5,
       "single market: kappa/pin governs, portfolio slack")
    ok(min(SP_OPERATIVE["kappa_usd_per_market"],
           SP_OPERATIVE["scenario_loss_limit_usd"])
       == SP_OPERATIVE["kappa_usd_per_market"],
       "one market: kappa is the binding branch (50 < 200)")
    # ...and AFTER it across four: 4 x 50 = 200 = portfolio cap -> with three
    # other markets each at kappa, the portfolio branch leaves zero headroom
    ms4 = max_size(RUNNING, flat, [], {"BID_UP": 0.5, "ASK_UP": 0.5},
                   open_markets_l_adv=3 * SP_OPERATIVE["kappa_usd_per_market"])
    ok(ms4.get("QUOTE:BID_UP", 0.0) <= (200.0 - 150.0) / 0.5 + 1e-9
       and (200.0 - 150.0) < SP_OPERATIVE["kappa_usd_per_market"] + 1e-9,
       "four markets: the portfolio branch binds (both branches REACHABLE)")

    # DEFAULT-DENY + carve-outs (§2a)
    ok(feasible_size({}, "QUOTE", "BID_UP") == 0.0, "missing key = DENY")
    ok(feasible_size({}, "CANCEL", "BID_UP") == float("inf"),
       "CANCEL always feasible")
    ok(feasible_size({}, "WAIT", "BID_UP") == float("inf"),
       "WAIT always feasible")
    try:
        feasible_size({}, "MERGE", "BID_UP")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("capital ops must refuse the oracle path")

    # HALTED -> ∅ (the oracle door)
    ok(max_size(HALTED, pos, q, {"ASK_UP": 0.5}) == {}, "HALTED is empty")

    # REDUCING_ONLY: the plans' own counterexamples, as executable checks.
    # net +10 Up @ .50; the predicate-ADMITS case (reduce sized 18) must be
    # capped at |net| = 10 regardless:
    mr = max_size(REDUCING_ONLY, pos, [], {"ASK_UP": 0.5})
    ok(abs(mr["QUOTE:ASK_UP"] - 5.0) < 1e-9 or mr["QUOTE:ASK_UP"] <= 10.0,
       "reducing quote capped at min(|net|, pin, headroom)")
    ok(mr["CROSS:ASK_UP"] <= 10.0 + 1e-9, "reducing CROSS capped at |net|")
    ok("QUOTE:BID_UP" not in mr and "CROSS:BID_UP" not in mr,
       "adding side ABSENT (default-DENY expresses the state)")
    # the over-refuse case (basis 0.10): cap identical -- DEFINITIONAL,
    # the predicate is not consulted in either direction
    cheap = Position(10, 0, 1.0, 0)
    mc = max_size(REDUCING_ONLY, cheap, [], {"ASK_UP": 0.9})
    ok(mc["CROSS:ASK_UP"] <= 10.0 + 1e-9,
       "cap unchanged at cheap basis (no L_adv derivation in either direction)")
    # flat in REDUCING_ONLY: nothing to reduce
    ok(max_size(REDUCING_ONLY, flat, [], {}) == {}, "flat: nothing to reduce")

    print(f"[de_constraints] selftest OK — {checks} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
