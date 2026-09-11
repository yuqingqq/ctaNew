"""STEP 5: the policy seam -- and the decorative-seam falsifier.

`fair_value_plan.md` §5 gate 5, plus §5's required falsifier: *changing
only `fairprice_estimator` metadata while leaving the consumed value
unchanged is detected as a decorative seam.*

THE SEAM IS A VALUE, NOT A LABEL. The plan's §3 names the failure it is
built against: *"a policy seam that consumes the value rather than merely
writing `fairprice_estimator` metadata"* and *"an unused schema field is
not a partial implementation"*. So this file's whole job is to make the
difference between those two MECHANICAL:

  * substituting Identity yields BIT-IDENTICAL quotes and trajectories;
  * a non-Identity value MOVES the quote anchor (positive control);
  * an absent challenger falls back to Identity and the fallback is
    COUNTED;
  * a run that changes only the ESTIMATOR NAME while the consumed value
    is unchanged is DETECTED as decorative.

`quote_anchor` here is the minimal consuming function -- the anchor a
quoter would post around. It is deliberately simple: the point is not the
quoting model, it is that the seam READS the number, so a change in the
number must appear downstream and a change in the label must not.

Usage:  de_fair_value_policy_seam.py --falsify
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_fair_price_identity as FP              # noqa: E402
import de_fair_price_wrapper as W                # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_POLICY_SEAM_V1"
#: §7's quote mapping. NOTHING NUMERIC IS INVENTED HERE: the tick and the
#: placement latency are READ from DA's declarations and REFUSED when
#: absent, because a quoter that supplies its own tick is quoting a market
#: it made up.
LEGAL_TICK_NOT_DECLARED = "LEGAL_TICK_IS_NOT_DECLARED"
PLACEMENT_LATENCY_NOT_DECLARED = "PLACEMENT_LATENCY_IS_NOT_DECLARED"
PLACE_WITHHELD = "PLACE_WITHHELD"
MARKETABLE_CROSS = "MARKETABLE_CROSS"
PROB_LO, PROB_HI = 0.0, 1.0
SIDES = ("UP", "DOWN")
DECORATIVE = "FAIRPRICE_SEAM_IS_DECORATIVE_VALUE_UNCHANGED"
NO_VALUE = "POLICY_SEAM_CONSUMED_NO_VALUE"


class SeamRefused(ValueError):
    """The seam cannot be exercised as declared."""


def _walk_for(doc, names):
    """The first non-container value under any of `names`, wherever nested."""
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k.lower() in names and not isinstance(v, (dict, list)):
                return v
        for v in doc.values():
            got = _walk_for(v, names)
            if got is not None:
                return got
    elif isinstance(doc, list):
        for v in doc:
            got = _walk_for(v, names)
            if got is not None:
                return got
    return None


def _declared(names, refusal: str, decl_dir=None, what: str = ""):
    """A NUMBER FROM A DECLARATION, or a refusal naming what is missing."""
    d = Path(decl_dir) if decl_dir else HERE / "declarations"
    hits = []
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            continue
        got = _walk_for(doc, names)
        if isinstance(got, (int, float)) and not isinstance(got, bool):
            hits.append((f.name, float(got)))
    if not hits:
        raise SeamRefused(
            f"REFUSED {refusal}: no declaration under {d} carries "
            f"{sorted(names)} as a number{(' -- ' + what) if what else ''}. "
            f"This file will not supply one: a quoter that invents its own "
            f"tick or latency is quoting a market it made up.")
    values = {v for _, v in hits}
    if len(values) > 1:
        raise SeamRefused(
            f"REFUSED {refusal}: {sorted(names)} is declared more than "
            f"once with different values {sorted(values)} in "
            f"{[n for n, _ in hits]}. Two declarations of one number is "
            f"not a number.")
    return {"value": hits[0][1], "declared_by": hits[0][0]}


def legal_tick(decl_dir=None) -> dict:
    """§7's LEGAL TICK, from DA's declaration. Refuses when absent."""
    return _declared({"tick_size", "legal_tick", "min_tick",
                      "tick"}, LEGAL_TICK_NOT_DECLARED, decl_dir,
                     "DA is establishing it from the market artifacts")


def placement_latency_ms(decl_dir=None) -> dict:
    """§7's placement latency, BOUND HERE by reading the declaration.

    Clause (g) asks for it to be bound in a file the freeze covers. It is
    bound by being READ here, not by being retyped here: one source, and
    the record names which declaration supplied it.
    """
    return _declared({"placement_latency_ms"},
                     PLACEMENT_LATENCY_NOT_DECLARED, decl_dir,
                     "§7 fixes it at 250 ms")


def _floor_tick(x: float, tick: float) -> float:
    return math.floor(round(x / tick, 9)) * tick


def _ceil_tick(x: float, tick: float) -> float:
    return math.ceil(round(x / tick, 9)) * tick


@dataclass(frozen=True)
class Quote:
    """One posted pair, derived FROM THE CONSUMED PROBABILITY."""
    slug: str
    generation_id: str
    anchor: float
    bid: float
    ask: float
    priced_by: str            # the estimator whose VALUE was consumed
    fell_back: bool
    side: str = "UP"          # (a) UP uses p; DOWN uses 1 - p
    p_side: float = None      # the probability THIS side quotes around
    tick: float = None
    bounded: tuple = ()       # (c)/(e) every bound applied, never silent
    withheld: bool = False    # (d) PLACE_WITHHELD
    withheld_reason: str = ""
    decision_ms: float = None
    effective_ms: float = None  # (f) decision + placement latency
    latency_ms: float = None

    def as_dict(self) -> dict:
        return asdict(self)


def quote_from(value: float, *, slug: str, generation_id: str,
               half_spread: float, priced_by: str,
               fell_back: bool = False, side: str = "UP",
               tick: float = None, best_bid: float = None,
               best_ask: float = None, decision_ms: float = None,
               latency_ms: float = None, decl_dir=None) -> Quote:
    """THE CONSUMING FUNCTION, and §7's mapping in seven named steps.

    (a) DOWN quotes `1 - p`; there is one probability and one complement.
    (b) the bid rounds DOWN and the ask rounds UP, to the LEGAL TICK --
        directional, because symmetric rounding can post a bid ABOVE the
        price it was derived from.
    (c) both sides are bounded to the legal binary range [0, 1].
    (d) a quote that would cross the book emits PLACE_WITHHELD with reason
        MARKETABLE_CROSS.
    (e) and it is NEVER silently clamped into an apparently passive order:
        every bound applied is recorded in `bounded`, and a cross is
        withheld rather than pulled back to the touch.
    (f)/(g) a candidate-induced change gets NO zero-latency privilege: the
        quote is effective at `decision + placement latency`, and the
        latency comes from the declaration.
    """
    if not FP._finite_real(value):
        raise SeamRefused(
            f"REFUSED {NO_VALUE}: the seam was handed {value!r}. A quote "
            f"built from no value is the decorative case with extra steps.")
    if side not in SIDES:
        raise SeamRefused(
            f"REFUSED: side {side!r} is not one of {SIDES}; an unnamed "
            f"side is how a complement gets quoted as an outright.")
    if tick is None:
        tick = legal_tick(decl_dir)["value"]          # refuses when absent
    if latency_ms is None:
        latency_ms = placement_latency_ms(decl_dir)["value"]
    p_side = float(value) if side == "UP" else 1.0 - float(value)   # (a)
    raw_bid, raw_ask = p_side - half_spread, p_side + half_spread
    bid = _floor_tick(raw_bid, tick)                                # (b)
    ask = _ceil_tick(raw_ask, tick)
    bounded = []
    if bid < PROB_LO:                                               # (c)
        bounded.append({"field": "bid", "from": bid, "to": PROB_LO,
                        "why": "below the legal binary range"})
        bid = PROB_LO
    if ask > PROB_HI:
        bounded.append({"field": "ask", "from": ask, "to": PROB_HI,
                        "why": "above the legal binary range"})
        ask = PROB_HI
    withheld, reason = False, ""
    crosses = ((best_ask is not None and bid >= best_ask)            # (d)
               or (best_bid is not None and ask <= best_bid))
    if crosses:
        withheld, reason = True, MARKETABLE_CROSS
    return Quote(slug=slug, generation_id=generation_id,
                 anchor=float(value), bid=round(bid, 12),
                 ask=round(ask, 12), priced_by=priced_by,
                 fell_back=fell_back, side=side, p_side=p_side,
                 tick=tick, bounded=tuple(
                     json.dumps(b, sort_keys=True) for b in bounded),
                 withheld=withheld, withheld_reason=reason,
                 decision_ms=decision_ms, latency_ms=latency_ms,
                 effective_ms=(None if decision_ms is None
                               else decision_ms + latency_ms))


def trajectory(quotes) -> str:
    """A digest of the posted path -- the thing 'bit-identical' is about."""
    return hashlib.sha256(json.dumps(
        [q.as_dict() for q in quotes], sort_keys=True).encode()).hexdigest()


def run_seam(actions, value_of, *, half_spread=0.01,
             counter: W.FallbackCounter = None, tick: float = None,
             decl_dir=None, side: str = "UP", decision_ms=None) -> dict:
    """Drive the seam over actions, returning quotes AND the trajectory.

    `value_of(action)` returns `(value, priced_by, fell_back)` -- the
    POLICY's answer, which is where the Identity fallback lives (the
    estimator never substitutes itself; DE 353).
    """
    counter = counter or W.FallbackCounter()
    quotes = []
    for a in actions:
        value, priced_by, fell_back = value_of(a)
        if fell_back:
            counter.fell_back += 1
        else:
            counter.used_candidate += 1
        quotes.append(quote_from(
            value, slug=a.slug, generation_id=a.generation_id,
            half_spread=half_spread, priced_by=priced_by,
            fell_back=fell_back, side=side, tick=tick,
            decl_dir=decl_dir, decision_ms=decision_ms))
    return {"protocol": PROTOCOL, "n_quotes": len(quotes),
            "quotes": quotes, "trajectory": trajectory(quotes),
            "fallbacks": counter.as_dict()}


def compare_runs(baseline: dict, challenger: dict) -> dict:
    """IS THE SEAM LOAD-BEARING? Anchors first, labels second.

    A run whose ANCHORS are identical while its `priced_by` LABELS differ
    is decorative: the schema field moved and the consumed value did not.
    That is the §5 falsifier, expressed as a predicate a reader can run
    over any two runs rather than a claim about the code.
    """
    a = [q.anchor for q in baseline["quotes"]]
    b = [q.anchor for q in challenger["quotes"]]
    la = [q.priced_by for q in baseline["quotes"]]
    lb = [q.priced_by for q in challenger["quotes"]]
    same_anchor = a == b
    same_label = la == lb
    moved = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    out = {"n": len(a), "anchors_identical": same_anchor,
           "labels_identical": same_label,
           "trajectories_identical":
               baseline["trajectory"] == challenger["trajectory"],
           "n_anchors_moved": len(moved), "first_moved_index":
               moved[0] if moved else None,
           "verdict": None}
    if same_anchor and not same_label:
        out["verdict"] = DECORATIVE
        out["reading"] = (
            "only the estimator LABEL changed; every consumed value is "
            "identical, so nothing downstream can depend on the "
            "challenger. An unused schema field is not a partial "
            "implementation.")
    elif same_anchor and same_label:
        out["verdict"] = "BIT_IDENTICAL"
        out["reading"] = ("the same estimator produced the same values: "
                          "substitution is a no-op, as it must be")
    else:
        out["verdict"] = "VALUE_IS_LOAD_BEARING"
        out["reading"] = (f"{len(moved)} of {len(a)} anchors moved, so the "
                          f"consumed value reaches the quote")
    return out


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    import de_fair_value_actions as A
    W_START = 1788825600
    rows = [{"coin": "btc", "slug": "btc-updown-5m-1788825600",
             "generation_id": f"g{i}", "decision_recv_ns": 1000 + i,
             "quote_side": "BID", "up_probability_consumed": 0.5,
             "window_start": W_START, "on_identity_reference_path": True}
            for i in range(4)]
    # GATE 4 NOW REQUIRES THE CANONICAL POPULATION (REVIEW 201): its
    # membership is derived, not read off the row. This fixture supplies
    # one -- and the fact that tightening gate 4 broke this fixture is why
    # the drive is done from the REF's bytes and not only in the tree
    # where the change was made.
    POP = {"actions": [(r["slug"], r["generation_id"]) for r in rows],
           "population": "P003_NEUTRAL_REFERENCE_PATH_FIXTURE",
           "as_of": "2026-09-11T19:00:00Z",
           "source_identity": f"{Path(__file__).name}.falsify fixture"}
    acts = A.build_actions(
        rows, canonical_population=POP)["actions"]
    acts = sorted(acts, key=lambda a: a.decision_recv_ns)

    ident = {a.generation_id: 0.50 + 0.01 * i for i, a in enumerate(acts)}
    chall = {a.generation_id: 0.70 - 0.02 * i for i, a in enumerate(acts)}

    def identity_only(a):
        return ident[a.generation_id], FP.IDENTITY, False

    def identity_again(a):
        return ident[a.generation_id], FP.IDENTITY, False

    def challenger(a):
        return chall[a.generation_id], FP.BN_BOOKTICKER, False

    def decorative(a):
        # THE VALUE IS IDENTITY'S; ONLY THE LABEL SAYS OTHERWISE.
        return ident[a.generation_id], FP.BN_BOOKTICKER, False

    def absent_challenger(a):
        return ident[a.generation_id], FP.IDENTITY, True

    # --- §7 QUOTE MAPPING: seven clauses, each two ways ----------------
    import tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        DECL = Path(td)
        (DECL / "fixture_tick.json").write_text(json.dumps(
            {"market": {"tick_size": 0.01}, "placement_latency_ms": 250.0}))

        def q(v, **kw):
            kw.setdefault("slug", "s")
            kw.setdefault("generation_id", "g")
            kw.setdefault("half_spread", 0.013)
            kw.setdefault("priced_by", FP.BN_BOOKTICKER)
            kw.setdefault("decl_dir", DECL)
            return quote_from(v, **kw)

        # (a) DOWN uses 1 - p
        up, down = q(0.62), q(0.62, side="DOWN")
        ck("(a) UP quotes p and DOWN quotes 1 - p",
           up.p_side == 0.62 and abs(down.p_side - 0.38) < 1e-12
           and up.side == "UP" and down.side == "DOWN",
           f"UP {up.p_side} / DOWN {down.p_side}")
        try:
            q(0.62, side="SIDEWAYS")
            bad_side = ""
        except SeamRefused as exc:
            bad_side = str(exc)
        ck("  and an unnamed side REFUSES rather than defaulting to UP",
           "SIDEWAYS" in bad_side, bad_side[:52] or "DEFAULTED")

        # (b) directional rounding to the legal tick
        r = q(0.615)                       # raw bid .602, raw ask .628
        ck("(b) the bid rounds DOWN and the ask rounds UP to the tick",
           r.bid == 0.60 and r.ask == 0.63 and r.tick == 0.01,
           f"raw [0.602, 0.628] -> [{r.bid}, {r.ask}] at tick {r.tick}")
        ck("  and neither is the symmetric round -- the bid is never "
           "above its own raw price",
           r.bid <= 0.602 + 1e-12 and r.ask >= 0.628 - 1e-12
           and round(0.602, 2) == 0.6 and round(0.628, 2) == 0.63,
           "floor/ceil, not round()")

        # (c) bounded to the legal binary range
        hi, lo = q(0.999), q(0.001)
        ck("(c) prices are BOUNDED to [0, 1] at both extremes",
           hi.ask <= 1.0 and lo.bid >= 0.0,
           f"p=0.999 -> ask {hi.ask}; p=0.001 -> bid {lo.bid}")
        mid = q(0.50)
        ck("  and an in-range quote is NOT bounded, so the bound is not "
           "a no-op that always fires",
           mid.bounded == () and hi.bounded != (),
           f"mid bounded {len(mid.bounded)}, extreme bounded "
           f"{len(hi.bounded)}")

        # (d) a crossing quote is withheld, by name
        cross = q(0.80, best_ask=0.70)
        ck("(d) a quote that would CROSS emits PLACE_WITHHELD with reason "
           "MARKETABLE_CROSS",
           cross.withheld and cross.withheld_reason == MARKETABLE_CROSS,
           f"bid {cross.bid} vs best_ask 0.70 -> {cross.withheld_reason}")
        passive = q(0.50, best_ask=0.70, best_bid=0.30)
        ck("  and a passive quote inside the book is NOT withheld",
           not passive.withheld and passive.withheld_reason == "",
           f"bid {passive.bid} ask {passive.ask} inside [0.30, 0.70]")

        # (e) never silently clamped
        ck("(e) every bound applied is RECORDED, never silent",
           all("from" in b and "to" in b and "why" in b
               for b in (json.loads(x) for x in hi.bounded)),
           json.loads(hi.bounded[0])["why"])
        ck("  and a CROSS is withheld rather than pulled back to the "
           "touch -- the quote keeps its own price",
           cross.withheld and cross.bid > 0.70,
           f"withheld at its own bid {cross.bid}, not clamped to 0.70")

        # (f)/(g) no zero-latency privilege, and the latency is declared
        t = q(0.55, decision_ms=1000.0)
        ck("(f) a candidate-induced quote is effective at decision + "
           "placement latency -- no zero-latency privilege",
           t.effective_ms == 1250.0 and t.latency_ms == 250.0,
           f"decision {t.decision_ms} -> effective {t.effective_ms}")
        ck("  and the latency is READ from a declaration, not typed here",
           placement_latency_ms(DECL)["value"] == 250.0
           and placement_latency_ms()["declared_by"].endswith(".json"),
           placement_latency_ms()["declared_by"])

    # (g)/(b) the numbers are REFUSED when undeclared -- the state today
    with _tf.TemporaryDirectory() as td2:
        try:
            quote_from(0.5, slug="s", generation_id="g", half_spread=0.01,
                       priced_by=FP.IDENTITY, decl_dir=Path(td2))
            undeclared = ""
        except SeamRefused as exc:
            undeclared = str(exc)
        ck("an UNDECLARED tick REFUSES -- this file will not invent one",
           LEGAL_TICK_NOT_DECLARED in undeclared,
           undeclared[:58] or "INVENTED A TICK")
    try:
        legal_tick()
        tick_today = "declared"
    except SeamRefused:
        tick_today = "NOT YET DECLARED by DA -- clauses (b) and (c) refuse "\
                     "on the real declarations until it lands"
    ck("  and the REAL declarations are reported as they are, not as I "
       "would like them",
       tick_today.startswith("NOT YET") or tick_today == "declared",
       tick_today[:72])

    # THE TICK IS AN INPUT THESE FIXTURES MUST SUPPLY: without it they
    # would be testing the tick's absence, not the seam's properties.
    TICK = 0.01
    def run(av, **kw):
        return run_seam(acts, av, tick=TICK, **kw)

    base = run(identity_only)
    same = run(identity_again)
    ck("substituting Identity for Identity is BIT-IDENTICAL, trajectory "
       "included",
       base["trajectory"] == same["trajectory"]
       and compare_runs(base, same)["verdict"] == "BIT_IDENTICAL",
       base["trajectory"][:16])

    moved = run(challenger)
    cmp_moved = compare_runs(base, moved)
    ck("a NON-IDENTITY value MOVES the quote anchor (positive control)",
       cmp_moved["verdict"] == "VALUE_IS_LOAD_BEARING"
       and cmp_moved["n_anchors_moved"] == len(acts)
       and not cmp_moved["trajectories_identical"],
       f"{cmp_moved['n_anchors_moved']} of {cmp_moved['n']} anchors moved")
    ck("  and the move is visible in the POSTED pair, not only the anchor",
       base["quotes"][0].bid != moved["quotes"][0].bid
       and base["quotes"][0].ask != moved["quotes"][0].ask,
       f"bid {base['quotes'][0].bid} -> {moved['quotes'][0].bid}")

    deco = run(decorative)
    cmp_deco = compare_runs(base, deco)
    ck("changing ONLY the estimator metadata is DETECTED as decorative",
       cmp_deco["verdict"] == DECORATIVE
       and cmp_deco["anchors_identical"]
       and not cmp_deco["labels_identical"],
       cmp_deco["verdict"])
    ck("  and the decorative run's TRAJECTORY differs only by the label, "
       "which is why anchors are compared first",
       not cmp_deco["trajectories_identical"]
       and cmp_deco["n_anchors_moved"] == 0,
       f"anchors moved {cmp_deco['n_anchors_moved']}, trajectory differs "
       f"by label only")

    fb = run(absent_challenger)
    ck("an ABSENT challenger falls back to Identity, and the fallback is "
       "COUNTED",
       fb["trajectory"] != base["trajectory"]  # labels differ
       and fb["fallbacks"]["fell_back"] == len(acts)
       and fb["fallbacks"]["fallback_share"] == 1.0
       and all(q.priced_by == FP.IDENTITY for q in fb["quotes"]),
       json.dumps(fb["fallbacks"]))
    ck("  and its ANCHORS equal the Identity run exactly -- the fallback "
       "is Identity's value, not an approximation of it",
       compare_runs(base, fb)["anchors_identical"],
       "anchors identical")

    try:
        run(lambda a: (None, FP.BN_BOOKTICKER, False))
        no_val = ""
    except SeamRefused as exc:
        no_val = str(exc)
    ck("a seam handed NO VALUE refuses rather than quoting around nothing",
       NO_VALUE in no_val, no_val[:56] or "QUOTED AROUND None")

    # THE CELL THAT WOULD CATCH A REGRESSION TO METADATA-ONLY: run the
    # REAL wrapper's policy_value through the seam and require the anchor
    # to follow the candidate.
    counter = W.FallbackCounter()
    p_ident = FP.FairPrice(coin="btc", window_start=W_START, outcome="UP",
                           value=0.5, source_timestamp=1.0,
                           local_knowledge_timestamp=1.5, freshness_s=0.5,
                           status=FP.OK, estimator=FP.IDENTITY)
    p_cand = FP.FairPrice(coin="btc", window_start=W_START, outcome="UP",
                          value=0.8, source_timestamp=1.0,
                          local_knowledge_timestamp=1.5, freshness_s=0.5,
                          status=FP.OK, estimator=FP.BN_BOOKTICKER)
    cand = W.Candidate(price=p_cand, identity=W.identity_of(FP.BN_BOOKTICKER),
                       cause=W.OK, inputs={}, decision_local_time=1.5)
    got = W.policy_value(cand, p_ident, counter)
    q = quote_from(got["value"], slug="s", generation_id="g",
                   half_spread=0.01, priced_by=got["priced_by"],
                   fell_back=got["fell_back"], tick=TICK, latency_ms=250.0)
    ck("the REAL policy_value feeds the seam, and the anchor is the "
       "CANDIDATE's value",
       q.anchor == 0.8 and q.priced_by == FP.BN_BOOKTICKER
       and counter.as_dict()["used_candidate"] == 1,
       f"anchor {q.anchor} priced_by {q.priced_by}")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL,
                      "decorative_refusal": DECORATIVE}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
