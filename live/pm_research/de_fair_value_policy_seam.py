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
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_fair_price_identity as FP              # noqa: E402
import de_fair_price_wrapper as W                # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_POLICY_SEAM_V1"
DECORATIVE = "FAIRPRICE_SEAM_IS_DECORATIVE_VALUE_UNCHANGED"
NO_VALUE = "POLICY_SEAM_CONSUMED_NO_VALUE"


class SeamRefused(ValueError):
    """The seam cannot be exercised as declared."""


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

    def as_dict(self) -> dict:
        return asdict(self)


def quote_from(value: float, *, slug: str, generation_id: str,
               half_spread: float, priced_by: str,
               fell_back: bool = False) -> Quote:
    """THE CONSUMING FUNCTION. The anchor IS the probability."""
    if not FP._finite_real(value):
        raise SeamRefused(
            f"REFUSED {NO_VALUE}: the seam was handed {value!r}. A quote "
            f"built from no value is the decorative case with extra steps.")
    return Quote(slug=slug, generation_id=generation_id, anchor=float(value),
                 bid=round(float(value) - half_spread, 12),
                 ask=round(float(value) + half_spread, 12),
                 priced_by=priced_by, fell_back=fell_back)


def trajectory(quotes) -> str:
    """A digest of the posted path -- the thing 'bit-identical' is about."""
    return hashlib.sha256(json.dumps(
        [q.as_dict() for q in quotes], sort_keys=True).encode()).hexdigest()


def run_seam(actions, value_of, *, half_spread=0.01,
             counter: W.FallbackCounter = None) -> dict:
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
            fell_back=fell_back))
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
    POP = [(r["slug"], r["generation_id"]) for r in rows]
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

    base = run_seam(acts, identity_only)
    same = run_seam(acts, identity_again)
    ck("substituting Identity for Identity is BIT-IDENTICAL, trajectory "
       "included",
       base["trajectory"] == same["trajectory"]
       and compare_runs(base, same)["verdict"] == "BIT_IDENTICAL",
       base["trajectory"][:16])

    moved = run_seam(acts, challenger)
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

    deco = run_seam(acts, decorative)
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

    fb = run_seam(acts, absent_challenger)
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
        run_seam(acts, lambda a: (None, FP.BN_BOOKTICKER, False))
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
                   fell_back=got["fell_back"])
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
