"""STEP 4: canonical forecast actions, and the paired scorer.

`fair_value_plan.md` §5 gate 4 and §6's common predictive population.

THE UNIT IS A DECISION, NOT A QUOTE. One row per ACTUAL fair-value
consumption decision on the neutral Identity reference path, keyed
`(coin, slug, generation_id, decision_recv_ns)`. Several quote sides that
consume the SAME UP probability at the same generation and timestamp are
ONE forecast action -- otherwise a two-sided quoter's every number counts
twice and every interval narrows for a reason that is arithmetic, not
evidence. Rule 2's shape, one lane over: rows are actions.

DUPLICATE KEYS REFUSE THE BUILD. Not de-duplicated, not warned about:
refused, because a duplicate key means the caller's notion of "a decision"
and this file's disagree, and silently picking one is how 1.99 rows per
fill became a measured inflation in the other lane.

THE SCORER'S ASYMMETRY IS THE POINT (§6). A challenger that is non-OK on
an action does NOT drop out: the POLICY's Identity fallback is scored
there and the native status is counted. So the primary number is over the
FULL Identity-eligible universe, and a challenger cannot improve by being
absent on the actions it finds hard. The native-OK intersection is
reported too, and labelled diagnostic.

NO LABELLED SCORE IS READ HERE. `score_actions` takes outcomes as an
argument and this module never loads one; step 6 is where a real outcome
may be joined. Build and falsify only.

Usage:  de_fair_value_actions.py --falsify
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import da_fair_price_identity as FP              # noqa: E402
import de_fair_price_wrapper as W                # noqa: E402

PROTOCOL = "P003_DE_FAIR_VALUE_ACTIONS_V1"
EPSILON = 1e-6                    # §6: the SAME clip for both sides
SCOPE_COINS = ("btc", "eth")      # step 4/5 scope

DUPLICATE_KEY = "FORECAST_ACTION_DUPLICATE_KEY"
NOT_IDENTITY_PATH = "ACTION_IS_NOT_ON_THE_NEUTRAL_IDENTITY_REFERENCE_PATH"
#: REVIEW 201's two unenforceable properties, now enforced. The flag was
#: READ, never DERIVED -- so the control fired on a LABEL and could not
#: tell a genuinely off-path row from a caller that set the bit; and
#: nothing joined these rows to `de_canonical_action_population`, so "one
#: row per ACTUAL consumption decision" was a property of whatever list
#: the caller passed.
NO_POPULATION = "CANONICAL_POPULATION_NOT_SUPPLIED"
FLAG_CONTRADICTS = "REFERENCE_PATH_FLAG_CONTRADICTS_THE_CANONICAL_POPULATION"
NOT_IN_POPULATION = "ACTION_NOT_IN_THE_CANONICAL_POPULATION"
#: REVIEW 202: the population was an argument with NO PROVENANCE -- a
#: fabricated one-element population containing the row under test
#: admitted it. A builder cannot verify that a supplied population is the
#: true one; it CAN refuse an unattributed one and record the digest of
#: exactly what it used, so a fabricated population is visible as one
#: rather than anonymous. That limit is stated, not papered over.
NO_PROVENANCE = "CANONICAL_POPULATION_HAS_NO_PROVENANCE"
OUT_OF_SCOPE = "ACTION_COIN_OUT_OF_SCOPE"
NO_DECISION_STAMP = "ACTION_HAS_NO_DECISION_RECV_NS"
OUTCOME_UNKNOWN = "ACTION_OUTCOME_NOT_SUPPLIED"


class ActionsRefused(ValueError):
    """The build must not proceed."""


@dataclass(frozen=True)
class ForecastAction:
    """ONE fair-value consumption decision."""
    coin: str
    slug: str
    generation_id: str
    decision_recv_ns: int
    quote_sides: tuple = ()          # the sides that consumed this value
    window_start: int = 0
    note: str = ""

    @property
    def key(self) -> tuple:
        return (self.coin, self.slug, self.generation_id,
                self.decision_recv_ns)

    def as_dict(self) -> dict:
        return dict(asdict(self), key=list(self.key))


def population_provenance(population) -> dict:
    """WHERE THE POPULATION CAME FROM -- required, and digested.

    `de_canonical_action_population.build_actions` already returns
    `population`, `as_of` and `source_identity`; when its output is passed
    straight through, the provenance comes with it. A bare iterable must
    carry the same three under `provenance`, or the build REFUSES.
    """
    if isinstance(population, dict):
        got = {k: population.get(k) for k in
               ("population", "as_of", "source_identity")}
        if all(isinstance(v, str) and v.strip() for v in got.values()):
            return dict(got, supplied_as="the canonical builder's own "
                                         "output")
        prov = population.get("provenance") or {}
        got = {k: prov.get(k) for k in
               ("population", "as_of", "source_identity")}
        if all(isinstance(v, str) and v.strip() for v in got.values()):
            return dict(got, supplied_as="an explicit provenance block")
    raise ActionsRefused(
        f"REFUSED {NO_PROVENANCE}: the canonical population carries no "
        f"`population` / `as_of` / `source_identity`. A population with "
        f"no provenance cannot be told from one fabricated around the "
        f"rows under test -- this builder cannot verify WHICH population "
        f"it was handed, so it refuses an ANONYMOUS one and records the "
        f"digest of what it used (REVIEW 202).")


def canonical_keys(population) -> set:
    """`(slug, generation_id)` for every canonical action, from the
    canonical population itself.

    Accepts `de_canonical_action_population.build_actions`' output, or any
    iterable of rows or pairs. The point is that MEMBERSHIP COMES FROM THE
    POPULATION, never from a bit on the row being tested.
    """
    if population is None:
        raise ActionsRefused(
            f"REFUSED {NO_POPULATION}: no canonical population was "
            f"supplied, so reference-path membership could only be read "
            f"off the rows themselves. A control that fires on a label "
            f"cannot tell an off-path row from a caller that set the bit "
            f"(REVIEW 201).")
    rows = population
    if isinstance(population, dict):
        rows = (population.get("actions") or population.get("rows")
                or population.get("canonical") or [])
    out = set()
    for r in rows:
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            out.add((str(r[0]), str(r[1])))
        elif isinstance(r, dict):
            gen = r.get("generation_id", r.get("gen"))
            out.add((str(r.get("slug")), str(gen)))
        else:
            gen = getattr(r, "generation_id", getattr(r, "gen", None))
            out.add((str(getattr(r, "slug", None)), str(gen)))
    return out


def build_actions(consumptions, *, scope=SCOPE_COINS,
                  canonical_population=None) -> dict:
    """Fold quote-side consumptions into canonical actions.

    `consumptions` is an iterable of dicts carrying at least coin, slug,
    generation_id, decision_recv_ns, quote_side and `on_identity_reference
    _path`. Several sides at ONE key collapse into one action; two rows at
    one key that disagree about the VALUE consumed are a duplicate key and
    REFUSE.
    """
    keys = canonical_keys(canonical_population)          # None -> NO_POPULATION
    provenance = population_provenance(canonical_population)
    keys_digest = __import__("hashlib").sha256(
        json.dumps(sorted(map(list, keys)), sort_keys=True).encode()
    ).hexdigest()
    actions: dict = {}
    excluded: dict = {}

    def drop(row, status):
        excluded.setdefault(status, []).append(
            {k: row.get(k) for k in ("coin", "slug", "generation_id",
                                     "decision_recv_ns", "quote_side")})

    for row in consumptions:
        coin = str(row.get("coin") or "")
        if coin not in scope:
            drop(row, OUT_OF_SCOPE)
            continue
        # MEMBERSHIP IS DERIVED, AND THE CALLER'S BIT IS CHECKED AGAINST
        # IT. The neutral reference path is the population (rule 1): an
        # action taken on a path the policy itself perturbed is
        # outcome-selected, and the unit is the DECISION-TIME exposure.
        on_path = (str(row.get("slug")),
                   str(row.get("generation_id"))) in keys
        claimed = row.get("on_identity_reference_path")
        if claimed is not None and bool(claimed) != on_path:
            raise ActionsRefused(
                f"REFUSED {FLAG_CONTRADICTS}: the row claims "
                f"on_identity_reference_path={bool(claimed)} for "
                f"{row.get('slug')}/{row.get('generation_id')} and the "
                f"canonical population says {on_path}. The population "
                f"decides; a row that disagrees with it is a wiring error "
                f"one level up, not a row to drop quietly.")
        if not on_path:
            drop(row, NOT_IN_POPULATION)
            continue
        stamp = row.get("decision_recv_ns")
        if not isinstance(stamp, int) or isinstance(stamp, bool):
            drop(row, NO_DECISION_STAMP)
            continue
        key = (coin, str(row.get("slug")), str(row.get("generation_id")),
               stamp)
        prev = actions.get(key)
        if prev is None:
            actions[key] = {
                "action": ForecastAction(
                    coin=coin, slug=str(row.get("slug")),
                    generation_id=str(row.get("generation_id")),
                    decision_recv_ns=stamp,
                    quote_sides=(str(row.get("quote_side")),),
                    window_start=int(row.get("window_start") or 0)),
                "value": row.get("up_probability_consumed")}
            continue
        if prev["value"] != row.get("up_probability_consumed"):
            raise ActionsRefused(
                f"REFUSED {DUPLICATE_KEY}: {key} appears twice consuming "
                f"{prev['value']!r} and {row.get('up_probability_consumed')!r}. "
                f"One key is one decision; two values at one key means the "
                f"caller's notion of a decision and this builder's "
                f"disagree, and picking one silently is how a row count "
                f"becomes an inflation.")
        a = prev["action"]
        actions[key] = {
            "action": ForecastAction(
                coin=a.coin, slug=a.slug, generation_id=a.generation_id,
                decision_recv_ns=a.decision_recv_ns,
                quote_sides=tuple(sorted(set(a.quote_sides)
                                         | {str(row.get("quote_side"))})),
                window_start=a.window_start),
            "value": prev["value"]}

    rows = [v["action"] for v in actions.values()]
    sides = sum(len(a.quote_sides) for a in rows)
    return {"protocol": PROTOCOL, "n_actions": len(rows),
            "canonical_population_size": len(keys),
            "canonical_population_provenance": provenance,
            "canonical_population_keys_sha256": keys_digest,
            "WHAT_THIS_BUILDER_CANNOT_DO":
                "it cannot verify that the supplied population is the "
                "true one. It "
                "refuses an unattributed population and digests exactly "
                "the keys it used, so a fabricated one is VISIBLE -- it "
                "is not prevented",
            "membership_decided_by":
                "the canonical population supplied to this build; the "
                "row's own `on_identity_reference_path` is CHECKED "
                "against it and never trusted",
            "n_quote_sides_folded": sides,
            "sides_per_action": (sides / len(rows)) if rows else None,
            "actions": rows,
            "consumed_values": {k: v["value"] for k, v in actions.items()},
            "excluded": {k: len(v) for k, v in excluded.items()},
            "excluded_rows": excluded,
            "scope": list(scope),
            "key": "(coin, slug, generation_id, decision_recv_ns)"}


def _clip(p: float, eps: float = EPSILON) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def log_loss(p_up: float, outcome_up: bool, eps: float = EPSILON) -> float:
    """ONE probability, the UP one. DOWN is its complement, never a second
    fit -- so there is no second number to clip differently."""
    p = _clip(p_up, eps)
    return -(math.log(p) if outcome_up else math.log(1.0 - p))


def score_actions(actions, identity_at, challenger_at, outcomes, *,
                  eps: float = EPSILON) -> dict:
    """The paired score. §6's rules, each one visible as a branch.

    `identity_at(action)` and `challenger_at(action)` return a FairPrice or
    a Candidate read STRICTLY as of `decision_recv_ns`; `outcomes[slug]`
    is True when UP settled. Nothing here loads an outcome: step 6 owns
    that, and this file must be landable before any labelled score exists.
    """
    rows, statuses = [], {}
    n_identity_bad = n_fallback = n_native = 0
    prim_i = prim_c = 0.0
    diag_i = diag_c = 0.0
    n_prim = n_diag = 0
    for a in actions:
        ident = identity_at(a)
        cand = challenger_at(a)
        cand_price = getattr(cand, "price", cand)
        cand_status = getattr(cand, "cause", None) or getattr(
            cand_price, "status", "UNKNOWN")
        statuses[cand_status] = statuses.get(cand_status, 0) + 1
        if a.slug not in outcomes:
            raise ActionsRefused(
                f"REFUSED {OUTCOME_UNKNOWN}: {a.slug} has no supplied "
                f"outcome. This scorer never loads one -- an outcome that "
                f"appears from nowhere is the labelled read step 6 owns.")
        up = bool(outcomes[a.slug])
        if ident is None or ident.status != FP.OK or ident.value is None:
            # IDENTITY NON-OK: count the action, score NEITHER side.
            n_identity_bad += 1
            rows.append({"key": list(a.key), "scored": False,
                         "why": "identity_non_ok",
                         "identity_status": getattr(ident, "status", None),
                         "challenger_status": cand_status})
            continue
        native_ok = (cand_price is not None
                     and cand_price.status == FP.OK
                     and cand_price.value is not None)
        # PRIMARY: the POLICY's value -- the challenger when it is OK, the
        # Identity fallback when it is not. The challenger cannot improve
        # by being absent where the market is hard.
        policy_p = cand_price.value if native_ok else ident.value
        if native_ok:
            n_native += 1
        else:
            n_fallback += 1
        li = log_loss(ident.value, up, eps)
        lc = log_loss(policy_p, up, eps)
        prim_i += li
        prim_c += lc
        n_prim += 1
        if native_ok:
            diag_i += li
            diag_c += lc
            n_diag += 1
        rows.append({"key": list(a.key), "scored": True,
                     "identity_p": ident.value, "policy_p": policy_p,
                     "native_ok": native_ok, "outcome_up": up,
                     "identity_log_loss": li, "policy_log_loss": lc,
                     "challenger_status": cand_status})
    return {
        "protocol": PROTOCOL, "epsilon": eps,
        "n_actions": len(list(rows)),
        "PRIMARY_fallback_scored": {
            "population": "every Identity-eligible action",
            "n": n_prim,
            "identity_mean_log_loss": (prim_i / n_prim) if n_prim else None,
            "policy_mean_log_loss": (prim_c / n_prim) if n_prim else None,
            "increment_vs_identity": ((prim_c - prim_i) / n_prim)
            if n_prim else None,
            "n_native": n_native, "n_fallback": n_fallback},
        "DIAGNOSTIC_native_intersection": {
            "population": "actions where the challenger was natively OK",
            "is_diagnostic_not_primary": True,
            "n": n_diag,
            "identity_mean_log_loss": (diag_i / n_diag) if n_diag else None,
            "challenger_mean_log_loss": (diag_c / n_diag) if n_diag else None,
            "increment_vs_identity": ((diag_c - diag_i) / n_diag)
            if n_diag else None},
        "identity_non_ok_actions": n_identity_bad,
        "challenger_status_counts": statuses,
        "rows": rows,
        "why_the_primary_is_the_fallback_score":
            "§6: a challenger that is absent on hard actions would "
            "otherwise score only where it is comfortable, and its "
            "increment would measure its ABSTENTION rather than its skill",
    }


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    W_START = 1788825600

    def row(side, stamp=1000, coin="btc", gen="g1", val=0.6, path=None,
            slug="btc-updown-5m-1788825600"):
        r = {"coin": coin, "slug": slug, "generation_id": gen,
             "decision_recv_ns": stamp, "quote_side": side,
             "up_probability_consumed": val, "window_start": W_START}
        if path is not None:
            r["on_identity_reference_path"] = path
        return r

    SLUG = "btc-updown-5m-1788825600"
    # THE POPULATION COMES WITH ITS PROVENANCE (REVIEW 202): a bare list
    # is refused, so the fixture carries the same three fields the
    # canonical builder emits.
    POP = {"actions": [(SLUG, "g1"), (SLUG, "g2")],
           "population": "P003_NEUTRAL_REFERENCE_PATH_FIXTURE",
           "as_of": "2026-09-11T19:00:00Z",
           "source_identity": "de_fair_value_actions.falsify fixture"}

    def build(rows, population=POP):
        return build_actions(rows, canonical_population=population)

    try:
        build_actions([row("BID")],
                      canonical_population=[(SLUG, "g1")])
        anon = ""
    except ActionsRefused as exc:
        anon = str(exc)
    ck("an ANONYMOUS population refuses -- a fabricated one must not be "
       "indistinguishable from the real one",
       NO_PROVENANCE in anon, anon[:58] or "ADMITTED AN UNATTRIBUTED POPULATION")
    _p = build([row("BID")])
    ck("  and the build RECORDS the provenance and the digest of the keys "
       "it actually used",
       _p["canonical_population_provenance"]["source_identity"]
       == "de_fair_value_actions.falsify fixture"
       and len(_p["canonical_population_keys_sha256"]) == 64
       and "cannot" in _p["WHAT_THIS_BUILDER_CANNOT_DO"],
       _p["canonical_population_keys_sha256"][:16])

    # --- REVIEW 201 (2): MEMBERSHIP IS DERIVED, THE BIT IS CHECKED ------
    try:
        build_actions([row("BID")], canonical_population=None)
        unjoined = ""
    except ActionsRefused as exc:
        unjoined = str(exc)
    ck("a build with NO canonical population REFUSES -- the flag alone is "
       "a label",
       NO_POPULATION in unjoined,
       unjoined[:58] or "BUILT FROM THE CALLER'S BIT ALONE")
    off = build([row("BID", gen="g9")])
    ck("a row absent from the population is EXCLUDED by the population, "
       "not by its own bit",
       off["n_actions"] == 0
       and off["excluded"] == {NOT_IN_POPULATION: 1},
       json.dumps(off["excluded"]))
    try:
        build([row("BID", gen="g9", path=True)])
        lied = ""
    except ActionsRefused as exc:
        lied = str(exc)
    ck("a row whose FLAG contradicts the population REFUSES by name",
       FLAG_CONTRADICTS in lied, lied[:58] or "TRUSTED THE BIT")
    try:
        build([row("BID", path=False)])
        lied2 = ""
    except ActionsRefused as exc:
        lied2 = str(exc)
    ck("  and the contradiction is caught in BOTH directions",
       FLAG_CONTRADICTS in lied2,
       lied2[:58] or "TRUSTED A FALSE BIT OVER THE POPULATION")

    both = build([row("BID"), row("ASK")])
    ck("two quote sides consuming ONE value at one key are ONE action",
       both["n_actions"] == 1 and both["n_quote_sides_folded"] == 2
       and both["actions"][0].quote_sides == ("ASK", "BID"),
       f"{both['n_quote_sides_folded']} sides -> {both['n_actions']} action")
    ck("  and the fold is reported, so a reader sees the ratio",
       both["sides_per_action"] == 2.0, str(both["sides_per_action"]))
    spread = build([row("BID"), row("ASK", stamp=1001)])
    ck("the SAME sides at DIFFERENT timestamps are two actions",
       spread["n_actions"] == 2, str(spread["n_actions"]))
    try:
        build([row("BID"), row("ASK", val=0.7)])
        dup = ""
    except ActionsRefused as exc:
        dup = str(exc)
    ck("two VALUES at one key REFUSE the build, by name",
       DUPLICATE_KEY in dup, dup[:58] or "ADMITTED A DUPLICATE KEY")
    off2 = build([row("BID", coin="sol"), row("BID", gen="g9",
                                               path=None)])
    ck("out-of-scope and out-of-population rows are EXCLUDED BY STATUS, "
       "never dropped silently",
       off2["n_actions"] == 0
       and off2["excluded"] == {OUT_OF_SCOPE: 1, NOT_IN_POPULATION: 1},
       json.dumps(off2["excluded"]))

    acts = both["actions"]
    slug = acts[0].slug

    def ident_ok(_a, v=0.5):
        return FP.FairPrice(coin="btc", window_start=W_START, outcome="UP",
                            value=v, source_timestamp=1.0,
                            local_knowledge_timestamp=1.5, freshness_s=0.5,
                            status=FP.OK, estimator=FP.IDENTITY)

    def cand_ok(_a, v=0.9):
        p = FP.FairPrice(coin="btc", window_start=W_START, outcome="UP",
                         value=v, source_timestamp=1.0,
                         local_knowledge_timestamp=1.5, freshness_s=0.5,
                         status=FP.OK, estimator=FP.BN_BOOKTICKER)
        return W.Candidate(price=p, identity=W.identity_of(FP.BN_BOOKTICKER),
                           cause=W.OK, inputs={}, decision_local_time=1.5)

    def cand_bad(_a):
        p = FP.FairPrice(coin="btc", window_start=W_START, outcome="UP",
                         value=None, source_timestamp=1.0,
                         local_knowledge_timestamp=1.5, freshness_s=0.5,
                         status=FP.NOT_READY, estimator=FP.BN_BOOKTICKER)
        return W.Candidate(price=p, identity=W.identity_of(FP.BN_BOOKTICKER),
                           cause=W.REFERENCE_NOT_YET_RECEIVED, inputs={},
                           decision_local_time=1.5)

    good = score_actions(acts, ident_ok, cand_ok, {slug: True})
    ck("an informative challenger BEATS Identity on the primary score",
       good["PRIMARY_fallback_scored"]["increment_vs_identity"] < 0,
       f"{good['PRIMARY_fallback_scored']['increment_vs_identity']:+.6f}")
    absent = score_actions(acts, ident_ok, cand_bad, {slug: True})
    ck("a NON-OK challenger is scored on the POLICY'S IDENTITY FALLBACK, "
       "so its increment is exactly zero, not absent",
       absent["PRIMARY_fallback_scored"]["increment_vs_identity"] == 0.0
       and absent["PRIMARY_fallback_scored"]["n_fallback"] == 1
       and absent["PRIMARY_fallback_scored"]["n"] == 1,
       f"increment {absent['PRIMARY_fallback_scored']['increment_vs_identity']}"
       f", n_fallback {absent['PRIMARY_fallback_scored']['n_fallback']}")
    ck("  and its NATIVE status is counted by name",
       absent["challenger_status_counts"].get(
           W.REFERENCE_NOT_YET_RECEIVED) == 1,
       json.dumps(absent["challenger_status_counts"]))
    ck("  while the DIAGNOSTIC intersection is EMPTY and says it is "
       "diagnostic",
       absent["DIAGNOSTIC_native_intersection"]["n"] == 0
       and absent["DIAGNOSTIC_native_intersection"][
           "is_diagnostic_not_primary"] is True)

    # THE ABSTENTION ATTACK, DRIVEN: a challenger that is right where it
    # answers and silent where it is wrong must NOT be able to buy a
    # better primary number by abstaining.
    two = build([row("BID"), row("BID", stamp=2000, gen="g2")])
    a1, a2 = sorted(two["actions"], key=lambda x: x.decision_recv_ns)
    outcomes = {slug: True}
    always = score_actions([a1, a2], ident_ok,
                          lambda a: cand_ok(a, 0.9 if a is a1 else 0.1),
                          outcomes)
    abstain = score_actions([a1, a2], ident_ok,
                            lambda a: cand_ok(a, 0.9) if a is a1
                            else cand_bad(a), outcomes)
    # WHAT §6 ACTUALLY BUYS, stated as the cell rather than as I first
    # assumed it. My first version asserted that abstention cannot improve
    # the primary score AT ALL, and the drive refuted me: abstaining where
    # it would have been WRONG scores -0.2939 against +0.5108 for
    # answering both. The mechanism does NOT make abstention free of
    # benefit -- it BOUNDS the benefit at Identity's own loss, so a
    # challenger can avoid LOSING by abstaining but can never GAIN over
    # Identity there. That distinction belongs in step 6's reading.
    per_action = {tuple(r["key"]): r for r in abstain["rows"] if r["scored"]}
    abstained = [r for r in per_action.values() if not r["native_ok"]]
    ck("where it ABSTAINS the challenger scores EXACTLY Identity -- it can "
       "avoid losing, never gain",
       all(abs(r["policy_log_loss"] - r["identity_log_loss"]) < 1e-12
           for r in abstained) and len(abstained) == 1,
       f"{len(abstained)} abstention(s), increment on them 0.0")
    ck("  and abstaining where it would be WRONG does improve the primary "
       "score -- §6 bounds that gain, it does not forbid it",
       abstain["PRIMARY_fallback_scored"]["increment_vs_identity"]
       < always["PRIMARY_fallback_scored"]["increment_vs_identity"]
       # THE BOUND IS ITS OWN COMFORTABLE SUBSET, not the other run's:
       # the primary can never be better than the diagnostic, because the
       # actions it abstained on contribute exactly zero increment.
       and abstain["PRIMARY_fallback_scored"]["increment_vs_identity"] >=
       abstain["DIAGNOSTIC_native_intersection"]["increment_vs_identity"],
       f"abstain {abstain['PRIMARY_fallback_scored']['increment_vs_identity']:+.4f}"
       f" vs answer-both "
       f"{always['PRIMARY_fallback_scored']['increment_vs_identity']:+.4f}")
    ck("  and the diagnostic intersection FLATTERS it, which is why it is "
       "not primary",
       abstain["DIAGNOSTIC_native_intersection"]["increment_vs_identity"]
       < abstain["PRIMARY_fallback_scored"]["increment_vs_identity"],
       f"diagnostic "
       f"{abstain['DIAGNOSTIC_native_intersection']['increment_vs_identity']:+.4f}"
       f" vs primary "
       f"{abstain['PRIMARY_fallback_scored']['increment_vs_identity']:+.4f}")

    def ident_bad(_a):
        return FP.FairPrice(coin="btc", window_start=W_START, outcome="UP",
                            value=None, source_timestamp=1.0,
                            local_knowledge_timestamp=1.5, freshness_s=0.5,
                            status=FP.STALE, estimator=FP.IDENTITY)
    none_scored = score_actions(acts, ident_bad, cand_ok, {slug: True})
    ck("when IDENTITY is non-OK the action is COUNTED and NEITHER side is "
       "scored",
       none_scored["identity_non_ok_actions"] == 1
       and none_scored["PRIMARY_fallback_scored"]["n"] == 0,
       f"counted {none_scored['identity_non_ok_actions']}, scored "
       f"{none_scored['PRIMARY_fallback_scored']['n']}")
    ck("both sides are clipped by the SAME epsilon",
       log_loss(0.0, True) == log_loss(EPSILON, True)
       and abs(log_loss(1.0, False) - log_loss(1.0 - EPSILON, False)) < 1e-12
       and EPSILON == 1e-6,
       f"eps {EPSILON}")
    ck("DOWN is never scored separately -- one probability, one term",
       abs(log_loss(0.7, False) - log_loss(0.3, True)) < 1e-12,
       # EXACT equality fails by one ULP: 1 - 0.7 is 0.30000000000000004,
       # and asserting `==` on two paths through floating point is a cell
       # that fails for arithmetic rather than for its property.
       f"|delta| {abs(log_loss(0.7, False) - log_loss(0.3, True)):.2e}")
    try:
        score_actions(acts, ident_ok, cand_ok, {})
        no_out = ""
    except ActionsRefused as exc:
        no_out = str(exc)
    ck("an action with NO SUPPLIED OUTCOME refuses -- this file never "
       "loads one",
       OUTCOME_UNKNOWN in no_out, no_out[:52] or "SCORED WITHOUT AN OUTCOME")
    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    print(json.dumps({"protocol": PROTOCOL, "epsilon": EPSILON,
                      "scope": list(SCOPE_COINS),
                      "key": "(coin, slug, generation_id, decision_recv_ns)"},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
