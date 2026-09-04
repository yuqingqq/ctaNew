"""Offline stateful cancel/hold/repost state machine -- TODO Phase 3 replay.

SURFACE AUTHORISATION (R-126, in-file): R-162(4), coordinator DE hold,
STATEFUL_HARMFUL_CANCEL_TODO.md Phase 3 (section 6). RESEARCH-ONLY, OFFLINE:
no live-trading semantics, no exchange calls, no order/cancel/venue port.
This file is a library plus its parity battery; the only entry point is
--selftest. Real-data wiring is deliverable 5 (harmful_stateful_comparison),
not this file.

WHAT IT CONSUMES.  Three inputs, none optional:
  1. a REFERENCE TRAJECTORY: the QR_SKEW_ONLY no-cancel shadow, generation-
     native -- per (slug, side) an ordered list of generations, each with its
     interval, level, displayed size, status, and fill tranches valued at
     their own times (rule 3: timestamps come from the events that carry
     them).  The skew rules' placement choices live entirely in this input;
     the predictor NEVER chooses placement (TODO section 2.2).
  2. a SCORE STREAM: timestamped harm scores per (slug, side).  Scores are
     computed on the no-cancel shadow (candidate-independent state) and are
     consumed unchanged -- the policy overlay cannot feed itself.
  3. DECLARED PARAMETERS.  Every threshold, latency, dwell, cost, protection
     mode and rate limit is an input with NO default that encodes a policy
     choice; an undeclared parameter is REFUSED (UndeclaredParameter), and
     theta_repost >= theta_cancel is REFUSED (InvalidParameter).  The single
     spec-sanctioned default is enable_reduce=False ("explicit ablation flag,
     default OFF", TODO section 6).

STATE MACHINE per (slug, side, generation):

    LIVE -> CANCEL_PENDING -> HELD -> REPOST_ELIGIBLE -> LIVE

  * first score crossing >= theta_cancel issues ONE simulated cancel for the
    generation; later crossings are counted, never double-acted;
  * CANCEL_PENDING: fills with t < t_request + latency stay CHARGED as stale
    (the v1 zero-latency defect is structurally impossible: latency <= 0 is
    refused at construction);
  * cancel effective strictly before reference generation end removes the
    order (fills at t >= t_effective are PREVENTED) and enters HELD; a cancel
    whose effectiveness lands at/after the generation's own end resolves
    STALE and removes nothing;
  * HELD: repost only after the score has been < theta_repost continuously
    for repost_dwell_s (score is a step function between its own events; the
    below-clock anchors at the score event that went below);
  * REPOST_ELIGIBLE -> LIVE: the join takes the CURRENT reference
    generation's side/level/size -- ordinary skew rules, never the predictor.
    A fresh policy generation gets a fresh queue position and a fresh cancel
    entitlement, and incurs the declared queue-reset cost (whether a repost
    landing exactly on a reference generation start is charged is itself a
    declared parameter, charge_reset_cost_at_generation_start -- the spec is
    ambiguous there and this module refuses to guess).

INVENTORY IS PER-SLUG (v2 changelog 2026-08-27; authorization R-184 step
(vii), coordinator DE hold).  Each 5-minute slug is an independent binary
market that settles at expiry, so the net-inventory state feeding
REDUCING_SIDE_PROTECTION (and reduce suppression) RESETS TO FLAT at every
slug boundary: no slug's protection decision reads another slug's net.
The v1 wiring allocated ONE inventory dict and reused it across every
slug (found by user audit, verified by a two-market falsifier: a SELL
cancel in market 2 was suppressed solely because market 1 left positive
inventory, while market 2 replayed alone issued it).  RESET-to-flat, not
settle-and-carry, is the chosen semantics -- each binary settles at
expiry, and this module carries no settlement-value model (if one is ever
added it must stay reporting-only).  The result's top-level "inventory"
block is a reporting-only cross-slug aggregate (sum of per-slug terminal
nets, max of per-slug peaks) beside a per_slug breakdown; no decision
reads it.  Selftest group O keeps the falsifier permanent: the per-slug
arm must issue the market-2 cancel, a reconstructed v1 global-inventory
arm must suppress it (rule 15, both arms), and single-market replays must
be bit-identical under the two wirings.

DETERMINISM / EVENT ORDERING (declared, load-bearing):
  * merged per-slug stream sorted by (t, rank, side, input_seq) with ranks
    GEN_START=0 < FILL=1 < GEN_END=2 < SCORE=3: at one timestamp a placement
    precedes its fills, fills precede the generation's end, and scores act
    on the post-turnover state.  Ties across sides break BUY_UP before
    SELL_UP.  Fills carry their generation EXPLICITLY, so attribution never
    depends on this ordering -- only bookkeeping does.
  * cancel/reduce effectiveness times are derived (t_request + latency) and
    settle lazily at the next processed event; trajectory events carry their
    own semantic time, so the list is in deterministic processing order and
    each event's `t` is its true time.
  * with latency > 0 (enforced) a fill simultaneous with its own cancel
    request is inside the latency window and stays charged regardless of
    tie-break.

STRUCTURALLY INCAPABLE OF EMITTING TRAINING ROWS.  There is no output path
that carries features or scores: trajectory events and per-cancel records
hold times, identities, shares and markout values only (EVENT_KEYS is the
closed schema; check_invariants refuses any event with extra keys), and the
one function whose name invites it, emit_training_rows(), unconditionally
raises TrainingPopulationRefused.  A policy-generated trajectory therefore
cannot be re-used as its own training population through this module
(TODO section 6, last parity item; reliability rule 1).

NO VERDICT STRINGS.  Every claimed property is a computed predicate
(check_invariants) or a number beside its n; nothing prints PASS/FAIL prose
next to a table (rule 10).  All exclusions and edge cases are counted
statuses, never silent drops (rule 4).

    python3 live/pm_research/harmful_stateful_policy.py --selftest

Selftest: 89 checks (EXPECTED_CHECKS below; the run asserts the count, so
the claim is computed at run time, not remembered here).
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

EXPECTED_CHECKS = 90          # asserted by selftest(); update together

SIDES = ("BUY_UP", "SELL_UP")
RANK_GEN_START, RANK_FILL, RANK_GEN_END, RANK_SCORE = 0, 1, 2, 3
PROTECTION_MODES = ("REDUCING_SIDE_PROTECTION", "ALL_ORDERS_OVERRIDE")
REPOST_FILL_MODELS = ("REFERENCE_FILLS", "NO_FILLS_UNTIL_NEXT_GENERATION")
OK = "OK"
EPS = 1e-9
RATE_WINDOW_S = 60.0          # rolling window of the per-slug cancel limiter


class UndeclaredParameter(ValueError):
    """A required policy parameter was not declared by the caller."""


class InvalidParameter(ValueError):
    """A declared parameter violates its contract (e.g. theta_repost >=
    theta_cancel, non-positive latency, NaN threshold)."""


class ReferenceIntegrityError(ValueError):
    """The reference trajectory input is malformed (overlapping generations,
    tranche outside its generation, shares exceeding displayed, ...)."""


class TrajectoryIntegrityError(ValueError):
    """A trajectory handed to the invariant checker is malformed -- unknown
    event kind, keys outside the closed schema, or an event referencing a
    generation that never existed.  Refusal, not a False predicate."""


class VacuousParity(ValueError):
    """A parity comparison was attempted on empty trajectories.  A zero from
    an instrument that never proved it can fire is not a result (rule 15)."""


class TrainingPopulationRefused(RuntimeError):
    """This module cannot emit training rows.  See the header."""


def emit_training_rows(*_args: Any, **_kwargs: Any) -> None:
    """UNCONDITIONAL REFUSAL -- the executable form of the ban.

    No policy-generated trajectory may be reused as its own training
    population (TODO section 6; reliability rule 1: never train on an
    outcome-selected population).  The tombstone exists so the ban has a
    falsifier: the selftest calls it and requires the refusal to fire."""
    raise TrainingPopulationRefused(
        "harmful_stateful_policy emits NO training rows, by construction. "
        "Training populations come from the no-cancel reference trajectory "
        "(harmful_exposure_rows), never from a policy-generated one.")


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS = (
    "predictor_enabled",
    "theta_cancel",
    "theta_repost",
    "repost_dwell_s",
    "cancel_effective_latency_ms",
    "queue_reset_cost_cents",
    "protection_mode",
    "max_cancels_per_minute",
    "repost_fill_model",
    "charge_reset_cost_at_generation_start",
)
_OPTIONAL_PARAMS = ("enable_reduce", "theta_reduce", "reduce_remaining_fraction")


def _num(p: dict, key: str, *, allow_nan: bool = False) -> float:
    v = p[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidParameter(f"{key} must be a number, got {v!r}")
    v = float(v)
    if not allow_nan and math.isnan(v):
        raise InvalidParameter(f"{key} is NaN; a NaN threshold compares False "
                               f"everywhere and would be a silent no-op")
    return v


def validate_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize and REFUSE.  Every policy-encoding constant is an input;
    absence is UndeclaredParameter, contract violation is InvalidParameter.
    Returns a normalized copy; the input dict is never mutated."""
    if not isinstance(params, dict):
        raise InvalidParameter("params must be a dict of declared parameters")
    unknown = set(params) - set(_REQUIRED_PARAMS) - set(_OPTIONAL_PARAMS)
    if unknown:
        raise InvalidParameter(
            f"unknown parameter(s) {sorted(unknown)}; a typo here would be a "
            f"silently ignored declaration, which is worse than a refusal")
    missing = [k for k in _REQUIRED_PARAMS if k not in params]
    if missing:
        raise UndeclaredParameter(
            f"undeclared parameter(s) {missing}: thresholds, latency, dwell, "
            f"queue-reset cost, protection mode, rate limit and repost fill "
            f"model are POLICY CHOICES and must be declared by the caller "
            f"(TODO section 6; house rule: no defaults that encode policy)")
    p: dict[str, Any] = {}
    for key in ("predictor_enabled", "charge_reset_cost_at_generation_start"):
        if not isinstance(params[key], bool):
            raise InvalidParameter(f"{key} must be a bool, got "
                                   f"{params[key]!r}")
        p[key] = params[key]
    p["theta_cancel"] = _num(params, "theta_cancel")
    p["theta_repost"] = _num(params, "theta_repost")
    if not p["theta_repost"] < p["theta_cancel"]:
        raise InvalidParameter(
            f"theta_repost ({p['theta_repost']}) must be strictly below "
            f"theta_cancel ({p['theta_cancel']}): without hysteresis the "
            f"machine would repost into the state it just cancelled")
    p["repost_dwell_s"] = _num(params, "repost_dwell_s")
    if p["repost_dwell_s"] < 0.0:
        raise InvalidParameter("repost_dwell_s must be >= 0")
    p["cancel_effective_latency_ms"] = _num(
        params, "cancel_effective_latency_ms")
    if (not math.isfinite(p["cancel_effective_latency_ms"])
            or p["cancel_effective_latency_ms"] <= 0.0):
        raise InvalidParameter(
            "cancel_effective_latency_ms must be finite and > 0: "
            "zero-latency cancellation was the named v1 evaluator defect "
            "and is structurally refused here")
    p["queue_reset_cost_cents"] = _num(params, "queue_reset_cost_cents")
    if (not math.isfinite(p["queue_reset_cost_cents"])
            or p["queue_reset_cost_cents"] < 0.0):
        raise InvalidParameter("queue_reset_cost_cents must be finite, >= 0")
    if params["protection_mode"] not in PROTECTION_MODES:
        raise InvalidParameter(
            f"protection_mode must be one of {PROTECTION_MODES}, got "
            f"{params['protection_mode']!r} -- the two protection cells are "
            f"explicit configuration, not behavior")
    p["protection_mode"] = params["protection_mode"]
    p["max_cancels_per_minute"] = _num(params, "max_cancels_per_minute")
    if p["max_cancels_per_minute"] <= 0.0:
        raise InvalidParameter("max_cancels_per_minute must be > 0 "
                               "(float('inf') is an acceptable explicit "
                               "declaration of 'unlimited')")
    if params["repost_fill_model"] not in REPOST_FILL_MODELS:
        raise InvalidParameter(
            f"repost_fill_model must be one of {REPOST_FILL_MODELS}, got "
            f"{params['repost_fill_model']!r}: whether a mid-generation "
            f"repost receives the reference tranches is a modelling choice "
            f"this module refuses to default")
    p["repost_fill_model"] = params["repost_fill_model"]
    p["enable_reduce"] = params.get("enable_reduce", False)
    if not isinstance(p["enable_reduce"], bool):
        raise InvalidParameter("enable_reduce must be a bool")
    if p["enable_reduce"]:
        for key in ("theta_reduce", "reduce_remaining_fraction"):
            if key not in params:
                raise UndeclaredParameter(
                    f"enable_reduce=True requires {key} to be declared")
        p["theta_reduce"] = _num(params, "theta_reduce")
        if not p["theta_reduce"] < p["theta_cancel"]:
            raise InvalidParameter(
                "theta_reduce must be strictly below theta_cancel "
                "(TODO section 6: reduce happens BEFORE a full cancel)")
        p["reduce_remaining_fraction"] = _num(
            params, "reduce_remaining_fraction")
        if not 0.0 < p["reduce_remaining_fraction"] < 1.0:
            raise InvalidParameter(
                "reduce_remaining_fraction must be in (0, 1) exclusive: 0 is "
                "a cancel and 1 is a no-op, both misdeclared as a reduce")
    else:
        for key in ("theta_reduce", "reduce_remaining_fraction"):
            if key in params:
                raise InvalidParameter(
                    f"{key} declared while enable_reduce is False/absent -- "
                    f"an inert declaration is ambiguous; refuse, not ignore")
    return p


# ---------------------------------------------------------------------------
# reference trajectory + score stream validation
# ---------------------------------------------------------------------------

_GEN_REQUIRED = ("gen", "t0", "t1", "level", "displayed", "status", "tranches")
_TRANCHE_REQUIRED = ("t", "shares", "markout_cents_per_share")


def _fin(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def validate_reference(reference: dict[str, Any]) -> None:
    """Refuse malformed reference input.  Extra keys on generation/tranche
    records are permitted and NEVER read (real exposure-pipeline records
    carry more fields); missing required keys, overlapping generations,
    out-of-interval tranches, shares exceeding displayed, or an OK
    generation with a missing markout are refused loudly."""
    if not isinstance(reference, dict) or not reference:
        raise ReferenceIntegrityError("reference must be a non-empty dict "
                                      "slug -> {side: [generations]}")
    for slug, sides in reference.items():
        if not isinstance(sides, dict) or set(sides) != set(SIDES):
            raise ReferenceIntegrityError(
                f"{slug}: reference[slug] must have exactly the side keys "
                f"{SIDES} (empty lists are fine); got "
                f"{sorted(sides) if isinstance(sides, dict) else sides!r}")
        for side in SIDES:
            gens = sides[side]
            if not isinstance(gens, list):
                raise ReferenceIntegrityError(f"{slug}/{side}: not a list")
            prev_t1 = -math.inf
            prev_gen = None
            for g in gens:
                for key in _GEN_REQUIRED:
                    if key not in g:
                        raise ReferenceIntegrityError(
                            f"{slug}/{side}: generation missing {key!r}")
                if not isinstance(g["gen"], int) or isinstance(g["gen"], bool):
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}: gen id must be an int")
                if prev_gen is not None and g["gen"] <= prev_gen:
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}: gen ids must strictly increase")
                if not (_fin(g["t0"]) and _fin(g["t1"])
                        and g["t0"] < g["t1"]):
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}/gen {g['gen']}: need finite t0 < t1")
                if g["t0"] < prev_t1 - EPS:
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}/gen {g['gen']}: overlaps previous "
                        f"generation (t0 {g['t0']} < prev t1 {prev_t1})")
                if not (_fin(g["level"]) and _fin(g["displayed"])
                        and g["displayed"] > 0.0):
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}/gen {g['gen']}: level must be finite "
                        f"and displayed finite > 0")
                if not isinstance(g["status"], str):
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}/gen {g['gen']}: status must be a str")
                total = 0.0
                for tr in g["tranches"]:
                    for key in _TRANCHE_REQUIRED:
                        if key not in tr:
                            raise ReferenceIntegrityError(
                                f"{slug}/{side}/gen {g['gen']}: tranche "
                                f"missing {key!r}")
                    if not (_fin(tr["t"])
                            and g["t0"] - EPS <= tr["t"] <= g["t1"] + EPS):
                        raise ReferenceIntegrityError(
                            f"{slug}/{side}/gen {g['gen']}: tranche time "
                            f"{tr['t']!r} outside [{g['t0']}, {g['t1']}]")
                    if not (_fin(tr["shares"]) and tr["shares"] > 0.0):
                        raise ReferenceIntegrityError(
                            f"{slug}/{side}/gen {g['gen']}: tranche shares "
                            f"must be finite > 0")
                    mo = tr["markout_cents_per_share"]
                    if g["status"] == OK:
                        if not _fin(mo):
                            raise ReferenceIntegrityError(
                                f"{slug}/{side}/gen {g['gen']}: OK "
                                f"generation with non-finite markout {mo!r} "
                                f"-- unvalued fills belong under a non-OK "
                                f"status, never inside OK")
                    elif mo is not None and not _fin(mo):
                        raise ReferenceIntegrityError(
                            f"{slug}/{side}/gen {g['gen']}: markout must be "
                            f"finite or None")
                    total += tr["shares"]
                if total > g["displayed"] + EPS:
                    raise ReferenceIntegrityError(
                        f"{slug}/{side}/gen {g['gen']}: tranche shares "
                        f"{total} exceed displayed {g['displayed']}")
                prev_t1 = g["t1"]
                prev_gen = g["gen"]


def validate_scores(scores: Sequence[dict[str, Any]]) -> None:
    """Refuse malformed score events.  A NaN score compares False against
    every threshold -- a silent semantic drop -- so it is refused, not
    counted.  An unknown slug is a legitimate mismatch and is COUNTED at
    replay time instead (rule 4)."""
    for i, s in enumerate(scores):
        if not isinstance(s, dict):
            raise ReferenceIntegrityError(f"score[{i}]: not a dict")
        for key in ("t", "slug", "side", "score"):
            if key not in s:
                raise ReferenceIntegrityError(f"score[{i}]: missing {key!r}")
        if not _fin(s["t"]):
            raise ReferenceIntegrityError(f"score[{i}]: non-finite t")
        if s["side"] not in SIDES:
            raise ReferenceIntegrityError(
                f"score[{i}]: side {s['side']!r} not in {SIDES}")
        v = s["score"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or math.isnan(v):
            raise ReferenceIntegrityError(
                f"score[{i}]: score {v!r} refused (NaN/non-number would "
                f"compare False everywhere and become a silent no-op)")


# ---------------------------------------------------------------------------
# trajectory event schema (closed) and emit helpers
# ---------------------------------------------------------------------------

EVENT_KEYS: dict[str, frozenset] = {
    "PLACE": frozenset({"kind", "t", "slug", "side", "ref_gen", "policy_gen",
                        "level", "displayed", "source"}),
    "FILL_CHARGED": frozenset({"kind", "t", "slug", "side", "ref_gen",
                               "policy_gen", "shares", "stale",
                               "markout_cents_per_share"}),
    "FILL_PREVENTED": frozenset({"kind", "t", "slug", "side", "ref_gen",
                                 "policy_gen", "shares",
                                 "markout_cents_per_share",
                                 "cancel_t_effective"}),
    "FILL_PREVENTED_REDUCED": frozenset({"kind", "t", "slug", "side",
                                         "ref_gen", "policy_gen", "shares",
                                         "markout_cents_per_share",
                                         "reduce_t_effective"}),
    "FILL_MISSED_HELD": frozenset({"kind", "t", "slug", "side", "ref_gen",
                                   "shares", "markout_cents_per_share"}),
    "FILL_MISSED_POST_REPOST": frozenset({"kind", "t", "slug", "side",
                                          "ref_gen", "policy_gen", "shares",
                                          "markout_cents_per_share"}),
    "GEN_END": frozenset({"kind", "t", "slug", "side", "ref_gen",
                          "policy_gen", "reason"}),
    "GEN_START_MISSED_HELD": frozenset({"kind", "t", "slug", "side",
                                        "ref_gen"}),
    "CANCEL_ISSUED": frozenset({"kind", "t", "slug", "side", "ref_gen",
                                "policy_gen", "t_effective",
                                "reducing_at_request"}),
    "CANCEL_EFFECTIVE": frozenset({"kind", "t", "slug", "side", "ref_gen",
                                   "policy_gen"}),
    "CANCEL_STALE": frozenset({"kind", "t", "slug", "side", "ref_gen",
                               "policy_gen", "reference_end_t"}),
    "REDUCE_ISSUED": frozenset({"kind", "t", "slug", "side", "ref_gen",
                                "policy_gen", "t_effective"}),
    "REDUCE_EFFECTIVE": frozenset({"kind", "t", "slug", "side", "ref_gen",
                                   "policy_gen", "remaining_after"}),
    "HOLD_START": frozenset({"kind", "t", "slug", "side", "ref_gen",
                             "policy_gen"}),
    "REPOST": frozenset({"kind", "t", "slug", "side", "ref_gen",
                         "policy_gen", "source", "cost_charged_cents"}),
    "REPOST_ELIGIBLE": frozenset({"kind", "t", "slug", "side"}),
    "REPOST_ELIGIBILITY_REVOKED": frozenset({"kind", "t", "slug", "side"}),
}


def _ev_place(slug: str, side: str, g: dict, policy_gen: str, t: float,
              source: str) -> dict:
    return {"kind": "PLACE", "t": t, "slug": slug, "side": side,
            "ref_gen": g["gen"], "policy_gen": policy_gen,
            "level": g["level"], "displayed": g["displayed"],
            "source": source}


def _ev_fill_charged(slug: str, side: str, g: dict, policy_gen: str,
                     t: float, shares: float, stale: bool,
                     markout: Any) -> dict:
    return {"kind": "FILL_CHARGED", "t": t, "slug": slug, "side": side,
            "ref_gen": g["gen"], "policy_gen": policy_gen, "shares": shares,
            "stale": stale, "markout_cents_per_share": markout}


def _ev_gen_end(slug: str, side: str, g: dict, policy_gen: str, t: float,
                reason: str) -> dict:
    return {"kind": "GEN_END", "t": t, "slug": slug, "side": side,
            "ref_gen": g["gen"], "policy_gen": policy_gen, "reason": reason}


def canonical_trajectory(events: Sequence[dict]) -> str:
    """Byte-stable serialization for bit-parity: one sorted-key JSON line per
    event, in processing order.  allow_nan=False makes a NaN in any emitted
    field a refusal rather than a silently unequal artifact."""
    return "\n".join(
        json.dumps(e, sort_keys=True, allow_nan=False) for e in events)


def bit_identical(events_a: Sequence[dict], events_b: Sequence[dict]) -> bool:
    """True iff the two trajectories serialize to identical bytes.  REFUSES
    a vacuous comparison: two empty trajectories prove nothing about the
    machine (rule 15 -- the instrument must be able to fire)."""
    if not events_a and not events_b:
        raise VacuousParity(
            "both trajectories are empty; a parity pass on nothing is not "
            "evidence -- feed a fixture with at least one event")
    return canonical_trajectory(events_a) == canonical_trajectory(events_b)


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------

class _SideRun:
    """Mutable per-(slug, side) policy state.  The generation-scoped state
    (LIVE / CANCEL_PENDING / CANCELLED / ENDED) lives on the policy-
    generation records; held/eligibility is side-scoped because HELD means
    'no order on this side at all'."""

    __slots__ = ("held", "hold_start", "holds", "below_since", "was_eligible",
                 "current_ref", "current_pol", "pending", "repost_seq",
                 "recs_by_gen", "missed_gens")

    def __init__(self) -> None:
        self.held = False
        self.hold_start: float | None = None
        self.holds: list[dict] = []
        self.below_since: float | None = None
        self.was_eligible = False
        self.current_ref: dict | None = None
        self.current_pol: dict | None = None
        self.pending: dict | None = None       # rec with an unresolved cancel
        self.repost_seq = 0
        # Generations whose START was missed because this side was HELD.  A
        # GENERATION-LEVEL fact, deliberately not derived from the side's
        # CURRENT held flag: the two disagree at a generation boundary, and
        # the disagreement is what real data found (see _on_fill).
        self.missed_gens: set = set()
        self.recs_by_gen: dict[int, list[dict]] = {}


def _merged_events(sides: dict[str, list], slug_scores: list) -> list:
    """(t, rank, side_idx, seq, kind, side, payload) sorted deterministically.
    seq is the assembly index, so equal (t, rank, side) ties keep input
    order.  Scores keep their stream order among themselves."""
    evs = []
    seq = 0
    for side_idx, side in enumerate(SIDES):
        for g in sides[side]:
            evs.append((g["t0"], RANK_GEN_START, side_idx, seq,
                        "start", side, g)); seq += 1
            for tr in g["tranches"]:
                evs.append((tr["t"], RANK_FILL, side_idx, seq,
                            "fill", side, (g, tr))); seq += 1
            evs.append((g["t1"], RANK_GEN_END, side_idx, seq,
                        "end", side, g)); seq += 1
    for s in slug_scores:
        evs.append((s["t"], RANK_SCORE, SIDES.index(s["side"]), seq,
                    "score", s["side"], s)); seq += 1
    evs.sort(key=lambda e: (e[0], e[1], e[2], e[3]))
    return evs


def _is_reducing(side: str, net: float) -> bool:
    """Inventory-reducing = the fill direction shrinks |net|.  net == 0 means
    NEITHER side reduces (both grow |net|), so it classifies as increasing."""
    return (net > EPS and side == "SELL_UP") or (net < -EPS
                                                 and side == "BUY_UP")


class _SlugReplay:
    """One slug's replay.  Everything is deterministic in the inputs."""

    def __init__(self, slug: str, sides: dict[str, list], slug_scores: list,
                 p: dict, counters: dict, trajectory: list, cancels: list,
                 econ: dict, inv: dict) -> None:
        self.slug = slug
        self.p = p
        self.c = counters
        self.traj = trajectory
        self.cancel_records = cancels
        self.econ = econ
        self.inv = inv          # THIS slug's inventory only (fresh per slug)
        self.sides_ref = sides
        self.scores = slug_scores
        self.side_runs = {s: _SideRun() for s in SIDES}
        self.issued_times: list[float] = []       # per-slug rate window
        self.latency_s = p["cancel_effective_latency_ms"] / 1000.0

    # -- accounting helpers -------------------------------------------------

    def _bucket(self, name: str, shares: float, markout: Any) -> None:
        b = self.econ["not_received"][name]
        b["shares"] += shares
        if markout is None:
            b["unvalued_shares"] += shares
            return
        v = -markout * shares          # dataset sign: >0 == harm avoided
        b["value_cents"] += v
        if v > 0:
            b["harm_avoided_cents"] += v
        else:
            b["sacrifice_cents"] += -v

    def _charge_fill(self, side: str, g: dict, rec: dict, t: float,
                     shares: float, stale: bool, markout: Any) -> None:
        f = self.econ["fills"]
        f["received_shares"] += shares
        if markout is None:
            f["received_unvalued_shares"] += shares
        else:
            f["received_markout_cents"] += markout * shares
        if stale:
            f["stale_shares"] += shares
            if markout is not None:
                f["stale_markout_cents"] += markout * shares
            rec["stale_shares"] += shares
            if markout is not None:
                rec["stale_value_cents"] += markout * shares
        net0 = self.inv["net"]
        net1 = net0 + (shares if side == "BUY_UP" else -shares)
        self.inv["net"] = net1
        self.inv["peak_abs_net"] = max(self.inv["peak_abs_net"], abs(net1))
        if abs(net1) < abs(net0) - EPS:
            self.inv["received_reducing_shares"] += shares
        else:
            self.inv["received_increasing_shares"] += shares
        self.traj.append(_ev_fill_charged(
            self.slug, side, g, rec["policy_gen"], t, shares, stale, markout))

    # -- lazy effectiveness -------------------------------------------------

    def _settle(self, rec: dict | None, t: float) -> None:
        """Apply reduce/cancel effectiveness times that have been reached.
        Reduce and cancel share one declared latency, so effectiveness order
        equals request order; reduce first when both are due."""
        if rec is None:
            return
        rd = rec["reduce"]
        if (rd is not None and not rd["applied"] and rd["t_effective"] <= t
                and rec["state"] in ("LIVE", "CANCEL_PENDING")
                and rd["t_effective"] < rec["ref"]["t1"]):
            rd["applied"] = True
            rec["remaining"] = rec["remaining"] * self.p[
                "reduce_remaining_fraction"]
            self.c["reduce_effective"] += 1
            self.traj.append({
                "kind": "REDUCE_EFFECTIVE", "t": rd["t_effective"],
                "slug": self.slug, "side": rec["side"],
                "ref_gen": rec["ref"]["gen"],
                "policy_gen": rec["policy_gen"],
                "remaining_after": rec["remaining"]})
        cn = rec["cancel"]
        if (cn is not None and rec["state"] == "CANCEL_PENDING"
                and cn["t_effective"] <= t
                and cn["t_effective"] < rec["ref"]["t1"]):
            rec["state"] = "CANCELLED"
            cn["outcome"] = "EFFECTIVE"
            self.c["cancels_effective"] += 1
            run = self.side_runs[rec["side"]]
            if run.pending is rec:
                run.pending = None
            if run.current_pol is rec:
                run.current_pol = None
            if rd is not None and not rd["applied"]:
                rd["superseded"] = True
                self.c["reduce_superseded_by_cancel"] += 1
            self.traj.append({
                "kind": "CANCEL_EFFECTIVE", "t": cn["t_effective"],
                "slug": self.slug, "side": rec["side"],
                "ref_gen": rec["ref"]["gen"],
                "policy_gen": rec["policy_gen"]})
            self.traj.append(_ev_gen_end(
                self.slug, rec["side"], rec["ref"], rec["policy_gen"],
                cn["t_effective"], "CANCELLED"))
            run.held = True
            run.hold_start = cn["t_effective"]
            run.was_eligible = False
            run.holds.append({"side": rec["side"],
                              "start": cn["t_effective"], "end": None,
                              "cancelled_policy_gen": rec["policy_gen"]})
            self.c["hold_entries"] += 1
            self.traj.append({
                "kind": "HOLD_START", "t": cn["t_effective"],
                "slug": self.slug, "side": rec["side"],
                "ref_gen": rec["ref"]["gen"],
                "policy_gen": rec["policy_gen"]})

    # -- joins / reposts ----------------------------------------------------

    def _join(self, side: str, g: dict, t: float, source: str) -> dict:
        run = self.side_runs[side]
        if source == "TRACKING":
            policy_gen = str(g["gen"])
        else:
            run.repost_seq += 1
            policy_gen = f"{g['gen']}.r{run.repost_seq}"
        rec = {"slug": self.slug, "side": side, "ref": g,
               "policy_gen": policy_gen, "state": "LIVE", "joined_at": t,
               "source": source, "remaining": g["displayed"],
               "no_fills": (source == "REPOST_MID_GENERATION"
                            and self.p["repost_fill_model"]
                            == "NO_FILLS_UNTIL_NEXT_GENERATION"),
               "non_ok": g["status"] != OK,
               "cancel": None, "reduce": None,
               "prevented_shares": 0.0, "prevented_value_cents": 0.0,
               "stale_shares": 0.0, "stale_value_cents": 0.0}
        run.recs_by_gen.setdefault(g["gen"], []).append(rec)
        run.current_pol = rec
        if source != "TRACKING":
            run.held = False
            run.was_eligible = False
            for h in run.holds:
                if h["end"] is None:
                    h["end"] = t
            run.hold_start = None
            cost = self.p["queue_reset_cost_cents"]
            if (source == "REPOST_AT_GEN_START"
                    and not self.p["charge_reset_cost_at_generation_start"]):
                cost = 0.0
            self.econ["queue_reset_cost_cents_total"] += cost
            self.c["reposts"] += 1
            self.c["reposts_mid_generation" if source ==
                   "REPOST_MID_GENERATION" else
                   "reposts_at_generation_start"] += 1
            self.traj.append({
                "kind": "REPOST", "t": t, "slug": self.slug, "side": side,
                "ref_gen": g["gen"], "policy_gen": policy_gen,
                "source": source, "cost_charged_cents": cost})
        self.traj.append(_ev_place(self.slug, side, g, policy_gen, t, source))
        return rec

    def _repost_check(self, side: str, t: float) -> None:
        """HELD -> REPOST_ELIGIBLE -> LIVE, evaluated at event times only.
        Eligibility = the score has been < theta_repost continuously since
        below_since (step semantics) for repost_dwell_s, and never before
        the hold began."""
        run = self.side_runs[side]
        if not run.held or run.below_since is None:
            return
        eligible_at = max(run.hold_start,
                          run.below_since + self.p["repost_dwell_s"])
        if t < eligible_at or math.isinf(eligible_at):
            return
        if not run.was_eligible:
            run.was_eligible = True
            self.c["repost_eligible_transitions"] += 1
            self.traj.append({"kind": "REPOST_ELIGIBLE", "t": eligible_at,
                              "slug": self.slug, "side": side})
        if run.current_ref is not None:
            self._join(side, run.current_ref, t, "REPOST_MID_GENERATION")

    # -- event handlers -----------------------------------------------------

    def _on_gen_start(self, side: str, g: dict, t: float) -> None:
        run = self.side_runs[side]
        # SETTLE FIRST.  The header declares that derived effectiveness times
        # "settle lazily at the next processed event"; a generation start IS
        # a processed event, and this was the one handler that read `held`
        # without settling.  Real data made it routine rather than exotic:
        # consecutive generations abut (t1 of N == t0 of N+1) and GEN_START
        # outranks GEN_END at equal times, so N+1's start is processed BEFORE
        # N's end -- and when no fill or score of this side falls between the
        # cancel's effectiveness and N's end, nothing had settled it.  The
        # side then PLACED on N+1 during what should have been a hold, and
        # every subsequent fill of N+1 was CHARGED to a policy that should
        # have had no order there.  check_invariants passed throughout: the
        # trajectory was internally consistent and economically wrong.
        self._settle(run.pending, t)
        run.current_ref = g
        if g["status"] != OK:
            self.c["non_ok_generations"] += 1
        if not run.held:
            self._join(side, g, t, "TRACKING")
            return
        eligible = (run.below_since is not None
                    and t >= max(run.hold_start,
                                 run.below_since + self.p["repost_dwell_s"]))
        if eligible:
            if not run.was_eligible:
                run.was_eligible = True
                self.c["repost_eligible_transitions"] += 1
                self.traj.append({
                    "kind": "REPOST_ELIGIBLE",
                    "t": max(run.hold_start,
                             run.below_since + self.p["repost_dwell_s"]),
                    "slug": self.slug, "side": side})
            self._join(side, g, t, "REPOST_AT_GEN_START")
        else:
            self.c["gen_starts_missed_held"] += 1
            run.missed_gens.add(g["gen"])
            self.traj.append({"kind": "GEN_START_MISSED_HELD", "t": t,
                              "slug": self.slug, "side": side,
                              "ref_gen": g["gen"]})

    def _on_fill(self, side: str, g: dict, tr: dict, t: float) -> None:
        run = self.side_runs[side]
        self._settle(run.pending, t)
        self._repost_check(side, t)
        self.econ["fills"]["reference_shares"] += tr["shares"]
        recs = run.recs_by_gen.get(g["gen"], [])
        rec = None
        for r in recs:
            if r["joined_at"] <= t + 1e-12:
                rec = r                       # latest join at/before t wins
        markout = tr["markout_cents_per_share"]
        if rec is None:
            # THE LICENCE IS A FACT ABOUT THE GENERATION, NOT ABOUT THE SIDE.
            # Real data (2026-09-01, btc-updown-5m-1787580000/BUY_UP/gen 449)
            # found the disagreement: gen 449's only tranche lands at exactly
            # its own t1, which is also gen 450's t0, and GEN_START outranks
            # FILL at equal times -- so gen 450's start is processed FIRST,
            # reposts (the side was newly eligible) and clears `held` and
            # `was_eligible`.  The fill of 449 then arrives at a side that is
            # no longer held, on a generation that was never joined, and the
            # guard fired on a case that is entirely correct: 449 WAS missed
            # while held.  The side flag is one event stale by construction;
            # `missed_gens` records the fact at the unit it belongs to.
            # The guard is NOT weakened -- a fill on a generation that was
            # neither joined nor missed still raises (selftest group P).
            if not (run.held or run.was_eligible
                    or g["gen"] in run.missed_gens):
                raise RuntimeError(
                    f"{self.slug}/{side}/gen {g['gen']}: fill at {t} has no "
                    f"policy record, the side is not held, and the "
                    f"generation was never missed-while-held -- machine bug")
            self._bucket("missed_while_held", tr["shares"], markout)
            self.traj.append({"kind": "FILL_MISSED_HELD", "t": t,
                              "slug": self.slug, "side": side,
                              "ref_gen": g["gen"], "shares": tr["shares"],
                              "markout_cents_per_share": markout})
            return
        self._settle(rec, t)
        if rec["state"] == "CANCELLED":
            cn = rec["cancel"]
            self._bucket("prevented_after_cancel", tr["shares"], markout)
            rec["prevented_shares"] += tr["shares"]
            if markout is not None:
                rec["prevented_value_cents"] += -markout * tr["shares"]
            self.traj.append({"kind": "FILL_PREVENTED", "t": t,
                              "slug": self.slug, "side": side,
                              "ref_gen": g["gen"],
                              "policy_gen": rec["policy_gen"],
                              "shares": tr["shares"],
                              "markout_cents_per_share": markout,
                              "cancel_t_effective": cn["t_effective"]})
            return
        if rec["no_fills"]:
            self._bucket("missed_post_repost", tr["shares"], markout)
            self.traj.append({"kind": "FILL_MISSED_POST_REPOST", "t": t,
                              "slug": self.slug, "side": side,
                              "ref_gen": g["gen"],
                              "policy_gen": rec["policy_gen"],
                              "shares": tr["shares"],
                              "markout_cents_per_share": markout})
            return
        stale = rec["state"] == "CANCEL_PENDING"
        charge = min(tr["shares"], max(0.0, rec["remaining"]))
        if charge > 0.0:
            rec["remaining"] -= charge
            self._charge_fill(side, g, rec, t, charge, stale, markout)
        cut = tr["shares"] - charge
        if cut > EPS:
            rd = rec["reduce"]
            self._bucket("prevented_reduced", cut, markout)
            self.traj.append({"kind": "FILL_PREVENTED_REDUCED", "t": t,
                              "slug": self.slug, "side": side,
                              "ref_gen": g["gen"],
                              "policy_gen": rec["policy_gen"], "shares": cut,
                              "markout_cents_per_share": markout,
                              "reduce_t_effective":
                                  rd["t_effective"] if rd else None})

    def _on_gen_end(self, side: str, g: dict, t1: float) -> None:
        run = self.side_runs[side]
        for rec in run.recs_by_gen.get(g["gen"], []):
            self._settle(rec, t1)
            rd = rec["reduce"]
            if (rec["state"] in ("LIVE", "CANCEL_PENDING") and rd is not None
                    and not rd["applied"] and not rd.get("superseded")):
                self.c["reduce_stale"] += 1        # landed after the gen died
            if rec["state"] == "LIVE":
                rec["state"] = "ENDED"
                self.traj.append(_ev_gen_end(
                    self.slug, side, g, rec["policy_gen"], t1,
                    "REFERENCE_END"))
            elif rec["state"] == "CANCEL_PENDING":
                # effectiveness lands at/after the reference end: STALE --
                # the reference order died on its own; nothing was removed.
                rec["state"] = "ENDED"
                rec["cancel"]["outcome"] = "STALE"
                self.c["cancels_stale"] += 1
                if run.pending is rec:
                    run.pending = None
                self.traj.append({
                    "kind": "CANCEL_STALE", "t": rec["cancel"]["t_effective"],
                    "slug": self.slug, "side": side, "ref_gen": g["gen"],
                    "policy_gen": rec["policy_gen"], "reference_end_t": t1})
                self.traj.append(_ev_gen_end(
                    self.slug, side, g, rec["policy_gen"], t1,
                    "REFERENCE_END"))
        if run.current_ref is g:
            run.current_ref = None
        if run.current_pol is not None and run.current_pol["ref"] is g:
            run.current_pol = None

    def _on_score(self, side: str, s: dict, t: float) -> None:
        if not self.p["predictor_enabled"]:
            self.c["scores_ignored_predictor_disabled"] += 1
            return
        run = self.side_runs[side]
        self._settle(run.pending, t)
        if run.current_pol is not None and run.current_pol is not run.pending:
            self._settle(run.current_pol, t)
        self._repost_check(side, t)
        sc = float(s["score"])
        # below-clock: a pure function of the score stream (step semantics)
        if sc < self.p["theta_repost"]:
            if run.below_since is None:
                run.below_since = t
        else:
            if run.held and run.was_eligible:
                run.was_eligible = False
                self.c["repost_eligibility_revoked"] += 1
                self.traj.append({"kind": "REPOST_ELIGIBILITY_REVOKED",
                                  "t": t, "slug": self.slug, "side": side})
            run.below_since = None
        crossing = sc >= self.p["theta_cancel"]
        reduce_band = (self.p["enable_reduce"] and not crossing
                       and sc >= self.p["theta_reduce"])
        if not crossing and not reduce_band:
            return
        if run.held:
            if crossing:
                self.c["crossings_while_held"] += 1
            else:
                self.c["reduce_crossings_while_held"] += 1
            return
        rec = run.current_pol
        if rec is None:
            self.c["crossings_while_idle" if crossing
                   else "reduce_crossings_while_idle"] += 1
            return
        if rec["non_ok"]:
            self.c["crossings_on_non_ok_generation"] += 1
            return
        if rec["state"] == "CANCEL_PENDING":
            self.c["crossings_while_pending" if crossing
                   else "reduce_crossings_while_pending"] += 1
            return
        if rec["state"] != "LIVE":
            raise RuntimeError(f"score against a {rec['state']} record -- "
                               f"machine bug")
        reducing = _is_reducing(side, self.inv["net"])
        if crossing:
            if (self.p["protection_mode"] == "REDUCING_SIDE_PROTECTION"
                    and reducing):
                self.c["cancel_suppressed_protected"] += 1
                return
            self.c["cancels_requested"] += 1
            self.issued_times = [x for x in self.issued_times
                                 if x > t - RATE_WINDOW_S]
            if len(self.issued_times) >= self.p["max_cancels_per_minute"]:
                self.c["cancels_suppressed_rate_limited"] += 1
                return
            self.c["cancels_rate_passed"] += 1
            self.issued_times.append(t)
            t_eff = t + self.latency_s
            rec["state"] = "CANCEL_PENDING"
            rec["cancel"] = {"t_request": t, "t_effective": t_eff,
                             "outcome": None,
                             "reducing_at_request": reducing}
            run.pending = rec
            self.c["cancels_issued"] += 1
            self.traj.append({"kind": "CANCEL_ISSUED", "t": t,
                              "slug": self.slug, "side": side,
                              "ref_gen": rec["ref"]["gen"],
                              "policy_gen": rec["policy_gen"],
                              "t_effective": t_eff,
                              "reducing_at_request": reducing})
            return
        # reduce band, state LIVE
        if rec["reduce"] is not None:
            self.c["reduce_crossings_while_reduced"] += 1
            return
        if (self.p["protection_mode"] == "REDUCING_SIDE_PROTECTION"
                and reducing):
            self.c["reduce_suppressed_protected"] += 1
            return
        t_eff = t + self.latency_s
        rec["reduce"] = {"t_request": t, "t_effective": t_eff,
                         "applied": False}
        self.c["reduce_requested"] += 1
        self.traj.append({"kind": "REDUCE_ISSUED", "t": t,
                          "slug": self.slug, "side": side,
                          "ref_gen": rec["ref"]["gen"],
                          "policy_gen": rec["policy_gen"],
                          "t_effective": t_eff})

    # -- run ----------------------------------------------------------------

    def run(self) -> None:
        for t, _rank, _sidx, _seq, kind, side, payload in _merged_events(
                self.sides_ref, self.scores):
            if kind == "start":
                self._on_gen_start(side, payload, t)
            elif kind == "fill":
                self._on_fill(side, payload[0], payload[1], t)
            elif kind == "end":
                self._on_gen_end(side, payload, t)
            else:
                self._on_score(side, payload, t)
        window_end = max((g["t1"] for s in SIDES for g in self.sides_ref[s]),
                         default=0.0)
        for side, run in self.side_runs.items():
            if run.pending is not None:
                self.c["cancels_unresolved"] += 1      # structurally 0
            for h in run.holds:
                if h["end"] is None:
                    h["permanent"] = True
                    h["end"] = window_end
                    self.c["permanent_holds"] += 1
                else:
                    h["permanent"] = False
                self.c["holds_total"] += 1
                dur = h["end"] - h["start"]
                self.econ["hold_seconds_total"] += dur
                self.econ["hold_seconds_max"] = max(
                    self.econ["hold_seconds_max"], dur)
            self.econ["holds"].extend(run.holds)
            for recs in run.recs_by_gen.values():
                for rec in recs:
                    cn = rec["cancel"]
                    if cn is None or cn["outcome"] is None:
                        continue
                    if (cn["outcome"] == "EFFECTIVE"
                            and rec["prevented_shares"] <= EPS):
                        self.c["cancels_zero_value"] += 1
                    self.cancel_records.append({
                        "slug": self.slug, "side": side,
                        "ref_gen": rec["ref"]["gen"],
                        "policy_gen": rec["policy_gen"],
                        "t_request": cn["t_request"],
                        "t_effective": cn["t_effective"],
                        "outcome": cn["outcome"],
                        "reducing_at_request": cn["reducing_at_request"],
                        "prevented_shares": rec["prevented_shares"],
                        "prevented_value_cents": rec["prevented_value_cents"],
                        "stale_shares_charged": rec["stale_shares"],
                        "stale_markout_cents": rec["stale_value_cents"]})


_NOT_RECEIVED_BUCKETS = ("prevented_after_cancel", "prevented_reduced",
                         "missed_while_held", "missed_post_repost")

_COUNTER_NAMES = (
    "cancels_requested", "cancels_rate_passed",
    "cancels_suppressed_rate_limited", "cancel_suppressed_protected",
    "cancels_issued", "cancels_effective", "cancels_stale",
    "cancels_zero_value", "cancels_unresolved",
    "crossings_while_pending", "crossings_while_held",
    "crossings_while_idle", "crossings_on_non_ok_generation",
    "reduce_requested", "reduce_effective", "reduce_stale",
    "reduce_superseded_by_cancel", "reduce_suppressed_protected",
    "reduce_crossings_while_reduced", "reduce_crossings_while_pending",
    "reduce_crossings_while_held", "reduce_crossings_while_idle",
    "hold_entries", "holds_total", "permanent_holds",
    "repost_eligible_transitions", "repost_eligibility_revoked",
    "reposts", "reposts_mid_generation", "reposts_at_generation_start",
    "gen_starts_missed_held", "non_ok_generations",
    "scores_ignored_predictor_disabled", "scores_unknown_slug",
)


def replay_policy(reference: dict[str, Any],
                  scores: Sequence[dict[str, Any]],
                  params: dict[str, Any]) -> dict[str, Any]:
    """Replay the stateful cancel/hold/repost policy over the QR_SKEW_ONLY
    reference trajectory.  Pure function of its three inputs; deterministic;
    consumes the score stream, never re-emits it.  Output carries NO feature
    fields and NO score values -- see the header's structural-ban paragraph.
    Inventory is PER-SLUG: each slug's replay starts flat and its protection
    decisions read only that slug's own net (R-184 step (vii)); the result's
    "inventory" block is a reporting-only aggregate beside per_slug detail.
    """
    p = validate_params(params)
    validate_reference(reference)
    validate_scores(scores)
    # THE DENOMINATOR EVERY CANCEL COUNTER IS COUNTED OVER. Taken at the
    # top, from the stream as supplied, so a zero counter below can be
    # told apart from a path that never ran (DA's routed finding).
    n_score_events = len(scores)
    counters = {k: 0 for k in _COUNTER_NAMES}
    trajectory: list[dict] = []
    cancel_records: list[dict] = []
    econ: dict[str, Any] = {
        "queue_reset_cost_cents_total": 0.0,
        "hold_seconds_total": 0.0, "hold_seconds_max": 0.0,
        "holds": [],
        "fills": {"reference_shares": 0.0, "received_shares": 0.0,
                  "received_markout_cents": 0.0,
                  "received_unvalued_shares": 0.0,
                  "stale_shares": 0.0, "stale_markout_cents": 0.0},
        "not_received": {b: {"shares": 0.0, "value_cents": 0.0,
                             "harm_avoided_cents": 0.0,
                             "sacrifice_cents": 0.0, "unvalued_shares": 0.0}
                         for b in _NOT_RECEIVED_BUCKETS},
    }
    # Inventory is PER-SLUG (v2; R-184 step (vii), coordinator DE hold):
    # each slug is an independent binary market settling at expiry, so the
    # net feeding REDUCING_SIDE_PROTECTION resets to flat at every slug
    # boundary.  v1 allocated ONE dict here and reused it across slugs, so
    # a SELL cancel in market 2 could be suppressed solely by market 1's
    # leftover positive net (user audit; permanent falsifier: selftest
    # group O).  inv_by_slug feeds ONLY the reporting aggregate below.
    inv_by_slug: dict[str, dict[str, float]] = {}
    by_slug: dict[str, list] = {slug: [] for slug in reference}
    for s in scores:
        if s["slug"] in by_slug:
            by_slug[s["slug"]].append(s)
        else:
            counters["scores_unknown_slug"] += 1     # counted, never dropped
    for slug, sides in reference.items():
        inv = {"net": 0.0, "peak_abs_net": 0.0,      # fresh: reset to flat
               "received_increasing_shares": 0.0,
               "received_reducing_shares": 0.0}
        inv_by_slug[slug] = inv
        _SlugReplay(slug, sides, by_slug[slug], p, counters, trajectory,
                    cancel_records, econ, inv).run()

    nr = econ["not_received"]
    harm = sum(b["harm_avoided_cents"] for b in nr.values())
    sac = sum(b["sacrifice_cents"] for b in nr.values())
    n_ref_gens = sum(len(reference[sl][sd]) for sl in reference for sd in
                     SIDES)
    fills = econ["fills"]
    result = {
        "unit": "ACTION",
        "params": p,
        "n_slugs": len(reference),
        "n_reference_generations": n_ref_gens,
        "n_actions_cancel": counters["cancels_issued"],
        "counters": counters,
        # DA's routed finding: a COUNTER WITH NO COMPANION DENOMINATOR
        # makes "the path ran and counted nothing" indistinguishable from
        # "the path never ran". DA hit the same shape in its own battery
        # and it produced a PHANTOM DETERMINISM FAILURE -- children that
        # could not import, reported as `identical: false`, which reads as
        # nondeterminism when neither interpreter had run. Every counter
        # block below now carries what it was counted OVER, and a block
        # whose denominator is zero says UNEVALUATED rather than reporting
        # a zero that looks like a measurement.
        "counters_evaluated": {
            "n_reference_generations": n_ref_gens,
            "n_slugs": len(reference),
            "n_score_events": n_score_events,
            "cancellation_was_reachable": n_score_events > 0
            and n_ref_gens > 0,
            "why": "zeros in `rate_limit`, `cancel_lifecycle` and `holds` "
                   "are only measurements when these are non-zero; with "
                   "no score events no cancel decision was ever taken and "
                   "every cancel counter is UNEVALUATED, not zero"},
        "rate_limit": {"requested": counters["cancels_requested"],
                       "passed": counters["cancels_rate_passed"],
                       "suppressed":
                           counters["cancels_suppressed_rate_limited"],
                       "evaluated_over_score_events": n_score_events,
                       "status": ("UNEVALUATED_NO_SCORE_EVENTS"
                                  if n_score_events == 0 else "COUNTED")},
        "cancel_lifecycle": {"issued": counters["cancels_issued"],
                             "effective": counters["cancels_effective"],
                             "stale": counters["cancels_stale"],
                             "zero_value": counters["cancels_zero_value"],
                             "unresolved": counters["cancels_unresolved"],
                             "evaluated_over_score_events": n_score_events,
                             "evaluated_over_generations": n_ref_gens,
                             "status": ("UNEVALUATED_NO_SCORE_EVENTS"
                                        if n_score_events == 0
                                        else "COUNTED")},
        "economics": {
            "harm_avoided_cents": harm,
            "sacrifice_cents": sac,
            "not_received_net_cents": harm - sac,
            "not_received": nr,
            "queue_reset_cost_cents_total":
                econ["queue_reset_cost_cents_total"],
            # includes queue-reset costs and every lifecycle bucket; still a
            # PARTIAL objective (received-fill P&L is reported beside it,
            # not netted in), so it is not labelled strategy profit.
            "cost_adjusted_value_cents":
                harm - sac - econ["queue_reset_cost_cents_total"],
            "received_markout_cents": fills["received_markout_cents"],
        },
        "fills": dict(fills),
        "retention_share_fraction": (
            fills["received_shares"] / fills["reference_shares"]
            if fills["reference_shares"] > 0 else None),
        "holds": {"n": counters["holds_total"],
                  "permanent": counters["permanent_holds"],
                  "total_s": econ["hold_seconds_total"],
                  "max_s": econ["hold_seconds_max"],
                  "records": econ["holds"],
                  "evaluated_over_score_events": n_score_events,
                  "evaluated_over_generations": n_ref_gens,
                  "status": ("UNEVALUATED_NO_SCORE_EVENTS"
                             if n_score_events == 0 else "COUNTED")},
        # REPORTING-ONLY cross-slug aggregate over the per-slug
        # inventories (decisions read only each slug's own dict; the
        # per-slug terminal nets are what settle at expiry, so the summed
        # terminal_net carries no decision meaning)
        "inventory": {
            "terminal_net": sum(v["net"] for v in inv_by_slug.values()),
            "peak_abs_net": max(
                (v["peak_abs_net"] for v in inv_by_slug.values()),
                default=0.0),
            "received_increasing_shares": sum(
                v["received_increasing_shares"]
                for v in inv_by_slug.values()),
            "received_reducing_shares": sum(
                v["received_reducing_shares"]
                for v in inv_by_slug.values()),
            "per_slug": {
                slug: {"terminal_net": v["net"],
                       "peak_abs_net": v["peak_abs_net"],
                       "received_increasing_shares":
                           v["received_increasing_shares"],
                       "received_reducing_shares":
                           v["received_reducing_shares"]}
                for slug, v in inv_by_slug.items()}},
        "cancels": cancel_records,
        "trajectory": trajectory,
    }
    return result


# ---------------------------------------------------------------------------
# pass-through baseline (QR_SKEW_ONLY) -- independent direct construction
# ---------------------------------------------------------------------------

def build_passthrough_trajectory(reference: dict[str, Any]) -> list[dict]:
    """The QR_SKEW_ONLY no-cancel trajectory, built DIRECTLY from the
    reference input by a plain loop -- no state machine involved, so the
    bit-parity tests compare the machine against an object it did not
    produce.  Shares/levels/markouts are passed through untouched (no
    arithmetic), which is what makes bit-identity a fair demand."""
    validate_reference(reference)
    events: list[dict] = []
    for slug, sides in reference.items():
        for t, _rank, _sidx, _seq, kind, side, payload in _merged_events(
                sides, []):
            if kind == "start":
                events.append(_ev_place(slug, side, payload,
                                        str(payload["gen"]), t, "TRACKING"))
            elif kind == "fill":
                g, tr = payload
                events.append(_ev_fill_charged(
                    slug, side, g, str(g["gen"]), t, tr["shares"], False,
                    tr["markout_cents_per_share"]))
            elif kind == "end":
                events.append(_ev_gen_end(slug, side, payload,
                                          str(payload["gen"]), t,
                                          "REFERENCE_END"))
    return events


# ---------------------------------------------------------------------------
# computed invariants over a result (rule 10: predicates, never verdicts)
# ---------------------------------------------------------------------------

def check_invariants(result: dict[str, Any]) -> dict[str, bool]:
    """Compute the parity-battery predicates from the result's own data.

    Malformed trajectories are REFUSED (TrajectoryIntegrityError): unknown
    event kind, keys outside the closed EVENT_KEYS schema, a cancel event
    for a generation never placed, or a stale-flagged fill on a generation
    with no cancel.  Refusal is the known-bad arm; a False predicate is the
    positive-control arm.  Nothing here prints a verdict."""
    traj = result.get("trajectory")
    if not isinstance(traj, list):
        raise TrajectoryIntegrityError("result has no trajectory list")
    placed: set = set()
    issued: dict = {}          # (slug, side, policy_gen) -> issued event
    effective: dict = {}
    issue_counts: dict = {}
    charged_by_gen: dict = {}
    stale_fills: list[dict] = []
    repost_cost = 0.0
    for e in traj:
        if not isinstance(e, dict) or "kind" not in e:
            raise TrajectoryIntegrityError(f"event without kind: {e!r}")
        kind = e["kind"]
        if kind not in EVENT_KEYS:
            raise TrajectoryIntegrityError(f"unknown event kind {kind!r}")
        if set(e) != set(EVENT_KEYS[kind]):
            raise TrajectoryIntegrityError(
                f"{kind} keys {sorted(e)} differ from the closed schema "
                f"{sorted(EVENT_KEYS[kind])} -- an extra field is how a "
                f"feature would leak into a policy trajectory")
        key = (e.get("slug"), e.get("side"), e.get("policy_gen"))
        if kind == "PLACE":
            placed.add(key)
        elif kind == "CANCEL_ISSUED":
            if key not in placed:
                raise TrajectoryIntegrityError(
                    f"CANCEL_ISSUED for never-placed generation {key}")
            issue_counts[key] = issue_counts.get(key, 0) + 1
            issued[key] = e
        elif kind == "CANCEL_EFFECTIVE":
            if key not in issued:
                raise TrajectoryIntegrityError(
                    f"CANCEL_EFFECTIVE without CANCEL_ISSUED for {key}")
            effective[key] = e
        elif kind == "FILL_CHARGED":
            charged_by_gen.setdefault(key, []).append(e)
            if e["stale"]:
                stale_fills.append(e)
        elif kind == "REPOST":
            repost_cost += e["cost_charged_cents"]
    for e in stale_fills:
        key = (e["slug"], e["side"], e["policy_gen"])
        if key not in issued:
            raise TrajectoryIntegrityError(
                f"stale-flagged fill on {key} which has no cancel -- a "
                f"stale charge without a cancel window is meaningless")
    out: dict[str, bool] = {}
    out["one_cancel_per_generation"] = all(
        n <= 1 for n in issue_counts.values()) if issue_counts else True
    out["effectiveness_time_consistent"] = all(
        abs(e["t"] - issued[k]["t_effective"]) <= 1e-9
        for k, e in effective.items())
    out["no_fill_charged_after_effective_cancel"] = all(
        f["t"] < e["t"]          # a charge AT effectiveness is a violation
        for k, e in effective.items()
        for f in charged_by_gen.get(k, ()))
    out["stale_fills_inside_latency_window"] = all(
        issued[(f["slug"], f["side"], f["policy_gen"])]["t"] - 1e-12
        <= f["t"]
        < issued[(f["slug"], f["side"], f["policy_gen"])]["t_effective"]
        for f in stale_fills)
    fills = result.get("fills", {})
    out["stale_accounting_consistent"] = (
        abs(sum(f["shares"] for f in stale_fills)
            - fills.get("stale_shares", math.nan)) <= 1e-9)
    econ = result.get("economics", {})
    out["queue_reset_cost_consistent"] = (
        abs(repost_cost - econ.get("queue_reset_cost_cents_total",
                                   math.nan)) <= 1e-9)
    return out


# ---------------------------------------------------------------------------
# selftest: the Phase-3 parity battery, synthetic fixtures only
# ---------------------------------------------------------------------------

def _gen(gen: int, t0: float, t1: float, tranches, displayed: float = 5.0,
         level: float = 0.5, status: str = OK) -> dict:
    return {"gen": gen, "t0": t0, "t1": t1, "level": level,
            "displayed": displayed, "status": status,
            "tranches": [{"t": t, "shares": sh,
                          "markout_cents_per_share": mo}
                         for t, sh, mo in tranches]}


def _ref1() -> dict:
    """Two BUY generations and one SELL generation; hand-checkable numbers.
    markout sign: positive = favorable fill, negative = adverse."""
    return {"w1": {
        "BUY_UP": [
            _gen(1, 0.0, 10.0, [(2.0, 2.0, -10.0), (6.5, 1.0, -4.0),
                                (8.0, 1.0, -25.0)]),
            _gen(2, 10.0, 20.0, [(15.0, 1.0, -20.0)], level=0.49),
        ],
        "SELL_UP": [
            _gen(1, 0.0, 20.0, [(5.0, 1.0, 3.0)], level=0.51),
        ],
    }}


def _scores1() -> list:
    return [
        {"t": 1.0, "slug": "w1", "side": "BUY_UP", "score": 0.1},
        {"t": 3.0, "slug": "w1", "side": "SELL_UP", "score": 0.05},
        {"t": 6.0, "slug": "w1", "side": "BUY_UP", "score": 0.9},
        {"t": 7.0, "slug": "w1", "side": "BUY_UP", "score": 0.95},
        {"t": 9.0, "slug": "w1", "side": "BUY_UP", "score": 0.1},
        {"t": 12.0, "slug": "w1", "side": "BUY_UP", "score": 0.2},
    ]


def _params(**over: Any) -> dict:
    base = {"predictor_enabled": True, "theta_cancel": 0.8,
            "theta_repost": 0.3, "repost_dwell_s": 2.0,
            "cancel_effective_latency_ms": 1000.0,
            "queue_reset_cost_cents": 3.0,
            "protection_mode": "ALL_ORDERS_OVERRIDE",
            "max_cancels_per_minute": float("inf"),
            "repost_fill_model": "REFERENCE_FILLS",
            "charge_reset_cost_at_generation_start": True}
    base.update(over)
    return base


def _fk_place(g: str = "1") -> dict:
    return {"kind": "PLACE", "t": 0.0, "slug": "w", "side": "BUY_UP",
            "ref_gen": 1, "policy_gen": g, "level": 0.5, "displayed": 5.0,
            "source": "TRACKING"}


def _fk_cancel(t: float, teff: float, g: str = "1") -> dict:
    return {"kind": "CANCEL_ISSUED", "t": t, "slug": "w", "side": "BUY_UP",
            "ref_gen": 1, "policy_gen": g, "t_effective": teff,
            "reducing_at_request": False}


def _fk_eff(t: float, g: str = "1") -> dict:
    return {"kind": "CANCEL_EFFECTIVE", "t": t, "slug": "w",
            "side": "BUY_UP", "ref_gen": 1, "policy_gen": g}


def _fk_fill(t: float, stale: bool, g: str = "1") -> dict:
    return {"kind": "FILL_CHARGED", "t": t, "slug": "w", "side": "BUY_UP",
            "ref_gen": 1, "policy_gen": g, "shares": 1.0, "stale": stale,
            "markout_cents_per_share": -1.0}


def _fake_result(evs: list, stale_sh: float = 0.0,
                 reset: float = 0.0) -> dict:
    return {"trajectory": evs, "fills": {"stale_shares": stale_sh},
            "economics": {"queue_reset_cost_cents_total": reset}}


#: The COUNTER keys, named once. The emission also carries denominators
#: and a status beside them (DA's routed finding), so a check that
#: compares the whole block by equality breaks the moment a denominator
#: is added -- and a denominator should be addable without editing five
#: assertions.
_CNT = ("requested", "passed", "suppressed")
_LC = ("issued", "effective", "stale", "zero_value", "unresolved")


def selftest() -> int:
    checks = 0

    def ok(cond: Any, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    def refuses(exc: type, fn, label: str) -> None:
        nonlocal checks
        try:
            fn()
        except exc:
            checks += 1
            return
        raise AssertionError(f"NOT REFUSED: {label}")

    ref = _ref1()
    scores = _scores1()

    # ---- group A: parity gates 1 and 2 (disabled / theta=+inf) ----------
    pt = build_passthrough_trajectory(ref)
    ok(len(pt) == 11 and pt[0]["kind"] == "PLACE"
       and pt[0]["side"] == "BUY_UP",
       "passthrough is the direct QR_SKEW_ONLY construction (11 events)")
    dis = replay_policy(ref, scores, _params(predictor_enabled=False))
    ok(bit_identical(dis["trajectory"], pt),
       "GATE 1: disabled predictor is BIT-IDENTICAL to QR_SKEW_ONLY")
    ok(dis["cancel_lifecycle"]["issued"] == 0
       and dis["economics"]["queue_reset_cost_cents_total"] == 0.0
       and dis["retention_share_fraction"] == 1.0,
       "disabled: zero cancels, zero reset cost, retention exactly 1.0")
    ok(dis["counters"]["scores_ignored_predictor_disabled"] == 6
       and dis["counters"]["cancels_requested"] == 0
       and dis["counters"]["repost_eligible_transitions"] == 0,
       "disabled: every score is a COUNTED ignore, and none touched state")
    inf_run = replay_policy(ref, scores,
                            _params(theta_cancel=float("inf")))
    ok(bit_identical(inf_run["trajectory"], pt),
       "GATE 2: theta_cancel=+inf is BIT-IDENTICAL to QR_SKEW_ONLY")
    ok(bit_identical(inf_run["trajectory"], dis["trajectory"]),
       "theta=+inf equals disabled: score evaluation has no side effects")
    ena = replay_policy(ref, scores, _params())
    ok(not bit_identical(ena["trajectory"], pt),
       "POSITIVE CONTROL: an acting policy is NOT bit-identical, so the "
       "parity comparator can fire")
    refuses(VacuousParity, lambda: bit_identical([], []),
            "KNOWN-BAD: empty-vs-empty parity is refused as vacuous")

    # ---- group B: the enabled run's hand-checkable lifecycle ------------
    c = ena["counters"]
    lc = ena["cancel_lifecycle"]
    ok({k: ena["rate_limit"][k] for k in _CNT}
       == {"requested": 1, "passed": 1, "suppressed": 0}
       and {k: lc[k] for k in _LC}
       == {"issued": 1, "effective": 1, "stale": 0, "zero_value": 0,
           "unresolved": 0},
       "enabled run: one requested, one issued, one effective cancel")
    # DA's routed finding, driven: the counters carry WHAT THEY WERE
    # COUNTED OVER, so a zero from a path that never ran is not a zero
    # that was measured.
    ok(lc["evaluated_over_score_events"] > 0
       and lc["status"] == "COUNTED"
       and ena["rate_limit"]["status"] == "COUNTED"
       and ena["holds"]["status"] == "COUNTED"
       and ena["counters_evaluated"]["cancellation_was_reachable"] is True,
       f"and every cancel counter carries its DENOMINATOR: counted over "
       f"{lc['evaluated_over_score_events']} score events and "
       f"{lc['evaluated_over_generations']} generations, status "
       f"{lc['status']}. A counter with no denominator makes 'the path "
       f"ran and counted nothing' indistinguishable from 'the path never "
       f"ran' -- the shape that reported DA's non-importing children as "
       f"`identical: false`, which reads as nondeterminism when neither "
       f"interpreter had run")

    ec = ena["economics"]
    ok(abs(ec["harm_avoided_cents"] - 25.0) < 1e-9
       and ec["sacrifice_cents"] == 0.0,
       "harm avoided = 25c (the prevented t=8 adverse fill), no sacrifice")
    ok(abs(ec["queue_reset_cost_cents_total"] - 3.0) < 1e-9
       and abs(ec["cost_adjusted_value_cents"] - 22.0) < 1e-9,
       "declared queue-reset cost charged once; 25 - 3 = 22")
    fl = ena["fills"]
    ok(abs(fl["stale_shares"] - 1.0) < 1e-12
       and abs(fl["stale_markout_cents"] - (-4.0)) < 1e-12,
       "the t=6.5 fill inside the latency window stays CHARGED as stale")
    ok(abs(fl["received_shares"] - 5.0) < 1e-12
       and abs(fl["reference_shares"] - 6.0) < 1e-12
       and abs(ena["retention_share_fraction"] - 5.0 / 6.0) < 1e-12,
       "share retention 5/6 (one reference fill prevented)")
    iv = ena["inventory"]
    ok(abs(iv["terminal_net"] - 3.0) < 1e-12
       and abs(iv["peak_abs_net"] - 3.0) < 1e-12
       and abs(iv["received_increasing_shares"] - 4.0) < 1e-12
       and abs(iv["received_reducing_shares"] - 1.0) < 1e-12,
       "inventory tracked from CHARGED fills; increasing/reducing split")
    ok(ena["holds"]["n"] == 1 and ena["holds"]["permanent"] == 0
       and abs(ena["holds"]["total_s"] - 5.0) < 1e-9,
       "one hold, 7s..12s, released by the repost")
    ok(c["reposts"] == 1 and c["reposts_mid_generation"] == 1
       and any(e["kind"] == "REPOST"
               and abs(e["cost_charged_cents"] - 3.0) < 1e-12
               for e in ena["trajectory"]),
       "one mid-generation repost, charged the declared reset cost")
    ok(any(e["kind"] == "REPOST_ELIGIBLE" and abs(e["t"] - 11.0) < 1e-12
           for e in ena["trajectory"]),
       "eligibility at below_since(9) + dwell(2) = 11, its own time")
    ok(c["crossings_while_held"] == 1,
       "the t=7 crossing lands on HELD state and is counted, not acted")
    cr = ena["cancels"][0]
    ok(cr["outcome"] == "EFFECTIVE" and cr["t_request"] == 6.0
       and cr["t_effective"] == 7.0
       and abs(cr["prevented_shares"] - 1.0) < 1e-12
       and abs(cr["prevented_value_cents"] - 25.0) < 1e-9
       and abs(cr["stale_shares_charged"] - 1.0) < 1e-12,
       "per-cancel record carries request/effective times and both sides "
       "of the ledger (prevented AND stale-charged)")
    ok(all(check_invariants(ena).values()),
       "every computed invariant holds on the enabled run")

    # ---- group C: gate 3 -- permanent hold == cancel-and-hold -----------
    hold_run = replay_policy(ref, scores, _params(theta_repost=0.0))
    gb1 = ref["w1"]["BUY_UP"][0]
    gs1 = ref["w1"]["SELL_UP"][0]
    expected_hold = [
        _ev_place("w1", "BUY_UP", gb1, "1", 0.0, "TRACKING"),
        _ev_place("w1", "SELL_UP", gs1, "1", 0.0, "TRACKING"),
        _ev_fill_charged("w1", "BUY_UP", gb1, "1", 2.0, 2.0, False, -10.0),
        _ev_fill_charged("w1", "SELL_UP", gs1, "1", 5.0, 1.0, False, 3.0),
        {"kind": "CANCEL_ISSUED", "t": 6.0, "slug": "w1", "side": "BUY_UP",
         "ref_gen": 1, "policy_gen": "1", "t_effective": 7.0,
         "reducing_at_request": False},
        _ev_fill_charged("w1", "BUY_UP", gb1, "1", 6.5, 1.0, True, -4.0),
        {"kind": "CANCEL_EFFECTIVE", "t": 7.0, "slug": "w1",
         "side": "BUY_UP", "ref_gen": 1, "policy_gen": "1"},
        _ev_gen_end("w1", "BUY_UP", gb1, "1", 7.0, "CANCELLED"),
        {"kind": "HOLD_START", "t": 7.0, "slug": "w1", "side": "BUY_UP",
         "ref_gen": 1, "policy_gen": "1"},
        {"kind": "FILL_PREVENTED", "t": 8.0, "slug": "w1", "side": "BUY_UP",
         "ref_gen": 1, "policy_gen": "1", "shares": 1.0,
         "markout_cents_per_share": -25.0, "cancel_t_effective": 7.0},
        {"kind": "GEN_START_MISSED_HELD", "t": 10.0, "slug": "w1",
         "side": "BUY_UP", "ref_gen": 2},
        {"kind": "FILL_MISSED_HELD", "t": 15.0, "slug": "w1",
         "side": "BUY_UP", "ref_gen": 2, "shares": 1.0,
         "markout_cents_per_share": -20.0},
        _ev_gen_end("w1", "SELL_UP", gs1, "1", 20.0, "REFERENCE_END"),
    ]
    ok(bit_identical(hold_run["trajectory"], expected_hold),
       "GATE 3: theta_repost=0 with non-negative scores never reposts and "
       "matches the hand-built cancel-and-hold trajectory exactly")
    nr3 = hold_run["economics"]["not_received"]
    ok(hold_run["holds"]["permanent"] == 1 and hold_run["holds"]["n"] == 1
       and abs(nr3["missed_while_held"]["harm_avoided_cents"] - 20.0) < 1e-9
       and abs(hold_run["economics"]["cost_adjusted_value_cents"]
               - 45.0) < 1e-9,
       "permanent hold counted; held-out generation's fill valued as "
       "missed-while-held; 25 + 20 - 0 = 45")
    ok(not bit_identical(ena["trajectory"], expected_hold),
       "POSITIVE CONTROL: the reposting run does NOT match cancel-and-hold")

    # ---- group D: gate 4 -- at most one cancel per generation -----------
    adv_scores = [{"t": t, "slug": "w1", "side": "BUY_UP", "score": s}
                  for t, s in ((6.0, 0.9), (6.2, 0.99), (6.4, 0.99),
                               (7.5, 0.99), (8.5, 0.99))]
    adv = replay_policy(ref, adv_scores, _params())
    ok(adv["cancel_lifecycle"]["issued"] == 1,
       "GATE 4: five crossings on one generation issue exactly ONE cancel")
    ok(adv["counters"]["crossings_while_pending"] == 2
       and adv["counters"]["crossings_while_held"] == 2,
       "the four later crossings are COUNTED states, never actions")
    ok(check_invariants(adv)["one_cancel_per_generation"],
       "the one-cancel invariant is computed from the trajectory itself")
    bad_double = _fake_result([_fk_place(), _fk_cancel(1.0, 2.0),
                               _fk_cancel(3.0, 4.0)])
    ok(not check_invariants(bad_double)["one_cancel_per_generation"],
       "FLAG: a fabricated double-cancel trajectory trips the predicate")
    refuses(TrajectoryIntegrityError,
            lambda: check_invariants(_fake_result([_fk_cancel(1.0, 2.0)])),
            "KNOWN-BAD: a cancel for a never-placed generation is refused")

    # ---- group E: gate 5 -- no fill after cancel effectiveness ----------
    ok(not any(e["kind"] == "FILL_CHARGED" and e["policy_gen"] == "1"
               and e["t"] >= 7.0 for e in ena["trajectory"])
       and any(e["kind"] == "FILL_PREVENTED" and e["t"] == 8.0
               and e["cancel_t_effective"] == 7.0
               for e in ena["trajectory"]),
       "GATE 5: the cancelled generation cannot fill at/after t_effective; "
       "its t=8 tranche is PREVENTED, not charged")
    bad_late = _fake_result([_fk_place(), _fk_cancel(6.0, 7.0),
                             _fk_eff(7.0), _fk_fill(8.0, False)])
    ok(not check_invariants(bad_late)[
        "no_fill_charged_after_effective_cancel"],
       "FLAG: a fabricated post-effectiveness charge trips the predicate")
    refuses(TrajectoryIntegrityError,
            lambda: check_invariants(_fake_result([_fk_place(),
                                                   _fk_eff(7.0)])),
            "KNOWN-BAD: CANCEL_EFFECTIVE without CANCEL_ISSUED is refused")

    # ---- group F: gate 6 -- pre-effectiveness fills stay charged --------
    ok(any(e["kind"] == "FILL_CHARGED" and e["stale"] and e["t"] == 6.5
           for e in ena["trajectory"])
       and abs(ena["cancels"][0]["stale_shares_charged"] - 1.0) < 1e-12,
       "GATE 6: the fill inside [t_request, t_effective) is charged and "
       "flagged stale on both the trajectory and the cancel record")
    bad_window = _fake_result([_fk_place(), _fk_cancel(6.0, 7.0),
                               _fk_fill(7.5, True)], stale_sh=1.0)
    ok(not check_invariants(bad_window)["stale_fills_inside_latency_window"],
       "FLAG: a stale-flagged fill OUTSIDE the latency window trips the "
       "predicate")
    refuses(TrajectoryIntegrityError,
            lambda: check_invariants(_fake_result([_fk_place(),
                                                   _fk_fill(1.0, True)])),
            "KNOWN-BAD: a stale fill on a generation with no cancel is "
            "refused")
    tampered = {"trajectory": ena["trajectory"],
                "fills": {**ena["fills"], "stale_shares": 9.9},
                "economics": ena["economics"]}
    ok(not check_invariants(tampered)["stale_accounting_consistent"],
       "FLAG: tampered stale-share accounting trips the consistency "
       "predicate")

    # ---- group G: gate 7 -- constructor refusals ------------------------
    refuses(InvalidParameter, lambda: validate_params(
        _params(theta_repost=0.9)),
        "theta_repost > theta_cancel is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(theta_repost=0.8)),
        "theta_repost == theta_cancel is refused (strict inequality)")
    refuses(UndeclaredParameter, lambda: validate_params(
        {k: v for k, v in _params().items()
         if k != "queue_reset_cost_cents"}),
        "an undeclared queue-reset cost is refused")
    for key in _REQUIRED_PARAMS:
        try:
            validate_params({k: v for k, v in _params().items() if k != key})
        except UndeclaredParameter:
            continue
        raise AssertionError(f"missing {key} was not refused")
    ok(True, "EVERY required parameter is refused when undeclared "
             f"({len(_REQUIRED_PARAMS)} parameters exercised)")
    refuses(InvalidParameter, lambda: validate_params(
        _params(cancel_effective_latency_ms=0.0)),
        "zero latency (the v1 defect) is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(cancel_effective_latency_ms=-5.0)),
        "negative latency is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(theta_cancel=float("nan"))),
        "NaN theta_cancel is refused (it would compare False everywhere)")
    refuses(InvalidParameter, lambda: validate_params(
        _params(protection_mode="BOTH")),
        "an unknown protection mode is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(repost_fill_model="OPTIMISTIC")),
        "an unknown repost fill model is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(max_cancels_per_minute=0.0)),
        "a zero rate limit is refused (inf is the explicit 'unlimited')")
    refuses(UndeclaredParameter, lambda: validate_params(
        _params(enable_reduce=True,
                reduce_remaining_fraction=0.5)),
        "enable_reduce without theta_reduce is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(enable_reduce=True, theta_reduce=0.9,
                reduce_remaining_fraction=0.5)),
        "theta_reduce >= theta_cancel is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(enable_reduce=True, theta_reduce=0.5,
                reduce_remaining_fraction=1.0)),
        "reduce fraction 1.0 (a no-op posing as a reduce) is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(enable_reduce=True, theta_reduce=0.5,
                reduce_remaining_fraction=0.0)),
        "reduce fraction 0.0 (a cancel posing as a reduce) is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(theta_reduce=0.5)),
        "theta_reduce declared while reduce is disabled is refused")
    refuses(InvalidParameter, lambda: validate_params(
        _params(bogus_knob=1.0)),
        "an unknown parameter name is refused, so a typo cannot become a "
        "silently ignored declaration")
    ok(validate_params(_params())["enable_reduce"] is False,
       "POSITIVE CONTROL: a fully declared config constructs; the reduce "
       "ablation defaults OFF (the one spec-sanctioned default)")
    refuses(InvalidParameter, lambda: validate_params(
        _params(predictor_enabled=False, theta_repost=0.9)),
        "a disabled predictor still validates its declared thetas")

    _selftest_more(ok, refuses, ref, scores, ena)

    if checks != EXPECTED_CHECKS:
        raise AssertionError(
            f"selftest ran {checks} checks but the header declares "
            f"EXPECTED_CHECKS={EXPECTED_CHECKS}; update both together")
    print(f"harmful_stateful_policy selftest: {checks} checks OK")
    return 0


def _selftest_more(ok, refuses, ref, scores, ena) -> None:
    """Groups H..N of the battery (same closures, same running count)."""
    # ---- group H: data refusals and counted exclusions ------------------
    refuses(ReferenceIntegrityError, lambda: validate_scores(
        [{"t": 1.0, "slug": "w1", "side": "BUY_UP",
          "score": float("nan")}]),
        "a NaN score is refused, never a silent below-threshold no-op")
    refuses(ReferenceIntegrityError, lambda: validate_scores(
        [{"t": 1.0, "slug": "w1", "side": "UP", "score": 0.5}]),
        "an unknown maker side is refused")
    bad_ref = _ref1()
    bad_ref["w1"]["BUY_UP"][0]["tranches"][0]["t"] = 55.0
    refuses(ReferenceIntegrityError, lambda: validate_reference(bad_ref),
            "a tranche outside its generation interval is refused")
    bad_ref2 = _ref1()
    bad_ref2["w1"]["BUY_UP"][0]["tranches"][0]["shares"] = 99.0
    refuses(ReferenceIntegrityError, lambda: validate_reference(bad_ref2),
            "tranche shares exceeding displayed are refused")
    bad_ref3 = _ref1()
    bad_ref3["w1"]["BUY_UP"][1]["t0"] = 5.0
    refuses(ReferenceIntegrityError, lambda: validate_reference(bad_ref3),
            "overlapping generations on one side are refused")
    bad_ref4 = _ref1()
    bad_ref4["w1"]["BUY_UP"][0]["tranches"][0][
        "markout_cents_per_share"] = None
    refuses(ReferenceIntegrityError, lambda: validate_reference(bad_ref4),
            "an OK generation with an unvalued fill is refused -- unvalued "
            "belongs under a non-OK status")
    ref_nok = {"w3": {
        "BUY_UP": [_gen(1, 0.0, 10.0, [(2.0, 1.0, None)],
                        status="GAP_IN_HORIZON")],
        "SELL_UP": []}}
    nok = replay_policy(ref_nok,
                        [{"t": 1.0, "slug": "w3", "side": "BUY_UP",
                          "score": 0.9}], _params())
    ok(nok["counters"]["crossings_on_non_ok_generation"] == 1
       and nok["cancel_lifecycle"]["issued"] == 0
       and abs(nok["fills"]["received_unvalued_shares"] - 1.0) < 1e-12,
       "POSITIVE CONTROL for the markout refusal: a non-OK generation IS "
       "accepted, its crossing is a counted status (no cancel), and its "
       "unvalued fill is charged and counted, never dropped")
    extra = replay_policy(ref, scores + [{"t": 2.5, "slug": "w9",
                                          "side": "BUY_UP", "score": 0.99}],
                          _params())
    ok(extra["counters"]["scores_unknown_slug"] == 1
       and extra["cancel_lifecycle"]["issued"] == 1,
       "an unknown-slug score is a COUNTED exclusion and changes nothing")

    # ---- group I: the two protection-mode config cells ------------------
    ref_p = {"w2": {
        "BUY_UP": [_gen(1, 0.0, 10.0, [(2.0, 2.0, -10.0), (6.5, 1.0, -4.0),
                                       (8.0, 1.0, -25.0)])],
        "SELL_UP": [_gen(1, 0.0, 20.0, [(1.0, 3.0, 2.0)], level=0.51)]}}
    sc_p = [{"t": 6.0, "slug": "w2", "side": "BUY_UP", "score": 0.9},
            {"t": 7.0, "slug": "w2", "side": "BUY_UP", "score": 0.95}]
    prot = replay_policy(ref_p, sc_p, _params(
        protection_mode="REDUCING_SIDE_PROTECTION"))
    ovr = replay_policy(ref_p, sc_p, _params())
    ok(prot["counters"]["cancel_suppressed_protected"] == 1
       and prot["cancel_lifecycle"]["issued"] == 1
       and prot["cancels"][0]["t_request"] == 7.0,
       "reducing-side protection: the t=6 crossing on the reducing side is "
       "suppressed; the entitlement survives and the t=7 crossing (net "
       "back to 0, no longer reducing) cancels")
    ok(prot["fills"]["stale_shares"] == 0.0
       and abs(prot["economics"]["harm_avoided_cents"] - 25.0) < 1e-9,
       "under protection the t=6.5 fill is an ordinary charge (no pending "
       "cancel), and the later cancel still prevents the t=8 harm")
    ok(ovr["counters"]["cancel_suppressed_protected"] == 0
       and ovr["cancels"][0]["t_request"] == 6.0
       and ovr["cancels"][0]["reducing_at_request"] is True
       and abs(ovr["fills"]["stale_shares"] - 1.0) < 1e-12,
       "all-orders override: the same crossing cancels the reducing side "
       "immediately, and the record says it was reducing at request")
    ok(not bit_identical(prot["trajectory"], ovr["trajectory"]),
       "the two protection cells are distinct trajectories, not labels")

    # ---- group J: rate limiting counts requested/passed/suppressed ------
    sc_r = [{"t": 6.0, "slug": "w1", "side": "BUY_UP", "score": 0.9},
            {"t": 8.0, "slug": "w1", "side": "BUY_UP", "score": 0.1},
            {"t": 12.0, "slug": "w1", "side": "BUY_UP", "score": 0.9}]
    lim = replay_policy(ref, sc_r, _params(max_cancels_per_minute=1.0))
    ok({k: lim["rate_limit"][k] for k in _CNT}
       == {"requested": 2, "passed": 1, "suppressed": 1}
       and lim["cancel_lifecycle"]["issued"] == 1,
       "rate limiter: second in-window request suppressed, all three "
       "counts reported separately")
    unlim = replay_policy(ref, sc_r, _params())
    ok({k: unlim["rate_limit"][k] for k in _CNT}
       == {"requested": 2, "passed": 2, "suppressed": 0}
       and unlim["cancel_lifecycle"]["issued"] == 2,
       "with an explicit unlimited declaration both requests issue")
    ok(lim["counters"]["reposts_at_generation_start"] == 1
       and abs(lim["economics"]["queue_reset_cost_cents_total"] - 3.0)
       < 1e-12,
       "a repost landing on a generation start is charged when "
       "charge_reset_cost_at_generation_start=True")
    lim0 = replay_policy(ref, sc_r, _params(
        max_cancels_per_minute=1.0,
        charge_reset_cost_at_generation_start=False))
    ok(lim0["counters"]["reposts"] == 1
       and lim0["economics"]["queue_reset_cost_cents_total"] == 0.0,
       "...and NOT charged when the declared parameter says False -- the "
       "ambiguity is a parameter, not a hidden default")

    # ---- group K: reduce ablation (explicit flag, one reduce per gen) ---
    ref_k = {"w1": {
        "BUY_UP": [_gen(1, 0.0, 10.0, [(2.0, 2.0, -10.0), (6.5, 2.0, -4.0),
                                       (8.0, 1.0, -25.0)])],
        "SELL_UP": []}}
    sc_k = [{"t": 4.0, "slug": "w1", "side": "BUY_UP", "score": 0.6},
            {"t": 7.0, "slug": "w1", "side": "BUY_UP", "score": 0.9}]
    red = replay_policy(ref_k, sc_k, _params(
        enable_reduce=True, theta_reduce=0.5,
        reduce_remaining_fraction=0.5))
    ok(red["counters"]["reduce_requested"] == 1
       and red["counters"]["reduce_effective"] == 1
       and any(e["kind"] == "REDUCE_EFFECTIVE"
               and abs(e["remaining_after"] - 1.5) < 1e-12
               for e in red["trajectory"]),
       "reduce: requested at t=4, effective at t=5, remaining 3 -> 1.5")
    nr_k = red["economics"]["not_received"]
    ok(abs(nr_k["prevented_reduced"]["shares"] - 0.5) < 1e-12
       and abs(nr_k["prevented_reduced"]["harm_avoided_cents"] - 2.0) < 1e-9
       and any(e["kind"] == "FILL_PREVENTED_REDUCED"
               for e in red["trajectory"]),
       "the size cut prevents the overflow half-share of the t=6.5 fill")
    ok(red["cancel_lifecycle"]["issued"] == 1
       and abs(nr_k["prevented_after_cancel"]["harm_avoided_cents"] - 25.0)
       < 1e-9,
       "a reduce does not consume the cancel entitlement: the t=7 crossing "
       "still cancels and prevents the t=8 fill")

    # ---- group L: the declared repost fill model ------------------------
    nofill = replay_policy(ref, scores, _params(
        repost_fill_model="NO_FILLS_UNTIL_NEXT_GENERATION"))
    nr_l = nofill["economics"]["not_received"]
    ok(abs(nr_l["missed_post_repost"]["shares"] - 1.0) < 1e-12
       and abs(nr_l["missed_post_repost"]["harm_avoided_cents"] - 20.0)
       < 1e-9
       and abs(nofill["fills"]["received_shares"] - 4.0) < 1e-12,
       "under NO_FILLS_UNTIL_NEXT_GENERATION the mid-generation repost "
       "receives nothing from the joined generation -- counted, valued, "
       "and distinct from REFERENCE_FILLS")

    # ---- group M: non-OK pass-through parity ----------------------------
    ref_nok2 = {"w3": {
        "BUY_UP": [_gen(1, 0.0, 10.0, [(2.0, 1.0, None)],
                        status="TRUNCATED_HORIZON")],
        "SELL_UP": []}}
    ok(replay_policy(ref_nok2, [], _params(
        predictor_enabled=False))["counters"]["non_ok_generations"] == 1,
       "non-OK generations are counted on every run")
    ok(bit_identical(
        replay_policy(ref_nok2, [], _params(predictor_enabled=False))[
            "trajectory"],
        build_passthrough_trajectory(ref_nok2)),
       "disabled-predictor parity holds through non-OK generations too "
       "(charged as-is, unvalued, never dropped)")

    # ---- group N: structural training-row ban + schema closure ----------
    refuses(TrainingPopulationRefused, lambda: emit_training_rows(ena),
            "emit_training_rows refuses UNCONDITIONALLY: the no-training-"
            "population ban is executable, not documentation")
    leaky = _fk_place()
    leaky["feature_x"] = 1.23
    refuses(TrajectoryIntegrityError,
            lambda: check_invariants(_fake_result([leaky])),
            "an event with a key outside the closed schema is refused -- "
            "the path a feature would need to leak through does not exist")
    scores_z = scores + [{"t": 6.0, "slug": "w1", "side": "SELL_UP",
                          "score": 0.9}]
    zed = replay_policy(ref, scores_z, _params())
    ok(zed["cancel_lifecycle"]["zero_value"] == 1
       and zed["cancel_lifecycle"]["effective"] == 2
       and zed["holds"]["permanent"] == 1,
       "a cancel that prevents nothing is counted zero-value; the SELL "
       "hold with no below-threshold score stays permanent")

    # ---- group O: PER-SLUG inventory (R-184 step (vii) falsifier) -------
    # Each slug is an independent binary market settling at expiry, so the
    # net feeding REDUCING_SIDE_PROTECTION resets to flat at slug start.
    # The v1 defect (ONE inventory dict reused across slugs; user audit,
    # two-market falsifier) is reconstructed below as the known-bad arm --
    # both arms per rule 15.
    def v1_global_inventory_replay(reference, score_events, params):
        """Reconstruct the v1 wiring: ONE shared inventory dict across
        every slug of the replay.  Test instrument only."""
        shared = {"net": 0.0, "peak_abs_net": 0.0,
                  "received_increasing_shares": 0.0,
                  "received_reducing_shares": 0.0}
        orig_init = _SlugReplay.__init__

        def patched(self, slug, sides, slug_scores, p, counters,
                    trajectory, cancels, econ, inv):
            orig_init(self, slug, sides, slug_scores, p, counters,
                      trajectory, cancels, econ, shared)

        _SlugReplay.__init__ = patched
        try:
            return replay_policy(reference, score_events, params)
        finally:
            _SlugReplay.__init__ = orig_init

    ref_o = {
        "m1": {"BUY_UP": [_gen(1, 0.0, 10.0, [(2.0, 3.0, -1.0)])],
               "SELL_UP": []},
        "m2": {"BUY_UP": [],
               "SELL_UP": [_gen(1, 0.0, 10.0, [(8.0, 1.0, -30.0)],
                                level=0.51)]},
    }
    sc_o = [{"t": 6.0, "slug": "m2", "side": "SELL_UP", "score": 0.9}]
    p_o = _params(protection_mode="REDUCING_SIDE_PROTECTION")
    fix_two = replay_policy(ref_o, sc_o, p_o)
    ok(fix_two["cancel_lifecycle"]["issued"] == 1
       and fix_two["counters"]["cancel_suppressed_protected"] == 0
       and fix_two["cancels"][0]["slug"] == "m2"
       and fix_two["cancels"][0]["side"] == "SELL_UP"
       and fix_two["cancels"][0]["reducing_at_request"] is False
       and abs(fix_two["cancels"][0]["prevented_value_cents"] - 30.0) < 1e-9
       and all(check_invariants(fix_two).values()),
       "PER-SLUG ARM: market 2's SELL crossing cancels -- market 1's +3 "
       "net never reaches market 2's protection decision (each slug "
       "starts flat) and the t=8 adverse fill is prevented")
    glob_two = v1_global_inventory_replay(ref_o, sc_o, p_o)
    ok(glob_two["cancel_lifecycle"]["issued"] == 0
       and glob_two["counters"]["cancel_suppressed_protected"] == 1
       and abs(glob_two["fills"]["received_shares"] - 4.0) < 1e-12,
       "GLOBAL-INVENTORY ARM (v1 wiring reconstructed): the SAME crossing "
       "is suppressed solely by market 1's leftover net and the adverse "
       "fill stays charged -- the falsifier fires (rule 15)")
    ref_o2 = {"m2": ref_o["m2"]}
    alone = replay_policy(ref_o2, sc_o, p_o)
    ok(alone["cancel_lifecycle"]["issued"] == 1
       and bit_identical([e for e in fix_two["trajectory"]
                          if e["slug"] == "m2"], alone["trajectory"]),
       "market 2 replayed ALONE issues the cancel, and its slice of the "
       "two-market replay is bit-identical to the standalone replay: no "
       "cross-slug effect under per-slug inventory")
    ok(bit_identical(
        alone["trajectory"],
        v1_global_inventory_replay(ref_o2, sc_o, p_o)["trajectory"])
       and bit_identical(
        prot["trajectory"],
        v1_global_inventory_replay(
            ref_p, sc_p,
            _params(protection_mode="REDUCING_SIDE_PROTECTION"))[
            "trajectory"]),
       "single-market replays are BIT-IDENTICAL under per-slug and "
       "v1-global inventory (two fixtures): the fix cannot change any "
       "single-market result")
    iv_o = fix_two["inventory"]
    ok(abs(iv_o["per_slug"]["m1"]["terminal_net"] - 3.0) < 1e-12
       and abs(iv_o["per_slug"]["m2"]["terminal_net"] - 0.0) < 1e-12
       and abs(iv_o["terminal_net"]
               - sum(v["terminal_net"]
                     for v in iv_o["per_slug"].values())) < 1e-12
       and abs(iv_o["peak_abs_net"]
               - max(v["peak_abs_net"]
                     for v in iv_o["per_slug"].values())) < 1e-12,
       "inventory reported per slug; the top-level block equals the "
       "sum/max aggregate of the per-slug dicts (reporting-only)")

    # ---- group P: the boundary fill REAL DATA found (2026-09-01, DE) -----
    # A generation whose only tranche lands at exactly its own t1 -- which is
    # also the NEXT generation's t0.  GEN_START outranks FILL at equal times,
    # so the next generation's start is processed FIRST; if the side is HELD
    # and has just become eligible, that start REPOSTS and clears `held` and
    # `was_eligible`.  The missed generation's fill then arrives at a side
    # that is no longer held, and the pre-fix guard raised "machine bug" on a
    # case that was entirely correct.
    # Found at btc-updown-5m-1787580000/BUY_UP/gen 449 (t1 = 198.186235413 =
    # gen 450's t0) in the first real-data parity run; reduced to this
    # fixture, which RAISES on the pre-fix code and is valued correctly after.
    ref_p2 = {"w": {
        "BUY_UP": [_gen(1, 0.0, 5.0, []),
                   _gen(2, 5.0, 10.0, [(10.0, 1.0, -5.0)]),
                   _gen(3, 10.0, 15.0, [])],
        "SELL_UP": []}}
    sc_p2 = [{"t": 1.0, "slug": "w", "side": "BUY_UP", "score": 0.99},
             {"t": 2.0, "slug": "w", "side": "BUY_UP", "score": 0.5},
             {"t": 6.0, "slug": "w", "side": "BUY_UP", "score": 0.0}]
    p_p2 = _params(protection_mode="ALL_ORDERS_OVERRIDE",
                   repost_dwell_s=0.5)
    bnd = replay_policy(ref_p2, sc_p2, p_p2)
    ok(bnd["counters"]["gen_starts_missed_held"] == 1
       and abs(bnd["economics"]["not_received"]["missed_while_held"]["shares"]
               - 1.0) < 1e-12
       and bnd["counters"]["reposts_at_generation_start"] == 1
       and all(check_invariants(bnd).values()),
       "BOUNDARY FILL: a fill on a generation missed while held is valued "
       "as missed_while_held even though the side reposted on the NEXT "
       "generation at the same instant -- the licence is a fact about the "
       "GENERATION, not the side's one-event-stale flag")
    ok(any(e["kind"] == "GEN_START_MISSED_HELD" and e["ref_gen"] == 2
           for e in bnd["trajectory"])
       and any(e["kind"] == "REPOST" and e["ref_gen"] == 3
               for e in bnd["trajectory"]),
       "and the trajectory shows BOTH facts that collided: generation 2 "
       "missed while held, generation 3 reposted at its own start")
    # THE GUARD IS NOT WEAKENED: a fill on a generation that was neither
    # joined nor missed still raises.  Driven at the unit, because the
    # replay cannot legitimately reach it -- a control that could not fire
    # would be worse than none (rule 16).
    def _unreachable_fill():
        counters = {k: 0 for k in _COUNTER_NAMES}
        econ = {"queue_reset_cost_cents_total": 0.0,
                "hold_seconds_total": 0.0, "hold_seconds_max": 0.0,
                "holds": [],
                "fills": {"reference_shares": 0.0, "received_shares": 0.0,
                          "received_markout_cents": 0.0,
                          "received_unvalued_shares": 0.0,
                          "stale_shares": 0.0, "stale_markout_cents": 0.0},
                "not_received": {b: {"shares": 0.0, "value_cents": 0.0,
                                     "harm_avoided_cents": 0.0,
                                     "sacrifice_cents": 0.0,
                                     "unvalued_shares": 0.0}
                                 for b in _NOT_RECEIVED_BUCKETS}}
        inv = {"net": 0.0, "peak_abs_net": 0.0,
               "received_increasing_shares": 0.0,
               "received_reducing_shares": 0.0}
        r = _SlugReplay("w", ref_p2["w"], [], validate_params(p_p2),
                        counters, [], [], econ, inv)
        ghost = _gen(99, 0.0, 1.0, [])
        r._on_fill("BUY_UP", ghost,
                   {"t": 0.5, "shares": 1.0,
                    "markout_cents_per_share": -1.0}, 0.5)
    refuses(RuntimeError, _unreachable_fill,
            "POSITIVE CONTROL, both directions: a fill on a generation that "
            "was NEITHER joined NOR missed-while-held still raises -- the "
            "fix admits the real case without disarming the guard")

    # ---- group Q: lazy settlement at a GENERATION START (2026-09-01, DE) --
    # The header declares effectiveness settles "at the next processed
    # event"; _on_gen_start was the one handler that read `held` without
    # settling.  Consecutive generations abut and GEN_START outranks GEN_END
    # at equal times, so when no fill or score of the side falls between a
    # cancel's effectiveness and the generation's end, the side PLACED during
    # what should have been a hold and CHARGED every fill of the next
    # generation.  Pre-fix this fixture reported received_shares 1.0 and
    # gen_starts_missed_held 0, with every invariant TRUE -- internally
    # consistent and economically wrong.
    ref_q = {"w": {"BUY_UP": [_gen(1, 0.0, 5.0, []),
                              _gen(2, 5.0, 10.0, [(7.0, 1.0, -9.0)])],
                   "SELL_UP": []}}
    sc_q = [{"t": 1.0, "slug": "w", "side": "BUY_UP", "score": 0.99}]
    q = replay_policy(ref_q, sc_q, _params(
        protection_mode="ALL_ORDERS_OVERRIDE"))
    ok(q["counters"]["gen_starts_missed_held"] == 1
       and abs(q["fills"]["received_shares"]) < 1e-12
       and abs(q["economics"]["not_received"]["missed_while_held"]["shares"]
               - 1.0) < 1e-12,
       "KNOWN-BAD (pre-fix: charged 1.0 share, 0 missed starts): a cancel "
       "effective INSIDE a generation holds the side before the NEXT "
       "generation starts, even with no fill or score in between")
    kinds_q = [e["kind"] for e in q["trajectory"]]
    ok(kinds_q.index("HOLD_START") < kinds_q.index("GEN_START_MISSED_HELD")
       and all(q["trajectory"][i]["t"] <= q["trajectory"][i + 1]["t"] + EPS
               for i in range(len(q["trajectory"]) - 1)),
       "and the trajectory is now time-ordered through the settle: pre-fix "
       "a PLACE at t=5 preceded the CANCEL_EFFECTIVE at t=2 it should have "
       "been held by")
    # POSITIVE CONTROL: settling early must not invent a hold where the
    # cancel never became effective (t_effective at/after the generation end
    # resolves STALE and removes nothing).
    ref_q2 = {"w": {"BUY_UP": [_gen(1, 0.0, 1.5, []),
                               _gen(2, 1.5, 10.0, [(7.0, 1.0, -9.0)])],
                    "SELL_UP": []}}
    q2 = replay_policy(ref_q2, sc_q, _params(
        protection_mode="ALL_ORDERS_OVERRIDE"))
    ok(q2["cancel_lifecycle"]["stale"] == 1
       and q2["counters"]["gen_starts_missed_held"] == 0
       and abs(q2["fills"]["received_shares"] - 1.0) < 1e-12,
       "POSITIVE CONTROL: the same fixture with the cancel landing AFTER "
       "its generation's end resolves STALE, holds nothing, and the next "
       "generation is placed and charged normally -- the settle-first fix "
       "does not manufacture holds")


def main(argv: Sequence[str] | None = None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args != ["--selftest"]:
        print("usage: harmful_stateful_policy.py [--selftest]\n"
              "This module is a library plus its parity battery; there is "
              "no run mode and no training-row output path (see header).")
        return 2
    return selftest()


if __name__ == "__main__":
    import sys
    sys.exit(main())
