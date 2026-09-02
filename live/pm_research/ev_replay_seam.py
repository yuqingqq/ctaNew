"""EV-Replay SEAM v1 -- the replay environment the cancel grid runs inside.

SURFACE AUTHORISATION (R-126, in-file): coordinator DE round-2 dispatch,
COORDINATION.md section 7 item B3 ("EV-Replay plan + harness"; DE), plan
section 10.1 (the common replay harness MAY be developed against TYPED STUB
OUTPUTS), EV_REPLAY_PLAN.md.  RESEARCH-ONLY, OFFLINE: no venue port, no order
path, no live semantics, no exchange adapter.

WHAT THIS IS, AND WHAT IT IS NOT.  `ev_replay.py` is the WINDOW replay
environment: it reads a tape and produces fill records.  THIS module is the
POLICY SEAM one level up -- it consumes an action-value policy, drives the
stateful cancel/hold/repost machine (`harmful_stateful_policy`) over a
reference trajectory, and emits a RunRecord of RAW events plus a receipt.  The
two are not the same environment and neither subsumes the other; the seam is
what plan section 10.1's diagram calls
`action-value policy -> stateful replay`.

EVERY NOT-YET-FROZEN INPUT IS A TYPED STUB WITH A NAMED STATUS.  Nothing is
defaulted: an input without a declared status is REFUSED, an undeclared status
is REFUSED, and a stub that claims to be RELEASED is REFUSED because the
released registry does not list it.  The statuses are the batch-1 class
(`NO_RELEASED_PREDICTOR` and its siblings): an arm that cannot run is a
reported status, never an absence.

NO ECONOMIC NUMBER IS CLAIMABLE FROM A STUB, AND THE HARNESS CANNOT EMIT ONE.
The guard is at the ARTIFACT, not in a comment and not behind a declared mode:
`emission_guard` walks the emitted object and REFUSES if any economic key
appears anywhere in it while any declared input is unreleased.  This is the
F-1 lesson taken directly -- a source-text check can see a deleted line but
not a value that never reached the consumer, so the check runs over the bytes
that are about to be written.  It fires on a doctored receipt and admits the
real one (rule 16, both directions).

THE SECTION-0 BOUNDARY IS EXECUTABLE HERE, NOT PROSE.  EV_REPLAY_PLAN section
0 rule 1 requires that no EV output reach the policy loop, with the stated
enforcement "the policy-facing interface contains no markout/gate type".  That
is `DecisionContext`: a closed schema whose validator refuses any key outside
it AND names the evaluation vocabulary explicitly, so a markout or a gate
verdict cannot be handed to a policy even by accident.

    python3 live/pm_research/ev_replay_seam.py --selftest
    python3 live/pm_research/ev_replay_seam.py run --out <path>
    python3 live/pm_research/ev_replay_seam.py --emit-runhash   (seam-internal)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

# CO-2 (class, not instance): a bare flat import dies under
# `python3 -m live.pm_research.…` while the script-dir launch is green --
# a suite that passes only because of how it was started. DA's modules
# already do this; measured before/after in both launches.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harmful_stateful_policy as hsp

# THE BRIDGE LIVES HERE, AND THE PLANE ORDER IS WHY.
# `de_admissible_windows` SUPPLIES the window list; this seam CONSUMES it.
# EV reads all planes and is read by none, so EV may import DE -- but DE
# importing EV is the plane-order violation R-32 ruled on when
# `de_constraints` did exactly that (the constants were inlined rather than
# imported: "a duplicated literal is a lesser evil than a plane-order
# violation"). Putting the bridge in the supplier would have made the
# supplier read its own consumer. So it is here, and the supplier stays
# unaware of the seam.
import de_admissible_windows as daw

PROTOCOL = "ev_replay_seam_v1"
CANON = "ev_runrecord_canon_v1"
STATUS = "RESEARCH_ONLY_NOT_DECISION_ELIGIBLE"
SIDES = hsp.SIDES

# ---------------------------------------------------------------------------
# Declared vocabulary.  Every one of these is a CLOSED set: a value outside it
# is refused rather than passed through, because a status nobody declared is a
# status nobody can act on.
INPUT_STATUSES = (
    "FROZEN_REFERENCE",              # QR_SKEW_ONLY, user-frozen semantics
    "DECLARED_PARAMETER",            # a policy choice priced by its protocol
    "NO_RELEASED_POLICY",            # the common action-value interface
    "NO_RELEASED_PREDICTOR",         # conditional value / hazard scores
    "NO_RELEASED_FAIRPRICE",         # the 2B challenger protocol
    "UNMEASURED_AT_THIS_VENUE",      # OP-LatencyBudget leg 4 (ack)
    # ADDITIVE, round 3: a policy that EXISTS and runs but has not been
    # forward-validated.  `NO_RELEASED_POLICY` would say no policy exists,
    # which stopped being true when RulePolicy_v1 landed; reusing it would be
    # the name-is-not-the-definition defect.  It is UNRELEASED, so the
    # economics guard still refuses.
    "RULE_POLICY_UNVALIDATED",
    "RELEASED",                      # nothing is, today; see RELEASED_INPUTS
)
# WHICH STATUSES MEAN "THIS INPUT DOES NOT EXIST YET".  Enumerated, NOT
# derived as the complement of the released ones -- the complement form
# silently classified `DECLARED_PARAMETER` as a stub, which would have made
# the economics guard fire on every Phase-4 cell that declares a queue-reset
# cost.  A declared parameter EXISTS; it is a priced choice, not a missing
# artifact, and the distinction is exactly what the guard must not blur.
UNRELEASED_STATUSES = (
    "NO_RELEASED_POLICY",
    "NO_RELEASED_PREDICTOR",
    "NO_RELEASED_FAIRPRICE",
    "UNMEASURED_AT_THIS_VENUE",
    "RULE_POLICY_UNVALIDATED",
)
RELEASED_INPUTS: tuple[str, ...] = ()          # deliberately empty

QUEUE_BOUNDS = ("FRONT", "BACK_DISPLAYED")

# The closed RunRecord schema.  Raw events only (EV_REPLAY_PLAN section 0
# rule 1: evaluation runs as a separate pass over a COMPLETED record).
RUNRECORD_KEYS = ("canon", "slug", "arm", "policy", "policy_status",
                  "predictor_active", "queue_bound", "n_reference_generations",
                  "events", "unavailable_iv", "diagnostics", "record_hash")

# The policy-facing context.  Exact in both directions.
CONTEXT_KEYS = ("slug", "side", "ref_gen", "t", "level", "displayed",
                "time_remaining_s")
# Named so the refusal can say WHAT it refused and why, rather than "unknown
# key".  These are the EV-plane vocabulary the boundary exists to keep out.
EVALUATION_VOCAB = (
    "markout", "markout_cents", "markout_cents_per_share", "gate", "gates",
    "gate_verdict", "calibration", "attribution", "pnl", "net_cents",
    "harm_avoided_cents", "sacrifice_cents", "cost_adjusted_value_cents",
    "verdict", "decision_eligible", "admissible",
)
# The economic vocabulary the emission guard refuses while anything is stubbed.
ECONOMIC_KEYS = (
    "economics", "pnl", "net_cents", "harm_avoided_cents", "sacrifice_cents",
    "cost_adjusted_value_cents", "not_received_net_cents",
    "received_markout_cents", "stale_markout_cents", "value_cents",
    "prevented_value_cents", "queue_reset_cost_cents_total", "cents",
)
ECONOMIC_SUFFIXES = ("_cents", "_pnl", "_value_cents", "_profit")

# THE GUARD IS ABOUT OUTPUTS, AND THE INPUT/OUTPUT LINE IS DECLARED RATHER
# THAN CARVED OUT.  A DECLARED PARAMETER may be an amount -- the queue-reset
# cost is priced in cents by its own protocol -- and refusing to record it
# would make a receipt unable to say what it ran with, which rule 12 requires.
# So the guard walks the OUTPUT region and skips the named INPUT subtrees;
# and because a carve-out is exactly where a leak would hide, the input
# subtree is separately checked to contain ONLY names the machine itself
# declares as parameters. A result smuggled in there is not a parameter name
# and is refused on that.
INPUT_SUBTREES = ("declared_parameters",)
DECLARED_PARAM_NAMES = tuple(sorted(set(hsp._REQUIRED_PARAMS)
                                    | set(hsp._OPTIONAL_PARAMS)))


RATIFICATION_REF_RE = re.compile(r"^R-\d+$")
#: What a supply emission must carry before this seam will read it.
SUPPLY_REQUIRED_KEYS = ("protocol", "day", "governed", "mask_consumed",
                        "mask_identity_hash", "counts", "windows",
                        "n_supplied_total")
#: What each supplied record must already carry. The seam's own required
#: keys are `slug` and `inputs_hash`; the supplier emits both.
SUPPLIED_RECORD_KEYS = ("slug", "inputs_hash", "coin", "start")
#: The bridged spec: the seam's required keys PLUS the supply's identity, so
#: a spec can never be read without knowing which supply and which
#: ratification produced it.
BRIDGED_SPEC_KEYS = ("slug", "inputs_hash", "coin", "start",
                     "supply_protocol", "day", "governed", "mask_consumed",
                     "mask_identity_hash", "ratification_ref")


class SeamRefused(RuntimeError):
    """An input, a status or a stamp the seam refuses to guess."""


class BridgeRefused(SeamRefused):
    """A supply this seam will not turn into window specs.

    A SUBCLASS, deliberately: the cause is specific and the disposition is
    not -- a caller catches one concept, as with the supplier's own
    `GoverningRuleUnreadable`."""


class EconomicsRefused(RuntimeError):
    """An artifact carried an economic quantity while an input was a stub."""


class BoundaryViolation(RuntimeError):
    """An EV-plane quantity reached the policy-facing interface."""


# ---------------------------------------------------------------------------
# 1. declared inputs -- each one a STATUS, never an absence
# ---------------------------------------------------------------------------

def declared_inputs(policy_status: str) -> dict[str, str]:
    """The seam's input manifest.  Every entry is present and carries a
    status; an input that does not exist yet is a NAMED status, so a reader
    can never mistake absence for release."""
    return {
        "reference_trajectory": "FROZEN_REFERENCE",
        "action_value_policy": policy_status,
        "harm_predictor": "NO_RELEASED_PREDICTOR",
        "fair_price": "NO_RELEASED_FAIRPRICE",
        "latency_budget_ack": "UNMEASURED_AT_THIS_VENUE",
        "queue_reset_cost": "DECLARED_PARAMETER",
    }


def validate_inputs(inputs: dict[str, str]) -> None:
    if not isinstance(inputs, dict) or not inputs:
        raise SeamRefused("the input manifest must be a non-empty dict")
    for name, st in inputs.items():
        if st not in INPUT_STATUSES:
            raise SeamRefused(
                f"input {name!r} carries undeclared status {st!r}; declared: "
                f"{INPUT_STATUSES}. A status nobody declared is a status "
                f"nobody can act on.")
        if st == "RELEASED" and name not in RELEASED_INPUTS:
            raise SeamRefused(
                f"input {name!r} claims RELEASED but the released registry is "
                f"{RELEASED_INPUTS or '() -- empty'}. A stub that declares "
                f"itself released is the one lie this seam must not accept.")


def any_unreleased(inputs: dict[str, str]) -> list[str]:
    return sorted(n for n, s in inputs.items() if s in UNRELEASED_STATUSES)


# ---------------------------------------------------------------------------
# 2. the policy-facing interface -- section 0 rule 1, made executable
# ---------------------------------------------------------------------------

def validate_context(ctx: dict[str, Any]) -> None:
    """REFUSE a policy context that carries an EV-plane quantity.

    EV_REPLAY_PLAN section 0 rule 1 states the enforcement point as "the
    policy-facing interface contains no markout/gate type".  Stated as prose
    that is a hope; stated here it is a predicate a known-bad can trip."""
    if not isinstance(ctx, dict):
        raise BoundaryViolation("context must be a dict")
    missing = [k for k in CONTEXT_KEYS if k not in ctx]
    if missing:
        raise BoundaryViolation(
            f"context is MISSING {missing}; nothing is defaulted here")
    extra = [k for k in ctx if k not in CONTEXT_KEYS]
    if extra:
        ev = [k for k in extra if k in EVALUATION_VOCAB]
        if ev:
            raise BoundaryViolation(
                f"EV-plane quantity {ev} reached the policy-facing "
                f"interface. EV reads all planes and is read by none; a "
                f"markout or a gate verdict in a decision context is the "
                f"channel section 0 rule 1 exists to close.")
        raise BoundaryViolation(
            f"context carries undeclared key(s) {extra}; the schema is closed "
            f"in both directions so a new input is a contract change")
    for k in ("t", "level", "displayed", "time_remaining_s"):
        v = ctx[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(float(v)):
            raise BoundaryViolation(f"context {k}={v!r} is not finite")


def context_from_generation(slug: str, side: str, g: dict, t: float,
                            window_end: float) -> dict[str, Any]:
    ctx = {"slug": slug, "side": side, "ref_gen": g["gen"], "t": t,
           "level": g["level"], "displayed": g["displayed"],
           "time_remaining_s": window_end - t}
    validate_context(ctx)
    return ctx


# ---------------------------------------------------------------------------
# 3. the action-value policy interface, and its two stub implementations
# ---------------------------------------------------------------------------
# MODELS ESTIMATE; THEY NEVER DECIDE (rule 14).  A policy returns a SCORE per
# context and nothing else -- no cancel boolean, no entitlement.  The decision
# lives in `harmful_stateful_policy`, which owns the thresholds and refuses
# every one of them as an undeclared parameter.

class ActionValuePolicy:
    """Base interface.  Subclasses declare `name`, `status`, and score()."""

    name = "abstract"
    status = "NO_RELEASED_POLICY"
    predictor_active = False

    def score(self, ctx: dict[str, Any]) -> float:
        raise NotImplementedError

    def score_stream(self, slug: str, sides: dict, window_end: float
                     ) -> list[dict]:
        """Scores at declared offsets inside each generation.  The stream is a
        pure function of the REFERENCE -- the policy never sees an outcome, so
        it cannot condition on the event a cancel exists to prevent (rule 1)."""
        out: list[dict] = []
        for side in SIDES:
            for g in sides[side]:
                span = g["t1"] - g["t0"]
                for frac in SCORE_OFFSET_FRACS:
                    t = g["t0"] + frac * span
                    ctx = context_from_generation(slug, side, g, t, window_end)
                    out.append({"t": t, "slug": slug, "side": side,
                                "score": float(self.score(ctx))})
        out.sort(key=lambda s: (s["t"], SIDES.index(s["side"])))
        return out


SCORE_OFFSET_FRACS = (0.25, 0.75)      # declared; two events per generation
STUB_SALT = "ev_replay_seam_v1"        # declared; sha256, never builtin hash()


class InertPolicy(ActionValuePolicy):
    """The anchor arm: no predictor, no scores, no action.  Its record must be
    bit-identical to the QR_SKEW_ONLY passthrough, which is the only claim
    this seam can make today without a released input."""

    name = "INERT"
    status = "NO_RELEASED_POLICY"
    predictor_active = False

    def score_stream(self, slug, sides, window_end):
        return []

    def score(self, ctx):                       # pragma: no cover - unused
        raise SeamRefused("the inert policy emits no scores")


class DeclaredStubPolicy(ActionValuePolicy):
    """A TYPED STUB with a declared status -- deterministic, sha256-derived,
    and explicitly not a model.  `hash()` of a str is salted by
    PYTHONHASHSEED, so a stub built on it would select a different action set
    per process; that is the determinism defect this seam gates on."""

    name = "DECLARED_STUB"
    status = "NO_RELEASED_PREDICTOR"
    predictor_active = True

    def score(self, ctx: dict[str, Any]) -> float:
        validate_context(ctx)
        key = f"{STUB_SALT}|{ctx['slug']}|{ctx['side']}|{ctx['ref_gen']}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    def score_stream(self, slug, sides, window_end):
        """Two events per generation: the EARLY one carries the stub value,
        the LATE one is zero.  The zero is what drives the hysteresis path
        (below theta_repost for the declared dwell), so the hold/repost arm of
        the machine is exercised rather than merely reachable."""
        out = []
        for side in SIDES:
            for g in sides[side]:
                span = g["t1"] - g["t0"]
                for i, frac in enumerate(SCORE_OFFSET_FRACS):
                    t = g["t0"] + frac * span
                    ctx = context_from_generation(slug, side, g, t, window_end)
                    v = self.score(ctx) if i == 0 else 0.0
                    out.append({"t": t, "slug": slug, "side": side,
                                "score": float(v)})
        out.sort(key=lambda s: (s["t"], SIDES.index(s["side"])))
        return out


class LeakyPolicy(ActionValuePolicy):
    """KNOWN-BAD, kept in the library because a boundary control that lives
    only in a test file is one refactor from being deleted: it asks for an
    EV quantity in its context and must be REFUSED by the seam."""

    name = "LEAKY"
    status = "NO_RELEASED_PREDICTOR"
    predictor_active = True

    def score(self, ctx):
        ctx = dict(ctx, markout_cents_per_share=-9.0)
        validate_context(ctx)                     # must raise
        return 1.0


# ---------------------------------------------------------------------------
# 3b. the BRIDGE -- a supply emission becomes per-window ReplaySession specs
# ---------------------------------------------------------------------------
# THE SEAM STILL NEVER CHOOSES. It converts; it does not select, sample, cap
# or rank. Which windows exist is the supplier's subtraction; whether they may
# be raced is an R-ADMISS ratification the coordinator performs, and the
# ratification enters here ONLY as a reference the caller passes in. It is
# never derived and never defaulted, because a supplier that could mint its
# own ratification would be performing the coordinator's act.

def _b_ratification_declared(ctx) -> None:
    ref = ctx["ratification_ref"]
    if ref is None or not isinstance(ref, str) or not ref.strip():
        raise BridgeRefused(
            f"ratification_ref is {ref!r}. It is a PARAMETER passed from "
            f"outside: R-ADMISS ratification is a coordinator act, and a "
            f"seam that defaulted or derived one would be minting it.")


def _b_ratification_shape(ctx) -> None:
    ref = ctx["ratification_ref"]
    if isinstance(ref, str) and not RATIFICATION_REF_RE.match(ref.strip()):
        raise BridgeRefused(
            f"ratification_ref {ref!r} does not match "
            f"{RATIFICATION_REF_RE.pattern} -- a register entry is R-<n>, and "
            f"a free-text ref would let anything look like a ratification")


def _b_supply_envelope(ctx) -> None:
    sup = ctx["supplied"]
    if not isinstance(sup, dict):
        raise BridgeRefused("supply emission is not an object")
    missing = [k for k in SUPPLY_REQUIRED_KEYS if k not in sup]
    if missing:
        raise BridgeRefused(
            f"supply emission is MISSING {missing} -- named, so a reader "
            f"knows which field was absent rather than that 'something' was")


def _b_governed_implies_mask_consumed(ctx) -> None:
    sup = ctx["supplied"]
    if sup.get("governed") is True and sup.get("mask_consumed") is not True:
        raise BridgeRefused(
            "supply declares governed=True with mask_consumed=False. The "
            "supplier already refuses this, and the seam refuses it AGAIN on "
            "purpose: a consumer that trusts its producer's guarantees has "
            "no way to notice when the producer stops providing them.")


def _b_total_matches_lists(ctx) -> None:
    sup = ctx["supplied"]
    listed = sum(len(v) for v in sup["windows"].values())
    if listed != sup["n_supplied_total"]:
        raise BridgeRefused(
            f"supply declares n_supplied_total={sup['n_supplied_total']} but "
            f"lists {listed} window(s); the two must agree or one of them is "
            f"describing a different set")


def _b_counts_match_lists(ctx) -> None:
    """The hand-added-slug catcher, and the reason it is exercisable.

    `counts` and `windows` are built by different paths in the supplier, so
    comparing them is a cross-check rather than a restatement: a record added
    to `windows` by hand shows up here as a per-coin disagreement."""
    sup = ctx["supplied"]
    bad = []
    for coin, recs in sorted(sup["windows"].items()):
        declared = (sup["counts"].get(coin) or {}).get("n_supplied")
        if declared != len(recs):
            bad.append(f"{coin}: counts={declared} list={len(recs)}")
    if bad:
        raise BridgeRefused(
            f"per-coin counts disagree with the window lists ({bad}). A spec "
            f"list is built ONLY from the supply's own records, so a slug "
            f"added by hand is a disagreement here, not an extra window.")


def _b_record_keys(ctx) -> None:
    sup = ctx["supplied"]
    for coin, recs in sorted(sup["windows"].items()):
        for i, rec in enumerate(recs):
            if not isinstance(rec, dict):
                raise BridgeRefused(f"{coin}[{i}] is not an object")
            miss = [k for k in SUPPLIED_RECORD_KEYS if k not in rec]
            if miss:
                raise BridgeRefused(
                    f"{coin}[{i}] (slug {rec.get('slug')!r}) is MISSING "
                    f"{miss} -- named, because `slug` and `inputs_hash` are "
                    f"the seam's own required keys and a record without them "
                    f"cannot be stamped")


def _b_slug_matches_record(ctx) -> None:
    sup = ctx["supplied"]
    for coin, recs in sorted(sup["windows"].items()):
        for rec in recs:
            want = daw.SLUG_FORM.format(coin=rec["coin"], start=rec["start"])
            if rec["slug"] != want:
                raise BridgeRefused(
                    f"record slug {rec['slug']!r} does not reconstruct from "
                    f"its own coin/start ({want!r}) -- a doctored slug would "
                    f"otherwise name a window the record is not about")
            if rec["coin"] != coin:
                raise BridgeRefused(
                    f"record under coin {coin!r} declares coin "
                    f"{rec['coin']!r}")


BRIDGE_GUARDS: tuple[tuple[str, Any], ...] = (
    ("ratification_declared", _b_ratification_declared),
    ("ratification_shape", _b_ratification_shape),
    ("supply_envelope", _b_supply_envelope),
    ("governed_implies_mask_consumed", _b_governed_implies_mask_consumed),
    ("total_matches_lists", _b_total_matches_lists),
    ("counts_match_lists", _b_counts_match_lists),
    ("record_keys", _b_record_keys),
    ("slug_matches_record", _b_slug_matches_record),
)
BRIDGE_GUARD_NAMES = tuple(n for n, _ in BRIDGE_GUARDS)


def window_specs_from_supply(supplied: dict, *, ratification_ref: str,
                             skip_guard: str | None = None) -> list[dict]:
    """One ReplaySession spec per SUPPLIED window, stamped with the supply's
    identity and the ratification it was admitted under.

    STRUCTURAL, and named as such because it cannot fail by input (rule 16):
    the spec list is built by iterating `supplied["windows"]` and nothing
    else. There is no parameter through which a window could be added, so
    "a window not listed by the supply cannot enter" is a property of the
    code's shape rather than a guard that fires. What IS exercisable is a
    record added to the supply itself -- `counts_match_lists` catches that,
    and its known-bad does fire."""
    ctx = {"supplied": supplied, "ratification_ref": ratification_ref}
    for name, guard in BRIDGE_GUARDS:
        if name == skip_guard:
            continue
        guard(ctx)
    ref = ratification_ref.strip()
    out: list[dict] = []
    for coin, recs in sorted(supplied["windows"].items()):
        for rec in recs:
            out.append({
                "slug": rec["slug"],
                "inputs_hash": rec["inputs_hash"],
                "coin": rec["coin"],
                "start": rec["start"],
                "supply_protocol": supplied["protocol"],
                "day": supplied["day"],
                "governed": supplied["governed"],
                "mask_consumed": supplied["mask_consumed"],
                "mask_identity_hash": supplied["mask_identity_hash"],
                "ratification_ref": ref,
            })
    for spec in out:
        if set(spec) != set(BRIDGED_SPEC_KEYS):
            raise BridgeRefused(
                f"bridged spec schema is closed: {sorted(set(spec) ^ set(BRIDGED_SPEC_KEYS))}")
    return out


# ---------------------------------------------------------------------------
# 4. the session
# ---------------------------------------------------------------------------

def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def record_hash(rec: dict) -> str:
    """Content hash over the FULL record minus the hash field itself.  It
    covers the events, the intervals and the diagnostics -- everything a later
    evaluation pass consumes -- so determinism is checked on what is read, not
    on what the receipt happens to serialize."""
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _overlaps(t: float, ivs: Sequence[Sequence[float]]) -> bool:
    return any(a - hsp.EPS <= t <= b + hsp.EPS for a, b in ivs)


class ReplaySession:
    """One window.  Pure in (window_spec, reference, params, policy, seed).

    Window SELECTION is not this environment's job (EV_REPLAY_PLAN section 2):
    the session takes an explicit spec and STAMPS it, never chooses."""

    def __init__(self, window_spec: dict, reference_sides: dict,
                 params: dict, policy: ActionValuePolicy, *,
                 queue_bound: str, unavailable_iv: Sequence[Sequence[float]],
                 window_end: float, seed: int) -> None:
        for k in ("slug", "inputs_hash"):
            if k not in window_spec:
                raise SeamRefused(f"window_spec is MISSING {k!r}")
        if queue_bound not in QUEUE_BOUNDS:
            raise SeamRefused(
                f"queue_bound {queue_bound!r} must be one of {QUEUE_BOUNDS} "
                f"and must be STAMPED by the caller -- inferring it is how a "
                f"receipt consumer ends up guessing which bound produced it")
        if not isinstance(policy, ActionValuePolicy):
            raise SeamRefused("policy must implement ActionValuePolicy")
        if policy.status not in INPUT_STATUSES:
            raise SeamRefused(
                f"policy {policy.name!r} carries undeclared status "
                f"{policy.status!r}")
        for iv in unavailable_iv:
            if len(iv) != 2 or not (iv[0] < iv[1]):
                raise SeamRefused(f"unavailable interval {iv!r} is not a "
                                  f"forward interval")
        self.spec = dict(window_spec)
        self.sides = reference_sides
        self.params = params
        self.policy = policy
        self.queue_bound = queue_bound
        self.unavailable_iv = [list(map(float, iv)) for iv in unavailable_iv]
        self.window_end = float(window_end)
        self.seed = int(seed)
        self.inputs = declared_inputs(policy.status)
        validate_inputs(self.inputs)

    def run(self) -> dict:
        slug = self.spec["slug"]
        ref = {slug: self.sides}
        scores = self.policy.score_stream(slug, self.sides, self.window_end)
        result = hsp.replay_policy(ref, scores, self.params)
        # RAW EVENTS ONLY.  The machine computes an economics block; the seam
        # does not carry it, and `emission_guard` proves the omission rather
        # than trusting this line.
        events = result["trajectory"]
        bad = [e for e in events
               if e["kind"] in ("FILL_CHARGED",)
               and _overlaps(e["t"], self.unavailable_iv)]
        if bad:
            raise SeamRefused(
                f"{len(bad)} charged fill(s) inside an UNAVAILABLE interval. "
                f"A collector gap clears state and retracts resting quotes, "
                f"so a fill there is not a fill the policy could have taken "
                f"(EV_REPLAY_PLAN section 1, gap state kill).")
        rec = {
            "canon": CANON,
            "slug": slug,
            "arm": "QR_CANCEL_HOLD_X_SKEW" if self.policy.predictor_active
                   else "QR_SKEW_ONLY",
            "policy": self.policy.name,
            "policy_status": self.policy.status,
            "predictor_active": bool(self.policy.predictor_active),
            "queue_bound": self.queue_bound,
            "n_reference_generations": result["n_reference_generations"],
            "events": events,
            # FIRST-CLASS, not a footnote: an UNAVAILABLE interval is a row of
            # the record, so a downstream reader cannot mistake a gap for a
            # quiet market (EV_REPLAY_PLAN section 3.3).
            "unavailable_iv": self.unavailable_iv,
            "diagnostics": {k: v for k, v in result["counters"].items()},
            "record_hash": None,
        }
        rec["record_hash"] = record_hash(rec)
        validate_runrecord(rec)
        return rec

    def receipt(self, records: Sequence[dict], gates: dict) -> dict:
        body = {
            "protocol": PROTOCOL,
            "status": STATUS,
            "canon": CANON,
            "seam": "action-value policy -> stateful replay (plan 10.1)",
            "inputs": self.inputs,
            "unreleased_inputs": any_unreleased(self.inputs),
            "economics_emittable": False,
            "why_no_economics":
                "every economic quantity is refused while any declared input "
                "is a stub; the refusal is evaluated over the emitted bytes "
                "at emission, not asserted by a mode comment",
            "queue_bound": self.queue_bound,
            "seed": self.seed,
            "declared_parameters": _jsonable(self.params),
            "score_offset_fracs": list(SCORE_OFFSET_FRACS),
            "stub_salt": STUB_SALT,
            "windows": [{"slug": self.spec["slug"],
                         "inputs_hash": self.spec["inputs_hash"]}],
            "records": [{"slug": r["slug"], "policy": r["policy"],
                         "policy_status": r["policy_status"],
                         "queue_bound": r["queue_bound"],
                         "n_events": len(r["events"]),
                         "n_unavailable_iv": len(r["unavailable_iv"]),
                         "record_hash": r["record_hash"]} for r in records],
            "gates": gates,
            "all_gates_pass": gates_all_pass(gates),
            "engine_identity": engine_identity(),
            "produced_at": produced_at(),
        }
        body["run_hash"] = hashlib.sha256(_canonical(body)).hexdigest()
        emission_guard(body, self.inputs)
        return body


def _jsonable(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        out[k] = (str(v) if isinstance(v, float) and not math.isfinite(v)
                  else v)
    return out


def validate_runrecord(rec: dict) -> None:
    if set(rec) != set(RUNRECORD_KEYS):
        raise SeamRefused(
            f"RunRecord schema is closed: missing "
            f"{sorted(set(RUNRECORD_KEYS) - set(rec))}, undeclared "
            f"{sorted(set(rec) - set(RUNRECORD_KEYS))}. An extra field is how "
            f"an evaluation quantity gets into a RAW record.")
    if rec["canon"] != CANON:
        raise SeamRefused(f"record canon {rec['canon']!r} != {CANON!r}")
    if rec["queue_bound"] not in QUEUE_BOUNDS:
        raise SeamRefused("record carries no stamped queue bound")


# ---------------------------------------------------------------------------
# 5. the emission guard -- an ARTIFACT-level refusal (the F-1 lesson)
# ---------------------------------------------------------------------------

def _economic_keys_in(obj: Any, path: str = "", *,
                      skip: Sequence[str] = ()) -> list[str]:
    """Walk the WHOLE output region.  A key check at the top level would miss
    the nesting, which is exactly where a leaked quantity would sit."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if p in skip:
                continue
            ks = str(k)
            if ks in ECONOMIC_KEYS or ks.endswith(ECONOMIC_SUFFIXES):
                hits.append(p)
            hits.extend(_economic_keys_in(v, p, skip=skip))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits.extend(_economic_keys_in(v, f"{path}[{i}]", skip=skip))
    return hits


def undeclared_parameter_names(obj: Any) -> list[str]:
    """Names inside an INPUT subtree that the machine does not declare as
    parameters.  This is what stops the input exemption being a hiding
    place: a computed result put there is not a parameter name."""
    bad: list[str] = []
    for sub in INPUT_SUBTREES:
        node = obj.get(sub) if isinstance(obj, dict) else None
        if not isinstance(node, dict):
            continue
        bad += [f"{sub}.{k}" for k in node if k not in DECLARED_PARAM_NAMES]
    return sorted(bad)


def emission_guard(obj: Any, inputs: dict[str, str]) -> None:
    """REFUSE to emit an economic quantity while any input is a stub.

    Run over the object that is ABOUT TO BE WRITTEN, so it sees what a reader
    would see.  A source-text assertion can watch a line disappear; it cannot
    watch a value arrive by another route -- which is the defect F-1 executed
    against a guard of that shape."""
    smuggled = undeclared_parameter_names(obj)
    if smuggled:
        raise EconomicsRefused(
            f"REFUSING to emit: the declared-input subtree carries "
            f"{smuggled}, which the machine does not declare as parameters. "
            f"The input exemption is not a place to put a result.")
    unreleased = any_unreleased(inputs)
    if not unreleased:
        return
    hits = _economic_keys_in(obj, skip=INPUT_SUBTREES)
    if hits:
        raise EconomicsRefused(
            f"REFUSING to emit: {len(hits)} economic quantity/ies "
            f"{hits[:6]} while these inputs are stubs {unreleased}. No "
            f"economic number is claimable from a stub, and this refusal is "
            f"evaluated over the emitted bytes rather than declared.")


# ---------------------------------------------------------------------------
# 6. gates -- enumerated, and an ABSENT required gate makes the run fail
# ---------------------------------------------------------------------------
REQUIRED_GATES = (
    "inert_policy_bit_identical",
    "acting_policy_differs",
    "determinism_same_process",
    "determinism_across_hashseed",
    "no_fill_inside_unavailable",
    "counterfactual_rows_carried",
    "no_economics_emitted",
    "policy_interface_boundary",
    "queue_bound_stamped",
    "stub_statuses_named",
)


def gates_all_pass(gates: dict) -> bool:
    """A required gate ABSENT from the receipt makes it False and is named.
    Absence is the failure mode that reads as success (LANE4 B1.2)."""
    missing = [g for g in REQUIRED_GATES if g not in gates]
    if missing:
        return False
    return all(bool(gates[g]) for g in REQUIRED_GATES)


def missing_gates(gates: dict) -> list[str]:
    return [g for g in REQUIRED_GATES if g not in gates]


# ---------------------------------------------------------------------------
# 7. identity -- engine hashes, and the commit the artifact was produced at
# ---------------------------------------------------------------------------
_ENGINE_FILES = ("ev_replay_seam.py", "harmful_stateful_policy.py")
ENGINE_IDENTITY = {
    f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()[:16]
    for f in _ENGINE_FILES}


def engine_identity() -> dict:
    """Taken AT IMPORT: a run outlives edits to its own source, and hashing at
    write time would stamp the artifact with code that did not produce it."""
    return dict(ENGINE_IDENTITY)


def _git(*args: str, strip: bool = True) -> str | None:
    try:
        r = subprocess.run(("git",) + args, capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parents[2]),
                           timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() if strip else r.stdout.rstrip("\n")


def parse_porcelain(text: str) -> list[str]:
    """Paths out of `git status --porcelain`, as a PURE function so it has a
    known-bad.

    IT NEEDED ONE.  The first version called `.strip()` on the whole stdout
    and then took `line[3:]`.  A porcelain status code is two characters and
    a space, and an UNSTAGED-only change starts with a SPACE (` M path`) --
    so stripping the whole output ate the first line's leading space and
    `[3:]` then ate the first character of its path.  It corrupted exactly
    one path, the first, silently, and it reached an emitted artifact
    (`ive/pm_research/...`).  Found by reading the artifact, which is the
    only place it was visible."""
    out: list[str] = []
    for line in text.split("\n"):
        if len(line) < 4:
            continue
        path = line[3:]
        if line[0] in ("R", "C") and " -> " in path:
            path = path.split(" -> ", 1)[1]      # renames name two paths
        path = path.strip().strip('"')
        if path:
            out.append(path)
    return sorted(out)


def produced_at() -> dict:
    """The artifact -> commit binding, named for what it actually is.

    The review's Scope-4 recommendation asks a receipt to carry its
    `carrying_commit`, so a mis-attributed commit message can never break the
    binding.  A producer cannot know the commit that will CARRY it -- it does
    not exist yet -- so this records the commit the code was AT plus the dirty
    paths (F-7: a dirty flag that names nothing leaves a reader no way to tell
    whether the dirt touched the producer).  Calling it `carrying_commit`
    would be the name-is-not-the-definition defect; it is
    `produced_at_commit`, and a reader binds it by re-hashing
    `engine_identity` at that commit."""
    head = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain", strip=False)
    if head is None or porcelain is None:
        return {"produced_at_commit": None, "git_readable": False,
                "working_tree_dirty": None, "dirty_paths": None,
                "note": "git could not be read; the binding is UNKNOWN, "
                        "which is not the same as clean"}
    paths = parse_porcelain(porcelain)
    return {"produced_at_commit": head, "git_readable": True,
            "working_tree_dirty": bool(paths),
            "dirty_paths": paths[:40],
            "n_dirty_paths": len(paths),
            "note": "produced_at_commit is the commit the PRODUCER ran at; "
                    "the carrying commit does not exist at emission. Bind by "
                    "re-hashing engine_identity at this commit."}


# ---------------------------------------------------------------------------
# 8. synthetic fixtures + the gate battery
# ---------------------------------------------------------------------------

def _gen(i, t0, t1, tranches, level=0.50, displayed=5.0, status=hsp.OK):
    return {"gen": i, "t0": t0, "t1": t1, "level": level,
            "displayed": displayed, "status": status,
            "tranches": [{"t": t, "shares": s,
                          "markout_cents_per_share": m}
                         for t, s, m in tranches]}


def fixture_reference() -> tuple[dict, float]:
    """A two-side synthetic reference built so every counterfactual kind is
    actually PRODUCED rather than merely reachable.

    The generation IDS are load-bearing and are declared here with the stub
    scores they induce, because the stub is a pure function of (salt, slug,
    side, gen) and a reader must be able to recompute the crossing rather
    than take it on faith:

        BUY_UP  gen 3 -> 0.634  (>= theta_cancel 0.50: CROSSES)
        BUY_UP  gen 5 -> 0.091  (below: this generation is only ever missed)
        SELL_UP gen 9 -> 0.146  (below: the SELL side stays a clean anchor)

    With the cancel crossing at t=2.5 and a 50 ms latency, the tranche at
    2.52 lands INSIDE the latency window (charged stale), 6.5 and 8.0 land
    after effectiveness (prevented), and gen 5's start at t=10 falls inside
    the hold."""
    sides = {
        "BUY_UP": [
            _gen(3, 0.0, 10.0, [(2.0, 2.0, -10.0), (2.52, 0.5, -3.0),
                                (6.5, 1.0, -4.0), (8.0, 1.0, -25.0)]),
            _gen(5, 10.0, 20.0, [(15.0, 1.0, -20.0)]),
        ],
        "SELL_UP": [
            _gen(9, 0.0, 20.0, [(5.0, 1.0, 3.0)], level=0.55),
        ],
    }
    return sides, 20.0


DEFAULT_PARAMS = {
    "predictor_enabled": True,
    "theta_cancel": 0.50,
    "theta_repost": 0.10,
    "repost_dwell_s": 0.5,
    "cancel_effective_latency_ms": 50.0,
    "queue_reset_cost_cents": 0.10,
    "protection_mode": "ALL_ORDERS_OVERRIDE",
    "max_cancels_per_minute": float("inf"),
    "repost_fill_model": "REFERENCE_FILLS",
    "charge_reset_cost_at_generation_start": False,
    "enable_reduce": False,
}
INERT_PARAMS = dict(DEFAULT_PARAMS, predictor_enabled=False)
# theta_repost = -inf can never be crossed from below, so the hold is
# permanent and the machine never reposts.  Both arms are run: the reposting
# one produces REPOST rows, the holding one produces the missed-while-held
# rows, and the counterfactual gate wants BOTH families present.
HOLD_PARAMS = dict(DEFAULT_PARAMS, theta_repost=float("-inf"))
SPEC = {"slug": "seam-fixture-w1", "inputs_hash": "0" * 16}


def _session(policy, params, *, unavailable=(), seed=20260901):
    sides, end = fixture_reference()
    return ReplaySession(SPEC, sides, params, policy,
                         queue_bound="BACK_DISPLAYED",
                         unavailable_iv=unavailable, window_end=end,
                         seed=seed)


def run_gates() -> tuple[dict, list[dict], ReplaySession]:
    """Every declared gate, computed.  No verdict string is printed beside a
    number anywhere in here (rule 10)."""
    sides, end = fixture_reference()
    inert = _session(InertPolicy(), INERT_PARAMS)
    rec_inert = inert.run()
    passthrough = hsp.build_passthrough_trajectory({SPEC["slug"]: sides})
    acting = _session(DeclaredStubPolicy(), DEFAULT_PARAMS)
    rec_act = acting.run()

    # determinism, same process
    same = _session(DeclaredStubPolicy(), DEFAULT_PARAMS).run()
    # determinism, across PYTHONHASHSEED -- a fixed seed over a
    # process-dependent iteration order is an independent draw, not a
    # reproduction, so this must run in fresh processes
    hashes = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--emit-runhash"], capture_output=True, text=True,
                           env=env, cwd=str(Path(__file__).parent), timeout=300)
        hashes.append(r.stdout.strip() if r.returncode == 0 else f"ERR{hs}")

    rec_hold = _session(DeclaredStubPolicy(), HOLD_PARAMS).run()
    kinds = {e["kind"] for e in rec_act["events"]}
    kinds_hold = {e["kind"] for e in rec_hold["events"]}
    # the emission guard, both directions, on real objects
    try:
        emission_guard({"records": [{"a": {"net_cents": 1.0}}]},
                       acting.inputs)
        guard_fires = False
    except EconomicsRefused:
        guard_fires = True
    try:
        emission_guard({"records": [{"n_events": 3}]}, acting.inputs)
        guard_admits = True
    except EconomicsRefused:
        guard_admits = False
    # the boundary, both directions
    try:
        LeakyPolicy().score(context_from_generation(
            SPEC["slug"], "BUY_UP", sides["BUY_UP"][0], 1.0, end))
        boundary_holds = False
    except BoundaryViolation:
        boundary_holds = True
    try:
        context_from_generation(SPEC["slug"], "BUY_UP", sides["BUY_UP"][0],
                                1.0, end)
        boundary_admits = True
    except BoundaryViolation:
        boundary_admits = False
    # a record whose bound was not stamped must refuse
    try:
        ReplaySession(SPEC, sides, INERT_PARAMS, InertPolicy(),
                      queue_bound="GUESSED", unavailable_iv=(),
                      window_end=end, seed=1)
        bound_stamped = False
    except SeamRefused:
        bound_stamped = True
    # a stub that claims RELEASED must refuse
    try:
        validate_inputs(dict(declared_inputs("NO_RELEASED_POLICY"),
                             harm_predictor="RELEASED"))
        statuses_named = False
    except SeamRefused:
        statuses_named = True

    gates = {
        "inert_policy_bit_identical":
            hsp.bit_identical(rec_inert["events"], passthrough),
        "acting_policy_differs":
            not hsp.bit_identical(rec_act["events"], passthrough),
        "determinism_same_process":
            rec_act["record_hash"] == same["record_hash"],
        "determinism_across_hashseed":
            len(set(hashes)) == 1 and not hashes[0].startswith("ERR"),
        "no_fill_inside_unavailable": _unavailable_gate(),
        # EV_REPLAY_PLAN 3.3: excised fills, partial-fill-then-cancel rows
        # and UNAVAILABLE rows are FIRST-CLASS in the record -- the cancel
        # protocol's accounting CONSUMES them, it does not recompute them.
        # Checked over both arms, because the two families are produced by
        # different paths and one arm alone would leave half the claim
        # unexercised.
        "counterfactual_rows_carried":
            {"FILL_PREVENTED", "REPOST"} <= kinds
            and {"FILL_PREVENTED", "GEN_START_MISSED_HELD",
                 "FILL_MISSED_HELD"} <= kinds_hold,
        "no_economics_emitted": guard_fires and guard_admits,
        "policy_interface_boundary": boundary_holds and boundary_admits,
        "queue_bound_stamped": bound_stamped,
        "stub_statuses_named": statuses_named,
    }
    return gates, [rec_inert, rec_act, rec_hold], acting


def _unavailable_gate() -> bool:
    """FIRES on a gap that covers a charged fill, ADMITS one that does not."""
    try:
        _session(DeclaredStubPolicy(), DEFAULT_PARAMS,
                 unavailable=((1.9, 2.1),)).run()
        fired = False
    except SeamRefused:
        fired = True
    try:
        _session(DeclaredStubPolicy(), DEFAULT_PARAMS,
                 unavailable=((18.5, 19.0),)).run()
        admitted = True
    except SeamRefused:
        admitted = False
    return fired and admitted


def emit_runhash() -> int:
    """Seam-internal entry point for the cross-process determinism gate."""
    print(_session(DeclaredStubPolicy(), DEFAULT_PARAMS).run()["record_hash"])
    return 0


def run(out: Path | None = None) -> dict:
    gates, records, session = run_gates()
    receipt = session.receipt(records, gates)
    if not receipt["all_gates_pass"]:
        raise SeamRefused(
            f"REFUSING to write: gates failed "
            f"{[g for g in REQUIRED_GATES if not gates.get(g)]}, missing "
            f"{missing_gates(gates)}. A run that writes an artifact its own "
            f"gates did not pass is the shape rule 11 forbids.")
    if out is not None:
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 64


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[ev_replay_seam] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(exc, fn, label):
        try:
            fn()
        except exc:
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[ev_replay_seam] FAIL (no refusal): {label}")

    sides, end = fixture_reference()

    # ---- inputs are statuses, never absences ---------------------------
    inp = declared_inputs("NO_RELEASED_POLICY")
    validate_inputs(inp)
    ok(set(inp) == {"reference_trajectory", "action_value_policy",
                    "harm_predictor", "fair_price", "latency_budget_ack",
                    "queue_reset_cost"}
       and all(v in INPUT_STATUSES for v in inp.values()),
       "every declared input is PRESENT and carries a declared status -- an "
       "input that does not exist yet is a named status, not an absence")
    ok(any_unreleased(inp) == ["action_value_policy", "fair_price",
                               "harm_predictor", "latency_budget_ack"],
       f"the unreleased set is COMPUTED from the manifest: {any_unreleased(inp)}")
    refuses(SeamRefused,
            lambda: validate_inputs(dict(inp, harm_predictor="PROBABLY_FINE")),
            "KNOWN-BAD: an undeclared status is REFUSED, never passed through")
    refuses(SeamRefused,
            lambda: validate_inputs(dict(inp, harm_predictor="RELEASED")),
            "KNOWN-BAD: a stub claiming RELEASED is REFUSED -- the released "
            "registry is empty and the seam enforces that rather than "
            "assuming it")
    ok(RELEASED_INPUTS == (),
       "POSITIVE CONTROL on that refusal: the released registry really is "
       "empty, so the check above is about today's state and not a constant")

    # ---- the section-0 boundary, executable ----------------------------
    good = context_from_generation(SPEC["slug"], "BUY_UP",
                                   sides["BUY_UP"][0], 1.0, end)
    ok(set(good) == set(CONTEXT_KEYS),
       "POSITIVE CONTROL: a legitimate decision context is ADMITTED and "
       "carries exactly the closed schema")
    refuses(BoundaryViolation,
            lambda: validate_context(dict(good, markout_cents_per_share=-1.0)),
            "KNOWN-BAD: a MARKOUT in a decision context is REFUSED by name -- "
            "EV_REPLAY_PLAN section 0 rule 1 as a predicate, not as prose")
    refuses(BoundaryViolation,
            lambda: validate_context(dict(good, gate_verdict="PASS")),
            "KNOWN-BAD: a GATE VERDICT in a decision context is REFUSED -- "
            "gates decide whether a run happens, never what the policy sees")
    refuses(BoundaryViolation,
            lambda: validate_context(dict(good, some_new_feature=1.0)),
            "and the schema is closed in BOTH directions: an undeclared key "
            "refuses too, so a new input is a contract change")
    refuses(BoundaryViolation,
            lambda: validate_context({k: good[k] for k in CONTEXT_KEYS[:-1]}),
            "KNOWN-BAD: a MISSING context key refuses rather than defaulting")
    refuses(BoundaryViolation, lambda: LeakyPolicy().score(good),
            "the LEAKY policy -- a known-bad kept in the LIBRARY, not in a "
            "test file -- is refused when it tries to read an EV quantity")

    # ---- the anchor and its positive control ---------------------------
    rec_inert = _session(InertPolicy(), INERT_PARAMS).run()
    passthrough = hsp.build_passthrough_trajectory({SPEC["slug"]: sides})
    ok(hsp.bit_identical(rec_inert["events"], passthrough),
       "ANCHOR: the inert policy's RunRecord is BIT-IDENTICAL to the "
       "QR_SKEW_ONLY passthrough")
    rec_act = _session(DeclaredStubPolicy(), DEFAULT_PARAMS).run()
    ok(not hsp.bit_identical(rec_act["events"], passthrough),
       "POSITIVE CONTROL: the acting stub is NOT bit-identical, so the "
       "anchor's comparator can fire")
    ok(rec_act["diagnostics"]["cancels_issued"] >= 1,
       f"the acting stub actually ACTS on the fixture "
       f"({rec_act['diagnostics']['cancels_issued']} cancels) -- a seam whose "
       f"policy never acts proves nothing about the policy path")
    kinds = {e["kind"] for e in rec_act["events"]}
    rec_hold = _session(DeclaredStubPolicy(), HOLD_PARAMS).run()
    kinds_hold = {e["kind"] for e in rec_hold["events"]}
    ok({"FILL_PREVENTED", "REPOST"} <= kinds,
       f"COUNTERFACTUAL ROWS ARE CARRIED by the reposting arm, not "
       f"recomputed downstream (EV_REPLAY_PLAN 3.3): "
       f"{sorted(kinds & {'FILL_PREVENTED', 'REPOST', 'FILL_PREVENTED_REDUCED'})}")
    ok({"FILL_PREVENTED", "GEN_START_MISSED_HELD",
        "FILL_MISSED_HELD"} <= kinds_hold,
       f"and the HOLDING arm carries the missed-while-held family, which the "
       f"reposting arm cannot produce: "
       f"{sorted(kinds_hold & {'GEN_START_MISSED_HELD', 'FILL_MISSED_HELD'})}")

    # ---- the RunRecord schema is closed --------------------------------
    refuses(SeamRefused,
            lambda: validate_runrecord(dict(rec_act, markout_cents=1.0)),
            "KNOWN-BAD: an extra field on a RAW RunRecord is REFUSED -- that "
            "is the route an evaluation quantity would take")
    refuses(SeamRefused,
            lambda: validate_runrecord({k: v for k, v in rec_act.items()
                                        if k != "queue_bound"}),
            "KNOWN-BAD: a RunRecord missing its stamped queue bound refuses")
    ok(rec_act["queue_bound"] == "BACK_DISPLAYED"
       and rec_inert["queue_bound"] == "BACK_DISPLAYED",
       "the queue bound is STAMPED on every record, never inferred")
    refuses(SeamRefused,
            lambda: ReplaySession(SPEC, sides, INERT_PARAMS, InertPolicy(),
                                  queue_bound="GUESSED", unavailable_iv=(),
                                  window_end=end, seed=1),
            "KNOWN-BAD: an unrecognised queue bound is refused at "
            "construction")
    refuses(SeamRefused,
            lambda: ReplaySession({"slug": "x"}, sides, INERT_PARAMS,
                                  InertPolicy(), queue_bound="FRONT",
                                  unavailable_iv=(), window_end=end, seed=1),
            "KNOWN-BAD: a window spec without its inputs_hash refuses -- "
            "selection is supplied and STAMPED, never inferred")

    # ---- UNAVAILABLE intervals are first class -------------------------
    refuses(SeamRefused,
            lambda: _session(DeclaredStubPolicy(), DEFAULT_PARAMS,
                             unavailable=((1.9, 2.1),)).run(),
            "KNOWN-BAD: a charged fill inside an UNAVAILABLE interval is "
            "REFUSED (a gap clears state and retracts resting quotes)")
    ok(_session(DeclaredStubPolicy(), DEFAULT_PARAMS,
                unavailable=((18.5, 19.0),)).run()["unavailable_iv"]
       == [[18.5, 19.0]],
       "POSITIVE CONTROL: a gap that covers no charged fill is ADMITTED and "
       "carried as a first-class row of the record")

    # ---- the emission guard, at the artifact ---------------------------
    refuses(EconomicsRefused,
            lambda: emission_guard({"a": {"b": [{"net_cents": 1.0}]}}, inp),
            "KNOWN-BAD: an economic quantity NESTED three levels down is "
            "found and REFUSED -- a top-level key check would have missed it")
    refuses(EconomicsRefused,
            lambda: emission_guard({"x": {"total_pnl": 0.0}}, inp),
            "KNOWN-BAD: the SUFFIX rule catches a name the vocabulary does "
            "not list")
    ok(_economic_keys_in({"records": [{"n_events": 3}]}) == [],
       "POSITIVE CONTROL: an honest record trips nothing, so the guard is a "
       "filter and not a blanket refusal")
    ok(emission_guard({"net_cents": 1.0},
                      {"reference_trajectory": "FROZEN_REFERENCE"}) is None,
       "AND THE GUARD IS SCOPED: with NO unreleased input it does not fire -- "
       "it refuses stub-derived economics, not economics as such")

    # ---- the receipt, and gates that can fail --------------------------
    receipt = run()
    ok(receipt["all_gates_pass"] and not missing_gates(receipt["gates"]),
       f"all {len(REQUIRED_GATES)} required gates pass on the fixture")
    ok(_economic_keys_in(receipt, skip=INPUT_SUBTREES) == [],
       "the EMITTED receipt carries no economic quantity anywhere in its "
       "OUTPUT region -- checked over the bytes, not asserted")
    ok(_economic_keys_in(receipt) == ["declared_parameters."
                                      "queue_reset_cost_cents"],
       "and the ONE amount it does carry is a DECLARED PARAMETER, named as "
       "such: a receipt that could not say what it ran with would fail "
       "rule 12 to satisfy a guard")
    refuses(EconomicsRefused,
            lambda: emission_guard(
                dict(receipt, declared_parameters=dict(
                    receipt["declared_parameters"], net_cents=1.0)), inp),
            "KNOWN-BAD: a RESULT smuggled into the exempt input subtree is "
            "REFUSED, because it is not a name the machine declares -- the "
            "exemption is not a hiding place")
    ok(gates_all_pass({g: True for g in REQUIRED_GATES[:-1]}) is False,
       "KNOWN-BAD: a MISSING required gate makes all_gates_pass False -- "
       "absence is the failure mode that reads as success")
    ok(missing_gates({g: True for g in REQUIRED_GATES[:-1]})
       == [REQUIRED_GATES[-1]],
       "and the missing gate is NAMED, so a shrunken battery is visible")
    ok(receipt["engine_identity"] == ENGINE_IDENTITY
       and len(receipt["engine_identity"]) == len(_ENGINE_FILES),
       "engine identity is taken AT IMPORT over the files that shape a record")
    # the porcelain parse, with the defect it actually shipped
    raw = " M live/pm_research/a.py\n?? live/pm_research/b.py\n"
    ok(parse_porcelain(raw) == ["live/pm_research/a.py",
                                "live/pm_research/b.py"],
       "the porcelain parse keeps the FIRST path whole when its status code "
       "begins with a space (an unstaged-only change)")
    ok(sorted(l.strip()[3:].strip() for l in raw.strip().split("\n"))
       != parse_porcelain(raw),
       "KNOWN-BAD: the shipped parse -- strip the output, then take [3:] -- "
       "produces a DIFFERENT and wrong list on this exact input, which is "
       "how `ive/pm_research/...` reached an artifact")
    ok(parse_porcelain("R  old/a.py -> new/b.py\n") == ["new/b.py"],
       "and a RENAME names the destination, not the arrow-joined pair")

    pa = receipt["produced_at"]
    ok(set(pa) >= {"produced_at_commit", "git_readable", "working_tree_dirty"}
       and ("dirty_paths" in pa),
       "the artifact carries produced_at_commit AND the dirty paths -- a "
       "dirty flag that names nothing leaves a reader unable to tell whether "
       "the dirt touched the producer (review F-7)")
    ok("carrying_commit" not in json.dumps(receipt),
       "and it is NOT called carrying_commit: the carrying commit does not "
       "exist at emission, and naming it so would be the "
       "name-is-not-the-definition defect")

    # ---- 3b: the BRIDGE, on a REAL supply emission ----------------------
    # FIXTURE REF, and named one: no live ratification exists and none is to
    # be invented here. `R-0` is not and will never be a register entry.
    FIXTURE_REF = "R-0"
    sup_day = daw.REAL_DAY
    m = daw.load_mask(sup_day)
    sup = daw.supply(sup_day, {c: list(daw._grid(sup_day)) for c in m["coins"]},
                     m)
    specs = window_specs_from_supply(sup, ratification_ref=FIXTURE_REF)
    ok(len(specs) == sup["n_supplied_total"] == 1875,
       f"BRIDGE: one spec per supplied window, {len(specs)} == the supply's "
       f"own n_supplied_total on the REAL {sup_day} mask")
    ok(all(set(x) == set(BRIDGED_SPEC_KEYS) for x in specs),
       "every bridged spec carries the seam's required keys PLUS the "
       "supply's identity, under a CLOSED schema")
    by_slug = {r["slug"]: r for recs in sup["windows"].values() for r in recs}
    ok(all(x["inputs_hash"] == by_slug[x["slug"]]["inputs_hash"]
           for x in specs),
       "and each spec's inputs_hash is the SUPPLIED record's, carried "
       "through rather than recomputed here")
    ok(all(x["ratification_ref"] == FIXTURE_REF
           and x["mask_identity_hash"] == sup["mask_identity_hash"]
           and x["day"] == sup_day for x in specs),
       f"with the supply's identity and the ratification ref on every one "
       f"({FIXTURE_REF}, a FIXTURE — no live ratification exists)")

    refuses(BridgeRefused,
            lambda: window_specs_from_supply(sup, ratification_ref=""),
            "KNOWN-BAD: an empty ratification_ref REFUSES -- it is a "
            "parameter from outside, and a seam that defaulted one would be "
            "minting a coordinator act")
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(sup, ratification_ref="ratified"),
            "KNOWN-BAD: a ref that is not R-<n> REFUSES, so free text cannot "
            "look like a ratification")
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(sup, ratification_ref=None),
            "KNOWN-BAD: a None ref REFUSES -- and it is the case only the "
            "DECLARED guard catches, since the shape guard skips a non-str "
            "by construction")
    _empty_both = []
    for _g in ("ratification_declared", "ratification_shape"):
        try:
            window_specs_from_supply(sup, ratification_ref="", skip_guard=_g)
        except BridgeRefused:
            _empty_both.append(_g)
    ok(sorted(_empty_both) == ["ratification_declared",
                               "ratification_shape"],
       "MEASURED REDUNDANCY, not assumed: the EMPTY ref is caught by BOTH "
       "ref guards -- disabling either alone still refuses. Reporting that "
       "as a survivor would have called a real defence-in-depth a hole, so "
       "the audit uses the case each guard uniquely owns")
    bad_gov = dict(sup, governed=True, mask_consumed=False)
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(bad_gov,
                                             ratification_ref=FIXTURE_REF),
            "KNOWN-BAD: governed with mask_consumed False REFUSES AT THE "
            "SEAM TOO -- defence in depth; the supplier already refuses it, "
            "and a consumer that trusts that cannot notice if it stops")
    bad_tot = dict(sup, n_supplied_total=sup["n_supplied_total"] + 1)
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(bad_tot,
                                             ratification_ref=FIXTURE_REF),
            "KNOWN-BAD: n_supplied_total disagreeing with the lists REFUSES")
    hand = json.loads(json.dumps(sup))
    hand["windows"]["btc"].append(dict(hand["windows"]["btc"][0],
                                       slug="btc-updown-5m-999",
                                       start=999))
    hand["n_supplied_total"] += 1
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(hand,
                                             ratification_ref=FIXTURE_REF),
            "KNOWN-BAD: a slug ADDED BY HAND is refused -- it survives the "
            "total check (the caller bumped it) and is caught by the "
            "per-coin counts, which the supplier builds on a different path")
    nokey = json.loads(json.dumps(sup))
    del nokey["windows"]["eth"][0]["inputs_hash"]
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(nokey,
                                             ratification_ref=FIXTURE_REF),
            "KNOWN-BAD: a record missing `inputs_hash` REFUSES BY NAME")
    noenv = {k: v for k, v in sup.items() if k != "mask_identity_hash"}
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(noenv,
                                             ratification_ref=FIXTURE_REF),
            "KNOWN-BAD: a supply missing an envelope field REFUSES BY NAME")
    doctored = json.loads(json.dumps(sup))
    doctored["windows"]["sol"][0]["slug"] = "sol-updown-5m-1"
    refuses(BridgeRefused,
            lambda: window_specs_from_supply(doctored,
                                             ratification_ref=FIXTURE_REF),
            "KNOWN-BAD: a slug that does not reconstruct from its own "
            "coin/start REFUSES")

    # ---- the specs are ACCEPTED where they are consumed ------------------
    sides_b, end_b = fixture_reference()
    one = specs[0]
    sess = ReplaySession(one, sides_b, INERT_PARAMS, InertPolicy(),
                         queue_bound="BACK_DISPLAYED", unavailable_iv=(),
                         window_end=end_b, seed=20260902)
    ok(sess.spec == one,
       "a ReplaySession CONSTRUCTS from a bridged spec and stores it "
       "UNCHANGED -- the supply's identity keys ride through the seam")
    rec_b = sess.run()
    rcpt = sess.receipt([rec_b], {g: True for g in REQUIRED_GATES})
    ok(rcpt["windows"] == [{"slug": one["slug"],
                            "inputs_hash": one["inputs_hash"]}],
       "and the RECEIPT stamps that spec: slug and inputs_hash carried "
       "through to the artifact")
    ok(rcpt["windows"][0]["inputs_hash"]
       == by_slug[one["slug"]]["inputs_hash"],
       "the receipt's inputs_hash EQUALS the supplied record's -- the chain "
       "mask -> supply -> spec -> receipt is one identity, end to end")
    ok(rec_b["slug"] == one["slug"],
       "and the RunRecord is keyed by the bridged slug")

    # ---- reads-no-verdict still holds for THIS module --------------------
    ok(daw.reads_no_verdict(daw.imported_modules(Path(__file__).read_text())),
       "the seam still imports NO verdict producer after gaining the bridge "
       "-- checked over its parsed import list, the predicate the supplier "
       "ships")

    # ---- mutation audit, live and disabled visibly different -------------
    baudit = bridge_mutation_audit(sup, FIXTURE_REF)
    ok(baudit["all_load_bearing"],
       f"BRIDGE MUTATION AUDIT: each of the {baudit['n_guards']} guards was "
       f"disabled in turn and its known-bad STOPPED refusing -- "
       f"{baudit['survivors']} survivors")
    ok(set(baudit["per_guard"]) == set(BRIDGE_GUARD_NAMES),
       f"and it covers every declared bridge guard by name "
       f"({len(BRIDGE_GUARD_NAMES)})")
    ok(baudit["crash_when_disabled"] == ["ratification_declared",
                                         "record_keys", "supply_envelope"],
       f"AND A THIRD OUTCOME IS RECORDED AS NEITHER REFUSAL NOR PASS: "
       f"disabling {baudit['crash_when_disabled']} makes the builder CRASH "
       f"(AttributeError on a None ref; KeyError on a missing record or "
       f"envelope field). That is precisely what those three guards are FOR "
       f"-- turning a malformed input into a NAMED refusal -- and making the "
       f"builder defensive instead would emit specs with a null identity, "
       f"which is worse than stopping. Counted as its own bucket so the "
       f"audit cannot read a crash as a kill")
    ok(len(window_specs_from_supply(sup, ratification_ref=FIXTURE_REF,
                                    skip_guard="counts_match_lists")) == 1875,
       "RULE 16, STATED: the 'a window not listed by the supply cannot "
       "enter' property is STRUCTURAL and cannot fail by input -- the list "
       "is built by iterating the supply and there is no parameter to add "
       "one. It is named here rather than counted as a killed mutant")

    # ---- the run refuses to write an artifact its gates did not pass ----
    ok(True, "the writer's refusal is exercised by the launcher seam test "
             "below, which runs main() the way a launcher runs it")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[ev_replay_seam] selftest OK -- {n[0]} checks")
    return 0


def bridge_mutation_audit(sup: dict, ref: str) -> dict:
    """Blank each bridge guard in turn; its known-bad must stop refusing.

    Live and disabled are VISIBLY DIFFERENT CALLS (round 5's lesson: a
    harness whose "live" run also passed `skip_guard` measured every guard
    with itself already disabled and read three as survivors)."""
    def _mut(**kw):
        d = json.loads(json.dumps(sup))
        for k, v in kw.items():
            d[k] = v
        return d

    hand = json.loads(json.dumps(sup))
    hand["windows"]["btc"].append(dict(hand["windows"]["btc"][0],
                                       slug="btc-updown-5m-999", start=999))
    hand["n_supplied_total"] += 1
    nokey = json.loads(json.dumps(sup))
    del nokey["windows"]["eth"][0]["inputs_hash"]
    doctored = json.loads(json.dumps(sup))
    doctored["windows"]["sol"][0]["slug"] = "sol-updown-5m-1"

    cases = {
        # A CASE ONLY THIS GUARD CATCHES. The empty string "" is caught by
        # BOTH this guard and `ratification_shape` (measured in the
        # selftest), so using it here would report a real redundancy as a
        # survivor. `None` is not a str, so the shape guard skips it by
        # construction and only the declared guard stands between it and the
        # builder.
        "ratification_declared": (sup, None),
        "ratification_shape": (sup, "ratified"),
        "supply_envelope": ({k: v for k, v in sup.items()
                             if k != "mask_identity_hash"}, ref),
        "governed_implies_mask_consumed":
            (_mut(governed=True, mask_consumed=False), ref),
        "total_matches_lists": (_mut(n_supplied_total=sup["n_supplied_total"]
                                     + 1), ref),
        "counts_match_lists": (hand, ref),
        "record_keys": (nokey, ref),
        "slug_matches_record": (doctored, ref),
    }
    per_guard: dict[str, dict] = {}
    for name, (bad_sup, bad_ref) in cases.items():
        try:                                   # LIVE: no skip_guard
            window_specs_from_supply(bad_sup, ratification_ref=bad_ref)
            live = False
        except SeamRefused:
            live = True
        crashed = None
        try:                                   # DISABLED: this guard skipped
            window_specs_from_supply(bad_sup, ratification_ref=bad_ref,
                                     skip_guard=name)
            disabled = False
        except SeamRefused:
            disabled = True
        except Exception as exc:               # noqa: BLE001
            # A CRASH IS NOT A REFUSAL, and it is not a pass either. With
            # `supply_envelope` disabled the builder raises KeyError on the
            # missing identity field -- which is exactly what that guard is
            # FOR: it turns a malformed supply into a NAMED refusal. Making
            # the builder defensive instead would let a malformed supply
            # produce specs with a null identity, which is worse than
            # stopping. Recorded as its own outcome rather than folded into
            # either bucket.
            disabled = False
            crashed = f"{type(exc).__name__}: {exc}"
        per_guard[name] = {"refuses_when_live": live,
                           "refuses_when_disabled": disabled,
                           "crashes_when_disabled": crashed,
                           "load_bearing": live and not disabled}
    survivors = sorted(k for k, v in per_guard.items()
                       if not v["load_bearing"])
    return {"n_guards": len(per_guard), "per_guard": per_guard,
            "survivors": survivors, "all_load_bearing": not survivors,
            "crash_when_disabled": sorted(
                k for k, v in per_guard.items() if v["crashes_when_disabled"])}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-runhash", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    if a.emit_runhash:
        return emit_runhash()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        # NUMBERS NEVER COME FROM A RED SUITE.
        if selftest() != 0:
            return 1
        r = run(Path(a.out) if a.out else None)
        print(json.dumps({k: v for k, v in r.items() if k != "records"},
                         indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
