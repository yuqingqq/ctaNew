"""RulePolicy_v1 -- the registered rule solver, and the registry that checks it.

SURFACE AUTHORISATION (R-126, in-file): coordinator DE round-3 dispatch;
COORDINATION.md section 7 ("RulePolicy_v1 exists precisely so the DE plane is
not idle behind these"); DE_MODULE_PLAN.md Rev 3 sections 3.1/3.2/3.4/6.1;
DE_PLACEMENT_POLICY_PLAN.md Rev 4 sections 3/6/7.  RESEARCH-ONLY, OFFLINE: no
venue port, no order path, no live semantics.

WHAT IT IS.  The composed policy the placement plan section 3 describes --
where to rest, when to leave, when to cross, when to stop -- as DECLARATIVE
rules over the typed action menu, registered as a solver plugin beside the
named-but-empty seams `ClosedFormGLFT`, `PerLevel`, `HJBQVI`.  It consumes NO
belief, competition, outcome, incentive or coupling input, and its manifest is
exactly (view, self, actions, portfolio, risk_scenarios, constraints,
horizon).

THE MANIFEST IS CHECKED AT REGISTRATION, BY REVELATION, NOT BY ASSERTION.
DE_MODULE_PLAN section 3.1's R-42 upgrade, shipped here WITH the wiring: the
registry builds a `DecisionProblem` whose NON-manifest fields are SENTINEL
objects that record any access, runs the solver on it, and REFUSES to
register if a sentinel was touched.  The check does not ask the policy what it
consumes; it makes the policy reveal it -- a declared manifest would otherwise
let the solver certify its own boundary, which is the self-certification class
R-42 names.

TWO LAYERS OF REVELATION, because one is not enough.  The sentinel VALUES
catch use; the problem CONTAINER also records which KEYS were read, because a
bare `problem["belief"]` that is fetched and discarded touches no sentinel and
would otherwise pass.  Registration refuses on either.

NO ECONOMIC NUMBER IS CLAIMABLE HERE.  RulePolicy_v1 is built, runnable and
UNVALIDATED: it carries the seam status `RULE_POLICY_UNVALIDATED`, which is an
unreleased status, so `ev_replay_seam.emission_guard` refuses any economic
quantity in anything this module emits.

    python3 live/pm_research/rule_policy_v1.py --selftest
    python3 live/pm_research/rule_policy_v1.py run --out <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import de_constraints as dc
import de_actionspace as das
import ev_replay_seam as seam
import harmful_stateful_policy as hsp

PROTOCOL = "rule_policy_v1"

# DE_MODULE_PLAN section 3.1, verbatim and load-bearing: "its manifest is
# exactly (view, self, actions, portfolio, risk_scenarios, constraints,
# horizon), and that list is load-bearing ... it cannot silently grow".
MANIFEST = ("view", "self", "actions", "portfolio", "risk_scenarios",
            "constraints", "horizon")
# The inputs the solver does NOT consume.  Section 3.4 completed this list
# (iteration 3 added `coupling`).
NON_MANIFEST = ("belief", "competition", "outcomes", "incentives", "coupling")
PROBLEM_FIELDS = MANIFEST + NON_MANIFEST

# Section 7's terminal schedule.  `r = 60` is an SP-owned parameter with
# provisional provenance (the notional peak was located on the withdrawn
# uniform grid), so it is a DECLARED PARAMETER here and never a literal in a
# rule body.
R_CUT_S = 60.0
R_DECISION_BAND_S = 5.0        # the width of "r is approximately 60"

# Section 6's dump threshold.  `gamma` is a risk-appetite choice reported
# across a ladder; `0.07` and `h` are SP-Venue facts read through handles.
# N* IS A DIAGNOSTIC -- the plan says so twice -- and the side-aware scenario
# cap binds independently and WINS EVERY CONFLICT.
DUMP_FEE_TERM = 0.07


class ManifestViolation(RuntimeError):
    """A solver touched an input outside its declared manifest."""


class RegistrationRefused(RuntimeError):
    """A registration failed its own checks; nothing is registered."""


class UtilityRefused(RuntimeError):
    """`utility_none` was evaluated.  It exists to be declared, not called."""


class PolicyRefused(RuntimeError):
    """The policy refuses rather than emitting an action it cannot justify."""


# ---------------------------------------------------------------------------
# 1. revelation apparatus
# ---------------------------------------------------------------------------

class AccessSentinel:
    """Records ANY interaction.  Every dunder that a normal consumer would
    reach for is instrumented -- attribute read, item read, iteration, length,
    truthiness, comparison, arithmetic and repr -- because a boundary probe
    that only watches attribute access is a probe with a hole in it."""

    def __init__(self, name: str, log: list) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_log", log)

    def _touch(self, how: str) -> None:
        object.__getattribute__(self, "_log").append(
            f"{object.__getattribute__(self, '_name')}:{how}")

    def __getattr__(self, item):
        self._touch(f"getattr({item})")
        return self

    def __getitem__(self, item):
        self._touch(f"getitem({item})")
        return self

    def __iter__(self):
        self._touch("iter")
        return iter(())

    def __len__(self):
        self._touch("len")
        return 0

    def __bool__(self):
        self._touch("bool")
        return False

    def __eq__(self, other):
        self._touch("eq")
        return False

    def __hash__(self):
        self._touch("hash")
        return 0

    def __float__(self):
        self._touch("float")
        return 0.0

    def __add__(self, other):
        self._touch("add")
        return self

    __radd__ = __add__

    def __repr__(self):
        self._touch("repr")
        return "<AccessSentinel>"


class ProbeProblem(dict):
    """A DecisionProblem that records which KEYS were read.

    The second layer.  A bare `problem["belief"]` fetched and thrown away
    touches no sentinel, so the sentinels alone would call that policy clean.
    Reading the key IS the boundary crossing, whatever the policy then does
    with it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.keys_read: list[str] = []

    def __getitem__(self, k):
        self.keys_read.append(str(k))
        return super().__getitem__(k)

    def get(self, k, default=None):
        self.keys_read.append(str(k))
        return super().get(k, default)


def revelation_probe(solver: "Solver", problem_fields: dict[str, Any]
                     ) -> dict[str, Any]:
    """Run `solver.decide` on a problem whose non-manifest fields are
    sentinels, and REPORT what it touched.  Reports; it does not judge."""
    touched: list[str] = []
    fields = dict(problem_fields)
    for name in NON_MANIFEST:
        fields[name] = AccessSentinel(name, touched)
    problem = ProbeProblem(fields)
    error = None
    try:
        solver.decide(problem)
    except Exception as exc:                      # a crash is a result too
        error = f"{type(exc).__name__}: {exc}"
    return {
        "sentinels_touched": sorted(set(touched)),
        "keys_read": sorted(set(problem.keys_read)),
        "non_manifest_keys_read": sorted(
            set(problem.keys_read) & set(NON_MANIFEST)),
        "manifest_keys_read": sorted(
            set(problem.keys_read) & set(MANIFEST)),
        "decide_error": error,
    }


# ---------------------------------------------------------------------------
# 2. the registry -- checks at REGISTRATION, and registers nothing on failure
# ---------------------------------------------------------------------------

class Solver:
    """Interface.  A solver declares its manifest and emits a Decision."""

    name = "abstract"
    manifest: tuple[str, ...] = ()

    def decide(self, problem) -> "Decision":
        raise NotImplementedError


@dataclass
class Decision:
    """What `DE-DecisionScheme` emits.  `duals` is DECLARED EMPTY for a rule
    policy (duals become load-bearing with optimizing solvers and incentive
    obligations); `rationale` is what EV-Attribution consumes."""
    actions: list = field(default_factory=list)
    rationale: dict = field(default_factory=dict)
    duals: dict = field(default_factory=dict)


class UtilityNone:
    """DE_MODULE_PLAN section 3.2.  `utility` is a required config field and
    RulePolicy_v1 evaluates no utility; declaring `risk_neutral` would be the
    silent-misdeclaration class n-ary validation exists to reject.  So the
    registered utility is one whose evaluation is a TYPED REFUSAL."""

    name = "utility_none"
    consumes_utility = False

    def evaluate(self, *_a, **_kw):
        raise UtilityRefused(
            "utility_none has no evaluation. It is registered so that "
            "'no utility' can be DECLARED rather than misdeclared as "
            "risk_neutral; a caller reaching this line has wired a utility "
            "into a solver whose manifest says it evaluates none.")


class IncentiveNone:
    """Section 3.4's mirror: a registered null IncentiveModel with empty
    contributions, replaced when rho/rewards facts exist."""

    name = "incentive_none"

    def contributions(self, *_a, **_kw) -> dict:
        return {}


# Section 3.4: `by_input` carries an entry for every CONSUMED input whose type
# has an `Unavailable` arm -- today exactly `risk_scenarios`.  Consumed
# always-Known inputs need no entry and can never trigger the policy; an
# omitted key on a consumed can-be-Unavailable input is a WIRING ERROR, not a
# default; and a non-consumed input carries NO entry, its absence licensed by
# the manifest.
CAN_BE_UNAVAILABLE = ("risk_scenarios",)
UNAVAILABLE_POLICY = {"risk_scenarios": "Halt"}
UNWRAP_POLICY: dict[str, str] = {}


def validate_unavailable_policy(by_input: dict[str, str],
                                manifest: Sequence[str]) -> None:
    required = sorted(set(CAN_BE_UNAVAILABLE) & set(manifest))
    missing = [k for k in required if k not in by_input]
    if missing:
        raise RegistrationRefused(
            f"unavailable_policy.by_input is MISSING {missing}. An omitted "
            f"key on a consumed input whose type has an Unavailable arm is a "
            f"WIRING ERROR, not a default (section 3.4).")
    extra = [k for k in by_input if k not in required]
    if extra:
        raise RegistrationRefused(
            f"unavailable_policy.by_input carries {extra}, which this solver "
            f"either does not consume or whose type is always-Known. A "
            f"non-consumed input carries NO entry -- its absence is licensed "
            f"by the manifest, and an entry claims a consumption that is not "
            f"in it.")
    for k, v in by_input.items():
        if v not in ("Halt", "RefuseAction", "FallBack"):
            raise RegistrationRefused(
                f"unavailable_policy[{k}]={v!r} is not a legal "
                f"UnavailableAction (v22: Halt | RefuseAction | FallBack)")


@dataclass
class Registration:
    name: str
    manifest: tuple[str, ...]
    utility: str
    incentive: str
    probe: dict
    unavailable_policy: dict
    unwrap_policy: dict


class SolverRegistry:
    """Registration is a CHECK, not a record of intent.

    Nothing is stored unless the revelation probe comes back clean and the
    R-COMPAT triple validates, so a registry entry is evidence rather than a
    claim."""

    def __init__(self) -> None:
        self._entries: dict[str, Registration] = {}
        self.named_seams = ("ClosedFormGLFT", "PerLevel", "HJBQVI")

    def register(self, solver: Solver, *, utility, incentive,
                 probe_fields: dict[str, Any],
                 unavailable_policy: dict[str, str],
                 unwrap_policy: dict[str, str]) -> Registration:
        declared = tuple(solver.manifest)
        grew = [k for k in declared if k not in MANIFEST]
        if grew:
            raise RegistrationRefused(
                f"solver {solver.name!r} declares manifest fields {grew} that "
                f"are not in the plan's manifest {MANIFEST}. The list is "
                f"load-bearing and cannot silently grow (section 3.1).")
        validate_unavailable_policy(unavailable_policy, declared)
        if unwrap_policy:
            raise RegistrationRefused(
                f"unwrap_policy must be empty for a rule policy; got "
                f"{unwrap_policy}")
        # R-COMPAT: utility_none is valid ONLY with a solver that evaluates
        # no utility.  `utility` is not a manifest field at all here, so a
        # utility that DOES evaluate is incompatible by construction.
        if getattr(utility, "consumes_utility", True):
            raise RegistrationRefused(
                f"R-COMPAT: solver {solver.name!r} declares no utility "
                f"consumption, so it may only be registered with a utility "
                f"that evaluates none; {utility.name!r} evaluates one.")
        probe = revelation_probe(solver, probe_fields)
        if probe["decide_error"] is not None:
            raise RegistrationRefused(
                f"the revelation probe could not run {solver.name!r}: "
                f"{probe['decide_error']}. A probe that crashed proves "
                f"nothing about the boundary, so registration refuses rather "
                f"than recording a vacuous pass.")
        if probe["sentinels_touched"] or probe["non_manifest_keys_read"]:
            raise RegistrationRefused(
                f"MANIFEST VIOLATION, revealed not asserted: solver "
                f"{solver.name!r} touched {probe['sentinels_touched']} and "
                f"read non-manifest keys {probe['non_manifest_keys_read']}. "
                f"Its manifest says it consumes none of these.")
        outside = [k for k in probe["keys_read"] if k not in declared]
        if outside:
            raise RegistrationRefused(
                f"solver {solver.name!r} read {outside}, which is outside its "
                f"OWN declared manifest {declared}")
        reg = Registration(solver.name, declared, utility.name, incentive.name,
                           probe, dict(unavailable_policy),
                           dict(unwrap_policy))
        self._entries[solver.name] = reg
        return reg

    def get(self, name: str) -> Registration:
        if name not in self._entries:
            raise RegistrationRefused(f"{name!r} is not registered")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)


# ---------------------------------------------------------------------------
# 3. RulePolicy_v1 -- declarative placement / cancel / cross / stop
# ---------------------------------------------------------------------------
CANCEL_RULES = ("NO_CANCEL", "DECLARED_STUB_TRIGGER")


@dataclass
class RuleConfig:
    """Every knob is a DECLARED parameter with no default that encodes a
    policy choice.  `cancel_rule` defaults to NO_CANCEL because the
    cancellation family is UNMEASURED at the achievable rungs (placement plan
    section 4 / section 11): shipping a trigger as the default would be
    choosing a policy the measurement does not support."""
    cancel_rule: str = "NO_CANCEL"
    dump_enabled: bool = False
    terminal_enabled: bool = True
    gamma: float = 1e-2                 # risk-appetite; report across a ladder
    half_spread_h: float = 0.005        # SP-Venue fact read through a handle
    r_cut_s: float = R_CUT_S
    decision_band_s: float = R_DECISION_BAND_S
    carry_size_shares: float = 0.0      # what section 7 intends to carry

    def validate(self) -> None:
        if self.cancel_rule not in CANCEL_RULES:
            raise PolicyRefused(
                f"cancel_rule {self.cancel_rule!r} not in {CANCEL_RULES}")
        for k in ("gamma", "half_spread_h", "r_cut_s", "decision_band_s",
                  "carry_size_shares"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(float(v)):
                raise PolicyRefused(f"{k}={v!r} must be a finite number")
        if self.gamma <= 0.0:
            raise PolicyRefused("gamma must be > 0 (it divides N*)")


def dump_threshold(p: float, gamma: float, h: float) -> float:
    """N*(p) = (2/gamma) * [0.07 + h / (p(1-p))]  -- placement plan section 6.

    A DIAGNOSTIC.  The plan says so in its own words twice, and the
    side-aware scenario cap binds independently and wins every conflict, so
    this number never authorises a CROSS by itself."""
    if not (0.0 < p < 1.0):
        raise PolicyRefused(
            f"dump threshold undefined at p={p!r}: p(1-p) is the variance of "
            f"the binary payoff and vanishes at the boundary")
    return (2.0 / gamma) * (DUMP_FEE_TERM + h / (p * (1.0 - p)))


class RulePolicyV1(Solver):
    """The composed policy of placement plan section 3, as rules."""

    name = "RulePolicy_v1"
    manifest = MANIFEST

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()
        self.config.validate()

    # -- the four rules, each its own function so each has its own control --

    def where_to_rest(self, net: float) -> dict[str, str]:
        """Skew on net, as MEASURED (the SKEW_LB arm): long Up -> the reducing
        side fronts on genuine level re-formation, the adding side joins;
        short Up mirrors; near flat both JOIN.

        `FRONT_ON_FORMATION`, never NEW_BBO: at a 1-tick book -- the modal
        btc/eth state -- price improvement would cross or lock and a
        same-price queue jump is impossible, so fronting is only meaningful
        at genuine re-formation."""
        if net > 1e-12:                       # long Up
            return {"ASK_UP": "FRONT_ON_FORMATION", "BID_UP": "JOIN"}
        if net < -1e-12:                      # short Up
            return {"BID_UP": "FRONT_ON_FORMATION", "ASK_UP": "JOIN"}
        return {"BID_UP": "JOIN", "ASK_UP": "JOIN"}

    def when_to_leave(self, ctx: dict[str, Any]) -> bool:
        """The cancellation rule.  UNMEASURED at the achievable rungs, so the
        only configurations are NO_CANCEL and an explicitly-named declared
        stub; there is no third option that quietly encodes a trigger."""
        if self.config.cancel_rule == "NO_CANCEL":
            return False
        key = (f"rule_policy_v1|{ctx['slug']}|{ctx['side']}|"
               f"{ctx['ref_gen']}")
        h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) >= 0.5

    def regime(self, r_seconds: float) -> str:
        """Section 7's terminal schedule."""
        if not self.config.terminal_enabled:
            return "NORMAL"
        if r_seconds > self.config.r_cut_s + self.config.decision_band_s:
            return "NORMAL"
        if r_seconds >= self.config.r_cut_s - self.config.decision_band_s:
            return "DECISION_POINT"
        return "REDUCING_ONLY"

    # -- the emitted decision --------------------------------------------

    def decide(self, problem) -> Decision:
        view = problem["view"]
        selfstate = problem["self"]
        portfolio = problem["portfolio"]
        scenarios = problem["risk_scenarios"]
        constraints = problem["constraints"]
        horizon = problem["horizon"]
        _vocab = problem["actions"]           # the ActionSpace words

        pos: dc.Position = portfolio["position"]
        resting: dict = view["resting"]
        prices: dict = view["prices"]
        p_mid = float(view["p_mid"])
        r = float(horizon["r_seconds"])
        halt = selfstate["halt_state"]
        sp = constraints["sp"]

        reg = self.regime(r)
        # The terminal schedule tightens the STATE the oracle is asked about;
        # it never loosens it.  A HALTED book stays halted: halt is latched
        # and all venue actions are blocked, and the residual is carried --
        # the designed degradation, not an unhandled path.
        state = halt
        if halt == dc.RUNNING and reg in ("DECISION_POINT", "REDUCING_ONLY"):
            state = dc.REDUCING_ONLY

        menu = das.enumerate_actions(state, pos, resting, prices, sp,
                                     scenarios["losses"])
        by_verb: dict[str, list] = {}
        for a in menu:
            by_verb.setdefault(a.verb, []).append(a)

        net = pos.net
        rs = dc.reducing_side(pos)
        adding = None if rs is None else ("BID_UP" if rs == "ASK_UP"
                                          else "ASK_UP")
        placement = self.where_to_rest(net)
        actions: list = []
        fired: list[str] = []

        # (1) WHEN TO STOP -- cancel the adding side at the decision point and
        #     below it.  Retracting an already-resting quote is the SCHEME's
        #     action: a feasibility oracle only refuses NEW actions, so no
        #     oracle can remove a quote that is already in the market.
        cancel_refs: list[str] = []
        if reg in ("DECISION_POINT", "REDUCING_ONLY") and adding is not None:
            cancel_refs = sorted(ref for ref, q in resting.items()
                                 if q.side == adding)
            if cancel_refs:
                fired.append(f"terminal:{reg}:cancel_adding_side")

        # (2) WHEN TO LEAVE -- the cancellation rule, per resting quote.
        for ref in sorted(resting):
            if ref in cancel_refs:
                continue
            q = resting[ref]
            ctx = {"slug": view["slug"], "side": q.side,
                   "ref_gen": view["ref_gen"]}
            if self.when_to_leave(ctx):
                cancel_refs.append(ref)
                fired.append(f"cancel_rule:{self.config.cancel_rule}:{ref}")
        cancel_by_ref = {a.order_ref: a for a in by_verb.get("CANCEL", [])}
        for ref in sorted(set(cancel_refs)):
            if ref in cancel_by_ref:
                actions.append(cancel_by_ref[ref])

        # (3) WHEN TO CROSS -- the dump, and the two things that gate it.
        n_star = None
        if self.config.dump_enabled and rs is not None and 0.0 < p_mid < 1.0:
            n_star = dump_threshold(p_mid, self.config.gamma,
                                    self.config.half_spread_h)
            cross = [a for a in by_verb.get("CROSS", []) if a.side == rs]
            if abs(net) > n_star and cross:
                # SECTION 4.7: a CROSS into a side is ALWAYS preceded by
                # cancelling our own resting quote on that side.  Enforced by
                # ORDER here and asserted by `check_self_trade_sequencing`.
                for ref, q in sorted(resting.items()):
                    if q.side == rs and ref in cancel_by_ref \
                            and cancel_by_ref[ref] not in actions:
                        actions.append(cancel_by_ref[ref])
                actions.append(cross[0])
                fired.append("dump:above_N_star")
            elif abs(net) > (n_star or math.inf) and not cross:
                # The diagnostic said dump; the oracle did not admit it. The
                # cap wins every conflict, and the disagreement is REPORTED
                # rather than resolved in the diagnostic's favour.
                fired.append("dump:refused_by_scenario_cap")

        # (4) WHERE TO REST -- quote the sides the menu still offers, at the
        #     placement the skew rule chose.  In REDUCING_ONLY the adding side
        #     is absent from the menu by construction, which is the oracle
        #     enforcing section 7's "the adding side must not rest".
        for a in by_verb.get("QUOTE", []):
            if a.placement != placement.get(a.side):
                continue
            if reg == "REDUCING_ONLY" and a.side != rs:
                continue
            actions.append(a)
            fired.append(f"skew:{a.side}:{a.placement}")
        if not actions:
            actions.append(dc.Action("WAIT"))
            fired.append("wait:no_expressible_action")

        for a in actions:                     # emission-time, fail-loud
            dc.validate_action(a)
        check_self_trade_sequencing(actions)

        return Decision(
            actions=actions,
            rationale={
                "regime": reg, "oracle_state": state, "net": net,
                "reducing_side": rs, "adding_side": adding,
                "placement": placement, "rules_fired": fired,
                "n_star_diagnostic": n_star,
                "n_star_is_a_diagnostic":
                    "the side-aware scenario cap binds independently and wins "
                    "every conflict; N* never authorises a CROSS by itself",
                "menu_size": len(menu),
            },
            duals={},                          # declared empty for a rule policy
        )


def check_self_trade_sequencing(actions: Sequence[dc.Action]) -> None:
    """Placement plan section 4.7: a CROSS into a side must be preceded by a
    CANCEL of our own resting quote on that side.  Checked over the emitted
    ORDER, because the ordering IS the mechanism."""
    seen_cancel: set = set()
    for i, a in enumerate(actions):
        if a.verb == "CANCEL":
            seen_cancel.add(a.order_ref)
        elif a.verb == "CROSS":
            for j in range(i + 1, len(actions)):
                if actions[j].verb == "CANCEL":
                    raise PolicyRefused(
                        f"CANCEL at index {j} follows a CROSS at index {i}: "
                        f"section 4.7 requires the cancel to PRECEDE the "
                        f"cross into that side")


# ---------------------------------------------------------------------------
# 4. the seam adapter -- RulePolicy_v1 driven through EV-Replay
# ---------------------------------------------------------------------------

class RulePolicySeamAdapter(seam.ActionValuePolicy):
    """Runs RulePolicy_v1's CANCEL RULE through the EV-Replay seam.

    THE {0,1} IN THE SCORE CHANNEL IS A DECISION, NOT AN ESTIMATE, and saying
    so is the point.  The seam's score stream feeds the state machine's
    declared thresholds; a rule policy has no probability to offer, so its
    cancel rule enters as a degenerate indicator against a threshold of 0.5.
    The machine still owns the LIFECYCLE -- one cancel per generation,
    latency, hold, repost -- and every threshold there remains a declared
    parameter it refuses to default."""

    name = "RULE_POLICY_V1"
    status = "RULE_POLICY_UNVALIDATED"

    def __init__(self, policy: RulePolicyV1) -> None:
        self.policy = policy
        self.predictor_active = policy.config.cancel_rule != "NO_CANCEL"

    def score(self, ctx: dict[str, Any]) -> float:
        seam.validate_context(ctx)            # the section-0 boundary holds
        return 1.0 if self.policy.when_to_leave(
            {"slug": ctx["slug"], "side": ctx["side"],
             "ref_gen": ctx["ref_gen"]}) else 0.0

    def score_stream(self, slug, sides, window_end):
        if not self.predictor_active:
            # NO_CANCEL emits no scores at all, which is what makes the
            # reduction to QR_SKEW_ONLY exact rather than approximate.
            return []
        return super().score_stream(slug, sides, window_end)


SEAM_PARAMS = dict(seam.DEFAULT_PARAMS, theta_cancel=0.5, theta_repost=0.1)
# NOTE THE ABSENCE OF AN INERT PARAMETER SET, AND WHY.
# The first version ran the NO_CANCEL parity under `predictor_enabled=False`.
# That gate could not fail: with the machine's predictor disabled EVERY score
# stream is ignored, so ANY policy would have been bit-identical, and the
# check would have proved the machine's own disabled-predictor anchor (which
# batch 1 already proved) rather than the claim it was written for. A mutant
# that made NO_CANCEL emit scores exited 0 against it -- rule 16's control
# that cannot fail, found by the mutant rather than by reading.
# The parity now runs with the predictor ENABLED, so the reduction to
# QR_SKEW_ONLY is a property of the POLICY's configuration -- it emits no
# scores at all -- and not of a flag on the machine.

# The window this module replays through the seam.  It reuses the SEAM's
# reference geometry (so the parity anchor is against the seam's own
# documented fixture) under its OWN slug, because the declared trigger is a
# pure function of (slug, side, gen) and the slug is what makes it fire.
#
# THE THRESHOLD IS NOT THE FREE VARIABLE, THE FIXTURE IS.  The trigger fires
# at >= 0.5 and that number is declared once and never moved; the fixture is
# chosen so the rule has something to act on, and the induced values are
# recorded here so a reader recomputes the crossings rather than trusting
# them:
#     BUY_UP  gen 3 -> 0.669  FIRES        SELL_UP gen 3 -> 0.212
#     BUY_UP  gen 5 -> 0.103                SELL_UP gen 5 -> 0.670
#     BUY_UP  gen 9 -> 0.787                SELL_UP gen 9 -> 0.835  FIRES
# (the reference carries BUY_UP 3 and 5 and SELL_UP 9, so two of its three
# generations cross).  Moving the threshold to fit a fixture would be the
# choosing-after-seeing move; moving the fixture to exercise a fixed rule is
# not, and the difference is worth naming.
RULE_SPEC = {"slug": "rule-policy-w1", "inputs_hash": "1" * 16}


def seam_record(policy: RulePolicyV1, *, params: dict) -> dict:
    sides, end = seam.fixture_reference()
    return seam.ReplaySession(
        RULE_SPEC, sides, params, RulePolicySeamAdapter(policy),
        queue_bound="BACK_DISPLAYED", unavailable_iv=(), window_end=end,
        seed=20260901).run()


# ---------------------------------------------------------------------------
# 5. fixtures, gates, receipt
# ---------------------------------------------------------------------------

def fixture_problem(*, net_up: float = 8.0, r_seconds: float = 300.0,
                    halt: str = dc.RUNNING) -> dict[str, Any]:
    pos = dc.Position(q_up=max(net_up, 0.0), q_down=max(-net_up, 0.0),
                      cost_up=0.5 * max(net_up, 0.0),
                      cost_down=0.5 * max(-net_up, 0.0))
    resting = {"ord-1": dc.RestingQuote("BID_UP", 0.49, 5.0),
               "ord-2": dc.RestingQuote("ASK_UP", 0.51, 5.0)}
    return {
        "view": {"prices": {"BID_UP": 0.49, "ASK_UP": 0.51}, "p_mid": 0.50,
                 "resting": resting, "slug": "rule-fixture-w1", "ref_gen": 1},
        "self": {"halt_state": halt},
        "actions": {"verbs": dc.VERBS},
        "portfolio": {"position": pos},
        "risk_scenarios": {"losses": {"S1": 0.0}},
        "constraints": {"sp": dict(dc.SP_OPERATIVE)},
        "horizon": {"r_seconds": r_seconds},
    }


class LeakySolver(Solver):
    """KNOWN-BAD kept in the LIBRARY: a rule policy that quietly reads a
    belief field.  It must be REFUSED at registration."""

    name = "LeakyRule_v0"
    manifest = MANIFEST

    def decide(self, problem):
        _ = problem["belief"].mean          # both layers trip
        return Decision(actions=[dc.Action("WAIT")])


class QuietLeakSolver(Solver):
    """KNOWN-BAD, subtler: it FETCHES a non-manifest field and discards it.
    No sentinel is ever used, so only the key-read layer can catch this."""

    name = "QuietLeak_v0"
    manifest = MANIFEST

    def decide(self, problem):
        problem["coupling"]                  # fetched and thrown away
        return Decision(actions=[dc.Action("WAIT")])


REQUIRED_GATES = (
    "manifest_revealed_clean",
    "leak_refused_at_registration",
    "quiet_leak_refused_at_registration",
    "utility_none_is_a_typed_refusal",
    "unavailable_policy_validated",
    "no_cancel_reduces_to_qr_skew_only",
    "no_cancel_bit_identical_through_da_contract",
    "cancel_rule_differs",
    "terminal_schedule_regimes",
    "self_trade_sequencing",
    "sizing_within_feasible_set",
    "no_economics_emitted",
)


def run_gates() -> tuple[dict, dict]:
    reg = SolverRegistry()
    policy = RulePolicyV1(RuleConfig())
    registration = reg.register(
        policy, utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY),
        unwrap_policy=dict(UNWRAP_POLICY))

    def _refuses(fn) -> bool:
        try:
            fn()
            return False
        except RegistrationRefused:
            return True

    leak_refused = _refuses(lambda: reg.register(
        LeakySolver(), utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}))
    quiet_refused = _refuses(lambda: reg.register(
        QuietLeakSolver(), utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}))

    try:
        UtilityNone().evaluate()
        utility_refuses = False
    except UtilityRefused:
        utility_refuses = True

    up_ok = (_refuses(lambda: validate_unavailable_policy({}, MANIFEST))
             and _refuses(lambda: validate_unavailable_policy(
                 dict(UNAVAILABLE_POLICY, belief="Halt"), MANIFEST))
             and validate_unavailable_policy(
                 dict(UNAVAILABLE_POLICY), MANIFEST) is None)

    # PARITY: NO_CANCEL reduces to QR_SKEW_ONLY, bit-identically
    sides, _end = seam.fixture_reference()
    passthrough = hsp.build_passthrough_trajectory({RULE_SPEC["slug"]: sides})
    rec_nc = seam_record(policy, params=SEAM_PARAMS)
    parity_native = (hsp.bit_identical(rec_nc["events"], passthrough)
                     # and the predictor was LIVE while it happened
                     and rec_nc["diagnostics"][
                         "scores_ignored_predictor_disabled"] == 0
                     and rec_nc["diagnostics"]["cancels_issued"] == 0)

    # ... and through the BATCH-1 INSTRUMENT: the same two trajectories,
    # exported under DA's canon and compared by DA's own digest.
    import de_lane4_real_parity as lane4
    import da_replay_parity_battery as da
    o_pol = lane4.export_da(rec_nc["events"], "QR_SKEW_ONLY", "none", False,
                            None, stale_cancel_reading="DROP")
    o_ref = lane4.export_da(passthrough, "QR_SKEW_ONLY", "none", False, None,
                            stale_cancel_reading="DROP")
    parity_contract = (
        da.load_external_trajectory(o_pol).digest()
        == da.load_external_trajectory(o_ref).digest())

    acting = RulePolicyV1(RuleConfig(cancel_rule="DECLARED_STUB_TRIGGER"))
    rec_act = seam_record(acting, params=SEAM_PARAMS)
    differs = (not hsp.bit_identical(rec_act["events"], passthrough)
               and rec_act["diagnostics"]["cancels_issued"] >= 1)

    regimes = {r: policy.regime(r) for r in (300.0, 60.0, 10.0)}
    regime_ok = (regimes[300.0] == "NORMAL"
                 and regimes[60.0] == "DECISION_POINT"
                 and regimes[10.0] == "REDUCING_ONLY")

    # terminal behaviour, at the decision point and below it
    d_dp = policy.decide(fixture_problem(r_seconds=60.0))
    d_ro = policy.decide(fixture_problem(r_seconds=10.0))
    verbs_ro = {a.verb for a in d_ro.actions}
    sides_quoted = {a.side for a in d_ro.actions if a.verb == "QUOTE"}
    terminal_ok = (
        any(a.verb == "CANCEL" and a.order_ref == "ord-1"
            for a in d_dp.actions)                     # adding side cancelled
        and "CANCEL" in verbs_ro
        and sides_quoted <= {"ASK_UP"})                # reducing side only

    # sizing: every emitted size-bearing action within FeasibleSet.max_size
    prob = fixture_problem()
    ms = dc.max_size(dc.RUNNING, prob["portfolio"]["position"],
                     list(prob["view"]["resting"].values()),
                     prob["view"]["prices"], prob["constraints"]["sp"],
                     prob["risk_scenarios"]["losses"])
    d_run = policy.decide(prob)
    sizing_ok = all(dc.action_feasible(a, ms) for a in d_run.actions)

    # self-trade sequencing, both directions
    bad = [dc.Action("CROSS", side="ASK_UP", size=5.0),
           dc.Action("CANCEL", order_ref="ord-2")]
    try:
        check_self_trade_sequencing(bad)
        seq_ok = False
    except PolicyRefused:
        seq_ok = True
    good = [dc.Action("CANCEL", order_ref="ord-2"),
            dc.Action("CROSS", side="ASK_UP", size=5.0)]
    seq_ok = seq_ok and check_self_trade_sequencing(good) is None

    inputs = seam.declared_inputs(RulePolicySeamAdapter(policy).status)
    try:
        seam.emission_guard({"r": [{"net_cents": 1.0}]}, inputs)
        guard = False
    except seam.EconomicsRefused:
        guard = True

    gates = {
        "manifest_revealed_clean":
            not registration.probe["sentinels_touched"]
            and not registration.probe["non_manifest_keys_read"]
            and bool(registration.probe["manifest_keys_read"]),
        "leak_refused_at_registration": leak_refused,
        "quiet_leak_refused_at_registration": quiet_refused,
        "utility_none_is_a_typed_refusal": utility_refuses,
        "unavailable_policy_validated": bool(up_ok),
        "no_cancel_reduces_to_qr_skew_only": parity_native,
        "no_cancel_bit_identical_through_da_contract": parity_contract,
        "cancel_rule_differs": differs,
        "terminal_schedule_regimes": regime_ok and terminal_ok,
        "self_trade_sequencing": seq_ok,
        "sizing_within_feasible_set": sizing_ok,
        "no_economics_emitted": guard,
    }
    return gates, {"registration": registration, "registry": reg,
                   "records": [rec_nc, rec_act], "regimes": regimes,
                   "rationale_normal": d_run.rationale}


def gates_all_pass(gates: dict) -> bool:
    if [g for g in REQUIRED_GATES if g not in gates]:
        return False
    return all(bool(gates[g]) for g in REQUIRED_GATES)


def missing_gates(gates: dict) -> list[str]:
    return [g for g in REQUIRED_GATES if g not in gates]


_IDENTITY_FILES = ("rule_policy_v1.py", "ev_replay_seam.py",
                   "harmful_stateful_policy.py", "de_constraints.py",
                   "de_actionspace.py")
ENGINE_IDENTITY = {
    f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()[:16]
    for f in _IDENTITY_FILES}


def code_binding() -> dict:
    """Per-file artifact->commit binding.

    A commit cannot contain its own id, so an artifact cannot literally carry
    the commit that will carry it.  What it CAN carry, and what makes the
    binding survive a mis-attributed commit message, is the commit that
    carries each PRODUCER FILE plus that file's content hash -- verifiable by
    re-hashing the blob at that commit.  A file not yet committed is stamped
    `UNCOMMITTED_AT_EMISSION` rather than given a plausible-looking commit."""
    out = {}
    for f, h in ENGINE_IDENTITY.items():
        rel = f"live/pm_research/{f}"
        commit = seam._git("log", "-1", "--format=%H", "--", rel)
        blob = seam._git("rev-parse", f"HEAD:{rel}") if commit else None
        status = "BOUND" if commit else "UNCOMMITTED_AT_EMISSION"
        if commit:
            head_blob_sha = seam._git("show", f"HEAD:{rel}")
            if head_blob_sha is None or hashlib.sha256(
                    (head_blob_sha + "\n").encode()).hexdigest()[:16] != h:
                status = "DIRTY_AT_EMISSION"
        out[f] = {"sha256_16": h, "carrying_commit": commit,
                  "blob": blob, "status": status}
    return out


def verify_binding(binding: dict, repo_file: str, commit: str) -> bool:
    """Recompute a file's hash AT a commit and compare.  This is what makes
    the binding a predicate rather than a stored string."""
    blob = seam._git("show", f"{commit}:live/pm_research/{repo_file}")
    if blob is None:
        return False
    got = hashlib.sha256((blob + "\n").encode()).hexdigest()[:16]
    return got == binding[repo_file]["sha256_16"]


def run(out: Path | None = None) -> dict:
    gates, ctx = run_gates()
    reg: Registration = ctx["registration"]
    inputs = seam.declared_inputs("RULE_POLICY_UNVALIDATED")
    body = {
        "protocol": PROTOCOL,
        "status": seam.STATUS,
        "solver": reg.name,
        "registered_beside": list(ctx["registry"].named_seams),
        "manifest": list(reg.manifest),
        "non_manifest": list(NON_MANIFEST),
        "manifest_check": "REVELATION at registration, not assertion",
        "revelation_probe": reg.probe,
        "utility": reg.utility,
        "incentive": reg.incentive,
        "unavailable_policy": reg.unavailable_policy,
        "unwrap_policy": reg.unwrap_policy,
        "duals": "DECLARED EMPTY for a rule policy",
        "inputs": inputs,
        "unreleased_inputs": seam.any_unreleased(inputs),
        "economics_emittable": False,
        "regimes": ctx["regimes"],
        "rationale_normal_regime": ctx["rationale_normal"],
        "records": [{"policy": r["policy"], "policy_status": r["policy_status"],
                     "n_events": len(r["events"]),
                     "record_hash": r["record_hash"]}
                    for r in ctx["records"]],
        "gates": gates,
        "all_gates_pass": gates_all_pass(gates),
        "engine_identity": dict(ENGINE_IDENTITY),
        "code_binding": code_binding(),
        "produced_at": seam.produced_at(),
    }
    body["run_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()
    seam.emission_guard(body, inputs)
    if not body["all_gates_pass"]:
        raise PolicyRefused(
            f"REFUSING to write: gates failed "
            f"{[g for g in REQUIRED_GATES if not gates.get(g)]}, missing "
            f"{missing_gates(gates)}")
    if out is not None:
        out.write_text(json.dumps(body, indent=2, sort_keys=True, default=str))
    return body


EXPECTED_CHECKS = 37


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[rule_policy_v1] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(exc, fn, label):
        try:
            fn()
        except exc:
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[rule_policy_v1] FAIL (no refusal): {label}")

    # ---- the manifest, revealed --------------------------------------
    reg = SolverRegistry()
    policy = RulePolicyV1(RuleConfig())
    r = reg.register(policy, utility=UtilityNone(), incentive=IncentiveNone(),
                     probe_fields=fixture_problem(),
                     unavailable_policy=dict(UNAVAILABLE_POLICY),
                     unwrap_policy={})
    ok(tuple(r.manifest) == MANIFEST and len(MANIFEST) == 7,
       f"RulePolicy_v1 registers with the plan's exact manifest {MANIFEST}")
    ok(not r.probe["sentinels_touched"]
       and not r.probe["non_manifest_keys_read"],
       "REVEALED, not asserted: the probe ran the solver on sentinels for "
       "belief/competition/outcomes/incentives/coupling and NONE was touched")
    ok(set(r.probe["manifest_keys_read"]) == set(MANIFEST),
       f"POSITIVE CONTROL: the probe DID observe the solver reading its whole "
       f"manifest ({len(r.probe['manifest_keys_read'])} keys) -- a probe that "
       f"saw nothing would report a clean boundary for a solver that never ran")
    refuses(RegistrationRefused, lambda: reg.register(
        LeakySolver(), utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}),
        "KNOWN-BAD: a solver that reads `belief` is REFUSED AT REGISTRATION")
    refuses(RegistrationRefused, lambda: reg.register(
        QuietLeakSolver(), utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}),
        "KNOWN-BAD, THE SUBTLE ONE: a solver that FETCHES `coupling` and "
        "discards it touches no sentinel -- only the key-read layer catches "
        "it, which is why there are two layers")
    ok(reg.names() == ["RulePolicy_v1"],
       f"and NOTHING is registered on a failed check: {reg.names()}")

    class GrownManifest(Solver):
        name = "Grown_v0"
        manifest = MANIFEST + ("belief",)

        def decide(self, problem):
            return Decision(actions=[dc.Action("WAIT")])

    refuses(RegistrationRefused, lambda: reg.register(
        GrownManifest(), utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}),
        "KNOWN-BAD: a manifest that GREW past the plan's list is refused -- "
        "the list is load-bearing and cannot silently grow")

    class CrashingSolver(Solver):
        name = "Crash_v0"
        manifest = MANIFEST

        def decide(self, problem):
            raise ValueError("boom")

    refuses(RegistrationRefused, lambda: reg.register(
        CrashingSolver(), utility=UtilityNone(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}),
        "KNOWN-BAD: a solver whose probe CRASHED is refused rather than "
        "recorded clean -- a probe that could not run proves nothing")

    # ---- utility / incentive / unavailable policy ---------------------
    refuses(UtilityRefused, lambda: UtilityNone().evaluate(),
            "utility_none is a TYPED REFUSAL, not a silent risk_neutral")
    ok(IncentiveNone().contributions() == {},
       "incentive_none contributes nothing, as a registration rather than "
       "as an omission")

    class RealUtility:
        name = "risk_neutral"
        consumes_utility = True

    refuses(RegistrationRefused, lambda: reg.register(
        RulePolicyV1(), utility=RealUtility(), incentive=IncentiveNone(),
        probe_fields=fixture_problem(),
        unavailable_policy=dict(UNAVAILABLE_POLICY), unwrap_policy={}),
        "KNOWN-BAD (R-COMPAT): a solver that evaluates no utility may not be "
        "registered with a utility that evaluates one")
    ok(validate_unavailable_policy(dict(UNAVAILABLE_POLICY), MANIFEST) is None,
       "POSITIVE CONTROL: the declared unavailable policy validates")
    refuses(RegistrationRefused,
            lambda: validate_unavailable_policy({}, MANIFEST),
            "KNOWN-BAD: an omitted entry for a consumed can-be-Unavailable "
            "input is a WIRING ERROR, not a default")
    refuses(RegistrationRefused, lambda: validate_unavailable_policy(
        dict(UNAVAILABLE_POLICY, belief="Halt"), MANIFEST),
        "KNOWN-BAD: an entry for a NON-consumed input is refused too -- it "
        "claims a consumption the manifest does not have")

    # ---- the rules ----------------------------------------------------
    ok(policy.where_to_rest(8.0) == {"ASK_UP": "FRONT_ON_FORMATION",
                                     "BID_UP": "JOIN"}
       and policy.where_to_rest(-8.0) == {"BID_UP": "FRONT_ON_FORMATION",
                                          "ASK_UP": "JOIN"}
       and policy.where_to_rest(0.0) == {"BID_UP": "JOIN", "ASK_UP": "JOIN"},
       "WHERE TO REST: long fronts the reducing side and joins the adding "
       "side, short mirrors, flat joins both -- and it is "
       "FRONT_ON_FORMATION, never NEW_BBO")
    ok(policy.regime(300.0) == "NORMAL" and policy.regime(60.0)
       == "DECISION_POINT" and policy.regime(10.0) == "REDUCING_ONLY",
       "WHEN TO STOP: the three regimes of the terminal schedule")
    d_ro = policy.decide(fixture_problem(r_seconds=10.0))
    ok({a.side for a in d_ro.actions if a.verb == "QUOTE"} <= {"ASK_UP"}
       and any(a.verb == "CANCEL" for a in d_ro.actions),
       "below r_cut the book is REDUCING-ONLY: only the reducing side rests "
       "and the adding side's resting quote is cancelled by the SCHEME "
       "(an oracle only refuses NEW actions)")
    d_h = policy.decide(fixture_problem(r_seconds=10.0, halt=dc.HALTED))
    ok(all(a.verb in dc.ALWAYS_FEASIBLE for a in d_h.actions),
       "and a HALTED book stays halted through the terminal schedule: "
       "CANCEL/WAIT only, residual carried -- the designed degradation")
    ok(abs(dump_threshold(0.50, 1e-2, 0.005) - (2 / 1e-2)
           * (0.07 + 0.005 / 0.25)) < 1e-12
       and dump_threshold(0.10, 1e-2, 0.005)
       > dump_threshold(0.50, 1e-2, 0.005),
       "WHEN TO CROSS: N*(p) evaluates its own formula, and the tails "
       "tolerate a LARGER imbalance than ATM")
    refuses(PolicyRefused, lambda: dump_threshold(0.0, 1e-2, 0.005),
            "KNOWN-BAD: N* is undefined where p(1-p) vanishes and refuses "
            "rather than returning an infinity")
    ok(policy.decide(fixture_problem()).rationale["n_star_diagnostic"] is None,
       "with dump_enabled False no N* is computed at all, so the diagnostic "
       "cannot leak into a decision it was never asked for")
    refuses(PolicyRefused, lambda: check_self_trade_sequencing(
        [dc.Action("CROSS", side="ASK_UP", size=5.0),
         dc.Action("CANCEL", order_ref="ord-2")]),
        "KNOWN-BAD (section 4.7): a CANCEL that FOLLOWS its CROSS is refused")
    ok(check_self_trade_sequencing(
        [dc.Action("CANCEL", order_ref="ord-2"),
         dc.Action("CROSS", side="ASK_UP", size=5.0)]) is None,
       "POSITIVE CONTROL: the correct order is admitted")
    refuses(PolicyRefused, lambda: RuleConfig(cancel_rule="MAYBE").validate(),
            "KNOWN-BAD: an undeclared cancel rule is refused; there is no "
            "third option that quietly encodes a trigger")
    ok(RuleConfig().cancel_rule == "NO_CANCEL",
       "and the DEFAULT is NO_CANCEL, because the cancellation family is "
       "UNMEASURED at the achievable rungs")

    # ---- parity: NO_CANCEL is QR_SKEW_ONLY ----------------------------
    sides, _ = seam.fixture_reference()
    passthrough = hsp.build_passthrough_trajectory({RULE_SPEC["slug"]: sides})
    rec_nc = seam_record(policy, params=SEAM_PARAMS)
    ok(hsp.bit_identical(rec_nc["events"], passthrough),
       "PARITY: RulePolicy_v1 in its NO_CANCEL configuration is "
       "BIT-IDENTICAL to the frozen QR_SKEW_ONLY reference, through the "
       "EV-Replay seam")
    ok(rec_nc["diagnostics"]["scores_ignored_predictor_disabled"] == 0
       and rec_nc["diagnostics"]["cancels_issued"] == 0,
       "AND THE PREDICTOR WAS LIVE WHILE IT HAPPENED: zero scores were "
       "ignored-as-disabled, so the reduction is a property of the POLICY "
       "emitting no scores, not of a disabled flag on the machine -- the "
       "first version of this gate ran with the predictor OFF and could not "
       "have failed")
    import de_lane4_real_parity as lane4
    import da_replay_parity_battery as da
    dig = [da.load_external_trajectory(
        lane4.export_da(ev, "QR_SKEW_ONLY", "none", False, None,
                        stale_cancel_reading="DROP")).digest()
        for ev in (rec_nc["events"], passthrough)]
    ok(dig[0] == dig[1],
       "and bit-identical THROUGH THE BATCH-1 INSTRUMENT too: the same two "
       "trajectories exported under DA's canon carry the same digest")
    acting = RulePolicyV1(RuleConfig(cancel_rule="DECLARED_STUB_TRIGGER"))
    rec_act = seam_record(acting, params=SEAM_PARAMS)
    ok(not hsp.bit_identical(rec_act["events"], passthrough)
       and rec_act["diagnostics"]["cancels_issued"] >= 1,
       f"POSITIVE CONTROL: with the declared trigger the policy ACTS "
       f"({rec_act['diagnostics']['cancels_issued']} cancels) and parity "
       f"BREAKS, so the parity claim is a comparison and not a tautology")

    # ---- economics and the receipt ------------------------------------
    receipt = run()
    ok(receipt["all_gates_pass"] and not missing_gates(receipt["gates"]),
       f"all {len(REQUIRED_GATES)} required gates pass")
    ok(receipt["economics_emittable"] is False
       and "RULE_POLICY_UNVALIDATED" in receipt["inputs"].values()
       and "action_value_policy" in receipt["unreleased_inputs"],
       "the policy is BUILT and UNVALIDATED, and says so: its status is an "
       "UNRELEASED one, so the guard still refuses economics")
    ok(seam._economic_keys_in(receipt, skip=seam.INPUT_SUBTREES) == [],
       "the emitted receipt carries no economic quantity in its output region")
    ok(gates_all_pass({g: True for g in REQUIRED_GATES[:-1]}) is False,
       "KNOWN-BAD: a MISSING required gate makes all_gates_pass False")
    cb = receipt["code_binding"]
    ok(all(v["status"] in ("BOUND", "UNCOMMITTED_AT_EMISSION",
                           "DIRTY_AT_EMISSION") for v in cb.values())
       and "harmful_stateful_policy.py" in cb,
       f"every producer file carries a per-file binding and a STATUS: "
       f"{ {k: v['status'] for k, v in cb.items()} }")
    bound = [f for f, v in cb.items() if v["status"] == "BOUND"]
    ok(bound and all(verify_binding(cb, f, cb[f]["carrying_commit"])
                     for f in bound),
       f"POSITIVE CONTROL: every BOUND file re-hashes to its recorded value "
       f"at its own carrying commit ({len(bound)} files)")
    ok(not verify_binding(cb, bound[0], "HEAD~50"),
       "KNOWN-BAD: the same predicate returns False at the WRONG commit, so "
       "the binding is a check and not a stored string")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[rule_policy_v1] selftest OK -- {n[0]} checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        if selftest() != 0:
            return 1
        r = run(Path(a.out) if a.out else None)
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("records", "revelation_probe")},
                         indent=2, sort_keys=True, default=str))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
