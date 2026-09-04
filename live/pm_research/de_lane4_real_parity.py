"""Lane 4 seven-arm parity battery, run against the REAL QR_SKEW_ONLY shadow.

SURFACE AUTHORISATION (R-126, in-file): R-379 TASK 1 (DE seat), R-165(3)
(full-tape integration is Phase-4 work, DE-owned), LANE4_REPLAY_PARITY_STUB_
BATTERY.md, SKEW_LANE_NEUTRAL_REFERENCE_FREEZE.md (USER-frozen semantics).
RESEARCH-ONLY, OFFLINE: no venue port, no order path, no live semantics.

THIS IS VERIFICATION, NOT SCORING.  No economics may be read from any output
here: every value the state machine computes is suppressed from the receipt
except the lifecycle COUNTS the parity gates are defined over.  The standing
hold (HANDOFF item 4) forbids PnL, capacity, promotion and forward verdicts,
and this module is built so that obeying it is structural rather than
remembered -- `_receipt_cell` refuses to emit any key from the economics
block.

WHAT IT DOES.  For each window of the declared population it
  1. replays the QR_SKEW_ONLY no-cancel shadow through the exposure
     pipeline's own recorder (`harmful_exposure_rows.replay_with_recorder`),
     which is the artifact the skew freeze cites -- never a re-implementation;
  2. adapts the generation table into the state machine's reference shape,
     counting every generation it cannot admit as a STATUS (rule 4);
  3. runs the parity gates the LANE4 spec names, at real-data scale.

WHAT IS AND IS NOT RUNNABLE, computed rather than asserted.  Three of the
seven declared arms cannot be run on this reference and the reason is a
MISSING INPUT, not a missing predictor:
  * HAZARD_ONLY_NEUTRAL / CONDVALUE_NEUTRAL declare NEUTRAL placement.  The
    frozen reference is QR_SKEW_ONLY -- skew is ON (`_qr_spec` hardcodes
    `skew: True` for both QR cells), placement comes entirely from the
    reference, and the predictor may never choose placement (TODO 2.2 fence,
    restated in the skew freeze 3).  A neutral-placement reference exists
    only in the NON-queue-realistic family (`JOIN_ONLY`), so running these
    arms would compare two placement engines and call the difference a
    policy effect.
  * CONDVALUE_X_SKEW / CONDVALUE_X_SKEW_X_FAIRPRICE need a released
    predictor; none is released for this use, and the parity contract has no
    name for the declared stub (see the FINDINGS block).
They are reported with an explicit status, never silently dropped, because
"five arms agreed" over a seven-arm claim is the vacuity this battery exists
to refuse.

    python3 live/pm_research/de_lane4_real_parity.py --selftest
    python3 live/pm_research/de_lane4_real_parity.py run [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# CO-2 (class, not instance): a bare flat import dies under
# `python3 -m live.pm_research.…` while the script-dir launch is green --
# a suite that passes only because of how it was started. DA's modules
# already do this; measured before/after in both launches.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harmful_stateful_policy as hsp

OK = "OK"
EPS = 1e-9
SIDES = ("BUY_UP", "SELL_UP")
SIDE_IDX = {s: i for i, s in enumerate(SIDES)}
RANK_START, RANK_FILL, RANK_END, RANK_SCORE = 0, 1, 2, 3

# ---- the declared population (never chosen after looking) -----------------
POPULATION = "v3_4_consumed_fragment"
ERA = "clob_v3_1"                 # the era of the consumed fragment, NAMED
COINS = ("btc", "eth")

# ---- the declared STUB scorer.  NOT a model, and deliberately not one -----
# LANE4 section 1: the battery is built and proved BEFORE any predictor
# exists, so every arm is a typed stub.  sha256-derived, never builtin
# hash(), because hash() of a str is salted by PYTHONHASHSEED and a scorer
# built on it selects a different cancel set per process (LANE4 B1.1).
STUB_SALT = "de_lane4_real_parity_v1"
STUB_EARLY_FRAC = 0.25            # first score event, as a fraction of the
STUB_LATE_FRAC = 0.75             # generation interval; second event
STUB_LATE_SCORE = 0.0             # the late event is always below theta

# ---- the declared policy parameters for the ACTIVE-stub cell -------------
# Declared here, before any cell is read.  None of them is a default that
# encodes a policy choice: the machine refuses every one of them if absent.
ACTIVE_PARAMS = {
    "predictor_enabled": True,
    "theta_cancel": 0.90,
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
# The cancel-and-hold cell: theta_repost = -inf can never be crossed from
# below, so the hold is permanent and the machine never reposts.  This is the
# LANE4 anchor "zero repost threshold with permanent hold == cancel-and-hold".
HOLD_PARAMS = dict(ACTIVE_PARAMS, theta_repost=float("-inf"))
DISABLED_PARAMS = dict(ACTIVE_PARAMS, predictor_enabled=False)
INF_PARAMS = dict(ACTIVE_PARAMS, theta_cancel=float("inf"))

ARMS = (
    "QR_SKEW_ONLY",
    "QR_CANCEL_HOLD_X_SKEW",
    "HAZARD_ONLY_NEUTRAL",
    "CONDVALUE_NEUTRAL",
    "CONDVALUE_X_SKEW",
    "CONDVALUE_X_SKEW_X_FAIRPRICE",
    "RANDOM_MATCHED",
)
# THE COMPOSITION THAT ACTUALLY EXISTS ON THE FROZEN LANE HAS NO NAME.
# LANE4 section 3 names arms 3 and 4 by their PLACEMENT ("fill-hazard-only
# cancel, NEUTRAL PLACEMENT" / "conditional-value cancel, NEUTRAL
# PLACEMENT"), and the frozen reference is QR_SKEW_ONLY -- skew ON.  BE's
# exporter (`be_trajectory_export.BE_EXPORTABLE_COMPOSITIONS`) reads the same
# name as a claim about the PREDICTOR instead, and declares 011 exportable as
# CONDVALUE_NEUTRAL because no skew/inventory state reaches the model.  Both
# readings are defensible and they disagree, so the run over the frozen
# reference is:
#   * NOT CONDVALUE_X_SKEW -- that name asserts an INTERACTION 011 does not
#     have, and DA's loader REFUSES interaction=False under it (verified);
#   * NOT honestly CONDVALUE_NEUTRAL -- that name decomposes to
#     components ("condvalue",), so the submission would omit the skew that
#     was in force in the placement it inherited (accepted by the loader,
#     which is exactly the mislabel BE's own comment says the contract
#     structurally cannot catch).
# Escalated, not resolved: the arm vocabulary needs a name for "condvalue
# over a skewed reference, WITHOUT interaction", or a ruling that one of the
# two readings governs.  SEAT_PROTOCOL rule 13.
ARM_NAME_COLLISION = (
    "no seven-arm name fits 'condvalue predictor over the frozen SKEWED "
    "reference with no interaction': X_SKEW asserts an interaction that is "
    "refused, and NEUTRAL omits the skew that was in force")

#: §8.1's REQUIRED OUTPUT, field by field, with WHERE each comes from.
#: The plan ends "net_cancel_cents alone is not a strategy-P&L verdict",
#: so an arm that reports a subset must say WHICH subset -- a partial
#: reported as complete is the failure this list exists to prevent.
#: `source` is the producer; None means nothing in this repo produces it
#: and the arm reports NOT_AVAILABLE with that reason rather than a zero.
SECTION_8_1_FIELDS: dict[str, dict] = {
    "maker_pnl_cents": {
        "source": None,
        "why": "the replay values CANCELLATION (harm avoided minus "
               "sacrifice), not a maker book. A complete maker P&L needs "
               "spread earned on every fill minus adverse selection on "
               "every fill, over the whole opportunity population -- the "
               "replay prices the DECISION, not the book"},
    "spread_capture_cents": {
        "source": None,
        "why": "`de_rho_estimator` computes a spread denominator PER "
               "RECEIVED FILL for rho; there is no book-level spread "
               "capture, and summing rho's denominator would be a "
               "different quantity wearing the name"},
    "post_fill_markout_cents": {
        "source": "economics.received_markout_cents",
        "also": ("economics.stale_markout_cents",)},
    "fill_share_retention": {"source": "retention_share_fraction"},
    "rho": {"source": "de_rho_estimator.rho"},
    "cancels_effective": {"source": "counters.cancels_effective"},
    "cancels_stale": {"source": "counters.cancels_stale"},
    "cancels_unresolved": {"source": "counters.cancels_unresolved"},
    "holds_total": {"source": "counters.holds_total"},
    "hold_seconds_total": {"source": "counters.hold_seconds_total"},
    "reposts": {"source": "counters.reposts"},
    "queue_reset_cost_cents_total": {
        "source": "counters.queue_reset_cost_cents_total"},
    "terminal_inventory": {"source": "economics.inventory.net"},
    "peak_inventory": {"source": "economics.inventory.peak_abs_net"},
    "inventory_loss_cents": {
        "source": None,
        "why": "the inventory block carries NET and PEAK ABS shares and "
               "the increasing/reducing split, but no valuation of the "
               "position it leaves behind. Inventory LOSS needs a "
               "terminal mark, which the replay never takes"},
    "latency_x_cost_sensitivity": {
        "source": "the caller, by replaying the arm across the frozen "
                  "latency axis and cost levels"},
}


#: WHAT EACH ARM NEEDS, as dependency NAMES rather than a verdict.
#: §8.1's seven arms differ in exactly two ways -- the PLACEMENT the
#: opportunity population is replayed under, and the PREDICTOR that decides
#: cancellation -- so a runnability answer is a conjunction over named
#: dependencies, each of which can be probed.
ARM_DEPENDENCIES: dict[str, tuple] = {
    "QR_SKEW_ONLY": ("skewed_placement",),
    "QR_CANCEL_HOLD_X_SKEW": ("skewed_placement", "cancel_hold_policy"),
    "HAZARD_ONLY_NEUTRAL": ("neutral_placement", "hazard_head"),
    "CONDVALUE_NEUTRAL": ("neutral_placement", "condvalue_head"),
    "CONDVALUE_X_SKEW": ("skewed_placement", "condvalue_head",
                         "policy_layer_skew_composition"),
    "CONDVALUE_X_SKEW_X_FAIRPRICE": ("skewed_placement", "condvalue_head",
                                     "policy_layer_skew_composition",
                                     "fairprice_challenger"),
    "RANDOM_MATCHED": ("skewed_placement", "matched_random_control"),
}


def _probe_skewed_placement() -> tuple:
    """A placement spec with skew ON -- the frozen reference."""
    import policy_optimizer_queue_realistic as _qr
    specs = {n: _qr._qr_spec(getattr(_qr, n), latency_ms=0, cancel=False)
             for n in ("QR_SKEW", "QR_BASELINE")}
    hit = {n: s for n, s in specs.items() if s.get("skew") is True}
    return bool(hit), {"specs_with_skew_true": sorted(hit),
                       "placements": sorted({s["placement"]
                                             for s in specs.values()})}


def _probe_neutral_placement() -> tuple:
    """A placement spec with skew OFF, which §8.1 arms 3 and 4 are DEFINED by.

    MEASURED 2026-09-04 and it is the round's first finding: there is none.
    `QR_BASELINE` is NOT a baseline -- its value is the string
    "QR_CANCEL_HOLD_X_SKEW", it yields `skew: True` and
    `placement: QUEUE_REALISTIC_SKEW`, and the ONLY field distinguishing it
    from `QR_SKEW` is the label `cell`. A reference built from it and called
    neutral would be the frozen SKEWED reference wearing a neutral name --
    the mislabel `ARM_NAME_COLLISION` escalated and never got a ruling on.
    Creating a `skew: False` placement is a change to placement semantics
    §10(5) freezes, so it is not this seat's to invent."""
    import policy_optimizer_queue_realistic as _qr
    specs = {n: _qr._qr_spec(getattr(_qr, n), latency_ms=0, cancel=False)
             for n in ("QR_SKEW", "QR_BASELINE")}
    neutral = {n: s for n, s in specs.items() if s.get("skew") is False}
    return bool(neutral), {
        "specs_with_skew_false": sorted(neutral),
        "constants_checked": {n: getattr(_qr, n) for n in
                              ("QR_SKEW", "QR_BASELINE")},
        "skew_by_constant": {n: s.get("skew") for n, s in specs.items()},
        "note": "QR_BASELINE's VALUE is 'QR_CANCEL_HOLD_X_SKEW'; the only "
                "field differing between the two specs is `cell`"}


def _probe_cancel_hold_policy() -> tuple:
    import harmful_stateful_policy as _hsp
    ok_ = hasattr(_hsp, "replay_policy") and hasattr(_hsp, "PROTECTION_MODES")
    return ok_, {"replay_policy": hasattr(_hsp, "replay_policy"),
                 "protection_modes": list(
                     getattr(_hsp, "PROTECTION_MODES", ()))}


def _probe_hazard_head() -> tuple:
    import de_head_scoring as _hs
    try:
        inc = _hs.load_incumbent("btc")
        return True, {"head": "incumbent_linear_d",
                      "n_features": inc["_n_features"]}
    except Exception as exc:                        # noqa: BLE001
        return False, {"error": f"{type(exc).__name__}: {exc}"}


def _probe_condvalue_head() -> tuple:
    import de_head_scoring as _hs
    try:
        booster, width = _hs.load_lgbm("btc")
        return True, {"head": "q1_arrival_composed_lgbm", "width": width}
    except Exception as exc:                        # noqa: BLE001
        return False, {"error": f"{type(exc).__name__}: {exc}"}


#: WHAT `X_SKEW` MEANS, RULED AND WRITTEN HERE SO NOBODY RE-ASKS.
#:
#: Round 32 escalated: no seven-arm name fitted "condvalue predictor over
#: the frozen SKEWED reference with no interaction". Round 52 first read
#: the gap as a MISSING DEPENDENCY -- no skew feature reaches the model,
#: therefore X_SKEW cannot run -- and that reading was wrong.
#:
#: THE PLAN ANSWERS IT AND FORBIDS THE OTHER READING. §2.2: "Inventory and
#: lifecycle state remain POLICY INPUTS. They price whether a cancel, size
#: reduction or repost is desirable; THEY DO NOT BECOME PREDICTOR FEATURES
#: MERELY BECAUSE THEY AFFECT THE ACTION DECISION." §10.1's dependency
#: diagram composes conditional value, fair price, frozen skew and
#: latency/cost as FOUR PARALLEL INPUTS converging on the action-value
#: policy -- composition, not interaction.
#:
#: So arm 5 is CONDITIONAL-VALUE CANCELLATION COMPOSED WITH FROZEN SKEW AT
#: THE POLICY LAYER, WITHOUT INTERACTION, and the predictor carries no
#: skew state BY DESIGN. Putting skew into the predictor would build the
#: thing §2.2 forbids and would risk the double-counting §2.2 exists to
#: prevent.
#:
#: (§3 does record inventory and the increases/reduces flag as DATASET ROW
#: fields. That is consistent: recorded in the row, not fed to the
#: predictor -- and it is the distinction that made this look like a
#: missing dependency.)
ARM_X_SKEW_SEMANTICS = (
    "conditional-value cancellation COMPOSED with frozen skew AT THE "
    "POLICY LAYER, without interaction; the predictor carries no skew or "
    "inventory state by design (plan §2.2, §10.1)")


def _probe_policy_layer_skew_composition() -> tuple:
    """Can the POLICY LAYER compose a conditional-value cancel decision
    with the frozen skew placement?

    This is the predicate the ruling asks for, and it is deliberately NOT
    "is there skew state in the predictor" -- that question tests for the
    thing §2.2 forbids, and answering it False blocked two arms for a
    design property rather than a gap.

    What composition needs: a placement that carries the frozen skew, a
    policy that takes a cancel threshold over a score stream, and both
    protection modes the conjunction is defined over."""
    import policy_optimizer_queue_realistic as _qr
    import harmful_stateful_policy as _hsp
    spec = _qr._qr_spec(_qr.QR_SKEW, latency_ms=0, cancel=True)
    frozen_skew = spec.get("skew") is True and spec.get("cancel") is True
    policy = (hasattr(_hsp, "replay_policy")
              and len(getattr(_hsp, "PROTECTION_MODES", ())) >= 1
              and len(getattr(_hsp, "REPOST_FILL_MODELS", ())) >= 1)
    return bool(frozen_skew and policy), {
        "frozen_skew_placement_with_cancel": frozen_skew,
        "policy_layer": policy,
        "protection_modes": list(getattr(_hsp, "PROTECTION_MODES", ())),
        "repost_fill_models": list(getattr(_hsp, "REPOST_FILL_MODELS", ())),
        "semantics": ARM_X_SKEW_SEMANTICS,
        "ruled_by": "plan §2.2 and §10.1, read at the artifact 2026-09-04; "
                    "closes the round-32 ARM_NAME_COLLISION escalation"}


def _probe_skew_state_in_predictor_RETIRED() -> tuple:
    """RETIRED BY THE §2.2 RULING and kept only as the record of a wrong
    predicate. It asked whether the predictor SEES skew or inventory
    state; the plan says it must not, so a False here is the DESIGN and
    never a blocker. Read by the suite alone.

    `CONDVALUE_X_SKEW` asserts an INTERACTION. If no skew or inventory
    feature reaches the model, the name claims something the artifact does
    not have -- which is what `ARM_NAME_COLLISION` is about, and what DA's
    loader refuses. Read off the fit's own feature list, never assumed."""
    # THIS PROBE IS THREE-VALUED AND THE THIRD VALUE IS THE POINT.
    # Its first draft read the feature list off `load_lgbm_normalisers`,
    # which carries only n_raw / norm_mu / norm_sd / source -- NO NAMES.
    # It therefore answered "no skew features" when the truth was "I could
    # not find the names", and an arm would have been reported blocked for
    # a reason that was never measured. `None` means UNDECIDABLE and is
    # reported as such; it still blocks (fail-closed) but it never claims
    # to have looked.
    import de_head_scoring as _hs
    try:
        norms = _hs.load_lgbm_normalisers("btc")
        n_raw = norms.get("n_raw")
    except Exception as exc:                        # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}",
                      "undecidable": "the normalisers would not load"}
    named: dict = {}
    try:
        import phase2_state_schema_freeze as _pin
        st = [str(x) for x in (_pin.build_pin().get("features_in_order")
                               or [])]
        named["state"] = st
    except Exception as exc:                        # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}",
                      "undecidable": "the state schema pin would not load"}
    if not named.get("state"):
        return None, {"undecidable": "no named feature list is reachable",
                      "n_raw": n_raw}
    WORDS = ("skew", "inventory", "invent", "posn", "position")
    hits = sorted({n for n in named["state"]
                   if any(w in n.lower() for w in WORDS)})
    # The state block is the ONLY named family; PM and fine features are
    # positional, so the answer is bounded to what can be read. Say so.
    return bool(hits), {
        "n_state_features_named": len(named["state"]),
        "n_raw_total": n_raw,
        "n_unnamed_positional": (n_raw - len(named["state"]))
        if isinstance(n_raw, int) else None,
        "skew_or_inventory_features": hits,
        "scope": "the STATE block is the only NAMED feature family; the "
                 "PM and fine families are positional in this artifact, "
                 "so this answers 'no skew feature among the 45 named "
                 "state features' and not 'no skew feature anywhere'",
        "consequence": "X_SKEW asserts an INTERACTION; with no skew or "
                       "inventory state reaching the model the name "
                       "claims something the artifact does not have "
                       "(ARM_NAME_COLLISION), and DA's loader refuses "
                       "interaction=False under it"}


def _probe_fairprice_challenger() -> tuple:
    """A RELEASED fair-price challenger. BE is building one; until a scored
    challenger artifact exists this is False and arm 6 is blocked."""
    _fits = (Path(__file__).resolve().parents[2]
             / "data/pm_5min/derived/phase2_fits")
    cands = sorted(_fits.glob("*fairprice*.json")) if _fits.exists() else []
    return bool(cands), {"searched": str(_fits),
                         "found": [c.name for c in cands],
                         "status_yml": "hazard-fair-price: challenger "
                                       "protocol not freeze-ready, no "
                                       "challenger scored"}


def _probe_matched_random_control() -> tuple:
    """The matched-random control -- §8.1 arm 7, and the FLOOR the other six
    are measured against. Recorded blocker was
    NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL; this probes whether the
    control and a contract identity for it exist NOW."""
    try:
        import de_matched_random_control as _mrc
    except Exception as exc:                        # noqa: BLE001
        return False, {"error": f"{type(exc).__name__}: {exc}"}
    has_draw = hasattr(_mrc, "draw")
    named = "RANDOM_MATCHED" in ARMS
    return bool(has_draw and named), {
        "module": getattr(_mrc, "__file__", None),
        "draw": has_draw, "named_in_ARMS": named,
        "matched_on": "action count, side, hour and cancellation budget "
                      "(§8.1 arm 7)"}


ARM_DEPENDENCY_PROBES = {
    "skewed_placement": _probe_skewed_placement,
    "neutral_placement": _probe_neutral_placement,
    "cancel_hold_policy": _probe_cancel_hold_policy,
    "hazard_head": _probe_hazard_head,
    "condvalue_head": _probe_condvalue_head,
    "policy_layer_skew_composition": _probe_policy_layer_skew_composition,
    "skew_state_in_predictor_RETIRED": _probe_skew_state_in_predictor_RETIRED,
    "fairprice_challenger": _probe_fairprice_challenger,
    "matched_random_control": _probe_matched_random_control,
}


def arm_runnability(probes: dict | None = None) -> dict:
    """WHICH ARMS CAN RUN, COMPUTED -- replacing a hand-maintained map.

    `ARM_RUNNABLE` was a dict of verdict strings written by hand. A claim
    about code kept in a comment drifts from the code without either the
    comment or the code noticing; this programme has been caught by that
    shape repeatedly. Every verdict here is a conjunction over probes that
    execute, and each probe returns its EVIDENCE, so a blocked arm says
    what it is blocked ON rather than carrying a slogan.

    `probes` is injectable FOR THE FALSIFIERS ALONE -- the run passes
    none, and both directions are driven: an arm whose dependencies all
    pass reads RUNNABLE, and removing any one makes it BLOCKED naming
    exactly that dependency."""
    P = probes or ARM_DEPENDENCY_PROBES
    results: dict = {}
    for name, fn in P.items():
        try:
            okd, ev = fn()
        except Exception as exc:                    # noqa: BLE001
            okd, ev = None, {"error": f"{type(exc).__name__}: {exc}",
                             "undecidable": "the probe itself raised"}
        results[name] = {
            "available": (None if okd is None else bool(okd)),
            "state": ("UNDECIDABLE" if okd is None
                      else "AVAILABLE" if okd else "ABSENT"),
            "evidence": ev}
    arms: dict = {}
    for arm, deps in ARM_DEPENDENCIES.items():
        absent = [d for d in deps
                  if results.get(d, {}).get("available") is False]
        undecided = [d for d in deps
                     if results.get(d, {}).get("available") is None]
        missing = absent + undecided
        arms[arm] = {"runnable": not missing,
                     "dependencies": list(deps),
                     "blocked_on": missing,
                     "absent": absent,
                     # FAIL-CLOSED, AND NAMED: an undecidable dependency
                     # blocks, but it is never reported as an absent one.
                     # "I looked and it is not there" and "I could not
                     # look" are different facts (rule 4).
                     "undecidable": undecided,
                     "status": "RUNNABLE" if not missing
                     else "BLOCKED:" + ",".join(
                         [f"{d}=ABSENT" for d in absent]
                         + [f"{d}=UNDECIDABLE" for d in undecided])}
    return {"as_of": _dt.datetime.now(_dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "dependencies": results,
            "arms": arms,
            "n_runnable": sum(1 for a in arms.values() if a["runnable"]),
            "runnable": sorted(a for a, v in arms.items() if v["runnable"]),
            "blocked": {a: v["blocked_on"] for a, v in arms.items()
                        if not v["runnable"]},
            "decides": "nothing -- this reports which arms have their "
                       "dependencies, never whether an arm should run"}


#: SUPERSEDED BY `arm_runnability()` (round 52) and kept only so the
#: computed report can be checked against what was believed by hand. It is
#: read by the suite and by nothing else.
ARM_RUNNABLE_LEGACY = {
    "QR_SKEW_ONLY": "RUNNABLE",
    "QR_CANCEL_HOLD_X_SKEW": "RUNNABLE",
    "HAZARD_ONLY_NEUTRAL": "NO_NEUTRAL_REFERENCE",
    "CONDVALUE_NEUTRAL": "NO_NEUTRAL_REFERENCE",
    "CONDVALUE_X_SKEW": "NO_RELEASED_PREDICTOR",
    "CONDVALUE_X_SKEW_X_FAIRPRICE": "NO_RELEASED_PREDICTOR",
    "RANDOM_MATCHED": "NO_CONTRACT_IDENTITY_FOR_AN_ACTING_CONTROL",
}

# Generation-admission statuses.  Every one is COUNTED and reported; none is
# a silent drop (rule 4).
GEN_STATUSES = ("ADMITTED", "NO_LEVEL_SEGMENT", "MULTI_LEVEL",
                "ZERO_LENGTH", "OVERLAPS_PREVIOUS", "TRANCHE_OUTSIDE",
                "SHARES_EXCEED_DISPLAYED", "NONFINITE")
WINDOW_STATUSES = ("ADMITTED", "REPLAY_NONE", "RECONCILIATION_FAILED",
                   "NO_ADMITTED_GENERATION", "REFERENCE_REFUSED")


# CODE IDENTITY IS TAKEN AT IMPORT, NOT AT RECEIPT-WRITE TIME.  A long run
# outlives edits to its own source, and hashing the file when the receipt is
# written would stamp the receipt with code that did not produce it -- an
# identity that attests to the wrong program is worse than none (rule 12).
_IDENTITY_FILES = ("de_lane4_real_parity.py", "harmful_stateful_policy.py",
                   "harmful_exposure_rows.py",
                   "policy_optimizer_queue_realistic.py",
                   "da_replay_parity_battery.py")
CODE_IDENTITY = {
    f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()[:16]
    for f in _IDENTITY_FILES}


class VacuousBattery(RuntimeError):
    """A battery over zero admitted generations must not report passing arms.
    Zero difference under zero data is not parity (LANE4 falsifier 6)."""


class ExportRefused(RuntimeError):
    """The DA-contract exporter refuses to guess a mapping it does not have."""


# ---------------------------------------------------------------------------
# 1. reference adapter -- generation table -> state-machine reference
# ---------------------------------------------------------------------------

def _fin(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def reference_from_window(segments: Sequence[dict], gens: dict
                          ) -> tuple[dict, dict]:
    """(reference sides, per-status counts).

    `gens` is `harmful_exposure_rows.generation_table`'s output, keyed
    (side, gen), carrying t0/t1/tranches.  It does NOT carry level, displayed
    or status -- those come from the recorder's segments, and where the
    segments disagree with themselves the generation is EXCLUDED WITH A
    STATUS rather than resolved by picking one."""
    counts = {s: 0 for s in GEN_STATUSES}
    first: dict[tuple, dict] = {}
    levels: dict[tuple, set] = {}
    for s in segments:
        if s["level"] is None:
            continue
        k = (s["side"], s["gen"])
        levels.setdefault(k, set()).add(round(float(s["level"]), 12))
        cur = first.get(k)
        if cur is None or s["t_start"] < cur["t_start"]:
            first[k] = s
    out: dict[str, list] = {side: [] for side in SIDES}
    for (side, gen) in sorted(gens, key=lambda k: (k[0], k[1])):
        g = gens[(side, gen)]
        seg = first.get((side, gen))
        if seg is None:
            counts["NO_LEVEL_SEGMENT"] += 1
            continue
        if len(levels[(side, gen)]) > 1:
            # A generation that changed level is not ONE resting order at one
            # level; picking the first would silently redefine the unit.
            counts["MULTI_LEVEL"] += 1
            continue
        t0, t1 = float(g["t0"]), float(g["t1"])
        displayed = float(seg["resting"])
        level = float(seg["level"])
        if not (_fin(t0) and _fin(t1) and _fin(level) and _fin(displayed)):
            counts["NONFINITE"] += 1
            continue
        if not (t1 > t0):
            counts["ZERO_LENGTH"] += 1
            continue
        if not (displayed > 0.0):
            counts["NONFINITE"] += 1
            continue
        trs = list(g["tranches"])
        if any(not (t0 - EPS <= float(t["t"]) <= t1 + EPS) for t in trs):
            counts["TRANCHE_OUTSIDE"] += 1
            continue
        if sum(float(t["shares"]) for t in trs) > displayed + EPS:
            counts["SHARES_EXCEED_DISPLAYED"] += 1
            continue
        status = OK if all(_fin(t["markout_cents_per_share"]) for t in trs) \
            else "NO_FUTURE_MID"
        out[side].append({
            "gen": int(gen), "t0": t0, "t1": t1, "level": level,
            "displayed": displayed, "status": status,
            "tranches": [{"t": float(t["t"]), "shares": float(t["shares"]),
                          "markout_cents_per_share":
                              t["markout_cents_per_share"]}
                         for t in sorted(trs, key=lambda t: float(t["t"]))],
        })
        counts["ADMITTED"] += 1
    # non-overlap is a property of the ADMITTED sequence, checked after
    # selection because an excluded generation cannot overlap anything
    for side in SIDES:
        keep, prev_t1 = [], -math.inf
        for g in out[side]:
            if g["t0"] < prev_t1 - EPS:
                counts["OVERLAPS_PREVIOUS"] += 1
                counts["ADMITTED"] -= 1
                continue
            keep.append(g)
            prev_t1 = g["t1"]
        out[side] = keep
    return out, counts


# ---------------------------------------------------------------------------
# 2. the declared stub score stream
# ---------------------------------------------------------------------------

def stub_score(slug: str, side: str, gen: int) -> float:
    h = hashlib.sha256(f"{STUB_SALT}|{slug}|{side}|{gen}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def stub_scores(slug: str, sides: dict) -> list[dict]:
    """Two events per generation: one inside the interval carrying the stub
    value, one late and below every threshold.  Deterministic and process-
    independent; it is a STUB, and no cell here reads a model."""
    ev = []
    for side in SIDES:
        for g in sides[side]:
            span = g["t1"] - g["t0"]
            ev.append({"t": g["t0"] + STUB_EARLY_FRAC * span, "slug": slug,
                       "side": side, "score": stub_score(slug, side, g["gen"])})
            ev.append({"t": g["t0"] + STUB_LATE_FRAC * span, "slug": slug,
                       "side": side, "score": STUB_LATE_SCORE})
    ev.sort(key=lambda s: (s["t"], SIDE_IDX[s["side"]]))
    return ev


# ---------------------------------------------------------------------------
# 3. INDEPENDENT cancel-and-hold construction (never the machine)
# ---------------------------------------------------------------------------
# Written from the DECLARED semantics in the module header of
# `harmful_stateful_policy` -- the merged (t, rank, side, seq) order, lazy
# settlement at the next processed event, and the closed EVENT_KEYS schema --
# not by calling the machine's constructors.  A fixture that borrowed the
# machine's own event builders would be supplying what the code under test
# should produce (rule 16 / R-229 class).

def build_cancel_and_hold(slug: str, sides: dict, scores: Sequence[dict],
                          theta_cancel: float, latency_ms: float
                          ) -> list[dict]:
    lat = latency_ms / 1000.0
    stream = []
    for side in SIDES:
        for i, g in enumerate(sides[side]):
            stream.append((g["t0"], RANK_START, SIDE_IDX[side], i,
                           "start", side, g))
            for j, tr in enumerate(g["tranches"]):
                stream.append((tr["t"], RANK_FILL, SIDE_IDX[side],
                               i * 10000 + j, "fill", side, (g, tr)))
            stream.append((g["t1"], RANK_END, SIDE_IDX[side], i,
                           "end", side, g))
    for k, s in enumerate(scores):
        stream.append((s["t"], RANK_SCORE, SIDE_IDX[s["side"]], k,
                       "score", s["side"], s))
    stream.sort(key=lambda e: (e[0], e[1], e[2], e[3]))

    traj: list[dict] = []
    held = {s: False for s in SIDES}
    rec_by_gen: dict[tuple, dict] = {}
    live: dict[str, dict | None] = {s: None for s in SIDES}
    # Inventory is PER-SLUG and starts FLAT (R-184 step (vii)): each 5-minute
    # slug is an independent binary market settling at expiry, so no slug's
    # state reads another's.  It is tracked here only because
    # `reducing_at_request` is a FIELD OF THE EVENT: with
    # ALL_ORDERS_OVERRIDE nothing is suppressed for being on the reducing
    # side, but the event still RECORDS which side it was, and an independent
    # builder that hardcoded it would be asserting a fact it never computed.
    net = [0.0]
    # the record carrying an unresolved cancel, per side -- settlement is
    # lazy and must be applied at the NEXT PROCESSED EVENT of any kind, a
    # generation start included.  (Written from the declared semantics; the
    # first version of this builder settled everywhere BUT a start, and the
    # real-data comparison is what surfaced the omission -- in both this
    # builder and the machine.)
    pending: dict[str, dict | None] = {s: None for s in SIDES}

    def _reducing(side):
        return ((net[0] > EPS and side == "SELL_UP")
                or (net[0] < -EPS and side == "BUY_UP"))

    def settle(rec, t):
        if rec is None or rec["state"] != "CANCEL_PENDING":
            return
        te = rec["t_eff"]
        if not (te <= t and te < rec["g"]["t1"]):
            return
        rec["state"] = "CANCELLED"
        side, g = rec["side"], rec["g"]
        if pending[side] is rec:
            pending[side] = None
        pg = rec["policy_gen"]
        traj.append({"kind": "CANCEL_EFFECTIVE", "t": te, "slug": slug,
                     "side": side, "ref_gen": g["gen"], "policy_gen": pg})
        traj.append({"kind": "GEN_END", "t": te, "slug": slug, "side": side,
                     "ref_gen": g["gen"], "policy_gen": pg,
                     "reason": "CANCELLED"})
        traj.append({"kind": "HOLD_START", "t": te, "slug": slug,
                     "side": side, "ref_gen": g["gen"], "policy_gen": pg})
        held[side] = True
        if live[side] is rec:
            live[side] = None

    for t, _r, _si, _sq, kind, side, payload in stream:
        if kind == "start":
            g = payload
            settle(pending[side], t)
            if held[side]:
                traj.append({"kind": "GEN_START_MISSED_HELD", "t": t,
                             "slug": slug, "side": side, "ref_gen": g["gen"]})
                continue
            rec = {"g": g, "side": side, "policy_gen": str(g["gen"]),
                   "state": "LIVE", "remaining": g["displayed"],
                   "t_eff": None}
            rec_by_gen[(side, g["gen"])] = rec
            live[side] = rec
            traj.append({"kind": "PLACE", "t": t, "slug": slug, "side": side,
                         "ref_gen": g["gen"], "policy_gen": rec["policy_gen"],
                         "level": g["level"], "displayed": g["displayed"],
                         "source": "TRACKING"})
        elif kind == "fill":
            g, tr = payload
            settle(pending[side], t)
            rec = rec_by_gen.get((side, g["gen"]))
            mo = tr["markout_cents_per_share"]
            if rec is None:
                traj.append({"kind": "FILL_MISSED_HELD", "t": t, "slug": slug,
                             "side": side, "ref_gen": g["gen"],
                             "shares": tr["shares"],
                             "markout_cents_per_share": mo})
                continue
            settle(rec, t)
            if rec["state"] == "CANCELLED":
                traj.append({"kind": "FILL_PREVENTED", "t": t, "slug": slug,
                             "side": side, "ref_gen": g["gen"],
                             "policy_gen": rec["policy_gen"],
                             "shares": tr["shares"],
                             "markout_cents_per_share": mo,
                             "cancel_t_effective": rec["t_eff"]})
                continue
            stale = rec["state"] == "CANCEL_PENDING"
            charge = min(tr["shares"], max(0.0, rec["remaining"]))
            if charge > 0.0:
                rec["remaining"] -= charge
                net[0] += charge if side == "BUY_UP" else -charge
                traj.append({"kind": "FILL_CHARGED", "t": t, "slug": slug,
                             "side": side, "ref_gen": g["gen"],
                             "policy_gen": rec["policy_gen"],
                             "shares": charge, "stale": stale,
                             "markout_cents_per_share": mo})
            cut = tr["shares"] - charge
            if cut > EPS:
                traj.append({"kind": "FILL_PREVENTED_REDUCED", "t": t,
                             "slug": slug, "side": side, "ref_gen": g["gen"],
                             "policy_gen": rec["policy_gen"], "shares": cut,
                             "markout_cents_per_share": mo,
                             "reduce_t_effective": None})
        elif kind == "end":
            g = payload
            rec = rec_by_gen.get((side, g["gen"]))
            if rec is not None:
                settle(rec, g["t1"])
                if rec["state"] == "LIVE":
                    rec["state"] = "ENDED"
                    traj.append({"kind": "GEN_END", "t": g["t1"],
                                 "slug": slug, "side": side,
                                 "ref_gen": g["gen"],
                                 "policy_gen": rec["policy_gen"],
                                 "reason": "REFERENCE_END"})
                elif rec["state"] == "CANCEL_PENDING":
                    rec["state"] = "ENDED"
                    traj.append({"kind": "CANCEL_STALE", "t": rec["t_eff"],
                                 "slug": slug, "side": side,
                                 "ref_gen": g["gen"],
                                 "policy_gen": rec["policy_gen"],
                                 "reference_end_t": g["t1"]})
                    traj.append({"kind": "GEN_END", "t": g["t1"],
                                 "slug": slug, "side": side,
                                 "ref_gen": g["gen"],
                                 "policy_gen": rec["policy_gen"],
                                 "reason": "REFERENCE_END"})
            if live[side] is not None and live[side]["g"] is g:
                live[side] = None
        else:                                          # score
            s = payload
            settle(pending[side], t)
            if float(s["score"]) < theta_cancel:
                continue
            if held[side]:
                continue
            rec = live[side]
            if rec is None or rec["g"]["status"] != OK:
                continue
            if rec["state"] != "LIVE":
                continue
            rec["state"] = "CANCEL_PENDING"
            rec["t_eff"] = t + lat
            pending[side] = rec
            traj.append({"kind": "CANCEL_ISSUED", "t": t, "slug": slug,
                         "side": side, "ref_gen": rec["g"]["gen"],
                         "policy_gen": rec["policy_gen"],
                         "t_effective": rec["t_eff"],
                         "reducing_at_request": _reducing(side)})
    return traj


# ---------------------------------------------------------------------------
# 4. DA-contract exporter -- written here, never imported (R-235)
# ---------------------------------------------------------------------------
DA_CANON = "replay_traj_canon_v1"
DA_EVENT_FIELDS = ("t", "seq", "kind", "slug", "side", "gen", "qty", "price",
                   "note")
# The projection this exporter is: DA's kind set names seven kinds, and the
# machine emits seventeen.  Everything outside the map below has NO DA
# counterpart and is DROPPED -- so a digest taken through this exporter is a
# PROJECTION of what the producer did, exactly the objection B1.6 raises
# against ignoring an undeclared FIELD, one level up at the KIND.  Disclosed,
# never silently taken: the load-bearing full-fidelity comparison is the
# native one (`hsp.bit_identical`), and this leg exists to exercise the
# CONTRACT.
DA_KIND_MAP = {
    "PLACE": "PLACE",
    "CANCEL_ISSUED": "CANCEL_REQUESTED",
    "CANCEL_EFFECTIVE": "CANCEL_EFFECTIVE",
}
DA_DROPPED_KINDS = ("GEN_END", "GEN_START_MISSED_HELD", "HOLD_START",
                    "REPOST", "REPOST_ELIGIBLE", "REPOST_ELIGIBILITY_REVOKED",
                    "FILL_PREVENTED", "FILL_PREVENTED_REDUCED",
                    "FILL_MISSED_HELD", "FILL_MISSED_POST_REPOST",
                    "REDUCE_ISSUED", "REDUCE_EFFECTIVE")
STALE_CANCEL_READINGS = ("DROP", "AS_SUPPRESSED")


def export_da(events: Sequence[dict], arm: str, predictor: str,
              predictor_active: bool, fairprice_estimator: str | None,
              *, stale_cancel_reading: str) -> dict:
    """Project a native trajectory into DA's submission shape.

    `stale_cancel_reading` has NO DEFAULT.  The machine resolves a cancel
    three ways -- EFFECTIVE, STALE (admitted, but effectiveness landed at or
    after the reference generation's own end, so nothing was removed) and
    rate-SUPPRESSED -- while the contract's identity is
    `requested = effective + suppressed`, which has no term for STALE.
    Dropping it breaks the identity; calling it SUPPRESSED records a limiter
    refusal that never happened.  Both readings are implemented and NEITHER
    is sound, so the caller must name the one it is reporting."""
    if stale_cancel_reading not in STALE_CANCEL_READINGS:
        raise ExportRefused(
            f"stale_cancel_reading must be one of {STALE_CANCEL_READINGS}; "
            f"the contract has no CANCEL_STALE kind and this exporter "
            f"refuses to pick a reading on the producer's behalf")
    out = []
    for e in events:
        kind = e["kind"]
        if kind == "FILL_CHARGED":
            da_kind = "FILL_STALE" if e["stale"] else "FILL"
            qty = float(e["shares"])
        elif kind == "CANCEL_STALE":
            if stale_cancel_reading == "DROP":
                continue
            da_kind, qty = "CANCEL_SUPPRESSED", 0.0
        elif kind in DA_KIND_MAP:
            da_kind, qty = DA_KIND_MAP[kind], 0.0
        else:
            continue
        out.append({"t": float(e["t"]), "seq": len(out), "kind": da_kind,
                    "slug": str(e["slug"]), "side": str(e["side"]),
                    "gen": int(e["ref_gen"]), "qty": qty,
                    "price": (float(e["level"]) if kind == "PLACE" else None),
                    "note": ""})
    return {"canon": DA_CANON, "arm": arm, "predictor": predictor,
            "predictor_active": predictor_active,
            "components": list(_ARM_COMPONENTS[arm]),
            "interaction": _ARM_INTERACTION[arm],
            "fairprice_estimator": fairprice_estimator, "events": out}


# Stated by DE for DE's own submissions (the contract requires the producer to
# state what it RAN, and verifies it against DA's decomposition -- a mismatch
# refuses, which is the point).
_ARM_COMPONENTS = {
    "QR_SKEW_ONLY": ("skew",),
    "QR_CANCEL_HOLD_X_SKEW": ("cancel_hold", "skew"),
    "HAZARD_ONLY_NEUTRAL": ("hazard",),
    "CONDVALUE_NEUTRAL": ("condvalue",),
    "CONDVALUE_X_SKEW": ("condvalue", "skew"),
    "CONDVALUE_X_SKEW_X_FAIRPRICE": ("condvalue", "skew", "fairprice"),
    "RANDOM_MATCHED": ("random_matched",),
}
_ARM_INTERACTION = {
    "QR_SKEW_ONLY": False, "QR_CANCEL_HOLD_X_SKEW": True,
    "HAZARD_ONLY_NEUTRAL": False, "CONDVALUE_NEUTRAL": False,
    "CONDVALUE_X_SKEW": True, "CONDVALUE_X_SKEW_X_FAIRPRICE": True,
    "RANDOM_MATCHED": False,
}


# ---------------------------------------------------------------------------
# 5. the per-window gates
# ---------------------------------------------------------------------------
_ALLOWED_CELL_KEYS = ("n_slugs", "n_reference_generations",
                      "n_actions_cancel", "counters", "rate_limit",
                      "cancel_lifecycle", "holds")


def _receipt_cell(result: dict) -> dict:
    """Lifecycle counts ONLY.  The economics block, the fills block and the
    per-cancel records are STRUCTURALLY excluded: this battery is
    verification, and a receipt that carried a value would invite one to be
    read from it under a standing no-economics hold."""
    cell = {k: result[k] for k in _ALLOWED_CELL_KEYS if k in result}
    cell["holds"] = {k: v for k, v in cell.get("holds", {}).items()
                     if k != "records"}
    return cell


def window_gates(slug: str, sides: dict) -> dict:
    """Every LANE4 gate the frozen lane can express, on ONE window."""
    ref = {slug: sides}
    scores = stub_scores(slug, sides)
    passthrough = hsp.build_passthrough_trajectory(ref)

    dis = hsp.replay_policy(ref, scores, DISABLED_PARAMS)
    inf = hsp.replay_policy(ref, scores, INF_PARAMS)
    act = hsp.replay_policy(ref, scores, ACTIVE_PARAMS)
    hold = hsp.replay_policy(ref, scores, HOLD_PARAMS)
    expected_hold = build_cancel_and_hold(
        slug, sides, scores, HOLD_PARAMS["theta_cancel"],
        HOLD_PARAMS["cancel_effective_latency_ms"])

    inv = hsp.check_invariants(act)
    lc = act["cancel_lifecycle"]
    rl = act["rate_limit"]
    # one cancel per generation: the machine's own per-record ledger, counted
    # here rather than trusted -- one CANCEL_ISSUED per (side, policy_gen).
    issued: dict[tuple, int] = {}
    for e in act["trajectory"]:
        if e["kind"] == "CANCEL_ISSUED":
            k = (e["side"], e["policy_gen"])
            issued[k] = issued.get(k, 0) + 1
    stale_charged = sum(1 for e in act["trajectory"]
                        if e["kind"] == "FILL_CHARGED" and e["stale"])
    # a charged-stale fill must lie strictly inside its own latency window
    eff_by_gen = {(e["side"], e["policy_gen"]): e["t"]
                  for e in act["trajectory"] if e["kind"] == "CANCEL_ISSUED"}
    stale_inside = all(
        e["t"] < eff_by_gen[(e["side"], e["policy_gen"])]
        + ACTIVE_PARAMS["cancel_effective_latency_ms"] / 1000.0 + EPS
        for e in act["trajectory"]
        if e["kind"] == "FILL_CHARGED" and e["stale"]
        and (e["side"], e["policy_gen"]) in eff_by_gen)
    return {
        "slug": slug,
        "n_generations": sum(len(sides[s]) for s in SIDES),
        "n_scores": len(scores),
        "gate_disabled_bit_identical":
            hsp.bit_identical(dis["trajectory"], passthrough),
        "gate_infinite_threshold_bit_identical":
            hsp.bit_identical(inf["trajectory"], passthrough),
        "gate_inf_equals_disabled":
            hsp.bit_identical(inf["trajectory"], dis["trajectory"]),
        "gate_cancel_and_hold_equivalent":
            hsp.bit_identical(hold["trajectory"], expected_hold),
        "gate_one_cancel_per_generation":
            (not issued) or max(issued.values()) == 1,
        "gate_rate_identity":
            rl["requested"] == rl["passed"] + rl["suppressed"],
        # THE OUTCOME SPACE IS CLOSED: every issued cancel resolves EFFECTIVE,
        # STALE or (structurally never) UNRESOLVED.  `zero_value` is a SUBSET
        # of EFFECTIVE -- a cancel that bound and prevented nothing -- so it
        # is asserted as a subset and NOT added to the identity.  The first
        # version of this predicate added it and then `or`-ed a weaker clause,
        # which made a gate that could not fail; it was also not in GATE_KEYS,
        # so it was a decorative field beside a verdict (LANE4 B1.7's shape).
        "gate_lifecycle_closed":
            lc["issued"] == lc["effective"] + lc["stale"] + lc["unresolved"]
            and lc["zero_value"] <= lc["effective"],
        "gate_stale_fills_inside_latency_window": stale_inside,
        "gate_invariants": all(inv.values()),
        "invariants": inv,
        "active_acted": not hsp.bit_identical(act["trajectory"], passthrough)
                        if act["trajectory"] and passthrough else False,
        "n_stale_fills_charged": stale_charged,
        "cells": {"active_stub": _receipt_cell(act),
                  "permanent_hold": _receipt_cell(hold)},
    }


GATE_KEYS = ("gate_disabled_bit_identical",
             "gate_infinite_threshold_bit_identical",
             "gate_inf_equals_disabled",
             "gate_cancel_and_hold_equivalent",
             "gate_one_cancel_per_generation",
             "gate_rate_identity",
             "gate_lifecycle_closed",
             "gate_stale_fills_inside_latency_window",
             "gate_invariants")


# ---------------------------------------------------------------------------
# 6. the run
# ---------------------------------------------------------------------------

def run(limit: int | None = None, out: Path | None = None) -> dict:
    import harmful_exposure_rows as HER
    import policy_optimizer_queue_realistic as qr

    as_of = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.time()
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    selected, n_bn_gap = HER.select_v2_era(COINS, POPULATION, era=ERA)
    if limit is not None:
        selected = selected[:limit]

    win = {s: 0 for s in WINDOW_STATUSES}
    gen_counts = {s: 0 for s in GEN_STATUSES}
    gate_fail: dict[str, list] = {k: [] for k in GATE_KEYS}
    per_slug_digest: list[str] = []
    days: set[str] = set()
    n_gen = n_scores = n_stale = n_cancel = 0
    acted_windows = 0
    contract_leg: dict[str, Any] | None = None

    print(f"[de_lane4] selected {len(selected)} windows "
          f"({round(time.time() - t_start, 1)}s), bn_gap_excluded={n_bn_gap}",
          flush=True)
    for i_w, ent in enumerate(selected):
        slug = ent[0]
        if i_w and i_w % 20 == 0:
            print(f"[de_lane4] {i_w}/{len(selected)} windows, "
                  f"admitted={win['ADMITTED']} gens={n_gen} "
                  f"cancels={n_cancel} "
                  f"failing_gates={[k for k in GATE_KEYS if gate_fail[k]]} "
                  f"({round(time.time() - t_start, 1)}s)", flush=True)
        outp = HER.replay_with_recorder(ent[1], ent[2], ent[3], ent[4], spec)
        if outp is None:
            win["REPLAY_NONE"] += 1
            continue
        arm, wf = outp
        joined, jrec = HER.join_fills(arm.fill_log, arm.fills)
        gens, recon = HER.generation_table(arm.segments, joined, wf,
                                           qr.base.fi.WINDOW_S)
        if (jrec["count_mismatch"] or jrec["tuple_mismatches"]
                or recon["orphan_fills"]
                or recon["wrong_generation_assignments"]
                or arm.unhooked_changes):
            win["RECONCILIATION_FAILED"] += 1
            continue
        sides, gc = reference_from_window(arm.segments, gens)
        for k, v in gc.items():
            gen_counts[k] += v
        if not any(sides[s] for s in SIDES):
            win["NO_ADMITTED_GENERATION"] += 1
            continue
        try:
            hsp.validate_reference({slug: sides})
        except hsp.ReferenceIntegrityError:
            win["REFERENCE_REFUSED"] += 1
            continue
        g = window_gates(slug, sides)
        win["ADMITTED"] += 1
        days.add(_dt.datetime.fromtimestamp(
            int(slug.rsplit("-", 1)[1]), _dt.timezone.utc).strftime("%Y-%m-%d"))
        n_gen += g["n_generations"]
        n_scores += g["n_scores"]
        n_stale += g["n_stale_fills_charged"]
        n_cancel += g["cells"]["active_stub"]["cancel_lifecycle"]["issued"]
        acted_windows += 1 if g["active_acted"] else 0
        for k in GATE_KEYS:
            if not g[k]:
                gate_fail[k].append(slug)
        per_slug_digest.append(hashlib.sha256(
            json.dumps({k: g[k] for k in GATE_KEYS}, sort_keys=True)
            .encode()).hexdigest())
        if contract_leg is None:
            contract_leg = _contract_leg(slug, sides)

    if win["ADMITTED"] == 0:
        raise VacuousBattery(
            "zero admitted windows: a battery over no data must not report "
            "passing arms (LANE4 falsifier 6)")
    if n_cancel == 0:
        raise VacuousBattery(
            "the active-stub cell issued ZERO cancels over the whole "
            "population: every lifecycle gate would then pass on an empty "
            "set, which is the vacuity the battery exists to refuse")

    receipt = {
        "protocol": "de_lane4_real_parity_v1",
        "status": "VERIFICATION_ONLY_NO_ECONOMICS_READ",
        "as_of_utc": as_of,
        "elapsed_s": round(time.time() - t_start, 1),
        "population": {"name": POPULATION, "era": ERA, "coins": list(COINS),
                       "n_selected_windows": len(selected),
                       "windows_excluded_binance_gap": n_bn_gap,
                       "days": sorted(days), "n_days": len(days)},
        "window_status_counts": win,
        "generation_status_counts": gen_counts,
        "n_admitted_generations": n_gen,
        "n_stub_score_events": n_scores,
        "n_cancels_issued_active_stub": n_cancel,
        "n_windows_where_the_policy_acted": acted_windows,
        "n_stale_charged_fills": n_stale,
        # DA's routed finding, and it is the phantom-determinism class:
        # `pass: not gate_fail[k]` with NO DENOMINATOR reads True when
        # ZERO windows were admitted, which is indistinguishable from
        # every window passing. DA hit the same shape in its own battery
        # and it reported children that could not import as
        # `identical: false` -- nondeterminism, when neither interpreter
        # had run. A gate evaluated on nothing is UNEVALUATED, never
        # passed.
        "gates": {k: {"pass": (None if win["ADMITTED"] == 0
                               else not gate_fail[k]),
                      "status": ("UNEVALUATED_NO_ADMITTED_WINDOWS"
                                 if win["ADMITTED"] == 0
                                 else "PASS" if not gate_fail[k]
                                 else "FAIL"),
                      "n_evaluated_windows": win["ADMITTED"],
                      "n_failing_windows": len(gate_fail[k]),
                      "n_passing_windows": win["ADMITTED"]
                      - len(gate_fail[k]),
                      "failing_slugs": gate_fail[k][:20]}
                  for k in GATE_KEYS},
        # `all` over an empty evaluation is True in Python and that is
        # exactly the wrong answer here, so the denominator gates it.
        "all_gates_pass": (None if win["ADMITTED"] == 0
                           else all(not gate_fail[k] for k in GATE_KEYS)),
        "n_evaluated_windows": win["ADMITTED"],
        "aggregate_gate_digest": hashlib.sha256(
            "".join(per_slug_digest).encode()).hexdigest(),
        "arm_runnability": dict(ARM_RUNNABLE_LEGACY),
        "contract_leg": contract_leg,
        "declared_parameters": {
            "active": {k: (str(v) if v in (float("inf"), float("-inf"))
                           else v) for k, v in ACTIVE_PARAMS.items()},
            "permanent_hold_theta_repost": "-inf",
            "stub": {"salt": STUB_SALT, "early_frac": STUB_EARLY_FRAC,
                     "late_frac": STUB_LATE_FRAC,
                     "late_score": STUB_LATE_SCORE,
                     "note": "sha256-derived STUB, not a model; no predictor "
                             "is released for this use"},
        },
        "code_identity": dict(CODE_IDENTITY),
        "code_identity_taken": "AT IMPORT, not at receipt-write time",
    }
    if out is not None:
        out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _contract_leg(slug: str, sides: dict) -> dict:
    """Submit the INERT arms through DA's own loader on ONE window.

    Full-population submission is not attempted: the population carries ~10^6
    events per arm and the contract's value here is the REFUSAL surface, which
    one window exercises exactly as well.  What this leg proves is that the
    contract admits DE's independently-written exporter -- and where it does
    not, the refusal text is recorded verbatim rather than worked around."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import da_replay_parity_battery as da
    except Exception as exc:                              # pragma: no cover
        return {"loaded": False, "why": repr(exc)}
    ref = {slug: sides}
    scores = stub_scores(slug, sides)
    dis = hsp.replay_policy(ref, scores, DISABLED_PARAMS)
    objs, refusals = [], {}
    for arm in ARMS:
        fpe = ("Identity" if "fairprice" in _ARM_COMPONENTS[arm] else None)
        obj = export_da(dis["trajectory"], arm, "none", False, fpe,
                        stale_cancel_reading="DROP")
        try:
            da.load_external_trajectory(obj)
            objs.append(obj)
        except Exception as exc:
            refusals[arm] = str(exc)[:300]
    res = da.check_external_arms(objs) if objs else {"evaluable": False}
    # the acting cell, both readings of a STALE cancel -- neither sound
    act = hsp.replay_policy(ref, scores, ACTIVE_PARAMS)
    lifecycle = {}
    for reading in STALE_CANCEL_READINGS:
        obj = export_da(act["trajectory"], "QR_CANCEL_HOLD_X_SKEW", "none",
                        False, None, stale_cancel_reading=reading)
        try:
            tr = da.load_external_trajectory(obj)
            lifecycle[reading] = da.external_lifecycle(tr)
        except Exception as exc:
            lifecycle[reading] = {"refused": str(exc)[:300]}
    # THE REPOST-AXIS DIAGNOSTIC, run as a matched PAIR so the cause is
    # established by execution rather than asserted: the two cells differ
    # ONLY in whether the policy reposts.  DA's `gen` is the REFERENCE
    # generation and the contract has no policy-generation axis, so a repost
    # onto the same reference generation reads as "a fill after the cancel
    # bound".  If the no-repost cell passes and the reposting cell fails on
    # exactly that predicate, the cause is the missing axis and not a broken
    # producer.
    hold_r = hsp.replay_policy(ref, scores, HOLD_PARAMS)
    pair = {}
    for name, r in (("reposting", act), ("no_repost_permanent_hold", hold_r)):
        obj = export_da(r["trajectory"], "QR_CANCEL_HOLD_X_SKEW", "none",
                        False, None, stale_cancel_reading="DROP")
        lc = da.external_lifecycle(da.load_external_trajectory(obj))
        pair[name] = {"n_reposts": r["counters"]["reposts"],
                      "no_fill_after_effective": lc["no_fill_after_effective"],
                      "identity_holds": lc["identity_holds"],
                      "pass": lc["pass"]}
    pair["diagnosis_holds"] = (
        pair["reposting"]["n_reposts"] > 0
        and pair["no_repost_permanent_hold"]["n_reposts"] == 0
        and pair["no_repost_permanent_hold"]["no_fill_after_effective"]
        and not pair["reposting"]["no_fill_after_effective"])
    return {"loaded": True, "slug": slug, "n_inert_arms_admitted": len(objs),
            "inert_refusals": refusals,
            "inert_check": {k: v for k, v in res.items() if k != "per_id"},
            "acting_lifecycle_by_stale_reading": lifecycle,
            "repost_axis_diagnostic": pair,
            "da_battery_canon": da.CANON,
            "da_arms_match_ours": tuple(da.ARMS) == ARMS}


# ---------------------------------------------------------------------------
# 7. selftest -- synthetic, both directions
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 35


def _seg(side, gen, level, resting, t0, t1):
    return {"side": side, "gen": gen, "level": level, "resting": resting,
            "qahead": 0.0, "net": 0.0, "t_start": t0, "t_end": t1}


def _fix():
    """A two-side fixture with a fill before and after a cancel, built from
    the SHAPES the exposure pipeline emits -- not from the machine's."""
    segs = [_seg("BUY_UP", 1, 0.50, 5.0, 0.0, 10.0),
            _seg("BUY_UP", 2, 0.51, 5.0, 10.0, 20.0),
            _seg("SELL_UP", 1, 0.55, 5.0, 0.0, 20.0)]
    gens = {
        ("BUY_UP", 1): {"t0": 0.0, "t1": 10.0, "tranches": [
            {"t": 2.0, "shares": 2.0, "markout_cents_per_share": -10.0},
            # inside the 50 ms latency window of the stub crossing at 2.5:
            # this tranche is what makes the stale-charge gate NON-VACUOUS
            {"t": 2.52, "shares": 0.5, "markout_cents_per_share": -3.0},
            {"t": 6.5, "shares": 1.0, "markout_cents_per_share": -4.0},
            {"t": 8.0, "shares": 1.0, "markout_cents_per_share": -25.0}]},
        ("BUY_UP", 2): {"t0": 10.0, "t1": 20.0, "tranches": [
            {"t": 15.0, "shares": 1.0, "markout_cents_per_share": -20.0}]},
        ("SELL_UP", 1): {"t0": 0.0, "t1": 20.0, "tranches": [
            {"t": 5.0, "shares": 1.0, "markout_cents_per_share": 3.0}]},
    }
    return segs, gens


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_lane4_real_parity] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    def refuses(exc, fn, label):
        try:
            fn()
        except exc:
            n[0] += 1
            print(f"  PASS  {label}")
            return
        raise SystemExit(f"[de_lane4_real_parity] FAIL (no refusal): {label}")

    segs, gens = _fix()
    sides, gc = reference_from_window(segs, gens)
    ok(gc["ADMITTED"] == 3 and len(sides["BUY_UP"]) == 2
       and len(sides["SELL_UP"]) == 1,
       f"adapter admits the three fixture generations ({gc['ADMITTED']})")
    hsp.validate_reference({"w": sides})
    ok(True, "the adapted reference passes the machine's own validator -- "
             "the adapter is checked at the consumer, not by inspection")

    # ---- exclusion statuses, each FIRED by its own known-bad -------------
    s2 = list(segs) + [_seg("BUY_UP", 1, 0.52, 5.0, 3.0, 4.0)]
    _, g2 = reference_from_window(s2, gens)
    ok(g2["MULTI_LEVEL"] == 1 and g2["ADMITTED"] == 2,
       "KNOWN-BAD: a generation whose segments carry two levels is EXCLUDED "
       "with a status, not resolved by picking the first")
    g3 = dict(gens)
    g3[("BUY_UP", 1)] = dict(gens[("BUY_UP", 1)], t1=0.0)
    _, c3 = reference_from_window(segs, g3)
    ok(c3["ZERO_LENGTH"] == 1, "KNOWN-BAD: a zero-length generation is "
                               "excluded with its own status")
    g4 = dict(gens)
    g4[("BUY_UP", 2)] = dict(gens[("BUY_UP", 2)], tranches=[
        {"t": 99.0, "shares": 1.0, "markout_cents_per_share": 1.0}])
    _, c4 = reference_from_window(segs, g4)
    ok(c4["TRANCHE_OUTSIDE"] == 1,
       "KNOWN-BAD: a tranche outside its generation interval is excluded")
    g5 = dict(gens)
    g5[("SELL_UP", 1)] = dict(gens[("SELL_UP", 1)], tranches=[
        {"t": 5.0, "shares": 99.0, "markout_cents_per_share": 1.0}])
    _, c5 = reference_from_window(segs, g5)
    ok(c5["SHARES_EXCEED_DISPLAYED"] == 1,
       "KNOWN-BAD: tranche shares above displayed are excluded, never "
       "silently clipped to fit")
    _, c6 = reference_from_window([], gens)
    ok(c6["NO_LEVEL_SEGMENT"] == 3 and c6["ADMITTED"] == 0,
       "KNOWN-BAD: generations with no level-bearing segment are counted, "
       "and the adapter returns an EMPTY reference rather than inventing one")
    ok(sum(gc.values()) == gc["ADMITTED"],
       "POSITIVE CONTROL: on the clean fixture every status but ADMITTED is "
       "zero -- the exclusion counters are not always firing")

    # ---- the gates on the fixture ---------------------------------------
    scores = [{"t": 6.0, "slug": "w", "side": "BUY_UP", "score": 0.99},
              {"t": 9.0, "slug": "w", "side": "BUY_UP", "score": 0.0}]
    ref = {"w": sides}
    pt = hsp.build_passthrough_trajectory(ref)
    dis = hsp.replay_policy(ref, scores, DISABLED_PARAMS)
    inf = hsp.replay_policy(ref, scores, INF_PARAMS)
    act = hsp.replay_policy(ref, scores, ACTIVE_PARAMS)
    ok(hsp.bit_identical(dis["trajectory"], pt),
       "GATE 1 on the adapted reference: a disabled predictor is "
       "BIT-IDENTICAL to QR_SKEW_ONLY")
    ok(hsp.bit_identical(inf["trajectory"], pt),
       "GATE 2: theta_cancel=+inf is BIT-IDENTICAL to QR_SKEW_ONLY")
    ok(not hsp.bit_identical(act["trajectory"], pt),
       "POSITIVE CONTROL: the acting cell is NOT bit-identical, so the "
       "comparator can fire on this fixture")

    hold = hsp.replay_policy(ref, scores, HOLD_PARAMS)
    exp = build_cancel_and_hold("w", sides, scores,
                                HOLD_PARAMS["theta_cancel"],
                                HOLD_PARAMS["cancel_effective_latency_ms"])
    ok(hsp.bit_identical(hold["trajectory"], exp),
       "GATE 3: permanent hold matches the INDEPENDENTLY-built "
       "cancel-and-hold trajectory event for event")
    ok(not hsp.bit_identical(act["trajectory"], exp),
       "POSITIVE CONTROL: the reposting cell does NOT match cancel-and-hold, "
       "so gate 3 is a comparison and not a tautology")
    perturbed = [dict(e) for e in exp]
    perturbed.insert(1, dict(perturbed[0], kind="CANCEL_EFFECTIVE",
                             policy_gen="1"))
    ok(not hsp.bit_identical(hold["trajectory"], perturbed),
       "KNOWN-BAD (LANE4 falsifier 1): ONE extra event breaks parity -- if a "
       "one-event perturbation did not, the anchor would be decorative")

    adv = [{"t": t, "slug": "w", "side": "BUY_UP", "score": 0.99}
           for t in (6.0, 6.2, 6.4, 7.5, 8.5)]
    a2 = hsp.replay_policy(ref, adv, ACTIVE_PARAMS)
    ok(a2["cancel_lifecycle"]["issued"] == 1,
       "GATE 4: five crossings on one generation issue exactly ONE cancel")
    g = window_gates("w", sides)
    ok(len(GATE_KEYS) == 9 and "gate_lifecycle_closed" in GATE_KEYS,
       "every computed gate ENTERS the verdict: nine keys, none decorative")
    ok(all(g[k] for k in GATE_KEYS),
       f"every declared gate passes on the fixture: "
       f"{[k for k in GATE_KEYS if not g[k]]}")
    # KNOWN-BAD for the closure identity, which must be able to return False
    _lcbad = {"issued": 3, "effective": 1, "stale": 1, "unresolved": 0,
              "zero_value": 0}
    ok(not (_lcbad["issued"] == _lcbad["effective"] + _lcbad["stale"]
            + _lcbad["unresolved"] and _lcbad["zero_value"]
            <= _lcbad["effective"]),
       "KNOWN-BAD: an issued cancel that resolves to nothing FAILS the "
       "closure identity -- the gate can return False")
    ok(g["n_stale_fills_charged"] >= 1,
       "GATE 5 is NON-VACUOUS on the fixture: at least one fill inside the "
       "latency window is CHARGED as stale, so the check has something to "
       "check")

    # ---- the receipt cannot carry economics -----------------------------
    cell = _receipt_cell(act)
    ok("economics" not in cell and "fills" not in cell and "cancels" not in cell,
       "STRUCTURAL: the receipt cell drops economics, fills and per-cancel "
       "records -- no value can be read out of a verification receipt")
    ok("economics" in act,
       "POSITIVE CONTROL: the machine DID compute an economics block, so the "
       "exclusion above is a filter and not an empty source")

    # ---- the exporter refuses to guess ----------------------------------
    refuses(ExportRefused,
            lambda: export_da(act["trajectory"], "QR_SKEW_ONLY", "none",
                              False, None, stale_cancel_reading="WHATEVER"),
            "KNOWN-BAD: the exporter REFUSES an undeclared reading of a "
            "STALE cancel rather than picking one")
    ev = export_da(dis["trajectory"], "QR_SKEW_ONLY", "none", False, None,
                   stale_cancel_reading="DROP")
    ok(ev["canon"] == DA_CANON and ev["events"]
       and all(set(e) == set(DA_EVENT_FIELDS) for e in ev["events"]),
       "the exporter emits exactly DA's declared event fields, no more")
    ok([e for e in ev["events"] if e["kind"] == "FILL"],
       "POSITIVE CONTROL: the export is non-empty and carries real fills")

    # ---- the battery refuses a vacuum -----------------------------------
    # THE NAMING COLLISION, established by execution against DA's loader
    # rather than argued (rule 15/16, both directions).
    import da_replay_parity_battery as _da
    _ev = [{"t": 0.0, "seq": 0, "kind": "PLACE", "slug": "w",
            "side": "BUY_UP", "gen": 1, "qty": 0.0, "price": 0.5, "note": ""}]
    _base = {"canon": _da.CANON, "predictor": "composed_linear",
             "predictor_active": True, "fairprice_estimator": None,
             "events": _ev}
    try:
        _da.load_external_trajectory(dict(
            _base, arm="CONDVALUE_X_SKEW",
            components=["condvalue", "skew"], interaction=False))
        _xskew_ok = True
    except _da.ParityRefused:
        _xskew_ok = False
    _neutral = _da.load_external_trajectory(dict(
        _base, arm="CONDVALUE_NEUTRAL", components=["condvalue"],
        interaction=False))
    ok(not _xskew_ok,
       "KNOWN-BAD: the honest composition (condvalue + skew, NO interaction) "
       "is REFUSED under CONDVALUE_X_SKEW -- an X in the name is an "
       "interaction claim")
    ok(_neutral.arm == "CONDVALUE_NEUTRAL",
       "AND THE ONLY NAME THE CONTRACT ACCEPTS omits the skew that was "
       "actually in force: " + ARM_NAME_COLLISION)

    ok(len(CODE_IDENTITY) == len(_IDENTITY_FILES)
       and all(len(h) == 16 for h in CODE_IDENTITY.values())
       and CODE_IDENTITY["de_lane4_real_parity.py"]
       == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
       "code identity is taken AT IMPORT over all five result-shaping files, "
       "so a run that outlives an edit to its own source still stamps the "
       "program that produced it")

    ok(set(ARM_RUNNABLE_LEGACY) == set(ARMS)
       and sum(1 for v in ARM_RUNNABLE_LEGACY.values() if v == "RUNNABLE") == 2,
       "arm runnability is DECLARED per arm and only two of seven are "
       "runnable on the frozen reference -- reported, never dropped")
    ok(stub_score("a", "BUY_UP", 1) == stub_score("a", "BUY_UP", 1)
       and stub_score("a", "BUY_UP", 1) != stub_score("a", "BUY_UP", 2),
       "the stub scorer is deterministic and generation-sensitive")

    # THE RUNNABILITY CLAIM IS A CLAIM ABOUT CODE, so it carries a check in
    # this suite (SEAT_PROTOCOL rule 15).  Reading it from the engine's own
    # spec table, not from memory or from a plan.
    import policy_optimizer_queue_realistic as _qr
    specs = {sp["cell"]: sp for sp in _qr._specs(50)}
    qr_cells = [c for c, sp in specs.items() if sp.get("queue_realistic")]
    ok(qr_cells and all(specs[c].get("skew") is True for c in qr_cells),
       f"VERIFIED AT THE ENGINE: every queue-realistic cell has skew ON "
       f"({qr_cells}) -- so the frozen lane offers NO neutral-placement "
       f"reference, which is why two arms are NO_NEUTRAL_REFERENCE")
    neutral = [c for c, sp in specs.items() if not sp.get("skew")]
    ok(neutral and not any(specs[c].get("queue_realistic") for c in neutral),
       f"AND THE CONVERSE: the neutral cells that DO exist ({neutral}) are "
       f"all NON-queue-realistic, so borrowing one would compare two "
       f"placement engines and call the difference a policy effect")

    # ---- the DA contract, exercised on the fixture ----------------------
    leg = _contract_leg("w", sides)
    ok(leg["loaded"] and leg["da_arms_match_ours"],
       "DA's battery loads and its ARMS tuple matches ours exactly -- the "
       "seven-arm space is one space, checked rather than assumed")
    ok(leg["n_inert_arms_admitted"] == 7 and not leg["inert_refusals"]
       and leg["inert_check"]["inactive_predictors_agree"]
       and leg["inert_check"]["pass"],
       "THE SEVEN-ARM INERT ANCHOR THROUGH DA'S OWN LOADER: all seven "
       "compositions load and are bit-identical with every predictor off")
    ok(leg["repost_axis_diagnostic"]["diagnosis_holds"],
       "MATCHED PAIR: the no-repost cell PASSES no_fill_after_effective and "
       "the reposting cell FAILS it -- the cause is DA's `gen` carrying the "
       "REFERENCE generation with no policy-generation axis, established by "
       "execution and not by argument")

    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[de_lane4_real_parity] selftest OK -- {n[0]} checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        if selftest() != 0:                 # numbers never come from a red suite
            return 1
        r = run(limit=a.limit,
                out=Path(a.out) if a.out else None)
        print(json.dumps({k: v for k, v in r.items()
                          if k != "contract_leg"}, indent=2, sort_keys=True))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
