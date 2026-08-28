#!/usr/bin/env python3
"""BE-side trajectory export for DA's external-arm parity contract.

CONTRACT: da_replay_parity_battery.py, canon `replay_traj_canon_v1`. That module
states its external interface is a DATA contract and imports nothing from BE.

THIS MODULE DECLARES ITS OWN FIELD LIST AND DOES NOT IMPORT DA'S.

That is deliberate and it is the whole point. If BE imported DA's EVENT_FIELDS,
the two sides would agree BY CONSTRUCTION and the contract check would be
vacuous -- it would verify that one list equals itself. A data contract exists
so that two independent implementations must be SHOWN to agree, and the showing
is a falsifier that compares the two declarations (see agreement_with_contract).
Same shape as annotation_canon_v1: agreement PROVEN, never assumed.

The first real BE trajectory is the first thing that can falsify this contract,
so the export path is built and driven on synthetic data BEFORE any real run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

# --- BE's INDEPENDENT declaration of the contract, transcribed from the spec --
BE_CANON = "replay_traj_canon_v1"
BE_EVENT_FIELDS = ("t", "seq", "kind", "slug", "side", "gen", "qty", "price",
                   "note")
BE_KINDS = ("PLACE", "PLACE_WITHHELD", "CANCEL_REQUESTED", "CANCEL_EFFECTIVE",
            "CANCEL_SUPPRESSED", "FILL", "FILL_STALE")
# B2: identity is TWO-DIMENSIONAL (composition x predictor) and lives in
# top-level fields, exact in both directions like the event fields.
BE_TRAJ_FIELDS = ("canon", "arm", "predictor", "predictor_active",
                  "components", "interaction", "fairprice_estimator",
                  "events")
# What BE claims each exportable composition IS. Declared independently; the
# contract compares it to its own spec and refuses a mismatch.
BE_COMPONENTS = {"CONDVALUE_NEUTRAL": ("condvalue",)}


class ExportRefused(RuntimeError):
    """Refuse at the producer rather than emit something the consumer rejects.

    A producer that emits a malformed trajectory and lets the consumer refuse it
    has moved the error one process away from its cause."""


def make_event(t, seq, kind, slug, side, gen, qty, price, note="") -> dict:
    """One event, with EXACTLY the declared fields and nothing else."""
    if kind not in BE_KINDS:
        raise ExportRefused(f"unknown kind {kind!r}; declared: {BE_KINDS}")
    return {"t": float(t), "seq": int(seq), "kind": kind, "slug": str(slug),
            "side": str(side), "gen": int(gen), "qty": float(qty),
            "price": float(price), "note": str(note)}


def export_trajectory(arm: str, events: list, predictor: str = "none",
                      predictor_active: bool = False,
                      components=None, interaction: bool = False,
                      fairprice_estimator=None) -> dict:
    """A contract-shaped trajectory object, B2 identity included.

    Refuses EMPTY, because the contract refuses it downstream and an empty
    trajectory is a producer bug rather than a population statement."""
    if not events:
        raise ExportRefused(
            f"arm {arm!r} has NO events. An empty trajectory is a producer "
            f"defect; the contract refuses it and so should the producer.")
    for i, e in enumerate(events):
        missing = [f for f in BE_EVENT_FIELDS if f not in e]
        extra = [k for k in e if k not in BE_EVENT_FIELDS]
        if missing or extra:
            raise ExportRefused(
                f"event {i}: missing={missing} undeclared={extra}. The field "
                f"set is exact in both directions.")
    obj = {"canon": BE_CANON, "arm": arm, "predictor": predictor,
           "predictor_active": bool(predictor_active),
           "components": list(components if components is not None
                              else BE_COMPONENTS.get(arm, ())),
           "interaction": bool(interaction),
           "fairprice_estimator": fairprice_estimator,
           "events": list(events)}
    miss = [f for f in BE_TRAJ_FIELDS if f not in obj]
    extra = [k for k in obj if k not in BE_TRAJ_FIELDS]
    if miss or extra:
        raise ExportRefused(
            f"trajectory: missing={miss} undeclared={extra}. B2 identity "
            f"fields are exact in both directions, like the event fields.")
    return obj



# --- BE's DECLARED MAPPING (amendment B2) ----------------------------------
# WHICH composition each 011 predictor may export under, and with what
# interaction claim. Stated FROM THE CODE, verified below, not from intent.
#
# BOTH 011 predictors implement NO SKEW INTERACTION today, and the evidence is
# three independent readings:
#   1. build_design (phase2_iter011_run.py:41-44) shows each predictor x as
#      PM + fine + state. Both arms see the SAME features and differ only in
#      MODEL CLASS (R-232 9.1).
#   2. NO skew, inventory, net or front/reducing state reaches any of the three
#      blocks. The 45 pinned features contain none; the only near-hits are
#      queue_ahead_* , which is ORDER-BOOK POSITION at decision time, spatial
#      rather than inventory. harmful_hazard_model.features/fine_feats mention
#      none of skew/inventory/front/reducing at all.
#   3. The state machine states the fence in its own words: "the skew rules'
#      placement choices live entirely in this input; the predictor NEVER
#      chooses placement" and, on repost, "ordinary skew rules, never the
#      predictor" (harmful_stateful_policy.py:15-16, :43-45).
#
# So each predictor exports under CONDVALUE_NEUTRAL with interaction False, and
# the X_SKEW / X_SKEW_X_FAIRPRICE compositions have NO VALID 011 EXPORTER until
# the Phase-3 skew wiring exists. Those pairs are declared ABSENT, not emitted
# with a label the code cannot support: a mislabelled arm is the one error the
# contract structurally cannot catch, because the label is the thing it trusts.
BE_PREDICTORS = ("composed_linear", "composed_lgbm")
BE_EXPORTABLE_COMPOSITIONS = ("CONDVALUE_NEUTRAL",)
BE_ABSENT_COMPOSITIONS = {
    "CONDVALUE_X_SKEW": "no skew interaction is implemented in 011; awaiting "
                        "Phase-3 skew wiring",
    "CONDVALUE_X_SKEW_X_FAIRPRICE": "no skew interaction AND no fair-price "
                                    "component in 011",
}


def declared_pairs() -> dict:
    """The (composition x predictor) pairs BE can submit TODAY, and those it
    cannot. Race multiplicity counts PAIRS (B2), so absent pairs must be
    declared absent rather than silently uncounted."""
    present = [(c, p) for c in BE_EXPORTABLE_COMPOSITIONS
               for p in BE_PREDICTORS]
    absent = [(c, p) for c in sorted(BE_ABSENT_COMPOSITIONS)
              for p in BE_PREDICTORS]
    return {"present_pairs": present, "n_present": len(present),
            "absent_pairs": absent, "n_absent": len(absent),
            "absent_reasons": dict(BE_ABSENT_COMPOSITIONS),
            "interaction_claim": False,
            "why_interaction_false": "both predictors see PM+fine+state and no "
                                     "skew/inventory/front state reaches any "
                                     "block; the predictor never chooses "
                                     "placement (harmful_stateful_policy)"}


def export_trajectory_b2(composition: str, predictor: str,
                         events: list) -> dict:
    """B2 export: identity is composition x predictor, stated not inferred.

    REFUSES a composition BE cannot support today. The refusal is at the
    PRODUCER because the contract cannot catch a wrong label -- the label is
    what it trusts."""
    if predictor not in BE_PREDICTORS:
        raise ExportRefused(
            f"unknown BE predictor {predictor!r}; declared: {BE_PREDICTORS}")
    if composition in BE_ABSENT_COMPOSITIONS:
        raise ExportRefused(
            f"REFUSED: {composition} has NO valid 011 exporter -- "
            f"{BE_ABSENT_COMPOSITIONS[composition]}. Emitting it would put a "
            f"label on a trajectory the code cannot produce, which is the one "
            f"error the contract cannot catch.")
    if composition not in BE_EXPORTABLE_COMPOSITIONS:
        raise ExportRefused(
            f"REFUSED: {composition!r} is not a composition BE declares; "
            f"exportable today: {BE_EXPORTABLE_COMPOSITIONS}")
    return export_trajectory(
        composition, events, predictor=predictor, predictor_active=True,
        components=BE_COMPONENTS[composition], interaction=False,
        fairprice_estimator=None)


# --- BE's COMPOSITION-SEMANTICS DECLARATION (B4 / R-261) --------------------
# DA's multiplicity derivation REFUSES until the owner of composition semantics
# declares, per composition: does it CONSUME A PREDICTOR ESTIMATE, and is it a
# CANDIDATE (adoptable) or a CONTROL (null apparatus)?
#
# CITATION NOTE, stated because the ask specified plan text: the seven arms are
# NOT enumerated by name in STATEFUL_HARMFUL_CANCEL_TODO. Its section 9 item 5
# names "seven-arm offline replay" and section 10 sequences it, but the NAMES
# and their components exist only in code (da_replay_parity_battery.ARM_SPEC).
# So every entry below is cited to CODE; where the plan is silent and the code
# does not settle it, the entry is a QUESTION, not a resolution.
#
# consumes_predictor is READ OFF THE COMPONENTS, which are code:
#   a composition consumes an estimate iff "hazard" or "condvalue" is among its
#   components. "skew", "cancel_hold" and "random_matched" are RULES, not
#   estimators -- skew is _target_front(net, band) (placement_skew.py:64-80),
#   cancel_hold is the rule arm _qr_spec(..., cancel=True)
#   (policy_optimizer_queue_realistic.py:66-67), and random_matched is the
#   matched-random apparatus in harmful_action_eval (randoms "matched within
#   (side x hour) strata", :51-52).
BE_CONSUMES_PREDICTOR = {
    "QR_SKEW_ONLY":                 False,   # components ("skew",) -- a rule
    "QR_CANCEL_HOLD_X_SKEW":        False,   # cancel_hold is a RULE cancel
    "HAZARD_ONLY_NEUTRAL":          True,    # "hazard" IS an estimate
    "CONDVALUE_NEUTRAL":            True,    # "condvalue" IS an estimate
    "CONDVALUE_X_SKEW":             True,
    "CONDVALUE_X_SKEW_X_FAIRPRICE": True,
    "RANDOM_MATCHED":               False,   # random by construction
}

# ROLE. DA's own words at da_replay_parity_battery.py:208-213: "BEING IN THE
# PARITY SPACE AND BEING IN THE CANDIDATE SPACE ARE DIFFERENT". RANDOM_MATCHED
# is the null apparatus -- harmful_action_eval builds it as the CONTROL the
# decision metric is compared against -- so it runs in the replay and is not a
# selectable winner.
BE_ROLE = {
    "HAZARD_ONLY_NEUTRAL":          "candidate",
    "CONDVALUE_NEUTRAL":            "candidate",
    "CONDVALUE_X_SKEW":             "candidate",
    "CONDVALUE_X_SKEW_X_FAIRPRICE": "candidate",
    "RANDOM_MATCHED":               "control",
    # the two rule arms: see BE_ROLE_QUESTIONS. Declared "candidate" pending a
    # ruling, because that is the ANSWER THAT COUNTS THEM, and a multiplicity
    # that is too large is conservative while one that is too small is not.
    "QR_SKEW_ONLY":                 "candidate",
    "QR_CANCEL_HOLD_X_SKEW":        "candidate",
}

BE_ROLE_QUESTIONS = {
    "QR_SKEW_ONLY":
        "GENUINELY OPEN. It is the NEUTRAL INTEGRATION REFERENCE (my "
        "skew-lane draft freezes it as the no-cancel shadow every arm is "
        "measured against), which argues CONTROL. But 'do not cancel at all' "
        "is also an adoptable policy, and if no cancel arm beats it the race's "
        "honest answer IS this arm -- which argues CANDIDATE. Being the "
        "reference and being adoptable are not exclusive, and the code does "
        "not settle which the race means. Declared candidate pending a ruling "
        "because that choice is the conservative one for multiplicity.",
    "QR_CANCEL_HOLD_X_SKEW":
        "GENUINELY OPEN. A RULE-based cancel arm with no estimator "
        "(_qr_spec cancel=True). It is adoptable, which argues CANDIDATE; it "
        "is also the natural incumbent comparator for the predictor arms, "
        "which is a CONTROL-like use. The code says what it IS, not what the "
        "race intends it FOR.",
}


def composition_semantics() -> dict:
    """BE's declaration for DA's multiplicity derivation (B4)."""
    return {"consumes_predictor": dict(BE_CONSUMES_PREDICTOR),
            "role": dict(BE_ROLE),
            "open_questions": dict(BE_ROLE_QUESTIONS),
            "citation_note": "the seven arm NAMES are not in the plan text; "
                             "they exist in da_replay_parity_battery.ARM_SPEC, "
                             "so every entry is cited to CODE",
            "derivation_rule": "consumes_predictor is TRUE iff 'hazard' or "
                               "'condvalue' is among the composition's "
                               "components; skew / cancel_hold / "
                               "random_matched are rules, not estimators"}


def agreement_with_contract() -> dict:
    """PROVE that BE's independent declaration matches DA's. Not assume it.

    Imported ONLY here, in the check -- never in the producer -- so the export
    path stays independent while the agreement is demonstrated."""
    import da_replay_parity_battery as DA
    disagreements = {}
    if BE_CANON != DA.CANON:
        disagreements["canon"] = (BE_CANON, DA.CANON)
    if tuple(BE_EVENT_FIELDS) != tuple(DA.EVENT_FIELDS):
        disagreements["event_fields"] = (BE_EVENT_FIELDS, DA.EVENT_FIELDS)
    if tuple(sorted(BE_KINDS)) != tuple(sorted(DA.KINDS)):
        disagreements["kinds"] = (sorted(BE_KINDS), sorted(DA.KINDS))
    if tuple(BE_TRAJ_FIELDS) != tuple(DA.TRAJ_FIELDS):
        disagreements["traj_fields"] = (BE_TRAJ_FIELDS, DA.TRAJ_FIELDS)
    for _c, _comp in BE_COMPONENTS.items():
        _spec = DA.ARM_SPEC[_c]
        if tuple(sorted(_comp)) != tuple(sorted(_spec["components"])):
            disagreements[f"components:{_c}"] = (_comp, _spec["components"])
        if _spec["interaction"] is not False:
            disagreements[f"interaction:{_c}"] = _spec["interaction"]
    if disagreements:
        raise ExportRefused(
            f"BE's declaration DISAGREES with the contract: {disagreements}. "
            f"Two implementations that have never been shown to agree are not "
            f"a contract.")
    return {"agreed": True, "canon": BE_CANON,
            "n_event_fields": len(BE_EVENT_FIELDS),
            "n_traj_fields": len(BE_TRAJ_FIELDS), "n_kinds": len(BE_KINDS),
            "components_agree": sorted(BE_COMPONENTS),
            "declared_independently": True}


def selftest() -> int:
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    import da_replay_parity_battery as DA

    ok(agreement_with_contract()["agreed"],
       "0 BE's INDEPENDENT declaration is PROVEN to match DA's contract "
       "(importing DA's list would make this vacuous)")

    def ev(seq, kind, gen=1, t=None, **kw):
        return make_event(t if t is not None else seq * 1.0, seq, kind,
                          "btc-updown-5m-1787650200", "BUY_UP", gen, 5.0, 0.5,
                          **kw)

    good = export_trajectory("CONDVALUE_NEUTRAL",
                             [ev(1, "PLACE"), ev(2, "FILL")])
    tr = DA.load_external_trajectory(good)
    ok(tr.arm == "CONDVALUE_NEUTRAL" and len(tr.events) == 2,
       "1 a well-formed export LOADS through the real contract loader "
       "(the guard is not a wall)")
    ok(DA.external_lifecycle(tr)["arm"] == "CONDVALUE_NEUTRAL",
       "2 and passes the real lifecycle invariants")

    # --- the contract's refusal rules, each driven against the REAL loader ---
    for lbl, obj in (
        ("not an object", ["nope"]),
        ("wrong canon", dict(good, canon="canon_v99")),
        ("unknown arm", dict(good, arm="BE_MADE_THIS_UP")),
        ("empty events", dict(good, events=[])),
        ("event not an object", dict(good, events=["x"])),
    ):
        try:
            DA.load_external_trajectory(obj)
            ok(False, f"3 the contract REFUSES: {lbl}")
        except DA.ParityRefused:
            ok(True, f"3 the contract REFUSES: {lbl}")

    _miss = dict(good["events"][0]); _miss.pop("note")
    try:
        DA.load_external_trajectory(dict(good, events=[_miss]))
        ok(False, "4 a MISSING field is refused")
    except DA.ParityRefused as e:
        ok("MISSING" in str(e), "4 a MISSING field is refused by the contract")
    _ext = dict(good["events"][0]); _ext["be_extra"] = 1
    try:
        DA.load_external_trajectory(dict(good, events=[_ext]))
        ok(False, "5 an UNDECLARED field is refused")
    except DA.ParityRefused as e:
        ok("UNDECLARED" in str(e),
           "5 an UNDECLARED field is refused — the set is exact in BOTH "
           "directions, so a helpful extra column is a refusal")

    # --- BE refuses at the PRODUCER, not only at the consumer ---
    try:
        export_trajectory("CONDVALUE_NEUTRAL", [])
        ok(False, "6 BE refuses an EMPTY trajectory at the producer")
    except ExportRefused:
        ok(True, "6 BE refuses an EMPTY trajectory at the PRODUCER, not one "
                 "process away at the consumer")
    try:
        make_event(1.0, 1, "TELEPORT", "s", "BUY_UP", 1, 5.0, 0.5)
        ok(False, "7 BE refuses an unknown kind at the producer")
    except ExportRefused:
        ok(True, "7 BE refuses an unknown KIND at the producer")
    try:
        export_trajectory("X", [{"t": 1.0}])
        ok(False, "8 BE refuses an incomplete event at the producer")
    except ExportRefused as e:
        ok("missing=" in str(e), "8 BE refuses an INCOMPLETE event at the "
                                 "producer, naming the missing fields")

    # --- lifecycle invariants BE must satisfy, driven on synthetic arms ------
    two_req = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_REQUESTED"), ev(3, "CANCEL_REQUESTED"),
        ev(4, "CANCEL_EFFECTIVE")])
    r = DA.external_lifecycle(DA.load_external_trajectory(two_req))
    ok(any(v is False for k, v in r.items() if isinstance(v, bool)),
       "9 TWO cancel requests on one generation FAIL a lifecycle invariant")

    fill_after = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_REQUESTED"), ev(3, "CANCEL_EFFECTIVE"),
        ev(4, "FILL_STALE")])
    r2 = DA.external_lifecycle(DA.load_external_trajectory(fill_after))
    ok(any(v is False for k, v in r2.items() if isinstance(v, bool)),
       "10 a FILL_STALE AFTER effectiveness fails — STALE is DEFINED as "
       "pre-effectiveness, so that mislabel is exactly what the check catches")

    orphan = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_EFFECTIVE")])
    r3 = DA.external_lifecycle(DA.load_external_trajectory(orphan))
    ok(any(v is False for k, v in r3.items() if isinstance(v, bool)),
       "11 an EFFECTIVE cancel with no REQUEST fails — otherwise "
       "requested==effective+suppressed could be satisfied by two "
       "compensating errors")

    clean = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_REQUESTED"), ev(3, "CANCEL_EFFECTIVE")])
    r4 = DA.external_lifecycle(DA.load_external_trajectory(clean))
    ok(all(v is not False for k, v in r4.items() if isinstance(v, bool)),
       "12 a CLEAN cancel lifecycle passes every invariant")

    # ------------------------------------------------ B2: the MAPPING ------
    _d = declared_pairs()
    ok(_d["n_present"] == 2 and _d["n_absent"] == 4,
       "B2 BE declares TWO submittable pairs today (CONDVALUE_NEUTRAL x each "
       "predictor) and FOUR absent — race multiplicity counts PAIRS, so the "
       "ones that cannot exist are DECLARED absent, not silently uncounted")
    ok(_d["interaction_claim"] is False,
       "B2 the interaction claim is FALSE, read from the code: no skew, "
       "inventory, net or front state reaches any feature block")

    _b2 = export_trajectory_b2("CONDVALUE_NEUTRAL", "composed_linear",
                               [ev(1, "PLACE"), ev(2, "FILL")])
    ok(_b2["predictor"] == "composed_linear" and _b2["interaction"] is False
       and _b2["arm"] == "CONDVALUE_NEUTRAL",
       "B2 a valid pair exports with identity as TOP-LEVEL fields")
    ok(DA.load_external_trajectory(_b2).arm == "CONDVALUE_NEUTRAL",
       "B2 and still loads through DA's real loader — identity fields sit "
       "OUTSIDE the canonical bytes, so they do not disturb the digest")

    for _c in ("CONDVALUE_X_SKEW", "CONDVALUE_X_SKEW_X_FAIRPRICE"):
        try:
            export_trajectory_b2(_c, "composed_lgbm", [ev(1, "PLACE")])
            ok(False, f"B2 BE REFUSES to emit {_c} today")
        except ExportRefused as _e:
            ok("NO valid 011 exporter" in str(_e),
               f"B2 BE REFUSES to emit {_c} today — a label the code cannot "
               f"support is the ONE error the contract cannot catch, because "
               f"the label is what it trusts")
    try:
        export_trajectory_b2("QR_SKEW_ONLY", "composed_linear",
                             [ev(1, "PLACE")])
        ok(False, "B2 BE refuses a composition it does not own")
    except ExportRefused:
        ok(True, "B2 BE refuses a composition it does not own (QR_SKEW_ONLY is "
                 "the neutral reference, not a BE predictor arm)")
    try:
        export_trajectory_b2("CONDVALUE_NEUTRAL", "composed_magic",
                             [ev(1, "PLACE")])
        ok(False, "B2 an undeclared PREDICTOR is refused")
    except ExportRefused:
        ok(True, "B2 an undeclared PREDICTOR is refused")

    # the mapping's evidence must remain TRUE of the code, not just asserted
    import phase2_state_schema_freeze as _PIN
    _f = _PIN.build_pin()["features_in_order"]
    _bad = [x for x in _f if any(t in x.lower() for t in
                                 ("skew", "invent", "_net", "net_", "front"))]
    ok(not _bad,
       f"B2 EVIDENCE HOLDS: no skew/inventory/net/front feature is in the "
       f"pinned set (found {_bad}); if one ever is, interaction:False becomes "
       f"a false claim and this falsifier fires")

    # ------------------------------------------------ B4 semantics --------
    _cs = composition_semantics()
    ok(set(_cs["consumes_predictor"]) == set(DA.ARM_SPEC),
       "B4 the declaration covers EVERY declared composition, none extra")
    _derived = {a: bool(set(sp["components"]) & {"hazard", "condvalue"})
                for a, sp in DA.ARM_SPEC.items()}
    ok(_cs["consumes_predictor"] == _derived,
       "B4 consumes_predictor agrees with an INDEPENDENT derivation from the "
       "components (hazard|condvalue are estimates; skew, cancel_hold and "
       "random_matched are rules) — the declaration is not free-hand")
    ok(_cs["role"]["RANDOM_MATCHED"] == "control",
       "B4 RANDOM_MATCHED is a CONTROL — the null apparatus, not a selectable "
       "winner (DA: being in the parity space and the candidate space differ)")
    ok(set(_cs["role"].values()) <= set(DA.ROLES),
       f"B4 every role is one DA declares {DA.ROLES}")
    ok(set(_cs["open_questions"]) == {"QR_SKEW_ONLY", "QR_CANCEL_HOLD_X_SKEW"},
       "B4 the two RULE arms are flagged as GENUINELY OPEN rather than "
       "resolved by resemblance — the code says what they ARE, not what the "
       "race intends them FOR")

    # RED-FIRST: DA's derivation must REFUSE a bad declaration.
    #
    # MY FIRST VERSION OF THESE TWO PASSED FOR THE WRONG REASON. It omitted the
    # keyword-only controls_are_candidates, so BOTH cases raised TypeError from
    # the missing argument — never reaching the typo or the omission they claim
    # to test. A test that passes because the call could not be made has tested
    # the call signature, not the guard. Every case below now supplies a
    # COMPLETE call and asserts the refusal is the RIGHT one.
    _CAC = False        # see BE_ROLE_QUESTIONS; a ruling, not BE's to make
    _base = dict(consumes_predictor=_cs["consumes_predictor"],
                 roles=_cs["role"], controls_are_candidates=_CAC)
    _m = DA.candidate_multiplicity(**_base)
    ok(isinstance(_m, dict) and _m,
       f"B4 a COMPLETE declaration DERIVES a multiplicity rather than "
       f"asserting one: {sorted(_m)[:5]}")

    # A REAL TYPO both removes and adds, so the MISSING side catches it —
    # measured, not assumed:
    _typo = {("RANDOM_MATCHD" if k == "RANDOM_MATCHED" else k): v
             for k, v in _cs["consumes_predictor"].items()}
    try:
        DA.candidate_multiplicity(**dict(_base, consumes_predictor=_typo))
        ok(False, "B4 a deliberate TYPO is REFUSED")
    except TypeError as _e:
        ok(False, f"B4 typo case raised TypeError — guard not reached: {_e}")
    except Exception as _e:
        ok("RANDOM_MATCHED" in str(_e),
           "B4 a deliberate TYPO is REFUSED — it removes the real arm as well "
           "as adding a fake one, and the MISSING side is what catches it")

    # OBSERVATION, reported not asserted as a defect: a PURELY ADDITIVE unknown
    # arm is accepted and ignored. The candidate list is identical with and
    # without it, so nothing is miscounted today; but a declaration naming a
    # composition the contract does not have passes silently, which would
    # matter if someone declared a future arm early and assumed it was counted.
    _extra = DA.candidate_multiplicity(
        **dict(_base, consumes_predictor=dict(_cs["consumes_predictor"],
                                              FUTURE_ARM=True)))
    ok(sorted(_extra["candidates"]) == sorted(_m["candidates"]),
       "B4 a purely ADDITIVE unknown arm changes NO candidate (it is ignored, "
       "not counted) — reported to DA as an observation, since a declaration "
       "naming a non-existent composition passes silently")

    try:
        DA.candidate_multiplicity(
            **dict(_base, consumes_predictor={
                k: v for k, v in _cs["consumes_predictor"].items()
                if k != "RANDOM_MATCHED"}))
        ok(False, "B4 a MISSING arm declaration is REFUSED")
    except TypeError as _e:
        ok(False, f"B4 missing-arm case raised TypeError — guard not reached: {_e}")
    except Exception as _e:
        ok("RANDOM_MATCHED" in str(_e),
           "B4 a MISSING arm declaration is REFUSED, naming it — the "
           "derivation has no default, so an unstated composition cannot be "
           "silently counted")

    print(f"\n{'BE TRAJECTORY EXPORT SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
