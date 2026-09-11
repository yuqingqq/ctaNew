"""THE FAIR-VALUE LANE'S PROGRESS LEDGER, KEYED ON §5's SIX BUILD GATES.

DA 277. The previous version was keyed on §11's EIGHT implementation steps
while the rule it enforces -- "no labelled score before the gates" -- is stated
over §5's SIX BUILD GATES. THE TWO LISTS DIVERGE AT 6:

    §5 gate 6  = the REPLAY SEAM (arms share inputs, paths stay free)
    §11 step 6 = FREEZE THE FULL PIPELINE and both candidate identities

so `gates_satisfied` counted out of eight against a threshold of six, and the
number was not comparable with any other seat's count of the same thing. The
six gates are the rows here; the §11 step is a COLUMN, because the plan uses
both and a reader needs the correspondence.

EVERY STATUS IS COMPUTED WHEN THIS RUNS. Not copied from a report, a register
row, a dispatch, or an earlier run of this ledger. Three independent
measurements make a row:

  1. PRESENCE   `git ls-tree` counts at a FETCHED ref.
  2. BEHAVIOUR  the gate's own falsifier, DRIVEN HERE, in a worktree cut at
                that ref -- exit code and cell count, not a remembered verdict.
  3. PROPERTIES DA's own adversarial cells for the properties §5 states in
                words, driven against the same bytes.

WHY BEHAVIOUR IS DRIVEN AND NOT RECORDED. v4 of this file carried a BEHAVIOUR
table of verdicts I had typed in after driving them by hand, each pinned to the
blob it was earned against. That is better than a bare verdict and still wrong
in the same direction: it is a REPORT ABOUT A DRIVE, and it ages the moment the
blob moves. Measured today, 2026-09-11: I drove all six gates at 70ab5b8 at
19:29Z; by 19:44Z the ref had moved twice, to 43c590e and then f0c303c, and
DE's own commit message for the second says "tightening gate 4 broke gates 5
and 6, and only the REF drive caught it". Fifteen minutes. A ledger that
remembers verdicts would have reported six green gates against bytes that no
longer existed.

A LEDGER ROW CAN BE WRONG IN BOTH DIRECTIONS, AND BOTH HAPPENED AT ONCE.
Measured 2026-09-11T20:19Z, this ledger read 5/6 and REV's read 5/6 -- on
DIFFERENT rows, and both of mine were wrong:

  * GATE 4 read UNSATISFIED because MY FIXTURE was broken. It passed a bare
    list as the canonical population, which REVIEW 202's provenance rule
    correctly refuses, so every property cell returned
    CANONICAL_POPULATION_HAS_NO_PROVENANCE. The properties were intact. A
    false NEGATIVE manufactured by the instrument -- the third of this shape
    I have made, after the cwd bug and the wrong-path attribution.

  * GATE 6 read SATISFIED because MY PROPERTY LIST HAD NO ENTRY for the
    shared action population. Eight declared properties, all green, and the
    ninth was never written down. A property list that omits a property makes
    a gate look satisfied for EXACTLY the reason it is not: `run_arm` takes
    `actions` outside `ReplayInputs`, so two arms replaying 6 and 3 actions
    are reported `inputs_identical: True` with an identical `inputs_digest`.
    A challenger that drops half the population passes the seam.

So the property list is itself an artifact that can be incomplete, and a green
row is only as strong as the list it was scored against. `properties` carries
`n_declared` for that reason: the count is a claim about MY enumeration, not
about the plan's.

AND CELLS PASSING IS NOT PROPERTIES COVERED. REV measured gate 4 at 16/16
cells with four of §5's six properties enforced, and gate 6 at 11/11 cells
proving the arms DECLARED the same inputs rather than RAN on them -- a flat
tape passed. Both were green falsifiers over an incomplete property set, which
is precisely the shape a cell count cannot show. So `cells` and
`properties_covered` are SEPARATE FIELDS and a gate needs both.

THE STANDING RULE THIS CAME FROM:

    A LANDING IS PROVEN BY A COUNT AT A FETCHED REF, NEVER BY A COMMIT SHA
    IN PROSE.

A sha in a message proves somebody made a commit. It does not prove the commit
is reachable from any ref another seat will fetch. Measured 2026-09-11T19:20Z,
two shas reported as landed work: 8fe2a2e and f9a5bc7, remote refs containing
each: NONE.

Usage:  da_fair_value_ledger.py            # the ledger, as JSON
        da_fair_value_ledger.py --falsify  # this file's own cells
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PROTOCOL = "P003_DA_FAIR_VALUE_LEDGER_V5"
PLAN = "fair_value_plan.md v1.2"
REFS = ("origin/mm-research", "origin/de-freeze-chain-v2", "origin/be-build-runner")
#: `origin/mm-research` is the user's fork and is declared NON-EXECUTING
#: (da_shared_tree_non_executing_v1.json). Presence there is RECORDED and
#: never counts toward a gate.
EXECUTING_REFS = ("origin/de-freeze-chain-v2", "origin/be-build-runner")
REQUIRED_REFS = EXECUTING_REFS
NON_EXECUTING_REF = "origin/mm-research"

#: §5 OF THE PLAN, VERBATIM IN SUBSTANCE -- the six build gates, which are the
#: rows of this ledger. (gate, title, paths, owning seat, §11 step).
GATES = (
    (1, "Settlement verifier: official winner join, margin-aware Chainlink "
        "agreement status, gap/outage statuses, head-resolving supersession",
     ("live/pm_research/da_fair_value_gate1_labels.py",), "DA", 1),
    (2, "Sigma producer: the C2 contract with scale, minimum-count, gap, "
        "zero-volatility, stale-input, pre-era and future-knowledge falsifiers",
     ("live/pm_research/be_sigma_30m.py",), "BE", 2),
    (3, "Estimator wrapper: a valid FairPrice for C1/C2; source-event and "
        "local-knowledge timestamps stay distinct at every hop",
     ("live/pm_research/de_fair_price_wrapper.py",), "DE", 3),
    (4, "Canonical forecast-action builder: one row per actual consumption "
        "decision on the neutral Identity path, key (coin, slug, "
        "generation_id, decision_recv_ns), sides fold, duplicates refuse",
     ("live/pm_research/de_canonical_action_population.py",
      "live/pm_research/de_fair_value_actions.py"), "DE", 4),
    (5, "Policy seam: Identity substitution is bit-identical, a non-Identity "
        "value moves the anchor, absent challenger falls back to Identity",
     ("live/pm_research/de_fair_value_policy_seam.py",), "DE", 5),
    (6, "Replay seam: baseline and challenger share all non-fair-value "
        "parameters, input snapshot and initial state, while their own "
        "resulting order paths stay free",
     ("live/pm_research/de_fair_value_replay_seam.py",), "DE", 6),
)

#: THE §11 CORRESPONDENCE, AND WHERE IT BREAKS. Gates 1-5 are §11 steps 1-5.
#: Gate 6 is NOT §11 step 6: the gate is the replay seam, the step is freezing
#: the whole pipeline, which SUBSUMES the seam and is a strictly larger claim.
#: §11 steps 7 and 8 (ten-day predictive validation; economic clock) have NO
#: §5 gate at all -- they are what the gates unlock, not gates themselves.
STEP11_DIVERGENCE = {
    "gates_1_to_5_are_steps_1_to_5": True,
    "gate_6_vs_step_6": ("§5 gate 6 is the REPLAY SEAM. §11 step 6 is FREEZE "
                         "THE FULL PIPELINE and both candidate identities, "
                         "which subsumes the seam. Satisfying gate 6 does NOT "
                         "satisfy step 6."),
    "step11_steps_with_no_gate": {7: "run ten-day predictive validation",
                                  8: "economic clock, predictive winners only"},
    "why_it_matters": ("a count of six out of EIGHT steps against a threshold "
                       "of SIX gates is not the same number, and two seats "
                       "reporting it disagreed for that reason alone"),
}

#: The threshold is the gate count, and it comes from §5's list length.
N_GATES = len(GATES)

#: DA'S OWN PROPERTY CELLS, driven against the ref's bytes in a subprocess.
#: These are NOT the owning seat's falsifier re-run -- they are a SECOND
#: INSTRUMENT over the properties §5 states in words, written from the plan
#: text rather than from the module, so that agreement is evidence (rule 38).
#: Each probe prints one JSON object: {"properties": {name: bool, ...}}.
PROBE_GATE_4 = r'''
import json, sys
sys.path.insert(0, PM)
import de_fair_value_actions as A
NS = 1788980100000000000; W = 1788980100
SLUG = "btc-updown-5m-%d" % W; GEN = "7"
# REVIEW 202: a population with no provenance cannot be told from one
# fabricated around the rows under test, so a BARE LIST is refused. This probe
# passed one, and every cell came back CANONICAL_POPULATION_HAS_NO_PROVENANCE
# -- so the row read CELLS_GREEN_BUT_PROPERTY_UNCOVERED when the properties
# were intact and the FIXTURE was broken. A probe must be fixed like any other
# instrument; the gate was right to refuse it.
POP = {"rows": [(SLUG, GEN)],
       "provenance": {"population": "da_fair_value_ledger gate-4 property probe",
                      "as_of": "computed at run time by the ledger",
                      "source_identity": "DA synthetic fixture, not a real population"}}
def row(**kw):
    d = dict(coin="btc", slug=SLUG, generation_id=GEN, decision_recv_ns=NS,
             quote_side="BID", up_probability_consumed=0.5, window_start=W)
    d.update(kw); return d
def run(rows, pop=POP):
    try:
        r = A.build_actions(rows, canonical_population=pop)
        return "OK n_actions=%d folded=%d %s" % (
            r["n_actions"], r["n_quote_sides_folded"], sorted(r["excluded"]))
    except A.ActionsRefused as e:
        return "REFUSED " + str(e).split(":")[0].replace("REFUSED ", "")
    except Exception as e:
        return "ERROR " + type(e).__name__ + " " + str(e)[:60]
P = {}
P["a_one_row_per_decision_not_per_quote"] = (
    "n_actions=1 folded=2" in run([row(quote_side="BID"), row(quote_side="ASK")]))
P["b_membership_derived_no_population_refuses"] = (
    "CANONICAL_POPULATION_NOT_SUPPLIED" in run([row()], pop=None))
P["c_key_is_four_part_differing_stamp_is_two_actions"] = (
    "n_actions=2" in run([row(), row(decision_recv_ns=NS + 1)]))
P["d_same_value_both_sides_remains_one_action"] = (
    "n_actions=1" in run([row(), row(quote_side="ASK")]))
P["e_duplicate_key_different_value_refuses_the_build"] = (
    "FORECAST_ACTION_DUPLICATE_KEY" in run(
        [row(), row(quote_side="ASK", up_probability_consumed=0.9)]))
P["f_claimed_path_flag_checked_against_population"] = (
    "FLAG_CONTRADICTS" in run([row(slug="ghost", on_identity_reference_path=True)]))
P["g_off_population_row_is_a_status_not_a_silent_drop"] = (
    "ACTION_NOT_IN_THE_CANONICAL_POPULATION" in run([row(slug="ghost")]))
print(json.dumps({"properties": P}))
'''

PROBE_GATE_6 = r'''
import json, sys, inspect, dataclasses as _dc
sys.path.insert(0, PM)
import de_fair_value_actions as A, de_fair_value_replay_seam as R
NS = 1788980100000000000; W = 1788980100
rows = [{"coin": "btc", "slug": "btc-updown-5m-%d" % (W + 300 * i),
         "generation_id": str(i), "decision_recv_ns": NS + i, "quote_side": "BID",
         "up_probability_consumed": 0.5, "window_start": W + 300 * i,
         "on_identity_reference_path": True} for i in range(6)]
def popof(rr):
    return {"rows": [(r["slug"], r["generation_id"]) for r in rr],
            "provenance": {"population": "da_fair_value_ledger gate-6 probe",
                           "as_of": "computed at run time by the ledger",
                           "source_identity": "DA synthetic fixture"}}
def actsof(rr):
    return sorted(A.build_actions(rr, canonical_population=popof(rr))["actions"],
                  key=lambda a: a.decision_recv_ns)
acts = actsof(rows)
half = actsof(rows[:3])
REAL = (0.52, 0.48, 0.55, 0.45, 0.50, 0.60)
FLAT = tuple([0.99] * 6)          # REV's flat tape, rebuilt from the finding
# `action_keys_sha256` IS REQUIRED (Q-DE-371). Every ReplayInputs below supplies
# it. The previous probe omitted it, died with a TypeError BEFORE printing, and
# the harness recorded `probed: false` -- which the status branch then read as
# SATISFIED. That is why this probe now builds inputs through ONE helper.
def mk(tape, acts_for, hs=0.01, st=0.0, **kw):
    return R.ReplayInputs(non_fair_value_params={"max_inventory": 5},
                          initial_state={"inventory": st, "clock": 0},
                          price_path=tape, half_spread=hs,
                          action_keys_sha256=R.action_keys_digest(acts_for), **kw)
def vo(v): return lambda a: (v, "probe", False)
def arm(tape, v, acts_for=None, hs=0.01, st=0.0):
    acts_for = acts if acts_for is None else acts_for
    return R.run_arm(acts_for, vo(v), mk(tape, acts_for, hs, st))
def run(fn):
    try:
        r = fn(); return "OK " + str(r)[:40]
    except R.ReplayRefused as e:
        return "REFUSED " + str(e).split(":")[0].replace("REFUSED ", "")
    except Exception as e:
        return "ERROR " + type(e).__name__ + " " + str(e)[:70]
P = {}
P["a_flat_tape_challenger_is_refused_by_name"] = (
    "ARMS_DO_NOT_SHARE" in run(lambda: R.compare_arms(arm(REAL, 0.50), arm(FLAT, 0.58))))
P["b_same_tape_differing_value_still_compares"] = (
    run(lambda: R.compare_arms(arm(REAL, 0.50), arm(REAL, 0.58))).startswith("OK"))
P["c_differing_half_spread_refuses"] = (
    "ARMS_DO_NOT_SHARE" in run(lambda: R.compare_arms(arm(REAL, 0.50), arm(REAL, 0.50, hs=0.02))))
P["d_differing_initial_state_refuses"] = (
    "ARMS_DO_NOT_SHARE" in run(lambda: R.compare_arms(arm(REAL, 0.50), arm(REAL, 0.50, st=3.0))))
P["e_a_lied_declared_snapshot_digest_refuses"] = (
    "DECLARED_SNAPSHOT" in run(lambda: mk(REAL, acts, declared_snapshot_sha256="0" * 64)))
P["f_a_true_declared_snapshot_digest_is_accepted"] = (
    run(lambda: mk(REAL, acts,
                   declared_snapshot_sha256=mk(REAL, acts).digest())).startswith("OK"))
b, c = arm(REAL, 0.50), arm(REAL, 0.58)
P["g_order_paths_remain_free_to_differ"] = (b["path"].digest() != c["path"].digest())
P["h_tape_is_not_an_argument_to_run_arm"] = (
    "price_path" not in inspect.signature(R.run_arm).parameters
    and "price_path" in R.ReplayInputs.__dataclass_fields__)
# THE SHARED ACTION POPULATION (REV 203, closed structurally by Q-DE-371).
P["i_differing_action_populations_refuse"] = (
    "ARMS_DO_NOT_SHARE" in run(lambda: R.compare_arms(
        arm(REAL, 0.50), arm(REAL, 0.58, acts_for=half))))
P["j_a_LIED_action_digest_refuses"] = (
    "ACTIONS_ARE_NOT_THE_DECLARED_POPULATION" in run(
        lambda: R.run_arm(half, vo(0.58), mk(REAL, acts))))
P["k_the_action_population_is_REQUIRED_like_price_path"] = (
    R.ReplayInputs.__dataclass_fields__["action_keys_sha256"].default is _dc.MISSING)
print(json.dumps({"properties": P}))
'''

PROBE_GATE_3 = r'''
import json, sys
sys.path.insert(0, PM)
import de_fair_price_wrapper as Wm
def run(fn):
    try:
        return "OK " + str(fn())[:70]
    except Wm.WrapperRefused as e:
        return "REFUSED " + str(e)[:90]
    except Exception as e:
        return "ERROR " + type(e).__name__ + " " + str(e)[:60]
P = {}
# §5 gate 3's surviving clause: the two clocks stay DISTINCT at every hop, and
# equality is a DECLARATION rather than an accident. DRIVEN, not grepped: the
# previous version of this probe matched source text, which is a control that
# fires on a label -- the exact defect REVIEW 201 found in the module it tests.
P["a_collapsed_clocks_undeclared_REFUSE"] = "REFUSED" in run(
    lambda: Wm.Stamped(1.0, 1000.0, 1000.0, "feed"))
P["b_the_same_pair_DECLARED_is_admitted"] = run(
    lambda: Wm.Stamped(1.0, 1000.0, 1000.0, "feed", equal_clocks_declared=True)
    ).startswith("OK")
P["c_knowledge_predating_the_event_REFUSES"] = "REFUSED" in run(
    lambda: Wm.Stamped(1.0, 1000.0, 999.0, "feed"))
P["d_an_ordinary_distinct_pair_is_admitted"] = run(
    lambda: Wm.Stamped(1.0, 1000.0, 1000.5, "feed")).startswith("OK")
# AND THE DECLARATION IS KEPT. The refusal text promises it "is recorded";
# REVIEW 201 found the hop record dropped it, so a declared-equal feed landed
# as transport_s 0.0, indistinguishable from a MEASURED zero.
hop = Wm._hops(feed=Wm.Stamped(1.0, 1000.0, 1000.0, "feed",
                               equal_clocks_declared=True))["feed"]
P["e_the_declaration_is_KEPT_in_the_hop_record"] = (
    hop.get("equal_clocks_declared") is True)
P["f_a_declared_zero_is_not_confusable_with_a_measured_one"] = (
    hop.get("transport_s") == 0 and "DECLARED" in str(hop.get("zero_transport_is")))
meas = Wm._hops(feed=Wm.Stamped(1.0, 1000.0, 1000.5, "feed"))["feed"]
P["g_a_MEASURED_transport_is_labelled_measured"] = (
    str(meas.get("zero_transport_is")) == "measured")
print(json.dumps({"properties": P}))
'''

PROBE_GATE_5 = r'''
import json, sys
sys.path.insert(0, PM)
import de_fair_value_actions as A, de_fair_value_policy_seam as S
NS = 1788980100000000000; W = 1788980100
rows = [{"coin": "btc", "slug": "btc-updown-5m-%d" % (W + 300 * i),
         "generation_id": str(i), "decision_recv_ns": NS + i, "quote_side": "BID",
         "up_probability_consumed": 0.5, "window_start": W + 300 * i,
         "on_identity_reference_path": True} for i in range(4)]
def popof(rr):
    return {"rows": [(r["slug"], r["generation_id"]) for r in rr],
            "provenance": {"population": "da_fair_value_ledger gate-6 probe",
                           "as_of": "computed at run time by the ledger",
                           "source_identity": "DA synthetic fixture"}}
def actsof(rr):
    return sorted(A.build_actions(rr, canonical_population=popof(rr))["actions"],
                  key=lambda a: a.decision_recv_ns)
acts = actsof(rows)
def seam(v, label, fell=False):
    return S.run_seam(acts, lambda a: (v, label, fell), half_spread=0.01)
P = {}
# §5's last required falsifier, DRIVEN: changing only the estimator LABEL while
# the consumed value is unchanged must be DETECTED as a decorative seam.
P["a_label_only_change_is_DETECTED_as_decorative"] = (
    S.compare_runs(seam(0.50, "identity"), seam(0.50, "c2_bn_bookticker")
                   )["verdict"] == S.DECORATIVE)
P["b_identity_against_itself_is_BIT_IDENTICAL"] = (
    S.compare_runs(seam(0.50, "identity"), seam(0.50, "identity")
                   )["verdict"] == "BIT_IDENTICAL")
P["c_a_different_value_is_LOAD_BEARING"] = (
    S.compare_runs(seam(0.50, "identity"), seam(0.58, "c2_bn_bookticker")
                   )["verdict"] == "VALUE_IS_LOAD_BEARING")
P["d_the_moved_anchors_are_COUNTED_not_just_flagged"] = (
    S.compare_runs(seam(0.50, "identity"), seam(0.58, "c2")
                   )["n_anchors_moved"] == len(acts))
# THE FALLBACK LIVES IN THE POLICY, NOT THE ESTIMATOR (DE 353).
r = seam(0.50, "identity", fell=True)
P["e_an_absent_challenger_falls_back_and_is_COUNTED"] = (
    r["fallbacks"].get("fell_back") == len(acts))
r2 = seam(0.50, "c2", fell=False)
P["f_a_used_candidate_is_counted_separately"] = (
    r2["fallbacks"].get("used_candidate") == len(acts))
print(json.dumps({"properties": P}))
'''


PROBE_GATE_1 = r'''
import json, sys
sys.path.insert(0, PM)
import da_fair_value_gate1_labels as G
P = {}
def cls(x0, xT, up, **kw):
    sym, series, market, win = G._fixture(x0, xT, up)
    return G.classify_window(slug="p", market=market,
                             winners=kw.pop("winners", win),
                             streams={(sym, G.WINDOW_S): series}, **kw)
# §5 item 1, driven. THE OFFICIAL JOIN IS NOT OPTIONAL.
r = cls(100.0, 101.0, True, winners=None)
P["a_no_official_row_means_NO_OFFICIAL_and_no_label"] = (
    r["status"] == G.NO_OFFICIAL and not r.get("label"))
r = cls(100.0, 101.0, True)
P["b_an_exact_endpoint_label_admits_with_a_label"] = (
    r["status"] == G.LABEL_ADMISSIBLE and r["label"] == "UP")
r = cls(101.0, 100.0, False)
P["c_the_DOWN_direction_admits_too"] = (
    r["status"] == G.LABEL_ADMISSIBLE and r["label"] == "DOWN")
# AN INVERTED SETTLEMENT MAPPING IS DETECTED AS DISAGREEMENT, NOT ABSENCE.
r = cls(100.0, 101.0, False)
P["d_an_inverted_mapping_is_DETECTED_as_disagreement"] = (r["status"] == G.DISAGREE)
# MARGIN-AWARE: a margin below the feed's resolution is its OWN status.
r = cls(100.0, 100.0 + 1e-9, True)
P["e_a_margin_below_resolution_is_its_own_status"] = (
    r["status"] == G.MARGIN_UNRESOLVABLE)
# EXACTLY ONE STATUS ADMITS A LABEL, and the map is total over the grammar.
P["f_exactly_one_status_admits_a_label"] = (len(G.ADMITTING) == 1)
P["g_the_consumer_map_is_TOTAL_over_the_status_grammar"] = (
    all(st in G.STATUS_MAP for st in G.STATUSES)
    and all(v in G.CONSUMER_STATUSES for v in G.STATUS_MAP.values()))
print(json.dumps({"properties": P}))
'''

PROBE_GATE_2 = r'''
import json, sys
sys.path.insert(0, PM)
import be_sigma_30m as S
NOW = S.ERA_FLOOR_NS + 10 * 3600 * S.NS
def st(ticks, t=NOW):
    return S.sigma_30m(ticks, t)["status"]
P = {}
# §5 item 2's seven named falsifiers, each driven to ITS OWN status.
full = S._synth(NOW, 2e-5)
ok = S.sigma_30m(full, NOW)
P["a_scale_a_2e-5_path_measures_2e-5"] = (
    ok["status"] == "OK" and abs(ok["sigma_per_sqrt_s"] - 2e-5) < 2e-7)
dbl = S.sigma_30m(S._synth(NOW, 4e-5), NOW)
P["b_scale_DOUBLING_the_path_doubles_the_estimate"] = (
    dbl["status"] == "OK"
    and abs(dbl["sigma_per_sqrt_s"] / ok["sigma_per_sqrt_s"] - 2.0) < 1e-6)
P["i_NO_ANNUALISATION_the_estimate_is_per_sqrt_second"] = (
    ok.get("units") == "per_sqrt_s" or not ok.get("annualised"))
P["c_minimum_count_too_few_returns_REFUSES"] = (
    st(S._synth(NOW, 2e-5, first_index=S.WINDOW_S - 2)) == S.INSUFFICIENT_RETURNS)
P["d_a_source_GAP_over_the_limit_REFUSES"] = (
    st(S._synth(NOW, 2e-5, skip=tuple(range(100, 900)))) == S.SOURCE_GAP_OVER_LIMIT)
P["e_ZERO_volatility_is_its_own_status"] = (
    st(S._synth(NOW, 0.0)) == S.ZERO_VOLATILITY)
P["f_a_STALE_input_is_its_own_status"] = (
    st(S._synth(NOW, 2e-5, last_index=S.WINDOW_S - 600)) == S.STALE_INPUT)
P["g_PRE_ERA_is_refused_by_name"] = (
    st(S._synth(S.ERA_FLOOR_NS - 3600 * S.NS, 2e-5), S.ERA_FLOOR_NS - 3600 * S.NS)
    == S.PRE_ERA)
# FUTURE KNOWLEDGE. The property the PLAN requires is that no post-decision
# row reaches the estimate. This module satisfies it by EXCLUDING such rows and
# COUNTING them, not by refusing the record -- `FUTURE_KNOWLEDGE_IN_SOURCE` is a
# structural backstop on an already-filtered series. My first version of this
# cell asserted the REFUSAL, which is my guess at the mechanism rather than the
# plan's requirement, and it failed against correct code.
_fut = S.sigma_30m(S._synth(NOW, 2e-5, extra_offsets_ns=(5 * S.NS,)), NOW)
_wild = list(S._synth(NOW, 2e-5)) + [(NOW + k * S.NS, 1e6) for k in (1, 2, 3)]
_wr = S.sigma_30m(_wild, NOW)
P["h_post_decision_rows_change_NOTHING_and_are_COUNTED"] = (
    abs(_fut["sigma_per_sqrt_s"] - ok["sigma_per_sqrt_s"]) < 1e-18
    and _fut["n_rows_after_decision"] == 1
    and abs(_wr["sigma_per_sqrt_s"] - ok["sigma_per_sqrt_s"]) < 1e-18
    and _wr["n_rows_after_decision"] == 3)
print(json.dumps({"properties": P}))
'''

PROBES = {1: PROBE_GATE_1, 2: PROBE_GATE_2, 3: PROBE_GATE_3, 4: PROBE_GATE_4, 5: PROBE_GATE_5, 6: PROBE_GATE_6}
#: Gates 1 and 2 carry no second DA instrument, and the ledger says so rather
#: than leaving an empty dict that reads as coverage. Gate 1 IS DA's own
#: module, so DA's cells are the FIRST instrument there, not a second one;
#: gate 2's 27 cells are BE's, re-driven here against the ref's bytes.
#: EVERY GATE IS PROBED. There is no longer an exemption, because an exemption
#: and a CRASH were indistinguishable in the output: both produced
#: `probed: false`, and the status branch read that as SATISFIED.
#:
#: GATE 1'S PROBE IS SAME-SEAT and is recorded as such. It is a COVERAGE check
#: over the properties §5 item 1 states, not an independent instrument -- DA
#: wrote the module and the probe. It establishes `probed`, never independence.
NO_DA_PROBE = {}
SAME_SEAT_PROBE = {
    1: ("DA wrote both the module and this probe. It measures COVERAGE of §5 "
        "item 1's stated properties; it is not a second instrument, and "
        "agreement between it and gate 1's own cells is not convergence."),
}

FAIR_VALUE_FILE_RE = r"(fair|sigma|forecast|seam|canonical)"


def _root() -> str:
    """THE REPO ROOT, RESOLVED -- `git ls-tree` takes paths relative to CWD.

    The first version of this function ran `git ls-tree` without `-C`, so from
    `live/pm_research` every path resolved to
    `live/pm_research/live/pm_research/...` and EVERY COUNT CAME BACK 0: the
    ledger reported the whole lane unlanded, including a file verified as
    landed minutes earlier. It passed its falsifier because every cell was a
    NEGATIVE control, and a bug returning 0 for everything satisfies all of
    them. The positive control in `falsify()` is what catches it.

    THE FALLBACK WAS THE SECOND HALF OF THE BUG. `git rev-parse` without a
    directory is CWD-DEPENDENT, and returning "." when it fails meant that
    driving this file from anywhere outside the repo made every `git -C .`
    call fail silently and every count come back 0 -- the exact input that
    makes a row read UNKNOWN or, before DA 280, default-SATISFIED. DE
    reproduced it from /tmp with a minimal environment (REVIEW 205's dependent
    sweep) and the landed-path control fired, as it was written to.

    So: resolve from THIS MODULE'S OWN LOCATION first, then the cwd, and
    REFUSE if neither is a repository. A path this file cannot verify is not a
    path it may guess at.
    """
    here = Path(__file__).resolve().parent
    for base in (here, Path.cwd()):
        r = subprocess.run(["git", "-C", str(base), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    raise RuntimeError(
        "REFUSED NO_REPOSITORY_RESOLVABLE: every count in this ledger comes "
        "from a git ref, and neither this module's directory nor the cwd is "
        "inside a repository. Returning '.' here is what made a landed path "
        "count 0 from outside the repo.")


def _git(*args, text=True):
    return subprocess.run(["git", "-C", _root(), *args],
                          capture_output=True, text=text)


def _count(ref: str, path: str) -> int:
    r = _git("ls-tree", "--name-only", ref, path)
    return len([x for x in r.stdout.splitlines() if x.strip()])


def _blob_sha16(ref: str, path: str) -> str:
    r = _git("show", f"{ref}:{path}", text=False)
    return hashlib.sha256(r.stdout).hexdigest()[:16] if r.returncode == 0 else ""


def remote_refs_containing(sha: str) -> list:
    r = _git("branch", "-r", "--contains", sha)
    return sorted(x.strip() for x in r.stdout.splitlines() if x.strip())


def unattributed_files(ref: str) -> list:
    """Lane files on `ref` that NO gate claims. A missing attribution shows up
    here rather than as a silent zero in somebody's row.

    This guard exists because v1 of this file GUESSED gate 4's paths
    (`de_canonical_forecast_action`, `de_forecast_action_scorer`); the real
    files are `de_canonical_action_population` and `de_fair_value_actions`, so
    the row read NO_FILES_ON_ANY_EXECUTING_REF while two of its files sat on
    both refs -- a false negative produced by the instrument, not the world.
    """
    r = _git("ls-tree", "-r", "--name-only", ref, "live/pm_research/")
    claimed = {p for _, _, paths, _, _ in GATES for p in paths}
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".py") or line in claimed:
            continue
        if re.search(FAIR_VALUE_FILE_RE, Path(line).name):
            out.append(line)
    return sorted(out)


_CELL = re.compile(r"^\s*\[(PASS|FAIL)\]")
_NN = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")
_JSONN = re.compile(r'"n"\s*:\s*(\d+)\s*,\s*"failed"\s*:\s*(\d+)')


def _parse_cells(out: str) -> dict:
    """Cell counts from a falsifier's own stdout, by whichever convention it
    uses -- and the convention is NAMED, so a zero from an unparsed format is
    visible as `method: none` instead of reading as a module with no cells."""
    n_pass = len([l for l in out.splitlines() if _CELL.match(l) and "[PASS]" in l])
    n_fail = len([l for l in out.splitlines() if _CELL.match(l) and "[FAIL]" in l])
    if n_pass or n_fail:
        return {"n": n_pass + n_fail, "failed": n_fail, "method": "[PASS]/[FAIL] lines"}
    m = _JSONN.search(out)
    if m:
        return {"n": int(m.group(1)), "failed": int(m.group(2)), "method": '"n"/"failed" json'}
    m = _NN.findall(out)
    if m:
        a, b = m[-1]
        return {"n": int(b), "failed": int(b) - int(a), "method": "N/N summary"}
    return {"n": 0, "failed": 0, "method": "none"}


def drive_gate(gate: int, paths, worktree: str, timeout: int = 900) -> dict:
    """DRIVE the gate's falsifier FROM THE WORKTREE'S BYTES.

    The module is the LAST path of the gate (the one carrying the falsifier),
    run as `python3 <module>.py --falsify` with cwd inside the worktree, so the
    file under test is the ref's file and not whatever is in the shared tree.
    DE's own Q-DE-367b is the argument for doing it this way: his four modules
    were green in the tree where he changed them and REFUSED from the ref's
    bytes, because the fixtures had not been updated for gate 4's tightening.
    A drive that does not name its tree proves nothing about any other tree.
    """
    pm = str(Path(worktree) / "live" / "pm_research")
    mod = Path(paths[-1]).name
    try:
        r = subprocess.run(["python3", mod, "--falsify"], cwd=pm,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"driven": True, "rc": None, "cells": {"n": 0, "failed": 0,
                "method": "none"}, "verdict": "TIMEOUT", "module": mod}
    cells = _parse_cells(r.stdout + r.stderr)
    # A GREEN EXIT WITH NO CELLS IS NOT A PASS (rule 15): an instrument that
    # never proved it can fire has not reported anything.
    if r.returncode == 0 and cells["n"] > 0 and cells["failed"] == 0:
        verdict = "DRIVEN_GREEN"
    elif r.returncode == 0 and cells["n"] == 0:
        verdict = "EXIT_0_BUT_NO_CELLS_PARSED"
    else:
        verdict = "DRIVEN_FAILING"
    return {"driven": True, "rc": r.returncode, "cells": cells,
            "verdict": verdict, "module": mod,
            "tail": (r.stdout.strip().splitlines() or [""])[-1][:120]}


def probe_gate(gate: int, worktree: str, timeout: int = 300) -> dict:
    """Run DA's own property cells against the worktree's bytes."""
    if gate not in PROBES:
        return {"probed": False, "reason": NO_DA_PROBE.get(gate, "no probe")}
    pm = str(Path(worktree) / "live" / "pm_research")
    src = f"PM = {pm!r}\n" + PROBES[gate]
    try:
        r = subprocess.run(["python3", "-c", src], cwd=pm,
                           capture_output=True, text=True, timeout=timeout)
        props = json.loads(r.stdout.strip().splitlines()[-1])["properties"]
    except Exception as e:
        return {"probed": False, "reason": f"{type(e).__name__}: {str(e)[:140]}"}
    return {"probed": True, "properties": props,
            "n_covered": sum(1 for v in props.values() if v),
            "n_declared": len(props),
            "all_covered": all(props.values()),
            "uncovered": sorted(k for k, v in props.items() if not v)}



FREEZE_DECL_RE = r"(fair_value.*(freeze|frozen)|freeze.*fair_value)"


def step11_step6_freeze(refs=EXECUTING_REFS) -> dict:
    """IS §11 STEP 6 SATISFIED? It is NOT the same question as gate 6.

    §5 gate 6 is the replay seam. §11 step 6 is "freeze the full pipeline and
    both candidate identities" -- a strictly larger claim, and the one the
    sentence "No fair-value score is evidence before step 6" is about. With all
    six BUILD gates green, the §5 barrier is down and this one is not, so the
    ledger measures it rather than letting one boolean read as permission.
    """
    found = {}
    for ref in refs:
        r = _git("ls-tree", "-r", "--name-only", ref,
                 "orchestrator/PROGRAMS/P-2026-003-polymarket-5min/")
        hits = [l.strip() for l in r.stdout.splitlines()
                if re.search(FREEZE_DECL_RE, l, re.I)]
        found[ref] = sorted(hits)
    n = min(len(v) for v in found.values()) if found else 0
    return {"declarations_found": found, "n_on_every_executing_ref": n,
            "satisfied": n > 0,
            "measured": "git ls-tree -r <ref> <program dir> | match a freeze declaration",
            "why_separate": ("§5 gate 6 is the replay seam; §11 step 6 is the "
                             "FULL-PIPELINE freeze plus both candidate "
                             "identities, which the seam does not establish")}


UNPROBED = "PROPERTIES_NOT_PROBED"


def row_status(present_on, beh, props) -> str:
    """THE STATUS DECISION, as one testable function.

    IT USED TO END IN `else: status = "SATISFIED"`, AND THAT DEFAULT WAS THE
    WHOLE DEFECT. A probe that crashed before printing produced
    `probed: false`, which matched no earlier branch and fell through to the
    default -- so an UNPROBED gate read SATISFIED and counted toward
    `gates_satisfied`. Measured 2026-09-11T20:43Z at the live ref: THREE rows
    read SATISFIED with `probed: false`, and the ledger printed 6/6 while gate
    6's probe was dying with a TypeError. REV proved the same branch would have
    printed 6/6 the round before, when gate 6 genuinely failed.

    AN UNPROBED GATE MUST NEVER READ SATISFIED. `SATISFIED` is now reachable
    only through an explicit `probed is True`, and every other case has a name.
    """
    if not present_on:
        return "NOT_ON_ANY_EXECUTING_REF"
    if sorted(present_on) != sorted(REQUIRED_REFS):
        return "ON_ONE_EXECUTING_REF_ONLY"
    if beh.get("verdict") != "DRIVEN_GREEN":
        return "LANDED_BUT_" + str(beh.get("verdict"))
    if props.get("probed") is not True:
        return UNPROBED
    if not props.get("all_covered"):
        return "CELLS_GREEN_BUT_PROPERTY_UNCOVERED"
    return "SATISFIED"


def executing_refs_divergence() -> dict:
    """THE TWO EXECUTING REFS ARE NOT ALWAYS ONE SHA, AND THAT IS FINE.

    Recorded so a reader meeting two shas is not surprised: the refs can carry
    one commit each that the other does not, while every lane module is
    byte-identical across both. What matters for a gate is the BLOB, not the
    head, and the blob equality is computed here rather than assumed.
    """
    heads = {r: _git("rev-parse", r).stdout.strip() for r in EXECUTING_REFS}
    paths = sorted({p for _, _, ps, _, _ in GATES for p in ps}
                   | {"live/pm_research/da_fair_value_ledger.py"})
    blobs = {p: {r: _blob_sha16(r, p) for r in EXECUTING_REFS} for p in paths}
    identical = {p: len(set(v.values())) == 1 for p, v in blobs.items()}
    a, b = EXECUTING_REFS
    ahead = {
        a: len([x for x in _git("log", "--format=%h", f"{b}..{a}").stdout.split() if x]),
        b: len([x for x in _git("log", "--format=%h", f"{a}..{b}").stdout.split() if x]),
    }
    return {
        "heads": heads,
        "heads_equal": len(set(heads.values())) == 1,
        "commits_each_ref_has_that_the_other_does_not": ahead,
        "n_lane_modules_compared": len(paths),
        "all_lane_modules_byte_identical_across_both_refs": all(identical.values()),
        "modules_that_differ": sorted(k for k, v in identical.items() if not v),
        "reading": ("a gate is satisfied by the BLOB on both refs, not by the "
                    "heads matching. Two shas with identical modules is a "
                    "normal state and blocks nothing."),
    }


def build(fetch: bool = True, drive: bool = True, ref: str = EXECUTING_REFS[0]) -> dict:
    if fetch:
        subprocess.run(["git", "-C", _root(), "fetch", "--quiet", "origin"], check=False)
    head = _git("rev-parse", "--short", ref).stdout.strip()
    wt = None
    if drive:
        wt = tempfile.mkdtemp(prefix="da_ledger_drive_")
        shutil.rmtree(wt, ignore_errors=True)
        a = _git("worktree", "add", "--detach", wt, ref)
        if a.returncode != 0:
            wt = None
    rows, satisfied = [], 0
    for gate, title, paths, owner, step11 in GATES:
        per_ref = {r: {p: _count(r, p) for p in paths} for r in REFS}
        present_on = [r for r in EXECUTING_REFS if all(per_ref[r][p] > 0 for p in paths)]
        blobs = {p: _blob_sha16(ref, p) for p in paths}
        beh = drive_gate(gate, paths, wt) if (wt and present_on) else {
            "driven": False, "verdict": "NOT_DRIVEN",
            "reason": "no worktree" if not wt else "not present on the ref"}
        props = probe_gate(gate, wt) if (wt and present_on) else {
            "probed": False, "reason": "not present on the ref"}
        # THE STATUS, COMPUTED. Presence, then behaviour, then properties --
        # and a gate with a green falsifier but an uncovered declared property
        # is NOT satisfied, because that is exactly what REV found twice.
        status = row_status(present_on, beh, props)
        # A ROW COUNTS ONLY IF IT WAS PROBED. Stated twice on purpose: once in
        # `row_status`, and once here, so a future edit to either cannot
        # quietly restore a default-satisfied path.
        if status == "SATISFIED" and props.get("probed") is True:
            satisfied += 1
        rows.append({
            "gate": gate, "plan": f"{PLAN} §5 item {gate}", "title": title,
            "step11_step": step11,
            "step11_note": (STEP11_DIVERGENCE["gate_6_vs_step_6"]
                            if gate == 6 else "same as the §5 gate"),
            "owner": owner, "paths": list(paths),
            "required_refs": list(REQUIRED_REFS),
            "missing_from_required_refs": [r for r in REQUIRED_REFS if r not in present_on],
            "present_on_executing_refs": present_on,
            "landed_only_on_the_non_executing_fork": bool(
                not present_on and all(per_ref[NON_EXECUTING_REF][p] > 0 for p in paths)),
            "counts_per_ref": per_ref,
            "blob_sha256_16_at_ref": blobs,
            "cells": beh, "properties": props,
            "probed": props.get("probed") is True,
            "same_seat_probe": SAME_SEAT_PROBE.get(gate),
            "counts_toward_gates_satisfied": (
                status == "SATISFIED" and props.get("probed") is True),
            "status": status,
        })
    step6 = step11_step6_freeze()
    if wt:
        _git("worktree", "remove", "--force", wt)
        shutil.rmtree(wt, ignore_errors=True)
    return {
        "protocol": PROTOCOL, "plan": PLAN,
        "keyed_on": "§5's SIX BUILD GATES",
        "measured_at_ref": ref, "ref_head": head,
        "gates": rows,
        "gates_satisfied": satisfied,
        "gates_probed": sum(1 for r in rows if r["properties"].get("probed") is True),
        "gates_satisfied_counts_only_probed_rows": True,
        "unprobed_gates": [r["gate"] for r in rows
                           if r["properties"].get("probed") is not True],
        "executing_refs": executing_refs_divergence(),
        "n_gates": N_GATES,
        # DA 277's predicate, EXACTLY AS SPECIFIED -- the §5 build-gate barrier.
        "no_labelled_score_permitted": satisfied < N_GATES,
        "what_that_predicate_covers":
            "§5's SIX BUILD GATES ONLY. It goes False when the BUILD barrier "
            "is down. It is NOT a statement that a labelled score is now "
            "evidence -- see `step11_step6_freeze` and "
            "`score_is_evidence_permitted` below.",
        "step11_step6_freeze": step6,
        "score_is_evidence_permitted": (satisfied >= N_GATES) and step6["satisfied"],
        "WHY_TWO_PREDICATES":
            "with all six build gates green the §5 barrier is down, and §11's "
            "sentence 'No fair-value score is evidence before step 6' is still "
            "binding because step 6 is the FULL-PIPELINE FREEZE, not the "
            "replay seam. One boolean reading as blanket permission is how a "
            "build gate becomes a licence.",
        "the_rule_this_makes_checkable":
            f"{PLAN} §5: no labelled score before all {N_GATES} build gates; "
            f"§11: 'No fair-value score is evidence before step 6.'",
        "step11_divergence": STEP11_DIVERGENCE,
        "THE_STANDING_RULE":
            "A LANDING IS PROVEN BY A COUNT AT A FETCHED REF, NEVER BY A "
            "COMMIT SHA IN PROSE.",
        "CELLS_PASSING_IS_NOT_PROPERTIES_COVERED":
            "a green falsifier over an incomplete property set is the shape a "
            "cell count cannot show; REV found it twice, on gates 4 and 6",
        "owners": {g: o for g, _, _, o, _ in GATES},
        "no_gate_claimed_by_two_seats": len({g for g, *_ in GATES}) == N_GATES,
        "required_refs": list(REQUIRED_REFS),
        "non_executing_ref": NON_EXECUTING_REF,
        "unattributed_lane_files": {r: unattributed_files(r) for r in EXECUTING_REFS},
        "every_status_computed_here": True,
        "no_status_copied_from_a_report": True,
        "behaviour_is_driven_not_recorded": bool(drive),
    }


def falsify() -> int:
    bad = 0

    def ck(label, cond, shown=""):
        nonlocal bad
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -> {shown}" if shown else ""))
        if not cond:
            bad += 1

    # ---- the rows are §5's six gates, not §11's eight steps ----------------
    ck("the ledger is keyed on §5's SIX gates", N_GATES == 6 and len(GATES) == 6,
       f"{N_GATES} gates")
    ck("the §11 correspondence is CARRIED, not dropped",
       all(isinstance(s, int) for *_, s in GATES),
       {g: s for g, _, _, _, s in GATES})
    ck("...and the divergence at 6 is STATED, not smoothed over",
       "REPLAY SEAM" in STEP11_DIVERGENCE["gate_6_vs_step_6"]
       and "FREEZE" in STEP11_DIVERGENCE["gate_6_vs_step_6"])
    ck("§11's two gate-less steps are named",
       set(STEP11_DIVERGENCE["step11_steps_with_no_gate"]) == {7, 8})

    # ---- the driver must prove it can FAIL (rule 15) -----------------------
    tmp = tempfile.mkdtemp(prefix="da_ledger_fals_")
    pm = Path(tmp) / "live" / "pm_research"
    pm.mkdir(parents=True)
    (pm / "green_mod.py").write_text(
        "import sys\n"
        "if '--falsify' in sys.argv:\n"
        "    print('  [PASS] a cell'); print('  [PASS] another'); sys.exit(0)\n")
    (pm / "red_mod.py").write_text(
        "import sys\n"
        "if '--falsify' in sys.argv:\n"
        "    print('  [PASS] a cell'); print('  [FAIL] a broken cell'); sys.exit(1)\n")
    (pm / "silent_mod.py").write_text(
        "import sys\nsys.exit(0)\n")
    g = drive_gate(0, ("live/pm_research/green_mod.py",), tmp)
    r = drive_gate(0, ("live/pm_research/red_mod.py",), tmp)
    s = drive_gate(0, ("live/pm_research/silent_mod.py",), tmp)
    ck("POSITIVE CONTROL: a green module drives DRIVEN_GREEN with cells > 0",
       g["verdict"] == "DRIVEN_GREEN" and g["cells"]["n"] == 2, str(g["cells"]))
    ck("NEGATIVE CONTROL: a failing module drives DRIVEN_FAILING",
       r["verdict"] == "DRIVEN_FAILING" and r["cells"]["failed"] == 1, str(r["cells"]))
    ck("A SILENT exit-0 module is NOT counted as a pass (rule 15)",
       s["verdict"] == "EXIT_0_BUT_NO_CELLS_PARSED", s["verdict"])
    ck("...and the counting CONVENTION is named, so an unparsed format is visible",
       s["cells"]["method"] == "none" and g["cells"]["method"].startswith("[PASS]"))
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- counts are real, and the cwd bug stays dead -----------------------
    ck("A PATH KNOWN TO BE LANDED COUNTS > 0 -- the control that catches a cwd bug",
       all(_count(r_, "live/pm_research/da_fair_value_gate1_labels.py") == 1
           for r_ in EXECUTING_REFS),
       {r_.replace("origin/", ""): _count(r_, "live/pm_research/da_fair_value_gate1_labels.py")
        for r_ in EXECUTING_REFS})
    ck("a path absent everywhere counts 0 on every ref",
       all(_count(r_, "live/pm_research/de_nonexistent_seam.py") == 0 for r_ in REFS))
    ck("a sha reported as landed but on NO remote ref is proven unlanded",
       remote_refs_containing("f9a5bc7") == [], "f9a5bc7 -> []")
    ck("presence on the NON-EXECUTING fork is never an executing ref",
       NON_EXECUTING_REF not in EXECUTING_REFS)

    # ---- the ledger itself -------------------------------------------------
    led = build()
    ck("every gate has a COMPUTED status", all(r_["status"] for r_ in led["gates"]),
       f"{len(led['gates'])} gates")
    ck("the score predicate is COMPUTED from the count against SIX",
       led["no_labelled_score_permitted"] == (led["gates_satisfied"] < N_GATES),
       f"gates_satisfied={led['gates_satisfied']}/{N_GATES} -> "
       f"no_labelled_score_permitted={led['no_labelled_score_permitted']}")
    ck("CELLS and PROPERTIES are separate fields on every row",
       all("cells" in r_ and "properties" in r_ for r_ in led["gates"]))
    ck("a gate with an UNCOVERED declared property cannot read SATISFIED",
       all(not (r_["properties"].get("probed")
                and not r_["properties"].get("all_covered")
                and r_["status"] == "SATISFIED") for r_ in led["gates"]))
    ck("every gate has exactly ONE owning seat",
       len(led["owners"]) == N_GATES and all(r_["owner"] for r_ in led["gates"]),
       {r_["gate"]: r_["owner"] for r_ in led["gates"]})
    ck("a gate missing from a REQUIRED ref is named, not silently satisfied",
       all(isinstance(r_["missing_from_required_refs"], list) for r_ in led["gates"]))
    ck("no lane file is left UNATTRIBUTED without being named",
       isinstance(led["unattributed_lane_files"], dict),
       {k.replace("origin/", ""): len(v) for k, v in led["unattributed_lane_files"].items()})
    ck("the behaviour on every row was DRIVEN here, not recorded",
       all(r_["cells"].get("driven") for r_ in led["gates"])
       and led["behaviour_is_driven_not_recorded"])
    # ---- §11 step 6 is measured, and is NOT gate 6 ------------------------
    ck("§11 step 6 is measured SEPARATELY from §5 gate 6",
       "step11_step6_freeze" in led and "satisfied" in led["step11_step6_freeze"],
       f"step6 satisfied={led['step11_step6_freeze']['satisfied']} "
       f"(n={led['step11_step6_freeze']['n_on_every_executing_ref']})")
    ck("...and the two predicates are not the same boolean",
       led["score_is_evidence_permitted"] ==
       ((led["gates_satisfied"] >= N_GATES) and led["step11_step6_freeze"]["satisfied"]),
       f"no_labelled_score_permitted={led['no_labelled_score_permitted']} vs "
       f"score_is_evidence_permitted={led['score_is_evidence_permitted']}")
    ck("clearing the BUILD gates never by itself permits a labelled score",
       not (led["no_labelled_score_permitted"] is False
            and led["score_is_evidence_permitted"] is True
            and not led["step11_step6_freeze"]["satisfied"]))
    # THE GUARD AGAINST MY OWN DEFECT. v5's first gate-3 and gate-5 probes
    # reported coverage from `"...text..." in src` -- a control that fires on a
    # LABEL, which is the exact defect REVIEW 201 found in the code they test.
    ck("NO property probe reports coverage from a SOURCE-TEXT match",
       all(" in src" not in src and "open(PM" not in src for src in PROBES.values()),
       f"{len(PROBES)} probes")
    ck("every property probe IMPORTS and DRIVES its module",
       all(any(f"import {pre}" in src for pre in ("de_", "da_", "be_"))
           for src in PROBES.values()),
       f"{len(PROBES)} probes")
    ck("the repo root is resolved from THIS MODULE, not the cwd (DE/REVIEW 205)",
       _root() == subprocess.run(
           ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
            "--show-toplevel"], capture_output=True, text=True).stdout.strip(),
       _root())
    # ---- AN UNPROBED GATE MUST NEVER READ SATISFIED (DA 280) --------------
    _green = {"verdict": "DRIVEN_GREEN"}
    _both = list(REQUIRED_REFS)
    ck("NEGATIVE CONTROL: a probe that CRASHED reads PROPERTIES_NOT_PROBED",
       row_status(_both, _green,
                  {"probed": False, "reason": "IndexError: list index out of range"})
       == UNPROBED)
    ck("...and REV's exact case -- everything else green, probe absent -- is NOT satisfied",
       row_status(_both, _green, {"probed": False}) != "SATISFIED",
       row_status(_both, _green, {"probed": False}))
    ck("POSITIVE CONTROL: a probed, fully covered row still reaches SATISFIED",
       row_status(_both, _green, {"probed": True, "all_covered": True}) == "SATISFIED")
    ck("a probed row with an uncovered property is named, not satisfied",
       row_status(_both, _green, {"probed": True, "all_covered": False})
       == "CELLS_GREEN_BUT_PROPERTY_UNCOVERED")
    ck("`probed: true` is required EXACTLY -- a truthy non-True cannot pass",
       row_status(_both, _green, {"probed": 1, "all_covered": True}) == UNPROBED)
    ck("gates_satisfied counts ONLY rows with probed: true",
       led["gates_satisfied"] ==
       sum(1 for r in led["gates"] if r["status"] == "SATISFIED"
           and r["properties"].get("probed") is True)
       and led["gates_satisfied_counts_only_probed_rows"],
       f"satisfied={led['gates_satisfied']} probed={led['gates_probed']}")
    ck("EVERY gate is probed -- there is no exemption left to hide a crash in",
       led["gates_probed"] == N_GATES, f"unprobed: {led['unprobed_gates']}")
    ck("a same-seat probe is DECLARED as such, not passed off as independent",
       all(r["same_seat_probe"] for r in led["gates"] if r["gate"] in SAME_SEAT_PROBE))
    # ---- the two executing refs -------------------------------------------
    ck("the two executing refs are compared by BLOB, not by head",
       isinstance(led["executing_refs"]["all_lane_modules_byte_identical_across_both_refs"], bool),
       f"heads_equal={led['executing_refs']['heads_equal']} "
       f"modules_identical={led['executing_refs']['all_lane_modules_byte_identical_across_both_refs']}")
    ck("...and any module that DIFFERS across the refs is named",
       isinstance(led["executing_refs"]["modules_that_differ"], list),
       led["executing_refs"]["modules_that_differ"] or "none differ")
    ck("every row names the BLOB its verdict was earned against",
       all(all(v for v in r_["blob_sha256_16_at_ref"].values()) for r_ in led["gates"]))
    print(f"\n  {'LEDGER CELLS PASS' if not bad else str(bad) + ' FAILED'}")
    return bad


if __name__ == "__main__":
    import sys
    if "--falsify" in sys.argv:
        sys.exit(1 if falsify() else 0)
    print(json.dumps(build(), indent=1))
