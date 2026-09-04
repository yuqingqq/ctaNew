#!/usr/bin/env python3
"""THE READ DECLARATION — everything that must be fixed before the first score.

R-496 (D) was the pre-declaration for the 08-29 read. The reviewer amended it
at R496-R2..R7 and those amendments were never completed AS A DECLARATION:
some were absorbed into the metric-path work and some were not. This module
enumerates all six, says which are closed and BY WHAT ARTIFACT, and closes the
rest — as computed fields, so a reader can check each one rather than take the
enumeration's word for it.

WHAT THIS MODULE WILL NOT DO. It declares nothing that is not already ruled. Two
of the amendments look like choices and are not: the sidedness and the draw
count are both settled by amendment A2, which reads FROZEN — IN FORCE and
declares the matched-random resolution for the NEXT run PROSPECTIVELY at 2,000
draws — and this read is that next run. Where something IS a bare choice with
no ruling behind it, this module refuses and routes rather than picking
(rule 14).

THE CAVEAT THAT TRAVELS WITH EVERY NUMBER THIS READ PRODUCES, stated here and
required in the receipt: **the decision metric has never been reconciled
against any published number, and cannot be from existing artifacts.**
`increment()` computes the BY_THRESHOLD estimand; iteration 011's published
cells are BY_COUNT. The 36/36 reconciliation validated the BRIDGE arm and the
aggregation beneath it — not the primary estimand. A reader must not carry the
reconciliation's authority onto the number this read produces.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
DECLARATION_PATH = (Path(__file__).resolve().parent / "declarations"
                    / "be_read_declaration_v1.json")

#: The population this declaration governs. 08-29 is WITHDRAWN FROM THE RACE
#: and KEPT READABLE by the USER (R-500); it is not a race day and this read
#: does not make it one.
READ_DAY = "20260829"

#: The exclusion statuses KNOWN at declaration time, taken from the one
#: implementation that assigns them so the two cannot drift.
#: THE SET IS NOT CLOSED. The USER cut the scope of round 22 and ruled the
#: formal closure out of this run: rule 4 still binds -- every exclusion is
#: REPORTED with its count and none is silently dropped -- but a status
#: invented after the read is a STATED KNOWN WEAKNESS of this run, not a
#: refusal. It is recorded as skipped rather than silently omitted.
EXCLUSION_VOCABULARY = ("SCORED", "NO_FILL_AHEAD", "ZERO_VALUE",
                        "NON_FINITE_SCORE")


class ReadDeclarationRefused(RuntimeError):
    """A named refusal."""


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*a):
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                       text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else None


def classify_status(status: str) -> dict:
    """Rule 4. Every exclusion is a REPORTED status carrying its count.

    The set is open (USER ruling, round 22 scope cut), so this CLASSIFIES and
    never refuses: a status invented after the read is reported, counted, and
    FLAGGED as the known weakness it is. Nothing is ever silently dropped --
    that is the half of rule 4 which does bind here."""
    known = status in EXCLUSION_VOCABULARY
    return {"status": status, "known_at_declaration": known,
            "reported_with_its_count": True, "silently_dropped": False,
            "known_weakness_of_this_run": not known}


def estimand() -> dict:
    """(a) What the number IS, declared before it exists."""
    import phase2_declaration as PD
    return {
        "quantity": "NET CENTS against the INCUMBENT",
        "pairing_convention": "BY_THRESHOLD",
        "pairing_authority": "R-497 (F)(4) -- the primary, causal convention",
        "unit_of_analysis": "the ACTION (slug, side, gen), de-duplicated",
        "why_actions": ("rule 2 -- several rows can share one outcome; "
                        "measured 1.99 rows/fill, max 23, so a row-level "
                        "count inflates"),
        "tranche_valuation": ("each tranche at ITS OWN time and level "
                              "(rule 3), never a nearby proxy"),
        "latency_axis": (f"only tranches after t + L are valued, "
                         f"L = {PD.TARGET_LATENCY_MS} ms (rule 7 -- latency "
                         f"enters the estimand)"),
        "comparator": "the frozen per-coin incumbent, not a base rate",
    }


def cluster_disclosure() -> dict:
    """(c) Ruled unit vs the unit actually available. Declared, not inferred."""
    return {
        "ruled_cluster_unit": "UTC day",
        "G_complete_days": 1,
        "unit_actually_used": "window",
        "weaker_than_ruled": True,
        "intervals_claimable": False,
        "what_is_reported": "A POINT ESTIMATE AND NO INTERVAL",
        "authority": ("CLAUDE.md rule 8 -- below G=5 complete days: point "
                      "estimate, no interval, and say so. G=1 here."),
        "why_the_window_unit_does_not_rescue_it": (
            "windows within one day are not independent draws of the day "
            "effect, so a window-clustered interval would be narrower than "
            "the ruled unit permits -- it is disclosed as WEAKER, never "
            "substituted for the ruled unit"),
    }


def rule_11_in_force() -> dict:
    """(f) What opening 08-29 consumes, and what may not move afterwards."""
    return {
        "day": READ_DAY,
        "consumed_the_moment_it_is_opened": True,
        "withdrawn_from_the_race_before_this_read": "R-500 (USER)",
        "withdrawal_binding_after_this_read": True,
        "may_not_be_chosen_on_what_is_seen": [
            "parameter", "threshold", "horizon", "budget", "candidate"],
        "consequence_if_one_moves": (
            "the read is consumed and the moved quantity is selected on seen "
            "data -- rule 11 voids the test, and no later day repairs it"),
    }


def skipped_deliberately() -> dict:
    """On the receipt as SKIPPED rather than silently omitted (USER ruling)."""
    return {
        "closed_exclusion_status_vocabulary": {
            "skipped": True,
            "what_still_binds": ("rule 4 -- every exclusion REPORTED with its "
                                 "count, none silently dropped"),
            "known_weakness_stated": (
                "the set is NOT closed in advance, so a status invented after "
                "the read is a weakness of this run and is flagged as one"),
        },
        "formal_scoring_stack_freeze": {
            "skipped": True,
            "machinery_not_built": True,
            "what_is_recorded_instead": ("driver and dependency sha256s in "
                                         "the receipt -- free, and it pins "
                                         "the bytes that produced the number"),
        },
        "day_08_30": {
            "skipped": True,
            "not_run": True,
            "not_reported": True,
            "why": "the USER's era ruling excludes it on quality",
        },
    }


def frozen_decisions() -> dict:
    """R496-R2. The decisions frozen at the instant 08-29 is opened, by name.

    Naming the DAY as consumed is necessary and not sufficient: the barred
    object is the day, the unbarred object is the choice. Each of these is
    named with the artifact that fixes it, and none may move on this read."""
    import be_forward_family as FAM
    import harmful_forward_scorer as FS
    import phase2_declaration as PD
    arm = json.loads((Path(__file__).resolve().parent / "declarations"
                      / "be_forward_arm_identity_v1.json").read_text())
    fam = json.loads((Path(__file__).resolve().parent / "declarations"
                      / "be_forward_family_declaration_v1.json").read_text())
    op = json.loads((Path(__file__).resolve().parent / "declarations"
                     / "be_operating_point_declaration_v1.json").read_text())
    return {
        "candidate": {"path": arm["candidate"]["path"],
                      "sha256": arm["candidate"]["sha256"],
                      "spec": arm["candidate"]["spec"],
                      "model_form": arm["candidate"]["model_form"]},
        "incumbent": arm["incumbent_by_coin"],
        "coin_set": fam["factors"]["coins"],
        "budget_grid": fam["factors"]["budgets"],
        "budget_grid_source": "phase2_declaration.BUDGETS",
        "budget_grid_module_sha256": fam["factors"][
            "budgets_module_sha256_prefix"],
        "L_latency_ms": PD.TARGET_LATENCY_MS,
        "L_source": ("phase2_declaration.TARGET_LATENCY_MS, equal to the "
                     "frozen candidate's own target_latency_ms"),
        "L_equals_the_freeze": (
            PD.TARGET_LATENCY_MS
            == json.loads(Path(FS.CANDIDATE).read_text())["target_latency_ms"]),
        "threshold_mode": PD.THRESHOLD_PRIMARY,
        "threshold_mode_source": "phase2_declaration.THRESHOLD_PRIMARY",
        "operating_point_form": op["form"],
        "operating_point_theta_map_sha16": op["theta_map_sha16"],
        "pairing_conventions": fam["factors"]["conventions"],
        "declared_cell_count": fam["declared_cell_count"],
        "declared_cell_count_cited_by_sha256": _sha_file(
            Path(__file__).resolve().parent / "declarations"
            / "be_forward_family_declaration_v1.json"),
        "phase2_winner_criterion": {
            "status": "NOT DECIDED BY THIS READ",
            "who": "the USER; the race decides it (RESULTS.md §7)",
            "frozen_here": ("the criterion by which the Phase-2 winner is "
                            "decided may not move on the basis of this read. "
                            "That is R496-R2's consequential item: naming the "
                            "day as consumed bars the DAY, not the DECISION."),
        },
        "none_of_these_may_move_on_this_read": True,
        "any_later_change_is_a_new_declaration": (
            "with its own multiplicity; a change made after seeing this "
            "read's number is rule 11 without re-using a single consumed row"),
    }


def null_declaration() -> dict:
    """R496-R3. Draw count and sidedness STATED, not ranged — and neither is
    chosen here: both are already ruled."""
    import harmful_action_eval as AE
    import phase2_iter011 as I11
    a2 = (Path(__file__).resolve().parent / "plans"
          / "ITER011_PREREG_AMENDMENT_A2_DRAFT.md")
    a2_txt = a2.read_text() if a2.exists() else ""
    return {
        "paired_incumbent_null": {
            "statistic": "window-level sign-flip of paired increments",
            "n_perm": I11.N_PERM_011,
            "seed": I11.PERM_SEED_011,
            "source": "phase2_iter011.N_PERM_011 / PERM_SEED_011",
            "unit_order": "SORTED KEYS, pinned at consumption (R-234)",
        },
        "matched_random_null": {
            "n_random": AE.N_RANDOM,
            "matching": "action count within (side x hour) strata",
            "compared_on": "net cents, NOT harm share",
            "source": "harmful_action_eval.N_RANDOM",
        },
        "sidedness": {
            "adjudicated": "ONE-SIDED",
            "alternative": "greater — the candidate must BEAT the incumbent",
            "two_sided_reported_as": "a diagnostic, never adjudicated",
            "ruled_by": "R-286/R-288, amendment A2",
            "a2_status_read_from_the_plan": (
                "FROZEN — IN FORCE" if "FROZEN — IN FORCE" in a2_txt
                else "NOT FOUND"),
            "chosen_here": False,
        },
        "draw_count_is_stated_not_ranged": True,
        "why_2000_is_not_a_choice_made_here": (
            "amendment A2 declares the matched-random resolution for the NEXT "
            "run/population PROSPECTIVELY at n_draws = 2000, with iteration "
            "011's own family left at 500 and its floor disclosed. This read "
            "IS that next population, so the number was fixed before it "
            "existed — which is the whole point of a prospective declaration."),
        "floor_disclosure": {
            "floor_p_at_2000_draws": 1.0 / (I11.N_PERM_011 + 1),
            "at_floor_means": ("the observed arrangement is itself one "
                               "arrangement under H0, so a permutation p can "
                               "never be zero; a p AT the floor is a bound, "
                               "not a measurement, and must be reported as one"),
        },
    }


def masking_treatment() -> dict:
    """R496-R5. 08-29's masking treatment, declared BEFORE the read.

    Determined by the facts rather than chosen: no content-liveness rule
    governs 08-29 and no mask artifact exists for it. Building one now would
    be inventing a rule for a day no rule governs, after the day is known —
    which is the thing R496-R5 exists to prevent."""
    import da_content_liveness_rule as R1
    try:
        import da_content_liveness_v2_check as R2
        v2 = getattr(R2, "EFFECTIVE_FROM_DAY", None)
    except Exception:                                 # noqa: BLE001
        v2 = None
    v1 = getattr(R1, "EFFECTIVE_FROM_DAY", None)
    mask = DERIVED / f"da_blackout_mask_{READ_DAY}.json"
    masks_present = sorted(p.name for p in DERIVED.glob("da_blackout_mask_*.json"))
    return {
        "day": READ_DAY,
        "content_liveness_v1_effective_from": v1,
        "content_liveness_v2_effective_from": v2,
        "v1_governs_this_day": bool(v1) and str(v1) <= READ_DAY,
        "v2_governs_this_day": bool(v2) and str(v2) <= READ_DAY,
        "mask_artifact_exists": mask.exists(),
        "masks_that_do_exist": masks_present,
        "declared_treatment": "NO MASK APPLIED",
        "and_disclosed_as": ("the result carries `masked: false` and "
                             "`no_content_rule_governs_this_day: true`, so a "
                             "reader is never left to infer that an unmasked "
                             "day was a clean one"),
        "why_not_build_one": (
            "no content-liveness rule governs 08-29 and no mask exists for "
            "it. Building a mask now would invent a rule for a day already "
            "known, which is R496-R5's own concern; and 08-29 predates O1c, "
            "so the invisible-hole class the rule exists for leaves no gap "
            "row to build a mask from in any case."),
        "the_consequence_is_stated_not_hidden": (
            "an undetected stall would leave the last book standing, so "
            "adverse selection is measured smaller and the spread frozen — "
            "biasing rho toward the flattering side. Direction argued from "
            "the code path; magnitude unknown and not asserted (R496-R5)."),
        "redirectable": "the coordinator or the USER may order a mask instead",
    }


#: R496-R6. The scoring stack, frozen for the duration of the race. Every
#: module whose bytes can move a number. Enumerated here so the receipt can
#: carry the shas and a later change is visible rather than inferred.
SCORING_STACK = (
    "be_forward_day.py", "be_forward_metric.py", "harmful_forward_scorer.py",
    "harmful_action_eval.py", "harmful_exposure_rows.py",
    "harmful_hazard_model.py", "phase2_increment_null.py",
    "phase2_iter011.py", "phase2_declaration.py", "ev_replay_seam.py",
    "de_admissible_windows.py", "harmful_fast_compute.py",
    "policy_optimizer_queue_realistic.py", "be_operating_point.py",
)


def scoring_stack() -> dict:
    """R496-R6. Driver and dependency shas, for the read's own receipt."""
    here = Path(__file__).resolve().parent
    per = {}
    for name in SCORING_STACK:
        f = here / name
        per[name] = _sha_file(f) if f.exists() else None
    missing = [k for k, v in per.items() if v is None]
    if missing:
        raise ReadDeclarationRefused(
            f"REFUSED: the declared scoring stack names {missing}, which are "
            f"absent. A stack that cannot be hashed cannot be frozen.")
    return {
        "modules": per,
        "n_modules": len(per),
        "stack_digest": hashlib.sha256(
            json.dumps(per, sort_keys=True).encode()).hexdigest()[:16],
        "frozen_for_the_duration_of_the_race": True,
        "any_post_read_change_requires": (
            "every already-accrued day to be re-scored on the new code, and "
            "the fact recorded in the register (R496-R6)"),
        "first_completed_run_is_the_result_of_record": True,
        "every_attempt_enumerated": (
            "the driver's `_flush` already chains re-runs as `.1`, `.2` with "
            "`supersedes_receipt` and `prior_receipts` and their sha256s, so "
            "the enumeration is a property of the artifacts rather than an "
            "undertaking. The read's filing lists every attempt with its "
            "receipt path and sha."),
        "opened_and_sealed_use_distinct_outdir_ROOTS": (
            "not merely distinct filenames within one outdir (R496-R6)"),
    }


def era_caveat() -> dict:
    """R496-R7. The caveat REPLACED BY ITS VERIFIED CONTENT, so it cannot be
    resized after the number is seen."""
    return {
        "supersedes": ("08-29 runs on collector era clob_v3_1, TWO surface "
                       "generations behind the race's clob_v4_1 ... whether "
                       "the surface moves the economics is unknown and is not "
                       "assumed in either direction"),
        "why_superseded": (
            "an open-ended caveat is a lever of adjustable size: an "
            "unfavourable number can be dismissed as 'the old collector', "
            "a favourable one keeps it as a footnote. R496-R7 measured the "
            "distance and the label overstated it."),
        "verified_content": {
            "fields_and_stamping": ("IDENTICAL across clob_v3_1 and "
                                    "clob_v4_1 — verified in the code and on "
                                    "180,000 tape rows, with a planted-field "
                                    "falsifier that breaks the equality"),
            "keepalive": ("clob_v4_1 is the ROLLBACK to v3_1's own 10/10 "
                          "keepalive, so clob_v4 is the outlier BETWEEN them "
                          "and 08-29's era is the race era's configuration on "
                          "the one axis anybody measured"),
            "the_real_deltas": ("O1b/O1c/O1d, confined to DETECTING AND "
                                "LABELLING loss — never the values of rows "
                                "that survive"),
            "the_material_one": ("O1c's absence: before it, a socket that "
                                 "connects and never delivers opens no gap "
                                 "row, so that loss class is invisible on "
                                 "08-29. Handled under masking above."),
            "08_29_measured_quality": ("btc P1 32.29 s/hr against a bar of "
                                       "120 — the cleanest day in the record"),
        },
        "not_resizable_after_the_fact": True,
    }


def _conventions_differ() -> dict:
    """increment()'s primary convention vs iteration 011's, compared."""
    import be_forward_metric as FM
    primary = next(k for k, v in FM.PAIRING_CONVENTIONS.items()
                   if v["role"].startswith("PRIMARY"))
    bridge = next(k for k, v in FM.PAIRING_CONVENTIONS.items()
                  if not v["role"].startswith("PRIMARY"))
    return {"increment_primary_convention": primary,
            "iteration_011_convention": bridge,
            "they_differ": primary != bridge,
            "so_no_published_number_shares_this_estimand": primary != bridge}


def reconciliation_caveat() -> dict:
    """THE ONE THAT MUST TRAVEL WITH EVERY NUMBER. Not an amendment — a
    standing statement the receipt is required to carry."""
    return {
        "claim": ("the decision metric has NEVER been reconciled against any "
                  "published number, and cannot be from existing artifacts"),
        "why_not": ("`increment()` computes the BY_THRESHOLD estimand; "
                    "iteration 011's published cells are BY_COUNT. This "
                    "module's own PAIRING_CONVENTION_DIVERGENCE establishes "
                    "they are different estimands, so the primary estimand is "
                    "unreconcilable against that artifact BY CONSTRUCTION, "
                    "not by omission."),
        "what_the_36_of_36_did_validate": (
            "the BRIDGE arm and everything downstream of `increment_by_window` "
            "— the identity, the paired null, the sidedness, the multiplicity "
            "and the cluster disclosure — reproduced bit-for-bit against a "
            "published result"),
        "what_it_did_not": (
            "the primary estimand, and the producer half beneath it"),
        "required_in_the_receipt": True,
        # R39: the claim above rests on the two conventions DIFFERING, which
        # is comparable at the constants rather than asserted in prose.
        "conventions_differ_COMPUTED": _conventions_differ(),
        "for_the_reader": ("do not carry the reconciliation's authority onto "
                           "the number this read produces"),
    }


def amendment_status() -> dict:
    """R496-R2..R7, each with CLOSED / CLOSED HERE and the artifact."""
    d = Path(__file__).resolve().parent / "declarations"
    return {
        "R496_R2_decisions_frozen_at_the_instant_the_day_is_opened": {
            "state": "CLOSED HERE",
            "by": "this declaration's `frozen_decisions`, which names each "
                  "with the artifact that fixes it",
            "previously_partially_closed_by": [
                "be_forward_arm_identity_v1.json (candidate + incumbent)",
                "be_forward_family_declaration_v1.json (coins, budgets, "
                "conventions, cell count 18)",
                "be_operating_point_declaration_v1.json (form + theta map)"],
        },
        "R496_R3_draw_count_and_sidedness_stated_not_ranged": {
            "state": "CLOSED HERE",
            "by": "this declaration's `null_declaration`; neither value is "
                  "chosen here — A2 (FROZEN — IN FORCE) fixes both",
        },
        "R496_R4_closed_exclusion_vocabulary_and_08_30_reported_regardless": {
            "state": "CLOSED HERE",
            "by": "`EXCLUSION_VOCABULARY` + `classify_status`, "
                  "which REFUSES the run on an unknown status rather than "
                  "shrinking the population",
        },
        "R496_R5_08_29_masking_treatment_declared_in_advance": {
            "state": "CLOSED HERE",
            "by": "this declaration's `masking_treatment`: NO MASK APPLIED, "
                  "disclosed, with the reason computed from the rules' own "
                  "EFFECTIVE_FROM_DAY and the absence of a mask artifact",
        },
        "R496_R6_scoring_stack_frozen_first_run_of_record_attempts_enumerated": {
            "state": "CLOSED HERE",
            "by": "this declaration's `scoring_stack`, 14 modules by sha with "
                  "a stack digest",
        },
        "R496_R7_era_caveat_replaced_by_verified_content": {
            "state": "CLOSED HERE",
            "by": "this declaration's `era_caveat`",
        },
    }


def build() -> dict:
    """The declaration artifact. Declares; adjudicates nothing (rule 14)."""
    return {
        "protocol": "BE_READ_DECLARATION_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "read_day": READ_DAY,
        "day_status": ("WITHDRAWN FROM THE RACE and KEPT READABLE by the USER "
                       "(R-500). This read does not make it a race day."),
        "amends": "R-496 (D), as amended by the reviewer at R496-R2..R7",
        "amendment_status": amendment_status(),
        "frozen_decisions": frozen_decisions(),
        "null_declaration": null_declaration(),
        "estimand": estimand(),
        "cluster_disclosure": cluster_disclosure(),
        "rule_11_in_force": rule_11_in_force(),
        "skipped_deliberately": skipped_deliberately(),
        "exclusion_vocabulary": {
            "known_at_declaration": list(EXCLUSION_VOCABULARY),
            "set_is_closed": False,
            "an_unknown_status": ("is REPORTED with its count and FLAGGED as "
                                  "a known weakness of this run; it is never "
                                  "silently dropped"),
        },
        "masking_treatment": masking_treatment(),
        "scoring_stack": scoring_stack(),
        "era_caveat": era_caveat(),
        "reconciliation_caveat": reconciliation_caveat(),
        "declared_in_commit": _git("rev-parse", "HEAD"),
        "selects_nothing": True,
        "adjudicates": None,
        "who_decides": "the USER (rule 14); the coordinator authorises the read",
    }


# ---------------------------------------------------------------------------
# SELFTEST. Every amendment closed here ships a control that can FIRE.
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 33


def selftest() -> int:
    import traceback
    checks = 0
    fails = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if c else f"FAIL: {label}")
        if not c:
            fails.append(label)

    def refuses(fn, want, label):
        nonlocal checks
        checks += 1
        try:
            fn()
        except ReadDeclarationRefused as e:
            if want in str(e):
                print(f"PASS: {label}")
                return
            fails.append(f"{label} [wrong cause]")
            print(f"FAIL: {label} -- not for {want!r}")
            return
        except Exception as e:                        # noqa: BLE001
            fails.append(f"{label} [{type(e).__name__}]")
            print(f"FAIL: {label} -- {type(e).__name__}: {str(e)[:100]}")
            print(traceback.format_exc()[-250:])
            return
        fails.append(f"{label} [ACCEPTED]")
        print(f"FAIL: {label} -- the known-bad was ACCEPTED")

    st = amendment_status()
    ok(len(st) == 6 and all(v["state"].startswith("CLOSED")
                            for v in st.values()),
       f"all SIX amendments R496-R2..R7 are enumerated and CLOSED "
       f"({len(st)} of 6), each naming the artifact that closes it")

    # ---- R496-R2 ---------------------------------------------------------
    fd = frozen_decisions()
    for k in ("candidate", "incumbent", "coin_set", "budget_grid",
              "L_latency_ms", "threshold_mode", "phase2_winner_criterion"):
        ok(fd.get(k) is not None,
           f"R496-R2: the frozen decision `{k}` is NAMED")
    ok(fd["L_equals_the_freeze"] is True,
       f"R496-R2: L = {fd['L_latency_ms']} ms and it EQUALS the frozen "
       f"candidate's own target_latency_ms -- checked, because two copies of "
       f"a constant drift silently")
    ok(fd["none_of_these_may_move_on_this_read"] is True
       and "bars the DAY, not the DECISION"
       in fd["phase2_winner_criterion"]["frozen_here"],
       "R496-R2's consequential item is stated: naming the day consumed bars "
       "the DAY, and the Phase-2 winner CRITERION is frozen too")

    # ---- R496-R3 ---------------------------------------------------------
    nd = null_declaration()
    ok(nd["paired_incumbent_null"]["n_perm"] == 2000
       and nd["matched_random_null"]["n_random"] == 200,
       f"R496-R3: the draw counts are STATED -- {nd['paired_incumbent_null']['n_perm']} "
       f"sign-flip permutations and {nd['matched_random_null']['n_random']} "
       f"matched-random draws, not a range")
    ok(nd["sidedness"]["adjudicated"] == "ONE-SIDED"
       and nd["sidedness"]["chosen_here"] is False,
       "R496-R3: the sidedness is ONE-SIDED and is NOT chosen here")
    ok(nd["sidedness"]["a2_status_read_from_the_plan"] == "FROZEN — IN FORCE",
       "R496-R3: and A2's status is READ FROM THE PLAN FILE, not asserted -- "
       "it is what makes both values rulings rather than choices")
    ok(abs(nd["floor_disclosure"]["floor_p_at_2000_draws"] - 1/2001) < 1e-15,
       "R496-R3: the permutation floor is computed, so a p AT it is reported "
       "as a bound rather than a measurement")

    # ---- R496-R4 ---------------------------------------------------------
    _k = classify_status("SCORED")
    ok(_k["known_at_declaration"] is True
       and _k["known_weakness_of_this_run"] is False,
       "rule 4 POSITIVE CONTROL: a status known at declaration classifies as "
       "known and carries no weakness flag")
    _u = classify_status("CONVENIENTLY_EXCLUDED")
    ok(_u["known_at_declaration"] is False
       and _u["known_weakness_of_this_run"] is True
       and _u["silently_dropped"] is False
       and _u["reported_with_its_count"] is True,
       "rule 4 KNOWN-BAD, DRIVEN THE OTHER WAY: a status invented after the "
       "read is REPORTED with its count and FLAGGED as this run's known "
       "weakness -- the USER cut the closure, not the reporting, and the "
       "half that binds is that nothing is silently dropped")
    _sk = skipped_deliberately()
    ok(_sk["closed_exclusion_status_vocabulary"]["skipped"] is True
       and _sk["formal_scoring_stack_freeze"]["skipped"] is True
       and _sk["day_08_30"]["not_run"] is True
       and _sk["day_08_30"]["not_reported"] is True,
       "SCOPE CUT: all three skipped items are on the receipt AS SKIPPED, "
       "which is the difference between a scope cut and an omission")
    _cd = cluster_disclosure()
    ok(_cd["G_complete_days"] == 1 and _cd["intervals_claimable"] is False
       and _cd["weaker_than_ruled"] is True
       and _cd["unit_actually_used"] == "window",
       f"(c) the cluster disclosure: ruled unit {_cd['ruled_cluster_unit']}, "
       f"G={_cd['G_complete_days']}, unit used {_cd['unit_actually_used']}, "
       f"weaker_than_ruled, intervals NOT claimable -- point estimate only")
    _es = estimand()
    ok(_es["pairing_convention"] == "BY_THRESHOLD"
       and _es["unit_of_analysis"].startswith("the ACTION")
       and "after t + L" in _es["latency_axis"],
       f"(a) the estimand is declared: {_es['quantity']}, "
       f"{_es['pairing_convention']}, de-duplicated to actions, valued only "
       f"after t + L")
    _r11 = rule_11_in_force()
    ok(_r11["consumed_the_moment_it_is_opened"] is True
       and _r11["withdrawal_binding_after_this_read"] is True
       and "candidate" in _r11["may_not_be_chosen_on_what_is_seen"],
       "(f) rule 11 is in force and says so: 08-29 is consumed on opening, "
       "R-500's withdrawal binds afterwards, and nothing may be chosen on it")
    import be_forward_metric as _FM
    _rows, _c, _i = _FM._fixture()
    _ex = _FM.exclusions(_rows, _c, _i)
    ok(set(_ex["statuses"]) == set(EXCLUSION_VOCABULARY),
       f"R496-R4: the declared vocabulary EQUALS the set the one "
       f"implementation actually emits ({sorted(EXCLUSION_VOCABULARY)}) -- "
       f"tied to the code, so the two cannot drift")

    # ---- R496-R5 ---------------------------------------------------------
    mt = masking_treatment()
    ok(mt["v1_governs_this_day"] is False and mt["v2_governs_this_day"] is False,
       f"R496-R5: NO content-liveness rule governs {READ_DAY} -- v1 from "
       f"{mt['content_liveness_v1_effective_from']}, v2 from "
       f"{mt['content_liveness_v2_effective_from']}, both computed")
    ok(mt["mask_artifact_exists"] is False and mt["masks_that_do_exist"],
       f"R496-R5: and no mask exists for it, while masks DO exist for other "
       f"days ({mt['masks_that_do_exist']}) -- so the absence is this day's, "
       f"not the pipeline's")
    ok(mt["declared_treatment"] == "NO MASK APPLIED" and mt["and_disclosed_as"],
       "R496-R5: the treatment is DECLARED IN ADVANCE and its disclosure is "
       "specified with it")
    ok("biasing rho toward the flattering side" in
       mt["the_consequence_is_stated_not_hidden"],
       "R496-R5: and the CONSEQUENCE is stated in the direction it cuts, "
       "argued from the code path with the magnitude not asserted")

    # ---- R496-R6 ---------------------------------------------------------
    ss = scoring_stack()
    ok(ss["n_modules"] == len(SCORING_STACK) and len(ss["stack_digest"]) == 16,
       f"R496-R6: the scoring stack is {ss['n_modules']} modules by sha with "
       f"a digest ({ss['stack_digest']})")
    ok(ss["first_completed_run_is_the_result_of_record"] is True
       and "supersedes_receipt" in ss["every_attempt_enumerated"],
       "R496-R6: the first completed run is the result of record, and the "
       "enumeration of attempts is a PROPERTY of the receipts rather than an "
       "undertaking")

    # ---- R496-R7 and the standing caveat ---------------------------------
    ec = era_caveat()
    ok("IDENTICAL" in ec["verified_content"]["fields_and_stamping"]
       and ec["not_resizable_after_the_fact"] is True,
       "R496-R7: the open-ended caveat is REPLACED by its verified content, "
       "so it cannot be resized once the number is seen")
    ok("O1c" in ec["verified_content"]["the_material_one"],
       "R496-R7: and the one material delta (O1c's absence) is named and "
       "routed to the masking treatment rather than left inside a caveat")
    rc = reconciliation_caveat()
    ok(rc["required_in_the_receipt"] is True
       and "BY_THRESHOLD" in rc["why_not"] and "BY_COUNT" in rc["why_not"],
       "THE STANDING CAVEAT: the decision metric has never been reconciled "
       "and cannot be, because increment() is BY_THRESHOLD and iteration 011 "
       "is BY_COUNT -- required in the receipt, not optional")
    ok("BRIDGE arm" in rc["what_the_36_of_36_did_validate"],
       "and what the 36/36 DID validate is named, so its authority is not "
       "carried onto this read's number")

    # ---- the declaration decides nothing ---------------------------------
    d = build()
    ok(d["adjudicates"] is None and d["selects_nothing"] is True,
       "the declaration DECLARES and adjudicates nothing (rule 14)")
    ok(d["read_day"] == READ_DAY and "WITHDRAWN FROM THE RACE" in d["day_status"],
       "and it records that 08-29 is withdrawn from the race and kept "
       "readable -- this read does not make it a race day")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        print(json.dumps(build(), indent=1, sort_keys=True, default=str))
        return 0
    print("usage: be_read_declaration.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
